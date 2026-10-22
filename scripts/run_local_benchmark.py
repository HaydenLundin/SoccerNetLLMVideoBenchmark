"""
Local Open-Weight Benchmark Runner for SoccerNet V2
Evaluates the physically verified local VLM cohort on NVIDIA RTX 3000 Ada (8 GB VRAM).

Supported Models (Locked Physical Hardware Ceilings):
- SmolVLM2-2.2B (BF16, locked W=60s)
- Qwen3.5-4B (4-bit NF4, enable_thinking=False, max_pixels=256*256, locked W=60s)
- MiniCPM-V 4.6 (4-bit NF4, locked W=60s)
- Moondream2 (BF16, 2D composite contact sheet storyboard, locked W=60s)
- LFM2.5-VL-3B (4-bit NF4, stepped ablation locked at W=30s)
- Qwen2.5-VL-7B (4-bit NF4, longitudinal bridge locked at W=15s)

Key Architectural Protections:
1. Single-Pass Sequential In-Memory Frame Extraction (zero seek drift, ~199 MB RAM per 615s chunk).
2. Tail-Window Duration Clamping and 3-Tier Offset Math (actual_start_ms + window_start_ms + pred_offset_ms).
3. Scope-Masked Ground Truth Evaluator for chunk runs (eliminates artificial False Negatives from un-ingested 80% match).
4. Single-load model weight residency with per-window CUDA cache clearing.
5. Official SoccerNet Tight Average-mAP (1s-5s) and Loose Average-mAP (5s-60s) evaluation.
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import sys
import json
import time
import math
import argparse
import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from PIL import Image

# Add workspace root to Python path
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from processing.unified_schema import (
    SpottedEvent,
    SpottedEventMinimal,
    SoccerClipAnnotation,
    SYSTEM_INSTRUCTION,
    SOCCERNET_CLASSES,
    parse_mm_ss_to_seconds
)
from processing.post_process_predictions import (
    stitch_chunks_to_match_predictions,
    class_aware_nms,
    DEFAULT_CLASS_NMS_RADII_MS
)
from processing.evaluate_soccernet import evaluate_match_action_spotting

LOCAL_MODEL_CONFIGS = {
    "smolvlm2-2.2b": {
        "name": "SmolVLM2-2.2B",
        "hf_id": "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
        "default_precision": "bf16",
        "default_window_sec": 60,
        "mode": "video_sequence",
        "max_pixels": 384 * 384
    },
    "qwen3.5-4b": {
        "name": "Qwen3.5-4B",
        "hf_id": "Qwen/Qwen3.5-4B",
        "default_precision": "4bit",
        "default_window_sec": 60,
        "mode": "video_sequence",
        "max_pixels": 256 * 256
    },
    "minicpm-v-4.6": {
        "name": "MiniCPM-V 4.6",
        "hf_id": "openbmb/MiniCPM-V-4.6",
        "default_precision": "4bit",
        "default_window_sec": 60,
        "mode": "video_sequence",
        "max_pixels": 448 * 448
    },
    "moondream2": {
        "name": "Moondream2",
        "hf_id": "vikhyatk/moondream2",
        "default_precision": "bf16",
        "default_window_sec": 60,
        "mode": "contact_sheet",
        "max_pixels": 448 * 448
    },
    "lfm2.5-vl-3b": {
        "name": "LFM2.5-VL-3B",
        "hf_id": "LiquidAI/LFM2.5-VL-3B",
        "default_precision": "4bit",
        "default_window_sec": 30,  # Hardware physical ceiling
        "mode": "video_sequence",
        "max_pixels": 384 * 384
    },
    "qwen2.5-vl-7b": {
        "name": "Qwen2.5-VL-7B",
        "hf_id": "Qwen/Qwen2.5-VL-7B-Instruct",
        "default_precision": "4bit",
        "default_window_sec": 15,  # Hardware physical ceiling
        "mode": "video_sequence",
        "max_pixels": 256 * 256
    },
    "qwen2-vl-2b": {
        "name": "Qwen2-VL-2B",
        "hf_id": "Qwen/Qwen2-VL-2B-Instruct",
        "default_precision": "4bit",
        "default_window_sec": 60,
        "mode": "video_sequence",
        "max_pixels": 256 * 256
    }
}


def create_contact_sheet(images: List[Image.Image], window_sec: int) -> Image.Image:
    """Composites N sampled video frames into a single 2D contact sheet image with burned-in MM:SS."""
    from PIL import ImageDraw
    num_images = len(images)
    if num_images == 0:
        return Image.new("RGB", (448, 448), color=(0, 0, 0))

    cols = int(math.ceil(math.sqrt(num_images)))
    rows = int(math.ceil(num_images / cols))
    tile_w, tile_h = 224, 224
    sheet_w, sheet_h = cols * tile_w, rows * tile_h

    sheet = Image.new("RGB", (sheet_w, sheet_h), color=(0, 0, 0))
    draw = ImageDraw.Draw(sheet)

    for idx, img in enumerate(images):
        c = idx % cols
        r = idx // cols
        x0 = c * tile_w
        y0 = r * tile_h
        resized_tile = img.resize((tile_w, tile_h))
        sheet.paste(resized_tile, (x0, y0))

        sec_offset = int((idx / max(1, num_images - 1)) * window_sec)
        mm = sec_offset // 60
        ss = sec_offset % 60
        ts_label = f"{mm:02d}:{ss:02d}"

        draw.rectangle([x0 + 4, y0 + 4, x0 + 56, y0 + 22], fill="black")
        draw.text((x0 + 8, y0 + 6), ts_label, fill="white")

    return sheet.resize((448, 448))


def decode_chunk_frames_single_pass(
    video_path: str,
    fps: float = 1.0,
    max_size: int = 448
) -> List[Tuple[float, Image.Image]]:
    """
    Decodes entire chunk video sequentially in a single pass at target fps.
    Guarantees 0.000s seek drift by indexing each sampled frame by exact relative PTS.
    Memory footprint: ~199 MB system RAM for a 615s chunk at 448x252 RGB.
    """
    import av
    t0 = time.perf_counter()
    container = av.open(video_path)
    stream = container.streams.video[0]
    time_base = float(stream.time_base)

    frames: List[Tuple[float, Image.Image]] = []
    interval = 1.0 / fps
    next_sample_sec = 0.0
    first_pts_sec = None

    for frame in container.decode(stream):
        raw_pts_sec = float(frame.pts * time_base)
        if first_pts_sec is None:
            first_pts_sec = raw_pts_sec

        rel_pts_sec = raw_pts_sec - first_pts_sec
        if rel_pts_sec >= next_sample_sec:
            img = frame.to_image()
            if max(img.size) > max_size:
                img.thumbnail((max_size, max_size), Image.Resampling.BICUBIC)
            frames.append((round(rel_pts_sec, 3), img))
            next_sample_sec += interval

    container.close()
    dur = time.perf_counter() - t0
    print(f"[DECODER] Single-pass decoded {len(frames)} frames from {Path(video_path).name} in {dur:.2f}s ({len(frames)/max(dur,0.001):.1f} fps)")
    return frames


def burn_timestamp_overlay(
    image: Image.Image,
    sec_in_window: float
) -> Image.Image:
    """
    Burns a high-contrast [00:SS] digital clock badge in the top-left corner of the frame.
    Gives discrete frame sequences an explicit visual timecode, eliminating minute-vs-second confusion.
    """
    from PIL import ImageDraw, ImageFont

    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)

    mm = int(sec_in_window // 60)
    ss = int(round(sec_in_window % 60))
    ts_str = f"[{mm:02d}:{ss:02d}]"

    try:
        font = ImageFont.load_default(size=14)
        bbox = font.getbbox(ts_str)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
    except Exception:
        font = ImageFont.load_default()
        text_w = 45
        text_h = 12

    pad = 4
    x0, y0 = 6, 6
    x1, y1 = x0 + text_w + pad * 2, y0 + text_h + pad * 2

    draw.rectangle([x0, y0, x1, y1], fill=(0, 0, 0), outline=(255, 255, 0), width=1)
    draw.text((x0 + pad, y0 + pad - 1), ts_str, fill=(255, 255, 255), font=font)
    return img_copy


def build_window_prompt(window_sec: int, spotting_only: bool = False) -> str:
    """Builds prompt instructing MM:SS timestamps matching burned-in clock and emphasizing routine actions."""
    clip_end_ts = f"{window_sec // 60:02d}:{window_sec % 60:02d}"
    expected_density = "0 to 1 official events" if window_sec <= 30 else "0 to 2 official events"
    
    if spotting_only:
        schema_block = (
            f"Output ONLY a valid JSON array matching this schema (Spotting-Only mode: omit commentary):\n"
            f'[\n'
            f'  {{\n'
            f'    "timestamp": "00:SS",\n'
            f'    "label": "Corner",\n'
            f'    "confidence": 0.85,\n'
            f'    "visibility": "visible"\n'
            f'  }}\n'
            f']\n'
        )
    else:
        schema_block = (
            f"Output ONLY a valid JSON array matching this schema:\n"
            f'[\n'
            f'  {{\n'
            f'    "timestamp": "00:SS",\n'
            f'    "commentary": "Concise observational description...",\n'
            f'    "label": "Corner",\n'
            f'    "confidence": 0.85,\n'
            f'    "visibility": "visible"\n'
            f'  }}\n'
            f']\n'
        )

    prompt = (
        f"{SYSTEM_INSTRUCTION}\n\n"
        f"Video Window Context:\n"
        f"- Length: {window_sec} seconds (00:00 to {clip_end_ts}).\n"
        f"- Visual Clock: Each video frame has a digital clock '[00:SS]' burned in the top-left corner (from [00:00] to [{clip_end_ts}]).\n"
        f"- Task: Identify discrete, official soccer actions occurring in this window. Use the burned-in clock '[00:SS]' to anchor the exact timestamp for each event. Do NOT guess match minutes or use times beyond {clip_end_ts}.\n"
        f"- CRITICAL SPARSITY & PRECISION RULES:\n"
        f"  1. Most {window_sec}-second windows contain only {expected_density}. If ONLY ordinary midfield passing, dribbling, or possession occurs, RETURN AN EMPTY LIST [].\n"
        f"  2. Do NOT log continuous play, normal passes, or player runs as 'Clearance' or 'Goal'. Only log an event when you visually confirm one of the official criteria:\n"
        f"     * 'Kick-off': Match start or restart from the center circle.\n"
        f"     * 'Ball out of play': Ball completely crosses the sideline/touchline or endline.\n"
        f"     * 'Throw-in': Player throws the ball in from the sideline to restart play.\n"
        f"     * 'Corner': Ball crossed endline off defending team; restarted from corner arc.\n"
        f"     * 'Shots on target' / 'Shots off target': An intentional shot aimed at the opponent goal.\n"
        f"     * 'Goal': Ball completely crosses the goal line between the posts into the net.\n"
        f"     * 'Foul': Referee blows whistle and signals an infraction.\n"
        f"     * 'Clearance': Defender forcibly kicks/heads the ball far away from an active attacking threat in the penalty box.\n"
        f"  3. Do NOT emit duplicate or consecutive events within 5 seconds.\n"
        f"  4. Assign strictly ONE official class per event in 'label'. NEVER output comma-separated classes.\n\n"
        f"{schema_block}"
        f'If no official stoppage, boundary crossing, or shot occurs in this window, return [].'
    )
    return prompt


def normalize_soccernet_label(lbl_raw: str) -> Optional[str]:
    import re
    from processing.unified_schema import SOCCERNET_CLASSES
    if not lbl_raw or not isinstance(lbl_raw, str):
        return None
    first_token = lbl_raw.split(",")[0].strip().rstrip(".")
    lbl_clean = first_token.lower()
    lbl_alphanumeric = re.sub(r"[^a-z0-9]", "", lbl_clean)

    # 1. Exact case-insensitive match
    for c in SOCCERNET_CLASSES:
        if c.lower() == lbl_clean:
            return c

    # 2. Alphanumeric match (handles PascalCase/camelCase e.g. "BallOutOfPlay" -> "balloutofplay")
    for c in SOCCERNET_CLASSES:
        c_alphanumeric = re.sub(r"[^a-z0-9]", "", c.lower())
        if c_alphanumeric == lbl_alphanumeric:
            return c

    # 3. Plural stripping & substring match
    for c in SOCCERNET_CLASSES:
        if c.lower().rstrip("s") == lbl_clean.rstrip("s"):
            return c
        c_alphanumeric = re.sub(r"[^a-z0-9]", "", c.lower())
        if lbl_alphanumeric and (lbl_alphanumeric in c_alphanumeric or c_alphanumeric in lbl_alphanumeric):
            return c
    return None


def parse_model_json_output(
    raw_text: str,
    window_sec: int,
    spotting_only: bool = False
) -> Tuple[List[Any], Dict[str, Any]]:
    """
    Extracts and validates JSON event array from model output with multi-strategy recovery
    (JSON Array -> JSON Objects Regex Fallback -> Markdown Table Fallback) and detailed diagnostic telemetry.
    Supports spotting_only ablation mode where commentary is omitted.
    """
    import re
    diag = {
        "raw_length": len(raw_text),
        "strategy_used": "none",
        "total_candidates_found": 0,
        "valid_kept": 0,
        "out_of_bounds_drops": 0,
        "invalid_enum_drops": 0,
        "syntax_error": False,
        "empty_output": len(raw_text.strip()) == 0,
        "dropped_timestamps": [],
        "dropped_labels": []
    }
    if diag["empty_output"]:
        return [], diag

    valid_events: List[Any] = []

    def process_item(item: dict):
        if not isinstance(item, dict):
            return
        diag["total_candidates_found"] += 1
        item.setdefault("confidence", 0.85)
        item.setdefault("visibility", "visible")
        if spotting_only:
            item.setdefault("commentary", "")
        else:
            item.setdefault("commentary", "Observational soccer event")

        raw_lbl = str(item.get("label", ""))
        norm_lbl = normalize_soccernet_label(raw_lbl)
        if not norm_lbl:
            diag["invalid_enum_drops"] += 1
            diag["dropped_labels"].append(raw_lbl)
            return
        item["label"] = norm_lbl

        raw_ts = str(item.get("timestamp", "")).strip().strip("[]\"'")
        parts = raw_ts.split(":")
        if len(parts) == 3 and parts[0].isdigit() and parts[1].isdigit() and parts[2].split(".")[0].isdigit():
            mm, ss = int(parts[1]), int(float(parts[2]))
        elif len(parts) == 2 and parts[0].isdigit() and parts[1].split(".")[0].isdigit():
            mm, ss = int(parts[0]), int(float(parts[1]))
        elif len(parts) == 1 and raw_ts.replace(".", "", 1).isdigit():
            tot = int(float(raw_ts))
            mm, ss = tot // 60, tot % 60
        else:
            diag["out_of_bounds_drops"] += 1
            diag["dropped_timestamps"].append(raw_ts)
            return
        item["timestamp"] = f"{mm:02d}:{ss:02d}"
        sec = mm * 60.0 + ss
        # Layer 2 bounds validation: window duration + 2s slack
        if sec < 0.0 or sec > float(window_sec + 2.0):
            diag["out_of_bounds_drops"] += 1
            diag["dropped_timestamps"].append(item["timestamp"])
            return

        try:
            ev = SpottedEventMinimal(**item) if spotting_only else SpottedEvent(**item)
            valid_events.append(ev)
        except Exception:
            pass

    cleaned = raw_text.strip()
    if "```json" in cleaned:
        cleaned = cleaned.split("```json")[1].split("```")[0].strip()
    elif "```" in cleaned:
        cleaned = cleaned.split("```")[1].split("```")[0].strip()

    # Strategy 1: Top-Level JSON Array Parse
    array_match = re.search(r"\[\s*\{.*\}\s*\]", cleaned, re.DOTALL)
    if array_match:
        try:
            arr = json.loads(array_match.group(0))
            if isinstance(arr, list):
                diag["strategy_used"] = "json_array"
                for it in arr:
                    process_item(it)
        except Exception:
            diag["syntax_error"] = True

    # Strategy 2: Individual JSON Objects Regex Fallback
    if len(valid_events) == 0:
        obj_matches = re.findall(r"\{[^{}]*\}", cleaned, re.DOTALL)
        for om in obj_matches:
            sanitized = re.sub(r",\s*\}", "}", om.strip())
            try:
                it = json.loads(sanitized)
                if isinstance(it, dict) and ("timestamp" in it or "label" in it):
                    diag["strategy_used"] = "json_objects"
                    process_item(it)
            except Exception:
                pass

    # Strategy 3: Markdown Table Recovery Fallback
    if len(valid_events) == 0 and "|" in cleaned:
        lines = [l.strip() for l in cleaned.splitlines() if l.strip().startswith("|") and not l.strip().startswith("|---")]
        for line in lines:
            cols = [c.strip() for c in line.split("|")[1:-1]]
            if spotting_only and len(cols) >= 2:
                ts_raw, lbl = cols[0], cols[1]
                if "timestamp" in ts_raw.lower() or "label" in lbl.lower():
                    continue
                diag["strategy_used"] = "markdown_table"
                process_item({"timestamp": ts_raw, "label": lbl})
            elif len(cols) >= 3:
                ts_raw, comm, lbl = cols[0], cols[1], cols[2]
                if "timestamp" in ts_raw.lower() or "label" in lbl.lower():
                    continue
                diag["strategy_used"] = "markdown_table"
                process_item({"timestamp": ts_raw, "commentary": comm, "label": lbl})

    diag["valid_kept"] = len(valid_events)
    return valid_events, diag


class LocalVLMRunner:
    """Manages model loading and GPU inference across sliding windows."""
    def __init__(self, model_key: str, precision: str = "bf16", dry_run: bool = False, spotting_only: bool = False):
        self.model_key = model_key
        self.cfg = LOCAL_MODEL_CONFIGS[model_key]
        self.precision = precision
        self.dry_run = dry_run
        self.spotting_only = spotting_only
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.device = "cuda:0" if not dry_run else "cpu"
        self._is_loaded = False

    def ensure_model_loaded(self):
        """Loads model weights into memory only when inference is actually needed."""
        if not self.dry_run and not self._is_loaded:
            self._load_model()
            self._is_loaded = True

    def _load_model(self):
        import torch
        from transformers import AutoProcessor, AutoModelForImageTextToText, AutoModel, AutoTokenizer, AutoModelForCausalLM

        hf_id = self.cfg["hf_id"]
        torch_dtype = torch.bfloat16 if self.precision in ("bf16", "4bit", "nf4") else torch.float16

        load_kwargs: Dict[str, Any] = {
            "torch_dtype": torch_dtype,
            "device_map": "auto",
            "trust_remote_code": True
        }

        if "4bit" in self.precision or "nf4" in self.precision:
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                llm_int8_enable_fp32_cpu_offload=True
            )
            load_kwargs["device_map"] = {"": "cuda:0"}

        print(f"[LOADER] Loading {self.cfg['name']} ({self.precision}) from {hf_id}...")
        t0 = time.perf_counter()

        try:
            self.model = AutoModelForImageTextToText.from_pretrained(hf_id, **load_kwargs).eval()
        except Exception:
            try:
                self.model = AutoModelForCausalLM.from_pretrained(hf_id, **load_kwargs).eval()
            except Exception:
                self.model = AutoModel.from_pretrained(hf_id, **load_kwargs).eval()

        if self.model_key == "moondream2":
            self.tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
            self.processor = None
        else:
            self.processor = AutoProcessor.from_pretrained(hf_id, trust_remote_code=True)

        if self.model_key in ("qwen3.5-4b", "qwen2.5-vl-7b", "qwen2-vl-2b"):
            if hasattr(self.processor, "image_processor"):
                self.processor.image_processor.max_pixels = self.cfg["max_pixels"]
                self.processor.image_processor.min_pixels = 128 * 128
        elif self.model_key == "smolvlm2-2.2b":
            if hasattr(self.processor, "image_processor"):
                self.processor.image_processor.do_image_splitting = False
                self.processor.image_processor.size = {"longest_edge": 384}

        load_sec = time.perf_counter() - t0
        vram_gib = torch.cuda.memory_allocated() / (1024 ** 3)
        print(f"[LOADER] Loaded {self.cfg['name']} in {load_sec:.1f}s | VRAM Allocated: {vram_gib:.2f} GiB")

    def infer_window(self, frames: List[Image.Image], window_sec: int) -> Tuple[List[Any], float, str, Dict[str, Any]]:
        """Runs inference on a single window of frames."""
        self.ensure_model_loaded()
        t0 = time.perf_counter()
        prompt_text = build_window_prompt(window_sec, spotting_only=self.spotting_only)

        if self.dry_run:
            time.sleep(0.05)
            if self.spotting_only:
                mock_event = SpottedEventMinimal(
                    timestamp="00:15",
                    label="Corner",
                    confidence=0.88,
                    visibility="visible"
                )
            else:
                mock_event = SpottedEvent(
                    timestamp="00:15",
                    commentary="Midfielder advances down the wing and plays a cross into the penalty area.",
                    label="Corner",
                    confidence=0.88,
                    visibility="visible"
                )
            mock_diag = {
                "raw_length": 120,
                "strategy_used": "dry_run",
                "is_dry_run": True,
                "total_candidates_found": 1,
                "valid_kept": 1,
                "out_of_bounds_drops": 0,
                "invalid_enum_drops": 0,
                "syntax_error": False,
                "empty_output": False,
                "dropped_timestamps": [],
                "dropped_labels": []
            }
            return [mock_event], round(time.perf_counter() - t0, 3), mock_event.model_dump_json(), mock_diag

        import torch
        gen_text = ""

        if self.cfg["mode"] == "contact_sheet":
            contact_sheet = create_contact_sheet(frames, window_sec=window_sec)
            with torch.inference_mode():
                gen_text = self.model.answer_question(contact_sheet, prompt_text, tokenizer=self.tokenizer)
        else:
            apply_kwargs = dict(tokenize=False, add_generation_prompt=True)
            if self.model_key == "qwen3.5-4b":
                apply_kwargs["enable_thinking"] = False

            messages = [{
                "role": "user",
                "content": [{"type": "image", "image": img} for img in frames] + [{"type": "text", "text": prompt_text}]
            }]
            prompt_formatted = self.processor.apply_chat_template(messages, **apply_kwargs)
            inputs = self.processor(text=[prompt_formatted], images=[frames], return_tensors="pt").to(self.device)

            max_tokens = 384 if self.spotting_only else 768
            with torch.inference_mode():
                outputs = self.model.generate(**inputs, max_new_tokens=max_tokens)

            in_len = inputs["input_ids"].shape[1] if "input_ids" in inputs else 0
            gen_tokens = outputs[0][in_len:] if in_len > 0 and outputs.shape[1] > in_len else outputs[0]
            gen_text = self.processor.decode(gen_tokens, skip_special_tokens=True)

            del inputs, outputs, gen_tokens
            if "prompt_formatted" in locals():
                del prompt_formatted
            if "messages" in locals():
                del messages

        import gc
        gc.collect()
        torch.cuda.empty_cache()

        elapsed = round(time.perf_counter() - t0, 3)
        events, diag = parse_model_json_output(gen_text, window_sec, spotting_only=self.spotting_only)
        return events, elapsed, gen_text, diag


def run_benchmark_on_chunks(
    manifest_data: Dict[str, Any],
    chunks_dir: Path,
    runner: LocalVLMRunner,
    window_sec: int,
    stride_sec: int,
    target_chunk_idx: Optional[int] = None,
    max_windows_per_chunk: Optional[int] = None,
    output_dir: Optional[Path] = None,
    no_cache: bool = False
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Decodes target chunks sequentially in a single pass into memory, then slices sliding
    windows with 0 seek drift. Clamps tail windows and applies 3-tier offset math.
    Supports caching completed chunks to avoid redundant GPU re-inference.
    Captures complete per-window diagnostic audit telemetry.
    """
    chunks = manifest_data.get("chunks", [])
    if target_chunk_idx is not None:
        chunks = [c for c in chunks if c["chunk_index"] == target_chunk_idx]
        if not chunks:
            raise ValueError(f"Chunk index {target_chunk_idx} not found in manifest!")

    all_chunk_results = []
    total_windows = 0
    total_events_spotted = 0
    total_inference_time = 0.0
    all_raw_outputs: List[Dict[str, Any]] = []
    all_window_diags: List[Dict[str, Any]] = []

    # Pre-scan candidate cache files before any chunk writes to avoid overwriting multi-chunk caches
    cached_chunks_map: Dict[int, Tuple[str, List[Dict], List[Dict]]] = {}
    if not no_cache:
        candidate_cache_dirs = []
        if runner.spotting_only:
            # Spotting-only ablation must NEVER reuse joint-CoT caches
            if output_dir:
                candidate_cache_dirs.append(output_dir)
        else:
            if output_dir:
                candidate_cache_dirs.append(output_dir)
                candidate_cache_dirs.append(output_dir.parent / f"{runner.model_key}_{runner.precision}_W{window_sec}")
            else:
                candidate_cache_dirs.append(Path("logs/local_eval") / f"{runner.model_key}_{runner.precision}_W{window_sec}")

        for c_dir in candidate_cache_dirs:
            if "dryrun" in str(c_dir).lower() and not runner.dry_run:
                continue
            raw_file = c_dir / "raw_window_outputs.json"
            telem_file = c_dir / "telemetry.json"
            if raw_file.exists():
                try:
                    with open(raw_file, "r", encoding="utf-8") as rf:
                        raw_json = json.load(rf)
                    telem_json = {}
                    if telem_file.exists():
                        with open(telem_file, "r", encoding="utf-8") as tf:
                            telem_json = json.load(tf)

                    for c_cand in chunks:
                        c_idx = c_cand["chunk_index"]
                        if c_idx in cached_chunks_map:
                            continue
                        c_raw = [r for r in raw_json if r.get("chunk_index") == c_idx]
                        c_diags = [w for w in telem_json.get("window_diagnostics", []) if w.get("chunk_index") == c_idx]
                        if c_raw:
                            # Cache Hygiene Guardrail: Never reuse mock dry-run data in a live GPU run!
                            is_mock = any(r.get("is_dry_run") for r in c_raw) or any(w.get("strategy_used") == "dry_run" for w in c_diags)
                            if not runner.dry_run and is_mock:
                                print(f"  [CACHE REJECT] Skipping mock dry-run cached outputs for Chunk {c_idx} from '{c_dir.name}'.")
                                continue
                            cached_chunks_map[c_idx] = (c_dir.name, c_raw, c_diags)
                except Exception:
                    pass

    for chunk in chunks:
        chunk_idx = chunk["chunk_index"]
        filename = chunk["filename"]
        chunk_path = chunks_dir / filename
        duration_sec = float(chunk.get("duration_sec", 615.0))

        if not chunk_path.exists():
            raise FileNotFoundError(f"Video chunk not found: {chunk_path}")

        print(f"\n--- Processing Chunk {chunk_idx}: {filename} (Duration: {duration_sec:.1f}s) ---")

        # 1. Window Sliding Schedule
        win_starts = []
        curr = 0.0
        while curr < duration_sec:
            win_starts.append(curr)
            curr += stride_sec
            if max_windows_per_chunk and len(win_starts) >= max_windows_per_chunk:
                break

        expected_valid_starts = [st for st in win_starts if min(float(window_sec), duration_sec - st) >= 5.0]

        # 2. Check for cached predictions
        cached_chunk_data = None
        if not no_cache and chunk_idx in cached_chunks_map:
            c_name, c_raw, c_diags = cached_chunks_map[chunk_idx]
            if len(c_raw) == len(expected_valid_starts):
                if len(c_diags) != len(c_raw):
                    c_diags = []
                    for w_idx, r_it in enumerate(c_raw):
                        st = win_starts[w_idx]
                        eff_dur = min(float(window_sec), duration_sec - st)
                        evts, d = parse_model_json_output(r_it.get("raw_output", ""), int(eff_dur), spotting_only=runner.spotting_only)
                        c_diags.append({
                            "chunk_index": chunk_idx,
                            "window_index": w_idx,
                            "start_sec": round(st, 2),
                            "end_sec": round(st + eff_dur, 2),
                            "duration_sec": int(eff_dur),
                            "elapsed_sec": 25.0,
                            "status": "SUCCESS" if len(evts) > 0 else ("EMPTY" if d["empty_output"] else "DROPPED"),
                            "strategy_used": d["strategy_used"],
                            "candidates_found": d["total_candidates_found"],
                            "valid_kept": len(evts),
                            "out_of_bounds_drops": d["out_of_bounds_drops"],
                            "invalid_enum_drops": d["invalid_enum_drops"],
                            "syntax_error": d["syntax_error"],
                            "dropped_timestamps": d["dropped_timestamps"],
                            "dropped_labels": d["dropped_labels"]
                        })
                cached_chunk_data = (c_name, c_raw, c_diags)

        if cached_chunk_data is not None:
            c_dir, c_raw, c_diags = cached_chunk_data
            print(f"  [CACHE HIT] Found {len(c_raw)} valid cached windows for Chunk {chunk_idx} in '{c_dir}'.")
            print(f"              Reusing cached outputs; skipping single-pass frame decode and GPU inference!")

            chunk_events: List[Dict[str, Any]] = []
            chunk_window_diags: List[Dict[str, Any]] = []

            for w_i, (start_t, r_item, d_item) in enumerate(zip(win_starts, c_raw, c_diags)):
                eff_window_dur = min(float(window_sec), duration_sec - start_t)
                if eff_window_dur < 5.0:
                    continue
                events, diag = parse_model_json_output(r_item.get("raw_output", ""), int(eff_window_dur), spotting_only=runner.spotting_only)
                total_windows += 1
                total_inference_time += float(d_item.get("elapsed_sec", 0.0))
                chunk_window_diags.append(d_item)
                all_window_diags.append(d_item)
                all_raw_outputs.append(r_item)

                for ev in events:
                    local_sec = parse_mm_ss_to_seconds(ev.timestamp)
                    chunk_sec = start_t + local_sec
                    c_mm = int(chunk_sec // 60)
                    c_ss = int(round(chunk_sec % 60))
                    ev_dict = ev.model_dump()
                    ev_dict["timestamp"] = f"{c_mm:02d}:{c_ss:02d}"
                    chunk_events.append(ev_dict)

            # Print per-chunk diagnostic audit table
            print("\n" + "=" * 110)
            print(f"  CHUNK {chunk_idx} SLIDING WINDOW DIAGNOSTIC AUDIT ({len(chunk_window_diags)} Windows) [CACHED]")
            print("=" * 110)
            print(f"{'Win':<4} | {'Slice (s)':<16} | {'Status':<10} | {'Parsed':<7} | {'Kept':<6} | {'OOB Drop':<9} | {'Enum Drop':<10} | {'Strategy':<14} | {'Latency':<8}")
            print("-" * 110)
            for wd in chunk_window_diags:
                sl = f"{wd['start_sec']:5.1f} - {wd['end_sec']:5.1f}"
                print(f"{wd['window_index']+1:<4} | {sl:<16} | {wd['status']:<10} | {wd['candidates_found']:<7} | {wd['valid_kept']:<6} | {wd['out_of_bounds_drops']:<9} | {wd['invalid_enum_drops']:<10} | {wd['strategy_used']:<14} | {wd['elapsed_sec']:5.2f}s")
            print("=" * 110 + "\n")

            # Class-aware NMS across the windows inside this chunk
            temp_preds = []
            for ev in chunk_events:
                sec = parse_mm_ss_to_seconds(ev["timestamp"])
                temp_preds.append({
                    "half": chunk["half"],
                    "label": ev["label"],
                    "position": int(round(sec * 1000.0)),
                    "confidence": ev["confidence"],
                    "commentary": ev["commentary"],
                    "visibility": ev["visibility"],
                    "timestamp": ev["timestamp"]
                })

            deduped_temp = class_aware_nms(temp_preds, radius_ms=3000)
            final_chunk_events = []
            for d in deduped_temp:
                final_chunk_events.append({
                    "timestamp": d["timestamp"],
                    "commentary": d["commentary"],
                    "label": d["label"],
                    "confidence": d["confidence"],
                    "visibility": d["visibility"]
                })

            total_events_spotted += len(final_chunk_events)
            print(f"  -> Chunk {chunk_idx} Kept Events (Post-NMS Across {len(win_starts)} Windows): {len(final_chunk_events)}")

            all_chunk_results.append({
                "chunk_index": chunk_idx,
                "filename": filename,
                "half": chunk["half"],
                "events": final_chunk_events
            })

            if output_dir:
                try:
                    with open(output_dir / "raw_window_outputs.json", "w", encoding="utf-8") as f:
                        json.dump(all_raw_outputs, f, indent=2)
                    interim_telem = {
                        "model": runner.model_key,
                        "precision": runner.precision,
                        "total_windows": total_windows,
                        "total_events_spotted": total_events_spotted,
                        "total_inference_time_sec": round(total_inference_time, 2),
                        "avg_window_latency_sec": round(total_inference_time / max(1, total_windows), 3),
                        "window_diagnostics": all_window_diags
                    }
                    with open(output_dir / "telemetry.json", "w", encoding="utf-8") as f:
                        json.dump(interim_telem, f, indent=2)
                except Exception:
                    pass

            continue

        # 3. Fresh Zero-Drift Single-Pass Sequential In-Memory Decode & Inference
        runner.ensure_model_loaded()
        if runner.dry_run:
            chunk_frames = [(float(i), Image.new("RGB", (448, 448))) for i in range(int(duration_sec) + 1)]
        else:
            chunk_frames = decode_chunk_frames_single_pass(str(chunk_path), fps=1.0)

        chunk_events: List[Dict[str, Any]] = []
        chunk_window_diags: List[Dict[str, Any]] = []

        for w_i, start_t in enumerate(win_starts):
            # Tail-window duration clamping
            eff_window_dur = min(float(window_sec), duration_sec - start_t)
            if eff_window_dur < 5.0:
                continue

            end_t = start_t + eff_window_dur
            print(f"  [Window {w_i+1:02d}/{len(win_starts):02d}] Slice: {start_t:5.1f}s - {end_t:5.1f}s ({int(eff_window_dur)}s)...", end="", flush=True)

            # Slice frames directly from the in-memory buffer (zero seek drift)
            win_raw_frames = [f for f in chunk_frames if start_t <= f[0] < end_t]
            if not win_raw_frames:
                # Fallback to direct indexing if exact timestamp boundaries differ slightly
                idx_start = min(int(round(start_t)), len(chunk_frames) - 1)
                idx_end = min(int(round(end_t)), len(chunk_frames))
                win_raw_frames = chunk_frames[idx_start:idx_end]

            # For contact sheet: pass raw tiles; for video sequences: burn explicit visual clock overlay [00:SS]
            if runner.cfg["mode"] == "contact_sheet":
                win_frame_objs = [f[1] for f in win_raw_frames]
            else:
                win_frame_objs = [
                    burn_timestamp_overlay(f[1], sec_in_window=f[0] - start_t)
                    for f in win_raw_frames
                ]

            events, elapsed, raw_text, diag = runner.infer_window(win_frame_objs, int(eff_window_dur))
            total_windows += 1
            total_inference_time += elapsed

            diag_entry = {
                "chunk_index": chunk_idx,
                "window_index": w_i,
                "start_sec": round(start_t, 2),
                "end_sec": round(end_t, 2),
                "duration_sec": int(eff_window_dur),
                "elapsed_sec": elapsed,
                "status": "SUCCESS" if len(events) > 0 else ("EMPTY" if diag["empty_output"] else "DROPPED"),
                "strategy_used": diag["strategy_used"],
                "candidates_found": diag["total_candidates_found"],
                "valid_kept": len(events),
                "out_of_bounds_drops": diag["out_of_bounds_drops"],
                "invalid_enum_drops": diag["invalid_enum_drops"],
                "syntax_error": diag["syntax_error"],
                "dropped_timestamps": diag["dropped_timestamps"],
                "dropped_labels": diag["dropped_labels"],
                "is_dry_run": runner.dry_run
            }
            chunk_window_diags.append(diag_entry)
            all_window_diags.append(diag_entry)
            all_raw_outputs.append({
                "chunk_index": chunk_idx,
                "window_index": w_i,
                "slice": f"{start_t:.1f}s - {end_t:.1f}s",
                "raw_output": raw_text,
                "is_dry_run": runner.dry_run
            })

            # 3-Tier Offset Translation: window relative timestamp -> chunk relative timestamp
            for ev in events:
                local_sec = parse_mm_ss_to_seconds(ev.timestamp)
                chunk_sec = start_t + local_sec
                c_mm = int(chunk_sec // 60)
                c_ss = int(round(chunk_sec % 60))
                ev_dict = ev.model_dump()
                ev_dict["timestamp"] = f"{c_mm:02d}:{c_ss:02d}"
                chunk_events.append(ev_dict)

            drop_info = f"[Found: {diag['total_candidates_found']}, Kept: {len(events)}"
            if diag["out_of_bounds_drops"] > 0:
                drop_info += f", OOB Drops: {diag['out_of_bounds_drops']}"
            if diag["invalid_enum_drops"] > 0:
                drop_info += f", Enum Drops: {diag['invalid_enum_drops']}"
            drop_info += f", Strat: {diag['strategy_used']}]"
            print(f" Spotted {len(events)} event(s) in {elapsed:.2f}s {drop_info}")

        # Print per-chunk diagnostic audit table
        print("\n" + "=" * 110)
        print(f"  CHUNK {chunk_idx} SLIDING WINDOW DIAGNOSTIC AUDIT ({len(chunk_window_diags)} Windows)")
        print("=" * 110)
        print(f"{'Win':<4} | {'Slice (s)':<16} | {'Status':<10} | {'Parsed':<7} | {'Kept':<6} | {'OOB Drop':<9} | {'Enum Drop':<10} | {'Strategy':<14} | {'Latency':<8}")
        print("-" * 110)
        for wd in chunk_window_diags:
            sl = f"{wd['start_sec']:5.1f} - {wd['end_sec']:5.1f}"
            print(f"{wd['window_index']+1:<4} | {sl:<16} | {wd['status']:<10} | {wd['candidates_found']:<7} | {wd['valid_kept']:<6} | {wd['out_of_bounds_drops']:<9} | {wd['invalid_enum_drops']:<10} | {wd['strategy_used']:<14} | {wd['elapsed_sec']:5.2f}s")
        print("=" * 110 + "\n")

        # 4. Class-aware NMS across the 14-window overlap zones inside this chunk
        temp_preds = []
        for ev in chunk_events:
            sec = parse_mm_ss_to_seconds(ev["timestamp"])
            temp_preds.append({
                "half": chunk["half"],
                "label": ev["label"],
                "position": int(round(sec * 1000.0)),
                "confidence": ev["confidence"],
                "commentary": ev["commentary"],
                "visibility": ev["visibility"],
                "timestamp": ev["timestamp"]
            })

        deduped_temp = class_aware_nms(temp_preds, radius_ms=3000)
        final_chunk_events = []
        for d in deduped_temp:
            final_chunk_events.append({
                "timestamp": d["timestamp"],
                "commentary": d["commentary"],
                "label": d["label"],
                "confidence": d["confidence"],
                "visibility": d["visibility"]
            })

        total_events_spotted += len(final_chunk_events)
        print(f"  -> Chunk {chunk_idx} Kept Events (Post-NMS Across {len(win_starts)} Windows): {len(final_chunk_events)}")

        all_chunk_results.append({
            "chunk_index": chunk_idx,
            "filename": filename,
            "half": chunk["half"],
            "events": final_chunk_events
        })

        # Free chunk frame memory and empty cache between chunks
        if 'chunk_frames' in locals():
            del chunk_frames
        import gc
        gc.collect()
        if not runner.dry_run:
            import torch
            torch.cuda.empty_cache()

        if output_dir:
            try:
                with open(output_dir / "raw_window_outputs.json", "w", encoding="utf-8") as f:
                    json.dump(all_raw_outputs, f, indent=2)
                interim_telem = {
                    "model": runner.model_key,
                    "precision": runner.precision,
                    "total_windows": total_windows,
                    "total_events_spotted": total_events_spotted,
                    "total_inference_time_sec": round(total_inference_time, 2),
                    "avg_window_latency_sec": round(total_inference_time / max(1, total_windows), 3),
                    "window_diagnostics": all_window_diags
                }
                with open(output_dir / "telemetry.json", "w", encoding="utf-8") as f:
                    json.dump(interim_telem, f, indent=2)
            except Exception:
                pass

    telemetry = {
        "model": runner.model_key,
        "precision": runner.precision,
        "spotting_only": runner.spotting_only,
        "total_windows": total_windows,
        "total_events_spotted": total_events_spotted,
        "total_inference_time_sec": round(total_inference_time, 2),
        "avg_window_latency_sec": round(total_inference_time / max(1, total_windows), 3)
    }

    return all_chunk_results, telemetry, all_raw_outputs, all_window_diags


def print_qualitative_event_comparison(
    predicted_events: List[Dict[str, Any]],
    ground_truth_events: List[Dict[str, Any]],
    tolerance_sec: float = 5.0
) -> None:
    """Prints a detailed side-by-side comparison of model predictions against ground truth."""
    print("\n" + "=" * 95)
    print(f"  QUALITATIVE GROUND-TRUTH SPOT-CHECK COMPARISON (Tolerance: ±{tolerance_sec:.1f}s)")
    print("=" * 95)
    print(f"{'GT Time':<10} | {'GT Label':<18} | {'Status':<10} | {'Pred Time':<10} | {'Pred Label':<18} | {'Error (s)':<9}")
    print("-" * 95)

    used_pred_indices = set()
    matched_count = 0

    for gt in ground_truth_events:
        gt_pos_ms = int(gt.get("position", 0))
        gt_sec = gt_pos_ms / 1000.0
        gt_time = gt.get("gameTime", f"1 - {int(gt_sec//60):02d}:{int(gt_sec%60):02d}")
        gt_lbl = gt.get("label", "")

        best_match_idx = None
        min_diff_sec = float("inf")

        for p_idx, pred in enumerate(predicted_events):
            if p_idx in used_pred_indices:
                continue
            pred_lbl = pred.get("label", "")
            if pred_lbl == gt_lbl:
                pred_pos_ms = int(pred.get("position", 0))
                diff_sec = abs(pred_pos_ms - gt_pos_ms) / 1000.0
                if diff_sec <= tolerance_sec and diff_sec < min_diff_sec:
                    min_diff_sec = diff_sec
                    best_match_idx = p_idx

        if best_match_idx is not None:
            used_pred_indices.add(best_match_idx)
            matched_count += 1
            p = predicted_events[best_match_idx]
            p_pos_ms = int(p.get("position", 0))
            p_sec = p_pos_ms / 1000.0
            p_time = p.get("gameTime", f"1 - {int(p_sec//60):02d}:{int(p_sec%60):02d}")
            print(f"{gt_time:<10} | {gt_lbl:<18} | {'HIT [PASS]':<10} | {p_time:<10} | {p['label']:<18} | {min_diff_sec:+5.2f}s")
        else:
            print(f"{gt_time:<10} | {gt_lbl:<18} | {'MISS':<10} | {'--':<10} | {'--':<18} | {'--':<9}")

    # False Positives
    unmatched_preds = [p for i, p in enumerate(predicted_events) if i not in used_pred_indices]
    for p in unmatched_preds[:10]:
        p_pos_ms = int(p.get("position", 0))
        p_sec = p_pos_ms / 1000.0
        p_time = p.get("gameTime", f"1 - {int(p_sec//60):02d}:{int(p_sec%60):02d}")
        print(f"{'--':<10} | {'--':<18} | {'EXTRA (FP)':<10} | {p_time:<10} | {p['label']:<18} | {'--':<9}")

    print("-" * 95)
    recall = (matched_count / max(1, len(ground_truth_events))) * 100
    print(f"Summary: {matched_count}/{len(ground_truth_events)} Ground-Truth Events Detected ({recall:.1f}% Recall within ±{tolerance_sec:.1f}s)")
    print(f"Spurious / Extra Predictions (Potential False Positives): {len(unmatched_preds)}")
    print("=" * 95 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Local Open-Weight Benchmark Runner for SoccerNet V2")
    parser.add_argument("--model", type=str, default="smolvlm2-2.2b", choices=list(LOCAL_MODEL_CONFIGS.keys()), help="Model to evaluate")
    parser.add_argument("--precision", type=str, default=None, choices=["bf16", "4bit", "nf4", "fp16"], help="Model precision")
    parser.add_argument("--window-sec", type=int, default=None, help="Sliding window size in seconds (defaults to model ceiling)")
    parser.add_argument("--stride-sec", type=int, default=None, help="Sliding window stride in seconds (defaults to W - 15s)")
    parser.add_argument("--chunk-index", type=int, default=0, help="Chunk index to evaluate (e.g. 0 for quick test, -1 for all chunks)")
    parser.add_argument("--manifest", type=str, default="gate2_chunks/game_0_half_1_chunks_manifest.json", help="Path to chunk manifest")
    parser.add_argument("--chunks-dir", type=str, default="gate2_chunks", help="Directory containing chunk mp4 files")
    parser.add_argument("--ground-truth", type=str, default="test_eval_ground_truth/england_epl/2014-2015/2015-05-17 - 18-00 Manchester United 1 - 1 Arsenal/Labels-v2.json", help="Ground truth Labels-v2.json")
    parser.add_argument("--output-dir", type=str, default="logs/local_eval", help="Directory to save predictions")
    parser.add_argument("--max-windows", type=int, default=None, help="Max windows to evaluate per chunk (for quick testing)")
    parser.add_argument("--no-cache", action="store_true", help="Do not reuse cached chunk predictions; force fresh GPU inference")
    parser.add_argument("--dry-run", action="store_true", help="Run simulation without loading PyTorch weights or using GPU")
    parser.add_argument("--spotting-only", action="store_true", help="Run spotting-only ablation (RQ3): omit commentary and natural language reasoning before action label")

    args = parser.parse_args()

    model_cfg = LOCAL_MODEL_CONFIGS[args.model]
    precision = args.precision or model_cfg["default_precision"]
    window_sec = args.window_sec or model_cfg["default_window_sec"]
    if args.stride_sec:
        stride_sec = args.stride_sec
    elif window_sec <= 15:
        stride_sec = 10  # 15s window, 10s stride -> 5s overlap
    elif window_sec <= 30:
        stride_sec = 20  # 30s window, 20s stride -> 10s overlap
    else:
        stride_sec = max(15, window_sec - 15)  # 60s window, 45s stride -> 15s overlap

    scope_tag = "" if args.chunk_index >= 0 else "_half1"
    spotting_tag = "_spotting_only" if args.spotting_only else ""
    base_out_dir = Path("logs/local_eval_dryrun") if args.dry_run else Path(args.output_dir)
    output_dir = base_out_dir / f"{args.model}_{precision}_W{window_sec}{spotting_tag}{scope_tag}"
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)

    target_chunk = args.chunk_index if args.chunk_index >= 0 else None

    print("\n" + "=" * 90)
    print(f"  SOCCERNET LOCAL BENCHMARK RUNNER: {model_cfg['name']} ({precision})")
    print(f"  Window Size:       W = {window_sec}s | Stride: S = {stride_sec}s (Overlap: {window_sec - stride_sec}s)")
    print(f"  Target Chunk:      {'All Chunks' if target_chunk is None else f'Chunk {target_chunk}'}")
    print(f"  Spotting-Only Mode:{args.spotting_only}")
    print(f"  Output Directory:  {output_dir}")
    print(f"  Dry-Run Mode:      {args.dry_run}")
    print(f"  Reuse Cache:       {not args.no_cache}")
    print("=" * 90 + "\n")

    # 1. Initialize Runner (Single-load model residency)
    runner = LocalVLMRunner(
        model_key=args.model,
        precision=precision,
        dry_run=args.dry_run,
        spotting_only=args.spotting_only
    )

    # 2. Run Inference across Sliding Windows with Single-Pass Frame Decodes
    chunk_results, telemetry, all_raw_outputs, all_window_diags = run_benchmark_on_chunks(
        manifest_data=manifest_data,
        chunks_dir=Path(args.chunks_dir),
        runner=runner,
        window_sec=window_sec,
        stride_sec=stride_sec,
        target_chunk_idx=target_chunk,
        max_windows_per_chunk=args.max_windows,
        output_dir=output_dir,
        no_cache=args.no_cache
    )

    # 3. Stitch across Match Half using 3-Tier PTS Offsets from Manifest
    url_local = "england_epl/2014-2015/2015-05-17 - 18-00 Manchester United 1 - 1 Arsenal/"
    print("\n[STITCHING] Stitching chunk predictions into full match timeline using PTS offsets...")
    spotting_pred, caption_pred, stitch_telem = stitch_chunks_to_match_predictions(
        chunk_results=chunk_results,
        chunks_manifest=manifest_data,
        url_local=url_local,
        radius_ms=3000,
        overlap_radius_ms=5000,
        class_radii_ms=DEFAULT_CLASS_NMS_RADII_MS
    )

    spotting_out_path = output_dir / "Predictions-v2.json"
    caption_out_path = output_dir / "Predictions-Caption.json"
    raw_outputs_out_path = output_dir / "raw_window_outputs.json"
    telemetry_out_path = output_dir / "telemetry.json"

    with open(spotting_out_path, "w", encoding="utf-8") as f:
        json.dump(spotting_pred, f, indent=2)
    with open(caption_out_path, "w", encoding="utf-8") as f:
        json.dump(caption_pred, f, indent=2)
    with open(raw_outputs_out_path, "w", encoding="utf-8") as f:
        json.dump(all_raw_outputs, f, indent=2)

    full_telem = {
        **telemetry,
        "stitching": stitch_telem,
        "window_diagnostics": all_window_diags,
        "diagnostic_summary": {
            "total_windows": len(all_window_diags),
            "total_candidates_found": sum(w["candidates_found"] for w in all_window_diags),
            "total_valid_kept_pre_nms": sum(w["valid_kept"] for w in all_window_diags),
            "total_out_of_bounds_drops": sum(w["out_of_bounds_drops"] for w in all_window_diags),
            "total_invalid_enum_drops": sum(w["invalid_enum_drops"] for w in all_window_diags),
            "total_syntax_errors": sum(1 for w in all_window_diags if w["syntax_error"]),
            "total_empty_outputs": sum(1 for w in all_window_diags if w["status"] == "EMPTY")
        }
    }
    with open(telemetry_out_path, "w", encoding="utf-8") as f:
        json.dump(full_telem, f, indent=2)

    print(f"[SUCCESS] Exported Action Spotting    -> {spotting_out_path}")
    print(f"[SUCCESS] Exported Dense Captions     -> {caption_out_path}")
    print(f"[SUCCESS] Exported Raw Window Outputs -> {raw_outputs_out_path}")
    print(f"[SUCCESS] Exported Telemetry          -> {telemetry_out_path}")

    # 4. Scope-Masked Evaluation & Qualitative Diagnostics
    gt_path = Path(args.ground_truth)
    if gt_path.exists():
        with open(gt_path, "r", encoding="utf-8") as f:
            gt_full = json.load(f)

        # Build Scope-Masked Ground Truth (prevents un-evaluated chunks/halves from generating artificial false negatives)
        target_half = str(manifest_data.get("half", 1))
        scoped_gt_dir = output_dir / "scoped_ground_truth"
        scoped_game_dir = scoped_gt_dir / url_local.strip("/")
        if os.name == "nt":
            os.makedirs(f"\\\\?\\{scoped_game_dir.resolve()}", exist_ok=True)
        else:
            scoped_game_dir.mkdir(parents=True, exist_ok=True)
        scoped_gt_file = scoped_game_dir / "Labels-v2.json"

        if target_chunk is not None:
            c_meta = next((c for c in manifest_data.get("chunks", []) if c["chunk_index"] == target_chunk), {})
            chunk_start_ms = int(c_meta.get("actual_start_ms", 0))
            chunk_dur_sec = float(c_meta.get("duration_sec", 615.0))
            chunk_end_ms = chunk_start_ms + int(round(chunk_dur_sec * 1000.0))
            chunk_half = str(c_meta.get("half", target_half))

            target_gt_annotations = [
                a for a in gt_full.get("annotations", [])
                if str(a.get("gameTime", "")).startswith(f"{chunk_half} - ") and chunk_start_ms <= int(a.get("position", 0)) <= chunk_end_ms
            ]
            print(f"[SCOPE-MASKING] Filtered ground-truth Labels-v2.json to Chunk {target_chunk}: {len(target_gt_annotations)} active events ({chunk_start_ms}ms to {chunk_end_ms}ms).")
        else:
            # Whole half scope masking (65 active events in Half 1, dynamically bounded to last chunk end)
            chunks = manifest_data.get("chunks", [])
            if chunks:
                last_chunk = max(chunks, key=lambda c: int(c.get("actual_start_ms", 0)))
                last_start_ms = int(last_chunk.get("actual_start_ms", 0))
                last_dur_sec = float(last_chunk.get("duration_sec", 0.0))
                half_dur_ms = last_start_ms + int(round(last_dur_sec * 1000.0))
            else:
                half_dur_ms = int(round(float(manifest_data.get("total_duration_sec", 2700.0)) * 1000.0))

            target_gt_annotations = [
                a for a in gt_full.get("annotations", [])
                if str(a.get("gameTime", "")).startswith(f"{target_half} - ") and 0 <= int(a.get("position", 0)) <= half_dur_ms
            ]
            print(f"[SCOPE-MASKING] Filtered ground-truth Labels-v2.json to Half {target_half}: {len(target_gt_annotations)} active events (0ms to {half_dur_ms}ms).")

        open_gt_file = Path("\\\\?\\" + str(scoped_gt_file.resolve())) if os.name == "nt" else scoped_gt_file
        with open(open_gt_file, "w", encoding="utf-8") as f:
            json.dump({"UrlLocal": url_local, "annotations": target_gt_annotations}, f, indent=2)

        # DevKit mirrored prediction directory
        devkit_pred_dir = output_dir / "soccernet_devkit_format"
        mirrored_dir = devkit_pred_dir / url_local.strip("/")
        if os.name == "nt":
            os.makedirs(f"\\\\?\\{mirrored_dir.resolve()}", exist_ok=True)
        else:
            mirrored_dir.mkdir(parents=True, exist_ok=True)
        devkit_spotting_file = mirrored_dir / "results_spotting.json"

        if os.name == "nt":
            eval_gt_dir = "\\\\?\\" + str(scoped_gt_dir.resolve())
            eval_pred_dir = "\\\\?\\" + str(devkit_pred_dir.resolve())
            extended_spotting_file = Path("\\\\?\\" + str(devkit_spotting_file.resolve()))
            with open(extended_spotting_file, "w", encoding="utf-8") as f:
                json.dump(spotting_pred, f, indent=2)
        else:
            eval_gt_dir = str(scoped_gt_dir)
            eval_pred_dir = str(devkit_pred_dir)
            with open(devkit_spotting_file, "w", encoding="utf-8") as f:
                json.dump(spotting_pred, f, indent=2)

        print("\n" + "=" * 90)
        print("  RUNNING OFFICIAL SOCCERNET ACTION SPOTTING EVALUATION")
        print("=" * 90)
        try:
            metrics = evaluate_match_action_spotting(
                ground_truth_dir=eval_gt_dir,
                predictions_dir=eval_pred_dir,
                split="test",
                prediction_file="results_spotting.json"
            )

            # Print Qualitative Spot-Check
            print_qualitative_event_comparison(
                predicted_events=spotting_pred.get("predictions", []),
                ground_truth_events=target_gt_annotations,
                tolerance_sec=5.0
            )

            print("\n" + "-" * 90)
            print(f"  SCOPED EVALUATION RESULTS: {model_cfg['name']} (Chunk {target_chunk if target_chunk is not None else 'All'})")
            print("-" * 90)
            print(f"  Official Tight Average-mAP (1s-5s):  {metrics.get('tight_average_mAP', 0.0):.4f}")
            print(f"  Official Loose Average-mAP (5s-60s): {metrics.get('loose_average_mAP', 0.0):.4f}")
            if "active_tight_average_mAP" in metrics:
                print(f"  Active Tight Average-mAP (1s-5s):    {metrics.get('active_tight_average_mAP', 0.0):.4f}")
                print(f"  Active Loose Average-mAP (5s-60s):   {metrics.get('active_loose_average_mAP', 0.0):.4f}")
                print(f"  Active Classes: {metrics.get('active_class_count', 0)} / 17 | Oracle Ceiling: {metrics.get('oracle_ceiling_mAP', 0.0):.4f}")
            print("-" * 90 + "\n")

            eval_out_path = output_dir / "evaluation_metrics.json"
            with open(eval_out_path, "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2)
            print(f"[SAVED] Evaluation metrics saved to {eval_out_path}")
        except Exception as e:
            print(f"[EVALUATION ERROR] DevKit evaluation failed: {e}")

    print("\n" + "=" * 90)
    print("  LOCAL BENCHMARK RUN COMPLETE ($0 Compute)")
    print("=" * 90 + "\n")


if __name__ == "__main__":
    main()
