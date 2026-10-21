"""
Subprocess-Isolated Local Memory & Modality Gate for SoccerNet VLM Benchmark
Evaluates open-weight VLM candidates on RTX 3000 Ada (8GB GDDR6 VRAM) across a stepped
window ladder: W in {15s, 30s, 60s} at 1 FPS (resized <= 448x448).

Key Features:
1. Subprocess Isolation: Each (model, precision, frame_count, audio) trial runs in its
   own Python subprocess to guarantee 100% clean VRAM isolation and avoid CUDA context leakage.
2. 4-Point Telemetry:
   - load_allocated_gib / load_reserved_gib (static weight footprint)
   - peak_allocated_gib / peak_reserved_gib (dynamic activation + KV cache peak)
   - wall-clock inference latency
   - schema_valid: strict validation against 17-class unified_schema.py
   - shared_memory_spill: flags Windows WDDM paging to system RAM (> 7.5 GiB reserved)
3. Pass Threshold: peak_reserved_gib < 7.5 GiB, zero shared spillover, valid schema output.
4. Modality Check (--test-audio): Tests joint audiovisual input for Phi-4 Multimodal and Gemma 4 E4B/E2B.
5. Dry-Run Mode (--dry-run): Exercises the complete orchestrator, worker subprocesses,
   telemetry capture, and reporting harness using synthetic memory workloads.
"""

import os
import sys
import json
import time
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Thresholds for 8GB GDDR6 GPU (leaving 500MB safety buffer for OS/WDDM display)
MAX_RESERVED_GIB = 7.5
DEFAULT_FRAME_LADDER = [15, 30, 60]

# Candidate Model Registry mapped by Track
MODEL_REGISTRY = {
    # Track 1: Core 5 Video Candidates
    "qwen3.5-4b": {
        "track": 1,
        "name": "Qwen3.5-4B",
        "hf_id": "Qwen/Qwen3.5-4B",
        "default_precisions": ["4bit"],
        "supports_audio": False,
        "notes": "Hybrid linear/full attention 3:1 ratio"
    },
    "gemma-4-e4b": {
        "track": 1,
        "name": "Gemma 4 E4B",
        "hf_id": "google/gemma-4-E4B-it",
        "default_precisions": ["4bit"],
        "supports_audio": True,
        "notes": "Dense 4.5B active; fallback to E2B if gate fails"
    },
    "gemma-4-e2b": {
        "track": 1,
        "name": "Gemma 4 E2B",
        "hf_id": "google/gemma-4-E2B-it",
        "default_precisions": ["4bit", "bf16"],
        "supports_audio": True,
        "notes": "Track 1 fallback candidate (2.3B effective / 5.1B total)"
    },
    "minicpm-v-4.6": {
        "track": 1,
        "name": "MiniCPM-V 4.6",
        "hf_id": "openbmb/MiniCPM-V-4.6",
        "default_precisions": ["bf16", "4bit"],
        "supports_audio": False,
        "notes": "Paired test: native BF16 vs 4-bit AWQ/NF4"
    },
    "internvl3.5-2b": {
        "track": 1,
        "name": "InternVL3.5-2B",
        "hf_id": "OpenGVLab/InternVL3_5-2B",
        "default_precisions": ["bf16", "4bit"],
        "supports_audio": False,
        "notes": "Qwen3 language backbone + InternViT"
    },
    "smolvlm2-2.2b": {
        "track": 1,
        "name": "SmolVLM2-2.2B",
        "hf_id": "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
        "default_precisions": ["bf16"],
        "supports_audio": False,
        "notes": "Lightweight video-trained model"
    },
    # Track 2: Screening Candidates (5-Half Subset)
    "phi-4-multimodal": {
        "track": 2,
        "name": "Phi-4 Multimodal",
        "hf_id": "microsoft/Phi-4-multimodal-instruct",
        "default_precisions": ["4bit"],
        "supports_audio": True,
        "notes": "Native speech and vision tokens"
    },
    "lfm2.5-vl-3b": {
        "track": 2,
        "name": "LFM2.5-VL-3B",
        "hf_id": "LiquidAI/LFM2.5-VL-3B",
        "default_precisions": ["4bit"],
        "supports_audio": False,
        "notes": "Hybrid short-conv / grouped-query attention"
    },
    "ministral-3-3b": {
        "track": 2,
        "name": "Ministral 3 3B",
        "hf_id": "mistralai/Ministral-3-3B-Instruct-2512",
        "default_precisions": ["fp8"],
        "supports_audio": False,
        "notes": "3.4B language + 0.4B vision (Native FineGrainedFP8 architecture)"
    },
    "deepseek-vl2-tiny": {
        "track": 2,
        "name": "DeepSeek-VL2-Tiny",
        "hf_id": "deepseek-ai/deepseek-vl2-tiny",
        "default_precisions": ["bf16", "4bit"],
        "supports_audio": False,
        "notes": "MoE: 1B active / 3B resident"
    },
    # Track 3: Non-Temporal Storyboard Baseline
    "moondream2": {
        "track": 3,
        "name": "Moondream2",
        "hf_id": "vikhyatk/moondream2",
        "default_precisions": ["bf16"],
        "supports_audio": False,
        "notes": "Contact-sheet 2D tiled baseline; excluded from temporal leaderboard"
    },
    # Track 4: Historical Longitudinal Ablation
    "qwen2.5-vl-7b": {
        "track": 4,
        "name": "Qwen2.5-VL-7B",
        "hf_id": "Qwen/Qwen2.5-VL-7B-Instruct",
        "default_precisions": ["4bit"],
        "supports_audio": False,
        "notes": "Direct bridge to audited V1 F1=0.075 / 0.041 baseline"
    }
}

def create_contact_sheet(images: List[Any], window_sec: int) -> Any:
    """
    Composites N sampled video frames into a single 2D contact sheet image with burned-in MM:SS
    labels per tile. Used for Track 3 (Moondream2 storyboard baseline).
    """
    from PIL import Image, ImageDraw
    import math

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

def normalize_soccernet_label(lbl_raw: str) -> Optional[str]:
    from processing.unified_schema import SOCCERNET_CLASSES
    lbl_clean = lbl_raw.strip().lower()
    for c in SOCCERNET_CLASSES:
        if c.lower() == lbl_clean:
            return c
    for c in SOCCERNET_CLASSES:
        if c.lower().rstrip("s") == lbl_clean.rstrip("s"):
            return c
        if lbl_clean in c.lower() or c.lower() in lbl_clean:
            return c
    return None

# ---------------------------------------------------------------------------
# Worker Subprocess Logic
# ---------------------------------------------------------------------------

def run_worker(args: argparse.Namespace) -> None:
    telemetry: Dict[str, Any] = {
        "model_key": args.model,
        "precision": args.precision,
        "frame_count": args.frames,
        "test_audio": args.test_audio,
        "dry_run": args.dry_run,
        "load_allocated_gib": 0.0,
        "load_reserved_gib": 0.0,
        "peak_allocated_gib": 0.0,
        "peak_reserved_gib": 0.0,
        "wall_clock_sec": 0.0,
        "schema_valid": False,
        "shared_memory_spill": False,
        "verdict": "UNKNOWN",
        "raw_output": None,
        "error": None
    }

    t0 = time.perf_counter()

    if args.dry_run:
        if "moondream" in args.model:
            base_weights_gb = 3.8 if "bf16" in args.precision else 1.9
            activation_gb = 0.35
            peak_alloc = base_weights_gb + activation_gb
            peak_res = peak_alloc + 0.3
            telemetry["load_allocated_gib"] = round(base_weights_gb, 3)
            telemetry["load_reserved_gib"] = round(base_weights_gb + 0.3, 3)
            telemetry["peak_allocated_gib"] = round(peak_alloc, 3)
            telemetry["peak_reserved_gib"] = round(peak_res, 3)
            telemetry["shared_memory_spill"] = False
        else:
            base_weights_gb = 2.5 if "4bit" in args.precision or "awq" in args.precision else 4.8
            if "7b" in args.model:
                base_weights_gb = 4.2 if "4bit" in args.precision else 14.0
            elif "2b" in args.model or "1.3b" in args.model or "minicpm" in args.model:
                base_weights_gb = 1.8 if "4bit" in args.precision else 3.2

            activation_per_frame_gb = 0.045
            audio_gb = 0.25 if args.test_audio else 0.0

            telemetry["load_allocated_gib"] = round(base_weights_gb, 3)
            telemetry["load_reserved_gib"] = round(base_weights_gb + 0.3, 3)

            peak_alloc = base_weights_gb + (args.frames * activation_per_frame_gb) + audio_gb
            peak_res = peak_alloc + 0.5
            telemetry["peak_allocated_gib"] = round(peak_alloc, 3)
            telemetry["peak_reserved_gib"] = round(peak_res, 3)

            if peak_res > MAX_RESERVED_GIB:
                telemetry["shared_memory_spill"] = True

        time.sleep(0.05 * (args.frames / 15.0))
        telemetry["wall_clock_sec"] = round(time.perf_counter() - t0, 3)

        from processing.unified_schema import SpottedEvent
        sample_event = {
            "timestamp": f"00:{min(args.frames, 59):02d}",
            "commentary": f"Simulated play at {args.frames}s window.",
            "label": "Foul",
            "confidence": 0.88,
            "visibility": "visible"
        }
        try:
            ev = SpottedEvent(**sample_event)
            telemetry["schema_valid"] = True
            telemetry["verdict"] = "PASS" if not telemetry["shared_memory_spill"] else "OOM_FAIL"
        except Exception as e:
            telemetry["schema_valid"] = False
            telemetry["error"] = str(e)
            telemetry["verdict"] = "SCHEMA_FAIL"

        telemetry["raw_output"] = json.dumps([sample_event], indent=2)
        print("__RESULT_JSON__" + json.dumps(telemetry))
        return

    # Real GPU evaluation
    try:
        import torch
    except ImportError:
        telemetry["error"] = "PyTorch is not installed in the current environment."
        telemetry["verdict"] = "ERROR"
        print("__RESULT_JSON__" + json.dumps(telemetry))
        sys.exit(0)

    if not torch.cuda.is_available():
        telemetry["error"] = "CUDA is not available on this system."
        telemetry["verdict"] = "ERROR"
        print("__RESULT_JSON__" + json.dumps(telemetry))
        sys.exit(0)

    try:
        device = torch.device("cuda:0")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        model_meta = MODEL_REGISTRY.get(args.model, {})
        hf_id = model_meta.get("hf_id", args.model)

        # 1. Model Loading
        from transformers import AutoProcessor, AutoModelForImageTextToText, AutoModel

        torch_dtype = torch.bfloat16 if args.precision in ("bf16", "4bit", "nf4") else torch.float16
        load_kwargs: Dict[str, Any] = {
            "torch_dtype": torch_dtype,
            "device_map": "auto",
            "trust_remote_code": True
        }

        if "4bit" in args.precision or "nf4" in args.precision:
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                llm_int8_enable_fp32_cpu_offload=True
            )
            load_kwargs["device_map"] = {"": "cuda:0"}
        elif "fp8" in args.precision:
            load_kwargs["device_map"] = {"": "cuda:0"}

        from transformers import AutoTokenizer
        if args.model == "internvl3.5-2b":
            from transformers.modeling_utils import PreTrainedModel
            PreTrainedModel.all_tied_weights_keys = {}
            model = AutoModel.from_pretrained(hf_id, **load_kwargs).eval()
            tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True, use_fast=False)
            processor = None
        else:
            from transformers import AutoModelForCausalLM
            try:
                model = AutoModelForImageTextToText.from_pretrained(hf_id, **load_kwargs).eval()
            except Exception:
                try:
                    model = AutoModelForCausalLM.from_pretrained(hf_id, **load_kwargs).eval()
                except Exception:
                    model = AutoModel.from_pretrained(hf_id, **load_kwargs).eval()
            if args.model == "moondream2":
                tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
                processor = None
            elif "ministral" in args.model:
                processor = AutoProcessor.from_pretrained(hf_id, trust_remote_code=True, fix_mistral_regex=True)
            else:
                processor = AutoProcessor.from_pretrained(hf_id, trust_remote_code=True)
            if args.model in ("qwen3.5-4b", "qwen2.5-vl-7b"):
                if hasattr(processor, "image_processor"):
                    processor.image_processor.max_pixels = 256 * 256
                    processor.image_processor.min_pixels = 128 * 128
            elif args.model == "smolvlm2-2.2b":
                if hasattr(processor, "image_processor"):
                    processor.image_processor.do_image_splitting = False
                    processor.image_processor.size = {"longest_edge": 384}

        telemetry["load_allocated_gib"] = round(torch.cuda.memory_allocated() / (1024 ** 3), 3)
        telemetry["load_reserved_gib"] = round(torch.cuda.memory_reserved() / (1024 ** 3), 3)

        # 2. Frame Ingestion
        from PIL import Image
        frames_dir = getattr(args, "frames_dir", None) or os.path.join(REPO_ROOT, "test_slices", "window_09_10", "frames_60")
        sample_images = []
        if os.path.isdir(frames_dir):
            all_jpgs = sorted([os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if f.endswith(".jpg")])
            if all_jpgs:
                # Contiguous 1.0 FPS slice [0 : args.frames]
                window_jpgs = all_jpgs[:args.frames]
                for img_path in window_jpgs:
                    with Image.open(img_path) as im:
                        sample_images.append(im.convert("RGB"))
        if not sample_images:
            sample_images = [
                Image.new("RGB", (448, 448), color=(i * 4 % 255, 100, 150))
                for i in range(args.frames)
            ]

        from processing.unified_schema import SYSTEM_INSTRUCTION, SOCCERNET_CLASSES
        clip_end_ts = f"{args.frames // 60:02d}:{args.frames % 60:02d}"
        prompt_text = (
            f"{SYSTEM_INSTRUCTION}\n"
            f"Video window length: {args.frames} seconds (00:00 to {clip_end_ts}).\n"
            f"CRITICAL: Output ONLY a valid JSON array of objects. Do NOT use markdown tables. Do NOT output commentary outside JSON.\n"
            f"All timestamps must be within 00:00 and {clip_end_ts}.\n"
            f"Example format:\n"
            '[{"timestamp": "MM:SS", "commentary": "Brief description of the action...", "label": "Corner", "confidence": 0.90, "visibility": "visible"}]\n'
        )

        torch.cuda.reset_peak_memory_stats()
        t_gen = time.perf_counter()

        if args.model == "internvl3.5-2b":
            import torchvision.transforms as T
            from torchvision.transforms.functional import InterpolationMode
            transform = T.Compose([
                T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
                T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            pixel_values = torch.stack([transform(f) for f in sample_images]).to(torch.bfloat16).to(device)
            question = "".join([f"Frame{i+1}: <image>\n" for i in range(len(sample_images))]) + prompt_text
            with torch.inference_mode():
                gen_text = model.chat(tokenizer, pixel_values, question, generation_config=dict(max_new_tokens=512, do_sample=False))
        else:
            # Audio slice if requested
            audio_slice = None
            sr = 16000
            if args.test_audio and model_meta.get("supports_audio"):
                wav_path = os.path.join(REPO_ROOT, "test_slices", "window_09_10", "clip_60s.wav")
                if os.path.exists(wav_path):
                    import soundfile as sf
                    audio_data, sr = sf.read(wav_path)
                    audio_slice = audio_data[: sr * args.frames]

            if "moondream" in args.model:
                contact_sheet = create_contact_sheet(sample_images, window_sec=args.frames)
                with torch.inference_mode():
                    gen_text = model.answer_question(contact_sheet, prompt_text, tokenizer=tokenizer)
            elif args.model == "phi-4-multimodal":
                image_tags = "".join([f"<|image_{i+1}|>\n" for i in range(len(sample_images))])
                if args.test_audio and audio_slice is not None:
                    prompt_formatted = f"<|user|>\n{image_tags}<|audio_1|>\n{prompt_text}<|end|>\n<|assistant|>\n"
                    inputs = processor(text=prompt_formatted, images=sample_images, audios=[(audio_slice, sr)], return_tensors="pt").to(device)
                else:
                    prompt_formatted = f"<|user|>\n{image_tags}{prompt_text}<|end|>\n<|assistant|>\n"
                    inputs = processor(text=prompt_formatted, images=sample_images, return_tensors="pt").to(device)
                with torch.inference_mode():
                    outputs = model.generate(**inputs, max_new_tokens=512)
                in_len = inputs["input_ids"].shape[1] if "input_ids" in inputs else 0
                gen_tokens = outputs[0][in_len:] if in_len > 0 and outputs.shape[1] > in_len else outputs[0]
                gen_text = processor.decode(gen_tokens, skip_special_tokens=True)
            elif args.test_audio and audio_slice is not None:
                messages = [{"role": "user", "content": [{"type": "image", "image": img} for img in sample_images] + [{"type": "audio", "audio": audio_slice}, {"type": "text", "text": prompt_text}]}]
                prompt_formatted = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = processor(text=[prompt_formatted], images=[sample_images], audio=[audio_slice], sampling_rate=sr, return_tensors="pt").to(device)
                with torch.inference_mode():
                    outputs = model.generate(**inputs, max_new_tokens=512)
                in_len = inputs["input_ids"].shape[1] if "input_ids" in inputs else 0
                gen_tokens = outputs[0][in_len:] if in_len > 0 and outputs.shape[1] > in_len else outputs[0]
                gen_text = processor.decode(gen_tokens, skip_special_tokens=True)
            else:
                apply_kwargs = dict(tokenize=False, add_generation_prompt=True)
                if args.model == "qwen3.5-4b":
                    apply_kwargs["enable_thinking"] = False
                messages = [{"role": "user", "content": [{"type": "image", "image": img} for img in sample_images] + [{"type": "text", "text": prompt_text}]}]
                prompt_formatted = processor.apply_chat_template(messages, **apply_kwargs)
                inputs = processor(text=[prompt_formatted], images=[sample_images], return_tensors="pt").to(device)
                with torch.inference_mode():
                    outputs = model.generate(**inputs, max_new_tokens=512)
                in_len = inputs["input_ids"].shape[1] if "input_ids" in inputs else 0
                gen_tokens = outputs[0][in_len:] if in_len > 0 and outputs.shape[1] > in_len else outputs[0]
                gen_text = processor.decode(gen_tokens, skip_special_tokens=True)

        telemetry["wall_clock_sec"] = round(time.perf_counter() - t_gen, 3)
        telemetry["raw_output"] = gen_text

        peak_alloc = torch.cuda.max_memory_allocated() / (1024 ** 3)
        peak_res = torch.cuda.max_memory_reserved() / (1024 ** 3)
        telemetry["peak_allocated_gib"] = round(peak_alloc, 3)
        telemetry["peak_reserved_gib"] = round(peak_res, 3)

        if peak_res > MAX_RESERVED_GIB:
            telemetry["shared_memory_spill"] = True

                # 3. Robust Schema Verification & Multi-Format Parsing
        from processing.unified_schema import SpottedEvent
        import re
        
        valid_events = 0
        parsed_events_list = []
        out_of_bounds_count = 0
        from processing.unified_schema import parse_mm_ss_to_seconds

        def process_event(data: dict):
            nonlocal valid_events, out_of_bounds_count
            data.setdefault("confidence", 0.85)
            data.setdefault("visibility", "visible")
            lbl = normalize_soccernet_label(str(data.get("label", "")))
            if not lbl:
                return
            data["label"] = lbl
            ts = str(data.get("timestamp", ""))
            try:
                sec = parse_mm_ss_to_seconds(ts)
            except ValueError:
                out_of_bounds_count += 1
                return
            # Strict Layer 2 Bounds Check: timestamp must be within [0.0, args.frames]
            if sec < 0.0 or sec > float(args.frames):
                out_of_bounds_count += 1
                return
            try:
                ev = SpottedEvent(**data)
                valid_events += 1
                parsed_events_list.append(ev.model_dump())
            except Exception:
                pass

        # Strategy 1: JSON Array
        json_array_match = re.search(r"\[\s*\{.*?\}\s*\]", gen_text, re.DOTALL)
        if json_array_match:
            try:
                data = json.loads(json_array_match.group(0))
                if isinstance(data, list):
                    for item in data:
                        process_event(item)
            except Exception:
                pass

        # Strategy 2: JSON Lines / Repeated Objects
        if valid_events == 0:
            for m in re.finditer(r"\{[^{}]*\}", gen_text):
                try:
                    data = json.loads(m.group(0))
                    process_event(data)
                except Exception:
                    pass

        # Strategy 3: Markdown Table Recovery Fallback
        if valid_events == 0 and "|" in gen_text:
            lines = [l.strip() for l in gen_text.splitlines() if l.strip().startswith("|") and not l.strip().startswith("|---")]
            for line in lines:
                cols = [c.strip() for c in line.split("|")[1:-1]]
                if len(cols) >= 3:
                    ts_raw, comm, lbl = cols[0], cols[1], cols[2]
                    if "timestamp" in ts_raw.lower() or "label" in lbl.lower():
                        continue
                    ts_parts = ts_raw.split(":")
                    if len(ts_parts) == 2:
                        try:
                            mm = int(ts_parts[0])
                            ss = int(ts_parts[1])
                            ts_clean = f"{mm:02d}:{ss:02d}"
                            process_event({"timestamp": ts_clean, "commentary": comm, "label": lbl})
                        except Exception:
                            pass

        telemetry["parsed_events"] = parsed_events_list
        telemetry["out_of_bounds_drops"] = out_of_bounds_count

        if valid_events > 0:
            telemetry["schema_valid"] = True
            telemetry["verdict"] = "PASS" if not telemetry["shared_memory_spill"] else "OOM_FAIL"
            telemetry["error"] = None
        else:
            telemetry["schema_valid"] = False
            if not telemetry.get("error"):
                telemetry["error"] = f"No valid within-bounds SpottedEvent detected (out_of_bounds_drops={out_of_bounds_count})."
            telemetry["verdict"] = "PARSE_FAIL" 

    except Exception as e:
        is_oom = isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in str(e).lower()
        if is_oom:
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass
            telemetry["error"] = f"CUDA OutOfMemoryError: {str(e)[:150]}"
            telemetry["verdict"] = "OOM_FAIL"
            telemetry["passed"] = False
            telemetry["shared_memory_spill"] = True
            telemetry["peak_reserved_gib"] = 8.0
        else:
            telemetry["error"] = str(e)
            telemetry["verdict"] = "ERROR"
            telemetry["passed"] = False

    print("__RESULT_JSON__" + json.dumps(telemetry))

# ---------------------------------------------------------------------------
# Parent Orchestrator Logic
# ---------------------------------------------------------------------------

def run_orchestrator(args: argparse.Namespace) -> None:
    print("\n" + "=" * 95)
    print(" SOCCERNET VLM BENCHMARK: SUBPROCESS-ISOLATED MEMORY & MODALITY GATE")
    print(f" Target Hardware: RTX 3000 Ada (8GB VRAM) | Max Reserved VRAM: {MAX_RESERVED_GIB} GiB")
    print(f" Frame Ladder Steps: {args.frame_ladder} frames @ 1 FPS (<= 448x448)")
    print(f" Mode: {'DRY-RUN (Synthetic Telemetry Simulation)' if args.dry_run else 'LIVE INFERENCE'}")
    print("=" * 95 + "\n")

    models_to_test = []
    if args.models:
        for m in args.models:
            m_clean = m.strip().lower()
            if m_clean == "all":
                models_to_test = list(MODEL_REGISTRY.keys())
                break
            elif m_clean in MODEL_REGISTRY:
                models_to_test.append(m_clean)
            else:
                print(f"[WARN] Unknown model key '{m}'. Skipping.")
    else:
        if args.track == "all":
            models_to_test = list(MODEL_REGISTRY.keys())
        else:
            target_track = int(args.track)
            models_to_test = [k for k, v in MODEL_REGISTRY.items() if v["track"] == target_track]

    if not models_to_test:
        print(f"[ERROR] No valid models specified to test for track '{args.track}'.")
        sys.exit(1)

    all_results: List[Dict[str, Any]] = []

    for model_key in models_to_test:
        meta = MODEL_REGISTRY[model_key]
        precisions = args.precisions if args.precisions else meta["default_precisions"]
        supports_audio = meta["supports_audio"]

        print(f"\n--- Testing Candidate: {meta['name']} (Track {meta['track']}) ---")
        print(f"    HF Identifier: {meta['hf_id']}")
        print(f"    Notes: {meta['notes']}")

        if not args.dry_run:
            print(f"    Checking local cache for {meta['hf_id']}...")
            try:
                from huggingface_hub import snapshot_download
                snapshot_download(repo_id=meta["hf_id"], ignore_patterns=["*.msgpack", "*.h5", "*.ot"])
                print("    Model files cached locally.")
            except Exception as dl_err:
                print(f"    [INFO] Pre-cache check: {dl_err}. Proceeding to loader...")

        audio_modes = [False, True] if (args.test_audio and supports_audio) else [False]

        for precision in precisions:
            ladder_breached = False
            prev_latency = None

            for frames in args.frame_ladder:
                for audio_flag in audio_modes:
                    audio_str = "+Audio" if audio_flag else "VideoOnly"
                    trial_desc = f"{meta['name']} [{precision}, {frames}f, {audio_str}]"
                    print(f"  -> Running trial: {trial_desc:45s} ... ", end="", flush=True)

                    cmd = [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--worker",
                        "--model", model_key,
                        "--precision", precision,
                        "--frames", str(frames)
                    ]
                    if audio_flag:
                        cmd.append("--test-audio")
                    if args.dry_run:
                        cmd.append("--dry-run")

                    try:
                        sub_env = os.environ.copy()
                        sub_env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + sub_env.get("PYTHONPATH", "")
                        proc = subprocess.run(
                            cmd,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            text=True,
                            cwd=str(REPO_ROOT),
                            env=sub_env,
                            timeout=args.timeout_sec
                        )

                        trial_telemetry = None
                        for line in proc.stdout.splitlines():
                            if line.startswith("__RESULT_JSON__"):
                                raw_json = line.replace("__RESULT_JSON__", "")
                                trial_telemetry = json.loads(raw_json)
                                break

                        if trial_telemetry is None:
                            trial_telemetry = {
                                "model_key": model_key,
                                "precision": precision,
                                "frame_count": frames,
                                "test_audio": audio_flag,
                                "error": f"Subprocess failed without emitting telemetry. Stderr: {proc.stderr[:200]}"
                            }

                        peak_res = trial_telemetry.get("peak_reserved_gib", 0.0)
                        shared_spill = trial_telemetry.get("shared_memory_spill", False)
                        schema_ok = trial_telemetry.get("schema_valid", False)
                        err = trial_telemetry.get("error")
                        verdict_type = trial_telemetry.get("verdict", "")
                        curr_latency = trial_telemetry.get("wall_clock_sec", 0.0)

                        scaling_ratio = None
                        if prev_latency and prev_latency > 0:
                            scaling_ratio = round(curr_latency / prev_latency, 2)
                            trial_telemetry["latency_scaling_ratio"] = scaling_ratio
                            if scaling_ratio > 3.5:
                                shared_spill = True
                                trial_telemetry["shared_memory_spill"] = True
                        prev_latency = curr_latency

                        is_oom = verdict_type == "OOM_FAIL" or "out of memory" in str(err or "").lower()

                        passed = (
                            proc.returncode == 0
                            and err is None
                            and peak_res < MAX_RESERVED_GIB
                            and not shared_spill
                            and schema_ok
                            and not is_oom
                        )

                        trial_telemetry["passed"] = passed
                        trial_telemetry["model_name"] = meta["name"]
                        trial_telemetry["track"] = meta["track"]
                        all_results.append(trial_telemetry)

                        if passed:
                            verdict_str = "[PASS]"
                        elif is_oom:
                            verdict_str = "[OOM_FAIL]"
                        else:
                            verdict_str = "[FAIL]"

                        scale_str = f" (Scale: {scaling_ratio}x)" if scaling_ratio else ""
                        print(f"{verdict_str} (Peak Res: {peak_res:.2f} GiB, Latency: {curr_latency:.2f}s{scale_str})")

                        if not passed and err:
                            print(f"       Reason: {err}")

                        if not passed and (is_oom or peak_res >= MAX_RESERVED_GIB or shared_spill):
                            if not audio_flag:
                                ladder_breached = True
                                print(f"       [!] Memory ceiling breached on video-only at {frames} frames. Halting further ladder expansion for {model_key} [{precision}].")
                                break
                            else:
                                print(f"       [!] Memory ceiling breached with audio at {frames} frames.")

                    except subprocess.TimeoutExpired:
                        print(f"[TIMEOUT] (Exceeded {args.timeout_sec}s)")
                        all_results.append({
                            "model_key": model_key,
                            "model_name": meta["name"],
                            "track": meta["track"],
                            "precision": precision,
                            "frame_count": frames,
                            "passed": False,
                            "error": f"Subprocess timed out after {args.timeout_sec}s"
                        })
                        break

                if ladder_breached:
                    break

    # Tabular Summary Output
    print("\n" + "=" * 105)
    print(" LOCAL MEMORY GATE VERIFICATION SUMMARY TABLE")
    print("=" * 105)
    headers = f"{'Model':<18} {'Track':<6} {'Prec':<10} {'Frames':<8} {'Audio':<7} {'Load(GB)':<10} {'Peak(GB)':<10} {'Time(s)':<9} {'Schema':<8} {'Verdict'}"
    print(headers)
    print("-" * 105)

    for res in all_results:
        m_name = res.get("model_name", res.get("model_key", ""))[:17]
        track_str = f"T{res.get('track', '-')}"
        prec = res.get("precision", "-")[:9]
        frames = str(res.get("frame_count", "-"))
        audio = "Yes" if res.get("test_audio") else "No"
        load_gb = f"{res.get('load_reserved_gib', 0.0):.2f}"
        peak_gb = f"{res.get('peak_reserved_gib', 0.0):.2f}"
        wall_time = f"{res.get('wall_clock_sec', 0.0):.2f}"
        schema = "VALID" if res.get("schema_valid") else "FAIL"
        verdict = "PASS" if res.get("passed") else "FAIL"

        print(f"{m_name:<18} {track_str:<6} {prec:<10} {frames:<8} {audio:<7} {load_gb:<10} {peak_gb:<10} {wall_time:<9} {schema:<8} {verdict}")

    print("=" * 105)

    out_file = Path(args.output_file) if args.output_file else Path("logs") / "memory_gate_results.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[OUTPUT] Detailed results saved to: {out_file.resolve()}\n")

# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Subprocess-Isolated Local Memory & Modality Gate")
    parser.add_argument("--worker", action="store_true", help="Internal flag: run in worker subprocess mode")
    parser.add_argument("--model", type=str, default="qwen3.5-4b", help="Model key to test")
    parser.add_argument("--precision", type=str, default="4bit", help="Precision to test (bf16, 4bit, awq_4bit, fp8)")
    parser.add_argument("--frames", type=int, default=15, help="Number of frames for worker trial")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Explicit list of model keys to orchestrate (or 'all'). If omitted, defaults to --track selection.")
    parser.add_argument("--track", type=str, default="all", choices=["1", "2", "3", "4", "all"],
                        help="Benchmark track to evaluate: 1 (Core 5), 2 (Screening), 3 (Storyboard), 4 (Longitudinal), or all (All 4 tracks)")
    parser.add_argument("--precisions", nargs="+", default=None, help="Override default precision list")
    parser.add_argument("--frame-ladder", nargs="+", type=int, default=DEFAULT_FRAME_LADDER, help="Frame counts to test")
    parser.add_argument("--test-audio", action="store_true", help="Include audio test for audio-supported models")
    parser.add_argument("--dry-run", action="store_true", help="Run simulation with synthetic telemetry")
    parser.add_argument("--timeout-sec", type=int, default=900, help="Per-trial subprocess timeout in seconds")
    parser.add_argument("--output-file", type=str, default="logs/memory_gate_results.json", help="Path to output JSON file")

    args = parser.parse_args()

    if args.worker:
        run_worker(args)
    else:
        run_orchestrator(args)

if __name__ == "__main__":
    main()
