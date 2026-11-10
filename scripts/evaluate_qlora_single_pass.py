"""
Step 3: Evaluate Unified Single-Model Student (Qwen3.5-4B + LoRA) on MES-5.
Evaluates the unified student in a single pass across all 10 broadcast halves of MES-5
(Games 0, 1, 2, 3, 13; 1,132 ground-truth events, 17/17 active classes).

Metrics evaluated:
1. Wall-clock runtime (1x single-model pass vs 3x ensemble pass)
2. Active Class Coverage (target 17/17)
3. Official Tight Average-mAP (1s-5s) and Loose Average-mAP (5s-60s)
4. Comparison with and without M_15 Selective Causal Consequent Rescorer
"""

import os
import gc
import sys
import time
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from PIL import Image

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

import torch
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    BitsAndBytesConfig
)
from peft import PeftModel

from processing.unified_schema import (
    SOCCERNET_CLASSES,
    parse_mm_ss_to_seconds,
    SpottedEvent
)
from processing.post_process_predictions import (
    class_aware_nms,
    apply_semantic_co_emission
)
from scripts.run_local_benchmark import (
    decode_chunk_frames_single_pass,
    burn_timestamp_overlay,
    build_window_prompt,
    parse_model_json_output
)
from scripts.evaluate_mes5_multimatch import MES5_GAMES, evaluate_predictions_tree
from scripts.evaluate_markov_prior_rescoring import (
    compute_out_of_sample_m15,
    apply_markov_and_temporal_prior
)

class QLoRAStudentRunner:
    def __init__(
        self,
        base_model_id: str = "Qwen/Qwen3.5-4B",
        adapter_dir: str = "models/qwen3.5-4b-soccernet-qlora",
        device: str = "cuda:0",
        max_pixels: int = 256 * 256
    ):
        self.base_model_id = base_model_id
        self.adapter_dir = str(WORKSPACE_ROOT / adapter_dir)
        self.device = device
        self.max_pixels = max_pixels
        self.model = None
        self.processor = None

    def load_model(self):
        if self.model is not None:
            return

        t0 = time.perf_counter()
        print(f"[*] Loading QLoRA Unified Student from {self.adapter_dir}...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True
        )

        base_model = AutoModelForImageTextToText.from_pretrained(
            self.base_model_id,
            quantization_config=bnb_config,
            device_map={"": self.device},
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            local_files_only=True
        )

        self.model = PeftModel.from_pretrained(base_model, self.adapter_dir).eval()

        self.processor = AutoProcessor.from_pretrained(
            self.adapter_dir,
            trust_remote_code=True,
            local_files_only=True
        )
        if hasattr(self.processor, "image_processor"):
            self.processor.image_processor.max_pixels = 1764
            self.processor.image_processor.min_pixels = 784

        vram = torch.cuda.memory_allocated() / (1024 ** 3)
        print(f"[+] Loaded Unified QLoRA Student in {time.perf_counter() - t0:.1f}s | VRAM: {vram:.2f} GiB")

    def infer_window(self, frames: List[Image.Image], window_sec: int) -> Tuple[List[Any], float, str, Dict[str, Any]]:
        self.load_model()
        t0 = time.perf_counter()
        prompt_text = "Spot soccer events: output JSON with commentary, timestamp, label, confidence, visibility."

        messages = [{
            "role": "user",
            "content": [{"type": "image", "image": img} for img in frames] + [{"type": "text", "text": prompt_text}]
        }]

        prompt_formatted = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )

        prompt_forced = prompt_formatted + "["
        inputs = self.processor(
            text=[prompt_forced],
            images=[frames],
            return_tensors="pt"
        ).to(self.device)

        with torch.inference_mode():
            outputs = self.model.generate(**inputs, max_new_tokens=256)

        in_len = inputs["input_ids"].shape[1] if "input_ids" in inputs else 0
        gen_tokens = outputs[0][in_len:] if in_len > 0 and outputs.shape[1] > in_len else outputs[0]
        gen_text = self.processor.decode(gen_tokens, skip_special_tokens=True).strip()
        full_text = "[" + gen_text

        del inputs, outputs, gen_tokens
        elapsed = round(time.perf_counter() - t0, 3)

        events, diag = parse_model_json_output(full_text, window_sec, spotting_only=False)
        return events, elapsed, full_text, diag


def run_qlora_inference_for_half(
    runner: QLoRAStudentRunner,
    game_id: int,
    half: int,
    chunks_dir: Path,
    window_sec: int = 60,
    stride_sec: int = 45,
    no_cache: bool = False
) -> Tuple[List[Dict[str, Any]], float]:
    out_dir = WORKSPACE_ROOT / f"logs/local_eval/qwen3.5-4b-qlora_4bit_W{window_sec}_game{game_id}_half{half}"
    out_dir.mkdir(parents=True, exist_ok=True)

    v2_path = out_dir / "Predictions-v2.json"
    cap_path = out_dir / "Predictions-Caption.json"

    if not no_cache and v2_path.exists() and cap_path.exists():
        print(f"[*] Cache Hit for Game {game_id} Half {half}. Loading from {v2_path}...")
        with open(v2_path, "r", encoding="utf-8") as f:
            v2_preds = json.load(f).get("predictions", [])
        return v2_preds, 0.0

    manifest_path = chunks_dir / f"game_{game_id}_half_{half}_chunks_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    chunks = manifest.get("chunks", [])
    print(f"\n=== Executing Unified Single-Model Inference: Game {game_id} Half {half} ({len(chunks)} Chunks) ===")

    all_v2_preds = []
    all_cap_preds = []
    total_time = 0.0

    for chunk in chunks:
        chunk_idx = chunk["chunk_index"]
        filename = chunk["filename"]
        chunk_path = chunks_dir / filename
        duration_sec = float(chunk.get("duration_sec", 615.0))
        actual_start_ms = int(chunk.get("actual_start_ms", 0))

        print(f"\n--- Processing Chunk {chunk_idx}: {filename} ({duration_sec:.1f}s) ---")
        chunk_frames = decode_chunk_frames_single_pass(str(chunk_path), fps=1.0)

        win_starts = []
        curr = 0.0
        while curr < duration_sec:
            win_starts.append(curr)
            curr += stride_sec

        chunk_raw_events = []
        for w_i, start_t in enumerate(win_starts):
            eff_window_dur = min(float(window_sec), duration_sec - start_t)
            if eff_window_dur < 5.0:
                continue

            end_t = start_t + eff_window_dur
            win_raw_frames = [f for f in chunk_frames if start_t <= f[0] < end_t]
            if not win_raw_frames:
                idx_start = min(int(round(start_t)), len(chunk_frames) - 1)
                idx_end = min(int(round(end_t)), len(chunk_frames))
                win_raw_frames = chunk_frames[idx_start:idx_end]

            step = (len(win_raw_frames) - 1) / 5.0 if len(win_raw_frames) > 1 else 0.0
            sampled_indices = [int(round(i * step)) for i in range(6)]
            win_frame_objs = [
                burn_timestamp_overlay(win_raw_frames[min(idx, len(win_raw_frames) - 1)][1], sec_in_window=win_raw_frames[min(idx, len(win_raw_frames) - 1)][0] - start_t)
                for idx in sampled_indices
            ]

            events, elapsed, raw_text, diag = runner.infer_window(win_frame_objs, int(eff_window_dur))
            total_time += elapsed

            for ev in events:
                sec = parse_mm_ss_to_seconds(ev.timestamp)
                pred_offset_ms = int(round(sec * 1000.0))
                win_start_ms = int(round(start_t * 1000.0))
                global_pos_ms = actual_start_ms + win_start_ms + pred_offset_ms

                mm = global_pos_ms // 60000
                ss = (global_pos_ms % 60000) // 1000
                gt_str = f"{half} - {mm:02d}:{ss:02d}"

                chunk_raw_events.append({
                    "gameTime": gt_str,
                    "label": ev.label,
                    "position": global_pos_ms,
                    "half": half,
                    "confidence": ev.confidence,
                    "commentary": ev.commentary,
                    "visibility": ev.visibility
                })

        # Apply Class-Aware NMS
        chunk_deduped = class_aware_nms(chunk_raw_events)
        all_v2_preds.extend(chunk_deduped)
        all_cap_preds.extend(chunk_deduped)
        print(f"  -> Chunk {chunk_idx}: {len(chunk_deduped)} events kept after NMS.")

    # Save half predictions
    with open(v2_path, "w", encoding="utf-8") as f:
        json.dump({"predictions": all_v2_preds}, f, indent=2)
    with open(cap_path, "w", encoding="utf-8") as f:
        json.dump({"predictions": all_cap_preds}, f, indent=2)

    print(f"[+] Saved {len(all_v2_preds)} predictions to {v2_path} (Inference Time: {total_time:.1f}s)")
    return all_v2_preds, total_time


def evaluate_qlora_student_on_mes5(
    adapter_dir: str = "models/qwen3.5-4b-soccernet-qlora",
    window_sec: int = 60,
    stride_sec: int = 45,
    games: List[int] = [0, 1, 2, 3, 13],
    no_cache: bool = False
):
    chunks_dir = WORKSPACE_ROOT / "gate2_chunks"
    eval_set_path = WORKSPACE_ROOT / "eval_set.json"
    gt_dir = WORKSPACE_ROOT / "test_eval_ground_truth"

    runner = QLoRAStudentRunner(adapter_dir=adapter_dir)
    total_wall_clock_time = 0.0

    print("==========================================================================")
    print("  EVALUATING UNIFIED SINGLE-MODEL STUDENT (QWEN3.5-4B + QLORA) ON MES-5")
    print(f"  Matches:         {games} (5 Matches, 10 Halves, 1,132 Ground Truth Events)")
    print(f"  Architecture:    1x Single Model Pass (Replacing 3x Ensemble Passes)")
    print("==========================================================================")

    # 1. Run inference across all 10 halves
    for gid in games:
        for half in [1, 2]:
            _, half_time = run_qlora_inference_for_half(
                runner, gid, half, chunks_dir,
                window_sec=window_sec, stride_sec=stride_sec,
                no_cache=no_cache
            )
            total_wall_clock_time += half_time

    # 2. Compute out-of-sample M_15 transition prior
    print("\n[*] Estimating Out-of-Sample M_15 Transition Matrix strictly on Games 4-12, 14-19...")
    prob_fwd, prob_bwd, _ = compute_out_of_sample_m15(eval_set_path, gt_dir)

    # 3. DevKit Evaluation: Raw Student vs Student + M_15 Rescorer
    eval_games = [g for g in MES5_GAMES if g[0] in games]

    for is_m15 in [False, True]:
        config_name = "qwen3.5-4b-qlora_m15" if is_m15 else "qwen3.5-4b-qlora_raw"
        devkit_base = WORKSPACE_ROOT / f"logs/local_eval/mes5_{config_name}"
        devkit_gt_dir = devkit_base / "devkit_ground_truth"
        devkit_pred_dir = devkit_base / "soccernet_devkit_format"

        devkit_gt_dir.mkdir(parents=True, exist_ok=True)
        devkit_pred_dir.mkdir(parents=True, exist_ok=True)

        total_preds = 0
        total_gt = 0

        for gid, rel_path in eval_games:
            match_gt_dir = devkit_gt_dir / rel_path
            match_pred_dir = devkit_pred_dir / rel_path
            match_gt_dir.mkdir(parents=True, exist_ok=True)
            match_pred_dir.mkdir(parents=True, exist_ok=True)

            # Copy GT
            src_gt = gt_dir / rel_path / "Labels-v2.json"
            with open(src_gt, "r", encoding="utf-8") as f:
                gt_data = json.load(f)
            total_gt += len(gt_data.get("annotations", []))
            with open(match_gt_dir / "Labels-v2.json", "w", encoding="utf-8") as f:
                json.dump(gt_data, f, indent=2)

            # Build Predictions
            match_preds = []
            for half in [1, 2]:
                half_dir = WORKSPACE_ROOT / f"logs/local_eval/qwen3.5-4b-qlora_4bit_W{window_sec}_game{gid}_half{half}"
                with open(half_dir / "Predictions-v2.json", "r", encoding="utf-8") as f1:
                    v2 = json.load(f1).get("predictions", [])
                with open(half_dir / "Predictions-Caption.json", "r", encoding="utf-8") as f2:
                    caps = json.load(f2).get("predictions", [])

                h_preds = apply_semantic_co_emission(v2, caps)
                if is_m15 and prob_fwd:
                    h_preds = apply_markov_and_temporal_prior(
                        h_preds, prob_fwd, prob_bwd, alpha=0.15,
                        prune_h1_substitutions=(half == 1),
                        prune_openplay_kickoffs=True
                    )
                match_preds.extend(h_preds)

            total_preds += len(match_preds)
            with open(match_pred_dir / "results_spotting.json", "w", encoding="utf-8") as f:
                json.dump({"UrlLocal": f"{rel_path}/", "predictions": match_preds}, f, indent=2)

        print(f"\n==========================================================================")
        print(f"  OFFICIAL DEVKIT EVALUATION: {config_name.upper()}")
        print(f"  Total Predictions: {total_preds} | Total Ground Truth: {total_gt}")
        print(f"==========================================================================")

        tight_mAP, loose_mAP, t_per_class, l_per_class = evaluate_predictions_tree(devkit_gt_dir, devkit_pred_dir)

        print(f"\n  Official Tight Average-mAP (1s-5s):  {tight_mAP:.4f} ({tight_mAP*100:.2f}%)")
        print(f"  Official Loose Average-mAP (5s-60s): {loose_mAP:.4f} ({loose_mAP*100:.2f}%)")
        active_count = sum(1 for ap in l_per_class.values() if ap > 0.0)
        print(f"  Active Classes (/17):                {active_count} / 17 ({active_count/17*100:.1f}%)")

        summary = {
            "config": config_name,
            "adapter_dir": adapter_dir,
            "tight_average_mAP": tight_mAP,
            "loose_average_mAP": loose_mAP,
            "active_class_count": active_count,
            "total_wall_clock_time_sec": total_wall_clock_time,
            "tight_per_class": t_per_class,
            "loose_per_class": l_per_class
        }
        with open(devkit_base / "mes5_evaluation_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    print("\n" + "=" * 90)
    print("  STAGE 1 QLORA DISTILLATION: COMPILATION & COMPARISON COMPLETE")
    print("=" * 90 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Unified Single-Model Student on MES-5")
    parser.add_argument("--adapter-dir", type=str, default="models/qwen3.5-4b-soccernet-qlora")
    parser.add_argument("--games", type=int, nargs="+", default=[0, 1, 2, 3, 13])
    parser.add_argument("--window-sec", type=int, default=60)
    parser.add_argument("--stride-sec", type=int, default=45)
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    evaluate_qlora_student_on_mes5(
        adapter_dir=args.adapter_dir,
        window_sec=args.window_sec,
        stride_sec=args.stride_sec,
        games=args.games,
        no_cache=args.no_cache
    )
