"""
scripts/evaluate_v3_vqa.py
Master evaluation harness for SoccerNet Visual Question Answering (SN-VQA).
Evaluates open-weight VLMs (Qwen3.5-4B, MiniCPM-V 4.6) on multimodal MCQA over localized broadcast video clips.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

from processing.vqa_schema import VQAQuestion, VQAResponse, build_vqa_prompt, parse_vqa_model_output
from scripts.run_local_benchmark import (
    decode_chunk_frames_single_pass,
    burn_timestamp_overlay,
    create_contact_sheet,
    LOCAL_MODEL_CONFIGS,
    LocalVLMRunner
)


def run_vqa_benchmark(
    questions: List[VQAQuestion],
    chunks_dir: Path,
    manifest: Dict[str, Any],
    runner: LocalVLMRunner,
    output_dir: Path
) -> Dict[str, Any]:
    """Runs VQA evaluation across all question clips, caching decoded chunk streams."""
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    category_stats = {}

    # Chunk frame caches to avoid redundant re-decodes
    chunk_frames_cache: Dict[int, List[Tuple[float, Image.Image]]] = {}

    print(f"\n[VQA RUNNER] Starting SN-VQA Benchmark across {len(questions)} multimodal questions...")
    print(f"[VQA RUNNER] Model: {runner.cfg['name']} ({runner.precision}) | Dry-Run: {runner.dry_run}\n")

    start_bench_time = time.perf_counter()

    for idx, q in enumerate(questions, 1):
        print(f"[{idx:02d}/{len(questions):02d}] Question ID: {q.question_id} | Category: {q.category} | Chunk {q.chunk_index} ({q.start_sec:.1f}s - {q.end_sec:.1f}s)")

        # 1. Fetch frames for target chunk
        if q.chunk_index not in chunk_frames_cache:
            c_meta = next((c for c in manifest.get("chunks", []) if c["chunk_index"] == q.chunk_index), None)
            c_file = c_meta.get("filename", f"game_0_half_1_chunk_{q.chunk_index:02d}.mp4") if c_meta else f"game_0_half_1_chunk_{q.chunk_index:02d}.mp4"
            video_path = str(chunks_dir / c_file)
            if not os.path.exists(video_path):
                print(f"  [WARNING] Chunk video not found: {video_path}, skipping.")
                continue
            chunk_frames_cache[q.chunk_index] = decode_chunk_frames_single_pass(video_path, fps=1.0, max_size=448)

        all_chunk_frames = chunk_frames_cache[q.chunk_index]

        # 2. Extract localized slice and burn clock overlays
        slice_frames: List[Image.Image] = []
        for rel_sec, img in all_chunk_frames:
            if q.start_sec <= rel_sec <= q.end_sec:
                sec_in_clip = rel_sec - q.start_sec
                badged = burn_timestamp_overlay(img, sec_in_clip)
                slice_frames.append(badged)

        if not slice_frames:
            # Fallback placeholder if empty
            slice_frames = [Image.new("RGB", (448, 448), (0, 0, 0))]

        # Uniformly subsample to max 10 frames to enforce strict VRAM and token efficiency
        max_vqa_frames = 10
        if len(slice_frames) > max_vqa_frames:
            step = (len(slice_frames) - 1) / (max_vqa_frames - 1)
            sampled_indices = [int(round(i * step)) for i in range(max_vqa_frames)]
            slice_frames = [slice_frames[i] for i in sampled_indices]

        # 3. Prompt building and model inference
        prompt_text = build_vqa_prompt(q)
        t0 = time.perf_counter()

        if runner.dry_run:
            time.sleep(0.05)
            pred_response = VQAResponse(
                thought=f"Simulated dry-run inspection of {len(slice_frames)} frames.",
                selected_option=q.ground_truth_option,
                confidence=0.92
            )
            raw_output = pred_response.model_dump_json()
            parse_diag = {"strategy": "dry_run", "is_valid": True}
            inference_dur = 0.05
        else:
            runner.ensure_model_loaded()
            import torch
            import gc

            if runner.cfg.get("mode") == "contact_sheet":
                contact_sheet = create_contact_sheet(slice_frames, window_sec=int(round(q.end_sec - q.start_sec)))
                with torch.inference_mode():
                    raw_output = runner.model.answer_question(contact_sheet, prompt_text, tokenizer=runner.tokenizer)
            else:
                apply_kwargs = dict(tokenize=False, add_generation_prompt=True)
                if runner.model_key == "qwen3.5-4b":
                    apply_kwargs["enable_thinking"] = False

                messages = [
                    {"role": "system", "content": "You are a professional video referee. Respond strictly with a single raw JSON object starting with '{'. Do not write markdown text or conversational preamble outside the JSON."},
                    {"role": "user", "content": [{"type": "image", "image": img} for img in slice_frames] + [{"type": "text", "text": prompt_text}]}
                ]
                prompt_formatted = runner.processor.apply_chat_template(messages, **apply_kwargs)
                inputs = runner.processor(text=[prompt_formatted], images=[slice_frames], return_tensors="pt").to(runner.device)

                with torch.inference_mode():
                    outputs = runner.model.generate(**inputs, max_new_tokens=96)

                in_len = inputs["input_ids"].shape[1] if "input_ids" in inputs else 0
                gen_tokens = outputs[0][in_len:] if in_len > 0 and outputs.shape[1] > in_len else outputs[0]
                raw_output = runner.processor.decode(gen_tokens, skip_special_tokens=True)

                del inputs, outputs, gen_tokens
                if "prompt_formatted" in locals():
                    del prompt_formatted
                if "messages" in locals():
                    del messages

            gc.collect()
            torch.cuda.empty_cache()

            inference_dur = round(time.perf_counter() - t0, 2)
            pred_response, parse_diag = parse_vqa_model_output(raw_output, q)

        is_correct = False
        selected_opt = None
        if pred_response:
            selected_opt = pred_response.selected_option
            is_correct = (selected_opt == q.ground_truth_option)

        status_str = "CORRECT [HIT]" if is_correct else "INCORRECT [MISS]"
        print(f"  -> Pred: {selected_opt} | GT: {q.ground_truth_option} | {status_str} in {inference_dur:.2f}s")

        res_item = {
            "question_id": q.question_id,
            "category": q.category,
            "question": q.question,
            "ground_truth_option": q.ground_truth_option,
            "predicted_option": selected_opt,
            "is_correct": is_correct,
            "confidence": pred_response.confidence if pred_response else 0.0,
            "thought": pred_response.thought if pred_response else "",
            "raw_output": raw_output,
            "inference_sec": inference_dur,
            "parse_diag": parse_diag
        }
        results.append(res_item)

        # Update category tallies
        cat = q.category
        if cat not in category_stats:
            category_stats[cat] = {"total": 0, "correct": 0}
        category_stats[cat]["total"] += 1
        if is_correct:
            category_stats[cat]["correct"] += 1

    total_time = round(time.perf_counter() - start_bench_time, 2)
    total_q = len(results)
    total_correct = sum(1 for r in results if r["is_correct"])
    overall_acc = (total_correct / max(1, total_q)) * 100

    summary = {
        "model_key": runner.model_key,
        "model_name": runner.cfg["name"],
        "precision": runner.precision,
        "dry_run": runner.dry_run,
        "total_questions": total_q,
        "total_correct": total_correct,
        "overall_accuracy_pct": round(overall_acc, 2),
        "random_baseline_pct": 25.0,
        "total_wall_sec": total_time,
        "avg_latency_per_question_sec": round(total_time / max(1, total_q), 2),
        "per_category_accuracy": {
            cat: {
                "total": s["total"],
                "correct": s["correct"],
                "accuracy_pct": round((s["correct"] / s["total"]) * 100, 2)
            }
            for cat, s in category_stats.items()
        },
        "results": results
    }

    # Print Report
    print("\n" + "=" * 80)
    print("  SOCCERNET V3 — VISUAL QUESTION ANSWERING (SN-VQA) BENCHMARK REPORT")
    print("=" * 80)
    print(f"  Model:                 {runner.cfg['name']} ({runner.precision})")
    print(f"  Overall Top-1 Accuracy:{summary['overall_accuracy_pct']:.2f}% ({total_correct}/{total_q})")
    print(f"  Random Guess Baseline: 25.00%")
    print(f"  Average Latency:       {summary['avg_latency_per_question_sec']:.2f}s per question")
    print("-" * 80)
    print("  Category Performance Breakdown:")
    for cat, s in summary["per_category_accuracy"].items():
        print(f"   - {cat:<24}: {s['accuracy_pct']:>6.2f}% ({s['correct']}/{s['total']})")
    print("=" * 80 + "\n")

    out_file = output_dir / f"vqa_{runner.model_key}_{runner.precision}_metrics.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[SAVED] VQA benchmark metrics saved to {out_file}\n")

    return summary


def main():
    parser = argparse.ArgumentParser(description="SoccerNet V3 — Visual Question Answering (SN-VQA) Runner")
    parser.add_argument("--model", type=str, default="qwen3.5-4b", choices=list(LOCAL_MODEL_CONFIGS.keys()))
    parser.add_argument("--precision", type=str, default=None)
    parser.add_argument("--dataset", type=str, default="data/v3_vqa_dataset.json")
    parser.add_argument("--manifest", type=str, default="gate2_chunks/game_0_half_1_chunks_manifest.json")
    parser.add_argument("--chunks-dir", type=str, default="gate2_chunks")
    parser.add_argument("--output-dir", type=str, default="logs/v3_eval/vqa")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    model_cfg = LOCAL_MODEL_CONFIGS[args.model]
    precision = args.precision or model_cfg["default_precision"]

    with open(args.dataset, "r", encoding="utf-8") as f:
        q_raw = json.load(f)
    questions = [VQAQuestion(**item) for item in q_raw]

    with open(args.manifest, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    runner = LocalVLMRunner(
        model_key=args.model,
        precision=precision,
        dry_run=args.dry_run
    )

    run_vqa_benchmark(
        questions=questions,
        chunks_dir=Path(args.chunks_dir),
        manifest=manifest,
        runner=runner,
        output_dir=Path(args.output_dir)
    )


if __name__ == "__main__":
    main()
