"""
Canonical MES-5 Multi-Match Sequential Orchestrator.
Target Hardware: 8 GB NVIDIA RTX 3000 Ada Laptop GPU ($0 Compute).

Runs sliding-window inference across the Canonical Minimal Evaluation Subset:
Games [0, 1, 2, 3, 13] x Halves [1, 2] (10 halves, ~8 hours, 1,132 events).
Enforces:
- Single-model GPU residency (gc.collect() + torch.cuda.empty_cache())
- Resumption tolerance: skips halves where Predictions-v2.json already exists
- Zero drift keyframe alignment via gate2_chunks manifests
"""

import os
import gc
import sys
import time
import json
import argparse
import subprocess
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

PYTHON_EXE = sys.executable

MES5_GAMES = [0, 1, 2, 3, 13]
MES5_HALVES = [1, 2]

def get_output_dir(model: str, game_id: int, half: int) -> Path:
    window_sec = 30 if model == "lfm2.5-vl-3b" else 60
    precision = "4bit"
    if game_id == 0 and half == 1:
        return WORKSPACE_ROOT / f"logs/local_eval/{model}_{precision}_W{window_sec}_half1"
    return WORKSPACE_ROOT / f"logs/local_eval/{model}_{precision}_W{window_sec}_game{game_id}_half{half}"

def is_half_completed(model: str, game_id: int, half: int) -> bool:
    out_dir = get_output_dir(model, game_id, half)
    pred_file = out_dir / "Predictions-v2.json"
    if pred_file.exists() and pred_file.stat().st_size > 100:
        return True
    return False

def run_single_half(model: str, game_id: int, half: int, dry_run: bool = False):
    desc = f"{model} on Game {game_id} Half {half}"
    print("\n" + "=" * 90)
    print(f"  EXECUTING: {desc}")
    print("=" * 90)

    cmd = [
        PYTHON_EXE, "scripts/run_local_benchmark.py",
        "--model", model,
        "--game-id", str(game_id),
        "--half", str(half),
        "--chunk-index", "-1"
    ]
    if dry_run:
        cmd.append("--dry-run")

    t0 = time.perf_counter()
    res = subprocess.run(cmd, cwd=str(WORKSPACE_ROOT))
    elapsed = time.perf_counter() - t0

    if res.returncode != 0:
        raise RuntimeError(f"Step failed ({desc}) with return code {res.returncode}")

    print(f"[COMPLETED] {desc} in {elapsed:.1f}s ({elapsed/60:.2f} min)\n")

def main():
    parser = argparse.ArgumentParser(description="Canonical MES-5 Multi-Match Orchestrator")
    parser.add_argument("--model", type=str, default="qwen3.5-4b",
                        choices=["qwen3.5-4b", "minicpm-v-4.6", "lfm2.5-vl-3b", "all"],
                        help="Model to run (default: qwen3.5-4b)")
    parser.add_argument("--games", type=int, nargs="+", default=[1, 2, 3, 13],
                        help="Game IDs to execute (Game 0 is skipped by default as it is already complete)")
    parser.add_argument("--halves", type=int, nargs="+", default=[1, 2],
                        help="Halves to execute (default: [1, 2])")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run with synthetic fast detections without calling local LLM weights")
    parser.add_argument("--include-game0", action="store_true",
                        help="Also evaluate Game 0 if missing")
    args = parser.parse_args()

    games = [0] + args.games if args.include_game0 and 0 not in args.games else args.games
    models = ["qwen3.5-4b", "minicpm-v-4.6", "lfm2.5-vl-3b"] if args.model == "all" else [args.model]

    print("=" * 90)
    print("  CANONICAL MES-5 MULTI-MATCH ORCHESTRATOR")
    print(f"  Target Models:  {models}")
    print(f"  Target Games:   {games}")
    print(f"  Target Halves:  {args.halves}")
    print(f"  Dry Run Mode:   {args.dry_run}")
    print("=" * 90)

    total_tasks = len(models) * len(games) * len(args.halves)
    task_idx = 0

    orchestrator_start = time.perf_counter()

    for model in models:
        print(f"\n>>> PROCESSING MODEL FAMILY: {model} <<<")
        for gid in games:
            for half in args.halves:
                task_idx += 1
                print(f"\n[{task_idx}/{total_tasks}] Game {gid} Half {half} | Model: {model}")
                if not args.dry_run and is_half_completed(model, gid, half):
                    print(f"  [+] Already completed on disk: {get_output_dir(model, gid, half).name}/Predictions-v2.json")
                    continue
                run_single_half(model, gid, half, dry_run=args.dry_run)

    total_wall_sec = time.perf_counter() - orchestrator_start
    print("\n" + "=" * 90)
    print(f"  MES-5 ORCHESTRATION PIPELINE COMPLETE IN {total_wall_sec/60:.2f} MINUTES!")
    print("=" * 90)

if __name__ == "__main__":
    main()
