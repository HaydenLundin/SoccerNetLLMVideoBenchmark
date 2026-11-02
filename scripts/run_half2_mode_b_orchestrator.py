"""
Mode B Sequential Orchestrator for Game 0 Half 2.
Orchestrates:
1. Waits for Qwen3.5-4B (4bit, W=60s) completion if currently running.
2. Executes MiniCPM-V 4.6 (4bit, W=60s) across all 6 chunks of Half 2.
3. Executes LFM2.5-VL-3B (4bit, W=30s) across all 6 chunks of Half 2.
4. Executes evaluate_half2_specialist_and_substitutions.py to:
   - Form the 3-Model Domain-Routed Specialist Ensemble
   - Audit the 6 Substitutions in Half 2 (Validating Second-Half Substitution Law)
   - Evaluate the full 90-minute match (Half 1 + Half 2, 173 total events)
"""

import os
import sys
import time
import json
import subprocess
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

PYTHON_EXE = sys.executable

def run_step(cmd_args: list, desc: str):
    print("\n" + "=" * 90)
    print(f"  STARTING: {desc}")
    print(f"  Command:  {' '.join(cmd_args)}")
    print("=" * 90)
    t0 = time.perf_counter()
    res = subprocess.run(cmd_args, cwd=str(WORKSPACE_ROOT))
    elapsed = time.perf_counter() - t0
    if res.returncode != 0:
        raise RuntimeError(f"Step failed ({desc}) with returncode {res.returncode}")
    print(f"[COMPLETED] {desc} in {elapsed:.1f}s ({elapsed/60:.2f} min)\n")

def main():
    qwen_done_file = WORKSPACE_ROOT / "logs/local_eval/qwen3.5-4b_4bit_W60_game0_half2/Predictions-v2.json"
    
    # 1. Wait for Qwen3.5-4B if still running
    if not qwen_done_file.exists():
        print("[*] Waiting for Qwen3.5-4B on Game 0 Half 2 to complete...")
        while not qwen_done_file.exists():
            time.sleep(10)
        print("[+] Qwen3.5-4B predictions detected!")
    else:
        print("[+] Qwen3.5-4B predictions already present.")

    # 2. Run MiniCPM-V 4.6
    minicpm_done_file = WORKSPACE_ROOT / "logs/local_eval/minicpm-v-4.6_4bit_W60_game0_half2/Predictions-v2.json"
    if not minicpm_done_file.exists():
        run_step([
            PYTHON_EXE, "scripts/run_local_benchmark.py",
            "--model", "minicpm-v-4.6",
            "--game-id", "0",
            "--half", "2",
            "--chunk-index", "-1"
        ], "MiniCPM-V 4.6 (4bit, W=60s) on Game 0 Half 2")
    else:
        print("[+] MiniCPM-V 4.6 predictions already present; skipping inference.")

    # 3. Run LFM2.5-VL-3B
    lfm_done_file = WORKSPACE_ROOT / "logs/local_eval/lfm2.5-vl-3b_4bit_W30_game0_half2/Predictions-v2.json"
    if not lfm_done_file.exists():
        run_step([
            PYTHON_EXE, "scripts/run_local_benchmark.py",
            "--model", "lfm2.5-vl-3b",
            "--game-id", "0",
            "--half", "2",
            "--chunk-index", "-1"
        ], "LFM2.5-VL-3B (4bit, W=30s) on Game 0 Half 2")
    else:
        print("[+] LFM2.5-VL-3B predictions already present; skipping inference.")

    # 4. Run Specialist Ensemble & Substitution Benchmark
    run_step([
        PYTHON_EXE, "scripts/evaluate_half2_specialist_and_substitutions.py"
    ], "Half 2 Specialist Ensemble & Full 90-Minute Match Evaluation")

    print("\n" + "=" * 90)
    print("  MODE B BENCHMARK FOR GAME 0 FULL MATCH COMPLETE!")
    print("=" * 90 + "\n")

if __name__ == "__main__":
    main()
