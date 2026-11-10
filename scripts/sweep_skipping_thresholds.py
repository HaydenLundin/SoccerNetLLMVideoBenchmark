"""
Pareto Sweep for Stage 2 Window Skipping Thresholds across Game 0 Half 1.
Finds the optimal acoustic + optical gate parameters that maximize % windows skipped
while guaranteeing >= 98.5% (or 100%) Ground Truth Event Recall.
"""

import sys
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from scripts.test_stage2_window_skipping import extract_chunk_acoustic_profile, compute_optical_flow_activity
from scripts.run_local_benchmark import decode_chunk_frames_single_pass

def run_threshold_sweep():
    manifest_path = WORKSPACE_ROOT / "gate2_chunks/game_0_half_1_chunks_manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    gt_path = WORKSPACE_ROOT / "logs/local_eval/mes5_qwen3.5-4b/devkit_ground_truth/england_epl/2014-2015/2015-05-17 - 18-00 Manchester United 1 - 1 Arsenal/Labels-v2.json"
    with open(gt_path, "r", encoding="utf-8") as f:
        gt_data = json.load(f)["annotations"]

    h1_gt = [a for a in gt_data if a["gameTime"].startswith("1")]
    print(f"Total Ground Truth Events in Half 1: {len(h1_gt)}")

    window_sec = 60
    stride_sec = 45

    windows = []
    chunks = manifest.get("chunks", [])

    print("[*] Extracting acoustic and optical profiles across all 5 chunks...")
    for chunk in chunks:
        c_idx = chunk["chunk_index"]
        filename = chunk["filename"]
        chunk_path = WORKSPACE_ROOT / "gate2_chunks" / filename
        start_ms = int(chunk.get("actual_start_ms", 0))
        dur_sec = float(chunk.get("duration_sec", 615.0))

        w_z, c_db, _ = extract_chunk_acoustic_profile(chunk_path)
        chunk_frames = decode_chunk_frames_single_pass(str(chunk_path), fps=1.0)
        optical_diffs = compute_optical_flow_activity(chunk_frames)

        curr = 0.0
        w_i = 0
        while curr < dur_sec:
            w_start = curr
            w_end = min(curr + window_sec, dur_sec)
            eff_dur = w_end - w_start
            if eff_dur >= 5.0:
                bin_start = int(w_start / 0.5)
                bin_end = int(np.ceil(w_end / 0.5))
                max_w = float(np.max(w_z[bin_start:bin_end])) if len(w_z) > bin_start else 0.0
                max_c = float(np.max(c_db[bin_start:bin_end])) if len(c_db) > bin_start else 0.0

                f_start = min(int(round(w_start)), len(optical_diffs) - 1)
                f_end = min(int(round(w_end)), len(optical_diffs))
                w_d = optical_diffs[f_start:f_end]
                max_o = float(np.max(w_d)) if w_d else 0.0
                std_o = float(np.std(w_d)) if len(w_d) > 1 else 0.0
                mean_o = float(np.mean(w_d)) if w_d else 0.0

                g_start = start_ms + int(w_start * 1000)
                g_end = start_ms + int(w_end * 1000)

                # Find which GT events are in this window
                contained_gt = [ev for ev in h1_gt if g_start <= int(ev["position"]) <= g_end]

                windows.append({
                    "chunk": c_idx,
                    "win_idx": w_i,
                    "g_start": g_start,
                    "g_end": g_end,
                    "max_w": max_w,
                    "max_c": max_c,
                    "max_o": max_o,
                    "std_o": std_o,
                    "mean_o": mean_o,
                    "gt_count": len(contained_gt)
                })
                w_i += 1
            curr += stride_sec

    print(f"[+] Total Windows: {len(windows)}")

    # Parameter grids for sweep
    whistle_grid = [3.5, 4.0, 4.5, 5.0]
    crowd_grid = [5.0, 6.0, 7.0, 8.0]
    opt_cut_grid = [30.0, 40.0, 50.0, 60.0]
    opt_var_grid = [6.0, 8.0, 10.0, 12.0]

    best_configurations = []

    for w_th in whistle_grid:
        for c_th in crowd_grid:
            for o_th in opt_cut_grid:
                for v_th in opt_var_grid:
                    # Evaluate this configuration
                    active_indices = set()
                    for idx, win in enumerate(windows):
                        if (win["max_w"] >= w_th or 
                            win["max_c"] >= c_th or 
                            win["max_o"] >= o_th or 
                            win["std_o"] >= v_th):
                            active_indices.add(idx)

                    # Compute GT recall
                    covered_gt = 0
                    for ev in h1_gt:
                        pos = int(ev["position"])
                        for idx in active_indices:
                            win = windows[idx]
                            if win["g_start"] <= pos <= win["g_end"]:
                                covered_gt += 1
                                break

                    recall = covered_gt / len(h1_gt)
                    skipped_count = len(windows) - len(active_indices)
                    skip_rate = skipped_count / len(windows)

                    if recall >= 0.95:  # Only keep configurations with >= 95% recall
                        best_configurations.append({
                            "whistle_th": w_th,
                            "crowd_th": c_th,
                            "opt_cut_th": o_th,
                            "opt_var_th": v_th,
                            "active_windows": len(active_indices),
                            "skipped_windows": skipped_count,
                            "skip_rate_pct": round(skip_rate * 100, 1),
                            "speedup": round(len(windows) / max(len(active_indices), 1), 2),
                            "recall_pct": round(recall * 100, 1),
                            "covered_gt": covered_gt,
                            "total_gt": len(h1_gt)
                        })

    # Sort by skip_rate descending, then recall descending
    best_configurations.sort(key=lambda x: (x["recall_pct"] >= 98.0, x["skip_rate_pct"], x["recall_pct"]), reverse=True)

    print("\n" + "=" * 95)
    print("  TOP PARETO-OPTIMAL CONFIGURATIONS (High Skip Rate + High Recall)")
    print("=" * 95)
    print("Whistle_Z | Crowd_dB | Opt_Cut | Opt_Var | Active / Skip | Skip % | Speedup | Recall %")
    print("-" * 95)

    seen_rates = set()
    top_picks = []
    for cfg in best_configurations:
        key = (cfg["skip_rate_pct"], cfg["recall_pct"])
        if key not in seen_rates:
            seen_rates.add(key)
            top_picks.append(cfg)
            print(f"{cfg['whistle_th']:9.1f} | {cfg['crowd_th']:8.1f} | {cfg['opt_cut_th']:7.1f} | {cfg['opt_var_th']:7.1f} | {cfg['active_windows']:3d} / {cfg['skipped_windows']:2d}   | {cfg['skip_rate_pct']:5.1f}% | {cfg['speedup']:6.2f}x | {cfg['recall_pct']:6.1f}% ({cfg['covered_gt']}/{cfg['total_gt']})")
        if len(top_picks) >= 10:
            break

    # Save sweep results
    out_path = WORKSPACE_ROOT / "logs/local_eval/skipping_threshold_sweep.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(top_picks, f, indent=2)
    print(f"\n[+] Saved top Pareto configurations to {out_path}")

if __name__ == "__main__":
    run_threshold_sweep()
