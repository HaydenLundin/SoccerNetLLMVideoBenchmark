"""
Full 5-Match MES-5 Simulation: Stage 2 Window Skipping vs mAP Retention.
Evaluates the Pareto trade-off across all 10 halves of MES-5 (1,132 ground-truth events, 17 classes).
Simulates masking windows that fail the Stage 2 CPU acoustic/optical gate:
(z_whistle >= 4.5 or crowd_db >= 6.0 or opt_cut >= 30.0 or opt_var >= 6.0).
Zero GPU compute, pure CPU simulation.
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from processing.unified_schema import parse_mm_ss_to_seconds
from processing.post_process_predictions import (
    class_aware_nms,
    apply_semantic_co_emission,
    DEFAULT_CLASS_NMS_RADII_MS
)
from scripts.evaluate_full_match_90min import build_specialist_ensemble
from scripts.evaluate_markov_prior_rescoring import (
    compute_out_of_sample_m15,
    apply_markov_and_temporal_prior
)
from scripts.evaluate_mes5_multimatch import (
    MES5_GAMES,
    evaluate_predictions_tree,
    get_model_preds_for_half
)
from scripts.run_local_benchmark import (
    parse_model_json_output,
    decode_chunk_frames_single_pass
)
from scripts.test_stage2_window_skipping import (
    extract_chunk_acoustic_profile,
    compute_optical_flow_activity
)

DSP_CACHE_DIR = WORKSPACE_ROOT / "logs/local_eval/mes5_dsp_profiles"
DSP_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def get_or_compute_chunk_dsp_profile(chunk_path: Path) -> Tuple[np.ndarray, np.ndarray, List[float]]:
    """Caches and loads chunk acoustic (whistle, crowd) and optical (MAD) profiles."""
    cache_file = DSP_CACHE_DIR / f"{chunk_path.stem}_dsp.npz"
    if cache_file.exists():
        data = np.load(str(cache_file))
        return data["whistle_z"], data["crowd_db"], list(data["optical_diffs"])

    # Compute
    w_z, c_db, _ = extract_chunk_acoustic_profile(chunk_path)
    chunk_frames = decode_chunk_frames_single_pass(str(chunk_path), fps=1.0)
    opt_diffs = compute_optical_flow_activity(chunk_frames)

    np.savez_compressed(
        str(cache_file),
        whistle_z=w_z,
        crowd_db=c_db,
        optical_diffs=np.array(opt_diffs, dtype=np.float32)
    )
    return w_z, c_db, opt_diffs


def filter_half_predictions_by_dsp(
    model: str,
    game_id: int,
    half: int,
    manifest: Dict[str, Any],
    chunks_dir: Path,
    whistle_th: float = 4.5,
    crowd_th: float = 6.0,
    opt_cut_th: float = 30.0,
    opt_var_th: float = 6.0
) -> Tuple[List[Dict[str, Any]], int, int]:
    """
    Filters raw window outputs for a given model and half using the Stage 2 CPU gate,
    then re-stitches remaining predictions via class-aware NMS.
    """
    window_sec = 30 if model == "lfm2.5-vl-3b" else 60
    precision = "4bit"
    stride_sec = 20 if window_sec == 30 else 45

    if game_id == 0 and half == 1:
        folder = WORKSPACE_ROOT / f"logs/local_eval/{model}_{precision}_W{window_sec}_half1"
    else:
        folder = WORKSPACE_ROOT / f"logs/local_eval/{model}_{precision}_W{window_sec}_game{game_id}_half{half}"

    raw_file = folder / "raw_window_outputs.json"
    if not raw_file.exists():
        # Fallback to direct Predictions-v2.json if raw outputs missing
        v2_file = folder / "Predictions-v2.json"
        if v2_file.exists():
            with open(v2_file, "r", encoding="utf-8") as f:
                return json.load(f).get("predictions", []), 0, 0
        return [], 0, 0

    with open(raw_file, "r", encoding="utf-8") as f:
        raw_outputs = json.load(f)

    chunks = manifest.get("chunks", [])
    chunks_by_idx = {c["chunk_index"]: c for c in chunks}

    total_win = 0
    skipped_win = 0
    unmasked_raw_events = []

    for item in raw_outputs:
        c_idx = item["chunk_index"]
        c_meta = chunks_by_idx.get(c_idx)
        if not c_meta:
            continue

        c_file = c_meta["filename"]
        chunk_path = chunks_dir / c_file
        actual_start_ms = int(c_meta.get("actual_start_ms", 0))

        # Parse slice e.g. "0.0s - 60.0s"
        slice_str = item.get("slice", "")
        if "s -" in slice_str:
            parts = slice_str.replace("s", "").split("-")
            start_t = float(parts[0].strip())
            end_t = float(parts[1].strip())
        else:
            w_idx = item.get("window_index", 0)
            start_t = w_idx * float(stride_sec)
            end_t = start_t + float(window_sec)

        eff_dur = end_t - start_t
        total_win += 1

        # Check DSP profile
        w_z, c_db, opt_diffs = get_or_compute_chunk_dsp_profile(chunk_path)

        bin_start = int(start_t / 0.5)
        bin_end = int(np.ceil(end_t / 0.5))
        max_w = float(np.max(w_z[bin_start:bin_end])) if len(w_z) > bin_start else 0.0
        max_c = float(np.max(c_db[bin_start:bin_end])) if len(c_db) > bin_start else 0.0

        f_start = min(int(round(start_t)), len(opt_diffs) - 1) if opt_diffs else 0
        f_end = min(int(round(end_t)), len(opt_diffs)) if opt_diffs else 0
        w_d = opt_diffs[f_start:f_end] if opt_diffs else []
        max_o = float(np.max(w_d)) if w_d else 0.0
        std_o = float(np.std(w_d)) if len(w_d) > 1 else 0.0

        is_active = (max_w >= whistle_th or max_c >= crowd_th or max_o >= opt_cut_th or std_o >= opt_var_th)

        if not is_active:
            skipped_win += 1
            continue  # Skip window!

        # Parse events from raw output
        events, _ = parse_model_json_output(item.get("raw_output", ""), int(eff_dur))
        for ev in events:
            sec = parse_mm_ss_to_seconds(ev.timestamp)
            pred_offset_ms = int(round(sec * 1000.0))
            win_start_ms = int(round(start_t * 1000.0))
            global_pos_ms = actual_start_ms + win_start_ms + pred_offset_ms

            mm = global_pos_ms // 60000
            ss = (global_pos_ms % 60000) // 1000
            gt_str = f"{half} - {mm:02d}:{ss:02d}"

            unmasked_raw_events.append({
                "gameTime": gt_str,
                "label": ev.label,
                "position": global_pos_ms,
                "half": half,
                "confidence": ev.confidence,
                "commentary": ev.commentary,
                "visibility": ev.visibility
            })

    # Apply Class-Aware NMS
    stitched_preds = class_aware_nms(unmasked_raw_events, radius_ms=3000)
    return stitched_preds, total_win, skipped_win


def simulate_mes5_window_skipping():
    print("=" * 95)
    print("  MES-5 WINDOW-SKIPPING SPEED VS. mAP RETENTION SIMULATION")
    print("  Evaluates 5 Matches, 10 Halves, 1,132 Ground Truth Events, 17 Classes")
    print("=" * 95)

    eval_games = MES5_GAMES
    eval_set_path = WORKSPACE_ROOT / "eval_set.json"
    gt_dir = WORKSPACE_ROOT / "test_eval_ground_truth"
    chunks_dir = WORKSPACE_ROOT / "gate2_chunks"

    # Compute M_15 out-of-sample prior
    prob_fwd, prob_bwd, _ = compute_out_of_sample_m15(eval_set_path, gt_dir)

    # 1. Evaluate Stage 2 Filtered Predictions
    out_dir_sim = WORKSPACE_ROOT / "logs/local_eval/mes5_simulated_stage2"
    pred_dir = out_dir_sim / "soccernet_devkit_format"
    gt_eval_dir = out_dir_sim / "devkit_ground_truth"

    total_mes5_windows = 0
    total_mes5_skipped = 0

    print("\n[*] Processing all 10 halves with Stage 2 DSP Gate...")
    t0_sim = time.perf_counter()

    for gid, rel_path in eval_games:
        match_pred_dir = pred_dir / rel_path
        match_gt_dir = gt_eval_dir / rel_path
        if os.name == "nt":
            os.makedirs(f"\\\\?\\{match_pred_dir.resolve()}", exist_ok=True)
            os.makedirs(f"\\\\?\\{match_gt_dir.resolve()}", exist_ok=True)
        else:
            match_pred_dir.mkdir(parents=True, exist_ok=True)
            match_gt_dir.mkdir(parents=True, exist_ok=True)

        # Stage ground truth
        src_gt = gt_dir / rel_path / "Labels-v2.json"
        with open(src_gt, "r", encoding="utf-8") as f:
            gt_data = json.load(f)
        with open(match_gt_dir / "Labels-v2.json", "w", encoding="utf-8") as f:
            json.dump(gt_data, f, indent=2)

        match_preds = []
        for half in [1, 2]:
            manifest_file = chunks_dir / f"game_{gid}_half_{half}_chunks_manifest.json"
            with open(manifest_file, "r", encoding="utf-8") as f:
                manifest = json.load(f)

            # Filter each model
            qwen_p, t_w, s_w = filter_half_predictions_by_dsp("qwen3.5-4b", gid, half, manifest, chunks_dir)
            total_mes5_windows += t_w
            total_mes5_skipped += s_w

            minicpm_p, _, _ = filter_half_predictions_by_dsp("minicpm-v-4.6", gid, half, manifest, chunks_dir)
            lfm_p, _, _ = filter_half_predictions_by_dsp("lfm2.5-vl-3b", gid, half, manifest, chunks_dir)

            # Apply semantic co-emission on Qwen
            cap_folder = WORKSPACE_ROOT / f"logs/local_eval/qwen3.5-4b_4bit_W60_game{gid}_half{half}"
            if gid == 0 and half == 1:
                cap_folder = WORKSPACE_ROOT / "logs/local_eval/qwen3.5-4b_4bit_W60_half1"
            cap_file = cap_folder / "Predictions-Caption.json"
            if cap_file.exists():
                with open(cap_file, "r", encoding="utf-8") as cf:
                    caps = json.load(cf).get("predictions", [])
                qwen_p = apply_semantic_co_emission(qwen_p, caps)

            # Build MECE ensemble
            h_ens = build_specialist_ensemble(qwen_p, minicpm_p, lfm_p, half=half)

            # Apply M_15 Consequent Rescorer
            h_rescored = apply_markov_and_temporal_prior(
                h_ens, prob_fwd, prob_bwd, alpha=0.15,
                prune_h1_substitutions=(half == 1),
                prune_openplay_kickoffs=True
            )
            match_preds.extend(h_rescored)

        # Save match predictions
        with open(match_pred_dir / "results_spotting.json", "w", encoding="utf-8") as f:
            json.dump({"UrlLocal": f"{rel_path}/", "predictions": match_preds}, f, indent=2)

    sim_time = time.perf_counter() - t0_sim

    # 2. Evaluate with Official DevKit
    print("\n" + "=" * 95)
    print("  RUNNING OFFICIAL DEVKIT EVALUATION ON STAGE 2 FILTERED PREDICTIONS")
    print("=" * 95)
    tight_mAP, loose_mAP, t_dict, l_dict = evaluate_predictions_tree(gt_eval_dir, pred_dir)

    # Baseline comparison (Commit 73d4d35)
    baseline_loose = 0.1833
    baseline_tight = 0.0607
    retention_loose = (loose_mAP / baseline_loose) * 100.0
    retention_tight = (tight_mAP / baseline_tight) * 100.0
    skip_pct = (total_mes5_skipped / max(1, total_mes5_windows)) * 100.0
    active_classes = sum(1 for ap in l_dict.values() if ap > 0.0)

    print("\n" + "=" * 95)
    print("  DEFINITIVE MES-5 SPEED-VS-mAP PARETO COMPARISON TABLE")
    print("=" * 95)
    print(f"  Configuration                           | Loose-mAP | Tight-mAP | Active Classes | Windows Skipped | mAP Retention")
    print("-" * 95)
    print(f"  MECE Baseline (0% Skipped, 73d4d35)     |  {baseline_loose:.4f}   |  {baseline_tight:.4f}   |   17 / 17      |   0.0% (0/{total_mes5_windows})   |    100.0%")
    print(f"  Stage 2 Gate (Acoustic + Optical DSP)   |  {loose_mAP:.4f}   |  {tight_mAP:.4f}   |   {active_classes:2d} / 17      |  {skip_pct:5.1f}% ({total_mes5_skipped}/{total_mes5_windows})  |    {retention_loose:.1f}%")
    print("=" * 95)

    summary = {
        "baseline_loose_mAP": baseline_loose,
        "baseline_tight_mAP": baseline_tight,
        "stage2_loose_mAP": loose_mAP,
        "stage2_tight_mAP": tight_mAP,
        "retention_loose_pct": retention_loose,
        "retention_tight_pct": retention_tight,
        "active_classes": active_classes,
        "total_windows": total_mes5_windows,
        "skipped_windows": total_mes5_skipped,
        "skip_pct": skip_pct,
        "simulation_time_sec": sim_time,
        "per_class_loose": l_dict,
        "per_class_tight": t_dict
    }

    out_summary = WORKSPACE_ROOT / "logs/local_eval/mes5_stage2_simulation_summary.json"
    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[+] Saved simulation summary to {out_summary}")

if __name__ == "__main__":
    simulate_mes5_window_skipping()
