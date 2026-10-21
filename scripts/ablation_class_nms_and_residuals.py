"""
Zero-Compute Class-Specific NMS Ablation & Signed Temporal Residual Audit
Evaluates on saved Half 1 raw_window_outputs.json for Qwen3.5-4B and MiniCPM-V 4.6.
Executes in < 5 seconds without GPU inference.
"""

import os
import sys
import json
import shutil
import statistics
from pathlib import Path
from typing import Dict, List, Tuple, Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from processing.unified_schema import parse_mm_ss_to_seconds, SOCCERNET_CLASSES
from processing.post_process_predictions import (
    stitch_chunks_to_match_predictions,
    DEFAULT_CLASS_NMS_RADII_MS,
    class_aware_nms
)
from processing.evaluate_soccernet import evaluate_match_action_spotting


def load_raw_windows_to_chunk_results(
    raw_outputs_file: Path,
    manifest: Dict
) -> List[Dict]:
    """Reconstructs chunk_results hierarchy from saved raw_window_outputs.json."""
    with open(raw_outputs_file, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    # Need to extract SpottedEvent objects per chunk
    from processing.unified_schema import SpottedEvent
    from processing.post_process_predictions import parse_mm_ss_to_seconds

    # We can parse the raw_output field using the benchmark's parser logic
    from scripts.run_local_benchmark import parse_model_json_output

    manifest_chunks = {c["chunk_index"]: c for c in manifest.get("chunks", [])}
    chunk_windows_map = {}
    for item in raw_data:
        c_idx = item.get("chunk_index", 0)
        chunk_windows_map.setdefault(c_idx, []).append(item)

    chunk_results = []
    window_sec = 60
    stride_sec = 45

    for c_idx in sorted(chunk_windows_map.keys()):
        c_meta = manifest_chunks.get(c_idx, {})
        duration_sec = float(c_meta.get("duration_sec", 615.0))
        filename = c_meta.get("filename", f"chunk_{c_idx:02d}.mp4")
        half = c_meta.get("half", 1)

        win_items = chunk_windows_map[c_idx]
        chunk_events = []

        for w_i, r_item in enumerate(win_items):
            start_t = float(w_i * stride_sec)
            eff_window_dur = min(float(window_sec), duration_sec - start_t)
            if eff_window_dur < 5.0:
                continue

            events, diag = parse_model_json_output(r_item.get("raw_output", ""), int(eff_window_dur))
            for ev in events:
                local_sec = parse_mm_ss_to_seconds(ev.timestamp)
                chunk_sec = start_t + local_sec
                c_mm = int(chunk_sec // 60)
                c_ss = int(round(chunk_sec % 60))
                ev_dict = ev.model_dump()
                ev_dict["timestamp"] = f"{c_mm:02d}:{c_ss:02d}"
                chunk_events.append(ev_dict)

        # Intra-chunk deduplication
        temp_preds = []
        for ev in chunk_events:
            sec = parse_mm_ss_to_seconds(ev["timestamp"])
            temp_preds.append({
                "half": half,
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

        chunk_results.append({
            "chunk_index": c_idx,
            "filename": filename,
            "half": half,
            "events": final_chunk_events
        })

    return chunk_results


def evaluate_predictions_dataset(
    spotting_pred: Dict,
    scoped_gt_dir: Path,
    temp_dir: Path,
    url_local: str
) -> Dict:
    """Writes results_spotting.json and runs official SoccerNet evaluation."""
    mirrored_dir = temp_dir / url_local.strip("/")
    mirrored_dir.mkdir(parents=True, exist_ok=True)
    spotting_file = mirrored_dir / "results_spotting.json"

    if os.name == "nt":
        open_file = Path("\\\\?\\" + str(spotting_file.resolve()))
        eval_gt_dir = "\\\\?\\" + str(scoped_gt_dir.resolve())
        eval_pred_dir = "\\\\?\\" + str(temp_dir.resolve())
    else:
        open_file = spotting_file
        eval_gt_dir = str(scoped_gt_dir)
        eval_pred_dir = str(temp_dir)

    with open(open_file, "w", encoding="utf-8") as f:
        json.dump(spotting_pred, f, indent=2)

    metrics = evaluate_match_action_spotting(
        ground_truth_dir=eval_gt_dir,
        predictions_dir=eval_pred_dir,
        split="test",
        prediction_file="results_spotting.json"
    )
    return metrics


def compute_signed_residuals(
    predictions: List[Dict],
    gt_annotations: List[Dict],
    tolerance_sec: float = 15.0
) -> Dict[str, Any]:
    """Computes signed temporal residuals (t_pred - t_GT) across matched events."""
    matched_residuals = []
    per_class_residuals = {c: [] for c in SOCCERNET_CLASSES}
    matched_events_log = []

    used_pred_indices = set()

    for gt in gt_annotations:
        gt_pos_ms = int(gt.get("position", 0))
        gt_lbl = gt.get("label", "")
        gt_time = gt.get("gameTime", "")

        best_p_idx = None
        best_diff_ms = float("inf")

        for p_idx, pred in enumerate(predictions):
            if p_idx in used_pred_indices:
                continue
            if pred.get("label") != gt_lbl:
                continue
            diff_ms = int(pred.get("position", 0)) - gt_pos_ms
            if abs(diff_ms) < abs(best_diff_ms):
                best_diff_ms = diff_ms
                best_p_idx = p_idx

        if best_p_idx is not None and abs(best_diff_ms) <= (tolerance_sec * 1000.0):
            used_pred_indices.add(best_p_idx)
            dt_sec = best_diff_ms / 1000.0
            matched_residuals.append(dt_sec)
            per_class_residuals[gt_lbl].append(dt_sec)
            pred_obj = predictions[best_p_idx]
            matched_events_log.append({
                "gt_time": gt_time,
                "label": gt_lbl,
                "pred_time": pred_obj.get("gameTime"),
                "dt_sec": dt_sec
            })

    if matched_residuals:
        mean_dt = statistics.mean(matched_residuals)
        median_dt = statistics.median(matched_residuals)
        std_dt = statistics.stdev(matched_residuals) if len(matched_residuals) > 1 else 0.0
    else:
        mean_dt = median_dt = std_dt = 0.0

    return {
        "matched_count": len(matched_residuals),
        "mean_dt": round(mean_dt, 3),
        "median_dt": round(median_dt, 3),
        "std_dt": round(std_dt, 3),
        "residuals": matched_residuals,
        "per_class": {c: res for c, res in per_class_residuals.items() if res},
        "log": matched_events_log
    }


def main():
    manifest_path = WORKSPACE_ROOT / "gate2_chunks" / "game_0_half_1_chunks_manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    url_local = "england_epl/2014-2015/2015-05-17 - 18-00 Manchester United 1 - 1 Arsenal/"

    models_to_test = [
        ("qwen3.5-4b", "Qwen3.5-4B (4-bit, W60)", "logs/local_eval/qwen3.5-4b_4bit_W60_half1"),
        ("minicpm-v-4.6", "MiniCPM-V 4.6 (4-bit, W60)", "logs/local_eval/minicpm-v-4.6_4bit_W60_half1")
    ]

    temp_root = WORKSPACE_ROOT / "scratch" / "nms_ablation_eval"
    if temp_root.exists():
        shutil.rmtree(temp_root, ignore_errors=True)
    temp_root.mkdir(parents=True, exist_ok=True)

    results_table = []
    residual_reports = {}

    print("\n" + "=" * 110)
    print("  ZERO-COMPUTE POST-PROCESSING ABLATION & SIGNED TEMPORAL RESIDUAL AUDIT")
    print("=" * 110)

    for model_id, model_name, model_dir_rel in models_to_test:
        model_dir = WORKSPACE_ROOT / model_dir_rel
        raw_file = model_dir / "raw_window_outputs.json"
        scoped_gt_dir = model_dir / "scoped_ground_truth"

        gt_file = scoped_gt_dir / url_local.strip("/") / "Labels-v2.json"
        open_gt = Path("\\\\?\\" + str(gt_file.resolve())) if os.name == "nt" else gt_file
        with open(open_gt, "r", encoding="utf-8") as f:
            gt_annotations = json.load(f).get("annotations", [])

        chunk_results = load_raw_windows_to_chunk_results(raw_file, manifest)

        # Baseline: Flat 3000ms
        temp_baseline = temp_root / f"{model_id}_baseline"
        spotting_base, _, _ = stitch_chunks_to_match_predictions(
            chunk_results=chunk_results,
            chunks_manifest=manifest,
            url_local=url_local,
            radius_ms=3000,
            overlap_radius_ms=5000,
            class_radii_ms=None
        )
        base_metrics = evaluate_predictions_dataset(spotting_base, scoped_gt_dir, temp_baseline, url_local)

        # Ablation: Domain-Informed Class-Specific NMS
        temp_class_nms = temp_root / f"{model_id}_class_nms"
        spotting_class, _, _ = stitch_chunks_to_match_predictions(
            chunk_results=chunk_results,
            chunks_manifest=manifest,
            url_local=url_local,
            radius_ms=3000,
            overlap_radius_ms=5000,
            class_radii_ms=DEFAULT_CLASS_NMS_RADII_MS
        )
        class_metrics = evaluate_predictions_dataset(spotting_class, scoped_gt_dir, temp_class_nms, url_local)

        # Count specific event categories before/after
        base_preds = spotting_base.get("predictions", [])
        class_preds = spotting_class.get("predictions", [])

        def get_class_counts(preds):
            c_counts = {}
            for p in preds:
                lbl = p.get("label", "")
                c_counts[lbl] = c_counts.get(lbl, 0) + 1
            return c_counts

        base_counts = get_class_counts(base_preds)
        class_counts = get_class_counts(class_preds)

        results_table.append({
            "model": model_name,
            "policy": "Flat 3s (Baseline)",
            "kept": len(base_preds),
            "goal": base_counts.get("Goal", 0),
            "kickoff": base_counts.get("Kick-off", 0),
            "shots_target": base_counts.get("Shots on target", 0),
            "corner": base_counts.get("Corner", 0),
            "foul": base_counts.get("Foul", 0),
            "tight_map": base_metrics.get("tight_average_mAP", 0.0),
            "loose_map": base_metrics.get("loose_average_mAP", 0.0),
            "act_tight_map": base_metrics.get("active_tight_average_mAP", 0.0),
            "act_loose_map": base_metrics.get("active_loose_average_mAP", 0.0),
            "per_class": base_metrics.get("loose_per_class", {})
        })

        results_table.append({
            "model": model_name,
            "policy": "Class-Specific NMS",
            "kept": len(class_preds),
            "goal": class_counts.get("Goal", 0),
            "kickoff": class_counts.get("Kick-off", 0),
            "shots_target": class_counts.get("Shots on target", 0),
            "corner": class_counts.get("Corner", 0),
            "foul": class_counts.get("Foul", 0),
            "tight_map": class_metrics.get("tight_average_mAP", 0.0),
            "loose_map": class_metrics.get("loose_average_mAP", 0.0),
            "act_tight_map": class_metrics.get("active_tight_average_mAP", 0.0),
            "act_loose_map": class_metrics.get("active_loose_average_mAP", 0.0),
            "per_class": class_metrics.get("loose_per_class", {})
        })

        # Residual analysis on baseline predictions
        residuals = compute_signed_residuals(base_preds, gt_annotations, tolerance_sec=15.0)
        residual_reports[model_name] = residuals

    # 1. Print Comparative Table
    print("\n" + "=" * 125)
    print("  TABLE 1: CLASS-SPECIFIC NMS ABLATION (Flat 3,000ms vs Domain-Informed Suppression Radii)")
    print("=" * 125)
    print(f"{'Model & Policy':<34} | {'Kept':<4} | {'Goal':<4} | {'KOff':<4} | {'ShotT':<5} | {'Foul':<4} | {'Tight mAP':<9} | {'Loose mAP':<9} | {'Act L-mAP':<9}")
    print("-" * 125)

    for r in results_table:
        row = (
            f"{r['model'] + ' (' + r['policy'].split()[0] + ')':<34} | "
            f"{r['kept']:<4} | "
            f"{r['goal']:<4} | "
            f"{r['kickoff']:<4} | "
            f"{r['shots_target']:<5} | "
            f"{r['foul']:<4} | "
            f"{r['tight_map']:<9.4f} | "
            f"{r['loose_map']:<9.4f} | "
            f"{r['act_loose_map']:<9.4f}"
        )
        print(row)
    print("=" * 125)

    # 2. Print Signed Temporal Residual Audit
    print("\n" + "=" * 115)
    print("  TABLE 2: SIGNED TEMPORAL RESIDUAL AUDIT (Delta-t = t_pred - t_GT for Matched Events <= 15.0s)")
    print("=" * 115)
    print(f"{'Model Name':<28} | {'Matched':<7} | {'Mean dt (s)':<12} | {'Median dt (s)':<14} | {'Std dt (s)':<10} | Temporal Anchoring Profile")
    print("-" * 115)

    for model_name, rep in residual_reports.items():
        profile = "Initiation / Pre-Action Bias (dt < 0)" if rep["median_dt"] < 0 else "Culmination / Reaction Bias (dt > 0)"
        print(
            f"{model_name:<28} | "
            f"{rep['matched_count']:<7} | "
            f"{rep['mean_dt']:>+10.2f}s  | "
            f"{rep['median_dt']:>+12.2f}s  | "
            f"{rep['std_dt']:>8.2f}s  | "
            f"{profile}"
        )
    print("=" * 115)

    # Print Event-Level Residual Spot Check
    print("\n" + "=" * 115)
    print("  EVENT-LEVEL SIGNED RESIDUAL COMPARISON (Side-by-Side Anchor Comparison)")
    print("=" * 115)
    print(f"{'GT Time':<8} | {'Event Label':<18} | {'Qwen3.5-4B dt (s)':<22} | {'MiniCPM-V 4.6 dt (s)':<22} | Diagnostic Interpretation")
    print("-" * 115)

    qwen_log = {f"{e['gt_time']}_{e['label']}": e for e in residual_reports["Qwen3.5-4B (4-bit, W60)"]["log"]}
    minicpm_log = {f"{e['gt_time']}_{e['label']}": e for e in residual_reports["MiniCPM-V 4.6 (4-bit, W60)"]["log"]}

    all_keys = sorted(set(list(qwen_log.keys()) + list(minicpm_log.keys())))
    for k in all_keys:
        q_item = qwen_log.get(k)
        m_item = minicpm_log.get(k)
        gt_t, lbl = k.split("_", 1)

        q_str = f"{q_item['pred_time']} ({q_item['dt_sec']:>+5.2f}s)" if q_item else "-- (No Match)"
        m_str = f"{m_item['pred_time']} ({m_item['dt_sec']:>+5.2f}s)" if m_item else "-- (No Match)"

        diag = ""
        if q_item and m_item:
            diff = q_item['dt_sec'] - m_item['dt_sec']
            diag = f"MiniCPM leads Qwen by {diff:.2f}s"
        elif q_item:
            diag = "Qwen unique detection"
        elif m_item:
            diag = "MiniCPM unique detection"

        print(f"{gt_t:<8} | {lbl:<18} | {q_str:<22} | {m_str:<22} | {diag}")
    print("=" * 115 + "\n")


if __name__ == "__main__":
    main()
