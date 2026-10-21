"""
Full 45-Minute Half 1 Multi-Model Per-Class and Macro-Seam Audit
Compares local open-weight VLM models (Qwen3.5-4B vs MiniCPM-V 4.6) on game_0_half_1 (00:00 - 45:00).
Evaluates against all 65 ground-truth annotations across 12 active classes (Oracle Ceiling: 0.7059).
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent

SOCCERNET_CLASSES = [
    "Penalty", "Kick-off", "Goal", "Substitution", "Offside",
    "Shots on target", "Shots off target", "Clearance", "Ball out of play",
    "Throw-in", "Foul", "Indirect free-kick", "Direct free-kick",
    "Corner", "Yellow card", "Red card", "Yellow->red card"
]

MODEL_DISPLAY_NAMES = {
    "qwen3.5-4b_4bit_W60_half1": "Qwen3.5-4B (4-bit, W60)",
    "minicpm-v-4.6_4bit_W60_half1": "MiniCPM-V 4.6 (4-bit, W60)"
}


def analyze_model_predictions(gt_annotations: List[dict], preds: List[dict], metrics: dict):
    stats = {}
    for c in SOCCERNET_CLASSES:
        stats[c] = {
            "gt_count": 0,
            "pred_count": 0,
            "hits_5s": 0,
            "hits_30s": 0,
            "tight_ap": metrics.get("tight_per_class", {}).get(c, 0.0),
            "loose_ap": metrics.get("loose_per_class", {}).get(c, 0.0)
        }

    for gt in gt_annotations:
        lbl = gt.get("label")
        if lbl in stats:
            stats[lbl]["gt_count"] += 1

    for p in preds:
        lbl = p.get("label")
        if lbl in stats:
            stats[lbl]["pred_count"] += 1

    # Match GT to predictions (greedy nearest timestamp within tolerance)
    for c in SOCCERNET_CLASSES:
        c_gts = [g for g in gt_annotations if g.get("label") == c]
        c_preds = [p for p in preds if p.get("label") == c]

        used_p = set()
        for g in c_gts:
            g_pos = int(g.get("position", 0))
            best_diff = float("inf")
            best_idx = None
            for idx, p in enumerate(c_preds):
                if idx in used_p:
                    continue
                diff = abs(int(p.get("position", 0)) - g_pos) / 1000.0
                if diff < best_diff:
                    best_diff = diff
                    best_idx = idx
            if best_idx is not None:
                if best_diff <= 5.0:
                    stats[c]["hits_5s"] += 1
                if best_diff <= 30.0:
                    stats[c]["hits_30s"] += 1
                used_p.add(best_idx)

    return stats


def load_model_data(model_dir: Path):
    preds_file = model_dir / "Predictions-v2.json"
    metrics_file = model_dir / "evaluation_metrics.json"
    telem_file = model_dir / "telemetry.json"

    preds = json.load(open(preds_file, encoding="utf-8")).get("predictions", []) if preds_file.exists() else []
    metrics = json.load(open(metrics_file, encoding="utf-8")) if metrics_file.exists() else {}
    telem = json.load(open(telem_file, encoding="utf-8")) if telem_file.exists() else {}
    return preds, metrics, telem


def check_key_events(models_data: Dict[str, Any], gt_annotations: List[dict]):
    """Spot-checks iconic ground-truth events across models."""
    key_events = [
        {"desc": "Kick-off (Match Start)", "time": "1 - 00:00", "label": "Kick-off", "pos_ms": 0},
        {"desc": "Corner (Early Attack)", "time": "1 - 09:20", "label": "Corner", "pos_ms": 560000},
        {"desc": "Direct free-kick (Dangerous Area)", "time": "1 - 14:17", "label": "Direct free-kick", "pos_ms": 857000},
        {"desc": "Foul (Midfield Clash)", "time": "1 - 25:38", "label": "Foul", "pos_ms": 1538000},
        {"desc": "GOAL (Ander Herrera 1-0)", "time": "1 - 29:19", "label": "Goal", "pos_ms": 1759000},
        {"desc": "Foul (Late First Half)", "time": "1 - 44:04", "label": "Foul", "pos_ms": 2644000}
    ]

    print("\n" + "=" * 125)
    print("  ICONIC MATCH EVENT LOCALIZATION AUDIT (Ground Truth Spot Check)")
    print("=" * 125)
    print(f"{'Target Event':<32} | {'GT Time':<8} | {'GT Label':<16} | Model Predictions (Nearest Detection & Error)")
    print("-" * 125)

    for ke in key_events:
        target_pos = ke["pos_ms"]
        target_lbl = ke["label"]
        row = f"{ke['desc']:<32} | {ke['time']:<8} | {target_lbl:<16} | "

        model_reports = []
        for k, m in models_data.items():
            disp = m["name"].split()[0]
            preds = m["preds"]
            same_class_preds = [p for p in preds if p.get("label") == target_lbl]
            if same_class_preds:
                best = min(same_class_preds, key=lambda p: abs(int(p.get("position", 0)) - target_pos))
                err_s = (int(best.get("position", 0)) - target_pos) / 1000.0
                sign = "+" if err_s >= 0 else ""
                status = "HIT" if abs(err_s) <= 5.0 else ("NEAR" if abs(err_s) <= 30.0 else "MISS")
                model_reports.append(f"{disp}: {best.get('gameTime')} ({sign}{err_s:.2f}s) [{status}]")
            else:
                model_reports.append(f"{disp}: NONE [MISS]")

        row += "  ||  ".join(model_reports)
        print(row)
    print("=" * 125)


def main():
    parser = argparse.ArgumentParser(description="Half 1 Multi-Model Per-Class and Macro-Seam Audit")
    parser.add_argument("--eval-dir", type=str, default="logs/local_eval", help="Path to local_eval directory")
    args = parser.parse_args()

    eval_root = WORKSPACE_ROOT / args.eval_dir

    # Find Half 1 GT file
    gt_file = None
    for p in eval_root.glob("*half1/**/Labels-v2.json"):
        if "scoped_ground_truth" in str(p):
            gt_file = p
            break

    if not gt_file or not gt_file.exists():
        fallback_gt = WORKSPACE_ROOT / "test_eval_ground_truth" / "england_epl" / "2014-2015" / "2015-05-17 - 18-00 Manchester United 1 - 1 Arsenal" / "Labels-v2.json"
        with open(fallback_gt, "r", encoding="utf-8") as f:
            raw_gt = json.load(f).get("annotations", [])
        gt_annotations = [a for a in raw_gt if str(a.get("gameTime", "")).startswith("1 - ") and 0 <= int(a.get("position", 0)) <= 2700003]
    else:
        open_gt = Path("\\\\?\\" + str(gt_file.resolve())) if os.name == "nt" else gt_file
        with open(open_gt, "r", encoding="utf-8") as f:
            gt_annotations = json.load(f).get("annotations", [])

    # Select only the Half 1 model runs
    target_keys = ["qwen3.5-4b_4bit_W60_half1", "minicpm-v-4.6_4bit_W60_half1"]
    models_data = {}

    for k in target_keys:
        md = eval_root / k
        if md.exists() and (md / "evaluation_metrics.json").exists():
            preds, metrics, telem = load_model_data(md)
            stats = analyze_model_predictions(gt_annotations, preds, metrics)
            models_data[k] = {
                "name": MODEL_DISPLAY_NAMES.get(k, k),
                "preds": preds,
                "metrics": metrics,
                "telem": telem,
                "stats": stats
            }

    if not models_data:
        print("[ERROR] No evaluated Half 1 model runs found in", eval_root)
        return

    active_classes = [c for c in SOCCERNET_CLASSES if sum(1 for g in gt_annotations if g.get("label") == c) > 0]
    inactive_classes = [c for c in SOCCERNET_CLASSES if c not in active_classes]
    num_active = len(active_classes)

    print("\n" + "=" * 135)
    print(f"  FULL 45-MINUTE HALF 1 GROUND TRUTH VS MULTI-MODEL AUDIT (00:00 - 45:00 | {len(gt_annotations)} GT Events | {num_active}/17 Active Classes)")
    print("=" * 135)

    header = f"{'Class Name':<20} | {'GT':<3} | {'Status':<8}"
    for k, m in models_data.items():
        disp = m['name'][:22]
        header += f" | {disp:<24} (P/H5/H30/TAP/LAP)"
    print(header)
    print("-" * 135)

    for c in SOCCERNET_CLASSES:
        gt_c = sum(1 for g in gt_annotations if g.get("label") == c)
        c_status = "ACTIVE" if gt_c > 0 else "MASKED"
        row = f"{c:<20} | {gt_c:<3} | {c_status:<8}"
        for k, m in models_data.items():
            st = m["stats"][c]
            p_cnt = st["pred_count"]
            h5 = st["hits_5s"]
            h30 = st["hits_30s"]
            tap = st["tight_ap"]
            lap = st["loose_ap"]
            col_str = f"{p_cnt:>3}/{h5:>2}/{h30:>2}/{tap:>5.2f}/{lap:>5.2f}"
            row += f" | {col_str:<32}"
        print(row)

    print("-" * 135)
    total_row = f"{'Total Events':<20} | {len(gt_annotations):<3} | {'ALL':<8}"
    active_row = f"{'Active-Class Average':<20} | {num_active:<3} | {'ACTIVE':<8}"

    for k, m in models_data.items():
        tot_p = len(m["preds"])
        tot_h5 = sum(s["hits_5s"] for s in m["stats"].values())
        tot_h30 = sum(s["hits_30s"] for s in m["stats"].values())
        tot_tap = m["metrics"].get("tight_average_mAP", 0.0)
        tot_lap = m["metrics"].get("loose_average_mAP", 0.0)
        col_str = f"{tot_p:>3}/{tot_h5:>2}/{tot_h30:>2}/{tot_tap:>5.4f}/{tot_lap:>5.4f}"
        total_row += f" | {col_str:<32}"

        act_tap = m["metrics"].get("active_tight_average_mAP", 0.0)
        act_lap = m["metrics"].get("active_loose_average_mAP", 0.0)
        active_col = f"Act TAP: {act_tap:>5.4f} | Act LAP: {act_lap:>5.4f}"
        active_row += f" | {active_col:<32}"

    print(total_row)
    print(active_row)
    print("=" * 135)

    # Detailed Summary & Seam Suppression
    print("\n" + "=" * 115)
    print(f"  SYSTEMS & LOCALIZATION PERFORMANCE METRICS (Oracle Max: {num_active/17:.4f})")
    print("=" * 115)
    for k, m in models_data.items():
        preds = m["preds"]
        tot_p = len(preds)
        tight_map = m["metrics"].get("tight_average_mAP", 0.0)
        loose_map = m["metrics"].get("loose_average_mAP", 0.0)
        act_tap = m["metrics"].get("active_tight_average_mAP", 0.0)
        act_lap = m["metrics"].get("active_loose_average_mAP", 0.0)
        zero_gt_fp = sum(s["pred_count"] for s in m["stats"].values() if s["gt_count"] == 0)
        present_p = tot_p - zero_gt_fp
        h5 = sum(s["hits_5s"] for s in m["stats"].values())
        h30 = sum(s["hits_30s"] for s in m["stats"].values())
        ghost_pct = (zero_gt_fp / max(1, tot_p)) * 100
        over_seg = present_p / max(1, len(gt_annotations))
        avg_lat = m["telem"].get("avg_window_latency_sec", 0.0)
        stitching = m["telem"].get("stitching", {})
        raw_events = stitching.get("total_raw_events", tot_p)
        nms_suppressed = stitching.get("nms_suppressed_events", 0)

        print(f"[{m['name']}]")
        print(f"  - Total Ingested Windows: 64 windows across 5 chunks (00:00 - 45:00)")
        print(f"  - Macro Seam Stitching:   {raw_events} raw events -> {nms_suppressed} boundary duplicates suppressed -> {tot_p} kept events")
        print(f"  - Ground-Truth Recall:    {h5}/{len(gt_annotations)} within <= 5.0s ({(h5/max(1, len(gt_annotations)))*100:.1f}%) | {h30}/{len(gt_annotations)} within <= 30.0s ({(h30/max(1, len(gt_annotations)))*100:.1f}%)")
        print(f"  - Ghost False Positives:  {zero_gt_fp}/{tot_p} ({ghost_pct:.1f}% on 5 masked classes: {inactive_classes})")
        print(f"  - Event Density Ratio:    {present_p} active predictions / {len(gt_annotations)} GT ({over_seg:.2f}x density)")
        print(f"  - Official 17-Class mAP:  Tight mAP: {tight_map:.4f} ({tight_map*100:.2f}%) | Loose mAP: {loose_map:.4f} ({loose_map*100:.2f}%)")
        print(f"  - Active 12-Class mAP:    Tight mAP: {act_tap:.4f} ({act_tap*100:.2f}%) | Loose mAP: {act_lap:.4f} ({act_lap*100:.2f}%) [Ceiling: {num_active/17:.4f}]")
        print(f"  - Avg Window Latency:     {avg_lat:.2f}s per 60s window")
        print("-" * 115)

    check_key_events(models_data, gt_annotations)


if __name__ == "__main__":
    main()
