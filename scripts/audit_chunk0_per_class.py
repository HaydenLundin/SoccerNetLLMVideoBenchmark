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
    "qwen3.5-4b_4bit_W60": "Qwen3.5-4B (W60)",
    "minicpm-v-4.6_4bit_W60": "MiniCPM-V 4.6 (W60)",
    "lfm2.5-vl-3b_4bit_W30": "LFM2.5-VL-3B (W30)",
    "smolvlm2-2.2b_bf16_W60": "SmolVLM2-2.2B (W60)",
    "qwen2.5-vl-7b_4bit_W15": "Qwen2.5-VL-7B (W15)",
    "moondream2_bf16_W60": "Moondream2 (W60)"
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

    # Match GT to preds
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
                if diff <= 5.0:
                    stats[c]["hits_5s"] += 1
                if diff <= 30.0:
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


def main():
    parser = argparse.ArgumentParser(description="Chunk 0 Multi-Model Per-Class and False-Positive Audit")
    parser.add_argument("--eval-dir", type=str, default="logs/local_eval", help="Path to local_eval directory")
    args = parser.parse_args()

    eval_root = WORKSPACE_ROOT / args.eval_dir

    # Find GT file from any available model dir
    gt_file = None
    for p in eval_root.glob("**/Labels-v2.json"):
        if "scoped_ground_truth" in str(p):
            gt_file = p
            break

    if not gt_file or not gt_file.exists():
        fallback_gt = WORKSPACE_ROOT / "test_eval_ground_truth" / "england_epl" / "2014-2015" / "2015-05-17 - 18-00 Manchester United 1 - 1 Arsenal" / "Labels-v2.json"
        with open(fallback_gt, "r", encoding="utf-8") as f:
            raw_gt = json.load(f).get("annotations", [])
        gt_annotations = [a for a in raw_gt if str(a.get("gameTime", "")).startswith("1") and 0 <= int(a.get("position", 0)) <= 615083]
    else:
        with open(gt_file, "r", encoding="utf-8") as f:
            gt_annotations = json.load(f).get("annotations", [])

    model_dirs = sorted([d for d in eval_root.iterdir() if d.is_dir() and (d / "evaluation_metrics.json").exists()])

    models_data = {}
    for md in model_dirs:
        key = md.name
        preds, metrics, telem = load_model_data(md)
        stats = analyze_model_predictions(gt_annotations, preds, metrics)
        models_data[key] = {
            "name": MODEL_DISPLAY_NAMES.get(key, key),
            "preds": preds,
            "metrics": metrics,
            "telem": telem,
            "stats": stats
        }

    if not models_data:
        print("[ERROR] No evaluated model runs found in", eval_root)
        return

    print("\n" + "=" * 135)
    print(f"  CHUNK 0 PER-CLASS GROUND-TRUTH VS MULTI-MODEL PREDICTION AUDIT (00:00 - 10:15 | {len(gt_annotations)} GT Events)")
    print("=" * 135)

    header = f"{'Class Name':<18} | {'GT':<3}"
    for k, m in models_data.items():
        disp = m['name'][:16]
        header += f" | {disp:<16} (P/H5/H30/LAP)"
    print(header)
    print("-" * 135)

    for c in SOCCERNET_CLASSES:
        gt_c = sum(1 for g in gt_annotations if g.get("label") == c)
        row = f"{c:<18} | {gt_c:<3}"
        for k, m in models_data.items():
            st = m["stats"][c]
            p_cnt = st["pred_count"]
            h5 = st["hits_5s"]
            h30 = st["hits_30s"]
            lap = st["loose_ap"]
            col_str = f"{p_cnt:>3}/{h5:>2}/{h30:>2}/{lap:>5.2f}"
            row += f" | {col_str:<26}"
        print(row)

    print("-" * 135)
    total_row = f"{'Total Events':<18} | {len(gt_annotations):<3}"
    active_row = f"{'Active-Class mAP':<18} | {sum(1 for c in SOCCERNET_CLASSES if any(m['stats'][c]['gt_count'] > 0 for m in models_data.values())):<3}"
    active_classes = [c for c in SOCCERNET_CLASSES if any(m['stats'][c]['gt_count'] > 0 for m in models_data.values())]
    num_active = len(active_classes)

    for k, m in models_data.items():
        tot_p = len(m["preds"])
        tot_h5 = sum(s["hits_5s"] for s in m["stats"].values())
        tot_h30 = sum(s["hits_30s"] for s in m["stats"].values())
        tot_map = m["metrics"].get("loose_average_mAP", 0.0)
        col_str = f"{tot_p:>3}/{tot_h5:>2}/{tot_h30:>2}/{tot_map:>5.4f}"
        total_row += f" | {col_str:<26}"

        active_lap = sum(m["stats"][c]["loose_ap"] for c in active_classes) / max(1, num_active)
        active_col = f"Active Loose-mAP: {active_lap:>6.4f}"
        active_row += f" | {active_col:<26}"

    print(total_row)
    print(active_row)
    print("=" * 135)

    print("\n" + "=" * 115)
    print(f"  SUMMARY FALSE-POSITIVE & SPARSITY METRICS (Active Support: {num_active}/17 Classes | Oracle Max: {num_active/17:.4f})")
    print("=" * 115)
    for k, m in models_data.items():
        preds = m["preds"]
        tot_p = len(preds)
        tight_map = m["metrics"].get("tight_average_mAP", 0.0)
        loose_map = m["metrics"].get("loose_average_mAP", 0.0)
        active_tight_map = sum(m["stats"][c]["tight_ap"] for c in active_classes) / max(1, num_active)
        active_loose_map = sum(m["stats"][c]["loose_ap"] for c in active_classes) / max(1, num_active)
        zero_gt_fp = sum(s["pred_count"] for s in m["stats"].values() if s["gt_count"] == 0)
        present_p = tot_p - zero_gt_fp
        h5 = sum(s["hits_5s"] for s in m["stats"].values())
        ghost_pct = (zero_gt_fp / max(1, tot_p)) * 100
        over_seg = present_p / max(1, len(gt_annotations))
        avg_lat = m["telem"].get("avg_window_latency_sec", 0.0)

        print(f"[{m['name']}]")
        print(f"  - Total Predictions:     {tot_p} (Target GT: {len(gt_annotations)})")
        print(f"  - Ground-Truth Hits:     {h5}/{len(gt_annotations)} within <= 5.0s ({(h5/max(1, len(gt_annotations)))*100:.1f}% Recall)")
        print(f"  - Ghost False Positives: {zero_gt_fp}/{tot_p} ({ghost_pct:.1f}% on 10 absent classes)")
        print(f"  - Present Segmentation:  {present_p} preds across 7 present classes ({over_seg:.1f}x density)")
        print(f"  - Official 17-Class:     Tight mAP: {tight_map:.4f} | Loose mAP: {loose_map:.4f}")
        print(f"  - Active-Class ({num_active} cls):    Tight mAP: {active_tight_map:.4f} | Loose mAP: {active_loose_map:.4f} (Ceiling: {num_active/17:.4f})")
        print(f"  - Avg Window Latency:    {avg_lat:.2f}s")
        print("-" * 115)


if __name__ == "__main__":
    main()
