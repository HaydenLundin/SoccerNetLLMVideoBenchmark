"""
Full 90-Minute Match Aggregation and Evaluation for Game 0 (Manchester United 1 - 1 Arsenal).
Merges Half 1 and Half 2 predictions from the 3-Model Specialist Ensemble
and computes official SoccerNet Tight & Loose mAP across all 173 ground-truth events.
"""

import os
import sys
import json
import shutil
from pathlib import Path
from typing import Dict, List, Any

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from processing.unified_schema import SOCCERNET_CLASSES
from processing.post_process_predictions import DEFAULT_CLASS_NMS_RADII_MS, class_aware_nms
from processing.evaluate_soccernet import evaluate_match_action_spotting
from scripts.ablation_class_nms_and_residuals import evaluate_predictions_dataset

def load_predictions(pred_path: Path) -> List[Dict]:
    if not pred_path.exists():
        raise FileNotFoundError(f"Prediction file not found: {pred_path}")
    with open(pred_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("predictions", [])

def build_specialist_ensemble(qwen_preds: List[Dict], minicpm_preds: List[Dict], lfm_preds: List[Dict], half: int) -> List[Dict]:
    """Applies the Domain-Routed Specialist Ensemble rules for the given half."""
    # Empirical finding: Qwen3.5-4B is the true specialist for Substitutions (66.7% recall in H2,
    # 0 false positives in H1) while MiniCPM suffers from severe thumbnail compression on touchline boards.
    QWEN_SPECIALIST_CLASSES = {"Corner", "Throw-in", "Goal", "Direct free-kick", "Substitution"}
    MINICPM_SPECIALIST_CLASSES = {"Kick-off", "Foul", "Shots off target", "Clearance", "Offside", "Indirect free-kick"}
    LFM_SPECIALIST_CLASSES = {"Ball out of play"}

    # Apply temporal calibration (+2950ms to MiniCPM, -3100ms to LFM)
    cal_minicpm = []
    for p in minicpm_preds:
        p_c = dict(p)
        p_c["position"] = max(0, int(p_c.get("position", 0)) + 2950)
        cal_minicpm.append(p_c)

    cal_lfm = []
    for p in lfm_preds:
        p_c = dict(p)
        p_c["position"] = max(0, int(p_c.get("position", 0)) - 3100)
        cal_lfm.append(p_c)

    candidates = []
    for p in qwen_preds:
        if p.get("label") in QWEN_SPECIALIST_CLASSES:
            p_c = dict(p)
            p_c["half"] = str(half)
            p_c["model"] = "qwen3.5-4b"
            candidates.append(p_c)

    for p in cal_minicpm:
        if p.get("label") in MINICPM_SPECIALIST_CLASSES:
            p_c = dict(p)
            p_c["half"] = str(half)
            p_c["model"] = "minicpm-v-4.6"
            candidates.append(p_c)

    for p in cal_lfm:
        if p.get("label") in LFM_SPECIALIST_CLASSES:
            p_c = dict(p)
            p_c["half"] = str(half)
            p_c["model"] = "lfm2.5-vl-3b"
            candidates.append(p_c)

    deduped = class_aware_nms(candidates, radius_ms=3000, class_radii_ms=DEFAULT_CLASS_NMS_RADII_MS)
    return deduped

def evaluate_full_90min_match(
    h1_specialist_preds: List[Dict],
    h2_specialist_preds: List[Dict],
    output_dir: Path,
    url_local: str = "england_epl/2014-2015/2015-05-17 - 18-00 Manchester United 1 - 1 Arsenal"
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    full_preds = []

    # Tag and pool both halves
    for p in h1_specialist_preds:
        p_c = dict(p)
        p_c["half"] = "1"
        full_preds.append(p_c)

    for p in h2_specialist_preds:
        p_c = dict(p)
        p_c["half"] = "2"
        full_preds.append(p_c)

    print(f"\n[FULL MATCH] Combined {len(full_preds)} total predictions ({len(h1_specialist_preds)} H1 + {len(h2_specialist_preds)} H2)")

    # Ground truth
    gt_dir = WORKSPACE_ROOT / "test_eval_ground_truth"
    gt_file = gt_dir / url_local.strip("/") / "Labels-v2.json"
    open_gt_file = Path(f"\\\\?\\{gt_file.resolve()}") if os.name == "nt" else gt_file
    with open(open_gt_file, "r", encoding="utf-8") as f:
        gt_data = json.load(f)
    gt_annotations = gt_data.get("annotations", [])
    print(f"[FULL MATCH] Ground truth total events: {len(gt_annotations)} (65 in H1, 108 in H2)")

    # Write scoped ground truth for full match
    scoped_gt_dir = output_dir / "scoped_ground_truth"
    scoped_mirrored_gt = scoped_gt_dir / url_local.strip("/")
    if os.name == "nt":
        os.makedirs(f"\\\\?\\{scoped_mirrored_gt.resolve()}", exist_ok=True)
    else:
        scoped_mirrored_gt.mkdir(parents=True, exist_ok=True)
    labels_file = scoped_mirrored_gt / "Labels-v2.json"
    open_labels = Path(f"\\\\?\\{labels_file.resolve()}") if os.name == "nt" else labels_file
    with open(open_labels, "w", encoding="utf-8") as f:
        json.dump({"UrlLocal": f"{url_local}/", "annotations": gt_annotations}, f, indent=2)

    # DevKit evaluation
    devkit_pred_dir = output_dir / "soccernet_devkit_format"
    mirrored_dir = devkit_pred_dir / url_local.strip("/")
    if os.name == "nt":
        os.makedirs(f"\\\\?\\{mirrored_dir.resolve()}", exist_ok=True)
    else:
        mirrored_dir.mkdir(parents=True, exist_ok=True)

    spotting_file = mirrored_dir / "results_spotting.json"
    open_spot = Path(f"\\\\?\\{spotting_file.resolve()}") if os.name == "nt" else spotting_file
    with open(open_spot, "w", encoding="utf-8") as f:
        json.dump({"UrlLocal": f"{url_local}/", "predictions": full_preds}, f, indent=2)

    if os.name == "nt":
        eval_gt_dir = "\\\\?\\" + str(scoped_gt_dir.resolve())
        eval_pred_dir = "\\\\?\\" + str(devkit_pred_dir.resolve())
    else:
        eval_gt_dir = str(scoped_gt_dir)
        eval_pred_dir = str(devkit_pred_dir)

    metrics = evaluate_match_action_spotting(
        ground_truth_dir=eval_gt_dir,
        predictions_dir=eval_pred_dir,
        split="test",
        prediction_file="results_spotting.json"
    )

    # Save summary
    summary_path = output_dir / "full_match_90min_evaluation_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "match": url_local,
            "total_predictions": len(full_preds),
            "h1_predictions": len(h1_specialist_preds),
            "h2_predictions": len(h2_specialist_preds),
            "metrics": metrics
        }, f, indent=2)

    print("\n" + "=" * 90)
    print("  FULL 90-MINUTE MATCH EVALUATION RESULTS (Game 0: Man United 1 - 1 Arsenal)")
    print("=" * 90)
    print(f"  Official Tight Average-mAP (1s-5s):  {metrics.get('tight_average_mAP', 0.0):.4f}")
    print(f"  Official Loose Average-mAP (5s-60s): {metrics.get('loose_average_mAP', 0.0):.4f}")
    if "active_tight_average_mAP" in metrics:
        print(f"  Active Tight Average-mAP (1s-5s):    {metrics.get('active_tight_average_mAP', 0.0):.4f}")
        print(f"  Active Loose Average-mAP (5s-60s):   {metrics.get('active_loose_average_mAP', 0.0):.4f}")
        print(f"  Active Classes: {metrics.get('active_class_count', 0)} / 17 | Oracle Ceiling: {metrics.get('oracle_ceiling_mAP', 0.0):.4f}")
    print("=" * 90 + "\n")
    return metrics

if __name__ == "__main__":
    print("[*] Full 90-Minute Match Aggregator Ready.")
