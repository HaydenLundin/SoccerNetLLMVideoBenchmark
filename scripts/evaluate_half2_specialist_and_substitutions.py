"""
Half 2 Specialist Ensemble & Second-Half Substitution Detection Benchmark.
Evaluates:
1. 3-Model Domain-Routed Specialist Ensemble on Game 0 Half 2.
2. Second-Half Substitution Law validation: Evaluates model recall and precision
   against the 6 official ground-truth substitutions in Half 2.
3. Out-of-sample M19 Markov transition & Half-2 temporal prior rescoring.
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from processing.unified_schema import SOCCERNET_CLASSES
from processing.post_process_predictions import DEFAULT_CLASS_NMS_RADII_MS, class_aware_nms
from processing.evaluate_soccernet import evaluate_match_action_spotting
from scripts.evaluate_full_match_90min import build_specialist_ensemble, load_predictions, evaluate_full_90min_match
from scripts.evaluate_markov_prior_rescoring import compute_out_of_sample_m19, apply_markov_and_temporal_prior

def audit_substitution_detections(
    predictions: List[Dict],
    gt_substitutions: List[Dict],
    tolerance_sec: float = 30.0
) -> Dict[str, Any]:
    """Audits detection accuracy on the 6 ground-truth substitutions in Half 2."""
    tol_ms = tolerance_sec * 1000.0
    detected_hits = []
    used_pred_indices = set()

    for gt in gt_substitutions:
        gt_pos = int(gt.get("position", 0))
        gt_time = gt.get("gameTime", "")
        best_diff = float("inf")
        best_p_idx = None

        for p_idx, p in enumerate(predictions):
            if p_idx in used_pred_indices or p.get("label") != "Substitution":
                continue
            diff = abs(int(p.get("position", 0)) - gt_pos)
            if diff < best_diff:
                best_diff = diff
                best_p_idx = p_idx

        if best_p_idx is not None and best_diff <= tol_ms:
            used_pred_indices.add(best_p_idx)
            detected_hits.append({
                "gt_time": gt_time,
                "gt_pos_ms": gt_pos,
                "pred_time": predictions[best_p_idx].get("gameTime", ""),
                "pred_pos_ms": int(predictions[best_p_idx].get("position", 0)),
                "error_sec": round((int(predictions[best_p_idx].get("position", 0)) - gt_pos) / 1000.0, 2),
                "model": predictions[best_p_idx].get("model", "unknown"),
                "confidence": predictions[best_p_idx].get("confidence", 0.0)
            })

    total_sub_preds = sum(1 for p in predictions if p.get("label") == "Substitution")
    recall = len(detected_hits) / max(1, len(gt_substitutions))
    precision = len(detected_hits) / max(1, total_sub_preds)

    return {
        "gt_substitution_count": len(gt_substitutions),
        "pred_substitution_count": total_sub_preds,
        "matched_hits_count": len(detected_hits),
        "recall_at_tol": recall,
        "precision_at_tol": precision,
        "tolerance_sec": tolerance_sec,
        "hits": detected_hits
    }

def run_half2_and_full_match_evaluation():
    url_local = "england_epl/2014-2015/2015-05-17 - 18-00 Manchester United 1 - 1 Arsenal"
    gt_file = WORKSPACE_ROOT / "test_eval_ground_truth" / url_local / "Labels-v2.json"
    with open(gt_file, "r", encoding="utf-8") as f:
        gt_data = json.load(f)
    gt_annotations = gt_data.get("annotations", [])

    h1_gt = [a for a in gt_annotations if str(a.get("gameTime", "")).startswith("1 - ")]
    h2_gt = [a for a in gt_annotations if str(a.get("gameTime", "")).startswith("2 - ")]
    h2_subs_gt = [a for a in h2_gt if a.get("label") == "Substitution"]

    print("\n" + "=" * 90)
    print("  HALF 2 & FULL 90-MINUTE MATCH BENCHMARK EVALUATOR")
    print("=" * 90)
    print(f"  Ground Truth: {len(gt_annotations)} Total Events ({len(h1_gt)} in H1, {len(h2_gt)} in H2)")
    print(f"  Half 2 Ground Truth Substitutions: {len(h2_subs_gt)} events (Validating Second-Half Substitution Law)")

    # 1. Load Half 2 predictions
    qwen_h2_file = WORKSPACE_ROOT / "logs/local_eval/qwen3.5-4b_4bit_W60_game0_half2/Predictions-v2.json"
    minicpm_h2_file = WORKSPACE_ROOT / "logs/local_eval/minicpm-v-4.6_4bit_W60_game0_half2/Predictions-v2.json"
    lfm_h2_file = WORKSPACE_ROOT / "logs/local_eval/lfm2.5-vl-3b_4bit_W30_game0_half2/Predictions-v2.json"

    qwen_h2 = load_predictions(qwen_h2_file) if qwen_h2_file.exists() else []
    minicpm_h2 = load_predictions(minicpm_h2_file) if minicpm_h2_file.exists() else []
    lfm_h2 = load_predictions(lfm_h2_file) if lfm_h2_file.exists() else []

    print(f"  Loaded Predictions: Qwen3.5-4B={len(qwen_h2)}, MiniCPM-V 4.6={len(minicpm_h2)}, LFM2.5-VL-3B={len(lfm_h2)}")

    # 2. Build Specialist Ensemble for Half 2
    h2_specialist = build_specialist_ensemble(qwen_h2, minicpm_h2, lfm_h2, half=2)
    print(f"  [HALF 2 ENSEMBLE] Kept {len(h2_specialist)} post-NMS specialist predictions for Half 2.")

    # 3. Audit Substitution Detections on Half 2
    sub_audit_specialist = audit_substitution_detections(h2_specialist, h2_subs_gt, tolerance_sec=30.0)
    sub_audit_minicpm = audit_substitution_detections(minicpm_h2, h2_subs_gt, tolerance_sec=30.0) if minicpm_h2 else {}
    sub_audit_qwen = audit_substitution_detections(qwen_h2, h2_subs_gt, tolerance_sec=30.0) if qwen_h2 else {}

    print("\n" + "-" * 90)
    print("  SECOND-HALF SUBSTITUTION LAW VERIFICATION (Game 0 Half 2)")
    print("-" * 90)
    print(f"  Target Ground-Truth Substitutions: {len(h2_subs_gt)}")
    print(f"  Specialist Ensemble Detected:      {sub_audit_specialist['matched_hits_count']}/{len(h2_subs_gt)} (Recall: {sub_audit_specialist['recall_at_tol']*100:.1f}%)")
    print(f"  Specialist Substitution Precision: {sub_audit_specialist['precision_at_tol']*100:.1f}% ({sub_audit_specialist['matched_hits_count']}/{sub_audit_specialist['pred_substitution_count']})")
    for hit in sub_audit_specialist["hits"]:
        print(f"    -> [MATCHED] GT: {hit['gt_time']} vs Pred: {hit['pred_time']} (Error: {hit['error_sec']:+.2f}s, Model: {hit['model']})")

    # 4. Evaluate Half 2 Action Spotting with Official DevKit
    output_dir_h2 = WORKSPACE_ROOT / "logs/local_eval/specialist_3m_game0_half2"
    output_dir_h2.mkdir(parents=True, exist_ok=True)

    devkit_pred_dir_h2 = output_dir_h2 / "soccernet_devkit_format"
    mirrored_dir_h2 = devkit_pred_dir_h2 / url_local.strip("/")
    if os.name == "nt":
        os.makedirs(f"\\\\?\\{mirrored_dir_h2.resolve()}", exist_ok=True)
    else:
        mirrored_dir_h2.mkdir(parents=True, exist_ok=True)

    res_file_h2 = mirrored_dir_h2 / "results_spotting.json"
    open_res_h2 = Path(f"\\\\?\\{res_file_h2.resolve()}") if os.name == "nt" else res_file_h2
    with open(open_res_h2, "w", encoding="utf-8") as f:
        json.dump({"UrlLocal": f"{url_local}/", "predictions": h2_specialist}, f, indent=2)

    # Scoped ground truth for Half 2
    scoped_gt_dir_h2 = output_dir_h2 / "scoped_ground_truth"
    scoped_mirrored_h2 = scoped_gt_dir_h2 / url_local.strip("/")
    if os.name == "nt":
        os.makedirs(f"\\\\?\\{scoped_mirrored_h2.resolve()}", exist_ok=True)
    else:
        scoped_mirrored_h2.mkdir(parents=True, exist_ok=True)

    labels_file_h2 = scoped_mirrored_h2 / "Labels-v2.json"
    open_labels_h2 = Path(f"\\\\?\\{labels_file_h2.resolve()}") if os.name == "nt" else labels_file_h2
    with open(open_labels_h2, "w", encoding="utf-8") as f:
        json.dump({"UrlLocal": f"{url_local}/", "annotations": h2_gt}, f, indent=2)

    if os.name == "nt":
        eval_gt_dir_h2 = "\\\\?\\" + str(scoped_gt_dir_h2.resolve())
        eval_pred_dir_h2 = "\\\\?\\" + str(devkit_pred_dir_h2.resolve())
    else:
        eval_gt_dir_h2 = str(scoped_gt_dir_h2)
        eval_pred_dir_h2 = str(devkit_pred_dir_h2)

    metrics_h2 = evaluate_match_action_spotting(
        ground_truth_dir=eval_gt_dir_h2,
        predictions_dir=eval_pred_dir_h2,
        split="test",
        prediction_file="results_spotting.json"
    )

    print("\n" + "-" * 90)
    print("  HALF 2 SPECIALIST ENSEMBLE OFFICIAL DEVKIT RESULTS")
    print("-" * 90)
    print(f"  Official Tight Average-mAP (1s-5s):  {metrics_h2.get('tight_average_mAP', 0.0):.4f}")
    print(f"  Official Loose Average-mAP (5s-60s): {metrics_h2.get('loose_average_mAP', 0.0):.4f}")
    if "active_tight_average_mAP" in metrics_h2:
        print(f"  Active Tight Average-mAP (1s-5s):    {metrics_h2.get('active_tight_average_mAP', 0.0):.4f}")
        print(f"  Active Loose Average-mAP (5s-60s):   {metrics_h2.get('active_loose_average_mAP', 0.0):.4f}")
        print(f"  Active Classes: {metrics_h2.get('active_class_count', 0)} / 17 | Oracle Ceiling: {metrics_h2.get('oracle_ceiling_mAP', 0.0):.4f}")

    # 5. Full 90-Minute Match Aggregation (Half 1 + Half 2)
    # Load established Half 1 specialist predictions or build directly from Half 1 models
    h1_specialist_file = WORKSPACE_ROOT / "logs/local_eval/specialist_3m/Predictions-v2.json"
    if not h1_specialist_file.exists():
        h1_specialist_file = WORKSPACE_ROOT / "logs/local_eval/specialist_3m_half1/Predictions-v2.json"
    if not h1_specialist_file.exists():
        h1_specialist_file = WORKSPACE_ROOT / "scratch/synthesis_eval/specialist_3m/Predictions-v2.json"

    if h1_specialist_file.exists():
        h1_specialist_preds = load_predictions(h1_specialist_file)
        print(f"\n  Loaded Half 1 Specialist Predictions: {len(h1_specialist_preds)} events from {h1_specialist_file.name}")
    else:
        print("\n  Building Half 1 Specialist Predictions from saved Half 1 model predictions...")
        qwen_h1 = load_predictions(WORKSPACE_ROOT / "logs/local_eval/qwen3.5-4b_4bit_W60_half1/Predictions-v2.json")
        minicpm_h1 = load_predictions(WORKSPACE_ROOT / "logs/local_eval/minicpm-v-4.6_4bit_W60_half1/Predictions-v2.json")
        lfm_h1 = load_predictions(WORKSPACE_ROOT / "logs/local_eval/lfm2.5-vl-3b_4bit_W30_half1/Predictions-v2.json")
        h1_specialist_preds = build_specialist_ensemble(qwen_h1, minicpm_h1, lfm_h1, half=1)
        h1_spec_dir = WORKSPACE_ROOT / "logs/local_eval/specialist_3m_half1"
        h1_spec_dir.mkdir(parents=True, exist_ok=True)
        with open(h1_spec_dir / "Predictions-v2.json", "w", encoding="utf-8") as f:
            json.dump({"UrlLocal": f"{url_local}/", "predictions": h1_specialist_preds}, f, indent=2)
        print(f"  [HALF 1 ENSEMBLE] Built and saved {len(h1_specialist_preds)} specialist predictions for Half 1.")

    output_dir_full = WORKSPACE_ROOT / "logs/local_eval/full_match_90min_specialist_3m"
    metrics_full = evaluate_full_90min_match(
        h1_specialist_preds=h1_specialist_preds,
        h2_specialist_preds=h2_specialist,
        output_dir=output_dir_full,
        url_local=url_local
    )

    # 6. Out-of-Sample M19 Markov Transition & Prior Rescoring
    print("\n" + "=" * 90)
    print("  STAGE 4: OUT-OF-SAMPLE 19-MATCH MARKOV RESCORING ON HALF 2 & FULL MATCH")
    print("=" * 90)
    eval_set_path = WORKSPACE_ROOT / "eval_set.json"
    gt_dir = WORKSPACE_ROOT / "test_eval_ground_truth"
    prob_fwd, prob_bwd, h1_priors = compute_out_of_sample_m19(eval_set_path, gt_dir)

    # Rescore Half 2
    h2_rescored = apply_markov_and_temporal_prior(
        h2_specialist, prob_fwd, prob_bwd, alpha=0.15,
        prune_h1_substitutions=False, # Half 2 has substitutions!
        prune_openplay_kickoffs=True
    )
    output_dir_h2_m19 = WORKSPACE_ROOT / "logs/local_eval/specialist_3m_m19_game0_half2"
    output_dir_h2_m19.mkdir(parents=True, exist_ok=True)
    devkit_pred_dir_h2_m19 = output_dir_h2_m19 / "soccernet_devkit_format"
    mirrored_dir_h2_m19 = devkit_pred_dir_h2_m19 / url_local.strip("/")
    if os.name == "nt":
        os.makedirs(f"\\\\?\\{mirrored_dir_h2_m19.resolve()}", exist_ok=True)
    else:
        mirrored_dir_h2_m19.mkdir(parents=True, exist_ok=True)

    res_file_h2_m19 = mirrored_dir_h2_m19 / "results_spotting.json"
    open_res_h2_m19 = Path(f"\\\\?\\{res_file_h2_m19.resolve()}") if os.name == "nt" else res_file_h2_m19
    with open(open_res_h2_m19, "w", encoding="utf-8") as f:
        json.dump({"UrlLocal": f"{url_local}/", "predictions": h2_rescored}, f, indent=2)

    metrics_h2_m19 = evaluate_match_action_spotting(
        ground_truth_dir=eval_gt_dir_h2,
        predictions_dir="\\\\?\\" + str(devkit_pred_dir_h2_m19.resolve()) if os.name == "nt" else str(devkit_pred_dir_h2_m19),
        split="test",
        prediction_file="results_spotting.json"
    )

    # Rescore Half 1 + Half 2 for Full Match
    h1_rescored = apply_markov_and_temporal_prior(
        h1_specialist_preds, prob_fwd, prob_bwd, alpha=0.15,
        prune_h1_substitutions=True,
        prune_openplay_kickoffs=True
    )
    output_dir_full_m19 = WORKSPACE_ROOT / "logs/local_eval/full_match_90min_specialist_3m_m19"
    metrics_full_m19 = evaluate_full_90min_match(
        h1_specialist_preds=h1_rescored,
        h2_specialist_preds=h2_rescored,
        output_dir=output_dir_full_m19,
        url_local=url_local
    )

    # Save complete summary
    summary_file = WORKSPACE_ROOT / "logs/local_eval/half2_and_full_match_evaluation_summary.json"
    summary_data = {
        "match": url_local,
        "half2_metrics": metrics_h2,
        "half2_m19_metrics": metrics_h2_m19,
        "substitution_audit": sub_audit_specialist,
        "substitution_audit_minicpm": sub_audit_minicpm,
        "substitution_audit_qwen": sub_audit_qwen,
        "full_match_90min_metrics": metrics_full,
        "full_match_90min_m19_metrics": metrics_full_m19
    }
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    print("\n" + "=" * 90)
    print("  FINAL 90-MINUTE MATCH BENCHMARK SUMMARY (Game 0 Full Match)")
    print("=" * 90)
    print(f"  Half 2 Base Specialist:     Tight-mAP={metrics_h2['tight_average_mAP']:.4f} | Loose-mAP={metrics_h2['loose_average_mAP']:.4f} | Active Loose={metrics_h2.get('active_loose_average_mAP', 0):.4f}")
    print(f"  Half 2 M19 Rescored:        Tight-mAP={metrics_h2_m19['tight_average_mAP']:.4f} | Loose-mAP={metrics_h2_m19['loose_average_mAP']:.4f} | Active Loose={metrics_h2_m19.get('active_loose_average_mAP', 0):.4f}")
    print(f"  Full 90m Base Specialist:   Tight-mAP={metrics_full['tight_average_mAP']:.4f} | Loose-mAP={metrics_full['loose_average_mAP']:.4f} | Active Loose={metrics_full.get('active_loose_average_mAP', 0):.4f}")
    print(f"  Full 90m M19 Rescored:      Tight-mAP={metrics_full_m19['tight_average_mAP']:.4f} | Loose-mAP={metrics_full_m19['loose_average_mAP']:.4f} | Active Loose={metrics_full_m19.get('active_loose_average_mAP', 0):.4f}")
    print(f"  Substitutions (Second-Half): {sub_audit_specialist['matched_hits_count']}/{sub_audit_specialist['gt_substitution_count']} ({sub_audit_specialist['recall_at_tol']*100:.1f}% Recall, {sub_audit_specialist['precision_at_tol']*100:.1f}% Precision)")
    print("=" * 90 + "\n")

    print(f"\n[+] Complete summary exported to: {summary_file}")
    return summary_data

if __name__ == "__main__":
    run_half2_and_full_match_evaluation()
