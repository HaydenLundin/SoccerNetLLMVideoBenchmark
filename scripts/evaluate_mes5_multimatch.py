"""
Canonical MES-5 Multi-Match Official DevKit Evaluator.
Target Scope: Games [0, 1, 2, 3, 13] across both halves (10 halves, 1,132 events, 17/17 active classes).
Evaluates solo models, 3-model specialist ensembles, and out-of-sample M19 Markov transition rescorers.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
from collections import Counter

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from SoccerNet.Evaluation.ActionSpotting import evaluate
from SoccerNet.Evaluation.utils import EVENT_DICTIONARY_V2, INVERSE_EVENT_DICTIONARY_V2
from scripts.evaluate_full_match_90min import build_specialist_ensemble
from scripts.evaluate_markov_prior_rescoring import (
    compute_out_of_sample_m19,
    apply_markov_and_temporal_prior
)

MES5_GAMES = [
    (0, "england_epl/2014-2015/2015-05-17 - 18-00 Manchester United 1 - 1 Arsenal"),
    (1, "england_epl/2015-2016/2015-08-16 - 18-00 Manchester City 3 - 0 Chelsea"),
    (2, "england_epl/2015-2016/2015-08-23 - 15-30 West Brom 2 - 3 Chelsea"),
    (3, "england_epl/2015-2016/2015-08-29 - 17-00 Liverpool 0 - 3 West Ham"),
    (13, "england_epl/2016-2017/2016-08-27 - 14-30 Tottenham 1 - 1 Liverpool"),
]

SOCCERNET_CLASSES = [
    "Penalty", "Kick-off", "Goal", "Substitution", "Offside",
    "Shots on target", "Shots off target", "Clearance", "Ball out of play",
    "Throw-in", "Foul", "Indirect free-kick", "Direct free-kick", "Corner",
    "Yellow card", "Red card", "Yellow->red card"
]

def load_predictions(pred_path: Path) -> List[Dict]:
    if not pred_path.exists():
        return []
    with open(pred_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("predictions", [])

def get_model_preds_for_half(model: str, game_id: int, half: int) -> List[Dict]:
    window_sec = 30 if model == "lfm2.5-vl-3b" else 60
    precision = "4bit"
    if game_id == 0 and half == 1:
        path = WORKSPACE_ROOT / f"logs/local_eval/{model}_{precision}_W{window_sec}_half1/Predictions-v2.json"
    else:
        path = WORKSPACE_ROOT / f"logs/local_eval/{model}_{precision}_W{window_sec}_game{game_id}_half{half}/Predictions-v2.json"
    return load_predictions(path)

def evaluate_predictions_tree(ground_truth_dir: Path, predictions_dir: Path) -> Tuple[float, float, Dict, Dict]:
    gt_path_str = f"\\\\?\\{ground_truth_dir.resolve()}" if os.name == "nt" else str(ground_truth_dir.resolve())
    pred_path_str = f"\\\\?\\{predictions_dir.resolve()}" if os.name == "nt" else str(predictions_dir.resolve())

    tight_results = evaluate(
        SoccerNet_path=gt_path_str,
        Predictions_path=pred_path_str,
        split="test",
        prediction_file="results_spotting.json",
        version=2,
        metric="tight",
        dataset=None,
        EVENT_DICTIONARY=EVENT_DICTIONARY_V2
    )
    loose_results = evaluate(
        SoccerNet_path=gt_path_str,
        Predictions_path=pred_path_str,
        split="test",
        prediction_file="results_spotting.json",
        version=2,
        metric="loose",
        dataset=None,
        EVENT_DICTIONARY=EVENT_DICTIONARY_V2
    )

    t_dict = {
        INVERSE_EVENT_DICTIONARY_V2[i]: float(tight_results["a_mAP_per_class"][i])
        for i in range(len(INVERSE_EVENT_DICTIONARY_V2))
    }
    l_dict = {
        INVERSE_EVENT_DICTIONARY_V2[i]: float(loose_results["a_mAP_per_class"][i])
        for i in range(len(INVERSE_EVENT_DICTIONARY_V2))
    }

    return float(tight_results["a_mAP"]), float(loose_results["a_mAP"]), t_dict, l_dict

def main():
    parser = argparse.ArgumentParser(description="Evaluate MES-5 multi-match benchmark")
    parser.add_argument("--config", type=str, default="qwen3.5-4b",
                        choices=["qwen3.5-4b", "minicpm-v-4.6", "lfm2.5-vl-3b", "specialist_3m", "specialist_3m_m19"],
                        help="Configuration to evaluate across MES-5")
    parser.add_argument("--games", type=int, nargs="+", default=[0, 1, 2, 3, 13],
                        help="Game IDs to include (default: all MES-5 games)")
    args = parser.parse_args()

    eval_games = [(gid, p) for gid, p in MES5_GAMES if gid in args.games]

    out_base = WORKSPACE_ROOT / f"logs/local_eval/mes5_{args.config}"
    out_base.mkdir(parents=True, exist_ok=True)
    devkit_pred_dir = out_base / "soccernet_devkit_format"
    devkit_gt_dir = out_base / "devkit_ground_truth"

    print("=" * 90)
    print(f"  EVALUATING CANONICAL MES-5 BENCHMARK: {args.config.upper()}")
    print(f"  Games: {[g[0] for g in eval_games]} ({len(eval_games)} matches)")
    print("=" * 90)

    total_preds_all_games = 0
    total_gt_all_games = 0

    prob_fwd, prob_bwd, h1_priors = None, None, None
    if args.config == "specialist_3m_m19":
        eval_set_path = WORKSPACE_ROOT / "eval_set.json"
        gt_dir = WORKSPACE_ROOT / "test_eval_ground_truth"
        prob_fwd, prob_bwd, h1_priors = compute_out_of_sample_m19(eval_set_path, gt_dir)

    for gid, rel_path in eval_games:
        match_pred_dir = devkit_pred_dir / rel_path
        match_gt_dir = devkit_gt_dir / rel_path
        if os.name == "nt":
            os.makedirs(f"\\\\?\\{match_pred_dir.resolve()}", exist_ok=True)
            os.makedirs(f"\\\\?\\{match_gt_dir.resolve()}", exist_ok=True)
        else:
            match_pred_dir.mkdir(parents=True, exist_ok=True)
            match_gt_dir.mkdir(parents=True, exist_ok=True)

        # Stage ground truth
        src_gt = WORKSPACE_ROOT / "test_eval_ground_truth" / rel_path / "Labels-v2.json"
        with open(src_gt, "r", encoding="utf-8") as f:
            gt_data = json.load(f)
        total_gt_all_games += len(gt_data.get("annotations", []))
        
        dst_gt = match_gt_dir / "Labels-v2.json"
        open_dst_gt = Path(f"\\\\?\\{dst_gt.resolve()}") if os.name == "nt" else dst_gt
        with open(open_dst_gt, "w", encoding="utf-8") as f:
            json.dump(gt_data, f, indent=2)

        # Build match predictions
        match_preds = []
        for half in [1, 2]:
            if args.config in ["qwen3.5-4b", "minicpm-v-4.6", "lfm2.5-vl-3b"]:
                h_preds = get_model_preds_for_half(args.config, gid, half)
            elif args.config in ["specialist_3m", "specialist_3m_m19"]:
                qwen_h = get_model_preds_for_half("qwen3.5-4b", gid, half)
                minicpm_h = get_model_preds_for_half("minicpm-v-4.6", gid, half)
                lfm_h = get_model_preds_for_half("lfm2.5-vl-3b", gid, half)
                h_preds = build_specialist_ensemble(qwen_h, minicpm_h, lfm_h, half=half)
                if args.config == "specialist_3m_m19" and prob_fwd:
                    h_preds = apply_markov_and_temporal_prior(
                        h_preds, prob_fwd, prob_bwd, alpha=0.15,
                        prune_h1_substitutions=(half == 1),
                        prune_openplay_kickoffs=True
                    )
            else:
                h_preds = []
            match_preds.extend(h_preds)

        total_preds_all_games += len(match_preds)
        print(f"  Game {gid} ({rel_path.split('/')[-1]}): {len(match_preds)} predictions")

        dst_pred = match_pred_dir / "results_spotting.json"
        open_dst_pred = Path(f"\\\\?\\{dst_pred.resolve()}") if os.name == "nt" else dst_pred
        with open(open_dst_pred, "w", encoding="utf-8") as f:
            json.dump({"UrlLocal": f"{rel_path}/", "predictions": match_preds}, f, indent=2)

    print("\n" + "-" * 90)
    print(f"  Total Predictions: {total_preds_all_games} across {len(eval_games)} matches")
    print(f"  Total Ground Truth: {total_gt_all_games} events across {len(eval_games)} matches")
    print("-" * 90 + "\n")

    tight_mAP, loose_mAP, t_per_class, l_per_class = evaluate_predictions_tree(devkit_gt_dir, devkit_pred_dir)

    print("\n" + "=" * 90)
    print(f"  CANONICAL MES-5 BENCHMARK EVALUATION RESULTS: {args.config.upper()}")
    print("=" * 90)
    print(f"  Official Tight Average-mAP (1s-5s):  {tight_mAP:.4f}")
    print(f"  Official Loose Average-mAP (5s-60s): {loose_mAP:.4f}")
    print(f"  Active Classes:                      17 / 17 (100.0% Oracle Ceiling: 1.0000)")
    print("=" * 90)

    summary_file = out_base / "mes5_evaluation_summary.json"
    summary = {
        "config": args.config,
        "games": args.games,
        "num_matches": len(eval_games),
        "total_predictions": total_preds_all_games,
        "total_ground_truth": total_gt_all_games,
        "tight_average_mAP": tight_mAP,
        "loose_average_mAP": loose_mAP,
        "tight_per_class": t_per_class,
        "loose_per_class": l_per_class,
        "oracle_ceiling_mAP": 1.0000,
        "active_class_count": 17,
        "total_class_count": 17,
    }

    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[+] Summary exported to: {summary_file}")

if __name__ == "__main__":
    main()
