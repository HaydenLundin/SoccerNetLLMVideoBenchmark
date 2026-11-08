import os
import sys
import json
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from SoccerNet.Evaluation.ActionSpotting import evaluate
from SoccerNet.Evaluation.utils import EVENT_DICTIONARY_V2, INVERSE_EVENT_DICTIONARY_V2
from processing.post_process_predictions import apply_semantic_co_emission, class_aware_nms, DEFAULT_CLASS_NMS_RADII_MS
from scripts.evaluate_markov_prior_rescoring import compute_out_of_sample_m15

MES5_GAMES = [
    (0, "england_epl/2014-2015/2015-05-17 - 18-00 Manchester United 1 - 1 Arsenal"),
    (1, "england_epl/2015-2016/2015-08-16 - 18-00 Manchester City 3 - 0 Chelsea"),
    (2, "england_epl/2015-2016/2015-08-23 - 15-30 West Brom 2 - 3 Chelsea"),
    (3, "england_epl/2015-2016/2015-08-29 - 17-00 Liverpool 0 - 3 West Ham"),
    (13, "england_epl/2016-2017/2016-08-27 - 14-30 Tottenham 1 - 1 Liverpool"),
]

def load_predictions(pred_path: Path):
    if not pred_path.exists():
        return []
    with open(pred_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("predictions", [])

def get_model_preds_for_half(model: str, game_id: int, half: int, co_emission: bool = True):
    window_sec = 30 if model == "lfm2.5-vl-3b" else 60
    precision = "4bit"
    if game_id == 0 and half == 1:
        path = WORKSPACE_ROOT / f"logs/local_eval/{model}_{precision}_W{window_sec}_half1/Predictions-v2.json"
        cap_path = WORKSPACE_ROOT / f"logs/local_eval/{model}_{precision}_W{window_sec}_half1/Predictions-Caption.json"
    else:
        path = WORKSPACE_ROOT / f"logs/local_eval/{model}_{precision}_W{window_sec}_game{game_id}_half{half}/Predictions-v2.json"
        cap_path = WORKSPACE_ROOT / f"logs/local_eval/{model}_{precision}_W{window_sec}_game{game_id}_half{half}/Predictions-Caption.json"
    preds = load_predictions(path)
    if co_emission and model == "qwen3.5-4b":
        caps = load_predictions(cap_path)
        preds = apply_semantic_co_emission(preds, caps)
    return preds

def build_mece_specialist_ensemble(qwen_preds, minicpm_preds, lfm_preds, half=1):
    # Canonical 17-Class MECE Specialist Partition:
    # Qwen dominates Set-Pieces, Goals, Substitutions, Fouls, Cards, and Shots off target
    # LFM dominates Ball out of play and Shots on target
    # MiniCPM dominates Offside, Indirect free-kick, and Half 1 Clearances
    if half == 1:
        QWEN_CLASSES = {
            "Penalty", "Kick-off", "Goal", "Substitution", "Shots off target",
            "Throw-in", "Foul", "Direct free-kick", "Corner", "Yellow card",
            "Red card", "Yellow->red card"
        }
        LFM_CLASSES = {"Ball out of play", "Shots on target"}
        MINICPM_CLASSES = {"Offside", "Clearance", "Indirect free-kick"}
    else:
        # In Half 2: Qwen takes Clearance as well (superior touchline clearance perception)
        QWEN_CLASSES = {
            "Penalty", "Kick-off", "Goal", "Substitution", "Shots off target",
            "Throw-in", "Foul", "Clearance", "Direct free-kick", "Corner", "Yellow card",
            "Red card", "Yellow->red card"
        }
        LFM_CLASSES = {"Ball out of play", "Shots on target"}
        MINICPM_CLASSES = {"Offside", "Indirect free-kick"}

    # Calibration shifts: MiniCPM +2950ms, LFM -3100ms
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
        if p.get("label") in QWEN_CLASSES:
            p_c = dict(p)
            p_c["half"] = str(half)
            p_c["model"] = "qwen3.5-4b"
            candidates.append(p_c)

    for p in cal_minicpm:
        if p.get("label") in MINICPM_CLASSES:
            p_c = dict(p)
            p_c["half"] = str(half)
            p_c["model"] = "minicpm-v-4.6"
            candidates.append(p_c)

    for p in cal_lfm:
        if p.get("label") in LFM_CLASSES:
            p_c = dict(p)
            p_c["half"] = str(half)
            p_c["model"] = "lfm2.5-vl-3b"
            candidates.append(p_c)

    deduped = class_aware_nms(candidates, radius_ms=3000, class_radii_ms=DEFAULT_CLASS_NMS_RADII_MS)
    return deduped

def apply_selective_markov_prior(
    predictions,
    prob_fwd,
    prob_bwd,
    alpha=0.15,
    prune_h1_substitutions=True,
    prune_openplay_kickoffs=True
):
    # Only boost consequent classes; keep initiators at raw calibrated confidence
    CONSEQUENT_CLASSES = {
        "Ball out of play", "Corner", "Direct free-kick", "Indirect free-kick",
        "Kick-off", "Offside", "Shots on target", "Shots off target",
        "Yellow card", "Red card", "Clearance"
    }

    sorted_preds = sorted(predictions, key=lambda x: int(x.get("position", 0)))
    goal_positions = [int(p.get("position", 0)) for p in sorted_preds if p.get("label") == "Goal" and float(p.get("confidence", 0.0)) >= 0.5]

    rescored = []
    for i, p in enumerate(sorted_preds):
        p_pos = int(p.get("position", 0))
        p_lbl = p.get("label", "")
        p_half = str(p.get("half", "1"))
        orig_conf = float(p.get("confidence", 0.85))

        # Filter 1: Half 1 Substitution Ghost Pruning
        # Prunes low-confidence (<0.90) Half 1 ghost substitutions while preserving verified ones
        if prune_h1_substitutions and p_lbl == "Substitution" and p_half == "1" and orig_conf < 0.90 and p_pos <= 2700000:
            continue

        # Filter 2: Open-Play Kick-off Ghost Pruning
        if prune_openplay_kickoffs and p_lbl == "Kick-off":
            is_start = p_pos <= 60000
            is_after_goal = any(0 <= (p_pos - g_pos) <= 90000 for g_pos in goal_positions)
            if not is_start and not is_after_goal:
                continue

        # Check if class is eligible for Markov transition boost
        if p_lbl in CONSEQUENT_CLASSES:
            fwd_boost = 0.0
            for j in range(i - 1, -1, -1):
                prev_p = sorted_preds[j]
                dt = p_pos - int(prev_p.get("position", 0))
                if dt < 1000:
                    continue
                if dt > 30000:
                    break
                prev_lbl = prev_p.get("label", "")
                if prev_lbl in prob_fwd and p_lbl in prob_fwd[prev_lbl]:
                    fwd_boost = max(fwd_boost, prob_fwd[prev_lbl][p_lbl])

            bwd_boost = 0.0
            for k in range(i + 1, len(sorted_preds)):
                next_p = sorted_preds[k]
                dt = int(next_p.get("position", 0)) - p_pos
                if dt < 1000:
                    continue
                if dt > 30000:
                    break
                next_lbl = next_p.get("label", "")
                if next_lbl in prob_bwd and p_lbl in prob_bwd[next_lbl]:
                    bwd_boost = max(bwd_boost, prob_bwd[next_lbl][p_lbl])

            causal_support = max(fwd_boost, bwd_boost)
            new_conf = min(1.0, orig_conf + alpha * causal_support)
        else:
            causal_support = 0.0
            new_conf = orig_conf

        p_new = dict(p)
        p_new["confidence"] = round(new_conf, 4)
        p_new["causal_support"] = round(causal_support, 4)
        rescored.append(p_new)

    return rescored

def evaluate_predictions_tree(ground_truth_dir: Path, predictions_dir: Path):
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
    print("Testing MECE Specialist Routing on MES-5...")
    eval_set_path = WORKSPACE_ROOT / "eval_set.json"
    gt_dir = WORKSPACE_ROOT / "test_eval_ground_truth"
    prob_fwd, prob_bwd, _ = compute_out_of_sample_m15(eval_set_path, gt_dir)

    for mode in ["mece_base", "mece_m15"]:
        out_base = WORKSPACE_ROOT / f"logs/local_eval/test_{mode}"
        devkit_pred_dir = out_base / "soccernet_devkit_format"
        devkit_gt_dir = out_base / "devkit_ground_truth"

        for gid, rel_path in MES5_GAMES:
            match_pred_dir = devkit_pred_dir / rel_path
            match_gt_dir = devkit_gt_dir / rel_path
            os.makedirs(f"\\\\?\\{match_pred_dir.resolve()}", exist_ok=True)
            os.makedirs(f"\\\\?\\{match_gt_dir.resolve()}", exist_ok=True)

            src_gt = WORKSPACE_ROOT / "test_eval_ground_truth" / rel_path / "Labels-v2.json"
            with open(src_gt, "r", encoding="utf-8") as f:
                gt_data = json.load(f)
            dst_gt = match_gt_dir / "Labels-v2.json"
            with open(Path(f"\\\\?\\{dst_gt.resolve()}"), "w", encoding="utf-8") as f:
                json.dump(gt_data, f, indent=2)

            match_preds = []
            for half in [1, 2]:
                qwen_h = get_model_preds_for_half("qwen3.5-4b", gid, half)
                minicpm_h = get_model_preds_for_half("minicpm-v-4.6", gid, half)
                lfm_h = get_model_preds_for_half("lfm2.5-vl-3b", gid, half)
                h_preds = build_mece_specialist_ensemble(qwen_h, minicpm_h, lfm_h, half=half)
                if mode == "mece_m15":
                    h_preds = apply_selective_markov_prior(
                        h_preds, prob_fwd, prob_bwd, alpha=0.15,
                        prune_h1_substitutions=(half == 1),
                        prune_openplay_kickoffs=True
                    )
                match_preds.extend(h_preds)

            dst_pred = match_pred_dir / "results_spotting.json"
            with open(Path(f"\\\\?\\{dst_pred.resolve()}"), "w", encoding="utf-8") as f:
                json.dump({"UrlLocal": f"{rel_path}/", "predictions": match_preds}, f, indent=2)

        t_mAP, l_mAP, t_dict, l_dict = evaluate_predictions_tree(devkit_gt_dir, devkit_pred_dir)
        print(f"\n[{mode.upper()}] Tight-mAP: {t_mAP:.4f} | Loose-mAP: {l_mAP:.4f}")
        for cl, ap in l_dict.items():
            print(f"  {cl:<20} | Tight: {t_dict.get(cl, 0.0):.4f} | Loose: {ap:.4f}")

if __name__ == "__main__":
    main()
