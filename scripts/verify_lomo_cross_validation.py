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

def evaluate_subset(game_tuples, model_preds_dict, scratch_tag="eval"):
    out_base = WORKSPACE_ROOT / f"logs/local_eval/scratch_{scratch_tag}"
    devkit_pred_dir = out_base / "soccernet_devkit_format"
    devkit_gt_dir = out_base / "devkit_ground_truth"

    for gid, rel_path in game_tuples:
        match_pred_dir = devkit_pred_dir / rel_path
        match_gt_dir = devkit_gt_dir / rel_path
        os.makedirs(f"\\\\?\\{match_pred_dir.resolve()}", exist_ok=True)
        os.makedirs(f"\\\\?\\{match_gt_dir.resolve()}", exist_ok=True)

        src_gt = WORKSPACE_ROOT / "test_eval_ground_truth" / rel_path / "Labels-v2.json"
        with open(src_gt, "r", encoding="utf-8") as f:
            gt_data = json.load(f)
        with open(Path(f"\\\\?\\{(match_gt_dir / 'Labels-v2.json').resolve()}"), "w", encoding="utf-8") as f:
            json.dump(gt_data, f, indent=2)

        preds = model_preds_dict.get(gid, [])
        with open(Path(f"\\\\?\\{(match_pred_dir / 'results_spotting.json').resolve()}"), "w", encoding="utf-8") as f:
            json.dump({"UrlLocal": f"{rel_path}/", "predictions": preds}, f, indent=2)

    gt_path_str = f"\\\\?\\{devkit_gt_dir.resolve()}" if os.name == "nt" else str(devkit_gt_dir.resolve())
    pred_path_str = f"\\\\?\\{devkit_pred_dir.resolve()}" if os.name == "nt" else str(devkit_pred_dir.resolve())

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
    print("=" * 80)
    print("  LEAVE-ONE-MATCH-OUT (LOMO) 5-FOLD CROSS-VALIDATION ON MES-5")
    print("=" * 80)

    # Pre-cache all predictions per model and game
    models = ["qwen3.5-4b", "lfm2.5-vl-3b", "minicpm-v-4.6"]
    game_preds = {m: {} for m in models}

    for m in models:
        for gid, rel_path in MES5_GAMES:
            m_preds = []
            for half in [1, 2]:
                h_preds = get_model_preds_for_half(m, gid, half)
                # apply calibration shifts
                if m == "minicpm-v-4.6":
                    for p in h_preds:
                        p_c = dict(p)
                        p_c["position"] = max(0, int(p_c.get("position", 0)) + 2950)
                        p_c["half"] = str(half)
                        m_preds.append(p_c)
                elif m == "lfm2.5-vl-3b":
                    for p in h_preds:
                        p_c = dict(p)
                        p_c["position"] = max(0, int(p_c.get("position", 0)) - 3100)
                        p_c["half"] = str(half)
                        m_preds.append(p_c)
                else:
                    for p in h_preds:
                        p_c = dict(p)
                        p_c["half"] = str(half)
                        m_preds.append(p_c)
            game_preds[m][gid] = m_preds

    # Evaluate each fold
    lomo_test_preds = []
    
    print("\nDetermining optimal class routing per fold (training on 4 matches, evaluating on held-out match):")
    for fold_idx, held_out in enumerate(MES5_GAMES):
        held_out_gid, held_out_path = held_out
        train_games = [g for g in MES5_GAMES if g[0] != held_out_gid]

        # Compute per-class loose-AP on the 4 training matches for each model
        train_l_maps = {}
        for m in models:
            _, _, _, l_dict = evaluate_subset(train_games, game_preds[m], scratch_tag=f"fold_{fold_idx}_{m}")
            train_l_maps[m] = l_dict

        # Determine best model per class on train set
        best_routing = {}
        for cl in EVENT_DICTIONARY_V2.keys():
            scores = {m: train_l_maps[m].get(cl, 0.0) for m in models}
            best_m = max(scores, key=scores.get)
            best_routing[cl] = (best_m, scores[best_m])

        print(f"\n--- Fold {fold_idx + 1}/5: Held-out Game {held_out_gid} ---")
        qwen_cls = [cl for cl, (m, s) in best_routing.items() if m == "qwen3.5-4b" and s > 0]
        lfm_cls = [cl for cl, (m, s) in best_routing.items() if m == "lfm2.5-vl-3b" and s > 0]
        minicpm_cls = [cl for cl, (m, s) in best_routing.items() if m == "minicpm-v-4.6" and s > 0]
        print(f"  Qwen Assigned ({len(qwen_cls)}): {qwen_cls}")
        print(f"  LFM Assigned ({len(lfm_cls)}): {lfm_cls}")
        print(f"  MiniCPM Assigned ({len(minicpm_cls)}): {minicpm_cls}")

        # Assemble predictions for the held-out match using this out-of-sample routing
        held_out_candidates = []
        for cl, (best_m, _) in best_routing.items():
            for p in game_preds[best_m][held_out_gid]:
                if p.get("label") == cl:
                    held_out_candidates.append(dict(p))
        
        held_out_deduped = class_aware_nms(held_out_candidates, radius_ms=3000, class_radii_ms=DEFAULT_CLASS_NMS_RADII_MS)
        lomo_test_preds.append((held_out_gid, held_out_path, held_out_deduped))

    # Evaluate the aggregated out-of-sample predictions across all 5 folds
    lomo_dict = {gid: preds for gid, _, preds in lomo_test_preds}
    t_mAP, l_mAP, t_dict, l_dict = evaluate_subset(MES5_GAMES, lomo_dict, scratch_tag="lomo_final")

    print("\n" + "=" * 80)
    print("  FINAL 5-FOLD LOMO CROSS-VALIDATION RESULTS ACROSS MES-5")
    print("=" * 80)
    print(f"  Official Tight-mAP: {t_mAP:.4f} ({t_mAP*100:.2f}%)")
    print(f"  Official Loose-mAP: {l_mAP:.4f} ({l_mAP*100:.2f}%)")
    print("=" * 80)
    for cl in sorted(l_dict.keys()):
        print(f"  {cl:<20}: Tight = {t_dict.get(cl, 0.0):.4f} | Loose = {l_dict.get(cl, 0.0):.4f}")

if __name__ == "__main__":
    main()
