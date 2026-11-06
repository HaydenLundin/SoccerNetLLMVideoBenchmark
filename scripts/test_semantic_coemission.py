"""
Test semantic co-emission recovery for Penalty and Yellow->red card on Qwen3.5-4B MES-5 predictions.
"""

import os
import json
from pathlib import Path
from typing import List, Dict
from copy import deepcopy

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent

from SoccerNet.Evaluation.ActionSpotting import evaluate
from SoccerNet.Evaluation.utils import EVENT_DICTIONARY_V2, INVERSE_EVENT_DICTIONARY_V2

MES5_GAMES = [
    (0, "england_epl/2014-2015/2015-05-17 - 18-00 Manchester United 1 - 1 Arsenal"),
    (1, "england_epl/2015-2016/2015-08-16 - 18-00 Manchester City 3 - 0 Chelsea"),
    (2, "england_epl/2015-2016/2015-08-23 - 15-30 West Brom 2 - 3 Chelsea"),
    (3, "england_epl/2015-2016/2015-08-29 - 17-00 Liverpool 0 - 3 West Ham"),
    (13, "england_epl/2016-2017/2016-08-27 - 14-30 Tottenham 1 - 1 Liverpool"),
]

def apply_semantic_co_emission(predictions: List[Dict], captions: List[Dict]) -> List[Dict]:
    """
    Applies zero-compute semantic co-emission for compound/multi-label events:
    1. Penalty: Converted/saved penalties emit Penalty alongside Goal / Direct free-kick when penalty context is present.
    2. Yellow->red card: When a yellow card is shown and followed by ejection/dismissal cues or dual card signals.
    """
    pos_to_caption = {}
    for cap in captions:
        pos = int(cap.get("position", 0))
        txt = cap.get("commentary", "")
        # Map within 10s
        pos_to_caption[pos] = txt

    new_preds = list(predictions)

    # Sort predictions by position
    sorted_preds = sorted(predictions, key=lambda x: int(x.get("position", 0)))

    for i, p in enumerate(sorted_preds):
        pos = int(p.get("position", 0))
        lbl = p.get("label", "")
        conf = float(p.get("confidence", 0.9))
        half = str(p.get("half", "1"))
        gt_time = p.get("gameTime", "")

        # Find closest caption text within 25s
        caption_text = ""
        for c_pos, c_txt in pos_to_caption.items():
            if abs(c_pos - pos) <= 25000:
                caption_text += " " + c_txt.lower()

        # Rule 1: Penalty Co-Emission
        # Triggered when Direct free-kick, Goal, or Shots on target has explicit penalty cues
        is_penalty_cue = any(k in caption_text for k in [
            "penalty kick", "penalty spot", "spot kick", "spot-kick", "penalty area"
        ])

        if lbl in ["Direct free-kick", "Goal", "Shots off target"] and is_penalty_cue:
            if any(k in caption_text for k in ["takes a direct free-kick from the penalty", "takes a free-kick from the penalty", "penalty kick"]):
                penalty_cand = deepcopy(p)
                penalty_cand["label"] = "Penalty"
                penalty_cand["confidence"] = min(0.95, conf)
                new_preds.append(penalty_cand)

        # Rule 2: Yellow->red card Co-Emission
        # Triggered when a Yellow card has second-yellow / dismissal / frustration cues
        if lbl == "Yellow card":
            is_dismissal_cue = any(k in caption_text for k in [
                "second yellow", "sent off", "dismissed", "reacts with frustration", "tunnel", "red card"
            ])
            if is_dismissal_cue:
                yr_cand = deepcopy(p)
                yr_cand["label"] = "Yellow->red card"
                yr_cand["confidence"] = min(0.95, conf)
                new_preds.append(yr_cand)

    # Sort and return
    new_preds.sort(key=lambda x: int(x.get("position", 0)))
    return new_preds

def main():
    out_dir = WORKSPACE_ROOT / "logs/local_eval/mes5_qwen3.5-4b_coemission"
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_dir = out_dir / "soccernet_devkit_format"
    gt_dir = out_dir / "devkit_ground_truth"

    for gid, rel_path in MES5_GAMES:
        m_pred_dir = pred_dir / rel_path
        m_gt_dir = gt_dir / rel_path
        if os.name == "nt":
            os.makedirs(f"\\\\?\\{m_pred_dir.resolve()}", exist_ok=True)
            os.makedirs(f"\\\\?\\{m_gt_dir.resolve()}", exist_ok=True)
        else:
            m_pred_dir.mkdir(parents=True, exist_ok=True)
            m_gt_dir.mkdir(parents=True, exist_ok=True)

        # Ground truth
        src_gt = WORKSPACE_ROOT / "test_eval_ground_truth" / rel_path / "Labels-v2.json"
        gt_data = json.load(open(src_gt, 'r', encoding='utf-8'))
        open_gt = Path(f"\\\\?\\{m_gt_dir.resolve()}") / "Labels-v2.json" if os.name == "nt" else m_gt_dir / "Labels-v2.json"
        with open(open_gt, "w", encoding="utf-8") as f:
            json.dump(gt_data, f, indent=2)

        # Predictions
        match_preds = []
        for half in [1, 2]:
            p_file = WORKSPACE_ROOT / f"logs/local_eval/qwen3.5-4b_4bit_W60_game{gid}_half{half}/Predictions-v2.json"
            c_file = WORKSPACE_ROOT / f"logs/local_eval/qwen3.5-4b_4bit_W60_game{gid}_half{half}/Predictions-Caption.json"
            if gid == 0 and half == 1:
                p_file = WORKSPACE_ROOT / "logs/local_eval/qwen3.5-4b_4bit_W60_half1/Predictions-v2.json"
                c_file = WORKSPACE_ROOT / "logs/local_eval/qwen3.5-4b_4bit_W60_half1/Predictions-Caption.json"
            if p_file.exists():
                h_preds = json.load(open(p_file, 'r', encoding='utf-8')).get("predictions", [])
                h_caps = json.load(open(c_file, 'r', encoding='utf-8')).get("predictions", []) if c_file.exists() else []
                rescored_h_preds = apply_semantic_co_emission(h_preds, h_caps)
                match_preds.extend(rescored_h_preds)

        open_pred = Path(f"\\\\?\\{m_pred_dir.resolve()}") / "results_spotting.json" if os.name == "nt" else m_pred_dir / "results_spotting.json"
        with open(open_pred, "w", encoding="utf-8") as f:
            json.dump({"UrlLocal": f"{rel_path}/", "predictions": match_preds}, f, indent=2)

    gt_path_str = f"\\\\?\\{gt_dir.resolve()}" if os.name == "nt" else str(gt_dir.resolve())
    pred_path_str = f"\\\\?\\{pred_dir.resolve()}" if os.name == "nt" else str(pred_dir.resolve())

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

    print("=" * 85)
    print("  MES-5 EVALUATION WITH ZERO-COMPUTE SEMANTIC CO-EMISSION")
    print("=" * 85)
    print(f"  Official Tight Average-mAP: {tight_results['a_mAP']:.4f}")
    print(f"  Official Loose Average-mAP: {loose_results['a_mAP']:.4f}")
    print("-" * 85)
    for i, cls in enumerate(INVERSE_EVENT_DICTIONARY_V2.values()):
        t_ap = tight_results['a_mAP_per_class'][i]
        l_ap = loose_results['a_mAP_per_class'][i]
        print(f"  {cls:<20}: Tight = {t_ap:.4f} | Loose = {l_ap:.4f}")
    print("=" * 85)

if __name__ == "__main__":
    main()
