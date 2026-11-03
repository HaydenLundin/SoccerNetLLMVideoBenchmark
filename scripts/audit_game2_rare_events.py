"""
Audit Qwen3.5-4B detections on Game 2 Golden Fixture rare events:
- 1 Penalty (Half 1)
- 1 Red card (Half 2)
- 5 Goals (West Brom 2 - 3 Chelsea)
- 3 Yellow cards
- 6 Substitutions
Computes matched hits and latency error within tolerance.
"""

import sys
import json
from pathlib import Path
from typing import List, Dict

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent

TARGET_RARE_EVENTS = [
    {"half": 1, "time": "1 - 13:03", "label": "Penalty", "pos_ms": 783751, "desc": "Penalty awarded / taken"},
    {"half": 1, "time": "1 - 16:09", "label": "Yellow card", "pos_ms": 969099, "desc": "Matic yellow card"},
    {"half": 1, "time": "1 - 19:29", "label": "Goal", "pos_ms": 1169360, "desc": "Pedro debut goal (0-1)"},
    {"half": 1, "time": "1 - 29:09", "label": "Goal", "pos_ms": 1749200, "desc": "Diego Costa goal (0-2)"},
    {"half": 1, "time": "1 - 34:50", "label": "Goal", "pos_ms": 2090400, "desc": "Morrison goal (1-2)"},
    {"half": 1, "time": "1 - 36:03", "label": "Yellow card", "pos_ms": 2163005, "desc": "McClean yellow card"},
    {"half": 1, "time": "1 - 41:39", "label": "Goal", "pos_ms": 2499676, "desc": "Azpilicueta goal (1-3)"},
    {"half": 2, "time": "2 - 08:21", "label": "Red card", "pos_ms": 501099, "desc": "John Terry straight red card"},
    {"half": 2, "time": "2 - 10:09", "label": "Substitution", "pos_ms": 609240, "desc": "Cahill on for Willian"},
    {"half": 2, "time": "2 - 13:33", "label": "Goal", "pos_ms": 813600, "desc": "Morrison header goal (2-3)"},
    {"half": 2, "time": "2 - 14:23", "label": "Substitution", "pos_ms": 863160, "desc": "Lambert on for Olsson"},
    {"half": 2, "time": "2 - 25:21", "label": "Yellow card", "pos_ms": 1521000, "desc": "McManaman yellow card"},
    {"half": 2, "time": "2 - 32:15", "label": "Substitution", "pos_ms": 1935716, "desc": "Falcao on for Diego Costa"},
    {"half": 2, "time": "2 - 32:18", "label": "Substitution", "pos_ms": 1938120, "desc": "Gnabry on for McClean"},
    {"half": 2, "time": "2 - 38:55", "label": "Substitution", "pos_ms": 2335240, "desc": "Mikel on for Pedro"},
    {"half": 2, "time": "2 - 42:27", "label": "Substitution", "pos_ms": 2547868, "desc": "Anichebe on for Dawson"},
]

def load_preds(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f).get("predictions", [])

def audit_rare_events(tolerance_sec: float = 30.0):
    p_h1 = WORKSPACE_ROOT / "logs/local_eval/qwen3.5-4b_4bit_W60_game2_half1/Predictions-v2.json"
    p_h2 = WORKSPACE_ROOT / "logs/local_eval/qwen3.5-4b_4bit_W60_game2_half2/Predictions-v2.json"

    preds_h1 = load_preds(p_h1)
    preds_h2 = load_preds(p_h2)

    all_preds = preds_h1 + preds_h2

    print("=" * 95)
    print("  GAME 2 GOLDEN FIXTURE RARE EVENT AUDIT (Qwen3.5-4B, W=60s)")
    print(f"  Predictions Loaded: H1={len(preds_h1)}, H2={len(preds_h2)} (Total: {len(all_preds)})")
    print("=" * 95)
    print(f"{'Target Event':<26} | {'GT Time':<9} | {'Status':<10} | {'Pred Time':<9} | {'Error (s)':<9} | {'Confidence':<10}")
    print("-" * 95)

    tol_ms = int(tolerance_sec * 1000.0)
    matched_count = 0

    for target in TARGET_RARE_EVENTS:
        tgt_half = str(target["half"])
        tgt_pos = target["pos_ms"]
        tgt_lbl = target["label"]

        # Find closest prediction matching label and half
        best_p = None
        best_diff = float("inf")

        for p in all_preds:
            p_lbl = p.get("label")
            p_half = str(p.get("half", "1"))
            p_pos = int(p.get("position", 0))

            if p_half == tgt_half and p_lbl == tgt_lbl:
                diff = abs(p_pos - tgt_pos)
                if diff <= tol_ms and diff < best_diff:
                    best_diff = diff
                    best_p = p

        event_name = f"{target['label']} ({target['desc'].split()[0]})"
        if best_p:
            matched_count += 1
            err_sec = (int(best_p.get("position", 0)) - tgt_pos) / 1000.0
            conf = float(best_p.get("confidence", 0.0))
            pred_time = best_p.get("gameTime", "")
            print(f"{target['desc']:<26} | {target['time']:<9} | {'HIT':<10} | {pred_time:<9} | {err_sec:+6.2f}s  | {conf:6.2f}")
        else:
            print(f"{target['desc']:<26} | {target['time']:<9} | {'MISS':<10} | {'--':<9} | {'--':<9} | {'--':<10}")

    print("-" * 95)
    print(f"Summary: {matched_count} / {len(TARGET_RARE_EVENTS)} rare events matched within +/-{tolerance_sec}s ({matched_count/len(TARGET_RARE_EVENTS)*100:.1f}%)")
    print("=" * 95)

if __name__ == "__main__":
    audit_rare_events()
