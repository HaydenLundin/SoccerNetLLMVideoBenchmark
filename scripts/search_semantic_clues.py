import json
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent

print("=== SEARCHING ALL 10 HALVES FOR 'PENALTY' IN CAPTIONS ===")
for gid in [0, 1, 2, 3, 13]:
    for half in [1, 2]:
        p_cap = WORKSPACE_ROOT / f"logs/local_eval/qwen3.5-4b_4bit_W60_game{gid}_half{half}/Predictions-Caption.json"
        if not p_cap.exists():
            continue
        d = json.load(open(p_cap, 'r', encoding='utf-8'))
        for item in d.get("predictions", []):
            txt = item.get("commentary", "").lower()
            if any(k in txt for k in ["penalty", "spot kick", "spot-kick", "12 yards"]):
                print(f"Game {gid} Half {half} | {item.get('gameTime')} (pos: {item.get('position')} ms) | {item.get('commentary')}")

print("\n=== SEARCHING ALL 10 HALVES FOR CARDS / SENT OFF IN CAPTIONS ===")
for gid in [0, 1, 2, 3, 13]:
    for half in [1, 2]:
        p_cap = WORKSPACE_ROOT / f"logs/local_eval/qwen3.5-4b_4bit_W60_game{gid}_half{half}/Predictions-Caption.json"
        if not p_cap.exists():
            continue
        d = json.load(open(p_cap, 'r', encoding='utf-8'))
        for item in d.get("predictions", []):
            txt = item.get("commentary", "").lower()
            if any(k in txt for k in ["second yellow", "sent off", "red card", "dismissed", "booking"]):
                print(f"Game {gid} Half {half} | {item.get('gameTime')} (pos: {item.get('position')} ms) | {item.get('commentary')}")
