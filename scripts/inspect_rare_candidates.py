import json
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent

def inspect_event(game_id, half, min_pos, max_pos, target_name):
    print(f"\n=== INSPECTING {target_name} in Game {game_id} Half {half} ({min_pos}-{max_pos} ms) ===")
    p_v2 = WORKSPACE_ROOT / f"logs/local_eval/qwen3.5-4b_4bit_W60_game{game_id}_half{half}/Predictions-v2.json"
    p_cap = WORKSPACE_ROOT / f"logs/local_eval/qwen3.5-4b_4bit_W60_game{game_id}_half{half}/Predictions-Caption.json"
    p_raw = WORKSPACE_ROOT / f"logs/local_eval/qwen3.5-4b_4bit_W60_game{game_id}_half{half}/raw_window_outputs.json"

    if p_v2.exists():
        d_v2 = json.load(open(p_v2, 'r', encoding='utf-8'))
        for p in d_v2.get("predictions", []):
            pos = int(p.get("position", 0))
            if min_pos <= pos <= max_pos:
                print(f"  [Predictions-v2] {p.get('gameTime')} | Label: {p.get('label'):<18} | Conf: {p.get('confidence')} | Commentary: {p.get('commentary', '')}")

    if p_cap.exists():
        d_cap = json.load(open(p_cap, 'r', encoding='utf-8'))
        for p in d_cap.get("predictions", []):
            pos = int(p.get("position", 0))
            if min_pos <= pos <= max_pos:
                print(f"  [Predictions-Caption] {p.get('gameTime')} | Text: {p.get('commentary', '')}")

    if p_raw.exists():
        d_raw = json.load(open(p_raw, 'r', encoding='utf-8'))
        for w in d_raw:
            w_idx = w.get("window_idx")
            for c in w.get("parsed_candidates", []):
                t_str = c.get("time", "")
                label = c.get("label", "")
                desc = c.get("description", "")
                if any(k in desc.lower() for k in ["penalty", "spot", "second yellow", "red", "card", "sent off"]):
                    print(f"  [Raw Candidate W{w_idx}] Time: {t_str} | Label: {label} | Desc: {desc}")

inspect_event(13, 1, 2500000, 2600000, "James Milner Penalty Conversion (GT: 1 - 42:48)")
inspect_event(2, 1, 740000, 820000, "James Morrison Saved Penalty (GT: 1 - 13:03)")
inspect_event(3, 2, 360000, 420000, "Philippe Coutinho 2nd Yellow Dismissal (GT: 2 - 06:31)")
