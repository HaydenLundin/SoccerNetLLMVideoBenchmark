import json
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent

p_raw = WORKSPACE_ROOT / "logs/local_eval/qwen3.5-4b_4bit_W60_game3_half2/raw_window_outputs.json"
d_raw = json.load(open(p_raw, 'r', encoding='utf-8'))

print("=== ALL RAW CANDIDATES IN GAME 3 HALF 2 WITH 'CARD' OR 'SENT' OR 'RED' OR 'YELLOW' ===")
for w in d_raw:
    w_idx = w.get("window_idx")
    for c in w.get("parsed_candidates", []):
        t_str = c.get("time", "")
        label = c.get("label", "")
        desc = c.get("description", "")
        if any(k in desc.lower() or k in label.lower() for k in ["card", "sent", "red", "yellow"]):
            print(f"W{w_idx:02d} | Time: {t_str} | Label: {label} | Desc: {desc}")
