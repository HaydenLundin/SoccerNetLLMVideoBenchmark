import json
from pathlib import Path

p_raw = Path("logs/local_eval/qwen3.5-4b_4bit_W60_game13_half1/raw_window_outputs.json")
d = json.load(open(p_raw, 'r', encoding='utf-8'))

for w in d:
    txt = w.get("raw_output", "")
    if "42:" in txt or "Milner" in txt or "penalty" in txt.lower():
        print(f"Window {w['window_index']} (slice: {w['slice']}):")
        print(txt)
