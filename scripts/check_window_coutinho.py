import json
from pathlib import Path

p_raw = Path("logs/local_eval/qwen3.5-4b_4bit_W60_game3_half2/raw_window_outputs.json")
d = json.load(open(p_raw, 'r', encoding='utf-8'))

for w in d:
    txt = w.get("raw_output", "")
    if "06:" in txt or "07:" in txt or "Coutinho" in txt or "coutinho" in txt:
        print(f"Window {w['window_index']} (slice: {w['slice']}):")
        print(txt)
