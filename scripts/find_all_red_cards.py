import json
from pathlib import Path

eval_set = json.load(open('eval_set.json', 'r', encoding='utf-8'))
gt_dir = Path('test_eval_ground_truth')

print("=== ALL RED AND YELLOW->RED CARDS IN 20 GAMES ===")
for g in eval_set['games']:
    p = gt_dir / g['relative_path'] / 'Labels-v2.json'
    d = json.load(open(p, 'r', encoding='utf-8'))
    for a in d.get('annotations', []):
        lbl = a.get('label', '')
        if 'red' in lbl.lower():
            print(f"Game {g['game_id']:2d} ({g['match']}): {a.get('gameTime')} | Label: {lbl} | Pos: {a.get('position')} ms")
