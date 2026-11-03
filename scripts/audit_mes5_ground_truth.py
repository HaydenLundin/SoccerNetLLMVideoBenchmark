"""
Audit ground-truth class distribution across Canonical MES-5:
Games [0, 1, 2, 3, 13] across both halves.
"""

import json
from pathlib import Path
from collections import Counter

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent

MES5_GAMES = [
    (0, "england_epl/2014-2015/2015-05-17 - 18-00 Manchester United 1 - 1 Arsenal"),
    (1, "england_epl/2015-2016/2015-08-16 - 18-00 Manchester City 3 - 0 Chelsea"),
    (2, "england_epl/2015-2016/2015-08-23 - 15-30 West Brom 2 - 3 Chelsea"),
    (3, "england_epl/2015-2016/2015-08-29 - 17-00 Liverpool 0 - 3 West Ham"),
    (13, "england_epl/2016-2017/2016-08-27 - 14-30 Tottenham 1 - 1 Liverpool"),
]

SOCCERNET_CLASSES = [
    "Penalty", "Kick-off", "Goal", "Substitution", "Offside",
    "Shots on target", "Shots off target", "Clearance", "Ball out of play",
    "Throw-in", "Foul", "Indirect free-kick", "Direct free-kick", "Corner",
    "Yellow card", "Red card", "Yellow->red card"
]

def main():
    total_counter = Counter()
    game_counters = {}

    for gid, rel_path in MES5_GAMES:
        gt_file = WORKSPACE_ROOT / "test_eval_ground_truth" / rel_path / "Labels-v2.json"
        if not gt_file.exists():
            print(f"[!] Missing GT file for Game {gid}: {gt_file}")
            continue
        with open(gt_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        annotations = data.get("annotations", [])
        c = Counter()
        for ann in annotations:
            label = ann.get("label")
            c[label] += 1
            total_counter[label] += 1
        game_counters[gid] = (rel_path, len(annotations), c)

    print("=" * 85)
    print("  CANONICAL MES-5 GROUND-TRUTH CLASS TOPOLOGY AUDIT")
    print("=" * 85)
    print(f"{'Class':<20} | {'Total':<6} | {'G0':<5} | {'G1':<5} | {'G2':<5} | {'G3':<5} | {'G13':<5}")
    print("-" * 85)
    for cls in SOCCERNET_CLASSES:
        cnt = total_counter[cls]
        g0 = game_counters.get(0, ("", 0, Counter()))[2][cls]
        g1 = game_counters.get(1, ("", 0, Counter()))[2][cls]
        g2 = game_counters.get(2, ("", 0, Counter()))[2][cls]
        g3 = game_counters.get(3, ("", 0, Counter()))[2][cls]
        g13 = game_counters.get(13, ("", 0, Counter()))[2][cls]
        print(f"{cls:<20} | {cnt:<6} | {g0:<5} | {g1:<5} | {g2:<5} | {g3:<5} | {g13:<5}")
    print("-" * 85)
    print(f"{'TOTAL EVENTS':<20} | {sum(total_counter.values()):<6} | "
          f"{game_counters.get(0, ('', 0, None))[1]:<5} | "
          f"{game_counters.get(1, ('', 0, None))[1]:<5} | "
          f"{game_counters.get(2, ('', 0, None))[1]:<5} | "
          f"{game_counters.get(3, ('', 0, None))[1]:<5} | "
          f"{game_counters.get(13, ('', 0, None))[1]:<5}")
    print(f"Active Classes in MES-5: {len([c for c in SOCCERNET_CLASSES if total_counter[c] > 0])} / 17")
    print(f"Oracle Support Ceiling:   {len([c for c in SOCCERNET_CLASSES if total_counter[c] > 0]) / 17:.4f} (100.0%)")
    print("=" * 85)

if __name__ == "__main__":
    main()
