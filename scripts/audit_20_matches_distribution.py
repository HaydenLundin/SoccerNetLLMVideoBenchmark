"""
Audit ground truth event distribution across all 20 frozen EPL matches (40 halves)
in eval_set.json.
"""

import json
from pathlib import Path
from collections import Counter

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
EVAL_SET_PATH = WORKSPACE_ROOT / "eval_set.json"
GT_DIR = WORKSPACE_ROOT / "test_eval_ground_truth"

def audit_dataset():
    with open(EVAL_SET_PATH, "r", encoding="utf-8") as f:
        eval_set = json.load(f)

    classes = eval_set["classes"]
    total_counts = Counter()
    h1_counts = Counter()
    h2_counts = Counter()
    per_game_summary = []

    for g in eval_set["games"]:
        rel = g["relative_path"]
        p = GT_DIR / rel / "Labels-v2.json"
        if not p.exists():
            print(f"Warning: {p} not found")
            continue
        with open(p, "r", encoding="utf-8") as gf:
            data = json.load(gf)
        
        g_counts = Counter()
        g_h1 = Counter()
        g_h2 = Counter()
        
        for a in data.get("annotations", []):
            lbl = a.get("label")
            t = a.get("gameTime", "")
            half = t.split(" - ")[0] if " - " in t else ""
            total_counts[lbl] += 1
            g_counts[lbl] += 1
            if half == "1":
                h1_counts[lbl] += 1
                g_h1[lbl] += 1
            elif half == "2":
                h2_counts[lbl] += 1
                g_h2[lbl] += 1
                
        per_game_summary.append({
            "game_id": g["game_id"],
            "match": g["match"],
            "total": sum(g_counts.values()),
            "h1": sum(g_h1.values()),
            "h2": sum(g_h2.values()),
            "red_cards": g_counts.get("Red card", 0),
            "yellow_red_cards": g_counts.get("Yellow->red card", 0),
            "penalties": g_counts.get("Penalty", 0),
            "substitutions": g_counts.get("Substitution", 0),
            "yellow_cards": g_counts.get("Yellow card", 0),
            "goals": g_counts.get("Goal", 0)
        })

    total_events = sum(total_counts.values())
    total_h1 = sum(h1_counts.values())
    total_h2 = sum(h2_counts.values())

    print("=" * 70)
    print("SOCCERNET VLM BENCHMARK: 20-MATCH (40-HALF) GROUND TRUTH AUDIT")
    print("=" * 70)
    print(f"Total Matches: {len(per_game_summary)} | Total Halves: 40 | Total Events: {total_events}")
    print(f"Half 1 Events: {total_h1} ({total_h1/total_events*100:.1f}%) | Half 2 Events: {total_h2} ({total_h2/total_events*100:.1f}%)")
    print("-" * 70)
    print(f"{'Class':<20} | {'Total':<6} | {'Half 1':<6} | {'Half 2':<6} | {'% Total':<8} | {'Active Halves'}")
    print("-" * 70)

    for c in sorted(classes, key=lambda x: total_counts[x], reverse=True):
        tot = total_counts[c]
        h1 = h1_counts[c]
        h2 = h2_counts[c]
        pct = (tot / total_events) * 100 if total_events > 0 else 0
        active_halves = sum(
            1 for g in per_game_summary
            # check per-half presence
        )
        print(f"{c:<20} | {tot:<6} | {h1:<6} | {h2:<6} | {pct:>6.2f}% |")

    print("=" * 70)
    print("\nMATCH-BY-MATCH RARE EVENT BREAKDOWN (Target Candidates for Scale-Out):")
    print(f"{'ID':<3} | {'Match':<45} | {'Tot':<4} | {'Sub':<3} | {'Yel':<3} | {'Red':<3} | {'Pen':<3} | {'Gol':<3}")
    print("-" * 80)
    for g in per_game_summary:
        print(f"{g['game_id']:<3} | {g['match'][:45]:<45} | {g['total']:<4} | {g['substitutions']:<3} | {g['yellow_cards']:<3} | {g['red_cards']:<3} | {g['penalties']:<3} | {g['goals']:<3}")

if __name__ == "__main__":
    audit_dataset()
