"""
Verification suite for processing/sn_vqa_2027_events.py (SN-VQA-2027 event detection scoring).
Synthetic fixtures only, so it runs in CPU-only CI.

Optional regression check of the research note's section 11 numbers:
    python scripts/test_sn_vqa_2027_events.py --official-dir <sn-vqa-2027 clone>/data
(uses the detector outputs committed in logs/sn_vqa_2027/events/)
"""

import sys
import json
import random
import argparse
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))

from processing.sn_vqa_2027_events import (
    CLASS_SETS, LETTERS, EVENT_LAG, windows, truth_events, covers, match, merge_above, peaks, auc, worst_case_ranks, f1,
)


def test_windows_sampling_gaps_and_last_window():
    print("\n--- 1. Windows: sampling rate, gaps and the partial last window ---")
    # 25 fps from 0 to 10 s, a gap, then 40 to 44 s
    frames = [(i * 0.04, i) for i in range(250)] + [(40 + i * 0.04, i) for i in range(100)]
    got = [(s, e, [round(t, 2) for t, _ in kept]) for s, e, kept in windows(frames, window=6.0, fps=1.0)]
    assert got == [
        (0.0, 6.0, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]),
        (6.0, 12.0, [6.0, 7.0, 8.0, 9.0]),
        (36.0, 42.0, [40.0, 41.0]),  # empty windows in the gap are skipped, not misfiled
        (42.0, 43.0, [42.0, 43.0]),  # the last window ends at its last frame
    ], got
    converted = list(windows([(0.0, "a"), (0.5, "b")], window=6.0, fps=1.0, convert=str.upper))
    assert converted == [(0.0, 0.0, [(0.0, "A")])]
    print("  [PASS] 1 frame/s kept, gap skipped, last window clipped, frames converted")


def _q(qid, pos_sec, sub, text):
    return {"id": qid, "position_ms": int(pos_sec * 1000), "question": text, "metadata": {"subcategory": sub}}


def test_truth_events():
    print("\n--- 2. Truth: one event per moment, shifted back, goals as shots ---")
    gt = {"data": [
        {"metadata": {"match_key": "game1", "half": 1}, "streaming_vqa": [
            _q("q1", 100, "event_team", "Which team is taking the corner now?"),
            _q("q2", 100, "event_player", "Who is taking the corner now?"),
            _q("q3", 300, "event_team", "Which team is scoring the goal now?"),
            _q("q4", 400, "event_player", "Who is taking the free kick now?"),
            _q("q5", 500, "recall_event_time", "When did Willian last take a shot?"),
        ]},
        {"metadata": {"match_key": "game1", "half": 2}, "streaming_vqa": [
            _q("q6", 50, "event_team", "Which team is taking the shot now?"),
        ]},
    ]}
    assert truth_events(gt, "game1", 1) == [(100 - EVENT_LAG, "corner"), (300 - EVENT_LAG, "shot"), (400 - EVENT_LAG, "free_kick")]
    assert truth_events(gt, "game1", 2) == [(50 - EVENT_LAG, "shot")]
    assert truth_events(gt, "game2", 1) == []
    print("  [PASS] team + player merged, recall question ignored, halves kept apart")


def test_matching():
    print("\n--- 3. Matching is one-to-one, with an optional lead ---")
    assert covers((10, 16), 10) and covers((10, 16), 16) and not covers((10, 16), 9.9)
    assert covers((20, 26), 8, lead=15) and not covers((20, 26), 4, lead=15)
    # one detection, two events inside it: credits only one
    assert match([(0, 12)], [3, 9]) == (1, 1)
    # two detections on one event: only one is real
    assert match([(0, 6), (6, 12)], [6]) == (1, 1)
    # a cue window after the event counts only with a lead
    assert match([(20, 26)], [10]) == (0, 0) and match([(20, 26)], [10], lead=15) == (1, 1)
    print("  [PASS] no double credit either way; lead lets a later cue window find the event")


def _rows(ps, width=6.0, kind="shot"):
    return [{"start": i * width, "end": (i + 1) * width, "p": {kind: p}} for i, p in enumerate(ps)]


def test_merge_and_peaks():
    print("\n--- 4. Merging above a threshold, and peak picking ---")
    rows = _rows([0.1, 0.6, 0.7, 0.2, 0.9, 0.1])
    assert merge_above(rows, "shot", 0.5) == [(6.0, 18.0), (24.0, 30.0)]
    assert merge_above(rows, "corner", 0.5) == []  # absent class
    # peaks: highest first, nothing within 30 s of a taken window
    rows = _rows([0.9, 0.8, 0.1, 0.1, 0.1, 0.1, 0.7, 0.2])
    assert peaks(rows, "shot", thr=0.5) == [(0.0, 6.0), (36.0, 42.0)]
    assert peaks(rows, "shot", thr=0.95) == []
    # with no threshold, windows are taken in the given order up to n (the random baseline)
    assert peaks(list(reversed(rows)), "shot", n=1) == [(42.0, 48.0)]
    print("  [PASS] consecutive windows merge; peaks keep 30 s apart and stop at the cut-off or n")


def test_auc_and_ranks():
    print("\n--- 5. AUC counts ties half; ranks are worst-case among ties ---")
    rows = _rows([0.9, 0.5, 0.5, 0.1])
    assert auc(rows, "shot", [8.0]) == (0 + 0.5 + 1) / 3  # event window p=0.5 loses to 0.9, ties 0.5, beats 0.1
    assert worst_case_ranks(rows, "shot", [8.0, 1.0]) == [1, 3]  # the 0.5 window ties with another: rank 3, not 2
    assert auc(rows, "shot", []) is None
    assert f1(5, 10, 5, 10) == 0.5 and f1(0, 10, 0, 0) == 0.0
    print("  [PASS] tie handling for AUC and ranks; F1 edge cases")


def test_prompts():
    print("\n--- 6. Prompts fit the answer letters ---")
    for name, classes in CLASS_SETS.items():
        keys = [k for k, _ in classes]
        assert len(keys) == len(set(keys)) and len(keys) <= len(LETTERS), name
        assert "shot" in keys, name
    print(f"  [PASS] {len(CLASS_SETS)} prompts, unique classes, each scores shots")


def check_official(official_dir: Path):
    """Reproduces section 11 of the research note from the committed detector outputs."""
    print(f"\n--- 7. Section 11 numbers from logs/sn_vqa_2027/events and {official_dir} ---")
    sys.path.insert(0, str(WORKSPACE_ROOT / "scripts"))
    from score_events_sn_vqa_2027 import load_halves, shot_counts, CUTS
    gt = json.loads((official_dir / "test_GT.json").read_text(encoding="utf-8"))
    folder = WORKSPACE_ROOT / "logs" / "sn_vqa_2027" / "events"
    expected = {"v2": (0.02, 0.87, 18, 77, "0.32"), "shot_bin_896": (0.9, 0.78, 8, 20, "0.30")}
    for source, (cut_expected, mean_auc, found_expected, dets_expected, f1_expected) in expected.items():
        halves = load_halves(gt, folder, source)
        assert len(halves) == 4, f"{source}: {len(halves)} halves"
        aucs = [auc(rows, "shot", [t for t, k in ev if k == "shot"]) for rows, _, ev in halves.values()]
        assert f"{sum(aucs) / 4:.2f}" == f"{mean_auc:.2f}", (source, aucs)
        rows, _, events = halves["game1_h1"]
        tuned = {c: f1(*shot_counts(rows, events, c, 30.0, 15.0)[:4]) for c in CUTS}
        cut = max(CUTS, key=lambda c: (tuned[c], c))
        assert cut == cut_expected, (source, cut)
        held = [0, 0, 0, 0]
        for tag, (rows, _, events) in halves.items():
            if tag != "game1_h1":
                held = [a + b for a, b in zip(held, shot_counts(rows, events, cut, 30.0, 15.0)[:4])]
        assert (held[0], held[3]) == (found_expected, dets_expected) and f"{f1(*held):.2f}" == f1_expected, (source, held)
        print(f"  [PASS] {source}: mean shot AUC {mean_auc:.2f}; cut {cut}; held-out {held[0]}/{held[1]} found, "
              f"{held[2]}/{held[3]} real, F1 {f1_expected}")


def run_all_tests(official_dir=None):
    test_windows_sampling_gaps_and_last_window()
    test_truth_events()
    test_matching()
    test_merge_and_peaks()
    test_auc_and_ranks()
    test_prompts()
    if official_dir is not None:
        check_official(official_dir)
    print("\n" + "=" * 60)
    print(" ALL SN-VQA-2027 EVENT DETECTION TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-dir", type=Path, default=None,
                        help="Path to the data/ folder of a SoccerNet/sn-vqa-2027 clone")
    args = parser.parse_args()
    run_all_tests(args.official_dir)
