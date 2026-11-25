"""
Verification suite for processing/event_ledger.py (SN-VQA-2027 counting questions).
Synthetic fixtures only, so it runs in CPU-only CI.

Optional parser-coverage check against the official question files:
    python scripts/test_event_ledger.py --official-dir <sn-vqa-2027 clone>/data
"""

import sys
import json
import argparse
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))

from processing.event_ledger import (
    LedgerEvent,
    EventLedger,
    CountQuery,
    parse_counting_question,
    match_clock_sec,
    choose_count_option,
    from_soccernet_label,
    ledgers_from_now_questions,
)
from processing.sn_vqa_2027 import iter_questions


def _ledger():
    """Chelsea vs Liverpool: events with an observation time (when perception reported them)."""
    led = EventLedger()
    led.add(LedgerEvent(half=1, sec=100, kind="shot", team="Chelsea", players=("Diego Costa",)), observed_half=1, observed_sec=130)
    led.add(LedgerEvent(half=1, sec=900, kind="shot", team="Chelsea", players=("Willian", "Diego Costa")), observed_half=1, observed_sec=930)
    led.add(LedgerEvent(half=1, sec=2600, kind="corner", team="Liverpool", players=("James Milner",)), observed_half=1, observed_sec=2610)
    led.add(LedgerEvent(half=2, sec=200, kind="shot", team="Liverpool", players=("Jordan Henderson",)), observed_half=2, observed_sec=260)
    led.add(LedgerEvent(half=2, sec=600, kind="shot", team="Chelsea", players=("Diego Costa",)), observed_half=2, observed_sec=640)
    return led


# --- Parser ------------------------------------------------------------------

def test_parser_templates():
    print("\n--- 1. Parser covers each scope, event kind and verb ---")
    cases = {
        "As of now, how many shots has Chelsea taken?": CountQuery("shot", "Chelsea", "taken", None),
        "Over the whole match so far, how many free kicks has Willian had?": CountQuery("free_kick", "Willian", "had", None),
        "In the past 20 minutes, how many corners has Liverpool taken?": CountQuery("corner", "Liverpool", "taken", 20),
        "Over the past 10 minutes, how many shots has Diego Costa been involved in?": CountQuery("shot", "Diego Costa", "been involved in", 10),
        "In the past 20 minutes, how many times has Diego Costa been caught offside?": CountQuery("offside", "Diego Costa", "been caught offside", 20),
        "As of now, how many substitutions has Manchester United made?": CountQuery("substitution", "Manchester United", "made", None),
        "As of now, how many goals has Diego Costa scored?": CountQuery("goal", "Diego Costa", "scored", None),
        "Over the whole match so far, how many cards has Chelsea had?": CountQuery("card", "Chelsea", "had", None),
        "Over the whole match so far, how many offsides has West Bromwich Albion had?": CountQuery("offside", "West Bromwich Albion", "had", None),
    }
    for text, expected in cases.items():
        got = parse_counting_question(text)
        assert got == expected, f"{text!r}: {got} != {expected}"
    assert parse_counting_question("Who is taking the corner now?") is None
    print(f"  [PASS] {len(cases)} templates parsed; non-counting question rejected")


# --- Clock -------------------------------------------------------------------

def test_match_clock():
    print("\n--- 2. Match clock joins the halves ---")
    assert match_clock_sec(1, 125) == 125
    assert match_clock_sec(2, 0) == 2700
    assert match_clock_sec(2, 300) == 3000
    # first-half stoppage time is clamped to 45:00 so it never sorts after second-half play
    assert match_clock_sec(1, 2800) == 2700
    print("  [PASS] 1-02:05 -> 125 s, 2-05:00 -> 3000 s, 1-46:40 clamps to 2700 s")


# --- Ledger queries ----------------------------------------------------------

def test_cumulative_counts_span_halves():
    print("\n--- 3. 'whole match so far' counts both halves, team or player ---")
    led = _ledger()
    q_team = CountQuery("shot", "Chelsea", "taken", None)
    assert led.count(q_team, now_half=2, now_sec=700) == 3
    assert led.count(q_team, now_half=1, now_sec=1000) == 2
    q_player = CountQuery("shot", "Diego Costa", "taken", None)
    assert led.count(q_player, now_half=2, now_sec=700) == 2, "primary actor only"
    q_involved = CountQuery("shot", "Diego Costa", "been involved in", None)
    assert led.count(q_involved, now_half=2, now_sec=700) == 3, "any listed player"
    print("  [PASS] team 3, Costa taken 2, Costa involved 3")


def test_count_is_causal_on_observation_time():
    print("\n--- 4. An event counts only once it has been observed ---")
    led = _ledger()
    q = CountQuery("shot", "Chelsea", "taken", None)
    # shot at 1-15:00 is reported at 1-15:30: not yet known at 1-15:10
    assert led.count(q, now_half=1, now_sec=910) == 1
    assert led.count(q, now_half=1, now_sec=930) == 2
    print("  [PASS] 1 before the report, 2 after")


def test_observation_cannot_precede_event():
    print("\n--- 5. Ledger rejects an observation earlier than the event ---")
    led = EventLedger()
    try:
        led.add(LedgerEvent(half=1, sec=500, kind="shot", team="Chelsea"), observed_half=1, observed_sec=499)
    except ValueError:
        print("  [PASS] ValueError raised")
        return
    raise AssertionError("Expected ValueError")


def test_window_counts():
    print("\n--- 6. 'past N minutes' windows use the match clock ---")
    led = _ledger()
    q20 = CountQuery("shot", "Chelsea", "taken", 20)
    # now = 2-10:40 (3340 s): window starts at 2140 s (1-35:40), so only the 2-10:00 shot counts
    assert led.count(q20, now_half=2, now_sec=640) == 1
    q_corner = CountQuery("corner", "Liverpool", "taken", 10)
    # corner at 1-43:20 (2600 s) is inside a 10-minute window ending at 2-04:30 (2970 s)
    assert led.count(q_corner, now_half=2, now_sec=270) == 1
    # but not inside one ending at 2-10:00 (3300 s), which starts at 2700 s
    assert led.count(q_corner, now_half=2, now_sec=600) == 0
    print("  [PASS] windows cross the half boundary")


def test_rejects_unknown_kind():
    print("\n--- 7. Unknown event kinds are rejected ---")
    try:
        LedgerEvent(half=1, sec=1, kind="throw_in", team="Chelsea")
    except ValueError:
        print("  [PASS] ValueError raised")
        return
    raise AssertionError("Expected ValueError")


# --- Answer mapping ----------------------------------------------------------

def test_choose_count_option():
    print("\n--- 8. Count maps to the matching option, else the nearest ---")
    opts = [{"id": "o1", "text": "2"}, {"id": "o2", "text": "3"}, {"id": "o3", "text": "1"}, {"id": "o4", "text": "0"}]
    assert choose_count_option(opts, 1) == ("o3", True)
    assert choose_count_option(opts, 7) == ("o2", False)
    # tie between 4 and 6 for a count of 5: pick the lower value
    tie = [{"id": "o1", "text": "6"}, {"id": "o2", "text": "4"}]
    assert choose_count_option(tie, 5) == ("o2", False)
    print("  [PASS] exact, nearest, and tie handling")


def test_soccernet_label_mapping():
    print("\n--- 9. SoccerNet-v2 classes map onto ledger kinds ---")
    assert from_soccernet_label("Shots on target") == "shot"
    assert from_soccernet_label("Shots off target") == "shot"
    assert from_soccernet_label("Direct free-kick") == "free_kick"
    assert from_soccernet_label("Indirect free-kick") == "free_kick"
    assert from_soccernet_label("Yellow->red card") == "card"
    assert from_soccernet_label("Throw-in") is None
    print("  [PASS] shots, free kicks, cards mapped; throw-ins ignored")


def test_goal_counts_as_shot():
    print("\n--- 10. A goal also counts as a shot, not the other way round ---")
    led = EventLedger()
    led.add(LedgerEvent(half=1, sec=100, kind="shot", team="Chelsea"))
    led.add(LedgerEvent(half=1, sec=200, kind="goal", team="Chelsea"))
    assert led.count(CountQuery("shot", "Chelsea", "had", None), 1, 300) == 2
    assert led.count(CountQuery("goal", "Chelsea", "scored", None), 1, 300) == 1
    print("  [PASS] shots 2, goals 1")


def _now_q(sample_id, qid, pos_ms, sub, text, options, half=1):
    return {
        "sample_id": sample_id, "match_key": "game1", "video_path": None, "half": half,
        "question": {
            "id": qid, "position_ms": pos_ms, "question": text,
            "options": [{"id": f"o{i}", "text": t} for i, t in enumerate(options, 1)],
            "metadata": {"subcategory": sub},
        },
    }


def test_ledger_from_now_questions():
    print("\n--- 11. Event questions plus our answers become ledger events ---")
    recs = [
        _now_q("m_h1", "q1", 100000, "event_team", "Which team is taking the shot now?", ["Chelsea", "Liverpool"]),
        _now_q("m_h1", "q2", 100000, "event_player", "Who is taking the shot now?", ["Willian", "Diego Costa"]),
        _now_q("m_h1", "q3", 500000, "event_team", "Which team is scoring the goal now?", ["Chelsea", "Liverpool"]),
        _now_q("m_h1", "q4", 600000, "event_team", "Which team is taking the free kick now?", ["Chelsea", "Liverpool"]),
        _now_q("m_h1", "q5", 700000, "recall_event_time", "When did Willian last take a shot?", ["01:40", "02:00"]),
    ]
    answers = {("m_h1", "q1"): "o1", ("m_h1", "q2"): "o2", ("m_h1", "q3"): "o1", ("m_h1", "q4"): "o2", ("m_h1", "q5"): "o1"}
    led = ledgers_from_now_questions(recs, answers)["game1"]
    assert len(led.events_as_of(1, 10_000)) == 3, "team + player at one moment merge; recall questions are ignored"
    shots = CountQuery("shot", "Chelsea", "taken", None)
    assert led.count(shots, 1, 400) == 1
    assert led.count(shots, 1, 600) == 2, "the goal also counts as a shot"
    assert led.count(CountQuery("shot", "Diego Costa", "taken", None), 1, 600) == 1
    assert led.count(CountQuery("free_kick", "Liverpool", "had", None), 1, 599) == 0, "not observed yet"
    assert led.count(CountQuery("free_kick", "Liverpool", "had", None), 1, 600) == 1
    print("  [PASS] 3 events; counts follow question times")


def test_half_time_boundary():
    print("\n--- 12. First-half stoppage past 2700 s orders before the second half ---")
    q_all = CountQuery("shot", "Chelsea", "taken", None)
    led = EventLedger()
    # shot at 1-45:50, reported at 1-46:00: known to every second-half query
    led.add(LedgerEvent(half=1, sec=2750, kind="shot", team="Chelsea"), observed_half=1, observed_sec=2760)
    assert [led.count(q_all, 2, s) for s in (10, 50, 60)] == [1, 1, 1]
    # reported only after the break is still a valid observation
    late = EventLedger()
    late.add(LedgerEvent(half=1, sec=2750, kind="shot", team="Chelsea"), observed_half=2, observed_sec=5)
    assert late.count(q_all, 1, 2800) == 0 and late.count(q_all, 2, 5) == 1
    # a second-half event is never visible to a first-half stoppage query
    led.add(LedgerEvent(half=2, sec=0, kind="shot", team="Chelsea"))
    assert led.count(q_all, 1, 2800) == 1
    # "past 10 minutes" in first-half stoppage uses real elapsed time: 1-47:00 reaches back to 1-37:00
    stop = EventLedger()
    stop.add(LedgerEvent(half=1, sec=2150, kind="shot", team="Chelsea"))  # 1-35:50, outside
    stop.add(LedgerEvent(half=1, sec=2300, kind="shot", team="Chelsea"))  # 1-38:20, inside
    assert stop.count(CountQuery("shot", "Chelsea", "taken", 10), 1, 2820) == 1
    # a 0-minute window is not "since kick-off"
    assert stop.count(CountQuery("shot", "Chelsea", "taken", 0), 1, 2820) == 0
    print("  [PASS] stoppage events visible after the break, never the reverse; windows keep real length")


# --- Optional official-file coverage -----------------------------------------

def check_official(official_dir: Path):
    print(f"\n--- 13. Parser coverage on official files in {official_dir} ---")
    for name in ("test_GT.json", "challenge_public.json"):
        ref = json.loads((official_dir / name).read_text(encoding="utf-8"))
        counting = [r for r in iter_questions(ref) if r["question"]["metadata"]["subcategory"] == "counting"]
        unparsed = [r["question"]["question"] for r in counting if parse_counting_question(r["question"]["question"]) is None]
        assert not unparsed, unparsed[:5]
        assert all(r["match_key"] for r in counting)
        print(f"  [PASS] {name}: {len(counting)}/{len(counting)} counting questions parsed")

    # Oracle: with the test answers to the earlier event questions, the ledger
    # must reproduce every test counting answer. Diagnostic only (uses labels).
    gt = json.loads((official_dir / "test_GT.json").read_text(encoding="utf-8"))
    records = list(iter_questions(gt))
    truth = {(r["sample_id"], r["question"]["id"]): r["question"]["correct_option_id"] for r in records}
    ledgers = ledgers_from_now_questions(records, truth)
    counting = [r for r in records if r["question"]["metadata"]["subcategory"] == "counting"]
    right = 0
    for r in counting:
        q = r["question"]
        n = ledgers[r["match_key"]].count(parse_counting_question(q["question"]), r["half"], q["position_ms"] / 1000)
        right += choose_count_option(q["options"], n)[0] == q["correct_option_id"]
    assert right == len(counting), f"{right}/{len(counting)}"
    print(f"  [PASS] oracle question-stream ledger reproduces {right}/{len(counting)} test counting answers")


def run_all_tests(official_dir=None):
    test_parser_templates()
    test_match_clock()
    test_cumulative_counts_span_halves()
    test_count_is_causal_on_observation_time()
    test_observation_cannot_precede_event()
    test_window_counts()
    test_rejects_unknown_kind()
    test_choose_count_option()
    test_soccernet_label_mapping()
    test_goal_counts_as_shot()
    test_ledger_from_now_questions()
    test_half_time_boundary()
    if official_dir is not None:
        check_official(official_dir)
    print("\n" + "=" * 60)
    print(" ALL EVENT LEDGER TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-dir", type=Path, default=None,
                        help="Path to the data/ folder of a SoccerNet/sn-vqa-2027 clone")
    args = parser.parse_args()
    run_all_tests(args.official_dir)
