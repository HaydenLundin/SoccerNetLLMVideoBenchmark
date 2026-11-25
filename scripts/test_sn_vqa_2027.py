"""
Verification suite for processing/sn_vqa_2027.py (SoccerNet 2027 Streaming VQA).
Covers the causal frame cutoff and the OSL JSON submission writer/validator on
synthetic fixtures, so it runs in CI without the gated videos or the official files.

Optional end-to-end check against the official repo files (github.com/SoccerNet/sn-vqa-2027):
    python scripts/test_sn_vqa_2027.py --official-dir <clone>/data
rebuilds the organizers' baseline submission through our writer and checks it
reproduces their published 41.76% score. To confirm with the official scorer, run
<clone>/evaluation/scoring.py on the output yourself.
"""

import sys
import ast
import copy
import json
import argparse
import tempfile
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))

from processing.sn_vqa_2027 import (
    causal_frame_window,
    question_time_sec,
    iter_questions,
    build_submission,
    validate_submission,
    check_challenge_safe_model,
    CHALLENGE_EXCLUDED_NAME_RE,
    question_window,
    annotate_option,
)


def _reference():
    """Minimal OSL reference shaped like data/test_GT.json (two halves, repeated question ids)."""
    def q(qid, pos_ms, n_opts, sub, cat, answer):
        return {
            "id": qid,
            "position_ms": pos_ms,
            "question": f"Question {qid}?",
            "options": [{"id": f"o{i}", "text": f"Option {i}"} for i in range(1, n_opts + 1)],
            "correct_option_id": answer,
            "metadata": {"category": cat, "subcategory": sub, "game_time": "1 - 00:00", "num_of_options": n_opts},
        }

    return {
        "version": "2.0",
        "task": "streaming_vqa",
        "dataset_name": "Synthetic",
        "modalities": ["video"],
        "metadata": {"source_row_count": 3},
        "data": [
            {
                "id": "match_001_half_1",
                "inputs": [{"type": "video", "path": "video-test/game1/1_720p.mkv"}],
                "metadata": {"half": 1, "match_key": "game1"},
                "streaming_vqa": [
                    q("dim3_q1", 204000, 4, "event_player", "dim3", "o3"),
                    q("dim3_q2", 204000, 2, "event_team", "dim3", "o2"),
                ],
            },
            {
                "id": "match_001_half_2",
                "inputs": [{"type": "video", "path": "video-test/game1/2_720p.mkv"}],
                "metadata": {"half": 2, "match_key": "game1"},
                "streaming_vqa": [
                    q("dim3_q1", 5000, 4, "counting", "dim1", "o1"),
                ],
            },
        ],
    }


# --- Causal cutoff -----------------------------------------------------------

def test_causal_window_excludes_future_frames():
    print("\n--- 1. Causal window never includes frames after the question time ---")
    frames = [(float(t), f"frame{t}") for t in range(0, 300)]
    window = causal_frame_window(frames, query_sec=204.0, window_sec=60.0, max_frames=1000)
    times = [t for t, _ in window]
    assert max(times) <= 204.0, f"Future frame leaked: {max(times)}"
    assert min(times) >= 144.0, f"Frame older than window: {min(times)}"
    assert times[-1] == 204.0, "The frame at the question time must be kept"
    print(f"  [PASS] {len(times)} frames in [144, 204]")


def test_causal_window_subsample_keeps_latest_frame():
    print("\n--- 2. Subsampling keeps the newest frame and stays causal ---")
    frames = [(t * 0.5, t) for t in range(0, 1000)]
    window = causal_frame_window(frames, query_sec=250.2, window_sec=60.0, max_frames=10)
    times = [t for t, _ in window]
    assert len(times) == 10, len(times)
    assert times == sorted(times), "Frames must stay in temporal order"
    assert times[-1] == 250.0, f"Newest causal frame dropped: {times[-1]}"
    assert all(t <= 250.2 for t in times)
    print(f"  [PASS] 10 frames, newest at {times[-1]}s")


def test_causal_window_at_half_start():
    print("\n--- 3. Window clamps at the start of the half ---")
    frames = [(float(t), t) for t in range(0, 30)]
    window = causal_frame_window(frames, query_sec=5.0, window_sec=60.0, max_frames=100)
    assert [t for t, _ in window] == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    assert causal_frame_window(frames, query_sec=-1.0, window_sec=60.0, max_frames=10) == []
    print("  [PASS] clamped to [0, 5]")


def test_causal_window_rejects_bad_args():
    print("\n--- 4. Invalid window arguments are rejected ---")
    for kwargs in ({"window_sec": 0.0, "max_frames": 5}, {"window_sec": 10.0, "max_frames": 0}):
        try:
            causal_frame_window([(0.0, 0)], query_sec=1.0, **kwargs)
        except ValueError:
            continue
        raise AssertionError(f"Expected ValueError for {kwargs}")
    print("  [PASS] ValueError raised")


def test_question_time_and_iteration():
    print("\n--- 5. Question time comes from position_ms; ids are scoped by sample ---")
    ref = _reference()
    items = list(iter_questions(ref))
    assert [(i["sample_id"], i["question"]["id"]) for i in items] == [
        ("match_001_half_1", "dim3_q1"),
        ("match_001_half_1", "dim3_q2"),
        ("match_001_half_2", "dim3_q1"),
    ]
    assert items[0]["video_path"] == "video-test/game1/1_720p.mkv"
    assert items[2]["half"] == 2
    assert question_time_sec(items[0]["question"]) == 204.0
    print("  [PASS] 3 questions, repeated id kept distinct across halves")


# --- OSL submission writer / validator --------------------------------------

def test_submission_round_trip():
    print("\n--- 6. Writer emits a valid submission without leaking answers ---")
    ref = _reference()
    preds = {
        ("match_001_half_1", "dim3_q1"): "o3",
        ("match_001_half_1", "dim3_q2"): None,
        ("match_001_half_2", "dim3_q1"): "o4",
    }
    sub = build_submission(ref, preds)
    assert validate_submission(sub, ref) == []
    for sample in sub["data"]:
        for q in sample["streaming_vqa"]:
            assert "correct_option_id" not in q, "Answer key leaked into submission"
            assert "category" in q["metadata"], "category must be preserved"
            assert "prediction" in q
    assert sub["data"][0]["streaming_vqa"][1]["prediction"] is None
    assert "correct_option_id" in ref["data"][0]["streaming_vqa"][0], "Reference must not be mutated"
    print("  [PASS] valid, answers stripped, nulls kept")


def test_writer_rejects_missing_and_unknown_predictions():
    print("\n--- 7. Writer refuses incomplete or unknown predictions ---")
    ref = _reference()
    full = {
        ("match_001_half_1", "dim3_q1"): "o1",
        ("match_001_half_1", "dim3_q2"): "o1",
        ("match_001_half_2", "dim3_q1"): "o1",
    }
    missing = dict(full)
    del missing[("match_001_half_2", "dim3_q1")]
    extra = dict(full)
    extra[("match_001_half_9", "dim3_q1")] = "o1"
    for bad in (missing, extra):
        try:
            build_submission(ref, bad)
        except ValueError:
            continue
        raise AssertionError("Expected ValueError")
    print("  [PASS] ValueError on missing and unknown keys")


def test_validator_flags_scorer_failures():
    print("\n--- 8. Validator catches what the official scorer rejects or zeroes ---")
    ref = _reference()
    preds = {
        ("match_001_half_1", "dim3_q1"): "o1",
        ("match_001_half_1", "dim3_q2"): "o1",
        ("match_001_half_2", "dim3_q1"): "o1",
    }
    good = build_submission(ref, preds)

    dup = copy.deepcopy(good)
    dup["data"][0]["streaming_vqa"].append(copy.deepcopy(dup["data"][0]["streaming_vqa"][0]))
    assert any("Duplicate" in e for e in validate_submission(dup, ref))

    no_pred = copy.deepcopy(good)
    del no_pred["data"][1]["streaming_vqa"][0]["prediction"]
    assert any("Missing prediction" in e for e in validate_submission(no_pred, ref))

    dropped = copy.deepcopy(good)
    dropped["data"][1]["streaming_vqa"] = []
    assert any("Missing" in e for e in validate_submission(dropped, ref))

    # dim3_q2 has only two options, so "o3" is out of range and would score as incorrect
    bad_opt = copy.deepcopy(good)
    bad_opt["data"][0]["streaming_vqa"][1]["prediction"] = "o3"
    assert any("Invalid option" in e for e in validate_submission(bad_opt, ref))
    print("  [PASS] duplicate, missing, dropped and out-of-range cases flagged")


# --- Optional official-file check -------------------------------------------

def check_official(official_dir: Path):
    print(f"\n--- 13. Official files: rebuild baseline submission from {official_dir} ---")
    gt = json.loads((official_dir / "test_GT.json").read_text(encoding="utf-8"))
    baseline = json.loads((official_dir / "test_submission.json").read_text(encoding="utf-8"))
    preds = {
        (s["id"], q["id"]): q["prediction"]
        for s in baseline["data"] for q in s["streaming_vqa"]
    }
    sub = build_submission(gt, preds)
    errors = validate_submission(sub, gt)
    assert errors == [], errors[:5]
    n = sum(len(s["streaming_vqa"]) for s in sub["data"])
    print(f"  [PASS] {n} questions, submission valid")

    public = json.loads((official_dir / "challenge_public.json").read_text(encoding="utf-8"))
    nulls = {(i["sample_id"], i["question"]["id"]): None for i in iter_questions(public)}
    assert validate_submission(build_submission(public, nulls), public) == []
    print(f"  [PASS] challenge split writer: {len(nulls)} questions")

    # Re-implements the documented metric (unweighted mean of six subcategory
    # accuracies); the organizers report 41.76% / 274 of 700 for this baseline.
    answers = {
        (r["sample_id"], r["question"]["id"]): (r["question"]["correct_option_id"], r["question"]["metadata"]["subcategory"])
        for r in iter_questions(gt)
    }
    hits, totals = {}, {}
    for s in sub["data"]:
        for q in s["streaming_vqa"]:
            answer, subcat = answers[(s["id"], q["id"])]
            totals[subcat] = totals.get(subcat, 0) + 1
            hits[subcat] = hits.get(subcat, 0) + (q["prediction"] == answer)
    score = 100 * sum(hits[k] / totals[k] for k in totals) / len(totals)
    assert len(totals) == 6 and round(score, 2) == 41.76 and sum(hits.values()) == 274, (score, hits)
    print(f"  [PASS] baseline reproduces {score:.2f}% ({sum(hits.values())}/{n})")


def test_challenge_guard_refuses_adapters():
    print("[TEST 9] Model guard: the QLoRA adapter and any PEFT adapter are refused")
    check_challenge_safe_model("Qwen/Qwen3.5-4B")
    with tempfile.TemporaryDirectory(dir=WORKSPACE_ROOT) as base:
        check_challenge_safe_model(base)  # a local base model inside this repo's folder is fine
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "adapter_config.json").write_text("{}", encoding="utf-8")
        refused = [str(WORKSPACE_ROOT / "models" / "qwen3.5-4b-soccernet-qlora"), d,
                   "me/MES-5", "me/mes_5-router", "me/qwen-soccernet-lora", "me/qwen-sft-merged"]
        for ref in refused:
            try:
                check_challenge_safe_model(ref)
            except ValueError:
                continue
            raise AssertionError(f"guard accepted {ref}")
    print("  [PASS] base model accepted; QLoRA adapter and a synthetic adapter refused")


def _excluded_imports(sources):
    """Follows repo-local imports from `sources`; returns (file, module) pairs naming excluded artifacts."""
    hits, seen, queue = [], set(), list(sources)
    while queue:
        path = queue.pop()
        if path in seen:
            continue
        seen.add(path)
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                names = [node.module] + [f"{node.module}.{a.name}" for a in node.names]
            else:
                continue
            for name in names:
                if CHALLENGE_EXCLUDED_NAME_RE.search(name):
                    hits.append((path.name, name))
                module_file = WORKSPACE_ROOT / (name.replace(".", "/") + ".py")
                if module_file.is_file():
                    queue.append(module_file)
    return hits


def test_sn_vqa_code_never_imports_excluded_artifacts():
    print("[TEST 10] No SN-VQA-2027 file imports QLoRA or MES-5 routing code, directly or transitively")
    sources = (
        set(WORKSPACE_ROOT.glob("processing/*sn_vqa_2027*.py"))
        | set(WORKSPACE_ROOT.glob("scripts/*sn_vqa_2027*.py"))
        | {WORKSPACE_ROOT / "processing" / "event_ledger.py", WORKSPACE_ROOT / "scripts" / "test_event_ledger.py"}
    )
    hits = _excluded_imports(sorted(sources))
    assert not hits, f"excluded artifacts imported: {hits}"

    # The scan must catch direct and transitive imports, or the check above proves nothing.
    # evaluate_half2_specialist_and_substitutions has a neutral name but imports the rescoring.
    with tempfile.TemporaryDirectory() as d:
        probe = Path(d) / "probe.py"
        probe.write_text("from scripts.test_mece_routing import get_model_preds_for_half\n", encoding="utf-8")
        assert _excluded_imports([probe]), "scan missed a direct MES-5 routing import"
        probe.write_text("import scripts.evaluate_half2_specialist_and_substitutions\n", encoding="utf-8")
        assert _excluded_imports([probe]), "scan missed a transitive MES-5 rescoring import"
    print(f"  [PASS] {len(sources)} SN-VQA-2027 files clean; scan catches planted direct and transitive imports")


def _record(text, half, pos_sec, options=("A", "B")):
    return {"half": half, "question": {
        "question": text, "position_ms": int(pos_sec * 1000),
        "options": [{"id": f"o{i}", "text": t} for i, t in enumerate(options, 1)],
    }}


def test_question_window_picks_causal_spans():
    print("[TEST 11] question_window: named times, option times and recent spans, never past the question")
    cases = [
        (_record("Which team is taking the corner now?", 1, 600), (1, 584.0, 600.0)),
        (_record("As of now, how many corners has Chelsea taken?", 2, 10), (2, 0.0, 10.0)),
        (_record("Who took the corner that occurred at first half, 03:22?", 1, 300), (1, 192.0, 212.0)),
        # earlier half: no clipping to the current time
        (_record("Who took the shot that occurred at approximately 00:23 in the first half?", 2, 100), (1, 13.0, 33.0)),
        # option clocks span 16:08-16:39 (968-999 s), padded, clipped at now = 1000 s
        (_record("Looking back, when did X last take a shot?", 1, 1000,
                 ("First half, 16:27", "First half, 16:16", "First half, 16:08", "First half, 16:39")), (1, 958.0, 1000.0)),
        # match clock 88:53 is second half 43:53
        (_record("Following Lucas's foul around 88:53, which player struck the wall?", 2, 2650), (2, 2623.0, 2643.0)),
        # second-half stoppage options (45:45-46:15) are half-relative: 2745-2775 s, not 45-75 s
        (_record("When did X take the shot?", 2, 2796,
                 ("Second half, 45:45", "Second half, 45:55", "Second half, 46:05", "Second half, 46:15")), (2, 2735.0, 2785.0)),
        # options straddling 45:00 stay one contiguous span
        (_record("When did X take the corner?", 2, 2750,
                 ("Second half, 44:33", "Second half, 44:45", "Second half, 44:52", "Second half, 45:01")), (2, 2663.0, 2711.0)),
        (_record("Who scored the goal that occurred at second half, 45:45?", 2, 2903), (2, 2735.0, 2755.0)),
        # the 12th minute is 11:00-12:00, padded, clipped at now
        (_record("Around the 12th minute, what led to the corner?", 1, 730), (1, 650.0, 730.0)),
        # a named time in a later half cannot be shown: fall back to the recent span
        (_record("Who took the corner that occurred at second half, 03:22?", 1, 300), (1, 240.0, 300.0)),
        # no time and not "now": the last 60 s
        (_record("What happened when Matic met Hazard's cross in the first half?", 1, 1157), (1, 1097.0, 1157.0)),
    ]
    for record, expected in cases:
        got = question_window(record)
        assert got == expected, (record["question"]["question"], got, expected)
        half, start, end = got
        assert half < record["half"] or end <= record["question"]["position_ms"] / 1000
    print(f"  [PASS] {len(cases)} cases, all spans end at or before the question time")


def test_annotate_option_from_roster():
    print("[TEST 12] annotate_option adds team and shirt number from the pre-match roster")
    prematch = {
        "home": {"name": "Chelsea", "players": [{"long_name": "Diego Costa", "name": "Costa D.", "short_name": "Costa", "shirt_number": "19"}]},
        "away": {"name": "Liverpool", "players": [{"long_name": "James Milner", "name": "Milner J.", "short_name": "Milner", "shirt_number": "7"}]},
    }
    assert annotate_option("Diego Costa", prematch) == "Diego Costa (Chelsea #19)"
    assert annotate_option("james milner", prematch) == "james milner (Liverpool #7)"
    assert annotate_option("Gary Cahill", prematch) == "Gary Cahill"
    assert annotate_option("Chelsea", prematch) == "Chelsea"
    assert annotate_option("Diego Costa", None) == "Diego Costa"
    print("  [PASS] rostered players annotated; teams, unknown names and missing rosters unchanged")


def run_all_tests(official_dir=None):
    test_causal_window_excludes_future_frames()
    test_causal_window_subsample_keeps_latest_frame()
    test_causal_window_at_half_start()
    test_causal_window_rejects_bad_args()
    test_question_time_and_iteration()
    test_submission_round_trip()
    test_writer_rejects_missing_and_unknown_predictions()
    test_validator_flags_scorer_failures()
    test_challenge_guard_refuses_adapters()
    test_sn_vqa_code_never_imports_excluded_artifacts()
    test_question_window_picks_causal_spans()
    test_annotate_option_from_roster()
    if official_dir is not None:
        check_official(official_dir)
    print("\n" + "=" * 60)
    print(" ALL SN-VQA-2027 TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-dir", type=Path, default=None,
                        help="Path to the data/ folder of a SoccerNet/sn-vqa-2027 clone")
    args = parser.parse_args()
    run_all_tests(args.official_dir)
