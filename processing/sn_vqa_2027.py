"""
processing/sn_vqa_2027.py
Causal frame selection and OSL JSON submission writer for the SoccerNet 2027
Streaming VQA challenge (github.com/SoccerNet/sn-vqa-2027).

Rules this module enforces (from the challenge RULES.md / evaluation README):
- Streaming: a question asked at `position_ms` may only use frames at or before
  that time. `causal_frame_window` is the single place frames are chosen, so the
  cutoff cannot drift between runners.
- Submissions are OSL JSON: `data[].streaming_vqa[]`, each question carrying a
  `prediction` that is an option id (e.g. "o2") or null. Answers are matched by
  (sample id, question id), because question ids repeat across halves.

Pure Python, no model or video dependencies, so it runs in CPU-only CI.
"""

import copy
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, TypeVar

Frame = TypeVar("Frame")
QuestionKey = Tuple[str, str]


def question_time_sec(question: Dict[str, Any]) -> float:
    """Seconds from the start of the half at which the question is asked."""
    return question["position_ms"] / 1000.0


def causal_frame_window(
    frames: Sequence[Tuple[float, Frame]],
    query_sec: float,
    window_sec: float,
    max_frames: int,
) -> List[Tuple[float, Frame]]:
    """
    Returns the frames in [query_sec - window_sec, query_sec], oldest first,
    uniformly subsampled to at most `max_frames`. The newest causal frame is
    always kept. Frames after `query_sec` are never returned.

    `frames` are (seconds_from_half_start, frame) pairs, as produced by
    scripts/run_local_benchmark.decode_chunk_frames_single_pass on a full half.
    """
    if window_sec <= 0:
        raise ValueError(f"window_sec must be positive, got {window_sec}")
    if max_frames < 1:
        raise ValueError(f"max_frames must be at least 1, got {max_frames}")

    start_sec = query_sec - window_sec
    eligible = sorted(
        (f for f in frames if start_sec <= f[0] <= query_sec),
        key=lambda f: f[0],
    )
    if len(eligible) <= max_frames:
        return eligible
    if max_frames == 1:
        return [eligible[-1]]

    step = (len(eligible) - 1) / (max_frames - 1)
    return [eligible[int(round(i * step))] for i in range(max_frames)]


def iter_questions(osl: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
    """Yields one record per question with its sample id, video path and half."""
    for sample in osl["data"]:
        video_path = next(
            (i["path"] for i in sample.get("inputs", []) if i.get("type") == "video"),
            None,
        )
        half = sample.get("metadata", {}).get("half")
        for question in sample["streaming_vqa"]:
            yield {
                "sample_id": sample["id"],
                "video_path": video_path,
                "half": half,
                "question": question,
            }


def build_submission(
    reference: Dict[str, Any],
    predictions: Dict[QuestionKey, Optional[str]],
) -> Dict[str, Any]:
    """
    Builds an OSL submission from a reference file (test_GT.json or
    challenge_public.json) and a {(sample_id, question_id): option_id_or_None} map.

    Raises ValueError if any reference question lacks a prediction or if a
    prediction targets an unknown question, since the scorer rejects both.
    Answer keys are stripped; question metadata (including `category`) is kept.
    """
    expected = {(r["sample_id"], r["question"]["id"]) for r in iter_questions(reference)}
    missing = expected - predictions.keys()
    extra = predictions.keys() - expected
    if missing:
        raise ValueError(f"{len(missing)} question(s) without a prediction, e.g. {sorted(missing)[:3]}")
    if extra:
        raise ValueError(f"{len(extra)} prediction(s) for unknown questions, e.g. {sorted(extra)[:3]}")

    submission = copy.deepcopy(reference)
    for sample in submission["data"]:
        for question in sample["streaming_vqa"]:
            question.pop("correct_option_id", None)
            question["prediction"] = predictions[(sample["id"], question["id"])]
    return submission


def validate_submission(submission: Dict[str, Any], reference: Dict[str, Any]) -> List[str]:
    """
    Checks a submission against a reference the way evaluation/scoring.py does,
    plus option ranges. Returns a list of problems (empty means valid).

    Fatal for the scorer: duplicate, missing or unknown (sample, question) keys,
    and questions without a `prediction` field. Not fatal but scored as
    incorrect: a prediction that is not null and not one of the question's
    option ids; reported here so it is caught before upload.
    """
    errors: List[str] = []
    options: Dict[QuestionKey, set] = {}
    for record in iter_questions(reference):
        key = (record["sample_id"], record["question"]["id"])
        options[key] = {o["id"] for o in record["question"].get("options", [])}

    seen = set()
    for sample in submission.get("data", []):
        sample_id = sample.get("id")
        questions = sample.get("streaming_vqa")
        if not isinstance(questions, list):
            errors.append(f"{sample_id}: missing streaming_vqa list")
            continue
        for question in questions:
            key = (sample_id, question.get("id"))
            if key in seen:
                errors.append(f"Duplicate submitted question: {key[0]}/{key[1]}")
                continue
            seen.add(key)
            if key not in options:
                errors.append(f"Unknown question: {key[0]}/{key[1]}")
                continue
            if "prediction" not in question:
                errors.append(f"Missing prediction: {key[0]}/{key[1]}")
                continue
            pred = question["prediction"]
            if pred is not None and pred not in options[key]:
                errors.append(f"Invalid option {pred!r} for {key[0]}/{key[1]}")

    for key in sorted(options.keys() - seen):
        errors.append(f"Missing question: {key[0]}/{key[1]}")
    return errors
