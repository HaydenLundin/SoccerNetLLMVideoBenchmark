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
import re
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, TypeVar

Frame = TypeVar("Frame")
QuestionKey = Tuple[str, str]

# Artifacts built from matches that are also in the hidden challenge split
# (docs/research/soccernet-2027-fit-audit-2026-11-18.md, section 5). The rules
# forbid using reference annotations, so the SN-VQA-2027 pipeline never loads or
# imports them:
# - the QLoRA adapter (models/qwen3.5-4b-soccernet-qlora), trained on eval_set
#   games 4 and 7; game 7 is likely the challenge West Ham match;
# - MES-5 routing and its Markov-prior rescoring, tuned on games 1 and 2, which
#   are the challenge Man City and West Brom matches.
# scripts/test_sn_vqa_2027.py checks every SN-VQA-2027 file's imports against this.
CHALLENGE_EXCLUDED_NAME_RE = re.compile(r"qlora|mes[-_ ]?5|mece|markov[-_ ]?prior", re.I)
# Model names are also refused if they look fine-tuned: a merged fine-tune has no
# adapter_config.json, so only its name can give it away.
_FINE_TUNED_MODEL_RE = re.compile(r"lora|sft|soccernet", re.I)
_REPO_ROOT = Path(__file__).resolve().parent.parent


def check_challenge_safe_model(model_ref: str) -> None:
    """
    Raises ValueError unless `model_ref` is a base model. Any local PEFT adapter
    is refused, since ours were trained on SoccerNet labels of challenge matches.
    """
    name = model_ref
    if Path(model_ref).exists():  # a local path: ignore the repo's own folder name
        try:
            name = str(Path(model_ref).resolve().relative_to(_REPO_ROOT))
        except ValueError:
            pass
    if (CHALLENGE_EXCLUDED_NAME_RE.search(name) or _FINE_TUNED_MODEL_RE.search(name)
            or (Path(model_ref) / "adapter_config.json").is_file()):
        raise ValueError(
            f"{model_ref!r} is excluded from SN-VQA-2027: our QLoRA adapter and MES-5 routing were "
            "trained or tuned on challenge matches. Use a base model."
        )


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
    """Yields one record per question with its sample id, match key, video path and half."""
    for sample in osl["data"]:
        video_path = next(
            (i["path"] for i in sample.get("inputs", []) if i.get("type") == "video"),
            None,
        )
        half = sample.get("metadata", {}).get("half")
        match_key = sample.get("metadata", {}).get("match_key")
        for question in sample["streaming_vqa"]:
            yield {
                "sample_id": sample["id"],
                "match_key": match_key,
                "video_path": video_path,
                "half": half,
                "question": question,
            }


HALF_SEC = 45 * 60

# Times named in a question's own text, e.g. "at first half, 03:22", "00:23 in the
# first half", "around 88:53" (match clock), "around the 12th minute".
_HALF_CLOCK_RE = re.compile(
    r"(?:(?P<h1>first|second) half, (?P<m1>\d{1,2}):(?P<s1>\d\d))"
    r"|(?:(?P<m2>\d{1,2}):(?P<s2>\d\d) in the (?P<h2>first|second) half)",
    re.I,
)
_MATCH_CLOCK_RE = re.compile(r"\b(?P<m>\d{1,2}):(?P<s>\d\d)\b")
_MINUTE_RE = re.compile(r"\b(?P<n>\d{1,2})(?:st|nd|rd|th) minute\b", re.I)
_OPTION_CLOCK_RE = re.compile(r"^(?P<h>First|Second) half, (?P<m>\d{1,2}):(?P<s>\d\d)$")


def _half_clock(half_word: str, minutes: int, seconds: int) -> Tuple[int, float]:
    # "Second half, 45:45" is stoppage time counted from the start of that half,
    # not match clock: in both splits every such time precedes its question.
    half = 1 if half_word.lower() == "first" else 2
    return half, float(minutes * 60 + seconds)


def option_time(text: str) -> Optional[Tuple[int, float]]:
    """(half, seconds from the start of that half) for an option such as "First half, 03:22"; None otherwise."""
    m = _OPTION_CLOCK_RE.match(text)
    return _half_clock(m["h"], int(m["m"]), int(m["s"])) if m else None


def _match_clock(minutes: int, seconds: int) -> Tuple[int, float]:
    sec = minutes * 60 + seconds
    return (2, float(sec - HALF_SEC)) if sec >= HALF_SEC else (1, float(sec))


def question_window(
    record: Dict[str, Any],
    recent_sec: float = 16.0,
    context_sec: float = 60.0,
    pad_sec: float = 10.0,
) -> Tuple[int, float, float]:
    """
    Returns (half, start_sec, end_sec): the span of video to show for a question,
    in seconds from the start of that half. Never extends past the question time.

    - Options that are all clock times (recall_event_time): the span of the
      options, padded.
    - A time named in the question ("the corner at first half, 03:22", "around
      the 12th minute"): that time, padded.
    - "... now?" / "so far": the last `recent_sec`.
    - Anything else (e.g. "What happened when X met Y's cross?", asked just
      after the play): the last `context_sec`.
    """
    q = record["question"]
    half, now = record["half"], question_time_sec(q)
    lookback = recent_sec if re.search(r"\bnow\b|\bso far\b", q["question"], re.I) else context_sec
    recent = (half, max(0.0, now - lookback), now)

    option_times = [_OPTION_CLOCK_RE.match(o["text"]) for o in q["options"]]
    text = q["question"]
    if option_times and all(option_times):
        refs = [_half_clock(m["h"], int(m["m"]), int(m["s"])) for m in option_times]
        ref_half = refs[0][0]
        secs = [s for h, s in refs if h == ref_half]
        ref = (ref_half, min(secs) - pad_sec, max(secs) + pad_sec)
    elif m := _HALF_CLOCK_RE.search(text):
        if m["h1"]:
            h, s = _half_clock(m["h1"], int(m["m1"]), int(m["s1"]))
        else:
            h, s = _half_clock(m["h2"], int(m["m2"]), int(m["s2"]))
        ref = (h, s - pad_sec, s + pad_sec)
    elif m := _MINUTE_RE.search(text):
        h, s = _match_clock(int(m["n"]) - 1, 30)  # "the 12th minute" is 11:00-12:00
        ref = (h, s - 30 - pad_sec, s + 30 + pad_sec)
    elif m := _MATCH_CLOCK_RE.search(text):
        h, s = _match_clock(int(m["m"]), int(m["s"]))
        ref = (h, s - pad_sec, s + pad_sec)
    else:
        return recent

    ref_half, start, end = ref
    if ref_half > half:
        return recent
    if ref_half == half:
        end = min(end, now)
    start = max(0.0, start)
    if start >= end:
        return recent
    return ref_half, start, end


def annotate_option(text: str, prematch: Optional[Dict[str, Any]]) -> str:
    """
    Appends team and shirt number to an option naming a rostered player, e.g.
    "Diego Costa" -> "Diego Costa (Chelsea #19)". `prematch` is one match of the
    organizers' pre-match file: {"home": {"name", "players": [...]}, "away": ...}.
    Rosters are permitted metadata under RULES.md.
    """
    if not prematch:
        return text
    key = text.casefold()
    for side in ("home", "away"):
        team = prematch[side]
        for p in team.get("players", []):
            names = {p.get(k, "").casefold() for k in ("long_name", "name", "short_name")}
            if key in names:
                return f"{text} ({team['name']} #{p['shirt_number']})"
    return text


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
