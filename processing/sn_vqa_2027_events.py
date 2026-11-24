"""
processing/sn_vqa_2027_events.py
Event detection over fixed windows of a half, and how to score it.

Detection (scripts/detect_events_sn_vqa_2027.py) asks a VLM which kind of event
each short window shows and keeps the per-class probabilities. This module holds
the parts that need no model or video:
- `windows`: groups decoded frames into fixed windows at a sampling rate;
- the prompts (`CLASS_SETS`, `window_messages`);
- `recall_event_kind` and `option_window`, which let the video QA runner answer
  recall_event_time questions by scoring a window centred on each option (section 12);
- scoring against the test split's own event moments, which is diagnostic only:
  test labels measure the detector, never feed it.

Matching rules (research note section 11):
- an event happens about EVENT_LAG s before its "... now?" question;
- a detection (start, end) finds an event at t if start - lead <= t <= end;
- each detection is credited to at most one event.

Pure Python, no model or video dependencies.
"""

import re
import bisect
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

# Prompts: each class is (key, description shown to the model as a lettered option).
CLASS_SETS: Dict[str, List[Tuple[str, str]]] = {
    # v2: a strict shot definition plus decoy classes that absorb clips a looser prompt called shots.
    "v2": [
        ("none", "Open play away from the goal: passing, dribbling, tackles, players running"),
        ("replay", "A slow-motion or repeated replay of earlier play"),
        ("closeup", "A close-up of players, coaches, the referee or the crowd, with no live play"),
        ("goal_kick", "A goal kick, or the goalkeeper holding, rolling or kicking the ball out"),
        ("throw_in", "A throw-in from the touchline"),
        ("cross", "A cross or pass into the penalty area that is headed or kicked away, with no attempt on goal"),
        ("shot", "A deliberate attempt to score: a player strikes or heads the ball towards the goal and the keeper saves it, it is blocked, or it goes in, wide or over. Passes, crosses and clearances are not shots"),
        ("corner", "A corner kick being taken or set up at the corner flag"),
        ("free_kick", "A free kick being taken or set up after a foul"),
        ("offside", "An assistant referee raising the flag for offside"),
        ("substitution", "A substitution: players leaving or entering, the substitution board"),
        ("card", "The referee showing a yellow or red card"),
    ],
    # shot_bin: one yes/no question per longer window, using the broadcast cues that follow a shot.
    "shot_bin": [
        ("no", "No attempt on goal in this clip"),
        ("shot", "A player shoots or heads the ball at goal in this clip, or the clip shows the aftermath of a shot: "
                 "the keeper diving or saving, a player with hands on head after a near miss, celebrations, or a replay of the attempt"),
    ],
}
LETTERS = "ABCDEFGHIJKL"

EVENT_LAG = 2.5  # event questions come ~2-3 s after the event
_KIND_RE = re.compile(r"\b(corner|shot|free kick|goal|substitution|card|offside)\b", re.I)
_RECALL_KIND_RE = re.compile(r"\b(shot|goal|corner|free kick|substitut\w*|card|offside|injury)\b", re.I)
_RECALL_KIND = {"shot": "shot", "goal": "shot", "corner": "corner", "free kick": "free_kick", "card": "card",
                "offside": "offside", "injury": None}

Window = Tuple[float, float]


def window_messages(images: Sequence[object], window_sec: float, prompt: str = "v2") -> List[Dict]:
    """Chat messages asking which class best describes a clip, with the classes as lettered options."""
    options = "\n".join(f"{letter}. {desc}" for letter, (_, desc) in zip(LETTERS, CLASS_SETS[prompt]))
    content = [{"type": "text", "text": f"These are {len(images)} consecutive frames, {window_sec:.0f} seconds of a live soccer broadcast, oldest first."}]
    content += [{"type": "image", "image": image} for image in images]
    content.append({"type": "text", "text": f"\nWhich best describes what happens in this clip?\n{options}\n\nAnswer with a single letter."})
    return [{"role": "user", "content": content}]


def recall_event_kind(question: str) -> Optional[str]:
    """
    The v2 class of the event a "Looking back, when did ...?" question asks about (goals
    count as shots), or None when v2 has no class for it (injuries) or no kind is named.
    """
    m = _RECALL_KIND_RE.search(question)
    if not m:
        return None
    word = m.group(1).lower()
    return "substitution" if word.startswith("substitut") else _RECALL_KIND[word]


def option_window(option_half: int, option_sec: float, now_half: int, now_sec: float,
                  window: float) -> Optional[Window]:
    """
    A `window`-second span centred on an option time, in seconds from the start of the
    option's half. In the question's own half it is moved earlier so it never ends after
    the question time. None if the option is after the question time.
    """
    if (option_half, option_sec) > (now_half, now_sec):
        return None
    start, end = option_sec - window / 2, option_sec + window / 2
    if option_half == now_half and end > now_sec:
        start, end = now_sec - window, now_sec
    return max(0.0, start), end


def windows(frames: Iterable[Tuple[float, object]], window: float, fps: float,
            convert: Callable[[object], object] = lambda f: f) -> Iterator[Tuple[float, float, list]]:
    """
    Groups decoded (time, frame) pairs, oldest first, into consecutive windows of
    `window` s, keeping at most `fps` frames per second (converted with `convert`).
    Yields (start, end, [(time, converted frame)]). Empty windows (gaps) are skipped,
    and the last, partial window ends at its last kept frame.
    """
    step, next_t, buf, w_start = 1.0 / fps, 0.0, [], 0.0
    for t, frame in frames:
        while t >= w_start + window:
            if buf:
                yield w_start, w_start + window, buf
            w_start += window
            buf = []
        if t >= next_t:
            buf.append((t, convert(frame)))
            next_t = t + step
    if buf:
        yield w_start, buf[-1][0], buf


def truth_events(gt: Dict, match_key: str, half: int) -> List[Tuple[float, str]]:
    """
    (event_sec, kind) pairs from the "Which team / Who is taking the <kind> now?"
    questions of one half; team and player questions at one moment collapse to one
    event, and goals count as shots. Diagnostic only: these are test labels.
    """
    events = set()
    for sample in gt["data"]:
        md = sample["metadata"]
        if md.get("match_key") != match_key or md.get("half") != half:
            continue
        for q in sample["streaming_vqa"]:
            if q["metadata"]["subcategory"] not in ("event_team", "event_player"):
                continue
            m = _KIND_RE.search(q["question"])
            if not m:
                continue
            kind = m.group(1).lower().replace("free kick", "free_kick")
            events.add((q["position_ms"] / 1000 - EVENT_LAG, "shot" if kind == "goal" else kind))
    return sorted(events)


def covers(det: Window, t: float, lead: float = 0.0) -> bool:
    """Whether a detection finds an event at t. `lead` lets the detection start up to that
    many seconds after the event (cues such as reactions and replays follow it)."""
    return det[0] - lead <= t <= det[1]


def match(dets: Sequence[Window], events: Sequence[float], lead: float = 0.0) -> Tuple[int, int]:
    """One-to-one greedy matching in time order. Returns (events found, detections that are real)."""
    used, found = set(), 0
    for t in sorted(events):
        for i, d in enumerate(dets):
            if i not in used and covers(d, t, lead):
                used.add(i)
                found += 1
                break
    return found, len(used)


def merge_above(rows: Sequence[Dict], kind: str, thr: float) -> List[Window]:
    """Merges consecutive windows with p(kind) >= thr into one detection each."""
    out, cur = [], None
    for r in rows:
        if r["p"].get(kind, 0.0) >= thr:
            cur = (cur[0], r["end"]) if cur else (r["start"], r["end"])
        elif cur:
            out.append(cur)
            cur = None
    if cur:
        out.append(cur)
    return out


def peaks(rows: Sequence[Dict], kind: str, thr: Optional[float] = None, n: Optional[int] = None,
          sep: float = 30.0) -> List[Window]:
    """
    Peak picking: windows in the given order (by default highest p(kind) first), skipping
    any that start within `sep` s of one already taken; stops below `thr` or after `n`.
    Each detection is one window, so the number of detections is the predicted count.
    """
    order = sorted(rows, key=lambda r: -r["p"][kind]) if thr is not None else list(rows)
    taken: List[Dict] = []
    for r in order:
        if thr is not None and r["p"][kind] < thr:
            break
        if all(abs(r["start"] - t["start"]) > sep for t in taken):
            taken.append(r)
            if n is not None and len(taken) == n:
                break
    return sorted((r["start"], r["end"]) for r in taken)


def causal_percentile_rows(rows: Sequence[Dict], kind: str, warmup: float = 0.0) -> List[Dict]:
    """
    Rows (time order, one half) with p(kind) replaced by its causal percentile: the share of
    this half's windows so far, itself included, with p(kind) at or below its own. Later video
    never changes a window's value, so a cut-off on it is a per-half cut-off a streaming
    answerer may use. Windows starting before `warmup` s get 0: a percentile over a handful
    of windows is noise.
    """
    seen: List[float] = []
    out = []
    for r in rows:
        p = r["p"][kind]
        bisect.insort(seen, p)
        pct = bisect.bisect_right(seen, p) / len(seen) if r["start"] >= warmup else 0.0
        out.append({**r, "p": {kind: pct}})
    return out


def auc(rows: Sequence[Dict], kind: str, events: Sequence[float]) -> Optional[float]:
    """AUC of p(kind): windows containing an event vs all others (ties count half)."""
    pos = [r["p"][kind] for r in rows if any(covers((r["start"], r["end"]), t) for t in events)]
    neg = [r["p"][kind] for r in rows if not any(covers((r["start"], r["end"]), t) for t in events)]
    if not pos or not neg:
        return None
    return sum((a > b) + 0.5 * (a == b) for a in pos for b in neg) / (len(pos) * len(neg))


def worst_case_ranks(rows: Sequence[Dict], kind: str, events: Sequence[float]) -> List[int]:
    """For each event, the rank of the window containing it, counting every tie as ranked above."""
    ranks = []
    for t in events:
        w = next((r for r in rows if covers((r["start"], r["end"]), t)), None)
        if w is not None:
            ranks.append(sum(r["p"][kind] >= w["p"][kind] for r in rows))
    return sorted(ranks)


def f1(found: float, n_events: float, real: float, n_dets: float) -> float:
    precision = real / n_dets if n_dets else 0.0
    recall = found / n_events if n_events else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0
