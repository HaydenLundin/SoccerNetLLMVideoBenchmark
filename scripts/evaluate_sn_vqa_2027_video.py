"""
scripts/evaluate_sn_vqa_2027_video.py
Local video QA for the SoccerNet 2027 Streaming VQA challenge (Qwen3.5-4B, NF4, 8 GB GPU).

Per question: `question_window` picks a span of video that ends no later than the
question time (the last few seconds for "now" questions, the named time for recall
questions); frames are decoded from that span only and passed with the question,
its options (players annotated with team and shirt number from the pre-match
roster) and the pre-match facts. No other question, answer or label is an input,
so it stays within RULES.md's permitted inputs under either reading.

The answer is the argmax next-token logit over the option letters, as in the
text-only baseline, so runs are deterministic and every prediction is valid.

With --recall-detector, recall_event_time questions ("Looking back, when did X
last take a shot?") are answered differently: each option time gets a short
window centred on it (never past the question time), the window is classified
with the event detector's v2 prompt, and the option with the highest p(kind)
wins. Questions about kinds v2 has no class for (injuries) use the normal path.
Research note section 12.

Usage (test split):
    python scripts/evaluate_sn_vqa_2027_video.py --gt <sn-vqa-2027>/data/test_GT.json \
        --videos <SN-VQA-2027 dataset root> --prematch <MatchLive-baseline>/data/prematch/test.json
"""

import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

from processing.sn_vqa_2027 import (
    iter_questions, build_submission, validate_submission, check_challenge_safe_model,
    causal_frame_window, question_window, annotate_option, option_time, question_time_sec,
)
from processing.sn_vqa_2027_events import CLASS_SETS, LETTERS as CLASS_LETTERS, window_messages, recall_event_kind, option_window
from scripts.evaluate_sn_vqa_2027_text_only import LETTERS, format_clock, score_by_subcategory

DECODE_STEP_SEC = 0.2  # keep at most 5 decoded frames per second before subsampling


class HalfVideos:
    """Opens each half's video once and decodes short spans of it."""

    def __init__(self, root: Path, reference: Dict):
        import av
        self._av = av
        self._paths = {}
        for r in iter_questions(reference):
            self._paths[(r["match_key"], r["half"])] = root / r["video_path"]
        self._open = {}

    def frames(self, match_key: str, half: int, start: float, end: float, n: int, width: int) -> List[Tuple[float, "Image"]]:
        """Up to `n` frames in [start, end], oldest first, as (seconds, PIL image)."""
        path = self._paths[(match_key, half)]
        if path not in self._open:
            container = self._av.open(str(path))
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            offset = float(stream.start_time * stream.time_base) if stream.start_time else 0.0
            self._open[path] = (container, stream, offset)
        container, stream, offset = self._open[path]
        container.seek(int((start + offset) / stream.time_base), stream=stream, backward=True)

        height = int(round(width * 9 / 16 / 2)) * 2
        grid, next_t = [], start
        for f in container.decode(stream):
            t = f.time - offset
            if t > end:
                break
            if t >= next_t:
                grid.append((t, f.to_image(width=width, height=height)))
                next_t = t + DECODE_STEP_SEC
        return causal_frame_window(grid, end, end - start, n)


def prematch_lines(prematch: Dict, half: int) -> List[str]:
    if not prematch:
        return []
    lines = []
    for side in ("home", "away"):
        team = prematch[side]
        kit = team.get("kit_color", {}).get("name", "unknown")
        defends = team.get("defends", {}).get(str(half))
        where = f", defending the {defends} goal this half" if defends else ""
        lines.append(f"{team['name']} ({side}): {kit} kit{where}.")
    return lines


def build_messages(record: Dict, frames: List[Tuple[float, object]], half_shown: int, prematch: Dict) -> List[Dict]:
    q = record["question"]
    content = [{"type": "text", "text": (
        "You are watching a live professional soccer broadcast and answering a multiple-choice question.\n"
        + "\n".join(prematch_lines(prematch, half_shown))  # directions for the half the frames come from
        + "\nFrames from the broadcast follow, oldest first, each labelled with its match time. "
        "Nothing after the current time is shown. Replays may repeat earlier play."
    )}]
    for t, image in frames:
        content.append({"type": "text", "text": f"\n[{format_clock(int(t * 1000), half_shown)}]"})
        content.append({"type": "image", "image": image})
    options = "\n".join(
        f"{letter}. {annotate_option(o['text'], prematch)}" for letter, o in zip(LETTERS, q["options"])
    )
    content.append({"type": "text", "text": (
        f"\n\nCurrent time: {format_clock(q['position_ms'], record['half'])}\n"
        f"Question: {q['question']}\n{options}\n\nAnswer with a single letter."
    )})
    return [{"role": "user", "content": content}]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", type=Path, required=True, help="OSL questions file (test_GT.json, or challenge_public.json)")
    parser.add_argument("--videos", type=Path, required=True, help="Root that the OSL video paths are relative to")
    parser.add_argument("--prematch", type=Path, default=None, help="Organizers' pre-match file (rosters, kits)")
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--frame-width", type=int, default=448)
    parser.add_argument("--recent-sec", type=float, default=16.0)
    parser.add_argument("--context-sec", type=float, default=60.0)
    parser.add_argument("--recall-detector", action="store_true",
                        help="Answer recall_event_time by classifying a window centred on each option time")
    parser.add_argument("--recall-window", type=float, default=4.0, help="Seconds per option window")
    parser.add_argument("--recall-frames", type=int, default=12, help="Frames per option window")
    parser.add_argument("--subcategories", nargs="+", default=None, help="Score only these subcategories")
    parser.add_argument("--limit", type=int, default=None, help="Score only the first N questions (smoke test)")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "logs" / "sn_vqa_2027")
    parser.add_argument("--tag", default="video", help="Output folder name")
    args = parser.parse_args()
    check_challenge_safe_model(args.model)

    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
    import transformers.integrations.sdpa_attention as sdpa_attention

    # On the torch 2.6 Windows wheel there is no FlashAttention and the memory-efficient kernel
    # rejects enable_gqa, so transformers' GQA path falls back to the O(n^2) math kernel.
    # Expanding KV heads first (repeat_kv) lets SDPA use the memory-efficient kernel instead.
    sdpa_attention.use_gqa_in_sdpa = lambda *a, **k: False

    reference =json.loads(args.gt.read_text(encoding="utf-8"))
    prematch_all = json.loads(args.prematch.read_text(encoding="utf-8")) if args.prematch else {}
    records = list(iter_questions(reference))
    if args.subcategories:
        records = [r for r in records if r["question"]["metadata"]["subcategory"] in args.subcategories]
    if args.limit:
        records = records[: args.limit]
    has_labels = all("correct_option_id" in r["question"] for r in records)
    videos = HalfVideos(args.videos, reference)

    print(f"[VIDEO] Loading {args.model} (NF4 4-bit)...")
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
        ),
    ).eval()
    processor = AutoProcessor.from_pretrained(args.model)
    letter_ids = [processor.tokenizer.encode(l, add_special_tokens=False)[0] for l in LETTERS]
    class_keys = [k for k, _ in CLASS_SETS["v2"]]
    class_ids = [processor.tokenizer.encode(l, add_special_tokens=False)[0] for l in CLASS_LETTERS[: len(class_keys)]]

    def answer_by_option_windows(rec, kind, times):
        """Option whose centred window has the highest p(kind); (prediction, windows, n_frames, max prompt tokens)."""
        q, now = rec["question"], question_time_sec(rec["question"])
        best, spans, n_frames, n_tokens = (-1.0, None), [], 0, 0
        for option, (half, sec) in zip(q["options"], times):
            span = option_window(half, sec, rec["half"], now, args.recall_window)
            frames = videos.frames(rec["match_key"], half, span[0], span[1], args.recall_frames, args.frame_width) if span else []
            p = 0.0
            if frames:
                inputs = processor.apply_chat_template(
                    window_messages([image for _, image in frames], args.recall_window),
                    add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt",
                    enable_thinking=False,
                ).to("cuda:0")
                with torch.inference_mode():
                    logits = model(**inputs, logits_to_keep=1).logits[0, -1]
                p = float(torch.softmax(logits[class_ids].float(), 0)[class_keys.index(kind)])
                n_tokens = max(n_tokens, int(inputs["input_ids"].shape[1]))
                del inputs, logits
            spans.append([half, round(span[0], 1), round(span[1], 1)] if span else None)
            n_frames += len(frames)
            if p > best[0]:
                best = (p, option["id"])
        return best[1], spans, n_frames, n_tokens

    predictions: Dict = {}
    rows = []
    t_run = time.perf_counter()
    for i, rec in enumerate(records, 1):
        q = rec["question"]
        prematch = prematch_all.get(rec["match_key"])
        kind = times = None
        if args.recall_detector and q["metadata"]["subcategory"] == "recall_event_time":
            kind, times = recall_event_kind(q["question"]), [option_time(o["text"]) for o in q["options"]]
        if kind and all(times):
            pred, window, n_frames, n_tokens = answer_by_option_windows(rec, kind, times)
        else:
            kind = None
            half_shown, start, end = question_window(rec, args.recent_sec, args.context_sec)
            frames = videos.frames(rec["match_key"], half_shown, start, end, args.frames, args.frame_width)
            inputs = processor.apply_chat_template(
                build_messages(rec, frames, half_shown, prematch),
                add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt",
                enable_thinking=False,
            ).to("cuda:0")
            with torch.inference_mode():
                logits = model(**inputs).logits[0, -1]
            option_logits = logits[letter_ids[: len(q["options"])]].float()
            pred = q["options"][int(option_logits.argmax())]["id"]
            window, n_frames = [half_shown, round(start, 1), round(end, 1)], len(frames)
            n_tokens = int(inputs["input_ids"].shape[1])
            del inputs, logits
        predictions[(rec["sample_id"], q["id"])] = pred
        rows.append({
            "sample_id": rec["sample_id"], "question_id": q["id"],
            "subcategory": q["metadata"]["subcategory"], "prediction": pred,
            "correct": pred == q.get("correct_option_id"),
            "window": window, "n_frames": n_frames, "prompt_tokens": n_tokens,
            "answered_by": f"option windows, p({kind})" if kind else "question",
        })
        if i % 25 == 0 or i == len(records):
            acc = sum(r["correct"] for r in rows) / len(rows)
            print(f"  [{i}/{len(records)}] running accuracy {acc:.3f} | {time.perf_counter() - t_run:.0f}s")

    out_dir = args.out / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.limit and not args.subcategories:
        submission = build_submission(reference, predictions)
        errors = validate_submission(submission, reference)
        assert errors == [], errors[:5]
        (out_dir / "submission.json").write_text(json.dumps(submission), encoding="utf-8")

    metrics = {
        "model": args.model, "precision": "nf4", "input": "causal video frames + question + pre-match roster",
        "frames": args.frames, "frame_width": args.frame_width,
        "recent_sec": args.recent_sec, "context_sec": args.context_sec,
        "recall_detector": args.recall_detector, "recall_window": args.recall_window, "recall_frames": args.recall_frames,
        "subcategories": args.subcategories, "limit": args.limit,
        "wall_sec": round(time.perf_counter() - t_run, 1),
        "sec_per_question": round((time.perf_counter() - t_run) / max(1, len(rows)), 2),
        "max_prompt_tokens": max(r["prompt_tokens"] for r in rows),
    }
    if has_labels:
        totals = {}
        for r in rows:
            totals.setdefault(r["subcategory"], []).append(r["correct"])
        metrics["per_subcategory_pct"] = {k: round(100 * sum(v) / len(v), 2) for k, v in totals.items()}
        metrics["n_per_subcategory"] = {k: len(v) for k, v in totals.items()}
        if not args.limit and not args.subcategories:
            metrics.update(score_by_subcategory(records, predictions))
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with open(out_dir / "predictions.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
