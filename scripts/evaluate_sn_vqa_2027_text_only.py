"""
scripts/evaluate_sn_vqa_2027_text_only.py
Text-only (no video) baseline for the SoccerNet 2027 Streaming VQA test split.

Measures how much of each subcategory can be answered without seeing the match.
Two context modes:
  question  the question and its options only
  history   also every question asked earlier in the same half (questions only,
            never answers), i.e. what a streaming system has already been asked.
            Probes whether the question stream itself leaks facts.

Answers are chosen by the next-token logit over the option letters, so every
prediction is a valid option and the run is deterministic (no sampling, no parsing).

Usage:
    python scripts/evaluate_sn_vqa_2027_text_only.py --gt <sn-vqa-2027>/data/test_GT.json --mode question
"""

import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

from processing.sn_vqa_2027 import iter_questions, build_submission, validate_submission, check_challenge_safe_model

SUBCATEGORIES = [
    "counting", "recall_event_time", "recall_event_player",
    "event_team", "event_player", "causal reasoning",
]
LETTERS = "ABCD"


def format_clock(position_ms: int, half: int) -> str:
    sec = position_ms // 1000
    return f"{'First' if half == 1 else 'Second'} half, {sec // 60:02d}:{sec % 60:02d}"


def build_prompt(record: Dict, history: List[Dict]) -> str:
    q = record["question"]
    lines = [
        "You are answering a multiple-choice question about a live professional soccer broadcast.",
        "You cannot see the video. Use only the text below and pick the most likely option.",
    ]
    if history:
        lines.append("\nQuestions already asked earlier in this half (answers not given):")
        for h in history:
            hq = h["question"]
            opts = "; ".join(o["text"] for o in hq["options"])
            lines.append(f"- [{format_clock(hq['position_ms'], h['half'])}] {hq['question']} Options: {opts}")
    lines.append(f"\nCurrent time: {format_clock(q['position_ms'], record['half'])}")
    lines.append(f"Question: {q['question']}")
    for letter, opt in zip(LETTERS, q["options"]):
        lines.append(f"{letter}. {opt['text']}")
    lines.append("\nAnswer with a single letter.")
    return "\n".join(lines)


def score_by_subcategory(records: List[Dict], predictions: Dict) -> Dict:
    hits = {s: 0 for s in SUBCATEGORIES}
    totals = {s: 0 for s in SUBCATEGORIES}
    for r in records:
        q = r["question"]
        sub = q["metadata"]["subcategory"]
        totals[sub] += 1
        hits[sub] += predictions[(r["sample_id"], q["id"])] == q["correct_option_id"]
    # Subcategories absent from a partial run (--limit, --subcategories) are left out of the mean
    per = {s: 100.0 * hits[s] / totals[s] for s in SUBCATEGORIES if totals[s]}
    return {
        "per_subcategory_pct": {s: round(v, 2) for s, v in per.items()},
        "subcategory_mean_pct": round(sum(per.values()) / len(per), 2),
        "overall_correct": sum(hits.values()),
        "overall_total": sum(totals.values()),
        "n_per_subcategory": totals,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", type=Path, required=True, help="Path to official test_GT.json")
    parser.add_argument("--mode", choices=["question", "history"], default="question")
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--limit", type=int, default=None, help="Score only the first N questions (smoke test)")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "logs" / "sn_vqa_2027")
    args = parser.parse_args()
    check_challenge_safe_model(args.model)

    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig

    reference = json.loads(args.gt.read_text(encoding="utf-8"))
    records = list(iter_questions(reference))
    if args.limit:
        records = records[: args.limit]

    print(f"[TEXT-ONLY] Loading {args.model} (NF4 4-bit)...")
    t0 = time.perf_counter()
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
        ),
    ).eval()
    processor = AutoProcessor.from_pretrained(args.model)
    tokenizer = processor.tokenizer
    letter_ids = [tokenizer.encode(l, add_special_tokens=False)[0] for l in LETTERS]
    print(f"[TEXT-ONLY] Loaded in {time.perf_counter() - t0:.1f}s; letter token ids {letter_ids}")

    predictions: Dict = {}
    rows = []
    t_run = time.perf_counter()
    for i, rec in enumerate(records, 1):
        q = rec["question"]
        history = []
        if args.mode == "history":
            history = [
                r for r in iter_questions(reference)
                if r["sample_id"] == rec["sample_id"] and r["question"]["position_ms"] < q["position_ms"]
            ]
        prompt = build_prompt(rec, history)
        text = processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        inputs = tokenizer(text, return_tensors="pt").to("cuda:0")
        with torch.inference_mode():
            logits = model(**inputs).logits[0, -1]
        n_opts = len(q["options"])
        option_logits = logits[letter_ids[:n_opts]].float()
        choice = int(option_logits.argmax())
        pred = q["options"][choice]["id"]
        predictions[(rec["sample_id"], q["id"])] = pred
        rows.append({
            "sample_id": rec["sample_id"], "question_id": q["id"],
            "subcategory": q["metadata"]["subcategory"], "prediction": pred,
            "correct": pred == q["correct_option_id"], "prompt_tokens": int(inputs["input_ids"].shape[1]),
        })
        del inputs, logits
        if i % 50 == 0 or i == len(records):
            acc = sum(r["correct"] for r in rows) / len(rows)
            print(f"  [{i}/{len(records)}] running accuracy {acc:.3f} | {time.perf_counter() - t_run:.0f}s")

    if not args.limit:
        errors = validate_submission(build_submission(reference, predictions), reference)
        assert errors == [], errors[:5]

    metrics = score_by_subcategory(records, predictions)
    metrics.update({
        "model": args.model, "precision": "nf4", "mode": args.mode, "input": "text only (no video)",
        "selection": "argmax next-token logit over option letters",
        "limit": args.limit, "wall_sec": round(time.perf_counter() - t_run, 1),
        "max_prompt_tokens": max(r["prompt_tokens"] for r in rows),
    })
    out_dir = args.out / f"text_only_{args.mode}" if not args.limit else args.out / f"text_only_{args.mode}_smoke"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with open(out_dir / "predictions.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
