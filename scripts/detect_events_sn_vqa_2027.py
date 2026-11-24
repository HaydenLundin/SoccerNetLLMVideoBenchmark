"""
scripts/detect_events_sn_vqa_2027.py
Scores every short window of one half for "what kind of event is this?" with
local Qwen3.5-4B (NF4, 8 GB GPU), using next-token letter logits (no generation).
Saves per-window class probabilities so detection rules can be studied offline
with scripts/score_events_sn_vqa_2027.py. Research note section 11.

Each window sees only its own frames, so a detection is causal if it is
timestamped at the window's end.

Usage (one half; the defaults are the v2 prompt on 6 s windows at 2 fps, 448 px):
    python scripts/detect_events_sn_vqa_2027.py <video-test/game1/1_720p.mkv> \
        logs/sn_vqa_2027/events/game1_h1_v2.json
"""

import sys
import json
import time
import itertools
import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

from processing.sn_vqa_2027 import check_challenge_safe_model
from processing.sn_vqa_2027_events import CLASS_SETS, LETTERS, windows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--prompt", default="v2", choices=sorted(CLASS_SETS))
    ap.add_argument("--window", type=float, default=6.0, help="Window length in seconds")
    ap.add_argument("--fps", type=float, default=2.0, help="Frames kept per second")
    ap.add_argument("--width", type=int, default=448, help="Frame width in pixels (16:9)")
    ap.add_argument("--limit-sec", type=float, default=None, help="Stop after this many seconds (smoke test)")
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    a = ap.parse_args()
    check_challenge_safe_model(a.model)

    import av
    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
    import transformers.integrations.sdpa_attention as sdpa_attention

    # On the torch 2.6 Windows wheel there is no FlashAttention and the memory-efficient kernel
    # rejects enable_gqa, so transformers' GQA path falls back to the O(n^2) math kernel.
    # Expanding KV heads first (repeat_kv) lets SDPA use the memory-efficient kernel instead.
    sdpa_attention.use_gqa_in_sdpa = lambda *args, **kwargs: False

    model = AutoModelForImageTextToText.from_pretrained(
        a.model, torch_dtype=torch.bfloat16, device_map={"": "cuda:0"},
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
        ),
    ).eval()
    proc = AutoProcessor.from_pretrained(a.model)
    classes = CLASS_SETS[a.prompt]
    ids = [proc.tokenizer.encode(letter, add_special_tokens=False)[0] for letter in LETTERS[: len(classes)]]
    options = "\n".join(f"{letter}. {desc}" for letter, (_, desc) in zip(LETTERS, classes))

    container = av.open(str(a.video))
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"
    height = int(round(a.width * 9 / 16 / 2)) * 2
    frames = itertools.takewhile(lambda tf: a.limit_sec is None or tf[0] <= a.limit_sec,
                                 ((float(f.time), f) for f in container.decode(stream)))

    rows, t0 = [], time.perf_counter()
    for start, end, kept in windows(frames, a.window, a.fps, convert=lambda f: f.to_image(width=a.width, height=height)):
        content = [{"type": "text", "text": f"These are {len(kept)} consecutive frames, {a.window:.0f} seconds of a live soccer broadcast, oldest first."}]
        content += [{"type": "image", "image": image} for _, image in kept]
        content.append({"type": "text", "text": f"\nWhich best describes what happens in this clip?\n{options}\n\nAnswer with a single letter."})
        inputs = proc.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True, tokenize=True,
                                          return_dict=True, return_tensors="pt", enable_thinking=False).to("cuda:0")
        with torch.inference_mode():
            logits = model(**inputs, logits_to_keep=1).logits[0, -1]
        p = torch.softmax(logits[ids].float(), 0).tolist()
        rows.append({"start": start, "end": end, "p": {k: round(v, 4) for (k, _), v in zip(classes, p)}})
        if len(rows) % 50 == 0:
            print(f"{len(rows)} windows, t={end:.0f}s, {time.perf_counter() - t0:.0f}s elapsed")

    settings = {k: v for k, v in vars(a).items() if k not in ("video", "out")}
    settings["video"] = "/".join(a.video.parts[-2:])  # e.g. game1/1_720p.mkv; no local paths in logs
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"args": settings, "classes": [k for k, _ in classes], "rows": rows}, indent=0), encoding="utf-8")
    print(f"done: {len(rows)} windows in {time.perf_counter() - t0:.0f}s -> {a.out}")


if __name__ == "__main__":
    main()
