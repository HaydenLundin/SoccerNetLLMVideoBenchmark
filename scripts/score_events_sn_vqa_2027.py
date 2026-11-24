"""
scripts/score_events_sn_vqa_2027.py
Scores detect_events_sn_vqa_2027.py outputs against the test split's own event
moments (research note section 11). Diagnostic only: test labels measure the
detector, never feed it.

Reads <dir>/<match_key>_h<half>_<source>.json for every half in the questions file
and prints:
1. Ranking quality per half and class: AUC, and the worst-case rank of the window
   containing each event (strict matching).
2. Shots as counts: peak picking on p(shot) with the cut-off chosen on the tuning
   half only (best F1), scored on the other halves against a random baseline with
   the same number of detections. A detection may start up to --lead s after the shot.
   Two cut-offs: an absolute p(shot), and a per-half one on each window's causal
   percentile within its half (windows so far only, after a --warmup s start).
   Count errors are compared with a rate baseline: the tuning half's shots per
   second times the half's length.
3. Corners and offsides at p >= 0.5, consecutive windows merged, strict matching.

Usage:
    python scripts/score_events_sn_vqa_2027.py <test_GT.json> logs/sn_vqa_2027/events --source v2
"""

import sys
import json
import random
import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from processing.sn_vqa_2027_events import causal_percentile_rows, truth_events, match, merge_above, peaks, auc, worst_case_ranks, f1

CUTS = [0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.85, 0.9]
PERCENTILES = [0.8, 0.85, 0.9, 0.93, 0.95, 0.96, 0.97, 0.98, 0.99, 0.995]


def load_halves(gt, folder: Path, source: str):
    """{tag: (rows, classes, events)} for every half with an output file; events cover the scored span."""
    halves = sorted({(s["metadata"]["match_key"], s["metadata"]["half"]) for s in gt["data"]})
    out = {}
    for match_key, half in halves:
        path = folder / f"{match_key}_h{half}_{source}.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        rows = data["rows"]
        events = [(t, k) for t, k in truth_events(gt, match_key, half) if t <= rows[-1]["end"]]
        out[f"{match_key}_h{half}"] = (rows, data["classes"], events)
    return out


def shot_counts(rows, events, cut, sep, lead, rng=None, draws=300, warmup=None):
    """With `warmup` set, `cut` applies to the causal percentile of p(shot) instead of p(shot)."""
    shots = [t for t, k in events if k == "shot"]
    if warmup is not None:
        rows = causal_percentile_rows(rows, "shot", warmup)
    det = peaks(rows, "shot", thr=cut, sep=sep)
    found, real = match(det, shots, lead)
    rand_found = rand_real = 0.0
    if rng is not None and det:
        for _ in range(draws):
            order = list(rows)
            rng.shuffle(order)
            f, r = match(peaks(order, "shot", n=len(det), sep=sep), shots, lead)
            rand_found += f / draws
            rand_real += r / draws
    return found, len(shots), real, len(det), rand_found, rand_real


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gt", type=Path, help="test_GT.json")
    ap.add_argument("dir", type=Path, help="Folder of detector outputs")
    ap.add_argument("--source", default="v2", help="Output suffix, e.g. v2 or shot_bin_896")
    ap.add_argument("--tune", default="game1_h1", help="Half used to choose the shot cut-off")
    ap.add_argument("--sep", type=float, default=30.0, help="Minimum gap between shot detections (s)")
    ap.add_argument("--lead", type=float, default=15.0, help="A shot detection may start this long after the shot (s)")
    ap.add_argument("--warmup", type=float, default=120.0,
                    help="Per-half cut-off: no detections in a half's first seconds, while the percentile is noise")
    args = ap.parse_args()
    gt = json.loads(args.gt.read_text(encoding="utf-8"))
    halves = load_halves(gt, args.dir, args.source)
    if args.tune not in halves:
        raise SystemExit(f"No output for the tuning half {args.tune} in {args.dir}")

    print(f"== Ranking quality ({args.source}): AUC | worst-case rank of each event's window")
    for tag, (rows, classes, events) in halves.items():
        for kind in classes:
            times = [t for t, k in events if k == kind]
            if times:
                print(f"  {tag} {kind:12s} AUC {auc(rows, kind, times):.2f} | {len(times)} events in {len(rows)} windows: "
                      f"{worst_case_ranks(rows, kind, times)}")

    # Rate baseline for counts: the tuning half's shots per second times each half's scored span.
    span = lambda rows: rows[-1]["end"] - rows[0]["start"]
    rows, _, events = halves[args.tune]
    tune_rate = sum(k == "shot" for _, k in events) / span(rows)
    for cuts, warmup, label in ((CUTS, None, "p(shot)"),
                                (PERCENTILES, args.warmup, f"causal percentile of p(shot) in its half (warm-up {args.warmup:.0f} s)")):
        rows, _, events = halves[args.tune]
        tuned = {c: f1(*shot_counts(rows, events, c, args.sep, args.lead, warmup=warmup)[:4]) for c in cuts}
        cut = max(cuts, key=lambda c: (tuned[c], c))
        print(f"\n== Shots as counts: {label} >= {cut} (best F1 on {args.tune}), separation {args.sep:.0f} s, lead {args.lead:.0f} s")
        rng = random.Random(0)
        held = [0.0] * 6
        abs_err = rate_err = 0.0
        for tag, (rows, _, events) in halves.items():
            found, n, real, m, rf, rr = shot_counts(rows, events, cut, args.sep, args.lead, rng, warmup=warmup)
            rate_count = tune_rate * span(rows)
            print(f"  {tag}: found {found}/{n}, real {real}/{m}, count error {m - n:+d} (rate {rate_count - n:+.1f})"
                  f"   random: found {rf:.1f}, real {rr:.1f}")
            if tag != args.tune:
                held = [a + b for a, b in zip(held, (found, n, real, m, rf, rr))]
                abs_err += abs(m - n)
                rate_err += abs(rate_count - n)
        print(f"  held-out: found {held[0]:.0f}/{held[1]:.0f}, real {held[2]:.0f}/{held[3]:.0f}, F1 {f1(*held[:4]):.2f}"
              f"   random F1 {f1(held[4], held[1], held[5], held[3]):.2f}"
              f"   |count error| {abs_err:.0f} (rate {rate_err:.1f})")

    for kind in ("corner", "offside"):
        total = [0, 0, 0, 0]
        for rows, classes, events in halves.values():
            if kind not in classes:
                continue
            times = [t for t, k in events if k == kind]
            det = merge_above(rows, kind, 0.5)
            found, real = match(det, times)
            total = [a + b for a, b in zip(total, (found, len(times), real, len(det)))]
        if total[1]:
            print(f"{kind} (p >= 0.5, strict): found {total[0]}/{total[1]}, real {total[2]}/{total[3]}")


if __name__ == "__main__":
    main()
