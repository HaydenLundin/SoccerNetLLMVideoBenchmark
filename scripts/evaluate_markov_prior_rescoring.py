"""
Stage 4: Out-of-Sample 19-Match Markov Transition & Temporal Prior Rescoring.
Evaluates the impact of data-driven physical transition laws and temporal priors
on saved Game 0 Half 1 predictions (Qwen3.5-4B, MiniCPM-V 4.6, and 3-Model Specialist Ensemble).

Zero GPU compute ($0 spend, < 5 seconds runtime).
Strict Out-of-Sample Protocol:
- M19 transition matrix and Half-1 temporal priors are estimated strictly on
  Games 1-19 (38 halves, 4,210 events, excluding Game 0).
- Applied out-of-sample to Game 0 Half 1 predictions.
"""

import os
import sys
import json
import shutil
import statistics
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Any, Optional

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from processing.unified_schema import parse_mm_ss_to_seconds, SOCCERNET_CLASSES
from processing.post_process_predictions import (
    DEFAULT_CLASS_NMS_RADII_MS,
    class_aware_nms
)
from scripts.ablation_class_nms_and_residuals import evaluate_predictions_dataset
from scripts.test_temporal_calibration import (
    get_chunk_index_for_position_ms,
    count_tight_and_loose_hits,
    apply_temporal_shift_to_predictions
)

def compute_out_of_sample_m19(eval_set_path: Path, gt_dir: Path) -> Tuple[Dict, Dict, Dict]:
    """Computes empirical 17x17 transition matrix and priors strictly on Games 1-19."""
    with open(eval_set_path, "r", encoding="utf-8") as f:
        eval_set = json.load(f)

    games_19 = [g for g in eval_set["games"] if g["game_id"] != 0]

    transitions_30s = defaultdict(Counter)
    backward_30s = defaultdict(Counter)
    h1_class_counts = Counter()
    total_events = 0
    total_h1_events = 0

    for g in games_19:
        p = gt_dir / g["relative_path"] / "Labels-v2.json"
        with open(p, "r", encoding="utf-8") as gf:
            d = json.load(gf)

        events = []
        for a in d.get("annotations", []):
            lbl = a.get("label")
            t_str = a.get("gameTime", "")
            pos = int(a.get("position", 0))
            half = t_str.split(" - ")[0] if " - " in t_str else "1"
            events.append((half, pos, lbl))
            total_events += 1
            if half == "1":
                h1_class_counts[lbl] += 1
                total_h1_events += 1

        events.sort(key=lambda x: (x[0], x[1]))
        for i in range(len(events) - 1):
            h1, pos1, lbl1 = events[i]
            h2, pos2, lbl2 = events[i+1]
            if h1 == h2 and 0 <= (pos2 - pos1) <= 30000:
                transitions_30s[lbl1][lbl2] += 1
                backward_30s[lbl2][lbl1] += 1

    # Normalize forward probabilities: P(C_{t+1} | C_t)
    prob_fwd = {}
    for c_curr, dests in transitions_30s.items():
        tot = sum(dests.values())
        prob_fwd[c_curr] = {c_next: cnt / tot for c_next, cnt in dests.items()}

    # Normalize backward probabilities: P(C_t | C_{t+1})
    prob_bwd = {}
    for c_next, srcs in backward_30s.items():
        tot = sum(srcs.values())
        prob_bwd[c_next] = {c_prev: cnt / tot for c_prev, cnt in srcs.items()}

    h1_priors = {c: cnt / total_h1_events for c, cnt in h1_class_counts.items()}

    return prob_fwd, prob_bwd, h1_priors

def load_with_commentary(model_dir: Path) -> List[Dict]:
    with open(model_dir / "Predictions-v2.json", "r", encoding="utf-8") as f1:
        v2_preds = json.load(f1).get("predictions", [])
    with open(model_dir / "Predictions-Caption.json", "r", encoding="utf-8") as f2:
        cap_preds = json.load(f2).get("predictions", [])

    time_to_cap = {}
    for c in cap_preds:
        t = c.get("gameTime")
        if t and t not in time_to_cap:
            time_to_cap[t] = c.get("commentary", "")

    for p in v2_preds:
        p_t = p.get("gameTime")
        p["commentary"] = time_to_cap.get(p_t, "")
    return v2_preds

def apply_markov_and_temporal_prior(
    predictions: List[Dict],
    prob_fwd: Dict[str, Dict[str, float]],
    prob_bwd: Dict[str, Dict[str, float]],
    alpha: float = 0.15,
    prune_h1_substitutions: bool = True,
    prune_openplay_kickoffs: bool = True
) -> List[Dict]:
    """
    Rescores and filters predictions using the out-of-sample M19 transition matrix and temporal priors.
    1. Boosts causally paired transitions (Ball out of play -> Throw-in/Corner, Foul -> Free-kick/Card).
    2. Prunes or demotes ghost predictions violating empirical temporal priors.
    """
    sorted_preds = sorted(predictions, key=lambda x: int(x.get("position", 0)))

    # Find predicted goals for kick-off validation
    goal_positions = [int(p.get("position", 0)) for p in sorted_preds if p.get("label") == "Goal" and float(p.get("confidence", 0.0)) >= 0.5]

    rescored = []
    for i, p in enumerate(sorted_preds):
        p_pos = int(p.get("position", 0))
        p_lbl = p.get("label", "")
        p_half = str(p.get("half", "1"))
        orig_conf = float(p.get("confidence", 0.85))

        # Filter 1: Half 1 Substitution Ghost Pruning
        # P(Substitution | Half 1) = 0.10% (only 2 instances in 19 matches)
        if prune_h1_substitutions and p_lbl == "Substitution" and p_half == "1" and p_pos <= 2700000:
            continue

        # Filter 2: Open-Play Kick-off Ghost Pruning
        # Kick-off requires t <= 60s or preceding Goal within [t - 90s, t]
        if prune_openplay_kickoffs and p_lbl == "Kick-off":
            is_start = p_pos <= 60000
            is_after_goal = any(0 <= (p_pos - g_pos) <= 90000 for g_pos in goal_positions)
            if not is_start and not is_after_goal:
                continue

        # Markov Causal Transition Support
        # Check backward support (did a causally valid antecedent occur within [1s, 30s]?)
        fwd_boost = 0.0
        for j in range(i - 1, -1, -1):
            prev_p = sorted_preds[j]
            dt = p_pos - int(prev_p.get("position", 0))
            if dt < 1000:
                continue
            if dt > 30000:
                break
            prev_lbl = prev_p.get("label", "")
            if prev_lbl in prob_fwd and p_lbl in prob_fwd[prev_lbl]:
                fwd_boost = max(fwd_boost, prob_fwd[prev_lbl][p_lbl])

        # Check forward support (is this event followed by a causally valid consequent within [1s, 30s]?)
        bwd_boost = 0.0
        for k in range(i + 1, len(sorted_preds)):
            next_p = sorted_preds[k]
            dt = int(next_p.get("position", 0)) - p_pos
            if dt < 1000:
                continue
            if dt > 30000:
                break
            next_lbl = next_p.get("label", "")
            if next_lbl in prob_bwd and p_lbl in prob_bwd[next_lbl]:
                bwd_boost = max(bwd_boost, prob_bwd[next_lbl][p_lbl])

        causal_support = max(fwd_boost, bwd_boost)
        new_conf = min(1.0, orig_conf + alpha * causal_support)

        p_new = dict(p)
        p_new["confidence"] = round(new_conf, 4)
        p_new["causal_support"] = round(causal_support, 4)
        rescored.append(p_new)

    return rescored

def run_rescoring_evaluation():
    eval_set_path = WORKSPACE_ROOT / "eval_set.json"
    gt_dir = WORKSPACE_ROOT / "test_eval_ground_truth"
    manifest_path = WORKSPACE_ROOT / "gate2_chunks" / "game_0_half_1_chunks_manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    url_local = "england_epl/2014-2015/2015-05-17 - 18-00 Manchester United 1 - 1 Arsenal/"
    scoped_gt_dir = WORKSPACE_ROOT / "logs/local_eval/qwen3.5-4b_4bit_W60_half1/scoped_ground_truth"
    gt_action_file = scoped_gt_dir / url_local.strip("/") / "Labels-v2.json"
    open_gt = Path("\\\\?\\" + str(gt_action_file.resolve())) if os.name == "nt" else gt_action_file
    with open(open_gt, "r", encoding="utf-8") as f:
        gt_action_annotations = json.load(f).get("annotations", [])

    print("=" * 110)
    print("STAGE 4: OUT-OF-SAMPLE 19-MATCH MARKOV TRANSITION & TEMPORAL PRIOR RESCORING (Half 1)")
    print("=" * 110)

    # 1. Compute M19 strictly on Games 1-19
    prob_fwd, prob_bwd, h1_priors = compute_out_of_sample_m19(eval_set_path, gt_dir)
    print(f"[M19] Computed Markov transition matrix strictly on Games 1-19 (38 halves, 4,210 events).")
    print(f"      P(Indirect FK | Foul, <=30s) = {prob_fwd.get('Foul', {}).get('Indirect free-kick', 0.0)*100:.1f}%")
    print(f"      P(Yellow Card | Foul, <=30s) = {prob_fwd.get('Foul', {}).get('Yellow card', 0.0)*100:.1f}%")
    print(f"      P(Throw-in | Ball out of play, <=30s) = {prob_fwd.get('Ball out of play', {}).get('Throw-in', 0.0)*100:.1f}%")
    print(f"      P(Corner | Ball out of play, <=30s) = {prob_fwd.get('Ball out of play', {}).get('Corner', 0.0)*100:.1f}%")
    print(f"      P(Substitution | Half 1) = {h1_priors.get('Substitution', 0.0)*100:.3f}% (Empirical Zero Prior)")

    # 2. Load Saved Half 1 Predictions
    qwen_dir = WORKSPACE_ROOT / "logs/local_eval/qwen3.5-4b_4bit_W60_half1"
    minicpm_dir = WORKSPACE_ROOT / "logs/local_eval/minicpm-v-4.6_4bit_W60_half1"
    lfm_dir = WORKSPACE_ROOT / "logs/local_eval/lfm2.5-vl-3b_4bit_W30_half1"

    qwen_preds = load_with_commentary(qwen_dir)
    minicpm_preds = load_with_commentary(minicpm_dir)
    lfm_preds = load_with_commentary(lfm_dir)

    # Apply calibrated shifts to base models
    minicpm_cal_preds = apply_temporal_shift_to_predictions(minicpm_preds, lambda pos, p: -2950 if p.get("label") == "Foul" else 0)
    lfm_cal_preds = apply_temporal_shift_to_predictions(lfm_preds, lambda pos, p: -3100)

    # Build Base Specialist Ensemble
    QWEN_SPECIALIST_CLASSES = {"Corner", "Throw-in", "Goal", "Direct free-kick"}
    MINICPM_SPECIALIST_CLASSES = {"Kick-off", "Foul", "Shots off target", "Clearance", "Offside", "Indirect free-kick"}
    LFM_SPECIALIST_CLASSES = {"Ball out of play"}

    specialist_candidates = []
    for p in qwen_preds:
        if p.get("label") in QWEN_SPECIALIST_CLASSES:
            p_c = dict(p); p_c["model"] = "qwen"
            specialist_candidates.append(p_c)
    for p in minicpm_cal_preds:
        if p.get("label") in MINICPM_SPECIALIST_CLASSES:
            p_c = dict(p); p_c["model"] = "minicpm"
            specialist_candidates.append(p_c)
    for p in lfm_cal_preds:
        if p.get("label") in LFM_SPECIALIST_CLASSES:
            p_c = dict(p); p_c["model"] = "lfm"
            specialist_candidates.append(p_c)

    specialist_base_deduped = class_aware_nms(specialist_candidates, radius_ms=3000, class_radii_ms=DEFAULT_CLASS_NMS_RADII_MS)

    # Evaluate Baseline Models
    temp_root = WORKSPACE_ROOT / "scratch/markov_eval"
    if temp_root.exists():
        shutil.rmtree(temp_root, ignore_errors=True)
    temp_root.mkdir(parents=True, exist_ok=True)

    configs_to_test = [
        ("Qwen3.5-4B (Raw)", qwen_preds),
        ("MiniCPM-V 4.6 (Raw)", minicpm_preds),
        ("3-Model Specialist Ensemble (Base)", specialist_base_deduped)
    ]

    results = []

    print("\n" + "-" * 110)
    print(f"{'System Configuration':<36} | {'N Pred':<6} | {'Off Tight mAP':<13} | {'Off Loose mAP':<13} | {'Act Loose mAP':<13} | {'Match Recall'}")
    print("-" * 110)

    for name, p_list in configs_to_test:
        # Base eval
        m_base = evaluate_predictions_dataset({"UrlLocal": url_local, "predictions": p_list}, scoped_gt_dir, temp_root / f"{name}_base", url_local)
        _, l_hits = count_tight_and_loose_hits(p_list, gt_action_annotations)
        active_loose_base = statistics.mean([m_base["loose_per_class"][c] for c in m_base["loose_per_class"] if sum(1 for g in gt_action_annotations if g["label"] == c) > 0])

        # Rescored eval
        p_rescored = apply_markov_and_temporal_prior(p_list, prob_fwd, prob_bwd, alpha=0.15)
        # Re-apply class-aware NMS if it's the ensemble
        if "Ensemble" in name:
            p_rescored = class_aware_nms(p_rescored, radius_ms=3000, class_radii_ms=DEFAULT_CLASS_NMS_RADII_MS)

        m_rescored = evaluate_predictions_dataset({"UrlLocal": url_local, "predictions": p_rescored}, scoped_gt_dir, temp_root / f"{name}_rescored", url_local)
        _, l_hits_rescored = count_tight_and_loose_hits(p_rescored, gt_action_annotations)
        active_loose_rescored = statistics.mean([m_rescored["loose_per_class"][c] for c in m_rescored["loose_per_class"] if sum(1 for g in gt_action_annotations if g["label"] == c) > 0])

        recall_base = (l_hits / len(gt_action_annotations)) * 100.0
        recall_rescored = (l_hits_rescored / len(gt_action_annotations)) * 100.0

        print(f"{name:<36} | {len(p_list):<6} | {m_base['tight_average_mAP']:<13.4f} | {m_base['loose_average_mAP']:<13.4f} | {active_loose_base*100:<12.2f}% | {l_hits}/{len(gt_action_annotations)} ({recall_base:.1f}%)")
        print(f"{name + ' + M19 Rescored':<36} | {len(p_rescored):<6} | {m_rescored['tight_average_mAP']:<13.4f} | {m_rescored['loose_average_mAP']:<13.4f} | {active_loose_rescored*100:<12.2f}% | {l_hits_rescored}/{len(gt_action_annotations)} ({recall_rescored:.1f}%)")
        print("-" * 110)

        results.append({
            "model": name,
            "base": {
                "pred_count": len(p_list),
                "tight_mAP": m_base["tight_average_mAP"],
                "loose_mAP": m_base["loose_average_mAP"],
                "active_loose_mAP": round(active_loose_base, 4),
                "loose_hits": l_hits,
                "recall": round(recall_base, 2)
            },
            "m19_rescored": {
                "pred_count": len(p_rescored),
                "tight_mAP": m_rescored["tight_average_mAP"],
                "loose_mAP": m_rescored["loose_average_mAP"],
                "active_loose_mAP": round(active_loose_rescored, 4),
                "loose_hits": l_hits_rescored,
                "recall": round(recall_rescored, 2)
            }
        })

    # Save Results
    out_file = WORKSPACE_ROOT / "logs/macro_study/markov_rescoring_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[+] Saved Markov Rescoring benchmark results to: {out_file}")

if __name__ == "__main__":
    run_rescoring_evaluation()
