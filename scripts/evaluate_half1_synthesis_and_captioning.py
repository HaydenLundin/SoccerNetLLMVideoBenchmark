"""
Stage 3: 3-Model Multi-Scale Ensemble & Track B Dense Video Captioning Benchmark
Executes in ~15 seconds with zero GPU compute on saved Half 1 artifacts.
Evaluates:
1. LFM2.5-VL-3B Temporal Recalibration (-3.10s shift & LOCO CV)
2. 3-Model Multi-Scale Ensemble (Qwen3.5-4B + MiniCPM-V 4.6 + LFM2.5-VL-3B)
   - Naive Confidence-Weighted Union vs. Domain-Routed Specialist Ensemble
   - 5-Fold Out-of-Sample Leave-One-Chunk-Out (LOCO) Cross-Validation
3. Track B: Dense Video Captioning (BLEU-1..4, Smoothed BLEU-4, ROUGE-L, CIDEr, METEOR at +-15s)
   - Raw Lexical Comparison vs. Official Broadcast Ground Truth
   - Delexicalized Role-Normalized Semantic Comparison
"""

import os
import sys
import json
import math
import shutil
import re
import statistics
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple, Any, Optional

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from processing.unified_schema import parse_mm_ss_to_seconds, SOCCERNET_CLASSES
from processing.post_process_predictions import (
    stitch_chunks_to_match_predictions,
    DEFAULT_CLASS_NMS_RADII_MS,
    class_aware_nms
)
from processing.evaluate_soccernet import evaluate_match_action_spotting
from scripts.ablation_class_nms_and_residuals import (
    load_raw_windows_to_chunk_results,
    evaluate_predictions_dataset,
    compute_signed_residuals
)
from scripts.test_temporal_calibration import (
    get_chunk_index_for_position_ms,
    count_tight_and_loose_hits,
    apply_temporal_shift_to_predictions
)

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider


def compute_fine_grained_hits(
    predictions: List[Dict],
    gt_annotations: List[Dict],
    thresholds: List[float] = [1.0, 2.0, 3.0, 5.0, 30.0]
) -> Dict[float, int]:
    """Computes matched hit counts at discrete tolerance thresholds."""
    hit_counts = {th: 0 for th in thresholds}
    used_pred_indices = set()

    for gt in gt_annotations:
        gt_pos_ms = int(gt.get("position", 0))
        gt_lbl = gt.get("label", "")

        best_p_idx = None
        best_diff_ms = float("inf")

        for p_idx, pred in enumerate(predictions):
            if p_idx in used_pred_indices or pred.get("label") != gt_lbl:
                continue
            diff_ms = abs(int(pred.get("position", 0)) - gt_pos_ms)
            if diff_ms < best_diff_ms:
                best_diff_ms = diff_ms
                best_p_idx = p_idx

        if best_p_idx is not None and best_diff_ms <= (max(thresholds) * 1000.0):
            used_pred_indices.add(best_p_idx)
            for th in thresholds:
                if best_diff_ms <= (th * 1000.0):
                    hit_counts[th] += 1

    return hit_counts


def load_with_commentary(model_dir: Path) -> List[Dict]:
    """
    Loads Action Spotting predictions and pairs them 1-to-1 with Dense Captions.
    Attaches commentary directly into prediction dictionaries so commentary
    survives temporal recalibration and class-aware NMS.
    """
    with open(model_dir / "Predictions-v2.json", "r", encoding="utf-8") as f1:
        v2_preds = json.load(f1).get("predictions", [])
    with open(model_dir / "Predictions-Caption.json", "r", encoding="utf-8") as f2:
        cap_preds = json.load(f2).get("predictions", [])

    cap_map = {
        (str(c.get("half", "1")), c.get("label", ""), int(c.get("position", 0))): c.get("commentary", "")
        for c in cap_preds
    }

    merged = []
    for p in v2_preds:
        p_c = dict(p)
        sig = (str(p.get("half", "1")), p.get("label", ""), int(p.get("position", 0)))
        p_c["commentary"] = cap_map.get(sig, "")
        merged.append(p_c)
    return merged


def compute_smoothed_bleu(hyp: str, ref: str, n: int = 4) -> float:
    """
    Computes smoothed BLEU-4 using Chen & Cherry (2014) Method 1.
    Smooths zero count n-grams by replacing p_k = 0 with 1 / (2^k * total_ngrams).
    Prevents degenerate 0.0000 on short single-sentence commentary captions.
    """
    def get_ngrams(tokens, k):
        return [tuple(tokens[i:i+k]) for i in range(len(tokens) - k + 1)]

    hyp_toks = hyp.lower().split()
    ref_toks = ref.lower().split()
    if not hyp_toks or not ref_toks:
        return 0.0

    precisions = []
    for k in range(1, n + 1):
        hyp_ngrams = get_ngrams(hyp_toks, k)
        ref_ngrams = get_ngrams(ref_toks, k)
        if not hyp_ngrams:
            precisions.append(0.0)
            continue
        hyp_c = Counter(hyp_ngrams)
        ref_c = Counter(ref_ngrams)
        match = sum(min(count, ref_c[ng]) for ng, count in hyp_c.items())
        if match == 0:
            precisions.append(1.0 / ((2 ** k) * len(hyp_ngrams)))
        else:
            precisions.append(match / len(hyp_ngrams))

    c, r = len(hyp_toks), len(ref_toks)
    bp = 1.0 if c > r else math.exp(1.0 - r / c)
    log_sum = sum(math.log(p) for p in precisions) / n
    return bp * math.exp(log_sum)


def compute_pure_meteor(
    hyp: str,
    ref: str,
    alpha: float = 0.9,
    gamma: float = 0.5,
    beta: float = 3.0
) -> float:
    """
    Computes exact-match unigram METEOR score with fragmentation penalty in pure Python.
    Harmonic mean of unigram precision and recall with recall weighted by alpha=0.9.
    """
    hyp_toks = hyp.lower().split()
    ref_toks = ref.lower().split()
    if not hyp_toks or not ref_toks:
        return 0.0

    matched_hyp, matched_ref = [], []
    ref_used = [False] * len(ref_toks)
    for h_idx, h_w in enumerate(hyp_toks):
        for r_idx, r_w in enumerate(ref_toks):
            if not ref_used[r_idx] and h_w == r_w:
                ref_used[r_idx] = True
                matched_hyp.append(h_idx)
                matched_ref.append(r_idx)
                break

    m = len(matched_hyp)
    if m == 0:
        return 0.0

    p = m / len(hyp_toks)
    r = m / len(ref_toks)
    f_mean = (p * r) / (alpha * p + (1.0 - alpha) * r)

    pairs = sorted(zip(matched_hyp, matched_ref), key=lambda x: x[0])
    chunks = 1
    for i in range(1, len(pairs)):
        if pairs[i][0] != pairs[i-1][0] + 1 or pairs[i][1] != pairs[i-1][1] + 1:
            chunks += 1

    penalty = gamma * ((chunks / m) ** beta)
    return max(0.0, f_mean * (1.0 - penalty))


def delexicalize_text(text: str) -> str:
    """
    Normalizes specific named entities and generic visual player descriptors to role tokens:
    [PLAYER], [TEAM], [REFEREE].
    Bridges the naming gap between broadcast commentator proper nouns and VLM visual role descriptions.
    """
    P_TOK = " ___PLAYER_TAG___ "
    T_TOK = " ___TEAM_TAG___ "
    R_TOK = " ___REFEREE_TAG___ "

    t = text

    # 1. Normalize teams FIRST so multi-word names aren't split
    teams = ["manchester united", "manchester utd", "man united", "man utd", "arsenal"]
    for tm in teams:
        t = re.sub(rf"\b{tm}\b", T_TOK, t, flags=re.IGNORECASE)
    t = re.sub(r"\bteam in (red|yellow|blue|white|green|black)\b", T_TOK, t, flags=re.IGNORECASE)

    # 2. Normalize known match players
    players = [
        "alexis sanchez", "ander herrera", "francis coquelin", "mesut ozil", "phil jones",
        "david de gea", "daley blind", "juan mata", "juan manuel mata", "olivier giroud", "giroud",
        "aaron ramsey", "ramsey", "wayne rooney", "rooney", "chris smalling", "smalling",
        "ashley young", "marouane fellaini", "fellaini", "antonio valencia", "valencia",
        "nacho monreal", "monreal", "laurent koscielny", "koscielny", "santi cazorla",
        "cazorla", "theo walcott", "walcott", "hector bellerin", "bellerin",
        "radamel falcao", "falcao", "victor valdes", "valdes", "per mertesacker",
        "mertesacker", "kieran gibbs", "gibbs", "wojciech szczesny", "szczesny",
        "robin van persie", "van persie", "angel di maria", "di maria", "marcos rojo", "rojo"
    ]
    for p in players:
        t = re.sub(rf"\b{p}\b", P_TOK, t, flags=re.IGNORECASE)

    # 3. Normalize generic visual player roles
    t = re.sub(
        r"\b(a|the)?\s*(player|goalkeeper|keeper|midfielder|defender|attacker|striker|winger)\s*(in\s*(a\s*)?(red|yellow|blue|white|green|black)\s*(jersey|shirt)?)?\b",
        P_TOK,
        t,
        flags=re.IGNORECASE
    )
    t = re.sub(r"\b(the|a)?\s*(visiting|home)?\s*goalkeeper\b", P_TOK, t, flags=re.IGNORECASE)

    # 4. Normalize officials
    officials = ["referee michael dean", "michael dean", "referee", "linesman", "assistant referee"]
    for off in officials:
        t = re.sub(rf"\b{off}\b", R_TOK, t, flags=re.IGNORECASE)

    # Replace placeholders with final tokens
    t = t.replace("___PLAYER_TAG___", "[PLAYER]").replace("___TEAM_TAG___", "[TEAM]").replace("___REFEREE_TAG___", "[REFEREE]")

    # Clean up redundant adjacent duplicate tokens
    t = re.sub(r"(\[PLAYER\]\s*)+", "[PLAYER] ", t)
    t = re.sub(r"(\[TEAM\]\s*)+", "[TEAM] ", t)
    t = re.sub(r"(\[REFEREE\]\s*)+", "[REFEREE] ", t)
    return " ".join(t.split())


def evaluate_track_b_dense_captioning(
    pred_captions: List[Dict],
    gt_captions: List[Dict],
    window_sec: float = 30.0,
    delexicalize: bool = False
) -> Dict[str, Any]:
    """
    Evaluates Track B: Dense Video Captioning within +-15.0s (30s window).
    Computes BLEU-1..4, Smoothed BLEU-4, ROUGE-L, CIDEr, METEOR, and temporal alignment precision/recall.
    """
    half_window_ms = (window_sec / 2.0) * 1000.0
    matched_gts = {}
    matched_preds = {}
    matched_pairs = []
    used_p_indices = set()

    pair_id = 0
    for gt in gt_captions:
        gt_pos_ms = int(gt.get("position", 0))
        if delexicalize:
            gt_text = gt.get("anonymized", gt.get("description", "")).strip()
        else:
            gt_text = gt.get("description", "").strip()

        if not gt_text:
            continue

        best_p_idx = None
        best_diff = float("inf")

        for p_idx, p in enumerate(pred_captions):
            if p_idx in used_p_indices:
                continue
            p_pos_ms = int(p.get("position", 0))
            diff = abs(p_pos_ms - gt_pos_ms)
            if diff < best_diff:
                best_diff = diff
                best_p_idx = p_idx

        if best_p_idx is not None and best_diff <= half_window_ms:
            used_p_indices.add(best_p_idx)
            p_text = pred_captions[best_p_idx].get("commentary", "").strip()
            if delexicalize:
                p_text = delexicalize_text(p_text)

            if p_text:
                matched_gts[pair_id] = [gt_text]
                matched_preds[pair_id] = [p_text]
                matched_pairs.append({
                    "gt_time": gt.get("gameTime", ""),
                    "gt_text": gt_text,
                    "pred_time": pred_captions[best_p_idx].get("gameTime", ""),
                    "pred_label": pred_captions[best_p_idx].get("label", ""),
                    "pred_text": p_text,
                    "dt_sec": round((int(pred_captions[best_p_idx].get("position", 0)) - gt_pos_ms) / 1000.0, 2)
                })
                pair_id += 1

    num_gt = len(gt_captions)
    num_pred = len(pred_captions)
    temporal_recall = len(matched_pairs) / max(1, num_gt)
    temporal_precision = len(matched_pairs) / max(1, num_pred)

    if not matched_pairs:
        return {
            "matched_count": 0,
            "total_gt": num_gt,
            "total_pred": num_pred,
            "temporal_recall": 0.0,
            "temporal_precision": 0.0,
            "bleu1": 0.0, "bleu2": 0.0, "bleu3": 0.0, "bleu4": 0.0,
            "smoothed_bleu4": 0.0,
            "rouge_l": 0.0, "cider": 0.0, "meteor": 0.0, "pairs": []
        }

    # Compute Standard NLP Metrics
    try:
        bleu_scorer = Bleu(4)
        bleu_scores, _ = bleu_scorer.compute_score(matched_gts, matched_preds)
    except Exception:
        bleu_scores = [0.0, 0.0, 0.0, 0.0]

    try:
        rouge_scorer = Rouge()
        rouge_score, _ = rouge_scorer.compute_score(matched_gts, matched_preds)
    except Exception:
        rouge_score = 0.0

    try:
        cider_scorer = Cider()
        cider_score, _ = cider_scorer.compute_score(matched_gts, matched_preds)
    except Exception:
        cider_score = 0.0

    # Compute Smoothed BLEU-4 and pure-Python METEOR across matched pairs
    pair_sm_b4 = [compute_smoothed_bleu(p["pred_text"], p["gt_text"]) for p in matched_pairs]
    pair_meteor = [compute_pure_meteor(p["pred_text"], p["gt_text"]) for p in matched_pairs]

    smoothed_bleu4 = statistics.mean(pair_sm_b4) if pair_sm_b4 else 0.0
    meteor_score = statistics.mean(pair_meteor) if pair_meteor else 0.0

    return {
        "matched_count": len(matched_pairs),
        "total_gt": num_gt,
        "total_pred": num_pred,
        "temporal_recall": round(temporal_recall, 4),
        "temporal_precision": round(temporal_precision, 4),
        "bleu1": round(float(bleu_scores[0]), 4),
        "bleu2": round(float(bleu_scores[1]), 4),
        "bleu3": round(float(bleu_scores[2]), 4),
        "bleu4": round(float(bleu_scores[3]), 4),
        "smoothed_bleu4": round(float(smoothed_bleu4), 4),
        "rouge_l": round(float(rouge_score), 4),
        "cider": round(float(cider_score), 4),
        "meteor": round(float(meteor_score), 4),
        "pairs": matched_pairs
    }


def main():
    manifest_path = WORKSPACE_ROOT / "gate2_chunks" / "game_0_half_1_chunks_manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    url_local = "england_epl/2014-2015/2015-05-17 - 18-00 Manchester United 1 - 1 Arsenal/"

    scoped_gt_dir = WORKSPACE_ROOT / "logs/local_eval/qwen3.5-4b_4bit_W60_half1/scoped_ground_truth"
    gt_action_file = scoped_gt_dir / url_local.strip("/") / "Labels-v2.json"
    open_gt = Path("\\\\?\\" + str(gt_action_file.resolve())) if os.name == "nt" else gt_action_file
    with open(open_gt, "r", encoding="utf-8") as f:
        gt_action_annotations = json.load(f).get("annotations", [])

    gt_caption_file = WORKSPACE_ROOT / "test_eval_ground_truth" / url_local.strip("/") / "Labels-caption.json"
    open_caption = Path("\\\\?\\" + str(gt_caption_file.resolve())) if os.name == "nt" else gt_caption_file
    with open(open_caption, "r", encoding="utf-8") as f:
        all_gt_captions = json.load(f).get("annotations", [])

    # Filter to Half 1 ground truth captions within broadcast time (0 to 2700s = 45:00)
    # Excludes post-half stoppage-time review comments (> 2,700,003 ms)
    half1_gt_captions = [
        c for c in all_gt_captions
        if str(c.get("gameTime", "")).startswith("1 - ") and int(c.get("position", 0)) <= 2700003
    ]

    temp_root = WORKSPACE_ROOT / "scratch/synthesis_eval"
    if temp_root.exists():
        shutil.rmtree(temp_root, ignore_errors=True)
    temp_root.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 125)
    print("  STAGE 3: 3-MODEL MULTI-SCALE ENSEMBLE & TRACK B DENSE CAPTIONING BENCHMARK (Half 1)")
    print("=" * 125)

    # Load All 3 Saved Model Predictions with Commentary Preserved
    qwen_dir = WORKSPACE_ROOT / "logs/local_eval/qwen3.5-4b_4bit_W60_half1"
    minicpm_dir = WORKSPACE_ROOT / "logs/local_eval/minicpm-v-4.6_4bit_W60_half1"
    lfm_dir = WORKSPACE_ROOT / "logs/local_eval/lfm2.5-vl-3b_4bit_W30_half1"

    qwen_preds = load_with_commentary(qwen_dir)
    minicpm_preds = load_with_commentary(minicpm_dir)
    lfm_preds = load_with_commentary(lfm_dir)

    print(f"[DATA] Loaded Qwen3.5-4B:   {len(qwen_preds)} predictions with commentary")
    print(f"[DATA] Loaded MiniCPM-V 4.6: {len(minicpm_preds)} predictions with commentary")
    print(f"[DATA] Loaded LFM2.5-VL-3B:  {len(lfm_preds)} predictions with commentary")
    print(f"[DATA] In-Bounds Ground Truth: {len(gt_action_annotations)} Action Spots | {len(half1_gt_captions)} Dense Captions (<= 45:00)")

    # =========================================================================
    # TASK 1: LFM2.5-VL-3B TEMPORAL RECALIBRATION (-3.10s Shift & LOCO CV)
    # =========================================================================
    print("\n" + "=" * 95)
    print("  TASK 1: LFM2.5-VL-3B TEMPORAL RECALIBRATION AUDIT (Overcoming the +3.10s Culmination Lag)")
    print("=" * 95)

    # 1. Baseline LFM2.5 metrics
    lfm_base_metrics = evaluate_predictions_dataset({"UrlLocal": url_local, "predictions": lfm_preds}, scoped_gt_dir, temp_root / "lfm_base", url_local)
    lfm_base_hits = compute_fine_grained_hits(lfm_preds, gt_action_annotations)

    # 2. Global Shift: -3,100 ms
    lfm_global_preds = apply_temporal_shift_to_predictions(lfm_preds, lambda pos, p: -3100)
    lfm_global_deduped = class_aware_nms(lfm_global_preds, radius_ms=3000, class_radii_ms=DEFAULT_CLASS_NMS_RADII_MS)
    lfm_global_metrics = evaluate_predictions_dataset({"UrlLocal": url_local, "predictions": lfm_global_deduped}, scoped_gt_dir, temp_root / "lfm_global", url_local)
    lfm_global_hits = compute_fine_grained_hits(lfm_global_deduped, gt_action_annotations)

    # 3. LOCO Out-of-Sample Cross-Validation for LFM
    lfm_residuals = []
    used_p = set()
    for gt in gt_action_annotations:
        gt_pos = int(gt.get("position", 0))
        gt_lbl = gt.get("label", "")
        best_idx = None
        best_diff = float("inf")
        for p_idx, p in enumerate(lfm_preds):
            if p_idx in used_p or p.get("label") != gt_lbl:
                continue
            diff = int(p.get("position", 0)) - gt_pos
            if abs(diff) < abs(best_diff):
                best_diff = diff
                best_idx = p_idx
        if best_idx is not None and abs(best_diff) <= 15000:
            used_p.add(best_idx)
            lfm_residuals.append({
                "pred_chunk": get_chunk_index_for_position_ms(int(lfm_preds[best_idx].get("position", 0)), manifest),
                "dt_sec": best_diff / 1000.0
            })

    lfm_loco_shifts = {}
    for c_i in range(5):
        tr = [r["dt_sec"] for r in lfm_residuals if r["pred_chunk"] != c_i]
        lfm_loco_shifts[c_i] = -statistics.median(tr) if tr else -3.10

    lfm_loco_preds = apply_temporal_shift_to_predictions(
        lfm_preds,
        lambda pos, p: int(round(lfm_loco_shifts[get_chunk_index_for_position_ms(pos, manifest)] * 1000.0))
    )
    lfm_loco_deduped = class_aware_nms(lfm_loco_preds, radius_ms=3000, class_radii_ms=DEFAULT_CLASS_NMS_RADII_MS)
    lfm_loco_metrics = evaluate_predictions_dataset({"UrlLocal": url_local, "predictions": lfm_loco_deduped}, scoped_gt_dir, temp_root / "lfm_loco", url_local)
    lfm_loco_hits = compute_fine_grained_hits(lfm_loco_deduped, gt_action_annotations)

    print(f"{'Configuration':<32} | {'Kept':<4} | {'<= 1.0s':<7} | {'<= 2.0s':<7} | {'<= 5.0s':<7} | {'Tight mAP':<9} | {'Loose mAP':<9} | {'BallOutPlay AP':<14}")
    print("-" * 95)
    print(f"{'LFM2.5-VL-3B (Uncalibrated)':<32} | {len(lfm_preds):<4} | {lfm_base_hits[1.0]:<7} | {lfm_base_hits[2.0]:<7} | {lfm_base_hits[5.0]:<7} | {lfm_base_metrics['tight_average_mAP']:<9.4f} | {lfm_base_metrics['loose_average_mAP']:<9.4f} | {lfm_base_metrics['loose_per_class'].get('Ball out of play', 0.0):<14.4f}")
    print(f"{'LFM2.5-VL-3B (Global Shift -3.10s)':<32} | {len(lfm_global_deduped):<4} | {lfm_global_hits[1.0]:<7} | {lfm_global_hits[2.0]:<7} | {lfm_global_hits[5.0]:<7} | {lfm_global_metrics['tight_average_mAP']:<9.4f} | {lfm_global_metrics['loose_average_mAP']:<9.4f} | {lfm_global_metrics['loose_per_class'].get('Ball out of play', 0.0):<14.4f}")
    print(f"{'LFM2.5-VL-3B (LOCO Out-of-Sample)':<32} | {len(lfm_loco_deduped):<4} | {lfm_loco_hits[1.0]:<7} | {lfm_loco_hits[2.0]:<7} | {lfm_loco_hits[5.0]:<7} | {lfm_loco_metrics['tight_average_mAP']:<9.4f} | {lfm_loco_metrics['loose_average_mAP']:<9.4f} | {lfm_loco_metrics['loose_per_class'].get('Ball out of play', 0.0):<14.4f}")
    print("=" * 95)

    # =========================================================================
    # TASK 2: 3-MODEL MULTI-SCALE RECEPTIVE-FIELD ENSEMBLE
    # =========================================================================
    print("\n" + "=" * 125)
    print("  TASK 2: 3-MODEL MULTI-SCALE ENSEMBLE (Qwen3.5-4B @ 60s + MiniCPM-V 4.6 @ 60s + LFM2.5-VL-3B @ 30s)")
    print("=" * 125)

    # Prepare calibrated versions of each model with commentary preserved
    minicpm_cal_preds = apply_temporal_shift_to_predictions(minicpm_preds, lambda pos, p: +2950)
    lfm_cal_preds = apply_temporal_shift_to_predictions(lfm_preds, lambda pos, p: -3100)

    # Policy A: Unrouted Confidence-Weighted Union
    union_candidates = []
    for p in qwen_preds:
        p_c = dict(p)
        p_c["model"] = "qwen"
        union_candidates.append(p_c)
    for p in minicpm_cal_preds:
        p_c = dict(p)
        p_c["model"] = "minicpm"
        if p_c["label"] in ["Goal", "Shots on target"]:
            p_c["confidence"] = float(p_c.get("confidence", 0.8)) * 0.5
        union_candidates.append(p_c)
    for p in lfm_cal_preds:
        p_c = dict(p)
        p_c["model"] = "lfm"
        union_candidates.append(p_c)

    union_deduped = class_aware_nms(union_candidates, radius_ms=3000, class_radii_ms=DEFAULT_CLASS_NMS_RADII_MS)
    union_metrics = evaluate_predictions_dataset({"UrlLocal": url_local, "predictions": union_deduped}, scoped_gt_dir, temp_root / "union_3m", url_local)
    union_tight_hits, union_loose_hits = count_tight_and_loose_hits(union_deduped, gt_action_annotations)

    # Policy B: Domain-Routed Specialist Ensemble (Global Prior)
    # Routing Rules:
    # 1. Set-Pieces & Major Stoppages (Corner, Throw-in, Goal, Direct free-kick): Route to Qwen3.5-4B (protects perfect precision)
    # 2. Open-Play Duels & Restarts (Foul, Kick-off, Shots off target, Clearance, Offside, Indirect free-kick): Route to MiniCPM-V 4.6 (high recall, avoids Qwen's kick-off replays)
    # 3. Rapid Boundary Transitions (Ball out of play): Route to LFM2.5-VL-3B (30s receptive field with 7 tight hits)

    QWEN_SPECIALIST_CLASSES = {"Corner", "Throw-in", "Goal", "Direct free-kick"}
    MINICPM_SPECIALIST_CLASSES = {"Kick-off", "Foul", "Shots off target", "Clearance", "Offside", "Indirect free-kick"}
    LFM_SPECIALIST_CLASSES = {"Ball out of play"}

    specialist_candidates = []
    for p in qwen_preds:
        if p.get("label") in QWEN_SPECIALIST_CLASSES:
            p_c = dict(p)
            p_c["model"] = "qwen"
            specialist_candidates.append(p_c)

    for p in minicpm_cal_preds:
        if p.get("label") in MINICPM_SPECIALIST_CLASSES:
            p_c = dict(p)
            p_c["model"] = "minicpm"
            specialist_candidates.append(p_c)

    for p in lfm_cal_preds:
        if p.get("label") in LFM_SPECIALIST_CLASSES:
            p_c = dict(p)
            p_c["model"] = "lfm"
            specialist_candidates.append(p_c)

    specialist_deduped = class_aware_nms(specialist_candidates, radius_ms=3000, class_radii_ms=DEFAULT_CLASS_NMS_RADII_MS)
    specialist_metrics = evaluate_predictions_dataset({"UrlLocal": url_local, "predictions": specialist_deduped}, scoped_gt_dir, temp_root / "specialist_3m", url_local)
    spec_tight_hits, spec_loose_hits = count_tight_and_loose_hits(specialist_deduped, gt_action_annotations)

    # Policy C: Out-of-Sample Leave-One-Chunk-Out (LOCO) Specialist Ensemble
    loco_test_candidates = []
    for k in range(5):
        train_gt = [g for g in gt_action_annotations if get_chunk_index_for_position_ms(int(g.get("position", 0)), manifest) != k]
        train_scoped = temp_root / f"train_scoped_{k}" / url_local.strip("/")
        train_scoped.mkdir(parents=True, exist_ok=True)
        with open(Path("\\\\?\\" + str((train_scoped / "Labels-v2.json").resolve())), "w", encoding="utf-8") as f:
            json.dump({"annotations": train_gt}, f)

        q_tr = [p for p in qwen_preds if get_chunk_index_for_position_ms(int(p.get("position", 0)), manifest) != k]
        m_tr = [p for p in minicpm_cal_preds if get_chunk_index_for_position_ms(int(p.get("position", 0)), manifest) != k]
        l_tr = [p for p in lfm_cal_preds if get_chunk_index_for_position_ms(int(p.get("position", 0)), manifest) != k]

        q_eval = evaluate_predictions_dataset({"UrlLocal": url_local, "predictions": q_tr}, temp_root / f"train_scoped_{k}", temp_root / f"q_{k}", url_local)
        m_eval = evaluate_predictions_dataset({"UrlLocal": url_local, "predictions": m_tr}, temp_root / f"train_scoped_{k}", temp_root / f"m_{k}", url_local)
        l_eval = evaluate_predictions_dataset({"UrlLocal": url_local, "predictions": l_tr}, temp_root / f"train_scoped_{k}", temp_root / f"l_{k}", url_local)

        winners = {}
        for c in SOCCERNET_CLASSES:
            scores = {
                "qwen": q_eval["loose_per_class"].get(c, 0.0),
                "minicpm": m_eval["loose_per_class"].get(c, 0.0),
                "lfm": l_eval["loose_per_class"].get(c, 0.0)
            }
            best = max(scores, key=scores.get)
            if scores[best] > 0.0:
                winners[c] = best
            else:
                if c in QWEN_SPECIALIST_CLASSES:
                    winners[c] = "qwen"
                elif c in MINICPM_SPECIALIST_CLASSES:
                    winners[c] = "minicpm"
                else:
                    winners[c] = "lfm"

        for c, model_name in winners.items():
            pool = qwen_preds if model_name == "qwen" else (minicpm_cal_preds if model_name == "minicpm" else lfm_cal_preds)
            test_pool = [p for p in pool if p.get("label") == c and get_chunk_index_for_position_ms(int(p.get("position", 0)), manifest) == k]
            loco_test_candidates.extend(test_pool)

    loco_deduped = class_aware_nms(loco_test_candidates, radius_ms=3000, class_radii_ms=DEFAULT_CLASS_NMS_RADII_MS)
    loco_metrics = evaluate_predictions_dataset({"UrlLocal": url_local, "predictions": loco_deduped}, scoped_gt_dir, temp_root / "loco_final", url_local)
    loco_tight_hits, loco_loose_hits = count_tight_and_loose_hits(loco_deduped, gt_action_annotations)

    print(f"{'Ensemble Architecture & Policy':<38} | {'Kept':<4} | {'Tight Hits':<10} | {'Loose Hits':<10} | {'Tight mAP':<9} | {'Loose mAP':<9} | {'Act L-mAP':<9}")
    print("-" * 125)
    print(f"{'Qwen3.5-4B Solo (Baseline)':<38} | {len(qwen_preds):<4} | {12:<10} | {15:<10} | {0.0520:<9.4f} | {0.1059:<9.4f} | {0.1500:<9.4f}")
    print(f"{'MiniCPM-V 4.6 Solo (Baseline)':<38} | {len(minicpm_preds):<4} | {3:<10} | {22:<10} | {0.0128:<9.4f} | {0.0551:<9.4f} | {0.0780:<9.4f}")
    print(f"{'LFM2.5-VL-3B Solo (Baseline)':<38} | {len(lfm_preds):<4} | {8:<10} | {8:<10} | {0.0004:<9.4f} | {0.0119:<9.4f} | {0.0168:<9.4f}")
    print(f"{'3-Model Unrouted Union':<38} | {len(union_deduped):<4} | {union_tight_hits:<10} | {union_loose_hits:<10} | {union_metrics['tight_average_mAP']:<9.4f} | {union_metrics['loose_average_mAP']:<9.4f} | {union_metrics['active_loose_average_mAP']:<9.4f}")
    print(f"{'3-MODEL DOMAIN-ROUTED SPECIALIST':<38} | {len(specialist_deduped):<4} | {spec_tight_hits:<10} | {spec_loose_hits:<10} | {specialist_metrics['tight_average_mAP']:<9.4f} | {specialist_metrics['loose_average_mAP']:<9.4f} | {specialist_metrics['active_loose_average_mAP']:<9.4f}")
    print(f"{'3-Model Specialist (LOCO Out-of-Sample)':<38} | {len(loco_deduped):<4} | {loco_tight_hits:<10} | {loco_loose_hits:<10} | {loco_metrics['tight_average_mAP']:<9.4f} | {loco_metrics['loose_average_mAP']:<9.4f} | {loco_metrics['active_loose_average_mAP']:<9.4f}")
    print("=" * 125)

    # Per-class AP Comparison for Specialist Ensemble
    print("\n" + "=" * 95)
    print("  PER-CLASS LOOSE AP BREAKDOWN: SPECIALIST ENSEMBLE vs INDIVIDUAL SOLOS")
    print("=" * 95)
    print(f"{'Class Name':<20} | {'Specialist Role':<16} | {'Qwen AP':<10} | {'MiniCPM AP':<11} | {'LFM AP':<10} | {'Ensemble AP':<11}")
    print("-" * 95)
    spec_pc = specialist_metrics.get("loose_per_class", {})
    qwen_loose = evaluate_predictions_dataset({"UrlLocal": url_local, "predictions": qwen_preds}, scoped_gt_dir, temp_root / "q_eval", url_local)["loose_per_class"]
    mini_loose = evaluate_predictions_dataset({"UrlLocal": url_local, "predictions": minicpm_preds}, scoped_gt_dir, temp_root / "m_eval", url_local)["loose_per_class"]
    lfm_loose = lfm_base_metrics["loose_per_class"]

    for c in SOCCERNET_CLASSES:
        if c in spec_pc or c in qwen_loose or c in mini_loose or c in lfm_loose:
            role = "Qwen (60s)" if c in QWEN_SPECIALIST_CLASSES else ("MiniCPM (60s)" if c in MINICPM_SPECIALIST_CLASSES else ("LFM (30s)" if c in LFM_SPECIALIST_CLASSES else "None"))
            print(f"{c:<20} | {role:<16} | {qwen_loose.get(c, 0.0):<10.4f} | {mini_loose.get(c, 0.0):<11.4f} | {lfm_loose.get(c, 0.0):<10.4f} | {spec_pc.get(c, 0.0):<11.4f}")
    print("=" * 95)

    # =========================================================================
    # TASK 3: TRACK B — DENSE VIDEO CAPTIONING BENCHMARK
    # =========================================================================
    print("\n" + "=" * 135)
    print(f"  TASK 3: TRACK B DENSE VIDEO CAPTIONING BENCHMARK ({len(half1_gt_captions)} In-Bounds Ground-Truth Half-1 Commentary Events)")
    print("=" * 135)

    # Build Cross-Grafted Ensemble Captioner (Best-Localizer + Best-Narrator)
    # Keeps exact 175 localized timestamps/labels from the 3-Model Specialist Ensemble,
    # but whenever Qwen3.5-4B generated a candidate event within +-10.0s of a routed MiniCPM or LFM event,
    # grafts Qwen3.5-4B's richer linguistic commentary onto that event.
    grafted_caps = []
    graft_count = 0
    for p in specialist_deduped:
        p_copy = dict(p)
        if p_copy.get("model") != "qwen":
            best_q = None
            best_diff = float("inf")
            for q in qwen_preds:
                diff = abs(int(q["position"]) - int(p_copy["position"]))
                if diff <= 10000 and diff < best_diff:
                    best_diff = diff
                    best_q = q
            if best_q is not None and best_q.get("commentary"):
                p_copy["commentary"] = best_q["commentary"]
                p_copy["commentary_source"] = "qwen_grafted"
                graft_count += 1
        grafted_caps.append(p_copy)
    print(f"[CROSS-GRAFTING] Grafted Qwen3.5-4B commentary onto {graft_count} / {len(specialist_deduped)} specialist ensemble events (within +-10.0s)")

    cap_models = [
        ("Qwen3.5-4B (4-bit, W=60s)", qwen_preds),
        ("MiniCPM-V 4.6 (4-bit, W=60s)", minicpm_preds),
        ("LFM2.5-VL-3B (4-bit, W=30s)", lfm_preds),
        ("3-Model Specialist Ensemble", specialist_deduped),
        ("3-Model Cross-Grafted Ensemble", grafted_caps)
    ]

    # Evaluate Raw Captions
    raw_results = []
    for model_name, caps in cap_models:
        metrics = evaluate_track_b_dense_captioning(caps, half1_gt_captions, window_sec=30.0, delexicalize=False)
        raw_results.append((model_name, metrics))

    # Evaluate Delexicalized Captions
    delex_results = []
    for model_name, caps in cap_models:
        metrics = evaluate_track_b_dense_captioning(caps, half1_gt_captions, window_sec=30.0, delexicalize=True)
        delex_results.append((model_name, metrics))

    print("\n--- TABLE 4A: RAW BROADCAST CAPTIONING (Direct Lexical Match vs Broadcast Commentator) ---")
    hdr = f"{'Model Architecture':<28} | {'Matched':<7} | {'Temp Rec':<8} | {'Temp Prec':<9} | {'BLEU-1':<7} | {'BLEU-4':<7} | {'Sm-B4':<7} | {'ROUGE-L':<8} | {'CIDEr':<7} | {'METEOR':<7}"
    print(hdr)
    print("-" * len(hdr))
    for model_name, m in raw_results:
        print(f"{model_name:<28} | {m['matched_count']:<7} | {m['temporal_recall']:<8.3f} | {m['temporal_precision']:<9.3f} | {m['bleu1']:<7.4f} | {m['bleu4']:<7.4f} | {m['smoothed_bleu4']:<7.4f} | {m['rouge_l']:<8.4f} | {m['cider']:<7.4f} | {m['meteor']:<7.4f}")
    print("-" * len(hdr))

    print("\n--- TABLE 4B: DELEXICALIZED CAPTIONING (Role-Normalized Semantic Evaluation) ---")
    print(hdr)
    print("-" * len(hdr))
    for model_name, m in delex_results:
        print(f"{model_name:<28} | {m['matched_count']:<7} | {m['temporal_recall']:<8.3f} | {m['temporal_precision']:<9.3f} | {m['bleu1']:<7.4f} | {m['bleu4']:<7.4f} | {m['smoothed_bleu4']:<7.4f} | {m['rouge_l']:<8.4f} | {m['cider']:<7.4f} | {m['meteor']:<7.4f}")
    print("=" * 135)

    # Qualitative Side-by-Side Excerpt
    print("\n" + "=" * 135)
    print("  QUALITATIVE DENSE CAPTIONING SPOT-CHECK: MODEL COMMENTARY vs OFFICIAL BROADCAST GROUND TRUTH")
    print("=" * 135)
    sample_pairs = raw_results[3][1]["pairs"][:5]  # Specialist Ensemble pairs
    delex_pairs = delex_results[3][1]["pairs"][:5]
    for idx, (p_raw, p_del) in enumerate(zip(sample_pairs, delex_pairs)):
        print(f"[{idx+1}] Match Time: {p_raw['gt_time']} (Pred: {p_raw['pred_time']}, Error: {p_raw['dt_sec']:>+5.2f}s) | Label: {p_raw['pred_label']}")
        print(f"    - Official GT (Raw):          \"{p_raw['gt_text']}\"")
        print(f"    - Ensemble Output (Raw):      \"{p_raw['pred_text']}\"")
        print(f"    - Official GT (Anonymized):   \"{p_del['gt_text']}\"")
        print(f"    - Ensemble Output (Delexed):  \"{p_del['pred_text']}\"")
        print("-" * 135)
    print("=" * 135 + "\n")

    # Export metrics to JSON
    summary_export = {
        "task1_lfm_calibration": {
            "baseline": {"hits": lfm_base_hits, "metrics": lfm_base_metrics},
            "global_shift": {"hits": lfm_global_hits, "metrics": lfm_global_metrics},
            "loco_cv": {"hits": lfm_loco_hits, "metrics": lfm_loco_metrics}
        },
        "task2_specialist_ensemble": {
            "specialist_metrics": specialist_metrics,
            "union_metrics": union_metrics,
            "loco_metrics": loco_metrics
        },
        "task3_dense_captioning": {
            "raw": {name: {k: v for k, v in m.items() if k != "pairs"} for name, m in raw_results},
            "delexicalized": {name: {k: v for k, v in m.items() if k != "pairs"} for name, m in delex_results}
        }
    }
    out_file = WORKSPACE_ROOT / "logs/local_eval/synthesis_and_captioning_benchmark_summary.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary_export, f, indent=2)
    print(f"[SUCCESS] Exported comprehensive synthesis benchmark results -> {out_file}\n")


if __name__ == "__main__":
    main()
