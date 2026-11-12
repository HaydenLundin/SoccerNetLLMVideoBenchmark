"""
Track J: Game State Reconstruction (2D Pitch Minimap & Role Classification).

Evaluates 2D metric pitch localization and 4-role kit classification across
8 calibrated real broadcast frames from Game 0 Half 1:
  - Baseline: Uncalibrated Linear Viewport Projection + Raw RGB Binned Classifier
  - Track J : Track I Pinhole Homography (H^{-1}) + Chroma-Guided HSV Kit Classifier

Ground-truth player world positions (X_gt, Y_gt) are obtained by projecting
visually verified foot coordinates (u_gt, v_gt) through each frame's
independent DLT ground-truth homography (fit_independent_gt_homography).
Matching is performed strictly in world metric space (meters) via Hungarian
bipartite assignment with a 2.0m distance threshold.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import random
import numpy as np
from scipy.optimize import linear_sum_assignment

# Deterministic execution seed (Audit 5)
random.seed(42)
np.random.seed(42)

import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from processing.pitch_geometry import (
    HALF_LENGTH,
    HALF_WIDTH,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    PITCH_LENGTH,
    PITCH_WIDTH,
    detect_players_on_pitch,
    estimate_pitch_homography,
    project_image_to_world,
)
from scripts.evaluate_v5_camera_calibration import (
    FRAME_SCHEDULE,
    MANIFEST_PATH,
    _save_contact_sheet,
    decode_frame_at,
    fit_independent_gt_homography,
)

OUTPUT_DIR = ROOT / "logs" / "v5_eval" / "game_state_gsr"
SUMMARY_PATH = OUTPUT_DIR / "game_state_gsr_summary.json"
LOC_THRESHOLD_M = 2.0

# Visually verified on-pitch player annotations: (u_foot, v_foot, role)
GT_PLAYER_ANNOTATIONS: Dict[str, List[Tuple[float, float, str]]] = {
    "c0_0045": [
        (960.0, 312.0, "Goalkeeper"),
        (984.0, 304.0, "Away Team"),
        (392.0, 364.0, "Away Team"),
        (107.0, 400.0, "Home Team"),
        (1134.0, 422.0, "Home Team"),
        (743.0, 720.0, "Home Team"),
    ],
    "c0_0130": [
        (65.0, 178.0, "Home Team"),
        (173.0, 232.0, "Away Team"),
        (623.0, 256.0, "Away Team"),
        (692.0, 244.0, "Home Team"),
        (1077.0, 248.0, "Away Team"),
        (1104.0, 286.0, "Goalkeeper"),
        (248.0, 308.0, "Away Team"),
        (860.0, 292.0, "Home Team"),
        (199.0, 346.0, "Home Team"),
        (764.0, 368.0, "Away Team"),
        (407.0, 594.0, "Home Team"),
        (675.0, 608.0, "Away Team"),
    ],
    "c0_0730": [
        (896.0, 206.0, "Away Team"),
        (722.0, 248.0, "Home Team"),
        (1174.0, 304.0, "Away Team"),
        (1076.0, 286.0, "Home Team"),
        (497.0, 302.0, "Home Team"),
        (541.0, 326.0, "Away Team"),
        (833.0, 336.0, "Home Team"),
        (1006.0, 332.0, "Away Team"),
        (765.0, 394.0, "Away Team"),
        (629.0, 448.0, "Home Team"),
        (197.0, 486.0, "Home Team"),
        (519.0, 478.0, "Referee"),
        (445.0, 512.0, "Away Team"),
        (701.0, 640.0, "Home Team"),
        (12.0, 668.0, "Home Team"),
    ],
    "c1_1145": [
        (542.0, 194.0, "Home Team"),
        (886.0, 208.0, "Home Team"),
        (1026.0, 219.0, "Away Team"),
        (392.0, 270.0, "Away Team"),
        (542.0, 334.0, "Home Team"),
        (566.0, 338.0, "Away Team"),
        (1230.0, 374.0, "Away Team"),
        (621.0, 379.0, "Referee"),
        (237.0, 462.0, "Home Team"),
        (1262.0, 524.0, "Home Team"),
        (470.0, 510.0, "Away Team"),
        (745.0, 602.0, "Home Team"),
        (1018.0, 688.0, "Away Team"),
    ],
    "c2_2105": [
        (610.0, 178.0, "Away Team"),
        (231.0, 269.0, "Home Team"),
        (1013.0, 334.0, "Away Team"),
        (278.0, 350.0, "Home Team"),
        (130.0, 374.0, "Away Team"),
        (462.0, 476.0, "Home Team"),
        (676.0, 478.0, "Away Team"),
        (988.0, 494.0, "Home Team"),
        (368.0, 720.0, "Home Team"),
    ],
    "c2_2919": [
        (184.0, 240.0, "Home Team"),
        (186.0, 272.0, "Away Team"),
        (325.0, 248.0, "Away Team"),
        (587.0, 302.0, "Home Team"),
        (676.0, 356.0, "Away Team"),
        (297.0, 346.0, "Away Team"),
        (752.0, 414.0, "Home Team"),
        (785.0, 425.0, "Away Team"),
        (802.0, 488.0, "Away Team"),
        (418.0, 438.0, "Home Team"),
        (904.0, 478.0, "Home Team"),
        (952.0, 294.0, "Goalkeeper"),
    ],
    "c3_2949": [
        (690.0, 286.0, "Away Team"),
        (94.0, 280.0, "Referee"),
        (572.0, 290.0, "Away Team"),
        (372.0, 354.0, "Away Team"),
        (694.0, 494.0, "Away Team"),
        (726.0, 405.0, "Away Team"),
        (750.0, 448.0, "Away Team"),
        (577.0, 411.0, "Home Team"),
        (1036.0, 448.0, "Goalkeeper"),
        (631.0, 469.0, "Home Team"),
        (478.0, 451.0, "Home Team"),
        (485.0, 554.0, "Home Team"),
    ],
    "c4_4355": [
        (611.0, 192.0, "Home Team"),
        (934.0, 220.0, "Away Team"),
        (80.0, 255.0, "Away Team"),
        (145.0, 320.0, "Home Team"),
        (1233.0, 342.0, "Away Team"),
        (576.0, 344.0, "Home Team"),
        (338.0, 374.0, "Home Team"),
        (194.0, 400.0, "Away Team"),
        (1004.0, 444.0, "Away Team"),
        (403.0, 415.0, "Referee"),
        (620.0, 422.0, "Away Team"),
        (573.0, 456.0, "Home Team"),
        (855.0, 610.0, "Home Team"),
    ],
}


def linear_viewport_project(foot_uv: np.ndarray) -> np.ndarray:
    """Uncalibrated baseline mapping pixel (u, v) linearly onto center viewport meters."""
    pts = np.atleast_2d(np.asarray(foot_uv, dtype=np.float64))
    xw = ((pts[:, 0] / float(IMAGE_WIDTH)) - 0.5) * 68.0
    yw = ((pts[:, 1] / float(IMAGE_HEIGHT)) - 0.55) * 52.0
    return np.stack([xw, yw], axis=1)


def baseline_rgb_role(bgr_img: np.ndarray, bbox: List[int]) -> str:
    """Simple raw BGR torso dominance baseline (cannot cleanly separate GK/Referee)."""
    x1, y1, x2, y2 = bbox
    bh = max(1, y2 - y1)
    patch = bgr_img[y1 + int(0.18 * bh):y1 + int(0.55 * bh), x1:x2]
    if patch.size == 0:
        return "Home Team"
    mean_bgr = patch.reshape(-1, 3).mean(axis=0)
    b, g, r = float(mean_bgr[0]), float(mean_bgr[1]), float(mean_bgr[2])
    if r > b + 20 and g < r - 25:
        return "Home Team"
    return "Away Team"


def hungarian_match_world(
    gt_world: np.ndarray,
    gt_roles: List[str],
    pred_world: np.ndarray,
    pred_roles: List[str],
    thresh_m: float = LOC_THRESHOLD_M,
) -> Dict[str, Any]:
    """Bipartite Hungarian matching strictly in Euclidean world metric space (meters)."""
    n_gt = len(gt_world)
    n_pred = len(pred_world)
    if n_gt == 0 or n_pred == 0:
        return {
            "tp": 0,
            "fp": n_pred,
            "fn": n_gt,
            "role_correct": 0,
            "errors_m": [],
        }

    dists = np.linalg.norm(gt_world[:, None, :] - pred_world[None, :, :], axis=2)
    row_ind, col_ind = linear_sum_assignment(dists)

    tp = 0
    role_correct = 0
    errors_m: List[float] = []
    for r, c in zip(row_ind, col_ind):
        d_m = float(dists[r, c])
        if d_m <= thresh_m:
            tp += 1
            errors_m.append(d_m)
            if gt_roles[r] == pred_roles[c]:
                role_correct += 1

    return {
        "tp": tp,
        "fp": n_pred - tp,
        "fn": n_gt - tp,
        "role_correct": role_correct,
        "errors_m": errors_m,
    }


def render_2d_minimap_canvas(
    gt_world: np.ndarray,
    gt_roles: List[str],
    pred_world: np.ndarray,
    pred_roles: List[str],
    frame_id: str,
) -> np.ndarray:
    """Renders a 2D canonical 105m x 68m tactical pitch canvas comparing GT vs Pred."""
    w, h = 640, 420
    pad = 25
    canvas = np.full((h, w, 3), (24, 58, 34), dtype=np.uint8)

    def m2px(x_m: float, y_m: float) -> Tuple[int, int]:
        px = int(round(pad + ((x_m + HALF_LENGTH) / PITCH_LENGTH) * (w - 2 * pad)))
        py = int(round(pad + ((y_m + HALF_WIDTH) / PITCH_WIDTH) * (h - 2 * pad)))
        return px, py

    # Draw pitch lines
    line_col = (225, 235, 225)
    cv2.rectangle(canvas, m2px(-HALF_LENGTH, -HALF_WIDTH), m2px(HALF_LENGTH, HALF_WIDTH), line_col, 2)
    cv2.line(canvas, m2px(0, -HALF_WIDTH), m2px(0, HALF_WIDTH), line_col, 2)
    cx, cy = m2px(0, 0)
    rad_px = int(round((9.15 / PITCH_LENGTH) * (w - 2 * pad)))
    cv2.circle(canvas, (cx, cy), rad_px, line_col, 2)
    cv2.rectangle(canvas, m2px(-HALF_LENGTH, -20.16), m2px(-36.0, 20.16), line_col, 2)
    cv2.rectangle(canvas, m2px(36.0, -20.16), m2px(HALF_LENGTH, 20.16), line_col, 2)
    cv2.rectangle(canvas, m2px(-HALF_LENGTH, -9.16), m2px(-47.0, 9.16), line_col, 1)
    cv2.rectangle(canvas, m2px(47.0, -9.16), m2px(HALF_LENGTH, 9.16), line_col, 1)

    role_bgr = {
        "Home Team": (60, 60, 240),
        "Away Team": (30, 225, 255),
        "Referee": (245, 220, 45),
        "Goalkeeper": (20, 145, 255),
    }

    for (xg, yg), r in zip(gt_world, gt_roles):
        cv2.circle(canvas, m2px(float(xg), float(yg)), 7, role_bgr.get(r, (255, 255, 255)), 2)

    for (xp, yp), r in zip(pred_world, pred_roles):
        if abs(xp) <= HALF_LENGTH + 3.0 and abs(yp) <= HALF_WIDTH + 3.0:
            cv2.circle(canvas, m2px(float(xp), float(yp)), 4, role_bgr.get(r, (255, 255, 255)), -1)

    cv2.putText(canvas, f"{frame_id} | Circle=GT, Dot=Pred (m)", (16, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def compute_system_metrics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Computes SoccerNet GSR localization precision/recall, metric error, role acc & GS-HOTA."""
    total_tp = sum(r["tp"] for r in records)
    total_fp = sum(r["fp"] for r in records)
    total_fn = sum(r["fn"] for r in records)
    total_role_correct = sum(r["role_correct"] for r in records)
    all_errors = [e for r in records for e in r["errors_m"]]

    n_pred = total_tp + total_fp
    n_gt = total_tp + total_fn
    precision = total_tp / float(n_pred) if n_pred > 0 else 0.0
    recall = total_tp / float(n_gt) if n_gt > 0 else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    det_a = total_tp / float(total_tp + total_fp + total_fn) if (total_tp + total_fp + total_fn) > 0 else 0.0
    role_acc = (total_role_correct / float(total_tp) * 100.0) if total_tp > 0 else 0.0
    ass_a = total_role_correct / float(total_tp + 0.5 * (total_fp + total_fn)) if total_tp > 0 else 0.0
    gs_hota = math.sqrt(max(0.0, det_a * ass_a)) if total_tp > 0 else 0.0

    return {
        "n_gt": n_gt,
        "n_pred": n_pred,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "loc_precision_2m_pct": round(precision * 100.0, 2),
        "loc_recall_2m_pct": round(recall * 100.0, 2),
        "loc_f1_2m_pct": round(f1 * 100.0, 2),
        "mean_metric_error_m": round(float(np.mean(all_errors)), 3) if all_errors else 99.999,
        "median_metric_error_m": round(float(np.median(all_errors)), 3) if all_errors else 99.999,
        "role_classification_acc_pct": round(role_acc, 2),
        "det_a_pct": round(det_a * 100.0, 2),
        "ass_a_pct": round(ass_a * 100.0, 2),
        "composite_gs_hota_pct": round(gs_hota * 100.0, 2),
    }


def evaluate_track_j(save_sheets_dir: str = "") -> Dict[str, Any]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    chunk_map = {int(c["chunk_index"]): ROOT / c["relative_path"] for c in manifest["chunks"]}

    baseline_records: List[Dict[str, Any]] = []
    trackj_records: List[Dict[str, Any]] = []
    per_frame: List[Dict[str, Any]] = []
    minimap_tiles: List[np.ndarray] = []

    t_start = time.perf_counter()
    for fspec in FRAME_SCHEDULE:
        fid = fspec["frame_id"]
        if fid not in GT_PLAYER_ANNOTATIONS:
            continue

        bgr = decode_frame_at(chunk_map[fspec["chunk"]], fspec["rel_sec"])
        gt_info = fit_independent_gt_homography(bgr, fid)
        H_gt = np.asarray(gt_info["H_gt"], dtype=np.float64)

        gt_pts = GT_PLAYER_ANNOTATIONS[fid]
        gt_uv = np.array([[u, v] for u, v, _ in gt_pts], dtype=np.float64)
        gt_world = project_image_to_world(H_gt, gt_uv)
        gt_roles = [r for _, _, r in gt_pts]

        # Run Track I calibrated homography + Track J chroma-guided detector
        cal = estimate_pitch_homography(bgr)
        H_pred = np.asarray(cal["H"], dtype=np.float64)
        dets = detect_players_on_pitch(bgr, field_mask=cal.get("field_mask"), H=H_pred)

        pred_uv = np.array([d["foot_uv"] for d in dets], dtype=np.float64) if dets else np.zeros((0, 2))
        pred_world_tj = (
            np.array([d["world_xy"] for d in dets], dtype=np.float64) if dets else np.zeros((0, 2))
        )
        pred_roles_tj = [d["role"] for d in dets]

        # Baseline: same boxes projected linearly without homography + raw RGB role
        pred_world_base = linear_viewport_project(pred_uv) if len(pred_uv) else np.zeros((0, 2))
        pred_roles_base = [baseline_rgb_role(bgr, d["bbox"]) for d in dets]

        m_base = hungarian_match_world(gt_world, gt_roles, pred_world_base, pred_roles_base, LOC_THRESHOLD_M)
        m_tj = hungarian_match_world(gt_world, gt_roles, pred_world_tj, pred_roles_tj, LOC_THRESHOLD_M)

        baseline_records.append(m_base)
        trackj_records.append(m_tj)

        per_frame.append({
            "frame_id": fid,
            "view": fspec["view"],
            "n_gt": len(gt_pts),
            "n_pred": len(dets),
            "baseline_tp_2m": m_base["tp"],
            "trackj_tp_2m": m_tj["tp"],
            "trackj_mean_err_m": round(float(np.mean(m_tj["errors_m"])), 3) if m_tj["errors_m"] else 99.999,
            "trackj_role_acc_pct": round(m_tj["role_correct"] / float(m_tj["tp"]) * 100.0, 2) if m_tj["tp"] else 0.0,
        })

        if save_sheets_dir:
            minimap_tiles.append(render_2d_minimap_canvas(gt_world, gt_roles, pred_world_tj, pred_roles_tj, fid))

    elapsed_s = time.perf_counter() - t_start
    base_agg = compute_system_metrics(baseline_records)
    tj_agg = compute_system_metrics(trackj_records)

    summary = {
        "track": "Track J: Game State Reconstruction (2D Pitch Minimap)",
        "dataset": "Game 0 Half 1 (8 Multi-View Calibrated Broadcast Frames)",
        "matching_threshold_m": LOC_THRESHOLD_M,
        "pitch_dimensions_m": [PITCH_LENGTH, PITCH_WIDTH],
        "runtime_sec": round(elapsed_s, 2),
        "baseline_linear_viewport": base_agg,
        "track_j_calibrated_gsr": tj_agg,
        "per_frame": per_frame,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if save_sheets_dir and minimap_tiles:
        sdir = Path(save_sheets_dir)
        sdir.mkdir(parents=True, exist_ok=True)
        _save_contact_sheet(minimap_tiles[:4], sdir / "trackj_minimap_0.jpg")
        if len(minimap_tiles) > 4:
            _save_contact_sheet(minimap_tiles[4:8], sdir / "trackj_minimap_1.jpg")

    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate Track J Game State Reconstruction.")
    parser.add_argument("--save-sheets", type=str, default="", help="Optional dir for 2D minimap sheets.")
    args = parser.parse_args()
    res = evaluate_track_j(args.save_sheets)
    print(json.dumps({
        "baseline": res["baseline_linear_viewport"],
        "track_j": res["track_j_calibrated_gsr"],
    }, indent=2))
