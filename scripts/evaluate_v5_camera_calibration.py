"""
Track I — SoccerNet Camera Calibration & Pitch Line Registration Evaluation (v5.0)
=================================================================================
Evaluates zero-GT monocular stadium pitch calibration across 12 real Game 0 Half 1
1280x720 broadcast keyframes spanning main-gantry, box zooms, midfield panning,
and auxiliary 18-yard offside gantry views (`gate2_chunks/game_0_half_1_chunk_*.mp4`).

Compares:
  (a) Classical Hough Line-Pair RANSAC Homography Baseline
  (b) Track I Multi-Start Soft-Inlier Chamfer Pinhole Calibration

Against independently annotated unconstrained 8-DOF homography ground truth (H_gt)
fitted directly to visually verified semantic chalk line corridors and pitch circles
(zero shared camera model or Chamfer search code).

Outputs:
  logs/v5_eval/camera_calibration/camera_calibration_summary.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import av
import cv2
import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from processing.pitch_geometry import (
    CANONICAL_PITCH_POINTS,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    compute_pitch_part_iou,
    compute_reprojection_accuracy,
    estimate_pitch_homography,
    extract_turf_and_chalk_masks,
    project_image_to_world,
    project_world_to_image,
)

MANIFEST_PATH = ROOT / "gate2_chunks" / "game_0_half_1_chunks_manifest.json"
OUT_JSON_PATH = ROOT / "logs" / "v5_eval" / "camera_calibration" / "camera_calibration_summary.json"

# 12 real verified Game 0 Half 1 wide tactical keyframes across Chunks 00..04
FRAME_SCHEDULE: List[Dict[str, Any]] = [
    {"frame_id": "c0_0045", "chunk": 0, "rel_sec": 45.0, "view": "Right Penalty Box"},
    {"frame_id": "c0_0130", "chunk": 0, "rel_sec": 90.0, "view": "Right Box & Mid-Wing"},
    {"frame_id": "c0_0730", "chunk": 0, "rel_sec": 450.0, "view": "Midfield Left"},
    {"frame_id": "c1_1145", "chunk": 1, "rel_sec": 123.0, "view": "Center Circle"},
    {"frame_id": "c1_1417", "chunk": 1, "rel_sec": 275.0, "view": "Right Box Free Kick"},
    {"frame_id": "c2_2105", "chunk": 2, "rel_sec": 100.0, "view": "Midfield Right"},
    {"frame_id": "c2_2505", "chunk": 2, "rel_sec": 340.0, "view": "Right-Center Transition"},
    {"frame_id": "c2_2919", "chunk": 2, "rel_sec": 594.0, "view": "Right Box Goal Action"},
    {"frame_id": "c3_2949", "chunk": 3, "rel_sec": 35.0, "view": "Auxiliary 18-yd Offside Gantry"},
    {"frame_id": "c3_3310", "chunk": 3, "rel_sec": 236.0, "view": "Left Penalty Box"},
    {"frame_id": "c4_4120", "chunk": 4, "rel_sec": 145.0, "view": "Left Box Save"},
    {"frame_id": "c4_4355", "chunk": 4, "rel_sec": 300.0, "view": "Center Circle Wide"},
]

# Independent semantic line-corridor & circle/arc ROI specifications for unconstrained
# 8-DOF H_gt estimation (fit strictly against raw chalk pixels inside each named ROI).
# Format:
#   lines:  [(X0, Y0, X1, Y1, u0, v0, u1, v1, corridor_halfwidth_px), ...]
#   circles: [(cx_w, cy_w, radius_w, u_min, v_min, u_max, v_max), ...]
#   init_pairs: [(line_idx_a, line_idx_b, X_w, Y_w), ...] or explicit initial pts
GT_SPECS: Dict[str, Dict[str, Any]] = {
    "c0_0045": {
        "lines": [
            (36.0, -20.16, 36.0, 20.16, 212, 216, 691, 372, 10),   # Right Pen L
            (36.0, 20.16, 52.5, 20.16, 212, 216, 460, 192, 10),    # Right Pen T
            (52.5, -20.16, 52.5, 34.0, 409, 127, 1270, 324, 12),   # Right Goal Line
            (47.0, -9.16, 47.0, 9.16, 654, 229, 1058, 329, 10),    # Right 6yd L
            (47.0, 9.16, 52.5, 9.16, 654, 229, 865, 210, 10),      # Right 6yd T
            (47.0, -9.16, 52.5, -9.16, 1058, 329, 1245, 312, 10),  # Right 6yd B
            (-52.5, 34.0, 52.5, 34.0, 5, 168, 415, 127, 10),       # Top Touchline
        ],
        "circles": [(41.5, 0.0, 9.15, 290, 235, 580, 340)],
    },
    "c0_0130": {
        "lines": [
            (36.0, -20.16, 36.0, 20.16, 561, 272, 1265, 530, 12),  # Right Pen L
            (36.0, 20.16, 52.5, 20.16, 562, 272, 1027, 225, 10),   # Right Pen T
            (52.5, -20.16, 52.5, 34.0, 795, 169, 1275, 285, 12),   # Right Goal Line
            (-52.5, 34.0, 52.5, 34.0, 15, 205, 795, 169, 6),       # Top Touchline
            (47.0, -9.16, 47.0, 9.16, 1103, 301, 1278, 347, 10),   # Right 6yd L
            (47.0, 9.16, 52.5, 9.16, 1103, 301, 1278, 285, 10),    # Right 6yd T
        ],
        "circles": [(41.5, 0.0, 9.15, 700, 330, 1120, 470)],
    },
    "c0_0730": {
        "lines": [
            (0.0, -34.0, 0.0, 34.0, 845, 226, 835, 718, 12),       # Halfway Line
            (-52.5, 34.0, 52.5, 34.0, 4, 216, 1270, 234, 10),      # Top Touchline
        ],
        "circles": [(0.0, 0.0, 9.15, 370, 360, 1260, 700)],
    },
    "c1_1145": {
        "lines": [
            (0.0, -34.0, 0.0, 34.0, 530, 150, 533, 718, 12),       # Halfway Line
            (-52.5, 34.0, 52.5, 34.0, 10, 152, 1270, 156, 10),     # Top Touchline
        ],
        "circles": [(0.0, 0.0, 9.15, 95, 250, 975, 675)],
    },
    "c1_1417": {
        "lines": [
            (36.0, -20.16, 36.0, 20.16, 1, 322, 1010, 648, 10),    # Right Pen L
            (36.0, 20.16, 52.5, 20.16, 1, 226, 415, 180, 8),       # Right Pen T
            (52.5, -20.16, 52.5, 34.0, 104, 112, 1220, 358, 10),   # Right Goal Line
            (47.0, 9.16, 52.5, 9.16, 560, 260, 695, 243, 8),       # Right 6yd T (clear chalk)
            (47.0, -9.16, 47.0, 9.16, 905, 360, 1245, 435, 8),     # Right 6yd L (below wall)
            (47.0, -9.16, 52.5, -9.16, 1165, 417, 1275, 405, 7),   # Right 6yd B
            (-52.5, 34.0, 52.5, 34.0, 1, 120, 104, 112, 5),        # Top Touchline
        ],
        "circles": [(41.5, 0.0, 9.15, 45, 355, 580, 525)],
    },
    "c2_2105": {
        "lines": [
            (0.0, -34.0, 0.0, 34.0, 255, 203, 266, 718, 12),       # Halfway Line
            (-52.5, 34.0, 52.5, 34.0, 10, 217, 1260, 186, 10),     # Top Touchline
        ],
        "circles": [(0.0, 0.0, 9.15, 5, 392, 745, 705)],
    },
    "c2_2505": {
        "lines": [
            (36.0, -20.16, 36.0, 20.16, 756, 160, 1275, 360, 12),  # Right Pen L
            (36.0, 20.16, 52.5, 20.16, 756, 161, 1085, 132, 10),   # Right Pen T
            (52.5, -20.16, 52.5, 34.0, 914, 93, 1275, 182, 10),    # Right Goal Line
            (-52.5, 34.0, 52.5, 34.0, 10, 155, 914, 93, 8),        # Top Touchline
            (47.0, -9.16, 47.0, 9.16, 1118, 183, 1278, 225, 10),   # Right 6yd L
            (47.0, 9.16, 52.5, 9.16, 1118, 183, 1278, 173, 10),    # Right 6yd T
        ],
        "circles": [
            (0.0, 0.0, 9.15, 5, 325, 240, 545),
            (41.5, 0.0, 9.15, 890, 200, 1150, 290),
        ],
    },
    "c2_2919": {
        "lines": [
            (36.0, -20.16, 36.0, 20.16, 1, 360, 1035, 698, 10),    # Right Pen L
            (52.5, -20.16, 52.5, 34.0, 78, 156, 1268, 421, 10),    # Right Goal Line
            (36.0, 20.16, 52.5, 20.16, 1, 265, 385, 222, 6),       # Right Pen T
            (47.0, 9.16, 52.5, 9.16, 460, 310, 650, 281, 6),       # Right 6yd T
            (47.0, -9.16, 47.0, 9.16, 460, 310, 1035, 452, 8),     # Right 6yd L
            (47.0, -9.16, 52.5, -9.16, 1035, 452, 1235, 415, 8),   # Right 6yd B
            (-52.5, 34.0, 52.5, 34.0, 5, 160, 78, 156, 5),         # Top Touchline
        ],
        "circles": [(41.5, 0.0, 9.15, 10, 405, 410, 560)],
    },
    "c3_2949": {
        "lines": [
            (36.0, -20.16, 36.0, 20.16, 231, 293, 217, 718, 10),   # Right Pen L
            (36.0, 20.16, 52.5, 20.16, 235, 292, 970, 284, 10),    # Right Pen T
            (47.0, -9.16, 47.0, 9.16, 793, 370, 941, 567, 10),     # Right 6yd L
            (47.0, 9.16, 52.5, 9.16, 796, 368, 1065, 365, 10),     # Right 6yd T
            (47.0, -9.16, 52.5, -9.16, 942, 566, 1270, 565, 10),   # Right 6yd B
            (52.5, -20.16, 52.5, 34.0, 935, 262, 1216, 497, 10),   # Right Goal Line
        ],
        "circles": [(41.5, 0.0, 9.15, 1, 385, 224, 565)],
    },
    "c3_3310": {
        "lines": [
            (-36.0, -20.16, -36.0, 20.16, 1, 411, 542, 203, 12),   # Left Pen L
            (-36.0, 20.16, -52.5, 20.16, 160, 170, 540, 203, 10),  # Left Pen T
            (-52.5, -20.16, -52.5, 34.0, 3, 211, 355, 124, 10),    # Left Goal Line
            (-52.5, 34.0, 52.5, 34.0, 355, 124, 1275, 180, 10),    # Top Touchline
            (-47.0, -9.16, -47.0, 9.16, 1, 275, 210, 210, 10),     # Left 6yd L
        ],
        "circles": [(-41.5, 0.0, 9.15, 150, 255, 480, 350)],
    },
    "c4_4120": {
        "lines": [
            (-36.0, -20.16, -36.0, 20.16, 2, 433, 683, 175, 10),   # Left Pen L
            (-36.0, 20.16, -52.5, 20.16, 274, 139, 683, 175, 8),   # Left Pen T
            (-52.5, -20.16, -52.5, 34.0, 1, 210, 428, 92, 8),      # Left Goal Line
            (-47.0, 9.16, -52.5, 9.16, 30, 201, 112, 210, 6),      # Left 6yd T
            (-47.0, -9.16, -47.0, 9.16, 1, 252, 112, 210, 8),      # Left 6yd L
        ],
        "circles": [(-41.5, 0.0, 9.15, 45, 255, 465, 425)],
    },
    "c4_4355": {
        "lines": [
            (0.0, -34.0, 0.0, 34.0, 713, 156, 703, 718, 12),       # Halfway Line
            (-52.5, 34.0, 52.5, 34.0, 10, 167, 1270, 175, 10),     # Top Touchline
        ],
        "circles": [(0.0, 0.0, 9.15, 280, 315, 1150, 615)],
    },
}


def decode_frame_at(video_path: Path, rel_sec: float) -> np.ndarray:
    """Decodes the exact RGB->BGR broadcast frame nearest `rel_sec` using PyAV."""
    container = av.open(str(video_path))
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"
    seek_pts = int(max(0.0, rel_sec - 2.0) / float(stream.time_base))
    container.seek(seek_pts, any_frame=False, backward=True, stream=stream)
    best_frame = None
    best_diff = float("inf")
    for frame in container.decode(video=0):
        t_s = float(frame.pts * frame.time_base) if frame.pts is not None else rel_sec
        diff = abs(t_s - rel_sec)
        if diff < best_diff:
            best_diff = diff
            best_frame = frame.to_ndarray(format="bgr24")
        if t_s > rel_sec + 0.15 and best_frame is not None:
            break
    container.close()
    if best_frame is None:
        raise RuntimeError(f"Failed to decode frame at {rel_sec}s from {video_path}")
    return best_frame


def _extract_corridor_points(
    chalk_pts: np.ndarray,
    u0: float, v0: float, u1: float, v1: float, radius: float
) -> np.ndarray:
    dx, dy = u1 - u0, v1 - v0
    length_sq = dx * dx + dy * dy
    if length_sq < 1e-6:
        return np.empty((0, 2), dtype=np.float64)
    length = float(np.sqrt(length_sq))
    t = ((chalk_pts[:, 0] - u0) * dx + (chalk_pts[:, 1] - v0) * dy) / length_sq
    perp = np.abs((chalk_pts[:, 0] - u0) * dy - (chalk_pts[:, 1] - v0) * dx) / length
    mask = (t >= -0.02) & (t <= 1.02) & (perp <= radius)
    return chalk_pts[mask].astype(np.float64)


def _fit_2d_line(pts: np.ndarray) -> Tuple[float, float, float]:
    vx, vy, x0, y0 = cv2.fitLine(pts.astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).flatten()
    a, b = float(vy), -float(vx)
    c = -(a * float(x0) + b * float(y0))
    return (a, b, c)  # a*u + b*v + c = 0


def _intersect_2d_lines(l1: Tuple[float, float, float], l2: Tuple[float, float, float]) -> Optional[Tuple[float, float]]:
    a1, b1, c1 = l1
    a2, b2, c2 = l2
    det = a1 * b2 - a2 * b1
    if abs(det) < 1e-7:
        return None
    u = (b1 * c2 - b2 * c1) / det
    v = (c1 * a2 - c2 * a1) / det
    return (float(u), float(v))


def _initial_gt_homography(
    spec: Dict[str, Any],
    line_eqs: List[Optional[Tuple[float, float, float]]],
    chalk_pts: np.ndarray,
) -> np.ndarray:
    """
    Constructs an analytical initial 3x3 DLT homography solely from fitted 2D line
    intersections and fitted ellipse landmarks (no gantry pose model).
    """
    w_pts: List[Tuple[float, float]] = []
    i_pts: List[Tuple[float, float]] = []
    lines = spec["lines"]

    # Analytical intersections between distinct named orthogonal world lines
    for i in range(len(lines)):
        if line_eqs[i] is None:
            continue
        x0_i, y0_i, x1_i, y1_i = lines[i][:4]
        is_x_const_i = abs(x1_i - x0_i) < 1e-6
        is_y_const_i = abs(y1_i - y0_i) < 1e-6
        for j in range(i + 1, len(lines)):
            if line_eqs[j] is None:
                continue
            x0_j, y0_j, x1_j, y1_j = lines[j][:4]
            is_x_const_j = abs(x1_j - x0_j) < 1e-6
            is_y_const_j = abs(y1_j - y0_j) < 1e-6
            if is_x_const_i and is_y_const_j:
                pt = _intersect_2d_lines(line_eqs[i], line_eqs[j])
                if pt is not None and -1000.0 < pt[0] < 2400.0 and -600.0 < pt[1] < 1500.0:
                    w_pts.append((x0_i, y0_j))
                    i_pts.append(pt)
            elif is_y_const_i and is_x_const_j:
                pt = _intersect_2d_lines(line_eqs[i], line_eqs[j])
                if pt is not None and -1000.0 < pt[0] < 2400.0 and -600.0 < pt[1] < 1500.0:
                    w_pts.append((x0_j, y0_i))
                    i_pts.append(pt)

    # For midfield/circle views or partial boxes, add conic intersections / ellipse tangents
    for cx_w, cy_w, R_w, umin, vmin, umax, vmax in spec.get("circles", []):
        roi_m = (
            (chalk_pts[:, 0] >= umin) & (chalk_pts[:, 0] <= umax) &
            (chalk_pts[:, 1] >= vmin) & (chalk_pts[:, 1] <= vmax)
        )
        c_pts = chalk_pts[roi_m].astype(np.float64)
        # Filter out points lying directly on straight lines so ellipse fit is pure
        for li in range(len(lines)):
            _, _, _, _, u0, v0, u1, v1, rad = lines[li]
            dx, dy = u1 - u0, v1 - v0
            L = max(float(np.hypot(dx, dy)), 1e-6)
            perp = np.abs((c_pts[:, 0] - u0) * dy - (c_pts[:, 1] - v0) * dx) / L
            c_pts = c_pts[perp > max(6.0, rad * 0.7)]
        if len(c_pts) >= 25:
            # Projective center-circle landmarks via RANSAC 2D ellipse + pole geometry (angle-agnostic)
            if abs(cx_w) < 1e-3 and abs(cy_w) < 1e-3 and line_eqs[0] is not None:
                rng = np.random.default_rng(42)
                best_Q: Optional[np.ndarray] = None
                best_inliers = -1
                n_c = len(c_pts)
                for _ in range(80):
                    idx = rng.choice(n_c, size=min(14, n_c), replace=False)
                    sub = c_pts[idx].astype(np.float32)
                    try:
                        (ex, ey), (ew, eh), ang_d = cv2.fitEllipse(sub)
                    except cv2.error:
                        continue
                    if ew < 40.0 or eh < 20.0 or ew > 2500.0 or eh > 1500.0:
                        continue
                    th = np.deg2rad(ang_d)
                    ct, st = np.cos(th), np.sin(th)
                    ax_a, ax_b = 0.5 * ew, 0.5 * eh
                    dx_all = c_pts[:, 0] - ex
                    dy_all = c_pts[:, 1] - ey
                    rx = (dx_all * ct + dy_all * st) / max(ax_a, 1e-6)
                    ry = (-dx_all * st + dy_all * ct) / max(ax_b, 1e-6)
                    r_norm = np.sqrt(rx * rx + ry * ry)
                    # Sampson-like pixel residual around radial ellipse boundary
                    denom_g = np.maximum(np.hypot(rx * ct / ax_a - ry * st / ax_b, rx * st / ax_a + ry * ct / ax_b), 1e-6)
                    err_px = np.abs(0.5 * (r_norm * r_norm - 1.0)) / denom_g
                    inl = int((err_px <= 3.5).sum())
                    if inl > best_inliers:
                        best_inliers = inl
                        T = np.array([
                            [ct / ax_a, st / ax_a, -(ex * ct + ey * st) / ax_a],
                            [-st / ax_b, ct / ax_b, -(-ex * st + ey * ct) / ax_b],
                            [0.0, 0.0, 1.0],
                        ], dtype=np.float64)
                        best_Q = T.T @ np.diag([1.0, 1.0, -1.0]) @ T

                if best_Q is not None:
                    def _intersect_line_conic(line_abc: np.ndarray, Q: np.ndarray) -> List[Tuple[float, float]]:
                        la, lb, lc = float(line_abc[0]), float(line_abc[1]), float(line_abc[2])
                        if abs(lb) >= abs(la):
                            p0 = np.array([0.0, -lc / lb, 1.0])
                            dv = np.array([1.0, -la / lb, 0.0])
                        else:
                            p0 = np.array([-lc / la, 0.0, 1.0])
                            dv = np.array([-lb / la, 1.0, 0.0])
                        A = float(dv @ Q @ dv)
                        B = float(2.0 * (p0 @ Q @ dv))
                        C = float(p0 @ Q @ p0)
                        disc = B * B - 4.0 * A * C
                        if disc < 0.0 or abs(A) < 1e-14:
                            return []
                        sq = float(np.sqrt(disc))
                        res = []
                        for t in ((-B - sq) / (2.0 * A), (-B + sq) / (2.0 * A)):
                            pt = p0 + t * dv
                            res.append((float(pt[0]), float(pt[1])))
                        return res

                    pts_x0 = _intersect_line_conic(np.asarray(line_eqs[0], dtype=np.float64), best_Q)
                    if len(pts_x0) == 2:
                        pts_x0.sort(key=lambda p: p[1])
                        pt_top, pt_bot = pts_x0[0], pts_x0[1]  # smaller v (top) is +R_w, larger v (bot) is -R_w
                        l_y0_line = np.array([0.0, 0.0, 1.0], dtype=np.float64)
                        if len(line_eqs) > 1 and line_eqs[1] is not None:
                            l_touch = np.asarray(line_eqs[1], dtype=np.float64)
                            l_tan_top = best_Q @ np.array([pt_top[0], pt_top[1], 1.0], dtype=np.float64)
                            Vx = np.cross(l_tan_top, l_touch)
                            p034 = _intersect_2d_lines(line_eqs[0], line_eqs[1])
                            if p034 is not None and np.linalg.norm(Vx) > 1e-9:
                                # 1D projective map on X=0 from (34.0, +R_w, -R_w) to scalar arc position s
                                u34, v34 = p034
                                ut, vt = pt_top
                                ub, vb = pt_bot
                                du, dv = ub - u34, vb - v34
                                denom_len = max(float(np.hypot(du, dv)), 1e-6)
                                s_34 = 0.0
                                s_t = ((ut - u34) * du + (vt - v34) * dv) / denom_len
                                s_b = ((ub - u34) * du + (vb - v34) * dv) / denom_len
                                ys_1d = [34.0, R_w, -R_w]
                                ss_1d = [s_34, s_t, s_b]
                                A1 = []
                                for yv, sv in zip(ys_1d, ss_1d):
                                    A1.append([yv, 1.0, -sv * yv, -sv])
                                _, _, Vt = np.linalg.svd(np.asarray(A1, dtype=np.float64))
                                h1 = Vt[-1].reshape(2, 2)
                                for y_eval, half_x in ((0.0, R_w), (0.5 * R_w, float(np.sqrt(0.75) * R_w))):
                                    uv_s = h1 @ np.array([y_eval, 1.0], dtype=np.float64)
                                    if abs(uv_s[1]) > 1e-9:
                                        s_val = uv_s[0] / uv_s[1]
                                        p_center_y = np.array([u34 + s_val * du / denom_len, v34 + s_val * dv / denom_len, 1.0])
                                        l_horiz = np.cross(p_center_y, Vx)
                                        h_pts = _intersect_line_conic(l_horiz, best_Q)
                                        if len(h_pts) == 2:
                                            h_pts.sort(key=lambda p: p[0])
                                            w_pts.extend([(-half_x, y_eval), (half_x, y_eval)])
                                            i_pts.extend([h_pts[0], h_pts[1]])
                        w_pts.extend([(0.0, R_w), (0.0, -R_w)])
                        i_pts.extend([pt_top, pt_bot])
            # Penalty Arc landmarks ((+-41.5, 0.0, 9.15) intersecting X = +-36.0) only when line Y-span is narrow
            elif abs(abs(cx_w) - 41.5) < 1e-3 and abs(cy_w) < 1e-3:
                y_span = (max(p[1] for p in w_pts) - min(p[1] for p in w_pts)) if w_pts else 0.0
                if y_span >= 15.0 and len(w_pts) >= 5:
                    continue
                x_box = float(np.sign(cx_w) * 36.0)
                idx_36 = [li for li, ls in enumerate(lines) if abs(ls[0] - x_box) < 1e-6 and abs(ls[2] - x_box) < 1e-6]
                if idx_36 and line_eqs[idx_36[0]] is not None:
                    a36, b36, c36 = line_eqs[idx_36[0]]
                    n36 = max(float(np.hypot(a36, b36)), 1e-6)
                    d36 = np.abs(a36 * c_pts[:, 0] + b36 * c_pts[:, 1] + c36) / n36
                    near36 = c_pts[d36 <= 20.0]
                    if len(near36) >= 6:
                        v_mid36 = 0.5 * (float(near36[:, 1].min()) + float(near36[:, 1].max()))
                        top36 = near36[near36[:, 1] < v_mid36 - 18.0]
                        bot36 = near36[near36[:, 1] > v_mid36 + 18.0]
                        if len(top36) >= 2 and len(bot36) >= 2:
                            y_arc = float(np.sqrt(max(0.0, R_w * R_w - 5.5 * 5.5)))
                            w_pts.extend([(x_box, y_arc), (x_box, -y_arc)])
                            i_pts.extend([
                                (float(np.median(top36[:, 0])), float(np.median(top36[:, 1]))),
                                (float(np.median(bot36[:, 0])), float(np.median(bot36[:, 1]))),
                            ])
                            # Furthest arc apex toward pitch center (~ +-32.35, 0)
                            far_order = np.argsort(-d36)
                            apex_pts = c_pts[far_order[:max(4, min(10, len(c_pts) // 8))]]
                            w_pts.append((float(cx_w - np.sign(cx_w) * R_w), 0.0))
                            i_pts.append((float(np.median(apex_pts[:, 0])), float(np.median(apex_pts[:, 1]))))

    if len(w_pts) >= 4:
        w_arr = np.asarray(w_pts, dtype=np.float64)
        i_arr = np.asarray(i_pts, dtype=np.float64)
        in_view = (i_arr[:, 0] >= -150.0) & (i_arr[:, 0] <= 1430.0) & (i_arr[:, 1] >= -100.0) & (i_arr[:, 1] <= 820.0)
        if int(in_view.sum()) >= 5:
            w_arr, i_arr = w_arr[in_view], i_arr[in_view]
        H0, _ = cv2.findHomography(w_arr, i_arr, method=0)
        if H0 is not None and abs(H0[2, 2]) > 1e-9:
            return H0 / H0[2, 2]
    raise RuntimeError(f"Insufficient analytical landmark constraints ({len(w_pts)}) for initial H_gt")


def fit_independent_gt_homography(bgr_img: np.ndarray, frame_id: str) -> Dict[str, Any]:
    """
    Fits an unconstrained 8-DOF planar homography H_gt strictly from manually verified
    semantic line corridors and pitch circles in pixel space.
    Uses zero pinhole gantry parameters and zero Chamfer distance maps.
    """
    spec = GT_SPECS[frame_id]
    _, chalk_mask, _ = extract_turf_and_chalk_masks(bgr_img)
    ys, xs = np.where(chalk_mask > 0)
    chalk_pts = np.column_stack([xs, ys]).astype(np.float64)

    line_obs: List[Tuple[np.ndarray, np.ndarray]] = []  # ([P0_hom, P1_hom], Nx2 pixel coords)
    line_eqs: List[Optional[Tuple[float, float, float]]] = []
    for X0, Y0, X1, Y1, u0, v0, u1, v1, rad in spec["lines"]:
        pts = _extract_corridor_points(chalk_pts, u0, v0, u1, v1, rad)
        if len(pts) >= 15:
            eq = _fit_2d_line(pts)
            # Prune gross outlier chalk pixels > 4px from Huber line
            a, b, c = eq
            d_line = np.abs(a * pts[:, 0] + b * pts[:, 1] + c) / max(np.hypot(a, b), 1e-6)
            inliers = pts[d_line <= 4.0]
            if len(inliers) >= 12:
                line_eqs.append(_fit_2d_line(inliers))
                line_obs.append((
                    np.array([[X0, Y0, 1.0], [X1, Y1, 1.0]], dtype=np.float64),
                    inliers,
                ))
            else:
                line_eqs.append(None)
        else:
            line_eqs.append(None)

    circle_obs: List[Tuple[float, float, float, np.ndarray]] = []
    for cx_w, cy_w, R_w, umin, vmin, umax, vmax in spec.get("circles", []):
        roi_m = (
            (chalk_pts[:, 0] >= umin) & (chalk_pts[:, 0] <= umax) &
            (chalk_pts[:, 1] >= vmin) & (chalk_pts[:, 1] <= vmax)
        )
        c_pts = chalk_pts[roi_m]
        for X0, Y0, X1, Y1, u0, v0, u1, v1, rad in spec["lines"]:
            dx, dy = u1 - u0, v1 - v0
            L = max(float(np.hypot(dx, dy)), 1e-6)
            perp = np.abs((c_pts[:, 0] - u0) * dy - (c_pts[:, 1] - v0) * dx) / L
            c_pts = c_pts[perp > max(6.0, rad * 0.7)]
        if len(c_pts) >= 20:
            circle_obs.append((cx_w, cy_w, R_w, c_pts))

    H0 = _initial_gt_homography(spec, line_eqs, chalk_pts)
    h_init = H0.ravel()[:8].copy()

    # Subsample up to 250 pts per segment & 300 per circle for clean speed
    sampled_lines = []
    for world_end, pxy in line_obs:
        step = max(1, len(pxy) // 250)
        sampled_lines.append((world_end, pxy[::step]))
    sampled_circles = []
    H0_inv = np.linalg.inv(H0)
    for cx_w, cy_w, R_w, cxy in circle_obs:
        step = max(1, len(cxy) // 400)
        sub = cxy[::step]
        # Pre-prune interior player sock pixels > 18px away from H0-predicted pitch circle
        hom_c = np.column_stack([sub, np.ones(len(sub), dtype=np.float64)])
        w_est = (H0_inv @ hom_c.T).T
        w_xy = w_est[:, :2] / np.where(np.abs(w_est[:, 2:3]) < 1e-6, 1e-6, w_est[:, 2:3])
        vec = w_xy - np.array([cx_w, cy_w], dtype=np.float64)
        on_circ = np.array([cx_w, cy_w], dtype=np.float64) + vec * (R_w / np.maximum(np.linalg.norm(vec, axis=1, keepdims=True), 1e-6))
        uv_c = (H0 @ np.column_stack([on_circ, np.ones(len(on_circ))]).T).T
        uv_c = uv_c[:, :2] / np.where(np.abs(uv_c[:, 2:3]) < 1e-6, 1e-6, uv_c[:, 2:3])
        keep0 = np.linalg.norm(sub - uv_c, axis=1) <= 18.0
        sampled_circles.append((cx_w, cy_w, R_w, sub[keep0] if keep0.sum() >= 20 else sub))

    # Iterative reweighted least squares / outlier pruning against initial H0 so stray players don't pull GT
    def make_res_fn(lines_data, circles_data):
        def res_fn(h8: np.ndarray) -> np.ndarray:
            H = np.append(h8, 1.0).reshape(3, 3)
            res_parts = []
            for ends_w, pxy in lines_data:
                pr = (H @ ends_w.T).T
                # 2D image line in homogeneous coords = p0 x p1
                l_hom = np.cross(pr[0], pr[1])
                norm = max(float(np.hypot(l_hom[0], l_hom[1])), 1e-9)
                d = (l_hom[0] * pxy[:, 0] + l_hom[1] * pxy[:, 1] + l_hom[2]) / norm
                res_parts.append(d)
            if circles_data:
                try:
                    H_inv = np.linalg.inv(H)
                except np.linalg.LinAlgError:
                    return np.full(100, 1e4, dtype=np.float64)
                for cx_w, cy_w, R_w, cxy in circles_data:
                    hom_c = np.column_stack([cxy, np.ones(len(cxy), dtype=np.float64)])
                    w_est = (H_inv @ hom_c.T).T
                    w_xy = w_est[:, :2] / np.where(np.abs(w_est[:, 2:3]) < 1e-6, 1e-6, w_est[:, 2:3])
                    vec = w_xy - np.array([cx_w, cy_w], dtype=np.float64)
                    r_est = np.maximum(np.linalg.norm(vec, axis=1), 1e-6)
                    # Nearest point on true FIFA circle in world meters, reprojected back into pixel residual
                    on_circ = np.array([cx_w, cy_w], dtype=np.float64) + vec * (R_w / r_est[:, None])
                    hc = np.column_stack([on_circ, np.ones(len(on_circ), dtype=np.float64)])
                    uv_c = (H @ hc.T).T
                    uv_c = uv_c[:, :2] / np.where(np.abs(uv_c[:, 2:3]) < 1e-6, 1e-6, uv_c[:, 2:3])
                    diff = cxy - uv_c
                    res_parts.append(np.hypot(diff[:, 0], diff[:, 1]))
            return np.concatenate(res_parts) if res_parts else np.zeros(8, dtype=np.float64)
        return res_fn

    opt1 = least_squares(make_res_fn(sampled_lines, sampled_circles), h_init, loss="soft_l1", f_scale=4.0, max_nfev=300)
    H_gt_rough = np.append(opt1.x, 1.0).reshape(3, 3)

    # Filter circle/line chalk outliers > 5px after stage 1 (e.g., boots/players touching circle)
    pruned_circles = []
    H_inv_r = np.linalg.inv(H_gt_rough)
    for cx_w, cy_w, R_w, cxy in sampled_circles:
        hom_c = np.column_stack([cxy, np.ones(len(cxy))])
        w_est = (H_inv_r @ hom_c.T).T
        w_xy = w_est[:, :2] / w_est[:, 2:3]
        vec = w_xy - np.array([cx_w, cy_w])
        on_circ = np.array([cx_w, cy_w]) + vec * (R_w / np.maximum(np.linalg.norm(vec, axis=1, keepdims=True), 1e-6))
        uv_c = (H_gt_rough @ np.column_stack([on_circ, np.ones(len(on_circ))]).T).T
        uv_c = uv_c[:, :2] / uv_c[:, 2:3]
        keep = np.linalg.norm(cxy - uv_c, axis=1) <= 5.5
        pruned_circles.append((cx_w, cy_w, R_w, cxy[keep] if keep.sum() >= 15 else cxy))

    opt2 = least_squares(make_res_fn(sampled_lines, pruned_circles), opt1.x, loss="soft_l1", f_scale=1.5, max_nfev=400)
    H_gt = np.append(opt2.x, 1.0).reshape(3, 3)
    # Enforce positive depth orientation on visible world line endpoints so project_world_to_image behaves cleanly
    if sampled_lines:
        test_pts = np.vstack([ends_w for ends_w, _ in sampled_lines])
        if float(np.median((H_gt @ test_pts.T)[2])) < 0.0:
            H_gt = -H_gt
    all_res = np.abs(make_res_fn(sampled_lines, pruned_circles)(opt2.x))
    return {
        "H_gt": H_gt,
        "gt_median_fit_px": round(float(np.median(all_res)), 3),
        "gt_mean_fit_px": round(float(np.mean(all_res)), 3),
        "gt_inlier_3px_pct": round(float(np.mean(all_res <= 3.0) * 100.0), 2),
        "gt_sample_count": int(len(all_res)),
    }


def calibrate_hough_baseline(bgr_img: np.ndarray) -> Dict[str, Any]:
    """
    Classical Hough Line-Pair + RANSAC Homography Baseline (non-chamfer, fixed-line baseline).
    Detects long white Hough segments, clusters horizontal & vertical pitch lines, maps the
    strongest visible orthogonal line pair + scale heuristics to canonical penalty/halfway lines.
    """
    t0 = time.perf_counter()
    h_img, w_img = bgr_img.shape[:2]
    _, chalk_mask, _ = extract_turf_and_chalk_masks(bgr_img)
    dist = cv2.distanceTransform(255 - chalk_mask, cv2.DIST_L2, 5)
    raw = cv2.HoughLinesP(chalk_mask, 1, np.pi / 180.0, 45, minLineLength=80, maxLineGap=18)

    horiz: List[Tuple[float, float, float, float, float]] = []
    vert: List[Tuple[float, float, float, float, float]] = []
    if raw is not None:
        for x1, y1, x2, y2 in raw.reshape(-1, 4):
            dx, dy = float(x2 - x1), float(y2 - y1)
            length = float(np.hypot(dx, dy))
            angle = abs(float(np.rad2deg(np.arctan2(dy, dx))))
            if angle < 20.0 or angle > 160.0:
                horiz.append((x1, y1, x2, y2, length))
            elif 40.0 < angle < 140.0:
                vert.append((x1, y1, x2, y2, length))

    horiz.sort(key=lambda s: -s[4])
    vert.sort(key=lambda s: -s[4])

    # Fallback wide broadcast prior if fewer than 2 orthogonal segments exist
    best_H = np.array([
        [28.0, 0.0, w_img / 2.0],
        [0.0, 12.0, h_img * 0.48],
        [0.0, 0.004, 1.0],
    ], dtype=np.float64)
    best_sc = float("inf")

    if horiz and vert:
        # Pair longest nearly-horizontal touchline/box-top with strongest vertical line
        # against canonical 4-way pitch interpretations (left box, right box, midfield)
        x1, y1, x2, y2, _ = horiz[0]
        if x1 > x2:
            x1, y1, x2, y2 = x2, y2, x1, y1
        vx1, vy1, vx2, vy2, _ = vert[0]
        if vy1 > vy2:
            vx1, vy1, vx2, vy2 = vx2, vy2, vx1, vy1

        def line_abc(ax, ay, bx, by):
            return (by - ay, ax - bx, bx * ay - ax * by)

        def inter(l1, l2):
            det = l1[0] * l2[1] - l2[0] * l1[1]
            if abs(det) < 1e-6:
                return None
            return ((l1[1] * l2[2] - l2[1] * l1[2]) / det, (l2[0] * l1[2] - l1[0] * l2[2]) / det)

        lh = line_abc(x1, y1, x2, y2)
        lv = line_abc(vx1, vy1, vx2, vy2)
        origin = inter(lh, lv)
        if origin is not None and -400 < origin[0] < w_img + 400 and -300 < origin[1] < h_img + 300:
            # Construct local projective basis around longest horizontal & vertical lines
            im_quad = np.array([
                origin,
                (x2, y2),
                (vx2, vy2),
                (x2 + (vx2 - origin[0]) * 1.15, y2 + (vy2 - origin[1]) * 0.95),
            ], dtype=np.float64)
            candidate_worlds = [
                np.array([[36.0, 20.16], [52.5, 20.16], [36.0, -20.16], [52.5, -20.16]]),
                np.array([[-36.0, 20.16], [-52.5, 20.16], [-36.0, -20.16], [-52.5, -20.16]]),
                np.array([[0.0, 34.0], [20.0, 34.0], [0.0, -10.0], [20.0, -10.0]]),
                np.array([[36.0, 34.0], [52.5, 34.0], [36.0, 0.0], [52.5, 0.0]]),
            ]
            for w_quad in candidate_worlds:
                H_cand, _ = cv2.findHomography(w_quad, im_quad, method=0)
                if H_cand is None or abs(H_cand[2, 2]) < 1e-9:
                    continue
                H_cand /= H_cand[2, 2]
                uv, val = project_world_to_image(H_cand, CANONICAL_PITCH_POINTS)
                vis = val & (uv[:, 0] >= 5) & (uv[:, 0] < w_img - 5) & (uv[:, 1] >= 85) & (uv[:, 1] < h_img - 5)
                if vis.sum() < 40:
                    continue
                d_vals = dist[uv[vis, 1].astype(int), uv[vis, 0].astype(int)]
                sc = float(np.minimum(d_vals, 25.0).mean()) - 0.5 * float((d_vals <= 5.0).mean())
                if sc < best_sc:
                    best_sc = sc
                    best_H = H_cand

    return {
        "H": best_H,
        "elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 1),
    }


def _save_contact_sheet(images: List[np.ndarray], out_path: Path) -> None:
    tiles = [cv2.resize(im, (640, 360)) for im in images]
    while len(tiles) < 4:
        tiles.append(np.zeros((360, 640, 3), dtype=np.uint8))
    top = np.hstack(tiles[:2])
    bot = np.hstack(tiles[2:4])
    cv2.imwrite(str(out_path), np.vstack([top, bot]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Track I Camera Calibration across Game 0 Half 1")
    parser.add_argument("--save-sheets", type=str, default="", help="Optional dir to save GT & pred visual sheets")
    args = parser.parse_args()

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    chunk_paths = {
        c.get("chunk_index", c.get("index", idx)): ROOT / c["relative_path"]
        for idx, c in enumerate(manifest["chunks"])
    }

    per_frame: List[Dict[str, Any]] = []
    gt_imgs, pred_imgs = [], []

    print(f"[Track I] Evaluating {len(FRAME_SCHEDULE)} real Game 0 Half 1 tactical frames...")
    for item in FRAME_SCHEDULE:
        fid = item["frame_id"]
        vpath = chunk_paths[item["chunk"]]
        bgr = decode_frame_at(vpath, item["rel_sec"])

        # 1. Independent non-circular unconstrained GT homography
        gt_res = fit_independent_gt_homography(bgr, fid)
        H_gt = gt_res["H_gt"]

        # 2. Classical Hough Baseline
        base_res = calibrate_hough_baseline(bgr)
        base_iou = compute_pitch_part_iou(base_res["H"], H_gt)
        base_rep = compute_reprojection_accuracy(base_res["H"], H_gt)

        # 3. Track I Multi-Start Soft-Inlier Chamfer Pinhole Calibration (Zero hints)
        pred_res = estimate_pitch_homography(bgr)
        pred_iou = compute_pitch_part_iou(pred_res["H"], H_gt)
        pred_rep = compute_reprojection_accuracy(pred_res["H"], H_gt)

        row = {
            "frame_id": fid,
            "chunk": item["chunk"],
            "rel_sec": item["rel_sec"],
            "view": item["view"],
            "gt_chalk_fit": {
                "median_px": gt_res["gt_median_fit_px"],
                "mean_px": gt_res["gt_mean_fit_px"],
                "inlier_3px_pct": gt_res["gt_inlier_3px_pct"],
            },
            "hough_baseline": {
                "iou_part": base_iou["mean_iou_part"],
                "ac_5px_pct": round(100.0 * base_rep["ac_5px"], 2),
                "ac_10px_pct": round(100.0 * base_rep["ac_10px"], 2),
                "mean_err_px": base_rep["mean_err_px"],
                "elapsed_ms": base_res["elapsed_ms"],
            },
            "track_i_chamfer_pinhole": {
                "iou_part": pred_iou["mean_iou_part"],
                "ac_5px_pct": round(100.0 * pred_rep["ac_5px"], 2),
                "ac_10px_pct": round(100.0 * pred_rep["ac_10px"], 2),
                "mean_err_px": pred_rep["mean_err_px"],
                "median_err_px": pred_rep["median_err_px"],
                "focal_px": round(pred_res["focal_px"], 1),
                "camera_center_3d": [round(float(x), 2) for x in pred_res["camera_center_3d"]],
                "chamfer_median_px": pred_res["chamfer_median_px"],
                "chamfer_inlier_5px_pct": pred_res["chamfer_inlier_5px_pct"],
                "elapsed_ms": pred_res["elapsed_ms"],
            },
        }
        per_frame.append(row)
        print(
            f"  {fid:8s} ({item['view']:28s}) | GT fit={gt_res['gt_median_fit_px']:.2f}px | "
            f"Hough IoU={base_iou['mean_iou_part']:.3f} AC@5={100.0 * base_rep['ac_5px']:5.1f}% | "
            f"Track-I IoU={pred_iou['mean_iou_part']:.3f} AC@5={100.0 * pred_rep['ac_5px']:5.1f}% "
            f"AC@10={100.0 * pred_rep['ac_10px']:5.1f}% med={pred_rep['median_err_px']:4.2f}px ({pred_res['elapsed_ms']:.0f}ms)"
        )

        if args.save_sheets:
            uv_gt, val_gt = project_world_to_image(H_gt, CANONICAL_PITCH_POINTS)
            vis_gt = bgr.copy()
            in_gt = val_gt & (uv_gt[:, 0] >= 0) & (uv_gt[:, 0] < IMAGE_WIDTH) & (uv_gt[:, 1] >= 0) & (uv_gt[:, 1] < IMAGE_HEIGHT)
            for u, v in uv_gt[in_gt]:
                cv2.circle(vis_gt, (int(u), int(v)), 2, (0, 255, 0), -1)
            cv2.putText(vis_gt, f"{fid} GT fit={gt_res['gt_median_fit_px']:.2f}px", (18, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 0), 2)
            gt_imgs.append(vis_gt)

            uv_pr, val_pr = project_world_to_image(pred_res["H"], CANONICAL_PITCH_POINTS)
            vis_pr = bgr.copy()
            in_pr = val_pr & (uv_pr[:, 0] >= 0) & (uv_pr[:, 0] < IMAGE_WIDTH) & (uv_pr[:, 1] >= 0) & (uv_pr[:, 1] < IMAGE_HEIGHT)
            for u, v in uv_pr[in_pr]:
                cv2.circle(vis_pr, (int(u), int(v)), 2, (0, 255, 255), -1)
            cv2.putText(vis_pr, f"{fid} IoU={pred_iou['mean_iou_part']:.3f} AC@5={100.0 * pred_rep['ac_5px']:.1f}%", (18, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 2)
            pred_imgs.append(vis_pr)

    if args.save_sheets:
        sdir = Path(args.save_sheets)
        sdir.mkdir(parents=True, exist_ok=True)
        for p_idx in range(0, len(gt_imgs), 4):
            _save_contact_sheet(gt_imgs[p_idx:p_idx + 4], sdir / f"gt_verified_sheet_{p_idx // 4}.jpg")
            _save_contact_sheet(pred_imgs[p_idx:p_idx + 4], sdir / f"pred_verified_sheet_{p_idx // 4}.jpg")

    def _agg(key: str) -> Dict[str, Any]:
        return {
            "mean_iou_part": round(float(np.mean([r[key]["iou_part"] for r in per_frame])), 4),
            "mean_ac_5px_pct": round(float(np.mean([r[key]["ac_5px_pct"] for r in per_frame])), 2),
            "mean_ac_10px_pct": round(float(np.mean([r[key]["ac_10px_pct"] for r in per_frame])), 2),
            "mean_reproj_err_px": round(float(np.mean([r[key]["mean_err_px"] for r in per_frame])), 2),
            "mean_latency_ms": round(float(np.mean([r[key]["elapsed_ms"] for r in per_frame])), 1),
        }

    base_summary = _agg("hough_baseline")
    pred_summary = _agg("track_i_chamfer_pinhole")
    pred_summary["median_reproj_err_px"] = round(float(np.mean([r["track_i_chamfer_pinhole"]["median_err_px"] for r in per_frame])), 2)
    pred_summary["mean_gt_chalk_fit_px"] = round(float(np.mean([r["gt_chalk_fit"]["median_px"] for r in per_frame])), 3)

    summary = {
        "track": "Track I - Camera Calibration & Pitch Line Registration",
        "dataset": "SoccerNet Game 0 Half 1 (Man Utd vs Arsenal, 12 real wide/offside tactical 1280x720 keyframes)",
        "frames_evaluated": len(per_frame),
        "ground_truth_method": "Independent unconstrained 8-DOF planar homography fit to semantic chalk line corridors & pitch circles",
        "baseline_hough_ransac": base_summary,
        "track_i_chamfer_pinhole": pred_summary,
        "per_frame_results": per_frame,
    }

    OUT_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("-" * 92)
    print(f"[Track I] Hough Baseline : IoU_part={base_summary['mean_iou_part']:.4f} | AC@5px={base_summary['mean_ac_5px_pct']:.2f}% | AC@10px={base_summary['mean_ac_10px_pct']:.2f}% | MeanErr={base_summary['mean_reproj_err_px']:.2f}px")
    print(f"[Track I] Chamfer Pinhole: IoU_part={pred_summary['mean_iou_part']:.4f} | AC@5px={pred_summary['mean_ac_5px_pct']:.2f}% | AC@10px={pred_summary['mean_ac_10px_pct']:.2f}% | MeanErr={pred_summary['mean_reproj_err_px']:.2f}px | Latency={pred_summary['mean_latency_ms']:.1f}ms")
    print(f"[Track I] Saved summary -> {OUT_JSON_PATH}")


if __name__ == "__main__":
    main()
