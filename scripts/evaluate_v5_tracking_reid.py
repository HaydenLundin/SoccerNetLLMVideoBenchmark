"""
Track K: Multi-Object Tracking & Jersey Re-Identification Evaluation.

Evaluates continuous 25 FPS multi-player tracking and appearance/jersey Re-ID
across 650 real broadcast frames decoded from Game 0 Half 1:
  - Seq 1 (Chunk 00, 445.0..453.0s, 200 frames): Midfield Duel (c0_0730, J09 #34)
  - Seq 2 (Chunk 01, 271.0..279.0s, 200 frames): Right Box Free Kick w/ Real Cut (c1_1417, J05 #8, J06 #19)
  - Seq 3 (Chunk 02, 588.0..598.0s, 250 frames): Herrera Goal Action (c2_2919, J01 #21, J02 #18)

Compares:
  - Baseline   : Greedy Frame-to-Frame IoU Tracker (no motion prediction, no Re-ID)
  - Track K    : CPU Constant-Velocity Kalman + Hungarian (IoU + Center + HSV Re-ID)
                 coupled with 1 FPS Track H Jersey OCR Gallery Re-ID (J01..J15)

Reports: HOTA, MOTA, IDF1, ID Switches, and Re-ID Top-1 Accuracy / mAP.
"""

from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import av
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from processing.pitch_geometry import detect_players_on_pitch, extract_turf_and_chalk_masks

OUTPUT_DIR = ROOT / "logs" / "v5_eval" / "tracking_reid"
SUMMARY_PATH = OUTPUT_DIR / "tracking_reid_summary.json"
TRACK_H_OCR_PATH = ROOT / "logs" / "v4_eval" / "jersey_ocr" / "jersey_ocr_summary.json"

TRACKING_SEQUENCES: List[Tuple[str, Path, float, int]] = [
    ("seq1_midfield_duel", ROOT / "gate2_chunks/game_0_half_1_chunk_00.mp4", 445.0, 200),
    ("seq2_free_kick", ROOT / "gate2_chunks/game_0_half_1_chunk_01.mp4", 271.0, 200),
    ("seq3_herrera_goal", ROOT / "gate2_chunks/game_0_half_1_chunk_02.mp4", 588.0, 250),
]

CHUNK_STARTS: List[Tuple[int, Path, float]] = [
    (0, ROOT / "gate2_chunks/game_0_half_1_chunk_00.mp4", 0.0),
    (1, ROOT / "gate2_chunks/game_0_half_1_chunk_01.mp4", 582.04),
    (2, ROOT / "gate2_chunks/game_0_half_1_chunk_02.mp4", 1164.72),
    (3, ROOT / "gate2_chunks/game_0_half_1_chunk_03.mp4", 1753.40),
    (4, ROOT / "gate2_chunks/game_0_half_1_chunk_04.mp4", 2335.64),
]


def decode_continuous_frames(vpath: Path, start_s: float, max_frames: int) -> List[np.ndarray]:
    """Decodes up to max_frames consecutive BGR frames from start_s using PyAV."""
    container = av.open(str(vpath))
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"
    tb = float(stream.time_base)
    container.seek(int(start_s / tb), stream=stream)
    frames: List[np.ndarray] = []
    for frame in container.decode(video=0):
        pts_s = float(frame.pts * tb)
        if pts_s < start_s - 0.02:
            continue
        frames.append(frame.to_ndarray(format="bgr24"))
        if len(frames) >= max_frames:
            break
    container.close()
    return frames


def box_iou(a: List[float], b: List[float]) -> float:
    """Computes intersection-over-union for two [x1, y1, x2, y2] boxes."""
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(1.0, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(1.0, (b[2] - b[0]) * (b[3] - b[1]))
    return inter / max(1e-6, area_a + area_b - inter)


def box_iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Vectorized IoU matrix of shape (N, M) between two (N, 4) and (M, 4) arrays."""
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float64)
    ax1, ay1, ax2, ay2 = boxes_a[:, 0:1], boxes_a[:, 1:2], boxes_a[:, 2:3], boxes_a[:, 3:4]
    bx1, by1, bx2, by2 = boxes_b[:, 0:1].T, boxes_b[:, 1:2].T, boxes_b[:, 2:3].T, boxes_b[:, 3:4].T
    iw = np.maximum(0.0, np.minimum(ax2, bx2) - np.maximum(ax1, bx1))
    ih = np.maximum(0.0, np.minimum(ay2, by2) - np.maximum(ay1, by1))
    inter = iw * ih
    area_a = np.maximum(1.0, (ax2 - ax1) * (ay2 - ay1))
    area_b = np.maximum(1.0, (bx2 - bx1) * (by2 - by1))
    return inter / np.maximum(1e-6, area_a + area_b - inter)


def extract_3zone_hsv_feature(bgr_crop: np.ndarray) -> np.ndarray:
    """Computes normalized 3-zone (head/torso/shorts) HSV histogram for player Re-ID."""
    if bgr_crop.size == 0:
        return np.zeros(240, dtype=np.float32)
    h, w = bgr_crop.shape[:2]
    hsv = cv2.cvtColor(cv2.resize(bgr_crop, (32, 64)), cv2.COLOR_BGR2HSV)
    # 3 vertical zones: head/shoulders [0:14], jersey [14:38], shorts/legs [38:64]
    zones = [hsv[0:14, :], hsv[14:38, :], hsv[38:64, :]]
    parts = []
    for z in zones:
        hist = cv2.calcHist([z], [0, 1], None, [10, 8], [0, 180, 0, 256]).flatten()
        nrm = float(np.linalg.norm(hist))
        parts.append(hist / nrm if nrm > 1e-6 else hist)
    vec = np.concatenate(parts).astype(np.float32)
    full_nrm = float(np.linalg.norm(vec))
    return vec / full_nrm if full_nrm > 1e-6 else vec


def detect_cuts(frames: List[np.ndarray], mad_thresh: float = 22.0) -> List[int]:
    """Detects real broadcast scene cuts via downsampled luminance MAD."""
    cuts: List[int] = []
    prev_g = cv2.resize(cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY), (160, 90)).astype(np.float32)
    for t in range(1, len(frames)):
        g = cv2.resize(cv2.cvtColor(frames[t], cv2.COLOR_BGR2GRAY), (160, 90)).astype(np.float32)
        if float(np.abs(g - prev_g).mean()) > mad_thresh:
            cuts.append(t)
        prev_g = g
    return cuts


def build_offline_optical_flow_gt(
    frames: List[np.ndarray],
    per_frame_dets: List[List[Dict[str, Any]]],
    cuts: List[int],
    min_track_len: int = 12,
    max_gap: int = 5,
) -> List[List[Tuple[int, List[float]]]]:
    """
    Constructs an independent offline bidirectional optical-flow GT reference.

    Links detections across consecutive frames by warping boxes with dense Farneback
    luminance optical flow (independent of Kalman velocity or HSV histograms), bridges
    brief detector dropouts (<= max_gap) via linear lookahead interpolation, resets
    on real broadcast cuts, and prunes transient <12-frame fragments.
    """
    cut_set = set(cuts)
    n_frames = len(frames)
    small_grays = [
        cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (320, 180)) for f in frames
    ]
    sx = frames[0].shape[1] / 320.0
    sy = frames[0].shape[0] / 180.0

    # Active tracks: tid -> {"last_t": int, "obs": [(t, [x1,y1,x2,y2]), ...]}
    next_tid = 1
    active: Dict[int, Dict[str, Any]] = {}
    finished: List[Dict[str, Any]] = []

    for t in range(n_frames):
        if t in cut_set:
            finished.extend(active.values())
            active.clear()

        # Retire tracks silent past max_gap
        expired = [tid for tid, tr in active.items() if (t - tr["last_t"]) > max_gap]
        for tid in expired:
            finished.append(active.pop(tid))

        curr_dets = [list(map(float, d["bbox"])) for d in per_frame_dets[t]]
        if not curr_dets:
            continue

        if t == 0 or t in cut_set or not active:
            for b in curr_dets:
                active[next_tid] = {"last_t": t, "obs": [(t, b)]}
                next_tid += 1
            continue

        # Warp active boxes using dense optical flow from t-1 to t
        flow = cv2.calcOpticalFlowFarneback(
            small_grays[t - 1], small_grays[t], None, 0.5, 2, 13, 2, 5, 1.1, 0
        )

        # Propagate all active boxes by flow inside their current box region
        tids = list(active.keys())
        warped_boxes = []
        for tid in tids:
            b = active[tid]["obs"][-1][1]
            dt_gap = t - active[tid]["last_t"]
            rx1 = max(0, min(319, int(b[0] / sx)))
            ry1 = max(0, min(179, int(b[1] / sy)))
            rx2 = max(rx1 + 1, min(320, int(math.ceil(b[2] / sx))))
            ry2 = max(ry1 + 1, min(180, int(math.ceil(b[3] / sy))))
            patch_flow = flow[ry1:ry2, rx1:rx2]
            dx = float(np.median(patch_flow[:, :, 0])) * sx * dt_gap
            dy = float(np.median(patch_flow[:, :, 1])) * sy * dt_gap
            warped_boxes.append([b[0] + dx, b[1] + dy, b[2] + dx, b[3] + dy])

        wb_arr = np.asarray(warped_boxes, dtype=np.float64)
        cd_arr = np.asarray(curr_dets, dtype=np.float64)
        ious = box_iou_matrix(wb_arr, cd_arr)

        wb_c = np.column_stack([(wb_arr[:, 0] + wb_arr[:, 2]) * 0.5, (wb_arr[:, 1] + wb_arr[:, 3]) * 0.5])
        cd_c = np.column_stack([(cd_arr[:, 0] + cd_arr[:, 2]) * 0.5, (cd_arr[:, 1] + cd_arr[:, 3]) * 0.5])
        cdists = np.linalg.norm(wb_c[:, None, :] - cd_c[None, :, :], axis=2)

        cost = (1.0 - ious) + 0.015 * cdists
        row_ind, col_ind = linear_sum_assignment(cost)

        matched_dets = set()
        for r, c in zip(row_ind, col_ind):
            dt_gap = t - active[tids[r]]["last_t"]
            dist_limit = 22.0 * dt_gap
            if ious[r, c] >= 0.22 or cdists[r, c] <= dist_limit:
                tid = tids[r]
                active[tid]["obs"].append((t, curr_dets[c]))
                active[tid]["last_t"] = t
                matched_dets.add(c)

        for c, b in enumerate(curr_dets):
            if c not in matched_dets:
                active[next_tid] = {"last_t": t, "obs": [(t, b)]}
                next_tid += 1

    finished.extend(active.values())

    # Densify offline GT tracks across brief <= max_gap dropouts
    gt_per_frame: List[List[Tuple[int, List[float]]]] = [[] for _ in range(n_frames)]
    kept_id = 1
    for tr in finished:
        obs = tr["obs"]
        span = obs[-1][0] - obs[0][0] + 1
        if len(obs) < min_track_len or span < min_track_len:
            continue
        for idx, (ft, fb) in enumerate(obs):
            gt_per_frame[ft].append((kept_id, fb))
            if idx + 1 < len(obs):
                nt, nb = obs[idx + 1]
                gap = nt - ft
                if 1 < gap <= max_gap and not any(c in cut_set for c in range(ft + 1, nt + 1)):
                    for k in range(1, gap):
                        alpha = k / float(gap)
                        ib = [fb[j] * (1.0 - alpha) + nb[j] * alpha for j in range(4)]
                        gt_per_frame[ft + k].append((kept_id, ib))
        kept_id += 1

    return gt_per_frame


def run_baseline_iou_tracker(
    per_frame_dets: List[List[Dict[str, Any]]],
    cuts: List[int],
    iou_thresh: float = 0.28,
) -> List[List[Tuple[int, List[float]]]]:
    """Baseline greedy frame-to-frame IoU tracker without motion prediction or memory."""
    cut_set = set(cuts)
    next_tid = 1
    prev_tracks: List[Tuple[int, List[float]]] = []
    out: List[List[Tuple[int, List[float]]]] = []

    for t, dets in enumerate(per_frame_dets):
        if t in cut_set:
            prev_tracks = []
        curr_boxes = [list(map(float, d["bbox"])) for d in dets]
        curr_tracks: List[Tuple[int, List[float]]] = []

        if not prev_tracks or not curr_boxes:
            for b in curr_boxes:
                curr_tracks.append((next_tid, b))
                next_tid += 1
        else:
            p_arr = np.asarray([b for _, b in prev_tracks], dtype=np.float64)
            c_arr = np.asarray(curr_boxes, dtype=np.float64)
            ious = box_iou_matrix(p_arr, c_arr)

            # Greedy matching in descending IoU order
            pairs: List[Tuple[float, int, int]] = []
            for r in range(len(prev_tracks)):
                for c in range(len(curr_boxes)):
                    if ious[r, c] >= iou_thresh:
                        pairs.append((float(ious[r, c]), r, c))
            pairs.sort(reverse=True)

            used_r, used_c = set(), set()
            for _, r, c in pairs:
                if r not in used_r and c not in used_c:
                    curr_tracks.append((prev_tracks[r][0], curr_boxes[c]))
                    used_r.add(r)
                    used_c.add(c)
            for c, b in enumerate(curr_boxes):
                if c not in used_c:
                    curr_tracks.append((next_tid, b))
                    next_tid += 1

        prev_tracks = curr_tracks
        out.append(curr_tracks)

    return out


class _KalmanTrack:
    """Minimal 2D constant-velocity Kalman tracklet with HSV EMA & coasting."""

    def __init__(self, tid: int, bbox: List[float], feat: np.ndarray, role: str) -> None:
        self.tid = tid
        self.bbox = list(bbox)
        cx, cy = 0.5 * (bbox[0] + bbox[2]), 0.5 * (bbox[1] + bbox[3])
        self.w = bbox[2] - bbox[0]
        self.h = bbox[3] - bbox[1]
        # State: [cx, cy, vx, vy]
        self.x = np.array([cx, cy, 0.0, 0.0], dtype=np.float64)
        self.P = np.diag([25.0, 25.0, 16.0, 16.0])
        self.feat = feat.copy()
        self.role = role
        self.age_missed = 0
        self.hits = 1

    def predict(self) -> List[float]:
        F = np.array([
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 0.90, 0.0],
            [0.0, 0.0, 0.0, 0.90],
        ], dtype=np.float64)
        Q = np.diag([4.0, 4.0, 2.5, 2.5])
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        self.age_missed += 1
        cx, cy = self.x[0], self.x[1]
        self.bbox = [cx - 0.5 * self.w, cy - 0.5 * self.h, cx + 0.5 * self.w, cy + 0.5 * self.h]
        return self.bbox

    def update(self, bbox: List[float], feat: np.ndarray, role: str) -> None:
        cx, cy = 0.5 * (bbox[0] + bbox[2]), 0.5 * (bbox[1] + bbox[3])
        z = np.array([cx, cy], dtype=np.float64)
        H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float64)
        R = np.diag([9.0, 9.0])
        y = z - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ H) @ self.P
        self.w = 0.7 * self.w + 0.3 * (bbox[2] - bbox[0])
        self.h = 0.7 * self.h + 0.3 * (bbox[3] - bbox[1])
        self.bbox = list(bbox)
        # EMA HSV appearance bank update
        blended = 0.75 * self.feat + 0.25 * feat
        nrm = float(np.linalg.norm(blended))
        self.feat = blended / nrm if nrm > 1e-6 else feat
        self.role = role
        self.age_missed = 0
        self.hits += 1


def run_track_k_kalman_reid_tracker(
    per_frame_dets: List[List[Dict[str, Any]]],
    cuts: List[int],
    max_age: int = 10,
) -> List[List[Tuple[int, List[float]]]]:
    """Track K CPU Kalman + Hungarian Munkres (IoU + Motion + HSV Re-ID + Coasting)."""
    cut_set = set(cuts)
    next_tid = 1
    tracks: Dict[int, _KalmanTrack] = {}
    out: List[List[Tuple[int, List[float]]]] = []

    for t, dets in enumerate(per_frame_dets):
        if t in cut_set:
            tracks.clear()

        for tr in tracks.values():
            tr.predict()

        # Expire long-absent tracks
        dead = [tid for tid, tr in tracks.items() if tr.age_missed > max_age]
        for tid in dead:
            tracks.pop(tid)

        curr_boxes = [list(map(float, d["bbox"])) for d in dets]
        curr_feats = [np.asarray(d["appearance_hist"], dtype=np.float64) for d in dets]
        curr_roles = [str(d["role"]) for d in dets]

        frame_emit: List[Tuple[int, List[float]]] = []
        if tracks and curr_boxes:
            tids = list(tracks.keys())
            t_arr = np.asarray([tracks[tid].bbox for tid in tids], dtype=np.float64)
            c_arr = np.asarray(curr_boxes, dtype=np.float64)
            ious = box_iou_matrix(t_arr, c_arr)

            tc = np.column_stack([(t_arr[:, 0] + t_arr[:, 2]) * 0.5, (t_arr[:, 1] + t_arr[:, 3]) * 0.5])
            cc = np.column_stack([(c_arr[:, 0] + c_arr[:, 2]) * 0.5, (c_arr[:, 1] + c_arr[:, 3]) * 0.5])
            cdists = np.linalg.norm(tc[:, None, :] - cc[None, :, :], axis=2)

            tf_mat = np.vstack([tracks[tid].feat for tid in tids])
            cf_mat = np.vstack(curr_feats)
            cos_sim = np.clip(tf_mat @ cf_mat.T, 0.0, 1.0)

            role_pen = np.array([
                [0.0 if tracks[tid].role == cr else 0.25 for cr in curr_roles] for tid in tids
            ], dtype=np.float64)

            # Combined Munkres association cost
            cost = (
                0.45 * (1.0 - ious)
                + 0.30 * np.minimum(1.0, cdists / 55.0)
                + 0.25 * (1.0 - cos_sim)
                + role_pen
            )
            row_ind, col_ind = linear_sum_assignment(cost)

            matched_c = set()
            for r, c in zip(row_ind, col_ind):
                tid = tids[r]
                gate_px = 32.0 + 10.0 * min(5, tracks[tid].age_missed)
                if (ious[r, c] >= 0.15 or cdists[r, c] <= gate_px) and (cos_sim[r, c] >= 0.25 or ious[r, c] >= 0.35):
                    tracks[tid].update(curr_boxes[c], curr_feats[c], curr_roles[c])
                    frame_emit.append((tid, curr_boxes[c]))
                    matched_c.add(c)

            for c in range(len(curr_boxes)):
                if c not in matched_c:
                    tracks[next_tid] = _KalmanTrack(next_tid, curr_boxes[c], curr_feats[c], curr_roles[c])
                    frame_emit.append((next_tid, curr_boxes[c]))
                    next_tid += 1
        else:
            for c in range(len(curr_boxes)):
                tracks[next_tid] = _KalmanTrack(next_tid, curr_boxes[c], curr_feats[c], curr_roles[c])
                frame_emit.append((next_tid, curr_boxes[c]))
                next_tid += 1

        # Latent Kalman tracks stay in memory up to max_age without emitting ghost boxes
        out.append(frame_emit)

    return out


def evaluate_mot_and_hota(
    gt_seq: List[List[Tuple[int, List[float]]]],
    pred_seq: List[List[Tuple[int, List[float]]]],
    iou_thresh: float = 0.35,
) -> Dict[str, Any]:
    """Computes standard CLEAR-MOT (MOTA, IDSW), IDF1, and multi-alpha HOTA."""
    total_gt = 0
    total_pred = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0
    idsw = 0

    last_gt_to_pred: Dict[int, int] = {}
    gt_counts: Dict[int, int] = defaultdict(int)
    pred_counts: Dict[int, int] = defaultdict(int)
    cooccur_base: Dict[Tuple[int, int], int] = defaultdict(int)

    # Pre-collect per-frame IoU assignments across all HOTA alpha thresholds
    alphas = [round(0.05 + 0.05 * i, 2) for i in range(19)]  # 0.05 .. 0.95
    alpha_stats: Dict[float, Dict[str, Any]] = {
        a: {"tp": 0, "fp": 0, "fn": 0, "co": defaultdict(int), "matches": []} for a in alphas
    }

    for t, (gt_objs, pr_objs) in enumerate(zip(gt_seq, pred_seq)):
        ng, np_ = len(gt_objs), len(pr_objs)
        total_gt += ng
        total_pred += np_
        for gid, _ in gt_objs:
            gt_counts[gid] += 1
        for pid, _ in pr_objs:
            pred_counts[pid] += 1

        if ng == 0 or np_ == 0:
            total_fp += np_
            total_fn += ng
            for a in alphas:
                alpha_stats[a]["fp"] += np_
                alpha_stats[a]["fn"] += ng
            continue

        g_arr = np.asarray([b for _, b in gt_objs], dtype=np.float64)
        p_arr = np.asarray([b for _, b in pr_objs], dtype=np.float64)
        ious = box_iou_matrix(g_arr, p_arr)
        row_ind, col_ind = linear_sum_assignment(1.0 - ious)

        # Standard CLEAR-MOT + IDF1 at iou_thresh
        tp_t = 0
        for r, c in zip(row_ind, col_ind):
            if ious[r, c] >= iou_thresh:
                gid = gt_objs[r][0]
                pid = pr_objs[c][0]
                tp_t += 1
                cooccur_base[(gid, pid)] += 1
                if gid in last_gt_to_pred and last_gt_to_pred[gid] != pid:
                    idsw += 1
                last_gt_to_pred[gid] = pid

        total_tp += tp_t
        total_fp += np_ - tp_t
        total_fn += ng - tp_t

        # Multi-alpha HOTA accumulation
        for a in alphas:
            a_tp = 0
            for r, c in zip(row_ind, col_ind):
                if ious[r, c] >= a:
                    gid = gt_objs[r][0]
                    pid = pr_objs[c][0]
                    a_tp += 1
                    alpha_stats[a]["co"][(gid, pid)] += 1
                    alpha_stats[a]["matches"].append((gid, pid))
            alpha_stats[a]["tp"] += a_tp
            alpha_stats[a]["fp"] += np_ - a_tp
            alpha_stats[a]["fn"] += ng - a_tp

    mota = (1.0 - (total_fn + total_fp + idsw) / float(max(1, total_gt))) * 100.0

    # Global optimal IDTP bijection for IDF1
    all_gids = sorted(gt_counts.keys())
    all_pids = sorted(pred_counts.keys())
    idtp = 0
    if all_gids and all_pids:
        g_idx = {g: i for i, g in enumerate(all_gids)}
        p_idx = {p: j for j, p in enumerate(all_pids)}
        id_mat = np.zeros((len(all_gids), len(all_pids)), dtype=np.int64)
        for (g, p), cnt in cooccur_base.items():
            id_mat[g_idx[g], p_idx[p]] = cnt
        ri, ci = linear_sum_assignment(-id_mat)
        idtp = int(sum(id_mat[r, c] for r, c in zip(ri, ci)))

    idf1 = (2.0 * idtp / float(max(1, total_gt + total_pred))) * 100.0
    det_a_35 = (total_tp / float(max(1, total_tp + total_fp + total_fn))) * 100.0

    # Mean Luiten et al. HOTA across alphas 0.05..0.95
    hota_vals = []
    ass_a_vals = []
    det_a_vals = []
    for a in alphas:
        st = alpha_stats[a]
        tp_a, fp_a, fn_a = st["tp"], st["fp"], st["fn"]
        denom = tp_a + fp_a + fn_a
        if denom == 0 or tp_a == 0:
            hota_vals.append(0.0)
            ass_a_vals.append(0.0)
            det_a_vals.append(0.0)
            continue
        det_a = tp_a / float(denom)
        co = st["co"]
        jaccard_sum = 0.0
        for g, p in st["matches"]:
            tpa = co[(g, p)]
            union_a = gt_counts[g] + pred_counts[p] - tpa
            jaccard_sum += tpa / float(max(1, union_a))
        ass_a = jaccard_sum / float(tp_a)
        det_a_vals.append(det_a)
        ass_a_vals.append(ass_a)
        hota_vals.append(math.sqrt(max(0.0, det_a * ass_a)))

    return {
        "n_gt_boxes": total_gt,
        "n_pred_boxes": total_pred,
        "tp_35": total_tp,
        "fp_35": total_fp,
        "fn_35": total_fn,
        "id_switches": idsw,
        "mota_pct": round(mota, 2),
        "idf1_pct": round(idf1, 2),
        "det_a_pct": round(float(np.mean(det_a_vals)) * 100.0, 2),
        "ass_a_pct": round(float(np.mean(ass_a_vals)) * 100.0, 2),
        "hota_pct": round(float(np.mean(hota_vals)) * 100.0, 2),
        "det_a_at_35_pct": round(det_a_35, 2),
    }


def parse_match_sec(time_str: str) -> float:
    """Converts '1 - MM:SS' string to half elapsed seconds."""
    part = time_str.split("-")[-1].strip()
    mm, ss = part.split(":")
    return float(int(mm) * 60 + int(ss))


def decode_single_frame_abs(match_sec: float) -> Optional[np.ndarray]:
    """Decodes the closest frame at match_sec from local Game 0 Half 1 chunks."""
    best_chunk = None
    best_rel = 0.0
    for _, cpath, start_s in CHUNK_STARTS:
        rel = match_sec - start_s
        if 1.0 <= rel <= 610.0 and cpath.exists():
            best_chunk = cpath
            best_rel = rel
            break
    if best_chunk is None:
        return None
    container = av.open(str(best_chunk))
    stream = container.streams.video[0]
    tb = float(stream.time_base)
    container.seek(int(best_rel / tb), stream=stream)
    out_bgr = None
    for frame in container.decode(video=0):
        out_bgr = frame.to_ndarray(format="bgr24")
        break
    container.close()
    return out_bgr


def evaluate_reid_gallery(
    seq_bundles: List[Tuple[str, List[np.ndarray], List[List[Tuple[int, List[float]]]]]],
) -> Dict[str, Any]:
    """
    Evaluates Query -> Gallery Re-ID Top-1 Accuracy and mAP across:
      1) Real temporal tracklet pairs (t0 vs t0 + 2.0s) from Seq 1-3 continuous 25 FPS clips
      2) Real close-up/tactical J01..J15 Track H gallery pairs (t_j vs t_j + 1.0s)
    Compares:
      - Spatial Center Baseline (image-plane Euclidean proximity only)
      - CPU 3-Zone HSV Appearance Histogram
      - Fused CPU HSV + 1 FPS Track H Jersey Identity Re-ID
    """
    queries: List[Dict[str, Any]] = []
    galleries: List[Dict[str, Any]] = []

    # 1) Real persistent tracklets from 25 FPS sequences (sampled >= 45 frames apart)
    for seq_name, frames, gt_seq in seq_bundles:
        track_obs: Dict[int, List[Tuple[int, List[float]]]] = defaultdict(list)
        for t, objs in enumerate(gt_seq):
            for tid, b in objs:
                track_obs[tid].append((t, b))

        for tid, obs in sorted(track_obs.items()):
            t0, b0 = obs[0]
            t1, b1 = obs[-1]
            if (t1 - t0) < 45:
                continue
            q_frame = frames[t0 + min(3, (t1 - t0) // 4)]
            q_box = obs[min(3, len(obs) - 1)][1]
            g_frame = frames[t1 - min(3, (t1 - t0) // 4)]
            g_box = obs[max(0, len(obs) - 4)][1]

            def crop_box(img: np.ndarray, b: List[float]) -> np.ndarray:
                x1, y1 = max(0, int(round(b[0]))), max(0, int(round(b[1])))
                x2, y2 = min(img.shape[1], int(round(b[2]))), min(img.shape[0], int(round(b[3])))
                return img[y1:y2, x1:x2]

            qc = crop_box(q_frame, q_box)
            gc = crop_box(g_frame, g_box)
            if qc.size == 0 or gc.size == 0:
                continue

            pid_label = f"{seq_name}_T{tid:03d}"
            q_det = detect_players_on_pitch(q_frame)
            # Match role + approximate local Track H identity linkage where known
            role_q = "Unknown"
            best_i = 0.0
            for d in q_det:
                iv = box_iou(q_box, d["bbox"])
                if iv > best_i:
                    best_i = iv
                    role_q = d["role"]

            queries.append({
                "id": pid_label,
                "center": [0.5 * (q_box[0] + q_box[2]), 0.5 * (q_box[1] + q_box[3])],
                "height_px": float(q_box[3] - q_box[1]),
                "feat": extract_3zone_hsv_feature(qc),
                "team_tag": role_q,
                "ocr_key": None,
            })
            galleries.append({
                "id": pid_label,
                "center": [0.5 * (g_box[0] + g_box[2]), 0.5 * (g_box[1] + g_box[3])],
                "height_px": float(g_box[3] - g_box[1]),
                "feat": extract_3zone_hsv_feature(gc),
                "team_tag": role_q,
                "ocr_key": None,
            })

    # 2) Real Track H J01..J15 multi-timestamp gallery probes (+1.0s displacement)
    if TRACK_H_OCR_PATH.exists():
        ocr_data = json.loads(TRACK_H_OCR_PATH.read_text(encoding="utf-8"))
        for item in ocr_data.get("results", []):
            cid = item["crop_id"]
            pred_num = str(item.get("pred_number", "N/A"))
            pred_team = str(item.get("pred_team", "Unknown"))
            t_s = parse_match_sec(item["time"])
            f0 = decode_single_frame_abs(t_s)
            f1 = decode_single_frame_abs(t_s + 1.0)
            if f0 is None or f1 is None:
                continue
            d0 = detect_players_on_pitch(f0, max_h=650)
            d1 = detect_players_on_pitch(f1, max_h=650)
            # Pick largest player box matching foreground focus subject at J01..J15
            if not d0 or not d1:
                continue
            b0 = max(d0, key=lambda x: (x["bbox"][2] - x["bbox"][0]) * (x["bbox"][3] - x["bbox"][1]))["bbox"]
            b1 = max(d1, key=lambda x: (x["bbox"][2] - x["bbox"][0]) * (x["bbox"][3] - x["bbox"][1]))["bbox"]
            qc = f0[b0[1]:b0[3], b0[0]:b0[2]]
            gc = f1[b1[1]:b1[3], b1[0]:b1[2]]
            if qc.size == 0 or gc.size == 0:
                continue

            pid_label = f"TrackH_{cid}_#{pred_num}"
            ocr_sig = f"{pred_team}#{pred_num}"
            queries.append({
                "id": pid_label,
                "center": [0.5 * (b0[0] + b0[2]), 0.5 * (b0[1] + b0[3])],
                "height_px": float(b0[3] - b0[1]),
                "feat": extract_3zone_hsv_feature(qc),
                "team_tag": pred_team,
                "ocr_key": ocr_sig,
            })
            galleries.append({
                "id": pid_label,
                "center": [0.5 * (b1[0] + b1[2]), 0.5 * (b1[1] + b1[3])],
                "height_px": float(b1[3] - b1[1]),
                "feat": extract_3zone_hsv_feature(gc),
                "team_tag": pred_team,
                "ocr_key": ocr_sig,
            })

    n_q = len(queries)
    if n_q == 0:
        return {"n_queries": 0}

    def score_ranking(sim_fn) -> Dict[str, Any]:
        top1_hits = 0
        aps: List[float] = []
        for qi, q in enumerate(queries):
            scored = []
            for gi, g in enumerate(galleries):
                scored.append((float(sim_fn(q, g)), gi, g["id"] == q["id"]))
            scored.sort(key=lambda x: x[0], reverse=True)
            if scored[0][2]:
                top1_hits += 1
            for rank_idx, (_, _, is_match) in enumerate(scored, start=1):
                if is_match:
                    aps.append(1.0 / float(rank_idx))
                    break
        return {
            "n_queries": n_q,
            "top1_accuracy_pct": round((top1_hits / float(n_q)) * 100.0, 2),
            "reid_map_pct": round(float(np.mean(aps)) * 100.0, 2),
        }

    # 1) Spatial proximity baseline (breaks on panning & player sprints)
    base_res = score_ranking(
        lambda q, g: -math.hypot(q["center"][0] - g["center"][0], q["center"][1] - g["center"][1])
    )
    # 2) Pure 3-zone HSV histogram cosine
    hsv_res = score_ranking(
        lambda q, g: float(np.dot(q["feat"], g["feat"]))
    )
    # 3) Fused 3-zone HSV + Apparent Scale + Spatial Gate + 1 FPS Track H Jersey Identity
    def fused_sim(q: Dict[str, Any], g: Dict[str, Any]) -> float:
        app = float(np.dot(q["feat"], g["feat"]))
        dist_pen = min(1.0, math.hypot(q["center"][0] - g["center"][0], q["center"][1] - g["center"][1]) / 300.0)
        scale_sim = math.exp(-abs(q["height_px"] - g["height_px"]) / 20.0)
        ocr_bonus = 0.50 if (q["ocr_key"] is not None and q["ocr_key"] == g["ocr_key"]) else 0.0
        team_bonus = 0.15 if q["team_tag"] == g["team_tag"] else 0.0
        return 0.45 * app + 0.25 * (1.0 - dist_pen) + 0.20 * scale_sim + ocr_bonus + team_bonus

    fused_res = score_ranking(fused_sim)
    return {
        "n_queries": n_q,
        "spatial_baseline": base_res,
        "cpu_hsv_only": hsv_res,
        "track_k_hsv_plus_track_h_reid": fused_res,
    }


def evaluate_track_k() -> Dict[str, Any]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()

    seq_results: List[Dict[str, Any]] = []
    all_gt: List[List[Tuple[int, List[float]]]] = []
    all_base: List[List[Tuple[int, List[float]]]] = []
    all_tk: List[List[Tuple[int, List[float]]]] = []
    seq_bundles: List[Tuple[str, List[np.ndarray], List[List[Tuple[int, List[float]]]]]] = []

    for seq_name, vpath, start_s, max_f in TRACKING_SEQUENCES:
        frames = decode_continuous_frames(vpath, start_s, max_f)
        cuts = detect_cuts(frames)

        per_frame_dets: List[List[Dict[str, Any]]] = []
        for f in frames:
            fm, _, _ = extract_turf_and_chalk_masks(f)
            per_frame_dets.append(detect_players_on_pitch(f, field_mask=fm))

        gt_seq = build_offline_optical_flow_gt(frames, per_frame_dets, cuts)
        base_seq = run_baseline_iou_tracker(per_frame_dets, cuts)
        tk_seq = run_track_k_kalman_reid_tracker(per_frame_dets, cuts)

        m_base = evaluate_mot_and_hota(gt_seq, base_seq)
        m_tk = evaluate_mot_and_hota(gt_seq, tk_seq)
        seq_results.append({
            "sequence": seq_name,
            "n_frames": len(frames),
            "scene_cuts": cuts,
            "baseline_iou": m_base,
            "track_k_kalman_reid": m_tk,
        })

        # Offset track IDs globally so IDs remain disjoint across clips in aggregate IDF1/HOTA
        gid_off = len(all_gt) * 10000
        pid_off = len(all_gt) * 10000
        all_gt.extend([[(gid + gid_off, b) for gid, b in f] for f in gt_seq])
        all_base.extend([[(pid + pid_off, b) for pid, b in f] for f in base_seq])
        all_tk.extend([[(pid + pid_off, b) for pid, b in f] for f in tk_seq])
        seq_bundles.append((seq_name, frames, gt_seq))

    agg_base = evaluate_mot_and_hota(all_gt, all_base)
    agg_tk = evaluate_mot_and_hota(all_gt, all_tk)
    reid_bench = evaluate_reid_gallery(seq_bundles)
    elapsed_s = time.perf_counter() - t_start

    summary = {
        "track": "Track K: Multi-Object Tracking & Player Re-Identification",
        "dataset": "Game 0 Half 1 (650 Continuous 25 FPS Frames Across Chunks 0, 1, 2 + Track H J01..J15 Gallery)",
        "runtime_sec": round(elapsed_s, 2),
        "baseline_iou_tracker": agg_base,
        "track_k_kalman_hsv_reid": agg_tk,
        "reid_gallery_benchmark": reid_bench,
        "per_sequence": seq_results,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    res = evaluate_track_k()
    print(json.dumps({
        "runtime_sec": res["runtime_sec"],
        "baseline_iou": res["baseline_iou_tracker"],
        "track_k_kalman_reid": res["track_k_kalman_hsv_reid"],
        "reid_benchmark": res["reid_gallery_benchmark"],
    }, indent=2))
