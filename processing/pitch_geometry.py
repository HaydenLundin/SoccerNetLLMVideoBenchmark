"""
Shared Broadcast Pitch Geometry, Homography Calibration, and Player Detection Engine.
Powers V5 SoccerNet Challenge Tracks I, J, K, and L on local CPU/OpenCV/SciPy ($0 compute):
- Canonical FIFA Pitch Model (105m x 68m, X in [-52.5, +52.5], Y in [-34.0, +34.0])
- Broadcast Turf & Chalk Line Extraction (top-hat luminance filtering + HUD suppression)
- Multi-Start Soft-Inlier Chamfer Pinhole Calibration -> 3x3 Homography (H) & H^{-1}
- Chroma-Guided Foreground Player Blob Detection + HSV Kit Role Classification
- Pitch Part IoU and Pixel Reprojection Accuracy (@5px, @10px) Evaluation Helpers
"""

import time
from typing import Dict, List, Tuple, Any, Optional

import cv2
import numpy as np
from scipy.optimize import least_squares

# ---------------------------------------------------------------------------
# 1. Canonical FIFA Pitch Model (105.0m x 68.0m)
# ---------------------------------------------------------------------------

PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0
HALF_LENGTH = PITCH_LENGTH / 2.0   # 52.5 m
HALF_WIDTH = PITCH_WIDTH / 2.0     # 34.0 m

IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720


def build_canonical_pitch_polylines() -> List[np.ndarray]:
    """
    Returns ordered metric (N, 2) polyline arrays representing every canonical
    FIFA white line marking on Z=0.
    """
    polylines: List[np.ndarray] = []

    def add_seg(p1: Tuple[float, float], p2: Tuple[float, float], n: int = 30) -> None:
        t = np.linspace(0.0, 1.0, n)[:, None]
        a = np.asarray(p1, dtype=np.float64)[None, :]
        b = np.asarray(p2, dtype=np.float64)[None, :]
        polylines.append(a * (1.0 - t) + b * t)

    # Outer boundaries & halfway line
    add_seg((-52.5, -34.0), (52.5, -34.0), 90)
    add_seg((-52.5, 34.0), (52.5, 34.0), 90)
    add_seg((-52.5, -34.0), (-52.5, 34.0), 60)
    add_seg((52.5, -34.0), (52.5, 34.0), 60)
    add_seg((0.0, -34.0), (0.0, 34.0), 60)

    # Center circle (R = 9.15m)
    th = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)
    polylines.append(np.stack([9.15 * np.cos(th), 9.15 * np.sin(th)], axis=1))

    # Left (-1) and Right (+1) 18-yard box, 6-yard box, and penalty arc
    for sgn in (-1.0, 1.0):
        gx = sgn * 52.5
        bx18 = sgn * 36.0
        bx6 = sgn * 47.0
        add_seg((gx, -20.16), (bx18, -20.16), 25)
        add_seg((gx, 20.16), (bx18, 20.16), 25)
        add_seg((bx18, -20.16), (bx18, 20.16), 45)
        add_seg((gx, -9.16), (bx6, -9.16), 15)
        add_seg((gx, 9.16), (bx6, 9.16), 15)
        add_seg((bx6, -9.16), (bx6, 9.16), 22)
        ath = np.linspace(-0.925, 0.925, 24)
        ax = (41.5 - 9.15 * np.cos(ath)) if sgn > 0 else (-41.5 + 9.15 * np.cos(ath))
        ay = 9.15 * np.sin(ath)
        polylines.append(np.stack([ax, ay], axis=1))

    return polylines


CANONICAL_POLYLINES: List[np.ndarray] = build_canonical_pitch_polylines()
CANONICAL_PITCH_POINTS: np.ndarray = np.vstack(CANONICAL_POLYLINES).astype(np.float64)


# Pitch Part Polygons for Track I Pitch Part IoU (mIoU_part)
def get_pitch_part_polygons() -> Dict[str, np.ndarray]:
    """
    Defines canonical closed polygon regions (in world XY meters) for evaluating
    Pitch Part IoU (Left/Right 18-yd box, 6-yd box, Center Circle, Half-Pitch Zones).
    """
    th = np.linspace(0.0, 2.0 * np.pi, 48, endpoint=False)
    circle_poly = np.stack([9.15 * np.cos(th), 9.15 * np.sin(th)], axis=1)

    return {
        "left_half": np.array([[-52.5, -34.0], [0.0, -34.0], [0.0, 34.0], [-52.5, 34.0]]),
        "right_half": np.array([[0.0, -34.0], [52.5, -34.0], [52.5, 34.0], [0.0, 34.0]]),
        "left_penalty_box": np.array([[-52.5, -20.16], [-36.0, -20.16], [-36.0, 20.16], [-52.5, 20.16]]),
        "right_penalty_box": np.array([[36.0, -20.16], [52.5, -20.16], [52.5, 20.16], [36.0, 20.16]]),
        "left_six_yard_box": np.array([[-52.5, -9.16], [-47.0, -9.16], [-47.0, 9.16], [-52.5, 9.16]]),
        "right_six_yard_box": np.array([[47.0, -9.16], [52.5, -9.16], [52.5, 9.16], [47.0, 9.16]]),
        "center_circle": circle_poly,
    }


PITCH_PART_POLYGONS = get_pitch_part_polygons()


# ---------------------------------------------------------------------------
# 2. Pinhole Camera Model & Homography Projection Helpers
# ---------------------------------------------------------------------------

def camera_params_to_pose(
    params: np.ndarray,
    width: int = IMAGE_WIDTH,
    height: int = IMAGE_HEIGHT
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Converts 7-DOF broadcast gantry pinhole parameters:
      params = [tx, ty, roll_rad, focal_px, cx, cy, cz]
    into (K, R, t, H) where:
      - C = [cx, cy, cz]: 3D stadium gantry location (+Z up, camera sits around Y~-56m looking toward +Y)
      - T = [tx, ty, 0]: 3D look-at point on Z=0 pitch plane
      - H: 3x3 planar homography mapping [X_m, Y_m, 1]^T -> [u_px, v_px, 1]^T
    """
    tx, ty, roll, f, cx, cy, cz = [float(v) for v in params]
    C = np.array([cx, cy, cz], dtype=np.float64)
    T = np.array([tx, ty, 0.0], dtype=np.float64)

    fwd = T - C
    fwd /= max(np.linalg.norm(fwd), 1e-9)
    up_w = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    right = np.cross(fwd, up_w)
    right /= max(np.linalg.norm(right), 1e-9)
    down = np.cross(fwd, right)
    down /= max(np.linalg.norm(down), 1e-9)

    cr, sr = np.cos(roll), np.sin(roll)
    R_roll = np.array([[cr, -sr, 0.0], [sr, cr, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    R = R_roll @ np.vstack([right, down, fwd])

    K = np.array([[f, 0.0, width / 2.0], [0.0, f, height / 2.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    t = -R @ C
    H = K @ np.column_stack([R[:, 0], R[:, 1], t])
    if abs(H[2, 2]) > 1e-9:
        H = H / H[2, 2]
    return K, R, t, H


def camera_params_to_homography(
    params: np.ndarray,
    width: int = IMAGE_WIDTH,
    height: int = IMAGE_HEIGHT
) -> np.ndarray:
    """Returns only the 3x3 world-to-image homography matrix H."""
    _, _, _, H = camera_params_to_pose(params, width=width, height=height)
    return H


def project_world_to_image(
    H: np.ndarray,
    world_xy: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Projects Nx2 world pitch points (X, Y) in meters into pixel coordinates (u, v)."""
    pts = np.asarray(world_xy, dtype=np.float64).reshape(-1, 2)
    hom = np.column_stack([pts, np.ones(len(pts), dtype=np.float64)])
    proj = (H @ hom.T).T
    valid = proj[:, 2] > 1e-3
    uv = proj[:, :2] / np.maximum(proj[:, 2:3], 1e-6)
    return uv, valid


def project_image_to_world(H: np.ndarray, image_uv: np.ndarray) -> np.ndarray:
    """Projects Nx2 image pixel foot coordinates (u, v) via H^{-1} onto Z=0 pitch plane (X, Y) in meters."""
    pts = np.asarray(image_uv, dtype=np.float64).reshape(-1, 2)
    H_inv = np.linalg.inv(H)
    hom = np.column_stack([pts, np.ones(len(pts), dtype=np.float64)])
    world = (H_inv @ hom.T).T
    w = np.where(np.abs(world[:, 2:3]) < 1e-6, 1e-6, world[:, 2:3])
    return world[:, :2] / w


def homography_from_landmarks(
    world_xy: List[Tuple[float, float]],
    image_uv: List[Tuple[float, float]]
) -> np.ndarray:
    """Computes normalized 3x3 DLT/RANSAC homography from >=4 world-to-image correspondences."""
    src = np.asarray(world_xy, dtype=np.float64)
    dst = np.asarray(image_uv, dtype=np.float64)
    H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
    if H is None:
        H, _ = cv2.findHomography(src, dst, 0)
    return H / (H[2, 2] if abs(H[2, 2]) > 1e-9 else 1.0)


# ---------------------------------------------------------------------------
# 3. Broadcast Turf, White Chalk Line Extraction & Homography Calibration
# ---------------------------------------------------------------------------

def extract_turf_and_chalk_masks(bgr_img: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Extracts clean green pitch field mask and thin white chalk marking mask from
    a 1280x720 broadcast frame while suppressing top-left scorebugs, top-right
    channel logos, bottom-right league watermarks, and bright advertising boards.
    """
    h_img, w_img = bgr_img.shape[:2]
    hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)

    # Green grass hue/saturation band tuned for striped EPL turf
    turf_raw = (
        (hsv[:, :, 0] >= 33) & (hsv[:, :, 0] <= 74) &
        (hsv[:, :, 1] >= 40) & (hsv[:, :, 2] >= 38)
    ).astype(np.uint8) * 255

    turf_closed = cv2.morphologyEx(turf_raw, cv2.MORPH_CLOSE, np.ones((21, 21), np.uint8))
    n_cc, labels, stats, _ = cv2.connectedComponentsWithStats(turf_closed)
    field_mask = np.zeros_like(turf_raw)
    if n_cc > 1:
        best_lbl = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        comp = (labels == best_lbl).astype(np.uint8) * 255
        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(field_mask, cnts, -1, 255, -1)

    turf_pct = float(field_mask.mean() / 255.0 * 100.0)

    # Suppress broadcast HUD corners (scaled proportionally to frame size)
    sx, sy = w_img / 1280.0, h_img / 720.0
    field_clean = field_mask.copy()
    field_clean[0:int(100 * sy), 0:int(425 * sx)] = 0
    field_clean[0:int(95 * sy), int(1075 * sx):w_img] = 0
    field_clean[int(595 * sy):h_img, int(1035 * sx):w_img] = 0

    # Keep upper touchline boundary (2px dilate around filled turf, avoiding tall goalposts/adboards)
    field_line = cv2.dilate(field_clean, np.ones((3, 3), np.uint8))
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, np.ones((11, 11), np.uint8))

    chalk_mask = (
        (tophat > 20) &
        (hsv[:, :, 1] < 95) &
        (gray > 140) &
        (field_line > 0)
    ).astype(np.uint8) * 255

    return field_clean, chalk_mask, turf_pct


# Lazy cache for coarse gantry hypotheses so repeated calls take ~100ms
_COARSE_HYPOTHESES_CACHE: Dict[Tuple[int, int], List[Tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}


def _get_coarse_hypotheses(width: int, height: int) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    key = (width, height)
    if key in _COARSE_HYPOTHESES_CACHE:
        return _COARSE_HYPOTHESES_CACHE[key]

    hom_pitch = np.column_stack([CANONICAL_PITCH_POINTS, np.ones(len(CANONICAL_PITCH_POINTS))])
    cands = [
        np.array([tx, ty, 0.0, fl, 0.5, -56.0, 15.8], dtype=np.float64)
        for tx in np.linspace(-46.0, 46.0, 47)
        for ty in np.linspace(-10.0, 22.0, 17)
        for fl in (1900.0, 2200.0, 2550.0, 2900.0, 3300.0, 3800.0, 4450.0)
    ]
    pad_y_top = int(95 * (height / 720.0))
    cached = []
    for p in cands:
        H = camera_params_to_homography(p, width=width, height=height)
        pr = (H @ hom_pitch.T).T
        val = pr[:, 2] > 1e-3
        uv = pr[:, :2] / np.maximum(pr[:, 2:3], 1e-6)
        inb = val & (uv[:, 0] >= 10) & (uv[:, 0] < width - 10) & (uv[:, 1] >= pad_y_top) & (uv[:, 1] < height - 10)
        if int(inb.sum()) >= 50:
            cached.append((p, uv[inb, 0].astype(int), uv[inb, 1].astype(int)))
    _COARSE_HYPOTHESES_CACHE[key] = cached
    return cached


def estimate_pitch_homography(
    bgr_img: np.ndarray,
    init_params: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Estimates 3x3 world-to-image homography H and 7-DOF stadium camera parameters
    strictly from raw broadcast pixel chalk markings (no external labels).
    Uses multi-start coarse Chamfer scoring + bilinear Distance-Transform LSQ refinement.
    """
    t0 = time.perf_counter()
    h_img, w_img = bgr_img.shape[:2]
    field_mask, chalk_mask, turf_pct = extract_turf_and_chalk_masks(bgr_img)
    dist = cv2.distanceTransform(255 - chalk_mask, cv2.DIST_L2, 5)

    # Candidate initial seeds
    if init_params is not None:
        seeds = [np.asarray(init_params, dtype=np.float64)]
    else:
        coarse = _get_coarse_hypotheses(w_img, h_img)
        scored = [
            (
                float(np.minimum(dist[vi, ui], 14.0).mean())
                - 1.5 * float((dist[vi, ui] <= 4.0).mean()),
                idx,
            )
            for idx, (_, ui, vi) in enumerate(coarse)
        ]
        scored.sort(key=lambda x: x[0])
        seeds = []
        for _, idx in scored:
            p = coarse[idx][0]
            if all(
                abs(p[0] - s[0]) > 5.0
                or abs(p[1] - s[1]) > 4.0
                or abs(p[3] - s[3]) > 300.0
                for s in seeds
            ):
                seeds.append(p)
            if len(seeds) >= 8:
                break
        # Ensure top diverse zoomed-in candidates (fl >= 3550 and fl >= 4100) are tested for tight box views
        for min_fl, quota in ((3550.0, 1), (4100.0, 3)):
            added = 0
            for _, idx in scored:
                p = coarse[idx][0]
                if p[3] < min_fl:
                    continue
                if all(abs(p[0] - s[0]) > 4.0 or abs(p[1] - s[1]) > 3.0 or abs(p[3] - s[3]) > 350.0 for s in seeds):
                    seeds.append(p)
                    added += 1
                    if added >= quota:
                        break

    lb = np.array([-52.5, -24.0, -0.05, 1850.0, -3.5, -62.0, 14.0], dtype=np.float64)
    ub = np.array([52.5, 32.0, 0.05, 5200.0, 3.5, -50.0, 18.5], dtype=np.float64)
    if init_params is not None and abs(float(init_params[4])) > 3.5:
        lb = np.array([-52.5, -24.0, -0.05, 1650.0, -48.0, -76.0, 11.5], dtype=np.float64)
        ub = np.array([52.5, 32.0, 0.05, 4800.0, 48.0, -44.0, 22.0], dtype=np.float64)
    pad_top = int(95 * (h_img / 720.0))

    best_score = float("inf")
    best_params = np.clip(seeds[0].copy(), lb, ub)

    def _refine_seed(seed_vec: np.ndarray, lb_vec: np.ndarray, ub_vec: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
        cur = np.clip(seed_vec.copy(), lb_vec, ub_vec)
        for _ in range(2):
            Hc = camera_params_to_homography(cur, width=w_img, height=h_img)
            uvc, valc = project_world_to_image(Hc, CANONICAL_PITCH_POINTS)
            vm = valc & (uvc[:, 0] >= 10) & (uvc[:, 0] < w_img - 10) & (uvc[:, 1] >= pad_top) & (uvc[:, 1] < h_img - 10)
            if int(vm.sum()) < 45:
                break
            vp = CANONICAL_PITCH_POINTS[vm]

            def residual_fn(params: np.ndarray) -> np.ndarray:
                H = camera_params_to_homography(params, width=w_img, height=h_img)
                uv, _ = project_world_to_image(H, vp)
                uc = np.clip(uv[:, 0], 0.0, w_img - 1.05)
                vc = np.clip(uv[:, 1], 0.0, h_img - 1.05)
                ui = uc.astype(int)
                vi = vc.astype(int)
                du = uc - ui
                dv = vc - vi
                return (
                    (1.0 - du) * (1.0 - dv) * dist[vi, ui] +
                    du * (1.0 - dv) * dist[vi, ui + 1] +
                    (1.0 - du) * dv * dist[vi + 1, ui] +
                    du * dv * dist[vi + 1, ui + 1]
                )

            cur = least_squares(residual_fn, cur, bounds=(lb_vec, ub_vec), loss="soft_l1", f_scale=3.0).x

        Hf = camera_params_to_homography(cur, width=w_img, height=h_img)
        uvf, valf = project_world_to_image(Hf, CANONICAL_PITCH_POINTS)
        vis = valf & (uvf[:, 0] >= 10) & (uvf[:, 0] < w_img - 10) & (uvf[:, 1] >= pad_top) & (uvf[:, 1] < h_img - 10)
        if int(vis.sum()) < 50:
            return float("inf"), cur, np.array([50.0])
        dchk = dist[uvf[vis, 1].astype(int), uvf[vis, 0].astype(int)]
        ref_sc = float(np.minimum(dchk, 25.0).mean()) - 3.0 * float((dchk <= 5.0).mean())
        return ref_sc, cur, dchk

    for seed in seeds:
        ref_sc, cur, _ = _refine_seed(seed, lb, ub)
        if ref_sc < best_score:
            best_score = ref_sc
            best_params = cur

    # Box-centered zoom fallback: when free-kick player walls perturb coarse ranking
    if init_params is None and best_score > 0.2 and abs(float(best_params[0])) > 25.0:
        side = float(np.sign(best_params[0]))
        zoom_cands = [
            np.array([side * tx_z, ty_z, 0.0, fl_z, 0.5, -56.0, 15.8], dtype=np.float64)
            for tx_z in (44.0, 45.5)
            for ty_z in (4.0, 6.5)
            for fl_z in (4550.0,)
        ]
        for zseed in zoom_cands:
            ref_z, cur_z, _ = _refine_seed(zseed, lb, ub)
            if ref_z < best_score:
                best_score = ref_z
                best_params = cur_z

    K, R, t, H_final = camera_params_to_pose(best_params, width=w_img, height=h_img)
    uv_fin, val_fin = project_world_to_image(H_final, CANONICAL_PITCH_POINTS)
    vis_fin = val_fin & (uv_fin[:, 0] >= 10) & (uv_fin[:, 0] < w_img - 10) & (uv_fin[:, 1] >= pad_top) & (uv_fin[:, 1] < h_img - 10)
    chamfer_vals = dist[uv_fin[vis_fin, 1].astype(int), uv_fin[vis_fin, 0].astype(int)] if vis_fin.any() else np.array([50.0])

    # Fallback for offside-line / corner gantry angles (e.g. X_cam ~ +-24..36m) when center gantry fails
    if init_params is None and (float(np.median(chamfer_vals)) > 1.8 or float(np.mean(chamfer_vals <= 5.0)) < 0.72):
        aux_cands = [
            np.array([cx + off, ty, 0.0, fl, cx, cy, 14.0], dtype=np.float64)
            for cx in (-36.0, -24.0, 24.0, 36.0)
            for cy in (-68.0, -56.0)
            for off in (-6.0, 2.0, 9.0)
            for ty in (-2.0, 5.0, 11.0, 17.0)
            for fl in (2400.0, 3200.0, 4100.0)
        ]
        aux_scored = []
        for ap in aux_cands:
            H_a = camera_params_to_homography(ap, width=w_img, height=h_img)
            uva, vala = project_world_to_image(H_a, CANONICAL_PITCH_POINTS)
            ina = vala & (uva[:, 0] >= 10) & (uva[:, 0] < w_img - 10) & (uva[:, 1] >= pad_top) & (uva[:, 1] < h_img - 10)
            if int(ina.sum()) < 45:
                continue
            d_a = dist[uva[ina, 1].astype(int), uva[ina, 0].astype(int)]
            sc = float(np.minimum(d_a, 14.0).mean()) - 1.5 * float((d_a <= 4.0).mean())
            aux_scored.append((sc, ap))
        aux_scored.sort(key=lambda x: x[0])
        if aux_scored:
            lb_aux = np.array([-52.5, -24.0, -0.05, 1650.0, -48.0, -76.0, 11.5], dtype=np.float64)
            ub_aux = np.array([52.5, 32.0, 0.05, 4800.0, 48.0, -44.0, 22.0], dtype=np.float64)
            aux_seeds = []
            # Pick best diverse seeds across gantry side and focal-length regimes
            for _, ap in aux_scored:
                if all(abs(ap[0] - s[0]) > 5.0 or abs(ap[4] - s[4]) > 8.0 or abs(ap[3] - s[3]) > 400.0 for s in aux_seeds):
                    aux_seeds.append(ap)
                if len(aux_seeds) >= 6:
                    break
            for cx_sign in (-1.0, 1.0):
                for fl_hi in (False, True):
                    for _, ap in aux_scored:
                        if np.sign(ap[4]) == cx_sign and (ap[3] >= 3500.0) == fl_hi:
                            if all(np.linalg.norm(ap - s) > 1e-3 for s in aux_seeds):
                                aux_seeds.append(ap)
                            break
            for aseed in aux_seeds:
                ref_a, cur_a, ch_a = _refine_seed(aseed, lb_aux, ub_aux)
                if ref_a < best_score:
                    best_score = ref_a
                    best_params = cur_a
                    K, R, t, H_final = camera_params_to_pose(best_params, width=w_img, height=h_img)
                    chamfer_vals = ch_a

    return {
        "H": H_final,
        "params": best_params,
        "K": K,
        "R": R,
        "t": t,
        "camera_center_3d": best_params[4:7].copy(),
        "focal_px": float(best_params[3]),
        "turf_pct": round(turf_pct, 2),
        "chamfer_median_px": round(float(np.median(chamfer_vals)), 3),
        "chamfer_inlier_5px_pct": round(float(np.mean(chamfer_vals <= 5.0) * 100.0), 2),
        "elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 1),
    }


# ---------------------------------------------------------------------------
# 4. Track I Evaluation Metrics (Pitch Part IoU & Reprojection Accuracy)
# ---------------------------------------------------------------------------

def compute_pitch_part_iou(
    H_pred: np.ndarray,
    H_gt: np.ndarray,
    width: int = IMAGE_WIDTH,
    height: int = IMAGE_HEIGHT
) -> Dict[str, Any]:
    """
    Renders canonical FIFA pitch part polygons in 2D image coordinates via H_pred
    and H_gt, and computes per-part & mean visible Pitch Part IoU (IoU_part).
    """
    part_ious: Dict[str, float] = {}
    for name, poly_xy in PITCH_PART_POLYGONS.items():
        uv_gt, val_gt = project_world_to_image(H_gt, poly_xy)
        uv_pr, val_pr = project_world_to_image(H_pred, poly_xy)
        if not (val_gt.all() and val_pr.all()):
            continue

        mask_gt = np.zeros((height, width), dtype=np.uint8)
        mask_pr = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask_gt, [np.round(uv_gt).astype(np.int32)], 1)
        cv2.fillPoly(mask_pr, [np.round(uv_pr).astype(np.int32)], 1)

        gt_area = int(mask_gt.sum())
        if gt_area < 400:  # Part not substantially visible inside camera frustum
            continue

        inter = int(np.logical_and(mask_gt > 0, mask_pr > 0).sum())
        union = int(np.logical_or(mask_gt > 0, mask_pr > 0).sum())
        part_ious[name] = round(float(inter) / float(max(union, 1)), 4)

    mean_iou = float(np.mean(list(part_ious.values()))) if part_ious else 0.0
    return {
        "mean_iou_part": round(mean_iou, 4),
        "per_part_iou": part_ious,
    }


def compute_reprojection_accuracy(
    H_pred: np.ndarray,
    H_gt: np.ndarray,
    width: int = IMAGE_WIDTH,
    height: int = IMAGE_HEIGHT,
) -> Dict[str, Any]:
    """
    Computes SoccerNet-Calibration Reprojection Accuracy @ 5px and @ 10px across
    all canonical pitch polyline samples visible inside the ground-truth viewport.
    """
    uv_gt, val_gt = project_world_to_image(H_gt, CANONICAL_PITCH_POINTS)
    vis = (
        val_gt &
        (uv_gt[:, 0] >= 0.0) & (uv_gt[:, 0] < float(width)) &
        (uv_gt[:, 1] >= 0.0) & (uv_gt[:, 1] < float(height))
    )
    if not vis.any():
        return {"ac_5px": 0.0, "ac_10px": 0.0, "mean_err_px": 999.0, "median_err_px": 999.0, "visible_points": 0}

    vis_world = CANONICAL_PITCH_POINTS[vis]
    uv_pr, val_pr = project_world_to_image(H_pred, vis_world)
    errs = np.where(
        val_pr,
        np.linalg.norm(uv_pr - uv_gt[vis], axis=1),
        999.0,
    )
    return {
        "ac_5px": round(float(np.mean(errs <= 5.0)), 4),
        "ac_10px": round(float(np.mean(errs <= 10.0)), 4),
        "mean_err_px": round(float(np.mean(errs)), 3),
        "median_err_px": round(float(np.median(errs)), 3),
        "visible_points": int(vis.sum()),
    }


# ---------------------------------------------------------------------------
# 5. Chroma-Guided Player Detector & HSV Kit Classification (Tracks J, K, L)
# ---------------------------------------------------------------------------

def classify_kit_role(hsv_patch: np.ndarray) -> Tuple[str, float]:
    """
    Classifies upper-torso HSV pixels into match roles for Man Utd vs Arsenal:
    - Home Team (Man Utd Red/White)
    - Away Team (Arsenal Gold/Navy)
    - Referee (Cyan/Black)
    - Goalkeeper (Orange/Dark Grey)
    """
    if hsv_patch.size == 0:
        return "Home Team", 0.50

    h = hsv_patch[:, :, 0]
    s = hsv_patch[:, :, 1]
    v = hsv_patch[:, :, 2]
    total = float(h.size)

    red_frac = float((((h <= 6) | (h >= 168)) & (s >= 75) & (v >= 45)).sum()) / total
    orange_frac = float(((h >= 7) & (h <= 16) & (s >= 115) & (v >= 115)).sum()) / total
    yellow_frac = float(((h >= 17) & (h <= 34) & (s >= 75) & (v >= 70)).sum()) / total
    cyan_frac = float(((h >= 78) & (h <= 112) & (s >= 55) & (v >= 65)).sum()) / total
    dark_frac = float(((v <= 78) & (s <= 95)).sum()) / total

    scores = {
        "Home Team": red_frac,
        "Away Team": yellow_frac,
        "Referee": cyan_frac * 1.15,
        "Goalkeeper": max(orange_frac * 1.35, dark_frac * 0.48),
    }
    best_role = max(scores, key=lambda k: scores[k])
    best_val = scores[best_role]
    if best_val < 0.04:
        return "Home Team" if red_frac >= yellow_frac else "Away Team", 0.52
    return best_role, round(min(0.99, 0.55 + best_val), 3)


def detect_players_on_pitch(
    bgr_img: np.ndarray,
    field_mask: Optional[np.ndarray] = None,
    H: Optional[np.ndarray] = None,
    min_h: int = 20,
    max_h: int = 220
) -> List[Dict[str, Any]]:
    """
    Detects active player / goalkeeper / referee bounding boxes strictly inside
    the green turf field using chroma-guided silhouette extraction (suppressing
    white chalk pitch lines while preserving white shorts/socks attached to jerseys).
    Optionally splits wide multi-player wall blobs and projects foot points via H^{-1}.
    """
    if field_mask is None:
        field_mask, _, _ = extract_turf_and_chalk_masks(bgr_img)

    # Fill interior holes inside turf mask so non-green players in the goal mouth stay inside pitch ROI
    field_opened = cv2.morphologyEx(field_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    cnts_f, _ = cv2.findContours(field_opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    field_filled = np.zeros_like(field_mask)
    if cnts_f:
        cv2.drawContours(field_filled, cnts_f, -1, 255, cv2.FILLED)
    else:
        field_filled = field_mask
    field_inner = cv2.erode(field_filled, np.ones((7, 5), np.uint8))
    hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)

    chroma_player = (
        (((hsv[:, :, 0] <= 16) | (hsv[:, :, 0] >= 164)) & (hsv[:, :, 1] >= 75)) |
        ((hsv[:, :, 0] >= 17) & (hsv[:, :, 0] <= 34) & (hsv[:, :, 1] >= 72)) |
        ((hsv[:, :, 0] >= 78) & (hsv[:, :, 0] <= 112) & (hsv[:, :, 1] >= 55)) |
        ((hsv[:, :, 2] <= 85) & (hsv[:, :, 1] <= 90) & (hsv[:, :, 2] >= 14))
    )
    body = ((chroma_player | (hsv[:, :, 1] < 38)) & (field_inner > 0)).astype(np.uint8) * 255
    chroma_u8 = (chroma_player.astype(np.uint8) * 255) & field_inner
    chroma_gate = cv2.dilate(chroma_u8, np.ones((27, 13), np.uint8))

    players_mask = cv2.morphologyEx(body & chroma_gate, cv2.MORPH_CLOSE, np.ones((11, 5), np.uint8))
    n_cc, _, stats, _ = cv2.connectedComponentsWithStats(players_mask)
    img_w = bgr_img.shape[1]

    raw_boxes: List[Tuple[int, int, int, int]] = []
    for i in range(1, n_cc):
        x, y, w, h, area = [int(v) for v in stats[i]]
        if not (min_h <= h <= max_h and 7 <= w <= 155 and area >= 90):
            continue
        # Reject skinny vertical goalposts / border-clipped slivers
        if h > 4.3 * w or (w <= 10 and (x <= 1 or x + w >= img_w - 1)):
            continue
        # Reject pure-white goalposts / chalk stubs with negligible chromatic jersey pixels
        if float(chroma_u8[y:y + h, x:x + w].mean()) <= 12.0:
            continue
        # Reject flat horizontal banners / touchlines; allow wider aspect only for multi-player clusters
        if h < 0.72 * w or (w < 40 and h < 1.15 * w):
            continue
        # Split wide merged multi-player clusters when wide, torso-bimodal, or split between contrasting kits
        do_split = False
        if w >= 54 and h >= 32:
            do_split = True
        elif w >= 40 and h >= 32:
            ty1, ty2 = y + int(0.12 * h), y + max(2, int(0.66 * h))
            torso_bin = (chroma_u8[ty1:ty2, x:x + w] > 0).sum(axis=0).astype(np.float32)
            col_smooth = np.convolve(torso_bin, np.ones(5, dtype=np.float32) / 5.0, mode="same")
            mid_lo, mid_hi = max(1, w // 4), min(w - 1, (3 * w) // 4)
            if mid_hi > mid_lo and col_smooth[mid_lo:mid_hi].min() < 0.35 * max(1e-5, float(col_smooth.max())):
                do_split = True
            else:
                half_x = w // 2
                r_l, c_l = classify_kit_role(hsv[ty1:ty2, x:x + half_x])
                r_r, c_r = classify_kit_role(hsv[ty1:ty2, x + half_x:x + w])
                if r_l != r_r and min(c_l, c_r) >= 0.70:
                    do_split = True

        if do_split:
            sub_col = chroma_u8[y:y + h, x:x + w].sum(axis=0)
            n_splits = max(2, min(5, int(round(w / 26.0))))
            step = w / float(n_splits)
            for k in range(n_splits):
                sx = int(round(x + k * step))
                ex = int(round(x + (k + 1) * step))
                if ex - sx < 8 or sub_col[sx - x:ex - x].max() <= 0:
                    continue
                nz = np.where(players_mask[y:y + h, sx:ex].max(axis=1) > 0)[0]
                if len(nz) == 0:
                    continue
                sy1, sy2 = y + int(nz[0]), y + int(nz[-1]) + 1
                if (sy2 - sy1) >= min_h and (sy2 - sy1) >= 1.10 * (ex - sx):
                    raw_boxes.append((sx, sy1, ex, sy2))
        else:
            raw_boxes.append((x, y, x + w, y + h))

    detections: List[Dict[str, Any]] = []
    for idx, (x1, y1, x2, y2) in enumerate(raw_boxes):
        bh = max(1, y2 - y1)
        torso_y1 = y1 + int(0.14 * bh)
        torso_y2 = y1 + int(0.58 * bh)
        torso_hsv = hsv[torso_y1:max(torso_y1 + 1, torso_y2), x1:x2]
        role, role_conf = classify_kit_role(torso_hsv)

        # Normalized 16-bin Hue + 8-bin Sat histogram for Track K Re-ID
        if torso_hsv.size > 0:
            hist = cv2.calcHist([torso_hsv], [0, 1], None, [16, 8], [0, 180, 0, 256]).flatten()
            norm = float(np.linalg.norm(hist))
            feat = hist / norm if norm > 1e-6 else hist
        else:
            feat = np.zeros(128, dtype=np.float32)

        foot_u = 0.5 * (x1 + x2)
        foot_v = float(y2)
        det: Dict[str, Any] = {
            "det_id": idx,
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
            "foot_uv": [round(foot_u, 1), round(foot_v, 1)],
            "height_px": int(bh),
            "role": role,
            "role_conf": role_conf,
            "appearance_hist": feat,
        }
        if H is not None:
            w_xy = project_image_to_world(H, np.array([[foot_u, foot_v]]))[0]
            det["world_xy"] = [round(float(w_xy[0]), 2), round(float(w_xy[1]), 2)]
        detections.append(det)

    return detections
