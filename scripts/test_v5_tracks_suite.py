"""
Verification test suite for v5.0:
  - Tracks I-L evaluation summaries (Camera Calibration, Game State Reconstruction,
    Multi-Object Tracking & Re-ID, 3D Metric Depth Estimation)
  - Spatial/geometric primitives in processing/pitch_geometry.py (homography
    round-trip projection, kit-role classification, and real-frame player detection)
  - ReVal cross-sport normalization parity & comprehensive quantitative benchmark

Usage:
    python scripts/test_v5_tracks_suite.py
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from processing.pitch_geometry import (
    camera_params_to_homography,
    classify_kit_role,
    detect_players_on_pitch,
    extract_turf_and_chalk_masks,
    project_image_to_world,
    project_world_to_image,
)
from processing.reval_scoring import (
    CANONICAL_GAME_FREQUENCY_ANCHOR,
    compute_normalized_action_weight,
)


def _load_json(rel_path: str) -> dict:
    path = ROOT / rel_path
    assert path.exists(), f"Missing artifact: {path}"
    return json.loads(path.read_text(encoding="utf-8"))


def test_track_i_through_l_summaries() -> None:
    """Validate all four v5.0 SoccerNet spatial/geometric evaluation JSONs."""
    # Track I: Camera Calibration
    cam = _load_json("logs/v5_eval/camera_calibration/camera_calibration_summary.json")
    hough = cam["baseline_hough_ransac"]
    ti = cam["track_i_chamfer_pinhole"]
    assert ti["mean_iou_part"] > 0.90, f"Unexpected Track I IoU_part: {ti}"
    assert ti["mean_iou_part"] > hough["mean_iou_part"]
    assert ti["mean_ac_10px_pct"] > 70.0
    assert ti["mean_reproj_err_px"] < 15.0

    # Track J: Game State Reconstruction (2D Minimap)
    gsr = _load_json("logs/v5_eval/game_state_gsr/game_state_gsr_summary.json")
    lin = gsr["baseline_linear_viewport"]
    tj = gsr["track_j_calibrated_gsr"]
    assert tj["composite_gs_hota_pct"] > 80.0
    assert tj["composite_gs_hota_pct"] > lin["composite_gs_hota_pct"]
    assert tj["loc_f1_2m_pct"] > 85.0
    assert tj["role_classification_acc_pct"] > 95.0

    # Track K: Multi-Object Tracking & Player Re-ID
    trk = _load_json("logs/v5_eval/tracking_reid/tracking_reid_summary.json")
    biou = trk["baseline_iou_tracker"]
    tk = trk["track_k_kalman_hsv_reid"]
    reid = trk["reid_gallery_benchmark"]
    assert tk["hota_pct"] > biou["hota_pct"]
    assert tk["idf1_pct"] > biou["idf1_pct"]
    assert tk["id_switches"] < biou["id_switches"]
    assert (
        reid["track_k_hsv_plus_track_h_reid"]["top1_accuracy_pct"]
        > reid["spatial_baseline"]["top1_accuracy_pct"]
    )
    assert (
        reid["track_k_hsv_plus_track_h_reid"]["reid_map_pct"]
        > reid["cpu_hsv_only"]["reid_map_pct"]
    )

    # Track L: 3D Metric Depth Estimation
    dep = _load_json("logs/v5_eval/depth_estimation/depth_estimation_summary.json")
    bscale = dep["baseline_bbox_scale_prior"]
    ray = dep["ground_plane_raycast_only"]
    hyb = dep["track_l_hybrid_raycast_plus_scale"]
    assert ray["abs_rel"] < 0.10 and ray["abs_rel"] < bscale["abs_rel"]
    assert hyb["abs_rel"] < 0.10 and hyb["rmse_m"] < 10.0
    assert hyb["delta_1_pct"] == 100.0
    print("  [PASS] Track I-L JSON summaries verified cleanly.")


def test_pitch_geometry_round_trip_and_kit_classifier() -> None:
    """Verify exact inverse homography math and 4-role HSV kit classification."""
    params = np.array([0.0, 0.0, 0.0, 2200.0, 2.22, -50.29, 14.45], dtype=np.float64)
    H = camera_params_to_homography(params, width=1280, height=720)
    world_pts = np.array([[0.0, 0.0], [36.0, 0.0], [16.5, -10.0]], dtype=np.float64)
    uv, valid = project_world_to_image(H, world_pts)
    assert bool(np.all(valid)), f"Expected all test points in front of camera: {valid}"
    recovered = project_image_to_world(H, uv)
    max_err = float(np.max(np.abs(recovered - world_pts)))
    assert max_err < 1e-6, f"Homography round-trip error too high: {max_err}"

    # Verify all 4 kit roles with clean HSV patches
    def _patch(h: int, s: int, v: int) -> np.ndarray:
        arr = np.zeros((24, 16, 3), dtype=np.uint8)
        arr[:, :, 0] = h
        arr[:, :, 1] = s
        arr[:, :, 2] = v
        return arr

    home_role, _ = classify_kit_role(_patch(2, 210, 200))
    away_role, _ = classify_kit_role(_patch(24, 210, 210))
    gk_role, _ = classify_kit_role(_patch(11, 190, 210))
    ref_role, _ = classify_kit_role(_patch(90, 200, 190))
    assert home_role == "Home Team", f"Got {home_role}"
    assert away_role == "Away Team", f"Got {away_role}"
    assert gk_role == "Goalkeeper", f"Got {gk_role}"
    assert ref_role == "Referee", f"Got {ref_role}"
    print(f"  [PASS] Homography round-trip ({max_err:.2e} m) & 4-role kit classifier verified.")


def test_real_frame_c3_2949_detection() -> None:
    """Verify real-frame turf masking and 12-player pitch detection on Game 0 chunk 03."""
    chunk_path = ROOT / "gate2_chunks/game_0_half_1_chunk_03.mp4"
    if not chunk_path.exists():
        print("  [SKIP] Cached Game 0 chunk 03 MP4 absent; skipping real-frame test.")
        return

    cap = cv2.VideoCapture(str(chunk_path))
    cap.set(cv2.CAP_PROP_POS_MSEC, 35.0 * 1000.0)
    ok, bgr = cap.read()
    cap.release()
    assert ok and bgr is not None and bgr.shape == (720, 1280, 3)

    turf_mask, _, turf_cov = extract_turf_and_chalk_masks(bgr)
    assert turf_cov > 50.0, f"Low turf coverage on c3_2949: {turf_cov}"

    dets = detect_players_on_pitch(bgr, field_mask=turf_mask)
    roles = {d["role"] for d in dets}
    assert len(dets) == 12, f"Expected 12 detections on c3_2949, got {len(dets)}"
    for expected_role in ("Home Team", "Away Team", "Goalkeeper", "Referee"):
        assert expected_role in roles, f"Missing {expected_role} in c3_2949 roles: {roles}"
    print(f"  [PASS] Real frame c3_2949 detection verified (12 players, roles={sorted(roles)}).")


def test_reval_normalization_and_benchmark_integrity() -> None:
    """Verify ReVal frequency-normalized cross-sport math and benchmark JSON invariants."""
    assert CANONICAL_GAME_FREQUENCY_ANCHOR == 10.0

    # Rare goal vs high-frequency basketball field goal parity check
    soccer_goal_w = compute_normalized_action_weight("Goal", "soccer", normalize_frequency=True)
    fb_td_w = compute_normalized_action_weight("Touchdown", "football", normalize_frequency=True)
    bk_fg_w = compute_normalized_action_weight("Two-Point FG / Dunk", "basketball", normalize_frequency=True)
    assert soccer_goal_w > fb_td_w > bk_fg_w > 0.0

    bench = _load_json("logs/reval_highlights/reval_comprehensive_benchmark.json")
    b0 = bench["benchmark_0_cross_sport_normalization"]
    assert b0["cross_sport_cv_normalized_pct"] < b0["cross_sport_cv_unnormalized_pct"]
    assert b0["cv_reduction_pct"] > 50.0
    assert b0["uncapped_cv_reduction_pct"] > 70.0

    b1 = bench["benchmark_1_multimodal_attribution_ablation"]
    strict = b1["mes5_strict_high_stakes"]
    assert strict["fused_multimodal"]["f1_pct"] > strict["track_g_asr_ner_only"]["f1_pct"]
    assert strict["track_g_asr_ner_only"]["f1_pct"] > strict["track_h_ocr_only"]["f1_pct"]

    b2 = bench["benchmark_2_adaptive_dsp_highlight_splicing"]
    fixed = b2["fixed_window_t_minus4_to_plus6"]
    adapt = b2["adaptive_acoustic_optical_splicer"]
    assert adapt["action_apex_containment_pct"] > fixed["action_apex_containment_pct"]
    assert adapt["cut_boundary_cleanliness_pct"] > fixed["cut_boundary_cleanliness_pct"]
    assert adapt["mean_tiou_pct"] >= fixed["mean_tiou_pct"]

    b3 = bench["benchmark_3_construct_validity_and_sensitivity"]
    assert b3["spearman_rho"] > 0.70 and b3["spearman_p_value"] < 0.05
    assert b3["kendall_tau"] >= 0.50 and b3["kendall_p_value"] < 0.05
    assert b3["top5_mean_jaccard_stability_pct"] >= 80.0

    b4 = bench["benchmark_4_system_engineering_and_wcag"]
    assert b4["pyav_720p_transcode"]["throughput_fps"] > 25.0
    assert b4["pyav_720p_transcode"]["realtime_capable"] is True
    assert len(b4["av_packet_sync_audit"]) == 3
    for av_row in b4["av_packet_sync_audit"]:
        assert av_row["has_audio"] is True, f"Audio missing in {av_row['file']}"
        assert av_row["sync_within_25ms"] is True, f"A/V sync drift in {av_row}"
    for name, ratio in b4["wcag_21_contrast_ratios"].items():
        assert ratio >= 4.5, f"WCAG AA failure on {name}: {ratio}"
    print("  [PASS] ReVal cross-sport normalization & 5-benchmark integrity verified.")


def run_all_tests() -> None:
    print("Running v5.0 Tracks I-L & ReVal benchmark verification suite...")
    test_track_i_through_l_summaries()
    test_pitch_geometry_round_trip_and_kit_classifier()
    test_real_frame_c3_2949_detection()
    test_reval_normalization_and_benchmark_integrity()
    print("ALL V5 TRACKS SUITE CHECKS PASSED.")


if __name__ == "__main__":
    run_all_tests()
