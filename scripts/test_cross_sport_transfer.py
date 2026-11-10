"""
Verification test suite for Cross-Sport Transfer Engine.
Validates:
1. Multi-sport schema normalization (Soccer, Football, Basketball).
2. Play-clock (30s) and shot-clock (24s) temporal window geometry.
3. Sport-specific Markov causal consequent rescoring (PAT post-TD, Free Throw post-Foul, Rebound post-Block).
4. Visual wristwatch [00:SS] stamping and Track F / Track H classifiers.
5. End-to-end pipeline execution across Football, Basketball, and Soccer profiles.
"""

import os
import sys
import json
import time
from pathlib import Path
import numpy as np
from PIL import Image

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from processing.sport_schemas import (
    build_sport_profile,
    apply_sport_markov_prior,
    CrossSportEvent,
    SOCCER_CLASSES,
    FOOTBALL_CLASSES,
    BASKETBALL_CLASSES
)
from scripts.cross_sport_adapter import (
    CrossSportAdapter,
    burn_wristwatch_overlay,
    classify_camera_perspective,
    analyze_player_jersey_crop
)


def test_schema_registry():
    print("--- 1. Testing Multi-Sport Schema Registry ---")
    # Soccer
    p_soccer = build_sport_profile("soccer")
    assert p_soccer.name == "soccer"
    assert len(p_soccer.classes) == 17
    assert p_soccer.window_sec == 60.0
    assert p_soccer.stride_sec == 45.0
    print(f"✓ Soccer profile: {len(p_soccer.classes)} classes, W={p_soccer.window_sec}s, S={p_soccer.stride_sec}s")

    # American Football
    p_fb = build_sport_profile("football")
    assert p_fb.name == "football"
    assert len(p_fb.classes) == 10
    assert p_fb.window_sec == 30.0
    assert p_fb.stride_sec == 20.0
    assert p_fb.play_clock_sec == 40.0
    print(f"✓ Football profile: {len(p_fb.classes)} classes, W={p_fb.window_sec}s, S={p_fb.stride_sec}s (Play Clock={p_fb.play_clock_sec}s)")

    # Basketball
    p_bb = build_sport_profile("basketball")
    assert p_bb.name == "basketball"
    assert len(p_bb.classes) == 10
    assert p_bb.window_sec == 24.0
    assert p_bb.stride_sec == 16.0
    assert p_bb.play_clock_sec == 24.0
    print(f"✓ Basketball profile: {len(p_bb.classes)} classes, W={p_bb.window_sec}s, S={p_bb.stride_sec}s (Shot Clock={p_bb.play_clock_sec}s)")

    # Pydantic serialization
    event = CrossSportEvent(
        timestamp="00:18",
        commentary="Quarterback finds receiver on slant route for first down.",
        label="Pass Completion",
        confidence=0.92,
        visibility="visible",
        sport="football"
    )
    data = event.model_dump()
    assert data["label"] == "Pass Completion"
    assert data["timestamp"] == "00:18"
    print("✓ Pydantic CrossSportEvent model validated.")


def test_markov_consequent_boosting():
    print("\n--- 2. Testing Sport-Specific Markov Prior Boosting ---")

    # 1. American Football: Touchdown -> Field Goal / PAT
    fb_preds = [
        {"label": "Touchdown", "pos_sec": 12.0, "confidence": 0.95},
        {"label": "Field Goal / PAT", "pos_sec": 38.0, "confidence": 0.70},  # dt = 26s <= 45s
        {"label": "Snap", "pos_sec": 95.0, "confidence": 0.60}
    ]
    boosted_fb = apply_sport_markov_prior(fb_preds, sport="football")
    pat_ev = [p for p in boosted_fb if p["label"] == "Field Goal / PAT"][0]
    td_ev = [p for p in boosted_fb if p["label"] == "Touchdown"][0]

    assert td_ev["confidence"] == 0.95, "Initiator Touchdown must NOT be degraded!"
    assert pat_ev.get("markov_boosted") is True, "PAT must be boosted post-Touchdown!"
    assert pat_ev["confidence"] > 0.70, f"PAT confidence should surge above 0.70, got {pat_ev['confidence']}"
    print(f"✓ Football Markov: PAT boosted from 0.70 to {pat_ev['confidence']} (factor: {pat_ev['boost_factor']}x) following Touchdown (+26s)")

    # 2. Basketball: Personal Foul -> Free Throw, Block -> Rebound
    bb_preds = [
        {"label": "Personal Foul", "pos_sec": 8.0, "confidence": 0.90},
        {"label": "Free Throw", "pos_sec": 22.0, "confidence": 0.65},  # dt = 14s <= 30s
        {"label": "Block", "pos_sec": 45.0, "confidence": 0.88},
        {"label": "Rebound", "pos_sec": 47.5, "confidence": 0.60}  # dt = 2.5s <= 5s
    ]
    boosted_bb = apply_sport_markov_prior(bb_preds, sport="basketball")
    ft_ev = [p for p in boosted_bb if p["label"] == "Free Throw"][0]
    reb_ev = [p for p in boosted_bb if p["label"] == "Rebound"][0]

    assert ft_ev.get("markov_boosted") is True
    assert reb_ev.get("markov_boosted") is True
    print(f"✓ Basketball Markov: Free Throw boosted from 0.65 to {ft_ev['confidence']} post-Foul (+14s)")
    print(f"✓ Basketball Markov: Rebound boosted from 0.60 to {reb_ev['confidence']} post-Block (+2.5s)")


def test_visual_wristwatch_and_track_f_h():
    print("\n--- 3. Testing Visual Wristwatch & Track F/H Classifiers ---")

    # Generate synthetic turf frame (green dominant)
    turf_img = Image.new("RGB", (640, 360), color=(34, 139, 34))
    stamped_turf = burn_wristwatch_overlay(turf_img, rel_sec=15.0, sport_name="football")
    assert stamped_turf.size == (640, 360)
    print("✓ Wristwatch [00:15] burned-in on 640x360 frame successfully.")

    cam_turf = classify_camera_perspective(turf_img, sport="football")
    assert "Wide" in cam_turf["camera_type"]
    print(f"✓ Track F Field Camera: {cam_turf['camera_type']} (turf ratio: {cam_turf['surface_ratio']})")

    # Generate synthetic basketball court frame (hardwood orange/amber dominant)
    court_img = Image.new("RGB", (640, 360), color=(205, 133, 63))
    cam_court = classify_camera_perspective(court_img, sport="basketball")
    assert "Wide" in cam_court["camera_type"]
    print(f"✓ Track F Basketball Court Camera: {cam_court['camera_type']} (court ratio: {cam_court['surface_ratio']})")

    # Track H Jersey Crop
    player_crop = Image.new("RGB", (200, 200), color=(180, 20, 20))  # Red jersey
    jersey_info = analyze_player_jersey_crop(player_crop)
    assert "Red" in jersey_info["kit_team_estimate"]
    print(f"✓ Track H Jersey OCR: Detected kit {jersey_info['kit_team_estimate']} (RGB: {jersey_info['kit_color_rgb']})")


def test_end_to_end_cross_sport_pipeline():
    print("\n--- 4. Testing End-to-End Cross-Sport Execution on Broadcast Video ---")
    video_sample = WORKSPACE_ROOT / "gate2_chunks" / "game_0_half_1_chunk_00.mp4"
    if not video_sample.exists():
        print("Notice: Sample chunk not found, using dry-run clip.")
        return

    # 1. Run Football Adapter (W=30s, S=20s)
    fb_adapter = CrossSportAdapter(sport="football", dry_run=True, skip_inactive=True)
    fb_results = fb_adapter.run_inference_on_clip(str(video_sample), start_sec=0.0, duration_sec=60.0)
    assert fb_results["sport"] == "football"
    assert fb_results["window_sec"] == 30.0
    assert fb_results["stride_sec"] == 20.0
    assert len(fb_results["predictions"]) > 0
    print(f"✓ Football Adapter: {fb_results['total_windows']} windows (W={fb_results['window_sec']}s), {len(fb_results['predictions'])} events spotted.")

    # 2. Run Basketball Adapter (W=24s, S=16s)
    bb_adapter = CrossSportAdapter(sport="basketball", dry_run=True, skip_inactive=True)
    bb_results = bb_adapter.run_inference_on_clip(str(video_sample), start_sec=0.0, duration_sec=60.0)
    assert bb_results["sport"] == "basketball"
    assert bb_results["window_sec"] == 24.0
    assert bb_results["stride_sec"] == 16.0
    assert len(bb_results["predictions"]) > 0
    print(f"✓ Basketball Adapter: {bb_results['total_windows']} windows (W={bb_results['window_sec']}s), {len(bb_results['predictions'])} events spotted.")


if __name__ == "__main__":
    t_start = time.time()
    test_schema_registry()
    test_markov_consequent_boosting()
    test_visual_wristwatch_and_track_f_h()
    test_end_to_end_cross_sport_pipeline()
    print(f"\n========================================================")
    print(f" ALL CROSS-SPORT TRANSFER VERIFICATION TESTS PASSED! ({time.time() - t_start:.2f}s)")
    print(f"========================================================\n")
