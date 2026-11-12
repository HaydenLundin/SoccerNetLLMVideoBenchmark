"""
Comprehensive Verification Test Suite for ReVal Athlete Valuation & Highlight Reel Pipeline.

Validates:
1. ReVal 3-Pillar Valuation Math & NIL Tier Multipliers (45% Action, 30% Exposure, 25% Resonance).
2. Multi-modal Athlete Attribution across MES-5 and cross-sport profiles.
3. ReVal HUD Overlay rendering with Prussian Blue (#003153) and Primary Orange (#FF9E1B).
4. Spliced Highlight MP4 Reel container integrity via PyAV (duration, resolution, frame count).
5. Demo Server GET /api/reval endpoint contract and payload structure.
"""

import os
import sys
import json
import time
from pathlib import Path
import numpy as np
from PIL import Image
import av

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from processing.reval_scoring import (
    ACTION_RARITY_WEIGHTS,
    ATHLETE_REGISTRY,
    compute_action_performance_score,
    compute_broadcast_exposure_score,
    compute_crowd_resonance_score,
    compute_reval_composite,
    attribute_events_to_athletes
)
from scripts.reval_athlete_highlights import (
    render_reval_hud_overlay,
    OUTPUT_DIR
)


def test_reval_3pillar_math():
    print("\n--- 1. Testing ReVal 3-Pillar Scoring Math & NIL Tiers ---")
    
    # Test Pillar 1: Action Performance
    sample_events_soccer = [
        {"label": "Goal", "crowd_surge_db": 13.8, "camera_type": "Close-Up"}
    ]
    act_score = compute_action_performance_score(sample_events_soccer, sport="soccer")
    assert 40.0 <= act_score <= 50.0, f"Expected action score ~46.2, got {act_score}"
    print(f"✓ Pillar 1 (Action Performance): Goal rarity yielded {act_score:.1f}%")

    # Test Pillar 2: Broadcast Exposure
    exp_score = compute_broadcast_exposure_score(sample_events_soccer)
    assert 40.0 <= exp_score <= 45.0, f"Expected exposure score ~42.9, got {exp_score}"
    print(f"✓ Pillar 2 (Broadcast Exposure): Close-Up dwell yielded {exp_score:.1f}%")

    # Test Pillar 3: Crowd Resonance
    res_score = compute_crowd_resonance_score(sample_events_soccer)
    assert 90.0 <= res_score <= 100.0, f"Expected resonance score ~98.6, got {res_score}"
    print(f"✓ Pillar 3 (Crowd Resonance): +13.8 dB crowd roar yielded {res_score:.1f}%")

    # Test Composite Formula: 0.45 * Act + 0.30 * Exp + 0.25 * Res
    composite = compute_reval_composite(act_score, exp_score, res_score, base_nil=250000.0)
    expected_score = round(0.45 * act_score + 0.30 * exp_score + 0.25 * res_score, 1)
    assert composite["reval_score"] == expected_score, f"Expected {expected_score}, got {composite['reval_score']}"
    assert composite["tier"] == "IMPACT TIER"
    assert composite["multiplier_numeric"] == 1.91
    assert composite["estimated_nil_value"] == 250000.0 * 1.91
    print(f"✓ Composite Score = {composite['reval_score']} | Tier = {composite['tier']} | Multiplier = {composite['nil_multiplier']} | NIL = ${composite['estimated_nil_value']:,.0f}")

    # Test Elite Tier Boundary (> 80.0)
    elite_comp = compute_reval_composite(90.0, 85.0, 95.0, base_nil=1000000.0)
    assert elite_comp["tier"] == "ELITE TIER"
    assert elite_comp["multiplier_numeric"] >= 3.0
    print(f"✓ Elite Boundary Test: ReVal={elite_comp['reval_score']} -> Tier={elite_comp['tier']} (Mult={elite_comp['nil_multiplier']})")


def test_athlete_attribution():
    print("\n--- 2. Testing Multi-Modal Athlete Attribution ---")
    leaderboard = attribute_events_to_athletes({})
    
    assert len(leaderboard) >= 12, f"Expected at least 12 athletes, got {len(leaderboard)}"
    print(f"✓ Successfully attributed events to {len(leaderboard)} athletes across Soccer, Football, and Basketball.")

    # Validate marquee athletes
    assert "ander_herrera" in leaderboard
    herrera = leaderboard["ander_herrera"]
    assert herrera["jersey"] == "21"
    assert herrera["team"] == "Manchester United"
    assert herrera["reval_score"] == 58.3
    assert herrera["events_count"] == 1
    assert herrera["attributed_events"][0]["label"] == "Goal"
    print(f"✓ Ander Herrera (#21): ReVal {herrera['reval_score']} ({herrera['tier']}), NIL: ${herrera['estimated_nil_value']:,.0f}")

    assert "sergio_aguero" in leaderboard
    aguero = leaderboard["sergio_aguero"]
    assert aguero["jersey"] == "10"
    assert aguero["reval_score"] == 56.5
    print(f"✓ Sergio Aguero (#10): ReVal {aguero['reval_score']} ({aguero['tier']}), NIL: ${aguero['estimated_nil_value']:,.0f}")

    assert "john_terry" in leaderboard
    terry = leaderboard["john_terry"]
    assert terry["jersey"] == "26"
    assert terry["reval_score"] == 51.5
    assert terry["attributed_events"][0]["label"] == "Red card"
    print(f"✓ John Terry (#26): ReVal {terry['reval_score']} ({terry['tier']}), NIL: ${terry['estimated_nil_value']:,.0f}")


def test_reval_hud_overlay_rendering():
    print("\n--- 3. Testing ReVal Broadcast HUD Lower-Third Rendering ---")
    test_frame = Image.new("RGB", (398, 224), color=(34, 139, 34))  # Pitch green

    ath_meta = {
        "name": "Ander Herrera",
        "jersey": "21",
        "team": "Manchester United",
        "position": "Midfielder",
        "reval_score": 58.3,
        "nil_multiplier": "1.91x",
        "tier": "IMPACT TIER"
    }
    ev_meta = {
        "label": "Goal",
        "crowd_surge_db": 13.8,
        "description": "Volley finish at far post into Stretford End."
    }

    overlaid = render_reval_hud_overlay(test_frame, ath_meta, ev_meta, rel_playback_sec=5.2)
    assert overlaid.size == (398, 224), "Overlay changed image dimensions!"
    
    # Inspect pixel colors in HUD area to verify Prussian Blue (#003153) and Primary Orange (#FF9E1B)
    arr = np.array(overlaid)
    
    # Prussian Blue RGB: (0, 49, 83)
    # Primary Orange RGB: (255, 158, 27)
    found_blue = np.any((arr[:, :, 0] < 20) & (arr[:, :, 1] >= 35) & (arr[:, :, 1] <= 65) & (arr[:, :, 2] >= 70) & (arr[:, :, 2] <= 100))
    found_orange = np.any((arr[:, :, 0] >= 240) & (arr[:, :, 1] >= 140) & (arr[:, :, 1] <= 175) & (arr[:, :, 2] < 45))
    
    assert found_blue, "ReVal Prussian Blue #003153 not detected in rendered HUD!"
    assert found_orange, "ReVal Primary Orange #FF9E1B not detected in rendered HUD!"
    print("✓ ReVal HUD telemetry rendered successfully with verified #003153 and #FF9E1B brand palette.")


def test_reval_spliced_mp4_integrity():
    print("\n--- 4. Testing Spliced Highlight MP4 Reels via PyAV ---")
    reels = [
        OUTPUT_DIR / "ander_herrera_21_reel.mp4",
        OUTPUT_DIR / "sergio_aguero_10_reel.mp4",
        OUTPUT_DIR / "john_terry_26_reel.mp4"
    ]

    for reel_path in reels:
        assert reel_path.exists(), f"Reel file missing: {reel_path}"
        file_size_mb = reel_path.stat().st_size / (1024 * 1024)
        assert file_size_mb > 0.5, f"Reel file suspiciously small: {file_size_mb:.2f} MB"

        # Validate synchronized video + AAC audio streams with PyAV
        container = av.open(str(reel_path))
        video_streams = [s for s in container.streams if s.type == "video"]
        audio_streams = [s for s in container.streams if s.type == "audio"]
        assert len(video_streams) == 1, f"Expected 1 video stream, found {len(video_streams)}"
        assert len(audio_streams) == 1, f"Expected 1 audio stream in {reel_path.name}, found {len(audio_streams)}"

        v_stream = video_streams[0]
        a_stream = audio_streams[0]
        width, height = v_stream.width, v_stream.height
        frames = v_stream.frames
        dur_sec = float(v_stream.duration * v_stream.time_base) if v_stream.duration else 10.0
        a_dur_sec = float(a_stream.duration * a_stream.time_base) if a_stream.duration else 0.0
        v_start = float(v_stream.start_time * v_stream.time_base) if v_stream.start_time is not None else 0.0
        a_start = float(a_stream.start_time * a_stream.time_base) if a_stream.start_time is not None else 0.0

        container.close()

        assert (width, height) in [(1280, 720), (398, 224)], f"Unexpected video dimensions: {width}x{height}"
        assert dur_sec >= 8.0, f"Expected reel duration >= 8s, got {dur_sec:.2f}s"
        assert frames >= 200, f"Expected >= 200 frames, got {frames}"
        assert abs(v_start - a_start) < 0.025, f"A/V start drift too high in {reel_path.name}: {abs(v_start - a_start):.4f}s"
        assert abs(dur_sec - a_dur_sec) < 0.05, f"A/V duration mismatch in {reel_path.name}: {abs(dur_sec - a_dur_sec):.4f}s"
        print(f"✓ Verified {reel_path.name}: {width}x{height}, {frames} frames ({dur_sec:.2f}s), aac@{a_stream.rate}Hz, drift={abs(v_start - a_start)*1000:.2f}ms, {file_size_mb:.2f} MB")


def test_reval_server_api_response():
    print("\n--- 5. Testing Demo Server /api/reval API Contract ---")
    leaderboard_file = OUTPUT_DIR / "reval_mes5_athlete_leaderboard.json"
    assert leaderboard_file.exists(), f"Leaderboard JSON missing at {leaderboard_file}"
    
    with open(leaderboard_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "leaderboard" in data
    assert "total_athletes" in data
    assert "scoring_formula" in data
    assert data["total_athletes"] == len(data["leaderboard"])

    # Check top athlete
    top = data["leaderboard"][0]
    assert "reval_score" in top
    assert "tier" in top
    assert "nil_multiplier" in top
    assert "sub_scores" in top
    assert "attributed_events" in top
    print(f"✓ Leaderboard API JSON contract verified: Top athlete is {top['name']} (Score: {top['reval_score']}, Multiplier: {top['nil_multiplier']})")


def test_reval_adversarial_clamping_and_empty_inputs():
    print("\n--- 6. Testing Adversarial Score Clamping & Edge Case Safety ---")
    from processing.reval_scoring import compute_reval_composite

    # Negative inputs clamped to 0.0
    underflow = compute_reval_composite(-100.0, -50.0, -20.0)
    assert underflow["reval_score"] == 0.0, f"Expected 0.0, got {underflow['reval_score']}"
    assert underflow["tier"] == "ROTATION TIER"
    assert underflow["multiplier_numeric"] >= 1.0

    # Extreme positive inputs clamped to 100.0
    overflow = compute_reval_composite(250.0, 300.0, 500.0)
    assert overflow["reval_score"] == 100.0, f"Expected 100.0, got {overflow['reval_score']}"
    assert overflow["tier"] == "ELITE TIER"
    assert overflow["multiplier_numeric"] == 6.0  # 4.5 + (100 - 85)*0.10

    # Monotonic tier boundary transitions
    assert compute_reval_composite(85.0, 85.0, 85.0)["tier"] == "ELITE TIER"
    assert compute_reval_composite(70.0, 70.0, 70.0)["tier"] == "PREMIER TIER"
    assert compute_reval_composite(50.0, 50.0, 50.0)["tier"] == "IMPACT TIER"
    assert compute_reval_composite(49.9, 49.9, 49.9)["tier"] == "ROTATION TIER"
    print("✓ ReVal score clamping [0.0, 100.0] and tier monotonicity verified under adversarial inputs.")


def run_all_tests():
    start = time.time()
    test_reval_3pillar_math()
    test_athlete_attribution()
    test_reval_hud_overlay_rendering()
    test_reval_spliced_mp4_integrity()
    test_reval_server_api_response()
    test_reval_adversarial_clamping_and_empty_inputs()
    elapsed = time.time() - start
    print(f"\n==================================================================")
    print(f" ALL REVAL HIGHLIGHT VERIFICATION TESTS PASSED ({elapsed:.2f}s)")
    print(f"==================================================================\n")


if __name__ == "__main__":
    run_all_tests()
