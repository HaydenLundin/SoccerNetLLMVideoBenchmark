"""
ReVal Athlete Highlight Reel Generator and Attribution Pipeline.
Fuses Track A Action Spotting, Track F Camera Perspectives, Track G Commentary Entity Resolution,
and Track H Jersey OCR into:
1. Structured Athlete Valuation Leaderboard (logs/reval_highlights/reval_mes5_athlete_leaderboard.json).
2. Broadcast Highlight Video Reels with ReVal Brand HUD Overlay (#003153 Prussian Blue / #FF9E1B Primary Orange).
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from processing.reval_scoring import (
    ATHLETE_REGISTRY,
    attribute_events_to_athletes,
    compute_reval_composite
)

OUTPUT_DIR = WORKSPACE_ROOT / "logs" / "reval_highlights"
GATE2_DIR = WORKSPACE_ROOT / "gate2_chunks"


# ---------------------------------------------------------------------------
# 1. ReVal Branded Broadcast Telemetry HUD Overlay
# ---------------------------------------------------------------------------

def render_reval_hud_overlay(
    image: Image.Image,
    athlete_meta: Dict[str, Any],
    event_meta: Dict[str, Any],
    rel_playback_sec: float
) -> Image.Image:
    """
    Renders ReVal's signature broadcast lower-third HUD overlay using brand colors:
    - Primary Bar: Prussian Blue #003153 (rgb(0, 49, 83))
    - Accent Highlights: Primary Orange #FF9E1B (rgb(255, 158, 27))
    """
    img = image.copy()
    draw = ImageDraw.Draw(img, "RGBA")
    w, h = img.size

    # Banner dimensions (occupies bottom 70px)
    hud_h = 74
    hud_top = h - hud_h - 10
    hud_left = 16
    hud_right = w - 16

    # 1. Primary Orange Accent Top Border (4px)
    draw.rectangle(
        [(hud_left, hud_top), (hud_right, hud_top + 4)],
        fill=(255, 158, 27, 255)  # #FF9E1B
    )

    # 2. Prussian Blue Main Telemetry Bar (#003153)
    draw.rectangle(
        [(hud_left, hud_top + 4), (hud_right, hud_top + hud_h)],
        fill=(0, 49, 83, 235)  # #003153 with high opacity
    )

    # Try font loading, fallback to default
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    # 3. ReVal Pill Badge (Left)
    reval_score = athlete_meta.get("reval_score", 85.0)
    badge_w = 145
    badge_h = 24
    badge_x = hud_left + 12
    badge_y = hud_top + 12

    # Orange Pill
    draw.rectangle(
        [(badge_x, badge_y), (badge_x + badge_w, badge_y + badge_h)],
        fill=(255, 158, 27, 255)
    )
    score_text = f"REVAL SCORE: {reval_score:.1f}"
    draw.text((badge_x + 8, badge_y + 6), score_text, fill=(0, 49, 83), font=font)

    # 4. Athlete Name, Jersey, and Team Line
    name_str = f"#{athlete_meta.get('jersey', '00')} {athlete_meta.get('name', 'Athlete').upper()}"
    team_str = f"|  {athlete_meta.get('team', 'Club')}  ({athlete_meta.get('position', 'Pro')})"
    draw.text((badge_x + badge_w + 14, badge_y + 3), name_str, fill=(255, 255, 255), font=font)
    draw.text((badge_x + badge_w + 160, badge_y + 3), team_str, fill=(148, 163, 184), font=font)

    # 5. NIL Valuation Multiplier Badge (Right)
    nil_mult = athlete_meta.get("nil_multiplier", "3.50x")
    tier = athlete_meta.get("tier", "ELITE TIER")
    tier_str = f"{tier} ({nil_mult} MEDIA MULTIPLIER)"
    draw.text((hud_right - 230, badge_y + 3), tier_str, fill=(255, 158, 27), font=font)

    # 6. Action Telemetry Sub-line (Bottom)
    action_lbl = event_meta.get("label", "Key Action")
    crowd_db = event_meta.get("crowd_surge_db", 10.0)
    cam_type = event_meta.get("camera_type", "Close-Up")
    desc = event_meta.get("description", "Action spot in match window.")

    sub_y = badge_y + 30
    action_info = f"EVENT: {action_lbl}  |  CROWD SURGE: +{crowd_db:.1f} dB  |  CAMERA: {cam_type}  |  {desc[:65]}"
    draw.text((hud_left + 14, sub_y), action_info, fill=(226, 232, 240), font=font)

    # 7. Top-Right Time Badge
    time_badge = f"[+{rel_playback_sec:.1f}s]"
    draw.text((hud_right - 60, sub_y), time_badge, fill=(255, 158, 27), font=font)

    return img


# ---------------------------------------------------------------------------
# 2. Automated Highlight Reel Splicer via PyAV
# ---------------------------------------------------------------------------

def generate_athlete_highlight_reel(
    athlete_id: str,
    athlete_info: Dict[str, Any],
    output_path: Path,
    lead_in_sec: float = 4.0,
    lead_out_sec: float = 6.0
) -> Optional[Path]:
    """
    Slices video segments around athlete's key actions from gate2_chunks/ and
    assembles an MP4 highlight reel with ReVal broadcast lower-third HUD overlays.
    """
    try:
        import av
    except ImportError:
        print("Error: PyAV ('av') is required for highlight reel generation.")
        return None

    events = athlete_info.get("attributed_events", [])
    if not events:
        print(f"No events found for athlete {athlete_id}")
        return None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n--- Splicing Highlight Reel for {athlete_info['name']} (#{athlete_info['jersey']}) ---")

    # Select the top event for highlight rendering
    target_event = events[0]
    g_id = target_event["game_id"]
    half = target_event["half"]
    c_idx = target_event["chunk_index"]
    event_sec = target_event["rel_sec"]

    # Locate source chunk file
    mf_path = GATE2_DIR / f"game_{g_id}_half_{half}_chunks_manifest.json"
    if not mf_path.exists():
        print(f"Manifest not found for Game {g_id} Half {half}")
        return None

    with open(mf_path, "r", encoding="utf-8") as f:
        mf = json.load(f)

    target_chunk = None
    for c in mf.get("chunks", []):
        if c["chunk_index"] == c_idx:
            target_chunk = c
            break

    if not target_chunk:
        print(f"Chunk {c_idx} not found in manifest")
        return None

    chunk_file = WORKSPACE_ROOT / target_chunk["relative_path"]
    if not chunk_file.exists():
        print(f"Video file missing: {chunk_file}")
        return None

    chunk_dur = target_chunk["duration_sec"]
    slice_start = max(0.0, event_sec - lead_in_sec)
    slice_end = min(chunk_dur, event_sec + lead_out_sec)
    total_slice_dur = slice_end - slice_start

    print(f"Slicing clip [{slice_start:.1f}s -> {slice_end:.1f}s] ({total_slice_dur:.1f}s) from {chunk_file.name}...")

    # Decode frames in range
    container = av.open(str(chunk_file))
    video_stream = container.streams.video[0]
    time_base = float(video_stream.time_base)

    processed_images = []
    fps_sample = 25.0
    frame_step = 1.0 / fps_sample
    next_t = slice_start

    for packet in container.demux(video_stream):
        for frame in packet.decode():
            if frame.pts is None:
                continue
            pts_sec = float(frame.pts) * time_base
            if pts_sec < slice_start:
                continue
            if pts_sec > slice_end:
                break

            if pts_sec >= next_t:
                img = frame.to_image()
                rel_t = pts_sec - slice_start
                # Render HUD overlay on frame
                hud_frame = render_reval_hud_overlay(img, athlete_info, target_event, rel_playback_sec=rel_t)
                processed_images.append(hud_frame)
                next_t += frame_step
        if next_t > slice_end:
            break
    container.close()

    if not processed_images:
        print("Failed to decode frames for reel.")
        return None

    # Write output video with PyAV
    out_container = av.open(str(output_path), mode="w")
    w, h = processed_images[0].size
    out_stream = out_container.add_stream("libx264", rate=25)
    out_stream.width = w
    out_stream.height = h
    out_stream.pix_fmt = "yuv420p"

    for img in processed_images:
        v_frame = av.VideoFrame.from_image(img)
        for packet in out_stream.encode(v_frame):
            out_container.mux(packet)

    for packet in out_stream.encode():
        out_container.mux(packet)
    out_container.close()

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"✓ Highlight Reel exported: {output_path.name} ({len(processed_images)} frames, {size_mb:.2f} MB)")
    return output_path


# ---------------------------------------------------------------------------
# 3. Master Pipeline Runner
# ---------------------------------------------------------------------------

def run_reval_attribution_and_highlight_pipeline(generate_video_reels: bool = True) -> Dict[str, Any]:
    """
    Executes complete ReVal attribution, scoring, leaderboard generation,
    and video highlight reel rendering across MES-5.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print(f"\n========================================================")
    print(f" ReVal Athlete Valuation & Highlight Generation Pipeline")
    print(f" Target Output: {OUTPUT_DIR}")
    print(f"========================================================")

    # 1. Attribute events & compute 3-pillar scores
    leaderboard = attribute_events_to_athletes(predictions_by_game={})

    # Sort leaderboard by ReVal Score descending
    sorted_athletes = sorted(leaderboard.values(), key=lambda a: a["reval_score"], reverse=True)

    # 2. Save structured JSON leaderboard
    leaderboard_file = OUTPUT_DIR / "reval_mes5_athlete_leaderboard.json"
    with open(leaderboard_file, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "benchmark": "SoccerNet MES-5 & Cross-Sport",
            "scoring_formula": "ReVal = 0.45*Action + 0.30*Exposure + 0.25*Resonance",
            "total_athletes": len(sorted_athletes),
            "leaderboard": sorted_athletes
        }, f, indent=2)
    print(f"✓ Leaderboard saved to: {leaderboard_file.name} ({len(sorted_athletes)} athletes ranked)")

    # 3. Generate sample highlight reels for marquee athletes
    sample_reels = []
    if generate_video_reels:
        # Generate reels for top 3 premier athletes: Herrera, Aguero, Terry
        target_ids = ["ander_herrera", "sergio_aguero", "john_terry"]
        for a in sorted_athletes:
            if a["id"] in target_ids:
                reel_name = f"{a['id']}_{a['jersey']}_reel.mp4"
                reel_path = OUTPUT_DIR / reel_name
                res = generate_athlete_highlight_reel(a["id"], a, reel_path)
                if res and res.exists():
                    sample_reels.append({
                        "athlete_id": a["id"],
                        "athlete_name": a["name"],
                        "reval_score": a["reval_score"],
                        "tier": a["tier"],
                        "nil_multiplier": a["nil_multiplier"],
                        "video_file": reel_name,
                        "video_url": f"/video/logs/reval_highlights/{reel_name}"
                    })

    elapsed = time.time() - t0
    print(f"\n========================================================")
    print(f" ReVal Pipeline Completed in {elapsed:.2f}s:")
    print(f" Athletes Ranked: {len(sorted_athletes)}")
    print(f" Sample Video Reels Created: {len(sample_reels)}")
    print(f"========================================================\n")

    return {
        "leaderboard": sorted_athletes,
        "sample_reels": sample_reels,
        "elapsed_sec": round(elapsed, 2)
    }


def main():
    parser = argparse.ArgumentParser(description="ReVal Athlete Valuation & Highlight Generation")
    parser.add_argument("--athlete-id", type=str, default=None, help="Generate for specific athlete ID")
    parser.add_argument("--no-video", action="store_true", help="Skip video highlight reel rendering")
    args = parser.parse_args()

    run_reval_attribution_and_highlight_pipeline(generate_video_reels=not args.no_video)


if __name__ == "__main__":
    main()
