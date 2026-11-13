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
from functools import lru_cache
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

@lru_cache(maxsize=16)
def _load_bold_font(size: int) -> Any:
    """Loads and caches a bold TrueType font across Windows, Linux, and macOS."""
    candidates = [
        # Windows
        r"C:\Windows\Fonts\arialbd.ttf",
        r"C:\Windows\Fonts\segoeuib.ttf",
        # Linux (Debian, Ubuntu, Arch, Fedora)
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/liberation-sans/LiberationSans-Bold.ttf",
        # macOS
        "/Library/Fonts/Arial Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        # Local / PATH lookup
        "DejaVuSans-Bold.ttf",
        "LiberationSans-Bold.ttf",
        "arialbd.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists() or candidate.endswith(".ttf"):
            try:
                return ImageFont.truetype(candidate, size=size)
            except Exception:
                pass
    try:
        return ImageFont.load_default(size=size)
    except Exception:
        return ImageFont.load_default()


def render_reval_hud_overlay(
    image: Image.Image,
    athlete_meta: Dict[str, Any],
    event_meta: Dict[str, Any],
    rel_playback_sec: float
) -> Image.Image:
    """
    Renders ReVal's signature broadcast lower-third HUD overlay using WCAG-verified colors:
    - Primary Bar: Prussian Blue #003153 (rgb(0, 49, 83))
    - Accent Highlights: Primary Orange #FF9E1B (rgb(255, 158, 27))
    - Badge Ink: Deep Navy #001020 (9.28:1 AAA on #FF9E1B)
    - Large Broadcast Typography (>= 20px bold at 720p): #FF9E1B on #003153 (6.49:1)
      meets WCAG 2.1 AAA Large/Bold Text (>= 4.5:1) and AA Normal Text strictly.
    """
    img = image.copy()
    draw = ImageDraw.Draw(img, "RGBA")
    w, h = img.size
    hd_mode = w >= 800

    # Banner dimensions
    hud_h = 88 if hd_mode else 74
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
        fill=(0, 49, 83, 238)  # #003153
    )

    font_lg = _load_bold_font(20 if hd_mode else 10)
    font_md = _load_bold_font(19 if hd_mode else 10)

    # 3. ReVal Pill Badge (Left)
    reval_score = athlete_meta.get("reval_score", 85.0)
    badge_w = 212 if hd_mode else 145
    badge_h = 30 if hd_mode else 24
    badge_x = hud_left + 12
    badge_y = hud_top + 12

    # Orange Pill with #001020 AAA ink (9.28:1)
    draw.rectangle(
        [(badge_x, badge_y), (badge_x + badge_w, badge_y + badge_h)],
        fill=(255, 158, 27, 255)
    )
    score_text = f"REVAL SCORE: {reval_score:.1f}"
    draw.text((badge_x + 10, badge_y + (4 if hd_mode else 6)), score_text, fill=(0, 16, 32), font=font_lg)

    # 4. Athlete Name, Jersey, and Team Line
    name_str = f"#{athlete_meta.get('jersey', '00')} {athlete_meta.get('name', 'Athlete').upper()}"
    team_str = f"|  {athlete_meta.get('team', 'Club')} ({athlete_meta.get('position', 'Pro')})"
    draw.text((badge_x + badge_w + 16, badge_y + (4 if hd_mode else 3)), name_str, fill=(255, 255, 255), font=font_lg)
    name_offset = 270 if hd_mode else 160
    draw.text((badge_x + badge_w + name_offset, badge_y + (5 if hd_mode else 3)), team_str, fill=(226, 232, 240), font=font_md)

    # 5. NIL Valuation Multiplier Badge (Right) — >= 20px bold (#FF9E1B on #003153 = 6.49:1 AAA-Large)
    nil_mult = athlete_meta.get("nil_multiplier", "3.50x")
    tier = athlete_meta.get("tier", "ELITE TIER")
    tier_str = f"{tier} ({nil_mult} NIL)"
    tier_x = (hud_right - 340) if hd_mode else (hud_right - 230)
    draw.text((tier_x, badge_y + (4 if hd_mode else 3)), tier_str, fill=(255, 158, 27), font=font_lg)

    # 6. Action Telemetry Sub-line (Bottom)
    action_lbl = event_meta.get("label", "Key Action")
    crowd_db = event_meta.get("crowd_surge_db", 10.0)
    cam_type = event_meta.get("camera_type", "Close-Up")
    desc = event_meta.get("description", "Action spot in match window.")

    sub_y = badge_y + (40 if hd_mode else 30)
    action_info = f"EVENT: {action_lbl} | CROWD: +{crowd_db:.1f} dB | CAM: {cam_type} | {desc[:64]}"
    draw.text((hud_left + 14, sub_y), action_info, fill=(226, 232, 240), font=font_md)

    # 7. Bottom-Right Time Badge — >= 20px bold (#FF9E1B on #003153 = 6.49:1 AAA-Large)
    time_badge = f"[+{rel_playback_sec:.1f}s]"
    time_x = (hud_right - 95) if hd_mode else (hud_right - 60)
    draw.text((time_x, sub_y), time_badge, fill=(255, 158, 27), font=font_lg)

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

    # Decode synchronized video + AAC audio packets in range
    container = av.open(str(chunk_file))
    video_stream = container.streams.video[0]
    audio_stream = container.streams.audio[0] if container.streams.audio else None
    v_time_base = float(video_stream.time_base)

    # Fast keyframe seek slightly ahead of slice_start to avoid decoding entire chunk
    container.seek(int(max(0.0, slice_start - 1.5) * av.time_base), any_frame=False, backward=True)

    processed_images = []
    fps_sample = 25.0
    frame_step = 1.0 / fps_sample
    next_t = slice_start
    first_v_pts_sec: Optional[float] = None

    a_rate = int(audio_stream.rate) if audio_stream is not None else 44100
    a_time_base = float(audio_stream.time_base) if audio_stream is not None else (1.0 / a_rate)
    resampler = av.AudioResampler(format="fltp", layout="stereo", rate=a_rate) if audio_stream is not None else None
    audio_chunks: List[Tuple[float, np.ndarray]] = []

    demux_streams = [video_stream, audio_stream] if audio_stream is not None else [video_stream]
    for packet in container.demux(demux_streams):
        if packet.dts is None and packet.pts is None:
            continue
        if packet.stream.type == "video":
            for frame in packet.decode():
                if frame.pts is None:
                    continue
                pts_sec = float(frame.pts) * v_time_base
                if pts_sec < slice_start or pts_sec > slice_end + frame_step:
                    continue
                if pts_sec >= next_t - 1e-4:
                    if first_v_pts_sec is None:
                        first_v_pts_sec = pts_sec
                    rel_t = pts_sec - slice_start
                    hud_frame = render_reval_hud_overlay(frame.to_image(), athlete_info, target_event, rel_playback_sec=rel_t)
                    processed_images.append(hud_frame)
                    next_t += frame_step
        elif audio_stream is not None and packet.stream.type == "audio" and resampler is not None:
            for a_frame in packet.decode():
                if a_frame.pts is None:
                    continue
                at0 = float(a_frame.pts) * a_time_base
                at1 = at0 + (a_frame.samples / float(a_rate))
                if at1 < slice_start - 0.1 or at0 > slice_end + 0.15:
                    continue
                for r_frame in resampler.resample(a_frame):
                    audio_chunks.append((at0, r_frame.to_ndarray()))

        audio_done = (audio_stream is None) or (len(audio_chunks) > 0 and audio_chunks[-1][0] > slice_end + 0.1)
        if next_t > slice_end + frame_step and audio_done:
            break
    container.close()

    if not processed_images or first_v_pts_sec is None:
        print("Failed to decode frames for reel.")
        return None

    # Write synchronized H.264 + AAC output with PyAV
    out_container = av.open(str(output_path), mode="w")
    w, h = processed_images[0].size
    out_v_stream = out_container.add_stream("libx264", rate=25)
    out_v_stream.width = w
    out_v_stream.height = h
    out_v_stream.pix_fmt = "yuv420p"
    out_v_stream.options = {"preset": "veryfast", "crf": "23"}

    out_a_stream = None
    pcm_stereo = None
    n_target_samples = 0
    if audio_stream is not None and audio_chunks:
        out_a_stream = out_container.add_stream("aac", rate=a_rate, layout="stereo")
        target_dur = len(processed_images) / fps_sample
        n_target_samples = int(round(target_dur * a_rate))
        pcm_stereo = np.zeros((2, n_target_samples), dtype=np.float32)
        for at0, arr in audio_chunks:
            s0 = int(round((at0 - first_v_pts_sec) * a_rate))
            s1 = s0 + arr.shape[1]
            d0, d1 = max(0, s0), min(n_target_samples, s1)
            if d1 > d0:
                pcm_stereo[:, d0:d1] = arr[:2, (d0 - s0):(d1 - s0)]

    for idx, img in enumerate(processed_images):
        v_frame = av.VideoFrame.from_image(img)
        v_frame.pts = idx
        for packet in out_v_stream.encode(v_frame):
            out_container.mux(packet)

    if out_a_stream is not None and pcm_stereo is not None:
        for pos in range(0, n_target_samples, 1024):
            seg = np.ascontiguousarray(pcm_stereo[:, pos:pos + 1024])
            a_out = av.AudioFrame.from_ndarray(seg, format="fltp", layout="stereo")
            a_out.sample_rate = a_rate
            a_out.pts = pos
            for packet in out_a_stream.encode(a_out):
                out_container.mux(packet)

    for packet in out_v_stream.encode():
        out_container.mux(packet)
    if out_a_stream is not None:
        for packet in out_a_stream.encode():
            out_container.mux(packet)
    out_container.close()

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"✓ Highlight Reel exported: {output_path.name} ({len(processed_images)} frames, audio={out_a_stream is not None}@{a_rate}Hz, {size_mb:.2f} MB)")
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
