"""
Zero-Drift Macro-Chunk Slicer for SoccerNet Video Benchmark
Mitigates:
1. FFmpeg -c copy keyframe drift: Probes actual keyframe PTS around target start so clip_offset_sec is millisecond-accurate.
2. Container & Audio compatibility: Transcodes Vorbis audio to standard AAC (-c:a aac) while stream-copying video (-c:v copy).
3. Stoppage time truncation: Dynamically computes duration and extends the final chunk to EOF (45+/90+).
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

try:
    import static_ffmpeg
    static_ffmpeg.add_paths()
except ImportError:
    pass

CHUNK_DURATION_SEC = 600.0   # 10 minutes nominal
CHUNK_OVERLAP_SEC = 15.0     # 15 seconds overlap
CHUNK_STRIDE_SEC = CHUNK_DURATION_SEC - CHUNK_OVERLAP_SEC  # 585 seconds

def run_command(cmd: List[str]) -> str:
    """Executes a subprocess command and returns stdout."""
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"Command failed ({' '.join(cmd)}):\n{res.stderr}")
    return res.stdout.strip()

def get_video_duration(video_path: str) -> float:
    """Probes the total duration of a video file in seconds."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path
    ]
    out = run_command(cmd)
    return float(out)

def find_keyframe_at_or_before(video_path: str, target_sec: float) -> float:
    """
    Finds the exact keyframe (I-frame) presentation timestamp at or immediately before target_sec.
    Probes packets locally within [target-20s, target+10s] for high speed without full-file scan.
    """
    if target_sec <= 0.0:
        return 0.0

    start_window = max(0.0, target_sec - 20.0)
    read_interval = f"{start_window:.3f}%+30"

    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "packet=pts_time,flags",
        "-read_intervals", read_interval,
        "-of", "json",
        video_path
    ]
    try:
        out = run_command(cmd)
        data = json.loads(out)
        keyframes = [
            float(p["pts_time"]) for p in data.get("packets", [])
            if "K" in p.get("flags", "") and "pts_time" in p
        ]
        candidates = [k for k in keyframes if k <= target_sec]
        if candidates:
            return max(candidates)
        elif keyframes:
            return min(keyframes)
    except Exception as e:
        print(f"[WARN] Keyframe probe fallback: {e}")

    return target_sec

def slice_half_into_macro_chunks(
    input_video_path: str,
    output_dir: str,
    game_id: int,
    half: int,
    target_container: str = "mp4"
) -> List[Dict]:
    """
    Slices a 45+ minute match half into 10-minute macro-chunks with zero keyframe drift.
    Preserves video streams (-c:v copy) and encodes audio to AAC (-c:a aac).
    Ensures stoppage time (45+/90+) is never truncated by extending the final chunk to EOF.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    total_duration = get_video_duration(input_video_path)
    print(f"[SLICE] Slicing Half {half} | Total Duration: {total_duration:.2f}s ({total_duration/60:.2f} min)")

    chunks = []
    chunk_index = 0
    curr_target_start = 0.0

    while curr_target_start < total_duration:
        # Snap target start to the exact preceding keyframe PTS
        actual_slice_start = find_keyframe_at_or_before(input_video_path, curr_target_start)
        
        # Check if the next stride would reach/exceed total duration
        remaining_duration = total_duration - actual_slice_start
        is_final_chunk = remaining_duration <= (CHUNK_DURATION_SEC + CHUNK_OVERLAP_SEC * 2)

        chunk_filename = f"game_{game_id}_half_{half}_chunk_{chunk_index:02d}.{target_container}"
        chunk_filepath = out_path / chunk_filename

        cmd = ["ffmpeg", "-y"]
        if actual_slice_start > 0.0:
            cmd.extend(["-ss", f"{actual_slice_start:.3f}"])
        cmd.extend(["-i", input_video_path])

        if not is_final_chunk:
            # Slices 10m + 15s nominal duration
            cmd.extend(["-t", f"{CHUNK_DURATION_SEC + CHUNK_OVERLAP_SEC:.3f}"])
        else:
            # Final chunk runs until EOF to capture full stoppage time (45+ / 90+)
            print(f"   [CHUNK {chunk_index:02d}] FINAL CHUNK: Capturing until EOF to preserve stoppage time.")

        # Stream copy video, transcode Vorbis audio to standard AAC for Vertex AI compatibility
        cmd.extend(["-c:v", "copy", "-c:a", "aac", "-b:a", "128k", str(chunk_filepath)])
        run_command(cmd)

        chunk_actual_duration = get_video_duration(str(chunk_filepath))

        chunk_meta = {
            "game_id": game_id,
            "half": half,
            "chunk_index": chunk_index,
            "filename": chunk_filename,
            "relative_path": str(chunk_filepath.relative_to(out_path.parent) if out_path.parent else chunk_filename),
            "requested_start_sec": curr_target_start,
            "actual_start_sec": actual_slice_start,
            "actual_start_ms": int(round(actual_slice_start * 1000.0)),
            "keyframe_drift_sec": round(curr_target_start - actual_slice_start, 3),
            "duration_sec": chunk_actual_duration,
            "is_final_chunk": is_final_chunk
        }
        chunks.append(chunk_meta)
        print(f"   [CHUNK {chunk_index:02d}] Start: {actual_slice_start:.3f}s ({chunk_meta['actual_start_ms']} ms) | Drift: {chunk_meta['keyframe_drift_sec']:+.3f}s | Duration: {chunk_actual_duration:.1f}s -> {chunk_filename}")

        if is_final_chunk:
            break

        chunk_index += 1
        curr_target_start += CHUNK_STRIDE_SEC

    manifest_path = out_path / f"game_{game_id}_half_{half}_chunks_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({"game_id": game_id, "half": half, "total_duration_sec": total_duration, "chunks": chunks}, f, indent=2)

    print(f"[MANIFEST] Written to: {manifest_path}")
    return chunks

if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage: python slice_macro_chunks.py <input_video.mp4> <output_dir> <game_id> <half>")
        sys.exit(1)
    slice_half_into_macro_chunks(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
