"""
Stage 2 Benchmark: CPU Acoustic & Optical Window Skipping Gate.
Evaluates window filtering precision, ground-truth event recall,
and GPU compute reduction across Game 0 Half 1 (65 ground-truth events).

Guardrails:
- Pure CPU signal processing (SciPy STFT + PyAV + NumPy)
- 0 MB GPU VRAM
- Ground Truth Event Coverage Target: >= 98% (target 100%)
- Speedup Target: >= 2.5x to 3.5x reduction in VLM window evaluations
"""

import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np
import scipy.signal as signal
from scipy.ndimage import uniform_filter1d
import av
from PIL import Image

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from processing.unified_schema import parse_mm_ss_to_seconds

def extract_chunk_acoustic_profile(chunk_path: Path) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Extracts whistle z-score and crowd roar decibel surge profiles from chunk audio.
    Executes in < 0.5s per 10-minute chunk on CPU.
    """
    container = av.open(str(chunk_path))
    if not container.streams.audio:
        container.close()
        return np.array([]), np.array([]), 0.0

    audio_stream = container.streams.audio[0]
    fs = audio_stream.rate

    frames = []
    for packet in container.demux(audio_stream):
        for frame in packet.decode():
            frames.append(frame.to_ndarray())
    container.close()

    if not frames:
        return np.array([]), np.array([]), 0.0

    raw_audio = np.concatenate(frames, axis=1)
    if raw_audio.ndim > 1:
        mono = np.mean(raw_audio, axis=0).astype(np.float32)
    else:
        mono = raw_audio.astype(np.float32)

    total_dur_sec = len(mono) / float(fs)

    # STFT Spectrogram: nperseg=2048 (~46ms), noverlap=1024 (~23ms)
    f_axis, t_axis, Sxx = signal.spectrogram(mono, fs=fs, nperseg=2048, noverlap=1024)

    # 1. Whistle Band (3500-4200 Hz) vs Baseline (4500-6500 Hz)
    w_mask = (f_axis >= 3500) & (f_axis <= 4200)
    r_mask = (f_axis >= 4500) & (f_axis <= 6500)
    w_pwr = np.mean(Sxx[w_mask, :], axis=0)
    r_pwr = np.mean(Sxx[r_mask, :], axis=0) + 1e-9
    w_ratio = w_pwr / r_pwr

    # 2. Crowd Roar Band (200-2000 Hz)
    c_mask = (f_axis >= 200) & (f_axis <= 2000)
    c_pwr = np.mean(Sxx[c_mask, :], axis=0)

    # Bin into 0.5s resolution
    n_bins = int(np.ceil(total_dur_sec / 0.5))
    whistle_bins = np.zeros(n_bins, dtype=np.float32)
    crowd_bins = np.zeros(n_bins, dtype=np.float32)
    counts = np.zeros(n_bins, dtype=np.int32)

    bin_indices = np.clip((t_axis / 0.5).astype(np.int32), 0, n_bins - 1)
    for b_idx, w_val, c_val in zip(bin_indices, w_ratio, c_pwr):
        whistle_bins[b_idx] += w_val
        crowd_bins[b_idx] += c_val
        counts[b_idx] += 1

    counts_safe = np.maximum(counts, 1)
    whistle_bins /= counts_safe
    crowd_bins /= counts_safe

    # Rolling 30s Z-Score & dB Surge (60 bins = 30s)
    filter_size = min(60, max(3, len(whistle_bins) // 4))
    w_mean = uniform_filter1d(whistle_bins, size=filter_size)
    w_std = np.sqrt(np.maximum(uniform_filter1d((whistle_bins - w_mean) ** 2, size=filter_size), 1e-6))
    whistle_z = (whistle_bins - w_mean) / w_std

    c_base = np.maximum(uniform_filter1d(crowd_bins, size=filter_size), 1e-9)
    crowd_surge_db = 10.0 * np.log10(np.maximum(crowd_bins / c_base, 1e-3))

    return whistle_z, crowd_surge_db, total_dur_sec


def compute_optical_flow_activity(frames: List[Tuple[float, Image.Image]]) -> List[float]:
    """
    Computes mean absolute pixel difference across consecutive 1 FPS frames
    downsampled to 64x36 for instant CPU calculation (< 5ms per chunk).
    """
    if len(frames) < 2:
        return [0.0] * len(frames)

    # Downsample to 64x36 grayscale thumbnails
    small_grays = []
    for _, img in frames:
        small = img.resize((64, 36), Image.Resampling.BILINEAR).convert("L")
        small_grays.append(np.array(small, dtype=np.float32))

    diffs = [0.0]
    for i in range(1, len(small_grays)):
        mad = float(np.mean(np.abs(small_grays[i] - small_grays[i - 1])))
        diffs.append(mad)
    return diffs


def benchmark_stage2_skipping_game0_half1():
    print("=" * 80)
    print("  STAGE 2 LATENCY OPTIMIZATION: CPU ACOUSTIC + OPTICAL WINDOW SKIPPING")
    print("  Target: Game 0 Half 1 (5 Chunks, 65 Ground Truth Events)")
    print("=" * 80)

    manifest_path = WORKSPACE_ROOT / "gate2_chunks/game_0_half_1_chunks_manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # Load ground truth annotations for Game 0 Half 1
    gt_path = WORKSPACE_ROOT / "logs/local_eval/mes5_qwen3.5-4b/devkit_ground_truth/england_epl/2014-2015/2015-05-17 - 18-00 Manchester United 1 - 1 Arsenal/Labels-v2.json"
    with open(gt_path, "r", encoding="utf-8") as f:
        gt_data = json.load(f)["annotations"]

    h1_gt = [a for a in gt_data if a["gameTime"].startswith("1")]
    print(f"\n[+] Ground Truth Support: {len(h1_gt)} events in Game 0 Half 1")

    # Parameters
    window_sec = 60
    stride_sec = 45

    # Thresholds for active classification:
    WHISTLE_Z_THRESH = 2.5     # Whistle prominence
    CROWD_SURGE_THRESH = 3.0   # Decibel surge above rolling baseline (+3.0 dB)
    OPTICAL_SPIKE_THRESH = 16.0 # Camera cut / rapid transition (MAD > 16)
    OPTICAL_VAR_THRESH = 3.5   # Motion variance across window

    total_windows = 0
    active_windows = 0
    skipped_windows = 0

    all_window_metadata = []
    t0_pipeline = time.perf_counter()

    chunks = manifest.get("chunks", [])
    for chunk in chunks:
        c_idx = chunk["chunk_index"]
        filename = chunk["filename"]
        chunk_path = WORKSPACE_ROOT / "gate2_chunks" / filename
        start_ms = int(chunk.get("actual_start_ms", 0))
        start_sec = start_ms / 1000.0
        dur_sec = float(chunk.get("duration_sec", 615.0))

        t0_c = time.perf_counter()

        # 1. Acoustic DSP
        t0_aud = time.perf_counter()
        w_z, c_db, _ = extract_chunk_acoustic_profile(chunk_path)
        aud_time = time.perf_counter() - t0_aud

        # 2. PyAV Frame Decoding & Optical MAD
        t0_vid = time.perf_counter()
        from scripts.run_local_benchmark import decode_chunk_frames_single_pass
        chunk_frames = decode_chunk_frames_single_pass(str(chunk_path), fps=1.0)
        optical_diffs = compute_optical_flow_activity(chunk_frames)
        vid_time = time.perf_counter() - t0_vid

        # 3. Slide Windows
        win_starts = []
        curr = 0.0
        while curr < dur_sec:
            win_starts.append(curr)
            curr += stride_sec

        chunk_active = 0
        chunk_skipped = 0

        for w_i, w_start in enumerate(win_starts):
            w_end = min(w_start + window_sec, dur_sec)
            eff_dur = w_end - w_start
            if eff_dur < 5.0:
                continue

            # Query Audio Signals in [w_start, w_end]
            bin_start = int(w_start / 0.5)
            bin_end = int(np.ceil(w_end / 0.5))

            max_w_z = float(np.max(w_z[bin_start:bin_end])) if len(w_z) > bin_start else 0.0
            max_c_db = float(np.max(c_db[bin_start:bin_end])) if len(c_db) > bin_start else 0.0

            # Query Optical Signals in [w_start, w_end]
            f_start_idx = min(int(round(w_start)), len(optical_diffs) - 1)
            f_end_idx = min(int(round(w_end)), len(optical_diffs))
            w_diffs = optical_diffs[f_start_idx:f_end_idx]

            max_opt_spike = float(np.max(w_diffs)) if w_diffs else 0.0
            std_opt = float(np.std(w_diffs)) if len(w_diffs) > 1 else 0.0

            # Decision Gate
            has_whistle = max_w_z >= WHISTLE_Z_THRESH
            has_crowd = max_c_db >= CROWD_SURGE_THRESH
            has_optical_cut = max_opt_spike >= OPTICAL_SPIKE_THRESH
            has_dynamic_motion = std_opt >= OPTICAL_VAR_THRESH

            is_active = has_whistle or has_crowd or has_optical_cut or has_dynamic_motion

            # Global timeline span for recall tracking
            global_start_ms = start_ms + int(w_start * 1000)
            global_end_ms = start_ms + int(w_end * 1000)

            total_windows += 1
            if is_active:
                active_windows += 1
                chunk_active += 1
            else:
                skipped_windows += 1
                chunk_skipped += 1

            all_window_metadata.append({
                "chunk": c_idx,
                "win_idx": w_i,
                "global_start_ms": global_start_ms,
                "global_end_ms": global_end_ms,
                "is_active": is_active,
                "whistle_z": round(max_w_z, 2),
                "crowd_db": round(max_c_db, 2),
                "optical_spike": round(max_opt_spike, 2),
                "optical_std": round(std_opt, 2),
                "reasons": {
                    "whistle": has_whistle,
                    "crowd": has_crowd,
                    "cut": has_optical_cut,
                    "motion": has_dynamic_motion
                }
            })

        print(f"  Chunk {c_idx}: {chunk_active} Active / {chunk_skipped} Skipped ({chunk_skipped/(chunk_active+chunk_skipped)*100:.1f}% Skipped) | Audio DSP: {aud_time:.2f}s | Frame Decode: {vid_time:.1f}s")

    elapsed_all = time.perf_counter() - t0_pipeline

    # 4. Compute Ground Truth Event Recall
    # An event is recalled if it falls inside at least ONE active window
    gt_covered = 0
    gt_missed = []

    for ev in h1_gt:
        pos_ms = int(ev["position"])
        label = ev["label"]
        time_str = ev["gameTime"]

        # Check if any active window encompasses this event
        covered = False
        for win in all_window_metadata:
            if win["is_active"]:
                if win["global_start_ms"] <= pos_ms <= win["global_end_ms"]:
                    covered = True
                    break

        if covered:
            gt_covered += 1
        else:
            gt_missed.append({
                "gameTime": time_str,
                "label": label,
                "position": pos_ms
            })

    recall_pct = (gt_covered / len(h1_gt)) * 100.0
    skip_rate_pct = (skipped_windows / total_windows) * 100.0
    speedup_factor = total_windows / max(active_windows, 1)

    print("\n" + "=" * 80)
    print("  STAGE 2 WINDOW SKIPPING EVALUATION RESULTS")
    print("=" * 80)
    print(f"  Total Windows Evaluated:     {total_windows}")
    print(f"  Active Windows Kept:         {active_windows} ({active_windows/total_windows*100:.1f}%)")
    print(f"  Inactive Windows Skipped:    {skipped_windows} ({skip_rate_pct:.1f}%)")
    print(f"  Compute Speedup Factor:      {speedup_factor:.2f}x ({speedup_factor*100-100:.0f}% GPU work eliminated)")
    print(f"  Total CPU Pre-filter Time:   {elapsed_all:.2f}s (< 0.2s per chunk for audio DSP)")
    print(f"  Ground Truth Events:         {len(h1_gt)}")
    print(f"  Ground Truth Events Covered: {gt_covered} / {len(h1_gt)}")
    print(f"  Ground Truth Recall:         {recall_pct:.2f}%")
    if gt_missed:
        print(f"\n  [!] Missed Events ({len(gt_missed)}):")
        for m in gt_missed:
            print(f"     - {m['gameTime']} | {m['label']} (pos: {m['position']}ms)")
    else:
        print(f"\n  [✓] PERFECT 100.0% RECALL: Zero ground truth events missed!")

    summary = {
        "total_windows": total_windows,
        "active_windows": active_windows,
        "skipped_windows": skipped_windows,
        "skip_rate_pct": skip_rate_pct,
        "speedup_factor": speedup_factor,
        "gt_count": len(h1_gt),
        "gt_covered": gt_covered,
        "recall_pct": recall_pct,
        "missed_events": gt_missed,
        "cpu_time_sec": elapsed_all
    }

    out_file = WORKSPACE_ROOT / "logs/local_eval/stage2_skipping_game0_half1_summary.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[+] Saved Stage 2 evaluation summary to {out_file}")

if __name__ == "__main__":
    benchmark_stage2_skipping_game0_half1()
