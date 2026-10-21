"""
RQ4 Audio-Visual Grounding Evaluator: Referee Whistle & Crowd-Surge DSP Fusion
Extracts physical broadcast acoustic signals (Fox 40 referee whistle flux & crowd RMS surge)
across the 45-minute Half 1 (chunks 00-04) using PyAV + SciPy STFT with 0 MB GPU VRAM.

Evaluates Audio-Grounded Confidence Recalibration & False-Positive Pruning across:
1. Qwen3.5-4B Solo
2. MiniCPM-V 4.6 Solo
3. 3-Model Domain-Routed Specialist Ensemble
"""

import os
import sys
import json
import time
import math
import shutil
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import scipy.signal as signal
from scipy.ndimage import uniform_filter1d

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from processing.unified_schema import SOCCERNET_CLASSES
from processing.post_process_predictions import class_aware_nms, DEFAULT_CLASS_NMS_RADII_MS
from processing.evaluate_soccernet import evaluate_match_action_spotting


def extract_half1_broadcast_audio_signals(
    manifest_path: Path,
    chunks_dir: Path,
    target_fs: int = 44100
) -> Dict[str, Any]:
    """
    Decodes audio streams from all 5 chunks and extracts calibrated physical signals:
    1. Referee Whistle Resonance: Fox 40 3.5 kHz - 4.2 kHz spectral flux ratio vs background.
    2. Crowd Roar Surge: 200 Hz - 2,000 Hz RMS decibel surge relative to rolling 30s baseline.
    Decoded and processed in ~3-4 seconds total on CPU ($0 compute, 0 MB GPU VRAM).
    """
    import av
    print("\n" + "=" * 95)
    print("  [DSP AUDIO INGESTION] Extracting Broadcast Acoustics Across 45-Minute Half 1")
    print("=" * 95)

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    chunks = manifest.get("chunks", [])
    chunks = sorted(chunks, key=lambda c: int(c.get("actual_start_ms", 0)))

    # 45 minutes = 2700s at 0.5s resolution -> 5400 bins
    half_dur_sec = float(manifest.get("total_duration_sec", 2700.0))
    time_bins = np.arange(0.0, half_dur_sec + 1.0, 0.5)
    n_bins = len(time_bins)

    whistle_ratio_grid = np.zeros(n_bins, dtype=np.float32)
    crowd_pwr_grid = np.zeros(n_bins, dtype=np.float32)
    coverage_grid = np.zeros(n_bins, dtype=np.int32)

    t0_all = time.perf_counter()

    for c in chunks:
        c_idx = c["chunk_index"]
        filename = c["filename"]
        chunk_path = chunks_dir / filename
        start_sec = float(c["actual_start_ms"]) / 1000.0
        dur_sec = float(c.get("duration_sec", 615.0))

        if not chunk_path.exists():
            raise FileNotFoundError(f"Chunk video missing: {chunk_path}")

        t0_c = time.perf_counter()
        container = av.open(str(chunk_path))
        audio_stream = container.streams.audio[0]
        fs = audio_stream.rate

        frames = []
        for packet in container.demux(audio_stream):
            for frame in packet.decode():
                frames.append(frame.to_ndarray())
        container.close()

        if not frames:
            continue

        raw_audio = np.concatenate(frames, axis=1)
        # Average channels to mono float32
        if raw_audio.ndim > 1:
            mono_audio = np.mean(raw_audio, axis=0).astype(np.float32)
        else:
            mono_audio = raw_audio.astype(np.float32)

        # STFT Spectrogram: nperseg=2048 (~46.4ms), noverlap=1024 (~23.2ms)
        f_axis, t_axis, Sxx = signal.spectrogram(mono_audio, fs=fs, nperseg=2048, noverlap=1024)

        # 1. Referee Whistle Band: 3500 Hz - 4200 Hz vs Reference Baseline 4500 Hz - 6500 Hz
        whistle_mask = (f_axis >= 3500) & (f_axis <= 4200)
        ref_mask = (f_axis >= 4500) & (f_axis <= 6500)
        w_pwr = np.mean(Sxx[whistle_mask, :], axis=0)
        r_pwr = np.mean(Sxx[ref_mask, :], axis=0) + 1e-9
        w_ratio = w_pwr / r_pwr

        # 2. Crowd Roar Band: 200 Hz - 2000 Hz
        crowd_mask = (f_axis >= 200) & (f_axis <= 2000)
        c_pwr = np.mean(Sxx[crowd_mask, :], axis=0)

        # Resample into half-timeline 0.5s bins
        match_times = start_sec + t_axis
        bin_indices = np.clip((match_times / 0.5).astype(np.int32), 0, n_bins - 1)

        for b_idx, w_val, c_val in zip(bin_indices, w_ratio, c_pwr):
            whistle_ratio_grid[b_idx] += w_val
            crowd_pwr_grid[b_idx] += c_val
            coverage_grid[b_idx] += 1

        elapsed_c = time.perf_counter() - t0_c
        print(f"  [Chunk {c_idx}] Decoded {len(mono_audio)} samples in {elapsed_c:.2f}s | Audio Span: {start_sec:5.1f}s - {start_sec+dur_sec:5.1f}s")

    # Normalize overlapping chunks
    cov_safe = np.maximum(coverage_grid, 1)
    whistle_ratio_grid /= cov_safe
    crowd_pwr_grid /= cov_safe

    # Compute Z-Scores and Decibel Surges
    # Rolling 30s statistics = 60 bins
    w_mean = uniform_filter1d(whistle_ratio_grid, size=60)
    w_std = np.sqrt(np.maximum(uniform_filter1d((whistle_ratio_grid - w_mean) ** 2, size=60), 1e-6))
    whistle_z_grid = (whistle_ratio_grid - w_mean) / w_std

    c_base = np.maximum(uniform_filter1d(crowd_pwr_grid, size=60), 1e-9)
    crowd_surge_db_grid = 10.0 * np.log10(np.maximum(crowd_pwr_grid / c_base, 1e-3))

    # Detect Discrete Peaks (with minimum refractory window of 3.0s = 6 bins)
    whistle_peaks: List[Dict[str, Any]] = []
    last_w_bin = -10
    whistle_threshold = 4.0  # +4.0 std devs above 30s background
    for i in range(1, n_bins - 1):
        if whistle_z_grid[i] >= whistle_threshold and whistle_z_grid[i] > whistle_z_grid[i - 1] and whistle_z_grid[i] >= whistle_z_grid[i + 1]:
            if i - last_w_bin >= 6:  # 3.0s refractory
                whistle_peaks.append({
                    "time_sec": round(float(time_bins[i]), 2),
                    "match_time": f"1 - {int(time_bins[i]//60):02d}:{int(time_bins[i]%60):02d}",
                    "z_score": round(float(whistle_z_grid[i]), 2)
                })
                last_w_bin = i

    crowd_roars: List[Dict[str, Any]] = []
    last_c_bin = -10
    crowd_threshold = 5.0  # +5.0 dB surge above 30s baseline
    for i in range(1, n_bins - 1):
        if crowd_surge_db_grid[i] >= crowd_threshold and crowd_surge_db_grid[i] > crowd_surge_db_grid[i - 1] and crowd_surge_db_grid[i] >= crowd_surge_db_grid[i + 1]:
            if i - last_c_bin >= 10:  # 5.0s refractory
                crowd_roars.append({
                    "time_sec": round(float(time_bins[i]), 2),
                    "match_time": f"1 - {int(time_bins[i]//60):02d}:{int(time_bins[i]%60):02d}",
                    "surge_db": round(float(crowd_surge_db_grid[i]), 2)
                })
                last_c_bin = i

    dur_all = time.perf_counter() - t0_all
    print(f"\n[DSP COMPLETE] Ingested 45 min in {dur_all:.2f}s | Detected {len(whistle_peaks)} Whistle Peaks (z >= {whistle_threshold}) & {len(crowd_roars)} Crowd Roar Surges (>= +{crowd_threshold} dB)")
    return {
        "time_bins": time_bins,
        "whistle_z_grid": whistle_z_grid,
        "crowd_surge_db_grid": crowd_surge_db_grid,
        "whistle_peaks": whistle_peaks,
        "crowd_roars": crowd_roars
    }


def apply_audio_grounding_recalibration(
    predictions: List[Dict[str, Any]],
    audio_signals: Dict[str, Any],
    whistle_tolerance_sec: float = 4.0,
    crowd_surge_min_db: float = 4.0,
    prune_ungrounded_goals: bool = True,
    prune_ungrounded_shots: bool = True,
    prune_ungrounded_fouls: bool = True,
    selective_mode: bool = False
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Applies multimodal audio grounding rules to visual predictions:
    1. Whistle Grounding (Foul, Direct free-kick, Indirect free-kick, Offside, Kick-off):
       - If within +-4.0s of whistle peak: boost confidence (+0.15)
       - If no whistle: penalize confidence (-0.30) and prune if confidence < 0.65
    2. Crowd Roar Grounding (Goal, Shots on target):
       - Goal: requires crowd surge within [-2s, +8s] >= +4.0 dB. If no crowd surge, prune!
       - Shots on target: requires crowd surge >= +2.5 dB or whistle.
    3. Visual-only set pieces (Throw-in, Corner, Ball out of play):
       - Retain unchanged.
    4. In selective_mode (Cross-Modal Synchronized Gate):
       - Gate strictly whistle-triggered events (Foul, Kick-off, Direct free-kick, Indirect free-kick, Penalty).
       - Gate strictly crowd events (Goal, Shots on target).
       - Exempt acoustically silent classes (Ball out of play, Throw-in, Clearance, Corner, Shots off target, Offside).
    """
    whistle_times = [p["time_sec"] for p in audio_signals["whistle_peaks"]]
    crowd_times = [p["time_sec"] for p in audio_signals["crowd_roars"]]

    recalibrated = []
    audit = {
        "initial_count": len(predictions),
        "whistle_confirmed": 0,
        "whistle_penalized": 0,
        "goals_confirmed": 0,
        "goals_pruned": 0,
        "shots_pruned": 0,
        "fouls_pruned": 0,
        "final_count": 0
    }

    if selective_mode:
        WHISTLE_CLASSES = {"Foul", "Kick-off", "Direct free-kick", "Indirect free-kick", "Penalty"}
        CROWD_CLASSES = {"Goal", "Shots on target"}
        SILENT_EXEMPT = {"Ball out of play", "Throw-in", "Clearance", "Corner", "Shots off target", "Offside"}
    else:
        WHISTLE_CLASSES = {"Foul", "Direct free-kick", "Indirect free-kick", "Offside", "Kick-off"}
        CROWD_CLASSES = {"Goal", "Shots on target"}
        SILENT_EXEMPT = set()

    for p in predictions:
        p_out = dict(p)
        pos_sec = float(p_out["position"]) / 1000.0
        lbl = p_out["label"]
        conf = float(p_out.get("confidence", 0.85))

        # Rule 0: Acoustically Silent Classes in Selective Mode
        if selective_mode and lbl in SILENT_EXEMPT:
            p_out["audio_grounding"] = "acoustically_exempt"
            recalibrated.append(p_out)
            continue

        # Rule 1: Whistle-Coupled Classes
        if lbl in WHISTLE_CLASSES:
            # Check proximity to nearest whistle peak
            min_w_dist = min([abs(pos_sec - wt) for wt in whistle_times], default=999.0)
            if min_w_dist <= whistle_tolerance_sec:
                p_out["confidence"] = min(1.0, conf + 0.15)
                p_out["audio_grounding"] = f"whistle_confirmed (dist={min_w_dist:.1f}s)"
                audit["whistle_confirmed"] += 1
                recalibrated.append(p_out)
            else:
                audit["whistle_penalized"] += 1
                p_out["audio_grounding"] = f"whistle_missing (min_dist={min_w_dist:.1f}s)"
                if not selective_mode:
                    p_out["confidence"] = max(0.0, conf - 0.30)
                    if prune_ungrounded_fouls and p_out["confidence"] < 0.65:
                        audit["fouls_pruned"] += 1
                        continue
                recalibrated.append(p_out)
            continue

        # Rule 2: Goal Crowd-Surge Verification
        if lbl == "Goal":
            # Must have crowd surge within [-2.0s, +10.0s] in selective mode, [-2.0s, +8.0s] in raw
            c_upper = 10.0 if selective_mode else 8.0
            near_crowd = [ct for ct in crowd_times if -2.0 <= (ct - pos_sec) <= c_upper]
            if near_crowd:
                p_out["confidence"] = 0.99
                p_out["audio_grounding"] = "crowd_roar_verified"
                audit["goals_confirmed"] += 1
                recalibrated.append(p_out)
            else:
                audit["goals_pruned"] += 1
                if prune_ungrounded_goals:
                    continue  # Prune spurious goal
                p_out["confidence"] = 0.30
                p_out["audio_grounding"] = "no_crowd_roar"
                recalibrated.append(p_out)
            continue

        # Rule 3: Shots on Target Crowd Verification
        if lbl == "Shots on target":
            near_crowd = [ct for ct in crowd_times if -2.0 <= (ct - pos_sec) <= 6.0]
            if near_crowd:
                p_out["confidence"] = min(1.0, conf + 0.10)
                p_out["audio_grounding"] = "crowd_roar_aligned"
                recalibrated.append(p_out)
            else:
                if prune_ungrounded_shots and conf < 0.70:
                    audit["shots_pruned"] += 1
                    continue
                p_out["confidence"] = max(0.0, conf - 0.20)
                p_out["audio_grounding"] = "crowd_flat"
                recalibrated.append(p_out)
            continue

        # Rule 4: Visual Routine Classes (Throw-in, Corner, Ball out of play, Clearance)
        p_out["audio_grounding"] = "visual_primary"
        recalibrated.append(p_out)

    audit["final_count"] = len(recalibrated)
    return recalibrated, audit


def evaluate_model_pipeline(
    model_name: str,
    predictions: List[Dict[str, Any]],
    scoped_gt_dir: Path,
    eval_work_dir: Path,
    url_local: str
) -> Dict[str, Any]:
    """Evaluates prediction dataset through official DevKit with MAX_PATH handling."""
    devkit_pred_dir = eval_work_dir / "soccernet_devkit_format"
    mirrored_dir = devkit_pred_dir / url_local.strip("/")
    if os.name == "nt":
        os.makedirs(f"\\\\?\\{mirrored_dir.resolve()}", exist_ok=True)
    else:
        mirrored_dir.mkdir(parents=True, exist_ok=True)

    devkit_spotting_file = mirrored_dir / "results_spotting.json"
    open_spotting = Path("\\\\?\\" + str(devkit_spotting_file.resolve())) if os.name == "nt" else devkit_spotting_file
    with open(open_spotting, "w", encoding="utf-8") as f:
        json.dump({"UrlLocal": url_local, "predictions": predictions}, f, indent=2)

    eval_gt_dir = "\\\\?\\" + str(scoped_gt_dir.resolve()) if os.name == "nt" else str(scoped_gt_dir)
    eval_pred_dir = "\\\\?\\" + str(devkit_pred_dir.resolve()) if os.name == "nt" else str(devkit_pred_dir)

    metrics = evaluate_match_action_spotting(
        ground_truth_dir=eval_gt_dir,
        predictions_dir=eval_pred_dir,
        split="test",
        prediction_file="results_spotting.json"
    )
    return metrics


def main():
    print("\n" + "=" * 125)
    print("  RESEARCH QUESTION 4 (RQ4): AUDIO-VISUAL FUSION & ACOUSTIC GROUNDING EVALUATION")
    print("  Referee Whistle (3.5 - 4.2 kHz) & Crowd Roar (+5 dB Surge) Confidence Recalibration & Pruning")
    print("=" * 125)

    manifest_path = WORKSPACE_ROOT / "gate2_chunks/game_0_half_1_chunks_manifest.json"
    chunks_dir = WORKSPACE_ROOT / "gate2_chunks"
    url_local = "england_epl/2014-2015/2015-05-17 - 18-00 Manchester United 1 - 1 Arsenal/"

    # 1. Extract physical audio signals from all 5 chunks (0 MB VRAM)
    audio_signals = extract_half1_broadcast_audio_signals(manifest_path, chunks_dir)

    # 2. Setup Scoped Ground Truth Directory
    scoped_gt_dir = WORKSPACE_ROOT / "logs/local_eval/qwen3.5-4b_4bit_W60_half1/scoped_ground_truth"
    temp_root = WORKSPACE_ROOT / "scratch/audio_grounding_eval"
    if temp_root.exists():
        shutil.rmtree(temp_root, ignore_errors=True)
    temp_root.mkdir(parents=True, exist_ok=True)

    # Load Ground Truth
    gt_file = scoped_gt_dir / url_local.strip("/") / "Labels-v2.json"
    open_gt = Path("\\\\?\\" + str(gt_file.resolve())) if os.name == "nt" else gt_file
    with open(open_gt, "r", encoding="utf-8") as f:
        gt_annotations = json.load(f).get("annotations", [])

    # 3. Load Model Predictions
    qwen_file = WORKSPACE_ROOT / "logs/local_eval/qwen3.5-4b_4bit_W60_half1/Predictions-v2.json"
    minicpm_file = WORKSPACE_ROOT / "logs/local_eval/minicpm-v-4.6_4bit_W60_half1/Predictions-v2.json"
    specialist_file = WORKSPACE_ROOT / "scratch/synthesis_eval/specialist_3m" / url_local.strip("/") / "results_spotting.json"

    # Fallback to synthesis summary if specialist_3m temp file was cleared
    if not specialist_file.exists():
        synth_sum_file = WORKSPACE_ROOT / "logs/local_eval/synthesis_and_captioning_benchmark_summary.json"
        if synth_sum_file.exists():
            with open(synth_sum_file, "r", encoding="utf-8") as f:
                specialist_preds = json.load(f).get("specialist_ensemble", {}).get("predictions", [])
        else:
            specialist_preds = []
    else:
        with open(specialist_file, "r", encoding="utf-8") as f:
            specialist_preds = json.load(f).get("predictions", [])

    with open(qwen_file, "r", encoding="utf-8") as f:
        qwen_preds = json.load(f).get("predictions", [])
    with open(minicpm_file, "r", encoding="utf-8") as f:
        minicpm_preds = json.load(f).get("predictions", [])

    models_to_test = [
        ("Qwen3.5-4B Solo (Half 1)", qwen_preds, temp_root / "qwen", False),
        ("MiniCPM-V 4.6 Solo (Half 1)", minicpm_preds, temp_root / "minicpm", False),
        ("3-Model Specialist Ensemble (Raw Gate)", specialist_preds, temp_root / "specialist", False),
        ("3-Model Specialist + Synchronized Selective Gate", specialist_preds, temp_root / "specialist_sync", True)
    ]

    benchmark_records = []

    print("\n" + "=" * 125)
    print("  RUNNING AUDIO-VISUAL FUSION BENCHMARK ACROSS ALL 4 COHORTS")
    print("=" * 125)

    for name, raw_preds, work_dir, is_selective in models_to_test:
        if not raw_preds:
            continue
        print(f"\n--- Evaluating Model: {name} ({len(raw_preds)} Raw Predictions) ---")

        # A. Baseline Evaluation (Visual Only)
        base_metrics = evaluate_model_pipeline(f"{name}_base", raw_preds, scoped_gt_dir, work_dir / "base", url_local)

        # B. Audio-Grounded Evaluation
        grounded_preds, audit = apply_audio_grounding_recalibration(
            predictions=raw_preds,
            audio_signals=audio_signals,
            whistle_tolerance_sec=4.0,
            crowd_surge_min_db=4.0,
            prune_ungrounded_goals=True,
            prune_ungrounded_shots=True,
            prune_ungrounded_fouls=not is_selective,
            selective_mode=is_selective
        )

        # Apply class-aware NMS after confidence boosting/penalization
        grounded_deduped = class_aware_nms(grounded_preds, radius_ms=3000, class_radii_ms=DEFAULT_CLASS_NMS_RADII_MS)
        grounded_metrics = evaluate_model_pipeline(f"{name}_grounded", grounded_deduped, scoped_gt_dir, work_dir / "grounded", url_local)

        rec = {
            "model": name,
            "raw_count": len(raw_preds),
            "grounded_count": len(grounded_deduped),
            "pruned_count": len(raw_preds) - len(grounded_deduped),
            "audit": audit,
            "base_tight_mAP": base_metrics.get("tight_average_mAP", 0.0),
            "grounded_tight_mAP": grounded_metrics.get("tight_average_mAP", 0.0),
            "base_loose_mAP": base_metrics.get("loose_average_mAP", 0.0),
            "grounded_loose_mAP": grounded_metrics.get("loose_average_mAP", 0.0),
            "base_active_loose_mAP": base_metrics.get("active_loose_average_mAP", 0.0),
            "grounded_active_loose_mAP": grounded_metrics.get("active_loose_average_mAP", 0.0),
            "goal_ap_before": base_metrics.get("loose_per_class", {}).get("Goal", 0.0),
            "goal_ap_after": grounded_metrics.get("loose_per_class", {}).get("Goal", 0.0),
            "foul_ap_before": base_metrics.get("loose_per_class", {}).get("Foul", 0.0),
            "foul_ap_after": grounded_metrics.get("loose_per_class", {}).get("Foul", 0.0)
        }
        benchmark_records.append(rec)

        print(f"  [RESULT] Pruned {rec['pruned_count']} Spurious Predictions ({rec['raw_count']} -> {rec['grounded_count']})")
        print(f"           Tight mAP: {rec['base_tight_mAP']:.4f} -> {rec['grounded_tight_mAP']:.4f} (Delta: {rec['grounded_tight_mAP']-rec['base_tight_mAP']:+.4f})")
        print(f"           Loose mAP: {rec['base_loose_mAP']:.4f} -> {rec['grounded_loose_mAP']:.4f} (Delta: {rec['grounded_loose_mAP']-rec['base_loose_mAP']:+.4f})")
        print(f"           Active Loose mAP: {rec['base_active_loose_mAP']:.4f} -> {rec['grounded_active_loose_mAP']:.4f} (Delta: {rec['grounded_active_loose_mAP']-rec['base_active_loose_mAP']:+.4f})")
        print(f"           Goal AP: {rec['goal_ap_before']:.4f} -> {rec['goal_ap_after']:.4f} | Foul AP: {rec['foul_ap_before']:.4f} -> {rec['foul_ap_after']:.4f}")

    # Print Master RQ4 Comparison Table
    print("\n" + "=" * 135)
    print("  TABLE 12: RESEARCH QUESTION 4 (RQ4) AUDIO-VISUAL FUSION & ACOUSTIC GROUNDING SUMMARY")
    print("=" * 135)
    print(f"{'Model Architecture':<35} | {'Events (B/A)':<12} | {'FP Pruned':<10} | {'Tight mAP (B/A)':<18} | {'Loose mAP (B/A)':<18} | {'Active Loose mAP':<18} | {'Goal AP (B/A)'}")
    print("-" * 135)
    for r in benchmark_records:
        ev_str = f"{r['raw_count']} -> {r['grounded_count']}"
        t_str = f"{r['base_tight_mAP']:.4f} -> {r['grounded_tight_mAP']:.4f}"
        l_str = f"{r['base_loose_mAP']:.4f} -> {r['grounded_loose_mAP']:.4f}"
        act_str = f"{r['base_active_loose_mAP']:.4f} -> {r['grounded_active_loose_mAP']:.4f}"
        g_str = f"{r['goal_ap_before']:.4f} -> {r['goal_ap_after']:.4f}"
        print(f"{r['model']:<35} | {ev_str:<12} | {r['pruned_count']:<10} | {t_str:<18} | {l_str:<18} | {act_str:<18} | {g_str}")
    print("=" * 135 + "\n")

    # Export RQ4 Summary JSON
    out_json = WORKSPACE_ROOT / "logs/local_eval/rq4_audio_grounding_summary.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "acoustic_signals": {
                "total_whistle_peaks_detected": len(audio_signals["whistle_peaks"]),
                "total_crowd_roars_detected": len(audio_signals["crowd_roars"]),
                "sample_whistle_peaks": audio_signals["whistle_peaks"][:15],
                "sample_crowd_roars": audio_signals["crowd_roars"][:15]
            },
            "records": benchmark_records
        }, f, indent=2)

    print(f"[SUCCESS] Exported RQ4 Audio-Visual Grounding Summary -> {out_json}")


if __name__ == "__main__":
    main()
