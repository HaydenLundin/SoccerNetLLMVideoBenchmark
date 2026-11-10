"""
Step 1: Construct the Zero-Leakage Multi-Specialist Distillation Dataset.
Dataset source: Strictly out-of-sample held-out matches from M_15 (Game 4 and Game 7).
Zero leakage into MES-5 (Games 0, 1, 2, 3, 13).

Target schema:
[
  {
    "commentary": "<dense telegraphic 12-to-18 token visual observation>",
    "timestamp": "00:SS",
    "label": "<Official SoccerNet Class>",
    "confidence": 0.95,
    "visibility": "visible"
  }
]
"""

import os
import sys
import json
import random
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from PIL import Image

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from processing.unified_schema import SOCCERNET_CLASSES
from scripts.run_local_benchmark import (
    decode_chunk_frames_single_pass,
    burn_timestamp_overlay,
    build_window_prompt
)

# 12-to-18 token concise visual observation templates for all 17 SoccerNet classes
CONCISE_COT_TEMPLATES = {
    "Offside": [
        "Attacking runner breaks beyond the defensive backline into the box as the assistant referee raises the offside flag.",
        "Forward receives a forward pass in an illegal offside position behind the final defending line prompting the whistle.",
        "Attacker makes an early run across the defensive line and referee blows whistle to signal offside."
    ],
    "Shots on target": [
        "Forward strikes a low drive aimed towards the bottom corner, forcing the goalkeeper into a diving save.",
        "Attacker fires a firm shot on goal that tests the goalkeeper who parries the ball away.",
        "Midfielder unleashes a powerful shot aimed directly at the net requiring an active intervention by the keeper."
    ],
    "Shots off target": [
        "Attacker attempts a curling shot from outside the box that sails high and wide past the upright.",
        "Forward strikes a first-time volley towards the goal but the ball misses the frame entirely.",
        "Player cuts inside and takes an ambitious long-range strike that drifts wide of the target."
    ],
    "Clearance": [
        "Defender inside the penalty area makes a powerful clearing header away from danger under heavy attacking pressure.",
        "Center back forcibly kicks the loose ball out of the eighteen-yard box to relieve immediate attacking pressure.",
        "Defending player boots the ball high and far downfield away from an active threat in the box."
    ],
    "Ball out of play": [
        "Deflected pass rolls past the sideline touchline and completely exits the field of boundary play.",
        "Overhit cross crosses the endline boundary harmlessly behind the goal resulting in a stoppage of play.",
        "Loose ball travels beyond the boundary line on the far side prompting an immediate stoppage of live play."
    ],
    "Indirect free-kick": [
        "Referee blows whistle with vertical arm raised awarding an indirect restart following an offside infringement.",
        "Match official halts play and raises an arm vertically to signal an indirect free-kick set piece restart.",
        "Referee awards an indirect free-kick set piece with raised arm following a technical rules infraction."
    ],
    "Direct free-kick": [
        "Referee awards a direct set piece outside the penalty box following an illegal sliding challenge.",
        "Match official blows whistle and points forward signaling a direct free-kick opportunity in attacking territory.",
        "Player is brought down outside the penalty area and referee awards a direct free-kick restart."
    ],
    "Corner": [
        "Defending defender redirects ball over the endline resulting in an attacking corner kick from the arc.",
        "Goalkeeper tips the shot over the crossbar and the attacking team prepares to take a corner kick.",
        "Ball is deflected behind the goal line off a defender awarding a set piece from the corner."
    ],
    "Throw-in": [
        "Player stands outside the sideline touchline and executes a two-handed overhead restart into play.",
        "Sideline restart delivered overhead by a fullback returning the ball into active play from the touchline.",
        "Player takes a throw-in from the touchline with both hands overhead to resume active competition."
    ],
    "Foul": [
        "Defending player commits a late sliding tackle bringing down the ball carrier near midfield.",
        "Aggressive challenge from behind trips the attacking midfielder prompting an immediate whistle from the match official.",
        "Player makes an illegal tactical obstruction against an opponent and the referee halts play for a foul."
    ],
    "Yellow card": [
        "Referee halts play and brandishes a yellow card caution to the offending player for a cynical foul.",
        "Match official stops play and holds up a yellow card booking for a reckless mistimed tackle.",
        "Referee summons the offending defender and administers a yellow card warning following persistent tactical fouls."
    ],
    "Red card": [
        "Referee produces a straight red card dismissing the player from the pitch for serious foul play.",
        "Match official immediately issues an expulsion red card send-off following an extremely dangerous high challenge.",
        "Referee brandishes an uncontested red card directing the dismissed player to leave the field of play."
    ],
    "Yellow->red card": [
        "Referee shows a second yellow card followed immediately by the red card ejection for repeated fouls.",
        "Match official displays a second caution yellow before flashing the red card dismissal to send off the player.",
        "Player receives a second bookable yellow card infraction and is promptly sent off with a red card."
    ],
    "Substitution": [
        "Fourth official displays the electronic substitution board at the technical area as the player walks off.",
        "Substituted player jogs off the pitch and claps as the fresh teammate enters the field of play.",
        "Sideline substitution transaction conducted as the substitute player crosses the touchline onto the pitch."
    ],
    "Goal": [
        "Attacking forward strikes the ball past the goalkeeper into the back of the net for a goal.",
        "Ball is placed cleanly into the bottom corner of the net sending teammates into a goal celebration.",
        "Striker connects on the cross and converts past the keeper to register an official goal."
    ],
    "Penalty": [
        "Attacking player is brought down inside the eighteen-yard penalty box, prompting the referee to point to the spot.",
        "Referee blows whistle firmly and points to the twelve-yard penalty spot after a foul in the box.",
        "Defensive trip inside the active penalty area results in an immediate spot-kick penalty award."
    ],
    "Kick-off": [
        "Player restarts play from the center spot inside the center circle following a goal or half interval.",
        "Two forwards stand inside the center circle and pass the ball from the spot to commence play.",
        "Match action resumes from the center spot with a kickoff pass across the midfield circle."
    ]
}

def get_concise_cot(label: str, seed_idx: int = 0) -> str:
    templates = CONCISE_COT_TEMPLATES.get(label, [f"Observational soccer action corresponding to an official {label} event."])
    return templates[seed_idx % len(templates)]

def load_ground_truth(game_id: int) -> Dict[str, Any]:
    eval_set_path = WORKSPACE_ROOT / "eval_set.json"
    with open(eval_set_path, "r", encoding="utf-8") as f:
        eval_set = json.load(f)
    game_info = next((g for g in eval_set["games"] if g["game_id"] == game_id), None)
    if not game_info:
        raise ValueError(f"Game {game_id} not found in eval_set.json")
    
    gt_path = WORKSPACE_ROOT / "test_eval_ground_truth" / game_info["relative_path"] / "Labels-v2.json"
    if not gt_path.exists():
        raise FileNotFoundError(f"Labels-v2.json not found at {gt_path}")
    with open(gt_path, "r", encoding="utf-8") as f:
        return json.load(f)

def build_distillation_dataset(
    training_games: List[int] = [4, 7],
    window_sec: int = 60,
    stride_sec: int = 45,
    frames_per_window: int = 16,
    target_resolution: Tuple[int, int] = (384, 216),
    output_dir: str = "data/qlora_distillation",
    negative_sample_ratio: float = 0.30,
    random_seed: int = 42
):
    random.seed(random_seed)
    out_path = WORKSPACE_ROOT / output_dir
    frames_dir = out_path / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    chunks_dir = WORKSPACE_ROOT / "gate2_chunks"
    dataset_records = []
    class_distribution = {c: 0 for c in SOCCERNET_CLASSES}

    total_chunks_processed = 0
    total_candidate_windows = 0
    all_positive_windows = []
    all_negative_windows = []

    print("==========================================================================")
    print("  Constructing Multi-Specialist QLoRA Distillation Dataset (Zero-Leakage)")
    print(f"  Training Games: {training_games} (Strictly M_15, Out-of-Sample from MES-5)")
    print(f"  Window: {window_sec}s | Stride: {stride_sec}s | Keyframes/Win: {frames_per_window} ({target_resolution[0]}x{target_resolution[1]})")
    print("==========================================================================")

    for gid in training_games:
        gt_data = load_ground_truth(gid)
        annotations = gt_data.get("annotations", [])
        print(f"\n[*] Loaded Ground Truth for Game {gid}: {len(annotations)} total annotations.")

        for half in [1, 2]:
            manifest_file = chunks_dir / f"game_{gid}_half_{half}_chunks_manifest.json"
            if not manifest_file.exists():
                print(f"[!] Warning: Manifest not found for Game {gid} Half {half} at {manifest_file}. Skipping.")
                continue

            with open(manifest_file, "r", encoding="utf-8") as f:
                manifest = json.load(f)

            half_annotations = [
                a for a in annotations 
                if a.get("gameTime", "").startswith(f"{half} -")
            ]

            print(f"  -> Processing Game {gid} Half {half}: {len(manifest.get('chunks', []))} chunks, {len(half_annotations)} GT events.")

            for chunk_meta in manifest.get("chunks", []):
                chunk_idx = chunk_meta["chunk_index"]
                filename = chunk_meta["filename"]
                chunk_video_path = chunks_dir / filename
                if not chunk_video_path.exists():
                    print(f"     [!] Chunk video missing: {chunk_video_path}")
                    continue

                actual_start_ms = int(chunk_meta.get("actual_start_ms", 0))
                duration_sec = float(chunk_meta.get("duration_sec", 615.0))

                # Decode chunk frames at 1.0 fps (zero seek drift)
                chunk_frames = decode_chunk_frames_single_pass(str(chunk_video_path), fps=1.0, max_size=target_resolution[0])
                total_chunks_processed += 1

                # Generate window start schedule
                win_starts = []
                curr_t = 0.0
                while curr_t < duration_sec:
                    win_starts.append(curr_t)
                    curr_t += stride_sec

                for w_idx, win_start_t in enumerate(win_starts):
                    eff_window_dur = min(float(window_sec), duration_sec - win_start_t)
                    if eff_window_dur < 10.0:
                        continue  # Skip tiny sliver tail windows

                    total_candidate_windows += 1
                    win_start_ms = int(round(win_start_t * 1000.0))
                    win_end_ms = win_start_ms + int(round(eff_window_dur * 1000.0))

                    # 3-Tier Offset Math: find all GT events falling inside this window
                    matching_events = []
                    for a_idx, a in enumerate(half_annotations):
                        gt_pos_ms = int(a.get("position", 0))
                        rel_ms = gt_pos_ms - (actual_start_ms + win_start_ms)
                        if 0 <= rel_ms < (eff_window_dur * 1000.0):
                            rel_sec = rel_ms / 1000.0
                            lbl = a.get("label")
                            if lbl in SOCCERNET_CLASSES:
                                mm = int(rel_sec // 60)
                                ss = int(round(rel_sec % 60))
                                ts_str = f"{mm:02d}:{ss:02d}"
                                cot_text = get_concise_cot(lbl, seed_idx=a_idx + w_idx)
                                matching_events.append({
                                    "commentary": cot_text,
                                    "timestamp": ts_str,
                                    "label": lbl,
                                    "confidence": 0.95,
                                    "visibility": "visible",
                                    "rel_sec": rel_sec
                                })

                    # Sort matching events chronologically
                    matching_events.sort(key=lambda x: x["rel_sec"])

                    # Build window frames
                    # Slice frames from in-memory chunk_frames
                    win_raw_frames = [f for f in chunk_frames if win_start_t <= f[0] < (win_start_t + eff_window_dur)]
                    if not win_raw_frames:
                        idx_start = min(int(round(win_start_t)), len(chunk_frames) - 1)
                        idx_end = min(int(round(win_start_t + eff_window_dur)), len(chunk_frames))
                        win_raw_frames = chunk_frames[idx_start:idx_end]

                    if len(win_raw_frames) == 0:
                        continue

                    # Uniformly sample frames_per_window keyframes
                    step = (len(win_raw_frames) - 1) / max(1, (frames_per_window - 1))
                    sampled_indices = [int(round(i * step)) for i in range(frames_per_window)]
                    sampled_indices = [min(idx, len(win_raw_frames) - 1) for idx in sampled_indices]

                    win_id = f"g{gid}_h{half}_c{chunk_idx:02d}_w{w_idx:02d}"
                    frame_paths = []

                    for frame_seq_idx, frame_idx in enumerate(sampled_indices):
                        raw_f_ts, raw_img = win_raw_frames[frame_idx]
                        sec_in_win = raw_f_ts - win_start_t
                        # Burn digital clock badge [00:SS]
                        burned_img = burn_timestamp_overlay(raw_img, sec_in_window=sec_in_win)
                        # Resize to target resolution
                        if burned_img.size != target_resolution:
                            burned_img = burned_img.resize(target_resolution, Image.Resampling.BICUBIC)
                        
                        f_filename = f"{win_id}_f{frame_seq_idx:02d}.jpg"
                        f_path = frames_dir / f_filename
                        burned_img.save(f_path, "JPEG", quality=88)
                        frame_paths.append(str(f_path.relative_to(WORKSPACE_ROOT)))

                    # Clean events (remove rel_sec internal field)
                    clean_events = []
                    for ev in matching_events:
                        clean_events.append({
                            "commentary": ev["commentary"],
                            "timestamp": ev["timestamp"],
                            "label": ev["label"],
                            "confidence": ev["confidence"],
                            "visibility": ev["visibility"]
                        })

                    prompt_text = build_window_prompt(int(eff_window_dur), spotting_only=False)
                    target_json_str = json.dumps(clean_events, indent=2)

                    record = {
                        "window_id": win_id,
                        "game_id": gid,
                        "half": half,
                        "chunk_index": chunk_idx,
                        "window_index": w_idx,
                        "start_sec_in_chunk": round(win_start_t, 2),
                        "actual_start_sec_in_half": round((actual_start_ms / 1000.0) + win_start_t, 2),
                        "duration_sec": int(eff_window_dur),
                        "frames": frame_paths,
                        "num_frames": len(frame_paths),
                        "prompt": prompt_text,
                        "target_json": target_json_str,
                        "events": clean_events,
                        "has_events": len(clean_events) > 0
                    }

                    if len(clean_events) > 0:
                        for ev in clean_events:
                            class_distribution[ev["label"]] += 1
                        all_positive_windows.append(record)
                    else:
                        all_negative_windows.append(record)

    print(f"\n[*] Window Extraction Completed:")
    print(f"    Total Processed Chunks: {total_chunks_processed}")
    print(f"    Total Candidate Windows: {total_candidate_windows}")
    print(f"    Positive Windows (with >= 1 GT event): {len(all_positive_windows)}")
    print(f"    Negative Windows (0 events / background play): {len(all_negative_windows)}")

    # Balance dataset: include all positive windows + sampled negative windows
    num_negatives_to_keep = int(round(len(all_positive_windows) * negative_sample_ratio))
    sampled_negatives = random.sample(all_negative_windows, min(num_negatives_to_keep, len(all_negative_windows)))

    final_dataset = all_positive_windows + sampled_negatives
    random.shuffle(final_dataset)

    manifest_output = out_path / "dataset_manifest.json"
    with open(manifest_output, "w", encoding="utf-8") as f:
        json.dump({
            "total_samples": len(final_dataset),
            "positive_samples": len(all_positive_windows),
            "negative_samples": len(sampled_negatives),
            "frames_per_window": frames_per_window,
            "target_resolution": list(target_resolution),
            "class_distribution": class_distribution,
            "samples": final_dataset
        }, f, indent=2)

    print("\n" + "=" * 70)
    print("  FINAL DISTILLATION DATASET SUMMARY")
    print("=" * 70)
    print(f"  Total Training Windows: {len(final_dataset)}")
    print(f"    - Event Windows:     {len(all_positive_windows)} ({len(all_positive_windows)/len(final_dataset)*100:.1f}%)")
    print(f"    - Background Windows:{len(sampled_negatives)} ({len(sampled_negatives)/len(final_dataset)*100:.1f}%)")
    print("\n  Per-Class Event Ground Truth Representation:")
    for c, cnt in sorted(class_distribution.items(), key=lambda x: -x[1]):
        if cnt > 0:
            print(f"    * {c:20s}: {cnt:3d} events")
    print(f"\n  [+] Saved Manifest: {manifest_output}")
    print("=" * 70 + "\n")
    return manifest_output

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Construct QLoRA Distillation Dataset")
    parser.add_argument("--games", type=int, nargs="+", default=[4, 7], help="Held-out training games from M_15")
    parser.add_argument("--window-sec", type=int, default=60, help="Sliding window duration in seconds")
    parser.add_argument("--stride-sec", type=int, default=45, help="Sliding window stride in seconds")
    parser.add_argument("--frames", type=int, default=16, help="Sampled keyframes per window")
    parser.add_argument("--width", type=int, default=384, help="Frame width")
    parser.add_argument("--height", type=int, default=216, help="Frame height")
    parser.add_argument("--output-dir", type=str, default="data/qlora_distillation", help="Output directory")
    parser.add_argument("--neg-ratio", type=float, default=0.30, help="Ratio of negative background windows")
    args = parser.parse_args()

    build_distillation_dataset(
        training_games=args.games,
        window_sec=args.window_sec,
        stride_sec=args.stride_sec,
        frames_per_window=args.frames,
        target_resolution=(args.width, args.height),
        output_dir=args.output_dir,
        negative_sample_ratio=args.neg_ratio
    )
