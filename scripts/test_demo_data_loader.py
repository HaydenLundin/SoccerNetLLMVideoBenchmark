import json
from pathlib import Path
import numpy as np

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent

MES5_GAMES = {
    0: {
        "season": "2014-2015",
        "match": "2015-05-17 - 18-00 Manchester United 1 - 1 Arsenal",
        "title": "Manchester United 1 - 1 Arsenal",
        "date": "2015-05-17"
    },
    1: {
        "season": "2015-2016",
        "match": "2015-08-16 - 18-00 Manchester City 3 - 0 Chelsea",
        "title": "Manchester City 3 - 0 Chelsea",
        "date": "2015-08-16"
    },
    2: {
        "season": "2015-2016",
        "match": "2015-08-23 - 15-30 West Brom 2 - 3 Chelsea",
        "title": "West Brom 2 - 3 Chelsea",
        "date": "2015-08-23"
    },
    3: {
        "season": "2015-2016",
        "match": "2015-08-29 - 17-00 Liverpool 0 - 3 West Ham",
        "title": "Liverpool 0 - 3 West Ham",
        "date": "2015-08-29"
    },
    13: {
        "season": "2016-2017",
        "match": "2016-08-27 - 14-30 Tottenham 1 - 1 Liverpool",
        "title": "Tottenham 1 - 1 Liverpool",
        "date": "2016-08-27"
    }
}

def test_data():
    print(f"Workspace root: {WORKSPACE_ROOT}")
    total_chunks = 0
    total_gt = 0
    total_dsp = 0

    for g_id, info in MES5_GAMES.items():
        for half in [1, 2]:
            manifest_path = WORKSPACE_ROOT / f"gate2_chunks/game_{g_id}_half_{half}_chunks_manifest.json"
            if not manifest_path.exists():
                print(f"Missing manifest: {manifest_path}")
                continue
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            chunks = manifest.get("chunks", [])
            total_chunks += len(chunks)

            # Check GT
            gt_path = WORKSPACE_ROOT / f"logs/local_eval/mes5_specialist_3m_m15/devkit_ground_truth/england_epl/{info['season']}/{info['match']}/Labels-v2.json"
            if gt_path.exists():
                with open(gt_path, "r", encoding="utf-8") as f:
                    gt_data = json.load(f)
                half_gt = [a for a in gt_data.get("annotations", []) if str(a.get("half")) == str(half) or (half == 1 and a.get("gameTime", "").startswith("1 -")) or (half == 2 and a.get("gameTime", "").startswith("2 -"))]
                total_gt += len(half_gt)

            # Check DSP profiles
            for c in chunks:
                chunk_file = WORKSPACE_ROOT / c["relative_path"]
                dsp_file = WORKSPACE_ROOT / f"logs/local_eval/mes5_dsp_profiles/{chunk_file.stem}_dsp.npz"
                if dsp_file.exists():
                    total_dsp += 1

    print(f"Verification Summary:")
    print(f"  Total Chunks: {total_chunks}")
    print(f"  Total GT events found: {total_gt}")
    print(f"  Total DSP profiles found: {total_dsp}/{total_chunks}")

    # Check prediction sets
    model_sets = [
        "mes5_specialist_3m_m15",
        "mes5_specialist_3m",
        "mes5_qwen3.5-4b",
        "mes5_minicpm-v-4.6",
        "mes5_lfm2.5-vl-3b",
        "mes5_simulated_stage2"
    ]
    print("\nModel Prediction Sets Check:")
    for mset in model_sets:
        found_count = 0
        total_preds = 0
        for g_id, info in MES5_GAMES.items():
            pred_file = WORKSPACE_ROOT / f"logs/local_eval/{mset}/soccernet_devkit_format/england_epl/{info['season']}/{info['match']}/results_spotting.json"
            if pred_file.exists():
                found_count += 1
                with open(pred_file, "r", encoding="utf-8") as f:
                    preds = json.load(f).get("predictions", [])
                total_preds += len(preds)
        print(f"  {mset:<25}: {found_count}/5 matches loaded ({total_preds} total predictions)")

    # Check raw window outputs (CoT)
    total_raw_windows = 0
    for g_id in MES5_GAMES.keys():
        for half in [1, 2]:
            if g_id == 0 and half == 1:
                raw_path = WORKSPACE_ROOT / "logs/local_eval/qwen3.5-4b_4bit_W60_half1/raw_window_outputs.json"
            else:
                raw_path = WORKSPACE_ROOT / f"logs/local_eval/qwen3.5-4b_4bit_W60_game{g_id}_half{half}/raw_window_outputs.json"
            if raw_path.exists():
                with open(raw_path, "r", encoding="utf-8") as f:
                    rw = json.load(f)
                total_raw_windows += len(rw)
    print(f"\nRaw Window CoT outputs loaded: {total_raw_windows} windows across 10 halves")

    # Discover key highlight moments
    print("\n--- Key Highlight Moments Discovery ---")
    target_labels = ["Goal", "Red card", "Yellow->red card", "Penalty"]
    for g_id, info in MES5_GAMES.items():
        gt_path = WORKSPACE_ROOT / f"logs/local_eval/mes5_specialist_3m_m15/devkit_ground_truth/england_epl/{info['season']}/{info['match']}/Labels-v2.json"
        with open(gt_path, "r", encoding="utf-8") as f:
            gt = json.load(f)
        for a in gt.get("annotations", []):
            if a.get("label") in target_labels:
                half = int(a.get("half", 1))
                pos_sec = float(a.get("position", 0)) / 1000.0
                manifest_path = WORKSPACE_ROOT / f"gate2_chunks/game_{g_id}_half_{half}_chunks_manifest.json"
                if not manifest_path.exists():
                    continue
                with open(manifest_path, "r", encoding="utf-8") as mf:
                    manifest = json.load(mf)
                for c in manifest["chunks"]:
                    if c["actual_start_sec"] <= pos_sec <= c["actual_start_sec"] + c["duration_sec"]:
                        rel_sec = pos_sec - c["actual_start_sec"]
                        print(f"Game {g_id:<2} H{half} Chunk {c['chunk_index']} @ {rel_sec:5.1f}s | {a['gameTime']} | {a['label']:<16} | {info['title']}")
                        break

if __name__ == "__main__":
    test_data()
