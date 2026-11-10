"""
SoccerNet VLM Interactive Demonstration Server.
Zero-dependency, standalone HTTP server with HTTP 206 Partial Content (Range) video streaming,
REST API for multimodal predictions, acoustic/optical DSP curves, and Chain-of-Thought commentary.
"""

import os
import sys
import json
import re
import mimetypes
from pathlib import Path
from http import HTTPStatus
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
import urllib.parse
import numpy as np

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
STATIC_DIR = Path(__file__).resolve().parent / "static"
GATE2_DIR = WORKSPACE_ROOT / "gate2_chunks"
LOGS_DIR = WORKSPACE_ROOT / "logs/local_eval"
DSP_CACHE_DIR = LOGS_DIR / "mes5_dsp_profiles"

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

MODEL_CONFIGS = {
    "mece_m15": {
        "name": "MECE 3-Model Ensemble + M15 Prior (Best: 0.1833)",
        "folder": "mes5_specialist_3m_m15"
    },
    "mece_3m": {
        "name": "MECE 3-Model Domain Specialist (0.1733)",
        "folder": "mes5_specialist_3m"
    },
    "stage2_pareto": {
        "name": "Stage 2 CPU Filtered (100.4% Ret.: 0.1840)",
        "folder": "mes5_simulated_stage2"
    },
    "qwen3.5-4b": {
        "name": "Qwen3.5-4B Solo (0.1564)",
        "folder": "mes5_qwen3.5-4b"
    },
    "minicpm-v-4.6": {
        "name": "MiniCPM-V 4.6 Solo (0.0809)",
        "folder": "mes5_minicpm-v-4.6"
    },
    "lfm2.5-vl-3b": {
        "name": "LFM2.5-VL-3B Solo (0.0206)",
        "folder": "mes5_lfm2.5-vl-3b"
    }
}

# Cache containers
MANIFEST_CACHE = {}
GT_CACHE = {}
PRED_CACHE = {}
DSP_CACHE = {}
RAW_WINDOWS_CACHE = {}


def load_all_caches():
    print("Pre-loading demo metadata and annotations...")
    for g_id, info in MES5_GAMES.items():
        for half in [1, 2]:
            mf_path = GATE2_DIR / f"game_{g_id}_half_{half}_chunks_manifest.json"
            if mf_path.exists():
                with open(mf_path, "r", encoding="utf-8") as f:
                    MANIFEST_CACHE[(g_id, half)] = json.load(f)

        gt_path = LOGS_DIR / f"mes5_specialist_3m_m15/devkit_ground_truth/england_epl/{info['season']}/{info['match']}/Labels-v2.json"
        if gt_path.exists():
            with open(gt_path, "r", encoding="utf-8") as f:
                GT_CACHE[g_id] = json.load(f).get("annotations", [])

        # Load raw CoT windows for Qwen
        for half in [1, 2]:
            if g_id == 0 and half == 1:
                rw_path = LOGS_DIR / "qwen3.5-4b_4bit_W60_half1/raw_window_outputs.json"
            else:
                rw_path = LOGS_DIR / f"qwen3.5-4b_4bit_W60_game{g_id}_half{half}/raw_window_outputs.json"
            if rw_path.exists():
                with open(rw_path, "r", encoding="utf-8") as f:
                    RAW_WINDOWS_CACHE[(g_id, half)] = json.load(f)

    for mkey, mcfg in MODEL_CONFIGS.items():
        PRED_CACHE[mkey] = {}
        for g_id, info in MES5_GAMES.items():
            pred_file = LOGS_DIR / f"{mcfg['folder']}/soccernet_devkit_format/england_epl/{info['season']}/{info['match']}/results_spotting.json"
            if pred_file.exists():
                with open(pred_file, "r", encoding="utf-8") as f:
                    PRED_CACHE[mkey][g_id] = json.load(f).get("predictions", [])
    print(f"Loaded {len(MANIFEST_CACHE)} manifests, {len(GT_CACHE)} GT sets, {len(PRED_CACHE)} model prediction sets.")


def get_curated_highlights():
    return [
        {
            "id": "terry-red",
            "title": "John Terry Straight Red Card (West Brom vs Chelsea)",
            "game_id": 2,
            "half": 2,
            "chunk_index": 0,
            "rel_sec": 501.1,
            "game_time": "2 - 08:21",
            "label": "Red card",
            "model": "qwen3.5-4b",
            "confidence": 0.99,
            "drift_sec": -1.10,
            "badge": "CRITICAL DISCIPLINE",
            "description": "Chelsea captain dismissed for denying goalscoring opportunity. Qwen3.5-4B spotted at 500.0s (conf 0.99, drift -1.10s)."
        },
        {
            "id": "noble-red",
            "title": "Mark Noble Straight Red Card (Liverpool vs West Ham)",
            "game_id": 3,
            "half": 2,
            "chunk_index": 3,
            "rel_sec": 217.8,
            "game_time": "2 - 32:51",
            "label": "Red card",
            "model": "qwen3.5-4b",
            "confidence": 0.98,
            "drift_sec": -2.28,
            "badge": "CRITICAL DISCIPLINE",
            "description": "Noble tackles Ings at Anfield. Qwen3.5-4B spotted at 215.5s (conf 0.98, drift -2.28s) under full referee whistle surge."
        },
        {
            "id": "coutinho-second-yellow",
            "title": "Philippe Coutinho 2nd Yellow Dismissal (Liverpool vs West Ham)",
            "game_id": 3,
            "half": 2,
            "chunk_index": 0,
            "rel_sec": 391.7,
            "game_time": "2 - 06:31",
            "label": "Yellow->red card",
            "model": "qwen3.5-4b",
            "confidence": 0.95,
            "drift_sec": 0.85,
            "badge": "COMPOUND DISCIPLINE",
            "description": "Second bookable challenge. Spotted by Qwen3.5-4B with zero-compute semantic co-emission (0.4773 AP)."
        },
        {
            "id": "herrera-goal",
            "title": "Ander Herrera Volley Goal (Man United vs Arsenal)",
            "game_id": 0,
            "half": 1,
            "chunk_index": 2,
            "rel_sec": 594.6,
            "game_time": "1 - 29:19",
            "label": "Goal",
            "model": "qwen3.5-4b",
            "confidence": 0.95,
            "drift_sec": 1.40,
            "badge": "LIVE GOAL",
            "description": "Young crosses to Herrera at the far post. Qwen3.5-4B grounds volley with burned-in wristwatch overlay."
        },
        {
            "id": "aguero-goal",
            "title": "Sergio Aguero Opener (Man City vs Chelsea)",
            "game_id": 1,
            "half": 1,
            "chunk_index": 3,
            "rel_sec": 111.1,
            "game_time": "1 - 31:00",
            "label": "Goal",
            "model": "qwen3.5-4b",
            "confidence": 0.95,
            "drift_sec": 2.10,
            "badge": "LIVE GOAL",
            "description": "Aguero converts inside the penalty area following sustained crowd roar surge (+9.2 dB)."
        },
        {
            "id": "westbrom-penalty",
            "title": "James Morrison Penalty Award (West Brom vs Chelsea)",
            "game_id": 2,
            "half": 1,
            "chunk_index": 1,
            "rel_sec": 198.8,
            "game_time": "1 - 13:03",
            "label": "Penalty",
            "model": "qwen3.5-4b",
            "confidence": 0.90,
            "drift_sec": -1.80,
            "badge": "PENALTY BOX",
            "description": "Foul inside the box awarded as penalty. Qwen3.5-4B resolves penalty decision at 0.0303 AP."
        },
        {
            "id": "walker-sub",
            "title": "Kyle Walker Acute Injury Substitution (Tottenham vs Liverpool)",
            "game_id": 13,
            "half": 1,
            "chunk_index": 2,
            "rel_sec": 414.2,
            "game_time": "1 - 27:30",
            "label": "Substitution",
            "model": "qwen3.5-4b",
            "confidence": 0.95,
            "drift_sec": 1.20,
            "badge": "HALF 1 INJURY SUB",
            "description": "Rare first-half injury replacement. Qwen3.5-4B spots substitution board at high resolution (0.2381 AP)."
        },
        {
            "id": "falcao-sub",
            "title": "Radamel Falcao Farewell Substitution (Man United vs Arsenal)",
            "game_id": 0,
            "half": 2,
            "chunk_index": 3,
            "rel_sec": 138.0,
            "game_time": "2 - 31:21",
            "label": "Substitution",
            "model": "qwen3.5-4b",
            "confidence": 0.95,
            "drift_sec": 2.21,
            "badge": "BENCH SUB",
            "description": "Falcao replaces Mata. Qwen3.5-4B catches substitution board with 66.7% recall while MiniCPM scores 0% due to spatial compression."
        }
    ]


class DemoRequestHandler(SimpleHTTPRequestHandler):
    """Custom HTTP Request Handler supporting REST API, JSON payloads, and HTTP 206 Video Streaming."""

    def translate_path(self, path):
        # Override to serve from static folder by default
        clean_path = urllib.parse.urlsplit(path).path
        if clean_path.startswith("/static/"):
            return str(STATIC_DIR / clean_path[8:])
        if clean_path.startswith("/video/"):
            rel = clean_path[7:]
            return str(WORKSPACE_ROOT / rel)
        return str(STATIC_DIR / "index.html")

    def do_GET(self):
        url_parts = urllib.parse.urlsplit(self.path)
        path = url_parts.path
        query = urllib.parse.parse_qs(url_parts.query)

        # 1. API Routes
        if path.startswith("/api/"):
            self.handle_api(path, query)
            return

        # 2. Video Streaming (HTTP 206 Range Request)
        if path.startswith("/video/"):
            self.handle_video(path)
            return

        # 3. Static Files
        if path == "/" or path == "/index.html":
            self.serve_file(STATIC_DIR / "index.html", "text/html")
            return
        elif path.endswith(".css"):
            self.serve_file(STATIC_DIR / Path(path).name, "text/css")
            return
        elif path.endswith(".js"):
            self.serve_file(STATIC_DIR / Path(path).name, "application/javascript")
            return
        elif path.endswith(".json"):
            self.serve_file(STATIC_DIR / Path(path).name, "application/json")
            return

        super().do_GET()

    def handle_api(self, path: str, query: dict):
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        response_data = {}

        if path == "/api/matches":
            matches = []
            for g_id, info in MES5_GAMES.items():
                chunk_counts = {}
                for h in [1, 2]:
                    mf = MANIFEST_CACHE.get((g_id, h), {})
                    chunk_counts[h] = len(mf.get("chunks", []))
                matches.append({
                    "game_id": g_id,
                    "title": info["title"],
                    "date": info["date"],
                    "season": info["season"],
                    "halves": [1, 2],
                    "chunk_counts": chunk_counts
                })
            response_data = {"matches": matches, "models": MODEL_CONFIGS}

        elif path == "/api/highlights":
            response_data = {"highlights": get_curated_highlights()}

        elif path == "/api/scorecard":
            scorecard_path = LOGS_DIR / "mes5_stage2_simulation_summary.json"
            m15_summary_path = LOGS_DIR / "mes5_specialist_3m_m15/mes5_evaluation_summary.json"
            base_summary_path = LOGS_DIR / "mes5_specialist_3m/mes5_evaluation_summary.json"
            qwen_summary_path = LOGS_DIR / "mes5_qwen3.5-4b/mes5_evaluation_summary.json"
            mini_summary_path = LOGS_DIR / "mes5_minicpm-v-4.6/mes5_evaluation_summary.json"
            lfm_summary_path = LOGS_DIR / "mes5_lfm2.5-vl-3b/mes5_evaluation_summary.json"

            def safe_load(p):
                if p.exists():
                    with open(p, "r", encoding="utf-8") as f:
                        return json.load(f)
                return {}

            stage2_data = safe_load(scorecard_path)
            m15_data = safe_load(m15_summary_path)
            base_data = safe_load(base_summary_path)
            qwen_data = safe_load(qwen_summary_path)
            mini_data = safe_load(mini_summary_path)
            lfm_data = safe_load(lfm_summary_path)

            response_data = {
                "overall": {
                    "stage2_loose_mAP": stage2_data.get("stage2_loose_mAP", 0.1840),
                    "stage2_tight_mAP": stage2_data.get("stage2_tight_mAP", 0.0609),
                    "m15_loose_mAP": m15_data.get("loose_average_mAP", 0.1833),
                    "m15_tight_mAP": m15_data.get("tight_average_mAP", 0.0607),
                    "base_loose_mAP": base_data.get("loose_average_mAP", 0.1733),
                    "qwen_loose_mAP": qwen_data.get("loose_average_mAP", 0.1564),
                    "mini_loose_mAP": mini_data.get("loose_average_mAP", 0.0809),
                    "lfm_loose_mAP": lfm_data.get("loose_average_mAP", 0.0206),
                    "active_classes": 17,
                    "retention_pct": stage2_data.get("retention_loose_pct", 100.37),
                    "windows_skipped": stage2_data.get("skipped_windows", 18),
                    "total_windows": stage2_data.get("total_windows", 687),
                    "acceleration": "2.0x - 3.3x"
                },
                "per_class": {
                    "stage2_loose": stage2_data.get("per_class_loose", {}),
                    "m15_loose": m15_data.get("loose_per_class", {}),
                    "base_loose": base_data.get("loose_per_class", {}),
                    "qwen_loose": qwen_data.get("loose_per_class", {}),
                    "mini_loose": mini_data.get("loose_per_class", {}),
                    "lfm_loose": lfm_data.get("loose_per_class", {}),
                    "m15_tight": m15_data.get("tight_per_class", {}),
                    "stage2_tight": stage2_data.get("per_class_tight", {})
                }
            }

        elif path == "/api/reval":
            reval_dir = LOGS_DIR.parent / "reval_highlights"
            leaderboard_path = reval_dir / "reval_mes5_athlete_leaderboard.json"
            leaderboard = []
            if leaderboard_path.exists():
                with open(leaderboard_path, "r", encoding="utf-8") as f:
                    leaderboard = json.load(f).get("leaderboard", [])
            else:
                from processing.reval_scoring import attribute_events_to_athletes
                res = attribute_events_to_athletes({})
                leaderboard = sorted(res.values(), key=lambda a: a["reval_score"], reverse=True)

            reels = []
            if reval_dir.exists():
                for mp4 in reval_dir.glob("*.mp4"):
                    # Find athlete for reel
                    ath_name = mp4.stem.replace("_reel", "").replace("_", " ").title()
                    reels.append({
                        "filename": mp4.name,
                        "title": ath_name,
                        "video_url": f"/video/logs/reval_highlights/{mp4.name}"
                    })

            response_data = {
                "leaderboard": leaderboard,
                "reels": reels,
                "pillars": {
                    "action_performance": "45% (Rarity-weighted event impact)",
                    "broadcast_exposure": "30% (Track F Close-Up dwell seconds)",
                    "crowd_resonance": "25% (STFT decibel roar surge)"
                }
            }

        elif path == "/api/chunk_data":
            g_id = int(query.get("game_id", [0])[0])
            half = int(query.get("half", [1])[0])
            c_idx = int(query.get("chunk_index", [0])[0])
            model_key = query.get("model_set", ["mece_m15"])[0]

            manifest = MANIFEST_CACHE.get((g_id, half), {})
            chunks = manifest.get("chunks", [])
            selected_chunk = None
            for c in chunks:
                if c["chunk_index"] == c_idx:
                    selected_chunk = c
                    break

            if not selected_chunk and chunks:
                selected_chunk = chunks[0]

            if selected_chunk:
                c_start = selected_chunk["actual_start_sec"]
                c_dur = selected_chunk["duration_sec"]
                c_end = c_start + c_dur
                rel_path = selected_chunk["relative_path"].replace("\\", "/")

                # Ground Truth in chunk
                all_gt = GT_CACHE.get(g_id, [])
                chunk_gt = []
                for a in all_gt:
                    gt_half = int(a["gameTime"][0]) if ("gameTime" in a and a["gameTime"][0] in "12") else int(a.get("half", 1))
                    if gt_half == half:
                        pos_sec = float(a.get("position", 0)) / 1000.0
                        if c_start <= pos_sec <= c_end:
                            chunk_gt.append({
                                "label": a.get("label"),
                                "gameTime": a.get("gameTime"),
                                "pos_sec": round(pos_sec, 2),
                                "rel_sec": round(pos_sec - c_start, 2),
                                "team": a.get("team", "not applicable"),
                                "visibility": a.get("visibility", "visible")
                            })

                # Predictions in chunk
                preds = PRED_CACHE.get(model_key, {}).get(g_id, [])
                chunk_preds = []
                for p in preds:
                    p_half = int(p.get("half", 1))
                    if p_half == half:
                        pos_sec = float(p.get("position", 0)) / 1000.0
                        if c_start <= pos_sec <= c_end:
                            chunk_preds.append({
                                "label": p.get("label"),
                                "gameTime": p.get("gameTime"),
                                "pos_sec": round(pos_sec, 2),
                                "rel_sec": round(pos_sec - c_start, 2),
                                "confidence": round(float(p.get("confidence", 0.0)), 3),
                                "model": p.get("model", "ensemble"),
                                "causal_support": round(float(p.get("causal_support", 0.0)), 3)
                            })

                # DSP profiles
                dsp_file = DSP_CACHE_DIR / f"{Path(selected_chunk['filename']).stem}_dsp.npz"
                dsp_data = {
                    "whistle_z": [],
                    "crowd_db": [],
                    "optical_mad": []
                }
                w_z_arr, c_db_arr, opt_arr = None, None, None
                if dsp_file.exists():
                    npz = np.load(str(dsp_file))
                    w_z_arr = npz["whistle_z"]
                    c_db_arr = npz["crowd_db"]
                    opt_arr = npz["optical_diffs"]
                    dsp_data["whistle_z"] = [round(float(v), 2) for v in w_z_arr]
                    dsp_data["crowd_db"] = [round(float(v), 2) for v in c_db_arr]
                    dsp_data["optical_mad"] = [round(float(v), 2) for v in opt_arr]

                # Sliding windows & CoT
                raw_windows = RAW_WINDOWS_CACHE.get((g_id, half), [])
                chunk_windows = []
                for rw in raw_windows:
                    if rw.get("chunk_index") == c_idx:
                        w_idx = rw.get("window_index", 0)
                        # Window bounds (stride=45s, window=60s)
                        w_start = w_idx * 45.0
                        w_end = min(w_start + 60.0, c_dur)

                        # Gate evaluation
                        is_active = True
                        reason = "Active Play"
                        max_wz = 0.0
                        max_cdb = 0.0
                        max_opt = 0.0
                        std_opt = 0.0

                        if w_z_arr is not None and len(w_z_arr) > 0:
                            idx0 = min(int(w_start * 2), len(w_z_arr) - 1)
                            idx1 = min(int(w_end * 2), len(w_z_arr))
                            if idx1 > idx0:
                                max_wz = float(np.max(w_z_arr[idx0:idx1]))
                                max_cdb = float(np.max(c_db_arr[idx0:idx1]))

                        if opt_arr is not None and len(opt_arr) > 0:
                            opt0 = min(int(w_start), len(opt_arr) - 1)
                            opt1 = min(int(w_end), len(opt_arr))
                            if opt1 > opt0:
                                max_opt = float(np.max(opt_arr[opt0:opt1]))
                                std_opt = float(np.std(opt_arr[opt0:opt1]))

                        # Pareto decision gate
                        triggers = []
                        if max_wz >= 4.5:
                            triggers.append(f"Whistle Resonance (z={max_wz:.1f} >= 4.5)")
                        if max_cdb >= 6.0:
                            triggers.append(f"Crowd Surge (+{max_cdb:.1f} dB >= 6.0)")
                        if max_opt >= 30.0:
                            triggers.append(f"Scene Cut (MAD={max_opt:.1f} >= 30.0)")
                        if std_opt >= 6.0:
                            triggers.append(f"Motion Variance (σ={std_opt:.1f} >= 6.0)")

                        if triggers:
                            is_active = True
                            reason = " + ".join(triggers)
                        else:
                            is_active = False
                            reason = "Midfield Inactive (Skipped in 0.001s CPU)"

                        # Parse commentary
                        raw_txt = rw.get("raw_output", "")
                        commentary = ""
                        predicted_in_window = []
                        try:
                            clean_json = raw_txt.strip()
                            if clean_json.startswith("```json"):
                                clean_json = clean_json[7:]
                            if clean_json.endswith("```"):
                                clean_json = clean_json[:-3]
                            parsed = json.loads(clean_json.strip())
                            if isinstance(parsed, list):
                                for item in parsed:
                                    if "commentary" in item:
                                        commentary += item["commentary"] + " "
                                    if "label" in item:
                                        predicted_in_window.append(item)
                        except Exception:
                            commentary = raw_txt[:160]

                        chunk_windows.append({
                            "window_index": w_idx,
                            "start_sec": round(w_start, 1),
                            "end_sec": round(w_end, 1),
                            "is_active": is_active,
                            "trigger_reason": reason,
                            "max_whistle_z": round(max_wz, 2),
                            "max_crowd_db": round(max_cdb, 2),
                            "max_optical_mad": round(max_opt, 2),
                            "commentary": commentary.strip(),
                            "predicted_events": predicted_in_window
                        })

                response_data = {
                    "chunk": {
                        "game_id": g_id,
                        "half": half,
                        "chunk_index": c_idx,
                        "filename": selected_chunk["filename"],
                        "actual_start_sec": round(c_start, 2),
                        "duration_sec": round(c_dur, 2),
                        "video_url": f"/video/{rel_path}"
                    },
                    "ground_truth": chunk_gt,
                    "predictions": chunk_preds,
                    "dsp": dsp_data,
                    "windows": chunk_windows
                }

        self.wfile.write(json.dumps(response_data).encode("utf-8"))

    def handle_video(self, path: str):
        """Streams MP4 video chunks with proper HTTP 206 Partial Content (Range) support."""
        rel_path = path[7:]  # remove /video/
        file_path = WORKSPACE_ROOT / rel_path

        if not file_path.exists() or not file_path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND, f"Video chunk not found: {rel_path}")
            return

        total_size = file_path.stat().st_size
        range_header = self.headers.get("Range")

        if not range_header:
            # Full file requested
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Content-Length", str(total_size))
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            with open(file_path, "rb") as f:
                self.copyfile(f, self.wfile)
            return

        # Parse Range header: bytes=start-end
        match = re.match(r"bytes=(\d*)-(\d*)", range_header.strip())
        if not match:
            self.send_error(HTTPStatus.BAD_REQUEST, "Invalid Range Header")
            return

        start_str, end_str = match.groups()
        if start_str and end_str:
            start = int(start_str)
            end = int(end_str)
        elif start_str:
            start = int(start_str)
            end = total_size - 1
        elif end_str:
            end = total_size - 1
            start = max(0, total_size - int(end_str))
        else:
            start = 0
            end = total_size - 1

        if start >= total_size or end >= total_size or start > end:
            self.send_response(HTTPStatus.RANGE_NOT_SATISFIABLE)
            self.send_header("Content-Range", f"bytes */{total_size}")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            return

        length = end - start + 1
        self.send_response(HTTPStatus.PARTIAL_CONTENT)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Content-Range", f"bytes {start}-{end}/{total_size}")
        self.send_header("Content-Length", str(length))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        with open(file_path, "rb") as f:
            f.seek(start)
            remaining = length
            chunk_size = 64 * 1024
            while remaining > 0:
                chunk = f.read(min(remaining, chunk_size))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (ConnectionResetError, BrokenPipeError):
                    break
                remaining -= len(chunk)

    def serve_file(self, file_path: Path, content_type: str):
        if not file_path.exists():
            self.send_error(HTTPStatus.NOT_FOUND, f"File {file_path.name} not found")
            return
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(file_path.stat().st_size))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        with open(file_path, "rb") as f:
            self.copyfile(f, self.wfile)


def run_demo_server(port: int = 8080, host: str = "127.0.0.1"):
    load_all_caches()
    server_address = (host, port)
    httpd = ThreadingHTTPServer(server_address, DemoRequestHandler)
    print(f"\n========================================================")
    print(f" SoccerNet VLM Interactive Showcase Live on:")
    print(f" http://localhost:{port} (or http://{host}:{port})")
    print(f"========================================================\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server.")
        httpd.server_close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SoccerNet VLM Interactive Demonstration Server")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind server (default: 8080)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host interface (default: 127.0.0.1)")
    args = parser.parse_args()

    run_demo_server(port=args.port, host=args.host)

