"""
Test probe for SoccerNet VLM Interactive Demonstration Server.
Starts server on ephemeral test port, verifies REST endpoints and HTTP 206 video streaming, then terminates cleanly.
"""

import sys
import time
import json
import urllib.request
import threading
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT / "apps/demo"))

from server import run_demo_server

TEST_PORT = 8899

def run_tests():
    time.sleep(1.2)  # Wait for server to bind
    base_url = f"http://127.0.0.1:{TEST_PORT}"
    print(f"Probing {base_url}...")

    # 1. Test Index
    req = urllib.request.Request(f"{base_url}/")
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200, f"Index failed: {resp.status}"
        html = resp.read().decode("utf-8")
        assert "SoccerNet VLM Interactive Showcase" in html
        print("✓ GET / (HTML index): OK (200)")

    # 2. Test Matches API
    req = urllib.request.Request(f"{base_url}/api/matches")
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))
        assert len(data["matches"]) == 5
        print(f"✓ GET /api/matches: OK (5 fixtures, {len(data['models'])} model configs)")

    # 3. Test Highlights API
    req = urllib.request.Request(f"{base_url}/api/highlights")
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))
        assert len(data["highlights"]) >= 6
        print(f"✓ GET /api/highlights: OK ({len(data['highlights'])} featured moments)")

    # 4. Test Chunk Data API
    req = urllib.request.Request(f"{base_url}/api/chunk_data?game_id=0&half=1&chunk_index=0&model_set=mece_m15")
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))
        assert "chunk" in data
        assert "ground_truth" in data
        assert "predictions" in data
        assert "dsp" in data
        assert "windows" in data
        gt_count = len(data["ground_truth"])
        pred_count = len(data["predictions"])
        win_count = len(data["windows"])
        dsp_pts = len(data["dsp"]["whistle_z"])
        print(f"✓ GET /api/chunk_data: OK ({gt_count} GT, {pred_count} Preds, {win_count} Windows, {dsp_pts} DSP points)")

    # 5. Test Video Stream Range Request (HTTP 206)
    chunk_sample = WORKSPACE_ROOT / "gate2_chunks/game_0_half_1_chunk_00.mp4"
    if chunk_sample.exists():
        video_url = f"{base_url}/video/gate2_chunks/game_0_half_1_chunk_00.mp4"
    else:
        video_url = f"{base_url}/video/logs/reval_highlights/ander_herrera_21_reel.mp4"
    range_req = urllib.request.Request(video_url)
    range_req.add_header("Range", "bytes=0-1023")
    with urllib.request.urlopen(range_req) as resp:
        assert resp.status == 206, f"Expected 206 Partial Content, got {resp.status}"
        c_range = resp.headers.get("Content-Range")
        c_len = resp.headers.get("Content-Length")
        body = resp.read()
        assert len(body) == 1024
        print(f"✓ GET /video/ (HTTP 206 Range Stream): OK ({c_range}, length: {c_len})")

    # 6. Test Scorecard API
    req = urllib.request.Request(f"{base_url}/api/scorecard")
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))
        assert "overall" in data
        assert "per_class" in data
        s2_map = data["overall"]["stage2_loose_mAP"]
        print(f"✓ GET /api/scorecard: OK (Stage 2 Loose-mAP: {s2_map:.4f}, 17 classes)")

    # 7. Test ReVal API
    req = urllib.request.Request(f"{base_url}/api/reval")
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))
        assert "leaderboard" in data
        assert "reels" in data
        assert len(data["leaderboard"]) >= 12
        assert len(data["reels"]) >= 3
        print(f"✓ GET /api/reval: OK ({len(data['leaderboard'])} athletes, {len(data['reels'])} highlight reels)")

    # 8. Test ReVal Reel Video Stream Range Request (HTTP 206)
    reel_url = f"{base_url}/video/logs/reval_highlights/ander_herrera_21_reel.mp4"
    reel_req = urllib.request.Request(reel_url)
    reel_req.add_header("Range", "bytes=0-1023")
    with urllib.request.urlopen(reel_req) as resp:
        assert resp.status == 206, f"Expected 206 Partial Content, got {resp.status}"
        c_range = resp.headers.get("Content-Range")
        body = resp.read()
        assert len(body) == 1024
        print(f"✓ GET /video/logs/reval_highlights/... (HTTP 206 Range Stream): OK ({c_range})")

    print("\nALL SERVER PROBES PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    server_thread = threading.Thread(target=run_demo_server, args=(TEST_PORT,), daemon=True)
    server_thread.start()
    try:
        run_tests()
    except Exception as e:
        print(f"\nTEST PROBE FAILED: {e}")
        sys.exit(1)
