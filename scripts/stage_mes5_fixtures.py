"""
Stage all video fixtures for the Canonical MES-5 benchmark.
Downloads 720p MKVs from SoccerNet/SoccerNet_raw_HQ (authenticated via HF_TOKEN)
and slices them into zero-drift macro-chunks with exact keyframe compensation.
"""

import sys
import time
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from scripts.download_and_slice_game import download_and_slice

# Canonical MES-5 fixtures (Games 0, 1, 2, 3, 13)
# Game 0 is already fully downloaded, sliced, and evaluated.
MES5_REMAINING = [
    (1, 2),
    (2, 1),
    (2, 2),
    (3, 1),
    (3, 2),
    (13, 1),
    (13, 2),
]

def main():
    print("=" * 80)
    print("  CANONICAL MES-5 VIDEO STAGING PIPELINE")
    print(f"  Target: {len(MES5_REMAINING)} halves to download and slice")
    print("=" * 80)

    total_start = time.perf_counter()
    for idx, (game_id, half) in enumerate(MES5_REMAINING, 1):
        print(f"\n[{idx}/{len(MES5_REMAINING)}] Processing Game {game_id} Half {half}...")
        t0 = time.perf_counter()
        try:
            download_and_slice(game_id=game_id, half=half)
            elapsed = time.perf_counter() - t0
            print(f"[+] Completed Game {game_id} Half {half} in {elapsed:.1f}s ({elapsed/60:.2f} min)")
        except Exception as e:
            print(f"[!] Error processing Game {game_id} Half {half}: {e}")
            raise e

    total_elapsed = time.perf_counter() - total_start
    print("\n" + "=" * 80)
    print(f"  ALL MES-5 FIXTURES SUCCESSFULLY STAGED IN {total_elapsed/60:.2f} MINUTES!")
    print("=" * 80)

if __name__ == "__main__":
    main()
