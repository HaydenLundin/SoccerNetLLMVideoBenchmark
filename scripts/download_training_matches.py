"""
Download and slice remaining training halves for Game 4 and Game 7 sequentially.
Target: (4, 2), (7, 1), (7, 2).
"""

import sys
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from scripts.download_and_slice_game import download_and_slice

def main():
    targets = [
        (4, 2),
        (7, 1),
        (7, 2)
    ]
    for gid, half in targets:
        print(f"\n=======================================================")
        print(f"  Starting Download & Slice for Game {gid} - Half {half}")
        print(f"=======================================================")
        try:
            download_and_slice(game_id=gid, half=half)
            print(f"[+] Successfully finished Game {gid} - Half {half}")
        except Exception as e:
            print(f"[!] Error on Game {gid} - Half {half}: {e}")
            raise e

if __name__ == "__main__":
    main()
