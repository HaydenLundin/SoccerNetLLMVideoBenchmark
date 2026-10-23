"""
Download and slice any match half from eval_set.json into zero-drift macro-chunks.
"""

import os
import sys
import json
import argparse
from pathlib import Path

# Ensure repo root is on sys.path
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from SoccerNet.Downloader import SoccerNetDownloader
from processing.slice_macro_chunks import slice_half_into_macro_chunks

try:
    import static_ffmpeg
    static_ffmpeg.add_paths()
except ImportError:
    pass

def load_nda_password():
    env_path = WORKSPACE_ROOT / ".env"
    if not env_path.exists():
        raise FileNotFoundError(".env file not found!")
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("SOCCERNET_NDA_PASSWORD"):
                pw = line.split("=", 1)[1].strip().strip("\"'")
                if pw:
                    return pw
    raise ValueError("SOCCERNET_NDA_PASSWORD not set in .env")

def get_game_info(game_id: int):
    eval_set_path = WORKSPACE_ROOT / "eval_set.json"
    with open(eval_set_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for g in data.get("games", []):
        if g.get("game_id") == game_id:
            return g
    raise ValueError(f"Game ID {game_id} not found in eval_set.json")

def download_and_slice(game_id: int, half: int, raw_dir: str = "gate2_raw_video", chunks_dir: str = "gate2_chunks", resolution: str = "720p"):
    game_info = get_game_info(game_id)
    game_path = game_info["relative_path"]
    filename = f"{half}_{resolution}.mkv"

    raw_path = Path(raw_dir)
    target_mkv = raw_path / game_path / filename

    print(f"=== Downloading & Slicing: Game {game_id} ({game_info['match']}) - Half {half} ===")
    if not target_mkv.exists():
        print(f"[*] Downloading {filename} from SoccerNet HuggingFace...")
        password = load_nda_password()
        downloader = SoccerNetDownloader(LocalDirectory=str(raw_path))
        downloader.password = password
        try:
            downloader.downloadGame(files=[filename], game=game_path, source="HuggingFace")
        except Exception as e:
            print(f"[!] HuggingFace source failed: {e}. Retrying with source='EXRCSDrive'...")
            downloader.downloadGame(files=[filename], game=game_path, source="EXRCSDrive")
    else:
        print(f"[*] Raw video already exists: {target_mkv} ({target_mkv.stat().st_size / 1e6:.1f} MB)")

    if not target_mkv.exists():
        raise FileNotFoundError(f"Failed to find downloaded video at: {target_mkv}")

    out_chunks = Path(chunks_dir)
    out_chunks.mkdir(parents=True, exist_ok=True)

    manifest_file = out_chunks / f"game_{game_id}_half_{half}_chunks_manifest.json"
    if manifest_file.exists():
        print(f"[*] Manifest already exists: {manifest_file}")
        with open(manifest_file, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        print(f"[*] Found {len(manifest.get('chunks', []))} macro-chunks. Slicing skipped.")
        return manifest

    print(f"[*] Slicing into zero-drift macro-chunks (target_dir={out_chunks})...")
    chunks = slice_half_into_macro_chunks(
        input_video_path=str(target_mkv),
        output_dir=str(out_chunks),
        game_id=game_id,
        half=half,
        target_container="mp4"
    )
    print(f"[+] Successfully sliced {len(chunks)} macro-chunks. Manifest: {manifest_file}")
    with open(manifest_file, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    return manifest

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and slice a game half")
    parser.add_argument("--game-id", type=int, default=0, help="Game ID in eval_set.json")
    parser.add_argument("--half", type=int, default=2, choices=[1, 2], help="Half (1 or 2)")
    parser.add_argument("--raw-dir", type=str, default="gate2_raw_video", help="Raw video directory")
    parser.add_argument("--chunks-dir", type=str, default="gate2_chunks", help="Output chunks directory")
    parser.add_argument("--resolution", type=str, default="720p", help="Video resolution (720p or 224p)")
    args = parser.parse_args()

    download_and_slice(
        game_id=args.game_id,
        half=args.half,
        raw_dir=args.raw_dir,
        chunks_dir=args.chunks_dir,
        resolution=args.resolution
    )
