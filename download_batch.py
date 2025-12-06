import os
import sys
from SoccerNet.Downloader import SoccerNetDownloader
from SoccerNet.utils import getListGames

# ============================================================================
# CONFIGURATION
# ============================================================================

BATCH_NUMBER = 1
NDA_PASSWORD = "s0cc3rn3t"
RESOLUTION = "720p"

# HPC PATHS
BASE_DIR = os.path.join(os.environ['HOME'], "soccer_project")
OUTPUT_DIR = os.path.join(BASE_DIR, "raw_data")

# ============================================================================
# MAIN SCRIPT
# ============================================================================

print(f"📂 Setting up download to: {OUTPUT_DIR}")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Get Game List
print("🔍 Finding EPL games...")
all_epl_games = []
for split in ["train", "valid", "test", "challenge"]:
    games = getListGames(split=split)
    epl_games = [g for g in games if g.startswith('england_epl/')]
    all_epl_games.extend(epl_games)

# 2. Select Batch
batch_1 = all_epl_games[0:20]
batch_2 = all_epl_games[20:40]
batch_3 = all_epl_games[40:60]
batch_4 = all_epl_games[60:80]
batch_5 = all_epl_games[80:104]

batches = {1: batch_1, 2: batch_2, 3: batch_3, 4: batch_4, 5: batch_5}
current_batch = batches[BATCH_NUMBER]

print(f"📦 Batch {BATCH_NUMBER}: {len(current_batch)} games")

# 3. Start Download
downloader = SoccerNetDownloader(LocalDirectory=OUTPUT_DIR)
downloader.password = NDA_PASSWORD

files_to_download = ["1_720p.mkv", "2_720p.mkv", "Labels-v2.json"] if RESOLUTION == "720p" else ["1_224p.mkv", "2_224p.mkv", "Labels-v2.json"]

print(f"{'='*40}")
print("STARTING DOWNLOAD")
print(f"{'='*40}")

for i, game in enumerate(current_batch, 1):
    try:
        print(f"[{i}/{len(current_batch)}] Downloading: {game}")
        downloader.downloadGame(files=files_to_download, game=game)
        print(f"   ✅ Complete\n")
    except Exception as e:
        print(f"   ❌ Failed: {e}\n")

print("🎉 Batch Download Finished.")



