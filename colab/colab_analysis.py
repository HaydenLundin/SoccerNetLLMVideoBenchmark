pip install soccernet 

from SoccerNet.Downloader import SoccerNetDownloader
import os

# Define the local directory where the data will be saved
local_dir = os.path.join(os.getcwd(), "SoccerNet_Data")

# Create the downloader instance
mySoccerNetDownloader = SoccerNetDownloader(LocalDirectory=local_dir)

print(f"Downloading Labels-v2.json files to: {local_dir}")

# Download Labels-v2.json for all splits (train, valid, test)
# 'Labels-v2.json' contains the 17 action spotting classes for v2.
mySoccerNetDownloader.downloadGames(
    files=["Labels-v2.json"],
    split=["train", "valid", "test"]
)

print("Download complete.")

#for soccernet
import os
import json
import pandas as pd

# ⚠️ IMPORTANT — this must be your already-filtered list of 20 game paths
selected_games = [
    "/content/SoccerNet_Data/england_epl/2015-2016/2016-01-13 - 22-45 Chelsea 2 - 2 West Brom",
    "/content/SoccerNet_Data/england_epl/2015-2016/2016-04-09 - 19-30 Manchester City 2 - 1 West Brom",
    "/content/SoccerNet_Data/england_epl/2015-2016/2016-01-03 - 16-30 Crystal Palace 0 - 3 Chelsea",
    "/content/SoccerNet_Data/england_epl/2015-2016/2015-08-29 - 17-00 Chelsea 1 - 2 Crystal Palace",
    "/content/SoccerNet_Data/england_epl/2015-2016/2015-10-04 - 19-15 Manchester City 1 - 2 Tottenham",
    "/content/SoccerNet_Data/england_epl/2015-2016/2015-11-07 - 18-00 Chelsea 1 - 0 Aston Villa",
    "/content/SoccerNet_Data/england_epl/2015-2016/2015-09-12 - 14-45 Everton 3 - 1 Chelsea",
    "/content/SoccerNet_Data/england_epl/2015-2016/2015-12-05 - 18-00 Chelsea 1 - 0 Bournemouth",
    "/content/SoccerNet_Data/england_epl/2015-2016/2016-03-02 - 23-00 Liverpool 3 - 0 Manchester City",
    "/content/SoccerNet_Data/england_epl/2015-2016/2016-05-07 - 17-00 Sunderland 3 - 2 Chelsea",
    "/content/SoccerNet_Data/england_epl/2015-2016/2015-08-08 - 19-30 Chelsea 2 - 2 Swansea",
    "/content/SoccerNet_Data/england_epl/2015-2016/2015-12-19 - 18-00 Chelsea 3 - 1 Sunderland",
    "/content/SoccerNet_Data/england_epl/2015-2016/2015-10-31 - 15-45 Chelsea 1 - 3 Liverpool",
    "/content/SoccerNet_Data/england_epl/2015-2016/2015-08-29 - 17-00 Manchester City 2 - 0 Watford",
    "/content/SoccerNet_Data/england_epl/2015-2016/2016-03-19 - 18-00 Chelsea 2 - 2 West Ham",
    "/content/SoccerNet_Data/england_epl/2015-2016/2016-04-09 - 17-00 Swansea 1 - 0 Chelsea",
    "/content/SoccerNet_Data/england_epl/2015-2016/2015-11-07 - 18-00 Manchester United 2 - 0 West Brom",
    "/content/SoccerNet_Data/england_epl/2015-2016/2015-11-21 - 20-30 Manchester City 1 - 4 Liverpool",
    "/content/SoccerNet_Data/england_epl/2015-2016/2015-09-19 - 19-30 Manchester City 1 - 2 West Ham",
    "/content/SoccerNet_Data/england_epl/2015-2016/2015-09-12 - 17-00 Crystal Palace 0 - 1 Manchester City",
]


records = []

for game_dir in selected_games:

    season = os.path.basename(os.path.dirname(game_dir))
    game = os.path.basename(game_dir)

    label_path = os.path.join(game_dir, "Labels-v2.json")
    if not os.path.exists(label_path):
        continue

    with open(label_path, "r") as f:
        data = json.load(f)

    for ann in data.get("annotations", []):
        
        game_time = ann.get("gameTime", "")
        half, clock = None, None
        
        if " - " in game_time:
            half, clock = game_time.split(" - ")

        position = ann.get("position", [None, None])

        records.append({
            "season": season,
            "game": game,
            "game_path": game_dir,
            "half": half,
            "clock": clock,
            "label": ann.get("label"),
            "team": ann.get("team", None),
            "player": ann.get("player", None),
            "x": position[0] if isinstance(position, list) else None,
            "y": position[1] if isinstance(position, list) else None,
            "visibility": ann.get("visibility", None),
        })

df = pd.DataFrame(records)

print("Total events loaded:", len(df))
df.head()

#start of qwen json data 

import os
import shutil
import glob

# 1. Create folder
pred_folder = "predictions"
os.makedirs(pred_folder, exist_ok=True)

# 2. Find ALL prediction json files in current directory
pred_files = glob.glob("final_predictions_qwen7b_video_*.json")

print("Found files:")
for f in pred_files:
    print(" -", f)

# 3. Move each file into the new folder
for f in pred_files:
    shutil.move(f, os.path.join(pred_folder, os.path.basename(f)))

print(f"\nMoved {len(pred_files)} files into '{pred_folder}/'")

import os
import glob
import json
import pandas as pd
import re

# Folder you just created and moved files into
pred_folder = "predictions"

# Grab ALL prediction json files in that folder
pred_files = sorted(
    glob.glob(os.path.join(pred_folder, "final_predictions_qwen7b_video_*.json"))
)

print("Prediction files found:")
for f in pred_files:
    print(" -", f)

rows = []

def clean_raw_json(raw):
    """Strip ```json fences and parse JSON; return a list of event dictionaries."""
    raw = re.sub(r"```json|```", "", raw).strip()
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return parsed
        elif isinstance(parsed, dict):
            return [parsed] # Wrap single dict in a list for consistent output
        else:
            return [] # Unexpected type
    except Exception:
        return [] # Return empty list if parsing fails

def convert_seconds_to_half_clock(t):
    """Map absolute seconds to SoccerNet-style half + mm:ss clock."""
    if t < 45 * 60:
        half = "1"
        clock = f"{int(t//60):02d}:{int(t%60):02d}"
    else:
        half = "2"
        t2 = t - 45 * 60
        clock = f"{int(t2//60):02d}:{int(t2%60):02d}"
    return half, clock


for file in pred_files:
    with open(file, "r") as f:
        preds_raw = json.load(f)

    for entry in preds_raw:
        t = entry.get("time")
        raw_text = entry.get("raw", "")

        events_from_parsed = clean_raw_json(raw_text) # This will now always return a list
        if not events_from_parsed: # If the list is empty, skip
            continue

        half, clock = convert_seconds_to_half_clock(t)

        for event in events_from_parsed: # Iterate through each event in the list
            rows.append({
                "season": None,                # not in preds
                "game": None,                  # not in preds
                "game_path": file,             # which prediction file
                "half": half,
                "clock": clock,
                "label": event.get("label"),   # Use event.get() now
                "team": event.get("team"),
                "player": None,
                "x": None,
                "y": None,
                "visibility": None,
                "confidence": event.get("confidence"),
                "details": event.get("details"),
            })

# Build the final dataset
df_pred_all = pd.DataFrame(rows)

# 🔢 Count of events
print("Total prediction events loaded:", len(df_pred_all))

# Peek at the dataset
df_pred_all.head()


pred_label_counts = df_pred_all['label'].value_counts()
true_label_counts = df['label'].value_counts()

comparison = pd.concat([pred_label_counts, true_label_counts], axis=1)
comparison.columns = ["Qwen", "SoccerNet"]
print(comparison)
