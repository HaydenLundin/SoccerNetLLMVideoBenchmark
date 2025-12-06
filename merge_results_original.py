import json
import glob
import os
import argparse

#Argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--video_index", type=int, required=True)
args = parser.parse_args()
# CONFIG
INPUT_PATTERN = f"/users/hlundin/soccer_project/partial_results_qwen7b_vid{args.video_index}_gpu*.json"
OUTPUT_FILE = f"final_predictions_qwen7b_video_{args.video_index}.json"

print(f"🔍 Merging results for Video Index {args.video_index}...")

all_events = []

# 1. Load all partial files
print("📂 Loading partial files...")
files = glob.glob(INPUT_PATTERN)
for f in files:
    try:
        with open(f, 'r') as json_file:
            data = json.load(json_file)
            all_events.extend(data)
            print(f"   Loaded {len(data)} events from {f}")
    except Exception as e:
        print(f"   ❌ Error loading {f}: {e}")

# 2. Sort by Time
print(f"🔄 Sorting {len(all_events)} total detections...")
all_events.sort(key=lambda x: x['time'])

# 3. Deduplicate (The Fix for "Foul Storm")
# If the same event label appears within 3 seconds, keep only the first one.
final_events = []
if all_events:
    final_events.append(all_events[0])
    for current in all_events[1:]:
        last = final_events[-1]

        # Parse JSON strings if needed
        try:
            cur_json = json.loads(current['raw']) if isinstance(current['raw'], str) else current['raw']
            last_json = json.loads(last['raw']) if isinstance(last['raw'], str) else last['raw']

            # Check for duplicate (Same Label + Same Team + Close Time)
            # Note: You might need to adjust parsing depending on exact format output by model
            # For this quick pass, we assume 'raw' is the JSON string
            if (cur_json.get('label') == last_json.get('label') and
                abs(current['time'] - last['time']) < 3.0):
                continue # Skip duplicate
        except:
            pass # Keep if parsing fails (safety)

        final_events.append(current)

print(f"✅ Merged {len(all_events)} -> {len(final_events)} unique events.")

# 4. Save
with open(OUTPUT_FILE, 'w') as f:
    json.dump(final_events, f, indent=2)
print(f"💾 Saved to {OUTPUT_FILE}")
