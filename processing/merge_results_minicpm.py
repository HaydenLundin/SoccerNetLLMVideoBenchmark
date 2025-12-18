import json
import glob
import os
import argparse
import re

#Argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--video_index", type=int, required=True)
parser.add_argument("--debug", action="store_true", help="Show detailed deduplication info")
args = parser.parse_args()

# CONFIG
PROJECT_DIR = "/users/hlundin/soccer_project"
INPUT_PATTERN = f"{PROJECT_DIR}/partial_results_minicpm_vid{args.video_index}_gpu*.json"
OUTPUT_FILE = f"{PROJECT_DIR}/final_predictions_minicpm_video_{args.video_index}.json"
DEDUP_WINDOW = 3.0  # seconds

print(f"🔍 Merging MiniCPM-V 2.6 results for Video Index {args.video_index}...")

all_events = []

# 1. Load all partial files
print("📂 Loading partial files...")
files = sorted(glob.glob(INPUT_PATTERN))
if not files:
    print(f"❌ No files found matching pattern: {INPUT_PATTERN}")
    exit(1)

for f in files:
    try:
        with open(f, 'r') as json_file:
            data = json.load(json_file)
            all_events.extend(data)
            print(f"   Loaded {len(data)} events from {os.path.basename(f)}")
    except Exception as e:
        print(f"   ❌ Error loading {f}: {e}")

# 2. Sort by Time
print(f"🔄 Sorting {len(all_events)} total detections...")
all_events.sort(key=lambda x: x['time'])

# 3. Enhanced Deduplication
print(f"🧹 Deduplicating events (window: {DEDUP_WINDOW}s)...")

def parse_event_data(raw_text):
    """
    Parse LLM output to extract event info.
    Handles multiple formats:
    - Valid JSON with double quotes
    - JSON with single quotes
    - Multiple events separated by newlines
    - Text surrounding JSON
    """
    if not raw_text or raw_text == "None":
        return []

    events = []

    # Try to find JSON objects (with single or double quotes)
    # Pattern matches {...} structures
    json_pattern = r'\{[^{}]*\}'
    matches = re.findall(json_pattern, str(raw_text))

    for match in matches:
        try:
            # Try parsing as-is (double quotes)
            event = json.loads(match)
            events.append(event)
        except json.JSONDecodeError:
            try:
                # Try replacing single quotes with double quotes
                fixed = match.replace("'", '"')
                event = json.loads(fixed)
                events.append(event)
            except json.JSONDecodeError:
                if args.debug:
                    print(f"   ⚠️ Could not parse: {match[:100]}...")
                continue

    return events

def events_are_duplicate(event1, event2, time_diff):
    """
    Check if two events are duplicates.
    Same label + within time window = duplicate
    """
    if time_diff > DEDUP_WINDOW:
        return False

    # Parse both events
    parsed1 = parse_event_data(event1.get('raw', ''))
    parsed2 = parse_event_data(event2.get('raw', ''))

    if not parsed1 or not parsed2:
        return False

    # Compare labels from first parsed event in each
    label1 = parsed1[0].get('label', '').lower().strip()
    label2 = parsed2[0].get('label', '').lower().strip()

    if not label1 or not label2:
        return False

    return label1 == label2

# Deduplicate
final_events = []
duplicates_removed = 0
parsing_failures = 0

for current in all_events:
    is_duplicate = False

    # Check against all events in the recent time window
    for recent in reversed(final_events[-10:]):  # Check last 10 events for efficiency
        time_diff = abs(current['time'] - recent['time'])

        if time_diff > DEDUP_WINDOW:
            break  # Events are sorted, so we can stop checking

        try:
            if events_are_duplicate(current, recent, time_diff):
                is_duplicate = True
                duplicates_removed += 1

                if args.debug:
                    cur_parsed = parse_event_data(current.get('raw', ''))
                    label = cur_parsed[0].get('label', 'UNKNOWN') if cur_parsed else 'UNKNOWN'
                    print(f"   🗑️ Duplicate: {label} at {current['time']:.1f}s (within {time_diff:.1f}s of {recent['time']:.1f}s)")
                break
        except Exception as e:
            parsing_failures += 1
            if args.debug:
                print(f"   ⚠️ Error comparing events: {e}")
            # If we can't parse, keep it to be safe
            continue

    if not is_duplicate:
        final_events.append(current)

print(f"✅ Merged {len(all_events)} -> {len(final_events)} unique events")
print(f"   Removed {duplicates_removed} duplicates")
if parsing_failures > 0:
    print(f"   ⚠️ {parsing_failures} parsing failures (kept events to be safe)")

# 4. Save
with open(OUTPUT_FILE, 'w') as f:
    json.dump(final_events, f, indent=2)
print(f"💾 Saved to {OUTPUT_FILE}")

# 5. Optional: Show sample of what we kept
if args.debug and final_events:
    print(f"\n📊 Sample of merged events:")
    for event in final_events[:5]:
        parsed = parse_event_data(event.get('raw', ''))
        if parsed:
            label = parsed[0].get('label', 'UNKNOWN')
            print(f"   {event['time']:.1f}s: {label}")
