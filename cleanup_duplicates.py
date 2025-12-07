#!/usr/bin/env python3
"""
Cleanup/Deduplicate Already-Merged Result Files

This script takes already-merged prediction files and removes duplicates.
Use this for cleaning up files that were merged with the old script.

Usage:
    python cleanup_duplicates.py --input final_predictions_qwen7b_video_0.json
    python cleanup_duplicates.py --input final_predictions_qwen7b_video_0.json --output cleaned_video_0.json
    python cleanup_duplicates.py --input final_predictions_qwen7b_video_0.json --inplace --debug
"""

import json
import argparse
import re
import os

# Argument parser
parser = argparse.ArgumentParser(description="Deduplicate already-merged event detection results")
parser.add_argument("--input", type=str, required=True, help="Input JSON file to clean")
parser.add_argument("--output", type=str, help="Output file (default: adds '_cleaned' suffix)")
parser.add_argument("--inplace", action="store_true", help="Overwrite input file with cleaned version")
parser.add_argument("--debug", action="store_true", help="Show detailed deduplication info")
parser.add_argument("--window", type=float, default=3.0, help="Deduplication time window in seconds (default: 3.0)")
args = parser.parse_args()

# Determine output file
if args.inplace:
    output_file = args.input
elif args.output:
    output_file = args.output
else:
    # Add '_cleaned' before extension
    base, ext = os.path.splitext(args.input)
    output_file = f"{base}_cleaned{ext}"

print(f"🧹 Cleaning up duplicates from: {args.input}")
print(f"   Time window: {args.window}s")

# 1. Load the file
try:
    with open(args.input, 'r') as f:
        all_events = json.load(f)
    print(f"📂 Loaded {len(all_events)} events")
except FileNotFoundError:
    print(f"❌ File not found: {args.input}")
    exit(1)
except json.JSONDecodeError as e:
    print(f"❌ Invalid JSON: {e}")
    exit(1)

# 2. Sort by time (in case not already sorted)
all_events.sort(key=lambda x: x.get('time', 0))

# 3. Deduplication functions
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
    if time_diff > args.window:
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

# 4. Deduplicate
final_events = []
duplicates_removed = 0
parsing_failures = 0

for current in all_events:
    is_duplicate = False

    # Check against all events in the recent time window
    for recent in reversed(final_events[-10:]):  # Check last 10 events for efficiency
        time_diff = abs(current.get('time', 0) - recent.get('time', 0))

        if time_diff > args.window:
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
            continue

    if not is_duplicate:
        final_events.append(current)

# 5. Report results
print(f"✅ Cleaned {len(all_events)} -> {len(final_events)} unique events")
print(f"   Removed {duplicates_removed} duplicates ({duplicates_removed/len(all_events)*100:.1f}%)")
if parsing_failures > 0:
    print(f"   ⚠️ {parsing_failures} parsing failures (kept events to be safe)")

# 6. Save
with open(output_file, 'w') as f:
    json.dump(final_events, f, indent=2)

if args.inplace:
    print(f"💾 Overwrote {output_file}")
else:
    print(f"💾 Saved to {output_file}")

# 7. Optional: Show sample of what we kept
if args.debug and final_events:
    print(f"\n📊 Sample of cleaned events:")
    for event in final_events[:10]:
        parsed = parse_event_data(event.get('raw', ''))
        if parsed:
            label = parsed[0].get('label', 'UNKNOWN')
            team = parsed[0].get('team', '?')
            print(f"   {event['time']:6.1f}s: {label:20s} ({team})")
