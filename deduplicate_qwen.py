#!/usr/bin/env python3
"""
Robust Deduplication Script for Qwen Results

This script handles deduplication of already-merged prediction files.
It uses multiple parsing strategies to handle various LLM output formats.

Usage:
    python deduplicate_qwen.py --input final_predictions_qwen7b_video_18.json
    python deduplicate_qwen.py --input final_predictions_qwen7b_video_18.json --debug
    python deduplicate_qwen.py --input final_predictions_qwen7b_video_18.json --window 5.0
    python deduplicate_qwen.py --input final_predictions_qwen7b_video_18.json --inplace
"""

import json
import argparse
import re
import ast
from difflib import SequenceMatcher
from pathlib import Path

# ============================================================================
# ARGUMENT PARSING
# ============================================================================

parser = argparse.ArgumentParser(description="Deduplicate Qwen prediction files")
parser.add_argument("--input", type=str, required=True, help="Input JSON file to deduplicate")
parser.add_argument("--output", type=str, help="Output file (default: input_deduped.json)")
parser.add_argument("--window", type=float, default=3.0, help="Time window for deduplication (seconds)")
parser.add_argument("--debug", action="store_true", help="Show detailed debugging information")
parser.add_argument("--inplace", action="store_true", help="Overwrite input file")
parser.add_argument("--show-raw", action="store_true", help="Show raw LLM outputs (for debugging)")
args = parser.parse_args()

DEDUP_WINDOW = args.window

# ============================================================================
# ENHANCED JSON PARSING
# ============================================================================

def extract_json_robust(raw_text):
    """
    Extract JSON objects from LLM output using multiple strategies.

    Returns: List of parsed event dictionaries
    """
    if not raw_text or raw_text == "None" or raw_text.strip() == "":
        return []

    events = []
    raw_text = str(raw_text).strip()

    # Strategy 1: Try to parse the entire string as JSON
    try:
        parsed = json.loads(raw_text)
        if isinstance(parsed, dict):
            events.append(parsed)
            return events
        elif isinstance(parsed, list):
            return parsed
    except (json.JSONDecodeError, ValueError):
        pass

    # Strategy 2: Find JSON objects with a more robust regex
    # This handles nested braces better
    brace_count = 0
    start_idx = None

    for i, char in enumerate(raw_text):
        if char == '{':
            if brace_count == 0:
                start_idx = i
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0 and start_idx is not None:
                candidate = raw_text[start_idx:i+1]

                # Try parsing as JSON
                try:
                    parsed = json.loads(candidate)
                    events.append(parsed)
                    start_idx = None
                    continue
                except (json.JSONDecodeError, ValueError):
                    pass

                # Try parsing as Python literal (handles single quotes)
                try:
                    # Replace Python's True/False/None with JSON equivalents
                    candidate_fixed = candidate.replace("'", '"')
                    parsed = json.loads(candidate_fixed)
                    events.append(parsed)
                    start_idx = None
                    continue
                except (json.JSONDecodeError, ValueError, SyntaxError):
                    pass

                # Try ast.literal_eval for Python dict syntax
                try:
                    parsed = ast.literal_eval(candidate)
                    if isinstance(parsed, dict):
                        events.append(parsed)
                        start_idx = None
                        continue
                except (ValueError, SyntaxError):
                    pass

                start_idx = None

    # Strategy 3: Look for common patterns even without perfect JSON
    if not events:
        # Try to extract label and team using regex patterns
        label_match = re.search(r"['\"]?label['\"]?\s*:\s*['\"]([^'\"]+)['\"]", raw_text, re.IGNORECASE)
        team_match = re.search(r"['\"]?team['\"]?\s*:\s*['\"]([^'\"]+)['\"]", raw_text, re.IGNORECASE)

        if label_match:
            event = {
                'label': label_match.group(1),
                'team': team_match.group(1) if team_match else 'unknown'
            }
            events.append(event)

    return events

def get_event_signature(event_dict):
    """
    Extract a comparable signature from an event dictionary.

    Returns: tuple of (label, team, details_snippet)
    """
    label = event_dict.get('label', '').lower().strip()
    team = event_dict.get('team', '').lower().strip()
    details = event_dict.get('details', '')[:50].lower().strip()  # First 50 chars

    return (label, team, details)

def text_similarity(text1, text2):
    """
    Calculate similarity between two text strings (0-1 scale).
    """
    return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()

# ============================================================================
# DEDUPLICATION LOGIC
# ============================================================================

def events_are_duplicate(event1, event2, time_diff, debug=False):
    """
    Determine if two events are duplicates.
    Uses multiple strategies:
    1. Parse JSON and compare event signatures
    2. Compare raw text similarity if parsing fails
    """
    if time_diff > DEDUP_WINDOW:
        return False

    raw1 = event1.get('raw', '')
    raw2 = event2.get('raw', '')

    # Try JSON-based comparison first
    parsed1 = extract_json_robust(raw1)
    parsed2 = extract_json_robust(raw2)

    if parsed1 and parsed2:
        # Compare event signatures
        sig1 = get_event_signature(parsed1[0])
        sig2 = get_event_signature(parsed2[0])

        # Match if labels and teams are the same
        if sig1[0] and sig2[0]:  # Both have labels
            same_label = sig1[0] == sig2[0]
            same_team = sig1[1] == sig2[1] if (sig1[1] and sig2[1]) else True

            if debug and same_label and same_team:
                print(f"      Matched: {sig1[0]} ({sig1[1]}) ~= {sig2[0]} ({sig2[1]})")

            return same_label and same_team

    # Fallback: Compare raw text similarity
    similarity = text_similarity(raw1, raw2)

    if debug and similarity > 0.8:
        print(f"      Text similarity: {similarity:.2f}")
        print(f"        Event 1: {raw1[:80]}...")
        print(f"        Event 2: {raw2[:80]}...")

    return similarity > 0.8  # 80% similar = duplicate

# ============================================================================
# MAIN DEDUPLICATION
# ============================================================================

print(f"🧹 Deduplicating: {args.input}")
print(f"   Time window: {DEDUP_WINDOW}s")
if args.debug:
    print(f"   Debug mode: ON")
print()

# Load input file
try:
    with open(args.input, 'r') as f:
        all_events = json.load(f)
    print(f"📂 Loaded {len(all_events)} events")
except Exception as e:
    print(f"❌ Error loading file: {e}")
    exit(1)

# Sort by time (should already be sorted, but ensure it)
all_events.sort(key=lambda x: x['time'])

# Show sample raw outputs if requested
if args.show_raw and all_events:
    print(f"\n🔍 Sample raw outputs (first 3 events):")
    for i, event in enumerate(all_events[:3]):
        print(f"\n   Event {i+1} at {event['time']:.1f}s:")
        print(f"   Raw: {event.get('raw', 'N/A')[:200]}...")
        parsed = extract_json_robust(event.get('raw', ''))
        if parsed:
            print(f"   Parsed: {parsed[0]}")
        else:
            print(f"   Parsed: FAILED")
    print()

# Deduplicate
final_events = []
duplicates_removed = 0
parsing_success = 0
parsing_failed = 0

for i, current in enumerate(all_events):
    is_duplicate = False

    # Check if current can be parsed
    parsed_current = extract_json_robust(current.get('raw', ''))
    if parsed_current:
        parsing_success += 1
    else:
        parsing_failed += 1

    # Check against recent events in the time window
    for recent in reversed(final_events[-20:]):  # Check last 20 for thoroughness
        time_diff = abs(current['time'] - recent['time'])

        if time_diff > DEDUP_WINDOW:
            break  # Events are sorted, stop checking

        try:
            if events_are_duplicate(current, recent, time_diff, debug=args.debug):
                is_duplicate = True
                duplicates_removed += 1

                if args.debug:
                    print(f"   🗑️ Duplicate at {current['time']:.1f}s (within {time_diff:.1f}s)")
                    print(f"      Current: {current.get('raw', '')[:80]}...")
                    print(f"      Recent:  {recent.get('raw', '')[:80]}...")

                break
        except Exception as e:
            if args.debug:
                print(f"   ⚠️ Comparison error: {e}")
            continue

    if not is_duplicate:
        final_events.append(current)

# Report results
print(f"\n✅ Deduplication complete:")
print(f"   Input:     {len(all_events)} events")
print(f"   Output:    {len(final_events)} events")
print(f"   Removed:   {duplicates_removed} duplicates ({100*duplicates_removed/len(all_events):.1f}%)")
print(f"   Parsed:    {parsing_success}/{len(all_events)} ({100*parsing_success/len(all_events):.1f}%)")
if parsing_failed > 0:
    print(f"   ⚠️ Failed: {parsing_failed} events (used text similarity fallback)")

# Save output
if args.inplace:
    output_file = args.input
    print(f"\n💾 Overwriting: {output_file}")
elif args.output:
    output_file = args.output
    print(f"\n💾 Saving to: {output_file}")
else:
    input_path = Path(args.input)
    output_file = str(input_path.parent / f"{input_path.stem}_deduped{input_path.suffix}")
    print(f"\n💾 Saving to: {output_file}")

try:
    with open(output_file, 'w') as f:
        json.dump(final_events, f, indent=2)
    print(f"✅ Saved successfully!")
except Exception as e:
    print(f"❌ Error saving: {e}")
    exit(1)

# Show sample of final events
if args.debug and final_events:
    print(f"\n📊 Sample of deduplicated events (first 5):")
    for event in final_events[:5]:
        parsed = extract_json_robust(event.get('raw', ''))
        if parsed:
            sig = get_event_signature(parsed[0])
            print(f"   {event['time']:.1f}s: {sig[0]} ({sig[1]})")
        else:
            print(f"   {event['time']:.1f}s: [unparseable]")
