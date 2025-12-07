# Deduplication Scripts

## Overview

This directory contains two scripts for handling duplicate event detection:

1. **`merge_results.py`** - Improved merge script (merges 4 GPU outputs + deduplicates)
2. **`cleanup_duplicates.py`** - Standalone cleanup script for already-merged files
3. **`merge_results_original.py`** - Original merge script (backup)

## What Changed

### The Problem
The original `merge_results.py` had a **silent failure** in deduplication:
- Line 53-54: `except: pass` silently swallowed all JSON parsing errors
- Only compared against the **last** event, not all events in the time window
- Couldn't handle LLM output formats with single quotes or malformed JSON

Result: **0% deduplication** even though duplicates existed

### The Fix
Both new scripts now:
- ✅ Parse multiple JSON formats (single/double quotes, embedded text)
- ✅ Compare against ALL recent events within the time window
- ✅ Show detailed stats on duplicates removed
- ✅ Handle parsing failures gracefully (with warnings)
- ✅ Support debug mode to see what's happening

---

## Usage

### For New Runs (Merge + Deduplicate)

Use the improved `merge_results.py` as before:

```bash
# Standard usage (same as before)
python merge_results.py --video_index 0

# With debug output to see what's being filtered
python merge_results.py --video_index 0 --debug
```

**Output:**
```
🔍 Merging results for Video Index 0...
📂 Loading partial files...
   Loaded 45 events from partial_results_qwen7b_vid0_gpu0.json
   Loaded 38 events from partial_results_qwen7b_vid0_gpu1.json
   Loaded 42 events from partial_results_qwen7b_vid0_gpu2.json
   Loaded 51 events from partial_results_qwen7b_vid0_gpu3.json
🔄 Sorting 176 total detections...
🧹 Deduplicating events (window: 3.0s)...
✅ Merged 176 -> 89 unique events
   Removed 87 duplicates
💾 Saved to final_predictions_qwen7b_video_0.json
```

---

### For Already-Merged Files (Cleanup Only)

Use `cleanup_duplicates.py` to deduplicate existing final prediction files:

```bash
# Clean an existing file (creates new file with _cleaned suffix)
python cleanup_duplicates.py --input final_predictions_qwen7b_video_0.json

# Clean with debug output
python cleanup_duplicates.py --input final_predictions_qwen7b_video_0.json --debug

# Overwrite the original file
python cleanup_duplicates.py --input final_predictions_qwen7b_video_0.json --inplace

# Specify custom output file
python cleanup_duplicates.py --input final_predictions_qwen7b_video_0.json --output cleaned_v0.json

# Adjust deduplication window (default: 3 seconds)
python cleanup_duplicates.py --input final_predictions_qwen7b_video_0.json --window 5.0
```

**Example Output:**
```
🧹 Cleaning up duplicates from: final_predictions_qwen7b_video_0.json
   Time window: 3.0s
📂 Loaded 176 events
✅ Cleaned 176 -> 89 unique events
   Removed 87 duplicates (49.4%)
💾 Saved to final_predictions_qwen7b_video_0_cleaned.json
```

---

## Batch Cleanup

To clean all already-merged files at once:

```bash
# Clean all final prediction files
for i in {0..39}; do
    if [ -f "final_predictions_qwen7b_video_${i}.json" ]; then
        echo "Cleaning video $i..."
        python cleanup_duplicates.py --input "final_predictions_qwen7b_video_${i}.json" --inplace
    fi
done
```

---

## How Deduplication Works

1. **Sort events by timestamp** (earliest to latest)
2. **For each event**:
   - Compare against the last 10 events in the output list
   - If same event label appears within 3 seconds → **skip as duplicate**
   - Otherwise → **keep**
3. **Label matching**:
   - Extracts label from LLM JSON output
   - Case-insensitive comparison
   - Handles single quotes, double quotes, and text around JSON

**Example:**
```
Time    Event          Action
----    -----          ------
10.0s   Foul (home)    ✅ Keep (first)
12.0s   Foul (home)    🗑️ Skip (within 3s of 10.0s)
14.5s   Foul (home)    🗑️ Skip (within 3s of 12.0s and 10.0s)
20.0s   Foul (away)    ✅ Keep (different label: "away" vs "home")
25.0s   Corner (home)  ✅ Keep (different event type)
```

---

## Troubleshooting

### No duplicates removed?

Run with `--debug` to see what's happening:
```bash
python cleanup_duplicates.py --input your_file.json --debug
```

This will show:
- Which events are being filtered as duplicates
- Any JSON parsing failures
- Sample of cleaned events

### Want to see raw LLM outputs?

Check the `raw` field in your JSON files:
```bash
python -c "import json; data=json.load(open('final_predictions_qwen7b_video_0.json')); print(data[0]['raw'][:200])"
```

### Adjust the time window

If you're getting too much/too little deduplication:
```bash
# More aggressive (5 second window)
python cleanup_duplicates.py --input file.json --window 5.0

# Less aggressive (1 second window)
python cleanup_duplicates.py --input file.json --window 1.0
```

---

## Files

- `merge_results.py` - **Use this for new runs**
- `cleanup_duplicates.py` - **Use this for already-merged files**
- `merge_results_original.py` - Original script (kept for reference)
