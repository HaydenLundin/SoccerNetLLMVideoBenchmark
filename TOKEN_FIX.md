# Quick Fix for "Invalid user token" Error

## The Problem

Your jobs are failing with:
```
huggingface_hub.errors.HfHubHTTPError: Invalid user token.
The token from HF_TOKEN environment variable is invalid.
```

The scripts have a placeholder `HF_TOKEN="YOUR_TOKEN"` that needs your actual HuggingFace token.

---

## Fix Option 1: Set Environment Variable (Recommended)

Before submitting jobs, set your HF token in your shell:

```bash
export HF_TOKEN='hf_YourActualTokenHere'
sbatch submit_restart.sh 18 39
```

This way you don't have to edit the scripts.

---

## Fix Option 2: Edit the Script

Edit the script and replace the placeholder:

```bash
nano submit_restart.sh
```

Change line ~28 from:
```bash
export HF_TOKEN="YOUR_TOKEN"
```

To:
```bash
export HF_TOKEN="hf_YourActualTokenHere"
```

Then save and run:
```bash
sbatch submit_restart.sh 18 39
```

---

## How to Get Your HuggingFace Token

1. Go to https://huggingface.co/settings/tokens
2. Create a new token (or copy existing one)
3. Use it in one of the methods above

---

## Current Status Check

You have 18 videos complete (0-17). Video 18 needs reprocessing:

```bash
# Check video 18 status
ls -lh ~/soccer_project/*vid18* ~/soccer_project/*video_18*

# Clean up video 18 (it has empty/bad merge)
python cleanup_incomplete.py --video_index 18 --force

# After setting HF_TOKEN, restart from video 18
export HF_TOKEN='your_token_here'
sbatch submit_restart.sh 18 39
```

---

## Other Fixes in This Update

1. **Merge script now saves to correct directory** - Was saving to current dir, now saves to `~/soccer_project/`
2. **Better error handling** - One video failure won't kill entire batch
3. **Empty merge detection** - Warns if merge creates 0-event file
4. **Token validation** - Script warns if token is placeholder

---

## Verification

After restarting with proper token:

```bash
# Monitor progress
tail -f swarm_restart_*.log

# Count completed videos
ls ~/soccer_project/final_predictions_qwen7b_video_*.json | wc -l
# Should reach 40 when all done

# Check for empty merges
for i in {0..39}; do
    f=~/soccer_project/final_predictions_qwen7b_video_${i}.json
    if [ -f "$f" ]; then
        count=$(python -c "import json; print(len(json.load(open('$f'))))")
        if [ "$count" -eq 0 ]; then
            echo "⚠️ Video $i: EMPTY (needs reprocessing)"
        fi
    fi
done
```
