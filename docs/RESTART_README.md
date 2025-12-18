# Restart After Timeout - Quick Guide

Your job timed out on **video 17**. Here's how to safely clean up and restart.

## 🚨 What Happened

Your SLURM job hit the 1-hour time limit while processing video 17:
```
slurmstepd: error: *** JOB 12575 CANCELLED AT 2025-12-06T23:38:12 DUE TO TIME LIMIT ***
```

Videos 0-16 should be complete, but video 17 may have partial results.

---

## ✅ Safe Restart Process (3 Steps)

### Step 1: Check Video 17 Status

```bash
# See what files exist for video 17
ls -lh ~/soccer_project/*vid17* ~/soccer_project/*video_17*

# Or use the cleanup script to inspect
python cleanup_incomplete.py --video_index 17 --dry-run
```

**What to look for:**
- ✅ If `final_predictions_qwen7b_video_17.json` exists → video 17 completed
- ⚠️ If only `partial_results_qwen7b_vid17_gpu*.json` exist → video 17 incomplete

---

### Step 2: Clean Up Video 17 (if needed)

**If video 17 was incomplete:**
```bash
# Preview what will be deleted
python cleanup_incomplete.py --video_index 17 --dry-run

# Clean up (safe - won't delete if merge exists)
python cleanup_incomplete.py --video_index 17
```

**If video 17 completed but you want to reprocess:**
```bash
# Force cleanup (deletes everything including merged file)
python cleanup_incomplete.py --video_index 17 --force
```

---

### Step 3: Restart Processing

**Option A: Restart from video 17 to end (recommended)**
```bash
sbatch submit_restart.sh 17 39
```

**Option B: Process only a few videos for testing**
```bash
sbatch submit_restart.sh 17 20   # Just videos 17-20
```

**Option C: Process just video 17**
```bash
sbatch submit_restart.sh 17 17
```

---

## 📊 What the Restart Script Does

1. **Skips completed videos** - Checks if merged file exists, skips if yes
2. **Cleans partials automatically** - Removes any leftover partial files
3. **Processes remaining videos** - Runs 4-GPU swarm for each video
4. **Longer timeout** - 6 hours instead of 1 hour (more buffer)

---

## 🔍 Monitoring Progress

```bash
# Watch the log file
tail -f swarm_restart_*.log

# Check which videos completed
ls -lh ~/soccer_project/final_predictions_qwen7b_video_*.json | wc -l
# Should show 40 when all complete (0-39)

# See event counts for completed videos
for i in {0..39}; do
    FILE=~/soccer_project/final_predictions_qwen7b_video_${i}.json
    if [ -f "$FILE" ]; then
        COUNT=$(python -c "import json; print(len(json.load(open('$FILE'))))")
        echo "Video $i: $COUNT events"
    fi
done
```

---

## 🛠️ Troubleshooting

### If the restart script also times out

Videos take different amounts of time. If you keep timing out:

```bash
# Process in smaller batches
sbatch submit_restart.sh 17 25   # First batch
sbatch submit_restart.sh 26 32   # Second batch
sbatch submit_restart.sh 33 39   # Third batch
```

Or increase the time limit in `submit_restart.sh`:
```bash
#SBATCH --time=12:00:00   # Change from 06:00:00 to 12:00:00
```

### If a specific video keeps failing

```bash
# Check the worker logs
grep "Video 17" swarm_restart_*.log

# Try processing just that video
sbatch submit_restart.sh 17 17

# If it fails, check the partial results
python -c "import json; print(json.load(open('~/soccer_project/partial_results_qwen7b_vid17_gpu0.json'))[:3])"
```

### If you want to start completely fresh

```bash
# Nuclear option: Delete ALL results and start from video 0
rm ~/soccer_project/partial_results_qwen7b_*.json
rm ~/soccer_project/final_predictions_qwen7b_video_*.json

# Then restart from beginning
sbatch submit_restart.sh 0 39
```

---

## 📝 Files Created

- `cleanup_incomplete.py` - Safe cleanup script for incomplete videos
- `submit_restart.sh` - Restart SLURM job from any video index
- `submit_split.sh` - Original submission script (use for fresh runs)

---

## 🎯 TL;DR - Quick Commands

```bash
# 1. Check status
python cleanup_incomplete.py --video_index 17 --dry-run

# 2. Clean up
python cleanup_incomplete.py --video_index 17

# 3. Restart (will auto-skip completed videos)
sbatch submit_restart.sh 17 39

# 4. Monitor
tail -f swarm_restart_*.log
```

---

## ⏱️ Time Estimates

Based on your timeout:
- **Per video**: ~3-5 minutes (varies by content)
- **1 hour limit**: Processes ~12-20 videos
- **6 hour limit**: Should handle all 40 videos comfortably

If you need to process more than 40 videos or videos are longer, increase `--time` in the script.
