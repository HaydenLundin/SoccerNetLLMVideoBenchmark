# Idefics2-8B Pipeline

## Overview

This pipeline processes soccer videos using **Idefics2-8B**, HuggingFace's multi-image specialist model. It runs in parallel across 4 GPUs with settings matched to Qwen for fair comparison.

### Model Details

- **Model**: `HuggingFaceM4/idefics2-8b`
- **Size**: 8B parameters
- **Quantization**: 4-bit (BitsAndBytes NF4)
- **VRAM per GPU**: ~11-12GB (same as Qwen)
- **Architecture**: Optimized for multi-image understanding
- **Perfect for**: Analyzing multiple frames efficiently

### Settings (Matched to Qwen for Fair Comparison)

- **5 FPS** - Same as Qwen
- **3 second window** - 15 frames per inference
- **2 second step** - 50% overlap
- **Same quantization** - 4-bit NF4
- **Same GPU splitting** - 4 workers in parallel

### Why Idefics2-8B?

1. **Built for multi-image tasks** - Designed by HuggingFace specifically for this use case
2. **Efficient attention** - Handles 15+ images without exploding memory
3. **Proven in production** - Used in real-world multi-image applications
4. **Fair comparison** - Same settings as Qwen for apple-to-apple results
5. **Better than LLaVA** - More memory-efficient multi-image handling

**Note**: Idefics2 succeeded where LLaVA-v1.6-Mistral failed. LLaVA's multi-image attention OOM'd on every window with 15 frames, but Idefics2 is specifically optimized for this scenario.

---

## Quick Start

### 1. Copy Scripts to Project Directory

```bash
# Run the setup script to copy files
bash setup_idefics2.sh

# Or manually:
cp idefics2_split_worker.py ~/soccer_project/
cp merge_results_idefics2.py ~/soccer_project/
```

### 2. Set HuggingFace Token

```bash
export HF_TOKEN='hf_YourActualTokenHere'
```

### 3. Run Processing

```bash
# Process all 40 videos
sbatch submit_idefics2.sh 0 39

# Process first 10 videos (for testing)
sbatch submit_idefics2.sh 0 9

# Process specific range
sbatch submit_idefics2.sh 20 29
```

---

## Files in This Pipeline

### Core Scripts

1. **`idefics2_split_worker.py`** - GPU worker script
   - Loads Idefics2-8B with 4-bit quantization
   - Processes assigned time slice of video
   - Outputs partial results to JSON

2. **`submit_idefics2.sh`** - SLURM batch submission
   - Launches 4-GPU parallel processing
   - Handles 0-39 videos sequentially
   - Auto-skips completed videos
   - Merges results after each video

3. **`merge_results_idefics2.py`** - Result aggregation
   - Combines 4 partial results into one
   - Deduplicates events within 3-second window
   - Saves final predictions

4. **`cleanup_idefics2.py`** - Cleanup utility
   - Safe cleanup of incomplete videos
   - Dry-run mode to preview changes

5. **`setup_idefics2.sh`** - Setup script
   - Copies files to `~/soccer_project/`

---

## How It Works

### Processing Flow

```
Video 0 (50 min match)
├── GPU 0: 0-750s    (12.5 min)  ┐
├── GPU 1: 750-1500s (12.5 min)  ├─ Process in parallel
├── GPU 2: 1500-2250s (12.5 min) │
└── GPU 3: 2250-3000s (12.5 min) ┘
         ↓
    Wait for all GPUs
         ↓
    Merge 4 partial results
         ↓
    Deduplicate overlapping events
         ↓
    Save final_predictions_idefics2_video_0.json
         ↓
    Clean up partial files
         ↓
    Move to Video 1
```

### Temporal Processing

Each GPU processes its time slice with:
- **3-second windows** at 5 FPS (15 frames per window - same as Qwen)
- **2-second step** (50% overlap to catch events at boundaries)
- **Efficient multi-image attention** that scales linearly with frame count
- **Processor auto-handles** image resolution

Example for GPU 0 (0-750s):
```
Window 1: 0-3s    (15 frames at 5 FPS)
Window 2: 2-5s    (15 frames, overlaps 1s with Window 1)
Window 3: 4-7s    (15 frames, overlaps 1s with Window 2)
...
Window 375: 748-751s (partial window)
```

---

## Output Format

### Partial Results
```json
[
  {
    "time": 245.0,
    "raw": "{'label': 'Goal', 'team': 'away', 'confidence': 0.95, 'details': 'Player scores from close range'}"
  },
  {
    "time": 247.0,
    "raw": "{'label': 'Goal', 'team': 'away', 'confidence': 0.93, 'details': 'Goal celebration visible'}"
  }
]
```

### Final Merged Results (after deduplication)
```json
[
  {
    "time": 245.0,
    "raw": "{'label': 'Goal', 'team': 'away', 'confidence': 0.95, 'details': 'Player scores from close range'}"
  }
]
```
*(Duplicate at 247.0s removed - same event within 3-second window)*

---

## Monitoring Progress

### Check Job Status
```bash
# View live processing log
tail -f idefics2_*.log

# Count completed videos
ls ~/soccer_project/final_predictions_idefics2_video_*.json | wc -l

# Check specific video event count
python -c "import json; print(len(json.load(open('~/soccer_project/final_predictions_idefics2_video_0.json'))))"
```

### Expected Timeline

- **Per video**: ~4-6 minutes (varies by event density)
- **10 videos**: ~50 minutes
- **40 videos**: ~3-4 hours

With 6-hour SLURM limit, you have comfortable buffer for all 40 videos.

---

## Comparing with Qwen Results

After processing, you can compare Idefics2 and Qwen detections:

```bash
# Count events from both models
for i in {0..39}; do
    qwen_count=$(python -c "import json; print(len(json.load(open('~/soccer_project/final_predictions_qwen7b_video_${i}.json'))))" 2>/dev/null || echo "0")
    idefics2_count=$(python -c "import json; print(len(json.load(open('~/soccer_project/final_predictions_idefics2_video_${i}.json'))))" 2>/dev/null || echo "0")
    echo "Video $i: Qwen=$qwen_count, Idefics2=$idefics2_count"
done
```

### Ensemble Analysis

Use both models to:
1. **High confidence events** - Detected by both models
2. **Model-specific strengths** - Events only one model catches
3. **Disagreement analysis** - Where models differ, requires closer inspection

---

## Troubleshooting

### Model Download Issues

If the model fails to download:
```bash
# Pre-download the model
python -c "from transformers import Idefics2ForConditionalGeneration; Idefics2ForConditionalGeneration.from_pretrained('HuggingFaceM4/idefics2-8b')"
```

### VRAM Issues

Idefics2 is optimized for multi-image tasks and should fit with 15 frames.

If you get OOM errors:

1. **Check GPU memory**: `nvidia-smi`
2. **Check logs** for the exact error message
3. **Report the issue** - Idefics2 is supposed to handle this workload

Unlike LLaVA, Idefics2's architecture is designed for this use case. If it OOMs, something else is wrong.

### Empty Results

If a video produces 0 events:
```bash
# Check partial files
ls ~/soccer_project/partial_results_idefics2_vid0_gpu*.json

# View sample of what model generated
python -c "import json; d=json.load(open('~/soccer_project/partial_results_idefics2_vid0_gpu0.json')); print(d[:3])"
```

### Reprocess Specific Video

```bash
# Clean up video 5
python cleanup_idefics2.py --video_index 5 --force

# Reprocess just video 5
sbatch submit_idefics2.sh 5 5
```

---

## Cleanup

After processing is complete and you've verified results:

```bash
# Remove all partial files (saves space)
rm ~/soccer_project/partial_results_idefics2_vid*_gpu*.json

# Keep only final predictions
ls ~/soccer_project/final_predictions_idefics2_video_*.json
```

---

## Model Comparison Chart

| Feature | Qwen 2.5-VL 7B | Idefics2-8B | LLaVA-v1.6 7B |
|---------|----------------|-------------|---------------|
| **Size** | 7B | 8B | 7B |
| **Multi-image design** | General purpose | **Specialized** | General purpose |
| **15-frame handling** | ✅ Works | ✅ Works | ❌ OOM |
| **Frames per window** | 15 (5 FPS × 3s) | 15 (5 FPS × 3s) | N/A |
| **Memory usage** | ~11GB (4-bit) | ~11-12GB (4-bit) | ~20GB+ (OOM) |
| **Architecture** | Frame-independent | **Multi-image optimized** | Multi-image |
| **Base LLM** | Qwen 2.5 | Mistral 7B | Mistral 7B |
| **Best for** | General vision + text | **Multi-image analysis** | Single images |

Idefics2 is the first model besides Qwen that successfully handles 15 frames on 24GB GPUs!

---

## Next Steps

After running Idefics2 pipeline:
1. Compare results with Qwen
2. Analyze which events each model is better at detecting
3. Build ensemble predictions combining both models
4. Benchmark against SoccerNet ground truth
5. Calculate precision/recall metrics per event type

Happy processing! 🎬⚽
