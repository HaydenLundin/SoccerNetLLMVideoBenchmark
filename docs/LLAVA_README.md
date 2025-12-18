# LLaVA Pipeline (v1.6-Mistral-7B)

## Overview

This pipeline processes soccer videos using **LLaVA-v1.6-Mistral-7B**, a powerful vision-language model that processes frames independently. It runs in parallel across 4 GPUs for efficient processing.

### Model Details

- **Model**: `llava-hf/llava-v1.6-mistral-7b-hf`
- **Size**: 7B parameters
- **Quantization**: 4-bit (BitsAndBytes NF4)
- **VRAM per GPU**: ~11GB (same as Qwen)
- **Architecture**: Processes frames independently (like Qwen)
- **Perfect for**: Multi-frame event detection with consistent memory usage

### Settings (Matched to Qwen for Fair Comparison)

- **5 FPS** - Same as Qwen
- **3 second window** - 15 frames per inference
- **2 second step** - 50% overlap
- **Same quantization** - 4-bit NF4
- **Same resolution** - Processor handles image sizing

### Why LLaVA-v1.6-Mistral?

1. **Frame-independent processing** - Like Qwen, processes each frame separately
2. **Proven memory footprint** - Fits in ~11GB with 15 frames
3. **Multi-image support** - Can analyze up to 15 frames in one prompt
4. **Strong visual reasoning** - Excellent at detecting nuanced soccer events
5. **Fair comparison** - Same settings as Qwen for apple-to-apple results

**Note**: We switched from LLaVA-NeXT-Video (which has cross-frame attention) to LLaVA-v1.6 (frame-independent) because the Video variant required 284GB VRAM with 15 frames due to its temporal attention mechanism.

---

## Quick Start

### 1. Copy Scripts to Project Directory

```bash
# Run the setup script to copy files
bash setup_llava.sh

# Or manually:
cp llava_split_worker.py ~/soccer_project/
cp merge_results_llava.py ~/soccer_project/
```

### 2. Set HuggingFace Token

```bash
export HF_TOKEN='hf_YourActualTokenHere'
```

### 3. Run Processing

```bash
# Process all 40 videos
sbatch submit_llava.sh 0 39

# Process first 10 videos (for testing)
sbatch submit_llava.sh 0 9

# Process specific range
sbatch submit_llava.sh 20 29
```

---

## Files in This Pipeline

### Core Scripts

1. **`llava_split_worker.py`** - GPU worker script
   - Loads LLaVA-NeXT-Video-7B with 4-bit quantization
   - Processes assigned time slice of video
   - Outputs partial results to JSON

2. **`submit_llava.sh`** - SLURM batch submission
   - Launches 4-GPU parallel processing
   - Handles 0-39 videos sequentially
   - Auto-skips completed videos
   - Merges results after each video

3. **`merge_results_llava.py`** - Result aggregation
   - Combines 4 partial results into one
   - Deduplicates events within 3-second window
   - Saves final predictions

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
    Save final_predictions_llava_video_0.json
         ↓
    Clean up partial files
         ↓
    Move to Video 1
```

### Temporal Processing

Each GPU processes its time slice with:
- **3-second windows** at 5 FPS (15 frames per window - same as Qwen)
- **2-second step** (50% overlap to catch events at boundaries)
- **Frames processed independently** but analyzed together in one prompt
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
tail -f llava_*.log

# Count completed videos
ls ~/soccer_project/final_predictions_llava_video_*.json | wc -l

# Check specific video event count
python -c "import json; print(len(json.load(open('~/soccer_project/final_predictions_llava_video_0.json'))))"
```

### Expected Timeline

- **Per video**: ~4-6 minutes (varies by event density)
- **10 videos**: ~50 minutes
- **40 videos**: ~3-4 hours

With 6-hour SLURM limit, you have comfortable buffer for all 40 videos.

---

## Comparing with Qwen Results

After processing, you can compare LLaVA and Qwen detections:

```bash
# Count events from both models
for i in {0..39}; do
    qwen_count=$(python -c "import json; print(len(json.load(open('~/soccer_project/final_predictions_qwen7b_video_${i}.json'))))" 2>/dev/null || echo "0")
    llava_count=$(python -c "import json; print(len(json.load(open('~/soccer_project/final_predictions_llava_video_${i}.json'))))" 2>/dev/null || echo "0")
    echo "Video $i: Qwen=$qwen_count, LLaVA=$llava_count"
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
python -c "from transformers import LlavaNextForConditionalGeneration; LlavaNextForConditionalGeneration.from_pretrained('llava-hf/llava-v1.6-mistral-7b-hf')"
```

### VRAM Issues

The script uses the same settings as Qwen (15 frames, 5 FPS) and should fit comfortably in 24GB.

If you get OOM errors:

1. **Check GPU memory**: `nvidia-smi`
2. **The script auto-skips OOM windows** - Check log for `⚠️ OOM at` messages
3. **If persistent**, it may indicate a different issue - check the error logs

The model processes frames independently (not with cross-frame attention), so memory usage is predictable and similar to Qwen.

### Empty Results

If a video produces 0 events:
```bash
# Check partial files
ls ~/soccer_project/partial_results_llava_vid0_gpu*.json

# View sample of what model generated
python -c "import json; d=json.load(open('~/soccer_project/partial_results_llava_vid0_gpu0.json')); print(d[:3])"
```

### Reprocess Specific Video

```bash
# Clean up video 5
rm ~/soccer_project/partial_results_llava_vid5_gpu*.json
rm ~/soccer_project/final_predictions_llava_video_5.json

# Reprocess just video 5
sbatch submit_llava.sh 5 5
```

---

## Cleanup

After processing is complete and you've verified results:

```bash
# Remove all partial files (saves space)
rm ~/soccer_project/partial_results_llava_vid*_gpu*.json

# Keep only final predictions
ls ~/soccer_project/final_predictions_llava_video_*.json
```

---

## Model Comparison Chart

| Feature | Qwen 2.5-VL 7B | LLaVA-v1.6-Mistral 7B |
|---------|----------------|----------------------|
| **Architecture** | Frame-independent | Frame-independent |
| **Training** | Images + some video | Images |
| **Frames per window** | 15 (5 FPS × 3s) | 15 (5 FPS × 3s) |
| **Memory usage** | ~11GB (4-bit) | ~11GB (4-bit) |
| **Speed** | Fast | Fast |
| **Base LLM** | Qwen 2.5 | Mistral 7B |
| **Best for** | General vision + text | Multi-image analysis |

Both models use the same settings for fair comparison. Combining results enables ensemble predictions and confidence scoring!

---

## Advanced: Adjust Detection Sensitivity

Edit `llava_split_worker.py` to tune event detection:

```python
# Line 158 - Adjust sampling for more/less conservative predictions
output_ids = model.generate(
    **inputs,
    max_new_tokens=256,
    do_sample=True,        # Enable sampling
    temperature=0.7,       # Lower = more conservative (try 0.5-1.0)
    top_p=0.9             # Nucleus sampling
)
```

---

## Next Steps

After running LLaVA pipeline:
1. Compare results with Qwen
2. Analyze which events each model is better at detecting
3. Build ensemble predictions combining both models
4. Benchmark against SoccerNet ground truth
5. Calculate precision/recall metrics per event type

Happy processing! 🎬⚽
