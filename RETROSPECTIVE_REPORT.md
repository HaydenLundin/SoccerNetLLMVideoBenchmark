# SoccerNet LLM Video Benchmark - Technical Retrospective

**Project:** Benchmarking Open-Source Vision-Language Models for Soccer Event Detection
**Dataset:** SoccerNet English Premier League (EPL) Videos
**Date:** December 2025

---

## Executive Summary

This project evaluated multiple open-source Vision-Language Models (VLMs) on their ability to detect and annotate soccer events from professional match footage. We processed 40+ EPL match videos (~50 minutes each) using 10+ different VLMs, implementing a parallel HPC pipeline with 4-way GPU splitting for efficient batch processing.

---

## 1. Challenges & Technical Hurdles

### Implementation Issues

| Issue | Description | Resolution |
|-------|-------------|------------|
| **OOM with LLaVA-NeXT-Video** | Cross-frame attention mechanism required 284GB VRAM with 15 frames | Switched to LLaVA-v1.6-Mistral (frame-independent processing) |
| **SLURM Timeout** | 1-hour time limit insufficient; job terminated mid-video (e.g., video 17) | Extended to 6-hour limit; created `submit_restart.sh` for resumable processing |
| **HuggingFace Token Invalid** | Placeholder `HF_TOKEN="YOUR_TOKEN"` caused auth failures | Created `TOKEN_FIX.md` with environment variable setup |
| **Silent Deduplication Failure** | Original `merge_results.py` had `except: pass` swallowing JSON errors | Rewrote parser with multi-strategy fallback (see Deduplication section) |
| **Empty Merge Files** | Merge script saved to wrong directory (current dir vs `~/soccer_project/`) | Fixed path handling in `merge_results.py` |
| **Phi-3.5 Flash Attention Failure** | Flash attention incompatible with older Titan RTX GPUs | Set `attn_implementation="eager"` in model config |

### Data Handling

- **Video Format Inconsistency:** SoccerNet provides MKV files at various resolutions; standardized to 720p
- **JSON Output Variability:** LLMs produced inconsistent output formats:
  - Single quotes vs double quotes: `{'label': 'Goal'}` vs `{"label": "Goal"}`
  - Markdown wrappers: ` ```json {...} ``` `
  - Embedded text around JSON objects
- **Frame Extraction Quality:** Initial JPEG quality setting was too low; changed to `-q:v 2` (highest quality)

### Resolution: Multi-Strategy JSON Parsing

```python
# From deduplicate_qwen.py - handles varied LLM output formats
def parse_event_data(raw):
    # Strategy 1: Direct JSON parse
    # Strategy 2: Regex extract {...} patterns
    # Strategy 3: Single quotes → double quotes
    # Strategy 4: ast.literal_eval for Python dict syntax
    # Strategy 5: Similarity matching for near-duplicates
```

**Key Takeaway:** "The original merge script had a silent failure in deduplication... Line 53-54: `except: pass` silently swallowed all JSON parsing errors... Result: 0% deduplication even though duplicates existed."

---

## 2. Educational Insights

### Why Models Hallucinate in This Context

The models exhibited several hallucination patterns specific to soccer video analysis:

1. **Player Identification Without Visual Evidence:** Models confidently reported specific player numbers or names that weren't visible in the frames
2. **Event Invention from Static Frames:** With only 3-second windows, models sometimes "invented" events to fill gaps (e.g., reporting a "goal" when only showing a player running)
3. **Team Attribution Errors:** Difficulty distinguishing home vs away teams, especially with similar jersey colors
4. **Temporal Confusion:** Events from overlapping windows sometimes reported as distinct events

**Root Cause:** VLMs are trained on captioned images/videos but lack true temporal understanding - they process frames independently, not as a continuous video stream.

### LLM Mechanics: Multimodal Processing

Key insights on how these models handle video:

- **Frame-Independent Processing:** All production models (Qwen, LLaVA-v1.6, IDEFICS2) process each frame separately, not recurrently
- **No True Temporal Attention:** Despite receiving 15 frames, models don't learn cross-frame relationships like action-reaction
- **Token Budget Constraints:** 70 frames at 720p = ~110K tokens; approaching context limits for most models
- **Quantization Trade-offs:** 4-bit NF4 quantization reduces VRAM from 24GB+ to ~11GB, but may impact fine-grained visual reasoning

### Working with Video: Frame Sampling Strategy

The project settled on these parameters after experimentation:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| FPS | 5.0 | Captures fast action without token explosion |
| Window Size | 3 seconds | Sufficient context for most soccer events |
| Step Size | 2 seconds | 50% overlap catches events at boundaries |
| Frames/Window | 15 | Maximum before OOM on 24GB GPUs |
| Target Duration | 50 min | One half of match |

**Key Takeaway:** "We switched from LLaVA-NeXT-Video (which has cross-frame attention) to LLaVA-v1.6 (frame-independent) because the Video variant required 284GB VRAM with 15 frames due to its temporal attention mechanism."

---

## 3. LLM Selection & Methodology

### Model Lineup

#### Production Models (HPC Pipeline)

| Model | Parameters | Quantization | VRAM | Notes |
|-------|------------|--------------|------|-------|
| **Qwen 2.5-VL 7B** | 7B | 4-bit NF4 | ~11GB | Primary benchmark model |
| **LLaVA-v1.6-Mistral 7B** | 7B | 4-bit NF4 | ~11GB | Frame-independent fallback |
| **IDEFICS2-8B** | 8B | 4-bit NF4 | ~11-12GB | Multi-image specialist |
| **Phi-3.5-Vision** | 4.2B | 4-bit NF4 | ~10GB | Smallest, uses eager attention |
| **MiniCPM-V 2.6** | 8B | INT4 native | ~7GB | Custom `.chat()` API |

#### Test/Prototype Models (Colab)

- **Llama 3.2 11B Vision** - Meta's latest multimodal
- **Pixtral 12B** - Mistral's vision model
- **Phi-4 Multimodal** - Microsoft's latest
- **Qwen 2.5-VL 32B AWQ** - Larger Qwen variant

#### API Models (OpenRouter)

- NVIDIA Nemotron Nano 12B 2 VL (128K context)
- Meta Llama 4 Maverick 128B (128K context)
- Google Gemini 2.0 Flash Experimental (1M context)
- Mistral Small 3.2 24B (131K context)
- Google Gemma 3 27B (131K context)
- Meta Llama 4 Scout 17B (128K context)

### Selection Rationale

1. **VRAM Constraints:** 24GB Titan RTX = must fit in ~11GB with 4-bit quantization
2. **Multi-Image Support:** Must handle 15 frames in single inference
3. **Fair Comparison:** Matched settings across models (FPS, window, quantization)
4. **Open-Source Priority:** Focus on reproducible benchmarks
5. **Context Window:** Need 100K+ tokens for API models with 70 frames

**Key Takeaway:** "IDEFICS2 is the first model besides Qwen that successfully handles 15 frames on 24GB GPUs!"

### Prompt Engineering

#### Detection Prompt Template

```python
prompt = (
    f"Context: {match_context}\n"
    f"Analyze this {WINDOW_SECONDS}s clip. Detect ANY of these 17 events:\n"
    "- Goals/Shots: Goal, Shot on target, Shot off target\n"
    "- Fouls/Cards: Foul, Yellow card, Red card, Offside\n"
    "- Set Pieces: Corner, Free-kick, Penalty, Throw-in, Kick-off, Goal kick\n"
    "- Other: Substitution, Ball out of play, Clearance\n\n"
    "For EACH event, output a JSON object:\n"
    "{'label': 'EVENT_TYPE', 'team': 'home' OR 'away', "
    "'confidence': 0.0-1.0, 'details': 'DESC'}\n"
    "If nothing significant happens, output exactly: None."
)
```

#### Strategies Used

- **Zero-Shot:** All production models used zero-shot prompting
- **Constrained Output:** Explicitly requested JSON format with specific keys
- **Match Context:** Extracted team names from 15-second frame to ground predictions
- **Deterministic Generation:** `do_sample=False`, no temperature for consistency
- **Repetition Penalty:** Phi-3.5 used `repetition_penalty=1.2` to prevent hallucination loops

---

## 4. Infrastructure & HPC

### Compute Environment

```bash
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:TitanRTX:4
#SBATCH --exclusive
```

**Hardware:** 4x NVIDIA Titan RTX (24GB VRAM each)

### Parallel Processing Strategy

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
    Save final_predictions
```

### Memory Management

- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` for dynamic allocation
- `torch.cuda.empty_cache()` after each window
- Frames deleted immediately after processing
- `CUDA_VISIBLE_DEVICES` isolation per worker

### Performance Constraints

| Metric | Value |
|--------|-------|
| Per-video processing time | 3-5 minutes |
| 40 videos total | ~3-4 hours |
| Inference per window | ~1-2 seconds |
| SLURM time limit | 6 hours (2.7-hour buffer) |
| Storage per video (frames) | Temporary only |

---

## 5. Experimental Results

### Quantitative Results

**Qwen 2.5-VL 7B (21 videos processed):**

| Video | Events Detected |
|-------|-----------------|
| Video 0 | 950 events |
| Video 1 | 933 events |
| Video 10 | 880 events |
| Video 11 | 917 events |
| Video 12 | 929 events |

**Average:** ~900+ events per 50-minute half

**Deduplication Impact:**
- Pre-deduplication: 176 events (example)
- Post-deduplication: 89 unique events
- Reduction: 49.4% duplicate removal

**Phi-3.5 Results:** 12 videos processed, but output files appear empty (2 bytes each) - potential failure mode requiring investigation.

### Qualitative Observations

- **High Sensitivity:** Models detected events at nearly every window (950 events = ~1 per 3 seconds)
- **Over-Detection Likely:** Real soccer matches have ~50-100 significant events per half, not 900+
- **Confidence Scores:** Consistently high (0.8-0.95) regardless of actual event significance
- **Event Type Distribution:** Heavy bias toward common events (Foul, Shot off target, Ball out of play)

### Model Comparison (Preliminary)

| Model | 15-Frame Handling | VRAM | Speed | Notes |
|-------|-------------------|------|-------|-------|
| Qwen 2.5-VL 7B | Works | ~11GB | Fast | Primary model, consistent output |
| LLaVA-v1.6 7B | Works | ~11GB | Fast | Frame-independent processing |
| IDEFICS2-8B | Works | ~11-12GB | Fast | Multi-image specialist |
| LLaVA-NeXT-Video | OOM | 284GB+ | N/A | Cross-frame attention too expensive |
| Phi-3.5 Vision | Works* | ~10GB | Fast | *Empty outputs observed |

---

## 6. Academic & Technical References

### Key Concepts

| Concept | Definition | Application |
|---------|------------|-------------|
| **VLM (Vision-Language Model)** | Models that jointly process images and text | Core evaluation target |
| **Temporal Action Localization** | Identifying when events occur in video | Ground truth from SoccerNet |
| **Frame-Independent Processing** | Each frame processed separately, no cross-frame attention | Used by Qwen, LLaVA-v1.6 |
| **Cross-Frame Attention** | Attention mechanism across video frames | LLaVA-NeXT-Video (too expensive) |
| **4-bit NF4 Quantization** | BitsAndBytes normal-float 4-bit compression | Used to fit 7B models in 11GB VRAM |
| **AWQ (Activation-aware Weight Quantization)** | Quantization preserving important weights | Used for Qwen 32B variant |

### Libraries & Frameworks

- **Transformers** (HuggingFace) - Model loading and inference
- **BitsAndBytes** - 4-bit quantization
- **SoccerNet** - Dataset downloading and ground truth labels
- **FFmpeg** - Frame extraction from video
- **SLURM** - HPC job scheduling

### SoccerNet Dataset Details

- **Source:** Professional EPL broadcast footage
- **Annotations:** `Labels-v2.json` per game with 17 event types
- **Access:** Requires NDA password (`s0cc3rn3t`)
- **Splits:** train, valid, test, challenge (104 EPL games total)

---

## 7. Future Work & Gaps

### What's Missing

1. **Ground Truth Evaluation:** No precision/recall metrics computed against SoccerNet labels
2. **Event Type Analysis:** No per-event-type accuracy breakdown (e.g., "Model X was good at fouls but bad at offsides")
3. **Ensemble Methods:** Multi-model voting/fusion not implemented
4. **Temporal IoU:** No overlap metrics for predicted vs actual event timestamps
5. **Hallucination Quantification:** No formal metric for false positive rate

### Suggestions for V2

1. **Implement Evaluation Pipeline:**
   ```python
   # Compare predictions to Labels-v2.json
   # Calculate precision, recall, F1 per event type
   # Use temporal IoU with tolerance window
   ```

2. **Reduce Over-Detection:**
   - Increase confidence threshold (currently outputs 0.8+ for everything)
   - Post-process with NMS (Non-Maximum Suppression) for overlapping events
   - Train a classifier on top of LLM confidence scores

3. **Better Temporal Modeling:**
   - Explore video-native models (Video-LLaVA, VideoLLM)
   - Implement sliding window with memory/context carryover
   - Consider recurrent processing over longer clips

4. **Benchmark Expansion:**
   - Add more diverse leagues (not just EPL)
   - Test on different video qualities (480p, 1080p)
   - Include broadcast overlays vs clean footage

5. **Model Fine-Tuning:**
   - Create SoccerNet-specific training data
   - LoRA fine-tuning for event detection
   - Domain adaptation for soccer visual vocabulary

### Open Questions

- Why did Phi-3.5 produce empty outputs? GPU compatibility? Model loading issue?
- Is 900+ events per half over-detection or are subtle events valid?
- How do API models (Gemini, Llama 4) compare to local models?
- What's the optimal confidence threshold for precision-recall tradeoff?

---

## Appendix: File Reference

| Category | Files |
|----------|-------|
| **Worker Scripts** | `qwen_split_worker.py`, `llava_split_worker.py`, `phi3_split_worker.py`, `idefics2_split_worker.py`, `minicpm_split_worker.py` |
| **Merge/Dedup** | `merge_results.py`, `deduplicate_qwen.py`, `cleanup_duplicates.py` |
| **Orchestration** | `submit_restart.sh`, `download_batch.py` |
| **Documentation** | `LLAVA_README.md`, `IDEFICS2_README.md`, `DEDUPLICATION_README.md`, `RESTART_README.md`, `TOKEN_FIX.md` |
| **Test Scripts** | `Llama3.211BVision-TestScript`, `Pixtral12B-TestScript`, `Qwen2.5VL32B-AWQ-TestScript`, `Openrouter` |
| **Results** | `qwenjson/` (21 videos), `phi3.5/` (12 videos - empty) |

---

*Report generated from codebase analysis. Additional context from conversation history pending.*
