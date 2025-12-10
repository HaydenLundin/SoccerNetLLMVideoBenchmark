# SoccerNet LLM Video Benchmark

Benchmarking Vision-Language Models (VLMs) for automatic soccer event detection and annotation on the SoccerNet English Premier League dataset.

## Overview

This project evaluates **15+ open-source and API-based Vision-Language Models** on their ability to detect and annotate soccer events from professional match footage. The benchmark processes **40+ full-length EPL videos** (~50 minutes each) using distributed GPU computing across multiple environments.

### Key Research Questions

- How well can VLMs detect soccer events without fine-tuning?
- What are the memory and performance characteristics of different models?
- Which optimization techniques work best for video understanding with limited VRAM?
- Which models are most reliable for sports event detection?

## Features

- **Multi-Model Support**: Evaluate 15+ VLMs including Qwen, LLaVA, IDEFICS2, Phi-3.5, MiniCPM, and more
- **Distributed Processing**: SLURM-based HPC orchestration with multi-GPU parallelization
- **Memory Optimization**: 4-bit quantization (BitsAndBytes NF4) to fit large models in limited VRAM
- **Robust Parsing**: 5-strategy JSON parsing to handle diverse LLM output formats
- **Intelligent Deduplication**: Window-based event deduplication with similarity matching
- **Resumable Processing**: Checkpoint system for long-running batch jobs

## Supported Models

| Model | Parameters | Status | Notes |
|-------|-----------|--------|-------|
| Qwen2.5-VL-7B | 7B | Primary | Best results, 21 videos processed |
| Qwen2.5-VL-32B (AWQ) | 32B | Quality baseline | Higher accuracy |
| LLaVA-v1.6-Mistral-7B | 7B | Working | Reliable fallback |
| IDEFICS2-8B | 8B | Working | Multi-image specialist |
| MiniCPM-V 2.6 | 8B | Working | Low hallucination rate |
| Phi-3.5-Vision | 4.2B | Issues | GPU compatibility problems |
| Pixtral-12B | 12B | Working | Vision-language |
| InternVL3-8B | 8B | Testing | Strong benchmarks |
| SmolVLM2-500M-Video | 500M | Testing | Fast baseline |
| Llama-3.2-11B-Vision | 11B | Issues | Markdown output format |

## Project Structure

```
SoccerNetLLMVideoBenchmark/
├── Worker Scripts (Inference)
│   ├── qwen_split_worker.py        # Qwen 2.5-VL 7B inference
│   ├── llava_split_worker.py       # LLaVA-v1.6 inference
│   ├── phi3_split_worker.py        # Phi-3.5-Vision inference
│   ├── idefics2_split_worker.py    # IDEFICS2-8B inference
│   └── minicpm_split_worker.py     # MiniCPM-V 2.6 inference
│
├── Merge & Deduplication
│   ├── merge_results.py            # Generic merge with deduplication
│   ├── merge_results_*.py          # Model-specific merge scripts
│   ├── deduplicate_qwen.py         # Advanced multi-strategy parser
│   └── cleanup_*.py                # Cleanup scripts
│
├── Orchestration (SLURM)
│   ├── submit_*.sh                 # SLURM job submission scripts
│   └── setup_*.sh                  # Environment setup scripts
│
├── Utilities
│   └── download_batch.py           # SoccerNet dataset downloader
│
├── Results
│   ├── qwenjson/                   # Qwen model results (21 videos)
│   └── phi3.5/                     # Phi-3.5 results
│
└── Documentation
    ├── RETROSPECTIVE_REPORT.md     # Technical report & lessons learned
    ├── LLAVA_README.md             # LLaVA pipeline docs
    ├── IDEFICS2_README.md          # IDEFICS2 pipeline docs
    ├── DEDUPLICATION_README.md     # Deduplication strategy
    └── RESTART_README.md           # Resumable processing docs
```

## Requirements

### Python Dependencies

```
torch
transformers
accelerate
bitsandbytes
pillow
soccernet
```

### System Requirements

- **GPU**: NVIDIA GPU with 11GB+ VRAM (24GB recommended)
- **CUDA**: 11.8 or higher
- **FFmpeg**: For video frame extraction
- **HuggingFace Token**: For model downloads

### Tested Environments

- **Google Colab Pro**: NVIDIA A100-SXM4-80GB
- **UNC Charlotte HPC**:
  - Titan RTX: 4x 24GB GDDR6
  - Titan V: 8x 12GB HBM2
  - GTX-1080Ti: 8x 11GB GDDR5X
- **Cloud APIs**: OpenRouter, Gemini 2.0 Flash

## Installation

1. Clone the repository:
```bash
git clone https://github.com/HaydenLundin/SoccerNetLLMVideoBenchmark.git
cd SoccerNetLLMVideoBenchmark
```

2. Install dependencies:
```bash
pip install torch transformers accelerate bitsandbytes pillow soccernet
```

3. Set up HuggingFace token:
```bash
export HF_TOKEN=your_huggingface_token
```

4. Download SoccerNet dataset:
```bash
python download_batch.py --batch 0  # Downloads first 20 videos
```

## Usage

### Single Video Processing

```bash
# Process a single video with Qwen model
python qwen_split_worker.py --video_id 0 --gpu_id 0 --start 0 --end 750
```

### Batch Processing (SLURM)

```bash
# Submit batch job for multiple videos
sbatch submit_qwen.sh
```

### Merge Results

```bash
# Merge partial results from multiple GPUs
python merge_results_qwen.py --video_id 0
```

### Deduplicate Events

```bash
# Remove duplicate events with advanced parsing
python deduplicate_qwen.py --input qwenjson/ --debug
```

## Processing Pipeline

1. **Frame Extraction**: Extract frames at 5 FPS using FFmpeg
2. **Window Processing**: Process 3-second windows with 50% overlap
3. **Event Detection**: VLM analyzes frames and outputs JSON events
4. **Merging**: Combine results from parallel GPU workers
5. **Deduplication**: Remove duplicate events within 3-second windows

### Event Detection Prompt

The models detect 17 soccer event types:
- **Goals/Shots**: Goal, Shot on target, Shot off target
- **Fouls/Cards**: Foul, Yellow card, Red card, Offside
- **Set Pieces**: Corner, Free-kick, Penalty, Throw-in, Kick-off, Goal kick
- **Other**: Substitution, Ball out of play, Clearance

## Output Format

```json
{
  "time": 125.0,
  "raw": "{\"label\": \"Goal\", \"team\": \"home\", \"confidence\": 0.95, \"details\": \"Header from corner kick\"}"
}
```

## Technical Details

### Memory Optimization

- **4-bit Quantization**: NF4 quantization reduces 7B model to ~11GB VRAM
- **Chunk Processing**: 3-second windows with 15 frames each
- **Incremental Saving**: Results saved after each window to prevent data loss

### Parallelization Strategy

```
GPU 0 → Process 0-750s     ┐
GPU 1 → Process 750-1500s  ├→ Merge & Deduplicate → Final Results
GPU 2 → Process 1500-2250s │
GPU 3 → Process 2250-3000s ┘
```

### JSON Parsing Strategies

The deduplication system uses 5 fallback strategies:
1. Direct JSON parse
2. Regex extraction of `{...}` patterns
3. Single quotes → double quotes conversion
4. Python `ast.literal_eval` for dict syntax
5. Similarity matching for near-duplicates

## Results

- **Qwen2.5-VL-7B**: 21 videos processed, ~900 events per 50-minute half
- **Deduplication**: 49.4% reduction (176 → 89 events average)
- **Processing Time**: 3-5 minutes per video with 4 GPUs

## Documentation

- [RETROSPECTIVE_REPORT.md](RETROSPECTIVE_REPORT.md) - Comprehensive technical report
- [LLAVA_README.md](LLAVA_README.md) - LLaVA pipeline documentation
- [IDEFICS2_README.md](IDEFICS2_README.md) - IDEFICS2 pipeline documentation
- [DEDUPLICATION_README.md](DEDUPLICATION_README.md) - Deduplication strategy
- [RESTART_README.md](RESTART_README.md) - Resumable processing workflow

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## License

This project is for research purposes. Please ensure compliance with the [SoccerNet NDA](https://www.soccer-net.org/) for dataset usage.

## Acknowledgments

- [SoccerNet](https://www.soccer-net.org/) for the dataset and annotations
- [HuggingFace](https://huggingface.co/) for model hosting and Transformers library
- UNC Charlotte HPC for compute resources
