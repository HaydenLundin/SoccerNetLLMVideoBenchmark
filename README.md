# SoccerNet VLM Modernization Benchmark (V2)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![SoccerNet Evaluation](https://img.shields.io/badge/SoccerNet-Labels--v2%20%7C%20Caption-green.svg)](https://www.soccer-net.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Hardware: 8GB VRAM](https://img.shields.io/badge/Hardware-8GB%20VRAM%20Compatible-brightgreen.svg)]()
[![Cloud Cost: $0](https://img.shields.io/badge/Cloud%20Cost-%240.00-success.svg)]()

> **Benchmarking Multimodal Vision-Language Models on Untrimmed Broadcast Sports Video: Temporal Action Spotting, Dense Video Captioning, and Zero-VRAM Audio DSP Grounding on Consumer Hardware.**

---

## ⚽ Project Overview

Traditional sports action spotting systems (e.g., NetVLAD, E2E-Spot, ActionFormer) rely on pre-extracted 3D visual features fed into closed-set classifiers. While effective at detection, they lack semantic reasoning, cannot explain *why* an event occurred, and cannot write play-by-play commentary.

The **SoccerNet VLM Modernization Benchmark (V2)** evaluates modern **Multimodal Vision-Language Models (VLMs)** on untrimmed broadcast soccer footage (English Premier League). The benchmark simultaneously evaluates:
1. **Track A: Classic Action Spotting (`Labels-v2`):** Localizing 17 official canonical soccer actions down to sub-second precision ($\pm 1\text{s}$ to $\pm 5\text{s}$).
2. **Track B: Dense Video Captioning (`SoccerNet-Caption`):** Generating play-by-play narrative commentary aligned with match events.

By combining single-pass memory streaming, burned-in frame clock badges, 4-bit NF4 quantization, and zero-VRAM digital signal processing (DSP), **this entire academic benchmark suite was executed locally on an 8 GB consumer laptop GPU for $0 in cloud compute.**

---

## 🏆 Key Benchmark Results

Across 45 continuous minutes of broadcast video (Half 1, 65 official ground-truth events), the benchmark established several landmark findings:

| Research Track | System Configuration | Primary Metric | Result | Breakthrough / Takeaway |
|---|---|---|:---:|---|
| **Track A (Spotting)** | **3-Model Domain-Routed Specialist** | **Active Loose-mAP** | **20.87%** | **55.4% Match Recall (36 / 65 events)**; beats all single models. |
| **Track A (Spotting)** | **3-Model Ensemble (5-Fold LOCO CV)** | **Out-of-Sample mAP** | **20.00%** | Strict Leave-One-Chunk-Out cross-validation proves zero data leakage. |
| **Track A (Spotting)** | Qwen3.5-4B (Solo, $W=60\text{s}$) | Corner AP | **97.9%** | 2/2 hits, 0 false positives; $W=60\text{s}$ unlocks tactical play buildup. |
| **Track A (Spotting)** | LFM2.5-VL-3B (Solo, $W=30\text{s}$) | Boundary Latency | **12.95s/win** | Ultra-fast boundary specialist; 0.1486 AP on `Ball out of play`. |
| **Track B (Captioning)** | 3-Model Specialist Ensemble | Temporal Recall | **81.3%** | 26 / 32 official ground-truth captions matched within window. |
| **Track B (Captioning)** | "Best-Localizer + Best-Narrator" Graft | Raw CIDEr | **0.0520** | **+32.0% CIDEr boost** by grafting Qwen commentary onto specialist hits. |
| **Track B (Captioning)** | Delexicalized Role Normalization | ROUGE-L / METEOR | **+16% to +31%** | Proves open VLMs perceive tactical actions; roster naming is the bottleneck. |
| **RQ3 (Dual-Task CoT)** | Joint-CoT vs. Spotting-Only | Tight-mAP Gain | **19.4× Gain** | Omitting commentary causes "00:00" first-token prior mode collapse. |
| **RQ4 (Audio DSP)** | Zero-VRAM Whistle & Roar Gating (Raw) | FP Pruning | **52.1% Pruned** | Pruned 49 FP on Qwen, 47 phantom goals on MiniCPM in 3.22s CPU time. |
| **RQ4 (Audio DSP)** | **3-Model Specialist + Synchronized Gate** | **Acoustic Precision** | **0.0534 Tight-mAP** | **5 Phantom Goals Pruned**; preserves 20.63% mAP & proves Synchrony Law. |

---

## 🧠 Core Engineering & Scientific Breakthroughs

### 1. The Slotted Sliding-Window & PyAV Streaming Engine (199 MB RAM)
- Traditional video loaders (`cv2.VideoCapture`) leak memory and suffer 30-second seek latency on long files.
- Our custom **PyAV** streaming reader linearly decodes video packets at 1 FPS with **0.000s seek drift** while consuming only **199 MB of RAM**.
- Processed in 10-minute macro-chunks with 15s overlap to eliminate output token ceiling exhaustion.

### 2. The Burned-In Digital Clock Badge `[00:SS]` (Visual Wristwatch)
- Vision-Language Models cannot reliably count frame sequences or compute elapsed milliseconds.
- Stamping high-contrast digital clock badges `[00:00]` through `[00:59]` into frame corner pixels allows model visual attention heads to read elapsed time directly, completely eliminating minute-vs-second hallucinations.

### 3. RQ3: Autoregressive Chain-of-Thought (CoT) Synergy
- In our controlled ablation, removing the natural-language `commentary` field caused a **19.4× collapse in Tight-mAP** ($0.0389 \to 0.0020$).
- **The "00:00" First-Token Prior Collapse:** In Spotting-Only mode (`{"timestamp": "MM:SS", "label": "..."}`), `timestamp` is emitted first before visual reasoning occurs. The decoder collapses onto the first badge `[00:00]`, emitting 6 spurious `Kick-off` events at exact window-stride boundaries. Generating commentary first forces cross-attention to locate the active play before committing to a timestamp.

### 4. RQ4: Zero-VRAM Audio DSP Grounding & The Synchrony Law
- Bypassed GPU VRAM limits by running Short-Time Fourier Transforms (`scipy.signal` STFT) on CPU over 44.1 kHz AAC audio in **3.22 seconds**:
  - **Fox 40 Referee Whistle Flux ($3,500\text{ Hz} - 4,200\text{ Hz}$):** Extracted 55 discrete whistle peaks ($z \ge 4.0$).
  - **Crowd Roar Decibel Surge ($200\text{ Hz} - 2,000\text{ Hz}$):** Extracted 104 crowd roars ($\ge +5.0\text{ dB}$, peaking at $+16.32\text{ dB}$ on Goal).
- **The Cross-Modal Temporal Synchrony Law:** Models anchored on action initiation (MiniCPM-V 4.6, $\mu_{\Delta t} = -1.68\text{s}$) precede physical whistle blows by seconds. Initiator-biased models must be shifted by $+2.95\text{s}$ before causal acoustic gating. Synchronized selective gating pruned 5 phantom goals while preserving 20.63% Active Loose-mAP.

---

## 🏛️ The 20-Architecture Hardware Survival & Deployment Matrix (RTX 3000 Ada, 8 GB VRAM)

We evaluated **20 distinct multimodal model families** on consumer workstation hardware under Windows 11 WDDM ($< 7.50\text{ GiB}$ reserved ceiling):

```
The 20-Architecture Multimodal Deployment Matrix (8 GB GDDR6 VRAM, 63 GB RAM)
 ├── 🟢 10 Verified Live Hardware Survivors (50% Survival Rate)
 │    ├── 60s Full-Window Survivors (4-bit NF4 & Contact Sheet)
 │    │    ├── Qwen3.5-4B (W=60s; Peak 7.07 GiB; Tactical Buildup Anchor; 97.9% Corner AP)
 │    │    ├── MiniCPM-V 4.6 (W=60s; Peak 7.38 GiB; Duel/Offside Specialist; 81.3% Caption Recall)
 │    │    ├── Qwen2-VL-2B (W=60s; Peak 5.37 GiB; Ultra-fast 6.8s/win; Subject to RQ3 00:00 mode collapse)
 │    │    └── Moondream2 (W=60s; Peak 4.20 GiB; 4.1s 2D Contact Sheet Storyboard Baseline)
 │    ├── 30s Rapid Transition Specialist
 │    │    └── LFM2.5-VL-3B (W=30s; Peak 6.77 GiB; 12.95s/win; Boundary Specialist; -3.10s Recalibration)
 │    └── 15s Rescued & Edge Survivors
 │         ├── SmolVLM2-2.2B (W=15s default / 60s un-split; Peak 6.94 GiB; Native BF16)
 │         ├── Qwen2.5-VL-7B (W=15s; Peak 7.15 GiB; Temporal Receptive Field Starvation Baseline)
 │         ├── InternVL3.5-2B (W=15s; Peak 6.80 GiB; Rescued via num_patches_list=[1]*N from 17 GB OOM)
 │         ├── LLaVA-OneVision-0.5B (W=15s; Peak 4.14 GiB; Native BF16 Sub-1B Video Baseline)
 │         └── H2OVL-Mississippi-800M (W=15s; Peak 4.17 GiB; Rescued via max_tiles=1 from 12.5 GB spill)
 └── 🔴 10 Architectural, Library & Platform Blockers (Documented & Audited)
      ├── Dynamic High-Res Tile Explosions (>100 GB Virtual Allocation)
      │    └── Granite-Vision-3.2-2B (LLaVA-NeXT AnyRes 5 crops/frame × 15 frames = 216k tokens -> 113.8 GB OOM)
      ├── Physical VRAM Footprint (>7.50 GiB Reserved Ceiling)
      │    ├── Phi-4-Multimodal (5.6B + unquantized SigLIP/Conformer requires 15.18 GiB under Windows WDDM)
      │    ├── Gemma 4 E2B (Per-Layer Embedding tables unquantized in BNB 4-bit -> 9.11 GiB initial OOM)
      │    ├── Gemma 4 E4B (PLE unquantized tables require > 11.2 GiB VRAM on initial load)
      │    └── Aria-VLM (25.3B total MoE weights require > 16 GB VRAM resident memory)
      └── Platform, Library & Compilation Barriers on Windows
           ├── Ministral 3 3B (Mistral FP8 runtime blocked by missing Windows Triton JIT compiler)
           ├── PaliGemma2-3B (HTTP 401 Unauthorized gated repository; requires authenticated HF token)
           ├── Eagle2-2B (Deprecated from_args_and_dict config crash under Transformers >= 5.18)
           ├── DeepSeek-VL2-Tiny (DeepSeek MLA MoE custom CUDA kernel fails Windows PyTorch 2.6 build)
           └── InternVideo2-1B (VideoMAE deformable attention C++ operator fails Windows MSVC JIT)
```

---

## 📁 Repository Structure

```
SoccerNetLLMVideoBenchmark/
├── README.md                                    # This landing page
├── FORhayden.md                                 # Comprehensive plain-language architecture guide
├── eval_set.json                                # Frozen 20-match evaluation split
├── processing/
│   ├── unified_schema.py                        # Pydantic schemas (Joint-CoT & Spotting-Only)
│   ├── slice_macro_chunks.py                    # FFmpeg macro-chunking with packet PTS probing
│   ├── post_process_predictions.py              # PTS alignment & Class-Aware NMS
│   └── evaluate_soccernet.py                    # Official SoccerNet SDK metric wrapper
├── scripts/
│   ├── run_local_benchmark.py                   # Master GPU runner (PyAV streaming, clock overlays)
│   ├── evaluate_half1_synthesis_and_captioning.py # Track A & B multi-model fusion & delexicalization
│   └── evaluate_rq4_audio_grounding.py          # Zero-VRAM CPU audio STFT whistle & roar analyzer
├── docs/
│   ├── RESEARCH.md                              # 90 KB formal research monograph (12 tables, all proofs)
│   ├── IMPLEMENTATION_PLAN.md                   # Locked V2 blueprint & verification gates
│   ├── MODEL_CLAIM_AUDIT.md                     # Hardware & VRAM fit audit
│   └── SCOPE_GCP.md                             # Future Vertex AI cloud scale-up guide
└── logs/
    └── local_eval/                              # Machine-readable evaluation summaries & predictions
```

---

## 🚀 Quickstart & Reproduction

### 1. Prerequisites & Environment Setup

```bash
# Clone the repository
git clone https://github.com/HaydenLundin/SoccerNetLLMVideoBenchmark.git
cd SoccerNetLLMVideoBenchmark

# Set up Python virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # Windows PowerShell

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers accelerate bitsandbytes av scipy pillow soccernet pydantic
```

### 2. Run Local VLM Inference (Chunk 0 or Full Half 1)

```bash
# Run Qwen3.5-4B (4-bit NF4, W=60s, S=45s) on Chunk 0
python scripts/run_local_benchmark.py --model qwen3.5-4b --chunk-index 0

# Run full 45-minute Half 1 (Chunks 00-04)
python scripts/run_local_benchmark.py --model qwen3.5-4b --chunk-index -1
```

### 3. Evaluate 3-Model Specialist Ensemble & Track B Captioning

```bash
# Runs Task 1 (LFM calibration), Task 2 (Specialist Ensemble + LOCO CV), and Task 3 (Track B Captioning)
python scripts/evaluate_half1_synthesis_and_captioning.py
```

### 4. Run Zero-VRAM Audio DSP Grounding (RQ4)

```bash
# Ingests broadcast audio, extracts Fox 40 whistles + crowd roars, and runs 4-cohort benchmark in ~10s
python scripts/evaluate_rq4_audio_grounding.py
```

---

## 📚 Documentation Index

- **For Hayden (Plain-Language Guide):** Read [`FORhayden.md`](file:///c:/Users/hayde/Documents/UNC%20Charlotte/DSBA%206165%20-%20AI%20and%20Deep%20Learning/SoccerNetLLMVideoBenchmark/FORhayden.md) for an engaging breakdown of the system architecture, design decisions, and 7 debugging war stories.
- **Formal Academic Monograph:** Read [`docs/RESEARCH.md`](file:///c:/Users/hayde/Documents/UNC%20Charlotte/DSBA%206165%20-%20AI%20and%20Deep%20Learning/SoccerNetLLMVideoBenchmark/docs/RESEARCH.md) for all 12 formal tables, mathematical proofs, and conference writeup material.
- **Cloud Scale-Up Plan:** Read [`docs/SCOPE_GCP.md`](file:///c:/Users/hayde/Documents/UNC%20Charlotte/DSBA%206165%20-%20AI%20and%20Deep%20Learning/SoccerNetLLMVideoBenchmark/docs/SCOPE_GCP.md) for the Vertex AI Batch API blueprint for frontier models (Gemini 3.8 Flash / Gemini 4 Argon).

---

## 📄 Citation & Attribution

If you use this benchmark, codebase, or findings in your research, please cite:

```bibtex
@misc{lundin2026soccernetvlm,
  author = {Lundin, Hayden},
  title = {SoccerNet VLM Modernization Benchmark (V2): Multi-Modal Vision-Language Models for Untrimmed Sports Video Understanding},
  year = {2026},
  publisher = {University of North Carolina at Charlotte},
  howpublished = {\url{https://github.com/HaydenLundin/SoccerNetLLMVideoBenchmark}}
}
```
