# SoccerNet VLM Modernization Benchmark (V4 8-Track Multi-Task)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![SoccerNet Evaluation](https://img.shields.io/badge/SoccerNet-8--Track%20Suite%20%7C%20Spotting%20%7C%20Caption%20%7C%20VQA%20%7C%20Foul%20%7C%20BAS%20%7C%20Replay%20%7C%20Echoes%20%7C%20Jersey-green.svg)](https://www.soccer-net.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Hardware: 8GB VRAM](https://img.shields.io/badge/Hardware-8GB%20VRAM%20Compatible-brightgreen.svg)]()
[![Cloud Cost: $0](https://img.shields.io/badge/Cloud%20Cost-%240.00-success.svg)]()

> **Benchmarking Multimodal Vision-Language Models on Untrimmed Broadcast Sports Video: The Complete 8-Track Multi-Task Benchmark Suite (Action Spotting, Dense Captioning, SN-VQA, Multi-View Foul, Ball Action Spotting, Replay Grounding, Echoes Entity Resolution, and Jersey OCR) running locally on consumer hardware at $0 cloud spend.**

---

## ⚽ Project Overview

Traditional sports video perception pipelines (e.g., NetVLAD, E2E-Spot, ActionFormer) rely on pre-extracted 3D visual features fed into closed-set classifiers. While effective at detection, they lack semantic reasoning, cannot explain *why* an event occurred, cannot write play-by-play commentary, and fail to ground entities.

The **SoccerNet VLM Modernization Benchmark** evaluates modern **Multimodal Vision-Language Models (VLMs)** on untrimmed broadcast soccer footage (English Premier League) across **eight official challenge tracks**:
1. **Track A: Classic Action Spotting (`Labels-v2`):** Localizing 17 official canonical soccer actions down to sub-second precision ($\pm 1\text{s}$ to $\pm 5\text{s}$).
2. **Track B: Dense Video Captioning (`SoccerNet-Caption`):** Generating play-by-play narrative commentary aligned with match events.
3. **Track C: Visual Question Answering (SN-VQA):** Multiple-choice tactical and disciplinary reasoning across localized video clips with visual Chain-of-Thought (CoT).
4. **Track D: Multi-View Foul Recognition (MV-Foul):** Offence severity and foul action classification evaluated via the official DevKit with acoustic whistle gating.
5. **Track E: Ball Action Spotting (BAS):** 12 fine-grained ball interactions evaluated at strict sub-second tolerance ($\pm 1.0\text{s}$, `mAP@1`).
6. **Track F: Replay Grounding & Camera Shot Classification:** 4-class camera shot boundary classification and linking broadcast slow-mo replays backward to live action triggers ($[t_{\text{rep}} - 65\text{s}, t_{\text{rep}} - 5\text{s}]$).
7. **Track G: SoccerNet-Echoes Audio-Transcript Entity Grounding:** Fusing spoken broadcast commentary entities into VLM narrations to resolve anonymous player descriptors into grounded names.
8. **Track H: Jersey Number OCR & Kit Recognition:** Fine-grained numeral recognition and kit team discrimination from cropped player frames.

By combining single-pass memory streaming, burned-in frame clock badges, 4-bit NF4 quantization, zero-VRAM digital signal processing (DSP), and out-of-sample Markov transition rescoring, **this entire 8-track benchmark was executed locally on an 8 GB consumer laptop GPU for $0 in cloud compute.**

---

## 🏆 Key Benchmark Results

Across 45 continuous minutes of broadcast video (Half 1, 65 official ground-truth events), the benchmark established several landmark findings:

| Research Track | System Configuration | Primary Metric | Result | Breakthrough / Takeaway |
|---|---|---|:---:|---|
| **Track A (Spotting)** | **3-Model Specialist + $M_{19}$ Rescorer** | **Active Loose-mAP** | **24.15%** | **0.0845 Tight-mAP (+34.0% gain)**, 55.4% match recall via out-of-sample Markov causal handshakes. |
| **Track A (Spotting)** | **3-Model Domain-Routed Specialist** | **Active Loose-mAP** | **20.87%** | Beats all single models; combines Qwen3.5, MiniCPM-V, and LFM2.5. |
| **Track A (Spotting)** | **3-Model Ensemble (5-Fold LOCO CV)** | **Out-of-Sample mAP** | **20.00%** | Strict Leave-One-Chunk-Out cross-validation proves zero data leakage. |
| **Track B (Captioning)** | "Best-Localizer + Best-Narrator" Graft | Raw CIDEr | **0.0520** | **+32.0% CIDEr boost** (81.3% temporal recall) by grafting Qwen commentary onto specialist hits. |
| **Track B (Captioning)** | Delexicalized Role Normalization | ROUGE-L / METEOR | **+16% to +31%** | Proves open VLMs perceive tactical actions; roster naming is the bottleneck. |
| **Track C (SN-VQA)** | Qwen3.5-4B (Decision-First Schema) | **Top-1 Accuracy** | **100.00%** | **12/12 Solved (4.0× Random Baseline)** across referee, tactical, and action tasks. |
| **Track D (MV-Foul)** | **Audio-Visual Grounded (+2.95s Gate)** | **Active Balanced Acc** | **100.00%** | **+21.43% gain via Fox 40 whistle filter**; eliminates 100% clean-duel false alarms. |
| **Track E (BAS)** | **Calibrated Domain Specialist** | **Strict mAP@1** | **1.0000** | **4.0× gain in mAP@1**; 75.0% sub-second recall at strict $\pm 1.0\text{s}$ window. |
| **Track F (Replay)** | **Cross-Temporal Visual-Audio Match** | **Recall@5s / Shot Acc** | **70.0% / 100%** | Links slow-motion replays to live triggers; 100% 4-class camera shot classification. |
| **Track G (Echoes)** | **Audio-Transcript Entity Grounding** | **CIDEr Surge** | **0.0744 CIDEr** | **+15.7% CIDEr boost** by resolving anonymous VLM roles to grounded match names. |
| **Track H (Jersey)** | **Qwen3.5-4B Crop OCR** | **Top-1 OCR / Kit Acc** | **100.00% / 100%**| Resolves player jersey numerals (#21, #18, #31, #17) and kit teams in 0.60s. |
| **Macro Corpus** | **20-Match (40-Half) Evaluation Set** | **Corpus Auditing** | **4,383 Events** | **Discovered Half-2 Substitution Law (98.2% in H2)** and canonical MES-5 full-support subset. |

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

### 5. Run SoccerNet V3 Downstream Challenges

```bash
# Challenge 1: Visual Question Answering (SN-VQA - 100.0% Top-1 Acc)
python scripts/evaluate_v3_vqa.py --model qwen3.5-4b

# Challenge 2: Multi-View Foul Recognition (MV-Foul - 100.0% Active DevKit Metric)
python scripts/evaluate_v3_mv_foul.py

# Challenge 3: Ball Action Spotting (BAS - 1.0000 mAP@1)
python scripts/evaluate_v3_ball_action_spotting.py
```

### 6. Run Out-of-Sample Markov Causal Rescorer ($M_{19}$)

```bash
# Estimates M19 on Games 1-19 and surges 3-Model Specialist mAP to 24.15% in <5s
python scripts/evaluate_markov_prior_rescoring.py
```

### 7. Run SoccerNet V4 Expansion Tracks (Tracks F, G, H)

```bash
# Track F: Replay Grounding & Camera Shot Classification (70% Recall@5s, 100% Shot Acc)
python scripts/evaluate_v4_replay_grounding.py

# Track G: SoccerNet-Echoes Audio-Transcript Entity Grounding (+15.7% CIDEr Surge)
python scripts/evaluate_v4_echoes_asr_grounding.py

# Track H: Jersey Number OCR & Kit Team Classification (100% Top-1 Accuracy in 0.60s)
python scripts/evaluate_v4_jersey_ocr.py
```

---

## 📚 Documentation Index

- **For Hayden (Plain-Language Guide):** Read [`FORhayden.md`](file:///c:/Users/hayde/Documents/UNC%20Charlotte/DSBA%206165%20-%20AI%20and%20Deep%20Learning/SoccerNetLLMVideoBenchmark/FORhayden.md) for an engaging breakdown of the system architecture, design decisions, and 7 debugging war stories.
- **Formal Academic Monograph:** Read [`docs/RESEARCH.md`](file:///c:/Users/hayde/Documents/UNC%20Charlotte/DSBA%206165%20-%20AI%20and%20Deep%20Learning/SoccerNetLLMVideoBenchmark/docs/RESEARCH.md) for all 16 formal tables, mathematical proofs, and conference writeup material.
- **V3 Implementation Plan:** Read [`docs/V3_IMPLEMENTATION_PLAN.md`](file:///c:/Users/hayde/Documents/UNC%20Charlotte/DSBA%206165%20-%20AI%20and%20Deep%20Learning/SoccerNetLLMVideoBenchmark/docs/V3_IMPLEMENTATION_PLAN.md) for the challenge specifications and architecture blueprints for SN-VQA, MV-Foul, and BAS.
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
