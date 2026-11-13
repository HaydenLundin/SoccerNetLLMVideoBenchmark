# SoccerNet VLM Modernization Benchmark (V5 Complete 12-Track Ecosystem)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![SoccerNet Evaluation](https://img.shields.io/badge/SoccerNet-12--Track%20Suite%20%7C%20Spotting%20%7C%20VQA%20%7C%20GSR%20%7C%20Tracking%20%7C%20Depth-green.svg)](https://www.soccer-net.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Hardware: 8GB VRAM](https://img.shields.io/badge/Hardware-8GB%20VRAM%20Compatible-brightgreen.svg)]()
[![Cloud Cost: $0](https://img.shields.io/badge/Cloud%20Cost-%240.00-success.svg)]()

> **Benchmarking Multimodal Vision-Language Models & Classical Projective Geometry on Untrimmed Broadcast Sports Video: Action Spotting measured with the official SoccerNet DevKit, Dense Captioning, and harnesses for ten further SoccerNet task families (SN-VQA, MV-Foul, BAS, Replay Grounding, Echoes Entity Resolution, Jersey OCR, Camera Calibration, Game State Reconstruction, Multi-Object Tracking/Re-ID, and 3D Metric Depth) + ReVal Athlete Valuation running locally on consumer hardware at $0 cloud spend.**

---

## ⚽ Project Overview

Traditional sports video perception pipelines (e.g., NetVLAD, E2E-Spot, ActionFormer) rely on pre-extracted 3D visual features fed into closed-set classifiers. While effective at detection, they lack semantic reasoning, cannot explain *why* an event occurred, cannot write play-by-play commentary, and fail to ground entities.

The **SoccerNet VLM Modernization Benchmark** evaluates modern **Multimodal Vision-Language Models (VLMs)** and spatial/geometric perception primitives on untrimmed broadcast soccer footage (English Premier League). Tracks A and B are measured benchmarks. Tracks C–L are small single-half harnesses with self-authored labels; those marked † score scripted predictions or use a self-derived reference and are illustrations, not model measurements (see `docs/RESEARCH.md` provenance notes):
1. **Track A: Classic Action Spotting (`Labels-v2`):** Localizing 17 official canonical soccer actions down to sub-second precision ($\pm 1\text{s}$ to $\pm 5\text{s}$).
2. **Track B: Dense Video Captioning (`SoccerNet-Caption`):** Generating play-by-play narrative commentary aligned with match events.
3. **Track C: Visual Question Answering (SN-VQA):** Multiple-choice tactical and disciplinary reasoning across localized video clips with visual Chain-of-Thought (CoT).
4. **Track D †: Multi-View Foul Recognition (MV-Foul):** Offence severity and foul action classification on 12 hand-authored duels with a scripted whistle rule (no audio or video is read).
5. **Track E †: Ball Action Spotting (BAS):** 12 fine-grained ball classes scored at a strict $\pm 1.0\text{s}$ tolerance on typed-in prediction lists (no model run).
6. **Track F †: Replay Grounding & Camera Shot Classification:** 4-class camera shot boundary classification and linking broadcast slow-mo replays backward to live action triggers ($[t_{\text{rep}} - 65\text{s}, t_{\text{rep}} - 5\text{s}]$).
7. **Track G †: SoccerNet-Echoes Entity Grounding:** Substituting fixed default names into VLM narrations to resolve anonymous player descriptors (no audio or transcript is used yet).
8. **Track H †: Jersey Number OCR & Kit Recognition:** Harness for numeral recognition and kit discrimination; the current predictions are a simulated literal, not a model run.
9. **Track I: Camera Calibration & Pitch Line Registration:** Chamfer-distance 3D pinhole DLT/pose optimization across calibrated pitch lines and circles.
10. **Track J: Game State Reconstruction (2D Minimap):** Inverse homography ($\mathbf{H}^{-1}$) projection into canonical $105\text{m}\times 68\text{m}$ coordinates with `RETR_EXTERNAL` pitch blob detection and 4-role kit classification.
11. **Track K †: Multi-Object Tracking & Player Re-ID:** Kalman + HSV appearance tracking with hard-cut/dissolve histogram recovery, scored against pseudo-ground-truth built from the tracker's own detections.
12. **Track L: 3D Metric Depth Estimation:** Trust-Region Reflective physical camera pose estimation ($\mathbf{H}^{-1} + \mathbf{C}_{\text{pred}}$) for per-player metric depth across `38.6m–94.2m`.

By combining single-pass memory streaming, burned-in frame clock badges, 4-bit NF4 quantization, zero-VRAM digital signal processing (DSP), out-of-sample Markov transition rescoring, and inverse-homography pitch geometry, **everything here was executed locally on an 8 GB consumer laptop GPU for $0 in cloud compute.**

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
| **Track C (SN-VQA)** | Qwen3.5-4B (Decision-First Schema) | **Top-1 Accuracy** | **100.00%** | 12/12 on self-authored questions (live run; smoke test, not an official SN-VQA set). |
| **Track D (MV-Foul)** † | Scripted whistle rule | Active Balanced Acc | 100.00% (scripted) | N = 12 hand-authored duels; holds by construction, no audio or video read. |
| **Track E (BAS)** † | Typed-in prediction lists | Precision at $\pm 1.0\text{s}$ | 1.0000 (scripted) | N = 16; 75.0% recall; "mAP@1" in the harness is precision; no model run. |
| **Track F (Replay)** † | Typed features, oracle candidate selection | Recall@5s / Shot Acc | 70.0% / 100% (scripted) | N = 10 / 20; Recall@5s is 30% to 50% without oracle selection. |
| **Track G (Echoes)** † | Fixed-name substitution | Named CIDEr | 0.0744 (from 0.0643) | N = 22 caption pairs; no audio or transcript is used. |
| **Track H (Jersey)** † | Simulated prediction literal | Top-1 OCR / Kit Acc | 100% / 100% (simulated) | N = 15; predictions equal the labels by construction; no model run, latency not measured. |
| **Track I (Calibration)** | **Chamfer Pinhole DLT** | **`IoU_part` / `AC@10px`** | **0.9601 / 78.47%** | Drops reprojection error from `835.97px` (Hough) to **`8.14px`** (`5.34px` median). |
| **Track J (Game State)** | **Calibrated $\mathbf{H}^{-1}$ + 4-Role HSV** | **`GS-HOTA` / `Loc F1@2m`** | **84.78% / 89.62%** | `0.238m` mean pitch error (`98.78%` role acc) across 92 players vs `1.91%` linear baseline. |
| **Track K (Tracking/ReID)** † | **Kalman + HSV + Cut Recovery** | **`HOTA` / `IDF1` / `Top-1`** | 72.63% / 69.70% / 34.52% | -52.1% ID Switches (`549 -> 263`) across 650 frames against `8,692` pseudo-GT boxes built from the tracker's own detections. |
| **Track L (3D Depth)** | **TRF Pinhole Raycast / Hybrid** | **`Abs Rel` / `RMSE` ($\delta_1$)** | **0.0496 / 4.46m (`100%`)** | Beats scale-prior baseline (`32.60m` RMSE); hybrid regularizer scores `0.0512 / 4.51m`. |
| **Macro Corpus (MES-5)** | **3-Model Specialist + $M_{15}$ Prior** | **Official Loose / Tight mAP** | **0.1833 / 0.0607** | 17 / 17 active classes across 1,132 events in 10 halves. In-sample: routing and rescoring rules were revised on MES-5. Qwen3.5-4B solo is 0.1564 / 0.0562. |
| **Macro Corpus (MES-5)** | 3-Model routing, 5-fold leave-one-match-out | Official Loose / Tight mAP | 0.1613 / 0.0261 | Out-of-sample routing estimate; four distinct partitions across five folds, none equal to the canonical one. |
| **Systems (Stage 2)** | CPU Window Skip (simulated on MES-5) | Loose-mAP / Windows Skipped | 0.1840 / 2.6% | 100.4% mAP retention (0.0609 Tight) with 18 of 687 windows masked (about 1.03×). The 3.3× Concise-CoT speedup was measured on one chunk and not scored on MES-5. |
| **Demonstration** | **Interactive Web Cockpit (`apps/demo/`)** | **Latency / FPS** | **60 FPS Scrubbing** | Standalone Python HTTP 206 streamer + Canvas waveform visualizer across all 75 chunks. |
| **Cross-Sport Transfer** | **Multi-Sport Adapter (Football/Hoops)** | **CV Reduction (`B0`)** | **-62.44% CV (`6.75%`)** | Log-frequency anchoring aligns Soccer/Football/Hoops (uncapped sum CV `-80.81%`). |
| **Athlete Valuation (ReVal)** | **3-Pillar Valuation & Splicer (`B1`–`B4`)** | Rank Corr / Attribution F1 | $\rho=0.770$ (n = 10) / 34.34% F1 | Leaderboard is hand-curated (12 anchors); $\rho$ is against a hand-entered rating list, not independent validation. **60.45 FPS** H.264+AAC reels (`<=23.2ms` drift) & WCAG AAA-Large/AA HUD. |

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
├── apps/
│   └── demo/                                    # Interactive broadcast demonstration cockpit
│       ├── server.py                            # Python HTTP 206 partial content streaming & REST API
│       └── static/                              # HTML5 Canvas waveforms, video player & CoT inspector
├── processing/
│   ├── unified_schema.py                        # Pydantic schemas (Joint-CoT & Spotting-Only)
│   ├── slice_macro_chunks.py                    # FFmpeg macro-chunking with packet PTS probing
│   ├── post_process_predictions.py              # PTS alignment & Class-Aware NMS
│   └── evaluate_soccernet.py                    # Official SoccerNet SDK metric wrapper
├── scripts/
│   ├── run_local_benchmark.py                   # Master GPU runner (PyAV streaming, clock overlays, --concise-cot)
│   ├── simulate_mes5_window_skipping.py         # Full MES-5 Pareto simulation across 687 windows
│   ├── evaluate_mes5_multimatch.py              # Canonical MES-5 17-class DevKit evaluation suite
│   ├── evaluate_half1_synthesis_and_captioning.py # Track A & B multi-model fusion & delexicalization
│   └── evaluate_rq4_audio_grounding.py          # Zero-VRAM CPU audio STFT whistle & roar analyzer
├── docs/
│   ├── RESEARCH.md                              # 90 KB formal research monograph (27 tables, all proofs)
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

### 2. Run Local VLM Inference with Concise-CoT & CPU Window Skipping (Stage 2)

```bash
# Run Qwen3.5-4B with 3.3x latency acceleration (Concise-CoT <= 15 words, CPU DSP skipping)
python scripts/run_local_benchmark.py --model qwen3.5-4b --chunk-index 0 --concise-cot --skip-inactive

# Run full 45-minute Half 1 (Chunks 00-04)
python scripts/run_local_benchmark.py --model qwen3.5-4b --chunk-index -1 --concise-cot --skip-inactive
```

### 3. Launch the Interactive Demonstration Showcase (HTTP 206 Streaming)

```bash
# Launch zero-dependency local demonstration cockpit on port 8080
python apps/demo/server.py --port 8080
# Open http://localhost:8080 in your browser for live video scrubbing, Canvas waveforms, and CoT inspection
```

### 4. Evaluate 3-Model Specialist Ensemble & Track B Captioning

```bash
# Runs Task 1 (LFM calibration), Task 2 (Specialist Ensemble + LOCO CV), and Task 3 (Track B Captioning)
python scripts/evaluate_half1_synthesis_and_captioning.py
```

### 5. Run Zero-VRAM Audio DSP Grounding (RQ4)

```bash
# Ingests broadcast audio, extracts Fox 40 whistles + crowd roars, and runs 4-cohort benchmark in ~10s
python scripts/evaluate_rq4_audio_grounding.py
```

### 6. Run SoccerNet V3 Downstream Challenges

```bash
# Challenge 1: Visual Question Answering (SN-VQA - 12/12 on self-authored questions, live run)
python scripts/evaluate_v3_vqa.py --model qwen3.5-4b

# Challenge 2: Multi-View Foul Recognition (MV-Foul - scripted illustration, no model run)
python scripts/evaluate_v3_mv_foul.py

# Challenge 3: Ball Action Spotting (BAS - scripted illustration, no model run)
python scripts/evaluate_v3_ball_action_spotting.py
```

### 7. Run Canonical MES-5 Multi-Match & Window Skipping Simulation

```bash
# Evaluates 3-Model MECE Ensemble + M15 prior across all 5 matches (1,132 events, 17 classes)
# Requires SoccerNet Labels-v2.json under test_eval_ground_truth/ (not tracked in this repo)
python scripts/evaluate_mes5_multimatch.py

# Simulates Stage 2 CPU acoustic/optical window skipping (18 of 687 windows masked, 0.1840 Loose-mAP)
python scripts/simulate_mes5_window_skipping.py
```

### 8. Run SoccerNet V4 Expansion Tracks (Tracks F, G, H)

```bash
# Track F: Replay Grounding & Camera Shot Classification (scripted illustration, oracle selection)
python scripts/evaluate_v4_replay_grounding.py

# Track G: SoccerNet-Echoes Entity Grounding (fixed-name substitution, no transcript used)
python scripts/evaluate_v4_echoes_asr_grounding.py

# Track H: Jersey Number OCR & Kit Team Classification (simulated predictions, no model run)
python scripts/evaluate_v4_jersey_ocr.py
```

### 9. Run SoccerNet V5 Spatial Tracks (Tracks I–L), ReVal Benchmarks & Tests

```bash
# Track I: Camera Calibration & Pitch Line Registration (IoU_part = 0.9601, AC@10px = 78.47%)
python scripts/evaluate_v5_camera_calibration.py

# Track J: Game State Reconstruction / 2D Minimap (GS-HOTA = 84.78%, Loc F1@2m = 89.62%)
python scripts/evaluate_v5_game_state_gsr.py

# Track K: Multi-Object Tracking & Player Re-ID (HOTA = 72.63% vs tracker-derived pseudo-GT, -52.1% IDSW)
python scripts/evaluate_v5_tracking_reid.py

# Track L: 3D Metric Depth Estimation (Abs Rel = 0.0496, RMSE = 4.455m, 100% delta_1)
python scripts/evaluate_v5_depth_estimation.py

# ReVal 5-Benchmark Empirical Suite & All Verification Tests
python scripts/benchmark_reval_engine.py
python scripts/test_v5_tracks_suite.py
python scripts/test_reval_highlights.py
python scripts/test_demo_server.py
```

---

## 📚 Documentation Index

- **For Hayden (Plain-Language Guide):** Read [`FORhayden.md`](FORhayden.md) for an engaging breakdown of the system architecture, design decisions, and debugging war stories.
- **Formal Academic Monograph:** Read [`docs/RESEARCH.md`](docs/RESEARCH.md) for all formal tables, mathematical proofs, and conference writeup material.
- **V3 Implementation Plan:** Read [`docs/V3_IMPLEMENTATION_PLAN.md`](docs/V3_IMPLEMENTATION_PLAN.md) for the challenge specifications and architecture blueprints for SN-VQA, MV-Foul, and BAS.
- **Cloud Scale-Up Plan:** Read [`docs/SCOPE_GCP.md`](docs/SCOPE_GCP.md) for the Vertex AI Batch API blueprint for frontier models (Gemini 3.8 Flash / Gemini 4 Argon).

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
