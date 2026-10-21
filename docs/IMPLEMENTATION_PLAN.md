# SoccerNet VLM Modernization Benchmark (V2) — Implementation Plan

**Target Architecture:** Hybrid GCP Vertex AI (Gemini 3.8 Flash, Gemini 4 Argon) + Local Workstation (RTX 3000 Ada 8GB for Qwen/Gemma)  
**Primary Tasks:** Dual Evaluation — Classic Action Spotting (`Labels-v2`, 17 classes) + Dense Video Captioning (`SoccerNet-Caption`)  
**Frozen Eval Set:** 20 EPL matches (40 halves: 18 test matches = 36 halves, 2 valid matches = 4 halves) defined in `eval_set.json`  
**Budget Cap:** $300 GCP Free Trial Credit (targeting ~$120–$180 spend)

---

## 1. System Architecture

```
                       ┌──────────────────────────────────────────────┐
                       │   Local Pre-processing & Slicing             │
                       │   - Download 20 EPL games (.mkv, 720p)       │
                       │   - Convert to .mp4 (H.264/AAC, stream copy) │
                       │   - Slice 10-min macro-chunks (15s overlap)  │
                       └──────────────────────┬───────────────────────┘
                                              │
                      ┌───────────────────────┴───────────────────────┐
                      ▼                                               ▼
         ┌─────────────────────────┐                     ┌──────────────────────────┐
         │ GCS Bucket (Private)    │                     │ Local Workstation        │
         │ gs://.../mp4_chunks/    │                     │ (RTX 3000 Ada 8GB VRAM)  │
         └────────────┬────────────┘                     └─────────────┬────────────┘
                      │                                                │
         ┌────────────┴────────────┐                     ┌─────────────┴────────────┐
         │ GCP Vertex AI Batch API │                     │ 60-second Window / 1 FPS │
         │ - Gemini 3.8 Flash (40) │                     │ - Qwen2.5-VL-7B (4-bit)  │
         │ - Gemini 4 Argon (5-10) │                     │ - Gemma 4 / MiniCPM-V    │
         │ (50% token discount)    │                     │ - Strict local JSON mode │
         └────────────┬────────────┘                     └─────────────┬────────────┘
                      │                                                │
                      └───────────────────────┬────────────────────────┘
                                              ▼
                       ┌──────────────────────────────────────────────┐
                       │   Unified Response Validator & Converter     │
                       │   - Enforces 17-class enum                   │
                       │   - CoT commentary evaluated first           │
                       │   - Deterministic 'MM:SS' -> position (ms)   │
                       │   - Border NMS across overlapping windows    │
                       └──────────────────────┬───────────────────────┘
                                              │
                      ┌───────────────────────┴───────────────────────┐
                      ▼                                               ▼
         ┌─────────────────────────┐                     ┌──────────────────────────┐
         │ SoccerNet.Evaluation    │                     │ Caption Metric Evaluator │
         │ - Tight mAP (1s to 5s)  │                     │ - CIDEr, BLEU-4, METEOR, │
         │ - Loose mAP (5s to 60s) │                     │   ROUGE-L                │
         │ - Class P / R / F1      │                     │ - 30-second window (±15s)│
         └────────────┬────────────┘                     └─────────────┬────────────┘
                      │                                                │
                      └───────────────────────┬────────────────────────┘
                                              ▼
                       ┌──────────────────────────────────────────────┐
                       │ BigQuery Storage & Looker Studio Dashboard   │
                       └──────────────────────────────────────────────┘
```

---

## 2. Phased Work Breakdown

### Phase 0: Foundation & GCP Guardrails (< $5, Day 1)
- [x] **Eval Set Freezing:** Execute `processing/freeze_eval_set.py` to lock 20 EPL matches (40 halves) into `eval_set.json`.
- [ ] **GCP Project & Quotas:**
  - Create GCP project (`soccernet-vlm-benchmark`).
  - Enable APIs: `aiplatform.googleapis.com`, `storage.googleapis.com`, `bigquery.googleapis.com`.
  - Set hard quota limits in **IAM & Admin -> Quotas** on Vertex AI daily requests and token ceilings to guard against overruns.
- [ ] **Billing Alert & Mitigation of Lag:**
  - Standard Pub/Sub billing alerts lag by 1 to 6 hours.
  - Mitigation Strategy:
    1. Submit Vertex AI batch predictions in **controlled 5-half waves** rather than a single 40-half queue.
    2. Deploy Cloud Run Function listening to billing alerts at $75, $150, $225, and $270.
- [ ] **GCS Bucket Setup:**
  - Create private regional bucket: `gs://<PROJECT_ID>-soccernet-raw/` (us-central1).
  - IAM strictly private in accordance with the SoccerNet NDA.
- [ ] **Video Transcoding & Slicing:**
  - Standardize video files from `.mkv` to `.mp4` (H.264 video + AAC audio) using stream copying (`ffmpeg -i input.mkv -c copy output.mp4`) to prevent Vertex AI decoding failures.
  - Slice each 45-minute half into **10-minute macro-chunks** (600 seconds) with **15 seconds overlap** using stream copy:
    ```bash
    ffmpeg -ss {start} -i half.mp4 -t 615 -c copy chunk_{i}.mp4
    ```
- [ ] **Phase 0 Batch API Smoke Test:**
  - Generate a 2-row test JSONL manifest in GCS.
  - Execute a test job using `client.batches.create` via Vertex AI to validate Batch API service account permissions, GCS read/write access, and JSONL format compliance before processing any match data.

---

### Phase 1: Dual Evaluation Harness & Post-Processor ($0, Days 1–2)
- [x] **Implement `processing/unified_schema.py` & `processing/post_process_predictions.py`:**
  - Enforce official **17-class enum** (incorporating `Shots on target` and `Shots off target`).
  - Order `commentary` **before** `label` in the JSON schema so autoregressive token generation acts as an observational Chain-of-Thought prior to categorization.
  - Ingest timestamps as `"timestamp": "MM:SS"` with strict regex (`^\d{2}:[0-5]\d$`) and run deterministic conversion:
    $$\text{position\_ms} = (\text{clip\_offset\_sec} + \text{minutes} \times 60 + \text{seconds}) \times 1000$$
    $$\text{gameTime} = \text{f"{half} - \{total\_min:02d\}:\{total\_sec:02d\}"}$$
  - Implement event-level duration bounds checking against `chunks_manifest.json` measured keyframe timestamps (fault-isolated drop with telemetry counter).
- [x] **Implement Multi-Window Stitching & Class-Aware NMS:**
  - Stitch the 5 macro-chunks per half into a single timeline using exact keyframe start timestamps.
  - Apply class-aware Non-Maximum Suppression (NMS) within a 3-second radius across overlapping chunk borders, suppressing duplicate detections of the same class while strictly preserving valid co-occurring events (e.g., Foul followed immediately by Yellow Card).
- [x] **Build & Verify `evaluate_soccernet.py`:**
  - **Action Spotting Evaluator:** Integrate `SoccerNet.Evaluation.ActionSpotting.evaluate` to output:
    1. **Official Tight Average-mAP:** $\delta \in [1, 2, 3, 4, 5]$ seconds (1s steps).
    2. **Official Loose Average-mAP:** $\delta \in [5, 10, 15, \dots, 60]$ seconds (5s steps).
    3. Custom diagnostic thresholds: $\Delta = \{1\text{s}, 5\text{s}, 20\text{s}, 60\text{s}\}$.
    4. Per-class Precision, Recall, and F1.
  - **Dense Captioning Evaluator:** Integrate standard caption metrics (CIDEr, BLEU-4, METEOR, ROUGE-L) matched within the official 30-second window ($\pm 15\text{s}$).
  - **Synthetic Test Verified:** Validated harness with synthetic perfect predictor against ground truth match, achieving 1.0000 AP on all active classes (0.8235 global mAP across 14/17 active classes).

---

### Phase 2: Vertex AI Gemini Benchmark ($60–$90, Days 2–3)
- [ ] **Execution with 50% Token Discount:**
  - Run all inference via `client.batches.create` (Vertex AI Batch Prediction).
- [ ] **Gemini 3.8 Flash:**
  - Full evaluation across all 40 halves (5 macro-chunks per half = 200 chunks total).
  - Submit in 4 waves of 10 halves (50 chunks per wave) with billing verification between waves.
- [ ] **Gemini 4 Argon:**
  - Target a focused subset of 5 to 10 high-event halves to benchmark frontier reasoning without exhausting trial credit.
- [ ] **Format Outputs:**
  - Save outputs into standardized `Predictions-v2.json` and `Predictions-Caption.json`.

---

### Phase 3: Local Open-Weight Comparison ($0 Compute, Days 3–4)
- [x] **Local 8GB VRAM Constraint & Memory Gate Protocol (`scripts/test_local_memory_gate.py`):**
  - Evaluated on NVIDIA RTX 3000 Ada Laptop GPU (8,188 MiB GDDR6 VRAM, WDDM mode) across stepped frame ladder $W \in \{15\text{s}, 30\text{s}, 60\text{s}\}$ at 1.0 FPS ($\le 448 \times 448$).
  - Subprocess isolation verified 0% CUDA context leakage; pass threshold enforced $\text{Peak Reserved} < 7.50\text{ GiB}$, zero host shared memory spillover, and strict JSON Schema compliance.
- [ ] **Locked Model Lineups ($0 Local Compute):**
  - **Full $W = 60\text{s}$ Local Benchmark Cohort (All 20 Matches / 40 Halves):**
    - `SmolVLM2-2.2B` (Native BF16, 6.06 GiB peak, 17.73s latency)
    - `Qwen3.5-4B` (4-bit NF4, 7.07 GiB peak, 40.86s latency with `enable_thinking=False` and `max_pixels=256*256`)
    - `MiniCPM-V 4.6` (4-bit NF4, 7.38 GiB peak, 29.72s latency)
    - `Moondream2` (Native BF16, 4.34 GiB peak, 3.39s latency, 2D composite contact sheet storyboard baseline)
  - **Stepped Boundary Local Ablations (5-Half Subset):**
    - `LFM2.5-VL-3B` (4-bit NF4): Evaluated locally at its verified physical ceiling of **$W = 30\text{s}$** (7.25 GiB peak, 15.79s latency).
    - `Qwen2.5-VL-7B` (Track 4 Historical Longitudinal Bridge, 4-bit NF4): Evaluated locally at its verified physical ceiling of **$W = 15\text{s}$** (6.81 GiB peak, 7.77s latency with `max_pixels=256*256`) to anchor against legacy V1 3-second baseline.
  - **Empirically Disqualified Models:**
    - `InternVL3.5-2B` (ViT sequence explosion at 30f: 17.04 GiB)
    - `Gemma 4 E4B/E2B` (Unquantized PLE tables: 6.39 GiB static load, 9.11 GiB at 15f)
    - `Ministral 3 3B` (FineGrainedFP8 requires `triton` kernels unavailable on Windows)
    - `Phi-4 Multimodal` (Accelerate meta-tensor init crash in `speech_conformer_encoder.py`)
- [x] **Strict Budget Guardrail:**
  - Zero open-weight deployment to Vertex AI GPU endpoints (avoiding unallocated GPU VM quota locks and hourly idle billing).
- [x] **Local Inference Runner (`scripts/run_local_benchmark.py`):**
  - Implemented and verified end-to-end: in-memory PyAV video decoding, sliding-window inference ($W=60\text{s}, S=45\text{s}$), abstract `"MM:SS"` prompt formatting (preventing few-shot anchoring), PTS stitching with Class-Aware NMS, and official SoccerNet Action Spotting mAP evaluation ($0 compute).

---

### Phase 4: Controlled Research Ablations ($15–$30, Day 5)
Run on a designated 5-half test subset:
1. **RQ1 & Baseline Comparison:** Generalist VLMs vs. Supervised Baselines (NetVLAD, E2E-Spot).
2. **RQ2 Isolation (Native Video vs. Frame Sampling at Locked $W = 60\text{s}$):**
   - Run Gemini 3.8 Flash on the **locked macro-window duration $W = 60\text{s}$** comparing native `.mp4` container ingestion against extracted 1.0 FPS `.jpg` frames (strictly 60 frames) to isolate frame tokenization from window duration.
3. **RQ3 Decoupling (Spotting vs. Joint Spotting + Captioning):**
   - Control Run A: Spotting-Only Schema (`timestamp` + `label` only).
   - Control Run B: Joint Schema (`commentary` generated first as CoT before `label`).
4. **Temporal Ingestion (Windowing):**
   - 45-minute full-half single pass vs. 10-minute macro-chunks with border NMS.
5. **Modality (RQ4 Audio-Visual Grounding via Serverless Gemini 3.8 Flash):**
   - Video-only vs. Video + Audio (referee whistle and crowd dynamics) evaluated serverlessly via Vertex AI Batch Prediction without dedicated GPU VM spend.
6. **Quantization Impact (MiniCPM-V 4.6):**
   - BF16 vs. 4-bit NF4 on 17-class spotting and captioning quality across $W \in \{15\text{s}, 30\text{s}, 60\text{s}\}$.

---

### Phase 5: BigQuery Analytics, Dashboard & Reporting ($0–$10, Days 6–7)
- [ ] **BigQuery Tables:**
  - `soccernet_benchmark.predictions`
  - `soccernet_benchmark.evaluation_summary`
  - `soccernet_benchmark.ablation_experiments`
- [ ] **Looker Studio Dashboard:**
  - Interactive match timeline showing ground truth vs. model predictions.
  - Per-class PR curves and confusion matrices.
  - Side-by-side commentary comparison.
- [ ] **Final Report & README:**
  - Author comprehensive paper updating `README.md` and replacing the obsolete V1 retrospective with verified empirical results.
