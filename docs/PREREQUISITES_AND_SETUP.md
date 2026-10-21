# SoccerNet VLM Modernization Benchmark (V2) — Prerequisites & Setup Checklist

This document details all account requirements, environment configurations, quota allocations, and software dependencies required before executing Phase 0.

---

## 1. Antigravity IDE & Environment Health
- [x] **Pre-Tool Hook Fix:** The pre-tool hook quote resolution bug in `plugin.json` has been resolved; terminal command executions and file write tools are fully operational.
- [x] **Repository Files Stored:**
  - `eval_set.json` (frozen 20 EPL matches = 40 halves)
  - `processing/freeze_eval_set.py`
  - `processing/unified_schema.py`
  - `docs/IMPLEMENTATION_PLAN.md`
  - `docs/RESEARCH.md`

---

## 2. Google Cloud Platform (GCP) & Vertex AI Configuration

### A. Project & Billing Setup
- [ ] **GCP Project:** Verify that your active Google Cloud project has the $300 free trial billing account linked.
- [ ] **Enable Core APIs:**
  Execute via terminal:
  ```bash
  gcloud services enable aiplatform.googleapis.com storage.googleapis.com bigquery.googleapis.com
  ```
- [ ] **Application Default Credentials (ADC):**
  Authenticate your local workstation to access GCP:
  ```bash
  gcloud auth application-default login
  ```

### B. Vertex AI Quota Ceilings (Overrun Prevention)
- [ ] Navigate to **Google Cloud Console -> IAM & Admin -> Quotas**.
- [ ] Filter by Service: **Vertex AI API**.
- [ ] Set a hard ceiling on:
  - `Online prediction requests per base model per minute`
  - `Batch prediction requests per region`
- Setting these prevents unbounded spending if a loop or script runs unexpectedly.

### C. Cloud Storage (Private Bucket)
- [ ] Create the regional private bucket in `us-central1` (matching Vertex AI model endpoints):
  ```bash
  gcloud storage buckets create gs://<YOUR_PROJECT_ID>-soccernet-raw --location=us-central1
  ```
- [ ] **Compliance Guardrail:** Ensure Public Access Prevention is enforced on this bucket to comply with the SoccerNet NDA.

### D. Service Account Permissions
- [ ] Verify that your default compute service account (`<project-number>-compute@developer.gserviceaccount.com`) or custom service account has the following IAM roles:
  - `roles/aiplatform.user`
  - `roles/storage.objectAdmin`
  - `roles/bigquery.admin`

---

## 3. SoccerNet Dataset & Video Pre-Processing Tools

### A. Credentials & Environment File
- [ ] Verify that `.env` contains your approved SoccerNet NDA credentials:
  ```bash
  SOCCERNET_NDA_PASSWORD="your_actual_password_here"
  ```
- [ ] Ensure `.env` is listed in `.gitignore` (verified).

### B. FFmpeg Installation
- [ ] Ensure `ffmpeg` and `ffprobe` are installed and reachable on system PATH:
  ```powershell
  ffmpeg -version
  ```
  *(Used for stream-copying `.mkv` to `.mp4` and segmenting 10-minute macro-chunks without lossy re-encoding).*

### C. Local Storage Capacity
- [ ] Verify at least **60 GB of free space** on your local SSD:
  - 20 EPL matches (40 halves) at 720p: ~35 GB.
  - Sliced 10-minute `.mp4` chunks (H.264/AAC): ~35 GB temporary working directory.

---

## 4. Local Workstation Setup (RTX 3000 Ada 8GB)

### A. Python & Tooling
- [ ] Python 3.12+ 64-bit installed.
- [ ] Ensure `uv` package manager is available:
  ```powershell
  uv --version
  ```
- [ ] Python packages installed in virtual environment:
  ```bash
  uv pip install soccernet torch torchvision transformers accelerate bitsandbytes google-genai pydantic tqdm
  ```

### B. Ollama Vision Models (Local Phase 3)
- [ ] Confirm Ollama is installed and running (`ollama --version`).
- [ ] Pull vision-capable models (avoiding text-only models):
  ```bash
  ollama pull qwen2.5vl:7b
  ollama pull minicpm-v
  ```

---

## 5. Verification Checklist Before Triggering Phase 0

1. [ ] Run `python -c "import google.genai; print(google.genai.__version__)"` to verify Vertex AI SDK.
2. [ ] Submit the Phase 0 2-row JSONL test batch to Vertex AI to confirm GCS read/write and batch completion.
3. [ ] Confirm that `eval_set.json` is present in the repository root.
