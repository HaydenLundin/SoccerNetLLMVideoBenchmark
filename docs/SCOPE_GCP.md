# V2 Scope: Modernizing the SoccerNet VLM Benchmark on GCP

*Drafted 2026-10-09. Prices were checked against official GCP pages that day. Anything marked (unverified) has to be re-checked before spending money on it.*

---

## 1. What V1 actually delivered

Sources: `capstone_archive/reports/*`, `docs/RETROSPECTIVE_REPORT.md`, code, results.

| Claimed | Reality |
|---|---|
| "15+ models benchmarked" | **One model produced usable output** (Qwen2.5-VL-7B, 4-bit, 21 halves). Phi-3.5 files are empty `[]`. Pixtral, Llama and Phi-4 only ran in Colab smoke tests. |
| Evaluation vs ground truth | F1 = 0.075 against a 0.041 random baseline. **No tolerance window, matching rule or per-class P/R was ever defined.** |
| mAP table in the retrospective (Qwen-32B 0.78, InternVideo2, VideoLLaMA-2) | **Not backed by any code or run.** InternVideo2 and VideoLLaMA-2 never appear in the repo. Do not carry this table forward. |
| Deployment (Project 4) | A reference architecture on paper only. Nothing was deployed. |

**Root-cause problems V2 must fix:**

1. **Predictions can't be joined to labels.** Workers name output `video_{index}`, where `index` is unsorted `os.walk` order over game folders that each hold two halves. The game ID and half are never recorded. **V1 predictions can't be re-scored**, so V2 re-runs everything with game ID and half in every record.
2. **No defined metric.** V2 uses the official SoccerNet action-spotting metric (tight/loose Average-mAP via `SoccerNet.Evaluation`), plus easier-to-read precision/recall/F1 at ±5 s, ±10 s and ±30 s tolerances, per class.
3. **The output format was not enforced.** The model wrote JSON wrapped in markdown and invented labels ("Set Pieces", "Premier League Logo"). V2 uses **structured output / response schemas** with an enum over the 17 Labels-v2 classes, so invalid labels become impossible rather than something to clean up afterwards.
4. **Overlapping windows double-count.** A 3 s window with a 2 s step plus no non-max suppression produced about 950 events per half against about 85 real ones. V2 uses NMS per class at a fixed window.
5. **The NDA password was hardcoded** in `processing/download_batch.py` and `colab/video_download.py`, in a **public** repo. *Fixed:* both now read `SOCCERNET_NDA_PASSWORD` (from `.env` for HPC/local, or a Colab secret). It is still in git history (commits `bc91ef2`, `90fff28`, `7711221`, `d5361b7`); whether to rewrite history is still open.

---

## 2. GCP budget constraints that shape everything

- **The free trial gives $300 for 90 days, whichever runs out first.** It can't be paused.
- **GPUs cannot be attached to VMs on a trial billing account, and quota increases are blocked.** This is the same wall V1 hit (`NVIDIA_T4_GPUS limit 0`).
- To run open-weight models on GPUs you must **upgrade to a paid account**. Remaining credit carries over for the rest of the 90 days. *(Unverified: whether trial credit is applied to GPU SKUs after upgrade. Confirm on the billing page before running GPU jobs.)*
- **AI Studio Gemini API usage is NOT payable with trial credit.** Use **Vertex AI** (`google-genai` SDK with `vertexai=True`) so the spend comes out of credit.
- **TPUs are not on the trial's blocked list** (only GPUs are named). Documented default Cloud TPU quotas are nonzero (v5e/v6e: 512 on-demand / 1536 preemptible cores per zone), but trial projects may be throttled ("access may be limited to prevent abuse"). **Unverified until checked** on the new project's Quotas page. Running open VLMs on TPU means the vLLM TPU backend or JAX. That is less mature than CUDA for multimodal models, so treat it as a stretch experiment, not the main path.
- The trial is for new customers only. If this Google account already used a trial, credit requires a different account.

### Unit costs (us-central1)

**Gemini on Vertex, input cost per 45-min half.** A half is 2,700 s of video plus audio, which comes to roughly 783k tokens at default media resolution or 265k at low. Batch prediction is 50% off.

| Model | Low-res, batch | Default-res, batch | Default-res, standard |
|---|---|---|---|
| Gemini 2.5 Flash-Lite | $0.02 | $0.05 | $0.10 |
| Gemini 2.5 Flash | $0.07 | $0.15 | $0.30 |
| Gemini 3.5 Flash-Lite | $0.04 | $0.12 | $0.24 |
| Gemini 3.8 Flash (intro price to 2026-12-31) | $0.10 | $0.29 | $0.59 |
| Gemini 3.5 Flash | $0.20 | $0.59 | $1.18 |
| Gemini 2.5 Pro | $0.33 | $0.98 | $1.96 |
| Gemini 3.1 Pro (preview) | $0.53 | $1.57 | $3.13 |

Add output cost on top. With thinking enabled, budget about 20-50k output tokens per half, which is $0.04-0.30 depending on the model. *Token rates for the 3.x models are taken from the 2.5-era docs. Run `count_tokens` on one half before the full run.*

**Self-hosted open models (paid account required):**

| Machine | On-demand | Spot | Fits |
|---|---|---|---|
| g2-standard-8 + L4 24 GB | $0.85/h | $0.49/h | Qwen3-VL-8B / Qwen2.5-VL-7B (bf16), InternVL3-8B, MiniCPM-V |
| a2-ultragpu-1g, A100 80 GB | $5.07/h | $3.04/h | Qwen3-VL-32B or a 30B-A3B MoE (bf16) |
| Cloud Run GPU (L4) | $0.67/h + CPU/mem, per-second | — | Good for a scale-to-zero demo endpoint |

Storage: 40 halves at 720p is about 60 GB, which costs about $1.20/month in GCS Standard. Google Cloud Batch has no fee beyond the VMs it runs.

---

## 3. Recommended scope (fits $300 with ~25% reserve)

**Eval set:** 20 EPL games = **40 halves** from the SoccerNet **test** split, which has public Labels-v2. The game list is frozen in `eval_set.json`.

| Phase | Work | GCP services learned | Est. cost |
|---|---|---|---|
| **0. Foundation** | New project, **budget alerts at 25/50/75/90%**, service account, GCS bucket (private, NDA data), upload 40 halves, `count_tokens` smoke test on 1 half | IAM, Billing Budgets, GCS, Vertex AI | **< $5** |
| **1. Ground-truth scorer** | Rebuild evaluation first: Labels-v2 parser, official Average-mAP, ±tolerance P/R/F1, per-class report. Unit-tested against a fake "perfect" predictor and a random one. | (local or Cloud Run Job) | **$0** |
| **2. Gemini benchmark** | 5 models (2.5 Flash-Lite, 2.5 Flash, 3.5 Flash, 3.8 Flash, 3.1 Pro). Low-res batch on all 40 halves, plus default-res on Flash-tier models. Native video in 5-min clips, schema-enforced output. Results land in **BigQuery**. | Vertex AI Batch Prediction, BigQuery | **$60-90** |
| **3. Open-weight comparison** *(needs paid upgrade)* | Qwen2.5-VL-7B (direct V1 continuity) and Qwen3-VL-8B on L4 Spot via vLLM, same 40 halves, same scorer. Optional: Qwen3-VL-32B on A100 Spot for 10 halves. | Compute Engine Spot, Artifact Registry (Docker), Cloud Batch | **$25-70** |
| **4. Ablations** | On one cheap model over 10 halves: fps (0.5/1/2), resolution (low/default), clip length (1/5/15 min), with/without audio. This is the honest version of V1's claimed "FPS sweep". | Vertex AI | **$15-30** |
| **5. Report + demo** | New write-up, a results dashboard (BigQuery → Looker Studio or a static page), and an optional Cloud Run endpoint that annotates an uploaded clip | Cloud Run, Looker Studio | **$0-10** |
| **Reserve** | Re-runs, debugging, prompt iteration (standard, non-batch calls) | | **~$75** |
| **Total** | | | **~$180-280** |

### What does *not* fit

- Running all ~104 EPL games (about 200 halves). That is 5x the eval cost, roughly $300+ for Gemini alone.
- Default-resolution Pro models across the full eval set, which comes to about $130 per model.
- Any LoRA fine-tuning. That needs an A100 for 10+ hours plus data prep; it would be its own project.
- An H100 or 8-GPU nodes.
- Third-party models (Claude or Llama via Vertex Model Garden MaaS). Trial credit doesn't cover partner models.

### If you stay on the trial (no upgrade)

Drop Phase 3. Phases 0-2 and 4-5 come to about **$80-130**. That leaves room to add Gemini 2.5 Pro and to expand to 60 halves. The trade-off is that you lose the "open vs proprietary" comparison, which is the bridge back to V1.

---

## 4. Guardrails (non-negotiable)

1. **Budget alert plus a hard stop.** Set up a Pub/Sub budget alert → Cloud Run function that disables billing at 95%. That also teaches you the pattern.
2. Run every expensive job on **1 half first**, check cost in Billing → Reports, then fan out.
3. Spot VMs run inside Cloud Batch with checkpointing, so preemption only loses the current clip. V1's `submit_restart.sh` idea becomes a built-in.
4. Never make the GCS bucket public (SoccerNet NDA). No videos in git.
5. Tear down idle endpoints. Model Garden and Vertex endpoints bill per node-hour even when idle.
