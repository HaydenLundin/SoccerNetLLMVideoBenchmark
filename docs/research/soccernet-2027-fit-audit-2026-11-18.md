# SoccerNet 2027 fit audit (2026-11-18)

An AI-generated plan claimed v5.0.6 maps onto the SoccerNet 2027 challenges. This note records
(1) which of its external claims hold up against official pages and (2) how the repo measures up
against each target. Repo audits were read-only; file:line references are at commit 6eaf2e9.

## 1. External claims vs official sources

| Claim in the plan | Status | Evidence |
|---|---|---|
| Four 2027 tasks: GAR, Streaming VQA, Long Form GSR, Novel View Synthesis | Confirmed | soccer-net.org/challenges/2027; huggingface.co/SoccerNet |
| Deadline 2027-04-25 23:59 AoE | Confirmed (one deadline for all tasks) | same pages; sn-vqa-2027 RULES.md |
| `SoccerNet/SN-VQA-2027` gated HF dataset | Confirmed (SoccerNet NDA, manual approval) | huggingface.co/datasets/SoccerNet/SN-VQA-2027 |
| `github.com/SoccerNet/sn-vqa-2027` | Confirmed | repo + RULES.md |
| `github.com/SoccerNet/sn-gsr-2027` | **Not live (404)**; site marks GSR "Coming Soon" | soccer-net.org/challenges/2027 |
| `OpenSportsLab/SoccerNet-GAR` | Confirmed (64 WC2022 matches, tracking + video, CC-BY-NC-4.0) | HF dataset; arXiv 2511.12606 |
| QLoRA fine-tune on the official SN-VQA-2027 **training split** | **Refuted: no train split.** Test = 4 halves / 700 Q (labels public); challenge = 10 halves / 1,158 Q (hidden) | sn-vqa-2027 README |
| Strict causal streaming | Confirmed, and stricter: the cutoff also covers memory, caches, retrieval, tool calls, state updates | RULES.md |
| Answers are free-form | Multiple choice; submit an option ID or null | RULES.md |
| 2–4 page technical report | Report **required**; **no length stated** | RULES.md |
| Co-authorship on a CVPR 2027 CVSports summary paper | **Not found** on any official page | — |
| Permanent CodaBench phases ("Ends: Never") for Action Spotting and BAA | **Unverified.** soccer-net.org doesn't mention them; CodaBench sits behind an Anubis bot check we didn't pass | soccer-net.org/tasks/action-spotting; /challenges/2026 |
| 8 GB GPU is "genuinely competitive" | Unsupported. Remote LLM/VLM APIs are allowed. MatchLive baseline: Gemini 84.5 avg, local Qwen 27B 67.2 on test. Organizer sample predictions: 41.76% | RULES.md; github.com/Becomebright/MatchLive-baseline |

Other SN-VQA-2027 rules: visual only (no audio, ASR or commentary); no web search or retrieval;
pre-match metadata (rosters, kits, lineups, directions) allowed; metric = unweighted mean of six
subcategory accuracies; submissions to CodaBench competition 18329 as OSL JSON.

## 2. Repo vs each target

| Target | Verdict | Main gaps |
|---|---|---|
| SN-VQA-2027 | Best fit; still needs work | Window runner looks up to 60 s ahead (`scripts/run_local_benchmark.py:916`); VQA clips include frames after the question (`scripts/evaluate_v3_vqa.py:70`); clock overlay is clip-relative while questions use match time; no OSL JSON writer; Track C is 12 self-authored Qs on the base model. Our Stage 2 audio pre-gate is **not allowed** under the visual-only rule. |
| SN-GSR-2027 | Large effort; devkit not released | HSV-threshold detector with Man Utd/Arsenal kits hardcoded (`processing/pitch_geometry.py:537-605`); no ball detector; Track H OCR is a simulated literal; custom single-frame GS-HOTA; 8–12 hand-labelled frames |
| SoccerNet-GAR | Not realistic | No fixed-label clip classifier; tracking is a 650-frame evaluation script |
| Action Spotting (permanent) | Format ready, coverage not | Official `results_spotting.json` + devkit already in use; never run on the official test split (~140–300 GPU-h per config estimated); tight mAP about 0.06 |
| Ball Action Anticipation | Not a fit | Markov code rescores spotting output and doesn't forecast; Track E is scripted |
| Technical report | Trim: hours. Numbers: days–weeks | No result in the paper uses an official split; known caveats (MES-5 in-sample tuning, unconfirmed 2.95 s offset, scripted Tracks D–H) carry over |

## 3. Side findings (not challenge-specific)
- `paper/main.tex:99` figure shows MV-Foul 100% / BAS 1.0000 without "scripted" (body text discloses it).
- `CITATION.cff` still says version 5.0.2.
- LOMO 0.1613 / 0.0261 (`README.md:58`) has no committed JSON source.
- QLoRA adapter README is an unfilled template; dataset card says MIT, but SoccerNet-derived data is NDA/"other".
