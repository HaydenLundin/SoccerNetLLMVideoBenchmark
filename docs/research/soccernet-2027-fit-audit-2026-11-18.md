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

## 4. Text-only baseline on the SN-VQA-2027 test split (2026-11-18)

Script: `scripts/evaluate_sn_vqa_2027_text_only.py` (Qwen3.5-4B, NF4, no video; the answer is
the argmax next-token logit over the option letters, so every prediction is valid and deterministic).
Logs: `logs/sn_vqa_2027/text_only_question/`, `logs/sn_vqa_2027/text_only_history_smoke/`.

Subcategory accuracy (%), score = unweighted mean of the six:

| System | counting | recall time | recall player | event team | event player | causal | **Mean** |
|---|---|---|---|---|---|---|---|
| n (test) | 84 | 200 | 129 | 118 | 130 | 39 | 700 |
| Chance (2- or 4-way) | 25.0 | 25.0 | 25.0 | 50.0 | 25.0 | 25.0 | **29.2** |
| Ours: Qwen3.5-4B text-only | 31.0 | 28.5 | 28.7 | 50.0 | 33.1 | 35.9 | **34.5** |
| GPT-5.5 text-only (MatchLive README, not run by us) | 36.9 | 26.0 | 45.0 | 48.3 | 43.1 | 51.3 | **41.8** |
| Organizer sample predictions | | | | | | | **41.8** |
| MatchLive, local Qwen3.8-27B (README) | 42.9 | 70.5 | 79.1 | 84.7 | 72.3 | 53.8 | **67.2** |
| MatchLive, gemini-3.8-flash (README) | 58.3 | 82.0 | 96.1 | 96.6 | 96.9 | 76.9 | **84.5** |

Findings:
- **Answer positions are perfectly balanced** in every subcategory, so a constant option scores chance.
- **The 4B text-only model is only slightly above chance:** 236/700 vs 204.5 expected, z = 2.7. No single
  subcategory is clearly above chance except event_player (z = 2.1). Team questions are exactly 50%.
- **Text-only does almost nothing for event timing.** recall_event_time (200 Qs, the largest group) is at
  chance for both us (28.5) and GPT-5.5 (26.0). That group needs video plus memory.
- **Counting is the weak spot even for the best published system** (58.3 for Gemini MatchLive, vs 82–97
  elsewhere). With equal subcategory weights it is the highest-headroom target.
- **Earlier questions don't measurably leak answers at 4B scale.** On the first 120 questions (one half),
  adding every earlier question as context: 49 vs 45 correct, 13 vs 9 discordant, exact McNemar p = 0.52.
  This needs a stronger model before it can be ruled out.
- **Speed:** without `flash-linear-attention` / `causal_conv1d`, Qwen3.5's linear-attention layers fall back
  to PyTorch. Question-only prompts (≤197 tokens) take 0.7 s each; history prompts (≤6k tokens) take ~45 s.
