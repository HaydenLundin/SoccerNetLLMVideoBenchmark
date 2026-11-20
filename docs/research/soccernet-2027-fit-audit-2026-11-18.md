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

## 5. Counting event ledger, and challenge-match overlap with our own data (2026-11-18)

**Ledger** (`processing/event_ledger.py`, tests in `scripts/test_event_ledger.py`):
- **Question parsing:** every counting question follows one template,
  "{As of now | Over the whole match so far | In/Over the past N minutes}, how many {events} has {subject}
  {verb}?". The parser covers 84/84 test and 150/150 challenge questions.
- **Subjects:** a team in 31/84 test questions and 67/150 challenge questions; otherwise a player.
- **Event kinds:** shots dominate. The challenge adds "been involved in" and cards, which test does not have.
- **Scopes:** 51 of 84 test questions are cumulative and 33 are 10- or 20-minute windows. 63 of 84 are asked
  in the second half, so the ledger is keyed by `match_key` and spans both halves.
- **Causality:** each event stores when perception *observed* it. A detector that looks ahead (e.g. our
  60 s windows) stays causal by timestamping its events at the window end.
- **Untested assumptions:**
  - shots = on + off target;
  - free kicks = direct + indirect;
  - "past N minutes" uses a match clock with first-half stoppage time clamped to 45:00;
  - "as of now" = "whole match so far".
  These need event-level ground truth to check.

**Challenge matches overlap with data we already used.** The team options of each sample's `event_team`
questions identify the five hidden challenge matches as Chelsea v Man City, Chelsea v West Brom, Chelsea v
Newcastle, Chelsea v West Ham, and Arsenal v Tottenham. Our `eval_set.json` contains 2015-16 fixtures of
the first four:
- `2015-08-16 Man City 3-0 Chelsea` (game 1, MES-5)
- `2015-08-23 West Brom 2-3 Chelsea` (game 2, MES-5)
- `2015-09-26 Newcastle 2-2 Chelsea` (game 5)
- `2015-10-24 West Ham 2-1 Chelsea` (game 7)

For games 1 and 2 the identity is confirmed. Corner times in the challenge questions and options fall within
5 s of our SoccerNet-v2 corner labels 4/10 and 8/16 times, against about 0.2 and 0.9 expected by chance.
Games 5 and 7 are only matched by team and season so far.

Consequences under the "no external match information" rule:
- **QLoRA adapter:** trained on games 4 and 7 (`dataset_manifest.json`: 128 + 139 samples). If game 7 is the
  challenge West Ham match, the adapter must not be used for a challenge submission.
- **MES-5 routing and rescoring:** tuned on games 1 and 2. They can't be used either.
- **M15 prior:** excludes the MES-5 games but may include games 5 and 7. Check before use.
- **The ledger:** must never be filled from SoccerNet-v2 labels for these games.

Enforced in code (2026-11-19): `check_challenge_safe_model` in `processing/sn_vqa_2027.py` refuses the QLoRA adapter
and any local PEFT adapter, and `scripts/evaluate_sn_vqa_2027_text_only.py` calls it before loading a model.
`scripts/test_sn_vqa_2027.py` (CI) fails if any SN-VQA-2027 file imports QLoRA, MES-5, MECE-routing or Markov-prior
code, directly or through other repo modules.

Test-split matches are Chelsea v Manchester United and Chelsea v Liverpool. `eval_set.json` contains
`2015-12-28 Man Utd 0-0 Chelsea`; identity unconfirmed.

## 6. Oracle checks for the counting ledger (2026-11-19)

**Test matches identified.** I downloaded SoccerNet-v2 `Labels-v2.json` for five candidate fixtures and matched them
to the test answers to "Which team is taking the … now?". Diagnostic use only; these are labels, never submission inputs.
- `game1` = **2016-02-07 Chelsea 1-1 Manchester United**: 45/60 question times within 5 s of a label, 41 with the right team.
- `game2` = **2016-09-16 Chelsea 1-2 Liverpool**: 52/58 within 5 s, 51 with the right team.
- Other candidates matched 16–27 times and 3–8 teams. Neither test match is in `eval_set.json`, so the test split is clean.

**SoccerNet-v2 events as an oracle: 21/31 team counting questions.** I filled the ledger with SoccerNet-v2 events and
team labels and asked only the team-subject questions.
- The misses are coverage, not timing. VQA questions come about 2–3 s after the SoccerNet-v2 label, and looking up to
  60 s ahead changes nothing.
- The VQA set has its own event annotations. Game 1 has 19 corners against 17 in SoccerNet-v2.
- A perception model trained to SoccerNet-v2 definitions is therefore capped well below 100% on counting.
- Rule variants: counting goals as shots gives 22/31; on-target-only gives 18; direct free kicks only gives 18.

**The question stream defines the countable events: 84/84.** Each event question ("Which team / Who is taking the
<kind> now?") is asked at the moment an event happens. A ledger built only from earlier event questions plus their
correct answers reproduces 74/84 counting answers. With goals also counted as shots, it reproduces **84/84** (team
31/31, player 53/53). Counting is bookkeeping over one's own answers to the "now" questions.

Sensitivity: I replaced each "now" answer with a random wrong option with probability 1 − p (50 draws each):

| "now"-question accuracy p | 1.00 | 0.97 | 0.90 | 0.80 | 0.70 | 0.50 | 0.33 |
|---|---|---|---|---|---|---|---|
| counting accuracy (mean %) | 100 | 92.7 | 79.7 | 70.5 | 60.5 | 51.6 | 43.8 |

MatchLive's Gemini run scores 96.6/96.9 on event_team/event_player but 58.3 on counting. At p ≈ 0.97, bookkeeping
predicts about 93%.

**Caveats:**
1. **Rules:** this uses earlier questions as inputs, which the rules' list of permitted inputs does not include.
   See section 7. Don't rely on it for a submission without a written answer from the organizers.
2. **The challenge split differs:**
   - 83 of 150 counting questions use "been involved in" (shots, offsides, goals), whose meaning (main actor only,
     or assists too) is unknown.
   - Challenge substitution events come only as "Who is being substituted in/out now?" with no team question, so the
     4 substitution-count questions need the player's team, e.g. from the permitted rosters.
   - 100% on test does not transfer automatically.
3. **The error model is simple:** uniform random wrong answers. Real errors are correlated (e.g. confusing two
   teammates).

Code: `ledgers_from_now_questions` and the goal-as-shot rule in `processing/event_ledger.py`. `scripts/test_event_ledger.py
--official-dir` asserts the 84/84 oracle.

## 7. Permitted inputs: what the rules say (2026-11-19)

Sources: RULES.md and README of github.com/SoccerNet/sn-vqa-2027, and the code of github.com/Becomebright/MatchLive-baseline.
The HF dataset card (gated), CodaBench and Discord were not read.

**The list of permitted inputs is closed, and it does not include earlier questions.** RULES.md: at question time `t`
"a method may use only the match video observed up to and including `t`, the current question and answer options,
the question timestamp, and permitted match metadata". The cutoff covers "memory, cached features, retrieved frames,
tool calls, and state or model updates", and "Predictions must use only the permitted inputs listed above."
- Earlier questions, previous answers and question history are not mentioned anywhere in the repo.
- Read strictly, "the current question" excludes earlier ones. That rules out the counting ledger built from the
  question stream, and also our `--mode history` text-only run (a test-split diagnostic, never a submission).
- A lenient reading says earlier questions sit inside the cutoff. Nothing in the text supports that over the strict one.

**The official baseline uses no question history.** MatchLive's `scripts/run_qa.py` says the model sees "the pre-match
facts, the question and its options, and nothing else from the question file". It answers questions independently in a
thread pool. Its video memory is built from frames only. So the baseline neither uses nor tests the practice.

**Delivery.** The whole question file (`challenge_public.json`, 1,158 questions) is handed out up front and scored as
one JSON file. Nothing technical stops a method from reading other questions. Compliance is judged from the required
technical report: "rule compliance is assessed from the report and method review."

**Pre-match metadata** is limited to "team names, kit colors, rosters and jersey numbers, initial lineups and positions,
and playing directions once they have been established from the broadcast". Rosters cover the substitution-team gap
(section 6, caveat 2). The baseline's `data/prematch/` files key players by team.

**Where to ask:** the README points to the SoccerNet Discord (discord.gg/cPbqf2mAwF) for "rule clarifications". GitHub
issues are enabled on the repo; it has no issues or discussions yet.

**Consequence:** treat the ledger as video-fed only. Perception (a spotting model or VLM watching frames up to `t`)
appends events, and counting reads them. That is permitted under either reading. Filling it from the question stream
stays a diagnostic until the organizers answer.
