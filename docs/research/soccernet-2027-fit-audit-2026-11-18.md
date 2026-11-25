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
- **Speed:** question-only prompts (≤197 tokens) take 0.7 s each; history prompts (≤6k tokens) take ~45 s.
  (Corrected 2026-11-24: this was first blamed on the missing `flash-linear-attention` / `causal_conv1d` kernels. Profiling
  showed the main cost is the 8 full-attention layers. On this torch 2.6 Windows wheel, transformers' GQA path for SDPA falls
  back to the O(n²) math kernel; see section 11.)

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
  - "past N minutes" uses video time within a half, and match clock across the break with first-half stoppage
    clamped to 45:00 (changed 2026-11-24; before, stoppage was clamped within the first half too);
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

**Consequence (see section 8):** treat the ledger as video-fed only. Perception (a spotting model or VLM watching frames up to `t`)
appends events, and counting reads them. That is permitted under either reading. Filling it from the question stream
stays a diagnostic until the organizers answer.

## 8. Where MatchLive loses counting points (2026-11-20)

**How MatchLive builds memory** (read from github.com/Becomebright/MatchLive-baseline; nothing run):
- Perception runs on every frame: RF-DETR detection, BoT-SORT tracking, a jersey-number classifier and kit colours.
- A VLM then reads 30 s windows (5 s overlap) at 2 fps, about 60 frames of 960x540 per call. It returns typed events: shot, goal, card, foul, substitution, penalty, free kick, offside, corner, injury.
- Rules merge repeat sightings. An LLM arbiter settles conflicts.
- Each event enters an append-only log at `t_known` = window end + API latency. Questions see only `known_by(now)`, so memory lags events by 30–90 s.
- The QA agent is generic: up to 4 rounds of tools (event list, state, frames up to `now`). It is the same for every subcategory, and counting is left to the LLM reading event counts.
- Per subcategory, Gemini scores counting 58.3, recall_event_time 82.0, recall_event_player 96.1, event_team 96.6, event_player 96.9, causal 76.9.

**Lag is not the main loss: 30–90 s of lag costs 11–14 points at most.** I rebuilt the ledger from the test "now" questions with correct answers (a diagnostic, test split only) and delayed every event's observation by L seconds:

| lag L (s) | 0 | 5 | 10 | 20–60 | 90 | 120 |
|---|---|---|---|---|---|---|
| counting accuracy | 100 | 92.9 | 91.7 | 89.3 | 85.7 | 82.1 |

- The first 5 s cost 6 questions. These are asked seconds after an event the answer already includes.
- The latest relevant event (any subject) falls within 10 s of the question for 15/84 counting questions, and within 90 s for 32/84. The median gap is 124 s.

**So the 58.3 comes mostly from event quality.** MatchLive answers "now" questions at about 97% when asked directly, but its window analyzer's ledger counts at 58.3. With the lag above and 97%-accurate events, bookkeeping would predict about 80–90% (section 6 table).

**Levers, in order:**
1. Event recall and precision under the VQA's own event definitions, with replays de-duplicated.
2. Deterministic counting over the ledger instead of LLM counting.
3. A short-lag path: check the last few seconds of video when a count question arrives, to catch the event that just happened.

**Inputs we now have:** the official pre-match rosters (`data/prematch/test.json` and `challenge.json` in the MatchLive repo) list each team's players with shirt numbers, starters and bench. That covers team attribution for substitutions.

**Blocked:** the SN-VQA-2027 videos (test: 4 halves, 3.4 GB; challenge: 10 halves, 9.8 GB) are in the gated `SoccerNet/SN-VQA-2027` dataset. This machine has no Hugging Face login with access (HTTP 401), and none of the local SoccerNet videos are test matches.

## 9. Local video runner, stage 1 (2026-11-20)

Decision: event detection and QA run on local Qwen3.5-4B only (NF4, 8 GB GPU); no paid API.

`scripts/evaluate_sn_vqa_2027_video.py` answers each question from the video alone, with no memory across questions:
- `question_window` (in `processing/sn_vqa_2027.py`) picks the span to show, always ending at or before the question time. "Now" questions get the last 16 s. Questions naming a time get that time ±10 s, including recall_event_player ("the corner at first half, 03:22"), match clock ("around 88:53") and "the 12th minute". recall_event_time gets the span of its option times. Other questions get the last 60 s.
- On the test split the window covers 129/129 recall_event_player, 200/200 recall_event_time and 32/39 causal questions with a named time. On the challenge split it covers 215/215, 493/493 and 1/50; challenge causal questions are asked right after the play, so they get the 60 s span.
- Frames: PyAV seeks to the span and keeps at most 5 frames/s, then `causal_frame_window` takes 8, at 448x252 and labelled with the match clock. On a local non-challenge video (Tottenham v Liverpool 2016-08-27), 8 frames decode in 0.4–1.4 s.
- Player options are labelled with team and shirt number from the official pre-match roster (`annotate_option`); this covers 100% of player options on both splits. Kits and playing directions go in the prompt.
- Answers are picked by the next-token logit over the option letters, as in the text-only run. Smoke run: 2.0 s per question at about 1,200 prompt tokens, so the test split takes about 25 min.

Not yet measured: no test videos (section 8, Blocked).

Next levers, after the first test run:
- **Goals from the scorebug.** The broadcast overlay shows the score and clock in every frame. Goal counts can be read from the latest frame.
- **A sharper last frame for player questions.** Jersey numbers are unreadable at 448 px.
- **Counting from a video-fed ledger.** Run event detection over windows up to `t` with the same model, then count with code (sections 6 and 8).

## 10. First video run on the test split (2026-11-23)

**Videos.** HF access to `SoccerNet/SN-VQA-2027` is still refused for our account, so this run uses SoccerNet's own 720p halves of the two test matches:

| Test match | SoccerNet game |
|---|---|
| game1 | `england_epl/2015-2016/2016-02-07 - 19-00 Chelsea 1 - 1 Manchester United` (train split) |
| game2 | `england_epl/2016-2017/2016-09-16 - 22-00 Chelsea 1 - 2 Liverpool` (train split) |

How we know they match the SN-VQA files:
- **Lineups and goals.** The pre-match lineups and the goal-count answers identify the games.
- **Same video, audio removed.** Our copies are not byte-identical to the SN-VQA files. In all four halves, our Vorbis audio track accounts for 96.8% of the size gap, and the rest is consistent with that track's container overhead.
- **Time alignment.** All five goal times named in the questions show the goal within about 2 s in our copies.
- **No leakage.** Neither game appears in `eval_set.json`, `data/` or `hf_dataset/`, so nothing we trained or tuned has seen them.

The halves are hard-linked into `data/sn_vqa_2027/video-test/` in the main checkout. A challenge submission must still use the official files.

**Result.** The defaults are 8 frames, 448 px width, 16 s/60 s windows, Qwen3.5-4B NF4. The run took 2.15 s per question, 25 min in total, with at most 1,250 prompt tokens. Logs are in `logs/sn_vqa_2027/video_v1/`.

| Subcategory | n | Text-only | Video | Paired (video-only right / text-only right), McNemar p |
|---|---|---|---|---|
| counting | 84 | 30.95 | 25.00 | 6 / 11, p = 0.33 |
| recall_event_time | 200 | 28.50 | 36.50 | 57 / 41, p = 0.13 |
| recall_event_player | 129 | 28.68 | 52.71 | 53 / 22, p < 0.001 |
| event_team | 118 | 50.00 | 67.80 | 22 / 1, p < 0.001 |
| event_player | 130 | 33.08 | 46.92 | 36 / 18, p = 0.02 |
| causal reasoning | 39 | 35.90 | 28.21 | 5 / 8, p = 0.58 |
| **Subcategory mean** | 700 | **34.52** | **42.86** | +8.3 points, bootstrap 95% CI +3.5 to +13.3 |

For reference, MatchLive reports 67.2 with Qwen3.8-27B and 84.5 with Gemini.

What this says:
- **Video helps where one look at the right moment answers the question.** The gains are in event_team and recall_event_player, which stay significant after a Holm correction for the six subcategories. event_player's gain (p = 0.02) does not survive the correction (its threshold is 0.0125), so treat it as likely rather than shown.
- **Counting and causal show no measurable change.** Counting needs the whole half, and an 8-frame window cannot count. The causal sample (39) is too small to read.
- **recall_event_time's gain is not significant (p = 0.13).** The option span is about 50 s (median 51 s, p10–p90 47–53 s), so 8 frames are about 7 s apart and can miss a short event.

The next levers in section 9 still apply. They now have measured targets:
- **Counting:** the scorebug for goals, then a video-fed ledger for everything else.
- **recall_event_time:** more frames or a denser scan of the option span.

## 11. Speed fix, review fixes, and event detection on all four test halves (2026-11-24)

### Speed: the attention fallback, not the linear-attention kernels

transformers warns that `flash-linear-attention` and `causal_conv1d` are missing, but those layers weren't the bottleneck. I timed one 12-frame, 896 px window (5,462 tokens):
- The 24 linear-attention layers took about 0.1 s each.
- The 8 full-attention layers took 1.3 s each and added 4.6 GiB. Peak memory reached 8.38 GiB on an 8 GiB card, so Windows spilled to shared memory.

**Cause:** transformers calls SDPA with `enable_gqa=True` (16 query heads, 4 KV heads). The torch 2.6 Windows wheel has no FlashAttention, and its memory-efficient kernel rejects mismatched head counts, so SDPA falls back to the O(n²) math kernel.

Reproduced standalone at the model's shapes:

| Call | Kernel | Time | Extra memory |
|---|---|---|---|
| `enable_gqa=True` | math | 0.15 s | 4.5 GiB |
| KV heads expanded first | memory-efficient | 0.012 s | 0.13 GiB |

**Fix:** the video runner and the detector set `transformers.integrations.sdpa_attention.use_gqa_in_sdpa` to return False, so KV heads are expanded first. I also installed `triton-windows` 3.2 and `flash-linear-attention` 0.6, which matches the reference to 4 decimals.

| Setup | Time per window | Peak memory |
|---|---|---|
| Stock | 13.7 s | 8.38 GiB |
| fla only | 9.2 s | 8.38 GiB |
| Patch only | 3.85 s | 4.97 GiB |
| Patch + fla | **2.9 s** | 4.39 GiB |

The test QA run dropped from 2.15 to 1.1 s per question.

### Review of sections 4–10

Two independent reviews (pipeline; ledger and detector scoring) found no causality leak:
- All 700 logged windows end at or before their question time, and none comes from a later half.
- Every metric in sections 4 and 10 reproduces from the raw predictions and `test_GT.json`.

Bugs found and fixed (commits `77ab9f2`, `316ed1e`):
- **Second-half stoppage times were shifted by 45 minutes.** "Second half, 45:45" is counted from the start of the half, not the match clock; in both splits every such time precedes its question. 17 test questions (13 recall_event_time, 4 recall_event_player) were shown the wrong span of video, for example 0:35–1:25 instead of 45:35–46:25.
- **The prompt stated the goal directions for the question's half** even when the frames came from the earlier half (8 test questions).
- **The ledger put both halves on one seconds axis.** First-half stoppage past 2700 s therefore sorted after early second-half play. No current question triggers this, but a video-fed ledger would. It now orders events by (half, sec).
- **Smaller fixes:** text-only scoring crashed on partial runs, and the model guard missed names like `MES-5` or `*-sft-merged`.

**Re-run** (`logs/sn_vqa_2027/video_v2/`, same settings as `video_v1`, with fixes and new kernels):

| Subcategory | v1 | v2 |
|---|---|---|
| counting | 25.00 | 25.00 |
| recall_event_time | 36.50 | 37.00 |
| recall_event_player | 52.71 | 54.26 |
| event_team | 67.80 | 68.64 |
| event_player | 46.92 | 49.23 |
| causal reasoning | 28.21 | 25.64 |
| **Mean** | **42.86** | **43.30** |

The change is noise:

| Questions | n | v1 correct | v2 correct | Answers changed |
|---|---|---|---|---|
| Window moved (stoppage fix) | 17 | 8 | 8 | 8 (+3/−3) |
| Frames from an earlier half (direction fix) | 8 | 0 | 0 | 0 |
| Unchanged window (kernel change only) | 675 | 306 | 312 | 28 (+12/−6, p = 0.24) |

- The right video didn't help the 17 stoppage questions. Most are recall_event_time, where 8 frames about 7 s apart can't separate options 10 s apart.
- **Kernel rounding alone changes about 4% of answers.** Differences under about 1 point between runs on different kernels are not evidence.

### Event detection on all four test halves

Code: `scripts/detect_events_sn_vqa_2027.py` (GPU), `scripts/score_events_sn_vqa_2027.py`, and `processing/sn_vqa_2027_events.py`. Window outputs are in `logs/sn_vqa_2027/events/`. `scripts/test_sn_vqa_2027_events.py --official-dir` reproduces the numbers below.

**Detectors.** Both run with Qwen3.5-4B, scoring each window by letter logits. They are scored against the moments of the test "now" questions, which is diagnostic only.
- **v2:** a 12-class prompt (with decoys such as cross, close-up and throw-in) on 6 s windows at 448 px and 2 fps, about 7–10 min per half.
- **shot_bin:** a yes/no shot question on 12 s windows at 896 px and 1 fps, about 15–20 min per half.

**Strict scoring** replaces the looser rule used on game1 H1 earlier:
- A window must contain the event moment, taken as the question time minus 2.5 s.
- Each detection is credited to at most one event.
- Ranks are worst-case among ties: the probabilities are coarse, so 218 of 225 windows in one run share their value with another window.

The earlier rule credited any window within a 19–25 s zone, which a random top-20 pick passes 17% of the time.

Ranking quality (AUC: windows containing an event vs all others):

| | g1h1 | g1h2 | g2h1 | g2h2 | Mean |
|---|---|---|---|---|---|
| shot, shot_bin 896 px | 0.85 | 0.78 | 0.80 | 0.69 | 0.78 |
| shot, v2 448 px | 0.85 | 0.88 | 0.78 | 0.96 | **0.87** |
| corner, v2 | 0.89 | 0.91 | 0.98 | 0.98 | **0.94** |

Free kicks, offsides, cards and substitutions rank between 0.43 and 1.00, but each half has only 1–7 of them.

**Shots as counts.** Detection is peak picking: windows are taken in order of p(shot), and any window within 30 s of one already taken is dropped, so the number of detections is the predicted count.
- Cut-offs were chosen on game1 H1 only.
- A detection may start up to 15 s after the shot, because the clearest cues (reactions, the replay wipe) follow it.
- The random baseline uses the same number of detections, placed under the same rule.

Held-out halves (34 shots):

| Detector (cut-off) | Found | Real | F1 | Random F1 | Count error per half |
|---|---|---|---|---|---|
| v2 448 px (p ≥ 0.02) | 18/34 | 18/77 | 0.32 | 0.11 | +22, +8, +13 |
| shot_bin 896 px (p ≥ 0.9) | 8/34 | 8/20 | 0.30 | 0.09 | +2, −12, −4 |

- **The 896 px advantage seen on game1 H1, the half the prompts were tuned on, was a selection effect.** On held-out halves both detectors reach about 3× random. The pipeline now uses v2 at 448 px for every class: it's one pass at about 2.2× lower cost.
- **Neither gives usable counts.** v2 overcounts by 8–22 shots per half. shot_bin's probability scale collapses on game2 (12 too few in one half). Counting options differ by one or two, so a ledger fed by either would stay near chance.
- **Corners rank well but a fixed threshold fails.** At p ≥ 0.5, 10/28 are found, with 10 of 31 detections real, despite AUC 0.89–0.98. The probability scale shifts between halves.

**Caveats:**
- The truth set is the events the test asks about. Real shots it skips count as false positives; in game1 H1 the VQA set and SoccerNet-v2 share only 5 of their 9 and 7 shots.
- Each cut-off was tuned on one half, and there are four halves in total.

### Per-half shot cut-off (2026-11-24)

The cut-off now applies to each window's **causal percentile**: the share of this half's windows so far, itself included, whose p(shot) is at or below its own. Later video never changes the value, so a streaming answerer may use it.
- Windows in a half's first 120 s get no detections, because a percentile over a handful of windows is noise. The 120 s was set before scoring.
- The percentile cut-off was chosen on game1 H1 only, from 0.80–0.995. Peak picking, the lead and the random baseline are unchanged.
- **Rate baseline for counts:** game1 H1's shots per second times each half's length. A fixed percentile fires at a roughly fixed rate, so it has to beat this to show that it tracks the match.

Held-out halves (34 shots):

| Detector (cut-off) | Found | Real | F1 | Random F1 | Count error per half | Sum of \|count error\| |
|---|---|---|---|---|---|---|
| v2 (p ≥ 0.02) | 18/34 | 18/77 | 0.32 | 0.11 | +22, +8, +13 | 43 |
| v2 (percentile ≥ 0.93) | 16/34 | 16/61 | 0.34 | 0.10 | +9, +3, +15 | 27 |
| shot_bin (p ≥ 0.9) | 8/34 | 8/20 | 0.30 | 0.09 | +2, −12, −4 | 18 |
| shot_bin (percentile ≥ 0.97) | 10/34 | 10/27 | 0.33 | 0.09 | −3, −4, 0 | 7 |
| Rate baseline | | | | | −3.9, −3.7, +2.2 | 9.8 |

- **It fixes shot_bin's calibration collapse.** Game2 goes from 1 and 3 detections to 9 and 7, and the sum of count errors falls from 18 to 7.
- **Counts barely beat a constant rate.** shot_bin's errors follow the rate baseline's half by half (−3, −4, 0 against −3.9, −3.7, +2.2). v2 is worse than the rate baseline. A percentile cut-off makes the detection rate the same in every half, which also removes the signal that one half had more shots than another.
- **F1 gains are within noise:** 0.32 → 0.34 and 0.30 → 0.33 on 34 shots.
- **The warm-up matters little.** Over 0, 60, 120 and 240 s the chosen cut-off stays the same and F1 stays at 0.33–0.36. Three held-out shots fall in the first 120 s.
- **Peak picking still looks up to 30 s ahead.** The cut-off is causal, but a window can be dropped for a higher one up to 30 s later. A ledger would therefore confirm a shot up to 30 s late, or pick forward in time order instead.

### Where the counting points are: detection vs attribution (2026-11-24)

Exploratory, run from scratch scripts on the committed window outputs; no repo code changed.

**What the counting questions ask.**

| Split | Counting questions | About shots | Shots, about a player |
|---|---|---|---|
| test | 84 | 37 | 28 |
| challenge | 150 | 74 | 55 |

Offsides come second on the challenge set (31, of which 21 are about a player). Corners have 10 questions in each split. On the challenge set, 83 of the 150 counting questions are about a player, not a team.

**Corners.** The per-half percentile cut-off overcounts by 13–19 per held-out half, because game1 H1 has too few corners to tune on. The absolute p ≥ 0.5 cut-off lands near the right counts (+2, −1, +1), but finds only 5 of 16 held-out corners, so those counts are close by coincidence.

**Better shot evidence from what we already have: none.** Each cue's causal percentile in its half, alone and averaged with v2's p(shot):

| Score | Mean AUC (4 halves) |
|---|---|
| v2 p(shot) | 0.87 |
| shot_bin p(shot) | 0.78 |
| v2 replay in the next 30 s | 0.54 |
| v2 goal kick in the next 24 s | 0.63 |
| v2 corner in the next 30 s | 0.62 |
| v2 + shot_bin over the next 12 s | 0.87 (held-out F1 0.36) |

Adding the after-shot cues to the average lowers AUC to 0.75–0.81.

**A bound on shot-count accuracy.** Ledgers were built from detections and scored on the 37 test shot-count questions with `choose_count_option`:
- A detection matched to a true shot (strict rule, 15 s lead) takes that shot's time and attribution.
- A false detection gets a random team and a random player from the match.
- Detections become visible 30 s after their window ends, because of the peak-picking look-ahead.
- Every result is a mean over 50 draws of the random choices.

"Measured" attribution is right with the video QA run's accuracy on the shot/goal "now" questions: 70% for team (chance 50%) and 49% for player (chance 25%). Those choices are among 2 teams or 4 listed players. A ledger has to pick from a whole squad, so "measured" is optimistic.

| Ledger fed by | Attribution | Team (9) | Player (28) | Total (37) |
|---|---|---|---|---|
| Video QA run (no ledger) | | 3 | 7 | 10 |
| Empty ledger (always 0) | | 2 | 8 | 10 |
| Chance (expected) | | | | 9.2 |
| Question-stream oracle (diagnostic) | oracle | 9 | 28 | 37 |
| Perfect detection, visible 30 s late | oracle | 6.0 | 28.0 | 34.0 |
| v2, percentile ≥ 0.93 | oracle | 5.3 | 12.6 | 17.9 |
| shot_bin, percentile ≥ 0.97 | oracle | 5.1 | 14.4 | 19.5 |
| Perfect detection, visible 30 s late | measured | 4.6 | 14.9 | 19.5 |
| v2, percentile ≥ 0.93 | measured | 5.3 | 11.2 | 16.5 |
| shot_bin, percentile ≥ 0.97 | measured | 4.3 | 12.1 | 16.4 |

- **Current detections looked worth wiring in, if attribution held up.** At measured attribution the ledger answers about 16.5 of 37, against 10 for the video QA run. The next subsection shows that attribution does not hold up, and that most of this gain is an artefact of how the test options are built.
- **Player attribution is the bigger bottleneck.** Perfect detection with measured attribution (19.5) is barely better than our detections with measured attribution (16.5). Perfect detection with oracle attribution reaches 34.
- **The 30 s visibility delay costs 3 of 9 team questions** even with perfect detection: questions asked soon after a shot don't see it yet.
- **Scale:** the score is the unweighted mean of six subcategories. 6.5 more of the 84 counting questions is 7.7 points of counting, or about 1.3 points of the score. (Corrected 2026-11-25: this said 0.9 points of pooled accuracy.)

### Player attribution from the full squad (2026-11-24)

Exploratory, run from a scratch script; no repo code changed.

**Setup.** For each of the 43 true shots on test, the model sees 8 frames from 6 s before to 10 s after the shot. Video after the shot is fair, because a ledger records a shot only after its window. Qwen3.5-4B (NF4) is asked two questions, scored by letter logits:
1. Which team takes the shot? The two options give each team's kit, and the prompt gives which goal each team defends.
2. Which player of that team's full 18-man pre-match squad takes it? Each option gives shirt number and position.

A no-frames control asks the same questions with the roster and kits only.

| Frames | Team right | Player right (of 36) | Player right when team right | True player in top 3 |
|---|---|---|---|---|
| 8 at 896 px | 24/43 (56%) | 5/43 (12%) | 21% | 21% |
| 8 at 448 px | 22/43 (51%) | 4/43 (9%) | 18% | 14% |
| None (control) | 22/43 (51%) | 5/43 (12%) | 23% | 19% |

- **The frames add nothing measurable.** Video at either resolution scores like the no-frames control.
- **The answers collapse onto fixed picks.** The model names Manchester United or Liverpool for 41–43 of the 43 shots. At 896 px it names one of two players, Nathaniel Clyne or Memphis Depay, for 28 of them.
- **The 70% / 49% "measured" attribution above was far too optimistic.** In the "now" questions the 4 listed players do most of the work. Against a full squad, Qwen3.5-4B cannot tell who shot.

**Ledgers with this attribution** (false detections get a random team, and a random player of that team's squad):

| Ledger fed by | Attribution | Total (37) |
|---|---|---|
| Video QA run (no ledger) | | 10 |
| Perfect detection | 896 px | 12.0 |
| Perfect detection | none (control) | 9.0 |
| v2, percentile ≥ 0.93 | 896 px | 17.7 |
| v2, percentile ≥ 0.93 | none (control) | 15.8 |

**Why v2 still scores well: an artefact of the test options.**
- Every test counting question offers 4 consecutive numbers, and the correct answer's position is balanced overall (21 at each rank of 84).
- Within groups it is skewed. When the smallest option is above 0, the correct answer is the largest in 12 of 14. For team counts since kick-off, the largest option is correct in 12 of 19.
- v2's false detections overcount, and `choose_count_option` snaps an overcount to the largest option. That earns team points without any video evidence; the no-frames control scores almost the same.
- The challenge split builds its options differently: only 4 of 150 option sets are consecutive (e.g. [1, 3, 4, 6], [2, 5, 8, 10]). So neither this artefact nor a question-only rule built on it carries over.

**Conclusion.** With Qwen3.5-4B, a video-fed ledger is not worth building for shot counting. Detection is weak (about 1 detection in 3 is real), and player identity, which 55 of the 74 challenge shot questions need, is not recoverable from frames at 448 or 896 px. Test-split counting gains should also be checked against the option artefact before being trusted.

**Next, if counting stays a target:**
- Player attribution needs a different signal than a squad-wide multiple choice from a 4B model: shirt-number reading on crops, or a larger model.
- Otherwise move effort to question types where frames help the most.

## 12. Where the headroom is, and recall_event_time from the window detector (2026-11-24)

Exploratory, run from scratch scripts on the committed logs; no repo code changed.

### Headroom by subcategory

*Corrected 2026-11-25.* The first version of this table weighted each subcategory by its share of the challenge split, which would matter for pooled accuracy. The score is the **unweighted mean of the six subcategories** (section 1), so each one counts 1/6 however many questions it has. "Points left" is (100 − video accuracy) / 6: the score one subcategory would add at 100%.

| Subcategory | Test n | Challenge n | Chance | Text | Video (`video_v2`) | Points left | With `--recall-detector` (`video_v3_recall`) | Points left |
|---|---|---|---|---|---|---|---|---|
| recall_event_time | 200 | 493 | 0.25 | 0.28 | 0.37 | 10.5 | **0.67** | 5.5 |
| counting | 84 | 150 | 0.25 | 0.31 | 0.25 | 12.5 | 0.25 | **12.5** |
| causal reasoning | 39 | 50 | 0.25 | 0.36 | 0.26 | 12.4 | 0.26 | **12.4** |
| event_player | 130 | 158 | 0.25 | 0.33 | 0.49 | 8.5 | 0.49 | 8.5 |
| recall_event_player | 129 | 215 | 0.25 | 0.29 | 0.54 | 7.6 | 0.54 | 7.6 |
| event_team | 118 | 92 | 0.50 | 0.50 | 0.69 | 5.2 | 0.69 | 5.2 |

- **Before the recall mode,** `recall_event_time` had the most headroom of the video-strong subcategories. It still had less than counting and causal reasoning, which are at chance with video.
- **Video scores below text-only on counting and causal reasoning** (0.25 vs 0.31, and 0.26 vs 0.36). Both splits are small (84 and 39), and counting on test is inflated by the option artefact in section 11, so picking text-only for them on this evidence would be selection on test.

### recall_event_time

**The questions.** For example, "Looking back, when did Cameron Borthwick-Jackson last take a corner?" with options First half 03:22, 03:11, 03:03 and 02:54.
- The four options sit close together: on test the median gap is 10 s (10th–90th percentile 8–12 s).
- The correct option is the event moment. Here the matching "now" question came at 03:24.
- The correct option is spread evenly by time order and by listed position (50 at each of the 4).
- Kinds on test: 70 shots, 46 substitutions, 24 free kicks, 21 corners, 17 offsides, 14 cards and 8 injuries.

**Method.** For each option, read the v2 detector's p(kind) from the 6 s window that contains the option time, and pick the highest. Only windows that end by the question time are used, and each window's probabilities come from its own frames only, so the method is causal.
- The v2 window outputs are the committed `logs/sn_vqa_2027/events/*_v2.json`.
- Goals use p(shot). Injuries have no v2 class, so they keep the video QA answer.
- Player and "first" vs "last" are ignored: the options already narrow the question to one episode.

| Window(s) read per option | Correct (200) |
|---|---|
| Video QA run (`video_v2`) | 74 (0.37) |
| Window containing t | **133 (0.665)** |
| Windows overlapping [t − 3, t + 3], max | 131 |
| Windows overlapping [t, t + 6], max | 127 |
| Windows overlapping [t − 6, t + 6], mean | 130 |
| Windows overlapping [t − 6, t + 6], max | 118 |
| Windows overlapping [t − 3, t + 9], max | 83 |

The single-window rule was the first one tried, not chosen from this table.
- **By kind:** shots 60/70, cards 11/14, corners 14/21, offsides 11/17, free kicks 11/24, substitutions 22/46. Substitutions do better with the 12 s mean (29/46): they unfold slowly.
- **Every half improves:** game1 H1 14 vs 10 (of 28), game1 H2 52 vs 27 (of 72), game2 H1 31 vs 13 (of 43), game2 H2 36 vs 24 (of 57).
- **Paired against the video QA run:** 78 questions only the detector gets right and 19 only the video run gets right; exact McNemar p = 1.2e-9.
- **No artefacts found:** the correct option's position is uniform, only 5 questions have tied scores, and no option's window ends after the question time.

**Why it works when the video QA run doesn't.** The QA run shows 8 frames spread over the span of all 4 options and asks for a clock time. The detector instead asks, separately for each option's moment, whether that kind of event is happening. Telling which of four nearby moments holds the event is a local comparison, and that suits a per-window classifier.

**Expected value.** +29.5 points on `recall_event_time` is about +4.9 points of the score (1/6 of it), if it transfers. (Corrected 2026-11-25: this said +12.6, using the challenge question share.)

**Caveats for the challenge split.**
- **The options are closer together:** median 7 s, 10th percentile 5 s, 90th percentile 16 s. A fixed 6 s grid can put two options in one window, so a runner should score a window centred on each option instead.
- **The challenge videos are not on disk:** only `video-test/` is.

### Centred windows at question time (GPU check)

This is the form a runner would use. For each option time t, the model sees one window centred on t, shifted earlier if it would end after the question time. The prompt is v2 at 448 px, scored exactly as the detector does. The 8 injury questions are skipped; the video QA run gets 4 of them.

| Window per option | Correct (192, no injuries) | With injuries (200) | Only this right / only video right | McNemar p |
|---|---|---|---|---|
| Video QA run | 70 | 74 | | |
| 6 s grid (above) | 129 | 133 | 78 / 19 | 1.2e-9 |
| 6 s centred, 2 fps | **129** | **133** | 76 / 17 | 4e-10 |
| 4 s centred, 3 fps | 127 | 131 | 78 / 21 | 7e-9 |

- **Centred windows reproduce the grid result:** 129 of 192 both ways.
- **4 s windows lose almost nothing:** 127, with the same 12 frames. They suit the challenge split's tighter gaps (10th percentile 5 s), where a 6 s window would overlap the next option.
- **By kind (6 s centred):** shots 61/70, corners 16/21, cards 11/14, free kicks 11/24, offsides 8/17, substitutions 22/46. Shots and corners carry the gain. Free kicks and substitutions are near the QA run's level.
- **Cost:** about 6 s per question (4 windows), so the 493 challenge questions take about 50 minutes on the 8 GB card.

### In the runner: `--recall-detector`

`scripts/evaluate_sn_vqa_2027_video.py --recall-detector` answers `recall_event_time` this way: 4 s windows with 12 frames, the v2 prompt from `window_messages`, and option windows from `option_window`. Questions whose kind has no v2 class (injuries) use the normal question path. The detector script now builds its prompt with the same `window_messages`, and its output on game1 H1's first 30 s is unchanged.

Test, `--subcategories recall_event_time` (scratch output, not committed):

| | Correct (200) |
|---|---|
| Video QA run (`video_v2`) | 74 |
| Runner, `--recall-detector` | **134 (67.0%)** |
| Detector-answered | 130/192 |
| Injuries, question path | 4/8, the same answers as `video_v2` |

- **Cost:** 5.0 s per question; the largest prompt is 1634 tokens.
- **Individual answers depend on frame choice:** the runner picks 12 of the window's frames decoded every 0.2 s, while the scratch run kept 3 fps. Only 140 of the 192 picks agree, yet accuracy is the same (134 vs 131 of 200).
- **CPU tests** (`scripts/test_sn_vqa_2027_events.py`, test 8) cover kind parsing and option windows: centred, moved earlier so they never pass the question time, clipped at 0, and None for an option after the question.
  - With `--official-dir`, every test and challenge recall question gets a v2 class except injuries (8 on test, 21 on challenge), and every option parses as a clock time.

### Full test run with `--recall-detector` (2026-11-25)

`logs/sn_vqa_2027/video_v3_recall/`, the same settings as `video_v2` plus `--recall-detector`:

| | `video_v2` | `video_v3_recall` |
|---|---|---|
| **Score (mean of six subcategories)** | 43.30 | **48.30** |
| Pooled accuracy | 320/700 (45.7%) | 380/700 (54.3%) |
| recall_event_time | 74/200 | 134/200 |
| Each of the other five subcategories | | identical answers, question by question |
| Time | 768 s (1.1 s per question) | 1426 s (2.0 s per question) |

- **The recall mode is the whole gain:** 74 questions gained and 14 lost, all in `recall_event_time`; exact McNemar p = 5.1e-11.
- **The other subcategories reproduce exactly:** same kernels and the same frames, so their answers are deterministic. This also confirms that the runner change did not touch the normal path.
- **Against the published numbers:** the organizers' test baseline scores 41.76, so 48.30 is 6.5 points above it.

**Next:**
- Download the challenge videos and run the challenge split with `--recall-detector` (5 matches, 10 halves; about 40 minutes for the 472 detector-answered questions).
- By the score's equal weighting, counting (12.5 points left) and causal reasoning (12.4) are now the largest headroom, and both sit at chance with video.
