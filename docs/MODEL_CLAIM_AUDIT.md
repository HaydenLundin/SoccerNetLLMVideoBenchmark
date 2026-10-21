# Model claim audit

Checked 2026-10-14. Scope: primary-source verification of the proposed ten-organization local benchmark. This is a documentary audit, not an inference benchmark. No model was loaded, so no full-workload 8 GB fit is certified.

## Findings recorded during verification

| Claim | Verdict | Evidence and implication |
|---|---|---|
| MiniCPM-V 4.6 uses SigLIP2-400M + Qwen3.5-0.8B | Confirmed | Explicit in the [official card](https://huggingface.co/openbmb/MiniCPM-V-4.6/blob/main/README.md). |
| MiniCPM is exactly 1.3B and released May 2026 | Partially established | Rounded component counts support a small model, but the inspected card does not establish exact total parameters. May 17 dates the API release, not explicitly the initial weights release. |
| MiniCPM has 19x temporal compression | Incorrect interpretation | Card documents mixed 4x/16x visual compression. The 19x figure is a comparative token-cost result, not a universal compression factor. |
| Never quantize MiniCPM because strict JSON severely degrades | Unverified | Official AWQ, GPTQ and BNB 4-bit checkpoints exist. Their config JSON files document quantization settings, not output-JSON accuracy. BF16 is a reasonable first test, but the causal quality claim requires a paired benchmark. [AWQ config](https://huggingface.co/openbmb/MiniCPM-V-4.6-AWQ/blob/main/config.json) |
| Gemma E2B: 2.3B effective / 5B total; E4B: 4.5B / 8B | Mostly confirmed | Google lists 5.1B including embeddings for E2B and 8B for E4B. These are dense models with effective counts, not MoE activated counts. [Card](https://huggingface.co/google/gemma-4-E2B#dense-models) |
| PLE automatically offloads embeddings and E4B comfortably fits 60 frames plus audio | Unverified | PLE is documented; selective CPU offload by the intended runtime is not demonstrated by device_map=auto alone. Neither claimed full-workload memory number is measured here. |
| Gemma supports simultaneous audio and video | Not established by inspected cards | Separate audio and video examples establish individual modalities, not synchronized joint audiovisual inference. [Card](https://huggingface.co/google/gemma-4-E2B-it) |
| Qwen3.5-4B has a 3:1 linear/full attention ratio | Confirmed | Config lists 24 linear-attention and 8 full-attention decoder layers. Early-fusion training is documented, and there is still a separate vision encoder. [Config](https://huggingface.co/Qwen/Qwen3.5-4B/blob/main/config.json), [card](https://huggingface.co/Qwen/Qwen3.5-4B/blob/main/README.md) |
| Qwen hybrid attention guarantees 60-frame fit | Unverified | Reduced traditional KV storage is an architectural advantage, not a workload memory certificate. Traditional attention KV cache grows linearly with sequence length; attention computation can grow quadratically. |
| Phi supports joint vision/audio | Confirmed | Card gives image and audio placeholders in the same prompt and vision-speech tasks. This does not establish synchronized video streaming or whistle-recognition accuracy. [Card](https://huggingface.co/microsoft/Phi-4-multimodal-instruct/blob/main/README.md) |
| Moondream2 has a 2K context and single-image API | Confirmed for inspected current implementation | Config defaults to 2,048 positions; image prefix uses 730 positions. The remaining budget includes prompt and output. [Config](https://huggingface.co/vikhyatk/moondream2/blob/main/config.py), [implementation](https://huggingface.co/vikhyatk/moondream2/blob/main/moondream.py) |
| Current Moondream2 definitively uses Phi-1.5 + SigLIP | Not established | Legacy files reference PhiConfig, but the active loader uses custom TextConfig/MoondreamModel. Do not infer current initialization lineage from legacy files alone. [Loader](https://huggingface.co/vikhyatk/moondream2/blob/main/hf_moondream.py) |

The native vision encoder sizes and modality routers do not establish ability to interpret referee whistles or crowd noise. Speech-oriented audio benchmarks are a different task. Also, replacing FlashAttention with SDPA/eager for Phi is a runtime compatibility hypothesis until its specific adapter path is tested.

- Ministral 3 3B: official card specifies 3.4B language parameters and a 0.4B vision encoder, and says FP8 can fit in 8 GB. This does not establish fit for 60 images. [Source](https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512/raw/main/README.md)
- LFM2.5-VL-3B: official card specifies a 2.6B language backbone, SigLIP2 NaFlex 400M encoder, 3.1B total, and 32,768 context. It describes a hybrid architecture; calling it wholly non-Transformer is misleading. A reported memory number on other hardware does not certify this laptop workload. [Source](https://huggingface.co/LiquidAI/LFM2.5-VL-3B/raw/main/README.md)
- DeepSeek-VL2 Tiny: official card identifies DeepSeekMoE-3B with 1B activated parameters. Active parameters describe per-token computation, not all resident weights. For three or more images the documented processor switches to padded 384x384 images rather than dynamic tiling. The proposed 8–12-frame maximum is not a documented hard limit. [Source](https://huggingface.co/deepseek-ai/deepseek-vl2-tiny/raw/main/README.md)
- MolmoE-1B: the proposed replacement has 7.2B total and 1.5B active parameters, not 1B total. Its native weights alone require about 14.4 GB decimal at two bytes per parameter. A quantized implementation would need its own compatibility and memory test. [Source](https://huggingface.co/allenai/MolmoE-1B-0924/raw/main/README.md)

## Measurement boundary

- SmolVLM2's card confirms SigLIP-SO400M + SmolLM2-1.7B and video training, and reports 5.2 GB GPU RAM for video inference. That is developer-reported for its setup, not a certification of 60-frame inference here. [Source](https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct/raw/main/README.md)
- InternVL3.5's card confirms Qwen3-based language models and InternViT encoders. Its Flash variant has a separate resolution router/compression path; attributing Flash-specific compression to the ordinary InternVL3_5-2B checkpoint is incorrect. [Source](https://huggingface.co/OpenGVLab/InternVL3_5-2B)
- Liquid's underlying model mixes 22 short-convolution blocks with 8 grouped-query attention blocks. It is a hybrid convolution/attention model, not wholly free of Transformer attention. [Source](https://huggingface.co/LiquidAI/LFM2.5-2.6B)

## Experimental claims, not documentary facts

- Four-bit quantization severely degrading 17-class JSON/MM:SS adherence requires a paired test on the exact checkpoint, runtime and decoding method. No benchmark-specific evidence has been established.
- Better general multimodal scores for a larger model do not prove better soccer spotting or whistle interpretation.
- Suggested caps of 8–12, 15–20, or 15–30 frames are tuning starting points, not established architectural maxima. Processor tiling, visual token budget and output length matter.
- Reducing a 60-second window to 15 frames introduces an input-budget difference. Report a common-frame-budget comparison separately from each model's best feasible configuration.
- A timestamped contact sheet can be tested as a storyboard baseline, but reduces per-frame spatial detail and does not make a single-image model natively temporal.
- Native video input often means frame extraction by the processor, not continuous-motion access. Compare actual selected frames, timestamps, resolution and audio, not just file extensions.
- V1 narrative: SCOPE_GCP.md reports F1 0.075 and random baseline 0.041 with unspecified matching rules. Calling 0.04 the established V1 model score is unsupported. Re-running old weights with the new scorer creates a controlled comparison, but cannot certify improvement relative to an undefined historical metric.

Approximate BF16 weight bytes = total parameters x 2. This excludes activations, KV/recurrent state, vision/audio processing, quantization metadata, temporary buffers and framework allocations. Four-bit bytes = total parameters / 2 is only an idealized lower bound; many quantizers leave some modules at higher precision.

The existing RESEARCH.md and IMPLEMENTATION_PLAN.md specify 60-second local windows. Their wording that a frame cap strictly prevents OOM should be treated as a proposed configuration, not a measured guarantee.

## Local schema check

The existing processing/unified_schema.py has the 17-class Literal enum, but timestamp is currently an unconstrained string with an MM:SS description. Its parser also accepts scalar seconds and HH:MM:SS. Therefore the present schema does not enforce strict MM:SS syntax or timestamps within a 60-second clip. Pydantic validation is not itself constrained generation. These observations were read-only; no implementation was changed.

## Recommended measured gate

Use the exact planned backend and checkpoint revision, batch size one, bounded output length and the same spotting schema. For each candidate, test a short clip first, then 15, 30 and 60 frames. Record processor-expanded visual token count, peak allocated/reserved VRAM, CPU offload, wall time, schema validity and out-of-window timestamps. Audio-enabled cases need separate gates. BF16 MiniCPM and quantized Gemma E4B are candidates, not certified configurations.

Preserve a common-frame-budget comparison across models, then report optional best-feasible settings separately. Keep contact-sheet Moondream outside the comparable temporal-video leaderboard. Do not treat MolmoE's active parameter count or Llama 3.2 Vision 11B at four bits as proof of a safe 8 GB replacement.
