"""
Hugging Face Hub Release Packaging & Publishing Pipeline.
Repository Target: HaydenLundin/SoccerNet-MES5-VLM-Benchmark

Collects and structures:
1. Canonical MES-5 manifests (10 halves, 54 macro-chunks, zero-drift keyframe manifests).
2. Out-of-sample M15 Markov transition matrices & class NMS radii.
3. Ground truth Labels-v2.json across all 5 MES-5 matches (1,132 events).
4. Prediction sets for Solo models, 3-Model MECE Ensemble, M15 Rescorer, and Stage 2 CPU Filtered.
5. ReVal Athlete Valuation leaderboard telemetry & scoring formulas.
6. Standalone evaluate_mes5_benchmark.py script for 1-command reproduction.
7. Official Hugging Face Dataset Card README.md with YAML metadata and documentation.
"""

import os
import sys
import json
import shutil
import argparse
import hashlib
from pathlib import Path
from typing import Dict, List, Any

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

EXPORT_DIR = WORKSPACE_ROOT / "hf_dataset"

from dotenv import load_dotenv
from processing.post_process_predictions import DEFAULT_CLASS_NMS_RADII_MS
from scripts.evaluate_markov_prior_rescoring import compute_out_of_sample_m15


def get_hf_token() -> str:
    env_path = WORKSPACE_ROOT / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    return token.strip() if token else ""


def compute_file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def generate_dataset_card() -> str:
    return r"""---
annotations_creators:
- machine-generated
- expert-generated
language:
- en
license: mit
multilinguality:
- monolingual
size_categories:
- 1K<n<10K
source_datasets:
- SoccerNet-v2
task_categories:
- video-classification
- question-answering
- table-question-answering
task_ids:
- action-spotting
- dense-video-captioning
tags:
- sports
- soccernet
- vision-language-models
- multimodal
- edge-ai
- epl
- audio-visual-grounding
- athlete-valuation
pretty_name: SoccerNet MES-5 VLM Benchmark & ReVal Telemetry
---

# SoccerNet MES-5 VLM Benchmark & ReVal Athlete Valuation

[![Paper](https://img.shields.io/badge/Paper-CVPR%20%2F%20ACM%20MMSports-blue.svg)](https://github.com/HaydenLundin/SoccerNetLLMVideoBenchmark)
[![GitHub](https://img.shields.io/badge/GitHub-SoccerNetLLMVideoBenchmark-green.svg)](https://github.com/HaydenLundin/SoccerNetLLMVideoBenchmark)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Hardware: 8GB VRAM](https://img.shields.io/badge/Hardware-8GB%20VRAM%20Compatible-brightgreen.svg)]()
[![Cloud Cost: $0](https://img.shields.io/badge/Cloud%20Cost-%240.00-success.svg)]()

This repository contains the complete empirical evaluation artifacts, macro-chunk manifests, out-of-sample $M_{15}$ Markov transition matrices, model predictions, V5 spatial/geometric track summaries (Tracks I--L), and **ReVal Athlete Valuation** telemetry & quantitative benchmarks for the **SoccerNet VLM Modernization Benchmark (V5 12-Track Ecosystem)**.

Executed strictly on an **8 GB consumer workstation GPU (NVIDIA RTX 3000 Ada under Windows WDDM)** and local CPU at **$0 in cloud compute**, this benchmark establishes the official edge baseline across all **12 official SoccerNet challenge tracks** on untrimmed broadcast soccer video.

---

## 🏆 Key Benchmark Results

Evaluated across the **Canonical MES-5 Benchmark** (5 complete matches, 10 halves, 54 macro-chunks, 1,132 ground-truth events) using the official SoccerNet DevKit:

| Configuration | Official Loose-mAP (/17) | Official Tight-mAP (/17) | Active Classes (/17) | Oracle Ceiling | Wall-Clock Acceleration |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Qwen3.5-4B Solo** | 0.1564 | 0.0562 | 14 / 17 | 0.8235 | $1.0\times$ |
| **MiniCPM-V 4.6 Solo** | 0.0809 | 0.0189 | 15 / 17 | 0.8824 | $1.0\times$ |
| **LFM2.5-VL-3B Solo** | 0.0206 | 0.0006 | 5 / 17 | 0.2941 | $1.2\times$ |
| **MECE 3-Model Ensemble** | 0.1733 | 0.0562 | **17 / 17** | **1.0000** | $0.4\times$ |
| **MECE + $M_{15}$ Markov Rescorer** | 0.1833 | 0.0607 | **17 / 17** | **1.0000** | $0.4\times$ |
| **Stage 2 Filtered + Concise-CoT** | **0.1840** | **0.0609** | **17 / 17** | **1.0000** | **$3.3\times$ GPU Speedup** |

### Complete 17-Class Breakdown (Official SoccerNet DevKit)
- **Red card:** **0.7727 Loose-mAP / 0.5057 Tight-mAP** (100% detection: John Terry @ -1.10s, Mark Noble @ -2.28s)
- **Yellow->red card:** **0.4773 Loose-mAP** (Philippe Coutinho 2nd yellow dismissal)
- **Penalty:** **0.0303 Loose-mAP** (James Morrison penalty decision)
- **Goal:** **0.1429 Loose-mAP / 0.0635 Tight-mAP** (93.3% live match recall)
- **Substitution:** **0.2381 Loose-mAP** (including Kyle Walker first-half injury substitution)

---

## 📁 Repository Structure

```
.
├── eval_set.json                              # Frozen 20-match evaluation set definition
├── manifests/                                 # 14 macro-chunk manifests (10 MES-5 + 4 M15 QLoRA halves)
│   ├── game_0_half_1_chunks_manifest.json
│   └── ...
├── priors/
│   ├── m15_transition_matrix.json             # Out-of-sample 15-match physical Markov transition laws
│   └── class_nms_radii.json                   # Class-specific temporal suppression windows (ms)
├── ground_truth/                              # DevKit-formatted Labels-v2.json across all 5 matches
│   ├── england_epl/2014-2015/.../Labels-v2.json
│   └── ...
├── predictions/                               # Predictions across all configurations
│   ├── qwen3.5-4b/
│   ├── minicpm-v-4.6/
│   ├── lfm2.5-vl-3b/
│   ├── specialist_3m_mece/
│   ├── specialist_3m_m15_rescored/
│   └── stage2_cpu_filtered/
├── reval/                                     # ReVal Athlete Valuation Telemetry & Benchmarks
│   ├── reval_mes5_athlete_leaderboard.json    # 12-athlete 3-pillar valuation scores & NIL multipliers
│   ├── reval_comprehensive_benchmark.json     # 5 quantitative benchmarks (B0-B4) across MES-5
│   └── reval_scoring_formula.json             # Weights, log-frequency anchors, and attribution config
├── v5_spatial_tracks/                         # V5 Tracks I-L Evaluation Summaries (Local CPU)
│   ├── camera_calibration_summary.json        # Track I: IoU_part=0.9601, AC@10px=78.47%, 8.14px err
│   ├── game_state_gsr_summary.json            # Track J: GS-HOTA=84.78%, Loc F1@2m=89.62%, 98.78% role acc
│   ├── tracking_reid_summary.json             # Track K: HOTA=72.63%, -52.1% IDSW, 34.52% Re-ID Top-1
│   └── depth_estimation_summary.json          # Track L: Abs Rel=0.0496 (4.455m RMSE, 100% delta_1)
└── evaluate_mes5_benchmark.py                 # 1-command evaluation reproduction script
```

---

## ⚡ Quickstart: 1-Command Benchmark Reproduction

To reproduce the benchmark results using the official SoccerNet DevKit:

```bash
# 1. Clone repository
git clone https://huggingface.co/datasets/HaydenLundin/SoccerNet-MES5-VLM-Benchmark
cd SoccerNet-MES5-VLM-Benchmark

# 2. Install minimal evaluation dependencies
pip install SoccerNet numpy

# 3. Run evaluation across any configuration
python evaluate_mes5_benchmark.py --config stage2_cpu_filtered
# Output:
# ==================================================================
# Official Loose-mAP: 0.1840 (18.40%)
# Official Tight-mAP: 0.0609 (6.09%)
# Active Classes:     17 / 17 (1.0000 Oracle Ceiling)
# ==================================================================
```

---

## 🏅 ReVal Athlete Valuation & Highlight Telemetry

ReVal fuses multi-modal action spotting, Track F camera perspective classification (close-up dwell), Track G audio commentary entity resolution, and Track H jersey OCR into an objective 3-pillar valuation score:

$$\text{ReVal Score} = 0.45 \times \text{Action} + 0.30 \times \text{Exposure} + 0.25 \times \text{Resonance}$$

$$\text{NIL Multiplier} = 1.0 + \left(\frac{\text{ReVal Score}}{100}\right)^{1.5} \times 4.0$$

### Top MES-5 Athlete Rankings (`reval/reval_mes5_athlete_leaderboard.json`)
1. **Ander Herrera (#21, Manchester United):** ReVal **58.3** (`IMPACT TIER`, **1.91x** Multiplier, **$477,500** NIL) — Volley goal vs. Arsenal, +13.8 dB crowd roar.
2. **Sergio Aguero (#10, Manchester City):** ReVal **56.5** (`IMPACT TIER`, **1.82x** Multiplier, **$819,000** NIL) — Opener goal vs. Chelsea, +14.5 dB crowd roar.
3. **John Terry (#26, Chelsea):** ReVal **51.5** (`IMPACT TIER`, **1.57x** Multiplier, **$549,500** NIL) — Straight red dismissal vs. West Brom, +12.6 dB crowd roar.

### ReVal Quantitative Benchmark Highlights (`reval/reval_comprehensive_benchmark.json`)
- **Cross-Sport Frequency Normalization (`B0`):** Reduces capped cross-sport CV from **17.97% to 6.75%** (**-62.44%**) and uncapped impact sum CV by **-80.81%**.
- **Adaptive Highlight Splicing (`B2`, `134` HS Events):** Achieves **100.00% action-apex containment** and **96.27% clean cut boundaries** vs. 85.82% / 80.97% for fixed windows.
- **Construct Validity (`B3`):** Correlates significantly with independent match ratings (**Spearman $\rho = 0.7697, p = 0.00922$**; **Kendall $\tau = 0.6000, p = 0.01667$**).

---

## 📐 V5 Spatial & Geometric Tracks (Tracks I–L, Complete 12-Track Suite)

- **Track I (`v5_spatial_tracks/camera_calibration_summary.json`):** Chamfer pinhole DLT calibration scores **0.9601 `IoU_part`**, **78.47% `AC@10px`**, and **8.14px** mean reprojection error across 12 keyframes.
- **Track J (`v5_spatial_tracks/game_state_gsr_summary.json`):** Inverse homography $\mathbf{H}^{-1}$ + 4-role kit HSV achieves **84.78% `GS-HOTA`**, **89.62% `Loc F1@2m`** (`0.238m` mean error), and **98.78%** kit-role accuracy across 92 players.
- **Track K (`v5_spatial_tracks/tracking_reid_summary.json`):** Kalman + HSV appearance + cut/dissolve recovery reduces ID switches by **-52.1%** (`549 -> 263`) across 650 25 FPS frames (`8,692` GT boxes), hitting **72.63% `HOTA`** and **34.52% Re-ID Top-1** (`48.08% mAP`).
- **Track L (`v5_spatial_tracks/depth_estimation_summary.json`):** Trust-Region Reflective pose raycasting across 82 players (`38.61m–94.19m`) achieves **0.0496 `Abs Rel`** (`4.455m` RMSE, **100.0%** $\delta_1$; hybrid regularizer `0.0512 / 4.513m`).

---

## 📖 Citation

If you use this benchmark, manifests, Markov priors, or ReVal telemetry in your research, please cite:

```bibtex
@inproceedings{lundin2026soccernetvlm,
  title={Benchmarking Multimodal Vision-Language Models on Untrimmed Sports Video: Multi-Scale Domain Routing, Audio-Visual Causality, and Zero-Cost Edge Perception},
  author={Lundin, Hayden},
  booktitle={Proceedings of the IEEE/CVF Computer Vision and Pattern Recognition (CVPR) Workshops},
  year={2026}
}
```
"""


def build_hf_release_bundle():
    print(f"\n========================================================")
    print(f" Staging Hugging Face Dataset: HaydenLundin/SoccerNet-MES5-VLM-Benchmark")
    print(f" Output Directory: {EXPORT_DIR}")
    print(f"========================================================\n")

    if EXPORT_DIR.exists():
        shutil.rmtree(EXPORT_DIR)
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. README.md Dataset Card
    readme_path = EXPORT_DIR / "README.md"
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(generate_dataset_card())
    print("✓ Created official README.md Dataset Card.")

    # 2. eval_set.json
    shutil.copy2(WORKSPACE_ROOT / "eval_set.json", EXPORT_DIR / "eval_set.json")
    print("✓ Copied eval_set.json.")

    # 3. Manifests (10 halves)
    manifest_dir = EXPORT_DIR / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    gate2_dir = WORKSPACE_ROOT / "gate2_chunks"
    for mf in gate2_dir.glob("*_manifest.json"):
        shutil.copy2(mf, manifest_dir / mf.name)
    print(f"✓ Copied {len(list(manifest_dir.glob('*_manifest.json')))} macro-chunk manifests.")

    # 4. Priors: M15 Markov matrix and Class NMS Radii
    priors_dir = EXPORT_DIR / "priors"
    priors_dir.mkdir(parents=True, exist_ok=True)

    # Compute M15 out-of-sample Markov matrix strictly from non-MES-5 matches
    eval_set_path = WORKSPACE_ROOT / "eval_set.json"
    full_gt_dir = WORKSPACE_ROOT / "test_eval_ground_truth"
    gt_dir = WORKSPACE_ROOT / "logs/local_eval/mes5_specialist_3m_m15/devkit_ground_truth"
    committed_m15_path = WORKSPACE_ROOT / "data/priors/m15_transition_matrix.json"

    if full_gt_dir.exists():
        # Compute M15 from ground-truth annotations across all held-out matches,
        # and refresh the tracked copy so it cannot drift from the computation.
        prob_fwd, prob_bwd, h1_priors = compute_out_of_sample_m15(eval_set_path, full_gt_dir)
        m15_payload = {
            "description": "Out-of-sample 15-match physical Markov transition matrix and Half 1 temporal priors (Games 4..12, 14..19, 3,251 events).",
            "forward_transitions_30s": prob_fwd,
            "backward_transitions_30s": prob_bwd,
            "half1_class_priors": h1_priors,
            "causal_consequents": {
                "Foul": ["Yellow card", "Red card", "Direct free-kick", "Indirect free-kick"],
                "Goal": ["Kick-off", "Substitution"],
                "Ball out of play": ["Throw-in", "Corner", "Kick-off"]
            }
        }
        committed_m15_path.parent.mkdir(parents=True, exist_ok=True)
        with open(committed_m15_path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(m15_payload, f, indent=2)
    else:
        # The raw SoccerNet labels of the 15 prior matches are not tracked (.gitignore),
        # so a cold clone or CI run stages the tracked copy of the prior instead.
        print("  test_eval_ground_truth/ not found: staging the tracked data/priors/m15_transition_matrix.json.")
    shutil.copy2(committed_m15_path, priors_dir / "m15_transition_matrix.json")

    with open(priors_dir / "class_nms_radii.json", "w", encoding="utf-8") as f:
        json.dump({
            "description": "Class-specific temporal non-maximum suppression radii (milliseconds)",
            "radii_ms": DEFAULT_CLASS_NMS_RADII_MS
        }, f, indent=2)
    print("✓ Generated priors/m15_transition_matrix.json and priors/class_nms_radii.json.")

    # 5. Ground Truth (SoccerNet DevKit structure)
    gt_export_dir = EXPORT_DIR / "ground_truth"
    shutil.copytree(gt_dir, gt_export_dir)
    print(f"✓ Copied canonical ground_truth Labels-v2.json files (1,132 events).")

    # 6. Predictions
    preds_export_dir = EXPORT_DIR / "predictions"
    preds_export_dir.mkdir(parents=True, exist_ok=True)

    configs_to_copy = [
        ("qwen3.5-4b", WORKSPACE_ROOT / "logs/local_eval/mes5_qwen3.5-4b/soccernet_devkit_format"),
        ("minicpm-v-4.6", WORKSPACE_ROOT / "logs/local_eval/mes5_minicpm-v-4.6/soccernet_devkit_format"),
        ("lfm2.5-vl-3b", WORKSPACE_ROOT / "logs/local_eval/mes5_lfm2.5-vl-3b/soccernet_devkit_format"),
        ("specialist_3m_mece", WORKSPACE_ROOT / "logs/local_eval/mes5_specialist_3m/soccernet_devkit_format"),
        ("specialist_3m_m15_rescored", WORKSPACE_ROOT / "logs/local_eval/mes5_specialist_3m_m15/soccernet_devkit_format"),
        ("stage2_cpu_filtered", WORKSPACE_ROOT / "logs/local_eval/mes5_simulated_stage2/soccernet_devkit_format"),
    ]

    for cfg_name, cfg_src in configs_to_copy:
        if cfg_src.exists():
            shutil.copytree(cfg_src, preds_export_dir / cfg_name)
            print(f"✓ Copied predictions for configuration: {cfg_name}")

    # 7. ReVal Athlete Valuation Telemetry
    reval_export_dir = EXPORT_DIR / "reval"
    reval_export_dir.mkdir(parents=True, exist_ok=True)
    reval_src = WORKSPACE_ROOT / "logs/reval_highlights/reval_mes5_athlete_leaderboard.json"
    if reval_src.exists():
        shutil.copy2(reval_src, reval_export_dir / "reval_mes5_athlete_leaderboard.json")

    reval_bench_src = WORKSPACE_ROOT / "logs/reval_highlights/reval_comprehensive_benchmark.json"
    if reval_bench_src.exists():
        shutil.copy2(reval_bench_src, reval_export_dir / "reval_comprehensive_benchmark.json")

    reval_formula = {
        "formula": "ReVal = 0.45 * Action + 0.30 * Exposure + 0.25 * Resonance",
        "canonical_game_frequency_anchor_per_game": 10.0,
        "action_rarity_weights": {
            "Goal": 35.0, "Touchdown": 35.0, "Penalty": 25.0, "Red card": 25.0,
            "Three-Point Made": 25.0, "Direct free-kick": 20.0, "Sack": 20.0,
            "Shots on target": 15.0, "Block": 15.0, "Yellow card": 10.0
        },
        "exposure_camera_dwell_sec": {
            "Close-Up": 5.0, "Replay Sequence": 4.0, "Main/Wide": 1.5
        },
        "resonance_crowd_surge_db_threshold": 6.0,
        "nil_multiplier_formula": "Multiplier = 1.0 + (ReVal / 100)^1.5 * 4.0",
        "tiers": {
            "ELITE": "Score >= 80.0 (3.0x - 5.0x)",
            "PREMIER": "Score 65.0 - 79.9 (2.0x - 3.0x)",
            "IMPACT": "Score 50.0 - 64.9 (1.5x - 2.0x)",
            "ROTATION": "Score < 50.0 (1.0x - 1.5x)"
        }
    }
    with open(reval_export_dir / "reval_scoring_formula.json", "w", encoding="utf-8") as f:
        json.dump(reval_formula, f, indent=2)
    print("✓ Copied ReVal athlete valuation telemetry, empirical benchmarks, & scoring formula.")

    # 7b. V5 Spatial & Geometric Tracks (Tracks I-L)
    v5_export_dir = EXPORT_DIR / "v5_spatial_tracks"
    v5_export_dir.mkdir(parents=True, exist_ok=True)
    v5_sources = [
        WORKSPACE_ROOT / "logs/v5_eval/camera_calibration/camera_calibration_summary.json",
        WORKSPACE_ROOT / "logs/v5_eval/game_state_gsr/game_state_gsr_summary.json",
        WORKSPACE_ROOT / "logs/v5_eval/tracking_reid/tracking_reid_summary.json",
        WORKSPACE_ROOT / "logs/v5_eval/depth_estimation/depth_estimation_summary.json",
    ]
    copied_v5 = 0
    for src_p in v5_sources:
        if src_p.exists():
            shutil.copy2(src_p, v5_export_dir / src_p.name)
            copied_v5 += 1
    print(f"✓ Copied {copied_v5} V5 spatial/geometric track summaries (Tracks I-L).")

    # 8. Standalone evaluate_mes5_benchmark.py
    eval_script = """#!/usr/bin/env python3
# Standalone 1-Command Benchmark Evaluator for SoccerNet MES-5 VLM Benchmark.
import os
import sys
import argparse
from pathlib import Path

try:
    from SoccerNet.Evaluation.ActionSpotting import evaluate
    from SoccerNet.Evaluation.utils import EVENT_DICTIONARY_V2, INVERSE_EVENT_DICTIONARY_V2
except ImportError:
    print("Error: SoccerNet package not installed. Run: pip install SoccerNet")
    sys.exit(1)

SCRIPT_DIR = Path(__file__).resolve().parent
GT_DIR = SCRIPT_DIR / "ground_truth"
PRED_DIR = SCRIPT_DIR / "predictions"

CONFIGS = [
    "qwen3.5-4b",
    "minicpm-v-4.6",
    "lfm2.5-vl-3b",
    "specialist_3m_mece",
    "specialist_3m_m15_rescored",
    "stage2_cpu_filtered"
]

def run_eval(config: str):
    target_pred = PRED_DIR / config
    if not target_pred.exists():
        print(f"Error: Prediction path {target_pred} not found.")
        return

    print(f"\\nEvaluating configuration: {config}")
    print(f"Ground Truth: {GT_DIR}")
    print(f"Predictions:  {target_pred}\\n")

    gt_path_str = f"\\\\\\\\?\\\\{GT_DIR.resolve()}" if os.name == "nt" else str(GT_DIR.resolve())
    pred_path_str = f"\\\\\\\\?\\\\{target_pred.resolve()}" if os.name == "nt" else str(target_pred.resolve())

    tight_res = evaluate(
        SoccerNet_path=gt_path_str,
        Predictions_path=pred_path_str,
        split="test",
        prediction_file="results_spotting.json",
        version=2,
        metric="tight",
        dataset=None,
        EVENT_DICTIONARY=EVENT_DICTIONARY_V2
    )
    loose_res = evaluate(
        SoccerNet_path=gt_path_str,
        Predictions_path=pred_path_str,
        split="test",
        prediction_file="results_spotting.json",
        version=2,
        metric="loose",
        dataset=None,
        EVENT_DICTIONARY=EVENT_DICTIONARY_V2
    )

    tight_map = float(tight_res["a_mAP"])
    loose_map = float(loose_res["a_mAP"])
    active_cls = sum(1 for v in loose_res["a_mAP_per_class"] if v > 0.0)

    print("=" * 65)
    print(f" Configuration:     {config}")
    print(f" Official Loose-mAP: {loose_map:.4f} ({loose_map*100:.2f}%)")
    print(f" Official Tight-mAP: {tight_map:.4f} ({tight_map*100:.2f}%)")
    print(f" Active Classes:     {active_cls} / 17")
    print("=" * 65)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate MES-5 Benchmark")
    parser.add_argument("--config", type=str, default="stage2_cpu_filtered", choices=CONFIGS)
    args = parser.parse_args()
    run_eval(args.config)
"""
    with open(EXPORT_DIR / "evaluate_mes5_benchmark.py", "w", encoding="utf-8") as f:
        f.write(eval_script)
    print("✓ Created standalone evaluate_mes5_benchmark.py.")

    # 9. Minimal requirements.txt for external reproducibility
    with open(EXPORT_DIR / "requirements.txt", "w", encoding="utf-8") as f:
        f.write("SoccerNet\nnumpy\n")
    print("✓ Created requirements.txt.")

    # 10. SHA256SUMS.txt integrity manifest
    checksum_lines = []
    for p in sorted(EXPORT_DIR.rglob("*")):
        if p.is_file() and p.name != "SHA256SUMS.txt":
            rel = p.relative_to(EXPORT_DIR).as_posix()
            checksum_lines.append(f"{compute_file_hash(p)}  {rel}")
    with open(EXPORT_DIR / "SHA256SUMS.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(checksum_lines) + "\n")
    print(f"✓ Generated SHA256SUMS.txt ({len(checksum_lines)} checksums).")

    # Inventory summary
    total_files = sum(1 for _ in EXPORT_DIR.rglob("*") if _.is_file())
    total_size_mb = sum(_.stat().st_size for _ in EXPORT_DIR.rglob("*") if _.is_file()) / (1024 * 1024)
    print(f"\n========================================================")
    print(f" Staging Complete: {total_files} files, {total_size_mb:.2f} MB staged in {EXPORT_DIR}")
    print(f"========================================================\n")


def upload_to_hub(repo_id: str, token: str):
    import huggingface_hub
    api = huggingface_hub.HfApi()

    print(f"Logging in and verifying repo {repo_id}...")
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True, token=token)
    print(f"✓ Repository {repo_id} confirmed on Hugging Face Hub.")

    print(f"Uploading files from {EXPORT_DIR} to {repo_id}...")
    api.upload_folder(
        folder_path=str(EXPORT_DIR),
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="feat: initial release of SoccerNet MES-5 VLM Benchmark & ReVal Telemetry",
        token=token
    )
    print(f"\n========================================================")
    print(f" SUCCESS: Published to https://huggingface.co/datasets/{repo_id}")
    print(f"========================================================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Publish SoccerNet MES-5 Benchmark to Hugging Face Hub")
    parser.add_argument("--repo-id", type=str, default="HaydenLundin/SoccerNet-MES5-VLM-Benchmark")
    parser.add_argument("--token", type=str, default=None, help="Hugging Face token (defaults to HF_TOKEN in env/.env)")
    parser.add_argument("--stage-only", action="store_true", default=False, help="Only stage local export directory")
    parser.add_argument("--upload", action="store_true", default=False, help="Upload staged directory to Hugging Face Hub")
    args = parser.parse_args()

    build_hf_release_bundle()

    if args.upload:
        token = args.token or get_hf_token()
        if not token:
            print("Error: HF_TOKEN not provided. Pass via --token, .env, or set HF_TOKEN environment variable.")
            sys.exit(1)
        upload_to_hub(args.repo_id, token)
    else:
        print("Run with --upload to push staged files to the Hugging Face Hub.")
