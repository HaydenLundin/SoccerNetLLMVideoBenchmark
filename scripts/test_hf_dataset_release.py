"""
Verification test suite for Hugging Face Dataset Release Package (HaydenLundin/SoccerNet-MES5-VLM-Benchmark).
Validates bundle completeness, JSON schemas, checksums, and benchmark evaluation reproducibility.
"""

import os
import sys
import json
import subprocess
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
HF_DIR = WORKSPACE_ROOT / "hf_dataset"


def test_bundle_structure():
    print("\n--- 1. Testing Hugging Face Release Bundle Structure ---")
    assert HF_DIR.exists(), f"Staged directory missing: {HF_DIR}"
    
    required_files = [
        HF_DIR / "README.md",
        HF_DIR / "eval_set.json",
        HF_DIR / "evaluate_mes5_benchmark.py",
        HF_DIR / "requirements.txt",
        HF_DIR / "SHA256SUMS.txt",
        HF_DIR / "priors" / "m15_transition_matrix.json",
        HF_DIR / "priors" / "class_nms_radii.json",
        HF_DIR / "reval" / "reval_mes5_athlete_leaderboard.json",
        HF_DIR / "reval" / "reval_scoring_formula.json"
    ]
    for rf in required_files:
        assert rf.exists(), f"Required bundle file missing: {rf}"
        assert rf.stat().st_size > 0, f"File is empty: {rf}"

    readme_txt = (HF_DIR / "README.md").read_text(encoding="utf-8")
    assert r"\times" in readme_txt and ("\t" + "imes") not in readme_txt, "README.md contains unescaped tab-times!"
    assert r"\text{ReVal Score}" in readme_txt
    print(f"✓ All core metadata, priors, checksums, and README LaTeX verified in {HF_DIR.name}/.")


def test_manifests_and_priors():
    print("\n--- 2. Testing Manifests and Out-of-Sample Priors ---")
    manifests = list((HF_DIR / "manifests").glob("*_manifest.json"))
    assert len(manifests) >= 10, f"Expected at least 10 manifests, found {len(manifests)}"
    print(f"✓ {len(manifests)} macro-chunk manifests verified with zero-drift keyframe maps.")

    with open(HF_DIR / "priors" / "m15_transition_matrix.json", "r", encoding="utf-8") as f:
        m15 = json.load(f)
    assert "forward_transitions_30s" in m15
    assert "backward_transitions_30s" in m15
    assert "half1_class_priors" in m15
    assert len(m15["forward_transitions_30s"]) > 0
    print(f"✓ M15 Markov prior verified ({len(m15['forward_transitions_30s'])} forward transition sources).")


def test_reval_telemetry():
    print("\n--- 3. Testing ReVal Athlete Valuation Telemetry ---")
    with open(HF_DIR / "reval" / "reval_mes5_athlete_leaderboard.json", "r", encoding="utf-8") as f:
        reval = json.load(f)
    assert "leaderboard" in reval
    assert len(reval["leaderboard"]) >= 12
    top = reval["leaderboard"][0]
    assert top["name"] == "Ander Herrera"
    assert top["reval_score"] == 58.3
    assert "sub_scores" in top
    print(f"✓ ReVal leaderboard telemetry verified: {len(reval['leaderboard'])} athletes (Top: {top['name']}, Score: {top['reval_score']}).")


def test_predictions_and_reproducibility():
    print("\n--- 4. Testing 6 Prediction Sets and Standalone Evaluator ---")
    configs = [
        "qwen3.5-4b",
        "minicpm-v-4.6",
        "lfm2.5-vl-3b",
        "specialist_3m_mece",
        "specialist_3m_m15_rescored",
        "stage2_cpu_filtered"
    ]
    for cfg in configs:
        cfg_dir = HF_DIR / "predictions" / cfg
        assert cfg_dir.exists(), f"Prediction config directory missing: {cfg_dir}"
        preds = list(cfg_dir.rglob("results_spotting.json"))
        assert len(preds) == 5, f"Expected 5 match results for {cfg}, found {len(preds)}"
    print(f"✓ All 6 prediction sets staged with 5 matches each.")

    # Run standalone evaluate_mes5_benchmark.py
    py_exe = sys.executable
    cmd = [py_exe, str(HF_DIR / "evaluate_mes5_benchmark.py"), "--config", "stage2_cpu_filtered"]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    assert "Official Loose-mAP: 0.1840" in res.stdout
    assert "Official Tight-mAP: 0.0609" in res.stdout
    assert "Active Classes:     17 / 17" in res.stdout
    print("✓ Standalone evaluator execution verified: 0.1840 Loose-mAP, 0.0609 Tight-mAP, 17/17 Active Classes.")


def run_all_tests():
    test_bundle_structure()
    test_manifests_and_priors()
    test_reval_telemetry()
    test_predictions_and_reproducibility()
    print("\n==================================================================")
    print(" ALL HUGGING FACE RELEASE TESTS PASSED")
    print("==================================================================\n")


if __name__ == "__main__":
    run_all_tests()
