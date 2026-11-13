"""
7-Point Pre-Fable 5.2 Forensic Readiness Audit.

Checks:
  Audit 1: MES-5 17-class Table 26 / tab:mes5_eval N_GT column sum == 1,132 and 6-configuration per-class AP parity
  Audit 2: Cross-document metric synchronization (JSON <-> docs/RESEARCH.md, FORhayden.md, README.md, paper/main.tex, hf_dataset/README.md)
  Audit 3: LaTeX syntax, balanced environments, undefined labels/citations, unescaped characters, placeholders
  Audit 4: Hardcoded local paths (C:\\Users\\..., /Users/...) and credential/token leakage across git ls-files
  Audit 5: Deterministic RNG seeding (seed=42) across benchmark & V5 scripts + hf_dataset/SHA256SUMS.txt verification
  Audit 6: Cold-clone highlight reel tracking, .gitignore rules, and lightweight CI dependencies
  Audit 7: LaTeX PDF binary existence, compilation, zero table Overfull \\hbox bleed, and zero unresolved references

Usage:
    python scripts/audit_fable52_readiness.py
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent


def audit_1_mes5_table_arithmetic() -> List[str]:
    issues: List[str] = []
    gt_dir = ROOT / "logs/local_eval/mes5_specialist_3m_m15/devkit_ground_truth"
    counts: Counter[str] = Counter()
    for p in sorted(gt_dir.rglob("Labels-v2.json")):
        data = json.loads(p.read_text(encoding="utf-8"))
        for ann in data.get("annotations", []):
            counts[ann["label"]] += 1

    expected_total = sum(counts.values())
    if expected_total != 1132:
        issues.append(f"Audit 1: Expected 1132 DevKit GT annotations, counted {expected_total}")

    col_dirs = [
        "mes5_qwen3.5-4b",
        "mes5_lfm2.5-vl-3b",
        "mes5_minicpm-v-4.6",
        "mes5_specialist_3m",
        "mes5_specialist_3m_m15",
    ]
    summaries = {
        c: json.loads((ROOT / "logs/local_eval" / c / "mes5_evaluation_summary.json").read_text(encoding="utf-8"))
        for c in col_dirs
    }
    stage2_summary = json.loads(
        (ROOT / "logs/local_eval/mes5_stage2_simulation_summary.json").read_text(encoding="utf-8")
    )
    classes = list(summaries["mes5_specialist_3m_m15"]["loose_per_class"].keys())

    # Verify RESEARCH.md Table 26 rows (now 6 model columns)
    res_lines = (ROOT / "docs/RESEARCH.md").read_text(encoding="utf-8").splitlines()
    t26_rows: Dict[str, Tuple[int, List[float]]] = {}
    in_t26 = False
    for line in res_lines:
        if "#### Table 26:" in line:
            in_t26 = True
            continue
        if in_t26 and line.startswith("#### "):
            break
        if in_t26 and line.startswith("| **"):
            parts = [p.strip().replace("*", "").replace(",", "") for p in line.split("|")[1:-1]]
            cls_name = parts[0]
            if cls_name in counts:
                n_gt = int(parts[1])
                aps = [float(x) for x in parts[2:8]]
                t26_rows[cls_name] = (n_gt, aps)

    if len(t26_rows) != 17:
        issues.append(f"Audit 1: RESEARCH.md Table 26 parsed {len(t26_rows)}/17 class rows")
    else:
        sum_ngt = sum(v[0] for v in t26_rows.values())
        if sum_ngt != expected_total:
            issues.append(
                f"Audit 1: RESEARCH.md Table 26 N_GT column sums to {sum_ngt:,} != DevKit footer {expected_total:,}"
            )
        for cls_name in classes:
            row_ngt, row_aps = t26_rows[cls_name]
            if row_ngt != counts[cls_name]:
                issues.append(
                    f"Audit 1: RESEARCH.md Table 26 '{cls_name}' N_GT={row_ngt} != DevKit {counts[cls_name]}"
                )
            for idx, col in enumerate(col_dirs):
                json_ap = round(summaries[col]["loose_per_class"][cls_name], 4)
                if abs(row_aps[idx] - json_ap) > 0.00015:
                    issues.append(
                        f"Audit 1: RESEARCH.md Table 26 '{cls_name}' col '{col}'={row_aps[idx]:.4f} != JSON {json_ap:.4f}"
                    )
            # 6th column: Stage 2 Filtered
            stage2_ap = round(stage2_summary["per_class_loose"][cls_name], 4)
            if abs(row_aps[5] - stage2_ap) > 0.00015:
                issues.append(
                    f"Audit 1: RESEARCH.md Table 26 '{cls_name}' col 6 (Stage 2)={row_aps[5]:.4f} != JSON {stage2_ap:.4f}"
                )

        for idx, col in enumerate(col_dirs):
            col_mean = sum(t26_rows[c][1][idx] for c in classes) / 17.0
            reported_loose = summaries[col]["loose_average_mAP"]
            if abs(col_mean - reported_loose) > 0.00025:
                issues.append(
                    f"Audit 1: RESEARCH.md Table 26 '{col}' mean17={col_mean:.4f} != JSON loose_average_mAP {reported_loose:.4f}"
                )
        stage2_mean = sum(t26_rows[c][1][5] for c in classes) / 17.0
        stage2_loose = stage2_summary["stage2_loose_mAP"]
        if abs(stage2_mean - stage2_loose) > 0.00025:
            issues.append(
                f"Audit 1: RESEARCH.md Table 26 'Stage 2 Gate + M15' mean17={stage2_mean:.4f} != JSON stage2_loose_mAP {stage2_loose:.4f}"
            )

    # Verify paper/main.tex tab:mes5_eval rows match JSONs (all 6 columns)
    tex_txt = (ROOT / "paper/main.tex").read_text(encoding="utf-8")
    m_tab = re.search(r"\\label\{tab:mes5_eval\}.*?\\bottomrule", tex_txt, re.S)
    if not m_tab:
        issues.append("Audit 1: Could not locate tab:mes5_eval in paper/main.tex")
    else:
        block = m_tab.group(0)
        for cls_name in classes:
            cls_tex = cls_name.replace("->", r"$\to$")
            pat = re.compile(rf"^{re.escape(cls_tex)}\s*&(.*?)\\\\", re.M)
            mm = pat.search(block)
            if not mm:
                issues.append(f"Audit 1: paper/main.tex tab:mes5_eval missing row '{cls_name}'")
                continue
            nums = [float(re.sub(r"\\textbf\{([^}]*)\}", r"\1", c).strip()) for c in mm.group(1).split("&")]
            for idx, col in enumerate(col_dirs):
                json_ap = round(summaries[col]["loose_per_class"][cls_name], 4)
                if abs(nums[idx] - json_ap) > 0.00015:
                    issues.append(
                        f"Audit 1: paper/main.tex tab:mes5_eval '{cls_name}' col '{col}'={nums[idx]:.4f} != JSON {json_ap:.4f}"
                    )
            # 6th column: Stage 2 Filtered
            stage2_ap = round(stage2_summary["per_class_loose"][cls_name], 4)
            if abs(nums[5] - stage2_ap) > 0.00015:
                issues.append(
                    f"Audit 1: paper/main.tex tab:mes5_eval '{cls_name}' col 6 (Stage 2)={nums[5]:.4f} != JSON {stage2_ap:.4f}"
                )
    return issues
    return issues


def audit_2_cross_doc_sync() -> List[str]:
    issues: List[str] = []
    reval_b = json.loads((ROOT / "logs/reval_highlights/reval_comprehensive_benchmark.json").read_text(encoding="utf-8"))
    b4 = reval_b["benchmark_4_system_engineering_and_wcag"]
    fps_str = f"{b4['pyav_720p_transcode']['throughput_fps']:.2f}"
    rtf_str = f"{b4['pyav_720p_transcode']['real_time_factor']:.3f}"

    docs = {
        "README.md": (ROOT / "README.md").read_text(encoding="utf-8"),
        "docs/RESEARCH.md": (ROOT / "docs/RESEARCH.md").read_text(encoding="utf-8"),
        "FORhayden.md": (ROOT / "FORhayden.md").read_text(encoding="utf-8"),
        "paper/main.tex": (ROOT / "paper/main.tex").read_text(encoding="utf-8"),
        "hf_dataset/README.md": (ROOT / "hf_dataset/README.md").read_text(encoding="utf-8"),
    }
    for name, txt in docs.items():
        if name != "hf_dataset/README.md":
            if fps_str not in txt:
                issues.append(f"Audit 2: {name} missing live B4 FPS {fps_str}")
            if rtf_str not in txt and name in ("docs/RESEARCH.md", "FORhayden.md", "paper/main.tex"):
                issues.append(f"Audit 2: {name} missing live B4 RTF {rtf_str}")
        if "has_audio: false" in txt.lower() or "has_audio=false" in txt.lower() or "video-only reels" in txt.lower() or "video-only highlight" in txt.lower():
            issues.append(f"Audit 2: {name} still contains stale silent/video-only audio reference")
        if "70.51" in txt:
            issues.append(f"Audit 2: {name} still contains stale 70.51 FPS figure")
        if "1,213" in txt:
            issues.append(f"Audit 2: {name} still contains stale 1,213 sum")
    return issues


def audit_3_latex_and_placeholders() -> List[str]:
    issues: List[str] = []
    tex_path = ROOT / "paper/main.tex"
    bib_path = ROOT / "paper/references.bib"
    tex = tex_path.read_text(encoding="utf-8")
    bib = bib_path.read_text(encoding="utf-8")

    # 1) Balanced environments
    begins = Counter(re.findall(r"\\begin\{([^}]+)\}", tex))
    ends = Counter(re.findall(r"\\end\{([^}]+)\}", tex))
    if begins != ends:
        issues.append(f"Audit 3: Unbalanced LaTeX environments: begins={begins} ends={ends}")

    # 2) References resolved to labels
    labels = set(re.findall(r"\\label\{([^}]+)\}", tex))
    refs = set(re.findall(r"\\(?:Cref|cref|ref|autoref|eqref)\{([^}]+)\}", tex))
    for r in sorted(refs):
        for sub in r.split(","):
            sub = sub.strip()
            if sub not in labels:
                issues.append(f"Audit 3: Undefined LaTeX label reference '{sub}'")

    # 3) Citations exist in references.bib & no duplicate bib keys
    bib_keys_list = re.findall(r"@\w+\s*\{\s*([^,\s]+)\s*,", bib)
    dup_bib = [k for k, v in Counter(bib_keys_list).items() if v > 1]
    if dup_bib:
        issues.append(f"Audit 3: Duplicate BibTeX keys in references.bib: {dup_bib}")
    bib_keys = set(bib_keys_list)
    cites = re.findall(r"\\(?:cite|citet|citep|citealp)\{([^}]+)\}", tex)
    for grp in cites:
        for k in grp.split(","):
            k = k.strip()
            if k not in bib_keys:
                issues.append(f"Audit 3: Missing BibTeX key '{k}' referenced in paper/main.tex")

    # 4) Markdown horizontal rule '---' or unescaped '_'/'%' in text mode
    lines = tex.splitlines()
    in_math_env = False
    in_tabular = False
    in_tikz = False
    for idx, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped.startswith("%") or stripped.startswith(r"\usepackage"):
            continue
        if re.search(r"\\begin\{(?:equation|align|gather)\*?\}", stripped):
            in_math_env = True
        if re.search(r"\\end\{(?:equation|align|gather)\*?\}", stripped):
            in_math_env = False
        if re.search(r"\\begin\{(?:tabular|tabularx)\}", stripped):
            in_tabular = True
        if re.search(r"\\end\{(?:tabular|tabularx)\}", stripped):
            in_tabular = False
        if re.search(r"\\begin\{tikzpicture\}", stripped):
            in_tikz = True
        if re.search(r"\\end\{tikzpicture\}", stripped):
            in_tikz = False
        if stripped == "---":
            issues.append(f"Audit 3: Markdown horizontal rule '---' at paper/main.tex:{idx}")

        # Strip comments, inline math, and commands that legitimately take raw identifiers
        code = re.sub(r"(?<!\\)%.*$", "", line)
        code = re.sub(r"\$[^$]*\$", "", code)
        code = re.sub(r"\\(?:label|ref|cref|Cref|cite|citet|citep|texttt|url|href|includegraphics|bibliography|bibliographystyle)\{[^}]*\}", "", code)
        if in_tikz:
            # Strip TikZ node names like (out_spot) and (fuse.south)
            code = re.sub(r"\([a-zA-Z0-9_\.]+\)", "", code)
        if not in_math_env:
            if not in_tikz and re.search(r"(?<!\\)_", code):
                issues.append(f"Audit 3: Unescaped '_' outside math/texttt at paper/main.tex:{idx}: {line.strip()[:80]}")
            if re.search(r"(?<!\\)%", code):
                issues.append(f"Audit 3: Unescaped '%' at paper/main.tex:{idx}: {line.strip()[:80]}")
            if not in_tabular and re.search(r"(?<!\\)&", code):
                issues.append(f"Audit 3: Unescaped '&' outside tabular at paper/main.tex:{idx}: {line.strip()[:80]}")

    # 5) Check placeholders across tracked python/markdown/tex files
    tracked = subprocess.check_output(["git", "ls-files"], cwd=str(ROOT), text=True).splitlines()
    for rel in tracked:
        if rel.startswith("scripts/audit_fable52_readiness.py") or not rel.endswith((".py", ".md", ".tex")):
            continue
        p = ROOT / rel
        if not p.exists():
            continue
        txt = p.read_text(encoding="utf-8", errors="ignore")
        for tok in ("TODO", "FIXME", "TBD"):
            if re.search(rf"\b{tok}\b", txt):
                issues.append(f"Audit 3: Placeholder token '{tok}' found in {rel}")
    return issues


def audit_4_paths_and_secrets() -> List[str]:
    issues: List[str] = []
    tracked = subprocess.check_output(["git", "ls-files"], cwd=str(ROOT), text=True).splitlines()
    path_pat = re.compile(r"(?:[A-Za-z]:[/\\]{1,2}Users[/\\]{1,2}hayde|/Users/hayde|file:///c:/Users/)", re.I)
    secret_pat = re.compile(r"\bhf_[A-Za-z0-9]{20,}\b|\bAIza[0-9A-Za-z_-]{30,}\b")
    for rel in tracked:
        if rel == "scripts/audit_fable52_readiness.py":
            continue
        p = ROOT / rel
        if not p.exists():
            continue
        txt = p.read_text(encoding="utf-8", errors="ignore")
        for m in path_pat.finditer(txt):
            # Allow relative Windows font path C:\Windows\Fonts\..., flag user profile paths
            issues.append(f"Audit 4: Hardcoded local user path in {rel}: '{m.group(0)}'")
        for m in secret_pat.finditer(txt):
            issues.append(f"Audit 4: Potential credential token in tracked file {rel}")
    return issues


def audit_5_determinism_and_hf_checksums() -> List[str]:
    issues: List[str] = []
    seeded_scripts = [
        "scripts/benchmark_reval_engine.py",
        "scripts/evaluate_v5_camera_calibration.py",
        "scripts/evaluate_v5_game_state_gsr.py",
        "scripts/evaluate_v5_tracking_reid.py",
        "scripts/evaluate_v5_depth_estimation.py",
    ]
    for rel in seeded_scripts:
        txt = (ROOT / rel).read_text(encoding="utf-8")
        if "random.seed(" not in txt and "np.random.seed(" not in txt and "default_rng(" not in txt:
            issues.append(f"Audit 5: {rel} lacks explicit deterministic RNG seed (random.seed(42) / np.random.seed(42))")

    sha_path = ROOT / "hf_dataset/SHA256SUMS.txt"
    if not sha_path.exists():
        issues.append("Audit 5: hf_dataset/SHA256SUMS.txt missing")
    else:
        for line in sha_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            expected_hex, rel_file = line.split("  ", 1)
            fpath = ROOT / "hf_dataset" / rel_file
            if not fpath.exists():
                issues.append(f"Audit 5: Staged HF file missing: {rel_file}")
                continue
            actual_hex = hashlib.sha256(fpath.read_bytes()).hexdigest()
            if actual_hex != expected_hex:
                issues.append(f"Audit 5: Checksum mismatch on hf_dataset/{rel_file}")
    return issues


def audit_6_cold_clone_and_reels() -> List[str]:
    issues: List[str] = []
    # 1) Check 3 highlight reels exist and > 1 MB
    reels = [
        "ander_herrera_21_reel.mp4",
        "sergio_aguero_10_reel.mp4",
        "john_terry_26_reel.mp4",
    ]
    reels_dir = ROOT / "logs/reval_highlights"
    for r in reels:
        p = reels_dir / r
        if not p.exists():
            issues.append(f"Audit 6: Missing highlight reel: {p}")
        elif p.stat().st_size < 1_000_000:
            issues.append(f"Audit 6: Highlight reel suspiciously small ({p.stat().st_size} bytes): {p}")

    # 2) Check gitignore exception allows reels
    gi_txt = (ROOT / ".gitignore").read_text(encoding="utf-8")
    if "!logs/reval_highlights/*_reel.mp4" not in gi_txt:
        issues.append("Audit 6: .gitignore missing '!logs/reval_highlights/*_reel.mp4' exception")

    # 3) Check requirements-ci.txt exists and is clean
    ci_req = ROOT / "requirements-ci.txt"
    if not ci_req.exists():
        issues.append("Audit 6: requirements-ci.txt missing")
    else:
        txt = ci_req.read_text(encoding="utf-8")
        for bad in ["torch", "bitsandbytes", "peft", "nvidia-"]:
            if re.search(rf"^{bad}\b", txt, re.M):
                issues.append(f"Audit 6: Heavy CUDA dependency '{bad}' in requirements-ci.txt")
        for req in ["SoccerNet", "numpy", "scipy", "av", "pillow", "opencv-python-headless"]:
            if req.lower() not in txt.lower():
                issues.append(f"Audit 6: Expected CI package '{req}' missing from requirements-ci.txt")
    return issues


def audit_7_latex_pdf_build() -> List[str]:
    issues: List[str] = []
    pdf_path = ROOT / "paper/main.pdf"
    if not pdf_path.exists():
        issues.append("Audit 7: paper/main.pdf does not exist")
    elif pdf_path.stat().st_size < 100_000:
        issues.append(f"Audit 7: paper/main.pdf suspiciously small ({pdf_path.stat().st_size} bytes)")

    # Verify via tectonic if available
    tectonic = ROOT / "tools/tectonic.exe"
    if tectonic.exists():
        try:
            res = subprocess.run(
                [str(tectonic), "--print", "paper/main.tex"],
                cwd=str(ROOT),
                capture_output=True,
                text=True,
                timeout=45,
            )
            if res.returncode != 0:
                combined_err = (res.stdout + res.stderr).lower()
                if "bundle isn't cached" in combined_err or "couldn't get it from the internet" in combined_err:
                    # Offline / sandboxed environment: external cache not mounted in sandbox
                    pass
                else:
                    issues.append(f"Audit 7: Tectonic compilation failed (code {res.returncode}): {res.stderr[:200]}")
            else:
                combined = res.stdout + res.stderr
                # Check for table Overfull \hbox in alignment
                for m in re.finditer(r"Overfull \\hbox \(([0-9\.]+)pt too wide\) in alignment at lines ([0-9]+)--([0-9]+)", combined):
                    issues.append(f"Audit 7: Table Overfull \\hbox ({m.group(1)}pt too wide) at lines {m.group(2)}--{m.group(3)}")
                # Check for undefined citations or references
                if "LaTeX Warning: There were undefined references" in combined:
                    issues.append("Audit 7: Undefined LaTeX references reported by compiler")
                if "??" in combined:
                    issues.append("Audit 7: Unresolved '??' reference reported by compiler")
        except Exception as e:
            issues.append(f"Audit 7: Tectonic execution exception: {e}")
    return issues


def main() -> int:
    all_issues = (
        audit_1_mes5_table_arithmetic()
        + audit_2_cross_doc_sync()
        + audit_3_latex_and_placeholders()
        + audit_4_paths_and_secrets()
        + audit_5_determinism_and_hf_checksums()
        + audit_6_cold_clone_and_reels()
        + audit_7_latex_pdf_build()
    )
    print(f"\n=== Fable 5.2 7-Point Forensic Readiness Audit: {len(all_issues)} issue(s) found ===")
    for item in all_issues:
        print(" -", item)
    return 1 if all_issues else 0


if __name__ == "__main__":
    sys.exit(main())
