import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
logs_dir = REPO_ROOT / "logs"

master_results = []

log_files = [
    logs_dir / "live_memory_gate_wave2.json",
    logs_dir / "live_memory_gate_wave1.json",
    logs_dir / "live_memory_gate_internvl_4bit.json",
    logs_dir / "live_memory_gate_gemma_e2b.json",
    logs_dir / "live_memory_gate_lfm2_5.json",
    logs_dir / "live_memory_gate_ministral_3b.json",
    logs_dir / "live_memory_gate_moondream2.json",
    logs_dir / "live_memory_gate_qwen2_5_vl_7b.json",
]

seen_keys = set()

for lf in log_files:
    if not lf.exists():
        continue
    with open(lf, "r", encoding="utf-8") as f:
        data = json.load(f)
    for entry in data:
        m = entry.get("model_key")
        prec = entry.get("precision")
        frames = entry.get("frame_count")
        key = (m, prec, frames)
        if key not in seen_keys:
            seen_keys.add(key)
            master_results.append(entry)

out_file = logs_dir / "live_memory_gate_all_tracks_master.json"
with open(out_file, "w", encoding="utf-8") as f:
    json.dump(master_results, f, indent=2)

print(f"Master results consolidated: {len(master_results)} trials saved to {out_file}")
