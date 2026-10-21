import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
logs_dir = REPO_ROOT / "logs"

master_file = logs_dir / "live_memory_gate_all_tracks_master.json"
with open(master_file, "r", encoding="utf-8") as f:
    d = json.load(f)

print("\n| Model | Prec | Frames | Peak VRAM (GiB) | Latency (s) | Schema | Verdict |")
print("| :--- | :--- | :---: | :---: | :---: | :---: | :---: |")
for e in d:
    m = e.get("model_name") or e.get("model_key")
    p = e.get("precision", "-")
    fc = e.get("frame_count", "-")
    pk = f"{e.get('peak_reserved_gib', 0.0):.2f}"
    lat = f"{e.get('wall_clock_sec', 0.0):.2f}"
    sch = "VALID" if e.get("schema_valid") else "FAIL"
    v = "PASS" if e.get("passed") else "FAIL"
    if e.get("verdict") in ("OOM_FAIL", "PARSE_FAIL"):
        v = e.get("verdict")
    print(f"| {m} | {p} | {fc} | {pk} | {lat} | {sch} | **{v}** |")
