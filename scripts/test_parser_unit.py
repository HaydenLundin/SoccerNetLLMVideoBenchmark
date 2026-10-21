import json
import re
from typing import List, Tuple, Dict, Any, Optional

SOCCERNET_CLASSES = [
    "Penalty", "Kick-off", "Goal", "Substitution", "Offside",
    "Shots on target", "Shots off target", "Clearance", "Ball out of play",
    "Throw-in", "Foul", "Indirect free-kick", "Direct free-kick",
    "Corner", "Yellow card", "Red card", "Yellow->red card"
]

def normalize_soccernet_label(lbl_raw: str) -> Optional[str]:
    if not lbl_raw or not isinstance(lbl_raw, str):
        return None
    first_token = lbl_raw.split(",")[0].strip().rstrip(".")
    lbl_clean = first_token.lower()
    for c in SOCCERNET_CLASSES:
        if c.lower() == lbl_clean:
            return c
    for c in SOCCERNET_CLASSES:
        if c.lower().rstrip("s") == lbl_clean.rstrip("s"):
            return c
        if lbl_clean in c.lower() or c.lower() in lbl_clean:
            return c
    return None

def parse_model_json_output(raw_text: str, window_sec: int):
    diag = {
        "raw_length": len(raw_text),
        "strategy_used": "none",
        "total_candidates_found": 0,
        "valid_kept": 0,
        "out_of_bounds_drops": 0,
        "invalid_enum_drops": 0,
        "syntax_error": False,
        "empty_output": len(raw_text.strip()) == 0,
        "dropped_timestamps": [],
        "dropped_labels": []
    }
    if diag["empty_output"]:
        return [], diag

    valid_events = []

    def process_item(item: dict):
        if not isinstance(item, dict):
            return
        diag["total_candidates_found"] += 1
        item.setdefault("confidence", 0.85)
        item.setdefault("visibility", "visible")
        item.setdefault("commentary", "Observational event")

        raw_lbl = str(item.get("label", ""))
        norm_lbl = normalize_soccernet_label(raw_lbl)
        if not norm_lbl:
            diag["invalid_enum_drops"] += 1
            diag["dropped_labels"].append(raw_lbl)
            return
        item["label"] = norm_lbl

        raw_ts = str(item.get("timestamp", "")).strip().strip("[]\"'")
        parts = raw_ts.split(":")
        if len(parts) != 2 or not parts[0].isdigit() or not parts[1].isdigit():
            diag["out_of_bounds_drops"] += 1
            diag["dropped_timestamps"].append(raw_ts)
            return
        mm, ss = int(parts[0]), int(parts[1])
        item["timestamp"] = f"{mm:02d}:{ss:02d}"
        sec = mm * 60.0 + ss
        if sec < 0.0 or sec > float(window_sec + 2.0):
            diag["out_of_bounds_drops"] += 1
            diag["dropped_timestamps"].append(item["timestamp"])
            return
        valid_events.append(item)

    cleaned = raw_text.strip()
    if "```json" in cleaned:
        cleaned = cleaned.split("```json")[1].split("```")[0].strip()
    elif "```" in cleaned:
        cleaned = cleaned.split("```")[1].split("```")[0].strip()

    # Strategy 1: JSON Array
    array_match = re.search(r"\[\s*\{.*\}\s*\]", cleaned, re.DOTALL)
    if array_match:
        try:
            arr = json.loads(array_match.group(0))
            if isinstance(arr, list):
                diag["strategy_used"] = "json_array"
                for it in arr:
                    process_item(it)
        except Exception:
            diag["syntax_error"] = True

    # Strategy 2: Individual JSON Objects Fallback
    if len(valid_events) == 0:
        obj_matches = re.findall(r"\{[^{}]*\}", cleaned, re.DOTALL)
        for om in obj_matches:
            sanitized = re.sub(r",\s*\}", "}", om.strip())
            try:
                it = json.loads(sanitized)
                if isinstance(it, dict) and ("timestamp" in it or "label" in it):
                    diag["strategy_used"] = "json_objects"
                    process_item(it)
            except Exception:
                pass

    # Strategy 3: Markdown table fallback
    if len(valid_events) == 0 and "|" in cleaned:
        lines = [l.strip() for l in cleaned.splitlines() if l.strip().startswith("|") and not l.strip().startswith("|---")]
        for line in lines:
            cols = [c.strip() for c in line.split("|")[1:-1]]
            if len(cols) >= 3:
                ts_raw, comm, lbl = cols[0], cols[1], cols[2]
                if "timestamp" in ts_raw.lower() or "label" in lbl.lower():
                    continue
                diag["strategy_used"] = "markdown_table"
                process_item({"timestamp": ts_raw, "commentary": comm, "label": lbl})

    diag["valid_kept"] = len(valid_events)
    return valid_events, diag

if __name__ == "__main__":
    t1 = '[{"timestamp": "00:15", "label": "Corner", "commentary": "Player crosses ball"}, {"timestamp": "02:30", "label": "Goal", "commentary": "Ball in net"}]'
    evs, d1 = parse_model_json_output(t1, 60)
    print("T1 kept:", len(evs), "OOB drops:", d1["out_of_bounds_drops"], "strategy:", d1["strategy_used"])
    assert len(evs) == 1 and evs[0]["label"] == "Corner" and d1["out_of_bounds_drops"] == 1

    t2 = '''Here are the events spotted:
{"timestamp": "00:20", "label": "Throw-in, Ball out of play", "commentary": "Player throws ball"}
{"timestamp": "00:45", "label": "Clearance", "commentary": "Defensive clear"}'''
    evs, d2 = parse_model_json_output(t2, 60)
    print("T2 kept:", len(evs), "labels:", [e["label"] for e in evs], "strategy:", d2["strategy_used"])
    assert len(evs) == 2 and evs[0]["label"] == "Throw-in" and evs[1]["label"] == "Clearance"

    t3 = '''| Timestamp | Action Description | Label |
|---|---|---|
| 00:10 | Defender kicks ball away | Clearance |
| 00:35 | Referee blows for tackle | Foul |'''
    evs, d3 = parse_model_json_output(t3, 60)
    print("T3 kept:", len(evs), "labels:", [e["label"] for e in evs], "strategy:", d3["strategy_used"])
    assert len(evs) == 2 and evs[0]["label"] == "Clearance" and evs[1]["label"] == "Foul"

    print("ALL UNIT TESTS PASSED!")
