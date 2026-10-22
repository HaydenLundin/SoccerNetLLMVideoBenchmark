"""
processing/ball_action_schema.py
Unified schema and validation harness for SoccerNet Ball Action Spotting (BAS).
Enforces the 12 canonical fine-grained ball-interaction classes with tight 1.0s temporal tolerance.
"""

import json
import re
from typing import List, Dict, Any, Optional, Tuple, Literal
from pydantic import BaseModel, Field, field_validator

SOCCERNET_BALL_ACTION_CLASSES = [
    "Pass",
    "Drive",
    "Header",
    "High Pass",
    "Out",
    "Cross",
    "Throw In",
    "Shot",
    "Ball Player Block",
    "Player Successful Tackle",
    "Free Kick",
    "Goal"
]

BallActionLiteral = Literal[
    "Pass",
    "Drive",
    "Header",
    "High Pass",
    "Out",
    "Cross",
    "Throw In",
    "Shot",
    "Ball Player Block",
    "Player Successful Tackle",
    "Free Kick",
    "Goal"
]


class SpottedBallAction(BaseModel):
    """A fine-grained ball interaction event with exact timestamp and physical context."""
    timestamp: str = Field(description="MM:SS within the window")
    commentary: str = Field(default="", description="Chain-of-thought observational evidence")
    label: BallActionLiteral
    confidence: float = Field(default=0.85, ge=0.0, le=1.0)
    team: Optional[str] = Field(default=None, description="Optional executing team: 'home', 'away', or None")

    @field_validator("timestamp")
    @classmethod
    def validate_timestamp_format(cls, v: str) -> str:
        v_clean = v.strip().strip("[]\"'")
        parts = v_clean.split(":")
        if len(parts) != 2:
            raise ValueError(f"Timestamp must be in MM:SS format. Got: {v}")
        if not (parts[0].isdigit() and parts[1].isdigit()):
            raise ValueError(f"MM and SS must be integers. Got: {v}")
        return f"{int(parts[0]):02d}:{int(parts[1]):02d}"


def normalize_ball_action_label(lbl_raw: str) -> Optional[str]:
    """Robust normalizer mapping freeform VLM output to the 12 canonical BAS classes."""
    if not lbl_raw or not isinstance(lbl_raw, str):
        return None
    first_token = lbl_raw.split(",")[0].strip().rstrip(".")
    lbl_clean = first_token.lower()
    lbl_alphanumeric = re.sub(r"[^a-z0-9]", "", lbl_clean)

    # 1. Exact match
    for c in SOCCERNET_BALL_ACTION_CLASSES:
        if c.lower() == lbl_clean:
            return c

    # 2. Alphanumeric match (e.g. "ThrowIn" -> "Throw In", "HighPass" -> "High Pass")
    for c in SOCCERNET_BALL_ACTION_CLASSES:
        c_alphanumeric = re.sub(r"[^a-z0-9]", "", c.lower())
        if c_alphanumeric == lbl_alphanumeric:
            return c

    # 3. Domain synonym mappings from V2 classes
    synonyms = {
        "throw-in": "Throw In",
        "throwin": "Throw In",
        "balloutofplay": "Out",
        "ball out of play": "Out",
        "clearance": "Ball Player Block",
        "shots on target": "Shot",
        "shots off target": "Shot",
        "foul": "Player Successful Tackle",
        "corner": "Cross",
        "direct free-kick": "Free Kick",
        "indirect free-kick": "Free Kick",
        "freekick": "Free Kick",
        "tackle": "Player Successful Tackle"
    }
    if lbl_clean in synonyms:
        return synonyms[lbl_clean]
    if lbl_alphanumeric in synonyms:
        return synonyms[lbl_alphanumeric]

    return None


def build_ball_action_window_prompt(window_sec: int) -> str:
    """Constructs prompt focusing attention on the 12 discrete ball interactions."""
    clip_end_ts = f"{window_sec // 60:02d}:{window_sec % 60:02d}"
    classes_list_str = ", ".join([f"'{c}'" for c in SOCCERNET_BALL_ACTION_CLASSES])
    prompt = (
        f"You are an expert FIFA ball-tracking analyst.\n"
        f"Observe the video frames ({window_sec} seconds, from 00:00 to {clip_end_ts}).\n"
        f"Each frame has a digital clock '[00:SS]' burned in the top-left corner.\n"
        f"Task: Identify discrete, fine-grained BALL ACTIONS occurring in this window.\n"
        f"Allowed Classes:\n{classes_list_str}.\n\n"
        f"Definitions:\n"
        f"- 'Pass': Ground pass between teammates.\n"
        f"- 'Drive': Player carrying/dribbling the ball forward under control.\n"
        f"- 'Header': Intentional aerial head strike on the ball.\n"
        f"- 'High Pass': Long aerial lofted ball or cross-field switch.\n"
        f"- 'Out': Ball completely crosses touchline or goal line.\n"
        f"- 'Cross': Ball played into the penalty box from a wide wing position.\n"
        f"- 'Throw In': Restart by throwing ball in from sideline.\n"
        f"- 'Shot': Intentional strike at the opponent goal.\n"
        f"- 'Ball Player Block': Defending player blocks a pass or shot.\n"
        f"- 'Player Successful Tackle': Defender cleanly dispossesses opponent of the ball.\n"
        f"- 'Free Kick': Direct or indirect restart following a foul.\n"
        f"- 'Goal': Ball enters the net.\n\n"
        f"Output ONLY a valid JSON array matching this schema:\n"
        f'[\n'
        f'  {{\n'
        f'    "timestamp": "00:SS",\n'
        f'    "commentary": "Brief observational action description...",\n'
        f'    "label": "Pass",\n'
        f'    "confidence": 0.85\n'
        f'  }}\n'
        f']\n'
        f"If no discrete ball action occurs, return []."
    )
    return prompt


def parse_ball_action_model_output(
    raw_text: str,
    window_sec: int
) -> Tuple[List[SpottedBallAction], Dict[str, Any]]:
    """Extracts and validates ball action spots from raw text."""
    diag = {
        "raw_length": len(raw_text),
        "total_candidates": 0,
        "valid_kept": 0,
        "strategy": "none"
    }
    valid_events: List[SpottedBallAction] = []

    def process_item(item: dict):
        if not isinstance(item, dict):
            return
        diag["total_candidates"] += 1
        raw_lbl = str(item.get("label", ""))
        norm_lbl = normalize_ball_action_label(raw_lbl)
        if not norm_lbl:
            return
        item["label"] = norm_lbl

        raw_ts = str(item.get("timestamp", "")).strip().strip("[]\"'")
        parts = raw_ts.split(":")
        if len(parts) == 2 and parts[0].isdigit() and parts[1].split(".")[0].isdigit():
            mm, ss = int(parts[0]), int(float(parts[1]))
        elif len(parts) == 3 and parts[1].isdigit() and parts[2].split(".")[0].isdigit():
            mm, ss = int(parts[1]), int(float(parts[2]))
        elif len(parts) == 1 and raw_ts.replace(".", "", 1).isdigit():
            tot = int(float(raw_ts))
            mm, ss = tot // 60, tot % 60
        else:
            return
        item["timestamp"] = f"{mm:02d}:{ss:02d}"
        sec = mm * 60.0 + ss
        if sec < 0.0 or sec > float(window_sec + 2.0):
            return

        item.setdefault("confidence", 0.85)
        item.setdefault("commentary", "Observed ball interaction")
        try:
            ev = SpottedBallAction(**item)
            valid_events.append(ev)
        except Exception:
            pass

    cleaned = raw_text.strip()
    if "```json" in cleaned:
        cleaned = cleaned.split("```json")[1].split("```")[0].strip()
    elif "```" in cleaned:
        cleaned = cleaned.split("```")[1].split("```")[0].strip()

    arr_match = re.search(r"\[\s*\{.*\}\s*\]", cleaned, re.DOTALL)
    if arr_match:
        try:
            arr = json.loads(arr_match.group(0))
            if isinstance(arr, list):
                diag["strategy"] = "json_array"
                for it in arr:
                    process_item(it)
        except Exception:
            pass

    if len(valid_events) == 0:
        obj_matches = re.findall(r"\{[^{}]*\}", cleaned, re.DOTALL)
        for om in obj_matches:
            try:
                it = json.loads(om)
                if isinstance(it, dict) and ("timestamp" in it or "label" in it):
                    diag["strategy"] = "json_objects"
                    process_item(it)
            except Exception:
                pass

    diag["valid_kept"] = len(valid_events)
    return valid_events, diag
