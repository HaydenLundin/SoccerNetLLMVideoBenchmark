"""
Unified Response Schema and Parsing for SoccerNet V2 Benchmark
Integrates Classic Action Spotting (Labels-v2, 17 classes) and Dense Video Captioning (SoccerNet-Caption).
"""

from typing import List, Literal, Tuple, Optional
from pydantic import BaseModel, Field

# The exact 17 official SoccerNet Labels-v2 classes
SoccerNetActionLabel = Literal[
    "Penalty",
    "Kick-off",
    "Goal",
    "Substitution",
    "Offside",
    "Shots on target",
    "Shots off target",
    "Clearance",
    "Ball out of play",
    "Throw-in",
    "Foul",
    "Indirect free-kick",
    "Direct free-kick",
    "Corner",
    "Yellow card",
    "Red card",
    "Yellow->red card"
]

SOCCERNET_CLASSES: List[str] = [
    "Penalty",
    "Kick-off",
    "Goal",
    "Substitution",
    "Offside",
    "Shots on target",
    "Shots off target",
    "Clearance",
    "Ball out of play",
    "Throw-in",
    "Foul",
    "Indirect free-kick",
    "Direct free-kick",
    "Corner",
    "Yellow card",
    "Red card",
    "Yellow->red card"
]

VisibilityType = Literal["visible", "replay", "not shown"]

class SpottedEvent(BaseModel):
    """
    Spotted soccer event.
    NOTE: 'commentary' is ordered BEFORE 'label' so that autoregressive token generation
    produces natural-language observational reasoning (Chain-of-Thought) before predicting
    the categorical action class.
    """
    timestamp: str = Field(
        ...,
        pattern=r"^\d{2}:[0-5]\d$",
        description="Event timestamp formatted strictly as 'MM:SS' relative to the video clip start (e.g. '04:12', '10:05'). Seconds must be 00-59."
    )
    commentary: str = Field(
        ...,
        description=(
            "Detailed, factual play-by-play description of the event for SoccerNet-Caption. "
            "Describe the player movement, ball trajectory, contact, or referee action. "
            "This observational evidence grounds the subsequent categorical label."
        )
    )
    label: SoccerNetActionLabel = Field(
        ...,
        description="The exact categorical action class from the official 17 SoccerNet Labels-v2 classes."
    )
    confidence: float = Field(
        default=0.85,
        description=(
            "Calibrated confidence score between 0.0 and 1.0 based on visual clarity, "
            "referee confirmation, and unobstructed camera view."
        ),
        ge=0.0,
        le=1.0
    )
    visibility: VisibilityType = Field(
        default="visible",
        description="Camera view of event: 'visible' during live play, 'replay', or 'not shown' (deduced from context)."
    )

class SpottedEventMinimal(BaseModel):
    """
    Spotting-only soccer event (RQ3 Ablation).
    Omits 'commentary' so categorical prediction is emitted directly without prior
    natural-language Chain-of-Thought (CoT) reasoning.
    """
    timestamp: str = Field(
        ...,
        pattern=r"^\d{2}:[0-5]\d$",
        description="Event timestamp formatted strictly as 'MM:SS' relative to the video clip start (e.g. '04:12', '10:05'). Seconds must be 00-59."
    )
    label: SoccerNetActionLabel = Field(
        ...,
        description="The exact categorical action class from the official 17 SoccerNet Labels-v2 classes."
    )
    confidence: float = Field(
        default=0.85,
        description="Calibrated confidence score between 0.0 and 1.0.",
        ge=0.0,
        le=1.0
    )
    visibility: VisibilityType = Field(
        default="visible",
        description="Camera view of event: 'visible' during live play, 'replay', or 'not shown'."
    )
    commentary: str = Field(
        default="",
        description="Empty commentary placeholder for spotting-only ablation compatibility."
    )

class SoccerClipAnnotation(BaseModel):
    clip_index: int = Field(default=0, description="Index of this macro-window clip (0 to N)")
    half: int = Field(..., description="Match half (1 or 2)", ge=1, le=2)
    events: List[SpottedEvent] = Field(
        default_factory=list,
        description="Chronologically sorted list of spotted events within this video clip."
    )

def parse_mm_ss_to_seconds(time_str: str) -> float:
    """Safely converts strict MM:SS into total seconds float."""
    import re
    if not re.match(r"^\d{2}:[0-5]\d$", time_str.strip()):
        raise ValueError(f"Invalid timestamp format '{time_str}'. Must strictly match 'MM:SS' (e.g. '04:12', '10:05').")
    parts = time_str.strip().split(":")
    return float(parts[0]) * 60.0 + float(parts[1])

def convert_clip_events_to_soccernet_format(
    events: List[SpottedEvent],
    half: int,
    clip_start_sec: float = 0.0,
    clip_duration_sec: Optional[float] = None
) -> Tuple[List[dict], List[dict], int]:
    """
    Converts LLM output events into official SoccerNet formats:
    1. Action Spotting (Predictions-v2 format with position in milliseconds and gameTime).
    2. Dense Video Captioning (SoccerNet-Caption format).
    3. Fault-isolated bounds validation: drops out-of-bounds timestamps individually without failing clip.
    Returns: (spotting_predictions, caption_predictions, out_of_bounds_drops)
    """
    spotting_predictions = []
    caption_predictions = []
    out_of_bounds_drops = 0

    for ev in events:
        try:
            local_sec = parse_mm_ss_to_seconds(ev.timestamp)
        except ValueError:
            out_of_bounds_drops += 1
            continue

        # Event-level bounds check against actual clip duration from manifest
        if clip_duration_sec is not None:
            if local_sec < 0.0 or local_sec > clip_duration_sec:
                out_of_bounds_drops += 1
                continue

        total_sec = clip_start_sec + local_sec
        position_ms = int(round(total_sec * 1000.0))

        tot_min = int(total_sec // 60)
        tot_sec = int(round(total_sec % 60))
        game_time = f"{half} - {tot_min:02d}:{tot_sec:02d}"

        # 1. Action Spotting Record
        spotting_predictions.append({
            "gameTime": game_time,
            "label": ev.label,
            "position": position_ms,
            "half": str(half),
            "confidence": float(ev.confidence),
            "visibility": ev.visibility
        })

        # 2. Dense Captioning Record
        caption_predictions.append({
            "gameTime": game_time,
            "position": position_ms,
            "half": str(half),
            "label": ev.label,
            "commentary": ev.commentary
        })

    # Sort strictly by position ascending
    spotting_predictions.sort(key=lambda x: x["position"])
    caption_predictions.sort(key=lambda x: x["position"])

    return spotting_predictions, caption_predictions, out_of_bounds_drops

SYSTEM_INSTRUCTION = """You are an expert soccer video analyst and official match annotator for the SoccerNet Action Spotting and Dense Captioning benchmark.

Task:
Analyze the broadcast video and identify all discrete soccer events.

Rules:
1. OUTPUT FORMAT: For each event:
   - 'timestamp': Output the exact time as 'MM:SS' relative to the start of this clip.
   - 'commentary': Provide a factual, one-sentence play-by-play description (who, action, ball movement, outcome). Generate this description first to ground the visual evidence.
   - 'label': Select strictly one of the 17 official SoccerNet classes:
     Penalty, Kick-off, Goal, Substitution, Offside, Shots on target, Shots off target,
     Clearance, Ball out of play, Throw-in, Foul, Indirect free-kick, Direct free-kick,
     Corner, Yellow card, Red card, Yellow->red card.
   - 'confidence': Continuous probability score (0.0 to 1.0) calibrated by visual certainty and referee gestures.
   - 'visibility': 'visible', 'replay', or 'not shown'.
2. TEMPORAL ANCHORING: Anchor the timestamp at the exact moment the action occurs (e.g., ball crossing line, referee whistle blow, tackle contact, card raised).
3. SUPPRESSION: Do not output duplicate events for the same action within a 3-second radius.
4. ORDER: Order events strictly in ascending chronological order by timestamp.
"""
