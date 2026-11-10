"""
Multi-Sport Schemas and Markov Transition Registry for Cross-Sport Video Understanding.
Provides pluggable profiles for:
- Soccer (EPL / FIFA): 17 classes, W=60s, S=45s
- American Football (NCAAF / NFL): 10 classes, W=30s, S=20s (40s play clock)
- Basketball (NCAAB / NBA): 10 classes, W=24s, S=16s (24s shot clock)
"""

from typing import List, Dict, Tuple, Optional, Any, Literal
from dataclasses import dataclass, field
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# 1. Action Classes per Sport
# ---------------------------------------------------------------------------

SOCCER_CLASSES: List[str] = [
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

FOOTBALL_CLASSES: List[str] = [
    "Snap",
    "Pass Completion",
    "Incompletion",
    "Rush / Tackle",
    "Touchdown",
    "Sack",
    "Turnover",
    "Field Goal / PAT",
    "Penalty Flag",
    "Timeout"
]

BASKETBALL_CLASSES: List[str] = [
    "Three-Point Made",
    "Two-Point FG / Dunk",
    "Free Throw",
    "Rebound",
    "Steal / Turnover",
    "Block",
    "Personal Foul",
    "Out of Bounds",
    "Substitution",
    "Timeout"
]

SPORT_CLASSES_MAP: Dict[str, List[str]] = {
    "soccer": SOCCER_CLASSES,
    "football": FOOTBALL_CLASSES,
    "basketball": BASKETBALL_CLASSES
}

# ---------------------------------------------------------------------------
# 2. Pydantic Event Models
# ---------------------------------------------------------------------------

class CrossSportEvent(BaseModel):
    """
    Standardized cross-sport spotted event with Chain-of-Thought commentary ordering.
    """
    timestamp: str = Field(
        ...,
        pattern=r"^\d{2}:[0-5]\d$",
        description="Event timestamp formatted as 'MM:SS' relative to clip/window start."
    )
    commentary: str = Field(
        ...,
        description="Concise observational play-by-play Chain-of-Thought reasoning describing visual actions."
    )
    label: str = Field(
        ...,
        description="Categorical action class from the sport's ontology."
    )
    confidence: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Calibrated detection confidence score."
    )
    visibility: str = Field(
        default="visible",
        description="View status: 'visible', 'replay', or 'not shown'."
    )
    sport: str = Field(
        default="soccer",
        description="Sport name: 'soccer', 'football', or 'basketball'."
    )


# ---------------------------------------------------------------------------
# 3. Sport Profile Specifications
# ---------------------------------------------------------------------------

@dataclass
class SportProfile:
    name: str
    classes: List[str]
    window_sec: float
    stride_sec: float
    play_clock_sec: Optional[float]
    whistle_freq_range: Tuple[float, float]
    whistle_threshold_z: float
    crowd_threshold_db: float
    optical_cut_mad: float
    optical_var_threshold: float
    markov_consequents: Dict[Tuple[str, str], Dict[str, float]]
    system_prompt_template: str


SOCCER_MARKOV_CONSEQUENTS: Dict[Tuple[str, str], Dict[str, float]] = {
    ("Goal", "Kick-off"): {"max_gap_sec": 60.0, "boost_factor": 1.40},
    ("Foul", "Yellow card"): {"max_gap_sec": 30.0, "boost_factor": 1.35},
    ("Foul", "Red card"): {"max_gap_sec": 30.0, "boost_factor": 1.25},
    ("Foul", "Direct free-kick"): {"max_gap_sec": 45.0, "boost_factor": 1.25},
    ("Foul", "Indirect free-kick"): {"max_gap_sec": 45.0, "boost_factor": 1.20},
    ("Shots on target", "Corner"): {"max_gap_sec": 30.0, "boost_factor": 1.20},
    ("Penalty", "Goal"): {"max_gap_sec": 15.0, "boost_factor": 1.30}
}

FOOTBALL_MARKOV_CONSEQUENTS: Dict[Tuple[str, str], Dict[str, float]] = {
    ("Touchdown", "Field Goal / PAT"): {"max_gap_sec": 45.0, "boost_factor": 1.50},
    ("Rush / Tackle", "Snap"): {"max_gap_sec": 40.0, "boost_factor": 1.35},
    ("Incompletion", "Snap"): {"max_gap_sec": 40.0, "boost_factor": 1.40},
    ("Sack", "Snap"): {"max_gap_sec": 40.0, "boost_factor": 1.30},
    ("Penalty Flag", "Timeout"): {"max_gap_sec": 30.0, "boost_factor": 1.25},
    ("Turnover", "Snap"): {"max_gap_sec": 60.0, "boost_factor": 1.25}
}

BASKETBALL_MARKOV_CONSEQUENTS: Dict[Tuple[str, str], Dict[str, float]] = {
    ("Personal Foul", "Free Throw"): {"max_gap_sec": 30.0, "boost_factor": 1.45},
    ("Block", "Rebound"): {"max_gap_sec": 5.0, "boost_factor": 1.40},
    ("Three-Point Made", "Timeout"): {"max_gap_sec": 20.0, "boost_factor": 1.30},
    ("Personal Foul", "Substitution"): {"max_gap_sec": 25.0, "boost_factor": 1.25},
    ("Out of Bounds", "Substitution"): {"max_gap_sec": 20.0, "boost_factor": 1.20}
}


def build_sport_profile(sport: str) -> SportProfile:
    sport_lower = sport.lower().strip()
    if sport_lower in ["football", "american_football", "nfl", "ncaaf"]:
        prompt = (
            "You are an expert American football video analytics AI. Analyze the broadcast video window with "
            "burned-in clock badges [00:SS]. Identify all football action events strictly from the following classes: "
            f"{', '.join(FOOTBALL_CLASSES)}. "
            "Emit a valid JSON list where each item has 'timestamp' (MM:SS), 'commentary' (<15 words factual description), "
            "'label' (exact class), 'confidence' (0.0-1.0), and 'visibility' ('visible', 'replay', or 'not shown')."
        )
        return SportProfile(
            name="football",
            classes=FOOTBALL_CLASSES,
            window_sec=30.0,
            stride_sec=20.0,
            play_clock_sec=40.0,
            whistle_freq_range=(3500.0, 4200.0),
            whistle_threshold_z=4.5,
            crowd_threshold_db=6.0,
            optical_cut_mad=30.0,
            optical_var_threshold=6.0,
            markov_consequents=FOOTBALL_MARKOV_CONSEQUENTS,
            system_prompt_template=prompt
        )

    elif sport_lower in ["basketball", "nba", "ncaab", "hoops"]:
        prompt = (
            "You are an expert basketball video analytics AI. Analyze the broadcast court video window with "
            "burned-in clock badges [00:SS]. Identify all basketball events strictly from the following classes: "
            f"{', '.join(BASKETBALL_CLASSES)}. "
            "Emit a valid JSON list where each item has 'timestamp' (MM:SS), 'commentary' (<15 words factual description), "
            "'label' (exact class), 'confidence' (0.0-1.0), and 'visibility' ('visible', 'replay', or 'not shown')."
        )
        return SportProfile(
            name="basketball",
            classes=BASKETBALL_CLASSES,
            window_sec=24.0,
            stride_sec=16.0,
            play_clock_sec=24.0,
            whistle_freq_range=(3200.0, 4000.0),
            whistle_threshold_z=4.0,
            crowd_threshold_db=5.5,
            optical_cut_mad=28.0,
            optical_var_threshold=5.5,
            markov_consequents=BASKETBALL_MARKOV_CONSEQUENTS,
            system_prompt_template=prompt
        )

    else:
        # Default: Soccer
        prompt = (
            "You are an expert association football video analytics AI. Analyze the broadcast video window with "
            "burned-in clock badges [00:SS]. Identify all soccer events strictly from the official 17 classes: "
            f"{', '.join(SOCCER_CLASSES)}. "
            "Emit a valid JSON list where each item has 'timestamp' (MM:SS), 'commentary' (<15 words factual description), "
            "'label' (exact class), 'confidence' (0.0-1.0), and 'visibility' ('visible', 'replay', or 'not shown')."
        )
        return SportProfile(
            name="soccer",
            classes=SOCCER_CLASSES,
            window_sec=60.0,
            stride_sec=45.0,
            play_clock_sec=None,
            whistle_freq_range=(3500.0, 4200.0),
            whistle_threshold_z=4.5,
            crowd_threshold_db=6.0,
            optical_cut_mad=30.0,
            optical_var_threshold=6.0,
            markov_consequents=SOCCER_MARKOV_CONSEQUENTS,
            system_prompt_template=prompt
        )


def apply_sport_markov_prior(
    predictions: List[Dict[str, Any]],
    sport: str
) -> List[Dict[str, Any]]:
    """
    Applies sport-specific Markov consequent boost without initiator degradation.
    """
    profile = build_sport_profile(sport)
    rules = profile.markov_consequents
    if not rules or not predictions:
        return predictions

    # Sort chronologically by position_sec (or position in ms)
    def get_pos_sec(p):
        if "pos_sec" in p:
            return float(p["pos_sec"])
        if "position" in p:
            return float(p["position"]) / 1000.0
        return 0.0

    sorted_preds = sorted(predictions, key=get_pos_sec)
    rescored = []

    for i, curr in enumerate(sorted_preds):
        curr_copy = dict(curr)
        curr_label = curr_copy.get("label")
        curr_time = get_pos_sec(curr_copy)
        curr_conf = float(curr_copy.get("confidence", 0.5))

        # Check prior initiators in reverse order
        max_boost = 1.0
        for j in range(i - 1, -1, -1):
            prev = sorted_preds[j]
            prev_label = prev.get("label")
            prev_time = get_pos_sec(prev)
            dt = curr_time - prev_time

            if dt < 0:
                continue

            pair = (prev_label, curr_label)
            if pair in rules:
                rule = rules[pair]
                if dt <= rule["max_gap_sec"]:
                    # Boost decayed linearly with time gap
                    decay = 1.0 - (dt / (rule["max_gap_sec"] * 1.5))
                    boost = 1.0 + (rule["boost_factor"] - 1.0) * max(0.2, decay)
                    if boost > max_boost:
                        max_boost = boost

            if dt > 70.0:  # Beyond horizon
                break

        if max_boost > 1.0:
            new_conf = min(0.99, curr_conf * max_boost)
            curr_copy["confidence"] = round(new_conf, 4)
            curr_copy["markov_boosted"] = True
            curr_copy["boost_factor"] = round(max_boost, 3)

        rescored.append(curr_copy)

    return rescored
