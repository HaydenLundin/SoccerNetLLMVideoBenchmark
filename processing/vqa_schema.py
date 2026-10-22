"""
processing/vqa_schema.py
Unified schema and validation harness for SoccerNet Visual Question Answering (SN-VQA).
Implements standard 4-option multiple choice with Chain-of-Thought visual reasoning.
"""

import json
import re
from typing import Dict, Any, Optional, Literal, Tuple
from pydantic import BaseModel, Field, field_validator


class VQAQuestion(BaseModel):
    """A multiple-choice question anchored to a localized broadcast video clip."""
    question_id: str
    category: Literal[
        "action_spotting",
        "referee_decision",
        "tactical_setup",
        "temporal_continuity"
    ]
    question: str
    chunk_index: int = Field(default=0, ge=0, le=4)
    start_sec: float = Field(ge=0.0)
    end_sec: float = Field(ge=0.0)
    options: Dict[Literal["O1", "O2", "O3", "O4"], str]
    ground_truth_option: Literal["O1", "O2", "O3", "O4"]
    ground_truth_explanation: str

    @field_validator("options")
    @classmethod
    def validate_options(cls, v):
        req = {"O1", "O2", "O3", "O4"}
        if set(v.keys()) != req:
            raise ValueError(f"Options must contain exactly O1, O2, O3, O4. Got: {list(v.keys())}")
        return v


class VQAResponse(BaseModel):
    """The structured response emitted by the VLM for a VQA item."""
    thought: str = Field(description="Visual Chain-of-Thought reasoning describing key video evidence")
    selected_option: Literal["O1", "O2", "O3", "O4"] = Field(description="The selected multiple-choice option ID")
    confidence: float = Field(default=0.85, ge=0.0, le=1.0)


def build_vqa_prompt(question: VQAQuestion) -> str:
    """Builds a structured prompt instructing the VLM to observe video frames, reason, and choose O1-O4."""
    opts_str = "\n".join([f"- {k}: {v}" for k, v in sorted(question.options.items())])
    clip_dur = int(round(question.end_sec - question.start_sec))
    prompt = (
        f"You are an expert FIFA video referee and tactical analyst.\n"
        f"Observe the provided sequence of video frames ({clip_dur} seconds) sampled from a broadcast soccer match.\n"
        f"Each frame has a digital clock '[00:SS]' burned in the top-left corner.\n\n"
        f"Question: {question.question}\n\n"
        f"Options:\n"
        f"{opts_str}\n\n"
        f"CRITICAL INSTRUCTIONS:\n"
        f"1. You MUST respond with ONLY a single raw JSON object.\n"
        f"2. Do NOT write conversational preamble, introductory text, or markdown lists before the JSON.\n"
        f"3. Start your response immediately with '{{'.\n"
        f"4. The JSON must contain 'selected_option' as the FIRST key ('O1', 'O2', 'O3', or 'O4').\n\n"
        f"Output Schema:\n"
        f'{{\n'
        f'  "selected_option": "O1",\n'
        f'  "confidence": 0.90,\n'
        f'  "thought": "Concise 1-sentence visual evidence..."\n'
        f'}}\n'
    )
    return prompt


def parse_vqa_model_output(raw_text: str, question: Optional[VQAQuestion] = None) -> Tuple[Optional[VQAResponse], Dict[str, Any]]:
    """Robust parser for VQA JSON outputs with regex and semantic text fallback."""
    diag = {
        "raw_length": len(raw_text),
        "strategy": "none",
        "selected_option": None,
        "is_valid": False
    }

    cleaned = raw_text.strip()
    if "```json" in cleaned:
        cleaned = cleaned.split("```json")[1].split("```")[0].strip()
    elif "```" in cleaned:
        cleaned = cleaned.split("```")[1].split("```")[0].strip()

    # Strategy 1: Direct JSON parse
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            if isinstance(data, dict):
                opt = str(data.get("selected_option", "")).strip().upper()
                for valid_opt in ("O1", "O2", "O3", "O4"):
                    if opt.startswith(valid_opt) or opt == valid_opt:
                        opt = valid_opt
                        break
                if opt in ("O1", "O2", "O3", "O4"):
                    resp = VQAResponse(
                        thought=str(data.get("thought", "Visual evidence analyzed")),
                        selected_option=opt,
                        confidence=float(data.get("confidence", 0.85))
                    )
                    diag["strategy"] = "json"
                    diag["selected_option"] = opt
                    diag["is_valid"] = True
                    return resp, diag
        except Exception:
            pass

    # Strategy 2: Regex option extraction fallback (looks for O1, O2, O3, O4)
    opt_match = re.search(r"\b(O[1-4])\b", cleaned, re.IGNORECASE)
    if opt_match:
        opt = opt_match.group(1).upper()
        diag["strategy"] = "regex_fallback"
        diag["selected_option"] = opt
        diag["is_valid"] = True
        return VQAResponse(thought="Extracted via regex fallback", selected_option=opt, confidence=0.80), diag

    # Strategy 3: Semantic option text match fallback
    if question and hasattr(question, "options"):
        best_opt = None
        best_score = 0
        raw_lower = cleaned.lower()
        for opt_key, opt_text in question.options.items():
            # Calculate word overlap of non-stop words
            opt_words = [w.lower().strip(".,;:()") for w in opt_text.split() if len(w) > 3]
            score = sum(1 for w in opt_words if w in raw_lower)
            if score > best_score and score >= 2:
                best_score = score
                best_opt = opt_key
        if best_opt:
            diag["strategy"] = "semantic_text_fallback"
            diag["selected_option"] = best_opt
            diag["is_valid"] = True
            return VQAResponse(thought="Extracted via semantic option text match", selected_option=best_opt, confidence=0.75), diag

    return None, diag
