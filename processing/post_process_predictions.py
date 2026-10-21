"""
Post-Processor and Stitching Engine for SoccerNet V2 Benchmark
Features:
1. Manifest-anchored stitching using exact millisecond keyframe offsets.
2. Fault-isolated event bounds validation against actual clip duration (drops invalid events, not whole clip).
3. Class-aware Non-Maximum Suppression (NMS) to preserve valid co-occurring events.
4. Dual exporter for Action Spotting (Predictions-v2.json) and Dense Captioning.
"""

import os
import sys
import json
from collections import defaultdict
from typing import List, Dict, Tuple, Optional
from pydantic import ValidationError

from processing.unified_schema import (
    SpottedEvent,
    convert_clip_events_to_soccernet_format,
    parse_mm_ss_to_seconds,
    SOCCERNET_CLASSES
)

# Domain-Informed Class-Specific NMS Suppression Windows
# Accounts for physical duration of restarts, celebrations, and multi-angle broadcast replays
DEFAULT_CLASS_NMS_RADII_MS: Dict[str, int] = {
    # Macro-Stoppage Classes (celebrations + walk-back + multi-angle broadcast replays)
    "Goal": 45000,
    "Kick-off": 45000,
    "Penalty": 45000,
    "Red card": 45000,
    "Yellow->red card": 45000,
    # Set-Piece & Card Classes
    "Corner": 12000,
    "Direct free-kick": 12000,
    "Indirect free-kick": 12000,
    "Yellow card": 12000,
    "Substitution": 12000,
    "Offside": 12000,
    # Fast Play-Transition Classes
    "Shots on target": 4000,
    "Shots off target": 4000,
    "Foul": 4000,
    "Clearance": 4000,
    "Throw-in": 4000,
    "Ball out of play": 4000,
}

def class_aware_nms(
    predictions: List[Dict],
    radius_ms: int = 3000,
    overlap_zones: Optional[List[Tuple[int, int]]] = None,
    overlap_radius_ms: int = 5000,
    class_radii_ms: Optional[Dict[str, int]] = None
) -> List[Dict]:
    """
    Suppress duplicate predictions of the SAME (half, label) within radius_ms (or class-specific radius).
    If overlap_zones is provided, any candidate or existing prediction situated inside
    a macro-chunk boundary overlap zone uses overlap_radius_ms (default 5000ms) instead
    of standard radius_ms (3000ms) to prevent boundary duplicates across chunk seams.
    Valid co-occurring soccer events (e.g., Foul at 12:01, Yellow card at 12:02,
    or Shots on target + Goal) are strictly preserved across different classes.
    """
    kept = []
    grouped = defaultdict(list)

    for pred in predictions:
        half_key = str(pred.get("half", "1"))
        label_key = pred.get("label", "")
        grouped[(half_key, label_key)].append(pred)

    def is_in_overlap(pos: int) -> bool:
        if not overlap_zones:
            return False
        return any(start <= pos <= end for start, end in overlap_zones)

    for (half, label), group in grouped.items():
        # Sort candidates descending by confidence
        group.sort(key=lambda x: float(x.get("confidence", 1.0)), reverse=True)
        base_radius = class_radii_ms.get(label, radius_ms) if class_radii_ms else radius_ms
        active = []
        for candidate in group:
            cand_pos = int(candidate["position"])
            cand_in_overlap = is_in_overlap(cand_pos)
            suppressed = False
            for a in active:
                a_pos = int(a["position"])
                eff_radius = max(overlap_radius_ms, base_radius) if (cand_in_overlap or is_in_overlap(a_pos)) else base_radius
                if abs(cand_pos - a_pos) <= eff_radius:
                    suppressed = True
                    break
            if not suppressed:
                active.append(candidate)
        kept.extend(active)

    # Sort final predictions chronologically across halves
    return sorted(kept, key=lambda x: (int(x.get("half", 1)), int(x["position"])))

def stitch_chunks_to_match_predictions(
    chunk_results: List[Dict],
    chunks_manifest: Dict,
    url_local: str,
    radius_ms: int = 3000,
    overlap_radius_ms: int = 5000,
    class_radii_ms: Optional[Dict[str, int]] = None
) -> Tuple[Dict, Dict, Dict]:
    """
    Stitches macro-window outputs using exact keyframe start timestamps from manifest.
    Applies event-level bounds check against measured clip duration, dynamically detects
    macro-chunk boundary overlap seams, and applies cross-chunk class-aware NMS.
    Returns: (spotting_output, caption_output, telemetry)
    """
    manifest_index_lookup = {c["chunk_index"]: c for c in chunks_manifest.get("chunks", []) if "chunk_index" in c}
    manifest_filename_lookup = {c["filename"]: c for c in chunks_manifest.get("chunks", []) if "filename" in c}

    # Dynamically extract macro-chunk boundary overlap zones from chunks manifest
    overlap_zones: List[Tuple[int, int]] = []
    sorted_chunks = sorted(chunks_manifest.get("chunks", []), key=lambda c: int(c.get("actual_start_ms", 0)))
    for i in range(len(sorted_chunks) - 1):
        c1, c2 = sorted_chunks[i], sorted_chunks[i+1]
        c1_start_ms = int(c1.get("actual_start_ms", 0))
        c1_dur_sec = float(c1.get("duration_sec", 0.0))
        c1_end_ms = c1_start_ms + int(round(c1_dur_sec * 1000.0))
        c2_start_ms = int(c2.get("actual_start_ms", 0))
        if c1_end_ms > c2_start_ms:
            overlap_zones.append((c2_start_ms, c1_end_ms))

    all_raw_spotting = []
    all_raw_captions = []
    total_raw_events = 0
    total_out_of_bounds_drops = 0

    for chunk_res in chunk_results:
        chunk_meta = None
        # 1. Deterministic Provenance via Echoed GCS URI or Filename (immune to async reordering & hallucinated index)
        if "filename" in chunk_res and chunk_res["filename"] in manifest_filename_lookup:
            chunk_meta = manifest_filename_lookup[chunk_res["filename"]]
        elif "file_uri" in chunk_res:
            import os
            uri_fname = os.path.basename(chunk_res["file_uri"].replace("\\", "/"))
            if uri_fname in manifest_filename_lookup:
                chunk_meta = manifest_filename_lookup[uri_fname]

        # 2. Fallback to clip_index or chunk_index if provenance not explicitly attached
        if not chunk_meta:
            chunk_idx = chunk_res.get("clip_index") if chunk_res.get("clip_index") is not None else chunk_res.get("chunk_index", 0)
            chunk_meta = manifest_index_lookup.get(chunk_idx, {})

        actual_start_sec = float(chunk_meta.get("actual_start_sec", 0.0))
        actual_duration_sec = float(chunk_meta.get("duration_sec", 0.0)) if "duration_sec" in chunk_meta else None
        half = int(chunk_meta.get("half", chunk_res.get("half", 1)))

        events_data = chunk_res.get("events", [])
        total_raw_events += len(events_data)
        events = []
        for ev in events_data:
            if isinstance(ev, dict):
                try:
                    events.append(SpottedEvent(**ev))
                except (ValidationError, ValueError):
                    # Schema or regex failure on individual event
                    total_out_of_bounds_drops += 1
            elif isinstance(ev, SpottedEvent):
                events.append(ev)

        spotting, captions, drops = convert_clip_events_to_soccernet_format(
            events=events,
            half=half,
            clip_start_sec=actual_start_sec,
            clip_duration_sec=actual_duration_sec
        )
        total_out_of_bounds_drops += drops
        all_raw_spotting.extend(spotting)
        all_raw_captions.extend(captions)

    # Apply Class-Aware NMS across the stitched match timeline with boundary overlap radius
    deduped_spotting = class_aware_nms(
        all_raw_spotting,
        radius_ms=radius_ms,
        overlap_zones=overlap_zones,
        overlap_radius_ms=overlap_radius_ms,
        class_radii_ms=class_radii_ms
    )

    # Filter captions to match the deduplicated spotting timestamps and labels
    kept_signatures = set((d["half"], d["label"], d["position"]) for d in deduped_spotting)
    deduped_captions = [
        c for c in all_raw_captions
        if (c["half"], c["label"], c["position"]) in kept_signatures
    ]

    telemetry = {
        "total_raw_events": total_raw_events,
        "out_of_bounds_drops": total_out_of_bounds_drops,
        "valid_bounded_events": len(all_raw_spotting),
        "nms_suppressed_events": len(all_raw_spotting) - len(deduped_spotting),
        "final_kept_events": len(deduped_spotting),
        "macro_overlap_zones": overlap_zones,
        "overlap_radius_ms": overlap_radius_ms
    }

    spotting_output = {
        "UrlLocal": url_local,
        "predictions": deduped_spotting
    }

    caption_output = {
        "UrlLocal": url_local,
        "predictions": deduped_captions
    }

    return spotting_output, caption_output, telemetry

def run_self_test():
    """Unit tests validating regex hardening, bounds checks, and class-aware NMS."""
    print("[TEST] Running Checkpoint 1 Validation Suite...")

    # Test 1: Regex Enforcement on SpottedEvent
    print("  [1/3] Testing MM:SS regex enforcement...")
    malformed_timestamps = ["1:5", "00:60", "12000", "-01:00", "04:65", "abc:de"]
    for ts in malformed_timestamps:
        failed = False
        try:
            SpottedEvent(
                timestamp=ts,
                commentary="Player takes a shot.",
                label="Goal",
                confidence=0.9
            )
        except (ValidationError, ValueError):
            failed = True
        assert failed, f"Regex failed to reject invalid timestamp: '{ts}'"
    print("        [PASS] All malformed timestamp strings rejected by regex.")

    # Test 2: Event-Level Duration Bounds Checking
    print("  [2/3] Testing event-level duration bounds checking...")
    mock_manifest = {
        "chunks": [
            {
                "chunk_index": 0,
                "actual_start_sec": 0.0,
                "duration_sec": 608.5  # Actual duration ~10m 08s
            }
        ]
    }
    mock_chunk_results = [
        {
            "chunk_index": 0,
            "half": 1,
            "events": [
                # Valid in-bounds events
                {"timestamp": "01:15", "commentary": "Early corner taken", "label": "Corner", "confidence": 0.95},
                {"timestamp": "05:30", "commentary": "Foul in midfield", "label": "Foul", "confidence": 0.85},
                {"timestamp": "10:02", "commentary": "Shot near end of chunk", "label": "Shots on target", "confidence": 0.90},
                # Out-of-bounds event (10:45 = 645s > 608.5s duration)
                {"timestamp": "10:45", "commentary": "Hallucinated late event", "label": "Offside", "confidence": 0.70},
                # Malformed event that should be dropped without failing clip
                {"timestamp": "99:99", "commentary": "Malformed seconds", "label": "Goal", "confidence": 0.50}
            ]
        }
    ]

    spot, caps, telem = stitch_chunks_to_match_predictions(
        chunk_results=mock_chunk_results,
        chunks_manifest=mock_manifest,
        url_local="england_epl/test/game_1"
    )

    assert telem["total_raw_events"] == 5, f"Expected 5 raw events, got {telem['total_raw_events']}"
    assert telem["out_of_bounds_drops"] == 2, f"Expected 2 dropped events, got {telem['out_of_bounds_drops']}"
    assert len(spot["predictions"]) == 3, f"Expected 3 kept predictions, got {len(spot['predictions'])}"
    print(f"        [PASS] Fault-isolation confirmed: {telem['out_of_bounds_drops']} dropped, {len(spot['predictions'])} preserved.")

    # Test 3: Class-Aware NMS
    print("  [3/3] Testing class-aware NMS...")
    test_preds = [
        {"half": "1", "label": "Foul", "position": 750000, "confidence": 0.95},
        {"half": "1", "label": "Yellow card", "position": 751200, "confidence": 0.90},
        {"half": "1", "label": "Foul", "position": 751500, "confidence": 0.70},
        {"half": "1", "label": "Foul", "position": 795000, "confidence": 0.85},
        {"half": "2", "label": "Shots on target", "position": 1800000, "confidence": 0.92},
        {"half": "2", "label": "Goal", "position": 1801000, "confidence": 0.98},
        {"half": "2", "label": "Goal", "position": 1802500, "confidence": 0.80},
    ]

    result = class_aware_nms(test_preds, radius_ms=3000)
    assert len(result) == 5, f"Expected 5 kept events, got {len(result)}"
    print("        [PASS] Class-aware NMS correctly preserved co-occurring events while suppressing duplicates.")

    # Test 4: Macro-Chunk Boundary Overlap NMS
    print("  [4/4] Testing macro-chunk boundary overlap NMS deduplication...")
    boundary_preds = [
        {"half": "1", "label": "Corner", "position": 583000, "confidence": 0.90},
        {"half": "1", "label": "Corner", "position": 587000, "confidence": 0.85},  # 4000ms apart
    ]
    # Without overlap zone, 4000ms > 3000ms -> both kept
    standard_res = class_aware_nms(boundary_preds, radius_ms=3000)
    assert len(standard_res) == 2, f"Expected 2 events with standard radius, got {len(standard_res)}"

    # With overlap zone, 4000ms <= 5000ms -> suppressed
    overlap_res = class_aware_nms(boundary_preds, radius_ms=3000, overlap_zones=[(582043, 615083)], overlap_radius_ms=5000)
    assert len(overlap_res) == 1, f"Expected 1 event with overlap radius, got {len(overlap_res)}"
    assert overlap_res[0]["position"] == 583000, "Expected higher-confidence event to be preserved"
    print("        [PASS] Boundary overlap NMS successfully suppressed cross-chunk seam duplicate.")

    print("\n[CHECKPOINT 1 PASS] All Schema & Bounds Hardening unit tests passed successfully!\n")

if __name__ == "__main__":
    run_self_test()
