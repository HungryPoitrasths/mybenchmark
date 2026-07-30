from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DEFAULT_TEMPLATE_PATH = Path(__file__).resolve().parents[2] / "templates" / "cot_templates.json"


_QUESTION_TYPE_CONTEXTS = {
    "direction_agent": "This direction question uses the first camera's horizontal coordinate frame",
    "direction_object_centric": "This direction question uses an observer's object-centered heading",
    "direction_allocentric": "This direction question uses cardinal axes on the room's floor plan",
    "occlusion": "This question applies the benchmark's visibility and occlusion definitions",
    "distance": "This question classifies distance between the closest object surfaces",
    "object_move_agent": "This movement question keeps the first camera's coordinate frame fixed",
    "object_move_distance": "This movement question recomputes closest-surface distance",
    "object_move_occlusion": "This movement question uses the first view for motion and the last view for occlusion",
    "object_rotate_object_centric": "This orbit question recomputes an observer-centered direction after the orbit",
    "object_move_object_centric": "This movement question preserves the observer's initial horizontal heading",
    "object_move_allocentric": "This movement question uses cardinal directions on the floor plan",
    "object_remove": "This question compares visibility before and after removing one object",
    "attachment_chain": "This question follows directed support paths from the moved object",
    "coordinate_rotation_agent": "This rotation question keeps the camera pose fixed",
    "coordinate_rotation_object_centric": "This rotation question preserves the observer's original world heading",
    "coordinate_rotation_allocentric": "This rotation question uses fixed cardinal axes on the floor plan",
}


_DETAIL_CONTEXTS = {
    "single_axis": "one horizontal component is decisive",
    "diagonal": "two horizontal components are combined",
    "direct": "the queried object is moved directly",
    "supported": "the queried object follows the motion through its support chain",
    "stationary": "the reference object stays fixed",
    "co_moved": "the reference object shares the same translation",
    "co_orbited": "the reference object shares the same orbit",
    "changed": "the qualitative relation changes after the operation",
    "preserved": "the qualitative relation is preserved",
    "closer": "the closest-surface distance becomes smaller",
    "farther": "the closest-surface distance becomes larger",
    "within_bin": "the distance changes without leaving its interval",
    "very_close": "the final distance falls in the very-close interval",
    "close": "the final distance falls in the close interval",
    "moderate": "the final distance falls in the moderate interval",
    "far": "the final distance falls in the far interval",
    "camera": "the camera coordinate frame remains unchanged",
    "floor_plan": "the comparison is made on the floor plan",
    "ref_in_frame_1": "the observer is identified in the first main view",
    "face_in_frame_1": "the facing object is identified in the first main view",
    "in_frame": "the target projection lies within the image frame",
    "partly_out_of_frame": "part of the target projection lies outside the image frame",
    "unknown": "no extra framing claim is needed",
    "not_visible": "the target has no visible portion",
    "not_occluded": "no other object blocks the visible target",
    "occluded": "another object blocks part of the visible target",
    "query_occluded_by_ref": "the reference object blocks the queried object",
    "ref_occluded_by_query": "the queried object blocks the reference object",
    "neither": "neither specified object blocks the other",
    "none": "none of the listed objects follows the moved root",
    "exactly_one": "exactly one listed object follows the moved root",
    "multiple_not_all": "multiple listed objects, but not all of them, follow the moved root",
    "all": "all listed objects follow the moved root",
}

_DIRECTION_CONTEXTS = {
    "front": "the final relation lies to the front",
    "back": "the final relation lies to the back",
    "left": "the final relation lies to the left",
    "right": "the final relation lies to the right",
    "front-left": "the final relation combines front and left",
    "front-right": "the final relation combines front and right",
    "back-left": "the final relation combines back and left",
    "back-right": "the final relation combines back and right",
    "north": "the final relation lies to the north",
    "south": "the final relation lies to the south",
    "east": "the final relation lies to the east",
    "west": "the final relation lies to the west",
    "northeast": "the final relation combines north and east",
    "northwest": "the final relation combines north and west",
    "southeast": "the final relation combines south and east",
    "southwest": "the final relation combines south and west",
}


def _detail_context(detail: str, *, question_type: str) -> str:
    if detail in _DETAIL_CONTEXTS:
        return _DETAIL_CONTEXTS[detail]
    if detail in _DIRECTION_CONTEXTS:
        return _DIRECTION_CONTEXTS[detail]
    if detail.isdigit():
        if question_type == "object_rotate_object_centric":
            return f"the prescribed orbit angle is {detail} degrees"
        return f"the prescribed clockwise rotation is {detail} degrees"
    if "_to_" in detail:
        before, after = detail.split("_to_", 1)
        return f"the saved state changes from {before.replace('_', ' ')} to {after.replace('_', ' ')}"
    return detail.replace("_", " ").replace("-", " ")


def _signature_context(signature_id: str) -> str:
    head, *details = signature_id.split(".")
    question_type = head.removeprefix("L1_").removeprefix("L2_").removeprefix("L3_")
    base = _QUESTION_TYPE_CONTEXTS.get(question_type, "This question uses the saved spatial relations")
    detail_text = "; ".join(
        _detail_context(detail, question_type=question_type) for detail in details
    )
    return f"{base}; {detail_text}" if detail_text else base


def generated_templates_for_signature(signature_id: str) -> list[dict[str, str]]:
    """Create twelve deterministic templates with natural signature semantics."""
    context = _signature_context(signature_id)
    patterns = (
        "{context}. Initial scene fact: {observation}. Spatial reasoning: {transformation}. Therefore, {conclusion}.",
        "{context}. Given relation: {observation}. Required update: {transformation}. Final conclusion: {conclusion}.",
        "{context}. Starting condition: {observation}. Applying the stated rule: {transformation}. Hence, {conclusion}.",
        "{context}. Relevant setup: {observation}. Decisive spatial step: {transformation}. Result: {conclusion}.",
        "{context}. Recorded scene relation: {observation}. Recomputed relation: {transformation}. It follows that {conclusion}.",
        "{context}. Spatial evidence: {observation}. Evaluation: {transformation}. Thus, {conclusion}.",
        "{context}. Initial state: {observation}. Consequence of the stated operation: {transformation}. Therefore, {conclusion}.",
        "{context}. Reference condition: {observation}. Derived spatial consequence: {transformation}. Final result: {conclusion}.",
        "{context}. Fixed reference convention: {observation}. Comparison under that convention: {transformation}. Therefore, {conclusion}.",
        "{context}. Initial fact: {observation}. Updated fact: {transformation}. Accordingly, {conclusion}.",
        "{context}. Saved spatial facts: {observation}. Requested reasoning step: {transformation}. This establishes that {conclusion}.",
        "{context}. Grounding relation: {observation}. Decisive comparison: {transformation}. We conclude that {conclusion}.",
    )
    return [
        {
            "id": f"signature_{index:02d}",
            "template": pattern.replace("{context}", context),
        }
        for index, pattern in enumerate(patterns)
    ]


def load_template_library(path: str | Path | None = None) -> dict[str, Any]:
    template_path = Path(path) if path else DEFAULT_TEMPLATE_PATH
    with template_path.open(encoding="utf-8") as handle:
        library = json.load(handle)
    styles = library.get("styles")
    if not isinstance(styles, list) or len(styles) != 12:
        raise ValueError(f"{template_path} must contain exactly 12 styles")
    for index, style in enumerate(styles):
        if not isinstance(style, dict) or not isinstance(style.get("template"), str):
            raise ValueError(f"invalid style at index {index}")
        required = {"{observation}", "{transformation}", "{conclusion}"}
        if not all(token in style["template"] for token in required):
            raise ValueError(f"style {index} is missing a required placeholder")
    return library


def templates_for_signature(library: dict[str, Any], signature_id: str) -> list[dict[str, str]]:
    overrides = library.get("signature_templates", {})
    selected = overrides.get(signature_id) if isinstance(overrides, dict) else None
    if selected is None:
        selected = generated_templates_for_signature(signature_id)
    if not isinstance(selected, list) or len(selected) != 12:
        raise ValueError(f"signature {signature_id} must have exactly 12 templates")
    normalized: list[dict[str, str]] = []
    for index, item in enumerate(selected):
        if isinstance(item, str):
            normalized.append({"id": f"custom_{index:02d}", "template": item})
        else:
            normalized.append({"id": str(item.get("id", f"style_{index:02d}")), "template": str(item["template"])})
    required = {"{observation}", "{transformation}", "{conclusion}"}
    for item in normalized:
        text = item["template"]
        if not all(slot in text for slot in required):
            raise ValueError(f"template {item['id']} for {signature_id} is missing a required slot")
        if "answer:" in text.lower() or "<think>" in text.lower():
            raise ValueError(
                f"template {item['id']} for {signature_id} must not contain an answer line or think tag"
            )
    return normalized
