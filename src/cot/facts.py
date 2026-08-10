from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any

from .images import collect_image_names
from .models import CotFactRecord, FactExtractionError


SUPPORTED_TYPES = {
    "direction_agent",
    "occlusion",
    "distance",
    "direction_object_centric",
    "direction_allocentric",
    "object_move_agent",
    "object_move_distance",
    "object_move_occlusion",
    "object_rotate_object_centric",
    "object_move_object_centric",
    "object_move_allocentric",
    "object_remove",
    "attachment_chain",
    "coordinate_rotation_agent",
    "coordinate_rotation_object_centric",
    "coordinate_rotation_allocentric",
}

EGOCENTRIC_DIRECTIONS = {
    "front",
    "back",
    "left",
    "right",
    "front-left",
    "front-right",
    "back-left",
    "back-right",
}
CARDINAL_DIRECTIONS = {
    "north",
    "south",
    "east",
    "west",
    "northeast",
    "northwest",
    "southeast",
    "southwest",
}
VISIBILITY_STATES = {"not visible", "occluded", "not occluded"}
PAIRWISE_OCCLUSION = {
    "query_occluded_by_ref",
    "ref_occluded_by_query",
    "neither",
}
PAIRWISE_VALUE_MAP = {
    "query occluded by reference": "query_occluded_by_ref",
    "query_occluded_by_ref": "query_occluded_by_ref",
    "query_occluded_by_reference": "query_occluded_by_ref",
    "target occluded by reference": "query_occluded_by_ref",
    "the query object is occluded by the reference object": "query_occluded_by_ref",
    "reference occluded by query": "ref_occluded_by_query",
    "ref_occluded_by_query": "ref_occluded_by_query",
    "reference_occluded_by_query": "ref_occluded_by_query",
    "the reference object is occluded by the query object": "ref_occluded_by_query",
    "neither": "neither",
    "neither object occludes the other": "neither",
}


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def question_uid(question: dict[str, Any]) -> str:
    existing = str(question.get("question_uid") or "").strip()
    if existing:
        return existing
    identity = {
        key: question.get(key)
        for key in (
            "_dataset",
            "scene_id",
            "level",
            "type",
            "question",
            "options",
            "answer",
        )
    }
    identity["image_route"] = collect_image_names(question)
    return hashlib.sha256(_canonical_json(identity).encode("utf-8")).hexdigest()


def _answer_letters(question: dict[str, Any]) -> list[str]:
    raw = str(question.get("answer") or "").upper()
    letters = re.findall(r"[A-Z]", raw)
    options = question.get("options")
    if not letters:
        raise FactExtractionError("missing_answer", "answer contains no option letter")
    if not isinstance(options, list) or not options:
        raise FactExtractionError("missing_options", "options must be a non-empty list")
    for letter in letters:
        if ord(letter) - ord("A") >= len(options):
            raise FactExtractionError("answer_out_of_range", f"answer {letter} exceeds options")
    return letters


def _semantic_answer(question: dict[str, Any]) -> Any:
    if question.get("multi_select"):
        values = question.get("correct_values")
        if isinstance(values, list) and values:
            return [str(value) for value in values]
        raw = str(question.get("correct_value") or "")
        parsed = [part.strip() for part in raw.split(";") if part.strip()]
        if parsed:
            return parsed
        raise FactExtractionError("missing_semantic_answer", "multi-select correct_values missing")
    value = question.get("new_correct_value", question.get("correct_value"))
    if value is None or str(value).strip() == "":
        raise FactExtractionError("missing_semantic_answer", "correct_value missing")
    return str(value).strip()


def _label(question: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = question.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    raise FactExtractionError("missing_object_label", f"none of {keys} is present")


def _id(question: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if question.get(key) is not None:
            return question[key]
    return None


def _direction_components(direction: str, *, cardinal: bool) -> dict[str, str]:
    direction = direction.lower()
    allowed = CARDINAL_DIRECTIONS if cardinal else EGOCENTRIC_DIRECTIONS
    if direction not in allowed:
        raise FactExtractionError(
            "unsupported_direction",
            f"only eight horizontal directions are supported, got {direction!r}",
        )
    if cardinal:
        vertical = "north" if "north" in direction else "south" if "south" in direction else "neutral"
        horizontal = "east" if "east" in direction else "west" if "west" in direction else "neutral"
    else:
        vertical = "front" if "front" in direction else "back" if "back" in direction else "neutral"
        horizontal = "left" if "left" in direction else "right" if "right" in direction else "neutral"
    return {
        "horizontal_axis": horizontal,
        "depth_axis": vertical,
        "layout": "diagonal" if horizontal != "neutral" and vertical != "neutral" else "single_axis",
    }


def _movement_mode(question: dict[str, Any]) -> str:
    moved_id = _id(question, "moved_obj_id")
    query_id = _id(question, "query_obj_id", "obj_b_id")
    if bool(question.get("attachment_remapped")) or (
        moved_id is not None and query_id is not None and str(moved_id) != str(query_id)
    ):
        return "supported"
    return "direct"


def _reference_motion_mode(question: dict[str, Any], *, orbit: bool = False) -> str:
    moved_id = _id(question, "moved_obj_id")
    ref_id = _id(question, "obj_ref_id", "obj_c_id")
    child_id = _id(question, "attachment_child_id")
    if ref_id is not None and (str(ref_id) == str(moved_id) or str(ref_id) == str(child_id)):
        return "co_orbited" if orbit else "co_moved"
    trace_reason = str(question.get("trace_reason") or "")
    if bool(question.get("relation_unchanged")) and trace_reason.endswith("preserved_fallback"):
        return "co_orbited" if orbit else "co_moved"
    return "stationary"


def _movement_distance(question: dict[str, Any]) -> float | None:
    delta = question.get("delta")
    if isinstance(delta, list) and len(delta) >= 2:
        try:
            return round(math.sqrt(sum(float(value) ** 2 for value in delta[:3])), 3)
        except (TypeError, ValueError):
            pass
    match = re.search(r"\b(\d+(?:\.\d+)?)m\b", str(question.get("question") or ""))
    return float(match.group(1)) if match else None


def _movement_direction(question: dict[str, Any]) -> str:
    explicit = str(question.get("movement_direction") or "").strip().lower()
    if explicit:
        return explicit
    text = str(question.get("question") or "").lower()
    movement_verbs = list(re.finditer(r"\b(?:moving|move|moved|shifted)\b", text))
    start = movement_verbs[-1].start() if movement_verbs else 0
    segment = text[start : start + 220]
    patterns = (
        "forward-right",
        "forward-left",
        "backward-right",
        "backward-left",
        "northeast",
        "northwest",
        "southeast",
        "southwest",
        "forward",
        "backward",
        "north",
        "south",
        "east",
        "west",
        "left",
        "right",
    )
    for value in patterns:
        if re.search(rf"\b{re.escape(value)}\b", segment):
            return value
    raise FactExtractionError("missing_movement_direction", "movement direction could not be recovered")


def _changed(question: dict[str, Any]) -> str:
    if question.get("relation_unchanged") is not None:
        return "preserved" if bool(question["relation_unchanged"]) else "changed"
    old = question.get("old_correct_value")
    new = question.get("new_correct_value", question.get("correct_value"))
    return "preserved" if old == new else "changed"


def _base_facts(question: dict[str, Any]) -> dict[str, Any]:
    return {
        "level": str(question.get("level") or ""),
        "scene_id": str(question.get("scene_id") or ""),
        "image_name": str(question.get("image_name") or ""),
    }


def _direction_facts(question: dict[str, Any], qtype: str, answer: str) -> tuple[dict[str, Any], str]:
    cardinal = qtype.endswith("allocentric")
    components = _direction_components(answer, cardinal=cardinal)
    facts = _base_facts(question)
    facts.update(components)
    facts["result"] = answer
    if qtype == "direction_agent":
        facts.update(
            subject=_label(question, "obj_a_label"),
            reference=_label(question, "obj_b_label"),
            reference_frame="first_camera",
        )
    elif qtype == "direction_object_centric":
        facts.update(
            observer=_label(question, "obj_ref_label"),
            facing_object=_label(question, "obj_face_label"),
            subject=_label(question, "obj_target_label"),
            reference_frame="object_heading",
        )
    else:
        facts.update(
            subject=_label(question, "obj_a_label"),
            reference=_label(question, "obj_b_label"),
            camera_cardinal=_label(question, "camera_cardinal"),
            reference_frame="floor_plan",
        )
    signature = f"L1_{qtype}.{components['layout']}.{answer}"
    return facts, signature


def _occlusion_facts(question: dict[str, Any], answer: str) -> tuple[dict[str, Any], str]:
    if answer not in VISIBILITY_STATES:
        raise FactExtractionError("unsupported_occlusion_state", answer)
    ratio = question.get("geometry_in_frame_ratio", question.get("bbox_in_frame_ratio"))
    framing = "unknown"
    if ratio is not None:
        try:
            framing = "partly_out_of_frame" if float(ratio) < 0.98 else "in_frame"
        except (TypeError, ValueError):
            framing = "unknown"
    facts = _base_facts(question)
    facts.update(target=_label(question, "obj_a_label"), status=answer, framing=framing)
    blocker = question.get("primary_occluder_label") or question.get("dominant_blocker_label")
    confidence = question.get("primary_occluder_confidence", question.get("dominant_blocker_ratio"))
    valid_hits = question.get("primary_occluder_valid_hit_count", question.get("blocker_valid_hit_count"))
    runner_up = question.get("primary_occluder_runner_up_ratio", question.get("runner_up_blocker_ratio"))
    if answer == "occluded" and blocker and confidence is not None and valid_hits is not None and runner_up is not None:
        if float(confidence) >= 0.8 and int(valid_hits) >= 32 and float(runner_up) <= 0.1:
            facts["blocker"] = str(blocker)
    return facts, f"L1_occlusion.{answer.replace(' ', '_')}.{framing}"


def _distance_facts(question: dict[str, Any], answer: str) -> tuple[dict[str, Any], str]:
    bin_id = str(question.get("distance_bin_id") or "").strip()
    if not bin_id:
        bin_id = re.sub(r"[^a-z0-9]+", "_", answer.lower()).strip("_")
    facts = _base_facts(question)
    facts.update(
        object_a=_label(question, "obj_a_label"),
        object_b=_label(question, "obj_b_label"),
        distance_bin=answer,
        distance_bin_id=bin_id,
        distance_definition="closest_surface_points",
    )
    return facts, f"L1_distance.{bin_id}"


def _move_common(question: dict[str, Any]) -> dict[str, Any]:
    facts = _base_facts(question)
    facts.update(
        moved_object=_label(question, "moved_obj_label"),
        query_object=_label(question, "query_obj_label", "obj_b_label"),
        movement_mode=_movement_mode(question),
        movement_direction=_movement_direction(question),
        movement_distance_m=_movement_distance(question),
        change_state=_changed(question),
    )
    return facts


def _move_direction_facts(
    question: dict[str, Any], qtype: str, answer: str
) -> tuple[dict[str, Any], str]:
    cardinal = qtype == "object_move_allocentric"
    components = _direction_components(answer, cardinal=cardinal)
    facts = _move_common(question)
    facts.update(components)
    facts.update(
        old_result=str(question.get("old_correct_value") or "unknown"),
        result=answer,
        reference_motion_mode=_reference_motion_mode(question),
    )
    if qtype == "object_move_agent":
        facts.update(
            reference_object=_label(question, "obj_c_label"),
            reference_frame="first_camera",
        )
    else:
        facts.update(reference_object=_label(question, "obj_ref_label"))
        if qtype == "object_move_object_centric":
            facts.update(
                reference_frame="query_original_camera_heading",
                movement_reference_frame="moved_object_facing_first_camera",
                answer_reference_frame="query_object_facing_first_camera",
            )
        else:
            facts.update(
                reference_frame="floor_plan",
                camera_cardinal=_label(question, "camera_cardinal"),
            )
    signature = ".".join(
        [
            f"L2_{qtype}",
            facts["movement_mode"],
            facts["reference_motion_mode"],
            facts["change_state"],
            components["layout"],
            answer,
        ]
    )
    return facts, signature


def _move_distance_facts(question: dict[str, Any], answer: str) -> tuple[dict[str, Any], str]:
    facts = _move_common(question)
    old_bin = str(question.get("old_correct_value") or "unknown")
    old_distance = question.get("old_distance_m")
    new_distance = question.get("new_distance_m")
    trend = "within_bin"
    if old_distance is not None and new_distance is not None:
        difference = float(new_distance) - float(old_distance)
        trend = "farther" if difference > 1e-6 else "closer" if difference < -1e-6 else "preserved"
    elif old_bin == answer:
        trend = "within_bin"
    facts.update(
        object_a=facts["query_object"],
        object_b=_label(question, "obj_c_label"),
        old_distance_bin=old_bin,
        distance_bin=answer,
        distance_trend=trend,
        reference_motion_mode=_reference_motion_mode(question),
        distance_definition="closest_surface_points",
    )
    if facts["reference_motion_mode"] == "co_moved" and facts["change_state"] == "preserved":
        facts["distance_trend"] = "preserved"
        trend = "preserved"
    signature = ".".join(
        [
            "L2_object_move_distance",
            facts["movement_mode"],
            facts["reference_motion_mode"],
            facts["change_state"],
            trend,
        ]
    )
    return facts, signature


def _pairwise_relation(question: dict[str, Any], answer: str) -> str:
    explicit = question.get("new_pairwise_occlusion_relation", question.get("pairwise_occlusion_relation"))
    candidates = [explicit, answer]
    query = str(question.get("query_obj_label") or "").lower()
    reference = str(question.get("obj_ref_label") or question.get("obj_b_label") or "").lower()
    for candidate in candidates:
        text = str(candidate or "").strip().lower()
        if text in PAIRWISE_VALUE_MAP:
            return PAIRWISE_VALUE_MAP[text]
        if "neither" in text:
            return "neither"
        if query and reference and query in text and reference in text:
            if re.search(rf"{re.escape(query)}\s+occluded by\s+(?:the\s+)?{re.escape(reference)}", text):
                return "query_occluded_by_ref"
            if re.search(rf"{re.escape(reference)}\s+occluded by\s+(?:the\s+)?{re.escape(query)}", text):
                return "ref_occluded_by_query"
    raise FactExtractionError("unsupported_pairwise_occlusion", answer)


def _move_occlusion_facts(question: dict[str, Any], answer: str) -> tuple[dict[str, Any], str]:
    if not question.get("occlusion_semantics_version") and not question.get("new_pairwise_occlusion_relation"):
        raise FactExtractionError("legacy_unary_move_occlusion", "only pairwise L2 occlusion is supported")
    relation = _pairwise_relation(question, answer)
    facts = _move_common(question)
    facts.update(
        query_object=_label(question, "query_obj_label", "target_obj_label"),
        reference_object=_label(question, "obj_ref_label", "obj_b_label"),
        pairwise_relation=relation,
        movement_reference_frame="first_camera",
        occlusion_viewpoint="last_main_view",
    )
    signature = f"L2_object_move_occlusion.{facts['movement_mode']}.{relation}"
    return facts, signature


def _rotate_object_facts(question: dict[str, Any], answer: str) -> tuple[dict[str, Any], str]:
    components = _direction_components(answer, cardinal=False)
    facts = _base_facts(question)
    angle = int(question.get("rotation_angle"))
    direction = str(question.get("rotation_direction") or "clockwise").strip().lower()
    if direction not in {"clockwise", "counterclockwise"}:
        raise FactExtractionError(
            "invalid_rotation_direction",
            f"unsupported rotation direction {direction!r}",
        )
    facts.update(
        moved_object=_label(question, "moved_obj_label"),
        observer=_label(question, "query_obj_label"),
        query_object=_label(question, "query_obj_label"),
        facing_object=_label(question, "obj_face_label"),
        target=_label(question, "obj_ref_label"),
        rotation_angle=angle,
        rotation_direction=direction,
        movement_mode=_movement_mode(question),
        reference_motion_mode=_reference_motion_mode(question, orbit=True),
        old_result=str(question.get("old_correct_value") or "unknown"),
        result=answer,
        change_state=_changed(question),
        heading_rule="refaces_orbit_center_after_orbit",
        **components,
    )
    signature = ".".join(
        [
            "L2_object_rotate_object_centric",
            str(angle),
            facts["movement_mode"],
            facts["reference_motion_mode"],
            facts["change_state"],
            components["layout"],
        ]
    )
    return facts, signature


def _remove_facts(question: dict[str, Any], answer: str) -> tuple[dict[str, Any], str]:
    old_state = str(question.get("old_visibility") or "").strip()
    new_state = str(question.get("new_visibility") or answer).strip()
    if old_state not in VISIBILITY_STATES or new_state not in VISIBILITY_STATES:
        raise FactExtractionError("missing_visibility_transition", f"{old_state!r} -> {new_state!r}")
    rank = {"not visible": 0, "occluded": 1, "not occluded": 2}
    if rank[new_state] < rank[old_state]:
        raise FactExtractionError("non_monotonic_removal", f"{old_state} -> {new_state}")
    facts = _base_facts(question)
    facts.update(
        removed_object=_label(question, "removed_obj_label"),
        target=_label(question, "obj_b_label"),
        old_status=old_state,
        new_status=new_state,
    )
    return facts, f"L2_object_remove.{old_state.replace(' ', '_')}_to_{new_state.replace(' ', '_')}"


def _normalized_option(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value).strip().lower())
    return re.sub(r"^(?:the|a|an)\s+", "", text)


def _attachment_facts(question: dict[str, Any], answer: list[str]) -> tuple[dict[str, Any], str]:
    options = [str(value) for value in question.get("options", [])]
    correct = {_normalized_option(value) for value in answer}
    roles = {
        _normalized_option(question.get("parent_label")): "direct_support_path",
        _normalized_option(question.get("grandchild_label")): "two_hop_support_path",
        _normalized_option(question.get("neighbor_label")): "no_support_path",
    }
    option_facts = []
    for option in options:
        normalized = _normalized_option(option)
        moves = normalized in correct
        option_facts.append(
            {
                "option": option,
                "moves": moves,
                "path_kind": roles.get(normalized, "recorded_support_path" if moves else "no_support_path"),
            }
        )
    count = sum(1 for item in option_facts if item["moves"])
    category = "none" if count == 0 else "all" if count == len(options) else "exactly_one" if count == 1 else "multiple_not_all"
    facts = _base_facts(question)
    facts.update(
        moved_object=_label(question, "grandparent_label"),
        option_facts=option_facts,
        correct_values=answer,
        selection_category=category,
    )
    return facts, f"L3_attachment_chain.{category}"


def _coordinate_rotation_facts(
    question: dict[str, Any], qtype: str, answer: str
) -> tuple[dict[str, Any], str]:
    cardinal = qtype.endswith("allocentric")
    components = _direction_components(answer, cardinal=cardinal)
    try:
        angle = int(question.get("rotation_angle"))
    except (TypeError, ValueError) as exc:
        raise FactExtractionError("missing_rotation_angle", "rotation_angle missing") from exc
    if angle not in {90, 180, 270}:
        raise FactExtractionError("unsupported_rotation_angle", str(angle))
    old_direction = str(question.get("old_direction") or "").strip()
    if not old_direction:
        raise FactExtractionError("missing_old_direction", "old_direction missing")
    _direction_components(old_direction, cardinal=cardinal)
    facts = _base_facts(question)
    facts.update(
        rotation_angle=angle,
        rotation_direction="clockwise",
        old_result=old_direction,
        result=answer,
        change_state=_changed(question),
        **components,
    )
    if qtype == "coordinate_rotation_object_centric":
        facts.update(
            observer=_label(question, "obj_ref_label"),
            facing_object=_label(question, "obj_face_label"),
            target=_label(question, "obj_target_label"),
            heading_rule="preserve_original_ref_to_face_heading",
            cross_frame_layout=str(question.get("cross_frame_layout") or "single_view"),
        )
        layout = facts["cross_frame_layout"]
    else:
        facts.update(
            subject=_label(question, "obj_a_label"),
            reference=_label(question, "obj_b_label"),
        )
        layout = "camera" if qtype.endswith("agent") else "floor_plan"
        if cardinal:
            facts["camera_cardinal"] = _label(question, "camera_cardinal")
    signature = ".".join(
        [f"L3_{qtype}", str(angle), layout, components["layout"]]
    )
    return facts, signature


def build_fact_record(question: dict[str, Any]) -> CotFactRecord:
    qtype = str(question.get("type") or "").strip()
    if qtype not in SUPPORTED_TYPES:
        raise FactExtractionError("unsupported_question_type", qtype or "missing")
    letters = _answer_letters(question)
    semantic = _semantic_answer(question)

    if qtype in {"direction_agent", "direction_object_centric", "direction_allocentric"}:
        facts, signature = _direction_facts(question, qtype, str(semantic))
    elif qtype == "occlusion":
        facts, signature = _occlusion_facts(question, str(semantic))
    elif qtype == "distance":
        facts, signature = _distance_facts(question, str(semantic))
    elif qtype in {"object_move_agent", "object_move_object_centric", "object_move_allocentric"}:
        facts, signature = _move_direction_facts(question, qtype, str(semantic))
    elif qtype == "object_move_distance":
        facts, signature = _move_distance_facts(question, str(semantic))
    elif qtype == "object_move_occlusion":
        facts, signature = _move_occlusion_facts(question, str(semantic))
    elif qtype == "object_rotate_object_centric":
        facts, signature = _rotate_object_facts(question, str(semantic))
    elif qtype == "object_remove":
        facts, signature = _remove_facts(question, str(semantic))
    elif qtype == "attachment_chain":
        if not isinstance(semantic, list):
            raise FactExtractionError("invalid_multiselect_answer", "correct_values must be a list")
        facts, signature = _attachment_facts(question, semantic)
    else:
        facts, signature = _coordinate_rotation_facts(question, qtype, str(semantic))

    return CotFactRecord(
        question_uid=question_uid(question),
        question_type=qtype,
        signature_id=signature,
        facts=facts,
        semantic_answer=semantic,
        answer_letters=letters,
        validation={"fact_source": "benchmark_oracle", "passed": True},
    )
