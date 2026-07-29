from __future__ import annotations

import hashlib
from typing import Any

from .models import CotFactRecord
from .templates import load_template_library, templates_for_signature


def _article(label: str) -> str:
    text = str(label).strip()
    if text.lower().startswith(("the ", "a ", "an ", "none ")):
        return text
    return f"the {text}"


def _axis_phrase(facts: dict[str, Any], *, cardinal: bool = False) -> str:
    horizontal = facts["horizontal_axis"]
    depth = facts["depth_axis"]
    if facts["layout"] == "diagonal":
        return f"the {depth} component and the {horizontal} component both apply"
    component = horizontal if horizontal != "neutral" else depth
    neutral_axis = "north-south" if cardinal and horizontal != "neutral" else "east-west" if cardinal else "front-back" if horizontal != "neutral" else "left-right"
    return f"the decisive component is {component}, while the {neutral_axis} component is neutral"


def _direction_clauses(record: CotFactRecord) -> tuple[str, str, str]:
    f = record.facts
    if record.question_type == "direction_agent":
        observation = f"Use {_article(f['reference'])} as the origin and the first camera's horizontal frame for the comparison"
        transformation = f"For {_article(f['subject'])}, {_axis_phrase(f)}"
        conclusion = f"{_article(f['subject']).capitalize()} is {f['result']} of {_article(f['reference'])}"
    elif record.question_type == "direction_object_centric":
        observation = f"Set {_article(f['observer'])}'s forward direction toward {_article(f['facing_object'])}"
        transformation = f"In that object-centered horizontal frame, {_axis_phrase(f)} for {_article(f['subject'])}"
        conclusion = f"From {_article(f['observer'])}'s perspective, {_article(f['subject'])} is {f['result']}"
    else:
        observation = f"The first camera faces {f['camera_cardinal']}, which anchors the room's floor-plan directions"
        transformation = f"With {_article(f['reference'])} as the origin, {_axis_phrase(f, cardinal=True)} for {_article(f['subject'])}"
        conclusion = f"{_article(f['subject']).capitalize()} is {f['result']} of {_article(f['reference'])} on the floor plan"
    return observation, transformation, conclusion


def _occlusion_clauses(record: CotFactRecord) -> tuple[str, str, str]:
    f = record.facts
    target = _article(f["target"])
    if f["status"] == "not visible":
        observation = f"No part of {target} is visible in the current image"
        transformation = "With no visible portion, the benchmark labels the object as not visible rather than inferring why it is absent"
    elif f["status"] == "occluded":
        blocker = _article(f["blocker"]) if f.get("blocker") else "another object"
        observation = f"Part of {target} remains visible, while another part is blocked by {blocker}"
        transformation = "That is occlusion under the stated rule because the missing part is caused by an intervening object"
    else:
        observation = f"{target.capitalize()} is visible and no other object blocks its visible extent"
        transformation = "The stated rule therefore does not classify it as occluded"
    if f["framing"] == "partly_out_of_frame":
        transformation += "; projected geometry outside the image boundary is ignored for occlusion"
    conclusion = f"The status of {target} is {f['status']}"
    return observation, transformation, conclusion


def _distance_clauses(record: CotFactRecord) -> tuple[str, str, str]:
    f = record.facts
    observation = f"Measure from the closest surface point of {_article(f['object_a'])} to the closest surface point of {_article(f['object_b'])}, not between their centers"
    transformation = f"The resulting separation falls in the public interval labeled {f['distance_bin']}"
    conclusion = f"The approximate shortest distance is {f['distance_bin']}"
    return observation, transformation, conclusion


def _movement_phrase(f: dict[str, Any], *, frame: str) -> str:
    distance = f.get("movement_distance_m")
    distance_text = f" by {distance:g} m" if isinstance(distance, (int, float)) else ""
    return f"Move {_article(f['moved_object'])} {f['movement_direction']}{distance_text} in {frame}"


def _support_clause(f: dict[str, Any]) -> str:
    if f["movement_mode"] == "supported":
        return f"{_article(f['query_object']).capitalize()} is carried with {_article(f['moved_object'])} through the support chain"
    return f"{_article(f['query_object']).capitalize()} is the object moved directly"


def _move_direction_clauses(record: CotFactRecord) -> tuple[str, str, str]:
    f = record.facts
    qtype = record.question_type
    if qtype == "object_move_agent":
        frame = "the first camera's frame"
        relation_frame = "that same camera frame"
    elif qtype == "object_move_object_centric":
        frame = f"{_article(f['moved_object'])}'s initial facing frame"
        relation_frame = f"{_article(f['query_object'])}'s unchanged initial heading"
    else:
        frame = "the room's cardinal frame"
        relation_frame = f"the floor plan anchored by the camera facing {f['camera_cardinal']}"
    observation = f"{_support_clause(f)}; {_movement_phrase(f, frame=frame)}"
    if f["reference_motion_mode"] == "co_moved":
        motion_effect = f"{_article(f['reference_object']).capitalize()} moves with the same common support, so their qualitative relation is preserved"
    else:
        motion_effect = f"Recompute {_article(f['query_object'])} relative to {_article(f['reference_object'])} in {relation_frame}; the relation is {f['change_state']} from {f['old_result']} to {f['result']}"
    if qtype == "object_move_object_centric":
        motion_effect += f", with {_article(f['query_object'])} keeping its original facing direction"
    transformation = motion_effect
    conclusion = f"{_article(f['query_object']).capitalize()} is {f['result']} of {_article(f['reference_object'])}"
    return observation, transformation, conclusion


def _move_distance_clauses(record: CotFactRecord) -> tuple[str, str, str]:
    f = record.facts
    movement = _movement_phrase(f, frame="the first camera's frame")
    observation = f"{_support_clause(f)}; {movement}"
    if f["reference_motion_mode"] == "co_moved" and f["change_state"] == "preserved":
        transformation = f"{_article(f['object_b']).capitalize()} moves with the same common support, so the closest-surface distance remains in {f['distance_bin']}"
    else:
        transformation = f"Recompute the closest points of {_article(f['object_a'])} and {_article(f['object_b'])}; the distance is {f['distance_trend']} and its interval changes from {f['old_distance_bin']} to {f['distance_bin']}"
    conclusion = f"The new approximate shortest distance is {f['distance_bin']}"
    return observation, transformation, conclusion


def _move_occlusion_clauses(record: CotFactRecord) -> tuple[str, str, str]:
    f = record.facts
    movement = _movement_phrase(f, frame="the first camera's coordinate frame")
    observation = f"{_support_clause(f)}; {movement}"
    relation = f["pairwise_relation"]
    if relation == "query_occluded_by_ref":
        transformation = f"From the last main view, {_article(f['reference_object'])} blocks {_article(f['query_object'])} along the line of sight"
        conclusion = f"{_article(f['query_object']).capitalize()} is occluded by {_article(f['reference_object'])}"
    elif relation == "ref_occluded_by_query":
        transformation = f"From the last main view, {_article(f['query_object'])} blocks {_article(f['reference_object'])} along the line of sight"
        conclusion = f"{_article(f['reference_object']).capitalize()} is occluded by {_article(f['query_object'])}"
    else:
        transformation = f"From the last main view, neither specified object blocks the other along the line of sight"
        conclusion = "Neither of the two specified objects occludes the other"
    return observation, transformation, conclusion


def _rotate_object_clauses(record: CotFactRecord) -> tuple[str, str, str]:
    f = record.facts
    support = _support_clause(f)
    observation = f"{support}; orbit {_article(f['moved_object'])} {f['rotation_angle']} degrees {f['rotation_direction']} around {_article(f['facing_object'])}"
    ref_motion = "also follows that orbit" if f["reference_motion_mode"] == "co_orbited" else "stays at its scene position"
    transformation = f"At the new position, {_article(f['observer'])} faces {_article(f['facing_object'])} again, while {_article(f['target'])} {ref_motion}; the object-centered relation becomes {f['result']}"
    conclusion = f"From {_article(f['observer'])}'s new perspective, {_article(f['target'])} is {f['result']}"
    return observation, transformation, conclusion


def _remove_clauses(record: CotFactRecord) -> tuple[str, str, str]:
    f = record.facts
    observation = f"Before removal, the saved status for {_article(f['target'])} is {f['old_status']}"
    if f["old_status"] == f["new_status"]:
        transformation = f"Removing {_article(f['removed_object'])} does not change that visibility class, so it remains {f['new_status']}"
    else:
        transformation = f"After {_article(f['removed_object'])} is absent, the visibility class improves from {f['old_status']} to {f['new_status']}"
    conclusion = f"The new status of {_article(f['target'])} is {f['new_status']}"
    return observation, transformation, conclusion


def _attachment_clauses(record: CotFactRecord) -> tuple[str, str, str]:
    f = record.facts
    moving = []
    stationary = []
    for item in f["option_facts"]:
        phrase = _article(item["option"])
        if phrase.lower().startswith("none "):
            stationary.append("the catch-all 'None of the above' option does not apply")
            continue
        if item["moves"]:
            if item["path_kind"] == "direct_support_path":
                moving.append(f"a direct support link connects the moved root to {phrase}")
            elif item["path_kind"] == "two_hop_support_path":
                moving.append(f"a two-hop support chain connects the moved root to {phrase}")
            else:
                moving.append(f"a recorded support path connects the moved root to {phrase}")
        else:
            stationary.append(f"{phrase} has no support path from the moved object")
    observation = f"Treat attachment as a directed support graph rooted at {_article(f['moved_object'])}"
    details = moving + stationary
    transformation = "; ".join(details) if details else "No option has a recorded support path from that root"
    transformation = transformation[:1].upper() + transformation[1:]
    selected = ", ".join(_article(value) for value in f["correct_values"]) or "none of the options"
    conclusion = f"The objects displaced with {_article(f['moved_object'])} are {selected}"
    return observation, transformation, conclusion


def _coordinate_rotation_clauses(record: CotFactRecord) -> tuple[str, str, str]:
    f = record.facts
    qtype = record.question_type
    if qtype == "coordinate_rotation_agent":
        observation = f"Before rotation, {_article(f['subject'])} is {f['old_result']} of {_article(f['reference'])} in the unchanged camera frame"
        transformation = f"Rotating the whole layout {f['rotation_angle']} degrees clockwise rotates that relative direction to {f['result']} while the camera pose stays fixed"
        conclusion = f"{_article(f['subject']).capitalize()} is {f['result']} of {_article(f['reference'])}"
    elif qtype == "coordinate_rotation_allocentric":
        observation = f"The camera facing {f['camera_cardinal']} anchors the room's cardinal axes, and the original relation is {f['old_result']}"
        transformation = f"A {f['rotation_angle']}-degree clockwise layout rotation maps that floor-plan direction to {f['result']}"
        conclusion = f"{_article(f['subject']).capitalize()} is {f['result']} of {_article(f['reference'])} on the floor plan"
    else:
        observation = f"Use the original direction from {_article(f['observer'])} toward {_article(f['facing_object'])} as the observer's fixed horizontal heading"
        transformation = f"Rotate all object positions {f['rotation_angle']} degrees clockwise around the room center but keep that heading unchanged; {_article(f['target'])}'s relation changes from {f['old_result']} to {f['result']}"
        conclusion = f"From {_article(f['observer'])}'s rotated position, {_article(f['target'])} is {f['result']}"
    return observation, transformation, conclusion


def reasoning_clauses(record: CotFactRecord) -> tuple[str, str, str]:
    qtype = record.question_type
    if qtype in {"direction_agent", "direction_object_centric", "direction_allocentric"}:
        return _direction_clauses(record)
    if qtype == "occlusion":
        return _occlusion_clauses(record)
    if qtype == "distance":
        return _distance_clauses(record)
    if qtype in {"object_move_agent", "object_move_object_centric", "object_move_allocentric"}:
        return _move_direction_clauses(record)
    if qtype == "object_move_distance":
        return _move_distance_clauses(record)
    if qtype == "object_move_occlusion":
        return _move_occlusion_clauses(record)
    if qtype == "object_rotate_object_centric":
        return _rotate_object_clauses(record)
    if qtype == "object_remove":
        return _remove_clauses(record)
    if qtype == "attachment_chain":
        return _attachment_clauses(record)
    return _coordinate_rotation_clauses(record)


def render_response(
    record: CotFactRecord,
    *,
    seed: int = 42,
    template_library: dict[str, Any] | None = None,
) -> tuple[str, str]:
    library = template_library or load_template_library()
    templates = templates_for_signature(library, record.signature_id)
    digest = hashlib.sha256(
        f"{seed}|{record.question_uid}|{record.signature_id}".encode("utf-8")
    ).digest()
    template = templates[int.from_bytes(digest[:8], "big") % len(templates)]
    observation, transformation, conclusion = reasoning_clauses(record)
    conclusion_after_connector = conclusion[:1].lower() + conclusion[1:]
    reasoning = template["template"].format(
        observation=observation,
        transformation=transformation,
        conclusion=conclusion_after_connector,
    ).strip()
    answer = " ".join(record.answer_letters)
    return f"{reasoning}\nAnswer: {answer}", template["id"]
