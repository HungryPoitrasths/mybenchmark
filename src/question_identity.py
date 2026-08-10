"""Shared question identity helpers used by generation and sampling."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


QUESTION_OBJECT_ID_FIELDS = (
    "query_obj_id",
    "obj_a_id",
    "target_obj_id",
    "obj_target_id",
    "removed_obj_id",
    "obj_ref_id",
    "obj_face_id",
    "moved_obj_id",
    "parent_id",
    "root_id",
    "grandchild_id",
    "grandparent_id",
    "neighbor_id",
    "obj_b_id",
)

QUESTION_PAIR_FIELDS_BY_TYPE: dict[str, tuple[str, str]] = {
    "direction_agent": ("obj_a_id", "obj_b_id"),
    "distance": ("obj_a_id", "obj_b_id"),
    "direction_allocentric": ("obj_a_id", "obj_b_id"),
    "coordinate_rotation_agent": ("obj_a_id", "obj_b_id"),
    "coordinate_rotation_allocentric": ("obj_a_id", "obj_b_id"),
    "direction_object_centric": ("obj_ref_id", "obj_target_id"),
    "coordinate_rotation_object_centric": ("obj_ref_id", "obj_target_id"),
    "object_move_agent": ("moved_obj_id", "query_obj_id"),
    "object_move_distance": ("moved_obj_id", "query_obj_id"),
    "object_move_occlusion": ("moved_obj_id", "query_obj_id"),
    "object_move_object_centric": ("moved_obj_id", "query_obj_id"),
    "object_rotate_object_centric": ("moved_obj_id", "query_obj_id"),
    "object_move_allocentric": ("moved_obj_id", "query_obj_id"),
    "object_remove": ("removed_obj_id", "obj_b_id"),
    "attachment_chain": ("grandparent_id", "grandchild_id"),
    "attachment_move": ("root_id", "query_obj_id"),
}


def canonical_question_type(question: Mapping[str, Any]) -> str:
    return str(question.get("type", "")).strip().lower()


def question_pair_key(question: Mapping[str, Any]) -> tuple[str, str, str] | None:
    """Return a type-scoped object-pair key, excluding the scene scope."""
    question_type = canonical_question_type(question)
    if question.get("cross_frame_layout"):
        groups = question.get("object_frame_groups")
        if isinstance(groups, Mapping):
            frame_1_ids = ",".join(str(value) for value in groups.get("frame_1", []))
            frame_2_ids = ",".join(str(value) for value in groups.get("frame_2", []))
            if frame_1_ids and frame_2_ids:
                return (
                    question_type,
                    str(question.get("cross_frame_layout")),
                    f"{frame_1_ids}->{frame_2_ids}",
                )

    field_pair = QUESTION_PAIR_FIELDS_BY_TYPE.get(question_type)
    if field_pair is not None:
        left = question.get(field_pair[0])
        right = question.get(field_pair[1])
        if left is not None and right is not None:
            pair = tuple(sorted((str(left), str(right))))
            if pair[0] != pair[1]:
                return (question_type, pair[0], pair[1])

    unique_ids: list[str] = []
    for field in QUESTION_OBJECT_ID_FIELDS:
        value = question.get(field)
        if value is None:
            continue
        text = str(value)
        if text not in unique_ids:
            unique_ids.append(text)
        if len(unique_ids) > 2:
            break
    if len(unique_ids) == 2:
        pair = tuple(sorted(unique_ids))
        if pair[0] != pair[1]:
            return (question_type, pair[0], pair[1])
    return None
