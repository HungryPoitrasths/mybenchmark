"""Stage 6: QA generation.

Generates multiple-choice questions from computed spatial relations and
virtual operation results.
"""

from __future__ import annotations

import json
import math
import random
import logging
from pathlib import Path
from typing import Any

import numpy as np

from .relation_engine import (
    ALL_DIRECTIONS_10,
    ALL_ALLOCENTRIC_10,
    CARDINAL_DIRECTIONS_8,
    HORIZONTAL_DIRECTIONS,
    MIN_DIRECTION_DISTANCE,
    compute_all_relations,
    find_changed_relations,
    primary_direction,
    primary_direction_object_centric,
    primary_direction_allocentric,
    camera_cardinal_direction,
    compute_distance,
)
from .virtual_ops import (
    apply_viewpoint_change,
    apply_removal,
    apply_coordinate_rotation,
    find_meaningful_movement,
)
from .utils.colmap_loader import CameraPose

logger = logging.getLogger(__name__)

# Default template file; can be overridden
_TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "templates"

ALL_DIRECTIONS = ALL_DIRECTIONS_10
ALL_DIRECTIONS_ALLOCENTRIC = ALL_ALLOCENTRIC_10
ALL_DISTANCES = ["touching (<0.5m)", "very close (0.5-1.5m)", "close (1.5-3m)", "far (>3m)"]
ALL_OCCLUSION = ["fully visible", "partially occluded", "not visible"]
YES_NO = ["Yes", "No"]


def _the(label: str) -> str:
    """Prepend 'the' to an object label for natural English grammar.

    'shoes' → 'the shoes', 'table' → 'the table'.
    Avoids 'where is shoes positioned' type errors.
    """
    return f"the {label}"


def _mention(role: str, label: str, obj_id: int | None = None) -> dict[str, Any]:
    """Create a normalised mentioned-object record for a question."""
    return {"role": role, "obj_id": obj_id, "label": label}

# Labels to exclude from ALL question types (not just L2).
# Structural elements, generic labels, and uninformative categories.
# Must stay in sync with scene_parser.EXCLUDED_LABELS.
EXCLUDED_LABELS = {
    # Structural / architectural
    "floor", "wall", "ceiling", "room", "ground",
    "door", "window", "stairs", "pillar", "column",
    "doorframe", "windowsill", "hand rail", "shower",
    "shower curtain rod", "bathroom stall", "bathroom stall door",
    "ledge", "structure", "closet", "breakfast bar", "shower curtain",
    # Generic / uninformative
    "object", "otherfurniture", "otherprop", "otherstructure",
    "unknown", "misc", "stuff",
    # Reflective / transparent — depth sensor unreliable
    "mirror", "glass", "monitor", "tv",
    # Ambiguous / vague
    "case", "tube", "board", "sign", "frame", "paper", "lotion",
    # Boundary-unclear / large amorphous / unreliable 3D annotation
    "counter", "couch", "clothing", "clothes", "cloth", "blanket", "rug",
    "shelf", "bookshelf", "shelves", "rack", "storage shelf",
    # Too small to reliably identify in images
    "power outlet", "light switch", "fire alarm", "controller",
    "power strip", "soda can", "starbucks cup", "battery disposal jar",
    "can", "water bottle", "paper cutter",
}


def _load_templates() -> dict:
    """Load question templates from the JSON file."""
    tpl_path = _TEMPLATE_DIR / "question_templates.json"
    if tpl_path.exists():
        with open(tpl_path, "r", encoding="utf-8") as f:
            return json.load(f)
    # Fallback inline templates
    return _default_templates()


def _default_templates() -> dict:
    return {
        # ==== L1 — Static perception ====

        # --- Ego-centric ---
        "L1_direction": [
            "From the camera's viewpoint, {obj_a} is in which direction relative to {obj_b}?",
            "Looking at the scene from the camera's perspective, where is {obj_a} positioned relative to {obj_b}?",
        ],
        "L1_distance": [
            "Approximately how far apart are {obj_a} and {obj_b}?",
            "What is the approximate distance between {obj_a} and {obj_b}?",
        ],
        "L1_occlusion": [
            "What is the visibility status of {obj_a} from the current viewpoint?",
            "Is {obj_a} fully visible from the current camera angle?",
        ],

        # --- Object-centric ---
        "L1_direction_object_centric": [
            "Imagine you are {obj_ref} and facing toward {obj_face}. From your perspective, in which direction is {obj_target}?",
            "If you were {obj_ref}, looking toward {obj_face}, where would {obj_target} be?",
        ],

        # --- Allocentric ---
        "L1_direction_allocentric": [
            "The camera is facing {camera_cardinal} in this scene. On the room's floor plan, in which cardinal direction is {obj_a} from {obj_b}?",
            "In this image the camera faces {camera_cardinal}. Viewed from above on the room's layout, {obj_a} is in which cardinal direction relative to {obj_b}?",
        ],

        # ==== L2 — Intervention ====

        # --- Ego-centric (existing) ---
        "L2_object_move": [
            "From the camera's perspective, imagine moving {obj_a} {direction} by {distance}. After this change, what is the relative position of {obj_b} to {obj_c}?",
            "From the camera's perspective, if we move {obj_a} {direction} by {distance}, where is {obj_b} relative to {obj_c}?",
        ],
        "L2_object_move_distance": [
            "From the camera's perspective, if {obj_a} is moved {direction} by {distance}, how far apart would {obj_b} and {obj_c} be?",
            "From the camera's perspective, imagine moving {obj_a} {direction} by {distance}. After this change, what is the distance between {obj_b} and {obj_c}?",
        ],
        "L2_viewpoint_move": [
            "If the observer moves {direction} by {distance} from the current position, would {obj_a} become visible or occluded?",
        ],
        "L2_object_remove": [
            "If {obj_a} were removed from the scene, what would be the visibility status of {obj_b} from the current viewpoint?",
        ],

        # --- Object-centric ---
        "L2_object_move_object_centric": [
            "On the room's floor plan (north-up), move {obj_moved} {distance} to the {direction}. Then imagine you are {obj_ref} facing {obj_face}. In which direction is {obj_moved}?",
            "Suppose {obj_moved} is shifted {distance} toward {direction} on the floor plan. After this change, if you stand at {obj_ref} and face {obj_face}, where is {obj_moved}?",
        ],

        # --- Allocentric ---
        "L2_object_move_allocentric": [
            "If {obj_moved} is moved {distance} to the {direction}, the camera faces {camera_cardinal}. On the floor plan, in which cardinal direction would {obj_moved} be from {obj_ref}?",
            "After moving {obj_moved} {distance} to the {direction}, on the room's layout (camera facing {camera_cardinal}), in which cardinal direction is {obj_moved} from {obj_ref}?",
        ],

        # ==== L3 — Counterfactual / multi-hop ====

        "L3_support_chain": [
            "Suppose {obj_a} were moved to a different location. Which of the following objects would also be displaced from their current positions?",
            "If {obj_a} were relocated elsewhere in the room, which of the following objects would also change position?",
            "Imagine {obj_a} is moved to a new spot. Which of the following objects would also be displaced as a result?",
        ],

        # --- Ego-centric (rewritten — 方案B) ---
        "L3_coordinate_rotation": [
            "Imagine all the furniture in the room is rearranged by rotating everything {angle} degrees clockwise around the center of the room (viewed from above). You remain standing in the exact same spot, looking in the same direction. After this rearrangement, in which direction is {obj_a} relative to {obj_b}?",
            "Suppose someone rotates all objects in the room {angle} degrees clockwise around the room's center (as seen from above), while you stay in place with the same viewing angle. In which direction would {obj_a} be relative to {obj_b}?",
        ],

        # --- Object-centric ---
        "L3_coordinate_rotation_object_centric": [
            "Imagine all furniture is rotated {angle} degrees clockwise around the room center (viewed from above). After this rearrangement, if you are {obj_ref} at its new position and face toward {obj_face}'s new position, in which direction is {obj_target}?",
        ],

        # --- Allocentric ---
        "L3_coordinate_rotation_allocentric": [
            "Imagine all furniture is rotated {angle} degrees clockwise around the room center (viewed from above). The camera, facing {camera_cardinal}, remains in place. On the floor plan, in which cardinal direction is {obj_a} from {obj_b}?",
        ],
    }


def generate_options(
    correct_answer: str,
    answer_pool: list[str],
    n_options: int = 4,
) -> tuple[list[str], str]:
    """Generate MCQ options from an answer pool.

    When the correct answer is a vertical direction (above/below), horizontal
    directions are excluded from distractors and vice versa — this prevents
    distractors that are also plausibly correct (e.g. an object that is
    "above" is also "front" if it happens to be slightly forward).

    Returns (shuffled_options, correct_letter).
    """
    VERTICAL = {"above", "below"}
    HORIZONTAL = set(HORIZONTAL_DIRECTIONS)  # front/back/left/right/…

    if correct_answer in VERTICAL:
        exclude = HORIZONTAL
    elif correct_answer in HORIZONTAL:
        exclude = VERTICAL
    else:
        exclude = set()

    options = [correct_answer]
    distractors = [a for a in answer_pool if a != correct_answer and a not in exclude]
    random.shuffle(distractors)
    options.extend(distractors[: n_options - 1])

    # If the strict pool is too small, fall back to the excluded pool before
    # introducing a single "None of the above" option.
    if len(options) < n_options:
        fallback = [
            a for a in answer_pool
            if a != correct_answer and a not in options
        ]
        random.shuffle(fallback)
        options.extend(fallback[: n_options - len(options)])

    if len(options) < n_options and "None of the above" not in options:
        options.append("None of the above")

    has_none_of_above = "None of the above" in options
    shuffled = [opt for opt in options if opt != "None of the above"]
    random.shuffle(shuffled)
    if has_none_of_above:
        shuffled.append("None of the above")

    options = shuffled
    correct_idx = options.index(correct_answer)
    correct_letter = chr(65 + correct_idx)  # A, B, C, D
    return options, correct_letter


# ---------------------------------------------------------------------------
#  L1 generators
# ---------------------------------------------------------------------------

def generate_l1_direction(
    relation: dict,
    templates: dict,
) -> dict | None:
    """Generate an L1-direction question from a precomputed relation."""
    if relation["ambiguity_score"] > 0.7:
        return None  # too ambiguous
    if relation["distance_m"] < MIN_DIRECTION_DISTANCE:
        return None  # too close — annotation errors dominate
    if relation["obj_a_label"] == relation["obj_b_label"]:
        return None  # same label → "chair relative to chair" is meaningless

    correct = relation["direction_b_rel_a"]
    tpl = random.choice(templates.get("L1_direction", _default_templates()["L1_direction"]))
    question_text = tpl.format(
        obj_a=_the(relation["obj_b_label"]),  # "where is B relative to A?"
        obj_b=_the(relation["obj_a_label"]),
    )
    options, answer = generate_options(correct, ALL_DIRECTIONS)

    return {
        "level": "L1",
        "type": "direction",
        "question": question_text,
        "options": options,
        "answer": answer,
        "correct_value": correct,
        "obj_a_id": relation["obj_a_id"],
        "obj_b_id": relation["obj_b_id"],
        "obj_a_label": relation["obj_a_label"],
        "obj_b_label": relation["obj_b_label"],
        "mentioned_objects": [
            _mention("reference", relation["obj_a_label"], relation["obj_a_id"]),
            _mention("target", relation["obj_b_label"], relation["obj_b_id"]),
        ],
        "ambiguity_score": relation["ambiguity_score"],
        "relation_unchanged": False,
    }


def generate_l1_distance(
    relation: dict,
    templates: dict,
) -> dict | None:
    """Generate an L1-distance question."""
    if relation["near_boundary"]:
        return None
    if relation["obj_a_label"] == relation["obj_b_label"]:
        return None  # same label → ambiguous question

    correct = relation["distance_bin"]
    tpl = random.choice(templates.get("L1_distance", _default_templates()["L1_distance"]))
    question_text = tpl.format(
        obj_a=_the(relation["obj_a_label"]),
        obj_b=_the(relation["obj_b_label"]),
    )
    options, answer = generate_options(correct, ALL_DISTANCES)

    return {
        "level": "L1",
        "type": "distance",
        "question": question_text,
        "options": options,
        "answer": answer,
        "correct_value": correct,
        "obj_a_id": relation["obj_a_id"],
        "obj_b_id": relation["obj_b_id"],
        "obj_a_label": relation["obj_a_label"],
        "obj_b_label": relation["obj_b_label"],
        "mentioned_objects": [
            _mention("obj_a", relation["obj_a_label"], relation["obj_a_id"]),
            _mention("obj_b", relation["obj_b_label"], relation["obj_b_id"]),
        ],
        "ambiguity_score": 0.0,
        "near_boundary": relation["near_boundary"],
        "relation_unchanged": False,
    }


def generate_l1_occlusion(
    obj: dict,
    occlusion_status: str,
    templates: dict,
) -> dict | None:
    """Generate an L1-occlusion question (per-object, not pairwise).

    Occlusion is now depth-map based: we know whether the object is visible
    from the camera, but not *which* other object occludes it.  Templates
    only reference the target object.
    """
    if occlusion_status == "unknown":
        return None

    correct = occlusion_status
    tpl = random.choice(templates.get("L1_occlusion", _default_templates()["L1_occlusion"]))
    question_text = tpl.format(obj_a=_the(obj["label"]))
    options, answer = generate_options(correct, ALL_OCCLUSION, n_options=3)

    return {
        "level": "L1",
        "type": "occlusion",
        "question": question_text,
        "options": options,
        "answer": answer,
        "correct_value": correct,
        "obj_a_id": obj["id"],
        "obj_a_label": obj["label"],
        "mentioned_objects": [_mention("target", obj["label"], obj["id"])],
        "ambiguity_score": 0.0,
        "relation_unchanged": False,
    }


# ---------------------------------------------------------------------------
#  L1 generators — new reference frames
# ---------------------------------------------------------------------------

def generate_l1_direction_object_centric(
    objects: list[dict],
    templates: dict,
    max_questions: int = 20,
) -> list[dict]:
    """Generate L1 object-centric direction questions.

    Uses triples (ref, face, target) to define a reference frame at *ref*
    facing *face*, then asks for the direction of *target*.
    """
    questions: list[dict] = []
    n = len(objects)
    if n < 3:
        return questions

    tpl_list = templates.get(
        "L1_direction_object_centric",
        _default_templates()["L1_direction_object_centric"],
    )

    candidates: list[dict] = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            for k in range(n):
                if k == i or k == j:
                    continue
                ref = objects[i]
                face = objects[j]
                target = objects[k]
                # All three labels must be distinct for unambiguous reference
                if len({ref["label"], face["label"], target["label"]}) < 3:
                    continue

                ref_c = np.array(ref["center"])
                face_c = np.array(face["center"])
                target_c = np.array(target["center"])

                # Require minimum distances to avoid annotation-error-dominated results
                if np.linalg.norm(face_c - ref_c) < MIN_DIRECTION_DISTANCE:
                    continue
                if np.linalg.norm(target_c - ref_c) < MIN_DIRECTION_DISTANCE:
                    continue

                direction, ambiguity = primary_direction_object_centric(
                    ref_c, face_c, target_c,
                )
                if ambiguity > 0.7:
                    continue

                tpl = random.choice(tpl_list)
                question_text = tpl.format(
                    obj_ref=_the(ref["label"]),
                    obj_face=_the(face["label"]),
                    obj_target=_the(target["label"]),
                )
                options, answer = generate_options(direction, ALL_DIRECTIONS)
                candidates.append({
                    "level": "L1",
                    "type": "direction_object_centric",
                    "reference_frame": "object_centric",
                    "question": question_text,
                    "options": options,
                    "answer": answer,
                    "correct_value": direction,
                    "obj_ref_id": ref["id"],
                    "obj_ref_label": ref["label"],
                    "obj_face_id": face["id"],
                    "obj_face_label": face["label"],
                    "obj_target_id": target["id"],
                    "obj_target_label": target["label"],
                    "mentioned_objects": [
                        _mention("reference_origin", ref["label"], ref["id"]),
                        _mention("reference_facing", face["label"], face["id"]),
                        _mention("target", target["label"], target["id"]),
                    ],
                    "ambiguity_score": ambiguity,
                    "relation_unchanged": False,
                })

    if len(candidates) > max_questions:
        candidates = random.sample(candidates, max_questions)
    return candidates


def generate_l1_direction_allocentric(
    objects: list[dict],
    camera_pose: CameraPose,
    templates: dict,
    max_questions: int = 20,
) -> list[dict]:
    """Generate L1 allocentric (cardinal) direction questions.

    Provides the camera's cardinal facing direction so the model can anchor
    absolute directions from the image.
    """
    questions: list[dict] = []
    cam_cardinal = camera_cardinal_direction(camera_pose)
    n = len(objects)

    tpl_list = templates.get(
        "L1_direction_allocentric",
        _default_templates()["L1_direction_allocentric"],
    )

    candidates: list[dict] = []
    for i in range(n):
        for j in range(i + 1, n):
            a, b = objects[i], objects[j]
            if a["label"] == b["label"]:
                continue

            a_c = np.array(a["center"])
            b_c = np.array(b["center"])
            if np.linalg.norm(b_c - a_c) < MIN_DIRECTION_DISTANCE:
                continue
            direction, ambiguity = primary_direction_allocentric(a_c, b_c)
            if ambiguity > 0.7:
                continue

            tpl = random.choice(tpl_list)
            question_text = tpl.format(
                camera_cardinal=cam_cardinal,
                obj_a=_the(a["label"]),
                obj_b=_the(b["label"]),
            )
            options, answer = generate_options(direction, ALL_DIRECTIONS_ALLOCENTRIC)
            candidates.append({
                "level": "L1",
                "type": "direction_allocentric",
                "reference_frame": "allocentric",
                "question": question_text,
                "options": options,
                "answer": answer,
                "correct_value": direction,
                "camera_cardinal": cam_cardinal,
                "obj_a_id": a["id"],
                "obj_a_label": a["label"],
                "obj_b_id": b["id"],
                "obj_b_label": b["label"],
                "mentioned_objects": [
                    _mention("obj_a", a["label"], a["id"]),
                    _mention("obj_b", b["label"], b["id"]),
                ],
                "ambiguity_score": ambiguity,
                "relation_unchanged": False,
            })

    if len(candidates) > max_questions:
        candidates = random.sample(candidates, max_questions)
    return candidates


# ---------------------------------------------------------------------------
#  L2 generators
# ---------------------------------------------------------------------------

def generate_l2_object_move(
    objects: list[dict],
    support_graph: dict[int, list[int]],
    camera_pose: CameraPose,
    templates: dict,
    max_per_object: int = 3,
    room_bounds: dict | None = None,
) -> list[dict]:
    """Generate L2.1 object-movement questions for a scene."""
    questions: list[dict] = []
    obj_map = {o["id"]: o for o in objects}

    for obj in objects:
        # Skip structural room elements — they cannot be "moved" in any
        # meaningful physical sense and confuse human annotators.
        if obj.get("label", "").lower() in EXCLUDED_LABELS:
            continue

        delta, changed = find_meaningful_movement(
            objects, support_graph, obj["id"], camera_pose,
            room_bounds=room_bounds,
        )
        if delta is None:
            continue

        # Only keep relation changes involving the moved object or its
        # support-chain dependents — otherwise the question is nonsensical
        # (e.g., "move cabinet → what happens between tv and table?").
        chain_ids = set(get_support_chain_ids(obj["id"], support_graph))
        chain_ids.add(obj["id"])
        changed = [
            ch for ch in changed
            if ch["obj_a_id"] in chain_ids or ch["obj_b_id"] in chain_ids
        ]
        if not changed:
            continue

        # Describe the movement in natural language
        direction_desc = _delta_to_description(delta, camera_pose)
        distance_desc = f"{np.linalg.norm(delta):.1f}m"

        tpl_list = templates.get("L2_object_move", _default_templates()["L2_object_move"])

        obj_questions: list[dict] = []
        for ch in changed:
            # Ensure the changed relation directly involves the moved object
            # so the question makes intuitive sense ("move A → what happens
            # between A and B?" is clear; "move A → what happens between C
            # and D?" is confusing even if C sits on A's support chain).
            if obj["id"] not in (ch["obj_a_id"], ch["obj_b_id"]):
                continue

            # Pick a changed relation field
            for field, vals in ch["changes"].items():
                if field == "direction_b_rel_a":
                    pool = ALL_DIRECTIONS
                    field_tpl_key = "L2_object_move"
                elif field == "distance_bin":
                    pool = ALL_DISTANCES
                    field_tpl_key = "L2_object_move_distance"
                elif field in ("occlusion_a", "occlusion_b"):
                    pool = ALL_OCCLUSION
                    field_tpl_key = "L2_object_move"
                else:
                    continue

                obj_b_label = obj_map.get(ch["obj_b_id"], {}).get("label", "object")
                obj_c_label = obj_map.get(ch["obj_a_id"], {}).get("label", "object")

                field_tpl_list = templates.get(field_tpl_key, _default_templates()[field_tpl_key])
                tpl = random.choice(field_tpl_list)
                question_text = tpl.format(
                    obj_a=_the(obj["label"]),
                    direction=direction_desc,
                    distance=distance_desc,
                    obj_b=_the(obj_b_label),
                    obj_c=_the(obj_c_label),
                )
                options, answer = generate_options(vals["new"], pool)

                obj_questions.append({
                    "level": "L2",
                    "type": "object_move",
                    "question": question_text,
                    "options": options,
                    "answer": answer,
                    "correct_value": vals["new"],
                    "moved_obj_id": obj["id"],
                    "moved_obj_label": obj["label"],
                    "obj_b_id": ch["obj_b_id"],
                    "obj_b_label": obj_b_label,
                    "obj_c_id": ch["obj_a_id"],
                    "obj_c_label": obj_c_label,
                    "mentioned_objects": [
                        _mention("moved_object", obj["label"], obj["id"]),
                        _mention("relation_obj_b", obj_b_label, ch["obj_b_id"]),
                        _mention("relation_obj_c", obj_c_label, ch["obj_a_id"]),
                    ],
                    "delta": delta.tolist(),
                    "relation_unchanged": False,
                    "has_support_chain": len(get_support_chain_ids(obj["id"], support_graph)) > 0,
                })

        # Cap per moved object to avoid flooding from high-connectivity objects
        if len(obj_questions) > max_per_object:
            obj_questions = random.sample(obj_questions, max_per_object)
        questions.extend(obj_questions)

    return questions


def generate_l2_viewpoint_move(
    objects: list[dict],
    camera_pose: CameraPose,
    templates: dict,
) -> list[dict]:
    """Generate L2.2 viewpoint-movement questions.

    Detects direction/distance changes only (no occlusion — can't synthesise
    a new depth map for the moved viewpoint).
    """
    questions: list[dict] = []
    tpl_list = templates.get("L2_viewpoint_move", _default_templates()["L2_viewpoint_move"])

    # No depth for either side — can't synthesise a new depth map after
    # viewpoint change.  These L2 questions only detect direction/distance changes.
    original_relations = compute_all_relations(objects, camera_pose, None, None)

    for direction in ("right", "left", "forward"):
        for dist in (2.0, 3.0):
            new_pose = apply_viewpoint_change(camera_pose, direction, dist)
            new_relations = compute_all_relations(objects, new_pose, None, None)
            changed = find_changed_relations(original_relations, new_relations)

            for ch in changed:
                for field, vals in ch["changes"].items():
                    if field == "direction_b_rel_a":
                        pool = ALL_DIRECTIONS
                    elif field == "distance_bin":
                        pool = ALL_DISTANCES
                    else:
                        continue
                    obj_label = next(
                        (o["label"] for o in objects if o["id"] == ch["obj_a_id"]),
                        "object",
                    )
                    tpl = random.choice(tpl_list)
                    question_text = tpl.format(
                        direction=direction,
                        distance=f"{dist:.0f}m",
                        obj_a=_the(obj_label),
                    )
                    options, answer = generate_options(vals["new"], pool)
                    questions.append({
                        "level": "L2",
                        "type": "viewpoint_move",
                        "question": question_text,
                        "options": options,
                        "answer": answer,
                        "correct_value": vals["new"],
                        "obj_a_id": ch["obj_a_id"],
                        "obj_a_label": obj_label,
                        "mentioned_objects": [
                            _mention("target", obj_label, ch["obj_a_id"]),
                        ],
                        "relation_unchanged": False,
                    })

    return questions


def generate_l2_object_remove(
    objects: list[dict],
    support_graph: dict[int, list[int]],
    camera_pose: CameraPose,
    templates: dict,
) -> list[dict]:
    """Generate L2.3 object-removal questions."""
    questions: list[dict] = []
    tpl_list = templates.get("L2_object_remove", _default_templates()["L2_object_remove"])

    # No depth for either side — can't synthesise depth after removing an
    # object.  These L2 questions detect direction/distance changes.
    original_relations = compute_all_relations(objects, camera_pose, None, None)

    for obj in objects:
        remaining = apply_removal(objects, support_graph, obj["id"], cascade=False)
        if len(remaining) < 2:
            continue
        new_relations = compute_all_relations(remaining, camera_pose, None, None)
        changed = find_changed_relations(original_relations, new_relations)

        for ch in changed:
            for field, vals in ch["changes"].items():
                if field == "direction_b_rel_a":
                    pool = ALL_DIRECTIONS
                elif field == "distance_bin":
                    pool = ALL_DISTANCES
                else:
                    continue
                other_label = next(
                    (o["label"] for o in objects if o["id"] == ch["obj_a_id"]),
                    "object",
                )
                tpl = random.choice(tpl_list)
                question_text = tpl.format(
                    obj_a=_the(obj["label"]),
                    obj_b=_the(other_label),
                )
                options, answer = generate_options(vals["new"], pool)
                questions.append({
                    "level": "L2",
                    "type": "object_remove",
                    "question": question_text,
                    "options": options,
                    "answer": answer,
                    "correct_value": vals["new"],
                    "removed_obj_id": obj["id"],
                    "removed_obj_label": obj["label"],
                    "obj_b_id": ch["obj_a_id"],
                    "obj_b_label": other_label,
                    "mentioned_objects": [
                        _mention("removed_object", obj["label"], obj["id"]),
                        _mention("remaining_object", other_label, ch["obj_a_id"]),
                    ],
                    "relation_unchanged": False,
                })

    return questions


# ---------------------------------------------------------------------------
#  L2 generators — new reference frames
# ---------------------------------------------------------------------------

def generate_l2_object_move_object_centric(
    objects: list[dict],
    support_graph: dict[int, list[int]],
    camera_pose: CameraPose,
    templates: dict,
    max_per_object: int = 3,
    room_bounds: dict | None = None,
) -> list[dict]:
    """L2 object-move questions answered in object-centric frame.

    After moving *obj_moved*, asks: standing at *obj_ref* facing *obj_face*,
    where is *obj_moved* now?
    """
    questions: list[dict] = []
    cam_cardinal = camera_cardinal_direction(camera_pose)
    obj_map = {o["id"]: o for o in objects}
    tpl_list = templates.get(
        "L2_object_move_object_centric",
        _default_templates()["L2_object_move_object_centric"],
    )

    for obj in objects:
        if obj.get("label", "").lower() in EXCLUDED_LABELS:
            continue

        delta, _changed = find_meaningful_movement(
            objects, support_graph, obj["id"], camera_pose,
            room_bounds=room_bounds,
        )
        if delta is None:
            continue

        direction_desc = _delta_to_cardinal_description(delta)
        distance_desc = f"{np.linalg.norm(delta):.1f}m"

        # Compute the moved object's new center
        new_center = np.array(obj["center"]) + delta

        obj_questions: list[dict] = []
        for ref in objects:
            if ref["id"] == obj["id"]:
                continue
            for face in objects:
                if face["id"] == obj["id"] or face["id"] == ref["id"]:
                    continue
                if len({obj["label"], ref["label"], face["label"]}) < 3:
                    continue

                ref_c = np.array(ref["center"])
                face_c = np.array(face["center"])
                new_dir, amb = primary_direction_object_centric(
                    ref_c, face_c, new_center,
                )
                if amb > 0.7:
                    continue

                # Check that direction actually changed from the original
                old_dir, _ = primary_direction_object_centric(
                    ref_c, face_c, np.array(obj["center"]),
                )
                if old_dir == new_dir:
                    continue

                tpl = random.choice(tpl_list)
                question_text = tpl.format(
                    obj_moved=_the(obj["label"]),
                    direction=direction_desc,
                    distance=distance_desc,
                    obj_ref=_the(ref["label"]),
                    obj_face=_the(face["label"]),
                    camera_cardinal=cam_cardinal,
                )
                options, answer = generate_options(new_dir, ALL_DIRECTIONS)
                obj_questions.append({
                    "level": "L2",
                    "type": "object_move_object_centric",
                    "reference_frame": "object_centric",
                    "question": question_text,
                    "options": options,
                    "answer": answer,
                    "correct_value": new_dir,
                    "moved_obj_id": obj["id"],
                    "moved_obj_label": obj["label"],
                    "obj_ref_id": ref["id"],
                    "obj_ref_label": ref["label"],
                    "obj_face_id": face["id"],
                    "obj_face_label": face["label"],
                    "mentioned_objects": [
                        _mention("moved_object", obj["label"], obj["id"]),
                        _mention("reference_origin", ref["label"], ref["id"]),
                        _mention("reference_facing", face["label"], face["id"]),
                    ],
                    "delta": delta.tolist(),
                    "relation_unchanged": False,
                })

        if len(obj_questions) > max_per_object:
            obj_questions = random.sample(obj_questions, max_per_object)
        questions.extend(obj_questions)

    return questions


def generate_l2_object_move_allocentric(
    objects: list[dict],
    support_graph: dict[int, list[int]],
    camera_pose: CameraPose,
    templates: dict,
    max_per_object: int = 3,
    room_bounds: dict | None = None,
) -> list[dict]:
    """L2 object-move questions answered in allocentric (cardinal) frame."""
    questions: list[dict] = []
    cam_cardinal = camera_cardinal_direction(camera_pose)
    tpl_list = templates.get(
        "L2_object_move_allocentric",
        _default_templates()["L2_object_move_allocentric"],
    )

    for obj in objects:
        if obj.get("label", "").lower() in EXCLUDED_LABELS:
            continue

        delta, _changed = find_meaningful_movement(
            objects, support_graph, obj["id"], camera_pose,
            room_bounds=room_bounds,
        )
        if delta is None:
            continue

        direction_desc = _delta_to_cardinal_description(delta)
        distance_desc = f"{np.linalg.norm(delta):.1f}m"
        new_center = np.array(obj["center"]) + delta

        obj_questions: list[dict] = []
        for ref in objects:
            if ref["id"] == obj["id"]:
                continue
            if ref["label"] == obj["label"]:
                continue

            ref_c = np.array(ref["center"])
            new_dir, amb = primary_direction_allocentric(new_center, ref_c)
            if amb > 0.7:
                continue

            old_dir, _ = primary_direction_allocentric(np.array(obj["center"]), ref_c)
            if old_dir == new_dir:
                continue

            tpl = random.choice(tpl_list)
            question_text = tpl.format(
                obj_moved=_the(obj["label"]),
                direction=direction_desc,
                distance=distance_desc,
                camera_cardinal=cam_cardinal,
                obj_ref=_the(ref["label"]),
            )
            options, answer = generate_options(new_dir, ALL_DIRECTIONS_ALLOCENTRIC)
            obj_questions.append({
                "level": "L2",
                "type": "object_move_allocentric",
                "reference_frame": "allocentric",
                "question": question_text,
                "options": options,
                "answer": answer,
                "correct_value": new_dir,
                "camera_cardinal": cam_cardinal,
                "moved_obj_id": obj["id"],
                "moved_obj_label": obj["label"],
                "obj_ref_id": ref["id"],
                "obj_ref_label": ref["label"],
                "mentioned_objects": [
                    _mention("moved_object", obj["label"], obj["id"]),
                    _mention("reference_object", ref["label"], ref["id"]),
                ],
                "delta": delta.tolist(),
                "relation_unchanged": False,
            })

        if len(obj_questions) > max_per_object:
            obj_questions = random.sample(obj_questions, max_per_object)
        questions.extend(obj_questions)

    return questions


# ---------------------------------------------------------------------------
#  L3 generators
# ---------------------------------------------------------------------------

def generate_l3_support_chain(
    objects: list[dict],
    support_graph: dict[int, list[int]],
    supported_by: dict[int, int],
    camera_pose: CameraPose,
    templates: dict,
) -> list[dict]:
    """Generate L3.1 support-chain membership questions (two-hop: A→B→C).

    Tests whether the model can identify ALL objects displaced when A moves.
    Requires 2-hop inference: A supports B (1-hop) AND B supports C (2-hop),
    so both B and C are displaced — not just the direct child B.

    The question does NOT state which objects are on which; the model must
    infer the full chain from the image.

    Options:
      - "{B}"               (1-hop child only  — wrong, misses C)
      - "{C}"               (2-hop grandchild only — wrong, misses B)
      - "Both {B} and {C}" (correct — full chain)
      - "{D}"               (non-chain neighbour — wrong)
    """
    questions: list[dict] = []
    obj_map = {o["id"]: o for o in objects}
    tpl_list = templates.get("L3_support_chain", _default_templates()["L3_support_chain"])

    for grandparent_id, parent_ids in support_graph.items():
        grandparent_id = int(grandparent_id)
        grandparent = obj_map.get(grandparent_id)
        if grandparent is None:
            continue

        for parent_id in parent_ids:
            parent_id = int(parent_id)
            # Second hop: does parent itself support anything?
            grandchild_ids = (
                support_graph.get(parent_id)
                or support_graph.get(str(parent_id))
                or []
            )
            if not grandchild_ids:
                continue  # no depth-2 chain here

            parent = obj_map.get(parent_id)
            if parent is None:
                continue

            # All objects in this grandparent's chain (not eligible as neighbour D)
            this_chain: set[int] = (
                set(get_support_chain_ids(grandparent_id, support_graph))
                | {grandparent_id}
            )
            non_chain = [o for o in objects if o["id"] not in this_chain]
            if not non_chain:
                continue

            # Pick closest non-chain object as the distractor (option D)
            gp_center = np.array(grandparent["center"])
            neighbor = min(
                non_chain,
                key=lambda o: np.linalg.norm(np.array(o["center"]) - gp_center),
            )

            for grandchild_id in grandchild_ids:
                grandchild_id = int(grandchild_id)
                grandchild = obj_map.get(grandchild_id)
                if grandchild is None:
                    continue

                # Skip when labels collide — options would be ambiguous
                if len({parent["label"], grandchild["label"], neighbor["label"]}) < 3:
                    continue

                tpl = random.choice(tpl_list)
                question_text = tpl.format(obj_a=_the(grandparent["label"]))

                opt_parent    = _the(parent["label"])
                opt_grandchild = _the(grandchild["label"])
                opt_both      = f"Both {_the(parent['label'])} and {_the(grandchild['label'])}"
                opt_neighbor  = _the(neighbor["label"])

                options = [opt_parent, opt_grandchild, opt_both, opt_neighbor]
                random.shuffle(options)
                answer_letter = chr(65 + options.index(opt_both))

                questions.append({
                    "level": "L3",
                    "type": "support_chain",
                    "question": question_text,
                    "options": options,
                    "answer": answer_letter,
                    "correct_value": opt_both,
                    "chain_depth": 2,
                    "grandparent_id": grandparent_id,
                    "grandparent_label": grandparent["label"],
                    "parent_id": parent_id,
                    "parent_label": parent["label"],
                    "grandchild_id": grandchild_id,
                    "grandchild_label": grandchild["label"],
                    "neighbor_id": neighbor["id"],
                    "neighbor_label": neighbor["label"],
                    "mentioned_objects": [
                        _mention("grandparent", grandparent["label"], grandparent_id),
                        _mention("parent", parent["label"], parent_id),
                        _mention("grandchild", grandchild["label"], grandchild_id),
                        _mention("neighbor", neighbor["label"], neighbor["id"]),
                    ],
                    "relation_unchanged": False,
                })

    return questions


def generate_l3_coordinate_rotation(
    objects: list[dict],
    camera_pose: CameraPose,
    templates: dict,
    max_per_angle: int = 5,
) -> list[dict]:
    """Generate L3.2 coordinate-rotation counterfactual questions.

    Objects are rotated around the room centre; the camera pose is held fixed.
    The question asks for the *new direction* of A relative to B as seen from
    the unchanged camera — a 4-option direction question, not Yes/No.

    Using the actual direction as the answer prevents the trivial shortcut of
    always answering "No" (which would be correct for most 90°/180° cases).

    max_per_angle caps questions per rotation angle to avoid flooding when
    the scene has many objects (O(n²) pairs).
    """
    questions: list[dict] = []
    tpl_list = templates.get("L3_coordinate_rotation", _default_templates()["L3_coordinate_rotation"])

    # Direction changes don't need depth/occlusion; skip it for speed.
    original_relations = compute_all_relations(objects, camera_pose, None, None)

    for angle in (90, 180, 270):
        # rotation_matrix_z uses math convention (positive = counterclockwise).
        # Templates say "clockwise", so negate the angle for the actual rotation.
        rotated = apply_coordinate_rotation(objects, float(-angle))
        # camera_pose intentionally unchanged — objects rotate, camera does not
        new_relations = compute_all_relations(rotated, camera_pose, None, None)
        changed = find_changed_relations(original_relations, new_relations)

        # Collect only direction-changed pairs, then sample to cap
        changed_dir = [ch for ch in changed if "direction_b_rel_a" in ch["changes"]]
        if len(changed_dir) > max_per_angle:
            changed_dir = random.sample(changed_dir, max_per_angle)

        for ch in changed_dir:
            vals = ch["changes"]["direction_b_rel_a"]
            obj_a_label = next((o["label"] for o in objects if o["id"] == ch["obj_a_id"]), "object")
            obj_b_label = next((o["label"] for o in objects if o["id"] == ch["obj_b_id"]), "object")

            tpl = random.choice(tpl_list)
            question_text = tpl.format(
                angle=angle,
                obj_a=_the(obj_a_label),
                obj_b=_the(obj_b_label),
            )
            # Correct answer is the new direction after rotation (4-option, not Yes/No)
            new_dir = vals["new"]
            options, answer_letter = generate_options(new_dir, ALL_DIRECTIONS)

            questions.append({
                "level": "L3",
                "type": "coordinate_rotation",
                "question": question_text,
                "options": options,
                "answer": answer_letter,
                "correct_value": new_dir,
                "rotation_angle": angle,
                "obj_a_id": ch["obj_a_id"],
                "obj_a_label": obj_a_label,
                "obj_b_id": ch["obj_b_id"],
                "obj_b_label": obj_b_label,
                "mentioned_objects": [
                    _mention("obj_a", obj_a_label, ch["obj_a_id"]),
                    _mention("obj_b", obj_b_label, ch["obj_b_id"]),
                ],
                "old_direction": vals["old"],
                "new_direction": new_dir,
                "relation_unchanged": False,
            })

    return questions


def generate_l3_coordinate_rotation_object_centric(
    objects: list[dict],
    camera_pose: CameraPose,
    templates: dict,
    max_per_angle: int = 5,
) -> list[dict]:
    """L3 coordinate-rotation questions in object-centric frame.

    After rotating all objects, asks: standing at obj_ref's NEW position and
    facing obj_face's NEW position, where is obj_target?
    """
    questions: list[dict] = []
    tpl_list = templates.get(
        "L3_coordinate_rotation_object_centric",
        _default_templates()["L3_coordinate_rotation_object_centric"],
    )

    for angle in (90, 180, 270):
        rotated = apply_coordinate_rotation(objects, float(-angle))
        rot_map = {o["id"]: o for o in rotated}

        candidates: list[dict] = []
        n = len(objects)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                for k in range(n):
                    if k == i or k == j:
                        continue
                    ref = objects[i]
                    face = objects[j]
                    target = objects[k]
                    if len({ref["label"], face["label"], target["label"]}) < 3:
                        continue

                    # Use the rotated positions for the reference frame
                    ref_rot = rot_map.get(ref["id"])
                    face_rot = rot_map.get(face["id"])
                    target_rot = rot_map.get(target["id"])
                    if ref_rot is None or face_rot is None or target_rot is None:
                        continue

                    new_dir, amb = primary_direction_object_centric(
                        np.array(ref_rot["center"]),
                        np.array(face_rot["center"]),
                        np.array(target_rot["center"]),
                    )
                    if amb > 0.7:
                        continue

                    # Check it differs from original
                    old_dir, _ = primary_direction_object_centric(
                        np.array(ref["center"]),
                        np.array(face["center"]),
                        np.array(target["center"]),
                    )
                    if old_dir == new_dir:
                        continue

                    tpl = random.choice(tpl_list)
                    question_text = tpl.format(
                        angle=angle,
                        obj_ref=_the(ref["label"]),
                        obj_face=_the(face["label"]),
                        obj_target=_the(target["label"]),
                    )
                    options, answer = generate_options(new_dir, ALL_DIRECTIONS)
                    candidates.append({
                        "level": "L3",
                        "type": "coordinate_rotation_object_centric",
                        "reference_frame": "object_centric",
                        "question": question_text,
                        "options": options,
                        "answer": answer,
                        "correct_value": new_dir,
                        "rotation_angle": angle,
                        "obj_ref_id": ref["id"],
                        "obj_ref_label": ref["label"],
                        "obj_face_id": face["id"],
                        "obj_face_label": face["label"],
                        "obj_target_id": target["id"],
                        "obj_target_label": target["label"],
                        "mentioned_objects": [
                            _mention("reference_origin", ref["label"], ref["id"]),
                            _mention("reference_facing", face["label"], face["id"]),
                            _mention("target", target["label"], target["id"]),
                        ],
                        "old_direction": old_dir,
                        "new_direction": new_dir,
                        "relation_unchanged": False,
                    })

        if len(candidates) > max_per_angle:
            candidates = random.sample(candidates, max_per_angle)
        questions.extend(candidates)

    return questions


def generate_l3_coordinate_rotation_allocentric(
    objects: list[dict],
    camera_pose: CameraPose,
    templates: dict,
    max_per_angle: int = 5,
) -> list[dict]:
    """L3 coordinate-rotation questions in allocentric (cardinal) frame.

    After rotating all objects, asks: on the floor plan, what cardinal
    direction is obj_a from obj_b?  (Camera + room axes stay fixed, so the
    objects' cardinal positions DO change.)
    """
    questions: list[dict] = []
    cam_cardinal = camera_cardinal_direction(camera_pose)
    tpl_list = templates.get(
        "L3_coordinate_rotation_allocentric",
        _default_templates()["L3_coordinate_rotation_allocentric"],
    )

    for angle in (90, 180, 270):
        rotated = apply_coordinate_rotation(objects, float(-angle))
        rot_map = {o["id"]: o for o in rotated}

        candidates: list[dict] = []
        n = len(objects)
        for i in range(n):
            for j in range(i + 1, n):
                a, b = objects[i], objects[j]
                if a["label"] == b["label"]:
                    continue

                a_rot = rot_map.get(a["id"])
                b_rot = rot_map.get(b["id"])
                if a_rot is None or b_rot is None:
                    continue

                new_dir, amb = primary_direction_allocentric(
                    np.array(a_rot["center"]), np.array(b_rot["center"]),
                )
                if amb > 0.7:
                    continue

                old_dir, _ = primary_direction_allocentric(
                    np.array(a["center"]), np.array(b["center"]),
                )
                if old_dir == new_dir:
                    continue

                tpl = random.choice(tpl_list)
                question_text = tpl.format(
                    angle=angle,
                    camera_cardinal=cam_cardinal,
                    obj_a=_the(a["label"]),
                    obj_b=_the(b["label"]),
                )
                options, answer = generate_options(new_dir, ALL_DIRECTIONS_ALLOCENTRIC)
                candidates.append({
                    "level": "L3",
                    "type": "coordinate_rotation_allocentric",
                    "reference_frame": "allocentric",
                    "question": question_text,
                    "options": options,
                    "answer": answer,
                    "correct_value": new_dir,
                    "camera_cardinal": cam_cardinal,
                    "rotation_angle": angle,
                    "obj_a_id": a["id"],
                    "obj_a_label": a["label"],
                    "obj_b_id": b["id"],
                    "obj_b_label": b["label"],
                    "mentioned_objects": [
                        _mention("obj_a", a["label"], a["id"]),
                        _mention("obj_b", b["label"], b["id"]),
                    ],
                    "old_direction": old_dir,
                    "new_direction": new_dir,
                    "relation_unchanged": False,
                })

        if len(candidates) > max_per_angle:
            candidates = random.sample(candidates, max_per_angle)
        questions.extend(candidates)

    return questions


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def get_support_chain_ids(obj_id: int, support_graph: dict) -> list[int]:
    """Get all dependent IDs (wrapper for import convenience)."""
    from .support_graph import get_support_chain
    return get_support_chain(obj_id, support_graph)


def _ensure_question_mentions(
    question: dict[str, Any],
    id_to_object: dict[int, dict],
    label_to_object: dict[str, dict],
) -> dict[str, Any]:
    """Backfill `mentioned_objects` for generators that do not set it explicitly."""
    if question.get("mentioned_objects"):
        return question

    mentions: list[dict[str, Any]] = []
    role_specs = [
        ("obj_a_id", "obj_a_label", "obj_a"),
        ("obj_b_id", "obj_b_label", "obj_b"),
        ("obj_ref_id", "obj_ref_label", "obj_ref"),
        ("obj_face_id", "obj_face_label", "obj_face"),
        ("obj_target_id", "obj_target_label", "obj_target"),
        ("moved_obj_id", "moved_obj_label", "moved_obj"),
        ("removed_obj_id", "removed_obj_label", "removed_obj"),
        ("grandparent_id", "grandparent_label", "grandparent"),
        ("parent_id", "parent_label", "parent"),
        ("grandchild_id", "grandchild_label", "grandchild"),
        ("neighbor_id", "neighbor_label", "neighbor"),
    ]
    seen: set[int] = set()
    for id_key, label_key, role in role_specs:
        obj_id = question.get(id_key)
        label = question.get(label_key)
        obj = None
        if obj_id is not None:
            obj = id_to_object.get(int(obj_id))
        elif label:
            obj = label_to_object.get(str(label))
        if obj is None:
            continue
        real_id = int(obj["id"])
        if real_id in seen:
            continue
        mentions.append(_mention(role, obj["label"], real_id))
        seen.add(real_id)

    question["mentioned_objects"] = mentions
    return question


def _enforce_strict_visibility(
    questions: list[dict[str, Any]],
    object_visibility: dict[int, dict[str, Any]] | None,
    id_to_object: dict[int, dict],
    label_to_object: dict[str, dict],
) -> list[dict[str, Any]]:
    """Drop questions whose mentioned objects are not all strict-eligible."""
    if not object_visibility:
        return questions

    kept: list[dict[str, Any]] = []
    removed = 0
    for question in questions:
        question = _ensure_question_mentions(question, id_to_object, label_to_object)
        mentions = question.get("mentioned_objects", [])
        if not mentions:
            removed += 1
            continue

        ok = True
        rejected_meta: list[dict[str, Any]] = []
        for mention in mentions:
            obj_id = mention.get("obj_id")
            meta = object_visibility.get(int(obj_id)) if obj_id is not None else None
            if meta is None or not meta.get("eligible_as_reference", False):
                ok = False
                rejected_meta.append({
                    "obj_id": obj_id,
                    "label": mention.get("label"),
                    "rejection_reasons": [] if meta is None else meta.get("rejection_reasons", []),
                })
        if ok:
            kept.append(question)
        else:
            removed += 1
            question["strict_visibility_rejections"] = rejected_meta

    if removed:
        logger.info("Strict visibility filter removed %d questions", removed)
    return kept


def _delta_to_description(delta: np.ndarray, camera_pose: CameraPose | None = None) -> str:
    """Convert a 3D world-frame delta to a camera-relative direction string.

    Projects the world-frame displacement into the camera coordinate system
    (x=right, y=down, z=forward in OpenCV convention) so that "right" in the
    question text matches what the viewer actually sees in the image.

    Used for ego-centric questions where the reference frame is the camera.
    Falls back to world-frame labels if *camera_pose* is not provided.
    """
    if camera_pose is not None:
        from .utils.coordinate_transform import world_to_camera
        # Transform delta as a direction (translate origin to camera, then difference)
        origin_cam = world_to_camera(np.zeros(3), camera_pose)
        delta_cam = world_to_camera(delta, camera_pose) - origin_cam
        dx, dy, dz = float(delta_cam[0]), float(delta_cam[1]), float(delta_cam[2])
        # OpenCV: x=right, y=down, z=forward
        horiz = abs(dx)
        depth = abs(dz)
        vert = abs(dy)
        dominant = max(horiz, depth, vert)
        if dominant == vert:
            return "down" if dy > 0 else "up"
        elif dominant == depth:
            return "forward" if dz > 0 else "backward"
        else:
            return "right" if dx > 0 else "left"

    # Fallback: world-frame (should not normally be used)
    axis = np.argmax(np.abs(delta))
    sign = delta[axis]
    labels = {0: ("right", "left"), 1: ("forward", "backward"), 2: ("up", "down")}
    pos_label, neg_label = labels[axis]
    return pos_label if sign > 0 else neg_label


def _delta_to_cardinal_description(delta: np.ndarray) -> str:
    """Convert a 3D world-frame delta to an absolute cardinal direction string.

    Convention after ScanNet axis alignment:  +x = east, +y = north, +z = up.

    Used for object-centric and allocentric questions where the movement
    should be described in unambiguous absolute terms.
    """
    dx, dy, dz = float(delta[0]), float(delta[1]), float(delta[2])
    horiz = math.sqrt(dx * dx + dy * dy)
    vert = abs(dz)

    if vert > horiz:
        return "up" if dz > 0 else "down"

    if horiz < 1e-6:
        return "north"

    angle = math.degrees(math.atan2(dx, dy))  # 0=north, 90=east
    if angle < 0:
        angle += 360

    # Bin into 4 cardinal directions (N/E/S/W) using 90-degree bins
    if angle < 45 or angle >= 315:
        return "north"
    elif angle < 135:
        return "east"
    elif angle < 225:
        return "south"
    else:
        return "west"


# ---------------------------------------------------------------------------
#  Main entry point
# ---------------------------------------------------------------------------

def generate_all_questions(
    objects: list[dict],
    support_graph: dict[int, list[int]],
    supported_by: dict[int, int],
    camera_pose: CameraPose,
    depth_image=None,
    depth_intrinsics=None,
    templates: dict | None = None,
    visible_object_ids: list[int] | None = None,
    object_visibility: dict[int, dict[str, Any]] | None = None,
    strict_mode: bool = False,
    room_bounds: dict | None = None,
) -> list[dict]:
    """Generate all question types for a single scene + frame.

    depth_image: float32 depth map in metres (from ScanNet depth PNG), or None.
    depth_intrinsics: CameraIntrinsics for the depth camera, or None.
    visible_object_ids: if provided, restrict all questions to objects whose
    centre projects into this frame.  Questions about off-screen objects are
    unanswerable from the image and should never be included.
    room_bounds: dict with bbox_min/bbox_max from wall/floor mesh, or None.

    Returns a list of question dicts.
    """
    if templates is None:
        templates = _load_templates()

    # Restrict to objects visible in this frame so every question can be
    # answered by looking at the image.
    if visible_object_ids is not None:
        vis_set = set(visible_object_ids)
        objects = [o for o in objects if o["id"] in vis_set]
        # Rebuild support graph restricted to visible objects
        support_graph = {
            k: [c for c in v if c in vis_set]
            for k, v in support_graph.items()
            if k in vis_set
        }
        supported_by = {k: v for k, v in supported_by.items() if k in vis_set and v in vis_set}

    # Remove objects with uninformative labels (wall, floor, object, etc.)
    objects = [o for o in objects if o.get("label", "").lower() not in EXCLUDED_LABELS]

    # Safety net: per-frame uniqueness check.  The primary unique-label
    # filtering now happens upstream in scene_parser.parse_scene(), but
    # visibility filtering above may reveal edge cases — keep this guard.
    from collections import Counter
    label_counts = Counter(o["label"] for o in objects)
    unique_label_ids = {o["id"] for o in objects if label_counts[o["label"]] == 1}
    objects_uniq = [o for o in objects if o["id"] in unique_label_ids]

    all_questions: list[dict] = []

    # Per-frame caps — keep the benchmark tractable when scenes have many objects
    MAX_L1_DIRECTION = 20
    MAX_L1_DIRECTION_OC = 15   # object-centric
    MAX_L1_DIRECTION_ALLO = 15 # allocentric
    MAX_L1_DISTANCE = 20

    # Compute baseline relations using only uniquely-labelled visible objects
    relations = compute_all_relations(objects_uniq, camera_pose, depth_image, depth_intrinsics)

    # Pre-compute per-object occlusion cache for L1 occlusion questions.
    # Only use visible objects (those whose centre projects into this frame)
    # so that every occlusion question references an object the viewer can
    # at least locate in the image.
    from .relation_engine import compute_occlusion_per_object
    occ_cache = compute_occlusion_per_object(
        objects_uniq, camera_pose, depth_image, depth_intrinsics
    )

    # L1 — collect separately so we can sample before adding
    MAX_L1_OCCLUSION = 15
    l1_dir_qs:  list[dict] = []
    l1_dist_qs: list[dict] = []
    l1_occ_qs:  list[dict] = []

    for rel in relations:
        q = generate_l1_direction(rel, templates)
        if q:
            l1_dir_qs.append(q)
        q = generate_l1_distance(rel, templates)
        if q:
            l1_dist_qs.append(q)

    # L1 occlusion: per-object (not pairwise).
    # Only uses visible objects — the viewer must be able to locate the object.
    seen_occ_objs: set[int] = set()
    for obj in objects_uniq:
        status, _ratio = occ_cache.get(obj["id"], ("unknown", 0.0))
        if obj["id"] in seen_occ_objs:
            continue
        q = generate_l1_occlusion(obj, status, templates)
        if q:
            l1_occ_qs.append(q)
            seen_occ_objs.add(obj["id"])

    # L1 new reference frames
    l1_dir_oc_qs = generate_l1_direction_object_centric(
        objects_uniq, templates, max_questions=MAX_L1_DIRECTION_OC,
    )
    l1_dir_allo_qs = generate_l1_direction_allocentric(
        objects_uniq, camera_pose, templates, max_questions=MAX_L1_DIRECTION_ALLO,
    )

    if len(l1_dir_qs) > MAX_L1_DIRECTION:
        l1_dir_qs = random.sample(l1_dir_qs, MAX_L1_DIRECTION)
    if len(l1_dist_qs) > MAX_L1_DISTANCE:
        l1_dist_qs = random.sample(l1_dist_qs, MAX_L1_DISTANCE)
    if len(l1_occ_qs) > MAX_L1_OCCLUSION:
        l1_occ_qs = random.sample(l1_occ_qs, MAX_L1_OCCLUSION)
    all_questions.extend(l1_dir_qs)
    all_questions.extend(l1_dist_qs)
    all_questions.extend(l1_occ_qs)
    all_questions.extend(l1_dir_oc_qs)
    all_questions.extend(l1_dir_allo_qs)

    # Rebuild support graph restricted to uniquely-labelled objects
    support_graph_uniq = {
        k: [c for c in v if c in unique_label_ids]
        for k, v in support_graph.items()
        if k in unique_label_ids
    }
    supported_by_uniq = {k: v for k, v in supported_by.items()
                         if k in unique_label_ids and v in unique_label_ids}

    # L2 — ego-centric (existing)
    all_questions.extend(
        generate_l2_object_move(objects_uniq, support_graph_uniq, camera_pose, templates,
                                room_bounds=room_bounds)
    )
    all_questions.extend(
        generate_l2_viewpoint_move(objects_uniq, camera_pose, templates)
    )
    all_questions.extend(
        generate_l2_object_remove(objects_uniq, support_graph_uniq, camera_pose, templates)
    )
    # L2 — new reference frames
    all_questions.extend(
        generate_l2_object_move_object_centric(
            objects_uniq, support_graph_uniq, camera_pose, templates,
            room_bounds=room_bounds,
        )
    )
    all_questions.extend(
        generate_l2_object_move_allocentric(
            objects_uniq, support_graph_uniq, camera_pose, templates,
            room_bounds=room_bounds,
        )
    )

    # L3
    all_questions.extend(
        generate_l3_support_chain(objects_uniq, support_graph_uniq, supported_by_uniq, camera_pose, templates)
    )
    # L3 coordinate rotation — all three reference frames
    all_questions.extend(
        generate_l3_coordinate_rotation(objects_uniq, camera_pose, templates)
    )
    all_questions.extend(
        generate_l3_coordinate_rotation_object_centric(objects_uniq, camera_pose, templates)
    )
    all_questions.extend(
        generate_l3_coordinate_rotation_allocentric(objects_uniq, camera_pose, templates)
    )

    id_to_object = {int(o["id"]): o for o in objects_uniq}
    label_to_object = {str(o["label"]): o for o in objects_uniq}
    for idx, question in enumerate(all_questions):
        all_questions[idx] = _ensure_question_mentions(
            question, id_to_object, label_to_object,
        )

    if strict_mode:
        all_questions = _enforce_strict_visibility(
            all_questions, object_visibility, id_to_object, label_to_object,
        )

    logger.info("Generated %d questions total", len(all_questions))
    return all_questions
