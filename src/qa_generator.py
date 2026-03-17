"""Stage 6: QA generation.

Generates multiple-choice questions from computed spatial relations and
virtual operation results.
"""

from __future__ import annotations

import json
import random
import logging
from pathlib import Path
from typing import Any

import numpy as np

from .relation_engine import (
    ALL_DIRECTIONS_10,
    compute_all_relations,
    find_changed_relations,
    primary_direction,
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
ALL_DISTANCES = ["touching (<0.5m)", "very close (0.5-1.5m)", "close (1.5-3m)", "far (>3m)"]
ALL_OCCLUSION = ["fully visible", "partially occluded", "fully occluded", "not in frame"]
YES_NO = ["Yes", "No"]


def _the(label: str) -> str:
    """Prepend 'the' to an object label for natural English grammar.

    'shoes' → 'the shoes', 'table' → 'the table'.
    Avoids 'where is shoes positioned' type errors.
    """
    return f"the {label}"

# Labels to exclude from ALL question types (not just L2).
# Structural elements, generic labels, and uninformative categories.
EXCLUDED_LABELS = {
    # Structural / architectural
    "floor", "wall", "ceiling", "room", "ground",
    "door", "window", "stairs", "pillar", "column",
    # Generic / uninformative
    "object", "otherfurniture", "otherprop", "otherstructure",
    "unknown", "misc", "stuff",
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
        "L1_direction": [
            "From the image, {obj_a} is in which direction relative to {obj_b}?",
            "Looking at the scene, where is {obj_a} positioned relative to {obj_b}?",
        ],
        "L1_occlusion": [
            "What is the visibility status of {obj_a} from the current viewpoint?",
            "Is {obj_a} fully visible from the current camera angle?",
        ],
        "L1_distance": [
            "Approximately how far apart are {obj_a} and {obj_b}?",
            "What is the approximate distance between {obj_a} and {obj_b}?",
        ],
        "L2_object_move": [
            "If {obj_a} is moved {direction} by {distance}, what would be the new spatial relationship between {obj_b} and {obj_c}?",
            "Imagine moving {obj_a} {direction} by {distance}. After this change, in which direction is {obj_b} relative to {obj_c}?",
        ],
        "L2_object_move_distance": [
            "If {obj_a} is moved {direction} by {distance}, how far apart would {obj_b} and {obj_c} be?",
            "Imagine moving {obj_a} {direction} by {distance}. After this change, what is the distance between {obj_b} and {obj_c}?",
        ],
        "L2_viewpoint_move": [
            "If the observer moves {direction} by {distance} from the current position, would {obj_a} become visible or occluded?",
        ],
        "L2_object_remove": [
            "If {obj_a} were removed from the scene, what would be the visibility status of {obj_b} from the current viewpoint?",
        ],
        "L3_support_chain": [
            "Suppose {obj_a} were moved to a different location. Which of the following objects would also be displaced from their current positions?",
            "If {obj_a} were relocated elsewhere in the room, which of the following objects would also change position?",
            "Imagine {obj_a} is moved to a new spot. Which of the following objects would also be displaced as a result?",
        ],
        "L3_coordinate_rotation": [
            "Suppose this room had originally been designed with its orientation rotated {angle} degrees clockwise (viewed from above), with all objects keeping their relative positions. Observed from the original camera position and viewing direction (unchanged), in which direction is {obj_a} relative to {obj_b}?",
        ],
    }


def generate_options(
    correct_answer: str,
    answer_pool: list[str],
    n_options: int = 4,
) -> tuple[list[str], str]:
    """Generate MCQ options from an answer pool.

    Returns (shuffled_options, correct_letter).
    """
    options = [correct_answer]
    distractors = [a for a in answer_pool if a != correct_answer]
    random.shuffle(distractors)
    options.extend(distractors[: n_options - 1])

    # Pad if pool is too small
    while len(options) < n_options:
        options.append(f"None of the above")

    random.shuffle(options)
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
    options, answer = generate_options(correct, ALL_OCCLUSION)

    return {
        "level": "L1",
        "type": "occlusion",
        "question": question_text,
        "options": options,
        "answer": answer,
        "correct_value": correct,
        "obj_a_id": obj["id"],
        "obj_a_label": obj["label"],
        "ambiguity_score": 0.0,
        "relation_unchanged": False,
    }


# ---------------------------------------------------------------------------
#  L2 generators
# ---------------------------------------------------------------------------

def generate_l2_object_move(
    objects: list[dict],
    support_graph: dict[int, list[int]],
    camera_pose: CameraPose,
    templates: dict,
    max_per_object: int = 3,
) -> list[dict]:
    """Generate L2.1 object-movement questions for a scene."""
    questions: list[dict] = []
    obj_map = {o["id"]: o for o in objects}

    for obj in objects:
        # Skip structural room elements — they cannot be "moved" in any
        # meaningful physical sense and confuse human annotators.
        if obj.get("label", "").lower() in EXCLUDED_LABELS:
            continue

        # L2.1 only makes sense when the moved object actually carries dependent
        # objects with it — otherwise there is no support-chain propagation and
        # the question degenerates into a simple single-object translation.
        children = support_graph.get(obj["id"]) or support_graph.get(str(obj["id"])) or []
        if not children:
            continue

        delta, changed = find_meaningful_movement(
            objects, support_graph, obj["id"], camera_pose,
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
        direction_desc = _delta_to_description(delta)
        distance_desc = f"{np.linalg.norm(delta):.1f}m"

        tpl_list = templates.get("L2_object_move", _default_templates()["L2_object_move"])

        obj_questions: list[dict] = []
        for ch in changed:
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

                obj_b_label = obj_map.get(ch["obj_a_id"], {}).get("label", "object")
                obj_c_label = obj_map.get(ch["obj_b_id"], {}).get("label", "object")

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
                    "obj_b_label": obj_b_label,
                    "obj_c_label": obj_c_label,
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
                    "relation_unchanged": False,
                })

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
                    "parent_id": parent_id,
                    "grandchild_id": grandchild_id,
                    "neighbor_id": neighbor["id"],
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
                "obj_a_label": obj_a_label,
                "obj_b_label": obj_b_label,
                "old_direction": vals["old"],
                "new_direction": new_dir,
                "relation_unchanged": False,
            })

    return questions


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def get_support_chain_ids(obj_id: int, support_graph: dict) -> list[int]:
    """Get all dependent IDs (wrapper for import convenience)."""
    from .support_graph import get_support_chain
    return get_support_chain(obj_id, support_graph)


def _delta_to_description(delta: np.ndarray) -> str:
    """Convert a 3D delta vector to a human-readable direction string."""
    axis = np.argmax(np.abs(delta))
    sign = delta[axis]
    labels = {0: ("right", "left"), 1: ("forward", "backward"), 2: ("up", "down")}
    pos_label, neg_label = labels[axis]
    return pos_label if sign > 0 else neg_label


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
) -> list[dict]:
    """Generate all question types for a single scene + frame.

    depth_image: float32 depth map in metres (from ScanNet depth PNG), or None.
    depth_intrinsics: CameraIntrinsics for the depth camera, or None.
    visible_object_ids: if provided, restrict all questions to objects whose
    centre projects into this frame.  Questions about off-screen objects are
    unanswerable from the image and should never be included.

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
    MAX_L1_DISTANCE = 20

    # Compute baseline relations using only uniquely-labelled visible objects
    relations = compute_all_relations(objects_uniq, camera_pose, depth_image, depth_intrinsics)

    # Pre-compute per-object occlusion cache for L1 occlusion questions
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

    # L1 occlusion: per-object (not pairwise)
    seen_occ_objs: set[int] = set()
    for obj in objects_uniq:
        status, _ratio = occ_cache.get(obj["id"], ("unknown", 0.0))
        if obj["id"] in seen_occ_objs:
            continue
        q = generate_l1_occlusion(obj, status, templates)
        if q:
            l1_occ_qs.append(q)
            seen_occ_objs.add(obj["id"])

    if len(l1_dir_qs) > MAX_L1_DIRECTION:
        l1_dir_qs = random.sample(l1_dir_qs, MAX_L1_DIRECTION)
    if len(l1_dist_qs) > MAX_L1_DISTANCE:
        l1_dist_qs = random.sample(l1_dist_qs, MAX_L1_DISTANCE)
    if len(l1_occ_qs) > MAX_L1_OCCLUSION:
        l1_occ_qs = random.sample(l1_occ_qs, MAX_L1_OCCLUSION)
    all_questions.extend(l1_dir_qs)
    all_questions.extend(l1_dist_qs)
    all_questions.extend(l1_occ_qs)

    # Rebuild support graph restricted to uniquely-labelled objects
    support_graph_uniq = {
        k: [c for c in v if c in unique_label_ids]
        for k, v in support_graph.items()
        if k in unique_label_ids
    }
    supported_by_uniq = {k: v for k, v in supported_by.items()
                         if k in unique_label_ids and v in unique_label_ids}

    # L2
    all_questions.extend(
        generate_l2_object_move(objects_uniq, support_graph_uniq, camera_pose, templates)
    )
    all_questions.extend(
        generate_l2_viewpoint_move(objects_uniq, camera_pose, templates)
    )
    all_questions.extend(
        generate_l2_object_remove(objects_uniq, support_graph_uniq, camera_pose, templates)
    )

    # L3
    all_questions.extend(
        generate_l3_support_chain(objects_uniq, support_graph_uniq, supported_by_uniq, camera_pose, templates)
    )
    all_questions.extend(
        generate_l3_coordinate_rotation(objects_uniq, camera_pose, templates)
    )

    logger.info("Generated %d questions total", len(all_questions))
    return all_questions
