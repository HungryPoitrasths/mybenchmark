"""Stage 6: QA generation.

Generates multiple-choice questions from computed spatial relations and
virtual operation results.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import random
import logging
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .relation_engine import (
    ALL_DIRECTIONS_10,
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
    find_meaningful_orbit_rotation,
)
from .utils.colmap_loader import CameraPose
from .utils.colmap_loader import CameraIntrinsics
from .utils.coordinate_transform import is_in_image, project_to_image
from .utils.depth_occlusion import (
    MIN_IN_FRAME_RATIO,
    MIN_PROJECTED_AREA_PX,
    FULLY_VISIBLE_RATIO_MIN,
    PARTIALLY_VISIBLE_RATIO_MIN,
    compute_mesh_depth_occlusion,
    compute_mesh_depth_occlusion_metrics,
)
from .scene_parser import EXCLUDED_LABELS, InstanceMeshData

logger = logging.getLogger(__name__)

# Default template file; can be overridden
_TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "templates"

ALL_DIRECTIONS = ALL_DIRECTIONS_10
ALL_DIRECTIONS_ALLOCENTRIC = list(CARDINAL_DIRECTIONS_8)
ALL_DISTANCES = ["touching (<0.5m)", "very close (0.5-1.5m)", "close (1.5-3m)", "far (>3m)"]
ALL_OCCLUSION = ["fully visible", "partially occluded", "not visible"]
L1_OCCLUSION_STATES = ["not occluded", "occluded", "not visible"]
YES_NO = ["Yes", "No"]

# Object-centric questions need a stable horizontal facing direction.
MIN_OBJECT_CENTRIC_FACING_HORIZONTAL_DISTANCE = 0.3
MIN_OBJECT_CENTRIC_FACING_HORIZONTAL_RATIO = 0.5
MIN_WALL_HEIGHT = 1.5
MIN_WALL_MAJOR_AXIS = 1.5
MAX_WALL_MINOR_AXIS = 1.0
MIN_WALL_AXIS_RATIO = 2.0
L1_OCCLUSION_MIN_IN_FRAME_RATIO = 0.05
L1_OCCLUSION_NOT_OCCLUDED_MAX = 0.05
L1_OCCLUSION_OCCLUDED_MIN = 0.20
L1_OCCLUSION_GRAYZONE_FALLBACK = 0.10
L1_OCCLUSION_SAMPLE_COUNT = 512
L1_OCCLUSION_MIN_EFFECTIVE_COUNT = 64
L1_OCCLUSION_MIN_EFFECTIVE_RATIO = 0.25
L1_OCCLUSION_MASK_PAD_RATIO = 0.20
L1_OCCLUSION_MASK_MIN_CROP_SIZE = 128
L1_OCCLUSION_MASK_ALPHA = 0.35
L1_OCCLUSION_MASK_FILL_BGR = (40, 40, 255)
L1_OCCLUSION_MASK_BORDER_BGR = (0, 255, 255)


@dataclass(frozen=True)
class _ModifiedSceneContext:
    ray_caster: Any
    ignored_tri_ids: frozenset[int]


@dataclass(frozen=True)
class _L1OcclusionMetrics:
    projected_area: float
    in_frame_ratio: float
    occlusion_ratio_in_frame: float
    valid_in_frame_count: int
    sampled_point_count: int
    in_frame_sample_count: int
    effective_ratio: float
    sufficient_evidence: bool
    decision: str
    backend: str


def _instance_triangle_id_set(
    instance_mesh_data: InstanceMeshData | None,
    obj_id: int,
) -> set[int]:
    if instance_mesh_data is None:
        return set()

    tri_parts = [
        arr for arr in (
            instance_mesh_data.triangle_ids_by_instance.get(int(obj_id)),
            instance_mesh_data.boundary_triangle_ids_by_instance.get(int(obj_id)),
        )
        if arr is not None and len(arr) > 0
    ]
    if not tri_parts:
        return set()
    tri_ids = np.unique(np.concatenate(tri_parts).astype(np.int64))
    return {int(tid) for tid in tri_ids.tolist()}


def _instance_surface_samples(
    instance_mesh_data: InstanceMeshData | None,
    obj_id: int,
) -> np.ndarray:
    if instance_mesh_data is None:
        return np.empty((0, 3), dtype=np.float64)
    samples = instance_mesh_data.surface_points_by_instance.get(int(obj_id))
    if samples is None:
        return np.empty((0, 3), dtype=np.float64)
    return np.asarray(samples, dtype=np.float64)


def _make_l1_occlusion_metrics(
    projected_area: float,
    in_frame_ratio: float,
    occlusion_ratio_in_frame: float,
    valid_in_frame_count: int,
    sampled_point_count: int,
    in_frame_sample_count: int,
    backend: str,
) -> _L1OcclusionMetrics:
    effective_ratio = (
        float(valid_in_frame_count / in_frame_sample_count)
        if in_frame_sample_count > 0 else 0.0
    )
    sufficient_evidence = (
        valid_in_frame_count >= L1_OCCLUSION_MIN_EFFECTIVE_COUNT
        and effective_ratio >= L1_OCCLUSION_MIN_EFFECTIVE_RATIO
    )
    metrics = _L1OcclusionMetrics(
        projected_area=float(projected_area),
        in_frame_ratio=float(in_frame_ratio),
        occlusion_ratio_in_frame=float(occlusion_ratio_in_frame),
        valid_in_frame_count=int(valid_in_frame_count),
        sampled_point_count=int(sampled_point_count),
        in_frame_sample_count=int(in_frame_sample_count),
        effective_ratio=effective_ratio,
        sufficient_evidence=bool(sufficient_evidence),
        decision="skip",
        backend=str(backend),
    )
    return _L1OcclusionMetrics(
        projected_area=metrics.projected_area,
        in_frame_ratio=metrics.in_frame_ratio,
        occlusion_ratio_in_frame=metrics.occlusion_ratio_in_frame,
        valid_in_frame_count=metrics.valid_in_frame_count,
        sampled_point_count=metrics.sampled_point_count,
        in_frame_sample_count=metrics.in_frame_sample_count,
        effective_ratio=metrics.effective_ratio,
        sufficient_evidence=metrics.sufficient_evidence,
        decision=_classify_l1_occlusion_metrics(metrics),
        backend=metrics.backend,
    )


def _depth_backend_inputs_ready(
    *,
    depth_image,
    depth_intrinsics,
    ray_caster,
    context: str,
) -> bool:
    """Validate optional depth-evaluation inputs.

    Returning ``False`` means the caller intentionally has no depth evidence
    available and should skip the depth-based check. Partial configuration is
    treated as an error because it usually means the caller requested the
    depth backend but forgot one of the required resources.
    """
    has_depth_image = depth_image is not None
    has_depth_intrinsics = depth_intrinsics is not None
    if has_depth_image != has_depth_intrinsics:
        missing = "depth_image" if not has_depth_image else "depth_intrinsics"
        raise RuntimeError(
            f"Depth backend for {context} requires both depth_image and depth_intrinsics; missing {missing}.",
        )
    if not has_depth_image:
        return False
    if ray_caster is None:
        raise RuntimeError(
            f"Depth backend for {context} requires ray_caster to compute target-mesh entry depth.",
        )
    return True


def _the(label: str) -> str:
    """Prepend 'the' to an object label for natural English grammar.

    'shoes' → 'the shoes', 'table' → 'the table'.
    Avoids 'where is shoes positioned' type errors.
    """
    return f"the {label}"


def _mention(role: str, label: str, obj_id: int | None = None) -> dict[str, Any]:
    """Create a normalised mentioned-object record for a question."""
    return {"role": role, "obj_id": obj_id, "label": label}


def _has_stable_object_centric_facing(
    anchor_center: np.ndarray,
    facing_center: np.ndarray,
) -> bool:
    """Whether anchor->facing defines a reliable horizontal heading.

    Reject pairs that are almost vertically aligned, because "face toward X"
    then fails to define an unambiguous left/right/front/back frame.
    """
    delta = facing_center - anchor_center
    horizontal_distance = float(np.linalg.norm(delta[:2]))
    total_distance = float(np.linalg.norm(delta))
    if horizontal_distance < MIN_OBJECT_CENTRIC_FACING_HORIZONTAL_DISTANCE:
        return False
    if total_distance < 1e-6:
        return False
    if horizontal_distance / total_distance < MIN_OBJECT_CENTRIC_FACING_HORIZONTAL_RATIO:
        return False
    return True


def _invert_direction(direction: str) -> str:
    """Convert a direction of B relative to A into A relative to B."""
    opposites = {
        "front": "back",
        "front-right": "back-left",
        "right": "left",
        "back-right": "front-left",
        "back": "front",
        "back-left": "front-right",
        "left": "right",
        "front-left": "back-right",
        "above": "below",
        "below": "above",
        "north": "south",
        "northeast": "southwest",
        "east": "west",
        "southeast": "northwest",
        "south": "north",
        "southwest": "northeast",
        "west": "east",
        "northwest": "southeast",
    }
    return opposites.get(direction, direction)


def _direction_with_camera_hint(direction: str) -> str:
    """Clarify camera-depth motions in rendered question text."""
    if direction == "forward":
        return "forward (away from the camera)"
    if direction == "backward":
        return "backward (toward the camera)"
    return direction


def _wall_image_side_phrase(side: str) -> str:
    if side == "left":
        return "on the left side of the image"
    if side == "right":
        return "on the right side of the image"
    if side == "top":
        return "near the top of the image"
    if side == "bottom":
        return "near the bottom of the image"
    return "in the image"


def _build_visible_wall_anchor(
    wall_objects: list[dict] | None,
    camera_pose: CameraPose,
    color_intrinsics: CameraIntrinsics | None,
) -> dict[str, Any] | None:
    """Select one visible, axis-aligned wall to anchor allocentric wording."""
    if not wall_objects or color_intrinsics is None:
        return None

    candidates: list[dict[str, Any]] = []
    width = float(color_intrinsics.width)
    height = float(color_intrinsics.height)
    if width <= 0 or height <= 0:
        return None

    for wall in wall_objects:
        dims = np.array(wall.get("dimensions", [0.0, 0.0, 0.0]), dtype=float)
        if dims.shape[0] < 3:
            continue
        dx, dy, dz = float(dims[0]), float(dims[1]), float(dims[2])
        if dz < MIN_WALL_HEIGHT:
            continue

        major_axis = max(dx, dy)
        minor_axis = min(dx, dy)
        if major_axis < MIN_WALL_MAJOR_AXIS or minor_axis > MAX_WALL_MINOR_AXIS:
            continue
        if minor_axis <= 1e-6:
            axis_ratio = float("inf")
        else:
            axis_ratio = major_axis / minor_axis
        if axis_ratio < MIN_WALL_AXIS_RATIO:
            continue

        if dy >= dx:
            wall_axis = "north_south"
            wall_axis_text = "north-south"
        else:
            wall_axis = "east_west"
            wall_axis_text = "east-west"

        center = np.array(wall["center"], dtype=float)
        uv, depth = project_to_image(center, camera_pose, color_intrinsics)
        if not (0.3 < depth <= 6.0):
            continue
        if not is_in_image(uv, color_intrinsics, margin=80):
            continue

        u = float(uv[0]) / width
        v = float(uv[1]) / height
        dx_img = u - 0.5
        dy_img = v - 0.5
        if abs(dx_img) >= abs(dy_img):
            image_side = "right" if dx_img >= 0 else "left"
            side_separation = abs(dx_img)
        else:
            image_side = "bottom" if dy_img >= 0 else "top"
            side_separation = abs(dy_img)

        side_phrase = _wall_image_side_phrase(image_side)
        note = (
            f"The wall {side_phrase} runs {wall_axis_text} on the floor plan."
        )
        candidates.append({
            "visible_wall_anchor_note": note,
            "anchor_wall_id": wall["id"],
            "anchor_wall_image_side": image_side,
            "anchor_wall_axis": wall_axis,
            "axis_ratio": float(axis_ratio),
            "side_separation": float(side_separation),
            "major_axis": float(major_axis),
        })

    if not candidates:
        return None

    candidates.sort(
        key=lambda c: (
            c["side_separation"],
            c["axis_ratio"],
            c["major_axis"],
        ),
        reverse=True,
    )
    best = candidates[0].copy()
    best.pop("axis_ratio", None)
    best.pop("side_separation", None)
    best.pop("major_axis", None)
    return best

_TEMPLATE_KEY_ALIASES: dict[str, tuple[str, ...]] = {
    "L2_object_move_object_centric": ("L2_object_rotate_object_centric",),
}


def _normalize_template_aliases(templates: dict) -> dict:
    """Backfill template aliases so old and new names both work."""
    normalized = dict(templates)
    for canonical_key, alias_keys in _TEMPLATE_KEY_ALIASES.items():
        source_value = normalized.get(canonical_key)
        if source_value is None:
            for alias_key in alias_keys:
                if alias_key in normalized:
                    source_value = normalized[alias_key]
                    normalized[canonical_key] = source_value
                    break
        if source_value is None:
            continue
        for alias_key in alias_keys:
            normalized.setdefault(alias_key, source_value)
    return normalized


def _load_templates() -> dict:
    """Load question templates from the JSON file."""
    tpl_path = _TEMPLATE_DIR / "question_templates.json"
    if tpl_path.exists():
        with open(tpl_path, "r", encoding="utf-8") as f:
            return _normalize_template_aliases(json.load(f))
    # Fallback inline templates
    return _normalize_template_aliases(_default_templates())


def _default_templates() -> dict:
    return {
        # ==== L1 — Static perception ====

        # --- Ego-centric ---
        "L1_direction_agent": [
            "From the camera's viewpoint, {obj_a} is in which direction relative to {obj_b}?",
            "Looking at the scene from the camera's perspective, where is {obj_a} positioned relative to {obj_b}?",
            "From the current camera perspective, what is the spatial relationship of {obj_a} to {obj_b}?",
        ],
        "L1_distance": [
            "Approximately how far apart are {obj_a} and {obj_b}?",
            "What is the approximate distance between {obj_a} and {obj_b}?",
        ],
        "L1_occlusion": [
            "What is the occlusion status of {obj_a} in the current view?",
            "From the current viewpoint, which best describes {obj_a}: not occluded, occluded, or not visible?",
            "In the current image, is {obj_a} unoccluded, occluded by another object, or not visible?",
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
        "L2_object_move_agent": [
            "From the camera's perspective, imagine moving {obj_a} {direction_with_camera_hint} by {distance}. After this change, what is the relative position of {obj_b} to {obj_c}?",
            "From the camera's perspective, if we move {obj_a} {direction_with_camera_hint} by {distance}, where is {obj_b} relative to {obj_c}?",
        ],
        "L2_object_move_distance": [
            "From the camera's perspective, if {obj_a} is moved {direction_with_camera_hint} by {distance}, how far apart would {obj_b} and {obj_c} be?",
            "From the camera's perspective, imagine moving {obj_a} {direction_with_camera_hint} by {distance}. After this change, what is the distance between {obj_b} and {obj_c}?",
        ],
        "L2_object_move_occlusion": [
            "From the camera's perspective, imagine moving {obj_a} {direction_with_camera_hint} by {distance}. After this change, what is the visibility status of {obj_b} relative to {obj_c}?",
            "From the camera's perspective, if {obj_a} is moved {direction_with_camera_hint} by {distance}, how visible would {obj_b} be with respect to {obj_c}?",
        ],
        "L2_viewpoint_move": [
            "If the camera translates {direction_with_camera_hint} by {distance} while keeping its intrinsics and orientation unchanged, would {obj_a} become visible or occluded?",
            "After the camera moves {direction_with_camera_hint} by {distance} without changing its viewing direction, what is the visibility status of {obj_a}?",
            "If the camera shifts {direction_with_camera_hint} by {distance} with no change in intrinsics or orientation, can you still see {obj_a}?",
        ],
        "L2_object_remove": [
            "If {obj_a} were removed from the scene, what would be the visibility status of {obj_b} from the current viewpoint?",
        ],

        # --- Object-centric ---
        "L2_object_move_object_centric": [
            "Imagine you are {obj_query} and facing toward {obj_face}. If {obj_move_source} were rotated {angle} degrees {rotation_direction} around {obj_face} (viewed from above), from your perspective, in which direction would {obj_ref} be?",
        ],
        "L2_object_rotate_object_centric": [
            "Imagine you are {obj_query} and facing toward {obj_face}. If {obj_move_source} were rotated {angle} degrees {rotation_direction} around {obj_face} (viewed from above), from your perspective, in which direction would {obj_ref} be?",
        ],

        # --- Allocentric ---
        "L2_object_move_allocentric": [
            "If {obj_move_source} is moved {distance} to the {direction}, the camera faces {camera_cardinal}. On the floor plan, in which cardinal direction would {obj_query} be from {obj_ref}?",
            "After moving {obj_move_source} {distance} to the {direction}, on the room's layout (camera facing {camera_cardinal}), in which cardinal direction is {obj_query} from {obj_ref}?",
        ],

        # ==== L3 — Counterfactual / multi-hop ====

        "L3_attachment_chain": [
            "Suppose {obj_a} were moved to a different location. Which of the following objects would also be displaced from their current positions?",
            "If {obj_a} were relocated elsewhere in the room, which of the following objects would also change position?",
            "Imagine {obj_a} is moved to a new spot. Which of the following objects would also be displaced as a result?",
        ],

        # --- Ego-centric (rewritten — 方案B) ---
        "L3_coordinate_rotation_agent": [
            "Suppose this room had originally been designed with its orientation rotated {angle} degrees clockwise (viewed from above), with all objects keeping their relative positions. Observed from the original camera position and viewing direction (unchanged), in which direction is {obj_a} relative to {obj_b}?",
            "If the room layout had been rotated {angle} degrees clockwise (top-down view) from the start, with all relative object positions preserved and camera position and orientation unchanged, from the camera's perspective, where would {obj_a} be relative to {obj_b}?",
            "Imagine the room was originally built rotated {angle} degrees clockwise (as seen from above). With all inter-object relationships intact and the camera at its original pose, from the camera's perspective, what is the direction of {obj_a} from {obj_b}?",
        ],

        # --- Object-centric ---
        "L3_coordinate_rotation_object_centric": [
            "Suppose this room had originally been oriented {angle} degrees clockwise (viewed from above), with all objects keeping their relative positions. If you were {obj_ref} at its rotated position and faced toward {obj_face}'s rotated position, in which direction would {obj_target} be?",
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

    HORIZONTAL.update(CARDINAL_DIRECTIONS_8)
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
            if a != correct_answer
            and a not in options
            and (not exclude or a in exclude)
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
    tpl_list = templates.get(
        "L1_direction_agent",
        templates.get("L1_direction", _default_templates()["L1_direction_agent"]),
    )
    tpl = random.choice(tpl_list)
    question_text = tpl.format(
        obj_a=_the(relation["obj_b_label"]),  # "where is B relative to A?"
        obj_b=_the(relation["obj_a_label"]),
    )
    options, answer = generate_options(correct, ALL_DIRECTIONS)

    return {
        "level": "L1",
        "type": "direction_agent",
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


def _normalize_label_counts(label_counts: dict[str, Any] | None) -> dict[str, int]:
    if not isinstance(label_counts, dict):
        return {}

    normalized: dict[str, int] = {}
    for label, count in label_counts.items():
        if not isinstance(label, str):
            continue
        label_text = label.strip().lower()
        if not label_text or label_text in EXCLUDED_LABELS:
            continue
        try:
            count_int = int(count)
        except (TypeError, ValueError):
            continue
        if count_int < 0:
            continue
        normalized[label_text] = count_int
    return normalized


def _l1_occlusion_question(
    label: str,
    correct: str,
    templates: dict,
    obj_id: int | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    tpl = random.choice(templates.get("L1_occlusion", _default_templates()["L1_occlusion"]))
    question_text = tpl.format(obj_a=_the(label))
    options, answer = generate_options(correct, L1_OCCLUSION_STATES, n_options=3)
    question = {
        "level": "L1",
        "type": "occlusion",
        "question": question_text,
        "options": options,
        "answer": answer,
        "correct_value": correct,
        "obj_a_id": obj_id,
        "obj_a_label": label,
        "mentioned_objects": [_mention("target", label, obj_id)],
        "ambiguity_score": 0.0,
        "relation_unchanged": False,
    }
    if extra:
        question.update(extra)
    return question


def _project_sample_point_records(
    sample_points: np.ndarray,
    camera_pose: CameraPose,
    color_intrinsics: CameraIntrinsics,
) -> list[dict[str, float | int | bool]]:
    records: list[dict[str, float | int | bool]] = []
    for idx, pt in enumerate(sample_points):
        uv, depth = project_to_image(pt, camera_pose, color_intrinsics)
        if uv is None or depth <= 0:
            continue
        u = float(uv[0])
        v = float(uv[1])
        records.append({
            "index": idx,
            "u": u,
            "v": v,
            "depth": float(depth),
            "in_frame": (
                0 <= u < color_intrinsics.width
                and 0 <= v < color_intrinsics.height
            ),
        })
    return records


def _projected_area_from_records(
    projected_records: list[dict[str, float | int | bool]],
    color_intrinsics: CameraIntrinsics,
) -> tuple[float, float]:
    if not projected_records:
        return 0.0, 0.0

    us = [float(rec["u"]) for rec in projected_records]
    vs = [float(rec["v"]) for rec in projected_records]
    u_min = max(0.0, min(us))
    v_min = max(0.0, min(vs))
    u_max = min(float(color_intrinsics.width), max(us))
    v_max = min(float(color_intrinsics.height), max(vs))
    area = max(0.0, u_max - u_min) * max(0.0, v_max - v_min)
    in_frame_count = sum(1 for rec in projected_records if bool(rec["in_frame"]))
    return float(area), float(in_frame_count / len(projected_records))


def _classify_l1_occlusion_metrics(metrics: _L1OcclusionMetrics) -> str:
    if (
        metrics.projected_area < MIN_PROJECTED_AREA_PX
        or metrics.in_frame_ratio < L1_OCCLUSION_MIN_IN_FRAME_RATIO
        or metrics.in_frame_sample_count <= 0
    ):
        return "skip"
    if (
        metrics.valid_in_frame_count <= 0
        or not metrics.sufficient_evidence
    ):
        return "skip"
    if metrics.occlusion_ratio_in_frame <= L1_OCCLUSION_NOT_OCCLUDED_MAX:
        return "not occluded"
    if metrics.occlusion_ratio_in_frame >= L1_OCCLUSION_OCCLUDED_MIN:
        return "occluded"
    return "grayzone"


def _compute_l1_occlusion_metrics(
    obj: dict[str, Any],
    camera_pose: CameraPose,
    color_intrinsics: CameraIntrinsics | None,
    depth_image,
    depth_intrinsics,
    occlusion_backend: str,
    ray_caster,
    instance_mesh_data: InstanceMeshData | None,
) -> _L1OcclusionMetrics:
    backend = str(occlusion_backend)
    if color_intrinsics is None:
        return _make_l1_occlusion_metrics(
            projected_area=0.0,
            in_frame_ratio=0.0,
            occlusion_ratio_in_frame=1.0,
            valid_in_frame_count=0,
            sampled_point_count=0,
            in_frame_sample_count=0,
            backend=backend,
        )

    obj_id = int(obj["id"])
    sample_points = _instance_surface_samples(instance_mesh_data, obj_id)
    target_tri_ids = _instance_triangle_id_set(instance_mesh_data, obj_id)
    sampled_point_count = int(len(sample_points))
    if sampled_point_count <= 0 or not target_tri_ids:
        return _make_l1_occlusion_metrics(
            projected_area=0.0,
            in_frame_ratio=0.0,
            occlusion_ratio_in_frame=1.0,
            valid_in_frame_count=0,
            sampled_point_count=sampled_point_count,
            in_frame_sample_count=0,
            backend=backend,
        )

    projected_records = _project_sample_point_records(
        sample_points, camera_pose, color_intrinsics,
    )
    projected_area, in_frame_ratio = _projected_area_from_records(
        projected_records, color_intrinsics,
    )
    in_frame_records = [rec for rec in projected_records if bool(rec["in_frame"])]
    in_frame_sample_count = len(in_frame_records)
    if (
        projected_area < MIN_PROJECTED_AREA_PX
        or in_frame_ratio < L1_OCCLUSION_MIN_IN_FRAME_RATIO
        or in_frame_sample_count <= 0
    ):
        return _make_l1_occlusion_metrics(
            projected_area=projected_area,
            in_frame_ratio=in_frame_ratio,
            occlusion_ratio_in_frame=1.0,
            valid_in_frame_count=0,
            sampled_point_count=sampled_point_count,
            in_frame_sample_count=in_frame_sample_count,
            backend=backend,
        )

    in_frame_indices = np.asarray(
        [int(rec["index"]) for rec in in_frame_records],
        dtype=np.int64,
    )
    in_frame_points = sample_points[in_frame_indices]

    if backend == "depth":
        if not _depth_backend_inputs_ready(
            depth_image=depth_image,
            depth_intrinsics=depth_intrinsics,
            ray_caster=ray_caster,
            context="L1 occlusion",
        ):
            return _make_l1_occlusion_metrics(
                projected_area=projected_area,
                in_frame_ratio=in_frame_ratio,
                occlusion_ratio_in_frame=1.0,
                valid_in_frame_count=0,
                sampled_point_count=sampled_point_count,
                in_frame_sample_count=in_frame_sample_count,
                backend=backend,
            )
        depth_metrics = compute_mesh_depth_occlusion_metrics(
            # Share the same color-in-frame sample pool with mesh_ray so the
            # evidence threshold uses the same denominator across backends.
            target_points=in_frame_points,
            target_tri_ids=target_tri_ids,
            camera_pose=camera_pose,
            intrinsics=depth_intrinsics,
            depth_image=depth_image,
            ray_caster=ray_caster,
        )
        return _make_l1_occlusion_metrics(
            projected_area=projected_area,
            in_frame_ratio=in_frame_ratio,
            occlusion_ratio_in_frame=float(depth_metrics["occlusion_ratio_in_frame"]),
            valid_in_frame_count=int(depth_metrics["valid_in_frame_count"]),
            sampled_point_count=sampled_point_count,
            in_frame_sample_count=in_frame_sample_count,
            backend=backend,
        )

    if backend == "mesh_ray" and ray_caster is not None:
        camera_pos = np.asarray(camera_pose.position, dtype=np.float64)
        visible_count, valid_count = ray_caster.mesh_visibility_stats(
            camera_pos=camera_pos,
            target_points=in_frame_points,
            target_tri_ids=target_tri_ids,
        )
        occlusion_ratio = 1.0
        if valid_count > 0:
            occlusion_ratio = float(1.0 - (visible_count / valid_count))
        return _make_l1_occlusion_metrics(
            projected_area=projected_area,
            in_frame_ratio=in_frame_ratio,
            occlusion_ratio_in_frame=occlusion_ratio,
            valid_in_frame_count=valid_count,
            sampled_point_count=sampled_point_count,
            in_frame_sample_count=in_frame_sample_count,
            backend=backend,
        )

    return _make_l1_occlusion_metrics(
        projected_area=projected_area,
        in_frame_ratio=in_frame_ratio,
        occlusion_ratio_in_frame=1.0,
        valid_in_frame_count=0,
        sampled_point_count=sampled_point_count,
        in_frame_sample_count=in_frame_sample_count,
        backend=backend,
    )


def _render_instance_projection_mask(
    obj_id: int,
    camera_pose: CameraPose,
    color_intrinsics: CameraIntrinsics,
    instance_mesh_data: InstanceMeshData | None,
) -> np.ndarray | None:
    if instance_mesh_data is None:
        return None
    tri_ids = _instance_triangle_id_set(instance_mesh_data, int(obj_id))
    if not tri_ids:
        return None

    import cv2

    mask = np.zeros(
        (int(color_intrinsics.height), int(color_intrinsics.width)),
        dtype=np.uint8,
    )
    vertices = np.asarray(instance_mesh_data.vertices, dtype=np.float64)
    faces = np.asarray(instance_mesh_data.faces, dtype=np.int64)
    for tri_id in sorted(tri_ids):
        tri_vertices = vertices[faces[int(tri_id)]]
        projected: list[list[int]] = []
        invalid = False
        for vertex in tri_vertices:
            uv, depth = project_to_image(vertex, camera_pose, color_intrinsics)
            if uv is None or depth <= 0:
                invalid = True
                break
            projected.append([int(round(float(uv[0]))), int(round(float(uv[1])))])
        if invalid or len(projected) != 3:
            continue
        xs = [pt[0] for pt in projected]
        ys = [pt[1] for pt in projected]
        if max(xs) < 0 or max(ys) < 0:
            continue
        if min(xs) >= color_intrinsics.width or min(ys) >= color_intrinsics.height:
            continue
        cv2.fillConvexPoly(mask, np.asarray(projected, dtype=np.int32), 255)

    if not np.any(mask):
        return None
    return mask


def _local_mask_overlay_image(
    frame_image: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray | None:
    if frame_image is None or mask is None or not np.any(mask):
        return None

    import cv2

    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x_min = int(xs.min())
    x_max = int(xs.max())
    y_min = int(ys.min())
    y_max = int(ys.max())
    width = x_max - x_min + 1
    height = y_max - y_min + 1
    pad_x = max(1, int(round(width * L1_OCCLUSION_MASK_PAD_RATIO)))
    pad_y = max(1, int(round(height * L1_OCCLUSION_MASK_PAD_RATIO)))

    crop_x0 = max(0, x_min - pad_x)
    crop_y0 = max(0, y_min - pad_y)
    crop_x1 = min(frame_image.shape[1], x_max + pad_x + 1)
    crop_y1 = min(frame_image.shape[0], y_max + pad_y + 1)

    crop_w = crop_x1 - crop_x0
    crop_h = crop_y1 - crop_y0
    need_w = max(0, L1_OCCLUSION_MASK_MIN_CROP_SIZE - crop_w)
    need_h = max(0, L1_OCCLUSION_MASK_MIN_CROP_SIZE - crop_h)
    if need_w > 0:
        extra_left = need_w // 2
        extra_right = need_w - extra_left
        crop_x0 = max(0, crop_x0 - extra_left)
        crop_x1 = min(frame_image.shape[1], crop_x1 + extra_right)
    if need_h > 0:
        extra_top = need_h // 2
        extra_bottom = need_h - extra_top
        crop_y0 = max(0, crop_y0 - extra_top)
        crop_y1 = min(frame_image.shape[0], crop_y1 + extra_bottom)

    crop = frame_image[crop_y0:crop_y1, crop_x0:crop_x1].copy()
    crop_mask = mask[crop_y0:crop_y1, crop_x0:crop_x1]
    if crop.size == 0 or not np.any(crop_mask):
        return None

    overlay = crop.copy()
    fill = np.zeros_like(overlay)
    fill[:, :] = L1_OCCLUSION_MASK_FILL_BGR
    mask_bool = crop_mask > 0
    overlay[mask_bool] = (
        crop[mask_bool] * (1.0 - L1_OCCLUSION_MASK_ALPHA)
        + fill[mask_bool] * L1_OCCLUSION_MASK_ALPHA
    ).astype(np.uint8)
    contours, _ = cv2.findContours(crop_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, L1_OCCLUSION_MASK_BORDER_BGR, 2)
    return overlay


def generate_l1_occlusion_questions(
    objects: list[dict[str, Any]],
    camera_pose: CameraPose,
    color_intrinsics: CameraIntrinsics | None,
    depth_image,
    depth_intrinsics,
    occlusion_backend: str,
    ray_caster,
    instance_mesh_data: InstanceMeshData | None,
    templates: dict,
    label_counts: dict[str, Any] | None = None,
    referable_object_ids: list[int] | None = None,
    frame_image: np.ndarray | None = None,
    occlusion_vlm_adjudicator: Callable[[np.ndarray, str], str | None] | None = None,
) -> list[dict[str, Any]]:
    normalized_counts = _normalize_label_counts(label_counts)
    referable_id_set = _normalize_object_id_set(
        referable_object_ids,
        "referable_object_ids_for_l1_occlusion",
    )
    label_to_objects: dict[str, list[dict[str, Any]]] = {}
    for obj in objects:
        label = str(obj.get("label", "")).strip().lower()
        if not label:
            continue
        label_to_objects.setdefault(label, []).append(obj)

    questions: list[dict[str, Any]] = []

    def _append_geometry_question(
        obj: dict[str, Any],
        label: str,
        source: str,
        vlm_count: int | None,
    ) -> None:
        metrics = _compute_l1_occlusion_metrics(
            obj=obj,
            camera_pose=camera_pose,
            color_intrinsics=color_intrinsics,
            depth_image=depth_image,
            depth_intrinsics=depth_intrinsics,
            occlusion_backend=occlusion_backend,
            ray_caster=ray_caster,
            instance_mesh_data=instance_mesh_data,
        )
        decision = metrics.decision
        if decision == "skip":
            return
        source_used = source
        overlay_available = False
        if (
            decision == "grayzone"
            and frame_image is not None
            and color_intrinsics is not None
            and occlusion_vlm_adjudicator is not None
        ):
            mask = _render_instance_projection_mask(
                int(obj["id"]),
                camera_pose,
                color_intrinsics,
                instance_mesh_data,
            )
            overlay = _local_mask_overlay_image(frame_image, mask) if mask is not None else None
            overlay_available = overlay is not None
            if overlay is not None:
                adjudicated = occlusion_vlm_adjudicator(overlay, label)
                if adjudicated in {"not occluded", "occluded"}:
                    decision = adjudicated
                    source_used = "vlm_mask_adjudication"
        if decision == "grayzone":
            decision = (
                "occluded"
                if metrics.occlusion_ratio_in_frame >= L1_OCCLUSION_GRAYZONE_FALLBACK
                else "not occluded"
            )
            source_used = "geometry_grayzone_fallback"

        questions.append(
            _l1_occlusion_question(
                label=label,
                correct=decision,
                templates=templates,
                obj_id=int(obj["id"]),
                extra={
                    "occlusion_decision_source": source_used,
                    "vlm_label_count": vlm_count,
                    "geometry_projected_area": metrics.projected_area,
                    "geometry_in_frame_ratio": metrics.in_frame_ratio,
                    "geometry_occlusion_ratio_in_frame": metrics.occlusion_ratio_in_frame,
                    "geometry_valid_in_frame_count": metrics.valid_in_frame_count,
                    "geometry_sampled_point_count": metrics.sampled_point_count,
                    "geometry_in_frame_sample_count": metrics.in_frame_sample_count,
                    "geometry_effective_ratio": metrics.effective_ratio,
                    "geometry_sufficient_evidence": metrics.sufficient_evidence,
                    "geometry_backend": metrics.backend,
                    "mask_overlay_available": overlay_available,
                },
            )
        )

    if normalized_counts:
        for label, count in sorted(normalized_counts.items()):
            if count == 0:
                questions.append(
                    _l1_occlusion_question(
                        label=label,
                        correct="not visible",
                        templates=templates,
                        obj_id=None,
                        extra={
                            "occlusion_decision_source": "vlm_count",
                            "vlm_label_count": 0,
                        },
                    )
                )
                continue
            if count != 1:
                continue

            candidates = label_to_objects.get(label, [])
            if len(candidates) == 1:
                _append_geometry_question(
                    obj=candidates[0],
                    label=label,
                    source="geometry",
                    vlm_count=count,
                )
                continue

            referable_candidates = [
                obj for obj in candidates
                if int(obj["id"]) in referable_id_set
            ]
            if len(referable_candidates) == 1:
                _append_geometry_question(
                    obj=referable_candidates[0],
                    label=label,
                    source="geometry",
                    vlm_count=count,
                )
        return questions

    for obj in objects:
        label = str(obj.get("label", "")).strip().lower()
        if not label:
            continue
        _append_geometry_question(
            obj=obj,
            label=label,
            source="geometry_only_fallback",
            vlm_count=None,
        )
    return questions


# ---------------------------------------------------------------------------
#  L1 generators – new reference frames
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
                if not _has_stable_object_centric_facing(ref_c, face_c):
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
                    "facing_anchor_center": ref_c.tolist(),
                    "facing_target_center": face_c.tolist(),
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
            if direction not in CARDINAL_DIRECTIONS_8:
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

def _visibility_status_from_ratio(visible_ratio: float) -> str:
    if visible_ratio >= FULLY_VISIBLE_RATIO_MIN:
        return "fully visible"
    if visible_ratio >= PARTIALLY_VISIBLE_RATIO_MIN:
        return "partially occluded"
    return "not visible"


def _projected_area_summary(
    sample_points: np.ndarray,
    camera_pose: CameraPose,
    color_intrinsics: CameraIntrinsics,
) -> tuple[float, float]:
    projected: list[tuple[float, float]] = []
    for pt in sample_points:
        uv, depth = project_to_image(pt, camera_pose, color_intrinsics)
        if uv is None or depth <= 0:
            continue
        projected.append((float(uv[0]), float(uv[1])))

    if not projected:
        return 0.0, 0.0

    us = [u for u, _ in projected]
    vs = [v for _, v in projected]
    u_min = max(0.0, min(us))
    v_min = max(0.0, min(vs))
    u_max = min(float(color_intrinsics.width), max(us))
    v_max = min(float(color_intrinsics.height), max(vs))
    area = max(0.0, u_max - u_min) * max(0.0, v_max - v_min)
    in_frame = sum(
        1 for u, v in projected
        if 0 <= u < color_intrinsics.width and 0 <= v < color_intrinsics.height
    )
    return float(area), float(in_frame / len(projected))


def _build_modified_scene(
    ray_caster,
    instance_mesh_data: InstanceMeshData | None,
    removed_ids: set[int],
) -> _ModifiedSceneContext | None:
    """Prepare a lightweight counterfactual scene query context."""
    if ray_caster is None:
        return None

    ignored_tri_ids: set[int] = set()
    if instance_mesh_data is not None:
        for obj_id in removed_ids:
            ignored_tri_ids.update(
                _instance_triangle_id_set(instance_mesh_data, int(obj_id))
            )

    return _ModifiedSceneContext(
        ray_caster=ray_caster,
        ignored_tri_ids=frozenset(ignored_tri_ids),
    )


def _compute_target_visibility(
    modified_scene: _ModifiedSceneContext | None,
    target_surface_points: np.ndarray,
    target_triangle_ids: set[int],
    camera_pos: np.ndarray,
) -> tuple[str, float]:
    """Evaluate target visibility against a possibly modified scene."""
    if (
        modified_scene is None
        or modified_scene.ray_caster is None
        or len(target_surface_points) == 0
        or not target_triangle_ids
    ):
        return "not visible", 0.0

    visible_ratio = modified_scene.ray_caster.mesh_visibility_ratio(
        camera_pos=camera_pos,
        target_points=target_surface_points,
        target_tri_ids=target_triangle_ids,
        ignored_tri_ids=set(modified_scene.ignored_tri_ids),
    )
    return _visibility_status_from_ratio(visible_ratio), float(visible_ratio)


def _compute_visibility_status_per_object(
    objects: list[dict],
    camera_pose: CameraPose,
    color_intrinsics: CameraIntrinsics | None,
    depth_image=None,
    depth_intrinsics=None,
    occlusion_backend: str = "depth",
    ray_caster=None,
    instance_mesh_data: InstanceMeshData | None = None,
    modified_scene: _ModifiedSceneContext | None = None,
) -> dict[int, tuple[str, float]]:
    if color_intrinsics is None:
        return {int(obj["id"]): ("not visible", 0.0) for obj in objects}

    backend = str(occlusion_backend)
    camera_pos = np.array(camera_pose.position, dtype=np.float64)
    if backend == "mesh_ray" and modified_scene is None:
        modified_scene = _build_modified_scene(
            ray_caster=ray_caster,
            instance_mesh_data=instance_mesh_data,
            removed_ids=set(),
        )

    visibility: dict[int, tuple[str, float]] = {}
    for obj in objects:
        obj_id = int(obj["id"])
        sample_points = _instance_surface_samples(instance_mesh_data, obj_id)
        target_tri_ids = _instance_triangle_id_set(instance_mesh_data, obj_id)
        projected_area, in_frame_ratio = _projected_area_summary(
            sample_points, camera_pose, color_intrinsics,
        )
        if (
            len(sample_points) == 0
            or not target_tri_ids
            or projected_area < MIN_PROJECTED_AREA_PX
            or in_frame_ratio < MIN_IN_FRAME_RATIO
        ):
            visibility[obj_id] = ("not visible", 0.0)
            continue

        if backend == "mesh_ray":
            status, visible_ratio = _compute_target_visibility(
                modified_scene=modified_scene,
                target_surface_points=sample_points,
                target_triangle_ids=target_tri_ids,
                camera_pos=camera_pos,
            )
            visibility[obj_id] = (status, float(visible_ratio))
            continue

        if backend == "depth":
            if not _depth_backend_inputs_ready(
                depth_image=depth_image,
                depth_intrinsics=depth_intrinsics,
                ray_caster=ray_caster,
                context="counterfactual visibility",
            ):
                visibility[obj_id] = ("not visible", 0.0)
                continue
            status, visible_ratio = compute_mesh_depth_occlusion(
                target_points=sample_points,
                target_tri_ids=target_tri_ids,
                camera_pose=camera_pose,
                intrinsics=depth_intrinsics,
                depth_image=depth_image,
                ray_caster=ray_caster,
            )
            visibility[obj_id] = (status, float(visible_ratio))
            continue

        raise ValueError(f"Unsupported occlusion backend: {backend}")

    return visibility


def _counterfactual_occlusion_backend(
    occlusion_backend: str,
    ray_caster,
    instance_mesh_data: InstanceMeshData | None,
) -> str:
    """Return the backend used for L2 counterfactual visibility comparisons.

    Even when the primary per-frame backend is ``depth``, L2 comparisons still
    use ``mesh_ray`` because a counterfactual camera/object edit does not come
    with a newly rendered depth map.
    """
    requested_backend = str(occlusion_backend)
    if requested_backend not in {"depth", "mesh_ray"}:
        raise ValueError(f"Unsupported occlusion backend: {requested_backend}")
    if ray_caster is None or instance_mesh_data is None:
        raise RuntimeError(
            "Counterfactual visibility requires mesh geometry for both depth and mesh_ray backends",
        )
    if requested_backend == "depth":
        logger.debug(
            "Counterfactual visibility requested with depth backend; falling back to mesh_ray because no counterfactual depth map exists.",
        )
    return "mesh_ray"


def _cap_question_groups(
    questions_by_key: dict[Any, list[dict]],
    max_per_group: int | None,
) -> list[dict]:
    """Downsample question pools so each grouping key contributes at most N items."""
    groups = list(questions_by_key.values())
    if max_per_group is None or max_per_group <= 0:
        return [q for group in groups for q in group]

    capped: list[dict] = []
    for group in groups:
        if len(group) > max_per_group:
            capped.extend(random.sample(group, max_per_group))
        else:
            capped.extend(group)
    return capped


def _balance_l2_object_move_attachment_counts(
    questions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Cap per-type unattached L2 object-move questions relative to attached count."""
    keep_mask = [True] * len(questions)
    grouped_indices: dict[str, list[int]] = {}
    for idx, question in enumerate(questions):
        if str(question.get("level", "")).strip() != "L2":
            continue
        qtype = str(question.get("type", "")).strip()
        if not qtype.startswith("object_move_"):
            continue
        grouped_indices.setdefault(qtype, []).append(idx)

    for qtype, indices in grouped_indices.items():
        attached = [
            idx for idx in indices
            if bool(questions[idx].get("attachment_remapped", False))
        ]
        unattached = [
            idx for idx in indices
            if not bool(questions[idx].get("attachment_remapped", False))
        ]
        allowed_unattached = len(attached) if attached else 3
        if len(unattached) <= allowed_unattached:
            continue
        for idx in unattached[allowed_unattached:]:
            keep_mask[idx] = False
        logger.info(
            "Balanced %s questions for one frame: kept %d attached and %d/%d unattached",
            qtype,
            len(attached),
            allowed_unattached,
            len(unattached),
        )

    return [
        question for idx, question in enumerate(questions)
        if keep_mask[idx]
    ]

def generate_l2_object_move(
    objects: list[dict],
    attachment_graph: dict[int, list[int]],
    attached_by: dict[int, int],
    camera_pose: CameraPose,
    templates: dict,
    max_per_object: int = 3,
    room_bounds: dict | None = None,
    collision_objects: list[dict] | None = None,
    movement_objects: list[dict] | None = None,
    object_map: dict[int, dict] | None = None,
) -> list[dict]:
    """Generate L2.1 object-movement questions for a scene."""
    questions_by_object: dict[int, list[dict]] = {}
    movement_scene_objects = movement_objects if movement_objects is not None else objects
    obj_map = object_map if object_map is not None else {
        int(o["id"]): o for o in movement_scene_objects
    }

    for obj in objects:
        # Skip structural room elements — they cannot be "moved" in any
        # meaningful physical sense and confuse human annotators.
        if obj.get("label", "").lower() in EXCLUDED_LABELS:
            continue

        move_source_id = _resolve_attachment_root_id(obj["id"], attached_by)
        move_source = obj_map.get(move_source_id)
        if move_source is None:
            continue
        attachment_remapped = move_source_id != obj["id"]

        delta, changed = find_meaningful_movement(
            movement_scene_objects, attachment_graph, move_source_id, camera_pose,
            room_bounds=room_bounds,
            collision_objects=collision_objects,
        )
        if delta is None:
            continue

        # Only keep relation changes involving the queried object.
        changed = [
            ch for ch in changed
            if obj["id"] in (ch["obj_a_id"], ch["obj_b_id"])
        ]
        if not changed:
            continue

        # Describe the movement in natural language
        direction_desc = _delta_to_description(delta, camera_pose)
        distance_desc = f"{np.linalg.norm(delta):.1f}m"

        tpl_list = templates.get(
            "L2_object_move_agent",
            templates.get("L2_object_move", _default_templates()["L2_object_move_agent"]),
        )

        obj_questions: list[dict] = []
        for ch in changed:
            # Ensure the changed relation directly involves the queried object.
            if obj["id"] not in (ch["obj_a_id"], ch["obj_b_id"]):
                continue

            # Pick a changed relation field
            for field, vals in ch["changes"].items():
                if field == "direction_b_rel_a":
                    pool = ALL_DIRECTIONS
                    field_tpl_key = "L2_object_move_agent"
                elif field == "distance_bin":
                    pool = ALL_DISTANCES
                    field_tpl_key = "L2_object_move_distance"
                elif field in ("occlusion_a", "occlusion_b"):
                    pool = ALL_OCCLUSION
                    field_tpl_key = "L2_object_move_occlusion"
                else:
                    continue

                if obj["id"] == ch["obj_b_id"]:
                    relation_obj_b_id = ch["obj_b_id"]
                    relation_obj_c_id = ch["obj_a_id"]
                    answer_value = vals["new"]
                elif obj["id"] == ch["obj_a_id"]:
                    relation_obj_b_id = ch["obj_a_id"]
                    relation_obj_c_id = ch["obj_b_id"]
                    answer_value = (
                        _invert_direction(vals["new"])
                        if field == "direction_b_rel_a"
                        else vals["new"]
                    )
                else:
                    continue

                obj_b_label = obj_map.get(relation_obj_b_id, {}).get("label", "object")
                obj_c_label = obj_map.get(relation_obj_c_id, {}).get("label", "object")

                field_tpl_list = templates.get(
                    field_tpl_key,
                    templates.get(
                        field_tpl_key.removesuffix("_agent"),
                        _default_templates()[field_tpl_key],
                    ),
                )
                tpl = random.choice(field_tpl_list)
                question_text = tpl.format(
                    obj_a=_the(move_source["label"]),
                    direction=direction_desc,
                    direction_with_camera_hint=_direction_with_camera_hint(direction_desc),
                    distance=distance_desc,
                    obj_b=_the(obj_b_label),
                    obj_c=_the(obj_c_label),
                )
                options, answer = generate_options(answer_value, pool)

                obj_questions.append({
                    "level": "L2",
                    "type": (
                        "object_move_distance"
                        if field_tpl_key == "L2_object_move_distance"
                        else (
                            "object_move_occlusion"
                            if field_tpl_key == "L2_object_move_occlusion"
                            else "object_move_agent"
                        )
                    ),
                    "question": question_text,
                    "options": options,
                    "answer": answer,
                    "correct_value": answer_value,
                    "moved_obj_id": move_source_id,
                    "moved_obj_label": move_source["label"],
                    "query_obj_id": obj["id"],
                    "query_obj_label": obj["label"],
                    "attachment_remapped": attachment_remapped,
                    "obj_b_id": relation_obj_b_id,
                    "obj_b_label": obj_b_label,
                    "obj_c_id": relation_obj_c_id,
                    "obj_c_label": obj_c_label,
                    "mentioned_objects": [
                        _mention("moved_object", move_source["label"], move_source_id),
                        _mention("query_object", obj["label"], obj["id"]),
                        _mention("relation_obj_b", obj_b_label, relation_obj_b_id),
                        _mention("relation_obj_c", obj_c_label, relation_obj_c_id),
                    ],
                    "delta": delta.tolist(),
                    "relation_unchanged": False,
                    "has_attachment_chain": len(get_attachment_chain_ids(move_source_id, attachment_graph)) > 0,
                })

        if obj_questions:
            object_key = int(obj["id"])
            questions_by_object.setdefault(object_key, []).extend(obj_questions)

    # Cap per query-object instance so repeated labels can still contribute.
    return _cap_question_groups(questions_by_object, max_per_object)


def generate_l2_viewpoint_move(
    objects: list[dict],
    camera_pose: CameraPose,
    color_intrinsics: CameraIntrinsics | None,
    depth_image,
    depth_intrinsics,
    occlusion_backend: str,
    ray_caster,
    instance_mesh_data: InstanceMeshData | None,
    templates: dict,
) -> list[dict]:
    """Generate L2.2 viewpoint-movement questions.

    Compares target-object visibility before/after moving the observer.
    """
    questions: list[dict] = []
    if color_intrinsics is None:
        return questions
    tpl_list = templates.get("L2_viewpoint_move", _default_templates()["L2_viewpoint_move"])
    compare_backend = _counterfactual_occlusion_backend(
        occlusion_backend, ray_caster, instance_mesh_data,
    )
    scene_context = _build_modified_scene(ray_caster, instance_mesh_data, set())

    original_visibility = _compute_visibility_status_per_object(
        objects, camera_pose, color_intrinsics,
        depth_image=None,
        depth_intrinsics=None,
        occlusion_backend=compare_backend,
        ray_caster=ray_caster,
        instance_mesh_data=instance_mesh_data,
        modified_scene=scene_context,
    )

    for direction, prompt_direction in (
        ("right", "right"),
        ("left", "left"),
        ("forward", "forward"),
        ("back", "backward"),
    ):
        for dist in (1.0, 2.0, 3.0):
            new_pose = apply_viewpoint_change(camera_pose, direction, dist)
            new_visibility = _compute_visibility_status_per_object(
                objects, new_pose, color_intrinsics,
                depth_image=None,
                depth_intrinsics=None,
                occlusion_backend=compare_backend,
                ray_caster=ray_caster,
                instance_mesh_data=instance_mesh_data,
                modified_scene=scene_context,
            )

            for obj in objects:
                old_status, _old_ratio = original_visibility.get(obj["id"], ("not visible", 0.0))
                new_status, _new_ratio = new_visibility.get(obj["id"], ("not visible", 0.0))
                if old_status == new_status:
                    continue

                tpl = random.choice(tpl_list)
                question_text = tpl.format(
                    direction=prompt_direction,
                    direction_with_camera_hint=_direction_with_camera_hint(prompt_direction),
                    distance=f"{dist:.0f}m",
                    obj_a=_the(obj["label"]),
                )
                options, answer = generate_options(new_status, ALL_OCCLUSION)
                questions.append({
                    "level": "L2",
                    "type": "viewpoint_move",
                    "question": question_text,
                    "options": options,
                    "answer": answer,
                    "correct_value": new_status,
                    "obj_a_id": obj["id"],
                    "obj_a_label": obj["label"],
                    "old_visibility": old_status,
                    "new_visibility": new_status,
                    "camera_motion_model": "translate_only",
                    "camera_intrinsics_unchanged": True,
                    "camera_orientation_unchanged": True,
                    "mentioned_objects": [
                        _mention("target", obj["label"], obj["id"]),
                    ],
                    "relation_unchanged": False,
                })

    return questions


def generate_l2_object_remove(
    objects: list[dict],
    attachment_graph: dict[int, list[int]],
    camera_pose: CameraPose,
    color_intrinsics: CameraIntrinsics | None,
    depth_image,
    depth_intrinsics,
    occlusion_backend: str,
    ray_caster,
    instance_mesh_data: InstanceMeshData | None,
    templates: dict,
) -> list[dict]:
    """Generate L2.3 object-removal questions from visibility changes."""
    questions: list[dict] = []
    if color_intrinsics is None:
        return questions
    tpl_list = templates.get("L2_object_remove", _default_templates()["L2_object_remove"])
    compare_backend = _counterfactual_occlusion_backend(
        occlusion_backend, ray_caster, instance_mesh_data,
    )
    original_scene_context = _build_modified_scene(ray_caster, instance_mesh_data, set())
    original_visibility = _compute_visibility_status_per_object(
        objects, camera_pose, color_intrinsics,
        depth_image=None,
        depth_intrinsics=None,
        occlusion_backend=compare_backend,
        ray_caster=ray_caster,
        instance_mesh_data=instance_mesh_data,
        modified_scene=original_scene_context,
    )

    original_ids = {int(obj["id"]) for obj in objects}
    for obj in objects:
        remaining = apply_removal(objects, attachment_graph, obj["id"], cascade=False)
        if len(remaining) < 2:
            continue
        remaining_ids = {int(other["id"]) for other in remaining}
        removed_ids = original_ids - remaining_ids
        removal_scene_context = _build_modified_scene(
            ray_caster, instance_mesh_data, removed_ids,
        )
        new_visibility = _compute_visibility_status_per_object(
            remaining, camera_pose, color_intrinsics,
            depth_image=None,
            depth_intrinsics=None,
            occlusion_backend=compare_backend,
            ray_caster=ray_caster,
            instance_mesh_data=instance_mesh_data,
            modified_scene=removal_scene_context,
        )

        for other in remaining:
            old_status, _old_ratio = original_visibility.get(other["id"], ("not visible", 0.0))
            new_status, _new_ratio = new_visibility.get(other["id"], ("not visible", 0.0))
            if old_status == new_status:
                continue

            tpl = random.choice(tpl_list)
            question_text = tpl.format(
                obj_a=_the(obj["label"]),
                obj_b=_the(other["label"]),
            )
            options, answer = generate_options(new_status, ALL_OCCLUSION)
            questions.append({
                "level": "L2",
                "type": "object_remove",
                "question": question_text,
                "options": options,
                "answer": answer,
                "correct_value": new_status,
                "removed_obj_id": obj["id"],
                "removed_obj_label": obj["label"],
                "obj_b_id": other["id"],
                "obj_b_label": other["label"],
                "old_visibility": old_status,
                "new_visibility": new_status,
                "mentioned_objects": [
                    _mention("removed_object", obj["label"], obj["id"]),
                    _mention("remaining_object", other["label"], other["id"]),
                ],
                "relation_unchanged": False,
            })

    return questions


# ---------------------------------------------------------------------------
#  L2 generators — new reference frames
# ---------------------------------------------------------------------------

def generate_l2_object_move_object_centric(
    objects: list[dict],
    attachment_graph: dict[int, list[int]],
    attached_by: dict[int, int],
    camera_pose: CameraPose,
    templates: dict,
    max_per_object: int = 3,
    room_bounds: dict | None = None,
    collision_objects: list[dict] | None = None,
    movement_objects: list[dict] | None = None,
    object_map: dict[int, dict] | None = None,
) -> list[dict]:
    """L2 object-move questions answered in a query-centric object-centric frame."""
    questions_by_object: dict[int, list[dict]] = {}
    movement_scene_objects = movement_objects if movement_objects is not None else objects
    obj_map = object_map if object_map is not None else {
        int(o["id"]): o for o in movement_scene_objects
    }
    tpl_list = templates.get(
        "L2_object_move_object_centric",
        _default_templates()["L2_object_move_object_centric"],
    )
    horizontal_answer_pool = list(HORIZONTAL_DIRECTIONS)

    for obj in objects:
        if obj.get("label", "").lower() in EXCLUDED_LABELS:
            continue

        move_source_id = _resolve_attachment_root_id(obj["id"], attached_by)
        move_source = obj_map.get(move_source_id)
        if move_source is None:
            continue
        moved_ids = set(get_attachment_chain_ids(move_source_id, attachment_graph)) | {move_source_id}
        if obj["id"] not in moved_ids:
            continue
        attachment_remapped = move_source_id != obj["id"]
        query_center = np.array(obj["center"], dtype=float)

        obj_questions: list[dict] = []
        for face in objects:
            if face["id"] in moved_ids:
                continue
            face_c = np.array(face["center"], dtype=float)
            if not _has_stable_object_centric_facing(query_center, face_c):
                continue

            valid_rotations = find_meaningful_orbit_rotation(
                movement_scene_objects,
                attachment_graph,
                move_source_id,
                face["id"],
                room_bounds=room_bounds,
                collision_objects=collision_objects,
            )
            if not valid_rotations:
                continue

            for rotation in valid_rotations:
                rotated_map = {o["id"]: o for o in rotation["objects"]}
                rotated_query = rotated_map.get(obj["id"])
                if rotated_query is None:
                    continue
                new_query_center = np.array(rotated_query["center"], dtype=float)
                if not _has_stable_object_centric_facing(new_query_center, face_c):
                    continue
                query_delta = new_query_center - query_center

                for ref in objects:
                    if ref["id"] in moved_ids or ref["id"] == face["id"]:
                        continue

                    labels = [
                        obj["label"],
                        ref["label"],
                        face["label"],
                    ]
                    if move_source_id != obj["id"]:
                        labels.append(move_source["label"])
                    if len(set(labels)) < len(labels):
                        continue

                    ref_c = np.array(ref["center"], dtype=float)
                    old_dir, old_amb = primary_direction_object_centric(
                        query_center, face_c, ref_c,
                    )
                    new_dir, new_amb = primary_direction_object_centric(
                        new_query_center, face_c, ref_c,
                    )
                    if old_dir not in horizontal_answer_pool or new_dir not in horizontal_answer_pool:
                        continue
                    if max(old_amb, new_amb) > 0.7:
                        continue
                    if old_dir == new_dir:
                        continue

                    tpl = random.choice(tpl_list)
                    question_text = tpl.format(
                        obj_move_source=_the(move_source["label"]),
                        obj_query=_the(obj["label"]),
                        obj_ref=_the(ref["label"]),
                        obj_face=_the(face["label"]),
                        angle=rotation["angle"],
                        rotation_direction=rotation["rotation_direction"],
                    )
                    options, answer = generate_options(new_dir, horizontal_answer_pool)
                    obj_questions.append({
                        "level": "L2",
                        "type": "object_move_object_centric",
                        "reference_frame": "object_centric",
                        "question": question_text,
                        "options": options,
                        "answer": answer,
                        "correct_value": new_dir,
                        "moved_obj_id": move_source_id,
                        "moved_obj_label": move_source["label"],
                        "query_obj_id": obj["id"],
                        "query_obj_label": obj["label"],
                        "attachment_remapped": attachment_remapped,
                        "obj_ref_id": ref["id"],
                        "obj_ref_label": ref["label"],
                        "obj_face_id": face["id"],
                        "obj_face_label": face["label"],
                        "facing_anchor_center": new_query_center.tolist(),
                        "facing_target_center": face_c.tolist(),
                        "rotation_angle": rotation["angle"],
                        "rotation_direction": rotation["rotation_direction"],
                        "mentioned_objects": [
                            _mention("moved_object", move_source["label"], move_source_id),
                            _mention("query_object", obj["label"], obj["id"]),
                            _mention("reference_object", ref["label"], ref["id"]),
                            _mention("reference_facing", face["label"], face["id"]),
                        ],
                        "delta": query_delta.tolist(),
                        "relation_unchanged": False,
                    })

        if obj_questions:
            object_key = int(obj["id"])
            questions_by_object.setdefault(object_key, []).extend(obj_questions)

    return _cap_question_groups(questions_by_object, max_per_object)


def generate_l2_object_move_allocentric(
    objects: list[dict],
    attachment_graph: dict[int, list[int]],
    attached_by: dict[int, int],
    camera_pose: CameraPose,
    templates: dict,
    max_per_object: int = 3,
    room_bounds: dict | None = None,
    collision_objects: list[dict] | None = None,
    movement_objects: list[dict] | None = None,
    object_map: dict[int, dict] | None = None,
) -> list[dict]:
    """L2 object-move questions answered in allocentric (cardinal) frame."""
    questions_by_object: dict[int, list[dict]] = {}
    movement_scene_objects = movement_objects if movement_objects is not None else objects
    obj_map = object_map if object_map is not None else {
        int(o["id"]): o for o in movement_scene_objects
    }
    cam_cardinal = camera_cardinal_direction(camera_pose)
    tpl_list = templates.get(
        "L2_object_move_allocentric",
        _default_templates()["L2_object_move_allocentric"],
    )

    for obj in objects:
        if obj.get("label", "").lower() in EXCLUDED_LABELS:
            continue

        move_source_id = _resolve_attachment_root_id(obj["id"], attached_by)
        move_source = obj_map.get(move_source_id)
        if move_source is None:
            continue
        moved_ids = set(get_attachment_chain_ids(move_source_id, attachment_graph)) | {move_source_id}
        if obj["id"] not in moved_ids:
            continue
        attachment_remapped = move_source_id != obj["id"]

        delta, _changed = find_meaningful_movement(
            movement_scene_objects, attachment_graph, move_source_id, camera_pose,
            room_bounds=room_bounds,
            collision_objects=collision_objects,
        )
        if delta is None:
            continue

        direction_desc = _delta_to_cardinal_description(delta)
        distance_desc = f"{np.linalg.norm(delta):.1f}m"
        new_center = np.array(obj["center"]) + delta

        obj_questions: list[dict] = []
        for ref in objects:
            if ref["id"] in (obj["id"], move_source_id):
                continue
            if ref["label"] == obj["label"]:
                continue

            ref_c = np.array(ref["center"])
            new_dir, amb = primary_direction_allocentric(new_center, ref_c)
            if amb > 0.7:
                continue
            if new_dir not in CARDINAL_DIRECTIONS_8:
                continue

            old_dir, _ = primary_direction_allocentric(np.array(obj["center"]), ref_c)
            if old_dir == new_dir:
                continue

            tpl = random.choice(tpl_list)
            question_text = tpl.format(
                obj_move_source=_the(move_source["label"]),
                obj_query=_the(obj["label"]),
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
                "moved_obj_id": move_source_id,
                "moved_obj_label": move_source["label"],
                "query_obj_id": obj["id"],
                "query_obj_label": obj["label"],
                "attachment_remapped": attachment_remapped,
                "obj_ref_id": ref["id"],
                "obj_ref_label": ref["label"],
                "mentioned_objects": [
                    _mention("moved_object", move_source["label"], move_source_id),
                    _mention("query_object", obj["label"], obj["id"]),
                    _mention("reference_object", ref["label"], ref["id"]),
                ],
                "delta": delta.tolist(),
                "relation_unchanged": False,
            })

        if obj_questions:
            object_key = int(obj["id"])
            questions_by_object.setdefault(object_key, []).extend(obj_questions)

    return _cap_question_groups(questions_by_object, max_per_object)


# ---------------------------------------------------------------------------
#  Attachment relation generators
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
#  L3 generators
# ---------------------------------------------------------------------------

def generate_l3_attachment_chain(
    objects: list[dict],
    attachment_graph: dict[int, list[int]],
    attached_by: dict[int, int],
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
    tpl_list = templates.get("L3_attachment_chain", _default_templates()["L3_attachment_chain"])

    for grandparent_id, parent_ids in attachment_graph.items():
        grandparent_id = int(grandparent_id)
        grandparent = obj_map.get(grandparent_id)
        if grandparent is None:
            continue

        for parent_id in parent_ids:
            parent_id = int(parent_id)
            # Second hop: does the intermediate object itself attach to anything?
            grandchild_ids = attachment_graph.get(parent_id) or []
            if not grandchild_ids:
                continue  # no depth-2 chain here

            parent = obj_map.get(parent_id)
            if parent is None:
                continue

            # All objects in this grandparent's chain (not eligible as neighbour D)
            this_chain: set[int] = (
                set(get_attachment_chain_ids(grandparent_id, attachment_graph))
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
                    "type": "attachment_chain",
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
    tpl_list = templates.get(
        "L3_coordinate_rotation_agent",
        templates.get(
            "L3_coordinate_rotation",
            _default_templates()["L3_coordinate_rotation_agent"],
        ),
    )

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
            # The relation engine stores "B relative to A", while this template
            # asks for "A relative to B", so invert the direction before use.
            old_dir = _invert_direction(vals["old"])
            new_dir = _invert_direction(vals["new"])
            options, answer_letter = generate_options(new_dir, ALL_DIRECTIONS)

            questions.append({
                "level": "L3",
                "type": "coordinate_rotation_agent",
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
                "old_direction": old_dir,
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
                    ref_rot_c = np.array(ref_rot["center"])
                    face_rot_c = np.array(face_rot["center"])
                    target_rot_c = np.array(target_rot["center"])
                    if not _has_stable_object_centric_facing(ref_rot_c, face_rot_c):
                        continue

                    new_dir, amb = primary_direction_object_centric(
                        ref_rot_c,
                        face_rot_c,
                        target_rot_c,
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
                        "facing_anchor_center": ref_rot_c.tolist(),
                        "facing_target_center": face_rot_c.tolist(),
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
                if new_dir not in CARDINAL_DIRECTIONS_8:
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

def get_attachment_chain_ids(obj_id: int, attachment_graph: dict) -> list[int]:
    """Get all dependent IDs (wrapper for import convenience)."""
    from .support_graph import get_attachment_chain
    return get_attachment_chain(obj_id, attachment_graph)


def _resolve_attachment_root_id(obj_id: int, attached_by: dict[int, int]) -> int:
    """Return the lowest attachment ancestor of *obj_id*, or itself if unattached."""
    current = int(obj_id)
    seen: set[int] = set()

    while current not in seen:
        seen.add(current)
        parent = attached_by.get(current)
        if parent is None:
            break
        try:
            current = int(parent)
        except (TypeError, ValueError):
            break

    return current


def _normalize_object_id_set(
    object_ids: list[int] | list[str] | None,
    field_name: str,
) -> set[int]:
    """Best-effort int normalization for object-ID filters."""
    normalized: set[int] = set()
    if object_ids is None:
        return normalized

    for obj_id in object_ids:
        try:
            normalized.add(int(obj_id))
        except (TypeError, ValueError):
            logger.warning("Skipping non-integer %s entry: %r", field_name, obj_id)
    return normalized


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
        ("query_obj_id", "query_obj_label", "query_obj"),
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


def _enforce_stable_facing_references(
    questions: list[dict[str, Any]],
    id_to_object: dict[int, dict],
) -> list[dict[str, Any]]:
    """Drop any question whose "stand at A, face B" reference frame is unstable.

    This is a semantic safety net: any question carrying both `obj_ref_id` and
    `obj_face_id` is treated as using an object-defined facing frame, regardless
    of its formal question type.
    """
    kept: list[dict[str, Any]] = []
    removed = 0
    for question in questions:
        ref_id = question.get("obj_ref_id")
        face_id = question.get("obj_face_id")
        if ref_id is None or face_id is None:
            kept.append(question)
            continue

        anchor_center = question.get("facing_anchor_center")
        facing_center = question.get("facing_target_center")
        if anchor_center is None or facing_center is None:
            ref_obj = id_to_object.get(int(ref_id))
            face_obj = id_to_object.get(int(face_id))
            if ref_obj is None or face_obj is None:
                kept.append(question)
                continue
            anchor_center = ref_obj["center"]
            facing_center = face_obj["center"]

        if _has_stable_object_centric_facing(
            np.array(anchor_center, dtype=float),
            np.array(facing_center, dtype=float),
        ):
            kept.append(question)
            continue

        removed += 1

    if removed:
        logger.info("Stable-facing filter removed %d questions", removed)
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
    attachment_graph: dict[int, list[int]],
    attached_by: dict[int, int],
    camera_pose: CameraPose,
    support_chain_graph: dict[int, list[int]] | None = None,
    support_chain_by: dict[int, int] | None = None,
    color_intrinsics: CameraIntrinsics | None = None,
    depth_image=None,
    depth_intrinsics=None,
    occlusion_backend: str = "depth",
    ray_caster=None,
    instance_mesh_data: InstanceMeshData | None = None,
    templates: dict | None = None,
    visible_object_ids: list[int] | None = None,
    referable_object_ids: list[int] | None = None,
    label_counts: dict[str, Any] | None = None,
    frame_image: np.ndarray | None = None,
    occlusion_vlm_adjudicator: Callable[[np.ndarray, str], str | None] | None = None,
    room_bounds: dict | None = None,
    wall_objects: list[dict] | None = None,
    attachment_edges: list[dict] | None = None,
) -> list[dict]:
    """Generate all question types for a single scene + frame.

    depth_image: float32 depth map in metres (from ScanNet depth PNG), or None.
    depth_intrinsics: CameraIntrinsics for the depth camera, or None.
    visible_object_ids: if provided, restrict all questions to objects whose
    centre projects into this frame.  Questions about off-screen objects are
    unanswerable from the image and should never be included.
    referable_object_ids: if provided, restrict question generation to the
    object_id subset judged referable by the VLM for this frame.
    label_counts: if provided, use per-label VLM counts to generate L1
    occlusion questions where count==0 => not visible and count==1 routes to
    geometry-based not-occluded / occluded classification.
    frame_image: optional BGR frame image used for gray-zone local mask VLM
    adjudication in L1 occlusion.
    occlusion_vlm_adjudicator: optional callback that receives a local masked
    BGR crop and the target label, and returns "not occluded", "occluded", or
    None.
    room_bounds: dict with bbox_min/bbox_max from wall/floor mesh, or None.
    wall_objects: visible filtering for ordinary objects does not touch these;
    they are only used to construct allocentric wall-anchor wording.

    Returns a list of question dicts.
    """
    if templates is None:
        templates = _load_templates()
    if attachment_edges is None:
        attachment_edges = []
    if support_chain_graph is None:
        support_chain_graph = attachment_graph
    if support_chain_by is None:
        support_chain_by = attached_by
    attachment_edge_input = len(attachment_edges)

    # Restrict to objects visible in this frame so every question can be
    # answered by looking at the image.
    if visible_object_ids is not None:
        vis_set = _normalize_object_id_set(visible_object_ids, "visible_object_ids")
        objects = [o for o in objects if o["id"] in vis_set]
        attachment_graph = {
            k: [c for c in v if c in vis_set]
            for k, v in attachment_graph.items()
            if k in vis_set
        }
        attached_by = {k: v for k, v in attached_by.items() if k in vis_set and v in vis_set}
        support_chain_graph = {
            k: [c for c in v if c in vis_set]
            for k, v in support_chain_graph.items()
            if k in vis_set
        }
        support_chain_by = {
            k: v for k, v in support_chain_by.items() if k in vis_set and v in vis_set
        }
        attachment_edges = [
            e for e in attachment_edges
            if int(e["parent_id"]) in vis_set and int(e["child_id"]) in vis_set
        ]

    attachment_edge_count_visible = len(attachment_edges)

    # Apply the shared hard blacklist once for both graph construction and
    # question generation.
    all_objects_for_graph = [
        o for o in objects
        if o.get("label", "").lower() not in EXCLUDED_LABELS
    ]
    objects_for_questions = list(all_objects_for_graph)
    # L1 occlusion can rely on per-label VLM counts, so keep a pre-referable
    # pool of question-eligible visible objects for that path only.
    l1_occlusion_objects = list(objects_for_questions)
    l2_collision_objects = list(objects_for_questions)
    graph_ids = {int(o["id"]) for o in all_objects_for_graph}
    attachment_edges = [
        e for e in attachment_edges
        if int(e["parent_id"]) in graph_ids and int(e["child_id"]) in graph_ids
    ]
    attachment_edge_count_nonexcluded = len(attachment_edges)
    attachment_context_ids: set[int] = set()

    if referable_object_ids is None:
        raise ValueError(
            "generate_all_questions requires referable_object_ids from VLM referability filtering"
        )

    referable_set = _normalize_object_id_set(
        referable_object_ids,
        "referable_object_ids",
    )
    attachment_parent_ids = {
        int(e["parent_id"])
        for e in attachment_edges
        if int(e["child_id"]) in referable_set
    }
    attachment_context_ids = attachment_parent_ids
    graph_allowed_ids = referable_set | attachment_context_ids
    all_objects_for_graph = [
        o for o in all_objects_for_graph
        if int(o["id"]) in graph_allowed_ids
    ]
    objects_for_questions = [
        o for o in objects_for_questions
        if int(o["id"]) in referable_set
    ]
    attachment_graph = {
        k: [c for c in v if c in graph_allowed_ids]
        for k, v in attachment_graph.items()
        if k in graph_allowed_ids
    }
    attached_by = {
        k: v for k, v in attached_by.items()
        if k in graph_allowed_ids and v in graph_allowed_ids
    }
    attachment_edges = [
        e for e in attachment_edges
        if int(e["parent_id"]) in graph_allowed_ids and int(e["child_id"]) in graph_allowed_ids
    ]
    objects_uniq = list(objects_for_questions)
    referable_question_ids = {int(o["id"]) for o in objects_uniq}
    support_chain_graph = {
        k: filtered_children
        for k, v in support_chain_graph.items()
        if k in referable_question_ids
        for filtered_children in ([c for c in v if c in referable_question_ids],)
        if filtered_children
    }
    support_chain_by = {
        k: v for k, v in support_chain_by.items()
        if k in referable_question_ids and v in referable_question_ids
    }

    graph_eligible_ids = referable_set | attachment_context_ids
    attachment_graph = {
        k: [c for c in v if c in graph_eligible_ids]
        for k, v in attachment_graph.items()
        if k in graph_eligible_ids
    }
    attached_by = {
        k: v for k, v in attached_by.items()
        if k in graph_eligible_ids and v in graph_eligible_ids
    }
    attachment_edges = [
        e for e in attachment_edges
        if int(e["parent_id"]) in graph_eligible_ids and int(e["child_id"]) in graph_eligible_ids
    ]
    movement_objects = [
        o for o in all_objects_for_graph
        if int(o["id"]) in graph_eligible_ids
    ]
    movement_object_map = {int(o["id"]): o for o in movement_objects}

    logger.info(
        "Attachment filter stats: edges input=%d visible=%d nonexcluded=%d final=%d, graph_objects=%d, question_objects=%d, referable_question_objects=%d",
        attachment_edge_input,
        attachment_edge_count_visible,
        attachment_edge_count_nonexcluded,
        len(attachment_edges),
        len(all_objects_for_graph),
        len(objects_for_questions),
        len(objects_uniq),
    )

    all_questions: list[dict] = []

    # Per-frame caps — keep the benchmark tractable when scenes have many objects
    MAX_L1_DIRECTION = 20
    MAX_L1_DIRECTION_OC = 15   # object-centric
    MAX_L1_DIRECTION_ALLO = 15 # allocentric
    MAX_L1_DISTANCE = 20

    # Ordinary L1 relations ignore depth in normal generation.
    relations = compute_all_relations(objects_uniq, camera_pose, None, None)

    # L1 – collect separately so we can sample before adding
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

    l1_occlusion_subjects = l1_occlusion_objects if label_counts else objects_uniq
    l1_occ_qs = generate_l1_occlusion_questions(
        objects=l1_occlusion_subjects,
        camera_pose=camera_pose,
        color_intrinsics=color_intrinsics,
        depth_image=depth_image,
        depth_intrinsics=depth_intrinsics,
        occlusion_backend=occlusion_backend,
        ray_caster=ray_caster,
        instance_mesh_data=instance_mesh_data,
        templates=templates,
        label_counts=label_counts,
        referable_object_ids=referable_object_ids,
        frame_image=frame_image,
        occlusion_vlm_adjudicator=occlusion_vlm_adjudicator,
    )

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
    # Rebuild the movement graph for question-eligible objects plus any
    # attachment-context ancestors kept to preserve attachment roots.
    attachment_graph_uniq = {
        k: [c for c in v if c in graph_eligible_ids]
        for k, v in attachment_graph.items()
        if k in graph_eligible_ids
    }
    attached_by_uniq = {
        k: v for k, v in attached_by.items()
        if k in graph_eligible_ids and v in graph_eligible_ids
    }
    support_chain_graph_uniq = {
        k: filtered_children
        for k, v in support_chain_graph.items()
        if k in referable_question_ids
        for filtered_children in ([c for c in v if c in referable_question_ids],)
        if filtered_children
    }
    support_chain_by_uniq = {
        k: v for k, v in support_chain_by.items()
        if k in referable_question_ids and v in referable_question_ids
    }

    # L2 — ego-centric (existing)
    all_questions.extend(
        generate_l2_object_move(
            objects_uniq,
            attachment_graph_uniq,
            attached_by_uniq,
            camera_pose,
            templates,
            room_bounds=room_bounds,
            collision_objects=l2_collision_objects,
            movement_objects=movement_objects,
            object_map=movement_object_map,
        )
    )
    all_questions.extend(
        generate_l2_viewpoint_move(
            objects_uniq,
            camera_pose,
            color_intrinsics,
            depth_image,
            depth_intrinsics,
            occlusion_backend,
            ray_caster,
            instance_mesh_data,
            templates,
        )
    )
    all_questions.extend(
        generate_l2_object_remove(
            objects_uniq,
            attachment_graph_uniq,
            camera_pose,
            color_intrinsics,
            depth_image,
            depth_intrinsics,
            occlusion_backend,
            ray_caster,
            instance_mesh_data,
            templates,
        )
    )
    # L2 — new reference frames
    all_questions.extend(
        generate_l2_object_move_object_centric(
            objects_uniq, attachment_graph_uniq, attached_by_uniq, camera_pose, templates,
            room_bounds=room_bounds,
            collision_objects=l2_collision_objects,
            movement_objects=movement_objects,
            object_map=movement_object_map,
        )
    )
    all_questions.extend(
        generate_l2_object_move_allocentric(
            objects_uniq, attachment_graph_uniq, attached_by_uniq, camera_pose, templates,
            room_bounds=room_bounds,
            collision_objects=l2_collision_objects,
            movement_objects=movement_objects,
            object_map=movement_object_map,
        )
    )
    # L3
    all_questions.extend(
        generate_l3_attachment_chain(
            objects_uniq,
            support_chain_graph_uniq,
            support_chain_by_uniq,
            camera_pose,
            templates,
        )
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

    all_questions = _enforce_stable_facing_references(
        all_questions, id_to_object,
    )
    all_questions = _balance_l2_object_move_attachment_counts(all_questions)

    logger.info("Generated %d questions total", len(all_questions))
    return all_questions
