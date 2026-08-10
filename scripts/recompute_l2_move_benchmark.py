#!/usr/bin/env python3
"""Repair existing L2 object-movement questions under movement semantics v2.

The input benchmark is treated as immutable.  Existing scene metadata is
preferred over reparsing raw scans because it contains the attachment graph
that was actually used when the questions were generated.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import logging
import math
from pathlib import Path
import random
import re
import sys
from typing import Any, Iterable, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets import make_data_source
from src.qa_generator import (
    ALL_DIRECTIONS,
    ALL_DIRECTIONS_ALLOCENTRIC,
    ALL_DISTANCES,
    L2_OBJECT_MOVE_SEMANTICS_VERSION,
    MIN_DISTANCE_QUESTION_DISTANCE_M,
    _allocentric_ground_move_directions,
    _annotate_attachment_trace_reason,
    _camera_ground_move_directions,
    _default_templates,
    _delta_to_cardinal_description,
    _delta_to_description,
    _delta_to_object_facing_description,
    _direction_with_camera_hint,
    _load_templates,
    _movement_metadata,
    _object_bottom_hull_xy,
    _object_camera_ground_move_directions,
    _object_facing_ground_axes,
    _object_pair_ground_move_directions,
    build_multi_frame_split_note,
    enrich_objects_with_distance_geometry,
    generate_options,
)
from src.quality_control import balance_l2_attachment_per_scene, compute_statistics
from src.relation_engine import (
    CARDINAL_DIRECTIONS_8,
    HORIZONTAL_DIRECTIONS,
    compute_distance_details,
    compute_pairwise_direction,
    primary_direction_allocentric,
    primary_direction_object_centric,
)
from src.scene_parser import load_instance_mesh_data, parse_scene
from src.support_graph import enrich_scene_with_attachment, get_scene_attachment_graph
from src.utils.colmap_loader import CameraPose
from src.virtual_ops import (
    apply_movement_selective,
    compute_room_bounds,
    get_moved_object_ids,
    has_terminal_bbox_collision,
    is_within_room,
)


LOGGER = logging.getLogger("recompute_l2_move_benchmark")

TARGET_TYPES = frozenset({
    "object_move_agent",
    "object_move_distance",
    "object_move_object_centric",
    "object_move_allocentric",
})
MOVE_MAGNITUDES_M = (3.0, 2.5, 2.0, 1.5, 1.0, 0.5)
AMBIGUITY_LIMIT = 0.7
OVERALL_RETENTION_THRESHOLD = 0.80
TYPE_RETENTION_THRESHOLD = 0.60
SCANNET_SCENE_RE = re.compile(r"^scene\d{4}_\d{2}$")
DISTANCE_RE = re.compile(r"(?<![\d.])(\d+(?:\.\d+)?)m\b", re.IGNORECASE)
LEGACY_OBJECT_CENTRIC_MOVE_RE = re.compile(
    r"(?:shifted|moved)\s+"
    r"(forward-right|forward-left|backward-right|backward-left|"
    r"forward|backward|left|right)\s+by\s+"
    r"(\d+(?:\.\d+)?)m\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class CandidateSpec:
    direction: str
    distance_m: float
    delta: np.ndarray
    phase: str
    rank: int


@dataclass
class CandidateEvaluation:
    valid: bool
    reason: str | None = None
    new_value: str | None = None
    ambiguity: float | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class MotionState:
    valid: bool
    reason: str | None
    moved_map: dict[int, dict[str, Any]] | None
    moved_ids: set[int]


@dataclass
class SceneResources:
    scene_id: str
    dataset: str
    scene_dir: Path | None
    metadata_path: Path | None
    objects: list[dict[str, Any]]
    attachment_graph: dict[int, list[int]]
    room_bounds: dict[str, Any] | None
    poses: dict[str, CameraPose]
    distance_geometry: str
    motion_cache: dict[tuple[int, tuple[float, float, float]], MotionState] = field(
        default_factory=dict
    )

    @property
    def objects_by_id(self) -> dict[int, dict[str, Any]]:
        return {int(obj["id"]): obj for obj in self.objects}


def detect_dataset(scene_id: str) -> str:
    """Infer the dataset from its stable scene-ID convention."""
    return "scannet" if SCANNET_SCENE_RE.fullmatch(scene_id) else "scannetpp"


def _json_load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _normalise_attachment_graph(raw: Any) -> dict[int, list[int]]:
    if not isinstance(raw, dict):
        return {}
    graph: dict[int, list[int]] = {}
    for parent_id, child_ids in raw.items():
        try:
            parent = int(parent_id)
        except (TypeError, ValueError):
            continue
        if not isinstance(child_ids, (list, tuple, set)):
            continue
        graph[parent] = []
        for child_id in child_ids:
            try:
                graph[parent].append(int(child_id))
            except (TypeError, ValueError):
                continue
    return graph


def _metadata_index(roots: Sequence[Path]) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = defaultdict(list)
    visited: set[Path] = set()
    for root in roots:
        root = root.resolve()
        if root in visited or not root.exists():
            continue
        visited.add(root)
        if root.is_file() and root.suffix.lower() == ".json":
            candidates = [root]
        else:
            candidates = [
                path
                for path in root.rglob("*.json")
                if path.parent.name == "scene_metadata"
            ]
        for path in candidates:
            index[path.stem].append(path)
    for paths in index.values():
        paths.sort(key=lambda path: str(path))
    return dict(index)


def _select_metadata_path(
    scene_id: str,
    index: dict[str, list[Path]],
) -> Path | None:
    paths = index.get(scene_id, [])
    if not paths:
        return None
    if len(paths) > 1:
        LOGGER.warning(
            "Multiple scene metadata files found for %s; using %s",
            scene_id,
            paths[0],
        )
    return paths[0]


def _scene_dir_for(
    scene_id: str,
    dataset: str,
    *,
    scannet_root: Path | None,
    scannetpp_root: Path | None,
) -> Path:
    root = scannet_root if dataset == "scannet" else scannetpp_root
    if root is None:
        flag = "--scannet-root" if dataset == "scannet" else "--scannetpp-root"
        raise RuntimeError(f"{flag} is required to load camera poses for {scene_id}")
    scene_dir = root / scene_id
    if not scene_dir.is_dir():
        raise RuntimeError(f"scene directory does not exist: {scene_dir}")
    return scene_dir


def _load_scene_resources(
    scene_id: str,
    *,
    metadata_index: dict[str, list[Path]],
    scannet_root: Path | None,
    scannetpp_root: Path | None,
    distance_geometry: str,
    needs_distance: bool,
    needs_camera_pose: bool = True,
) -> SceneResources:
    dataset = detect_dataset(scene_id)
    metadata_path = _select_metadata_path(scene_id, metadata_index)
    needs_raw_scene = (
        needs_camera_pose
        or metadata_path is None
        or (needs_distance and distance_geometry == "mesh")
    )
    scene_dir = (
        _scene_dir_for(
            scene_id,
            dataset,
            scannet_root=scannet_root,
            scannetpp_root=scannetpp_root,
        )
        if needs_raw_scene
        else None
    )
    if metadata_path is not None:
        scene = _json_load(metadata_path)
    else:
        if scene_dir is None:
            raise RuntimeError(f"scene directory is required to parse {scene_id}")
        parse_kwargs: dict[str, Any] = {}
        if dataset == "scannetpp":
            parse_kwargs["dataset"] = "scannetpp"
        scene = parse_scene(scene_dir, **parse_kwargs)
        if scene is None:
            raise RuntimeError(f"failed to parse scene geometry for {scene_id}")
        enrich_scene_with_attachment(scene)

    objects = scene.get("objects") if isinstance(scene, dict) else None
    if not isinstance(objects, list) or not objects:
        raise RuntimeError(f"scene {scene_id} has no object metadata")
    objects = copy.deepcopy(objects)
    graph = _normalise_attachment_graph(scene.get("attachment_graph"))
    if not graph:
        graph = _normalise_attachment_graph(
            get_scene_attachment_graph(scene, scene_id=scene_id)
        )

    poses: dict[str, CameraPose] = {}
    if needs_camera_pose:
        if scene_dir is None:
            raise RuntimeError(f"scene directory is required to load poses for {scene_id}")
        data_source = make_data_source(dataset, scene_dir, sensor="iphone")
        poses = data_source.load_poses()
        if not poses:
            raise RuntimeError(f"scene {scene_id} has no camera poses")

    geometry_used = "not_needed"
    if needs_distance:
        if distance_geometry == "mesh":
            if scene_dir is None:
                raise RuntimeError(
                    f"scene directory is required to load distance geometry for {scene_id}"
                )
            kwargs: dict[str, Any] = {
                "instance_ids": [int(obj["id"]) for obj in objects],
                "n_surface_samples": 512,
            }
            if dataset == "scannetpp":
                kwargs["dataset"] = "scannetpp"
            instance_mesh_data = load_instance_mesh_data(scene_dir, **kwargs)
            enrich_objects_with_distance_geometry(objects, instance_mesh_data)
            geometry_used = "surface_sample_closest_point"
        else:
            enrich_objects_with_distance_geometry(objects, None)
            geometry_used = "aabb_closest_point_approx"

    return SceneResources(
        scene_id=scene_id,
        dataset=dataset,
        scene_dir=scene_dir,
        metadata_path=metadata_path,
        objects=objects,
        attachment_graph=graph,
        room_bounds=scene.get("room_bounds"),
        poses=poses,
        distance_geometry=geometry_used,
    )


def _pose_for_question(
    question: dict[str, Any],
    resources: SceneResources,
) -> CameraPose:
    image_name = str(question.get("image_name", "")).strip()
    pose = resources.poses.get(image_name)
    if pose is not None:
        return pose
    basename = Path(image_name).name
    matches = [value for key, value in resources.poses.items() if Path(key).name == basename]
    if len(matches) == 1:
        return matches[0]
    raise KeyError(f"camera pose {image_name!r} not found in scene {resources.scene_id}")


def recover_legacy_movement(
    question: dict[str, Any],
    camera_pose: CameraPose,
    objects_by_id: dict[int, dict[str, Any]],
) -> tuple[str, float]:
    """Recover the word emitted by the pre-v2 generator from its old delta."""
    delta = np.asarray(question.get("delta", []), dtype=np.float64)
    if delta.shape != (3,) or not np.all(np.isfinite(delta)):
        raise ValueError("question has an invalid legacy delta")
    distance_m = float(np.linalg.norm(delta))
    qtype = str(question.get("type", ""))
    if qtype in {"object_move_agent", "object_move_distance"}:
        direction = _delta_to_description(delta, camera_pose)
    elif qtype == "object_move_allocentric":
        direction = _delta_to_cardinal_description(delta)
    elif qtype == "object_move_object_centric":
        # Legacy object-centric deltas were sometimes stored in the wrong
        # world direction. The visible action text is the source of intent.
        return recover_legacy_object_centric_movement_from_text(question)
    else:
        raise ValueError(f"unsupported target type: {qtype}")
    return str(direction), distance_m


def recover_legacy_object_centric_movement_from_text(
    question: dict[str, Any],
) -> tuple[str, float]:
    """Recover a legacy object-centric action without requiring a camera pose."""
    match = LEGACY_OBJECT_CENTRIC_MOVE_RE.search(str(question.get("question", "")))
    if match is None:
        raise ValueError("legacy object-centric movement text is not parseable")
    direction = match.group(1).lower()
    distance_m = float(match.group(2))
    return direction, distance_m


def cross_check_legacy_text(
    question: dict[str, Any],
    direction: str,
    distance_m: float,
) -> tuple[bool, list[str]]:
    """Verify that recovered legacy movement agrees with the visible prompt."""
    text = str(question.get("question", ""))
    text_lower = text.lower()
    qtype = str(question.get("type", ""))
    reasons: list[str] = []

    if qtype in {"object_move_agent", "object_move_distance"}:
        direction_phrase = f"{_direction_with_camera_hint(direction).lower()} by"
        direction_ok = direction_phrase in text_lower
    elif qtype == "object_move_allocentric":
        direction_ok = f"to the {direction.lower()}" in text_lower
    else:
        # Object-centric prompts may insert the moved-object label between the
        # verb and action ("moving the sofa forward by 1.0m"). The action
        # token immediately followed by "by" is the stable visible contract.
        direction_ok = re.search(
            rf"\b{re.escape(direction.lower())}\s+by\b",
            text_lower,
        )
        direction_ok = direction_ok is not None
    if not direction_ok:
        reasons.append("legacy_direction_text_mismatch")

    text_distances = [float(match.group(1)) for match in DISTANCE_RE.finditer(text)]
    expected_rounded = round(float(distance_m), 1)
    if not any(math.isclose(value, expected_rounded, abs_tol=1e-6) for value in text_distances):
        reasons.append("legacy_distance_text_mismatch")
    return not reasons, reasons


def _snap_magnitude(distance_m: float) -> float:
    closest = min(MOVE_MAGNITUDES_M, key=lambda value: abs(value - distance_m))
    if not math.isclose(float(distance_m), closest, abs_tol=2e-3, rel_tol=0.0):
        raise ValueError(
            f"legacy movement distance {distance_m:.6f}m is not in the discrete action set"
        )
    return float(closest)


def build_candidate_schedule(
    directions: Sequence[tuple[str, np.ndarray]],
    legacy_direction: str,
    legacy_distance_m: float,
) -> list[CandidateSpec]:
    """Build the locked repair order while deduplicating repeated candidates."""
    direction_map = {
        str(label): np.asarray(unit, dtype=np.float64)
        for label, unit in directions
    }
    if legacy_direction not in direction_map:
        raise ValueError(
            f"legacy direction {legacy_direction!r} is unavailable in the strict action frame"
        )
    original_distance = _snap_magnitude(legacy_distance_m)
    entries: list[tuple[str, float, str]] = []
    entries.append((legacy_direction, original_distance, "original_direction_original_distance"))
    for magnitude in sorted(
        (value for value in MOVE_MAGNITUDES_M if value > original_distance)
    ):
        entries.append((legacy_direction, magnitude, "same_direction_farther"))
    for magnitude in sorted(
        (value for value in MOVE_MAGNITUDES_M if value < original_distance),
        reverse=True,
    ):
        entries.append((legacy_direction, magnitude, "same_direction_nearer"))
    for label, _unit in directions:
        if label != legacy_direction:
            entries.append((str(label), original_distance, "other_direction_original_distance"))
    for magnitude in MOVE_MAGNITUDES_M:
        for label, _unit in directions:
            entries.append((str(label), float(magnitude), "remaining_candidates"))

    result: list[CandidateSpec] = []
    seen: set[tuple[str, float]] = set()
    for label, magnitude, phase in entries:
        key = (label, float(magnitude))
        if key in seen:
            continue
        seen.add(key)
        delta = np.round(direction_map[label] * magnitude, 6).astype(np.float64)
        result.append(CandidateSpec(label, magnitude, delta, phase, len(result)))
    return result


def strict_directions_for_question(
    question: dict[str, Any],
    camera_pose: CameraPose | None,
    objects_by_id: dict[int, dict[str, Any]],
) -> tuple[tuple[str, np.ndarray], ...]:
    qtype = str(question.get("type", ""))
    if qtype in {"object_move_agent", "object_move_distance"}:
        if camera_pose is None:
            raise ValueError(f"camera pose is required for {qtype}")
        return _camera_ground_move_directions(camera_pose)
    if qtype == "object_move_allocentric":
        return _allocentric_ground_move_directions()
    if qtype == "object_move_object_centric":
        if camera_pose is None:
            raise ValueError("camera pose is required for object_move_object_centric")
        moved = objects_by_id[int(question["moved_obj_id"])]
        return _object_camera_ground_move_directions(
            np.asarray(moved["center"], dtype=np.float64),
            np.asarray(camera_pose.position, dtype=np.float64),
        )
    raise ValueError(f"unsupported target type: {qtype}")


def _motion_key(moved_obj_id: int, delta: np.ndarray) -> tuple[int, tuple[float, float, float]]:
    return (
        int(moved_obj_id),
        tuple(float(value) for value in np.round(np.asarray(delta), 6)),
    )


def _motion_state(
    resources: SceneResources,
    moved_obj_id: int,
    delta: np.ndarray,
) -> MotionState:
    key = _motion_key(moved_obj_id, delta)
    cached = resources.motion_cache.get(key)
    if cached is not None:
        return cached

    moved_ids = get_moved_object_ids(moved_obj_id, resources.attachment_graph)
    moved_objects = apply_movement_selective(
        resources.objects,
        resources.attachment_graph,
        moved_obj_id,
        delta,
    )
    room_min, room_max = compute_room_bounds(
        resources.objects,
        room_bounds=resources.room_bounds,
    )
    if not is_within_room(moved_objects, room_min, room_max):
        state = MotionState(False, "outside_room", None, moved_ids)
    elif has_terminal_bbox_collision(
        resources.objects,
        moved_objects,
        moved_ids,
        collision_objects=resources.objects,
    ):
        state = MotionState(False, "terminal_collision", None, moved_ids)
    else:
        state = MotionState(
            True,
            None,
            {int(obj["id"]): obj for obj in moved_objects},
            moved_ids,
        )
    resources.motion_cache[key] = state
    return state


def _object_centric_direction(
    query_obj: dict[str, Any],
    reference_obj: dict[str, Any],
    facing_offset: np.ndarray,
) -> tuple[str, float]:
    query_center = np.asarray(query_obj["center"], dtype=np.float64)
    ref_center = np.asarray(reference_obj["center"], dtype=np.float64)
    return primary_direction_object_centric(
        query_center,
        query_center + facing_offset,
        ref_center,
        horizontal_only=True,
        anchor_hull_xy=_object_bottom_hull_xy(query_obj),
        target_hull_xy=_object_bottom_hull_xy(reference_obj),
        anchor_bbox_min=np.asarray(query_obj["bbox_min"], dtype=np.float64),
        anchor_bbox_max=np.asarray(query_obj["bbox_max"], dtype=np.float64),
        target_bbox_min=np.asarray(reference_obj["bbox_min"], dtype=np.float64),
        target_bbox_max=np.asarray(reference_obj["bbox_max"], dtype=np.float64),
    )


def _allocentric_direction(
    query_obj: dict[str, Any],
    reference_obj: dict[str, Any],
) -> tuple[str, float]:
    return primary_direction_allocentric(
        np.asarray(query_obj["center"], dtype=np.float64),
        np.asarray(reference_obj["center"], dtype=np.float64),
        obj_a_hull_xy=_object_bottom_hull_xy(query_obj),
        obj_b_hull_xy=_object_bottom_hull_xy(reference_obj),
        obj_a_bbox_min=np.asarray(query_obj["bbox_min"], dtype=np.float64),
        obj_a_bbox_max=np.asarray(query_obj["bbox_max"], dtype=np.float64),
        obj_b_bbox_min=np.asarray(reference_obj["bbox_min"], dtype=np.float64),
        obj_b_bbox_max=np.asarray(reference_obj["bbox_max"], dtype=np.float64),
    )


def recompute_baseline(
    question: dict[str, Any],
    camera_pose: CameraPose | None,
    objects_by_id: dict[int, dict[str, Any]],
) -> CandidateEvaluation:
    qtype = str(question["type"])
    try:
        if qtype == "object_move_agent":
            if camera_pose is None:
                return CandidateEvaluation(False, "missing_camera_pose")
            obj_b = objects_by_id[int(question["obj_b_id"])]
            obj_c = objects_by_id[int(question["obj_c_id"])]
            value, ambiguity = compute_pairwise_direction(obj_c, obj_b, camera_pose)
            if ambiguity > AMBIGUITY_LIMIT:
                return CandidateEvaluation(False, "baseline_ambiguous", value, ambiguity)
            return CandidateEvaluation(True, new_value=value, ambiguity=ambiguity)

        if qtype == "object_move_distance":
            obj_b = objects_by_id[int(question["obj_b_id"])]
            obj_c = objects_by_id[int(question["obj_c_id"])]
            details = compute_distance_details(obj_b, obj_c)
            if float(details["distance_m"]) < MIN_DISTANCE_QUESTION_DISTANCE_M:
                return CandidateEvaluation(False, "baseline_distance_too_small", details=details)
            if bool(details["near_boundary"]):
                return CandidateEvaluation(False, "baseline_distance_near_boundary", details=details)
            return CandidateEvaluation(
                True,
                new_value=str(details["distance_bin"]),
                details=details,
            )

        if qtype == "object_move_object_centric":
            if camera_pose is None:
                return CandidateEvaluation(False, "missing_camera_pose")
            query = objects_by_id[int(question["query_obj_id"])]
            reference = objects_by_id[int(question["obj_ref_id"])]
            facing_offset = (
                np.asarray(camera_pose.position, dtype=np.float64)
                - np.asarray(query["center"], dtype=np.float64)
            )
            value, ambiguity = _object_centric_direction(query, reference, facing_offset)
            if value not in HORIZONTAL_DIRECTIONS:
                return CandidateEvaluation(False, "baseline_non_horizontal", value, ambiguity)
            if ambiguity > AMBIGUITY_LIMIT:
                return CandidateEvaluation(False, "baseline_ambiguous", value, ambiguity)
            return CandidateEvaluation(True, new_value=value, ambiguity=ambiguity)

        if qtype == "object_move_allocentric":
            query = objects_by_id[int(question["query_obj_id"])]
            reference = objects_by_id[int(question["obj_ref_id"])]
            value, ambiguity = _allocentric_direction(query, reference)
            if value not in CARDINAL_DIRECTIONS_8:
                return CandidateEvaluation(False, "baseline_non_cardinal", value, ambiguity)
            if ambiguity > AMBIGUITY_LIMIT:
                return CandidateEvaluation(False, "baseline_ambiguous", value, ambiguity)
            return CandidateEvaluation(True, new_value=value, ambiguity=ambiguity)
    except (KeyError, TypeError, ValueError) as exc:
        return CandidateEvaluation(False, f"baseline_error:{exc}")
    return CandidateEvaluation(False, "unsupported_type")


def evaluate_candidate(
    question: dict[str, Any],
    candidate: CandidateSpec,
    baseline: CandidateEvaluation,
    camera_pose: CameraPose | None,
    resources: SceneResources,
) -> CandidateEvaluation:
    moved_obj_id = int(question["moved_obj_id"])
    motion = _motion_state(resources, moved_obj_id, candidate.delta)
    if not motion.valid or motion.moved_map is None:
        return CandidateEvaluation(False, motion.reason)
    moved_map = motion.moved_map
    qtype = str(question["type"])
    try:
        if qtype == "object_move_agent":
            if camera_pose is None:
                return CandidateEvaluation(False, "missing_camera_pose")
            obj_b = moved_map[int(question["obj_b_id"])]
            obj_c = moved_map[int(question["obj_c_id"])]
            value, ambiguity = compute_pairwise_direction(obj_c, obj_b, camera_pose)
            if ambiguity > AMBIGUITY_LIMIT:
                return CandidateEvaluation(False, "answer_ambiguous", value, ambiguity)
            return CandidateEvaluation(True, new_value=value, ambiguity=ambiguity)

        if qtype == "object_move_distance":
            obj_b = moved_map[int(question["obj_b_id"])]
            obj_c = moved_map[int(question["obj_c_id"])]
            details = compute_distance_details(obj_b, obj_c)
            if bool(details["near_boundary"]):
                return CandidateEvaluation(False, "distance_near_boundary", details=details)
            if (
                str(details["distance_bin"]) == baseline.new_value
                and float(details["distance_m"]) < MIN_DISTANCE_QUESTION_DISTANCE_M
            ):
                return CandidateEvaluation(False, "unchanged_distance_too_small", details=details)
            return CandidateEvaluation(
                True,
                new_value=str(details["distance_bin"]),
                details=details,
            )

        if qtype == "object_move_object_centric":
            if camera_pose is None:
                return CandidateEvaluation(False, "missing_camera_pose")
            original_map = resources.objects_by_id
            original_query = original_map[int(question["query_obj_id"])]
            facing_offset = (
                np.asarray(camera_pose.position, dtype=np.float64)
                - np.asarray(original_query["center"], dtype=np.float64)
            )
            value, ambiguity = _object_centric_direction(
                moved_map[int(question["query_obj_id"])],
                moved_map[int(question["obj_ref_id"])],
                facing_offset,
            )
            if value not in HORIZONTAL_DIRECTIONS:
                return CandidateEvaluation(False, "answer_non_horizontal", value, ambiguity)
            if ambiguity > AMBIGUITY_LIMIT:
                return CandidateEvaluation(False, "answer_ambiguous", value, ambiguity)
            return CandidateEvaluation(True, new_value=value, ambiguity=ambiguity)

        if qtype == "object_move_allocentric":
            value, ambiguity = _allocentric_direction(
                moved_map[int(question["query_obj_id"])],
                moved_map[int(question["obj_ref_id"])],
            )
            if value not in CARDINAL_DIRECTIONS_8:
                return CandidateEvaluation(False, "answer_non_cardinal", value, ambiguity)
            if ambiguity > AMBIGUITY_LIMIT:
                return CandidateEvaluation(False, "answer_ambiguous", value, ambiguity)
            return CandidateEvaluation(True, new_value=value, ambiguity=ambiguity)
    except (KeyError, TypeError, ValueError) as exc:
        return CandidateEvaluation(False, f"candidate_error:{exc}")
    return CandidateEvaluation(False, "unsupported_type")


def choose_candidate(
    question: dict[str, Any],
    schedule: Sequence[CandidateSpec],
    baseline: CandidateEvaluation,
    camera_pose: CameraPose | None,
    resources: SceneResources,
) -> tuple[CandidateSpec | None, CandidateEvaluation | None, Counter[str]]:
    """Prefer the first changed answer; retain the first legal unchanged fallback."""
    fallback: tuple[CandidateSpec, CandidateEvaluation] | None = None
    rejection_counts: Counter[str] = Counter()
    for candidate in schedule:
        evaluation = evaluate_candidate(
            question,
            candidate,
            baseline,
            camera_pose,
            resources,
        )
        if not evaluation.valid:
            rejection_counts[str(evaluation.reason or "invalid_candidate")] += 1
            continue
        if evaluation.new_value != baseline.new_value:
            return candidate, evaluation, rejection_counts
        if fallback is None:
            fallback = (candidate, evaluation)
    if fallback is not None:
        return fallback[0], fallback[1], rejection_counts
    return None, None, rejection_counts


def _stable_seed(question: dict[str, Any], source_index: int, purpose: str) -> int:
    payload = "|".join([
        purpose,
        str(source_index),
        str(question.get("scene_id", "")),
        str(question.get("type", "")),
        str(question.get("image_name", "")),
        str(question.get("reasoning_frame_2", "")),
        str(question.get("moved_obj_id", "")),
        str(question.get("query_obj_id", "")),
        str(question.get("obj_c_id", question.get("obj_ref_id", ""))),
    ])
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big")


def _deterministic_options(
    correct_value: str,
    answer_pool: list[str],
    seed: int,
) -> tuple[list[str], str]:
    state = random.getstate()
    try:
        random.seed(seed)
        return generate_options(correct_value, answer_pool)
    finally:
        random.setstate(state)


def _answer_pool(qtype: str) -> list[str]:
    if qtype == "object_move_agent":
        return list(ALL_DIRECTIONS)
    if qtype == "object_move_distance":
        return list(ALL_DISTANCES)
    if qtype == "object_move_object_centric":
        return list(HORIZONTAL_DIRECTIONS)
    if qtype == "object_move_allocentric":
        return list(ALL_DIRECTIONS_ALLOCENTRIC)
    raise ValueError(f"unsupported target type: {qtype}")


def _update_options(
    repaired: dict[str, Any],
    correct_value: str,
    source_index: int,
) -> bool:
    options = repaired.get("options")
    if isinstance(options, list) and correct_value in options:
        repaired["answer"] = chr(65 + options.index(correct_value))
        return True
    new_options, answer = _deterministic_options(
        correct_value,
        _answer_pool(str(repaired["type"])),
        _stable_seed(repaired, source_index, "options"),
    )
    repaired["options"] = new_options
    repaired["answer"] = answer
    return False


def _template_for(
    qtype: str,
    templates: dict[str, Any],
    question: dict[str, Any],
    source_index: int,
    *,
    object_centric_template: str = "configured",
) -> str:
    key = {
        "object_move_agent": "L2_object_move_agent",
        "object_move_distance": "L2_object_move_distance",
        "object_move_object_centric": "L2_object_move_object_centric",
        "object_move_allocentric": "L2_object_move_allocentric",
    }[qtype]
    candidates = templates.get(key) or _default_templates()[key]
    if not isinstance(candidates, list) or not candidates:
        raise ValueError(f"template list {key} is empty")
    if qtype == "object_move_object_centric" and object_centric_template == "freeze":
        freeze_templates = [
            candidate
            for candidate in candidates
            if "Freeze both objects' initial horizontal forward/right axes" in str(candidate)
        ]
        if len(freeze_templates) != 1:
            raise ValueError(
                "expected exactly one canonical strict object-centric template"
            )
        return str(freeze_templates[0])
    if object_centric_template not in {"configured", "freeze"}:
        raise ValueError(
            f"unsupported object-centric template mode: {object_centric_template}"
        )
    return str(candidates[_stable_seed(question, source_index, "template") % len(candidates)])


def _render_question_text(
    repaired: dict[str, Any],
    candidate: CandidateSpec,
    resources: SceneResources,
    templates: dict[str, Any],
    source_index: int,
    *,
    object_centric_template: str = "configured",
) -> str:
    qtype = str(repaired["type"])
    template = _template_for(
        qtype,
        templates,
        repaired,
        source_index,
        object_centric_template=object_centric_template,
    )
    distance = f"{candidate.distance_m:.1f}m"
    if qtype in {"object_move_agent", "object_move_distance"}:
        base = template.format(
            obj_a=f"the {repaired['moved_obj_label']}",
            direction=candidate.direction,
            direction_with_camera_hint=_direction_with_camera_hint(candidate.direction),
            distance=distance,
            obj_b=f"the {repaired['obj_b_label']}",
            obj_c=f"the {repaired['obj_c_label']}",
        )
    elif qtype == "object_move_object_centric":
        base = template.format(
            obj_a=f"the {repaired['moved_obj_label']}",
            obj_move_source=f"the {repaired['moved_obj_label']}",
            obj_query=f"the {repaired['query_obj_label']}",
            obj_ref=f"the {repaired['obj_ref_label']}",
            direction=candidate.direction,
            direction_with_camera_hint=_direction_with_camera_hint(candidate.direction),
            distance=distance,
        )
    else:
        base = template.format(
            obj_move_source=f"the {repaired['moved_obj_label']}",
            obj_query=f"the {repaired['query_obj_label']}",
            obj_ref=f"the {repaired['obj_ref_label']}",
            direction=candidate.direction,
            distance=distance,
            camera_cardinal=repaired.get("camera_cardinal", "unknown"),
        )

    groups = repaired.get("object_frame_groups")
    if not isinstance(groups, dict):
        return base
    objects_by_id = resources.objects_by_id
    try:
        group_a = [objects_by_id[int(obj_id)] for obj_id in groups["frame_1"]]
        group_b = [objects_by_id[int(obj_id)] for obj_id in groups["frame_2"]]
    except (KeyError, TypeError, ValueError):
        return base
    return f"{build_multi_frame_split_note(group_a, group_b)} {base}"


def build_repaired_question(
    question: dict[str, Any],
    source_index: int,
    candidate: CandidateSpec,
    baseline: CandidateEvaluation,
    selected: CandidateEvaluation,
    resources: SceneResources,
    camera_pose: CameraPose | None,
    templates: dict[str, Any],
    *,
    object_centric_template: str = "configured",
) -> tuple[dict[str, Any], bool]:
    repaired = copy.deepcopy(question)
    qtype = str(repaired["type"])
    new_value = str(selected.new_value)
    old_value = str(baseline.new_value)
    relation_unchanged = new_value == old_value

    repaired.update({
        "correct_value": new_value,
        "old_correct_value": old_value,
        "new_correct_value": new_value,
        "delta": [float(value) for value in candidate.delta.tolist()],
        "relation_unchanged": relation_unchanged,
    })
    repaired.update(_movement_metadata(
        direction=candidate.direction,
        delta=candidate.delta,
        reference_frame={
            "object_move_agent": "agent",
            "object_move_distance": "agent",
            "object_move_object_centric": "moved_object_facing_first_camera",
            "object_move_allocentric": "allocentric",
        }[qtype],
    ))
    if qtype in {"object_move_agent", "object_move_distance"}:
        repaired["movement_camera_binding"] = "frame_1"
    elif qtype == "object_move_object_centric":
        if camera_pose is None:
            raise ValueError("camera pose is required for object-centric frame metadata")
        camera_position = np.asarray(camera_pose.position, dtype=np.float64)
        moved_obj = resources.objects_by_id[int(repaired["moved_obj_id"])]
        query_obj = resources.objects_by_id[int(repaired["query_obj_id"])]
        movement_axes = _object_facing_ground_axes(
            np.asarray(moved_obj["center"], dtype=np.float64), camera_position
        )
        answer_axes = _object_facing_ground_axes(
            np.asarray(query_obj["center"], dtype=np.float64), camera_position
        )
        if movement_axes is None or answer_axes is None:
            raise ValueError("degenerate strict object-centric frame")
        movement_forward, movement_right = movement_axes
        answer_forward, answer_right = answer_axes
        repaired.pop("movement_frame_query_obj_id", None)
        repaired.pop("movement_frame_reference_obj_id", None)
        repaired.update({
            "movement_frame_anchor_obj_id": int(repaired["moved_obj_id"]),
            "movement_camera_binding": "frame_1",
            "movement_frame_forward_world": movement_forward.tolist(),
            "movement_frame_right_world": movement_right.tolist(),
            "movement_frame_frozen": True,
            "answer_reference_frame": "query_object_facing_first_camera",
            "answer_frame_anchor_obj_id": int(repaired["query_obj_id"]),
            "answer_camera_binding": "frame_1",
            "answer_frame_forward_world": answer_forward.tolist(),
            "answer_frame_right_world": answer_right.tolist(),
            "answer_frame_frozen": True,
        })
    else:
        repaired["movement_world_axes"] = "scannet_aligned_xy"

    if qtype == "object_move_distance":
        old_details = baseline.details
        new_details = selected.details
        repaired.update({
            "old_distance_m": float(old_details["distance_m"]),
            "new_distance_m": float(new_details["distance_m"]),
            "old_distance_bin_id": old_details.get("distance_bin_id"),
            "new_distance_bin_id": new_details.get("distance_bin_id"),
            "distance_definition": new_details.get("distance_definition"),
            "old_distance_definition": old_details.get("distance_definition"),
            "new_distance_definition": new_details.get("distance_definition"),
        })

    options_preserved = _update_options(repaired, new_value, source_index)
    repaired["question"] = _render_question_text(
        repaired,
        candidate,
        resources,
        templates,
        source_index,
        object_centric_template=object_centric_template,
    )
    repaired.pop("trace_reason", None)
    _annotate_attachment_trace_reason(repaired)
    return repaired, options_preserved


def normalize_v2_object_centric_template(
    question: dict[str, Any],
    source_index: int,
    resources: SceneResources | None,
    templates: dict[str, Any],
) -> dict[str, Any]:
    """Render an existing v2 object-centric question with canonical wording."""
    if str(question.get("type", "")) != "object_move_object_centric":
        return copy.deepcopy(question)
    if (
        question.get("movement_semantics_version")
        != L2_OBJECT_MOVE_SEMANTICS_VERSION
    ):
        raise ValueError("cannot normalize a non-v2 object-centric question")
    delta = np.asarray(question.get("delta", []), dtype=np.float64)
    if delta.shape != (3,) or not np.all(np.isfinite(delta)):
        raise ValueError("v2 object-centric question has an invalid delta")
    direction = str(question.get("movement_direction", "")).strip()
    if not direction:
        raise ValueError("v2 object-centric question has no movement direction")
    distance_m = float(
        question.get("movement_distance_m", float(np.linalg.norm(delta)))
    )
    normalized = copy.deepcopy(question)
    template = _template_for(
        "object_move_object_centric",
        templates,
        normalized,
        source_index,
        object_centric_template="freeze",
    )
    base = template.format(
        obj_a=f"the {normalized['moved_obj_label']}",
        obj_move_source=f"the {normalized['moved_obj_label']}",
        obj_query=f"the {normalized['query_obj_label']}",
        obj_ref=f"the {normalized['obj_ref_label']}",
        direction=direction,
        direction_with_camera_hint=_direction_with_camera_hint(direction),
        distance=f"{distance_m:.1f}m",
    )
    text = str(normalized.get("question", ""))
    starts = [
        position
        for marker in (
            "Use a fixed object-centric coordinate frame",
            "Initially, let the direction from",
            "In the initial scene, imagine",
            "Suppose the ",
        )
        if (position := text.find(marker)) >= 0
    ]
    if starts:
        normalized["question"] = f"{text[:min(starts)]}{base}"
    elif resources is not None:
        normalized["question"] = _render_question_text(
            normalized,
            CandidateSpec(direction, distance_m, delta, "template_normalization", 0),
            resources,
            templates,
            source_index,
            object_centric_template="freeze",
        )
    else:
        raise ValueError("cannot locate the object-centric template text")
    return normalized


def _role_key(question: dict[str, Any]) -> tuple[int, ...]:
    qtype = str(question.get("type", ""))
    fields = {
        "object_move_agent": ("moved_obj_id", "query_obj_id", "obj_b_id", "obj_c_id"),
        "object_move_distance": ("moved_obj_id", "query_obj_id", "obj_b_id", "obj_c_id"),
        "object_move_object_centric": ("moved_obj_id", "query_obj_id", "obj_ref_id"),
        "object_move_allocentric": ("moved_obj_id", "query_obj_id", "obj_ref_id"),
    }[qtype]
    return tuple(int(question[field]) for field in fields)


def repaired_dedup_key(question: dict[str, Any]) -> tuple[Any, ...]:
    delta = tuple(float(value) for value in np.round(question.get("delta", []), 6))
    return (
        str(question.get("scene_id", "")),
        str(question.get("type", "")),
        str(question.get("image_name", "")),
        str(question.get("reasoning_frame_2", "")),
        _role_key(question),
        delta,
    )


def _audit_base(index: int, question: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_index": index,
        "scene_id": question.get("scene_id"),
        "type": question.get("type"),
        "image_name": question.get("image_name"),
        "reasoning_frame_2": question.get("reasoning_frame_2"),
        "moved_obj_id": question.get("moved_obj_id"),
        "query_obj_id": question.get("query_obj_id"),
        "stored_delta": question.get("delta"),
        "stored_old_correct_value": question.get("old_correct_value"),
        "stored_correct_value": question.get("correct_value"),
    }


def _repair_one(
    index: int,
    question: dict[str, Any],
    resources: SceneResources,
    templates: dict[str, Any],
    *,
    recover_object_centric_from_text: bool = False,
    object_centric_template: str = "configured",
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    audit = _audit_base(index, question)
    objects_by_id = resources.objects_by_id
    try:
        qtype = str(question.get("type", ""))
        pose = _pose_for_question(question, resources)
        if qtype == "object_move_object_centric":
            structured_direction = str(question.get("movement_direction") or "").strip()
            structured_distance = question.get("movement_distance_m")
            if structured_direction and structured_distance is not None:
                legacy_direction = structured_direction
                legacy_distance = float(structured_distance)
            else:
                legacy_direction, legacy_distance = (
                    recover_legacy_object_centric_movement_from_text(question)
                )
        else:
            legacy_direction, legacy_distance = recover_legacy_movement(
                question,
                pose,
                objects_by_id,
            )
        text_ok, text_reasons = cross_check_legacy_text(
            question,
            legacy_direction,
            legacy_distance,
        )
        audit["legacy_recovery"] = {
            "direction": legacy_direction,
            "distance_m": legacy_distance,
            "text_cross_check": "pass" if text_ok else "fail",
            "text_cross_check_reasons": text_reasons,
        }
        if not text_ok:
            audit.update(status="dropped", reason="legacy_text_mismatch")
            return None, audit

        moved_id = int(question["moved_obj_id"])
        query_id = int(question["query_obj_id"])
        moved_ids = get_moved_object_ids(moved_id, resources.attachment_graph)
        trusted_evidence = getattr(resources, "trusted_attachment_evidence", {}).get(
            (moved_id, query_id)
        )
        if trusted_evidence is not None:
            audit["attachment_source"] = "human_review_override"
            audit["attachment_evidence"] = copy.deepcopy(trusted_evidence)
        if bool(question.get("attachment_remapped", False)) and query_id not in moved_ids:
            audit.update(
                status="dropped",
                reason="attachment_graph_mismatch",
                moved_group_ids=sorted(moved_ids),
            )
            return None, audit

        baseline = recompute_baseline(question, pose, objects_by_id)
        audit["baseline"] = {
            "valid": baseline.valid,
            "value": baseline.new_value,
            "ambiguity": baseline.ambiguity,
            "reason": baseline.reason,
            "details": baseline.details,
        }
        if not baseline.valid:
            audit.update(status="dropped", reason=baseline.reason)
            return None, audit

        directions = strict_directions_for_question(question, pose, objects_by_id)
        if not directions:
            audit.update(status="dropped", reason="degenerate_action_frame")
            return None, audit
        schedule = build_candidate_schedule(
            directions,
            legacy_direction,
            legacy_distance,
        )
        original_strict_evaluation = evaluate_candidate(
            question,
            schedule[0],
            baseline,
            pose,
            resources,
        )
        audit["original_strict_candidate"] = {
            "direction": schedule[0].direction,
            "distance_m": schedule[0].distance_m,
            "delta": schedule[0].delta.tolist(),
            "valid": original_strict_evaluation.valid,
            "reason": original_strict_evaluation.reason,
            "new_value": original_strict_evaluation.new_value,
            "ambiguity": original_strict_evaluation.ambiguity,
            "details": original_strict_evaluation.details,
            "stored_answer_matches": (
                original_strict_evaluation.valid
                and question.get("correct_value") == original_strict_evaluation.new_value
            ),
        }
        candidate, selected, rejections = choose_candidate(
            question,
            schedule,
            baseline,
            pose,
            resources,
        )
        audit["candidate_rejection_counts"] = dict(sorted(rejections.items()))
        audit["candidate_count"] = len(schedule)
        if candidate is None or selected is None:
            audit.update(status="dropped", reason="no_legal_candidate")
            return None, audit

        repaired, options_preserved = build_repaired_question(
            question,
            index,
            candidate,
            baseline,
            selected,
            resources,
            pose,
            templates,
            object_centric_template=object_centric_template,
        )
        audit.update({
            "status": "candidate_repaired",
            "reason": None,
            "selected": {
                "rank": candidate.rank,
                "phase": candidate.phase,
                "direction": candidate.direction,
                "distance_m": candidate.distance_m,
                "delta": candidate.delta.tolist(),
                "new_value": selected.new_value,
                "ambiguity": selected.ambiguity,
                "details": selected.details,
                "relation_unchanged": selected.new_value == baseline.new_value,
            },
            "stored_answer_was_correct_under_recomputed_geometry": (
                question.get("correct_value") == selected.new_value
            ),
            "options_preserved": options_preserved,
            "movement_semantics_version": L2_OBJECT_MOVE_SEMANTICS_VERSION,
        })
        return repaired, audit
    except Exception as exc:
        audit.update(status="dropped", reason=f"repair_error:{type(exc).__name__}:{exc}")
        return None, audit


def _count_by_type(questions: Iterable[dict[str, Any]]) -> Counter[str]:
    return Counter(str(question.get("type", "")) for question in questions)


def _retention_report(
    input_questions: Sequence[dict[str, Any]],
    kept_questions: Sequence[dict[str, Any]],
    *,
    partial_run: bool,
    scene_errors: dict[str, str],
) -> dict[str, Any]:
    input_counts = _count_by_type(input_questions)
    kept_counts = _count_by_type(kept_questions)
    per_type: dict[str, Any] = {}
    failed_types: list[str] = []
    for qtype in sorted(TARGET_TYPES):
        total = input_counts[qtype]
        kept = kept_counts[qtype]
        retention = kept / total if total else 1.0
        passed = retention >= TYPE_RETENTION_THRESHOLD
        if not passed:
            failed_types.append(qtype)
        per_type[qtype] = {
            "input": total,
            "kept": kept,
            "dropped": total - kept,
            "retention": retention,
            "threshold": TYPE_RETENTION_THRESHOLD,
            "passed": passed,
        }
    total_input = sum(input_counts.values())
    total_kept = sum(kept_counts.values())
    overall_retention = total_kept / total_input if total_input else 1.0
    reasons: list[str] = []
    if partial_run:
        reasons.append("partial_scene_filter")
    if scene_errors:
        reasons.append("scene_load_errors")
    if overall_retention < OVERALL_RETENTION_THRESHOLD:
        reasons.append("overall_retention_below_threshold")
    if failed_types:
        reasons.append("type_retention_below_threshold")
    return {
        "accepted": not reasons,
        "failure_reasons": reasons,
        "overall": {
            "input": total_input,
            "kept": total_kept,
            "dropped": total_input - total_kept,
            "retention": overall_retention,
            "threshold": OVERALL_RETENTION_THRESHOLD,
            "passed": overall_retention >= OVERALL_RETENTION_THRESHOLD,
        },
        "per_type": per_type,
        "failed_types": failed_types,
    }


def repair_benchmark(
    benchmark: dict[str, Any],
    *,
    resources_by_scene: dict[str, SceneResources],
    scene_errors: dict[str, str],
    templates: dict[str, Any],
    selected_scene_ids: set[str] | None = None,
    target_types: set[str] | None = None,
    legacy_only: bool = False,
    rebalance: bool = True,
    deduplicate_against_preserved: bool = False,
    recover_object_centric_from_text: bool = False,
    object_centric_template: str = "configured",
) -> tuple[dict[str, Any], dict[str, Any]]:
    questions = benchmark.get("questions")
    if not isinstance(questions, list):
        raise ValueError("benchmark must contain a questions list")
    selected_types = set(TARGET_TYPES if target_types is None else target_types)
    unsupported_types = selected_types - TARGET_TYPES
    if not selected_types or unsupported_types:
        raise ValueError(
            f"invalid target types: {sorted(unsupported_types or selected_types)}"
        )
    partial_run = selected_scene_ids is not None
    in_scope_indices = [
        index
        for index, question in enumerate(questions)
        if str(question.get("type", "")) in selected_types
        and (
            not legacy_only
            or question.get("movement_semantics_version")
            != L2_OBJECT_MOVE_SEMANTICS_VERSION
        )
        and (
            selected_scene_ids is None
            or str(question.get("scene_id", "")) in selected_scene_ids
        )
    ]
    input_target_questions = [questions[index] for index in in_scope_indices]
    required_scene_ids = {
        str(question.get("scene_id", "")) for question in input_target_questions
    }
    unavailable_scene_ids = sorted(
        scene_id for scene_id in required_scene_ids if scene_id not in resources_by_scene
    )
    if unavailable_scene_ids:
        details = {
            scene_id: scene_errors.get(scene_id, "missing_resources")
            for scene_id in unavailable_scene_ids
        }
        raise RuntimeError(
            "refusing to drop questions for systemic scene-resource failures: "
            + json.dumps(details, ensure_ascii=False, sort_keys=True)
        )

    repaired_by_index: dict[int, dict[str, Any]] = {}
    audits_by_index: dict[int, dict[str, Any]] = {}
    for position, index in enumerate(in_scope_indices, start=1):
        question = questions[index]
        scene_id = str(question.get("scene_id", ""))
        LOGGER.info(
            "Repairing target question %d/%d: index=%d scene=%s type=%s",
            position,
            len(in_scope_indices),
            index,
            scene_id,
            question.get("type"),
        )
        resources = resources_by_scene[scene_id]
        repaired, audit = _repair_one(
            index,
            question,
            resources,
            templates,
            recover_object_centric_from_text=recover_object_centric_from_text,
            object_centric_template=object_centric_template,
        )
        audits_by_index[index] = audit
        if repaired is not None:
            repaired_by_index[index] = repaired

    in_scope_set = set(in_scope_indices)
    seen_keys: dict[tuple[Any, ...], int] = {}
    if deduplicate_against_preserved:
        for index, question in enumerate(questions):
            if (
                index not in in_scope_set
                and str(question.get("type", "")) in selected_types
            ):
                seen_keys.setdefault(repaired_dedup_key(question), index)
    deduped_by_index: dict[int, dict[str, Any]] = {}
    for index in in_scope_indices:
        repaired = repaired_by_index.get(index)
        if repaired is None:
            continue
        key = repaired_dedup_key(repaired)
        duplicate_of = seen_keys.get(key)
        if duplicate_of is not None:
            duplicate_reason = (
                "duplicate_repaired_question"
                if duplicate_of in in_scope_set
                else "duplicate_preserved_question"
            )
            audits_by_index[index].update(
                status="dropped",
                reason=duplicate_reason,
                duplicate_of_source_index=duplicate_of,
            )
            continue
        seen_keys[key] = index
        deduped_by_index[index] = repaired

    final_repaired_by_index: dict[int, dict[str, Any]] = {}
    if rebalance:
        balance_input: list[dict[str, Any]] = []
        for index in in_scope_indices:
            repaired = deduped_by_index.get(index)
            if repaired is None:
                continue
            tagged = copy.deepcopy(repaired)
            tagged["_repair_source_index"] = index
            balance_input.append(tagged)
        balanced = balance_l2_attachment_per_scene(balance_input)
        kept_indices = {
            int(question["_repair_source_index"]) for question in balanced
        }
        for tagged in balanced:
            source_index = int(tagged.pop("_repair_source_index"))
            final_repaired_by_index[source_index] = tagged
            audits_by_index[source_index]["status"] = "kept"
        for index in deduped_by_index:
            if index not in kept_indices:
                audits_by_index[index].update(
                    status="dropped", reason="attachment_balance"
                )
    else:
        final_repaired_by_index = dict(deduped_by_index)
        for index in final_repaired_by_index:
            audits_by_index[index]["status"] = "kept"

    output_questions: list[dict[str, Any]] = []
    output_index_by_source: dict[int, int] = {}
    for index, question in enumerate(questions):
        if index not in in_scope_set:
            output_questions.append(question)
            continue
        repaired = final_repaired_by_index.get(index)
        if repaired is not None:
            output_index_by_source[index] = len(output_questions)
            output_questions.append(repaired)

    for source_index, output_index in output_index_by_source.items():
        repaired_question = output_questions[output_index]
        audits_by_index[source_index].update(
            output_index=output_index,
            repaired_question_sha256=hashlib.sha256(
                json.dumps(
                    repaired_question,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
        )

    repaired_doc = copy.deepcopy(benchmark)
    repaired_doc["questions"] = output_questions
    suffix = (
        "object-centric-v2-repair"
        if selected_types == {"object_move_object_centric"} and legacy_only
        else "l2-move-v2-repair"
    )
    repaired_doc["version"] = f"{benchmark.get('version', 'unknown')}-{suffix}"
    repaired_doc["statistics"] = compute_statistics(output_questions)

    kept_scope = [final_repaired_by_index[index] for index in in_scope_indices if index in final_repaired_by_index]
    acceptance = _retention_report(
        input_target_questions,
        kept_scope,
        partial_run=partial_run,
        scene_errors=scene_errors,
    )
    status_counts = Counter(audit["status"] for audit in audits_by_index.values())
    drop_reasons = Counter(
        str(audit.get("reason"))
        for audit in audits_by_index.values()
        if audit.get("status") == "dropped"
    )
    changed_counts = Counter(
        str(question.get("type", ""))
        for question in kept_scope
        if question.get("relation_unchanged") is False
    )
    unchanged_counts = Counter(
        str(question.get("type", ""))
        for question in kept_scope
        if question.get("relation_unchanged") is True
    )
    audit_doc = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "movement_semantics_version": L2_OBJECT_MOVE_SEMANTICS_VERSION,
        "target_types": sorted(selected_types),
        "legacy_only": legacy_only,
        "attachment_rebalance_enabled": rebalance,
        "deduplicate_against_preserved": deduplicate_against_preserved,
        "object_centric_template": object_centric_template,
        "selected_scene_ids": sorted(selected_scene_ids) if selected_scene_ids else None,
        "partial_run": partial_run,
        "scene_errors": scene_errors,
        "publication_acceptance": acceptance,
        "aggregate": {
            "input_benchmark_question_count": len(questions),
            "output_benchmark_question_count": len(output_questions),
            "in_scope_target_question_count": len(in_scope_indices),
            "candidate_repaired_count": len(repaired_by_index),
            "deduplicated_candidate_count": len(deduped_by_index),
            "kept_repaired_count": len(kept_scope),
            "status_counts": dict(sorted(status_counts.items())),
            "drop_reasons": dict(sorted(drop_reasons.items())),
            "changed_kept_by_type": dict(sorted(changed_counts.items())),
            "unchanged_kept_by_type": dict(sorted(unchanged_counts.items())),
        },
        "questions": [audits_by_index[index] for index in in_scope_indices],
    }
    return repaired_doc, audit_doc


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("output_val/benchmark.json"),
        help="Existing benchmark (never overwritten).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output_val/benchmark_l2_move_corrected_v2.json"),
    )
    parser.add_argument(
        "--audit-output",
        type=Path,
        default=Path("output_val/benchmark_l2_move_corrected_v2_audit.json"),
    )
    parser.add_argument("--scannet-root", type=Path)
    parser.add_argument("--scannetpp-root", type=Path)
    parser.add_argument(
        "--scene-metadata-root",
        action="append",
        type=Path,
        default=None,
        help="Root to search for scene_metadata/*.json; repeatable. Defaults to input parent.",
    )
    parser.add_argument(
        "--distance-geometry",
        choices=("mesh", "aabb"),
        default="mesh",
        help="Use pipeline-equivalent surface samples (default) or explicit AABB approximation.",
    )
    parser.add_argument(
        "--scene-id",
        action="append",
        default=None,
        help="Repair only these pilot scenes. Unselected questions are preserved and publication acceptance fails.",
    )
    parser.add_argument(
        "--target-type",
        action="append",
        choices=sorted(TARGET_TYPES),
        default=None,
        help="Repair only this movement type; repeatable. Defaults to all target types.",
    )
    parser.add_argument(
        "--legacy-only",
        action="store_true",
        help="Repair only questions that are not already marked with movement semantics v2.",
    )
    parser.add_argument(
        "--object-centric-template",
        choices=("configured", "freeze"),
        default="configured",
        help="Use configured template selection or force the canonical Freeze wording.",
    )
    parser.add_argument(
        "--skip-attachment-rebalance",
        action="store_true",
        help="Preserve repaired question order without balancing the selected subset.",
    )
    parser.add_argument(
        "--deduplicate-against-preserved",
        action="store_true",
        help="Drop repaired questions that duplicate an out-of-scope preserved question.",
    )
    parser.add_argument(
        "--allow-retention-failure",
        action="store_true",
        help="Return success after writing even when publication retention thresholds fail.",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    input_path = args.input.resolve()
    output_path = args.output.resolve()
    audit_path = args.audit_output.resolve()
    if output_path == input_path or audit_path == input_path:
        raise SystemExit("Refusing to overwrite the input benchmark")

    benchmark = _json_load(input_path)
    if not isinstance(benchmark, dict):
        raise SystemExit("Input benchmark must be a JSON object")
    questions = benchmark.get("questions")
    if not isinstance(questions, list):
        raise SystemExit("Input benchmark has no questions list")

    selected_scene_ids = set(args.scene_id) if args.scene_id else None
    selected_types = set(args.target_type or TARGET_TYPES)
    target_questions = [
        question
        for question in questions
        if str(question.get("type", "")) in selected_types
        and (
            not args.legacy_only
            or question.get("movement_semantics_version")
            != L2_OBJECT_MOVE_SEMANTICS_VERSION
        )
        and (
            selected_scene_ids is None
            or str(question.get("scene_id", "")) in selected_scene_ids
        )
    ]
    scene_ids = sorted({str(question.get("scene_id", "")) for question in target_questions})
    metadata_roots = args.scene_metadata_root or [input_path.parent]
    metadata_index = _metadata_index(metadata_roots)
    LOGGER.info("Loaded metadata index for %d scenes", len(metadata_index))

    target_types_by_scene: dict[str, set[str]] = defaultdict(set)
    for question in target_questions:
        target_types_by_scene[str(question.get("scene_id", ""))].add(
            str(question.get("type", ""))
        )

    resources_by_scene: dict[str, SceneResources] = {}
    scene_errors: dict[str, str] = {}
    for position, scene_id in enumerate(scene_ids, start=1):
        LOGGER.info("Loading scene %s (%d/%d)", scene_id, position, len(scene_ids))
        try:
            resources_by_scene[scene_id] = _load_scene_resources(
                scene_id,
                metadata_index=metadata_index,
                scannet_root=args.scannet_root.resolve() if args.scannet_root else None,
                scannetpp_root=args.scannetpp_root.resolve() if args.scannetpp_root else None,
                distance_geometry=args.distance_geometry,
                needs_distance="object_move_distance" in target_types_by_scene[scene_id],
                needs_camera_pose=True,
            )
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            scene_errors[scene_id] = message
            LOGGER.error("Failed to load scene %s: %s", scene_id, message)

    repaired, audit = repair_benchmark(
        benchmark,
        resources_by_scene=resources_by_scene,
        scene_errors=scene_errors,
        templates=_load_templates(),
        selected_scene_ids=selected_scene_ids,
        target_types=selected_types,
        legacy_only=args.legacy_only,
        rebalance=not args.skip_attachment_rebalance,
        deduplicate_against_preserved=args.deduplicate_against_preserved,
        recover_object_centric_from_text=True,
        object_centric_template=args.object_centric_template,
    )
    audit["input_path"] = str(input_path)
    audit["output_path"] = str(output_path)
    audit["audit_path"] = str(audit_path)
    audit["distance_geometry_mode"] = args.distance_geometry
    audit["scene_metadata_paths"] = {
        scene_id: str(resources.metadata_path) if resources.metadata_path else None
        for scene_id, resources in resources_by_scene.items()
    }

    _write_json(output_path, repaired)
    _write_json(audit_path, audit)
    acceptance = audit["publication_acceptance"]
    LOGGER.info(
        "Wrote %s and %s; kept=%d/%d retention=%.3f publication_accepted=%s",
        output_path,
        audit_path,
        acceptance["overall"]["kept"],
        acceptance["overall"]["input"],
        acceptance["overall"]["retention"],
        acceptance["accepted"],
    )
    return (
        0
        if acceptance["accepted"]
        or selected_scene_ids is not None
        or args.allow_retention_failure
        else 2
    )


if __name__ == "__main__":
    raise SystemExit(main())
