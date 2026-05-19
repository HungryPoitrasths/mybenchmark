#!/usr/bin/env python3
"""Explain why edited attachment-pair children did not yield L2 questions.

Input is the JSON produced by scripts/diagnose_edited_pair_generation.py.
For each selected edited pair, this script reloads the scene/frame and checks
the child object as the L2 object-move query target for:

* direction questions (object_move_agent)
* distance questions (object_move_distance)
* occlusion questions (object_move_occlusion, when mesh-ray resources load)
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.qa_generator import (  # noqa: E402
    EXCLUDED_LABELS,
    MIN_DISTANCE_QUESTION_DISTANCE_M,
    _build_modified_scene,
    _classify_pair_movement,
    _compute_l1_style_visibility_metrics_for_static_target,
    _counterfactual_occlusion_backend,
    _direction_values_for_query_object,
    _distance_bin_index,
    _find_stable_distance_move_for_relation,
    _is_l2_occlusion_state_transition,
    _is_l2_occlusion_unchanged_candidate,
    _iter_valid_object_move_states,
    _query_visibility_for_object_move_state,
    _resolve_counterfactual_l1_visibility_status,
    _select_object_move_state,
)
from src.relation_engine import compute_all_relations, compute_distance_details  # noqa: E402
from src.scene_parser import load_instance_mesh_data, parse_scene  # noqa: E402
from src.support_graph import enrich_scene_with_attachment, get_scene_attachment_graph  # noqa: E402
from src.utils import RayCaster  # noqa: E402
from src.utils.colmap_loader import (  # noqa: E402
    CameraPose,
    load_axis_alignment,
    load_scannet_intrinsics,
)
from src.virtual_ops import (  # noqa: E402
    apply_movement,
    compute_room_bounds,
    get_moved_object_ids,
    has_terminal_bbox_collision,
    is_within_room,
)


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _resolve_scene_dir(data_root: Path, scene_id: str) -> Path:
    for candidate in (data_root / scene_id, data_root / "scans" / scene_id):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"scene directory not found for {scene_id}")


def _load_scene_poses(scene_dir: Path, axis_alignment: np.ndarray) -> dict[str, CameraPose]:
    poses: dict[str, CameraPose] = {}
    pose_dir = scene_dir / "pose"
    for pose_file in sorted(pose_dir.glob("*.txt"), key=lambda p: int(p.stem) if p.stem.isdigit() else 10**9):
        if not pose_file.stem.isdigit():
            continue
        matrix = np.loadtxt(str(pose_file))
        if not np.isfinite(matrix).all():
            continue
        aligned = axis_alignment @ matrix
        rotation = aligned[:3, :3]
        translation = aligned[:3, 3]
        image_name = f"{pose_file.stem}.jpg"
        poses[image_name] = CameraPose(
            image_name=image_name,
            rotation=rotation.T.astype(np.float64),
            translation=(-rotation.T @ translation).astype(np.float64),
        )
    return poses


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _normalize_ids(value: Any) -> set[int]:
    if not isinstance(value, (list, tuple, set)):
        return set()
    out: set[int] = set()
    for item in value:
        obj_id = _as_int(item)
        if obj_id is not None:
            out.add(obj_id)
    return out


def _merge_objects_by_id(*object_lists: list[dict[str, Any]]) -> list[dict[str, Any]]:
    objects_by_id: dict[int, dict[str, Any]] = {}
    for object_list in object_lists:
        for obj in object_list:
            obj_id = _as_int(obj.get("id"))
            if obj_id is None:
                continue
            objects_by_id[obj_id] = obj
    return list(objects_by_id.values())


def _merge_frame_attachment_rows(
    attachment_graph: dict[int, list[int]],
    frame_debug: dict[str, Any] | None,
) -> dict[int, list[int]]:
    merged = {int(parent): [int(child) for child in children] for parent, children in attachment_graph.items()}
    rows = (frame_debug or {}).get("attachment_rows", [])
    if not isinstance(rows, list):
        return merged
    for row in rows:
        if not isinstance(row, dict):
            continue
        parent_id = _as_int(row.get("parent_id"))
        child_id = _as_int(row.get("child_id"))
        if parent_id is None or child_id is None:
            continue
        children = merged.setdefault(parent_id, [])
        if child_id not in children:
            children.append(child_id)
    return {parent: sorted(children) for parent, children in merged.items()}


def _attachment_graph_closure_ids(
    *,
    attachment_graph: dict[int, list[int]],
    attachment_referable_ids: set[int],
) -> set[int]:
    allowed = set(attachment_referable_ids)
    stack = list(attachment_referable_ids)
    parent_by_child: dict[int, int] = {}
    for parent_id, child_ids in attachment_graph.items():
        for child_id in child_ids:
            parent_by_child[int(child_id)] = int(parent_id)
    visited: set[int] = set()
    while stack:
        current = int(stack.pop())
        if current in visited:
            continue
        visited.add(current)
        allowed.add(current)
        parent_id = parent_by_child.get(current)
        if parent_id is not None and parent_id not in allowed:
            allowed.add(parent_id)
            stack.append(parent_id)
        for child_id in attachment_graph.get(current, []) or []:
            child_id = int(child_id)
            if child_id not in allowed:
                allowed.add(child_id)
                stack.append(child_id)
    return allowed


def _load_frame_debug_map(pilot_root: Path) -> dict[tuple[str, str], dict[str, Any]]:
    frame_debug_dir = pilot_root / "frame_debug"
    out: dict[tuple[str, str], dict[str, Any]] = {}
    if not frame_debug_dir.exists():
        return out
    for path in sorted(frame_debug_dir.glob("*.json")):
        try:
            doc = _read_json(path)
        except Exception:
            continue
        if not isinstance(doc, dict):
            continue
        scene_id = str(doc.get("scene_id") or path.stem)
        for frame in doc.get("frames", []):
            if not isinstance(frame, dict):
                continue
            image_name = str(frame.get("image_name", "")).strip()
            if image_name:
                out[(scene_id, image_name)] = frame
    return out


def _relation_other_id(query_obj_id: int, relation: dict[str, Any]) -> int | None:
    obj_a_id = int(relation["obj_a_id"])
    obj_b_id = int(relation["obj_b_id"])
    if query_obj_id == obj_a_id:
        return obj_b_id
    if query_obj_id == obj_b_id:
        return obj_a_id
    return None


def _valid_move_states(
    *,
    objects: list[dict[str, Any]],
    attachment_graph: dict[int, list[int]],
    move_source_id: int,
    room_bounds: dict | None,
    collision_objects: list[dict[str, Any]] | None,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    room_min, room_max = compute_room_bounds(objects, room_bounds=room_bounds)
    moved_ids = set(get_moved_object_ids(move_source_id, attachment_graph))
    states: list[dict[str, Any]] = []
    invalid_reasons: Counter[str] = Counter()
    for delta, moved_objects, state_moved_ids in _iter_valid_object_move_states(
        objects,
        attachment_graph,
        move_source_id,
        room_bounds=room_bounds,
        collision_objects=collision_objects,
    ):
        states.append({
            "delta": np.asarray(delta, dtype=np.float64),
            "moved_objects": moved_objects,
            "moved_ids": set(int(obj_id) for obj_id in state_moved_ids),
        })

    # Count coarse invalid reasons over the same candidate deltas used by virtual_ops.
    from src.virtual_ops import MOVEMENT_CANDIDATES

    for delta in MOVEMENT_CANDIDATES:
        delta = np.asarray(delta, dtype=np.float64)
        if not np.allclose(delta[2], 0.0):
            continue
        moved_objects = apply_movement(objects, attachment_graph, move_source_id, delta)
        if not is_within_room(moved_objects, room_min, room_max):
            invalid_reasons["out_of_room"] += 1
            continue
        if has_terminal_bbox_collision(
            objects,
            moved_objects,
            moved_ids,
            collision_objects=collision_objects,
        ):
            invalid_reasons["collision"] += 1
    return states, invalid_reasons


def _diagnose_direction(
    *,
    query_obj_id: int,
    move_source_id: int,
    moved_ids: set[int],
    base_relations: list[dict[str, Any]],
    valid_states: list[dict[str, Any]],
    invalid_reasons: Counter[str],
    referable_ids: set[int],
    camera_pose: CameraPose,
) -> dict[str, Any]:
    reason_counts: Counter[str] = Counter()
    generated: list[dict[str, Any]] = []
    skipped_examples: dict[str, list[dict[str, Any]]] = defaultdict(list)

    if not valid_states:
        reason_counts.update(invalid_reasons or Counter({"no_valid_move_state": 1}))
        return {
            "candidate_possible": False,
            "generated_candidate_count": 0,
            "reason_counts": dict(sorted(reason_counts.items())),
            "generated_examples": [],
            "skipped_examples": dict(skipped_examples),
        }

    relation_maps = [
        {
            (int(rel["obj_a_id"]), int(rel["obj_b_id"])): rel
            for rel in compute_all_relations(state["moved_objects"], camera_pose, None, None)
        }
        for state in valid_states
    ]

    for relation in base_relations:
        if query_obj_id not in (int(relation["obj_a_id"]), int(relation["obj_b_id"])):
            continue
        other_id = _relation_other_id(query_obj_id, relation)
        if other_id is None:
            continue
        if other_id not in referable_ids:
            reason_counts["other_object_not_referable"] += 1
            continue

        relation_obj_b_id = query_obj_id
        relation_obj_c_id = other_id
        _pair_moves_apart, pair_moves_together = _classify_pair_movement(
            relation_obj_b_id,
            relation_obj_c_id,
            moved_ids,
        )
        same_direction = False
        changed_direction = False
        valid_relation_state_count = 0
        for state, relation_map in zip(valid_states, relation_maps):
            new_relation = relation_map.get((int(relation["obj_a_id"]), int(relation["obj_b_id"])))
            if new_relation is None:
                continue
            direction_values = _direction_values_for_query_object(query_obj_id, relation, new_relation)
            if direction_values is None:
                continue
            valid_relation_state_count += 1
            old_val, new_val = direction_values
            if old_val == new_val:
                same_direction = True
            else:
                changed_direction = True
                generated.append({
                    "relation_pair": [int(relation["obj_a_id"]), int(relation["obj_b_id"])],
                    "other_id": other_id,
                    "old_direction": old_val,
                    "new_direction": new_val,
                    "delta": state["delta"].tolist(),
                    "relation_unchanged": False,
                })
                break

        if not changed_direction and same_direction and pair_moves_together:
            generated.append({
                "relation_pair": [int(relation["obj_a_id"]), int(relation["obj_b_id"])],
                "other_id": other_id,
                "old_direction": "unchanged",
                "new_direction": "unchanged",
                "relation_unchanged": True,
            })
            continue
        if changed_direction:
            continue

        reasons: list[str] = []
        if valid_relation_state_count == 0:
            reasons.append("relation_missing_after_move")
        else:
            reasons.append("no_direction_change")
        if same_direction and not pair_moves_together:
            reasons.append("unchanged_fallback_not_allowed_because_other_object_does_not_move_with_child")
        for reason in reasons:
            reason_counts[reason] += 1
        key = ",".join(reasons)
        if len(skipped_examples[key]) < 3:
            skipped_examples[key].append({
                "relation_pair": [int(relation["obj_a_id"]), int(relation["obj_b_id"])],
                "other_id": other_id,
                "valid_relation_state_count": valid_relation_state_count,
                "same_direction_seen": same_direction,
                "pair_moves_together": pair_moves_together,
            })

    return {
        "candidate_possible": bool(generated),
        "generated_candidate_count": len(generated),
        "reason_counts": dict(sorted(reason_counts.items())),
        "generated_examples": generated[:3],
        "skipped_examples": dict(skipped_examples),
    }


def _diagnose_distance(
    *,
    objects: list[dict[str, Any]],
    movement_objects: list[dict[str, Any]],
    query_obj_id: int,
    move_source_id: int,
    moved_ids: set[int],
    attachment_graph: dict[int, list[int]],
    base_relations: list[dict[str, Any]],
    valid_states: list[dict[str, Any]],
    invalid_reasons: Counter[str],
    referable_ids: set[int],
    room_bounds: dict | None,
    collision_objects: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    reason_counts: Counter[str] = Counter()
    generated: list[dict[str, Any]] = []
    skipped_examples: dict[str, list[dict[str, Any]]] = defaultdict(list)

    if not valid_states:
        reason_counts.update(invalid_reasons or Counter({"no_valid_move_state": 1}))
        return {
            "candidate_possible": False,
            "generated_candidate_count": 0,
            "reason_counts": dict(sorted(reason_counts.items())),
            "generated_examples": [],
            "skipped_examples": dict(skipped_examples),
        }

    for relation in base_relations:
        if query_obj_id not in (int(relation["obj_a_id"]), int(relation["obj_b_id"])):
            continue
        other_id = _relation_other_id(query_obj_id, relation)
        if other_id is None:
            continue
        if other_id not in referable_ids:
            reason_counts["other_object_not_referable"] += 1
            continue

        pair_moves_apart, pair_moves_together = _classify_pair_movement(
            query_obj_id,
            other_id,
            moved_ids,
        )
        delta, old_value, answer_value, relation_unchanged = _find_stable_distance_move_for_relation(
            objects,
            attachment_graph,
            move_source_id,
            relation,
            room_bounds=room_bounds,
            collision_objects=collision_objects,
            movement_objects=movement_objects,
            allow_unchanged_fallback=pair_moves_together,
        )
        if delta is not None and old_value is not None and answer_value is not None:
            generated.append({
                "relation_pair": [int(relation["obj_a_id"]), int(relation["obj_b_id"])],
                "other_id": other_id,
                "old_distance_bin": old_value,
                "new_distance_bin": answer_value,
                "delta": np.asarray(delta).tolist(),
                "relation_unchanged": relation_unchanged,
            })
            continue

        old_label = str(relation.get("distance_bin", "")).strip()
        old_bin_id = str(relation.get("distance_bin_id", "")).strip() or None
        old_idx = _distance_bin_index(old_label, bin_id=old_bin_id)
        reasons: list[str] = []
        if old_idx is None:
            reasons.append("missing_old_distance_bin")
        else:
            same_bin = False
            near_boundary = False
            too_small = False
            changed_bin = False
            for state in valid_states:
                moved_map = {int(obj["id"]): obj for obj in state["moved_objects"]}
                if int(relation["obj_a_id"]) not in moved_map or int(relation["obj_b_id"]) not in moved_map:
                    continue
                new_distance = compute_distance_details(
                    moved_map[int(relation["obj_a_id"])],
                    moved_map[int(relation["obj_b_id"])],
                )
                if float(new_distance.get("distance_m", 0.0) or 0.0) < MIN_DISTANCE_QUESTION_DISTANCE_M:
                    too_small = True
                    continue
                if bool(new_distance.get("near_boundary", False)):
                    near_boundary = True
                    continue
                new_idx = _distance_bin_index(
                    str(new_distance.get("distance_bin", "")),
                    bin_id=str(new_distance.get("distance_bin_id", "")).strip() or None,
                )
                if new_idx is None:
                    continue
                if new_idx == old_idx:
                    same_bin = True
                else:
                    changed_bin = True
            if near_boundary:
                reasons.append("near_distance_bin_boundary")
            if too_small:
                reasons.append("distance_too_small")
            if not changed_bin and pair_moves_apart:
                reasons.append("no_distance_bin_crossing")
            if same_bin and not pair_moves_together:
                reasons.append("unchanged_fallback_not_allowed_because_other_object_does_not_move_with_child")
            if pair_moves_together and not same_bin:
                reasons.append("no_stable_unchanged_distance_fallback")
        if not reasons:
            reasons.append("no_stable_distance_move")
        for reason in reasons:
            reason_counts[reason] += 1
        key = ",".join(sorted(set(reasons)))
        if len(skipped_examples[key]) < 3:
            skipped_examples[key].append({
                "relation_pair": [int(relation["obj_a_id"]), int(relation["obj_b_id"])],
                "other_id": other_id,
                "old_distance_bin": old_label,
                "pair_moves_apart": pair_moves_apart,
                "pair_moves_together": pair_moves_together,
            })

    return {
        "candidate_possible": bool(generated),
        "generated_candidate_count": len(generated),
        "reason_counts": dict(sorted(reason_counts.items())),
        "generated_examples": generated[:3],
        "skipped_examples": dict(skipped_examples),
    }


def _load_occlusion_resources(scene_dir: Path, scene_id: str, axis_align: np.ndarray, objects: list[dict[str, Any]]) -> tuple[Any, Any, Any, list[str]]:
    errors: list[str] = []
    color_intrinsics = None
    ray_caster = None
    instance_mesh_data = None
    try:
        color_intrinsics = load_scannet_intrinsics(scene_dir)
    except Exception as exc:
        errors.append(f"missing_color_intrinsics:{exc}")
    mesh_path = scene_dir / f"{scene_id}_vh_clean.ply"
    if not mesh_path.exists():
        mesh_path = scene_dir / f"{scene_id}_vh_clean_2.ply"
    if RayCaster is None:
        errors.append("raycaster_unavailable")
    elif not mesh_path.exists():
        errors.append("scene_mesh_ply_missing")
    else:
        try:
            ray_caster = RayCaster.from_ply(str(mesh_path), axis_alignment=axis_align)
        except Exception as exc:
            errors.append(f"raycaster_init_failed:{exc}")
    try:
        instance_mesh_data = load_instance_mesh_data(
            scene_dir,
            instance_ids=[int(obj["id"]) for obj in objects],
            n_surface_samples=512,
        )
    except Exception as exc:
        errors.append(f"instance_mesh_data_failed:{exc}")
    return color_intrinsics, ray_caster, instance_mesh_data, errors


def _diagnose_occlusion(
    *,
    objects: list[dict[str, Any]],
    query_obj: dict[str, Any],
    move_source_id: int,
    attachment_graph: dict[int, list[int]],
    camera_pose: CameraPose,
    room_bounds: dict | None,
    collision_objects: list[dict[str, Any]] | None,
    color_intrinsics,
    ray_caster,
    instance_mesh_data,
    resource_errors: list[str],
) -> dict[str, Any]:
    if resource_errors or color_intrinsics is None or ray_caster is None or instance_mesh_data is None:
        return {
            "candidate_possible": False,
            "diagnosed": False,
            "reason_counts": {"occlusion_not_diagnosed_missing_inputs": 1},
            "resource_errors": resource_errors,
            "state_examples": [],
        }

    reason_counts: Counter[str] = Counter()
    state_examples: list[dict[str, Any]] = []
    generated: list[dict[str, Any]] = []
    compare_backend = _counterfactual_occlusion_backend("mesh_ray", ray_caster, instance_mesh_data)
    moved_ids = set(get_moved_object_ids(move_source_id, attachment_graph))

    original_scene_context = _build_modified_scene(ray_caster, instance_mesh_data, set())
    metrics, source = _compute_l1_style_visibility_metrics_for_static_target(
        obj=query_obj,
        camera_pose=camera_pose,
        color_intrinsics=color_intrinsics,
        depth_image=None,
        depth_intrinsics=None,
        occlusion_backend=compare_backend,
        ray_caster=ray_caster,
        instance_mesh_data=instance_mesh_data,
        modified_scene=original_scene_context,
    )
    original_status, old_reason_code, _old_reason_detail = _resolve_counterfactual_l1_visibility_status(metrics)
    original_visibility = {
        int(query_obj["id"]): (original_status, source, old_reason_code, metrics)
    }
    if original_status is None:
        return {
            "candidate_possible": False,
            "diagnosed": True,
            "reason_counts": {"original_visibility_unresolved": 1},
            "original_status": original_status,
            "original_reason_code": old_reason_code,
            "state_examples": [],
        }

    selected_state = _select_object_move_state(
        objects,
        attachment_graph,
        move_source_id,
        camera_pose,
        room_bounds=room_bounds,
        collision_objects=collision_objects,
        allow_unchanged_attachment=len(moved_ids) > 1,
        color_intrinsics=color_intrinsics,
        occlusion_backend="mesh_ray",
        ray_caster=ray_caster,
        instance_mesh_data=instance_mesh_data,
    )
    states = []
    if selected_state is not None:
        states.append(("selected", selected_state))
    excluded = [selected_state.delta] if selected_state is not None else []
    for idx, (delta, moved_objects, state_moved_ids) in enumerate(
        _iter_valid_object_move_states(
            objects,
            attachment_graph,
            move_source_id,
            room_bounds=room_bounds,
            collision_objects=collision_objects,
        )
    ):
        if any(np.allclose(delta, old_delta) for old_delta in excluded):
            continue
        from src.qa_generator import _make_selected_object_move_state

        states.append((
            f"fallback_{idx}",
            _make_selected_object_move_state(delta, moved_objects, set(state_moved_ids)),
        ))

    if not states:
        return {
            "candidate_possible": False,
            "diagnosed": True,
            "reason_counts": {"no_valid_move_state": 1},
            "original_status": original_status,
            "original_reason_code": old_reason_code,
            "state_examples": [],
        }

    unchanged_seen = False
    for state_name, state in states:
        visibility = _query_visibility_for_object_move_state(
            query_obj=query_obj,
            original_objects=objects,
            selected_state=state,
            original_visibility=original_visibility,
            camera_pose=camera_pose,
            color_intrinsics=color_intrinsics,
            compare_backend=compare_backend,
            ray_caster=ray_caster,
            instance_mesh_data=instance_mesh_data,
        )
        old_status, old_source, old_reason_code, _old_metrics, new_status, new_source, new_reason_code, _new_metrics = visibility
        if _is_l2_occlusion_state_transition(old_status, new_status):
            generated.append({
                "state": state_name,
                "delta": state.delta.tolist(),
                "old_status": old_status,
                "new_status": new_status,
                "old_reason_code": old_reason_code,
                "new_reason_code": new_reason_code,
            })
            continue
        if _is_l2_occlusion_unchanged_candidate(old_status, new_status):
            unchanged_seen = True
        if len(state_examples) < 5:
            state_examples.append({
                "state": state_name,
                "delta": state.delta.tolist(),
                "old_status": old_status,
                "new_status": new_status,
                "old_source": old_source,
                "new_source": new_source,
                "old_reason_code": old_reason_code,
                "new_reason_code": new_reason_code,
            })

    if generated:
        return {
            "candidate_possible": True,
            "diagnosed": True,
            "generated_candidate_count": len(generated),
            "generated_examples": generated[:3],
            "reason_counts": {},
            "original_status": original_status,
            "state_examples": state_examples,
        }
    if unchanged_seen:
        reason_counts["visibility_unchanged_for_valid_moves"] += 1
    else:
        reason_counts["no_l2_occlusion_transition"] += 1
    return {
        "candidate_possible": False,
        "diagnosed": True,
        "generated_candidate_count": 0,
        "reason_counts": dict(sorted(reason_counts.items())),
        "original_status": original_status,
        "original_reason_code": old_reason_code,
        "state_examples": state_examples,
    }


def _load_scene_context(data_root: Path, scene_id: str) -> dict[str, Any]:
    scene_dir = _resolve_scene_dir(data_root, scene_id)
    axis_align = load_axis_alignment(scene_dir)
    scene = parse_scene(scene_dir)
    if scene is None:
        raise RuntimeError(f"parse_scene returned None for {scene_id}")
    enrich_scene_with_attachment(scene)
    poses = _load_scene_poses(scene_dir, axis_alignment=axis_align)
    return {
        "scene_dir": scene_dir,
        "axis_align": axis_align,
        "scene": scene,
        "poses": poses,
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    pair_report = _read_json(args.pair_report)
    pilot_root = Path(pair_report.get("pilot_root") or args.pilot_root or args.pair_report.parent)
    frame_debug_by_key = _load_frame_debug_map(pilot_root)
    scene_cache: dict[str, dict[str, Any]] = {}
    occlusion_resource_cache: dict[str, tuple[Any, Any, Any, list[str]]] = {}
    results: list[dict[str, Any]] = []
    aggregate = {
        "direction": Counter(),
        "distance": Counter(),
        "occlusion": Counter(),
    }

    selected_diagnoses = set(args.diagnosis or [])
    pairs = pair_report.get("pairs", [])
    if not isinstance(pairs, list):
        raise ValueError("pair report has no pairs list")

    for item in pairs:
        if not isinstance(item, dict):
            continue
        diagnosis = str(item.get("diagnosis", ""))
        if selected_diagnoses and diagnosis not in selected_diagnoses:
            continue
        if not selected_diagnoses and diagnosis == "benchmark_contains_human_child_surface_text" and not args.all_pairs:
            continue
        scene_id = str(item.get("scene_id", "")).strip()
        image_name = str(item.get("image_name", "")).strip()
        pair = item.get("pair")
        if not scene_id or not image_name or not isinstance(pair, list) or len(pair) < 2:
            continue
        parent_id = int(pair[0])
        child_id = int(pair[1])

        if scene_id not in scene_cache:
            scene_cache[scene_id] = _load_scene_context(Path(args.data_root), scene_id)
        ctx = scene_cache[scene_id]
        scene = ctx["scene"]
        objects = list(scene["objects"])
        obj_map = {int(obj["id"]): obj for obj in objects}
        if parent_id not in obj_map or child_id not in obj_map:
            results.append({
                "scene_id": scene_id,
                "image_name": image_name,
                "pair": [parent_id, child_id],
                "error": "parent_or_child_not_in_scene_objects",
            })
            continue
        poses = ctx["poses"]
        if image_name not in poses:
            results.append({
                "scene_id": scene_id,
                "image_name": image_name,
                "pair": [parent_id, child_id],
                "error": "pose_missing_for_frame",
            })
            continue
        camera_pose = poses[image_name]
        frame_debug = frame_debug_by_key.get((scene_id, image_name))
        referable_ids = _normalize_ids((frame_debug or {}).get("pipeline_referable_object_ids_used_for_generation"))
        if not referable_ids:
            referable_ids = _normalize_ids((frame_debug or {}).get("referable_object_ids"))
        visible_ids = _normalize_ids((frame_debug or {}).get("pipeline_visible_object_ids_used_for_generation"))
        if not visible_ids:
            visible_ids = {int(obj["id"]) for obj in objects}
        attachment_referable_ids = _normalize_ids(
            (frame_debug or {}).get("pipeline_attachment_referable_object_ids_used_for_generation")
        )
        if not attachment_referable_ids:
            attachment_referable_ids = _normalize_ids((frame_debug or {}).get("attachment_referable_object_ids"))
        visible_graph_seed_objects = [
            obj for obj in objects
            if int(obj["id"]) in visible_ids
            and str(obj.get("label", "")).strip().lower() not in EXCLUDED_LABELS
        ]
        collision_objects = list(visible_graph_seed_objects)
        attachment_graph = _merge_frame_attachment_rows(
            get_scene_attachment_graph(scene, scene_id=scene_id),
            frame_debug,
        )
        attachment_graph = {
            int(parent_id): [
                int(child_id)
                for child_id in child_ids
                if int(child_id) in visible_ids
            ]
            for parent_id, child_ids in attachment_graph.items()
            if int(parent_id) in visible_ids
        }
        graph_eligible_ids = _attachment_graph_closure_ids(
            attachment_graph=attachment_graph,
            attachment_referable_ids=attachment_referable_ids or {parent_id, child_id},
        )
        movement_objects = [
            obj for obj in visible_graph_seed_objects
            if int(obj["id"]) in graph_eligible_ids
        ]
        objects_uniq = [
            obj for obj in visible_graph_seed_objects
            if int(obj["id"]) in referable_ids
        ]
        relation_objects = _merge_objects_by_id(objects_uniq, movement_objects)
        moved_ids = set(get_moved_object_ids(parent_id, attachment_graph))
        base_relations = compute_all_relations(relation_objects, camera_pose, None, None)
        valid_states, invalid_reasons = _valid_move_states(
            objects=movement_objects,
            attachment_graph=attachment_graph,
            move_source_id=parent_id,
            room_bounds=scene.get("room_bounds"),
            collision_objects=collision_objects,
        )

        direction = _diagnose_direction(
            query_obj_id=child_id,
            move_source_id=parent_id,
            moved_ids=moved_ids,
            base_relations=base_relations,
            valid_states=valid_states,
            invalid_reasons=invalid_reasons,
            referable_ids=referable_ids,
            camera_pose=camera_pose,
        )
        distance = _diagnose_distance(
            objects=relation_objects,
            movement_objects=movement_objects,
            query_obj_id=child_id,
            move_source_id=parent_id,
            moved_ids=moved_ids,
            attachment_graph=attachment_graph,
            base_relations=base_relations,
            valid_states=valid_states,
            invalid_reasons=invalid_reasons,
            referable_ids=referable_ids,
            room_bounds=scene.get("room_bounds"),
            collision_objects=collision_objects,
        )
        if args.no_occlusion:
            occlusion = {
                "candidate_possible": False,
                "diagnosed": False,
                "reason_counts": {"occlusion_diagnosis_disabled": 1},
            }
        else:
            if scene_id not in occlusion_resource_cache:
                occlusion_resource_cache[scene_id] = _load_occlusion_resources(
                    ctx["scene_dir"],
                    scene_id,
                    ctx["axis_align"],
                    objects,
                )
            color_intrinsics, ray_caster, instance_mesh_data, resource_errors = occlusion_resource_cache[scene_id]
            occlusion = _diagnose_occlusion(
                objects=movement_objects,
                query_obj=obj_map[child_id],
                move_source_id=parent_id,
                attachment_graph=attachment_graph,
                camera_pose=camera_pose,
                room_bounds=scene.get("room_bounds"),
                collision_objects=collision_objects,
                color_intrinsics=color_intrinsics,
                ray_caster=ray_caster,
                instance_mesh_data=instance_mesh_data,
                resource_errors=resource_errors,
            )

        for key, section in (("direction", direction), ("distance", distance), ("occlusion", occlusion)):
            if section.get("candidate_possible"):
                aggregate[key]["candidate_possible"] += 1
            reason_counts = section.get("reason_counts", {})
            if isinstance(reason_counts, dict):
                aggregate[key].update({str(reason): int(count) for reason, count in reason_counts.items()})

        results.append({
            "scene_id": scene_id,
            "image_name": image_name,
            "pair": [parent_id, child_id],
            "parent_surface_text": item.get("parent_surface_text"),
            "child_surface_text": item.get("child_surface_text"),
            "previous_pair_diagnosis": diagnosis,
            "child_in_moved_ids": child_id in moved_ids,
            "moved_ids": sorted(moved_ids),
            "referable_ids": sorted(referable_ids),
            "visible_ids": sorted(visible_ids),
            "attachment_referable_ids": sorted(attachment_referable_ids),
            "movement_object_ids": sorted(int(obj["id"]) for obj in movement_objects),
            "collision_object_ids": sorted(int(obj["id"]) for obj in collision_objects),
            "relation_object_ids": sorted(int(obj["id"]) for obj in relation_objects),
            "child_in_attachment_query_pool": child_id in attachment_referable_ids,
            "valid_move_state_count": len(valid_states),
            "invalid_move_reason_counts": dict(sorted(invalid_reasons.items())),
            "direction": direction,
            "distance": distance,
            "occlusion": occlusion,
        })

    return {
        "pair_report": str(args.pair_report),
        "pilot_root": str(pilot_root),
        "data_root": str(args.data_root),
        "pair_count": len(results),
        "aggregate": {
            key: dict(sorted(counter.items()))
            for key, counter in aggregate.items()
        },
        "pairs": results,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose L2 direction/distance/occlusion skip reasons for edited attachment-pair children."
    )
    parser.add_argument("--pair_report", type=Path, required=True, help="edited_pair_generation_debug.json")
    parser.add_argument("--data_root", type=Path, required=True, help="ScanNet scans root")
    parser.add_argument("--pilot_root", type=Path, default=None, help="Optional pilot root override")
    parser.add_argument("--output", type=Path, default=None, help="Output JSON path")
    parser.add_argument(
        "--diagnosis",
        action="append",
        default=[],
        help="Restrict to pair_report diagnosis value. May be passed multiple times.",
    )
    parser.add_argument("--all_pairs", action="store_true", help="Analyze pairs that already reached benchmark too")
    parser.add_argument("--no_occlusion", action="store_true", help="Skip mesh-ray occlusion diagnosis")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_report(args)
    output = args.output or args.pair_report.with_name("edited_pair_l2_skip_reasons.json")
    _write_json(output, report)
    print(f"Wrote {output}")
    print(f"diagnosed pairs: {report['pair_count']}")
    print("aggregate:")
    for section, counts in report["aggregate"].items():
        print(f"  {section}:")
        for reason, count in counts.items():
            print(f"    {reason}: {count}")


if __name__ == "__main__":
    main()
