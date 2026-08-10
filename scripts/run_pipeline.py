#!/usr/bin/env python3
"""One-click pipeline runner for PSR-Bench.

Usage:
    python scripts/run_pipeline.py --data_root data/scannet/scans \\
                                   --output_dir output \\
                                   --max_scenes 300 \\
                                   --max_frames 5
"""

from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import datetime, timezone
import gc
import glob
import json
import logging
import os
import random
import re
import shutil
import sys
import time
import zlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.frame_selector import compute_frame_object_visibility
from src.scene_parser import (
    EXCLUDED_LABELS,
    _load_scene_geometry,
    _sample_surface_points_from_triangles,
    load_instance_mesh_data,
    load_scannet_label_map,
    parse_scene,
)
from src.support_graph import (
    enrich_scene_with_attachment,
    get_scene_attached_by,
    get_scene_attachment_graph,
    get_scene_support_chain_by,
    get_scene_support_chain_graph,
    has_nontrivial_attachment,
)
from src.qa_generator import (
    _object_bottom_hull_xy,
    _in_frame_surface_sample_subset,
    _instance_triangle_id_set,
    _mesh_visibility_stats_compat,
    _apply_attachment_surface_text_overrides,
    _annotate_cross_frame_questions,
    _clear_cross_frame_distance_metadata,
    _cross_frame_distance_priority_key,
    _prioritize_cross_frame_questions_by_distance,
    build_multi_frame_split_note,
    find_two_frame_split_v2,
    generate_all_questions,
    generate_cross_frame_questions,
    ReasoningFrameContext,
    SceneMotionCache,
    OcclusionDirectedSearchBudget,
    L2_OBJECT_MOVE_SEMANTICS_VERSION,
    L2_OBJECT_MOVE_OCCLUSION_RELATION_NEITHER,
    L2_OBJECT_MOVE_OCCLUSION_RELATION_QUERY_BY_REF,
    L2_OBJECT_MOVE_OCCLUSION_RELATION_REF_BY_QUERY,
    recompute_coordinate_rotation_agent_answer,
    MAX_OCCLUSION_OBJECTS_AUTO,
)
from src.relation_engine import camera_cardinal_direction, primary_direction_allocentric
from src.virtual_ops import apply_orbit_rotation
from src.referability_checks import (
    QUESTION_MENTION_FIELDS,
    build_question_referability_audit as _shared_build_question_referability_audit,
    collect_question_mentions as _shared_collect_question_mentions,
    coerce_object_id as _shared_coerce_object_id,
    normalize_attachment_pairs as _shared_normalize_attachment_pairs,
    normalize_label_to_object_ids as _shared_normalize_label_to_object_ids,
)
from src.auxiliary_path import MAX_AUXILIARY_FRAMES, VisualPoseGraph
from src.depth_auxiliary_path import (
    DEFAULT_MAX_CANDIDATE_POSES,
    DepthRouteGeometryCache,
    DepthVisualRedundancyEvaluator,
    find_depth_corridor_auxiliary_route,
)
from src.datasets.scannetpp_depth import DEFAULT_DEPTH_CACHE_SIZE
from src.datasets.scannet import ScanNetDataSource
from src.hybrid_auxiliary_path import HybridAuxiliaryRouter
from src.legacy_auxiliary_path import (
    find_geometric_auxiliary_route,
    object_group_center,
)
from src.quality_control import full_quality_pipeline, compute_statistics
from src.question_identity import (
    QUESTION_OBJECT_ID_FIELDS as _QUESTION_CAP_OBJECT_ID_FIELDS,
    question_pair_key as _shared_question_pair_key,
)
from src.utils.colmap_loader import (
    load_axis_alignment,
    load_scannet_depth_intrinsics,
    load_scannet_intrinsics,
    load_scannet_poses,
)
from src.utils.depth_occlusion import load_depth_image
from src.utils import RayCaster
from scripts.run_vlm_referability import (
    SCENE_STATUS_VERSION as REFERABILITY_SCENE_STATUS_VERSION,
    SCANNET_METADATA_SPLIT_FILES,
    SCANNETPP_METADATA_SPLIT_FILES,
    SEGMENTATION_EXTREME_NOISE_MIN_SCORE as QUESTION_DINOX_LOOSE_MIN_SCORE,
    SEGMENTATION_STRONG_MIN_SCORE as QUESTION_DINOX_STRONG_MIN_SCORE,
    _apply_attachment_pair_salvage_html_review,
    _attachment_human_review_surface_text_by_object_id,
    _call_dinox_joint_detection as _referability_call_dinox_joint_detection,
    _compute_mesh_mask_quality_for_object,
    _compute_topology_quality_for_object,
    _derive_final_referability_fields,
    _dedupe_detections_by_mask_iou,
    _frame_entry_has_consistent_final_fields,
    _repair_final_referability_fields,
    _select_best_detection_for_object_review,
    _strong_detection_min_area,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("pipeline")
DEFAULT_VLM_URL = "http://183.129.178.195:60029/v1"
EXPECTED_REFERABILITY_CACHE_VERSION = "20.0"
MANUAL_ATTACHMENT_CACHE_SCHEMA = "two_hop_attachment_salvage_v1"
MANUAL_ATTACHMENT_QUESTION_TYPES = ["L3_attachment_chain"]
PIPELINE_SCENE_STATUS_VERSION = 10
OBJECT_MOVE_OBJECT_CENTRIC_SEMANTICS_PROFILE = "strict-camera-facing-frozen-v1"
PIPELINE_RANDOM_SEED = 20240506
RAW_QUESTIONS_SCENE_CACHE_DIRNAME = "_raw_questions_scene_cache"
CROSS_FRAME_SCENE_CACHE_DIRNAME = "_cross_frame_scene_cache"
CROSS_FRAME_CHECKPOINT_VERSION = 6
L1_CANDIDATE_BUDGET_BY_SPLIT = {"val": 75, "train": 300}
L2_CANDIDATE_BUDGET_BY_SPLIT = {"val": 400, "train": 600}
L3_CANDIDATE_BUDGET_BY_SPLIT = {"val": 300, "train": 600}
QUESTION_REVIEW_MAX_RETRIES = 4
QUESTION_REVIEW_RETRY_DELAY_SECONDS = 2.0
QUESTION_REVIEW_MAX_TOKENS_PER_TARGET = 128
QUESTION_REVIEW_MAX_TOKENS_CAP = 1024

SINGLE_FRAME_PUBLIC_QUESTION_TYPES = frozenset({
    "L1_occlusion",
    "L2_object_remove",
    "L3_attachment_chain",
})
CROSS_FRAME_PUBLIC_QUESTION_TYPES = frozenset({
    "L1_direction_agent",
    "L1_distance",
    "L1_direction_object_centric",
    "L1_direction_allocentric",
    "L2_object_move_agent",
    "L2_object_move_distance",
    "L2_object_move_occlusion",
    "L2_object_move_object_centric",
    "L2_object_rotate_object_centric",
    "L2_object_move_allocentric",
    "L3_attachment_move",
    "L3_coordinate_rotation_agent",
    "L3_coordinate_rotation_object_centric",
    "L3_coordinate_rotation_allocentric",
})
CROSS_FRAME_MAX_MAIN_PAIRS_PER_SEMANTIC_QUESTION = 1
AUXILIARY_ROUTE_METHOD_VISUAL_POSE_GRAPH = "visual_pose_graph"
AUXILIARY_ROUTE_METHOD_LEGACY_GEOMETRIC = "legacy_geometric"
AUXILIARY_ROUTE_METHOD_HYBRID_GEOMETRIC_VISUAL = "hybrid_geometric_visual"
AUXILIARY_ROUTE_METHOD_DEPTH_CORRIDOR_GEOMETRIC = "depth_corridor_geometric"
DEPTH_ROUTE_ENFORCE_CAMERA_MOTION_HARD_LIMITS = False
DEPTH_ROUTE_DISABLED_CAMERA_MOTION_HARD_LIMITS = (
    "forward_angle_deg",
    "height_change_m",
    "local_perpendicular_m",
    "global_perpendicular_m",
    "degenerate_xy_translation_m",
)
AUXILIARY_ROUTE_METHODS = (
    AUXILIARY_ROUTE_METHOD_DEPTH_CORRIDOR_GEOMETRIC,
    AUXILIARY_ROUTE_METHOD_VISUAL_POSE_GRAPH,
    AUXILIARY_ROUTE_METHOD_LEGACY_GEOMETRIC,
    AUXILIARY_ROUTE_METHOD_HYBRID_GEOMETRIC_VISUAL,
)
VLM_API_KEY_ENV_NAMES = ("DASHSCOPE_API_KEY", "OPENAI_API_KEY")
PLACEHOLDER_VLM_API_KEY = "EMPTY"
QUESTION_REVIEW_CROP_PADDING_RATIO = 0.10
QUESTION_REVIEW_CROP_MIN_PADDING_PX = 12
QUESTION_REVIEW_CROP_MAX_PADDING_PX = 80
QUESTION_REVIEW_CROP_MIN_DIM_PX = 16
# Calibrated against ScanNet v2's 640x480 sensor; scaled per-frame so
# higher-resolution sensors (e.g. ScanNet++ iPhone at 1920x1440) get a
# proportionally larger floor instead of the same 800px on ~6x more area.
QUESTION_REVIEW_CROP_MIN_PROJECTED_AREA_RATIO = 800.0 / (640 * 480)
QUESTION_REVIEW_CROP_MIN_IN_FRAME_RATIO = 0.35
CROSS_FRAME_EXCLUSION_BBOX_IN_FRAME_RATIO_MIN = 0.20
CROSS_FRAME_TRUSTED_VISIBILITY_SOURCES = frozenset({
    "mesh_ray_refined",
    "mesh_ray_depth_refined",
})


def _scene_resource_requirements(
    *,
    single_frame_requested_types: list[str],
    cross_frame_requested_types: list[str],
    occlusion_backend: str,
) -> tuple[bool, bool]:
    single_frame_mesh_types = {"L1_occlusion", "L2_object_remove"}
    single_frame_needs_mesh = bool(
        set(single_frame_requested_types) & single_frame_mesh_types
    )
    needs_mesh_resources = (
        occlusion_backend in ("depth", "mesh_ray")
        and (
            single_frame_needs_mesh
            or "L2_object_move_occlusion" in cross_frame_requested_types
        )
    )
    needs_instance_mesh_data = (
        needs_mesh_resources
        or "L1_distance" in cross_frame_requested_types
    )
    return needs_mesh_resources, needs_instance_mesh_data


def question_review_crop_min_projected_area_px(width: int, height: int) -> float:
    return QUESTION_REVIEW_CROP_MIN_PROJECTED_AREA_RATIO * float(width) * float(height)
QUESTION_MENTION_FALLBACK_FIELDS = QUESTION_MENTION_FIELDS
REFERABLE_OCCLUSION_VETO_DENSE_BASE_SAMPLE_COUNT = 512
REFERABLE_OCCLUSION_VETO_DENSE_BASE_PROJECTED_AREA_PX = 400.0
REFERABLE_OCCLUSION_VETO_DENSE_MAX_SAMPLE_COUNT = 4096
REFERABLE_OCCLUSION_VETO_MIN_VISIBLE_RATIO = 0.35
REFERABLE_OCCLUSION_VETO_DENSE_CHUNK_SIZE = 64
_GENERATE_ALL_QUESTIONS_ATTACHMENT_SURFACE_COMPAT_WARNING_EMITTED = False

# ---- generalized two-frame split (route-continuity beyond object_move_occlusion) ----
#
# Maps each eligible question type to which of its already-present id fields form
# group A (-> frame 1 / image_name) vs group B (-> frame 2), plus optional "bonus"
# ids (soft preference only, e.g. obj_face -- never required to land in either frame).
# attachment_chain and object_remove are deliberately absent and always stay single-
# frame: attachment_chain's support-chain answer needs the pair co-visible in one
# frame, and object_remove was decided to stay single-frame rather than attempt a
# split. Every other type here REQUIRES a valid mutually-exclusive split -- if
# find_two_frame_split_v2 can't find one, the question is dropped entirely rather
# than falling back to the single shared image_name (see _apply_two_frame_split).
TWO_FRAME_SPLIT_ID_FIELDS: dict[str, dict[str, list[str]]] = {
    "object_move_occlusion": {
        "group_a": ["moved_obj_id", "query_obj_id"],
        "group_b": ["obj_ref_id"],
    },
    "object_move_agent": {
        "group_a": ["moved_obj_id", "query_obj_id"],
        "group_b": ["obj_c_id"],
    },
    "object_move_distance": {
        "group_a": ["moved_obj_id", "query_obj_id"],
        "group_b": ["obj_c_id"],
    },
    "object_move_object_centric": {
        "group_a": ["moved_obj_id", "query_obj_id"],
        "group_b": ["obj_ref_id"],
    },
    "object_move_allocentric": {
        "group_a": ["moved_obj_id", "query_obj_id"],
        "group_b": ["obj_ref_id"],
    },
    "object_rotate_object_centric": {
        "group_a": ["moved_obj_id", "query_obj_id"],
        "group_b": ["obj_ref_id"],
        "bonus": ["obj_face_id"],
    },
    # Covers all 3 reference_frame variants (agent/object_centric/allocentric) --
    # they share the "type": "attachment_move" tag and the same object roles.
    "attachment_move": {
        "group_a": ["root_id", "query_obj_id"],
        "group_b": ["obj_ref_id"],
    },
    "coordinate_rotation_agent": {
        "group_a": ["obj_a_id"],
        "group_b": ["obj_b_id"],
    },
    "coordinate_rotation_allocentric": {
        "group_a": ["obj_a_id"],
        "group_b": ["obj_b_id"],
    },
    "coordinate_rotation_object_centric": {
        "group_a": ["obj_ref_id"],
        "group_b": ["obj_target_id"],
        "bonus": ["obj_face_id"],
    },
}

# coordinate_rotation_agent/_allocentric use frame_1's camera as both the rotation
# pivot and the front/back/left/right (or cardinal-text) reference, so a GT/text
# recompute is needed whenever _apply_two_frame_split reassigns frame_1 away from
# the scene's original shared camera. coordinate_rotation_object_centric's GT only
# depends on object-defined facing geometry, never the camera, so it needs no
# recompute branch.
_COORDINATE_ROTATION_CAMERA_ANCHORED_TYPES = frozenset(
    {"coordinate_rotation_agent", "coordinate_rotation_allocentric"}
)


def _recompute_coordinate_rotation_gt_for_new_anchor(
    q: dict[str, Any],
    *,
    objects_by_id: dict[int, dict[str, Any]],
    all_poses: dict[str, Any],
) -> None:
    """Re-derive answer/options (agent) or the camera_cardinal text (allocentric)
    after _apply_two_frame_split reassigned frame_1 to a real camera other than the
    scene's original shared one.
    """
    new_pose = all_poses.get(q.get("image_name"))
    if new_pose is None:
        return

    if q.get("type") == "coordinate_rotation_agent":
        obj_a = objects_by_id.get(int(q["obj_a_id"]))
        obj_b = objects_by_id.get(int(q["obj_b_id"]))
        if obj_a is None or obj_b is None:
            return
        new_dir, options, answer_letter = recompute_coordinate_rotation_agent_answer(
            obj_a, obj_b, q["rotation_angle"], new_pose,
        )
        q["options"] = options
        q["answer"] = answer_letter
        q["correct_value"] = new_dir
        q["new_direction"] = new_dir
        q["relation_unchanged"] = q.get("old_direction") == new_dir
    elif q.get("type") == "coordinate_rotation_allocentric":
        old_cardinal = q.get("camera_cardinal")
        new_cardinal = camera_cardinal_direction(new_pose)
        if old_cardinal and new_cardinal != old_cardinal and isinstance(q.get("question"), str):
            q["question"] = re.sub(
                rf"\b{re.escape(str(old_cardinal))}\b", new_cardinal, q["question"]
            )
        q["camera_cardinal"] = new_cardinal


def _resolve_objects_for_ids(
    id_values: list[Any],
    objects_by_id: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Look up distinct, present object dicts for a list of (possibly duplicate/None) ids."""
    resolved: list[dict[str, Any]] = []
    seen: set[int] = set()
    for raw_id in id_values:
        if raw_id is None:
            continue
        try:
            obj_id = int(raw_id)
        except (TypeError, ValueError):
            continue
        if obj_id in seen:
            continue
        obj = objects_by_id.get(obj_id)
        if obj is None:
            continue
        seen.add(obj_id)
        resolved.append(obj)
    return resolved


def _group_is_referable_in(
    entry: dict | None,
    group_ids: list[int],
    *,
    pool: str = "ordinary",
) -> bool:
    """True iff every group id belongs to the requested frame candidate pool."""
    if not entry or not group_ids:
        return False
    ordinary_ids = set(
        int(obj_id) for obj_id in (entry.get("referable_object_ids") or [])
    )
    attachment_ids = set(
        int(obj_id)
        for obj_id in (entry.get("attachment_referable_object_ids") or [])
    )
    if pool == "attachment":
        referable_ids = attachment_ids
    elif pool == "any":
        referable_ids = ordinary_ids | attachment_ids
    else:
        referable_ids = ordinary_ids
    return all(int(obj_id) in referable_ids for obj_id in group_ids)


def _referable_frame_names_for_group(
    scene_frames: dict[str, dict] | None,
    group_ids: list[int],
    *,
    pool: str = "ordinary",
) -> set[str]:
    """Frame names where *group_ids* are all referable and the frame is usable.

    This is the positive gate for picking a two-frame-split's frame_a/frame_b: only
    frames that scannetpp_flash/run_vlm_referability.py actually reviewed (clarity
    pass + per-object referability) and kept as usable are eligible -- frames that
    were rejected for blur/etc (frame_usable=False) or never reviewed at all (absent
    from scene_frames) are never candidates, no matter what pure geometry says.
    """
    names: set[str] = set()
    for image_name, entry in (scene_frames or {}).items():
        if not isinstance(entry, dict):
            continue
        if entry.get("frame_usable") is False:
            continue
        if _group_is_referable_in(entry, group_ids, pool=pool):
            names.add(image_name)
    return names


def _reasoning_frame_pair_violates_referability(
    *,
    frame_a_name: str,
    frame_b_name: str,
    group_a_ids: list[int],
    group_b_ids: list[int],
    scene_frames: dict[str, dict] | None,
) -> bool:
    """True if either final reasoning frame also refers-in the OTHER frame's group,
    per VLM referability_cache data (when available for that frame).

    This is a negative/rejection check on top of the positive gate in
    _referable_frame_names_for_group: ANY signal that the other group's object is
    also uniquely identifiable in a chosen frame contradicts the question's "this
    photo doesn't show it" premise.

    Only evaluated against the two frames find_two_frame_split_v2 actually chose --
    not a per-candidate filter during its search. No-ops (returns False) when
    scene_frames is empty or a frame has no cache entry: referability_cache only
    covers the narrow subset of frames actually run through run_vlm_referability.py,
    and the geometric exclusivity check (_group_visible_at_all, in
    find_two_frame_split_v2) already guards every candidate, referability-audited or
    not, so absence of data here is not a gap.
    """
    if not scene_frames:
        return False

    if _group_is_referable_in(
        scene_frames.get(frame_b_name), group_a_ids, pool="any"
    ):
        return True
    if _group_is_referable_in(
        scene_frames.get(frame_a_name), group_b_ids, pool="any"
    ):
        return True
    return False


def _apply_two_frame_split(
    q: dict[str, Any],
    *,
    objects_by_id: dict[int, dict[str, Any]],
    all_poses: dict[str, Any] | None,
    color_intrinsics: Any,
    camera_pose: Any,
    scene_frames: dict[str, dict] | None = None,
) -> bool:
    """Split a question's referenced objects across two real frames, in place.

    Returns True if the question should be kept, False if it must be dropped.

    Always returns True (no-op, keeps the single shared image_name) if the type isn't
    in TWO_FRAME_SPLIT_ID_FIELDS or required pose/intrinsics infra is missing -- those
    types/cases were never meant to split. For every type that IS in
    TWO_FRAME_SPLIT_ID_FIELDS, a valid mutually-exclusive frame split is required: if
    either group resolves empty, or no valid split + continuity chain can be found,
    this returns False and the caller must drop the question (its "first photo shows
    A, last photo shows B" premise cannot be satisfied by any real frame pair).

    scene_frames (the scene's referability_cache entries, keyed by image_name) is now
    a REQUIRED positive gate, not just an optional safeguard: frame_a and frame_b are
    the two frames the question is actually about, so each is only a valid candidate
    if scannetpp_flash/run_vlm_referability.py reviewed it, kept it as usable, and
    found that frame's own group of objects referable there (see
    _referable_frame_names_for_group). Pure geometric in-frame projection is not
    enough on its own -- a bbox can project fully in-frame while the object is
    actually occluded or the frame is too blurry to make out, which referability
    review already screens for. Only the auxiliary bridging/continuity frames may
    still come from the full raw scene pose set (all_poses), since those are never
    directly reasoned about by the model. Once a candidate split is found, it's also
    rejected if VLM referability data says the OTHER group is referable (visible +
    uniquely labeled) in either chosen frame -- see
    _reasoning_frame_pair_violates_referability.
    """
    split_spec = TWO_FRAME_SPLIT_ID_FIELDS.get(q.get("type"))
    if split_spec is None or color_intrinsics is None or not all_poses:
        return True

    group_a_objects = _resolve_objects_for_ids(
        [q.get(field) for field in split_spec["group_a"]], objects_by_id
    )
    group_b_objects = _resolve_objects_for_ids(
        [q.get(field) for field in split_spec["group_b"]], objects_by_id
    )
    if not group_a_objects or not group_b_objects:
        return False
    bonus_objects = _resolve_objects_for_ids(
        [q.get(field) for field in split_spec.get("bonus", [])], objects_by_id
    )

    group_a_ids = [int(o["id"]) for o in group_a_objects]
    group_b_ids = [int(o["id"]) for o in group_b_objects]
    group_a_pool = "ordinary"
    group_b_pool = "ordinary"
    if _question_uses_attachment_referability(q):
        group_a_pool = "attachment"
    frame_a_pool = {
        name: all_poses[name]
        for name in _referable_frame_names_for_group(
            scene_frames, group_a_ids, pool=group_a_pool
        )
        if name in all_poses
    }
    frame_b_pool = {
        name: all_poses[name]
        for name in _referable_frame_names_for_group(
            scene_frames, group_b_ids, pool=group_b_pool
        )
        if name in all_poses
    }
    if not frame_a_pool or not frame_b_pool:
        return False

    split = find_two_frame_split_v2(
        group_a_objects=group_a_objects,
        group_b_objects=group_b_objects,
        all_poses=all_poses,
        color_intrinsics=color_intrinsics,
        preferred_camera_pose=camera_pose,
        bonus_objects=bonus_objects,
        frame_a_candidate_pool=frame_a_pool,
        frame_b_candidate_pool=frame_b_pool,
    )
    if split is None:
        return False

    frame_a_name, frame_b_name, chain = split
    if _reasoning_frame_pair_violates_referability(
        frame_a_name=frame_a_name,
        frame_b_name=frame_b_name,
        group_a_ids=group_a_ids,
        group_b_ids=group_b_ids,
        scene_frames=scene_frames,
    ):
        return False
    q["image_name"] = frame_a_name
    q["reasoning_frame_2"] = frame_b_name
    q["auxiliary_image_names"] = chain
    q["object_frame_groups"] = {
        "frame_1": group_a_ids,
        "frame_2": group_b_ids,
    }
    # Lead with the note (not append it) so the model learns before reading the
    # question that it's getting a photo series, and which named objects are in
    # the first vs. last frame -- every split type gets this, including
    # coordinate_rotation_agent/_allocentric, whose own template text only names
    # the rotation-pivot camera, not which object ends up in which frame.
    if isinstance(q.get("question"), str):
        note = build_multi_frame_split_note(group_a_objects, group_b_objects)
        q["question"] = f"{note} {q['question']}"
    return True


def _build_reasoning_frame_contexts(
    *,
    frames: list[dict[str, object]],
    scene_frames: dict[str, dict],
    poses: dict[str, object],
    scene_objects: list[dict],
    color_intrinsics: object,
) -> list[ReasoningFrameContext]:
    objects_by_id = {int(obj["id"]): obj for obj in scene_objects}

    def _cached_bbox_ratios(entry: dict, candidate_ids: list[int]) -> dict[int, float]:
        ratios: dict[int, float] = {}

        def _ingest(container: object) -> None:
            if isinstance(container, dict):
                items = container.items()
            elif isinstance(container, list):
                items = ((None, value) for value in container)
            else:
                return
            for key, value in items:
                if not isinstance(value, dict):
                    continue
                try:
                    obj_id = int(value.get("obj_id", key))
                    ratio = float(value["bbox_in_frame_ratio"])
                except (KeyError, TypeError, ValueError):
                    continue
                if obj_id in candidate_ids:
                    ratios[obj_id] = ratio

        _ingest(entry.get("visibility_audit_by_object_id"))
        _ingest(entry.get("object_reviews"))
        return ratios

    contexts: list[ReasoningFrameContext] = []
    for frame in frames:
        image_name = str(frame.get("image_name", "")).strip()
        entry = scene_frames.get(image_name)
        pose = poses.get(image_name)
        if not image_name or not isinstance(entry, dict) or pose is None:
            continue
        if entry.get("frame_usable") is False:
            continue
        visibility_source = str(entry.get("candidate_visibility_source", "")).strip()
        if visibility_source not in CROSS_FRAME_TRUSTED_VISIBILITY_SOURCES:
            continue
        candidate_ids = _normalize_object_ids(entry.get("candidate_visible_object_ids"))
        bbox_ratios = _cached_bbox_ratios(entry, candidate_ids)
        missing_ratio_ids = [obj_id for obj_id in candidate_ids if obj_id not in bbox_ratios]
        if missing_ratio_ids and color_intrinsics is not None:
            fallback_visibility = compute_frame_object_visibility(
                objects=[
                    objects_by_id[obj_id]
                    for obj_id in missing_ratio_ids
                    if obj_id in objects_by_id
                ],
                pose=pose,
                color_intrinsics=color_intrinsics,
                image_path=None,
                depth_image=None,
                depth_intrinsics=None,
            )
            for obj_id, meta in fallback_visibility.items():
                try:
                    bbox_ratios[int(obj_id)] = float(
                        meta.get("bbox_in_frame_ratio", 0.0) or 0.0
                    )
                except (TypeError, ValueError):
                    continue
        if any(obj_id not in bbox_ratios for obj_id in candidate_ids):
            continue
        cross_frame_visible_ids = frozenset(
            obj_id
            for obj_id in candidate_ids
            if bbox_ratios[obj_id] >= CROSS_FRAME_EXCLUSION_BBOX_IN_FRAME_RATIO_MIN
        )
        regular_ids = frozenset(_normalize_object_ids(entry.get("referable_object_ids")))
        attachment_ids = frozenset(
            _normalize_object_ids(entry.get("attachment_referable_object_ids"))
        )
        if not regular_ids and not attachment_ids:
            continue
        contexts.append(
            ReasoningFrameContext(
                image_name=image_name,
                camera_pose=pose,
                regular_referable_ids=regular_ids,
                attachment_referable_ids=attachment_ids,
                cross_frame_visible_ids=cross_frame_visible_ids,
                cache_entry=entry,
            )
        )
    return contexts


def _restrict_context_for_semantic_exclusivity(
    own: ReasoningFrameContext,
    other: ReasoningFrameContext,
) -> ReasoningFrameContext:
    return ReasoningFrameContext(
        image_name=own.image_name,
        camera_pose=own.camera_pose,
        regular_referable_ids=frozenset(own.regular_referable_ids - other.any_referable_ids),
        attachment_referable_ids=frozenset(
            own.attachment_referable_ids - other.any_referable_ids
        ),
        cross_frame_visible_ids=own.cross_frame_visible_ids,
        cache_entry=own.cache_entry,
    )


def _cross_frame_semantic_key(question: dict[str, object]) -> tuple[object, ...]:
    groups = question.get("object_frame_groups")
    groups = groups if isinstance(groups, dict) else {}
    delta = question.get("delta")
    normalized_delta = tuple(
        round(float(value), 4) for value in delta
    ) if isinstance(delta, list) else ()
    return (
        question.get("type"),
        question.get("reference_frame"),
        question.get("cross_frame_layout"),
        tuple(groups.get("frame_1", [])),
        tuple(groups.get("frame_2", [])),
        normalized_delta,
        question.get("rotation_angle"),
        question.get("rotation_direction"),
        question.get("correct_value"),
    )


def _retain_best_cross_frame_views(
    questions: list[dict],
    *,
    max_views: int = CROSS_FRAME_MAX_MAIN_PAIRS_PER_SEMANTIC_QUESTION,
) -> list[dict]:
    grouped: dict[tuple[object, ...], list[dict]] = defaultdict(list)
    for question in questions:
        grouped[_cross_frame_semantic_key(question)].append(question)
    kept: list[dict] = []
    ordered_groups = sorted(
        grouped.items(),
        key=lambda item: (
            _cross_frame_distance_priority_key(item[1][0])[:3],
            repr(item[0]),
        ),
    )
    for _key, group in ordered_groups:
        ranked = sorted(
            group,
            key=lambda question: (
                float(question.get("_cross_frame_pair_score", float("inf"))),
                str(question.get("image_name", "")),
                str(question.get("reasoning_frame_2", "")),
            ),
        )[:max(1, int(max_views))]
        for question in ranked:
            question.pop("_cross_frame_pair_score", None)
            kept.append(question)
    return kept


def _cluster_attachment_reference_questions(
    pair_questions: list[dict],
    *,
    objects_by_id: dict[int, dict],
    radius_m: float,
) -> tuple[list[dict], int, int]:
    """Drop nearby attachment references with identical complete outcomes."""
    if radius_m <= 0.0:
        return pair_questions, 0, 0

    questions_by_chain: dict[tuple[int, int], dict[int, list[dict]]] = (
        defaultdict(lambda: defaultdict(list))
    )
    for question in pair_questions:
        if question.get("type") != "attachment_move":
            continue
        try:
            chain_key = (
                int(question["root_id"]),
                int(question["query_obj_id"]),
            )
            ref_id = int(question["obj_ref_id"])
        except (KeyError, TypeError, ValueError):
            continue
        questions_by_chain[chain_key][ref_id].append(question)

    dropped_ref_ids_by_chain: dict[tuple[int, int], set[int]] = {}
    for chain_key, questions_by_ref in questions_by_chain.items():
        representatives: list[tuple[np.ndarray, tuple[object, ...]]] = []
        dropped_ref_ids: set[int] = set()
        for ref_id, ref_questions in questions_by_ref.items():
            ref_obj = objects_by_id.get(ref_id)
            if ref_obj is None:
                continue
            try:
                center_xy = np.asarray(ref_obj["center"], dtype=np.float64)[:2]
                signatures = tuple(sorted(
                    (
                        str(question.get("reference_frame", "")),
                        str(question.get("old_correct_value", "")),
                        str(question.get("new_correct_value", "")),
                        tuple(
                            round(float(value), 4)
                            for value in question.get("delta", [])
                        ),
                    )
                    for question in ref_questions
                ))
            except (KeyError, TypeError, ValueError):
                continue
            if center_xy.shape != (2,) or not np.all(np.isfinite(center_xy)):
                continue
            duplicate = any(
                signatures == representative_signatures
                and float(np.linalg.norm(center_xy - representative_center))
                <= radius_m
                for representative_center, representative_signatures in representatives
            )
            if duplicate:
                dropped_ref_ids.add(ref_id)
                continue
            representatives.append((center_xy, signatures))
        if dropped_ref_ids:
            dropped_ref_ids_by_chain[chain_key] = dropped_ref_ids

    if not dropped_ref_ids_by_chain:
        return pair_questions, 0, 0

    clustered: list[dict] = []
    dropped_question_count = 0
    for question in pair_questions:
        if question.get("type") != "attachment_move":
            clustered.append(question)
            continue
        try:
            chain_key = (
                int(question["root_id"]),
                int(question["query_obj_id"]),
            )
            ref_id = int(question["obj_ref_id"])
        except (KeyError, TypeError, ValueError):
            clustered.append(question)
            continue
        if ref_id in dropped_ref_ids_by_chain.get(chain_key, set()):
            dropped_question_count += 1
            continue
        clustered.append(question)
    dropped_ref_count = sum(map(len, dropped_ref_ids_by_chain.values()))
    return clustered, dropped_ref_count, dropped_question_count


def _is_object_move_occlusion_question(question: dict) -> bool:
    return str(question.get("type", "")).strip() == "object_move_occlusion"


def _is_positive_object_move_occlusion(question: dict) -> bool:
    return (
        _is_object_move_occlusion_question(question)
        and str(question.get("new_pairwise_occlusion_relation", "")).strip()
        in {
            L2_OBJECT_MOVE_OCCLUSION_RELATION_QUERY_BY_REF,
            L2_OBJECT_MOVE_OCCLUSION_RELATION_REF_BY_QUERY,
        }
    )


def _has_strict_object_move_occlusion_frame_roles(question: dict) -> bool:
    if not _is_object_move_occlusion_question(question):
        return True
    groups = question.get("object_frame_groups")
    if not isinstance(groups, dict):
        return False
    try:
        frame_1_ids = [int(value) for value in groups.get("frame_1", [])]
        frame_2_ids = [int(value) for value in groups.get("frame_2", [])]
        expected_frame_1 = list(dict.fromkeys([
            int(question["moved_obj_id"]),
            int(question["query_obj_id"]),
        ]))
        expected_frame_2 = [int(question["obj_ref_id"])]
    except (KeyError, TypeError, ValueError):
        return False
    relation = str(question.get("new_pairwise_occlusion_relation", "")).strip()
    return (
        frame_1_ids == expected_frame_1
        and frame_2_ids == expected_frame_2
        and set(frame_1_ids).isdisjoint(frame_2_ids)
        and relation in {
            L2_OBJECT_MOVE_OCCLUSION_RELATION_QUERY_BY_REF,
            L2_OBJECT_MOVE_OCCLUSION_RELATION_REF_BY_QUERY,
            L2_OBJECT_MOVE_OCCLUSION_RELATION_NEITHER,
        }
        and bool(str(question.get("image_name", "")).strip())
        and bool(str(question.get("reasoning_frame_2", "")).strip())
        and str(question.get("image_name")) != str(question.get("reasoning_frame_2"))
    )


def _prioritize_object_move_occlusion_positives(questions: list[dict]) -> list[dict]:
    positives = [question for question in questions if _is_positive_object_move_occlusion(question)]
    other = [
        question for question in questions
        if not _is_object_move_occlusion_question(question)
    ]
    negatives = [
        question for question in questions
        if _is_object_move_occlusion_question(question)
        and not _is_positive_object_move_occlusion(question)
    ]
    return positives + other + negatives


def _balance_scene_object_move_occlusion_negatives(
    questions: list[dict],
) -> tuple[list[dict], dict[str, int]]:
    valid_questions = [
        question for question in questions
        if _has_strict_object_move_occlusion_frame_roles(question)
    ]
    positives = [
        question for question in valid_questions
        if _is_positive_object_move_occlusion(question)
    ]
    negatives = [
        question for question in valid_questions
        if _is_object_move_occlusion_question(question)
        and str(question.get("new_pairwise_occlusion_relation", "")).strip()
        == L2_OBJECT_MOVE_OCCLUSION_RELATION_NEITHER
    ]
    negative_limit = len(positives) if positives else 3

    def _negative_key(question: dict) -> tuple[float, float, str, str, int, int]:
        blocking = max(
            float(question.get("new_query_blocking_ratio", 0.0) or 0.0),
            float(question.get("new_ref_blocking_ratio", 0.0) or 0.0),
        )
        score = question.get("directed_search_score")
        predicted_coverage = (
            float(score.get("predicted_coverage", 0.0) or 0.0)
            if isinstance(score, dict) else 0.0
        )
        return (
            -blocking,
            -predicted_coverage,
            str(question.get("image_name", "")),
            str(question.get("reasoning_frame_2", "")),
            int(question.get("query_obj_id", -1)),
            int(question.get("obj_ref_id", -1)),
        )

    kept_negative_ids = {
        id(question)
        for question in sorted(negatives, key=_negative_key)[:negative_limit]
    }
    balanced = [
        question for question in valid_questions
        if not _is_object_move_occlusion_question(question)
        or _is_positive_object_move_occlusion(question)
        or id(question) in kept_negative_ids
    ]
    return balanced, {
        "positive_count": len(positives),
        "negative_input_count": len(negatives),
        "negative_kept_count": min(len(negatives), negative_limit),
        "negative_dropped_count": max(0, len(negatives) - negative_limit),
        "invalid_frame_role_dropped_count": len(questions) - len(valid_questions),
    }


def _object_move_occlusion_balance_key(
    question: dict,
) -> tuple[float, float, str, str, str, int, int, int]:
    blocking = max(
        float(question.get("new_query_blocking_ratio", 0.0) or 0.0),
        float(question.get("new_ref_blocking_ratio", 0.0) or 0.0),
    )
    score = question.get("directed_search_score")
    predicted_coverage = (
        float(score.get("predicted_coverage", 0.0) or 0.0)
        if isinstance(score, dict) else 0.0
    )
    return (
        -blocking,
        -predicted_coverage,
        str(question.get("scene_id", "")),
        str(question.get("image_name", "")),
        str(question.get("reasoning_frame_2", "")),
        int(question.get("moved_obj_id", -1)),
        int(question.get("query_obj_id", -1)),
        int(question.get("obj_ref_id", -1)),
    )


def _balance_global_object_move_occlusion_three_way(
    questions: list[dict],
) -> tuple[list[dict], dict[str, int | bool]]:
    valid_questions = [
        question for question in questions
        if _has_strict_object_move_occlusion_frame_roles(question)
    ]
    occlusion_questions = [
        question for question in valid_questions
        if _is_object_move_occlusion_question(question)
    ]
    nonself_positive = [
        question for question in occlusion_questions
        if _is_positive_object_move_occlusion(question)
        and int(question.get("moved_obj_id", -1))
        != int(question.get("query_obj_id", -1))
    ]
    self_positive = [
        question for question in occlusion_questions
        if _is_positive_object_move_occlusion(question)
        and int(question.get("moved_obj_id", -1))
        == int(question.get("query_obj_id", -1))
    ]
    neither = [
        question for question in occlusion_questions
        if str(question.get("new_pairwise_occlusion_relation", "")).strip()
        == L2_OBJECT_MOVE_OCCLUSION_RELATION_NEITHER
    ]
    target_per_class = min(
        len(nonself_positive),
        len(self_positive),
        len(neither),
    )
    balance_applied = target_per_class > 0
    if balance_applied:
        kept_occlusion_ids = {
            id(question)
            for group in (nonself_positive, self_positive, neither)
            for question in sorted(group, key=_object_move_occlusion_balance_key)[
                :target_per_class
            ]
        }
        balanced = [
            question for question in valid_questions
            if not _is_object_move_occlusion_question(question)
            or id(question) in kept_occlusion_ids
        ]
    else:
        balanced = valid_questions
    return balanced, {
        "balance_applied": balance_applied,
        "target_per_class": target_per_class,
        "nonself_positive_input_count": len(nonself_positive),
        "self_positive_input_count": len(self_positive),
        "neither_input_count": len(neither),
        "nonself_positive_kept_count": (
            target_per_class if balance_applied else len(nonself_positive)
        ),
        "self_positive_kept_count": (
            target_per_class if balance_applied else len(self_positive)
        ),
        "neither_kept_count": target_per_class if balance_applied else len(neither),
        "invalid_frame_role_dropped_count": len(questions) - len(valid_questions),
    }


def _question_dinox_mask_bounds(mask: object) -> list[int] | None:
    if not isinstance(mask, np.ndarray):
        return None
    mask_bool = np.asarray(mask, dtype=bool)
    ys, xs = np.where(mask_bool)
    if xs.size <= 0 or ys.size <= 0:
        return None
    return [
        int(xs.min()),
        int(xs.max()) + 1,
        int(ys.min()),
        int(ys.max()) + 1,
    ]


def _call_generate_all_questions_compat(**kwargs):
    """Tolerate older generate_all_questions signatures during mixed deploys."""
    global _GENERATE_ALL_QUESTIONS_ATTACHMENT_SURFACE_COMPAT_WARNING_EMITTED

    compat_kwargs = dict(kwargs)
    while True:
        try:
            return generate_all_questions(**compat_kwargs)
        except TypeError as exc:
            message = str(exc)
            if "attachment_object_surface_text_by_id" in message:
                if "attachment_object_surface_text_by_id" not in compat_kwargs:
                    raise
                if not _GENERATE_ALL_QUESTIONS_ATTACHMENT_SURFACE_COMPAT_WARNING_EMITTED:
                    logger.warning(
                        "generate_all_questions compatibility mode: runtime does not support "
                        "attachment_object_surface_text_by_id; attachment naming overrides "
                        "will be skipped for this run"
                    )
                    _GENERATE_ALL_QUESTIONS_ATTACHMENT_SURFACE_COMPAT_WARNING_EMITTED = True
                compat_kwargs.pop("attachment_object_surface_text_by_id", None)
                continue
            if "attachment_chain_role_overrides" in message:
                if "attachment_chain_role_overrides" not in compat_kwargs:
                    raise
                compat_kwargs.pop("attachment_chain_role_overrides", None)
                continue
            if "attachment_chain_role_override" in message:
                if "attachment_chain_role_override" not in compat_kwargs:
                    raise
                compat_kwargs.pop("attachment_chain_role_override", None)
                continue
            if "attachment_priority_pairs" in message:
                if "attachment_priority_pairs" not in compat_kwargs:
                    raise
                compat_kwargs.pop("attachment_priority_pairs", None)
                continue
            if "attachment_referable_pairs" in message:
                if "attachment_referable_pairs" not in compat_kwargs:
                    raise
                compat_kwargs.pop("attachment_referable_pairs", None)
                continue
            if "question_type_budgets" in message:
                if "question_type_budgets" not in compat_kwargs:
                    raise
                compat_kwargs.pop("question_type_budgets", None)
                continue
            if "max_occlusion_objects" in message:
                if "max_occlusion_objects" not in compat_kwargs:
                    raise
                compat_kwargs.pop("max_occlusion_objects", None)
                continue
            if "max_move_sources" in message:
                if "max_move_sources" not in compat_kwargs:
                    raise
                compat_kwargs.pop("max_move_sources", None)
                continue
            raise


def _manual_attachment_graph_for_scene(
    referability_cache: dict | None,
    scene_id: str,
) -> dict[int, list[int]] | None:
    """Return a human salvage graph when the cache supplies one."""
    if not isinstance(referability_cache, dict):
        return None
    raw_graphs = referability_cache.get("manual_attachment_graph")
    if not isinstance(raw_graphs, dict):
        return None
    raw_graph = raw_graphs.get(scene_id)
    if isinstance(raw_graph, dict) and "edges" in raw_graph:
        raw_graph = raw_graph.get("edges")
    graph: dict[int, list[int]] = {}
    if isinstance(raw_graph, dict):
        for parent_id, child_ids in raw_graph.items():
            try:
                parent = int(parent_id)
            except (TypeError, ValueError):
                continue
            if not isinstance(child_ids, list):
                continue
            children: list[int] = []
            for child_id in child_ids:
                try:
                    child = int(child_id)
                except (TypeError, ValueError):
                    continue
                if child != parent and child not in children:
                    children.append(child)
            if children:
                graph[parent] = sorted(children)
    elif isinstance(raw_graph, list):
        for edge in raw_graph:
            if not isinstance(edge, (list, tuple)) or len(edge) < 2:
                continue
            try:
                parent, child = int(edge[0]), int(edge[1])
            except (TypeError, ValueError):
                continue
            if parent == child:
                continue
            graph.setdefault(parent, [])
            if child not in graph[parent]:
                graph[parent].append(child)
        graph = {parent: sorted(children) for parent, children in graph.items()}
    return graph or None


def _is_manual_attachment_cache(cache: dict | None) -> bool:
    return bool(
        isinstance(cache, dict)
        and str(cache.get("schema", "")).strip() == MANUAL_ATTACHMENT_CACHE_SCHEMA
    )


def _has_manual_attachment_overrides(cache: dict | None) -> bool:
    if not isinstance(cache, dict):
        return False
    if _is_manual_attachment_cache(cache):
        return True
    scene_ids = cache.get("manual_attachment_scene_ids")
    return isinstance(scene_ids, list) and bool(scene_ids)


def _manual_attachment_role_records_for_frame(
    referability_entry: dict[str, object] | None,
) -> list[dict[str, dict[str, object]]]:
    if not isinstance(referability_entry, dict):
        return []
    raw_role_sets = referability_entry.get("manual_attachment_role_sets")
    if isinstance(raw_role_sets, list) and raw_role_sets:
        candidates = raw_role_sets
    else:
        candidates = [referability_entry.get("manual_attachment_roles")]
    normalized: list[dict[str, dict[str, object]]] = []
    for raw_roles in candidates:
        if not isinstance(raw_roles, dict):
            continue
        roles: dict[str, dict[str, object]] = {}
        ids: list[int] = []
        valid = True
        for role in ("moved", "child", "grandchild", "contrast"):
            value = raw_roles.get(role)
            if isinstance(value, dict):
                raw_id = value.get("id")
                label = str(value.get("label", "")).strip()
            else:
                raw_id = value
                label = ""
            try:
                obj_id = int(raw_id)
            except (TypeError, ValueError):
                valid = False
                break
            ids.append(obj_id)
            roles[role] = {"id": obj_id, "label": label}
        if valid and len(set(ids)) == len(ids):
            normalized.append(roles)
    return normalized


def _manual_attachment_role_sets_for_frame(
    referability_entry: dict[str, object] | None,
) -> list[dict[str, int]]:
    return [
        {role: int(value["id"]) for role, value in role_set.items()}
        for role_set in _manual_attachment_role_records_for_frame(referability_entry)
    ]


def _manual_attachment_roles_for_frame(
    referability_entry: dict[str, object] | None,
) -> dict[str, int] | None:
    role_sets = _manual_attachment_role_sets_for_frame(referability_entry)
    return role_sets[0] if role_sets else None


def _manual_attachment_surface_text_by_object_id(
    referability_entry: dict[str, object] | None,
) -> dict[int, str]:
    if not isinstance(referability_entry, dict):
        return {}
    surface_text_by_id: dict[int, str] = {}
    for role_set in _manual_attachment_role_records_for_frame(referability_entry):
        for value in role_set.values():
            obj_id = int(value["id"])
            label = str(value.get("label", "")).strip()
            if label:
                surface_text_by_id[obj_id] = label
    return surface_text_by_id


def _attachment_surface_text_by_object_id(
    referability_entry: dict[str, object] | None,
) -> dict[int, str]:
    if not isinstance(referability_entry, dict):
        return {}
    surface_text_by_id = _attachment_human_review_surface_text_by_object_id(
        referability_entry.get("attachment_human_review_cards")
    )
    surface_text_by_id.update(
        _manual_attachment_surface_text_by_object_id(referability_entry)
    )
    return dict(sorted(surface_text_by_id.items()))


def _serialize_question_dinox_detection(detection: dict[str, object]) -> dict[str, object]:
    bbox = detection.get("bbox")
    return {
        "bbox": [float(value) for value in bbox] if isinstance(bbox, list) else None,
        "score": float(detection.get("score", 0.0) or 0.0),
        "area_px": int(detection.get("area_px", 0) or 0),
        "category": str(detection.get("category", "")).strip().lower(),
        "mask_bounds_px": _question_dinox_mask_bounds(detection.get("mask")),
    }


def _call_question_dinox_detection(
    *,
    image_path: Path,
    label: str,
    image_shape: tuple[int, ...],
) -> list[dict[str, object]]:
    return _referability_call_dinox_joint_detection(
        client=None,
        image_path=image_path,
        alias_variants=[str(label).strip().lower()],
        image_shape=image_shape,
    )


def _collect_question_dinox_label_targets(question: dict[str, object]) -> list[dict[str, object]]:
    grouped: dict[str, dict[str, object]] = {}
    for mention in _shared_collect_question_mentions(question, {}):
        if not isinstance(mention, dict):
            continue
        label = str(mention.get("label", "")).strip().lower()
        role = str(mention.get("role", "")).strip().lower()
        obj_id = _shared_coerce_object_id(mention.get("obj_id"))
        key = label if label else "__missing__"
        entry = grouped.setdefault(
            key,
            {
                "label": label,
                "roles": set(),
                "mentioned_object_ids": set(),
            },
        )
        if role:
            entry["roles"].add(role)
        if obj_id is not None:
            entry["mentioned_object_ids"].add(int(obj_id))
    return [
        {
            "label": str(entry["label"]),
            "roles": sorted(str(role) for role in entry["roles"]),
            "mentioned_object_ids": sorted(int(obj_id) for obj_id in entry["mentioned_object_ids"]),
        }
        for entry in grouped.values()
    ]


def _copy_question_dinox_cache_entry(entry: dict[str, object]) -> dict[str, object]:
    return {
        "status": str(entry.get("status", "")),
        "label": str(entry.get("label", "")),
        "reason": entry.get("reason"),
        "prompt_variants": list(entry.get("prompt_variants", []) or []),
        "raw_detection_count": int(entry.get("raw_detection_count", 0) or 0),
        "loose_detection_count": int(entry.get("loose_detection_count", 0) or 0),
        "strong_detection_count": int(entry.get("strong_detection_count", 0) or 0),
        "raw_detections": [dict(item) for item in entry.get("raw_detections", []) if isinstance(item, dict)],
    }


def _get_question_dinox_cache_entry(
    *,
    scene_id: str,
    image_name: str,
    label: str,
    data_root: Path,
    detection_cache: dict[tuple[str, str, str], dict[str, object]],
    image_shape_cache: dict[tuple[str, str], tuple[int, ...] | None],
    dataset: str = "scannet",
    scannetpp_sensor: str = "iphone",
    scannetpp_frame_root: str | None = None,
) -> dict[str, object]:
    cache_key = (scene_id, image_name, label)
    cached = detection_cache.get(cache_key)
    if cached is not None:
        return cached

    image_cache_key = (scene_id, image_name)
    if not scene_id or not image_name:
        cached = {
            "status": "error",
            "label": label,
            "reason": "missing_scene_or_image_name",
            "prompt_variants": [label] if label else [],
            "raw_detection_count": 0,
            "loose_detection_count": 0,
            "strong_detection_count": 0,
            "raw_detections": [],
            "detections": [],
            "candidate_detections": [],
            "strong_detections": [],
            "image_shape": None,
        }
        detection_cache[cache_key] = cached
        return cached

    image_shape = image_shape_cache.get(image_cache_key)
    if image_cache_key not in image_shape_cache:
        from src.datasets import make_data_source
        ds_dinox = make_data_source(
            dataset, data_root / scene_id, sensor=scannetpp_sensor,
        )
        image_path = ds_dinox.image_path(image_name)
        image = cv2.imread(str(image_path))
        image_shape = None if image is None else tuple(image.shape)
        image_shape_cache[image_cache_key] = image_shape
    from src.datasets import make_data_source
    ds_dinox = make_data_source(
        dataset, data_root / scene_id, sensor=scannetpp_sensor,
    )
    image_path = ds_dinox.image_path(image_name)
    if image_shape is None:
        cached = {
            "status": "error",
            "label": label,
            "reason": "image_unavailable",
            "prompt_variants": [label] if label else [],
            "raw_detection_count": 0,
            "loose_detection_count": 0,
            "strong_detection_count": 0,
            "raw_detections": [],
            "detections": [],
            "candidate_detections": [],
            "strong_detections": [],
            "image_shape": None,
        }
        detection_cache[cache_key] = cached
        return cached

    try:
        detections = _call_question_dinox_detection(
            image_path=image_path,
            label=label,
            image_shape=image_shape,
        )
        candidate_detections = _dedupe_detections_by_mask_iou(
            [
                detection
                for detection in detections
                if float(detection.get("score", 0.0) or 0.0) >= QUESTION_DINOX_LOOSE_MIN_SCORE
            ]
        )
        strong_min_area = _strong_detection_min_area(image_shape)
        strong_detections = [
            detection
            for detection in candidate_detections
            if int(detection.get("area_px", 0) or 0) >= strong_min_area
            and float(detection.get("score", 0.0) or 0.0) >= QUESTION_DINOX_STRONG_MIN_SCORE
        ]
        cached = {
            "status": "ok",
            "label": label,
            "reason": None,
            "prompt_variants": [label] if label else [],
            "raw_detection_count": len(detections),
            "loose_detection_count": len(candidate_detections),
            "strong_detection_count": len(strong_detections),
            "raw_detections": [
                _serialize_question_dinox_detection(detection)
                for detection in detections
            ],
            "detections": detections,
            "candidate_detections": candidate_detections,
            "strong_detections": strong_detections,
            "image_shape": image_shape,
        }
    except Exception as exc:
        cached = {
            "status": "error",
            "label": label,
            "reason": str(exc),
            "prompt_variants": [label] if label else [],
            "raw_detection_count": 0,
            "loose_detection_count": 0,
            "strong_detection_count": 0,
            "raw_detections": [],
            "detections": [],
            "candidate_detections": [],
            "strong_detections": [],
            "image_shape": image_shape,
        }

    detection_cache[cache_key] = cached
    return cached


def _apply_question_dinox_audit(
    *,
    questions: list[dict[str, object]],
    data_root: Path,
    dataset: str = "scannet",
    scannetpp_sensor: str = "iphone",
) -> list[dict[str, object]]:
    detection_cache: dict[tuple[str, str, str], dict[str, object]] = {}
    image_shape_cache: dict[tuple[str, str], tuple[int, ...] | None] = {}

    for question in questions:
        label_targets = _collect_question_dinox_label_targets(question)
        label_audits: list[dict[str, object]] = []
        scene_id = str(question.get("scene_id", "")).strip()
        image_name = str(question.get("image_name", "")).strip()
        image_cache_key = (scene_id, image_name)

        for target in label_targets:
            label = str(target.get("label", "")).strip().lower()
            roles = list(target.get("roles", []))
            mentioned_object_ids = list(target.get("mentioned_object_ids", []))

            if not label:
                label_audits.append(
                    {
                        "status": "skipped",
                        "label": "",
                        "reason": "missing_label",
                        "roles": roles,
                        "mentioned_object_ids": mentioned_object_ids,
                        "raw_detection_count": 0,
                        "loose_detection_count": 0,
                        "raw_detections": [],
                    }
                )
                continue

            if label in EXCLUDED_LABELS:
                label_audits.append(
                    {
                        "status": "skipped",
                        "label": label,
                        "reason": "excluded_label",
                        "roles": roles,
                        "mentioned_object_ids": mentioned_object_ids,
                        "raw_detection_count": 0,
                        "loose_detection_count": 0,
                        "raw_detections": [],
                    }
                )
                continue

            cached = _get_question_dinox_cache_entry(
                scene_id=scene_id,
                image_name=image_name,
                label=label,
                data_root=data_root,
                detection_cache=detection_cache,
                image_shape_cache=image_shape_cache,
                dataset=dataset,
                scannetpp_sensor=scannetpp_sensor,
            )

            label_audit = _copy_question_dinox_cache_entry(cached)
            label_audit["roles"] = roles
            label_audit["mentioned_object_ids"] = mentioned_object_ids
            label_audits.append(label_audit)

        question["question_dinox_audit"] = {
            "status": _question_dinox_overall_status(label_audits),
            "labels": label_audits,
        }

    return questions


def _post_audit_review_stub(frame_context: dict[str, object], obj_id: int) -> dict[str, object]:
    crop_entry = {}
    crop_by_obj_id = frame_context.get("crop_by_obj_id", {})
    if isinstance(crop_by_obj_id, dict):
        crop_entry = crop_by_obj_id.get(int(obj_id), {}) or {}
    visibility_meta = {}
    visibility_by_obj_id = frame_context.get("visibility_by_obj_id", {})
    if isinstance(visibility_by_obj_id, dict):
        visibility_meta = visibility_by_obj_id.get(int(obj_id), {}) or {}
    return {
        "roi_bounds_px": crop_entry.get("roi_bounds_px") or visibility_meta.get("roi_bounds_px"),
        "crop_bounds_px": crop_entry.get("crop_bounds_px"),
    }


def _question_dinox_overall_status(label_audits: list[dict[str, object]]) -> str:
    if not label_audits:
        return "skipped"
    if any(str(item.get("status", "")).strip().lower() == "error" for item in label_audits):
        return "error"
    if any(str(item.get("status", "")).strip().lower() == "ok" for item in label_audits):
        return "ok"
    return "skipped"


def _build_question_post_dinox_skipped_label_audit(
    *,
    label: str,
    reason: str,
    roles: list[object],
    mentioned_object_ids: list[object],
) -> dict[str, object]:
    return {
        "status": "skipped",
        "label": label,
        "decision": "skipped",
        "reason": reason,
        "reason_codes": [reason],
        "roles": list(roles),
        "mentioned_object_ids": list(mentioned_object_ids),
        "matched_object_ids": [],
        "unmatched_object_ids": list(mentioned_object_ids),
        "raw_detection_count": 0,
        "loose_detection_count": 0,
        "strong_detection_count": 0,
        "raw_detections": [],
    }


def _match_question_post_dinox_object_ids(
    *,
    mentioned_object_ids: list[object],
    strong_detections: list[dict[str, object]],
    frame_context: dict[str, object],
    image_shape: tuple[int, ...],
) -> tuple[list[int], list[int]]:
    matched_object_ids: list[int] = []
    unmatched_object_ids: list[int] = []

    for obj_id in mentioned_object_ids:
        normalized_obj_id = int(obj_id)
        review_stub = _post_audit_review_stub(frame_context, normalized_obj_id)
        matched = any(
            _select_best_detection_for_object_review(
                detections=[detection],
                review=review_stub,
                image_shape=image_shape,
            ) is not None
            for detection in strong_detections
        )
        if matched:
            matched_object_ids.append(normalized_obj_id)
        else:
            unmatched_object_ids.append(normalized_obj_id)

    return matched_object_ids, unmatched_object_ids


def _build_question_post_dinox_label_review(
    *,
    target: dict[str, object],
    frame_context: dict[str, object],
    scene_id: str,
    image_name: str,
    data_root: Path,
    detection_cache: dict[tuple[str, str, str], dict[str, object]],
    image_shape_cache: dict[tuple[str, str], tuple[int, ...] | None],
    dataset: str = "scannet",
    scannetpp_sensor: str = "iphone",
) -> dict[str, object]:
    label = str(target.get("label", "")).strip().lower()
    roles = list(target.get("roles", []))
    mentioned_object_ids = list(target.get("mentioned_object_ids", []))

    if not label:
        return _build_question_post_dinox_skipped_label_audit(
            label="",
            reason="missing_label",
            roles=roles,
            mentioned_object_ids=mentioned_object_ids,
        )

    if label in EXCLUDED_LABELS:
        return _build_question_post_dinox_skipped_label_audit(
            label=label,
            reason="excluded_label",
            roles=roles,
            mentioned_object_ids=mentioned_object_ids,
        )

    cached = _get_question_dinox_cache_entry(
        scene_id=scene_id,
        image_name=image_name,
        label=label,
        data_root=data_root,
        detection_cache=detection_cache,
        image_shape_cache=image_shape_cache,
        dataset=dataset,
        scannetpp_sensor=scannetpp_sensor,
        scannetpp_frame_root=scannetpp_frame_root,
    )
    strong_detections = [
        detection for detection in cached.get("strong_detections", [])
        if isinstance(detection, dict)
    ]
    matched_object_ids, unmatched_object_ids = _match_question_post_dinox_object_ids(
        mentioned_object_ids=mentioned_object_ids,
        strong_detections=strong_detections,
        frame_context=frame_context,
        image_shape=tuple(cached.get("image_shape") or ()),
    )

    reason_codes: list[str] = []
    if str(cached.get("status", "")).strip().lower() != "ok":
        reason_codes.append("dinox_error")
    elif mentioned_object_ids:
        strong_count = int(cached.get("strong_detection_count", 0) or 0)
        if strong_count <= 0:
            reason_codes.append("dinox_no_strong_detection")
        elif strong_count >= 2:
            reason_codes.append("dinox_multiple_strong_detections")
        if strong_count > 0 and unmatched_object_ids:
            reason_codes.append("dinox_detection_misses_target")

    label_audit = _copy_question_dinox_cache_entry(cached)
    label_audit["decision"] = "manual_review" if reason_codes else "pass"
    label_audit["reason_codes"] = reason_codes
    label_audit["roles"] = roles
    label_audit["mentioned_object_ids"] = mentioned_object_ids
    label_audit["matched_object_ids"] = matched_object_ids
    label_audit["unmatched_object_ids"] = unmatched_object_ids
    return label_audit


def _run_question_post_dinox_stage(
    *,
    question: dict[str, object],
    frame_context: dict[str, object],
    data_root: Path,
    detection_cache: dict[tuple[str, str, str], dict[str, object]],
    image_shape_cache: dict[tuple[str, str], tuple[int, ...] | None],
    dataset: str = "scannet",
    scannetpp_sensor: str = "iphone",
) -> dict[str, object]:
    scene_id = str(question.get("scene_id", "")).strip()
    image_name = str(question.get("image_name", "")).strip()
    label_audits: list[dict[str, object]] = []
    dinox_reason_codes: list[str] = []
    flagged_labels: list[str] = []
    flagged_object_ids: list[int] = []

    for target in _collect_question_dinox_label_targets(question):
        label_audit = _build_question_post_dinox_label_review(
            target=target,
            frame_context=frame_context,
            scene_id=scene_id,
            image_name=image_name,
            data_root=data_root,
            detection_cache=detection_cache,
            image_shape_cache=image_shape_cache,
            dataset=dataset,
            scannetpp_sensor=scannetpp_sensor,
        )
        label_audits.append(label_audit)

        if str(label_audit.get("decision", "")).strip().lower() != "manual_review":
            continue
        label = str(label_audit.get("label", "")).strip().lower()
        dinox_reason_codes.extend(
            f"{code}:{label}"
            for code in label_audit.get("reason_codes", [])
        )
        if label and label not in flagged_labels:
            flagged_labels.append(label)
        for obj_id in label_audit.get("mentioned_object_ids", []):
            normalized_obj_id = int(obj_id)
            if normalized_obj_id not in flagged_object_ids:
                flagged_object_ids.append(normalized_obj_id)

    return {
        "audit": {
            "status": _question_dinox_overall_status(label_audits),
            "labels": label_audits,
        },
        "reason_codes": dinox_reason_codes,
        "flagged_labels": flagged_labels,
        "flagged_object_ids": flagged_object_ids,
    }


def _get_question_post_mesh_resources(
    *,
    scene_id: str,
    frame_context: dict[str, object],
    objects_by_id: dict[int, dict[str, object]],
    scene_mesh_cache: dict[str, object],
    scene_depth_intrinsics_cache: dict[str, object],
    dataset: str = "scannet",
    scannetpp_sensor: str = "iphone",
) -> dict[str, object]:
    scene_dir = frame_context.get("scene_dir") if isinstance(frame_context, dict) else None
    pose = frame_context.get("pose") if isinstance(frame_context, dict) else None
    color_intrinsics = frame_context.get("color_intrinsics") if isinstance(frame_context, dict) else None
    has_projection_context = bool(frame_context.get("has_projection_context", False)) if isinstance(frame_context, dict) else False

    if scene_id and scene_id not in scene_depth_intrinsics_cache:
        if isinstance(scene_dir, Path):
            try:
                from src.datasets import make_data_source
                ds_mesh = make_data_source(
                    dataset, scene_dir, sensor=scannetpp_sensor,
                )
                scene_depth_intrinsics_cache[scene_id] = ds_mesh.load_depth_intrinsics()
            except Exception:
                scene_depth_intrinsics_cache[scene_id] = None
        else:
            scene_depth_intrinsics_cache[scene_id] = None

    if scene_id and scene_id not in scene_mesh_cache:
        if isinstance(scene_dir, Path):
            try:
                mesh_kwargs = {
                    "instance_ids": sorted(int(obj_id) for obj_id in objects_by_id.keys()),
                    "n_surface_samples": 1,
                }
                if dataset == "scannetpp":
                    mesh_kwargs["dataset"] = "scannetpp"
                scene_mesh_cache[scene_id] = load_instance_mesh_data(scene_dir, **mesh_kwargs)
            except Exception as exc:
                logger.warning("Question post-generation mesh load failed for %s: %s", scene_id, exc)
                scene_mesh_cache[scene_id] = None
        else:
            scene_mesh_cache[scene_id] = None

    return {
        "pose": pose,
        "color_intrinsics": color_intrinsics,
        "has_projection_context": has_projection_context,
        "instance_mesh_data": scene_mesh_cache.get(scene_id),
        "depth_intrinsics": scene_depth_intrinsics_cache.get(scene_id),
    }


def _apply_question_post_review_results(
    question: dict[str, object],
    *,
    dinox_stage: dict[str, object],
    mesh_stage: dict[str, object],
) -> None:
    combined_reason_codes = _dedupe_strings(
        list(dinox_stage.get("reason_codes", [])) + list(mesh_stage.get("reason_codes", []))
    )
    question["question_dinox_audit"] = dict(dinox_stage.get("audit", {}))
    question["question_mesh_audit"] = dict(mesh_stage.get("audit", {}))
    question["question_post_generation_review"] = {
        "decision": "manual_review" if combined_reason_codes else "pass",
        "reason_codes": combined_reason_codes,
        "flagged_labels": _dedupe_strings(
            list(dinox_stage.get("flagged_labels", [])) + list(mesh_stage.get("flagged_labels", []))
        ),
        "flagged_object_ids": sorted(
            {
                int(obj_id)
                for obj_id in list(dinox_stage.get("flagged_object_ids", []))
                + list(mesh_stage.get("flagged_object_ids", []))
            }
        ),
        "dinox_label_reviews": list(question["question_dinox_audit"].get("labels", [])),
        "mesh_object_reviews": list(question["question_mesh_audit"].get("objects", [])),
    }


def _run_question_post_mesh_stage(
    *,
    question: dict[str, object],
    frame_context: dict[str, object],
    objects_by_id: dict[int, dict[str, object]],
    data_root: Path,
    detection_cache: dict[tuple[str, str, str], dict[str, object]],
    image_shape_cache: dict[tuple[str, str], tuple[int, ...] | None],
    scene_mesh_cache: dict[str, object],
    scene_depth_intrinsics_cache: dict[str, object],
    topology_cache: dict[tuple[str, int], dict[str, object]],
    dataset: str = "scannet",
    scannetpp_sensor: str = "iphone",
) -> dict[str, object]:
    scene_id = str(question.get("scene_id", "")).strip()
    image_name = str(question.get("image_name", "")).strip()
    mesh_object_reviews: list[dict[str, object]] = []
    mesh_reason_codes: list[str] = []
    flagged_labels: list[str] = []
    flagged_object_ids: list[int] = []
    seen_mesh_obj_ids: set[int] = set()
    mesh_resources = _get_question_post_mesh_resources(
        scene_id=scene_id,
        frame_context=frame_context,
        objects_by_id=objects_by_id,
        scene_mesh_cache=scene_mesh_cache,
        scene_depth_intrinsics_cache=scene_depth_intrinsics_cache,
        dataset=dataset,
        scannetpp_sensor=scannetpp_sensor,
    )
    pose = mesh_resources["pose"]
    color_intrinsics = mesh_resources["color_intrinsics"]
    has_projection_context = bool(mesh_resources["has_projection_context"])
    instance_mesh_data = mesh_resources["instance_mesh_data"]
    depth_intrinsics = mesh_resources["depth_intrinsics"]

    for mention in _iter_question_referability_mentions(question, objects_by_id):
        obj_id = _coerce_object_id(mention.get("obj_id"))
        label = str(mention.get("label", "")).strip().lower()
        roles = _dedupe_strings(
            [str(role).strip() for role in mention.get("observed_roles", []) if str(role).strip()]
            or [str(mention.get("role", "")).strip()]
        )
        if obj_id is None or obj_id in seen_mesh_obj_ids or label in EXCLUDED_LABELS:
            continue
        seen_mesh_obj_ids.add(int(obj_id))

        crop_entry = {}
        crop_by_obj_id = frame_context.get("crop_by_obj_id", {})
        if isinstance(crop_by_obj_id, dict):
            crop_entry = crop_by_obj_id.get(int(obj_id), {}) or {}
        mesh_review: dict[str, object] = {
            "label": label,
            "obj_id": int(obj_id),
            "roles": roles,
            "decision": "manual_review",
            "reason": "",
            "reason_codes": [],
            "topology_status": None,
            "topology_reason_codes": [],
            "mesh_mask_status": None,
            "mesh_mask_reason_codes": [],
            "mesh_mask_iou": None,
            "mesh_mask_under_coverage": None,
            "mesh_mask_over_coverage": None,
            "mesh_mask_area_ratio": None,
            "mesh_mask_depth_bad_ratio": None,
            "matched_detection": None,
        }

        if not has_projection_context or pose is None or color_intrinsics is None:
            mesh_review["reason"] = "missing_projection_context"
            mesh_review["reason_codes"] = ["missing_projection_context"]
        elif instance_mesh_data is None:
            mesh_review["reason"] = "missing_instance_mesh_data"
            mesh_review["reason_codes"] = ["missing_instance_mesh_data"]
        elif not bool(crop_entry.get("valid", False)):
            mesh_review["reason"] = str(crop_entry.get("reason", "")).strip() or "invalid_crop"
            mesh_review["reason_codes"] = [str(mesh_review["reason"])]
        else:
            topology_key = (scene_id, int(obj_id))
            topology_quality = topology_cache.get(topology_key)
            if topology_quality is None:
                topology_quality = _compute_topology_quality_for_object(
                    obj_id=int(obj_id),
                    instance_mesh_data=instance_mesh_data,
                )
                topology_cache[topology_key] = topology_quality
            mesh_review["topology_status"] = str(topology_quality.get("status", "")).strip().lower() or None
            mesh_review["topology_reason_codes"] = list(topology_quality.get("reason_codes", []))

            if str(topology_quality.get("status", "")).strip().lower() == "fail":
                mesh_review["reason"] = "topology_fail"
                mesh_review["reason_codes"] = ["topology_fail"]
            else:
                cached = _get_question_dinox_cache_entry(
                    scene_id=scene_id,
                    image_name=image_name,
                    label=label,
                    data_root=data_root,
                    detection_cache=detection_cache,
                    image_shape_cache=image_shape_cache,
                    dataset=dataset,
                    scannetpp_sensor=scannetpp_sensor,
                )
                if str(cached.get("status", "")).strip().lower() != "ok":
                    mesh_review["reason"] = "dinox_error"
                    mesh_review["reason_codes"] = ["dinox_error"]
                else:
                    matched_detection = _select_best_detection_for_object_review(
                        detections=list(cached.get("candidate_detections", [])),
                        review=_post_audit_review_stub(frame_context, int(obj_id)),
                        image_shape=tuple(cached.get("image_shape") or ()),
                    )
                    if matched_detection is None:
                        mesh_review["reason"] = (
                            "no_detection_overlap"
                            if list(cached.get("candidate_detections", []))
                            else "no_detection_mask"
                        )
                        mesh_review["reason_codes"] = [str(mesh_review["reason"])]
                    else:
                        mesh_review["matched_detection"] = _serialize_question_dinox_detection(matched_detection)
                        mesh_quality = _compute_mesh_mask_quality_for_object(
                            obj_id=int(obj_id),
                            detection_mask=np.asarray(matched_detection.get("mask"), dtype=bool),
                            topology_status=str(topology_quality.get("status", "")),
                            camera_pose=pose,
                            color_intrinsics=color_intrinsics,
                            depth_image=None,
                            depth_intrinsics=depth_intrinsics,
                            instance_mesh_data=instance_mesh_data,
                        )
                        mesh_review["mesh_mask_status"] = str(mesh_quality.get("status", "")).strip().lower() or None
                        mesh_review["mesh_mask_reason_codes"] = list(mesh_quality.get("reason_codes", []))
                        mesh_review["mesh_mask_iou"] = mesh_quality.get("iou")
                        mesh_review["mesh_mask_under_coverage"] = mesh_quality.get("under_coverage")
                        mesh_review["mesh_mask_over_coverage"] = mesh_quality.get("over_coverage")
                        mesh_review["mesh_mask_area_ratio"] = mesh_quality.get("area_ratio")
                        mesh_review["mesh_mask_depth_bad_ratio"] = mesh_quality.get("depth_bad_ratio")
                        if str(mesh_quality.get("status", "")).strip().lower() == "fail":
                            mesh_review["reason"] = "mesh_mask_mismatch"
                            mesh_review["reason_codes"] = [
                                f"mesh_{code}" for code in mesh_quality.get("reason_codes", [])
                            ]
                        else:
                            mesh_review["decision"] = "pass"
                            mesh_review["reason"] = "mesh_mask_match"
                            mesh_review["reason_codes"] = []

        if mesh_review["decision"] != "pass":
            mesh_reason_codes.extend(
                f"{code}:{label}#{int(obj_id)}"
                for code in mesh_review.get("reason_codes", [])
            )
            if label and label not in flagged_labels:
                flagged_labels.append(label)
            if int(obj_id) not in flagged_object_ids:
                flagged_object_ids.append(int(obj_id))
        mesh_object_reviews.append(mesh_review)

    return {
        "audit": {
            "status": "ok" if mesh_object_reviews else "skipped",
            "objects": mesh_object_reviews,
        },
        "reason_codes": mesh_reason_codes,
        "flagged_labels": flagged_labels,
        "flagged_object_ids": flagged_object_ids,
    }


def _apply_question_post_generation_audit(
    *,
    questions: list[dict[str, object]],
    data_root: Path,
    output_dir: Path,
    frame_context_by_key: dict[tuple[str, str], dict[str, object]] | None = None,
    dataset: str = "scannet",
    scannetpp_sensor: str = "iphone",
    scannetpp_frame_root: str | None = None,
) -> list[dict[str, object]]:
    detection_cache: dict[tuple[str, str, str], dict[str, object]] = {}
    image_shape_cache: dict[tuple[str, str], tuple[int, ...] | None] = {}
    if frame_context_by_key is None:
        frame_context_by_key = _prebuild_question_review_frame_contexts(
            questions=questions,
            data_root=data_root,
            output_dir=output_dir,
            dataset=dataset,
            scannetpp_sensor=scannetpp_sensor,
            scannetpp_frame_root=scannetpp_frame_root,
        )
    scene_mesh_cache: dict[str, object] = {}
    scene_depth_intrinsics_cache: dict[str, object] = {}
    topology_cache: dict[tuple[str, int], dict[str, object]] = {}

    for question in questions:
        if question.get("cross_frame_layout"):
            question["question_post_generation_review"] = {
                "decision": "pass",
                "mode": "cross_frame_flash_role_referability",
                "reason_codes": [],
            }
            continue
        scene_id = str(question.get("scene_id", "")).strip()
        image_name = str(question.get("image_name", "")).strip()
        frame_context = frame_context_by_key.get((scene_id, image_name), {})
        objects_by_id = (
            dict(frame_context.get("objects_by_id", {}))
            if isinstance(frame_context, dict) else {}
        )
        dinox_stage = _run_question_post_dinox_stage(
            question=question,
            frame_context=frame_context,
            data_root=data_root,
            detection_cache=detection_cache,
            image_shape_cache=image_shape_cache,
            dataset=dataset,
            scannetpp_sensor=scannetpp_sensor,
        )
        mesh_stage = _run_question_post_mesh_stage(
            question=question,
            frame_context=frame_context,
            objects_by_id=objects_by_id,
            data_root=data_root,
            detection_cache=detection_cache,
            image_shape_cache=image_shape_cache,
            scene_mesh_cache=scene_mesh_cache,
            scene_depth_intrinsics_cache=scene_depth_intrinsics_cache,
            topology_cache=topology_cache,
            dataset=dataset,
            scannetpp_sensor=scannetpp_sensor,
        )
        _apply_question_post_review_results(
            question,
            dinox_stage=dinox_stage,
            mesh_stage=mesh_stage,
        )

    return questions


def _iter_referability_cache_frame_entries(
    cache: dict | None,
) -> Iterator[tuple[dict[str, object], str, str, object]]:
    if not isinstance(cache, dict):
        return
    frames = cache.get("frames", cache)
    if not isinstance(frames, dict):
        return
    for scene_id, scene_frames in frames.items():
        if not isinstance(scene_frames, dict):
            continue
        if "/" in str(scene_id):
            scene_key, image_name = str(scene_id).split("/", 1)
            yield frames, scene_key, str(image_name), scene_frames
            continue
        for image_name, entry in scene_frames.items():
            yield scene_frames, str(scene_id), str(image_name), entry


def _find_inconsistent_referability_entry(cache: dict | None) -> str | None:
    for _scene_frames, scene_id, image_name, entry in _iter_referability_cache_frame_entries(cache):
        if (
            isinstance(entry, dict)
            and not _manual_attachment_roles_for_frame(entry)
            and not _frame_entry_has_consistent_final_fields(entry)
        ):
            return f"{scene_id}/{image_name}"
    return None


def _repair_referability_cache_entries(cache: dict | None) -> int:
    repaired_count = 0
    for scene_frames, _scene_id, image_name, entry in _iter_referability_cache_frame_entries(cache):
        if not isinstance(entry, dict):
            continue
        if _frame_entry_has_consistent_final_fields(entry):
            continue
        scene_frames[image_name] = _repair_final_referability_fields(entry)
        repaired_count += 1
    return repaired_count


def _referability_cache_edited_html_path(path: Path) -> Path:
    return path.parent / "edited.html"


def _referability_cache_legacy_edited_html_glob(path: Path) -> str:
    return str(path.parent / "edited*.html")


def _referability_cache_legacy_edited_html_paths(path: Path) -> list[Path]:
    return sorted(path.parent.glob("edited*.html"))


def _referability_cache_scene_edited_html_path(path: Path, scene_id: str) -> Path:
    return path.parent / f"{path.stem}_{str(scene_id).strip()}_edited.html"


def _referability_cache_scene_edited_html_glob(path: Path) -> str:
    return str(path.parent / f"{path.stem}_*_edited.html")


def _expected_referability_cache_scene_ids(cache: dict | None) -> list[str]:
    if not isinstance(cache, dict):
        return []
    scene_ids: set[str] = set()
    for field_name in ("scene_status", "scene_grouping"):
        field_value = cache.get(field_name)
        if not isinstance(field_value, dict):
            continue
        for scene_key, scene_value in field_value.items():
            if isinstance(scene_key, str) and scene_key.strip():
                scene_ids.add(scene_key.strip())
            if isinstance(scene_value, dict):
                nested_scene_id = str(scene_value.get("scene_id", "")).strip()
                if nested_scene_id:
                    scene_ids.add(nested_scene_id)

    frames = cache.get("frames", cache)
    if isinstance(frames, dict):
        for scene_key, scene_value in frames.items():
            if isinstance(scene_value, dict) and "frame_usable" not in scene_value:
                scene_id = str(scene_key).strip()
                if scene_id:
                    scene_ids.add(scene_id)
            elif isinstance(scene_key, str) and "/" in scene_key:
                scene_id = scene_key.split("/", 1)[0].strip()
                if scene_id:
                    scene_ids.add(scene_id)
    return sorted(scene_ids)


def _resolve_referability_cache_review_html_paths(
    *,
    path: Path,
    cache_doc: dict[str, object],
) -> tuple[list[Path], str]:
    legacy_html_paths = _referability_cache_legacy_edited_html_paths(path)
    if len(legacy_html_paths) > 1:
        candidate_lines = "\n".join(
            f"- {candidate.resolve()}"
            for candidate in legacy_html_paths
        )
        raise ValueError(
            "[legacy edited*.html review files have multiple candidates / å¤šä¸ªå€™é€‰]\n"
            f"referability_cache: {path}\n"
            f"expected_glob: {_referability_cache_legacy_edited_html_glob(path)}\n"
            "matched_paths:\n"
            f"{candidate_lines}\n"
            "Keep exactly one legacy edited*.html file to avoid pipeline misreads."
        )
    if legacy_html_paths:
        return [legacy_html_paths[0]], "legacy"

    scene_html_paths = sorted(path.parent.glob(f"{path.stem}_*_edited.html"))
    if scene_html_paths:
        expected_scene_ids = _expected_referability_cache_scene_ids(cache_doc)
        existing_scene_ids: list[str] = []
        missing_scene_ids: list[str] = []
        for scene_id in expected_scene_ids:
            if _referability_cache_scene_edited_html_path(path, scene_id).exists():
                existing_scene_ids.append(scene_id)
            else:
                missing_scene_ids.append(scene_id)
        if missing_scene_ids:
            missing_lines = "\n".join(
                f"- {scene_id}: {_referability_cache_scene_edited_html_path(path, scene_id).resolve()}"
                for scene_id in missing_scene_ids
            )
            logger.warning(
                "[缺少按 scene 划分的人工审核文件 / incomplete scene-scoped review HTML]\n"
                f"referability_cache: {path}\n"
                f"检测到新格式文件: {_referability_cache_scene_edited_html_glob(path)}\n"
                f"期望 scene: {', '.join(expected_scene_ids) or '<none>'}\n"
                "缺失文件:\n"
                f"{missing_lines}\n"
                "缺少人工审核文件的 scene 将跳过 salvage 回填。"
            )
        if existing_scene_ids:
            return [
                _referability_cache_scene_edited_html_path(path, scene_id)
                for scene_id in existing_scene_ids
            ], "scene-scoped"
        return [], "none"

    return [], "none"


def _load_single_referability_cache(
    path: Path,
    *,
    repair_inconsistent_entries: bool = False,
    persist_repaired_entries: bool = False,
    no_salvage: bool = False,
) -> dict | None:
    if not path.exists():
        logger.warning("Referability cache not found: %s", path)
        return None
    with open(path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    is_two_hop_manual_cache = _is_manual_attachment_cache(raw_data)
    if no_salvage or is_two_hop_manual_cache:
        review_html_paths: list[Path] = []
        review_html_mode = "none"
    else:
        review_html_paths, review_html_mode = _resolve_referability_cache_review_html_paths(
            path=path,
            cache_doc=raw_data,
        )
    data = raw_data
    for review_html_path in review_html_paths:
        html_text = review_html_path.read_text(encoding="utf-8")
        data = _apply_attachment_pair_salvage_html_review(
            html_text=html_text,
            cache_doc=data,
            cache_path=path,
        )
    version = str(data.get("version", ""))
    if version != EXPECTED_REFERABILITY_CACHE_VERSION:
        raise ValueError(
            f"Referability cache version mismatch: expected {EXPECTED_REFERABILITY_CACHE_VERSION}, got {version or '<missing>'}. "
            "Regenerate the referability cache with the updated VLM prompts before running the pipeline."
        )
    inconsistent_entry = _find_inconsistent_referability_entry(data)
    if inconsistent_entry is not None:
        if not repair_inconsistent_entries:
            raise ValueError(
                f"Referability cache entry for {inconsistent_entry} is inconsistent with cache version "
                f"{EXPECTED_REFERABILITY_CACHE_VERSION}. Rerun with --repair_referability_cache to repair "
                "deterministic final fields, or regenerate the referability cache if the underlying prompts changed."
            )
        repaired_count = _repair_referability_cache_entries(data)
        remaining_inconsistent_entry = _find_inconsistent_referability_entry(data)
        if remaining_inconsistent_entry is not None:
            raise ValueError(
                f"Referability cache entry for {remaining_inconsistent_entry} is inconsistent with cache version "
                f"{EXPECTED_REFERABILITY_CACHE_VERSION} even after repair. Regenerate the referability cache from scratch."
            )
        logger.warning(
            "Repaired %d inconsistent referability cache entr%s in %s",
            repaired_count,
            "y" if repaired_count == 1 else "ies",
            path,
        )
        if persist_repaired_entries and repaired_count > 0:
            persistable_data = json.loads(json.dumps(raw_data, ensure_ascii=False))
            persistable_repaired_count = _repair_referability_cache_entries(persistable_data)
            remaining_persistable_inconsistent_entry = _find_inconsistent_referability_entry(persistable_data)
            if remaining_persistable_inconsistent_entry is not None:
                raise ValueError(
                    f"Referability cache entry for {remaining_persistable_inconsistent_entry} is inconsistent with cache version "
                    f"{EXPECTED_REFERABILITY_CACHE_VERSION} even after repair. Regenerate the referability cache from scratch."
                )
            if persistable_repaired_count > 0:
                _write_json_file(path, persistable_data)
                logger.info("Wrote repaired referability cache to %s", path)
    if review_html_mode == "none":
        logger.info(
            "Loaded referability cache from %s without human salvage backfill (%s)",
            path,
            (
                "manual two-hop cache"
                if is_two_hop_manual_cache
                else "disabled via --no_salvage"
                if no_salvage
                else "no review HTML found"
            ),
        )
    else:
        logger.info(
            "Loaded referability cache from %s with automatic human salvage backfill enabled via %s (%s)",
            path,
            (
                review_html_paths[0]
                if len(review_html_paths) == 1
                else _referability_cache_scene_edited_html_glob(path)
            ),
            review_html_mode,
        )
    return data


def _expand_referability_cache_paths(path_or_pattern: str | Path) -> tuple[list[Path], bool]:
    raw_pattern = str(path_or_pattern)
    has_glob = glob.has_magic(raw_pattern)
    if not has_glob:
        return [Path(raw_pattern)], False
    matched_paths = [
        Path(match)
        for match in sorted(glob.glob(raw_pattern))
    ]
    if not matched_paths:
        raise ValueError(f"Referability cache glob matched no files: {raw_pattern}")
    return matched_paths, True


def _merge_referability_cache_docs(
    cache_docs: list[tuple[Path, dict[str, object]]],
    *,
    allow_mixed_models: bool = False,
) -> dict[str, object]:
    if not cache_docs:
        raise ValueError("No referability cache documents were provided for merging")

    base_path, base_doc = cache_docs[0]
    merged_doc = json.loads(json.dumps(base_doc, ensure_ascii=False))
    merged_doc["frames"] = {}
    merged_doc["scene_grouping"] = {}
    merged_doc["scene_status"] = {}

    metadata_fields = (
        "version",
        "referability_backend",
        "alias_config_version",
    )
    if not allow_mixed_models:
        metadata_fields = (*metadata_fields, "model")
    merge_fields = ("frames", "scene_grouping", "scene_status")

    for field_name in merge_fields:
        if not isinstance(merged_doc.get(field_name), dict):
            merged_doc[field_name] = {}

    for current_path, current_doc in cache_docs:
        for field_name in metadata_fields:
            base_value = base_doc.get(field_name)
            current_value = current_doc.get(field_name)
            if current_value != base_value:
                raise ValueError(
                    f"Referability cache metadata mismatch for {field_name}: "
                    f"{base_path} has {base_value!r}, but {current_path} has {current_value!r}"
                )

        for field_name in merge_fields:
            current_field = current_doc.get(field_name, {})
            if not isinstance(current_field, dict):
                continue
            merged_field = merged_doc.setdefault(field_name, {})
            if not isinstance(merged_field, dict):
                raise ValueError(f"Merged referability cache field {field_name} must be an object")
            for scene_id, scene_value in current_field.items():
                if scene_id in merged_field:
                    raise ValueError(
                        f"Duplicate referability cache scene {scene_id!r} found in field {field_name} "
                        f"while merging {current_path}"
                    )
                merged_field[scene_id] = scene_value

    return merged_doc


def _referability_cache_doc_contains_scene(
    cache_doc: dict[str, object],
    scene_id: str,
) -> bool:
    for field_name in ("scene_grouping", "scene_status"):
        field_value = cache_doc.get(field_name)
        if isinstance(field_value, dict) and scene_id in field_value:
            return True

    frames = cache_doc.get("frames", cache_doc)
    if not isinstance(frames, dict):
        return False
    if scene_id in frames:
        return True
    prefix = f"{scene_id}/"
    return any(isinstance(key, str) and key.startswith(prefix) for key in frames)


def _project_referability_cache_doc(
    cache_doc: dict[str, object],
    *,
    scene_ids: set[str],
) -> dict[str, object]:
    projected = {
        key: value
        for key, value in cache_doc.items()
        if key not in {"frames", "scene_grouping", "scene_status"}
    }
    for field_name in ("frames", "scene_grouping", "scene_status"):
        field_value = cache_doc.get(field_name, {})
        projected_field: dict[str, object] = {}
        if isinstance(field_value, dict):
            for key, value in field_value.items():
                key_text = str(key)
                scene_id = key_text.split("/", 1)[0] if field_name == "frames" else key_text
                if scene_id in scene_ids:
                    projected_field[key_text] = value
        projected[field_name] = projected_field
    return projected


def _load_referability_cache_from_scene_status(
    scene_status_path: Path,
    *,
    repair_inconsistent_entries: bool = False,
    persist_repaired_entries: bool = False,
    no_salvage: bool = False,
) -> dict[str, object]:
    if not scene_status_path.exists():
        raise ValueError(f"Referability scene status not found: {scene_status_path}")
    try:
        with open(scene_status_path, "r", encoding="utf-8") as f:
            scene_status_doc = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Failed to load referability scene status JSON: {scene_status_path}"
        ) from exc
    if not isinstance(scene_status_doc, dict):
        raise ValueError(
            f"Invalid referability scene status at {scene_status_path}: expected JSON object"
        )

    try:
        status_version = int(scene_status_doc.get("version", 0) or 0)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid referability scene status version at {scene_status_path}: "
            f"{scene_status_doc.get('version')!r}"
        ) from exc
    if status_version != REFERABILITY_SCENE_STATUS_VERSION:
        raise ValueError(
            f"Unsupported referability scene status version {status_version or '<missing>'} "
            f"at {scene_status_path}; expected {REFERABILITY_SCENE_STATUS_VERSION}."
        )

    completed_scenes = scene_status_doc.get("completed_scenes")
    if not isinstance(completed_scenes, dict):
        raise ValueError(
            f"Invalid referability scene status at {scene_status_path}: "
            "completed_scenes must be an object"
        )
    if not completed_scenes:
        raise ValueError(
            f"Referability scene status contains no completed scenes: {scene_status_path}"
        )

    batch_scene_ids: dict[Path, set[str]] = defaultdict(set)
    for raw_scene_id, record in completed_scenes.items():
        scene_id = str(raw_scene_id).strip()
        if not scene_id:
            raise ValueError(
                f"Invalid empty scene id in referability scene status: {scene_status_path}"
            )
        if not isinstance(record, dict):
            raise ValueError(
                f"Invalid referability scene status record for {scene_id} at "
                f"{scene_status_path}: expected object"
            )
        batch_file = str(record.get("batch_file", "")).strip()
        if not batch_file:
            raise ValueError(
                f"Invalid referability scene status record for {scene_id} at "
                f"{scene_status_path}: missing batch_file"
            )
        batch_path = (scene_status_path.parent / batch_file).resolve()
        if not batch_path.exists():
            raise ValueError(
                f"Referability scene status says {scene_id} is stored in {batch_file}, "
                f"but the batch file does not exist: {batch_path}"
            )
        batch_scene_ids[batch_path].add(scene_id)

    active_batch_docs: list[tuple[Path, dict[str, object]]] = []
    for batch_path, active_scene_ids in sorted(
        batch_scene_ids.items(),
        key=lambda item: str(item[0]),
    ):
        try:
            batch_doc = _load_single_referability_cache(
                batch_path,
                repair_inconsistent_entries=repair_inconsistent_entries,
                persist_repaired_entries=persist_repaired_entries,
                no_salvage=no_salvage,
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"Failed to load referability batch JSON {batch_path} referenced by "
                f"{scene_status_path}"
            ) from exc
        if batch_doc is None:
            raise ValueError(f"Referability cache not found: {batch_path}")
        for scene_id in sorted(active_scene_ids):
            if not _referability_cache_doc_contains_scene(batch_doc, scene_id):
                raise ValueError(
                    f"Referability scene status says {scene_id} is stored in "
                    f"{batch_path.name}, but that batch does not contain the scene: "
                    f"{batch_path}"
                )
        active_batch_docs.append(
            (
                batch_path,
                _project_referability_cache_doc(
                    batch_doc,
                    scene_ids=active_scene_ids,
                ),
            )
        )

    active_models = {
        str(batch_doc.get("model", "")).strip() or "<missing>"
        for _batch_path, batch_doc in active_batch_docs
    }
    if len(active_models) > 1:
        logger.warning(
            "Loading active referability scenes generated by multiple VLM models from %s: %s",
            scene_status_path,
            ", ".join(sorted(active_models)),
        )
    merged = _merge_referability_cache_docs(
        active_batch_docs,
        allow_mixed_models=True,
    )
    logger.info(
        "Loaded %d active referability scene(s) from %d batch cache file(s) "
        "referenced by %s",
        len(completed_scenes),
        len(active_batch_docs),
        scene_status_path,
    )
    return merged


def _load_referability_cache(
    path_or_pattern: str | Path,
    *,
    repair_inconsistent_entries: bool = False,
    persist_repaired_entries: bool = False,
    no_salvage: bool = False,
) -> dict | None:
    paths, used_glob = _expand_referability_cache_paths(path_or_pattern)
    if len(paths) == 1 and not used_glob:
        if paths[0].name.lower() == "scene_status.json":
            return _load_referability_cache_from_scene_status(
                paths[0],
                repair_inconsistent_entries=repair_inconsistent_entries,
                persist_repaired_entries=persist_repaired_entries,
                no_salvage=no_salvage,
            )
        return _load_single_referability_cache(
            paths[0],
            repair_inconsistent_entries=repair_inconsistent_entries,
            persist_repaired_entries=persist_repaired_entries,
            no_salvage=no_salvage,
        )

    loaded_docs: list[tuple[Path, dict[str, object]]] = []
    for path in paths:
        loaded = _load_single_referability_cache(
            path,
            repair_inconsistent_entries=repair_inconsistent_entries,
            persist_repaired_entries=persist_repaired_entries,
            no_salvage=no_salvage,
        )
        if loaded is None:
            raise ValueError(f"Referability cache not found: {path}")
        loaded_docs.append((path, loaded))

    merged = _merge_referability_cache_docs(loaded_docs)
    logger.info(
        "Merged %d referability batch cache files from %s",
        len(loaded_docs),
        path_or_pattern,
    )
    return merged


def _normalize_manual_attachment_role_sets(
    entry: dict[str, object],
    *,
    scene_id: str,
    image_name: str,
) -> list[dict[str, dict[str, object]]]:
    raw_role_sets = entry.get("manual_attachment_role_sets")
    if isinstance(raw_role_sets, list) and raw_role_sets:
        candidates = raw_role_sets
    else:
        candidates = [entry.get("manual_attachment_roles")]
    if not candidates or not all(isinstance(candidate, dict) for candidate in candidates):
        raise ValueError(
            f"Manual attachment frame {scene_id}/{image_name} is missing manual attachment roles"
        )
    projected_ids = set(_normalize_object_ids(entry.get("candidate_visible_object_ids")))
    normalized_sets: list[dict[str, dict[str, object]]] = []
    role_keys: set[tuple[tuple[int, str], ...]] = set()
    for set_index, raw_roles in enumerate(candidates, start=1):
        roles: dict[str, dict[str, object]] = {}
        role_ids: list[int] = []
        for role in ("moved", "child", "grandchild", "contrast"):
            raw_value = raw_roles.get(role)
            if not isinstance(raw_value, dict):
                raise ValueError(
                    f"Manual attachment frame {scene_id}/{image_name} annotation {set_index} "
                    f"role {role!r} must contain id and label"
                )
            try:
                obj_id = int(raw_value.get("id"))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Manual attachment frame {scene_id}/{image_name} annotation {set_index} "
                    f"role {role!r} has an invalid object id"
                ) from exc
            label = str(raw_value.get("label", "")).strip()
            if not label:
                raise ValueError(
                    f"Manual attachment frame {scene_id}/{image_name} annotation {set_index} "
                    f"role {role!r} has an empty label"
                )
            if obj_id not in projected_ids:
                raise ValueError(
                    f"Manual attachment frame {scene_id}/{image_name} annotation {set_index} "
                    f"role {role!r} uses object id {obj_id}, which is not one of the "
                    "projected frame objects"
                )
            role_ids.append(obj_id)
            roles[role] = {"id": obj_id, "label": label}
        if len(set(role_ids)) != len(role_ids):
            raise ValueError(
                f"Manual attachment frame {scene_id}/{image_name} annotation {set_index} "
                "must use four distinct object ids"
            )
        role_key = tuple(
            (int(roles[role]["id"]), str(roles[role]["label"]).strip().lower())
            for role in ("moved", "child", "grandchild", "contrast")
        )
        if role_key in role_keys:
            raise ValueError(
                f"Manual attachment frame {scene_id}/{image_name} contains duplicate "
                f"role set {tuple(role_ids)}"
            )
        role_keys.add(role_key)
        option_labels = {
            str(roles[role]["label"]).strip().lower()
            for role in ("child", "grandchild", "contrast")
        }
        if len(option_labels) != 3:
            raise ValueError(
                f"Manual attachment frame {scene_id}/{image_name} annotation {set_index} "
                "child, grandchild, and contrast labels must be distinct"
            )
        normalized_sets.append(roles)
    return normalized_sets


def _nested_scene_frames_for_update(
    frames: dict[str, object],
    scene_id: str,
) -> dict[str, dict[str, object]]:
    existing = frames.get(scene_id)
    if isinstance(existing, dict) and "frame_usable" not in existing:
        return existing
    scene_frames: dict[str, dict[str, object]] = {}
    prefix = f"{scene_id}/"
    for key in list(frames):
        value = frames.get(key)
        if isinstance(key, str) and key.startswith(prefix) and isinstance(value, dict):
            scene_frames[key[len(prefix):]] = value
            del frames[key]
    frames[scene_id] = scene_frames
    return scene_frames


def _merge_manual_attachment_cache(
    referability_cache: dict[str, object],
    manual_cache: dict[str, object],
) -> dict[str, object]:
    if not _is_manual_attachment_cache(manual_cache):
        raise ValueError(
            f"Manual attachment cache must use schema {MANUAL_ATTACHMENT_CACHE_SCHEMA}"
        )
    if str(manual_cache.get("version", "")).strip() != EXPECTED_REFERABILITY_CACHE_VERSION:
        raise ValueError(
            "Manual attachment cache version mismatch: expected "
            f"{EXPECTED_REFERABILITY_CACHE_VERSION}"
        )
    raw_manual_frames = manual_cache.get("frames")
    if not isinstance(raw_manual_frames, dict) or not raw_manual_frames:
        raise ValueError("Manual attachment cache contains no frames")

    merged = json.loads(json.dumps(referability_cache, ensure_ascii=False))
    merged_frames = merged.setdefault("frames", {})
    if not isinstance(merged_frames, dict):
        raise ValueError("Referability cache frames field must be an object")

    manual_graph: dict[str, dict[str, list[int]]] = {}
    manual_scene_ids: list[str] = []
    merged_frame_count = 0
    added_frame_count = 0

    for raw_scene_id, raw_scene_frames in raw_manual_frames.items():
        scene_id = str(raw_scene_id).strip()
        if not scene_id or not isinstance(raw_scene_frames, dict):
            raise ValueError("Manual attachment cache contains an invalid scene entry")
        manual_scene_ids.append(scene_id)
        target_scene_frames = _nested_scene_frames_for_update(merged_frames, scene_id)
        scene_graph = manual_graph.setdefault(scene_id, {})

        for raw_image_name, raw_entry in raw_scene_frames.items():
            image_name = str(raw_image_name).strip()
            if not image_name or not isinstance(raw_entry, dict):
                raise ValueError(f"Manual attachment scene {scene_id} contains an invalid frame")
            role_sets = _normalize_manual_attachment_role_sets(
                raw_entry,
                scene_id=scene_id,
                image_name=image_name,
            )
            role_ids: set[int] = set()
            attachment_pairs: list[list[int]] = []
            for roles in role_sets:
                moved_id = int(roles["moved"]["id"])
                child_id = int(roles["child"]["id"])
                grandchild_id = int(roles["grandchild"]["id"])
                contrast_id = int(roles["contrast"]["id"])
                role_ids.update((moved_id, child_id, grandchild_id, contrast_id))
                for parent_id, attached_id in (
                    (moved_id, child_id),
                    (child_id, grandchild_id),
                ):
                    pair = [parent_id, attached_id]
                    if pair not in attachment_pairs:
                        attachment_pairs.append(pair)
                    children = scene_graph.setdefault(str(parent_id), [])
                    if attached_id not in children:
                        children.append(attached_id)
                        children.sort()
            sorted_role_ids = sorted(role_ids)

            existing_entry = target_scene_frames.get(image_name)
            if isinstance(existing_entry, dict):
                merged_entry = dict(existing_entry)
                merged_frame_count += 1
            else:
                merged_entry = dict(raw_entry)
                merged_entry["referable_object_ids"] = list(sorted_role_ids)
                added_frame_count += 1

            for field_name in (
                "candidate_visible_object_ids",
                "selector_visible_object_ids",
                "pipeline_visible_object_ids_used_for_generation",
                "visible_object_ids",
            ):
                existing_ids = _normalize_object_ids(merged_entry.get(field_name))
                if field_name == "candidate_visible_object_ids" or existing_ids:
                    merged_entry[field_name] = sorted(set(existing_ids) | role_ids)

            merged_entry.update(
                {
                    "scene_id": scene_id,
                    "image_name": image_name,
                    "frame_usable": True,
                    "final_selection_rank": -1,
                    "attachment_referable_object_ids": list(sorted_role_ids),
                    "attachment_referable_pairs": attachment_pairs,
                    "attachment_referable_pair_count": len(attachment_pairs),
                    "attachment_selector_signal": {
                        "well_cropped_pair_count": len(attachment_pairs),
                        "viewpoint_exempt": True,
                    },
                    "attachment_final_referability": {
                        "object_ids": list(sorted_role_ids),
                        "pairs": attachment_pairs,
                        "pair_count": len(attachment_pairs),
                    },
                    "attachment_final_frame_selection": {
                        "selected_for_final_cache": True,
                        "selection_rank": -1,
                    },
                    "manual_attachment_override": True,
                    "manual_attachment_role_sets": role_sets,
                    "manual_attachment_roles": role_sets[0],
                }
            )
            target_scene_frames[image_name] = merged_entry

    existing_graphs = merged.get("manual_attachment_graph")
    if not isinstance(existing_graphs, dict):
        existing_graphs = {}
    existing_graphs.update(manual_graph)
    merged["manual_attachment_graph"] = existing_graphs
    merged["manual_attachment_scene_ids"] = sorted(set(manual_scene_ids))
    logger.info(
        "Applied manual attachment cache: scenes=%d overlapping_frames=%d added_frames=%d",
        len(set(manual_scene_ids)),
        merged_frame_count,
        added_frame_count,
    )
    return merged


def _get_referability_entry(cache: dict | None, scene_id: str, image_name: str) -> dict | None:
    if not cache:
        return None
    frames = cache.get("frames", cache)
    scene_frames = frames.get(scene_id)
    if isinstance(scene_frames, dict):
        entry = scene_frames.get(image_name)
        if not isinstance(entry, dict):
            return entry
        if not _manual_attachment_roles_for_frame(entry) and not _frame_entry_has_consistent_final_fields(entry):
            raise ValueError(
                f"Referability cache entry for {scene_id}/{image_name} is inconsistent with cache version "
                f"{EXPECTED_REFERABILITY_CACHE_VERSION}. Regenerate the referability cache instead of repairing it at read time."
            )
        return entry
    entry = frames.get(f"{scene_id}/{image_name}")
    if not isinstance(entry, dict):
        return entry
    if not _manual_attachment_roles_for_frame(entry) and not _frame_entry_has_consistent_final_fields(entry):
        raise ValueError(
            f"Referability cache entry for {scene_id}/{image_name} is inconsistent with cache version "
            f"{EXPECTED_REFERABILITY_CACHE_VERSION}. Regenerate the referability cache instead of repairing it at read time."
        )
    return entry


def _resolve_vlm_api_key(*, purpose: str, missing_key_hint: str | None = None) -> str:
    for env_name in VLM_API_KEY_ENV_NAMES:
        api_key = os.getenv(env_name)
        if api_key:
            return api_key

    hint = f" {missing_key_hint}" if missing_key_hint else ""
    logger.warning(
        "%s is using placeholder API key %r because neither %s nor %s is set.%s",
        purpose,
        PLACEHOLDER_VLM_API_KEY,
        VLM_API_KEY_ENV_NAMES[0],
        VLM_API_KEY_ENV_NAMES[1],
        hint,
    )
    return PLACEHOLDER_VLM_API_KEY


def _encode_review_image_to_base64(image: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", image)
    if not ok:
        raise ValueError("Failed to encode review image")
    return base64.b64encode(buf.tobytes()).decode()


def _extract_json_object(text: str) -> dict | None:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def _image_path_to_base64(path: Path) -> tuple[str, str]:
    ext = path.suffix.lstrip(".").lower()
    mime = "image/jpeg" if ext in ("jpg", "jpeg") else f"image/{ext}"
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode(), mime


def _is_question_review_retryable_error(exc: Exception) -> bool:
    text = str(exc or "").lower()
    return (
        "concurrent_request_limit_exceeded" in text
        or "too many concurrent requests" in text
    )


def _is_authentication_error(exc: Exception) -> bool:
    text = str(exc or "").lower()
    return (
        "401" in text
        or "unauthorized" in text
        or "authentication" in text
        or "invalid api key" in text
    )


def _call_question_review_vlm(create_fn, *, context: str):
    last_exc: Exception | None = None
    for attempt in range(1, QUESTION_REVIEW_MAX_RETRIES + 1):
        try:
            return create_fn()
        except Exception as exc:
            last_exc = exc
            if _is_authentication_error(exc):
                raise RuntimeError(
                    f"{context} failed with an authentication error: {exc}. "
                    "Set DASHSCOPE_API_KEY or OPENAI_API_KEY for the configured VLM endpoint, "
                    "or disable this step with --no-question_presence_review."
                ) from exc
            if (
                not _is_question_review_retryable_error(exc)
                or attempt >= QUESTION_REVIEW_MAX_RETRIES
            ):
                raise
            delay_seconds = QUESTION_REVIEW_RETRY_DELAY_SECONDS * attempt
            logger.warning(
                "%s hit a VLM concurrency limit (%d/%d). Retrying in %.1fs: %s",
                context,
                attempt,
                QUESTION_REVIEW_MAX_RETRIES,
                delay_seconds,
                exc,
            )
            time.sleep(delay_seconds)
    if last_exc is None:
        raise RuntimeError(f"{context} failed without raising a review error")
    raise last_exc


def _normalize_question_presence_status(value: object) -> str | None:
    text = str(value or "").strip().lower()
    if text in {"present", "visible", "in_image", "in image", "yes"}:
        return "present"
    if text in {"absent", "missing", "not_present", "not present", "no"}:
        return "absent"
    if text in {"unsure", "uncertain", "unknown", "cannot_tell", "can't tell"}:
        return "unsure"
    return None


def _dedupe_strings(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _normalize_label_to_object_ids(value: object) -> dict[str, list[int]]:
    return _shared_normalize_label_to_object_ids(value)


def _coerce_object_id(value: object) -> int | None:
    return _shared_coerce_object_id(value)


def _attachment_human_review_priority_pairs(cards: object) -> list[tuple[int, int]]:
    if not isinstance(cards, list):
        return []
    pairs: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for card in cards:
        if not isinstance(card, dict):
            continue
        parent_id = _coerce_object_id(card.get("parent_id"))
        child_id = _coerce_object_id(card.get("child_id"))
        if parent_id is None or child_id is None:
            continue
        pair = (int(parent_id), int(child_id))
        if pair in seen:
            continue
        seen.add(pair)
        pairs.append(pair)
    return pairs


def _iter_question_referability_mentions(
    question: dict[str, object],
    objects_by_id: dict[int, dict[str, object]],
) -> list[dict[str, object]]:
    return _shared_collect_question_mentions(question, objects_by_id)


def _question_uses_attachment_referability(question: dict[str, object]) -> bool:
    question_type = str(question.get("type", "")).strip().lower()
    if question_type == "attachment_chain" or question_type.startswith("attachment"):
        return True
    # ``attachment_remapped`` only means the move dragged attached children along
    # (``len(moved_ids) > 1``). That is still a plain object_move question when the
    # queried object is the moved object itself. It genuinely exercises the support
    # relation only when the query object is a dragged-along child (moved != query).
    if not bool(question.get("attachment_remapped", False)):
        return False
    moved_obj_id = _coerce_object_id(question.get("moved_obj_id"))
    query_obj_id = _coerce_object_id(question.get("query_obj_id"))
    return moved_obj_id is not None and query_obj_id is not None and moved_obj_id != query_obj_id


def _build_question_referability_audit(
    question: dict[str, object],
    *,
    objects_by_id: dict[int, dict[str, object]],
    referability_entry: dict[str, object] | None,
    frame_referable_ids: list[int],
    attachment_frame_referable_ids: list[int] | None = None,
    attachment_frame_referable_pairs: list[tuple[int, int]] | None = None,
) -> dict[str, object]:
    return _shared_build_question_referability_audit(
        question,
        objects_by_id=objects_by_id,
        label_statuses=(referability_entry or {}).get("label_statuses"),
        label_to_object_ids=(referability_entry or {}).get("label_to_object_ids"),
        frame_referable_ids=frame_referable_ids,
        attachment_frame_referable_ids=attachment_frame_referable_ids,
        attachment_referable_pairs=(
            attachment_frame_referable_pairs
            if attachment_frame_referable_pairs is not None
            else (
                (referability_entry or {}).get("attachment_referable_pairs")
                if referability_entry is not None
                else None
            )
        ),
    )


def _apply_question_referability_filter(
    questions: list[dict[str, object]],
    *,
    objects_by_id: dict[int, dict[str, object]],
    referability_entry: dict[str, object] | None,
    frame_referable_ids: list[int],
    attachment_frame_referable_ids: list[int] | None = None,
    attachment_frame_referable_pairs: list[tuple[int, int]] | None = None,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    kept_questions: list[dict[str, object]] = []
    audited_questions: list[dict[str, object]] = []
    dropped_summaries: list[str] = []

    for question in questions:
        audited_question = dict(question)
        audit = _build_question_referability_audit(
            audited_question,
            objects_by_id=objects_by_id,
            referability_entry=referability_entry,
            frame_referable_ids=frame_referable_ids,
            attachment_frame_referable_ids=attachment_frame_referable_ids,
            attachment_frame_referable_pairs=attachment_frame_referable_pairs,
        )
        audited_question["question_referability_audit"] = audit
        audited_questions.append(audited_question)
        if audit.get("decision") == "pass":
            kept_questions.append(audited_question)
            continue
        dropped_summaries.append(
            "  scene="
            f"{audited_question.get('scene_id', '<unknown>')} "
            "frame="
            f"{audited_question.get('image_name', '<unknown>')} "
            "type="
            f"{audited_question.get('type', '<unknown>')} "
            "reasons="
            f"{audit.get('reason_codes', [])}"
        )

    if dropped_summaries:
        raise AssertionError(
            "Referability backstop detected "
            f"{len(dropped_summaries)} question(s) that should have been prevented by the generator "
            "(generator bug):\n"
            + "\n".join(dropped_summaries)
        )
    return kept_questions, audited_questions


def _question_review_scene_metadata_path(output_dir: Path, scene_id: str) -> Path:
    return output_dir / "scene_metadata" / f"{scene_id}.json"


def _build_question_review_crop(
    image: np.ndarray,
    visibility_meta: dict[str, object],
) -> dict[str, object]:
    roi_bounds = visibility_meta.get("roi_bounds_px")
    projected_area_px = float(visibility_meta.get("projected_area_px", 0.0) or 0.0)
    bbox_in_frame_ratio = float(visibility_meta.get("bbox_in_frame_ratio", 0.0) or 0.0)
    edge_margin_px = float(visibility_meta.get("edge_margin_px", 0.0) or 0.0)
    result = {
        "valid": False,
        "reason": "missing_projection",
        "roi_bounds_px": None,
        "crop_bounds_px": None,
        "projected_area_px": projected_area_px,
        "bbox_in_frame_ratio": bbox_in_frame_ratio,
        "edge_margin_px": edge_margin_px,
        "image_b64": None,
        "mime": "image/jpeg",
    }
    if not isinstance(roi_bounds, (list, tuple)) or len(roi_bounds) != 4:
        return result

    try:
        u_min, u_max, v_min, v_max = [int(value) for value in roi_bounds]
    except (TypeError, ValueError):
        return result

    width = max(0, u_max - u_min)
    height = max(0, v_max - v_min)
    if width <= 0 or height <= 0:
        return result

    pad = int(round(
        max(
            QUESTION_REVIEW_CROP_MIN_PADDING_PX,
            min(
                QUESTION_REVIEW_CROP_PADDING_RATIO * max(width, height),
                QUESTION_REVIEW_CROP_MAX_PADDING_PX,
            ),
        )
    ))
    crop_u_min = max(0, u_min - pad)
    crop_u_max = min(int(image.shape[1]), u_max + pad)
    crop_v_min = max(0, v_min - pad)
    crop_v_max = min(int(image.shape[0]), v_max + pad)

    crop_width = max(0, crop_u_max - crop_u_min)
    crop_height = max(0, crop_v_max - crop_v_min)
    result["roi_bounds_px"] = [u_min, u_max, v_min, v_max]
    result["crop_bounds_px"] = [crop_u_min, crop_u_max, crop_v_min, crop_v_max]

    # Presence review uses looser thresholds than strict referability filtering:
    # the goal is to crop likely-visible instances, not to enforce benchmark quality.
    if (
        crop_width < QUESTION_REVIEW_CROP_MIN_DIM_PX
        or crop_height < QUESTION_REVIEW_CROP_MIN_DIM_PX
        or projected_area_px < question_review_crop_min_projected_area_px(
            int(image.shape[1]), int(image.shape[0])
        )
        or bbox_in_frame_ratio < QUESTION_REVIEW_CROP_MIN_IN_FRAME_RATIO
    ):
        result["reason"] = "invalid_crop"
        return result

    crop_image = image[crop_v_min:crop_v_max, crop_u_min:crop_u_max]
    if crop_image.size == 0:
        return result

    result["valid"] = True
    result["reason"] = ""
    result["image_b64"] = _encode_review_image_to_base64(crop_image)
    return result


def _build_question_review_scene_context(
    *,
    scene_id: str,
    data_root: Path,
    output_dir: Path,
    dataset: str = "scannet",
    scannetpp_sensor: str = "iphone",
) -> dict[str, object]:
    scene_dir = data_root / scene_id
    scene = None
    errors: list[str] = []
    metadata_path = _question_review_scene_metadata_path(output_dir, scene_id)

    if metadata_path.exists():
        try:
            scene = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning(
                "Failed to load scene metadata for question review %s: %s",
                scene_id,
                e,
            )
            errors.append("invalid_scene_metadata")
    elif scene_dir.exists():
        try:
            if dataset == "scannetpp":
                scene = parse_scene(scene_dir, dataset="scannetpp")
            else:
                scene = parse_scene(scene_dir)
        except Exception as e:
            logger.warning("Question review parse fallback failed for %s: %s", scene_id, e)
            errors.append("parse_scene_failed")
    else:
        errors.append("scene_dir_missing")

    objects = scene.get("objects", []) if isinstance(scene, dict) else []
    objects_by_id: dict[int, dict[str, object]] = {}
    if isinstance(objects, list):
        for obj in objects:
            if not isinstance(obj, dict):
                continue
            obj_id = _coerce_object_id(obj.get("id"))
            if obj_id is None:
                continue
            objects_by_id[obj_id] = obj
    if not objects_by_id:
        errors.append("missing_scene_objects")

    poses: dict[str, object] = {}
    color_intrinsics = None
    if scene_dir.exists():
        from src.datasets import make_data_source
        ds_review = make_data_source(dataset, scene_dir, sensor=scannetpp_sensor)
        try:
            poses = ds_review.load_poses()
        except Exception as e:
            logger.warning("Question review pose load failed for %s: %s", scene_id, e)
            errors.append("missing_pose_data")
        try:
            color_intrinsics = ds_review.load_intrinsics()
        except Exception as e:
            logger.warning(
                "Question review color intrinsics load failed for %s: %s",
                scene_id,
                e,
            )
            errors.append("missing_color_intrinsics")

    return {
        "scene_id": scene_id,
        "scene_dir": scene_dir if scene_dir.exists() else None,
        "objects": objects,
        "objects_by_id": objects_by_id,
        "poses": poses,
        "color_intrinsics": color_intrinsics,
        "errors": _dedupe_strings(errors),
    }


def _build_question_review_frame_context(
    *,
    scene_id: str,
    image_name: str,
    data_root: Path,
    scene_context: dict[str, object],
    dataset: str = "scannet",
    scannetpp_sensor: str = "iphone",
    scannetpp_frame_root: str | None = None,
) -> dict[str, object]:
    from src.datasets import make_data_source
    ds_review = make_data_source(dataset, data_root / scene_id, sensor=scannetpp_sensor,
                                 frame_root=scannetpp_frame_root)
    image_path = ds_review.image_path(image_name)
    image_exists = image_path.exists()
    image_b64 = None
    mime = "image/jpeg"
    image = None
    errors = list(scene_context.get("errors", []))

    if image_exists:
        try:
            image_b64, mime = _image_path_to_base64(image_path)
        except Exception as e:
            logger.warning(
                "Question review image encode failed for %s/%s: %s",
                scene_id,
                image_name,
                e,
            )
            errors.append("image_encode_failed")
        image = cv2.imread(str(image_path))
        if image is None:
            errors.append("image_unreadable")
    else:
        errors.append("image_not_found")

    objects = scene_context.get("objects", [])
    objects_by_id = dict(scene_context.get("objects_by_id", {}))
    poses = scene_context.get("poses", {})
    pose = poses.get(image_name) if isinstance(poses, dict) else None
    color_intrinsics = scene_context.get("color_intrinsics")
    if pose is None:
        errors.append("missing_pose")
    if color_intrinsics is None:
        errors.append("missing_color_intrinsics")
    if not objects_by_id:
        errors.append("missing_scene_objects")

    has_projection_context = (
        image is not None
        and pose is not None
        and color_intrinsics is not None
        and isinstance(objects, list)
        and bool(objects)
    )
    visibility_by_obj_id: dict[int, dict[str, object]] = {}
    crop_by_obj_id: dict[int, dict[str, object]] = {}
    if has_projection_context:
        try:
            raw_visibility = compute_frame_object_visibility(
                objects=objects,
                pose=pose,
                color_intrinsics=color_intrinsics,
                image_path=image_path,
                depth_image=None,
                depth_intrinsics=None,
            )
            visibility_by_obj_id = {
                int(obj_id): meta
                for obj_id, meta in raw_visibility.items()
            }
            for obj_id, meta in visibility_by_obj_id.items():
                crop_by_obj_id[int(obj_id)] = _build_question_review_crop(image, meta)
        except Exception as e:
            logger.warning(
                "Question review visibility build failed for %s/%s: %s",
                scene_id,
                image_name,
                e,
            )
            errors.append("visibility_compute_failed")
            has_projection_context = False

    return {
        "scene_id": scene_id,
        "image_name": image_name,
        "scene_dir": scene_context.get("scene_dir"),
        "image_path": image_path,
        "image_exists": image_exists,
        "image_b64": image_b64,
        "mime": mime,
        "objects_by_id": objects_by_id,
        "pose": pose,
        "color_intrinsics": color_intrinsics,
        "visibility_by_obj_id": visibility_by_obj_id,
        "crop_by_obj_id": crop_by_obj_id,
        "has_projection_context": has_projection_context,
        "context_errors": _dedupe_strings(errors),
    }


def _prebuild_question_review_frame_contexts(
    *,
    questions: list[dict[str, object]],
    data_root: Path,
    output_dir: Path,
    dataset: str = "scannet",
    scannetpp_sensor: str = "iphone",
    scannetpp_frame_root: str | None = None,
) -> dict[tuple[str, str], dict[str, object]]:
    frame_keys: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for question in questions:
        scene_id = str(question.get("scene_id", "")).strip()
        image_names = [str(question.get("image_name", "")).strip()]
        reasoning_frame_2 = str(question.get("reasoning_frame_2", "")).strip()
        if reasoning_frame_2:
            image_names.append(reasoning_frame_2)
        for image_name in image_names:
            key = (scene_id, image_name)
            if not image_name or key in seen:
                continue
            seen.add(key)
            frame_keys.append(key)

    scene_contexts: dict[str, dict[str, object]] = {}
    frame_contexts: dict[tuple[str, str], dict[str, object]] = {}
    for scene_id, image_name in frame_keys:
        if scene_id not in scene_contexts:
            scene_contexts[scene_id] = _build_question_review_scene_context(
                scene_id=scene_id,
                data_root=data_root,
                output_dir=output_dir,
                dataset=dataset,
                scannetpp_sensor=scannetpp_sensor,
            )
        frame_contexts[(scene_id, image_name)] = _build_question_review_frame_context(
            scene_id=scene_id,
            image_name=image_name,
            data_root=data_root,
            scene_context=scene_contexts[scene_id],
            dataset=dataset,
            scannetpp_sensor=scannetpp_sensor,
            scannetpp_frame_root=scannetpp_frame_root,
        )
    return frame_contexts


def _collect_question_presence_targets(
    question: dict[str, object],
    objects_by_id: dict[int, dict[str, object]],
) -> list[dict[str, object]]:
    targets: list[dict[str, object]] = []
    targets_by_obj_id: dict[int, dict[str, object]] = {}
    unresolved_targets: dict[str, dict[str, object]] = {}

    for idx, mention in enumerate(_iter_question_referability_mentions(question, objects_by_id)):
        label = str(mention.get("label", "")).strip()
        label_key = label.lower()
        if label_key in EXCLUDED_LABELS:
            continue
        role = str(mention.get("role", "mentioned")).strip() or "mentioned"
        obj_id = _coerce_object_id(mention.get("obj_id"))

        if obj_id is not None:
            target = targets_by_obj_id.get(obj_id)
            if target is None:
                target = {
                    "sort_index": idx,
                    "label": label,
                    "obj_id": obj_id,
                    "roles": [role],
                }
                targets_by_obj_id[obj_id] = target
                targets.append(target)
            else:
                if not str(target.get("label", "")).strip() and label:
                    target["label"] = label
                if role not in target["roles"]:
                    target["roles"].append(role)
            continue

        unresolved_key = label_key or f"unresolved:{idx}"
        target = unresolved_targets.get(unresolved_key)
        if target is None:
            target = {
                "sort_index": idx,
                "label": label,
                "obj_id": None,
                "roles": [role],
            }
            unresolved_targets[unresolved_key] = target
            targets.append(target)
        elif role not in target["roles"]:
            target["roles"].append(role)

    normalized_targets: list[dict[str, object]] = []
    for target in sorted(targets, key=lambda item: int(item.get("sort_index", 0))):
        normalized_targets.append(
            {
                "label": str(target.get("label", "")).strip(),
                "obj_id": _coerce_object_id(target.get("obj_id")),
                "roles": sorted(
                    {
                        str(role).strip()
                        for role in target.get("roles", [])
                        if str(role).strip()
                    }
                ),
            }
        )
    return normalized_targets


def _question_presence_prompt(
    question_text: str,
    targets: list[dict[str, object]],
) -> str:
    targets_json = json.dumps(
        [
            {
                "crop_index": idx + 1,
                "label": str(target.get("label", "")).strip(),
                "roles": list(target.get("roles", [])),
            }
            for idx, target in enumerate(targets)
        ],
        ensure_ascii=False,
    )
    return (
        "You are auditing whether specific object instances mentioned in a visual question are clearly visible "
        "in the frame.\n"
        "You will receive the full scene image first, followed by one crop for each target instance.\n"
        "Each crop appears in the same order as the Targets list, so crop_index 1 refers to the first crop after "
        "the full image.\n"
        "Use the crop as the primary evidence and the full image only as context.\n"
        "Judge each crop_index independently.\n"
        "Return present only if the crop clearly shows the target instance and the object in the crop is "
        "recognizable as the given label from the crop itself.\n"
        "For occlusion review, treat any visible blocking by another object as occlusion, even if the blocking "
        "object is very small. If the blocking pixels belong to a different object or label, that still counts "
        "as occlusion.\n"
        "Return unsure if the crop does not provide enough evidence to tell that the object is the given label.\n"
        "Return absent if the target instance is not visible in the image.\n"
        "Return strict JSON only with this schema:\n"
        '{"objects":[{"crop_index":1,"status":"present","reason":"short reason"}]}\n'
        f"Question: {question_text}\n"
        f"Targets: {targets_json}"
    )


def _should_run_question_presence_review(question: dict[str, object]) -> bool:
    if (
        str(question.get("level", "")).strip().upper() == "L1"
        and str(question.get("type", "")).strip() == "occlusion"
    ):
        return True
    return (
        str(question.get("level", "")).strip().upper() == "L2"
        and str(question.get("type", "")).strip() == "object_move_occlusion"
        and _coerce_object_id(question.get("moved_obj_id")) is not None
        and _coerce_object_id(question.get("moved_obj_id"))
        == _coerce_object_id(question.get("query_obj_id"))
    )


def _should_run_attachment_pair_review(question: dict[str, object]) -> bool:
    if str(question.get("level", "")).strip().upper() != "L2":
        return False
    qtype = str(question.get("type", "")).strip()
    if qtype not in {
        "object_move_agent",
        "object_move_distance",
        "object_move_occlusion",
        "object_move_object_centric",
        "object_rotate_object_centric",
        "object_move_allocentric",
    }:
        return False
    moved_obj_id = _coerce_object_id(question.get("moved_obj_id"))
    query_obj_id = _coerce_object_id(question.get("query_obj_id"))
    return moved_obj_id is not None and query_obj_id is not None and moved_obj_id != query_obj_id


def _build_presence_review_entry(
    target: dict[str, object],
    *,
    status: str,
    reason: str,
) -> dict[str, object]:
    roi_bounds = target.get("roi_bounds_px")
    normalized_roi = None
    if isinstance(roi_bounds, (list, tuple)) and len(roi_bounds) == 4:
        try:
            normalized_roi = [int(value) for value in roi_bounds]
        except (TypeError, ValueError):
            normalized_roi = None
    return {
        "label": str(target.get("label", "")).strip(),
        "obj_id": _coerce_object_id(target.get("obj_id")),
        "roles": _dedupe_strings(
            [str(role).strip() for role in target.get("roles", []) if str(role).strip()]
        ),
        "status": status,
        "reason": reason,
        "roi_bounds_px": normalized_roi,
    }


def _finalize_presence_review(
    object_reviews: list[dict[str, object]],
    *,
    raw_response: str,
) -> dict[str, object]:
    flagged_labels: list[str] = []
    flagged_object_ids: list[int] = []
    seen_labels: set[str] = set()
    seen_obj_ids: set[int] = set()
    flagged = False
    for item in object_reviews:
        if not isinstance(item, dict):
            continue
        status = str(item.get("status", "")).strip()
        if status not in {"absent", "unsure"}:
            continue
        flagged = True
        label = str(item.get("label", "")).strip()
        if label and label not in seen_labels:
            seen_labels.add(label)
            flagged_labels.append(label)
        obj_id = _coerce_object_id(item.get("obj_id"))
        if obj_id is not None and obj_id not in seen_obj_ids:
            seen_obj_ids.add(obj_id)
            flagged_object_ids.append(obj_id)
    return {
        "review_mode": "instance",
        "decision": "manual_review" if flagged else "pass",
        "flagged_labels": flagged_labels,
        "flagged_object_ids": flagged_object_ids,
        "object_reviews": object_reviews,
        "raw_response": raw_response,
    }


def _resolve_question_review_vlm(
    vlm_url: str | None,
    vlm_model: str | None,
    *,
    purpose: str,
):
    if not vlm_url:
        raise ValueError(f"{purpose} requires a VLM URL")

    from openai import OpenAI

    api_key = _resolve_vlm_api_key(
        purpose=purpose,
        missing_key_hint=(
            "If this endpoint requires authentication, set one of those environment "
            "variables before using this VLM endpoint."
        ),
    )
    client = OpenAI(api_key=api_key, base_url=vlm_url)
    model_name = vlm_model
    if not model_name:
        try:
            models = client.models.list()
            available = [m.id for m in models.data]
            if not available:
                raise RuntimeError("No VLM models available")
            model_name = available[0]
        except Exception as e:
            raise RuntimeError(f"Cannot reach {purpose} VLM at {vlm_url}: {e}") from e

    return client, model_name


def _make_question_presence_reviewer(client, model_name: str):
    logger.info("Using question presence review VLM model: %s", model_name)

    def _review(
        frame_context: dict[str, object],
        question: dict[str, object],
        targets: list[dict[str, object]],
    ) -> dict[str, object]:
        image_b64 = str(frame_context.get("image_b64", "") or "")
        mime = str(frame_context.get("mime", "") or "image/jpeg")
        content: list[dict[str, object]] = [
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{image_b64}"}}
        ]
        for target in targets:
            crop_b64 = str(target.get("crop_image_b64", "") or "")
            crop_mime = str(target.get("crop_mime", "") or "image/jpeg")
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{crop_mime};base64,{crop_b64}"},
                }
            )
        content.append(
            {
                "type": "text",
                "text": _question_presence_prompt(str(question.get("question", "")), targets),
            }
        )
        resp = _call_question_review_vlm(
            lambda: client.chat.completions.create(
                model=model_name,
                messages=[{
                    "role": "user",
                    "content": content,
                }],
                max_tokens=min(
                    QUESTION_REVIEW_MAX_TOKENS_CAP,
                    max(256, QUESTION_REVIEW_MAX_TOKENS_PER_TARGET * max(1, len(targets))),
                ),
                temperature=0,
            ),
            context=f"question presence review for {frame_context.get('image_name', '<unknown>')}",
        )
        raw_text = (resp.choices[0].message.content or "").strip()
        parsed = _extract_json_object(raw_text)

        target_by_obj_id = {
            int(target["obj_id"]): target
            for target in targets
            if _coerce_object_id(target.get("obj_id")) is not None
        }
        mapped_reviews: dict[int, dict[str, object]] = {}
        objects = parsed.get("objects") if isinstance(parsed, dict) else None
        if isinstance(objects, list):
            for item in objects:
                if not isinstance(item, dict):
                    continue
                target = None
                crop_index = _coerce_object_id(item.get("crop_index"))
                if crop_index is not None and 1 <= crop_index <= len(targets):
                    target = targets[crop_index - 1]
                else:
                    obj_id = _coerce_object_id(item.get("obj_id"))
                    if obj_id is not None:
                        target = target_by_obj_id.get(obj_id)
                if target is None:
                    continue
                status = _normalize_question_presence_status(item.get("status")) or "unsure"
                target_obj_id = _coerce_object_id(target.get("obj_id"))
                if target_obj_id is None:
                    continue
                mapped_reviews[target_obj_id] = _build_presence_review_entry(
                    target,
                    status=status,
                    reason=str(item.get("reason", "")).strip(),
                )

        object_reviews: list[dict[str, object]] = []
        for target in targets:
            obj_id = int(target["obj_id"])
            object_reviews.append(
                mapped_reviews.get(
                    obj_id,
                    _build_presence_review_entry(
                        target,
                        status="unsure",
                        reason="missing_obj_id_in_vlm_response",
                    ),
                )
            )
        return {
            "object_reviews": object_reviews,
            "raw_response": raw_text,
        }

    return model_name, _review


def _attachment_pair_prompt(question_text: str, moved_label: str, query_label: str) -> str:
    return (
        "You are reviewing whether an L2 attachment question is a valid attachment-pair judgment.\n"
        "Judge whether the moved object and query object are two distinct objects and whether the question is "
        "about their attachment relation rather than the same object.\n"
        "Return strict JSON only with this schema:\n"
        '{"decision":"pass","reason":""}\n'
        f"Question: {question_text}\n"
        f"Moved object: {moved_label}\n"
        f"Query object: {query_label}"
    )


def _normalize_attachment_pair_review_decision(value: object) -> str:
    text = str(value or "").strip().lower()
    if text in {"pass", "manual_review"}:
        return text
    return "manual_review"


def _build_attachment_pair_review_entry(
    question: dict[str, object],
    *,
    decision: str,
    reason: str,
    raw_response: str,
) -> dict[str, object]:
    return {
        "decision": decision,
        "reason": reason,
        "moved_obj_id": _coerce_object_id(question.get("moved_obj_id")),
        "query_obj_id": _coerce_object_id(question.get("query_obj_id")),
        "moved_obj_label": str(question.get("moved_obj_label", "")).strip(),
        "query_obj_label": str(question.get("query_obj_label", "")).strip(),
        "raw_response": raw_response,
    }


def _make_attachment_pair_reviewer(client, model_name: str):
    logger.info("Using attachment pair review VLM model: %s", model_name)

    def _review(
        frame_context: dict[str, object],
        question: dict[str, object],
        targets: list[dict[str, object]],
    ) -> dict[str, object]:
        image_b64 = str(frame_context.get("image_b64", "") or "")
        mime = str(frame_context.get("mime", "") or "image/jpeg")
        content: list[dict[str, object]] = [
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{image_b64}"}}
        ]
        for target in targets:
            crop_b64 = str(target.get("crop_image_b64", "") or "")
            crop_mime = str(target.get("crop_mime", "") or "image/jpeg")
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{crop_mime};base64,{crop_b64}"},
                }
            )
        content.append(
            {
                "type": "text",
                "text": _attachment_pair_prompt(
                    str(question.get("question", "")),
                    str(question.get("moved_obj_label", "")).strip(),
                    str(question.get("query_obj_label", "")).strip(),
                ),
            }
        )
        resp = _call_question_review_vlm(
            lambda: client.chat.completions.create(
                model=model_name,
                messages=[{
                    "role": "user",
                    "content": content,
                }],
                max_tokens=256,
                temperature=0,
            ),
            context=f"attachment pair review for {frame_context.get('image_name', '<unknown>')}",
        )
        raw_text = (resp.choices[0].message.content or "").strip()
        parsed = _extract_json_object(raw_text) or {}
        decision = _normalize_attachment_pair_review_decision(parsed.get("decision"))
        reason = str(parsed.get("reason", "")).strip()
        return {
            "pair_review": _build_attachment_pair_review_entry(
                question,
                decision=decision,
                reason=reason,
                raw_response=raw_text,
            ),
            "raw_response": raw_text,
        }

    return model_name, _review


def _manual_review_reason_from_presence_review(review: dict[str, object]) -> str:
    object_reviews = review.get("object_reviews", [])
    if not isinstance(object_reviews, list):
        return "VLM marked this question for manual review."
    parts: list[str] = []
    for item in object_reviews:
        if not isinstance(item, dict):
            continue
        status = str(item.get("status", "")).strip()
        if status not in {"absent", "unsure"}:
            continue
        label = str(item.get("label", "")).strip() or "object"
        obj_id = _coerce_object_id(item.get("obj_id"))
        if obj_id is not None:
            parts.append(f"{label}#{obj_id}={status}")
        else:
            parts.append(f"{label}={status}")
    if parts:
        return "VLM flagged mentioned objects: " + ", ".join(parts)
    return "VLM marked this question for manual review."


def _manual_review_reason_from_post_generation_review(review: dict[str, object]) -> str:
    reason_codes = [
        str(code).strip()
        for code in review.get("reason_codes", [])
        if str(code).strip()
    ] if isinstance(review.get("reason_codes", []), list) else []
    if reason_codes:
        return "Post-generation audit flagged: " + ", ".join(reason_codes)
    return "Post-generation audit marked this question for manual review."


def _manual_review_reason_from_attachment_pair_review(review: dict[str, object]) -> str:
    reason = str(review.get("reason", "")).strip()
    if reason:
        return "Attachment-pair review flagged: " + reason
    moved_label = str(review.get("moved_obj_label", "")).strip() or "moved object"
    query_label = str(review.get("query_obj_label", "")).strip() or "query object"
    moved_obj_id = _coerce_object_id(review.get("moved_obj_id"))
    query_obj_id = _coerce_object_id(review.get("query_obj_id"))
    moved_text = f"{moved_label}#{moved_obj_id}" if moved_obj_id is not None else moved_label
    query_text = f"{query_label}#{query_obj_id}" if query_obj_id is not None else query_label
    return f"Attachment-pair review flagged: {moved_text} vs {query_text}"


def _combine_manual_review_reasons(reasons: list[str]) -> str:
    cleaned: list[str] = []
    seen: set[str] = set()
    for reason in reasons:
        text = reason.strip()
        if not text or text in seen:
            continue
        seen.add(text)
        cleaned.append(text)
    return " | ".join(cleaned)


def _review_question_object_presence(
    review_fn,
    *,
    question_index: int,
    question: dict[str, object],
    data_root: Path,
    frame_context_by_key: dict[tuple[str, str], dict[str, object]],
) -> dict[str, object]:
    reviewed_question = dict(question)
    reviewed_question["benchmark_index"] = int(question_index)
    review_reasons: list[str] = []
    existing_post_review = reviewed_question.get("question_post_generation_review")
    if isinstance(existing_post_review, dict) and existing_post_review.get("decision") == "manual_review":
        review_reasons.append(_manual_review_reason_from_post_generation_review(existing_post_review))
    existing_manual_reason = str(reviewed_question.get("manual_review_reason", "")).strip()
    if existing_manual_reason:
        review_reasons.append(existing_manual_reason)

    scene_id = str(question.get("scene_id", "")).strip()
    image_name = str(question.get("image_name", "")).strip()
    frame_context = frame_context_by_key.get((scene_id, image_name))
    reasoning_frame_2 = str(question.get("reasoning_frame_2", "")).strip()
    frame_2_context = (
        frame_context_by_key.get((scene_id, reasoning_frame_2))
        if reasoning_frame_2 else None
    )

    objects_by_id = (
        dict(frame_context.get("objects_by_id", {}))
        if isinstance(frame_context, dict) else {}
    )
    if isinstance(frame_2_context, dict):
        objects_by_id.update(dict(frame_2_context.get("objects_by_id", {})))
    targets = _collect_question_presence_targets(question, objects_by_id)
    object_reviews: list[dict[str, object]] = []
    raw_responses: list[str] = []
    valid_targets_by_frame: dict[str, tuple[dict[str, object], list[dict[str, object]]]] = {}
    object_frame_groups = question.get("object_frame_groups")
    frame_2_ids = {
        int(value) for value in object_frame_groups.get("frame_2", [])
    } if isinstance(object_frame_groups, dict) else set()

    for target in targets:
        obj_id = _coerce_object_id(target.get("obj_id"))
        if obj_id is None:
            object_reviews.append(
                _build_presence_review_entry(
                    target,
                    status="unsure",
                    reason="missing_obj_id",
                )
            )
            continue
        target_context = frame_2_context if obj_id in frame_2_ids else frame_context
        if not isinstance(target_context, dict):
            object_reviews.append(
                _build_presence_review_entry(
                    target,
                    status="unsure",
                    reason="missing_frame_context",
                )
            )
            continue
        if not bool(target_context.get("image_exists", False)):
            object_reviews.append(
                _build_presence_review_entry(
                    target,
                    status="unsure",
                    reason="image_not_found",
                )
            )
            continue
        if obj_id not in objects_by_id:
            object_reviews.append(
                _build_presence_review_entry(
                    target,
                    status="unsure",
                    reason="object_not_in_scene",
                )
            )
            continue
        if (
            not bool(target_context.get("has_projection_context", False))
            or not str(target_context.get("image_b64", "") or "")
        ):
            object_reviews.append(
                _build_presence_review_entry(
                    target,
                    status="unsure",
                    reason="missing_frame_context",
                )
            )
            continue

        crop_entry = target_context.get("crop_by_obj_id", {}).get(obj_id)
        if not isinstance(crop_entry, dict):
            object_reviews.append(
                _build_presence_review_entry(
                    target,
                    status="unsure",
                    reason="missing_projection",
                )
            )
            continue
        if not bool(crop_entry.get("valid", False)):
            object_reviews.append(
                _build_presence_review_entry(
                    {
                        **target,
                        "roi_bounds_px": crop_entry.get("roi_bounds_px"),
                    },
                    status="unsure",
                    reason=str(crop_entry.get("reason", "")).strip() or "invalid_crop",
                )
            )
            continue

        target_image_name = str(target_context.get("image_name", "")).strip()
        grouped_context, grouped_targets = valid_targets_by_frame.setdefault(
            target_image_name,
            (target_context, []),
        )
        grouped_targets.append(
            {
                **target,
                "roi_bounds_px": crop_entry.get("roi_bounds_px"),
                "crop_image_b64": crop_entry.get("image_b64"),
                "crop_mime": crop_entry.get("mime", "image/jpeg"),
            }
        )

    for target_context, valid_targets in valid_targets_by_frame.values():
        try:
            vlm_review = review_fn(target_context, question, valid_targets)
            raw_response = str(vlm_review.get("raw_response", "") or "")
            if raw_response:
                raw_responses.append(raw_response)
            object_reviews.extend(list(vlm_review.get("object_reviews", [])))
        except Exception as e:
            object_reviews.extend(
                _build_presence_review_entry(
                    target,
                    status="unsure",
                    reason=f"VLM review failed: {e}",
                )
                for target in valid_targets
            )

    review = _finalize_presence_review(
        object_reviews,
        raw_response="\n".join(raw_responses),
    )
    if not targets:
        review = _finalize_presence_review([], raw_response="")
        review["decision"] = "pass"

    reviewed_question["question_presence_review"] = review
    if review.get("decision") == "manual_review":
        review_reasons.append(_manual_review_reason_from_presence_review(review))

    if review_reasons:
        reviewed_question["manual_review_reason"] = _combine_manual_review_reasons(review_reasons)
    else:
        reviewed_question.pop("manual_review_reason", None)
    return reviewed_question


def _review_question_attachment_pair(
    review_fn,
    *,
    question_index: int,
    question: dict[str, object],
    frame_context_by_key: dict[tuple[str, str], dict[str, object]],
) -> dict[str, object]:
    reviewed_question = dict(question)
    reviewed_question["benchmark_index"] = int(question_index)
    review_reasons: list[str] = []
    existing_post_review = reviewed_question.get("question_post_generation_review")
    if isinstance(existing_post_review, dict) and existing_post_review.get("decision") == "manual_review":
        review_reasons.append(_manual_review_reason_from_post_generation_review(existing_post_review))
    existing_manual_reason = str(reviewed_question.get("manual_review_reason", "")).strip()
    if existing_manual_reason:
        review_reasons.append(existing_manual_reason)
    scene_id = str(question.get("scene_id", "")).strip()
    image_name = str(question.get("image_name", "")).strip()
    frame_context = frame_context_by_key.get((scene_id, image_name))
    fallback_reason = ""

    moved_obj_id = _coerce_object_id(question.get("moved_obj_id"))
    query_obj_id = _coerce_object_id(question.get("query_obj_id"))
    if moved_obj_id is None or query_obj_id is None:
        fallback_reason = "missing moved/query object id"
    elif moved_obj_id == query_obj_id:
        fallback_reason = "self attachment pair should not be reviewed"
    elif not isinstance(frame_context, dict):
        fallback_reason = "missing frame context"
    elif not bool(frame_context.get("image_exists", False)):
        fallback_reason = "frame image missing"

    if fallback_reason:
        review = _build_attachment_pair_review_entry(
            question,
            decision="manual_review",
            reason=fallback_reason,
            raw_response="",
        )
        reviewed_question["question_attachment_pair_review"] = review
        review_reasons.append(_manual_review_reason_from_attachment_pair_review(review))
        reviewed_question["manual_review_reason"] = _combine_manual_review_reasons(review_reasons)
        return reviewed_question

    objects_by_id = dict(frame_context.get("objects_by_id", {}))
    moved_obj = objects_by_id.get(int(moved_obj_id))
    query_obj = objects_by_id.get(int(query_obj_id))
    if not isinstance(moved_obj, dict) or not isinstance(query_obj, dict):
        review = _build_attachment_pair_review_entry(
            question,
            decision="manual_review",
            reason="missing moved/query object metadata",
            raw_response="",
        )
        reviewed_question["question_attachment_pair_review"] = review
        review_reasons.append(_manual_review_reason_from_attachment_pair_review(review))
        reviewed_question["manual_review_reason"] = _combine_manual_review_reasons(review_reasons)
        return reviewed_question

    moved_crop = frame_context.get("crop_by_obj_id", {}).get(int(moved_obj_id))
    query_crop = frame_context.get("crop_by_obj_id", {}).get(int(query_obj_id))
    if not isinstance(moved_crop, dict) or not isinstance(query_crop, dict):
        review = _build_attachment_pair_review_entry(
            question,
            decision="manual_review",
            reason="missing moved/query object crop",
            raw_response="",
        )
        reviewed_question["question_attachment_pair_review"] = review
        review_reasons.append(_manual_review_reason_from_attachment_pair_review(review))
        reviewed_question["manual_review_reason"] = _combine_manual_review_reasons(review_reasons)
        return reviewed_question
    if not bool(moved_crop.get("valid", False)) or not bool(query_crop.get("valid", False)):
        review = _build_attachment_pair_review_entry(
            question,
            decision="manual_review",
            reason="invalid moved/query object crop",
            raw_response="",
        )
        reviewed_question["question_attachment_pair_review"] = review
        review_reasons.append(_manual_review_reason_from_attachment_pair_review(review))
        reviewed_question["manual_review_reason"] = _combine_manual_review_reasons(review_reasons)
        return reviewed_question

    pair_targets = [
        {
            "obj_id": int(moved_obj_id),
            "label": str(moved_obj.get("label", "")).strip(),
            "crop_image_b64": moved_crop.get("image_b64"),
            "crop_mime": moved_crop.get("mime", "image/jpeg"),
        },
        {
            "obj_id": int(query_obj_id),
            "label": str(query_obj.get("label", "")).strip(),
            "crop_image_b64": query_crop.get("image_b64"),
            "crop_mime": query_crop.get("mime", "image/jpeg"),
        },
    ]

    try:
        vlm_review = review_fn(frame_context, question, pair_targets)
    except Exception as e:
        review = _build_attachment_pair_review_entry(
            question,
            decision="manual_review",
            reason=f"VLM review failed: {e}",
            raw_response="",
        )
        reviewed_question["question_attachment_pair_review"] = review
        review_reasons.append(_manual_review_reason_from_attachment_pair_review(review))
        reviewed_question["manual_review_reason"] = _combine_manual_review_reasons(review_reasons)
        return reviewed_question

    pair_review = dict(vlm_review.get("pair_review", {}))
    review = _build_attachment_pair_review_entry(
        question,
        decision=_normalize_attachment_pair_review_decision(pair_review.get("decision")),
        reason=str(pair_review.get("reason", "")).strip(),
        raw_response=str(vlm_review.get("raw_response", "") or ""),
    )
    reviewed_question["question_attachment_pair_review"] = review
    if review.get("decision") == "manual_review":
        review_reasons.append(_manual_review_reason_from_attachment_pair_review(review))
    if review_reasons:
        reviewed_question["manual_review_reason"] = _combine_manual_review_reasons(review_reasons)
    else:
        reviewed_question.pop("manual_review_reason", None)
    return reviewed_question


def _run_question_presence_review(
    *,
    questions: list[dict[str, object]],
    data_root: Path,
    output_dir: Path,
    vlm_url: str | None,
    vlm_model: str | None,
    workers: int = 8,
    frame_context_by_key: dict[tuple[str, str], dict[str, object]] | None = None,
    dataset: str = "scannet",
    scannetpp_sensor: str = "iphone",
    scannetpp_frame_root: str | None = None,
) -> dict[str, object]:
    from scripts.make_viewer import build_viewer_html

    presence_review_targets = [
        (idx, question)
        for idx, question in enumerate(questions)
        if _should_run_question_presence_review(question)
    ]
    attachment_pair_review_targets = [
        (idx, question)
        for idx, question in enumerate(questions)
        if _should_run_attachment_pair_review(question)
    ]
    review_questions = [question for _, question in presence_review_targets + attachment_pair_review_targets]
    review_json_path = output_dir / "question_presence_review.json"
    flagged_json_path = output_dir / "question_presence_review_flagged.json"
    flagged_html_path = output_dir / "question_presence_review_flagged.html"
    viewer_image_root = data_root
    if dataset == "scannetpp" and scannetpp_sensor == "iphone":
        viewer_image_root = Path("output") / "scannetpp_iphone_frames"
    scannet_roots = [viewer_image_root] if dataset == "scannet" else []
    scannetpp_roots = [viewer_image_root] if dataset == "scannetpp" else []
    if not review_questions:
        review_payload = {
            "name": "PSR-Bench question presence review",
            "model": vlm_model,
            "reviewed_question_count": 0,
            "manual_review_count": 0,
            "referability_issue_count": 0,
            "attachment_pair_issue_count": 0,
            "post_generation_issue_count": 0,
            "questions": [],
        }
        flagged_payload = dict(review_payload)
        flagged_payload["name"] = "PSR-Bench question presence review (flagged)"
        with open(review_json_path, "w", encoding="utf-8") as f:
            json.dump(review_payload, f, indent=2, ensure_ascii=False)
        with open(flagged_json_path, "w", encoding="utf-8") as f:
            json.dump(flagged_payload, f, indent=2, ensure_ascii=False)
        flagged_html_path.write_text(
            build_viewer_html(
                [],
                scannet_roots,
                scannetpp_roots,
                scannetpp_sensor=scannetpp_sensor,
                title="question presence manual review",
                include_referability_audit=False,
                apply_filters=False,
            ),
            encoding="utf-8",
        )
        logger.info(
            "Question presence review skipped: no L1 occlusion or non-self L2 attachment-pair questions found."
        )
        return review_payload
    client, model_name = _resolve_question_review_vlm(
        vlm_url,
        vlm_model,
        purpose="question post-review",
    )
    _, presence_review_fn = _make_question_presence_reviewer(client, model_name)
    _, attachment_pair_review_fn = _make_attachment_pair_reviewer(client, model_name)
    reviewed_questions: list[dict[str, object]] = []
    if frame_context_by_key is None:
        frame_context_by_key = _prebuild_question_review_frame_contexts(
            questions=review_questions,
            data_root=data_root,
            output_dir=output_dir,
            dataset=dataset,
            scannetpp_sensor=scannetpp_sensor,
            scannetpp_frame_root=scannetpp_frame_root,
        )

    if presence_review_targets or attachment_pair_review_targets:
        with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
            futures = [
                pool.submit(
                    _review_question_object_presence,
                    presence_review_fn,
                    question_index=idx,
                    question=question,
                    data_root=data_root,
                    frame_context_by_key=frame_context_by_key,
                )
                for idx, question in presence_review_targets
            ]
            futures.extend(
                pool.submit(
                    _review_question_attachment_pair,
                    attachment_pair_review_fn,
                    question_index=idx,
                    question=question,
                    frame_context_by_key=frame_context_by_key,
                )
                for idx, question in attachment_pair_review_targets
            )
            for future in as_completed(futures):
                reviewed_questions.append(future.result())
    reviewed_questions.sort(key=lambda item: int(item.get("benchmark_index", -1)))

    referability_issue_count = sum(
        1
        for question in reviewed_questions
        if isinstance(question.get("question_presence_review"), dict)
        and question["question_presence_review"].get("decision") == "manual_review"
    )
    attachment_pair_issue_count = sum(
        1
        for question in reviewed_questions
        if isinstance(question.get("question_attachment_pair_review"), dict)
        and question["question_attachment_pair_review"].get("decision") == "manual_review"
    )
    post_generation_issue_count = sum(
        1
        for question in reviewed_questions
        if isinstance(question.get("question_post_generation_review"), dict)
        and question["question_post_generation_review"].get("decision") == "manual_review"
    )
    flagged_questions = [
        question for question in reviewed_questions
        if (
            isinstance(question.get("question_post_generation_review"), dict)
            and question["question_post_generation_review"].get("decision") == "manual_review"
        ) or (
            isinstance(question.get("question_presence_review"), dict)
            and question["question_presence_review"].get("decision") == "manual_review"
        ) or (
            isinstance(question.get("question_attachment_pair_review"), dict)
            and question["question_attachment_pair_review"].get("decision") == "manual_review"
        ) or bool(str(question.get("manual_review_reason", "")).strip())
    ]

    review_payload = {
        "name": "PSR-Bench question presence review",
        "model": model_name,
        "reviewed_question_count": len(reviewed_questions),
        "manual_review_count": len(flagged_questions),
        "referability_issue_count": referability_issue_count,
        "attachment_pair_issue_count": attachment_pair_issue_count,
        "post_generation_issue_count": post_generation_issue_count,
        "questions": reviewed_questions,
    }
    flagged_payload = {
        "name": "PSR-Bench question presence review (flagged)",
        "model": model_name,
        "reviewed_question_count": len(reviewed_questions),
        "manual_review_count": len(flagged_questions),
        "referability_issue_count": referability_issue_count,
        "attachment_pair_issue_count": attachment_pair_issue_count,
        "post_generation_issue_count": post_generation_issue_count,
        "questions": flagged_questions,
    }

    with open(review_json_path, "w", encoding="utf-8") as f:
        json.dump(review_payload, f, indent=2, ensure_ascii=False)
    with open(flagged_json_path, "w", encoding="utf-8") as f:
        json.dump(flagged_payload, f, indent=2, ensure_ascii=False)

    flagged_html = build_viewer_html(
        flagged_questions,
        scannet_roots,
        scannetpp_roots,
        scannetpp_sensor=scannetpp_sensor,
        title="question presence manual review",
        include_referability_audit=False,
        apply_filters=False,
    )
    flagged_html_path.write_text(flagged_html, encoding="utf-8")

    logger.info(
        "Question presence review complete for L1 occlusion and L2 attachment-pair questions: %d reviewed, %d flagged. JSON: %s HTML: %s",
        len(reviewed_questions),
        len(flagged_questions),
        flagged_json_path,
        flagged_html_path,
    )
    return {
        "model": model_name,
        "reviewed_question_count": len(reviewed_questions),
        "manual_review_count": len(flagged_questions),
        "referability_issue_count": referability_issue_count,
        "attachment_pair_issue_count": attachment_pair_issue_count,
        "post_generation_issue_count": post_generation_issue_count,
        "review_json_path": review_json_path,
        "flagged_json_path": flagged_json_path,
        "flagged_html_path": flagged_html_path,
        "questions": reviewed_questions,
    }

def _get_referability_scene_frames(cache: dict | None, scene_id: str) -> dict[str, dict]:
    if not cache:
        return {}
    frames = cache.get("frames", cache)
    scene_frames = frames.get(scene_id)
    if isinstance(scene_frames, dict):
        return scene_frames

    prefix = f"{scene_id}/"
    matched: dict[str, dict] = {}
    for key, value in frames.items():
        if isinstance(key, str) and key.startswith(prefix) and isinstance(value, dict):
            matched[key[len(prefix):]] = value
    return matched


def _get_referability_scene_ids(cache: dict | None) -> set[str]:
    if not cache:
        return set()
    frames = cache.get("frames", cache)
    scene_ids: set[str] = set()
    for key, value in frames.items():
        if isinstance(value, dict) and "frame_usable" not in value:
            scene_ids.add(str(key))
        elif isinstance(key, str) and "/" in key:
            scene_ids.add(key.split("/", 1)[0])
    return scene_ids


def _prioritize_manual_attachment_scene_dirs(
    scene_dirs: list[Path],
    referability_cache: dict[str, object],
) -> list[Path]:
    manual_scene_ids = {
        str(scene_id)
        for scene_id in (referability_cache.get("manual_attachment_scene_ids") or [])
    }
    return sorted(scene_dirs, key=lambda path: path.name not in manual_scene_ids)


def _normalize_label_list(value: object) -> list[str]:
    labels: list[str] = []
    seen: set[str] = set()
    if not isinstance(value, list):
        return labels
    for item in value:
        label = str(item or "").strip().lower()
        if not label or label in seen:
            continue
        seen.add(label)
        labels.append(label)
    return labels


def _has_l1_visibility_candidates(
    label_statuses: object,
    out_of_frame_not_visible_labels: object = None,
) -> bool:
    _ = label_statuses
    return bool(_normalize_label_list(out_of_frame_not_visible_labels))


def _frames_from_referability_cache(scene_frames: dict[str, dict]) -> list[dict[str, object]]:
    def _coerce_rank(value: object) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    frames: list[dict[str, object]] = []
    for image_name, entry in scene_frames.items():
        if not isinstance(entry, dict):
            continue
        if not entry.get("frame_usable", True):
            continue
        visible_object_ids: list[int] = []
        candidate_visible_object_ids = entry.get("candidate_visible_object_ids")
        if isinstance(candidate_visible_object_ids, list):
            for obj_id in candidate_visible_object_ids:
                try:
                    visible_object_ids.append(int(obj_id))
                except (TypeError, ValueError):
                    continue
        frames.append(
            {
                "image_name": image_name,
                "visible_object_ids": sorted(visible_object_ids),
                "final_selection_rank": _coerce_rank(
                    entry.get("final_selection_rank", 1_000_000)
                ),
                "attachment_referable_pair_count": _coerce_rank(
                    entry.get("attachment_referable_pair_count", 0) or 0
                ),
                "frame_selection_score": _coerce_rank(entry.get("frame_selection_score", 0) or 0),
                "selector_score": _coerce_rank(entry.get("selector_score", 0) or 0),
            }
        )
    # Prefer the attachment-aware selection order when present, then fall back
    # to the legacy rerank fields for older caches.
    frames.sort(
        key=lambda frame: (
            int(frame.get("final_selection_rank", 1_000_000)),
            -int(frame.get("attachment_referable_pair_count", 0) or 0),
            -int(frame.get("frame_selection_score", 0) or 0),
            -int(frame.get("selector_score", 0) or 0),
            str(frame.get("image_name", "")),
        )
    )
    return frames


def _support_chain_graph_has_two_hop_chain(
    support_chain_graph: dict[int, list[int]] | dict[str, list[int]],
) -> bool:
    graph = {
        int(parent_id): [int(child_id) for child_id in (child_ids or [])]
        for parent_id, child_ids in (support_chain_graph or {}).items()
    }
    return any(
        graph.get(int(child_id))
        for child_ids in graph.values()
        for child_id in child_ids
    )


def _attachment_graph_has_two_hop_chain(
    attachment_graph: dict[int, list[int]] | dict[str, list[int]],
) -> bool:
    graph = {
        int(parent_id): [int(child_id) for child_id in (child_ids or [])]
        for parent_id, child_ids in (attachment_graph or {}).items()
    }
    return any(
        graph.get(int(child_id))
        for child_ids in graph.values()
        for child_id in child_ids
    )


def _frame_has_l3_attachment_chain(
    frame: dict[str, object],
    referability_entry: dict[str, object] | None,
    support_chain_graph: dict[int, list[int]] | dict[str, list[int]],
) -> bool:
    visible_ids = set(_normalize_object_ids(frame.get("visible_object_ids")))
    if not visible_ids:
        visible_ids = set(_normalize_object_ids((referability_entry or {}).get("candidate_visible_object_ids")))

    raw_attachment_referable_ids = None
    if referability_entry is not None:
        raw_attachment_referable_ids = referability_entry.get("attachment_referable_object_ids")
        if raw_attachment_referable_ids is None:
            raw_attachment_referable_ids = _derive_final_referability_fields(
                referability_entry
            ).get("attachment_referable_object_ids", [])
    attachment_referable_ids = set(_normalize_object_ids(raw_attachment_referable_ids))
    if not visible_ids or not attachment_referable_ids:
        return False

    graph = {
        int(parent_id): [int(child_id) for child_id in (child_ids or [])]
        for parent_id, child_ids in (support_chain_graph or {}).items()
    }
    eligible_ids = visible_ids & attachment_referable_ids
    raw_pairs = (
        referability_entry.get("attachment_referable_pairs")
        if referability_entry is not None
        and "attachment_referable_pairs" in referability_entry
        else None
    )
    if raw_pairs is None:
        eligible_pairs = {
            (parent_id, child_id)
            for parent_id, child_ids in graph.items()
            for child_id in child_ids
            if parent_id in eligible_ids and child_id in eligible_ids
        }
    else:
        eligible_pairs = {
            (parent_id, child_id)
            for parent_id, child_id in _shared_normalize_attachment_pairs(raw_pairs)
            if parent_id in eligible_ids and child_id in eligible_ids
        }
    for grandparent_id, parent_ids in graph.items():
        if grandparent_id not in eligible_ids:
            continue
        for parent_id in parent_ids:
            parent_id = int(parent_id)
            if (
                parent_id not in eligible_ids
                or (grandparent_id, parent_id) not in eligible_pairs
            ):
                continue
            if any(
                int(grandchild_id) in eligible_ids
                and (parent_id, int(grandchild_id)) in eligible_pairs
                for grandchild_id in graph.get(parent_id, [])
            ):
                return True
    return False


def _normalize_object_ids(value: object) -> list[int]:
    object_ids: list[int] = []
    if not isinstance(value, list):
        return object_ids
    for item in value:
        try:
            object_ids.append(int(item))
        except (TypeError, ValueError):
            continue
    return sorted(set(object_ids))


def _frame_attachment_referable_pairs(
    *,
    referability_entry: dict[str, object] | None,
    attachment_graph: dict[int, list[int]] | dict[str, list[int]],
    attachment_referable_ids: list[int],
    visible_object_ids: list[int],
) -> list[tuple[int, int]]:
    graph_pairs = {
        (int(parent_id), int(child_id))
        for parent_id, child_ids in (attachment_graph or {}).items()
        for child_id in (child_ids or [])
    }
    attachment_ids = set(_normalize_object_ids(attachment_referable_ids))
    visible_ids = set(_normalize_object_ids(visible_object_ids))
    raw_pairs = (
        referability_entry.get("attachment_referable_pairs")
        if referability_entry is not None
        and "attachment_referable_pairs" in referability_entry
        else None
    )
    if raw_pairs is None:
        candidate_pairs = graph_pairs
    else:
        candidate_pairs = set(_shared_normalize_attachment_pairs(raw_pairs))
    return sorted(
        (parent_id, child_id)
        for parent_id, child_id in candidate_pairs
        if (parent_id, child_id) in graph_pairs
        and parent_id in attachment_ids
        and child_id in attachment_ids
        and (not visible_ids or (parent_id in visible_ids and child_id in visible_ids))
    )


def _build_visible_object_in_frame_ratio_map(
    *,
    visible_object_ids: list[int],
    referability_entry: dict[str, object] | None,
    scene_objects: list[dict],
    camera_pose: CameraPose | None,
    color_intrinsics: CameraIntrinsics | None,
) -> dict[int, float]:
    """Return per-visible-object projected bbox in-frame ratios."""
    visible_ids = _normalize_object_ids(visible_object_ids)
    if not visible_ids:
        return {}

    ratios_by_obj_id: dict[int, float] = {}

    def _ingest_review_container(container: object) -> None:
        if isinstance(container, dict):
            entries = container.items()
        elif isinstance(container, list):
            entries = [(None, item) for item in container]
        else:
            return

        for key, review in entries:
            if not isinstance(review, dict):
                continue
            try:
                obj_id = int(review.get("obj_id", key))
                ratio = float(review.get("bbox_in_frame_ratio", 0.0) or 0.0)
            except (TypeError, ValueError):
                continue
            if obj_id in visible_ids:
                ratios_by_obj_id[obj_id] = ratio

    _ingest_review_container((referability_entry or {}).get("object_reviews"))
    _ingest_review_container((referability_entry or {}).get("visibility_audit_by_object_id"))

    missing_ids = [
        int(obj_id)
        for obj_id in visible_ids
        if int(obj_id) not in ratios_by_obj_id
    ]
    if missing_ids and camera_pose is not None and color_intrinsics is not None:
        visible_set = set(missing_ids)
        fallback_visibility = compute_frame_object_visibility(
            objects=[
                obj for obj in scene_objects
                if int(obj.get("id", -1)) in visible_set
            ],
            pose=camera_pose,
            color_intrinsics=color_intrinsics,
            image_path=None,
            depth_image=None,
            depth_intrinsics=None,
        )
        for obj_id, meta in fallback_visibility.items():
            try:
                ratios_by_obj_id[int(obj_id)] = float(meta.get("bbox_in_frame_ratio", 0.0) or 0.0)
            except (TypeError, ValueError):
                continue

    return {
        int(obj_id): float(ratios_by_obj_id.get(int(obj_id), 0.0) or 0.0)
        for obj_id in visible_ids
    }


def _build_occlusion_eligible_object_ids(
    *,
    visible_object_ids: list[int],
    mention_in_frame_ratio_by_obj_id: dict[int, float] | None,
) -> list[int]:
    """Return visible object ids without any downstream bbox-ratio re-gating."""
    visible_ids = _normalize_object_ids(visible_object_ids)
    _ = mention_in_frame_ratio_by_obj_id
    return visible_ids


def _build_visible_object_projected_area_map(
    *,
    visible_object_ids: list[int],
    referability_entry: dict[str, object] | None,
    scene_objects: list[dict],
    camera_pose: CameraPose | None,
    color_intrinsics: CameraIntrinsics | None,
) -> dict[int, float]:
    """Return per-visible-object projected bbox areas in pixels."""
    visible_ids = _normalize_object_ids(visible_object_ids)
    if not visible_ids:
        return {}

    projected_area_by_obj_id: dict[int, float] = {}

    def _ingest_review_container(container: object) -> None:
        if isinstance(container, dict):
            entries = container.items()
        elif isinstance(container, list):
            entries = [(None, item) for item in container]
        else:
            return

        for key, review in entries:
            if not isinstance(review, dict):
                continue
            try:
                obj_id = int(review.get("obj_id", key))
                projected_area_px = float(review.get("projected_area_px", 0.0) or 0.0)
            except (TypeError, ValueError):
                continue
            if obj_id in visible_ids:
                projected_area_by_obj_id[obj_id] = projected_area_px

    _ingest_review_container((referability_entry or {}).get("object_reviews"))
    _ingest_review_container((referability_entry or {}).get("visibility_audit_by_object_id"))

    missing_ids = [
        int(obj_id)
        for obj_id in visible_ids
        if int(obj_id) not in projected_area_by_obj_id
    ]
    if missing_ids and camera_pose is not None and color_intrinsics is not None:
        visible_set = set(missing_ids)
        fallback_visibility = compute_frame_object_visibility(
            objects=[
                obj for obj in scene_objects
                if int(obj.get("id", -1)) in visible_set
            ],
            pose=camera_pose,
            color_intrinsics=color_intrinsics,
            image_path=None,
            depth_image=None,
            depth_intrinsics=None,
        )
        for obj_id, meta in fallback_visibility.items():
            try:
                projected_area_by_obj_id[int(obj_id)] = float(meta.get("projected_area_px", 0.0) or 0.0)
            except (TypeError, ValueError):
                continue

    return {
        int(obj_id): float(projected_area_by_obj_id.get(int(obj_id), 0.0) or 0.0)
        for obj_id in visible_ids
    }


def _referable_occlusion_veto_dense_sample_budget(projected_area_px: float) -> int:
    area_px = float(projected_area_px or 0.0)
    if not np.isfinite(area_px) or area_px <= 0.0:
        area_px = REFERABLE_OCCLUSION_VETO_DENSE_BASE_PROJECTED_AREA_PX
    scale = max(area_px / REFERABLE_OCCLUSION_VETO_DENSE_BASE_PROJECTED_AREA_PX, 1.0)
    budget = int(round(REFERABLE_OCCLUSION_VETO_DENSE_BASE_SAMPLE_COUNT * scale))
    return max(
        REFERABLE_OCCLUSION_VETO_DENSE_BASE_SAMPLE_COUNT,
        min(budget, REFERABLE_OCCLUSION_VETO_DENSE_MAX_SAMPLE_COUNT),
    )


def _resample_instance_surface_probe_points(
    *,
    instance_mesh_data: object | None,
    obj_id: int,
    sample_budget: int,
    frame_seed_key: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if instance_mesh_data is None or sample_budget <= 0:
        return (
            np.empty((0, 3), dtype=np.float64),
            np.empty((0,), dtype=np.int64),
            np.empty((0, 3), dtype=np.float64),
        )

    target_tri_ids = _instance_triangle_id_set(instance_mesh_data, int(obj_id))
    vertices = np.asarray(
        getattr(instance_mesh_data, "vertices", np.empty((0, 3), dtype=np.float64)),
        dtype=np.float64,
    )
    faces = np.asarray(
        getattr(instance_mesh_data, "faces", np.empty((0, 3), dtype=np.int64)),
        dtype=np.int64,
    )
    if vertices.ndim != 2 or faces.ndim != 2 or len(vertices) == 0 or len(faces) == 0 or not target_tri_ids:
        return (
            np.empty((0, 3), dtype=np.float64),
            np.empty((0,), dtype=np.int64),
            np.empty((0, 3), dtype=np.float64),
        )

    tri_ids = np.array(sorted(int(tri_id) for tri_id in target_tri_ids), dtype=np.int64)
    seed = zlib.crc32(
        f"{frame_seed_key}:{int(obj_id)}:{int(sample_budget)}".encode("utf-8")
    ) & 0xFFFFFFFF
    rng = np.random.RandomState(seed)
    return _sample_surface_points_from_triangles(
        vertices=vertices,
        faces=faces,
        triangle_ids=tri_ids,
        n_samples=int(sample_budget),
        rng=rng,
        return_metadata=True,
    )


def _mesh_visibility_counts_with_early_stop(
    *,
    ray_caster: object,
    camera_pos: np.ndarray,
    target_points: np.ndarray,
    target_tri_ids: set[int],
    sample_triangle_ids: np.ndarray,
    sample_barycentrics: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
    min_visible_ratio: float,
) -> dict[str, object]:
    total_points = int(len(target_points))
    if total_points <= 0 or not target_tri_ids:
        return {
            "visible_count": 0,
            "valid_count": 0,
            "processed_count": 0,
            "stopped_early": False,
            "stop_reason": "no_points",
        }

    visible_count = 0
    valid_count = 0
    processed_count = 0
    stopped_early = False
    stop_reason = "completed"

    for start_idx in range(0, total_points, REFERABLE_OCCLUSION_VETO_DENSE_CHUNK_SIZE):
        end_idx = min(start_idx + REFERABLE_OCCLUSION_VETO_DENSE_CHUNK_SIZE, total_points)
        chunk_visible, chunk_valid = _mesh_visibility_stats_compat(
            ray_caster,
            camera_pos=np.asarray(camera_pos, dtype=np.float64),
            target_points=np.asarray(target_points[start_idx:end_idx], dtype=np.float64),
            target_tri_ids=set(int(tri_id) for tri_id in target_tri_ids),
            sample_triangle_ids=np.asarray(sample_triangle_ids[start_idx:end_idx], dtype=np.int64),
            sample_barycentrics=np.asarray(sample_barycentrics[start_idx:end_idx], dtype=np.float64),
            vertices=np.asarray(vertices, dtype=np.float64),
            faces=np.asarray(faces, dtype=np.int64),
        )
        visible_count += int(chunk_visible)
        valid_count += int(chunk_valid)
        processed_count = int(end_idx)

        remaining_count = total_points - processed_count
        max_final_valid_count = valid_count + remaining_count
        best_case_visible_ratio = (
            float((visible_count + remaining_count) / max_final_valid_count)
            if max_final_valid_count > 0 else 0.0
        )
        worst_case_visible_ratio = (
            float(visible_count / max_final_valid_count)
            if max_final_valid_count > 0 else 0.0
        )
        if valid_count > 0 and worst_case_visible_ratio >= float(min_visible_ratio):
            stopped_early = True
            stop_reason = "ratio_guaranteed_pass"
            break
        if best_case_visible_ratio < float(min_visible_ratio):
            stopped_early = True
            stop_reason = "cannot_reach_ratio_threshold"
            break

    return {
        "visible_count": int(visible_count),
        "valid_count": int(valid_count),
        "processed_count": int(processed_count),
        "stopped_early": bool(stopped_early),
        "stop_reason": str(stop_reason),
    }


def _evaluate_referable_occlusion_veto_for_object(
    *,
    obj: dict[str, object] | None,
    obj_id: int,
    scene_id: str,
    image_name: str,
    projected_area_px: float,
    camera_pose: CameraPose | None,
    color_intrinsics: CameraIntrinsics | None,
    ray_caster: object | None,
    instance_mesh_data: object | None,
) -> dict[str, object]:
    audit: dict[str, object] = {
        "obj_id": int(obj_id),
        "projected_area_px": float(projected_area_px or 0.0),
        "status": "skipped",
        "keep_for_generation": True,
        "probe_visible_count": 0,
        "probe_valid_count": 0,
        "probe_visible_enough_threshold": 0,
        "dense_sample_budget": 0,
        "dense_in_frame_sample_count": 0,
        "dense_visible_count": 0,
        "dense_valid_count": 0,
        "dense_processed_count": 0,
        "dense_visible_ratio": None,
        "dense_visible_ratio_threshold": REFERABLE_OCCLUSION_VETO_MIN_VISIBLE_RATIO,
        "dense_visible_ratio_denominator": "valid_count",
        "dense_stop_reason": "not_run",
        "reason": "missing_occlusion_resources",
    }
    if obj is None or camera_pose is None or color_intrinsics is None:
        return audit
    if ray_caster is None or instance_mesh_data is None:
        return audit

    bbox_min = np.asarray(obj.get("bbox_min", []), dtype=np.float64)
    bbox_max = np.asarray(obj.get("bbox_max", []), dtype=np.float64)
    if bbox_min.shape != (3,) or bbox_max.shape != (3,):
        audit["reason"] = "invalid_bbox"
        return audit

    camera_pos = np.asarray(camera_pose.position, dtype=np.float64)
    sample_budget = _referable_occlusion_veto_dense_sample_budget(projected_area_px)
    audit["dense_sample_budget"] = int(sample_budget)
    sample_points, sample_triangle_ids, sample_barycentrics = _resample_instance_surface_probe_points(
        instance_mesh_data=instance_mesh_data,
        obj_id=int(obj_id),
        sample_budget=int(sample_budget),
        frame_seed_key=f"{scene_id}:{image_name}",
    )
    if len(sample_points) <= 0:
        audit["reason"] = "missing_surface_samples"
        return audit

    (
        _projected_area_unused,
        _in_frame_ratio_unused,
        in_frame_points,
        in_frame_triangle_ids,
        in_frame_barycentrics,
    ) = _in_frame_surface_sample_subset(
        sample_points,
        camera_pose,
        color_intrinsics,
        sample_triangle_ids=sample_triangle_ids,
        sample_barycentrics=sample_barycentrics,
    )
    dense_in_frame_count = int(len(in_frame_points))
    audit["dense_in_frame_sample_count"] = dense_in_frame_count
    if dense_in_frame_count <= 0:
        audit["reason"] = "no_in_frame_dense_samples"
        return audit

    target_tri_ids = _instance_triangle_id_set(instance_mesh_data, int(obj_id))
    if not target_tri_ids:
        audit["reason"] = "missing_target_triangles"
        return audit

    try:
        dense_counts = _mesh_visibility_counts_with_early_stop(
            ray_caster=ray_caster,
            camera_pos=camera_pos,
            target_points=np.asarray(in_frame_points, dtype=np.float64),
            target_tri_ids=target_tri_ids,
            sample_triangle_ids=np.asarray(in_frame_triangle_ids, dtype=np.int64),
            sample_barycentrics=np.asarray(in_frame_barycentrics, dtype=np.float64),
            vertices=np.asarray(
                getattr(instance_mesh_data, "vertices", np.empty((0, 3), dtype=np.float64)),
                dtype=np.float64,
            ),
            faces=np.asarray(
                getattr(instance_mesh_data, "faces", np.empty((0, 3), dtype=np.int64)),
                dtype=np.int64,
            ),
            min_visible_ratio=REFERABLE_OCCLUSION_VETO_MIN_VISIBLE_RATIO,
        )
    except Exception:
        audit["reason"] = "dense_visibility_error"
        return audit
    visible_count = int(dense_counts["visible_count"])
    valid_count = int(dense_counts["valid_count"])
    processed_count = int(dense_counts["processed_count"])
    visible_ratio = (
        float(visible_count / valid_count)
        if valid_count > 0 else 0.0
    )
    audit["dense_visible_count"] = visible_count
    audit["dense_valid_count"] = valid_count
    audit["dense_processed_count"] = processed_count
    audit["dense_visible_ratio"] = visible_ratio
    audit["dense_stop_reason"] = str(dense_counts["stop_reason"])

    if visible_count <= 0 or valid_count <= 0:
        audit["status"] = "not_visible"
        audit["keep_for_generation"] = False
        audit["reason"] = "dense_visible_ratio_zero"
    elif visible_ratio < REFERABLE_OCCLUSION_VETO_MIN_VISIBLE_RATIO:
        audit["status"] = "low_visible"
        audit["keep_for_generation"] = False
        audit["reason"] = "dense_visible_ratio_below_threshold"
    else:
        audit["status"] = "visible_enough"
        audit["keep_for_generation"] = True
        audit["reason"] = "dense_visible_ratio_above_threshold"
    return audit


def _filter_referable_object_ids_with_occlusion_veto(
    *,
    scene_id: str,
    image_name: str,
    referable_object_ids: list[int] | None,
    objects_by_id: dict[int, dict],
    projected_area_by_obj_id: dict[int, float] | None,
    camera_pose: CameraPose | None,
    color_intrinsics: CameraIntrinsics | None,
    ray_caster: object | None,
    instance_mesh_data: object | None,
) -> dict[str, object]:
    raw_ids = _normalize_object_ids(referable_object_ids)
    filtered_ids: list[int] = []
    low_visible_ids: list[int] = []
    not_visible_ids: list[int] = []
    skipped_ids: list[int] = []
    audit_by_obj_id: dict[str, dict[str, object]] = {}

    for obj_id in raw_ids:
        audit = _evaluate_referable_occlusion_veto_for_object(
            obj=objects_by_id.get(int(obj_id)),
            obj_id=int(obj_id),
            scene_id=scene_id,
            image_name=image_name,
            projected_area_px=float((projected_area_by_obj_id or {}).get(int(obj_id), 0.0) or 0.0),
            camera_pose=camera_pose,
            color_intrinsics=color_intrinsics,
            ray_caster=ray_caster,
            instance_mesh_data=instance_mesh_data,
        )
        audit_by_obj_id[str(int(obj_id))] = audit
        status = str(audit.get("status", "skipped"))
        if status == "low_visible":
            low_visible_ids.append(int(obj_id))
            continue
        if status == "not_visible":
            not_visible_ids.append(int(obj_id))
            continue
        filtered_ids.append(int(obj_id))
        if status == "skipped":
            skipped_ids.append(int(obj_id))

    return {
        "raw_object_ids": raw_ids,
        "filtered_object_ids": filtered_ids,
        "low_visible_object_ids": low_visible_ids,
        "not_visible_object_ids": not_visible_ids,
        "skipped_object_ids": skipped_ids,
        "audit_by_object_id": audit_by_obj_id,
    }


def _normalize_label_counts(value: object) -> dict[str, int]:
    counts: dict[str, int] = {}
    if not isinstance(value, dict):
        return counts
    for key, count in value.items():
        if not isinstance(key, str):
            continue
        try:
            counts[key] = int(count)
        except (TypeError, ValueError):
            continue
    return dict(sorted(counts.items()))


def _normalize_label_statuses(value: object) -> dict[str, str]:
    statuses: dict[str, str] = {}
    if not isinstance(value, dict):
        return statuses
    for key, status in value.items():
        if not isinstance(key, str):
            continue
        label = key.strip().lower()
        if not label:
            continue
        text = str(status or "").strip().lower()
        if text not in {"absent", "unique", "multiple", "unsure"}:
            continue
        statuses[label] = text
    return dict(sorted(statuses.items()))


def _count_labels_for_object_ids(
    object_ids: list[int],
    objects_by_id: dict[int, dict],
) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for obj_id in object_ids:
        obj = objects_by_id.get(int(obj_id))
        if obj is None:
            continue
        label = str(obj.get("label", "")).strip()
        if not label:
            continue
        counter[label] += 1
    return dict(sorted((str(label), int(count)) for label, count in counter.items()))


def _build_scene_attachment_rows(scene: dict) -> list[dict[str, object]]:
    obj_map = {int(obj["id"]): obj for obj in scene.get("objects", [])}
    rows: list[dict[str, object]] = []
    edges = scene.get("attachment_edges")
    if isinstance(edges, list) and edges:
        for edge in edges:
            try:
                parent_id = int(edge["parent_id"])
                child_id = int(edge["child_id"])
            except (KeyError, TypeError, ValueError):
                continue
            rows.append({
                "parent_id": parent_id,
                "parent_label": str(obj_map.get(parent_id, {}).get("label", "object")),
                "child_id": child_id,
                "child_label": str(obj_map.get(child_id, {}).get("label", "object")),
                "relation_type": str(edge.get("type") or edge.get("relation_type") or "attachment"),
                "confidence": edge.get("confidence", edge.get("score")),
            })
        rows.sort(key=lambda row: (row["parent_label"], row["child_label"], row["parent_id"], row["child_id"]))
        return rows

    graph = scene.get("attachment_graph") or scene.get("support_graph") or {}
    if not isinstance(graph, dict):
        return rows
    for parent_id, child_ids in graph.items():
        try:
            parent_int = int(parent_id)
        except (TypeError, ValueError):
            continue
        if not isinstance(child_ids, list):
            continue
        for child_id in child_ids:
            try:
                child_int = int(child_id)
            except (TypeError, ValueError):
                continue
            rows.append({
                "parent_id": parent_int,
                "parent_label": str(obj_map.get(parent_int, {}).get("label", "object")),
                "child_id": child_int,
                "child_label": str(obj_map.get(child_int, {}).get("label", "object")),
                "relation_type": "attachment",
                "confidence": None,
            })
    rows.sort(key=lambda row: (row["parent_label"], row["child_label"], row["parent_id"], row["child_id"]))
    return rows


def _filter_frame_attachment_rows(
    scene_attachment_rows: list[dict[str, object]],
    relevant_object_ids: set[int],
) -> list[dict[str, object]]:
    return [
        row for row in scene_attachment_rows
        if int(row["parent_id"]) in relevant_object_ids and int(row["child_id"]) in relevant_object_ids
    ]


def _attachment_summary_for_object(
    obj_id: int,
    frame_attachment_rows: list[dict[str, object]],
) -> str:
    attached_to = [
        f'{row["parent_label"]} #{row["parent_id"]}'
        for row in frame_attachment_rows
        if int(row["child_id"]) == obj_id
    ]
    carries = [
        f'{row["child_label"]} #{row["child_id"]}'
        for row in frame_attachment_rows
        if int(row["parent_id"]) == obj_id
    ]
    parts: list[str] = []
    if attached_to:
        parts.append("附着于 " + ", ".join(attached_to))
    if carries:
        parts.append("承载 " + ", ".join(carries))
    return "；".join(parts) if parts else "-"


def _build_object_debug_rows(
    scene_objects: list[dict],
    selector_visible_ids: list[int],
    pipeline_visible_ids: list[int],
    referability_entry: dict | None,
    frame_attachment_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    selector_set = set(int(obj_id) for obj_id in selector_visible_ids)
    pipeline_set = set(int(obj_id) for obj_id in pipeline_visible_ids)
    candidate_set = set(_normalize_object_ids((referability_entry or {}).get("candidate_visible_object_ids")))
    referable_set = set(_normalize_object_ids((referability_entry or {}).get("referable_object_ids")))
    attachment_set = {
        int(row["parent_id"])
        for row in frame_attachment_rows
    } | {
        int(row["child_id"])
        for row in frame_attachment_rows
    }
    relevant_ids = selector_set | pipeline_set | candidate_set | referable_set | attachment_set

    rows: list[dict[str, object]] = []
    for obj in scene_objects:
        obj_id = int(obj["id"])
        if relevant_ids and obj_id not in relevant_ids:
            continue
        tags: list[str] = []
        if obj_id in candidate_set:
            tags.append("VLM候选")
        if obj_id in referable_set:
            tags.append("VLM唯一")
        if obj_id in pipeline_set:
            tags.append("Pipeline可用")
        if obj_id in attachment_set:
            tags.append("被attachment约束")
        rows.append({
            "id": obj_id,
            "label": str(obj.get("label", "")),
            "tags": tags,
            "attachment_summary": _attachment_summary_for_object(obj_id, frame_attachment_rows),
        })

    rows.sort(key=lambda row: (
        "VLM唯一" not in row["tags"],
        "Pipeline可用" not in row["tags"],
        str(row["label"]),
        int(row["id"]),
    ))
    return rows


def _build_frame_debug_entry(
    image_name: str,
    scene_objects: list[dict],
    objects_by_id: dict[int, dict],
    selector_visible_ids: list[int],
    pipeline_visible_ids: list[int],
    occlusion_eligible_object_ids: list[int] | None,
    pipeline_referable_object_ids: list[int] | None = None,
    pipeline_attachment_referable_object_ids: list[int] | None = None,
    pipeline_attachment_referable_pairs: list[tuple[int, int]] | None = None,
    referability_entry: dict | None = None,
    frame_attachment_rows: list[dict[str, object]] | None = None,
    referable_occlusion_veto: dict[str, object] | None = None,
    generated_questions: list[dict] | None = None,
    pipeline_skip_reason: str | None = None,
) -> dict[str, object]:
    generated_questions = [] if generated_questions is None else [dict(q) for q in generated_questions]
    frame_attachment_rows = [] if frame_attachment_rows is None else list(frame_attachment_rows)
    label_to_object_ids = (referability_entry or {}).get("label_to_object_ids") or {}
    return {
        "image_name": image_name,
        "frame_usable": bool((referability_entry or {}).get("frame_usable", True)),
        "frame_reject_reason": (referability_entry or {}).get("frame_reject_reason"),
        "pipeline_skip_reason": pipeline_skip_reason,
        "selector_visible_object_ids": _normalize_object_ids(selector_visible_ids),
        "selector_visible_label_counts": _count_labels_for_object_ids(selector_visible_ids, objects_by_id),
        "pipeline_visible_object_ids_used_for_generation": _normalize_object_ids(pipeline_visible_ids),
        "pipeline_visible_label_counts": _count_labels_for_object_ids(pipeline_visible_ids, objects_by_id),
        "occlusion_eligible_object_ids": _normalize_object_ids(occlusion_eligible_object_ids),
        "pipeline_referable_object_ids_used_for_generation": _normalize_object_ids(pipeline_referable_object_ids),
        "pipeline_attachment_referable_object_ids_used_for_generation": _normalize_object_ids(
            pipeline_attachment_referable_object_ids
        ),
        "pipeline_attachment_referable_pairs_used_for_generation": [
            [parent_id, child_id]
            for parent_id, child_id in sorted(
                _shared_normalize_attachment_pairs(
                    pipeline_attachment_referable_pairs
                )
            )
        ],
        "candidate_visibility_source": (referability_entry or {}).get("candidate_visibility_source"),
        "candidate_visible_label_counts": _normalize_label_counts(
            (referability_entry or {}).get("candidate_visible_label_counts")
        ),
        "crop_label_statuses": _normalize_label_statuses((referability_entry or {}).get("crop_label_statuses")),
        "crop_label_counts": _normalize_label_counts((referability_entry or {}).get("crop_label_counts")),
        "crop_referable_object_ids": _normalize_object_ids((referability_entry or {}).get("crop_referable_object_ids")),
        "full_frame_label_reviews": list((referability_entry or {}).get("full_frame_label_reviews", [])),
        "full_frame_label_statuses": _normalize_label_statuses((referability_entry or {}).get("full_frame_label_statuses")),
        "full_frame_label_counts": _normalize_label_counts((referability_entry or {}).get("full_frame_label_counts")),
        "vlm_label_statuses": _normalize_label_statuses((referability_entry or {}).get("label_statuses")),
        "vlm_label_counts": _normalize_label_counts((referability_entry or {}).get("label_counts")),
        "out_of_frame_label_reviews": list((referability_entry or {}).get("out_of_frame_label_reviews", [])),
        "out_of_frame_not_visible_labels": _normalize_label_list(
            (referability_entry or {}).get("out_of_frame_not_visible_labels")
        ),
        "out_of_frame_label_to_object_ids": {
            str(label): _normalize_object_ids(obj_ids)
            for label, obj_ids in (
                _shared_normalize_label_to_object_ids(
                    (referability_entry or {}).get("out_of_frame_label_to_object_ids")
                )
            ).items()
        },
        "out_of_frame_vlm_early_stop": bool(
            (referability_entry or {}).get("out_of_frame_vlm_early_stop", False)
        ),
        "referable_object_ids": _normalize_object_ids((referability_entry or {}).get("referable_object_ids")),
        "attachment_referable_object_ids": _normalize_object_ids(
            (referability_entry or {}).get("attachment_referable_object_ids")
        ),
        "attachment_referable_pairs": [
            [int(pair[0]), int(pair[1])]
            for pair in ((referability_entry or {}).get("attachment_referable_pairs") or [])
            if isinstance(pair, (list, tuple)) and len(pair) >= 2
        ],
        "referable_occlusion_veto": dict(referable_occlusion_veto or {}),
        "candidate_labels": list((referability_entry or {}).get("candidate_labels", [])),
        "label_to_object_ids": {
            str(label): _normalize_object_ids(obj_ids)
            for label, obj_ids in label_to_object_ids.items()
        },
        "vlm_label_reviews": list(
            (referability_entry or {}).get("vlm_label_reviews")
            or (referability_entry or {}).get("full_frame_label_reviews", [])
        ),
        "object_reviews": dict((referability_entry or {}).get("object_reviews", {})),
        "object_rows": _build_object_debug_rows(
            scene_objects,
            selector_visible_ids,
            pipeline_visible_ids,
            referability_entry,
            frame_attachment_rows,
        ),
        "attachment_rows": frame_attachment_rows,
        "generated_questions": generated_questions,
    }


def _write_json_file(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _write_json_file_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with open(temporary_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary_path, path)
    finally:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass


def _cross_frame_scene_cache_dir(output_dir: Path, scene_id: str) -> Path:
    return output_dir / CROSS_FRAME_SCENE_CACHE_DIRNAME / str(scene_id)


def _counter_checkpoint_payload(counter: Counter) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for key, count in counter.items():
        rows.append({
            "key": list(key) if isinstance(key, tuple) else key,
            "tuple_key": isinstance(key, tuple),
            "count": int(count),
        })
    return rows


def _counter_from_checkpoint(payload: object) -> Counter:
    if not isinstance(payload, list):
        raise ValueError("checkpoint counter payload must be a list")
    result: Counter = Counter()
    for row in payload:
        if not isinstance(row, dict):
            raise ValueError("checkpoint counter rows must be objects")
        if "key" not in row or "count" not in row:
            raise ValueError("checkpoint counter rows require key and count")
        key = row.get("key")
        if bool(row.get("tuple_key")):
            if not isinstance(key, list):
                raise ValueError("checkpoint tuple counter keys must be lists")
            key = tuple(key)
        try:
            hash(key)
            count = int(row["count"])
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid checkpoint counter row") from exc
        result[key] = count
    return result


def _nested_tuple(value: object) -> object:
    if isinstance(value, list):
        return tuple(_nested_tuple(item) for item in value)
    return value


def _rng_checkpoint_payload() -> dict[str, object]:
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": {
            "bit_generator": numpy_state[0],
            "keys": numpy_state[1].tolist(),
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
    }


def _restore_rng_checkpoint(payload: object) -> None:
    if not isinstance(payload, dict):
        raise ValueError("checkpoint RNG payload must be an object")
    random.setstate(_nested_tuple(payload["python"]))
    numpy_payload = payload["numpy"]
    if not isinstance(numpy_payload, dict):
        raise ValueError("checkpoint NumPy RNG payload must be an object")
    np.random.set_state((
        str(numpy_payload["bit_generator"]),
        np.asarray(numpy_payload["keys"], dtype=np.uint32),
        int(numpy_payload["position"]),
        int(numpy_payload["has_gauss"]),
        float(numpy_payload["cached_gaussian"]),
    ))


def _set_pipeline_random_seed(seed: int = PIPELINE_RANDOM_SEED) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))


def _pipeline_scene_status_path(output_dir: Path) -> Path:
    return output_dir / "scene_status.json"


def _raw_scene_questions_cache_dir(output_dir: Path) -> Path:
    return output_dir / RAW_QUESTIONS_SCENE_CACHE_DIRNAME


def _validate_strict_object_centric_questions(
    questions: list[dict],
    *,
    source: str,
) -> None:
    direction_coefficients = {
        "forward": (1.0, 0.0),
        "forward-right": (1.0, 1.0),
        "right": (0.0, 1.0),
        "backward-right": (-1.0, 1.0),
        "backward": (-1.0, 0.0),
        "backward-left": (-1.0, -1.0),
        "left": (0.0, -1.0),
        "forward-left": (1.0, -1.0),
    }

    def fail(index: int, detail: str) -> None:
        raise RuntimeError(
            f"Invalid strict object_move_object_centric question at {source} "
            f"index {index}: {detail}"
        )

    def axes(index: int, question: dict, prefix: str) -> tuple[np.ndarray, np.ndarray]:
        try:
            forward = np.asarray(question[f"{prefix}_forward_world"], dtype=np.float64)
            right = np.asarray(question[f"{prefix}_right_world"], dtype=np.float64)
        except (KeyError, TypeError, ValueError) as exc:
            fail(index, f"missing or invalid {prefix} axes: {exc}")
        if (
            forward.shape != (3,)
            or right.shape != (3,)
            or not np.all(np.isfinite(forward))
            or not np.all(np.isfinite(right))
            or not np.isclose(forward[2], 0.0, atol=1e-8)
            or not np.isclose(right[2], 0.0, atol=1e-8)
            or not np.isclose(np.linalg.norm(forward), 1.0, atol=1e-6)
            or not np.isclose(np.linalg.norm(right), 1.0, atol=1e-6)
            or not np.allclose(right, [forward[1], -forward[0], 0.0], atol=1e-6)
        ):
            fail(index, f"{prefix} axes are not a valid frozen horizontal frame")
        return forward, right

    for index, question in enumerate(questions):
        if question.get("type") != "object_move_object_centric":
            continue
        if question.get("movement_semantics_version") != L2_OBJECT_MOVE_SEMANTICS_VERSION:
            fail(index, "movement_semantics_version is not current")
        legacy_fields = {
            "movement_frame_query_obj_id",
            "movement_frame_reference_obj_id",
        } & question.keys()
        if legacy_fields:
            fail(index, f"legacy query-to-reference frame fields remain: {sorted(legacy_fields)}")

        required = {
            "movement_reference_frame": "moved_object_facing_first_camera",
            "movement_camera_binding": "frame_1",
            "movement_frame_frozen": True,
            "answer_reference_frame": "query_object_facing_first_camera",
            "answer_camera_binding": "frame_1",
            "answer_frame_frozen": True,
        }
        for field, expected in required.items():
            if question.get(field) != expected:
                fail(index, f"{field}={question.get(field)!r}, expected {expected!r}")
        try:
            if int(question["movement_frame_anchor_obj_id"]) != int(question["moved_obj_id"]):
                fail(index, "movement frame is not anchored at the moved object")
            if int(question["answer_frame_anchor_obj_id"]) != int(question["query_obj_id"]):
                fail(index, "answer frame is not anchored at the query object")
        except (KeyError, TypeError, ValueError) as exc:
            fail(index, f"missing or invalid frame anchor: {exc}")

        movement_forward, movement_right = axes(index, question, "movement_frame")
        axes(index, question, "answer_frame")
        direction = str(question.get("movement_direction") or "")
        coefficients = direction_coefficients.get(direction)
        if coefficients is None:
            fail(index, f"unsupported movement_direction={direction!r}")
        try:
            distance = float(question["movement_distance_m"])
            delta = np.asarray(question["delta"], dtype=np.float64)
        except (KeyError, TypeError, ValueError) as exc:
            fail(index, f"missing or invalid movement delta: {exc}")
        if not np.isfinite(distance) or distance <= 0.0:
            fail(index, "movement_distance_m must be finite and positive")
        if delta.shape != (3,) or not np.all(np.isfinite(delta)):
            fail(index, "delta must be a finite 3-vector")
        expected_delta = (
            coefficients[0] * movement_forward + coefficients[1] * movement_right
        )
        expected_delta /= np.linalg.norm(expected_delta)
        if not np.allclose(delta, expected_delta * distance, atol=2e-6):
            fail(index, "delta does not match the frozen moved-object camera-facing frame")


def _build_empty_pipeline_scene_status_doc() -> dict[str, object]:
    return {
        "version": PIPELINE_SCENE_STATUS_VERSION,
        "object_move_semantics_version": L2_OBJECT_MOVE_SEMANTICS_VERSION,
        "object_move_object_centric_semantics": (
            OBJECT_MOVE_OBJECT_CENTRIC_SEMANTICS_PROFILE
        ),
        "completed_scenes": {},
    }


def _load_pipeline_scene_status_doc(path: Path) -> dict[str, object]:
    if not path.exists():
        return _build_empty_pipeline_scene_status_doc()

    with open(path, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    if not isinstance(loaded, dict):
        raise RuntimeError(f"Invalid scene status document at {path}: expected JSON object")

    version = int(loaded.get("version", 0) or 0)
    if version != PIPELINE_SCENE_STATUS_VERSION:
        raise RuntimeError(
            f"Unsupported scene status version {version or '<missing>'} at {path}; "
            f"expected {PIPELINE_SCENE_STATUS_VERSION}."
        )

    try:
        movement_semantics_version = int(
            loaded.get("object_move_semantics_version", 0) or 0
        )
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"Invalid object-move semantics version at {path}"
        ) from exc
    if movement_semantics_version != L2_OBJECT_MOVE_SEMANTICS_VERSION:
        raise RuntimeError(
            "Unsupported object-move semantics version "
            f"{movement_semantics_version or '<missing>'} at {path}; expected "
            f"{L2_OBJECT_MOVE_SEMANTICS_VERSION}."
        )
    object_centric_semantics = str(
        loaded.get("object_move_object_centric_semantics") or ""
    )
    if object_centric_semantics != OBJECT_MOVE_OBJECT_CENTRIC_SEMANTICS_PROFILE:
        raise RuntimeError(
            "Unsupported object_move_object_centric semantics profile "
            f"{object_centric_semantics or '<missing>'!r} at {path}; expected "
            f"{OBJECT_MOVE_OBJECT_CENTRIC_SEMANTICS_PROFILE!r}. Use a new "
            "--output_dir or rerun without --resume so stale raw-question caches "
            "cannot be reused."
        )

    completed_scenes = loaded.get("completed_scenes")
    if not isinstance(completed_scenes, dict):
        raise RuntimeError(f"Invalid scene status document at {path}: completed_scenes must be an object")

    result: dict[str, object] = {
        "version": PIPELINE_SCENE_STATUS_VERSION,
        "object_move_semantics_version": L2_OBJECT_MOVE_SEMANTICS_VERSION,
        "object_move_object_centric_semantics": (
            OBJECT_MOVE_OBJECT_CENTRIC_SEMANTICS_PROFILE
        ),
        "completed_scenes": dict(completed_scenes),
    }
    route_method = loaded.get("auxiliary_route_method")
    if isinstance(route_method, str) and route_method in AUXILIARY_ROUTE_METHODS:
        result["auxiliary_route_method"] = route_method
    depth_hard_limits = loaded.get(
        "depth_route_camera_motion_hard_limits_enabled"
    )
    if depth_hard_limits is not None:
        if not isinstance(depth_hard_limits, bool):
            raise RuntimeError(
                f"Invalid scene status document at {path}: "
                "depth_route_camera_motion_hard_limits_enabled must be boolean"
            )
        result["depth_route_camera_motion_hard_limits_enabled"] = depth_hard_limits
    for field_name in (
        "l1_candidate_budget",
        "l2_candidate_budget",
        "l3_candidate_budget",
        "l2_l3_candidate_budget",
        "occlusion_max_references_per_query",
        "occlusion_max_combinations_per_scene",
    ):
        if field_name not in loaded:
            continue
        try:
            recorded_budget = int(loaded[field_name])
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"Invalid scene status document at {path}: {field_name} must be an integer"
            ) from exc
        if recorded_budget < 0:
            raise RuntimeError(
                f"Invalid scene status document at {path}: {field_name} must be >= 0"
            )
        result[field_name] = recorded_budget
    return result


def _scene_status_updated_at_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_pipeline_scene_status_updated_at(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    candidate = value.strip()
    if not candidate:
        return None
    if candidate.endswith("Z"):
        candidate = f"{candidate[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(timezone.utc)


def _coerce_pipeline_scene_completion_index(value: object) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _pipeline_completed_scene_records(
    scene_status_doc: dict[str, object],
) -> dict[str, object]:
    completed_scenes = scene_status_doc.setdefault("completed_scenes", {})
    if not isinstance(completed_scenes, dict):
        raise RuntimeError("scene_status_doc.completed_scenes must be an object")
    return completed_scenes


def _build_pipeline_scene_status_record(
    scene_id: str,
    *,
    completion_index: int,
    raw_question_count: int,
    frame_count: int,
    pipeline_outcome: str,
    updated_at: str,
) -> dict[str, object]:
    return {
        "scene_id": str(scene_id),
        "status": "completed",
        "completion_index": max(1, int(completion_index)),
        "raw_question_count": max(0, int(raw_question_count)),
        "frame_count": max(0, int(frame_count)),
        "pipeline_outcome": str(pipeline_outcome),
        "updated_at": str(updated_at),
    }


def _mark_pipeline_scene_completed(
    scene_status_doc: dict[str, object],
    *,
    scene_id: str,
    raw_question_count: int,
    frame_count: int,
    pipeline_outcome: str,
) -> dict[str, object]:
    completed_scenes = _pipeline_completed_scene_records(scene_status_doc)
    completion_index = 0
    for record in completed_scenes.values():
        if not isinstance(record, dict):
            continue
        completion_index = max(
            completion_index,
            _coerce_pipeline_scene_completion_index(record.get("completion_index")) or 0,
        )
    updated_record = _build_pipeline_scene_status_record(
        scene_id,
        completion_index=completion_index + 1,
        raw_question_count=raw_question_count,
        frame_count=frame_count,
        pipeline_outcome=pipeline_outcome,
        updated_at=_scene_status_updated_at_now(),
    )
    completed_scenes[str(scene_id)] = updated_record
    return updated_record


def _reset_pipeline_completed_scenes(
    scene_status_doc: dict[str, object],
    *,
    count: int,
) -> list[str]:
    if int(count) <= 0:
        raise ValueError("count must be >= 1")
    completed_scenes = _pipeline_completed_scene_records(scene_status_doc)
    ranked_items = sorted(
        completed_scenes.items(),
        key=lambda item: (
            _coerce_pipeline_scene_completion_index(
                item[1].get("completion_index") if isinstance(item[1], dict) else None
            ) or -1,
            _parse_pipeline_scene_status_updated_at(
                item[1].get("updated_at") if isinstance(item[1], dict) else None
            ) or datetime.min.replace(tzinfo=timezone.utc),
            str(item[0]),
        ),
        reverse=True,
    )
    removed_scene_ids = [str(scene_id) for scene_id, _ in ranked_items[: int(count)]]
    for scene_id in removed_scene_ids:
        completed_scenes.pop(scene_id, None)
    return removed_scene_ids


def _clear_pipeline_resume_state(output_dir: Path) -> None:
    scene_status_path = _pipeline_scene_status_path(output_dir)
    raw_questions_dir = _raw_scene_questions_cache_dir(output_dir)
    cross_frame_cache_dir = output_dir / CROSS_FRAME_SCENE_CACHE_DIRNAME
    try:
        scene_status_path.unlink()
    except FileNotFoundError:
        pass
    shutil.rmtree(raw_questions_dir, ignore_errors=True)
    shutil.rmtree(cross_frame_cache_dir, ignore_errors=True)


def _delete_raw_scene_cache_files(raw_questions_dir: Path, scene_ids: list[str]) -> None:
    for scene_id in scene_ids:
        try:
            (raw_questions_dir / f"{scene_id}.json").unlink()
        except FileNotFoundError:
            pass


def _delete_cross_frame_scene_cache_dirs(output_dir: Path, scene_ids: list[str]) -> None:
    for scene_id in scene_ids:
        shutil.rmtree(
            _cross_frame_scene_cache_dir(output_dir, scene_id),
            ignore_errors=True,
        )


def _reconcile_pipeline_completed_scenes(
    scene_status_doc: dict[str, object],
    *,
    raw_questions_dir: Path,
    target_scene_ids: list[str],
) -> tuple[list[str], list[str], bool]:
    completed_scenes = _pipeline_completed_scene_records(scene_status_doc)
    changed = False
    corrupted_scene_ids: list[str] = []
    completed_scene_ids: list[str] = []
    for scene_id in target_scene_ids:
        if scene_id not in completed_scenes:
            continue
        raw_question_path = raw_questions_dir / f"{scene_id}.json"
        if not raw_question_path.exists():
            completed_scenes.pop(scene_id, None)
            corrupted_scene_ids.append(scene_id)
            changed = True
            continue
        completed_scene_ids.append(scene_id)
    return completed_scene_ids, corrupted_scene_ids, changed


def _build_benchmark_payload(questions: list[dict[str, object]]) -> dict[str, object]:
    _validate_strict_object_centric_questions(
        questions,
        source="benchmark output",
    )
    return {
        "name": "PSR-Bench",
        "version": "1.0",
        "object_move_semantics_version": L2_OBJECT_MOVE_SEMANTICS_VERSION,
        "object_move_object_centric_semantics": (
            OBJECT_MOVE_OBJECT_CENTRIC_SEMANTICS_PROFILE
        ),
        "statistics": compute_statistics(questions),
        "questions": questions,
    }


def _write_benchmark_file(output_dir: Path, questions: list[dict[str, object]]) -> Path:
    benchmark_path = output_dir / "benchmark.json"
    _write_json_file(benchmark_path, _build_benchmark_payload(questions))
    return benchmark_path


def _finalize_scene_debug_file(
    debug_path: Path,
    *,
    final_questions_by_frame: dict[str, list[dict[str, object]]],
) -> None:
    if not debug_path.exists():
        return

    with open(debug_path, "r", encoding="utf-8") as f:
        record = json.load(f)

    frames = record.get("frames", [])
    if isinstance(frames, list):
        total_generated = 0
        total_final = 0
        for frame_entry in frames:
            if not isinstance(frame_entry, dict):
                continue
            generated_questions = frame_entry.get("generated_questions", [])
            if isinstance(generated_questions, list):
                total_generated += len(generated_questions)
            final_frame_questions = list(
                final_questions_by_frame.get(str(frame_entry.get("image_name", "")), [])
            )
            frame_entry["final_questions"] = final_frame_questions
            frame_entry["final_question_count"] = len(final_frame_questions)
            total_final += len(final_frame_questions)
        record["summary"] = {
            "frame_count": len(frames),
            "generated_question_count": total_generated,
            "final_question_count": total_final,
        }

    _write_json_file(debug_path, record)


def _deduplicate_scene_questions(scene_questions: list[dict], max_per_key: int = 2) -> list[dict]:
    """Keep at most max_per_key questions with same (question text, object ID combo) per scene."""
    kept: list[dict] = []
    seen: dict[tuple, int] = {}
    for q in scene_questions:
        key = _make_dedup_key(q)
        count = seen.get(key, 0)
        if count >= max_per_key:
            continue
        kept.append(q)
        seen[key] = count + 1
    return kept


def _canonical_scene_question_type(question: dict) -> str:
    return str(question.get("type", "")).strip().lower()


_VERTICAL_RELATION_LABELS = frozenset({"above", "below"})


def _strict_vertical_object_pair(obj_a: dict, obj_b: dict) -> bool | None:
    """Return whether two objects satisfy the benchmark's strict above/below rule.

    ``None`` means the geometry is missing or malformed.  The caller deliberately
    keeps such questions instead of turning incomplete metadata into a false
    positive removal.
    """
    try:
        required_vectors = [
            np.asarray(obj_a[field], dtype=float)
            for field in ("center", "bbox_min", "bbox_max")
        ] + [
            np.asarray(obj_b[field], dtype=float)
            for field in ("center", "bbox_min", "bbox_max")
        ]
        if any(vector.shape != (3,) or not np.all(np.isfinite(vector)) for vector in required_vectors):
            return None

        direction, _ambiguity = primary_direction_allocentric(
            required_vectors[0],
            required_vectors[3],
            obj_a_hull_xy=_object_bottom_hull_xy(obj_a),
            obj_b_hull_xy=_object_bottom_hull_xy(obj_b),
            obj_a_bbox_min=required_vectors[1],
            obj_a_bbox_max=required_vectors[2],
            obj_b_bbox_min=required_vectors[4],
            obj_b_bbox_max=required_vectors[5],
        )
    except (AttributeError, IndexError, KeyError, TypeError, ValueError):
        return None
    return direction in _VERTICAL_RELATION_LABELS


def _question_orbit_signed_angle(question: dict) -> float | None:
    try:
        angle = abs(float(question["rotation_angle"]))
    except (KeyError, TypeError, ValueError):
        return None
    if not np.isfinite(angle):
        return None

    direction = str(question.get("rotation_direction", "")).strip().lower()
    if direction == "clockwise":
        return -angle
    if direction == "counterclockwise":
        return angle
    return None


def _filter_vertical_object_rotate_questions(
    questions: list[dict],
    *,
    scene_objects: list[dict],
    attachment_graph: dict[int, list[int]],
) -> list[dict]:
    """Drop L2 object-rotate/object-centric questions with vertical role pairs.

    Both the original scene and the question's simulated orbit-rotation state
    are checked.  A strict above/below relation between query-ref or query-face
    in either state makes the horizontal object-centric question invalid.
    """
    objects_by_id: dict[int, dict] = {}
    rotated_scene_cache: dict[tuple[int, int, float], dict[int, dict] | None] = {}
    malformed_geometry_count = 0
    for obj in scene_objects:
        obj_id = _coerce_object_id(obj.get("id")) if isinstance(obj, dict) else None
        if obj_id is not None:
            objects_by_id[obj_id] = obj

    kept: list[dict] = []
    removed = 0
    for question in questions:
        if _canonical_scene_question_type(question) != "object_rotate_object_centric":
            kept.append(question)
            continue

        query_id = _coerce_object_id(question.get("query_obj_id"))
        ref_id = _coerce_object_id(question.get("obj_ref_id"))
        face_id = _coerce_object_id(question.get("obj_face_id"))
        moved_id = _coerce_object_id(question.get("moved_obj_id"))
        role_ids = (query_id, ref_id, face_id, moved_id)
        if any(obj_id is None for obj_id in role_ids):
            malformed_geometry_count += 1
            kept.append(question)
            continue

        query_obj = objects_by_id.get(int(query_id))
        ref_obj = objects_by_id.get(int(ref_id))
        face_obj = objects_by_id.get(int(face_id))
        if query_obj is None or ref_obj is None or face_obj is None:
            malformed_geometry_count += 1
            kept.append(question)
            continue

        original_relations = (
            _strict_vertical_object_pair(query_obj, ref_obj),
            _strict_vertical_object_pair(query_obj, face_obj),
        )
        if any(relation is True for relation in original_relations):
            removed += 1
            continue
        if any(relation is None for relation in original_relations):
            malformed_geometry_count += 1
            kept.append(question)
            continue

        signed_angle = _question_orbit_signed_angle(question)
        if signed_angle is None:
            malformed_geometry_count += 1
            kept.append(question)
            continue

        rotation_key = (int(moved_id), int(face_id), float(signed_angle))
        if rotation_key not in rotated_scene_cache:
            try:
                rotated_objects = apply_orbit_rotation(
                    scene_objects,
                    attachment_graph,
                    int(moved_id),
                    int(face_id),
                    signed_angle,
                )
                rotated_scene_cache[rotation_key] = {
                    int(obj["id"]): obj for obj in rotated_objects
                }
            except (AttributeError, IndexError, KeyError, TypeError, ValueError):
                rotated_scene_cache[rotation_key] = None

        rotated_by_id = rotated_scene_cache[rotation_key]
        if rotated_by_id is None:
            malformed_geometry_count += 1
            kept.append(question)
            continue
        try:
            rotated_query = rotated_by_id[int(query_id)]
            rotated_ref = rotated_by_id[int(ref_id)]
            rotated_face = rotated_by_id[int(face_id)]
        except (KeyError, TypeError):
            malformed_geometry_count += 1
            kept.append(question)
            continue

        final_relations = (
            _strict_vertical_object_pair(rotated_query, rotated_ref),
            _strict_vertical_object_pair(rotated_query, rotated_face),
        )
        if any(relation is True for relation in final_relations):
            removed += 1
            continue
        if any(relation is None for relation in final_relations):
            malformed_geometry_count += 1
        kept.append(question)

    if removed:
        logger.info(
            "Vertical object-rotate filter removed %d question(s) with query-ref/query-face above/below geometry",
            removed,
        )
    if malformed_geometry_count:
        logger.warning(
            "Vertical object-rotate filter kept %d question(s) because required IDs, rotation fields, or geometry were incomplete",
            malformed_geometry_count,
        )
    return kept


_ALL_CANONICAL_QUESTION_TYPES = {
    "direction_agent",
    "occlusion",
    "distance",
    "direction_object_centric",
    "direction_allocentric",
    "object_move_agent",
    "object_move_distance",
    "object_move_occlusion",
    "object_move_object_centric",
    "object_rotate_object_centric",
    "object_move_allocentric",
    "object_remove",
    "attachment_chain",
    "attachment_move",
    "coordinate_rotation_agent",
    "coordinate_rotation_object_centric",
    "coordinate_rotation_allocentric",
}
# L1/L2/L3 have no final scene/type total cap. Candidate generation budgets
# are enforced separately; fixed per-primary-object limits are applied below.
_SCENE_TYPE_CAP_ELIGIBLE_TYPES = {
    "direction_agent",
    "occlusion",
    "distance",
    "direction_object_centric",
    "direction_allocentric",
}
_L2_CANONICAL_QUESTION_TYPES = {
    "object_move_agent",
    "object_move_distance",
    "object_move_occlusion",
    "object_move_object_centric",
    "object_rotate_object_centric",
    "object_move_allocentric",
    "object_remove",
}
_L3_CANONICAL_QUESTION_TYPES = {
    "attachment_chain",
    "attachment_move",
    "coordinate_rotation_agent",
    "coordinate_rotation_object_centric",
    "coordinate_rotation_allocentric",
}
_L3_FRAME_TYPE_OBJECT_CAP = 1
_L1_FRAME_TYPE_OBJECT_CAP = 1
_PUBLIC_TO_CANONICAL_QUESTION_TYPES = {
    "L1_direction_agent": "direction_agent",
    "L1_occlusion": "occlusion",
    "L1_distance": "distance",
    "L1_direction_object_centric": "direction_object_centric",
    "L1_direction_allocentric": "direction_allocentric",
    "L2_object_move_agent": "object_move_agent",
    "L2_object_move_distance": "object_move_distance",
    "L2_object_move_occlusion": "object_move_occlusion",
    "L2_object_move_object_centric": "object_move_object_centric",
    "L2_object_rotate_object_centric": "object_rotate_object_centric",
    "L2_object_move_allocentric": "object_move_allocentric",
    "L2_object_remove": "object_remove",
    "L3_attachment_chain": "attachment_chain",
    "L3_attachment_move": "attachment_move",
    "L3_coordinate_rotation_agent": "coordinate_rotation_agent",
    "L3_coordinate_rotation_object_centric": "coordinate_rotation_object_centric",
    "L3_coordinate_rotation_allocentric": "coordinate_rotation_allocentric",
}
_ATTACHMENT_ONLY_L2_PUBLIC_TYPES = {
    "L2_object_move_agent",
    "L2_object_move_distance",
    "L2_object_move_object_centric",
    "L2_object_rotate_object_centric",
    "L2_object_move_allocentric",
}
_QUESTION_CAP_OBJECT_ID_FIELD_BY_TYPE = {
    "direction_agent": "obj_a_id",
    "occlusion": "obj_a_id",
    "distance": "obj_a_id",
    "direction_object_centric": "obj_target_id",
    "direction_allocentric": "obj_a_id",
    "object_move_agent": "query_obj_id",
    "object_move_distance": "query_obj_id",
    "object_move_occlusion": "query_obj_id",
    "object_move_object_centric": "query_obj_id",
    "object_rotate_object_centric": "query_obj_id",
    "object_move_allocentric": "query_obj_id",
    "object_remove": "removed_obj_id",
    "attachment_chain": "grandparent_id",
    "attachment_move": "query_obj_id",
    "coordinate_rotation_agent": "obj_a_id",
    "coordinate_rotation_object_centric": "obj_target_id",
    "coordinate_rotation_allocentric": "obj_a_id",
}
def _question_cap_object_id(question: dict) -> str:
    canonical_type = _canonical_scene_question_type(question)
    primary_field = _QUESTION_CAP_OBJECT_ID_FIELD_BY_TYPE.get(canonical_type)
    if primary_field is not None and question.get(primary_field) is not None:
        return str(question[primary_field])
    for field in _QUESTION_CAP_OBJECT_ID_FIELDS:
        value = question.get(field)
        if value is not None:
            return str(value)
    trace_question_id = question.get("trace_question_id")
    if trace_question_id is not None:
        return f"trace:{trace_question_id}"
    return f"question:{question.get('question', '')}"


def _question_pair_key(question: dict) -> tuple[str, str, str] | None:
    return _shared_question_pair_key(question)


def _only_l2_attachment_types_requested(only_question_types: list[str] | None) -> bool:
    return bool(only_question_types) and all(
        str(question_type) in _ATTACHMENT_ONLY_L2_PUBLIC_TYPES
        for question_type in only_question_types
    )


def _normalize_only_question_types(only_question_types: list[str] | None) -> list[str] | None:
    if only_question_types is None:
        return None
    normalized: list[str] = []
    for question_type in only_question_types:
        text = str(question_type).strip()
        if not text:
            continue
        normalized.append(text)
    return normalized or None


def _frame_has_attachment_pair(
    frame: dict[str, object],
    referability_entry: dict[str, object] | None,
    attachment_graph: dict[int, list[int]] | dict[str, list[int]],
) -> bool:
    if referability_entry is not None:
        visible_ids = set(_normalize_object_ids(frame.get("visible_object_ids")))
        if not visible_ids:
            visible_ids = set(
                _normalize_object_ids((referability_entry or {}).get("candidate_visible_object_ids"))
            )
        if "attachment_referable_pairs" in referability_entry:
            return any(
                not visible_ids
                or (parent_id in visible_ids and child_id in visible_ids)
                for parent_id, child_id in _shared_normalize_attachment_pairs(
                    referability_entry.get("attachment_referable_pairs")
                )
            )

    if int(frame.get("attachment_referable_pair_count", 0) or 0) > 0:
        return True

    graph = {
        int(parent_id): [int(child_id) for child_id in (child_ids or [])]
        for parent_id, child_ids in (attachment_graph or {}).items()
    }
    return any(child_ids for child_ids in graph.values())


_L1_L3_PAIR_SCENE_CAP = 3


def _apply_incremental_question_caps(
    questions: list[dict],
    *,
    frame_type_cap: int,
    frame_type_object_cap: int,
    scene_type_counts: Counter[str] | None = None,
    frame_type_counts: Counter[tuple[str, str, str]] | None = None,
    frame_type_object_counts: Counter[tuple[str, str, str, str]] | None = None,
    pair_counts: Counter[tuple[str, str, str]] | None = None,
) -> list[dict]:
    kept: list[dict] = []
    scene_counts = scene_type_counts if scene_type_counts is not None else Counter()
    frame_counts = frame_type_counts if frame_type_counts is not None else Counter()
    frame_object_counts = (
        frame_type_object_counts if frame_type_object_counts is not None else Counter()
    )
    pair_counter = pair_counts if pair_counts is not None else Counter()
    # Keyed by (scene_id, frame_identity, *pair_key), scoped to this batch.
    frame_pair_counter: Counter[tuple[str, ...]] = Counter()

    for question in questions:
        if bool(question.get("manual_attachment_override", False)):
            kept.append(question)
            continue
        canonical_type = _canonical_scene_question_type(question)
        if not canonical_type:
            kept.append(question)
            continue
        scene_id = str(question.get("scene_id", "")).strip()
        image_name = str(question.get("image_name", "")).strip()
        reasoning_frame_2 = str(question.get("reasoning_frame_2", "")).strip()
        frame_identity = (
            f"{image_name}\0{reasoning_frame_2}"
            if reasoning_frame_2
            else image_name
        )
        object_id = _question_cap_object_id(question)
        pair_key = _question_pair_key(question)
        frame_key = (scene_id, frame_identity, canonical_type)
        frame_object_key = (scene_id, frame_identity, canonical_type, object_id)
        frame_pair_key = (
            (scene_id, frame_identity, *pair_key)
            if pair_key is not None
            else None
        )
        type_is_cap_eligible = canonical_type in _SCENE_TYPE_CAP_ELIGIBLE_TYPES
        if type_is_cap_eligible:
            # L1 follows the multi-image L3 distribution rule: no frame/type
            # total cap, and one question per primary object/frame identity.
            effective_frame_type_cap = 0
            effective_frame_type_object_cap = _L1_FRAME_TYPE_OBJECT_CAP
        elif canonical_type in _L2_CANONICAL_QUESTION_TYPES:
            effective_frame_type_cap = 0
            effective_frame_type_object_cap = 0
        elif canonical_type in _L3_CANONICAL_QUESTION_TYPES:
            effective_frame_type_cap = 0
            effective_frame_type_object_cap = _L3_FRAME_TYPE_OBJECT_CAP
        else:
            effective_frame_type_cap = 0
            effective_frame_type_object_cap = 0
        if effective_frame_type_cap > 0 and frame_counts[frame_key] >= effective_frame_type_cap:
            continue
        if (
            effective_frame_type_object_cap > 0
            and frame_object_counts[frame_object_key] >= effective_frame_type_object_cap
        ):
            continue
        pair_diversity_cap_applies = canonical_type not in _L2_CANONICAL_QUESTION_TYPES
        if pair_diversity_cap_applies:
            if frame_pair_key is not None and frame_pair_counter[frame_pair_key] >= 1:
                continue
            if pair_key is not None and pair_counter[pair_key] >= _L1_L3_PAIR_SCENE_CAP:
                continue
        kept.append(question)
        scene_counts[canonical_type] += 1
        frame_counts[frame_key] += 1
        frame_object_counts[frame_object_key] += 1
        if pair_diversity_cap_applies and pair_key is not None:
            pair_counter[pair_key] += 1
        if pair_diversity_cap_applies and frame_pair_key is not None:
            frame_pair_counter[frame_pair_key] += 1
    return kept


def _scene_question_target_types(only_question_types: list[str] | None) -> set[str]:
    if not only_question_types:
        return set(_ALL_CANONICAL_QUESTION_TYPES)
    target_types = {
        canonical_type
        for question_type in only_question_types
        for canonical_type in [_PUBLIC_TO_CANONICAL_QUESTION_TYPES.get(str(question_type))]
        if canonical_type
    }
    return target_types or set(_ALL_CANONICAL_QUESTION_TYPES)


def _apply_scene_type_cap(
    questions: list[dict],
    *,
    scene_type_cap: int,
    frame_type_cap: int = 0,
    frame_type_object_cap: int = 0,
    type_counts: Counter[str] | None = None,
    frame_type_counts: Counter[tuple[str, str, str]] | None = None,
    frame_type_object_counts: Counter[tuple[str, str, str, str]] | None = None,
    pair_counts: Counter[tuple[str, str, str]] | None = None,
    scene_question_hard_cap: int = 0,
) -> list[dict]:
    """Apply object/frame and pair diversity caps, never a type-total cap.

    ``scene_type_cap`` and ``scene_question_hard_cap`` remain accepted for
    compatibility with older internal callers. They now control candidate
    generation only and do not truncate retained questions.
    """
    _ = scene_type_cap, scene_question_hard_cap
    return _apply_incremental_question_caps(
        questions,
        frame_type_cap=frame_type_cap,
        frame_type_object_cap=frame_type_object_cap,
        scene_type_counts=type_counts,
        frame_type_counts=frame_type_counts,
        frame_type_object_counts=frame_type_object_counts,
        pair_counts=pair_counts,
    )


def _candidate_budget_for_type(
    canonical_type: str,
    *,
    l1_candidate_budget: int,
    l2_candidate_budget: int | None = None,
    l3_candidate_budget: int | None = None,
    l2_l3_candidate_budget: int | None = None,
) -> int | None:
    # Keep the shared argument as a compatibility alias for older internal callers.
    shared_budget = int(l2_l3_candidate_budget or 0)
    resolved_l2_budget = (
        shared_budget if l2_candidate_budget is None else int(l2_candidate_budget)
    )
    resolved_l3_budget = (
        shared_budget if l3_candidate_budget is None else int(l3_candidate_budget)
    )
    if canonical_type in _SCENE_TYPE_CAP_ELIGIBLE_TYPES:
        budget = l1_candidate_budget
    elif canonical_type in _L2_CANONICAL_QUESTION_TYPES:
        budget = resolved_l2_budget
    elif canonical_type in _L3_CANONICAL_QUESTION_TYPES:
        budget = resolved_l3_budget
    else:
        return None
    return int(budget) if int(budget) > 0 else None


def _remaining_candidate_type_budgets(
    type_counts: Counter[str],
    *,
    l1_candidate_budget: int,
    l2_candidate_budget: int | None = None,
    l3_candidate_budget: int | None = None,
    l2_l3_candidate_budget: int | None = None,
    allowed_types: set[str] | None = None,
) -> dict[str, int] | None:
    target_types = set(allowed_types) if allowed_types else set(_ALL_CANONICAL_QUESTION_TYPES)
    budgets: dict[str, int] = {}
    for question_type in sorted(target_types):
        budget = _candidate_budget_for_type(
            question_type,
            l1_candidate_budget=l1_candidate_budget,
            l2_candidate_budget=l2_candidate_budget,
            l3_candidate_budget=l3_candidate_budget,
            l2_l3_candidate_budget=l2_l3_candidate_budget,
        )
        if budget is not None:
            budgets[question_type] = max(budget - int(type_counts[question_type]), 0)
    return budgets or None


def _take_questions_within_candidate_budgets(
    questions: list[dict],
    type_counts: Counter[str],
    *,
    l1_candidate_budget: int,
    l2_candidate_budget: int | None = None,
    l3_candidate_budget: int | None = None,
    l2_l3_candidate_budget: int | None = None,
) -> list[dict]:
    kept: list[dict] = []
    for question in questions:
        if bool(question.get("manual_attachment_override", False)):
            kept.append(question)
            continue
        canonical_type = _canonical_scene_question_type(question)
        budget = _candidate_budget_for_type(
            canonical_type,
            l1_candidate_budget=l1_candidate_budget,
            l2_candidate_budget=l2_candidate_budget,
            l3_candidate_budget=l3_candidate_budget,
            l2_l3_candidate_budget=l2_l3_candidate_budget,
        )
        if budget is not None and type_counts[canonical_type] >= budget:
            continue
        kept.append(question)
        if canonical_type:
            type_counts[canonical_type] += 1
    return kept


def _candidate_type_budget_remaining(
    type_counts: Counter[str],
    canonical_type: str,
    *,
    l1_candidate_budget: int,
    l2_candidate_budget: int | None = None,
    l3_candidate_budget: int | None = None,
    l2_l3_candidate_budget: int | None = None,
) -> int | None:
    budgets = _remaining_candidate_type_budgets(
        type_counts,
        l1_candidate_budget=l1_candidate_budget,
        l2_candidate_budget=l2_candidate_budget,
        l3_candidate_budget=l3_candidate_budget,
        l2_l3_candidate_budget=l2_l3_candidate_budget,
        allowed_types={canonical_type},
    )
    if budgets is None:
        return None
    return budgets.get(canonical_type)


def _all_candidate_type_budgets_exhausted(
    type_counts: Counter[str],
    canonical_types: set[str],
    *,
    l1_candidate_budget: int,
    l2_candidate_budget: int | None = None,
    l3_candidate_budget: int | None = None,
    l2_l3_candidate_budget: int | None = None,
) -> bool:
    limited_types = {
        canonical_type
        for canonical_type in canonical_types
        if _candidate_budget_for_type(
            canonical_type,
            l1_candidate_budget=l1_candidate_budget,
            l2_candidate_budget=l2_candidate_budget,
            l3_candidate_budget=l3_candidate_budget,
            l2_l3_candidate_budget=l2_l3_candidate_budget,
        ) is not None
    }
    return bool(limited_types) and limited_types == canonical_types and all(
        _candidate_type_budget_remaining(
            type_counts,
            canonical_type,
            l1_candidate_budget=l1_candidate_budget,
            l2_candidate_budget=l2_candidate_budget,
            l3_candidate_budget=l3_candidate_budget,
            l2_l3_candidate_budget=l2_l3_candidate_budget,
        ) == 0
        for canonical_type in limited_types
    )


def _default_l1_candidate_budget(split: str | None) -> int:
    """Return the default per-scene/type L1 generation budget."""
    split_name = str(split or "").strip().lower()
    return L1_CANDIDATE_BUDGET_BY_SPLIT.get(split_name, 0)


def _default_l2_candidate_budget(split: str | None) -> int:
    """Return the default per-scene/type L2 generation budget."""
    split_name = str(split or "").strip().lower()
    return L2_CANDIDATE_BUDGET_BY_SPLIT.get(split_name, 0)


def _default_l3_candidate_budget(split: str | None) -> int:
    """Return the default per-scene/type L3 generation budget."""
    split_name = str(split or "").strip().lower()
    return L3_CANDIDATE_BUDGET_BY_SPLIT.get(split_name, 0)


def _make_dedup_key(q: dict) -> tuple:
    """Build a dedup key from question text + sorted object IDs."""
    obj_id_fields = [
        "obj_a_id", "obj_b_id", "obj_ref_id", "obj_face_id", "obj_target_id",
        "query_obj_id", "moved_obj_id", "removed_obj_id",
        "parent_id", "child_id", "grandparent_id", "grandchild_id",
        "neighbor_id", "obj_c_id", "target_obj_id",
    ]
    ids = []
    for field in obj_id_fields:
        val = q.get(field)
        if val is not None:
            ids.append(str(val))
    ids_sorted = tuple(sorted(ids))
    reasoning_frame_2 = str(q.get("reasoning_frame_2", "")).strip()
    main_frame_pair = (
        (str(q.get("image_name", "")).strip(), reasoning_frame_2)
        if reasoning_frame_2
        else ()
    )
    return (q.get("scene_id"), q.get("question"), ids_sorted, main_frame_pair)


def _load_cached_scene_questions(
    raw_questions_dir: Path,
    *,
    scene_ids: list[str],
    scene_type_cap: int,
    frame_type_cap: int,
    frame_type_object_cap: int,
    scene_question_hard_cap: int = 0,
    referability_cache: dict | None = None,
) -> tuple[list[dict], int]:
    all_questions: list[dict] = []
    raw_question_count = 0
    attachment_surface_text_by_scene_image: dict[tuple[str, str], dict[int, str]] = {}
    if referability_cache is not None:
        frames = referability_cache.get("frames", referability_cache)
        if isinstance(frames, dict):
            for scene_id, scene_frames in frames.items():
                if not isinstance(scene_frames, dict):
                    continue
                for image_name, entry in scene_frames.items():
                    if not isinstance(entry, dict):
                        continue
                    attachment_surface_text_by_scene_image[(str(scene_id), str(image_name))] = (
                        _attachment_surface_text_by_object_id(entry)
                    )
    for scene_id in scene_ids:
        raw_question_path = raw_questions_dir / f"{scene_id}.json"
        if not raw_question_path.exists():
            continue
        with open(raw_question_path, "r", encoding="utf-8") as f:
            scene_questions = json.load(f)
        if not isinstance(scene_questions, list):
            logger.warning(
                "Skipping malformed raw scene cache for %s at %s because it is not a JSON list",
                scene_id,
                raw_question_path,
            )
            continue
        _validate_strict_object_centric_questions(
            scene_questions,
            source=str(raw_question_path),
        )
        raw_question_count += len(scene_questions)
        scene_questions = _deduplicate_scene_questions(scene_questions)
        scene_questions = _apply_scene_type_cap(
            scene_questions,
            scene_type_cap=scene_type_cap,
            frame_type_cap=frame_type_cap,
            frame_type_object_cap=frame_type_object_cap,
            scene_question_hard_cap=scene_question_hard_cap,
        )
        if attachment_surface_text_by_scene_image:
            for idx, question in enumerate(scene_questions):
                image_name = str(question.get("image_name", "")).strip()
                if not image_name:
                    continue
                surface_text_by_obj_id = attachment_surface_text_by_scene_image.get((scene_id, image_name))
                if not surface_text_by_obj_id:
                    continue
                scene_questions[idx] = _apply_attachment_surface_text_overrides(
                    question,
                    surface_text_by_obj_id,
                )
        all_questions.extend(scene_questions)
    return all_questions, raw_question_count


def _rebuild_pipeline_outputs(
    *,
    data_root: Path,
    output_dir: Path,
    questions_dir: Path,
    frame_debug_dir: Path,
    raw_questions_dir: Path,
    scene_ids: list[str],
    referability_cache: dict | None,
    write_frame_debug: bool,
    run_question_dinox_audit: bool,
    run_question_presence_review: bool,
    vlm_url: str | None,
    vlm_model: str | None,
    question_presence_review_workers: int,
    scene_type_cap: int,
    frame_type_cap: int,
    frame_type_object_cap: int,
    scene_question_hard_cap: int = 0,
    dataset: str = "scannet",
    scannetpp_sensor: str = "iphone",
    scannetpp_frame_root: str | None = None,
) -> list[dict]:
    all_questions, raw_question_count = _load_cached_scene_questions(
        raw_questions_dir,
        scene_ids=scene_ids,
        scene_type_cap=scene_type_cap,
        frame_type_cap=frame_type_cap,
        frame_type_object_cap=frame_type_object_cap,
        scene_question_hard_cap=scene_question_hard_cap,
        referability_cache=referability_cache,
    )

    logger.info(
        "Running benchmark quality control on %d raw questions (viewer-only attachment filtering excluded)",
        raw_question_count,
    )
    final_questions = full_quality_pipeline(all_questions)
    question_review_frame_contexts: dict[tuple[str, str], dict[str, object]] | None = None
    if run_question_dinox_audit or run_question_presence_review:
        question_review_frame_contexts = _prebuild_question_review_frame_contexts(
            questions=final_questions,
            data_root=Path(data_root),
            output_dir=output_dir,
            dataset=dataset,
            scannetpp_sensor=scannetpp_sensor,
            scannetpp_frame_root=scannetpp_frame_root,
        )
    if run_question_dinox_audit:
        final_questions = _apply_question_post_generation_audit(
            questions=final_questions,
            data_root=Path(data_root),
            output_dir=output_dir,
            frame_context_by_key=question_review_frame_contexts,
            dataset=dataset,
            scannetpp_sensor=scannetpp_sensor,
            scannetpp_frame_root=scannetpp_frame_root,
        )
    else:
        logger.info("Skipping DINO-X-dependent post-generation audit")

    by_scene: dict[str, list[dict]] = defaultdict(list)
    final_by_scene_frame: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for question in final_questions:
        by_scene[str(question["scene_id"])].append(question)
        final_by_scene_frame[str(question["scene_id"])][str(question["image_name"])].append(question)

    for scene_id in scene_ids:
        _write_json_file(questions_dir / f"{scene_id}.json", by_scene.get(scene_id, []))

    if write_frame_debug:
        for scene_id in scene_ids:
            _finalize_scene_debug_file(
                frame_debug_dir / f"{scene_id}.json",
                final_questions_by_frame=final_by_scene_frame.get(scene_id, {}),
            )

    benchmark_path = _write_benchmark_file(output_dir, final_questions)

    if run_question_presence_review:
        _run_question_presence_review(
            questions=final_questions,
            data_root=data_root,
            output_dir=output_dir,
            vlm_url=vlm_url,
            vlm_model=vlm_model,
            workers=question_presence_review_workers,
            frame_context_by_key=question_review_frame_contexts,
            dataset=dataset,
            scannetpp_sensor=scannetpp_sensor,
            scannetpp_frame_root=scannetpp_frame_root,
        )

    logger.info(
        "Pipeline complete! %d questions saved to %s",
        len(final_questions),
        benchmark_path,
    )
    return final_questions


def run_pipeline(
    data_root: Path,
    output_dir: Path,
    dataset: str = "scannet",
    scannetpp_sensor: str = "iphone",
    split: str | None = None,
    scannetpp_split_file: str | None = None,
    scannetpp_frame_root: str | None = None,
    max_scenes: int = 300,
    max_frames: int = 5,
    use_occlusion: bool = True,
    referability_cache: dict | None = None,
    occlusion_backend: str = "mesh_ray",
    vlm_url: str | None = None,
    vlm_model: str | None = None,
    write_frame_debug: bool = True,
    run_question_dinox_audit: bool = False,
    run_question_presence_review: bool = True,
    question_presence_review_workers: int = 8,
    slow_frame_warn_seconds: float = 120.0,
    slow_phase_warn_seconds: float = 30.0,
    generator_progress_log_seconds: float = 15.0,
    slow_generator_warn_seconds: float = 60.0,
    resume: bool = False,
    rebuild_benchmark: bool = False,
    reset: int | None = None,
    only_question_types: list[str] | None = None,
    scene_type_cap: int | None = None,
    frame_type_cap: int = 2,
    frame_type_object_cap: int = 1,
    max_questions_per_scene_type: int | None = None,
    max_occlusion_objects: int | str | None = MAX_OCCLUSION_OBJECTS_AUTO,
    occlusion_max_references_per_query: int = 64,
    occlusion_max_combinations_per_scene: int = 2000,
    max_move_sources: int | None = None,
    auxiliary_route_method: str = AUXILIARY_ROUTE_METHOD_DEPTH_CORRIDOR_GEOMETRIC,
    auxiliary_max_pose_candidates: int = DEFAULT_MAX_CANDIDATE_POSES,
    scannetpp_depth_cache_size: int = DEFAULT_DEPTH_CACHE_SIZE,
    attachment_reference_cluster_radius_m: float = 0.5,
):
    """Execute the PSR-Bench generation pipeline or rebuild cached outputs."""
    _set_pipeline_random_seed()
    only_question_types = _normalize_only_question_types(only_question_types)

    auxiliary_route_method = str(auxiliary_route_method).strip().lower()
    if auxiliary_route_method not in AUXILIARY_ROUTE_METHODS:
        raise ValueError(
            f"Unknown auxiliary_route_method: {auxiliary_route_method!r}. "
            f"Expected one of {AUXILIARY_ROUTE_METHODS}."
        )
    auxiliary_max_pose_candidates = int(auxiliary_max_pose_candidates)
    if auxiliary_max_pose_candidates < 0:
        raise ValueError("auxiliary_max_pose_candidates must be >= 0")
    scannetpp_depth_cache_size = int(scannetpp_depth_cache_size)
    if scannetpp_depth_cache_size <= 0:
        raise ValueError("scannetpp_depth_cache_size must be > 0")
    attachment_reference_cluster_radius_m = float(
        attachment_reference_cluster_radius_m
    )
    if (
        not np.isfinite(attachment_reference_cluster_radius_m)
        or attachment_reference_cluster_radius_m < 0.0
    ):
        raise ValueError(
            "attachment_reference_cluster_radius_m must be finite and >= 0"
        )

    if referability_cache is None:
        raise ValueError(
            "run_pipeline requires a referability_cache or manual attachment cache"
        )
    if _has_manual_attachment_overrides(referability_cache):
        if only_question_types != MANUAL_ATTACHMENT_QUESTION_TYPES:
            logger.info(
                "Manual attachment cache detected; generating only %s",
                MANUAL_ATTACHMENT_QUESTION_TYPES[0],
            )
        only_question_types = list(MANUAL_ATTACHMENT_QUESTION_TYPES)
    if reset is not None and int(reset) <= 0:
        raise ValueError("reset must be >= 1")
    if reset is not None and not resume:
        raise ValueError("reset requires resume=True")
    if rebuild_benchmark and not resume:
        raise ValueError("rebuild_benchmark requires resume=True")
    if rebuild_benchmark and reset is not None:
        raise ValueError("rebuild_benchmark cannot be combined with reset")
    if max_questions_per_scene_type is not None:
        scene_type_cap = int(max_questions_per_scene_type)
    elif scene_type_cap is None:
        scene_type_cap = _default_l1_candidate_budget(split)
    scene_type_cap = int(scene_type_cap)
    l2_candidate_budget = _default_l2_candidate_budget(split)
    l3_candidate_budget = _default_l3_candidate_budget(split)
    frame_type_cap = int(frame_type_cap)
    frame_type_object_cap = int(frame_type_object_cap)
    if scene_type_cap < 0:
        raise ValueError("scene_type_cap must be >= 0")
    if frame_type_cap < 0:
        raise ValueError("frame_type_cap must be >= 0")
    if frame_type_object_cap < 0:
        raise ValueError("frame_type_object_cap must be >= 0")
    if max_occlusion_objects is not None and max_occlusion_objects != MAX_OCCLUSION_OBJECTS_AUTO:
        max_occlusion_objects = int(max_occlusion_objects)
        if max_occlusion_objects < 0:
            raise ValueError("max_occlusion_objects must be >= 0 or None")
    occlusion_max_references_per_query = int(occlusion_max_references_per_query)
    occlusion_max_combinations_per_scene = int(occlusion_max_combinations_per_scene)
    if occlusion_max_references_per_query < 0:
        raise ValueError("occlusion_max_references_per_query must be >= 0")
    if occlusion_max_combinations_per_scene < 0:
        raise ValueError("occlusion_max_combinations_per_scene must be >= 0")
    if dataset not in ("scannet", "scannetpp"):
        raise ValueError(f"Unknown dataset: {dataset!r}. Expected 'scannet' or 'scannetpp'.")
    if scene_type_cap > 0 or l2_candidate_budget > 0 or l3_candidate_budget > 0:
        logger.info(
            "Applying split=%s candidate budgets per (scene, type): L1=%s L2=%s L3=%s",
            split,
            scene_type_cap or "unlimited",
            l2_candidate_budget or "unlimited",
            l3_candidate_budget or "unlimited",
        )
    l3_attachment_chain_only = only_question_types == ["L3_attachment_chain"]
    l3_attachment_move_only = only_question_types == ["L3_attachment_move"]
    requested_public_types = (
        SINGLE_FRAME_PUBLIC_QUESTION_TYPES | CROSS_FRAME_PUBLIC_QUESTION_TYPES
        if only_question_types is None
        else {str(value).strip() for value in only_question_types}
    )
    single_frame_requested_types = sorted(
        requested_public_types & SINGLE_FRAME_PUBLIC_QUESTION_TYPES
    )
    cross_frame_requested_types = sorted(
        requested_public_types & CROSS_FRAME_PUBLIC_QUESTION_TYPES
    )
    attachment_chain_fast_path = single_frame_requested_types == ["L3_attachment_chain"]
    single_frame_scene_question_types = {
        _PUBLIC_TO_CANONICAL_QUESTION_TYPES[public_type]
        for public_type in single_frame_requested_types
    }
    if cross_frame_requested_types:
        logger.info("Cross-frame auxiliary route method: %s", auxiliary_route_method)
        if auxiliary_route_method == AUXILIARY_ROUTE_METHOD_DEPTH_CORRIDOR_GEOMETRIC:
            logger.info(
                "Depth-route camera-motion hard limits: disabled "
                "(soft motion costs remain enabled)"
            )
    attachment_only_l2_mode = False

    meta_dir = output_dir / "scene_metadata"
    questions_dir = output_dir / "questions"
    frame_debug_dir = output_dir / "frame_debug"
    cross_frame_funnel_dir = output_dir / "cross_frame_funnel"
    auxiliary_graph_cache_dir = output_dir / "auxiliary_graph_cache"
    meta_dir.mkdir(parents=True, exist_ok=True)
    questions_dir.mkdir(parents=True, exist_ok=True)
    if write_frame_debug:
        frame_debug_dir.mkdir(parents=True, exist_ok=True)
    if cross_frame_requested_types:
        cross_frame_funnel_dir.mkdir(parents=True, exist_ok=True)
        if auxiliary_route_method in {
            AUXILIARY_ROUTE_METHOD_VISUAL_POSE_GRAPH,
            AUXILIARY_ROUTE_METHOD_HYBRID_GEOMETRIC_VISUAL,
        }:
            auxiliary_graph_cache_dir.mkdir(parents=True, exist_ok=True)

    if dataset == "scannetpp":
        if scannetpp_split_file is not None:
            split_ids = [l.strip() for l in Path(scannetpp_split_file).read_text(encoding="utf-8").splitlines() if l.strip()]
            discovered_scene_dirs = [data_root / sid for sid in split_ids if (data_root / sid).is_dir()]
        elif split is not None and split in SCANNETPP_METADATA_SPLIT_FILES:
            resolved_split_file = SCANNETPP_METADATA_SPLIT_FILES[split]
            split_ids = [l.strip() for l in resolved_split_file.read_text(encoding="utf-8").splitlines() if l.strip()]
            discovered_scene_dirs = [data_root / sid for sid in split_ids if (data_root / sid).is_dir()]
        else:
            from src.datasets.scannetpp import has_scannetpp_geometry
            discovered_scene_dirs = sorted(p for p in data_root.iterdir() if p.is_dir() and has_scannetpp_geometry(p))
        if scannetpp_frame_root is None:
            scannetpp_frame_root = str(Path(data_root).parent / "iphone_frames")
    else:
        if split is not None and split in SCANNET_METADATA_SPLIT_FILES:
            resolved_split_file = SCANNET_METADATA_SPLIT_FILES[split]
            split_scene_ids = {l.strip() for l in resolved_split_file.read_text(encoding="utf-8").splitlines() if l.strip()}
            discovered_scene_dirs = sorted(
                p for p in data_root.iterdir()
                if p.is_dir() and (p / "pose").exists() and p.name in split_scene_ids
            )
        else:
            discovered_scene_dirs = sorted(
                p for p in data_root.iterdir()
                if p.is_dir() and (p / "pose").exists()
            )
    cached_scene_ids = _get_referability_scene_ids(referability_cache)
    scene_dirs = [p for p in discovered_scene_dirs if p.name in cached_scene_ids]
    scene_dirs = _prioritize_manual_attachment_scene_dirs(
        scene_dirs,
        referability_cache,
    )
    scene_limit = max(0, int(max_scenes))
    frame_limit = max(0, int(max_frames))
    discovered_cached_scene_count = len(scene_dirs)
    scene_dirs = scene_dirs[:scene_limit]
    logger.info(
        "Loaded %d cached scenes from referability cache; processing up to %d scene(s) and %d frame(s) per scene",
        discovered_cached_scene_count,
        len(scene_dirs),
        frame_limit,
    )

    total_scenes = len(scene_dirs)
    target_scene_ids = [scene_dir.name for scene_dir in scene_dirs]
    scene_status_path = _pipeline_scene_status_path(output_dir)
    raw_questions_dir = _raw_scene_questions_cache_dir(output_dir)
    if resume:
        scene_status_doc = _load_pipeline_scene_status_doc(scene_status_path)
        raw_questions_dir.mkdir(parents=True, exist_ok=True)
        if reset is not None:
            removed_scene_ids = _reset_pipeline_completed_scenes(
                scene_status_doc,
                count=int(reset),
            )
            _delete_raw_scene_cache_files(raw_questions_dir, removed_scene_ids)
            _delete_cross_frame_scene_cache_dirs(output_dir, removed_scene_ids)
            _write_json_file(scene_status_path, scene_status_doc)
            logger.info(
                "Reset cleared %d completed scene(s) from %s",
                len(removed_scene_ids),
                scene_status_path,
            )
        completed_route_records = _pipeline_completed_scene_records(scene_status_doc)
        recorded_l1_budget = scene_status_doc.get("l1_candidate_budget")
        legacy_l2_l3_budget = scene_status_doc.get("l2_l3_candidate_budget")
        recorded_l2_budget = scene_status_doc.get(
            "l2_candidate_budget", legacy_l2_l3_budget
        )
        recorded_l3_budget = scene_status_doc.get(
            "l3_candidate_budget", legacy_l2_l3_budget
        )
        recorded_occlusion_reference_budget = scene_status_doc.get(
            "occlusion_max_references_per_query"
        )
        recorded_occlusion_combination_budget = scene_status_doc.get(
            "occlusion_max_combinations_per_scene"
        )
        if completed_route_records and (
            recorded_l1_budget != scene_type_cap
            or recorded_l2_budget != l2_candidate_budget
            or recorded_l3_budget != l3_candidate_budget
            or recorded_occlusion_reference_budget
            != occlusion_max_references_per_query
            or recorded_occlusion_combination_budget
            != occlusion_max_combinations_per_scene
        ):
            raise RuntimeError(
                "Cannot resume with candidate/search budgets "
                f"L1={scene_type_cap}, L2={l2_candidate_budget}, "
                f"L3={l3_candidate_budget}, "
                f"occlusion references={occlusion_max_references_per_query}, "
                f"occlusion combinations={occlusion_max_combinations_per_scene}: "
                f"{len(completed_route_records)} completed scene(s) were generated "
                f"with L1={recorded_l1_budget!r}, L2={recorded_l2_budget!r}, "
                f"L3={recorded_l3_budget!r}, "
                f"occlusion references={recorded_occlusion_reference_budget!r}, "
                f"occlusion combinations={recorded_occlusion_combination_budget!r}. "
                "Use a new --output_dir, or reset all completed scenes before "
                "changing candidate budgets."
        )
        scene_status_doc["l1_candidate_budget"] = scene_type_cap
        scene_status_doc["l2_candidate_budget"] = l2_candidate_budget
        scene_status_doc["l3_candidate_budget"] = l3_candidate_budget
        scene_status_doc.pop("l2_l3_candidate_budget", None)
        scene_status_doc["occlusion_max_references_per_query"] = (
            occlusion_max_references_per_query
        )
        scene_status_doc["occlusion_max_combinations_per_scene"] = (
            occlusion_max_combinations_per_scene
        )
        recorded_route_method = scene_status_doc.get("auxiliary_route_method")
        if cross_frame_requested_types and completed_route_records:
            if recorded_route_method is None:
                # Status files written before this option existed used the visual graph.
                recorded_route_method = AUXILIARY_ROUTE_METHOD_VISUAL_POSE_GRAPH
            if recorded_route_method != auxiliary_route_method:
                raise RuntimeError(
                    "Cannot resume with auxiliary_route_method="
                    f"{auxiliary_route_method!r}: {len(completed_route_records)} completed "
                    f"scene(s) were generated with {recorded_route_method!r}. Use a new "
                    "--output_dir, or reset all completed scenes before switching methods."
                )
            if recorded_route_method == AUXILIARY_ROUTE_METHOD_DEPTH_CORRIDOR_GEOMETRIC:
                recorded_hard_limits = scene_status_doc.get(
                    "depth_route_camera_motion_hard_limits_enabled"
                )
                if recorded_hard_limits is None:
                    # Depth-route status files written before this policy field
                    # used the original hard camera-motion gates.
                    recorded_hard_limits = True
                if (
                    recorded_hard_limits
                    != DEPTH_ROUTE_ENFORCE_CAMERA_MOTION_HARD_LIMITS
                ):
                    raise RuntimeError(
                        "Cannot resume with depth-route camera-motion hard limits "
                        f"enabled={DEPTH_ROUTE_ENFORCE_CAMERA_MOTION_HARD_LIMITS}: "
                        f"{len(completed_route_records)} completed scene(s) were "
                        f"generated with enabled={recorded_hard_limits}. Use a new "
                        "--output_dir, or reset all completed scenes before changing "
                        "the depth-route camera-motion policy."
                    )
        scene_status_doc["auxiliary_route_method"] = auxiliary_route_method
        if auxiliary_route_method == AUXILIARY_ROUTE_METHOD_DEPTH_CORRIDOR_GEOMETRIC:
            scene_status_doc["depth_route_camera_motion_hard_limits_enabled"] = (
                DEPTH_ROUTE_ENFORCE_CAMERA_MOTION_HARD_LIMITS
            )
        else:
            scene_status_doc.pop(
                "depth_route_camera_motion_hard_limits_enabled", None
            )
        completed_scene_ids, corrupted_scene_ids, scene_status_changed = _reconcile_pipeline_completed_scenes(
            scene_status_doc,
            raw_questions_dir=raw_questions_dir,
            target_scene_ids=target_scene_ids,
        )
        if corrupted_scene_ids:
            recovery_action = (
                "excluded from the benchmark rebuild"
                if rebuild_benchmark
                else "regenerated"
            )
            logger.warning(
                "Resume found %d scene status record(s) with missing raw scene cache; they will be %s: %s",
                len(corrupted_scene_ids),
                recovery_action,
                ", ".join(corrupted_scene_ids),
            )
            if not rebuild_benchmark:
                _delete_cross_frame_scene_cache_dirs(output_dir, corrupted_scene_ids)
        if scene_status_changed:
            _write_json_file(scene_status_path, scene_status_doc)
        if not rebuild_benchmark:
            _delete_cross_frame_scene_cache_dirs(output_dir, completed_scene_ids)
    else:
        _clear_pipeline_resume_state(output_dir)
        scene_status_doc = _build_empty_pipeline_scene_status_doc()
        scene_status_doc["auxiliary_route_method"] = auxiliary_route_method
        if auxiliary_route_method == AUXILIARY_ROUTE_METHOD_DEPTH_CORRIDOR_GEOMETRIC:
            scene_status_doc["depth_route_camera_motion_hard_limits_enabled"] = (
                DEPTH_ROUTE_ENFORCE_CAMERA_MOTION_HARD_LIMITS
            )
        scene_status_doc["l1_candidate_budget"] = scene_type_cap
        scene_status_doc["l2_candidate_budget"] = l2_candidate_budget
        scene_status_doc["l3_candidate_budget"] = l3_candidate_budget
        scene_status_doc["occlusion_max_references_per_query"] = (
            occlusion_max_references_per_query
        )
        scene_status_doc["occlusion_max_combinations_per_scene"] = (
            occlusion_max_combinations_per_scene
        )
        raw_questions_dir.mkdir(parents=True, exist_ok=True)
        completed_scene_ids = []

    completed_scene_id_set = set(completed_scene_ids)
    pending_scene_entries = [
        (scene_index, scene_dir)
        for scene_index, scene_dir in enumerate(scene_dirs, start=1)
        if scene_dir.name not in completed_scene_id_set
    ]
    if resume:
        logger.info(
            "Resume state: %d completed scene(s), %d pending scene(s), %d total target scene(s)",
            len(completed_scene_ids),
            len(pending_scene_entries),
            total_scenes,
        )
        if not pending_scene_entries:
            logger.info(
                "All target scenes already have cached raw scene questions; rebuilding final outputs from cache only"
            )

    if rebuild_benchmark:
        logger.info(
            "Cache-only benchmark rebuild: using %d completed scene(s) and skipping %d pending scene(s)",
            len(completed_scene_ids),
            len(pending_scene_entries),
        )
        return _rebuild_pipeline_outputs(
            data_root=data_root,
            output_dir=output_dir,
            questions_dir=questions_dir,
            frame_debug_dir=frame_debug_dir,
            raw_questions_dir=raw_questions_dir,
            scene_ids=completed_scene_ids,
            referability_cache=referability_cache,
            write_frame_debug=write_frame_debug,
            run_question_dinox_audit=run_question_dinox_audit,
            run_question_presence_review=run_question_presence_review,
            vlm_url=vlm_url,
            vlm_model=vlm_model,
            question_presence_review_workers=question_presence_review_workers,
            scene_type_cap=scene_type_cap,
            frame_type_cap=frame_type_cap,
            frame_type_object_cap=frame_type_object_cap,
            dataset=dataset,
            scannetpp_sensor=scannetpp_sensor,
            scannetpp_frame_root=scannetpp_frame_root,
        )

    def _format_observability_value(value: object) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, float):
            return f"{value:.2f}"
        return str(value)

    def _log_frame_event(
        level_fn,
        event_name: str,
        frame_ctx: dict[str, object],
        **fields: object,
    ) -> None:
        message_parts = [
            f"scene={frame_ctx['scene_id']}",
            f"frame={frame_ctx['image_name']}",
            f"index={frame_ctx['frame_index']}/{frame_ctx['frame_total']}",
        ]
        for key, value in fields.items():
            if value is None:
                continue
            message_parts.append(f"{key}={_format_observability_value(value)}")
        level_fn("%s: %s", event_name, " ".join(message_parts))

    @contextmanager
    def _timed_frame_phase(
        frame_ctx: dict[str, object],
        phase_name: str,
    ) -> Iterator[None]:
        phase_started_at = time.perf_counter()
        _log_frame_event(
            logger.info,
            "frame phase start",
            frame_ctx,
            phase=phase_name,
        )
        try:
            yield
        finally:
            elapsed_seconds = time.perf_counter() - phase_started_at
            _log_frame_event(
                logger.info,
                "frame phase done",
                frame_ctx,
                phase=phase_name,
                elapsed=f"{elapsed_seconds:.2f}s",
            )
            if (
                slow_phase_warn_seconds > 0
                and elapsed_seconds >= slow_phase_warn_seconds
            ):
                _log_frame_event(
                    logger.warning,
                    "slow frame phase",
                    frame_ctx,
                    phase=phase_name,
                    elapsed=f"{elapsed_seconds:.2f}s",
                )

    def _log_frame_done(
        frame_ctx: dict[str, object],
        frame_started_at: float,
        *,
        status: str,
        skip_reason: str | None,
        raw_generated_count: int,
        kept_count: int,
    ) -> None:
        elapsed_seconds = time.perf_counter() - frame_started_at
        slow_frame = (
            slow_frame_warn_seconds > 0
            and elapsed_seconds >= slow_frame_warn_seconds
        )
        _log_frame_event(
            logger.info,
            "frame done",
            frame_ctx,
            status=status,
            skip_reason=skip_reason,
            elapsed=f"{elapsed_seconds:.2f}s",
            raw_generated=raw_generated_count,
            kept=kept_count,
            slow_frame=slow_frame,
        )
        if slow_frame:
            _log_frame_event(
                logger.warning,
                "slow frame",
                frame_ctx,
                status=status,
                skip_reason=skip_reason,
                elapsed=f"{elapsed_seconds:.2f}s",
                raw_generated=raw_generated_count,
                kept=kept_count,
            )

    def _persist_completed_scene(
        scene_id: str,
        *,
        scene_questions: list[dict],
        frame_count: int,
        pipeline_outcome: str,
    ) -> None:
        raw_question_path = raw_questions_dir / f"{scene_id}.json"
        scene_questions = _deduplicate_scene_questions(scene_questions)
        scene_questions = _apply_scene_type_cap(
            scene_questions,
            scene_type_cap=scene_type_cap,
            frame_type_cap=frame_type_cap,
            frame_type_object_cap=frame_type_object_cap,
        )
        _validate_strict_object_centric_questions(
            scene_questions,
            source=f"generated scene {scene_id}",
        )
        _write_json_file(raw_question_path, scene_questions)
        _mark_pipeline_scene_completed(
            scene_status_doc,
            scene_id=scene_id,
            raw_question_count=len(scene_questions),
            frame_count=frame_count,
            pipeline_outcome=pipeline_outcome,
        )
        _write_json_file(scene_status_path, scene_status_doc)
        logger.info(
            "Scene %s completed with outcome=%s frame_count=%d raw_questions=%d",
            scene_id,
            pipeline_outcome,
            frame_count,
            len(scene_questions),
        )

    for scene_index, scene_dir in pending_scene_entries:
        scene_id = scene_dir.name
        scene_question_type_counts: Counter[str] = Counter()
        scene_candidate_type_counts: Counter[str] = Counter()
        scene_pair_counts: Counter[tuple[str, str, str]] = Counter()
        logger.info(
            "=== Processing scene %s (%d/%d) ===",
            scene_id,
            scene_index,
            total_scenes,
        )

        scene_questions: list[dict] = []
        scene_frame_debug_entries: list[dict[str, object]] = []
        preloaded_geometry = None
        needs_mesh_resources, needs_instance_mesh_data = _scene_resource_requirements(
            single_frame_requested_types=single_frame_requested_types,
            cross_frame_requested_types=cross_frame_requested_types,
            occlusion_backend=occlusion_backend,
        )
        if needs_mesh_resources or needs_instance_mesh_data:
            try:
                preloaded_geometry = _load_scene_geometry(scene_dir)
            except Exception as e:
                logger.warning("Scene geometry preload failed for %s: %s", scene_id, e)

        parse_kwargs = {"preloaded_geometry": preloaded_geometry}
        if dataset == "scannetpp":
            parse_kwargs["dataset"] = "scannetpp"
        scene = parse_scene(scene_dir, **parse_kwargs)
        if scene is None:
            _persist_completed_scene(
                scene_id,
                scene_questions=scene_questions,
                frame_count=0,
                pipeline_outcome="scene_parse_skipped",
            )
            continue

        enrich_scene_with_attachment(scene)
        attachment_graph = get_scene_attachment_graph(scene, scene_id=scene_id)
        attached_by = get_scene_attached_by(scene, scene_id=scene_id)
        support_chain_graph = get_scene_support_chain_graph(scene, scene_id=scene_id)
        support_chain_by = get_scene_support_chain_by(scene, scene_id=scene_id)
        manual_attachment_graph = _manual_attachment_graph_for_scene(
            referability_cache,
            scene_id,
        )
        if manual_attachment_graph is not None:
            attachment_graph = manual_attachment_graph
            support_chain_graph = manual_attachment_graph
            attached_by = {
                int(child_id): int(parent_id)
                for parent_id, child_ids in manual_attachment_graph.items()
                for child_id in child_ids
            }
            support_chain_by = dict(attached_by)
            scene["attachment_graph"] = {
                str(parent_id): list(child_ids)
                for parent_id, child_ids in manual_attachment_graph.items()
            }
            scene["support_chain_graph"] = dict(scene["attachment_graph"])
            scene["attached_by"] = {
                str(child_id): parent_id
                for child_id, parent_id in attached_by.items()
            }
            scene["support_chain_by"] = dict(scene["attached_by"])
            manual_object_map = {
                int(obj["id"]): obj
                for obj in scene.get("objects", [])
                if isinstance(obj, dict) and obj.get("id") is not None
            }
            scene["attachment_edges"] = [
                {
                    "parent_id": int(parent_id),
                    "parent_label": str(manual_object_map.get(int(parent_id), {}).get("label", "")),
                    "child_id": int(child_id),
                    "child_label": str(manual_object_map.get(int(child_id), {}).get("label", "")),
                    "type": "human_salvage",
                    "confidence": 1.0,
                    "source": "two_hop_attachment_salvage",
                }
                for parent_id, child_ids in manual_attachment_graph.items()
                for child_id in child_ids
            ]
        scene_attachment_rows = _build_scene_attachment_rows(scene)
        objects_by_id = {int(obj["id"]): obj for obj in scene["objects"]}

        if not has_nontrivial_attachment(attachment_graph):
            logger.info("Scene %s has no support relations; skipping", scene_id)
            _persist_completed_scene(
                scene_id,
                scene_questions=scene_questions,
                frame_count=0,
                pipeline_outcome="no_support_relations",
            )
            continue
        if l3_attachment_chain_only and not _support_chain_graph_has_two_hop_chain(support_chain_graph):
            logger.info("Scene %s has no two-hop attachment chain; skipping", scene_id)
            _persist_completed_scene(
                scene_id,
                scene_questions=scene_questions,
                frame_count=0,
                pipeline_outcome="no_two_hop_attachment_chain",
            )
            continue
        if l3_attachment_move_only and not _attachment_graph_has_two_hop_chain(attachment_graph):
            logger.info("Scene %s has no two-hop attachment graph for attachment_move; skipping", scene_id)
            _persist_completed_scene(
                scene_id,
                scene_questions=scene_questions,
                frame_count=0,
                pipeline_outcome="no_two_hop_attachment_move_chain",
            )
            continue

        scene["depth_source"] = "none" if dataset == "scannetpp" else "sensor"
        _write_json_file(meta_dir / f"{scene_id}.json", scene)

        ds = None
        if dataset == "scannetpp":
            from src.datasets import make_data_source
            ds = make_data_source(dataset, scene_dir, sensor=scannetpp_sensor,
                                  frame_root=scannetpp_frame_root,
                                  depth_cache_size=scannetpp_depth_cache_size)

        scene_frames = _get_referability_scene_frames(referability_cache, scene_id)
        frames = _frames_from_referability_cache(scene_frames)
        if attachment_only_l2_mode:
            attachment_frames: list[dict[str, object]] = []
            skipped_attachment_frames: list[dict[str, object]] = []
            for frame in frames:
                image_name = str(frame.get("image_name", "")).strip()
                referability_entry = scene_frames.get(image_name)
                if _frame_has_attachment_pair(frame, referability_entry, attachment_graph):
                    attachment_frames.append(frame)
                else:
                    skipped_attachment_frames.append(frame)
            if write_frame_debug:
                for frame in skipped_attachment_frames:
                    image_name = str(frame.get("image_name", "")).strip()
                    selector_visible_ids = _normalize_object_ids(frame.get("visible_object_ids"))
                    frame_attachment_rows = _filter_frame_attachment_rows(
                        scene_attachment_rows,
                        set(selector_visible_ids),
                    )
                    scene_frame_debug_entries.append(
                        _build_frame_debug_entry(
                            image_name=image_name,
                            scene_objects=scene["objects"],
                            objects_by_id=objects_by_id,
                            selector_visible_ids=selector_visible_ids,
                            pipeline_visible_ids=selector_visible_ids,
                            occlusion_eligible_object_ids=[],
                            pipeline_referable_object_ids=[],
                            pipeline_attachment_referable_object_ids=_normalize_object_ids(
                                (scene_frames.get(image_name) or {}).get("attachment_referable_object_ids")
                            ),
                            referability_entry=scene_frames.get(image_name),
                            frame_attachment_rows=frame_attachment_rows,
                            pipeline_skip_reason="no_attachment_pair_for_attachment_only_l2",
                        )
                    )
            logger.info(
                "Attachment-only L2 mode kept %d/%d frame candidates for %s",
                len(attachment_frames),
                len(frames),
                scene_id,
            )
            frames = attachment_frames
        if l3_attachment_chain_only:
            l3_frames: list[dict[str, object]] = []
            skipped_l3_frames: list[dict[str, object]] = []
            for frame in frames:
                image_name = str(frame.get("image_name", "")).strip()
                referability_entry = scene_frames.get(image_name)
                if _frame_has_l3_attachment_chain(frame, referability_entry, support_chain_graph):
                    l3_frames.append(frame)
                else:
                    skipped_l3_frames.append(frame)
            if write_frame_debug:
                for frame in skipped_l3_frames:
                    image_name = str(frame.get("image_name", "")).strip()
                    selector_visible_ids = _normalize_object_ids(frame.get("visible_object_ids"))
                    frame_attachment_rows = _filter_frame_attachment_rows(
                        scene_attachment_rows,
                        set(selector_visible_ids),
                    )
                    scene_frame_debug_entries.append(
                        _build_frame_debug_entry(
                            image_name=image_name,
                            scene_objects=scene["objects"],
                            objects_by_id=objects_by_id,
                            selector_visible_ids=selector_visible_ids,
                            pipeline_visible_ids=selector_visible_ids,
                            occlusion_eligible_object_ids=[],
                            pipeline_referable_object_ids=[],
                            pipeline_attachment_referable_object_ids=_normalize_object_ids(
                                (scene_frames.get(image_name) or {}).get("attachment_referable_object_ids")
                            ),
                            referability_entry=scene_frames.get(image_name),
                            frame_attachment_rows=frame_attachment_rows,
                            pipeline_skip_reason="no_l3_attachment_chain_frame",
                        )
                    )
            logger.info(
                "L3 attachment-chain-only mode kept %d/%d frame candidates for %s",
                len(l3_frames),
                len(frames),
                scene_id,
            )
            frames = l3_frames
        if len(frames) > frame_limit:
            frames = frames[:frame_limit]
        if not frames:
            logger.info("No valid frames for scene %s after cache filtering; skipping", scene_id)
            _persist_completed_scene(
                scene_id,
                scene_questions=scene_questions,
                frame_count=0,
                pipeline_outcome="no_frame_candidates",
            )
            continue

        if ds is not None:
            axis_align = ds.load_axis_alignment()
            poses = ds.load_poses()
        else:
            axis_align = load_axis_alignment(scene_dir)
            poses = load_scannet_poses(scene_dir, axis_alignment=axis_align)
        ray_caster = None
        if needs_mesh_resources:
            if ds is not None:
                mesh_path = ds.mesh_path()
            else:
                mesh_path = scene_dir / f"{scene_id}_vh_clean.ply"
                if not mesh_path.exists():
                    mesh_path = scene_dir / f"{scene_id}_vh_clean_2.ply"
            if not mesh_path.exists():
                raise RuntimeError(
                    f"{occlusion_backend} backend requested for {scene_id}, "
                    f"but mesh not found at {mesh_path}"
                )
            if RayCaster is not None:
                try:
                    ray_caster = RayCaster.from_ply(str(mesh_path), axis_alignment=axis_align)
                except Exception as e:
                    raise RuntimeError(
                        f"{occlusion_backend} backend requested for {scene_id}, "
                        f"but ray caster initialization failed: {e}"
                    ) from e
            else:
                raise RuntimeError(
                    f"{occlusion_backend} backend requested for {scene_id}, "
                    "but mesh geometry or RayCaster is unavailable"
                )

        instance_mesh_data = None
        if needs_instance_mesh_data:
            try:
                instance_mesh_kwargs = {
                    "instance_ids": [int(o["id"]) for o in scene["objects"]],
                    "n_surface_samples": 512,
                    "preloaded_geometry": preloaded_geometry,
                }
                if dataset == "scannetpp":
                    instance_mesh_kwargs["dataset"] = "scannetpp"
                instance_mesh_data = load_instance_mesh_data(scene_dir, **instance_mesh_kwargs)
            except Exception as e:
                if needs_mesh_resources:
                    raise RuntimeError(
                        f"{occlusion_backend} backend requested for {scene_id}, "
                        f"but instance mesh data could not be loaded: {e}"
                    ) from e
                logger.warning(
                    "Instance mesh data load failed for %s; distance GT will fall back to AABB closest points: %s",
                    scene_id,
                    e,
                )

        # Release preloaded geometry — vertices/faces are now owned by
        # instance_mesh_data and ray_caster; keeping this around wastes memory.
        del preloaded_geometry

        # Share vertex/face arrays between instance_mesh_data and ray_caster
        # to avoid keeping two large copies of the same mesh in memory.
        if (
            instance_mesh_data is not None
            and ray_caster is not None
            and hasattr(ray_caster, "mesh")
        ):
            try:
                rc_verts = np.asarray(ray_caster.mesh.vertices, dtype=np.float64)
                rc_faces = np.asarray(ray_caster.mesh.faces, dtype=np.int64)
            except (TypeError, ValueError):
                rc_verts = None
                rc_faces = None
            if rc_verts is not None and rc_faces is not None:
                if (
                    rc_verts.shape == instance_mesh_data.vertices.shape
                    and rc_faces.shape == instance_mesh_data.faces.shape
                ):
                    instance_mesh_data.vertices = rc_verts
                    instance_mesh_data.faces = rc_faces

        depth_intrinsics = None
        if use_occlusion:
            try:
                if ds is not None:
                    depth_intrinsics = ds.load_depth_intrinsics()
                else:
                    depth_intrinsics = load_scannet_depth_intrinsics(scene_dir)
            except Exception as e:
                logger.warning("Depth intrinsics load failed for %s: %s", scene_id, e)

        try:
            if ds is not None:
                color_intrinsics = ds.load_intrinsics()
            else:
                color_intrinsics = load_scannet_intrinsics(scene_dir)
        except Exception as e:
            logger.warning("Color intrinsics load failed for %s: %s", scene_id, e)
            color_intrinsics = None

        for frame_index, frame in enumerate(frames, start=1):
            if not single_frame_requested_types:
                logger.info(
                    "Skipping per-frame single-image generation for %s; only cross-frame types were requested",
                    scene_id,
                )
                break
            pending_image_name = str(frame.get("image_name", "")).strip()
            pending_referability_entry = scene_frames.get(pending_image_name)
            pending_manual_role_sets = _manual_attachment_role_records_for_frame(
                pending_referability_entry
            )
            # Stop scanning frames once every requested type has exhausted its
            # generation budget. Final retained questions have no type-total cap.
            if (
                not pending_manual_role_sets
                and (
                    scene_type_cap > 0
                    or l2_candidate_budget > 0
                    or l3_candidate_budget > 0
                )
            ):
                if _all_candidate_type_budgets_exhausted(
                    scene_candidate_type_counts,
                    canonical_types=single_frame_scene_question_types,
                    l1_candidate_budget=scene_type_cap,
                    l2_candidate_budget=l2_candidate_budget,
                    l3_candidate_budget=l3_candidate_budget,
                ):
                    logger.info(
                        "Scene %s exhausted all active single-frame candidate budgets after %d frame(s); stopping early",
                        scene_id,
                        frame_index - 1,
                    )
                    break
            image_name = frame["image_name"]
            selector_visible_ids = _normalize_object_ids(frame.get("visible_object_ids"))
            visible_ids = list(selector_visible_ids)
            visible_id_set = {int(obj_id) for obj_id in visible_ids}
            referability_entry = _get_referability_entry(
                referability_cache,
                scene_id,
                image_name,
            )
            cache_referable_count = (
                len(referability_entry.get("referable_object_ids", []) or [])
                if referability_entry is not None
                else None
            )
            frame_ctx = {
                "scene_id": scene_id,
                "image_name": image_name,
                "frame_index": frame_index,
                "frame_total": len(frames),
            }
            frame_started_at = time.perf_counter()
            frame_status = "done"
            frame_skip_reason: str | None = None
            frame_raw_generated_count = 0
            frame_kept_count = 0
            _log_frame_event(
                logger.info,
                "frame start",
                frame_ctx,
                visible=len(selector_visible_ids),
                cache_referable=cache_referable_count,
            )

            referable_ids = None
            attachment_referable_ids = None
            frame_attachment_pairs: list[tuple[int, int]] = []
            attachment_object_surface_text_by_id: dict[int, str] = {}
            attachment_priority_pairs: list[tuple[int, int]] = []
            label_statuses = None
            label_counts = None
            out_of_frame_not_visible_labels: list[str] = []
            out_of_frame_label_to_object_ids: dict[str, list[int]] | None = None
            referable_occlusion_veto: dict[str, object] = {
                "raw_object_ids": [],
                "filtered_object_ids": [],
                "low_visible_object_ids": [],
                "not_visible_object_ids": [],
                "skipped_object_ids": [],
                "audit_by_object_id": {},
            }
            mention_in_frame_ratio_by_obj_id: dict[int, float] = {}
            projected_area_by_obj_id: dict[int, float] = {}
            occlusion_eligible_ids: list[int] = []
            camera_pose = None
            depth_image = None

            try:
                with _timed_frame_phase(frame_ctx, "pose_depth_load"):
                    if image_name in poses:
                        camera_pose = poses[image_name]
                        if use_occlusion and depth_intrinsics is not None:
                            if ds is not None:
                                depth_path = ds.depth_image_path(image_name)
                            else:
                                frame_id, _ = os.path.splitext(image_name)
                                depth_path = scene_dir / "depth" / f"{frame_id}.png"
                            if depth_path is not None and depth_path.exists():
                                try:
                                    depth_image = load_depth_image(depth_path)
                                except Exception as e:
                                    logger.warning("Depth load failed for %s/%s: %s", scene_id, image_name, e)

                if camera_pose is None:
                    if write_frame_debug:
                        with _timed_frame_phase(frame_ctx, "frame_debug_assembly"):
                            frame_attachment_rows = _filter_frame_attachment_rows(
                                scene_attachment_rows,
                                set(selector_visible_ids),
                            )
                            scene_frame_debug_entries.append(
                                _build_frame_debug_entry(
                                    image_name=image_name,
                                    scene_objects=scene["objects"],
                                    objects_by_id=objects_by_id,
                                    selector_visible_ids=selector_visible_ids,
                                    pipeline_visible_ids=[],
                                    occlusion_eligible_object_ids=[],
                                    pipeline_referable_object_ids=[],
                                    pipeline_attachment_referable_object_ids=[],
                                    referability_entry=referability_entry,
                                    frame_attachment_rows=frame_attachment_rows,
                                    pipeline_skip_reason="missing_pose",
                                )
                    )
                    frame_status = "skipped"
                    frame_skip_reason = "missing_pose"
                    continue

                with _timed_frame_phase(frame_ctx, "referability_entry_normalization"):
                    if referability_entry is not None:
                        label_statuses = _normalize_label_statuses(referability_entry.get("label_statuses"))
                        label_counts = _normalize_label_counts(referability_entry.get("label_counts"))
                        out_of_frame_not_visible_labels = _normalize_label_list(
                            referability_entry.get("out_of_frame_not_visible_labels")
                        )
                        out_of_frame_label_to_object_ids = _shared_normalize_label_to_object_ids(
                            referability_entry.get("out_of_frame_label_to_object_ids")
                        )
                        raw_referable_ids = [
                            int(obj_id)
                            for obj_id in referability_entry.get("referable_object_ids", [])
                            if int(obj_id) in visible_id_set
                        ]
                        raw_attachment_referable_ids = referability_entry.get(
                            "attachment_referable_object_ids"
                        )
                        attachment_object_surface_text_by_id = (
                            _attachment_surface_text_by_object_id(referability_entry)
                        )
                        attachment_priority_pairs = _attachment_human_review_priority_pairs(
                            referability_entry.get("attachment_human_review_cards")
                        )
                        if raw_attachment_referable_ids is None:
                            raw_attachment_referable_ids = _derive_final_referability_fields(
                                referability_entry
                            ).get("attachment_referable_object_ids", [])
                        attachment_referable_ids = [
                            int(obj_id)
                            for obj_id in (raw_attachment_referable_ids or [])
                            if int(obj_id) in visible_id_set
                        ]
                        frame_attachment_pairs = _frame_attachment_referable_pairs(
                            referability_entry=referability_entry,
                            attachment_graph=attachment_graph,
                            attachment_referable_ids=attachment_referable_ids,
                            visible_object_ids=visible_ids,
                        )
                    else:
                        raw_referable_ids = []
                        attachment_object_surface_text_by_id = {}
                        attachment_priority_pairs = []

                with _timed_frame_phase(frame_ctx, "in_frame_ratio_projected_area_map_build"):
                    mention_in_frame_ratio_by_obj_id = _build_visible_object_in_frame_ratio_map(
                        visible_object_ids=visible_ids,
                        referability_entry=referability_entry,
                        scene_objects=scene["objects"],
                        camera_pose=camera_pose,
                        color_intrinsics=color_intrinsics,
                    )
                    projected_area_by_obj_id = _build_visible_object_projected_area_map(
                        visible_object_ids=visible_ids,
                        referability_entry=referability_entry,
                        scene_objects=scene["objects"],
                        camera_pose=camera_pose,
                        color_intrinsics=color_intrinsics,
                    )
                    occlusion_eligible_ids = _build_occlusion_eligible_object_ids(
                        visible_object_ids=visible_ids,
                        mention_in_frame_ratio_by_obj_id=mention_in_frame_ratio_by_obj_id,
                    )

                with _timed_frame_phase(frame_ctx, "referable_occlusion_veto"):
                    if referability_entry is not None:
                        if attachment_chain_fast_path:
                            referable_occlusion_veto = {
                                "raw_object_ids": list(raw_referable_ids),
                                "filtered_object_ids": list(raw_referable_ids),
                                "low_visible_object_ids": [],
                                "not_visible_object_ids": [],
                                "skipped_object_ids": list(raw_referable_ids),
                                "audit_by_object_id": {},
                                "mode": "trusted_referability_cache_for_attachment_chain",
                            }
                        else:
                            referable_occlusion_veto = _filter_referable_object_ids_with_occlusion_veto(
                                scene_id=scene_id,
                                image_name=image_name,
                                referable_object_ids=raw_referable_ids,
                                objects_by_id=objects_by_id,
                                projected_area_by_obj_id=projected_area_by_obj_id,
                                camera_pose=camera_pose,
                                color_intrinsics=color_intrinsics,
                                ray_caster=ray_caster,
                                instance_mesh_data=instance_mesh_data,
                            )
                        referable_ids = list(referable_occlusion_veto["filtered_object_ids"])

                if (
                    referability_entry is not None
                    and not referable_ids
                    and not frame_attachment_pairs
                    and not _has_l1_visibility_candidates(
                        label_statuses,
                        out_of_frame_not_visible_labels,
                    )
                ):
                    if write_frame_debug:
                        with _timed_frame_phase(frame_ctx, "frame_debug_assembly"):
                            frame_attachment_rows = _filter_frame_attachment_rows(
                                scene_attachment_rows,
                                set(selector_visible_ids) | set(int(obj_id) for obj_id in visible_ids),
                            )
                            scene_frame_debug_entries.append(
                                _build_frame_debug_entry(
                                    image_name=image_name,
                                    scene_objects=scene["objects"],
                                    objects_by_id=objects_by_id,
                                    selector_visible_ids=selector_visible_ids,
                                    pipeline_visible_ids=list(visible_ids),
                                    occlusion_eligible_object_ids=occlusion_eligible_ids,
                                    pipeline_referable_object_ids=referable_ids,
                                    pipeline_attachment_referable_object_ids=attachment_referable_ids,
                                    referability_entry=referability_entry,
                                    frame_attachment_rows=frame_attachment_rows,
                                    referable_occlusion_veto=referable_occlusion_veto,
                                    pipeline_skip_reason="no_referable_objects_or_l1_candidates",
                                )
                            )
                    logger.debug(
                        "Frame %s/%s has no referable objects or L1 visibility candidates",
                        scene_id,
                        image_name,
                    )
                    frame_status = "skipped"
                    frame_skip_reason = "no_referable_objects_or_l1_candidates"
                    continue

                if attachment_only_l2_mode and not frame_attachment_pairs:
                    if write_frame_debug:
                        with _timed_frame_phase(frame_ctx, "frame_debug_assembly"):
                            frame_attachment_rows = _filter_frame_attachment_rows(
                                scene_attachment_rows,
                                set(selector_visible_ids) | set(int(obj_id) for obj_id in visible_ids),
                            )
                            scene_frame_debug_entries.append(
                                _build_frame_debug_entry(
                                    image_name=image_name,
                                    scene_objects=scene["objects"],
                                    objects_by_id=objects_by_id,
                                    selector_visible_ids=selector_visible_ids,
                                    pipeline_visible_ids=list(visible_ids),
                                    occlusion_eligible_object_ids=occlusion_eligible_ids,
                                    pipeline_referable_object_ids=referable_ids,
                                    pipeline_attachment_referable_object_ids=attachment_referable_ids,
                                    referability_entry=referability_entry,
                                    frame_attachment_rows=frame_attachment_rows,
                                    referable_occlusion_veto=referable_occlusion_veto,
                                    pipeline_skip_reason="no_attachment_pair_for_attachment_only_l2",
                                )
                            )
                    frame_status = "skipped"
                    frame_skip_reason = "no_attachment_pair_for_attachment_only_l2"
                    continue

                with _timed_frame_phase(frame_ctx, "generate_all_questions"):
                    try:
                        if not single_frame_requested_types:
                            questions = []
                            question_type_budgets = None
                        else:
                            question_type_budgets = _remaining_candidate_type_budgets(
                                scene_candidate_type_counts,
                                l1_candidate_budget=scene_type_cap,
                                l2_candidate_budget=l2_candidate_budget,
                                l3_candidate_budget=l3_candidate_budget,
                                allowed_types=single_frame_scene_question_types,
                            )
                            if pending_manual_role_sets and question_type_budgets is not None:
                                question_type_budgets.pop("attachment_chain", None)

                        def _pair_budget_remaining(
                            canonical_type: str, id_a: int, id_b: int,
                        ) -> bool:
                            _ = canonical_type, id_a, id_b
                            return True

                        if single_frame_requested_types:
                            questions = _call_generate_all_questions_compat(
                            objects=scene["objects"],
                            attachment_graph=attachment_graph,
                            attached_by=attached_by,
                            support_chain_graph=support_chain_graph,
                            support_chain_by=support_chain_by,
                            camera_pose=camera_pose,
                            color_intrinsics=color_intrinsics,
                            depth_image=depth_image,
                            depth_intrinsics=depth_intrinsics,
                            occlusion_backend=occlusion_backend,
                            ray_caster=ray_caster,
                            instance_mesh_data=instance_mesh_data,
                            visible_object_ids=visible_ids,
                            referable_object_ids=referable_ids,
                            attachment_referable_object_ids=attachment_referable_ids,
                            attachment_referable_pairs=frame_attachment_pairs,
                            attachment_chain_role_override=_manual_attachment_roles_for_frame(
                                referability_entry
                            ),
                            attachment_chain_role_overrides=(
                                _manual_attachment_role_records_for_frame(referability_entry)
                                or None
                            ),
                            attachment_object_surface_text_by_id=attachment_object_surface_text_by_id,
                            attachment_priority_pairs=attachment_priority_pairs,
                            occlusion_eligible_object_ids=occlusion_eligible_ids,
                            mention_in_frame_ratio_by_obj_id=mention_in_frame_ratio_by_obj_id,
                            label_statuses=label_statuses,
                            label_counts=label_counts,
                            label_to_object_ids=(referability_entry or {}).get("label_to_object_ids"),
                            out_of_frame_not_visible_labels=out_of_frame_not_visible_labels,
                            out_of_frame_label_to_object_ids=out_of_frame_label_to_object_ids,
                            room_bounds=scene.get("room_bounds"),
                            wall_objects=scene.get("wall_objects"),
                            attachment_edges=scene.get("attachment_edges", []),
                            generator_progress_log_seconds=generator_progress_log_seconds,
                            slow_generator_warn_seconds=slow_generator_warn_seconds,
                            only_question_types=single_frame_requested_types,
                            question_type_budgets=question_type_budgets,
                            max_occlusion_objects=max_occlusion_objects,
                            max_move_sources=max_move_sources,
                            pair_budget_remaining=_pair_budget_remaining,
                            )
                    except Exception:
                        logger.exception(
                            "Question generation failed for %s/%s (visible=%d referable=%d attachment_referable=%d occlusion_eligible=%d)",
                            scene_id,
                            image_name,
                            len(visible_ids or []),
                            len(referable_ids or []),
                            len(attachment_referable_ids or []),
                            len(occlusion_eligible_ids or []),
                        )
                        raise
                questions = _filter_vertical_object_rotate_questions(
                    questions,
                    scene_objects=scene["objects"],
                    attachment_graph=attachment_graph,
                )
                questions = _take_questions_within_candidate_budgets(
                    questions,
                    scene_candidate_type_counts,
                    l1_candidate_budget=scene_type_cap,
                    l2_candidate_budget=l2_candidate_budget,
                    l3_candidate_budget=l3_candidate_budget,
                )
                frame_raw_generated_count = len(questions)

                for q in questions:
                    q["scene_id"] = scene_id
                    q["image_name"] = image_name

                with _timed_frame_phase(frame_ctx, "referability_invariant_check"):
                    kept_questions, audited_questions = _apply_question_referability_filter(
                        questions,
                        objects_by_id=objects_by_id,
                        referability_entry=referability_entry,
                        frame_referable_ids=referable_ids or [],
                        attachment_frame_referable_ids=attachment_referable_ids or [],
                        attachment_frame_referable_pairs=frame_attachment_pairs,
                    )
                    frame_question_type_counts: Counter[tuple[str, str, str]] = Counter()
                    frame_question_type_object_counts: Counter[tuple[str, str, str, str]] = Counter()
                    kept_questions = _apply_scene_type_cap(
                        kept_questions,
                        scene_type_cap=scene_type_cap,
                        frame_type_cap=frame_type_cap,
                        frame_type_object_cap=frame_type_object_cap,
                        type_counts=scene_question_type_counts,
                        frame_type_counts=frame_question_type_counts,
                        frame_type_object_counts=frame_question_type_object_counts,
                        pair_counts=scene_pair_counts,
                    )
                frame_kept_count = len(kept_questions)
                scene_questions.extend(kept_questions)

                if write_frame_debug:
                    with _timed_frame_phase(frame_ctx, "frame_debug_assembly"):
                        frame_attachment_rows = _filter_frame_attachment_rows(
                            scene_attachment_rows,
                            set(selector_visible_ids) | set(int(obj_id) for obj_id in visible_ids),
                        )
                        scene_frame_debug_entries.append(
                            _build_frame_debug_entry(
                                image_name=image_name,
                                scene_objects=scene["objects"],
                                objects_by_id=objects_by_id,
                                selector_visible_ids=selector_visible_ids,
                                pipeline_visible_ids=list(visible_ids),
                                occlusion_eligible_object_ids=occlusion_eligible_ids,
                                pipeline_referable_object_ids=referable_ids,
                                pipeline_attachment_referable_object_ids=attachment_referable_ids,
                                pipeline_attachment_referable_pairs=frame_attachment_pairs,
                                referability_entry=referability_entry,
                                frame_attachment_rows=frame_attachment_rows,
                                referable_occlusion_veto=referable_occlusion_veto,
                                generated_questions=audited_questions,
                            )
                        )
            except Exception:
                frame_status = "error"
                raise
            finally:
                _log_frame_done(
                    frame_ctx,
                    frame_started_at,
                    status=frame_status,
                    skip_reason=frame_skip_reason,
                    raw_generated_count=frame_raw_generated_count,
                    kept_count=frame_kept_count,
                )

        if cross_frame_requested_types:
            scene_motion_cache = SceneMotionCache()
            flash_contexts = _build_reasoning_frame_contexts(
                frames=frames,
                scene_frames=scene_frames,
                poses=poses,
                scene_objects=scene["objects"],
                color_intrinsics=color_intrinsics,
            )
            cross_checkpoint_dir = _cross_frame_scene_cache_dir(output_dir, scene_id)
            cross_checkpoint_manifest_path = cross_checkpoint_dir / "manifest.json"
            cross_checkpoint_pre_path = cross_checkpoint_dir / "pre_cross.json"
            cross_checkpoint_signature = {
                "version": CROSS_FRAME_CHECKPOINT_VERSION,
                "object_move_semantics_version": L2_OBJECT_MOVE_SEMANTICS_VERSION,
                "object_move_object_centric_semantics": (
                    OBJECT_MOVE_OBJECT_CENTRIC_SEMANTICS_PROFILE
                ),
                "scene_id": scene_id,
                "frame_names": [context.image_name for context in flash_contexts],
                "question_types": list(cross_frame_requested_types),
                "auxiliary_route_method": auxiliary_route_method,
                "depth_route_camera_motion_hard_limits_enabled": (
                    DEPTH_ROUTE_ENFORCE_CAMERA_MOTION_HARD_LIMITS
                    if auxiliary_route_method
                    == AUXILIARY_ROUTE_METHOD_DEPTH_CORRIDOR_GEOMETRIC
                    else None
                ),
                "auxiliary_max_pose_candidates": auxiliary_max_pose_candidates,
                "scannetpp_depth_cache_size": scannetpp_depth_cache_size,
                "attachment_reference_cluster_radius_m": (
                    attachment_reference_cluster_radius_m
                ),
                "l1_candidate_budget": scene_type_cap,
                "l2_candidate_budget": l2_candidate_budget,
                "l3_candidate_budget": l3_candidate_budget,
                "max_move_sources": max_move_sources,
                "occlusion_max_references_per_query": occlusion_max_references_per_query,
                "occlusion_max_combinations_per_scene": occlusion_max_combinations_per_scene,
            }
            cross_checkpoint_manifest: dict[str, object] | None = None
            restored_cross_candidates: list[dict[str, object]] = []
            restored_cross_candidate_type_counts: Counter = Counter()
            restored_pair_stage_counts: Counter = Counter()
            restored_generated_counts: Counter = Counter()
            restored_distance_annotation_stats: Counter = Counter()
            restored_occlusion_combinations_attempted = 0
            restored_occlusion_raycast_states = 0
            restored_rng_state: object = None
            if resume and cross_checkpoint_manifest_path.is_file():
                try:
                    loaded_manifest = json.loads(
                        cross_checkpoint_manifest_path.read_text(encoding="utf-8")
                    )
                    if (
                        not isinstance(loaded_manifest, dict)
                        or loaded_manifest.get("signature") != cross_checkpoint_signature
                    ):
                        raise ValueError("checkpoint signature mismatch")
                    pre_cross_payload = json.loads(
                        cross_checkpoint_pre_path.read_text(encoding="utf-8")
                    )
                    if not isinstance(pre_cross_payload, dict):
                        raise ValueError("pre-cross checkpoint must be an object")
                    restored_scene_questions = pre_cross_payload.get("scene_questions", [])
                    restored_frame_debug_entries = pre_cross_payload.get(
                        "scene_frame_debug_entries",
                        [],
                    )
                    if not isinstance(restored_scene_questions, list):
                        raise ValueError("pre-cross scene_questions must be a list")
                    if not isinstance(restored_frame_debug_entries, list):
                        raise ValueError("pre-cross scene_frame_debug_entries must be a list")
                    completed_names = loaded_manifest.get("completed_deferred_frames", [])
                    if not isinstance(completed_names, list) or not all(
                        isinstance(value, str) for value in completed_names
                    ):
                        raise ValueError("completed_deferred_frames must be a string list")
                    completed_name_set = set(completed_names)
                    expected_completed_names = [
                        context.image_name
                        for context in flash_contexts
                        if context.image_name in completed_name_set
                    ]
                    if (
                        len(completed_name_set) != len(completed_names)
                        or completed_names != expected_completed_names
                    ):
                        raise ValueError(
                            "completed deferred frames are duplicated, unknown, or out of order"
                        )
                    occlusion_completed = loaded_manifest.get("occlusion_completed", False)
                    if not isinstance(occlusion_completed, bool):
                        raise ValueError("occlusion_completed must be boolean")
                    if completed_names and not occlusion_completed:
                        raise ValueError(
                            "deferred frames cannot be complete before occlusion phase"
                        )
                    restored_candidate_questions: list[dict[str, object]] = []
                    if occlusion_completed:
                        occlusion_checkpoint_path = cross_checkpoint_dir / "occlusion.json"
                        occlusion_payload = json.loads(
                            occlusion_checkpoint_path.read_text(encoding="utf-8")
                        )
                        if not isinstance(occlusion_payload, dict):
                            raise ValueError("occlusion checkpoint must be an object")
                        occlusion_questions = occlusion_payload.get("questions", [])
                        if not isinstance(occlusion_questions, list):
                            raise ValueError("occlusion checkpoint questions must be a list")
                        restored_candidate_questions.extend(occlusion_questions)
                        for deferred_index, deferred_name in enumerate(
                            completed_names,
                            start=1,
                        ):
                            shard_path = (
                                cross_checkpoint_dir
                                / "deferred"
                                / f"{deferred_index:04d}_{deferred_name}.json"
                            )
                            shard_payload = json.loads(
                                shard_path.read_text(encoding="utf-8")
                            )
                            if not isinstance(shard_payload, dict):
                                raise ValueError(
                                    f"deferred checkpoint must be an object: {shard_path}"
                                )
                            if shard_payload.get("frame_1") != deferred_name:
                                raise ValueError(
                                    f"deferred checkpoint frame mismatch: {shard_path}"
                                )
                            shard_questions = shard_payload.get("questions", [])
                            if not isinstance(shard_questions, list):
                                raise ValueError(
                                    f"deferred checkpoint questions must be a list: {shard_path}"
                                )
                            restored_candidate_questions.extend(shard_questions)
                        validated_cross_candidate_type_counts = _counter_from_checkpoint(
                            loaded_manifest.get("cross_candidate_type_counts")
                        )
                        validated_pair_stage_counts = _counter_from_checkpoint(
                            loaded_manifest.get("pair_stage_counts")
                        )
                        validated_generated_counts = _counter_from_checkpoint(
                            loaded_manifest.get("generated_counts")
                        )
                        validated_distance_annotation_stats = _counter_from_checkpoint(
                            loaded_manifest.get("distance_annotation_stats")
                        )
                        validated_occlusion_combinations_attempted = int(
                            loaded_manifest.get("occlusion_combinations_attempted", 0)
                        )
                        validated_occlusion_raycast_states = int(
                            loaded_manifest.get("occlusion_raycast_states", 0)
                        )
                        validated_rng_state = loaded_manifest.get("rng_state")
                        current_rng_state = _rng_checkpoint_payload()
                        try:
                            _restore_rng_checkpoint(validated_rng_state)
                        finally:
                            _restore_rng_checkpoint(current_rng_state)
                    else:
                        validated_cross_candidate_type_counts = Counter()
                        validated_pair_stage_counts = Counter()
                        validated_generated_counts = Counter()
                        validated_distance_annotation_stats = Counter()
                        validated_occlusion_combinations_attempted = 0
                        validated_occlusion_raycast_states = 0
                        validated_rng_state = None
                    restored_scene_question_type_counts = _counter_from_checkpoint(
                        pre_cross_payload.get("scene_question_type_counts")
                    )
                    restored_scene_candidate_type_counts = _counter_from_checkpoint(
                        pre_cross_payload.get("scene_candidate_type_counts")
                    )
                    restored_scene_pair_counts = _counter_from_checkpoint(
                        pre_cross_payload.get("scene_pair_counts")
                    )
                    scene_questions = list(restored_scene_questions)
                    scene_frame_debug_entries = list(restored_frame_debug_entries)
                    scene_question_type_counts = restored_scene_question_type_counts
                    scene_candidate_type_counts = restored_scene_candidate_type_counts
                    scene_pair_counts = restored_scene_pair_counts
                    restored_cross_candidates = restored_candidate_questions
                    restored_cross_candidate_type_counts = (
                        validated_cross_candidate_type_counts
                    )
                    restored_pair_stage_counts = validated_pair_stage_counts
                    restored_generated_counts = validated_generated_counts
                    restored_distance_annotation_stats = (
                        validated_distance_annotation_stats
                    )
                    restored_occlusion_combinations_attempted = (
                        validated_occlusion_combinations_attempted
                    )
                    restored_occlusion_raycast_states = validated_occlusion_raycast_states
                    restored_rng_state = validated_rng_state
                    cross_checkpoint_manifest = loaded_manifest
                    logger.info(
                        "Restored cross-frame checkpoint for scene %s (%d deferred frame(s))",
                        scene_id,
                        len(loaded_manifest.get("completed_deferred_frames", [])),
                    )
                except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
                    logger.warning(
                        "Ignoring invalid cross-frame checkpoint for scene %s: %s",
                        scene_id,
                        exc,
                    )
                    shutil.rmtree(cross_checkpoint_dir, ignore_errors=True)
            if cross_checkpoint_manifest is None:
                shutil.rmtree(cross_checkpoint_dir, ignore_errors=True)
                cross_checkpoint_manifest = {
                    "signature": cross_checkpoint_signature,
                    "occlusion_completed": False,
                    "completed_deferred_frames": [],
                }
                _write_json_file_atomic(
                    cross_checkpoint_pre_path,
                    {
                        "scene_questions": scene_questions,
                        "scene_frame_debug_entries": scene_frame_debug_entries,
                        "scene_question_type_counts": _counter_checkpoint_payload(
                            scene_question_type_counts
                        ),
                        "scene_candidate_type_counts": _counter_checkpoint_payload(
                            scene_candidate_type_counts
                        ),
                        "scene_pair_counts": _counter_checkpoint_payload(scene_pair_counts),
                    },
                )
                _write_json_file_atomic(
                    cross_checkpoint_manifest_path,
                    cross_checkpoint_manifest,
                )
            funnel: dict[str, object] = {
                "scene_id": scene_id,
                "raw_pose_frame_count": len(poses),
                "flash_frame_count": len(flash_contexts),
                "ordered_frame_pair_count": len(flash_contexts) * max(len(flash_contexts) - 1, 0),
                "requested_question_types": cross_frame_requested_types,
                "pair_stage_counts": Counter(),
                "question_type_generated_counts": Counter(),
                "question_type_kept_counts": Counter(),
                "pair_failures": [],
            }
            route_graph: VisualPoseGraph | None = None
            hybrid_router: HybridAuxiliaryRouter | None = None
            depth_visual_router: HybridAuxiliaryRouter | None = None
            depth_visual_redundancy: DepthVisualRedundancyEvaluator | None = None
            depth_route_geometry_cache = DepthRouteGeometryCache()
            depth_data_source = None
            hybrid_cache_path: Path | None = None
            hybrid_cache_hit = False
            depth_visual_cache_path: Path | None = None
            depth_visual_cache_hit = False
            if auxiliary_route_method == AUXILIARY_ROUTE_METHOD_DEPTH_CORRIDOR_GEOMETRIC:
                depth_data_source = ds if ds is not None else ScanNetDataSource(scene_dir)
                if dataset == "scannetpp" and (
                    getattr(depth_data_source, "sensor", None) != "iphone"
                    or not (scene_dir / "iphone" / "depth.bin").is_file()
                    or not (scene_dir / "iphone" / "pose_intrinsic_imu.json").is_file()
                ):
                    raise RuntimeError(
                        "depth_corridor_geometric requires ScanNet++ iPhone depth.bin "
                        "and pose_intrinsic_imu.json"
                    )
                if ds is not None:
                    image_path_for = ds.image_path
                else:
                    image_path_for = lambda name: scene_dir / "color" / name
                depth_visual_router = HybridAuxiliaryRouter(
                    poses=poses,
                    intrinsics=color_intrinsics,
                    image_path_for=image_path_for,
                )
                depth_visual_cache_path = (
                    auxiliary_graph_cache_dir / f"{scene_id}.depth_visual_prune.json"
                )
                depth_visual_cache_hit = depth_visual_router.load_cache(
                    depth_visual_cache_path
                )
                depth_visual_redundancy = DepthVisualRedundancyEvaluator(
                    poses=poses,
                    depth_frame_for=depth_data_source.load_depth_frame,
                    rgb_evidence_for=depth_visual_router.visual_continuity,
                )
            if auxiliary_route_method == AUXILIARY_ROUTE_METHOD_VISUAL_POSE_GRAPH:
                if ds is not None:
                    image_path_for = ds.image_path
                else:
                    image_path_for = lambda name: scene_dir / "color" / name
                route_graph = VisualPoseGraph(
                    poses=poses,
                    image_path_for=image_path_for,
                    flash_frame_names={context.image_name for context in flash_contexts},
                )
                route_graph_cache_path = auxiliary_graph_cache_dir / f"{scene_id}.json"
                route_graph_cache_hit = route_graph.load_cache(route_graph_cache_path)
                if not route_graph_cache_hit:
                    route_graph.build()
                    try:
                        route_graph.save_cache(route_graph_cache_path)
                    except OSError as exc:
                        logger.warning(
                            "Could not persist visual-pose graph cache for %s: %s",
                            scene_id,
                            exc,
                        )
                funnel["auxiliary_graph"] = {
                    **route_graph.diagnostics(),
                    "method": auxiliary_route_method,
                    "cache_hit": route_graph_cache_hit,
                    "cache_path": str(route_graph_cache_path),
                }
            elif auxiliary_route_method == AUXILIARY_ROUTE_METHOD_HYBRID_GEOMETRIC_VISUAL:
                if ds is not None:
                    image_path_for = ds.image_path
                else:
                    image_path_for = lambda name: scene_dir / "color" / name
                hybrid_router = HybridAuxiliaryRouter(
                    poses=poses,
                    intrinsics=color_intrinsics,
                    image_path_for=image_path_for,
                )
                hybrid_cache_path = auxiliary_graph_cache_dir / f"{scene_id}.hybrid.json"
                hybrid_cache_hit = hybrid_router.load_cache(hybrid_cache_path)
                funnel["auxiliary_graph"] = {
                    **hybrid_router.diagnostics(),
                    "method": auxiliary_route_method,
                    "cache_hit": hybrid_cache_hit,
                    "cache_path": str(hybrid_cache_path),
                    "note": "routes_are_computed_per_question_after_object_role_binding",
                }
            else:
                if depth_visual_router is not None:
                    funnel["auxiliary_graph"] = {
                        **depth_visual_router.diagnostics(),
                        "method": auxiliary_route_method,
                        "camera_motion_hard_limits_enabled": (
                            DEPTH_ROUTE_ENFORCE_CAMERA_MOTION_HARD_LIMITS
                        ),
                        "disabled_camera_motion_hard_limits": list(
                            DEPTH_ROUTE_DISABLED_CAMERA_MOTION_HARD_LIMITS
                        ),
                        "cache_hit": depth_visual_cache_hit,
                        "cache_path": str(depth_visual_cache_path),
                        "note": "visual evidence is used only for depth-route pruning",
                    }
                else:
                    funnel["auxiliary_graph"] = {
                        "method": auxiliary_route_method,
                        "pose_count": len(poses),
                        "cache_hit": False,
                        "note": "routes_are_computed_per_question_after_object_role_binding",
                    }
            cross_candidates: list[dict] = []
            occlusion_search_budget = OcclusionDirectedSearchBudget(
                max_combinations=occlusion_max_combinations_per_scene,
            )
            cross_candidate_type_counts = scene_candidate_type_counts

            def _cross_type_budget_available(canonical_type: str) -> bool:
                remaining = _candidate_type_budget_remaining(
                    cross_candidate_type_counts,
                    canonical_type,
                    l1_candidate_budget=scene_type_cap,
                    l2_candidate_budget=l2_candidate_budget,
                    l3_candidate_budget=l3_candidate_budget,
                )
                return remaining is None or remaining > 0

            answer_pair_distance_cache: dict[tuple[int, int], dict[str, Any]] = {}
            distance_annotation_stats: Counter[str] = Counter()
            pair_stage_counts: Counter = funnel["pair_stage_counts"]
            generated_counts: Counter = funnel["question_type_generated_counts"]
            routes_by_pair: dict[tuple[str, str], object | None] = {}
            contexts_by_name = {context.image_name: context for context in flash_contexts}
            for raw_frame_1 in flash_contexts:
                for raw_frame_2 in flash_contexts:
                    if raw_frame_1.image_name == raw_frame_2.image_name:
                        continue
                    pair_stage_counts["considered"] += 1
                    pair_key = (raw_frame_1.image_name, raw_frame_2.image_name)
                    if route_graph is None:
                        routes_by_pair[pair_key] = None
                        pair_stage_counts["path_deferred_until_question"] += 1
                    else:
                        route = route_graph.find_route(
                            raw_frame_1.image_name,
                            raw_frame_2.image_name,
                            max_auxiliary_frames=MAX_AUXILIARY_FRAMES,
                        )
                        if route is None:
                            pair_stage_counts["auxiliary_path_rejected"] += 1
                            failures = funnel["pair_failures"]
                            if isinstance(failures, list) and len(failures) < 100:
                                failures.append({
                                    "frame_1": raw_frame_1.image_name,
                                    "frame_2": raw_frame_2.image_name,
                                    "reason": "no_visual_pose_path_within_auxiliary_limit",
                                })
                            continue
                        routes_by_pair[pair_key] = route
                        pair_stage_counts["path_accepted"] += 1

            def _question_object_groups(
                question: dict,
            ) -> tuple[list[dict], list[dict]] | None:
                raw_groups = question.get("object_frame_groups")
                if not isinstance(raw_groups, dict):
                    return None

                def _objects_for_group(group_name: str) -> list[dict]:
                    raw_ids = raw_groups.get(group_name)
                    if not isinstance(raw_ids, (list, tuple, set)):
                        return []
                    resolved: list[dict] = []
                    for raw_id in raw_ids:
                        try:
                            obj = objects_by_id.get(int(raw_id))
                        except (TypeError, ValueError):
                            return []
                        if obj is None:
                            return []
                        resolved.append(obj)
                    return resolved

                group_a_objects = _objects_for_group("frame_1")
                group_b_objects = _objects_for_group("frame_2")
                if not group_a_objects or not group_b_objects:
                    return None
                return group_a_objects, group_b_objects

            def _legacy_route_for_question(
                question: dict,
                frame_1: ReasoningFrameContext,
                frame_2: ReasoningFrameContext,
            ):
                groups = _question_object_groups(question)
                if groups is None or color_intrinsics is None:
                    return None
                group_a_objects, group_b_objects = groups
                try:
                    return find_geometric_auxiliary_route(
                        center_a=object_group_center(group_a_objects),
                        center_b=object_group_center(group_b_objects),
                        frame_a_name=frame_1.image_name,
                        frame_b_name=frame_2.image_name,
                        poses=poses,
                        intrinsics=color_intrinsics,
                        group_a_objects=group_a_objects,
                        group_b_objects=group_b_objects,
                        max_auxiliary_frames=MAX_AUXILIARY_FRAMES,
                    )
                except (TypeError, ValueError):
                    return None

            def _depth_route_for_question(
                question: dict,
                frame_1: ReasoningFrameContext,
                frame_2: ReasoningFrameContext,
            ):
                if depth_data_source is None:
                    return None
                groups = _question_object_groups(question)
                if groups is None or color_intrinsics is None:
                    return None
                group_a_objects, group_b_objects = groups
                return find_depth_corridor_auxiliary_route(
                    center_a=object_group_center(group_a_objects),
                    center_b=object_group_center(group_b_objects),
                    frame_a_name=frame_1.image_name,
                    frame_b_name=frame_2.image_name,
                    poses=poses,
                    intrinsics=color_intrinsics,
                    depth_frame_for=depth_data_source.load_depth_frame,
                    group_a_objects=group_a_objects,
                    group_b_objects=group_b_objects,
                    visual_redundancy_for=depth_visual_redundancy,
                    geometry_cache=depth_route_geometry_cache,
                    max_auxiliary_frames=MAX_AUXILIARY_FRAMES,
                    max_candidate_poses=(
                        None
                        if auxiliary_max_pose_candidates == 0
                        else auxiliary_max_pose_candidates
                    ),
                    enforce_camera_motion_hard_limits=(
                        DEPTH_ROUTE_ENFORCE_CAMERA_MOTION_HARD_LIMITS
                    ),
                )

            def _hybrid_route_for_question(
                question: dict,
                frame_1: ReasoningFrameContext,
                frame_2: ReasoningFrameContext,
            ):
                if hybrid_router is None:
                    return None
                groups = _question_object_groups(question)
                if groups is None:
                    return None
                group_a_objects, group_b_objects = groups
                try:
                    return hybrid_router.find_route(
                        frame_a_name=frame_1.image_name,
                        frame_b_name=frame_2.image_name,
                        group_a_objects=group_a_objects,
                        group_b_objects=group_b_objects,
                        max_auxiliary_frames=MAX_AUXILIARY_FRAMES,
                    )
                except (TypeError, ValueError):
                    return None

            question_route_cache: dict[tuple[object, ...], object | None] = {}
            question_route_stats: Counter[str] = Counter()

            def _question_route_cache_key(
                question: dict,
                frame_1: ReasoningFrameContext,
                frame_2: ReasoningFrameContext,
            ) -> tuple[object, ...] | None:
                groups = _question_object_groups(question)
                if groups is None:
                    return None
                group_a_objects, group_b_objects = groups
                return (
                    auxiliary_route_method,
                    frame_1.image_name,
                    frame_2.image_name,
                    tuple(int(obj["id"]) for obj in group_a_objects),
                    tuple(int(obj["id"]) for obj in group_b_objects),
                )

            def _append_pair_questions(
                pair_questions: list[dict],
                frame_1: ReasoningFrameContext,
                frame_2: ReasoningFrameContext,
                route,
            ) -> None:
                surface_text_by_id: dict[int, str] = {}
                for context in (frame_1, frame_2):
                    surface_text_by_id.update(
                        _attachment_surface_text_by_object_id(context.cache_entry)
                    )
                frame_1_rank = int((frame_1.cache_entry or {}).get("final_selection_rank", 1_000_000))
                frame_2_rank = int((frame_2.cache_entry or {}).get("final_selection_rank", 1_000_000))
                for question in pair_questions:
                    canonical_type = _canonical_scene_question_type(question)
                    if not _cross_type_budget_available(canonical_type):
                        continue
                    question_route = route
                    if auxiliary_route_method in {
                        AUXILIARY_ROUTE_METHOD_DEPTH_CORRIDOR_GEOMETRIC,
                        AUXILIARY_ROUTE_METHOD_LEGACY_GEOMETRIC,
                        AUXILIARY_ROUTE_METHOD_HYBRID_GEOMETRIC_VISUAL,
                    }:
                        route_cache_key = _question_route_cache_key(
                            question,
                            frame_1,
                            frame_2,
                        )
                        route_cache_hit = (
                            route_cache_key is not None
                            and route_cache_key in question_route_cache
                        )
                        if route_cache_hit:
                            question_route_stats["cache_hits"] += 1
                            question_route = question_route_cache[route_cache_key]
                        else:
                            route_started_at = time.perf_counter()
                            if auxiliary_route_method == AUXILIARY_ROUTE_METHOD_DEPTH_CORRIDOR_GEOMETRIC:
                                question_route = _depth_route_for_question(question, frame_1, frame_2)
                            elif auxiliary_route_method == AUXILIARY_ROUTE_METHOD_LEGACY_GEOMETRIC:
                                question_route = _legacy_route_for_question(question, frame_1, frame_2)
                            else:
                                question_route = _hybrid_route_for_question(question, frame_1, frame_2)
                            question_route_stats["computed"] += 1
                            question_route_stats["compute_seconds"] += (
                                time.perf_counter() - route_started_at
                            )
                            if route_cache_key is not None:
                                question_route_cache[route_cache_key] = question_route
                        if auxiliary_route_method == AUXILIARY_ROUTE_METHOD_DEPTH_CORRIDOR_GEOMETRIC:
                            rejection_reason = "no_depth_corridor_path_within_auxiliary_limit"
                        elif auxiliary_route_method == AUXILIARY_ROUTE_METHOD_LEGACY_GEOMETRIC:
                            rejection_reason = "no_legacy_geometric_path_within_auxiliary_limit"
                        else:
                            rejection_reason = "no_hybrid_geometric_visual_path_within_auxiliary_limit"
                        if question_route is None:
                            question_route_stats["rejected"] += 1
                            pair_stage_counts["question_auxiliary_path_rejected"] += 1
                            failures = funnel["pair_failures"]
                            if isinstance(failures, list) and len(failures) < 100:
                                failures.append({
                                    "frame_1": frame_1.image_name,
                                    "frame_2": frame_2.image_name,
                                    "question_type": str(question.get("type", "")),
                                    "reason": rejection_reason,
                                })
                            continue
                        question_route_stats["accepted"] += 1
                        pair_stage_counts["question_path_accepted"] += 1
                    if question_route is None:
                        continue
                    pair_score = float(question_route.cost) + 0.001 * (
                        frame_1_rank + frame_2_rank
                    )
                    question.pop("_cross_frame_layout_hint", None)
                    question["scene_id"] = scene_id
                    question["auxiliary_image_names"] = list(
                        question_route.auxiliary_image_names
                    )
                    if auxiliary_route_method == AUXILIARY_ROUTE_METHOD_VISUAL_POSE_GRAPH:
                        question["auxiliary_route"] = {
                            "method": auxiliary_route_method,
                            "edge_count": question_route.edge_count,
                            "cost": question_route.cost,
                            "min_inliers": question_route.min_inliers,
                            "min_inlier_ratio": question_route.min_inlier_ratio,
                        }
                    elif auxiliary_route_method == AUXILIARY_ROUTE_METHOD_DEPTH_CORRIDOR_GEOMETRIC:
                        question["auxiliary_route"] = {
                            "method": auxiliary_route_method,
                            "search_method": question_route.search_method,
                            "edge_count": question_route.edge_count,
                            "cost": question_route.cost,
                            "route_sample_count": question_route.route_sample_count,
                            "frame_a_coverage_end": question_route.frame_a_coverage_end,
                            "frame_b_coverage_start": question_route.frame_b_coverage_start,
                            "auxiliary_responsibility_fraction": (
                                question_route.auxiliary_responsibility_fraction
                            ),
                            "transition_overlap_fraction": (
                                question_route.transition_overlap_fraction
                            ),
                            "min_progress_fraction": question_route.min_progress_fraction,
                            "min_depth_valid_fraction": (
                                question_route.min_depth_valid_fraction
                            ),
                            "min_depth_visible_fraction": (
                                question_route.min_depth_visible_fraction
                            ),
                            "max_local_perpendicular_m": (
                                question_route.max_local_perpendicular_m
                            ),
                            "max_global_perpendicular_m": (
                                question_route.max_global_perpendicular_m
                            ),
                            "max_height_change_m": question_route.max_height_change_m,
                            "max_parallel_change_m": question_route.max_parallel_change_m,
                            "max_forward_angle_deg": question_route.max_forward_angle_deg,
                            "camera_motion_hard_limits_enabled": (
                                DEPTH_ROUTE_ENFORCE_CAMERA_MOTION_HARD_LIMITS
                            ),
                            "disabled_camera_motion_hard_limits": list(
                                DEPTH_ROUTE_DISABLED_CAMERA_MOTION_HARD_LIMITS
                            ),
                            "depth_sources": list(question_route.depth_sources),
                            "pre_prune_auxiliary_count": (
                                question_route.pre_prune_auxiliary_count
                            ),
                            "pruned_auxiliary_frame_count": (
                                question_route.pruned_auxiliary_frame_count
                            ),
                            "visual_pruned_auxiliary_frame_count": getattr(
                                question_route,
                                "visual_pruned_auxiliary_frame_count",
                                0,
                            ),
                            "visual_duplicate_candidate_count": getattr(
                                question_route,
                                "visual_duplicate_candidate_count",
                                0,
                            ),
                            "visual_prune_relaxed_angle_edge_count": getattr(
                                question_route,
                                "visual_prune_relaxed_angle_edge_count",
                                0,
                            ),
                            "visual_redundancy_metric_version": getattr(
                                question_route,
                                "visual_redundancy_metric_version",
                                None,
                            ),
                            "semantic_rejected_frame_count": (
                                question_route.semantic_rejected_frame_count
                            ),
                        }
                    elif auxiliary_route_method == AUXILIARY_ROUTE_METHOD_LEGACY_GEOMETRIC:
                        question["auxiliary_route"] = {
                            "method": auxiliary_route_method,
                            "search_method": getattr(
                                question_route, "search_method", "legacy_greedy_backtracking"
                            ),
                            "edge_count": question_route.edge_count,
                            "cost": question_route.cost,
                            "route_sample_count": question_route.route_sample_count,
                            "frame_a_coverage_end": question_route.frame_a_coverage_end,
                            "frame_b_coverage_start": question_route.frame_b_coverage_start,
                            "auxiliary_responsibility_fraction": (
                                question_route.auxiliary_responsibility_fraction
                            ),
                            "transition_overlap_fraction": (
                                question_route.transition_overlap_fraction
                            ),
                            "min_progress_fraction": getattr(
                                question_route, "min_progress_fraction", None
                            ),
                            "near_duplicate_translation_m": getattr(
                                question_route, "near_duplicate_translation_m", None
                            ),
                            "near_duplicate_rotation_deg": getattr(
                                question_route, "near_duplicate_rotation_deg", None
                            ),
                            "pre_prune_auxiliary_count": getattr(
                                question_route,
                                "pre_prune_auxiliary_count",
                                len(question_route.auxiliary_image_names),
                            ),
                            "pruned_auxiliary_frame_count": getattr(
                                question_route, "pruned_auxiliary_frame_count", 0
                            ),
                            "semantic_rejected_frame_count": getattr(
                                question_route, "semantic_rejected_frame_count", 0
                            ),
                        }
                    else:
                        question["auxiliary_route"] = {
                            "method": auxiliary_route_method,
                            "edge_count": question_route.edge_count,
                            "cost": question_route.cost,
                            "route_sample_count": question_route.route_sample_count,
                            "frame_a_coverage_end": question_route.frame_a_coverage_end,
                            "frame_b_coverage_start": question_route.frame_b_coverage_start,
                            "auxiliary_responsibility_fraction": (
                                question_route.auxiliary_responsibility_fraction
                            ),
                            "transition_overlap_fraction": (
                                question_route.transition_overlap_fraction
                            ),
                            "min_mutual_matches": question_route.min_mutual_matches,
                            "min_inliers": question_route.min_inliers,
                            "min_inlier_ratio": question_route.min_inlier_ratio,
                            "min_grid_fraction": question_route.min_grid_fraction,
                            "visual_models": list(question_route.visual_models),
                            "semantic_rejected_frames": (
                                question_route.semantic_rejected_frames
                            ),
                        }
                    question["question_referability_audit"] = {
                        "decision": "pass",
                        "mode": "cross_frame_role_aware",
                        "frame_1": frame_1.image_name,
                        "frame_2": frame_2.image_name,
                    }
                    question["_cross_frame_pair_score"] = pair_score
                    if surface_text_by_id:
                        question = _apply_attachment_surface_text_overrides(
                            question,
                            surface_text_by_id,
                        )
                    generated_counts[str(question.get("type", ""))] += 1
                    cross_candidates.append(question)
                    cross_candidate_type_counts[canonical_type] += 1

            completed_deferred_frames = {
                str(value)
                for value in cross_checkpoint_manifest.get(
                    "completed_deferred_frames",
                    [],
                )
            }
            if bool(cross_checkpoint_manifest.get("occlusion_completed")):
                cross_candidates.extend(restored_cross_candidates)
                cross_candidate_type_counts.clear()
                cross_candidate_type_counts.update(restored_cross_candidate_type_counts)
                pair_stage_counts.clear()
                pair_stage_counts.update(restored_pair_stage_counts)
                generated_counts.clear()
                generated_counts.update(restored_generated_counts)
                distance_annotation_stats.clear()
                distance_annotation_stats.update(restored_distance_annotation_stats)
                occlusion_search_budget.combinations_attempted = (
                    restored_occlusion_combinations_attempted
                )
                occlusion_search_budget.raycast_states = restored_occlusion_raycast_states
                _restore_rng_checkpoint(restored_rng_state)

            def _save_cross_checkpoint_manifest() -> None:
                cross_checkpoint_manifest["cross_candidate_type_counts"] = (
                    _counter_checkpoint_payload(cross_candidate_type_counts)
                )
                cross_checkpoint_manifest["pair_stage_counts"] = (
                    _counter_checkpoint_payload(pair_stage_counts)
                )
                cross_checkpoint_manifest["generated_counts"] = (
                    _counter_checkpoint_payload(generated_counts)
                )
                cross_checkpoint_manifest["distance_annotation_stats"] = (
                    _counter_checkpoint_payload(distance_annotation_stats)
                )
                cross_checkpoint_manifest["occlusion_combinations_attempted"] = int(
                    occlusion_search_budget.combinations_attempted
                )
                cross_checkpoint_manifest["occlusion_raycast_states"] = int(
                    occlusion_search_budget.raycast_states
                )
                cross_checkpoint_manifest["rng_state"] = _rng_checkpoint_payload()
                _write_json_file_atomic(
                    cross_checkpoint_manifest_path,
                    cross_checkpoint_manifest,
                )

            occlusion_cross_type = "L2_object_move_occlusion"
            if (
                occlusion_cross_type in cross_frame_requested_types
                and not bool(cross_checkpoint_manifest.get("occlusion_completed"))
                and _cross_type_budget_available(
                    _PUBLIC_TO_CANONICAL_QUESTION_TYPES[occlusion_cross_type]
                )
            ):
                for (frame_1_name, frame_2_name), route in routes_by_pair.items():
                    if not _cross_type_budget_available(
                        _PUBLIC_TO_CANONICAL_QUESTION_TYPES[occlusion_cross_type]
                    ):
                        break
                    frame_1 = contexts_by_name[frame_1_name]
                    frame_2 = contexts_by_name[frame_2_name]
                    pair_questions = generate_cross_frame_questions(
                        objects=scene["objects"],
                        attachment_graph=attachment_graph,
                        attached_by=attached_by,
                        frame_1=frame_1,
                        frame_2=frame_2,
                        color_intrinsics=color_intrinsics,
                        room_bounds=scene.get("room_bounds"),
                        collision_objects=scene["objects"],
                        ray_caster=ray_caster,
                        instance_mesh_data=instance_mesh_data,
                        occlusion_backend=occlusion_backend,
                        only_question_types=[occlusion_cross_type],
                        max_occlusion_objects=max_occlusion_objects,
                        occlusion_max_references_per_query=occlusion_max_references_per_query,
                        max_move_sources=max_move_sources,
                        attachment_edges=scene.get("attachment_edges", []),
                        preserve_distance_metadata=True,
                        occlusion_search_budget=occlusion_search_budget,
                        motion_cache=scene_motion_cache,
                        answer_pair_distance_cache=answer_pair_distance_cache,
                    )
                    pair_questions = _filter_vertical_object_rotate_questions(
                        pair_questions,
                        scene_objects=scene["objects"],
                        attachment_graph=attachment_graph,
                    )
                    _append_pair_questions(pair_questions, frame_1, frame_2, route)

            if not bool(cross_checkpoint_manifest.get("occlusion_completed")):
                _write_json_file_atomic(
                    cross_checkpoint_dir / "occlusion.json",
                    {"questions": cross_candidates},
                )
                cross_checkpoint_manifest["occlusion_completed"] = True
                _save_cross_checkpoint_manifest()

            deferred_cross_types = [
                public_type
                for public_type in cross_frame_requested_types
                if public_type != occlusion_cross_type
            ]
            if deferred_cross_types:
                for frame_1 in flash_contexts:
                    if frame_1.image_name in completed_deferred_frames:
                        continue
                    active_cross_types = [
                        public_type
                        for public_type in deferred_cross_types
                        if _cross_type_budget_available(
                            _PUBLIC_TO_CANONICAL_QUESTION_TYPES[public_type]
                        )
                    ]
                    if not active_cross_types:
                        logger.info(
                            "Scene %s exhausted all requested cross-frame candidate budgets; stopping early",
                            scene_id,
                        )
                        break
                    destinations = [
                        contexts_by_name[frame_2_name]
                        for frame_1_name, frame_2_name in routes_by_pair
                        if frame_1_name == frame_1.image_name
                    ]
                    if not destinations:
                        continue
                    deferred_visibility_conflicts = [
                        {
                            "frame_2": destination.image_name,
                            "object_ids": sorted(
                                destination.regular_referable_ids
                                & frame_1.cross_frame_visible_ids
                            ),
                        }
                        for destination in destinations
                        if destination.regular_referable_ids
                        & frame_1.cross_frame_visible_ids
                    ]
                    for conflict in deferred_visibility_conflicts:
                        conflict_ids = conflict["object_ids"]
                        pair_stage_counts[
                            "deferred_reference_visibility_overlap_object_rejected"
                        ] += len(conflict_ids)
                        failures = funnel["pair_failures"]
                        if isinstance(failures, list) and len(failures) < 100:
                            failures.append({
                                "frame_1": frame_1.image_name,
                                "frame_2": conflict["frame_2"],
                                "reason": "deferred_reference_visible_in_frame_1",
                                "conflicting_object_ids": conflict_ids,
                            })
                    deferred_regular_ids = frozenset().union(*(
                        destination.regular_referable_ids
                        - frame_1.any_referable_ids
                        - frame_1.cross_frame_visible_ids
                        for destination in destinations
                    ))
                    if not deferred_regular_ids:
                        continue
                    frame_candidate_start = len(cross_candidates)
                    deferred_frame_2 = ReasoningFrameContext(
                        image_name="__deferred_frame_2__",
                        camera_pose=destinations[0].camera_pose,
                        regular_referable_ids=deferred_regular_ids,
                        attachment_referable_ids=frozenset(),
                        defer_annotation=True,
                    )
                    raw_questions = generate_cross_frame_questions(
                        objects=scene["objects"],
                        attachment_graph=attachment_graph,
                        attached_by=attached_by,
                        frame_1=frame_1,
                        frame_2=deferred_frame_2,
                        color_intrinsics=color_intrinsics,
                        room_bounds=scene.get("room_bounds"),
                        collision_objects=scene["objects"],
                        ray_caster=ray_caster,
                        instance_mesh_data=instance_mesh_data,
                        occlusion_backend=occlusion_backend,
                        only_question_types=active_cross_types,
                        max_occlusion_objects=max_occlusion_objects,
                        max_move_sources=max_move_sources,
                        attachment_edges=scene.get("attachment_edges", []),
                        motion_cache=scene_motion_cache,
                        answer_pair_distance_cache=answer_pair_distance_cache,
                    )
                    raw_questions = _filter_vertical_object_rotate_questions(
                        raw_questions,
                        scene_objects=scene["objects"],
                        attachment_graph=attachment_graph,
                    )
                    active_canonical_types = {
                        _PUBLIC_TO_CANONICAL_QUESTION_TYPES[public_type]
                        for public_type in active_cross_types
                    }
                    raw_questions = [
                        question
                        for question in raw_questions
                        if _canonical_scene_question_type(question)
                        in active_canonical_types
                    ]
                    for frame_2 in destinations:
                        pair_annotated_questions: list[dict] = []
                        route = routes_by_pair[(frame_1.image_name, frame_2.image_name)]
                        for raw_question in raw_questions:
                            layout_id = str(
                                raw_question.get("_cross_frame_layout_hint", "")
                            ).strip() or None
                            annotated = _annotate_cross_frame_questions(
                                [dict(raw_question)],
                                frame_1=frame_1,
                                frame_2=frame_2,
                                objects_by_id=objects_by_id,
                                layout_id=layout_id,
                                answer_pair_distance_cache=answer_pair_distance_cache,
                                distance_annotation_stats=distance_annotation_stats,
                                rejection_counts=pair_stage_counts,
                                rejection_details=funnel["pair_failures"],
                            )
                            if annotated:
                                pair_annotated_questions.extend(annotated)
                        if pair_annotated_questions:
                            (
                                pair_annotated_questions,
                                dropped_ref_count,
                                dropped_question_count,
                            ) = _cluster_attachment_reference_questions(
                                pair_annotated_questions,
                                objects_by_id=objects_by_id,
                                radius_m=attachment_reference_cluster_radius_m,
                            )
                            pair_stage_counts[
                                "attachment_reference_cluster_dropped_refs"
                            ] += dropped_ref_count
                            pair_stage_counts[
                                "attachment_reference_cluster_dropped_questions"
                            ] += dropped_question_count
                            _append_pair_questions(
                                pair_annotated_questions,
                                frame_1,
                                frame_2,
                                route,
                            )

                    deferred_frame_names = list(
                        cross_checkpoint_manifest.get("completed_deferred_frames", [])
                    )
                    deferred_frame_names.append(frame_1.image_name)
                    shard_index = len(deferred_frame_names)
                    _write_json_file_atomic(
                        cross_checkpoint_dir
                        / "deferred"
                        / f"{shard_index:04d}_{frame_1.image_name}.json",
                        {
                            "frame_1": frame_1.image_name,
                            "questions": cross_candidates[frame_candidate_start:],
                        },
                    )
                    cross_checkpoint_manifest["completed_deferred_frames"] = (
                        deferred_frame_names
                    )
                    completed_deferred_frames.add(frame_1.image_name)
                    _save_cross_checkpoint_manifest()
                    logger.info(
                        "Cross-frame checkpoint saved: scene=%s frame_1=%s completed=%d/%d",
                        scene_id,
                        frame_1.image_name,
                        len(deferred_frame_names),
                        len(flash_contexts),
                    )

            prioritized_cross_questions, distance_priority_diagnostics = (
                _prioritize_cross_frame_questions_by_distance(cross_candidates)
            )
            distance_priority_diagnostics["annotation_invalid_question_count"] = (
                int(distance_annotation_stats["invalid_answer_pair"])
            )
            funnel["distance_priority"] = distance_priority_diagnostics
            funnel["main_frame_visibility_overlap_rejected"] = int(
                pair_stage_counts["question_main_frame_visibility_overlap_rejected"]
            )
            funnel["deferred_reference_visibility_overlap_object_rejected"] = int(
                pair_stage_counts[
                    "deferred_reference_visibility_overlap_object_rejected"
                ]
            )
            retained_cross_questions = _retain_best_cross_frame_views(
                prioritized_cross_questions
            )
            retained_cross_questions = _prioritize_object_move_occlusion_positives(
                retained_cross_questions
            )
            retained_cross_questions = _apply_scene_type_cap(
                retained_cross_questions,
                scene_type_cap=scene_type_cap,
                frame_type_cap=frame_type_cap,
                frame_type_object_cap=frame_type_object_cap,
                type_counts=scene_question_type_counts,
                pair_counts=scene_pair_counts,
            )
            valid_retained_cross_questions = [
                question for question in retained_cross_questions
                if _has_strict_object_move_occlusion_frame_roles(question)
            ]
            invalid_frame_role_count = (
                len(retained_cross_questions) - len(valid_retained_cross_questions)
            )
            retained_cross_questions = valid_retained_cross_questions
            scene_occlusion_questions = [
                question for question in retained_cross_questions
                if _is_object_move_occlusion_question(question)
            ]
            occlusion_balance_diagnostics = {
                "balance_scope": "global_output",
                "nonself_positive_candidate_count": sum(
                    1 for question in scene_occlusion_questions
                    if _is_positive_object_move_occlusion(question)
                    and int(question.get("moved_obj_id", -1))
                    != int(question.get("query_obj_id", -1))
                ),
                "self_positive_candidate_count": sum(
                    1 for question in scene_occlusion_questions
                    if _is_positive_object_move_occlusion(question)
                    and int(question.get("moved_obj_id", -1))
                    == int(question.get("query_obj_id", -1))
                ),
                "neither_candidate_count": sum(
                    1 for question in scene_occlusion_questions
                    if str(question.get("new_pairwise_occlusion_relation", "")).strip()
                    == L2_OBJECT_MOVE_OCCLUSION_RELATION_NEITHER
                ),
                "invalid_frame_role_dropped_count": invalid_frame_role_count,
            }
            funnel["object_move_occlusion_balance"] = {
                **occlusion_balance_diagnostics,
                "directed_combinations_attempted": int(
                    occlusion_search_budget.combinations_attempted
                ),
                "raycast_states": int(occlusion_search_budget.raycast_states),
                "max_directed_combinations": int(
                    occlusion_search_budget.max_combinations
                ),
                "max_references_per_query": int(
                    occlusion_max_references_per_query
                ),
            }
            for question in retained_cross_questions:
                _clear_cross_frame_distance_metadata(question)
            kept_counts: Counter = funnel["question_type_kept_counts"]
            for question in retained_cross_questions:
                kept_counts[str(question.get("type", ""))] += 1
            if hybrid_router is not None:
                if hybrid_cache_path is not None:
                    try:
                        hybrid_router.save_cache(hybrid_cache_path)
                    except OSError as exc:
                        logger.warning(
                            "Could not persist hybrid visual cache for %s: %s",
                            scene_id,
                            exc,
                        )
                funnel["auxiliary_graph"] = {
                    **hybrid_router.diagnostics(),
                    "method": auxiliary_route_method,
                    "cache_hit": hybrid_cache_hit,
                    "cache_path": str(hybrid_cache_path) if hybrid_cache_path is not None else None,
                    "note": "routes_are_computed_per_question_after_object_role_binding",
                }
            if depth_visual_router is not None:
                if depth_visual_cache_path is not None:
                    try:
                        depth_visual_router.save_cache(depth_visual_cache_path)
                    except OSError as exc:
                        logger.warning(
                            "Could not persist depth visual-prune cache for %s: %s",
                            scene_id,
                            exc,
                        )
                funnel["auxiliary_graph"] = {
                    **depth_visual_router.diagnostics(),
                    "method": auxiliary_route_method,
                    "camera_motion_hard_limits_enabled": (
                        DEPTH_ROUTE_ENFORCE_CAMERA_MOTION_HARD_LIMITS
                    ),
                    "disabled_camera_motion_hard_limits": list(
                        DEPTH_ROUTE_DISABLED_CAMERA_MOTION_HARD_LIMITS
                    ),
                    "cache_hit": depth_visual_cache_hit,
                    "cache_path": (
                        str(depth_visual_cache_path)
                        if depth_visual_cache_path is not None
                        else None
                    ),
                    "note": "visual evidence is used only for depth-route pruning",
                }
            funnel["pair_stage_counts"] = dict(sorted(pair_stage_counts.items()))
            funnel["question_type_generated_counts"] = dict(sorted(generated_counts.items()))
            funnel["question_type_kept_counts"] = dict(sorted(kept_counts.items()))
            funnel["final_cross_frame_question_count"] = len(retained_cross_questions)
            funnel["scene_motion_cache"] = scene_motion_cache.diagnostics()
            funnel["question_route_cache_entry_count"] = len(question_route_cache)
            funnel["depth_route_geometry_cache"] = (
                depth_route_geometry_cache.diagnostics()
            )
            funnel["question_route_stats"] = {
                key: (
                    round(float(value), 6)
                    if key.endswith("_seconds")
                    else int(value)
                )
                for key, value in sorted(question_route_stats.items())
            }
            if ds is not None and hasattr(ds, "depth_cache_diagnostics"):
                funnel["depth_cache"] = ds.depth_cache_diagnostics()
            logger.info(
                "Scene %s motion cache diagnostics: %s",
                scene_id,
                scene_motion_cache.diagnostics(),
            )
            logger.info(
                "Scene %s auxiliary route diagnostics: %s",
                scene_id,
                funnel["question_route_stats"],
            )
            _write_json_file(cross_frame_funnel_dir / f"{scene_id}.json", funnel)
            scene_questions.extend(retained_cross_questions)

        if write_frame_debug:
            _write_json_file(
                frame_debug_dir / f"{scene_id}.json",
                {
                    "scene_id": scene_id,
                    "occlusion_backend": occlusion_backend,
                    "scene_attachment_rows": scene_attachment_rows,
                    "frames": scene_frame_debug_entries,
                },
            )

        _persist_completed_scene(
            scene_id,
            scene_questions=scene_questions,
            frame_count=len(frames),
            pipeline_outcome="processed",
        )
        if cross_frame_requested_types:
            shutil.rmtree(
                _cross_frame_scene_cache_dir(output_dir, scene_id),
                ignore_errors=True,
            )

        # Explicitly release heavy scene resources and force GC to reclaim
        # trimesh/Embree C-extension objects that have cyclic references.
        del ray_caster, instance_mesh_data
        gc.collect()

    completed_scene_ids, _, scene_status_changed = _reconcile_pipeline_completed_scenes(
        scene_status_doc,
        raw_questions_dir=raw_questions_dir,
        target_scene_ids=target_scene_ids,
    )
    if scene_status_changed:
        _write_json_file(scene_status_path, scene_status_doc)

    return _rebuild_pipeline_outputs(
        data_root=data_root,
        output_dir=output_dir,
        questions_dir=questions_dir,
        frame_debug_dir=frame_debug_dir,
        raw_questions_dir=raw_questions_dir,
        scene_ids=completed_scene_ids,
        referability_cache=referability_cache,
        write_frame_debug=write_frame_debug,
        run_question_dinox_audit=run_question_dinox_audit,
        run_question_presence_review=run_question_presence_review,
        vlm_url=vlm_url,
        vlm_model=vlm_model,
        question_presence_review_workers=question_presence_review_workers,
        scene_type_cap=scene_type_cap,
        frame_type_cap=frame_type_cap,
        frame_type_object_cap=frame_type_object_cap,
        dataset=dataset,
        scannetpp_sensor=scannetpp_sensor,
        scannetpp_frame_root=scannetpp_frame_root,
    )
    """
        if not has_nontrivial_attachment(attachment_graph):
            logger.info("Scene %s has no support relations — skipping", scene_id)
            continue

        # ---- Stage 3: Frame selection ----
        scene_frames = _get_referability_scene_frames(referability_cache, scene_id)
        frames = _frames_from_referability_cache(scene_frames)
        if not frames:
            logger.info("No valid frames for scene %s — skipping", scene_id)
            continue

        # Load camera poses (with axis alignment so coords match the mesh)
        axis_align = load_axis_alignment(scene_dir)
        poses = load_scannet_poses(scene_dir, axis_alignment=axis_align)
        ray_caster = None
        if needs_mesh_resources:
            mesh_path = scene_dir / f"{scene_id}_vh_clean.ply"
            if not mesh_path.exists():
                mesh_path = scene_dir / f"{scene_id}_vh_clean_2.ply"
            if mesh_path.exists() and RayCaster is not None:
                try:
                    ray_caster = RayCaster.from_ply(str(mesh_path), axis_alignment=axis_align)
                except Exception as e:
                    raise RuntimeError(
                        f"{occlusion_backend} backend requested for {scene_id}, "
                        f"but ray caster initialization failed: {e}"
                    ) from e
            else:
                raise RuntimeError(
                    f"{occlusion_backend} backend requested for {scene_id}, "
                    "but mesh geometry or RayCaster is unavailable"
                )

        instance_mesh_data = None
        try:
            instance_mesh_data = load_instance_mesh_data(
                scene_dir,
                instance_ids=[int(o["id"]) for o in scene["objects"]],
                n_surface_samples=512,
                preloaded_geometry=preloaded_geometry,
                dataset=dataset,
            )
        except Exception as e:
            if needs_mesh_resources:
                raise RuntimeError(
                    f"{occlusion_backend} backend requested for {scene_id}, "
                    f"but instance mesh data could not be loaded: {e}"
                ) from e
            logger.warning(
                "Instance mesh data load failed for %s; distance GT will fall back to AABB closest points: %s",
                scene_id,
                e,
            )

        # Load depth intrinsics once per scene (shared across all frames)
        depth_intrinsics = None
        if use_occlusion:
            try:
                depth_intrinsics = ds.load_depth_intrinsics()
            except Exception as e:
                logger.warning("Depth intrinsics load failed for %s: %s", scene_id, e)

        # Load colour intrinsics for local ROI blur check
        try:
            color_intrinsics = ds.load_intrinsics()
        except Exception as e:
            logger.warning("Color intrinsics load failed for %s: %s", scene_id, e)
            color_intrinsics = None

        scene_frame_debug_entries: list[dict[str, object]] = []

        # ---- Stages 4-6: Relations + Virtual ops + QA ----
        for frame in frames:
            image_name = frame["image_name"]
            if image_name not in poses:
                if write_frame_debug:
                    selector_visible_ids = _normalize_object_ids(frame.get("visible_object_ids"))
                    frame_attachment_rows = _filter_frame_attachment_rows(
                        scene_attachment_rows,
                        set(selector_visible_ids),
                    )
                    scene_frame_debug_entries.append(_build_frame_debug_entry(
                        image_name=image_name,
                        scene_objects=scene["objects"],
                        objects_by_id=objects_by_id,
                        selector_visible_ids=selector_visible_ids,
                        pipeline_visible_ids=[],
                        occlusion_eligible_object_ids=[],
                        referability_entry=_get_referability_entry(referability_cache, scene_id, image_name),
                        frame_attachment_rows=frame_attachment_rows,
                        pipeline_skip_reason="missing_pose",
                    ))
                continue
            camera_pose = poses[image_name]

            # Load depth map for this frame
            depth_image = None
            if use_occlusion and depth_intrinsics is not None:
                frame_id = image_name.replace(".jpg", "")
                depth_path = scene_dir / "depth" / f"{frame_id}.png"
                if depth_path.exists():
                    try:
                        depth_image = load_depth_image(depth_path)
                    except Exception as e:
                        logger.warning("Depth load failed for %s/%s: %s", scene_id, image_name, e)

            selector_visible_ids = _normalize_object_ids(frame.get("visible_object_ids"))
            visible_ids = list(selector_visible_ids)

            visible_id_set = set(int(obj_id) for obj_id in visible_ids)
            referable_ids = None
            attachment_referable_ids = None
            attachment_object_surface_text_by_id: dict[int, str] = {}
            attachment_priority_pairs: list[tuple[int, int]] = []
            label_statuses = None
            label_counts = None
            out_of_frame_not_visible_labels: list[str] = []
            out_of_frame_label_to_object_ids: dict[str, list[int]] | None = None
            referability_entry = _get_referability_entry(
                referability_cache, scene_id, image_name,
            )
            mention_in_frame_ratio_by_obj_id = _build_visible_object_in_frame_ratio_map(
                visible_object_ids=visible_ids,
                referability_entry=referability_entry,
                scene_objects=scene["objects"],
                camera_pose=camera_pose,
                color_intrinsics=color_intrinsics,
            )
            occlusion_eligible_ids = _build_occlusion_eligible_object_ids(
                visible_object_ids=visible_ids,
                mention_in_frame_ratio_by_obj_id=mention_in_frame_ratio_by_obj_id,
            )
            if referability_entry is not None:
                label_statuses = _normalize_label_statuses(referability_entry.get("label_statuses"))
                label_counts = _normalize_label_counts(referability_entry.get("label_counts"))
                out_of_frame_not_visible_labels = _normalize_label_list(
                    referability_entry.get("out_of_frame_not_visible_labels")
                )
                out_of_frame_label_to_object_ids = _shared_normalize_label_to_object_ids(
                    referability_entry.get("out_of_frame_label_to_object_ids")
                )
                referable_ids = [
                    int(obj_id) for obj_id in referability_entry.get("referable_object_ids", [])
                    if int(obj_id) in visible_id_set
                ]
                raw_attachment_referable_ids = referability_entry.get(
                    "attachment_referable_object_ids"
                )
                attachment_object_surface_text_by_id = (
                    _attachment_surface_text_by_object_id(referability_entry)
                )
                attachment_priority_pairs = _attachment_human_review_priority_pairs(
                    referability_entry.get("attachment_human_review_cards")
                )
                if raw_attachment_referable_ids is None:
                    raw_attachment_referable_ids = _derive_final_referability_fields(
                        referability_entry
                    ).get("attachment_referable_object_ids", [])
                attachment_referable_ids = [
                    int(obj_id)
                    for obj_id in (raw_attachment_referable_ids or [])
                    if int(obj_id) in visible_id_set
                ]
                if not referable_ids and not _has_l1_visibility_candidates(
                    label_statuses,
                    out_of_frame_not_visible_labels,
                ):
                    if write_frame_debug:
                        frame_attachment_rows = _filter_frame_attachment_rows(
                            scene_attachment_rows,
                            set(selector_visible_ids) | set(int(obj_id) for obj_id in visible_ids),
                        )
                        scene_frame_debug_entries.append(_build_frame_debug_entry(
                            image_name=image_name,
                            scene_objects=scene["objects"],
                            objects_by_id=objects_by_id,
                            selector_visible_ids=selector_visible_ids,
                            pipeline_visible_ids=list(visible_ids),
                            occlusion_eligible_object_ids=occlusion_eligible_ids,
                            referability_entry=referability_entry,
                            frame_attachment_rows=frame_attachment_rows,
                            pipeline_skip_reason="no_referable_objects_or_l1_candidates",
                        ))
                    logger.debug(
                        "Frame %s/%s has no referable objects or L1 visibility candidates",
                        scene_id, image_name,
                    )
                    continue

            questions = _call_generate_all_questions_compat(
                objects=scene["objects"],
                attachment_graph=attachment_graph,
                attached_by=attached_by,
                support_chain_graph=support_chain_graph,
                support_chain_by=support_chain_by,
                camera_pose=camera_pose,
                color_intrinsics=color_intrinsics,
                depth_image=depth_image,
                depth_intrinsics=depth_intrinsics,
                occlusion_backend=occlusion_backend,
                ray_caster=ray_caster,
                instance_mesh_data=instance_mesh_data,
                visible_object_ids=visible_ids,
                referable_object_ids=referable_ids,
                attachment_referable_object_ids=attachment_referable_ids,
                attachment_chain_role_override=_manual_attachment_roles_for_frame(
                    referability_entry
                ),
                attachment_chain_role_overrides=(
                    _manual_attachment_role_records_for_frame(referability_entry)
                    or None
                ),
                attachment_object_surface_text_by_id=attachment_object_surface_text_by_id,
                attachment_priority_pairs=attachment_priority_pairs,
                occlusion_eligible_object_ids=occlusion_eligible_ids,
                mention_in_frame_ratio_by_obj_id=mention_in_frame_ratio_by_obj_id,
                label_statuses=label_statuses,
                label_counts=label_counts,
                label_to_object_ids=(referability_entry or {}).get("label_to_object_ids"),
                out_of_frame_not_visible_labels=out_of_frame_not_visible_labels,
                out_of_frame_label_to_object_ids=out_of_frame_label_to_object_ids,
                room_bounds=scene.get("room_bounds"),
                wall_objects=scene.get("wall_objects"),
                attachment_edges=scene.get("attachment_edges", []),
                only_question_types=only_question_types,
                max_occlusion_objects=max_occlusion_objects,
                max_move_sources=max_move_sources,
            )
            questions = _filter_vertical_object_rotate_questions(
                questions,
                scene_objects=scene["objects"],
                attachment_graph=attachment_graph,
            )

            for q in questions:
                q["scene_id"]   = scene_id
                q["image_name"] = image_name

            kept_questions, audited_questions = _apply_question_referability_filter(
                questions,
                objects_by_id=objects_by_id,
                referability_entry=referability_entry,
                frame_referable_ids=referable_ids or [],
            )

            all_questions.extend(kept_questions)
            frame_attachment_rows = _filter_frame_attachment_rows(
                scene_attachment_rows,
                set(selector_visible_ids) | set(int(obj_id) for obj_id in visible_ids),
            )
            if write_frame_debug:
                scene_frame_debug_entries.append(_build_frame_debug_entry(
                    image_name=image_name,
                    scene_objects=scene["objects"],
                    objects_by_id=objects_by_id,
                    selector_visible_ids=selector_visible_ids,
                    pipeline_visible_ids=list(visible_ids),
                    occlusion_eligible_object_ids=occlusion_eligible_ids,
                    referability_entry=referability_entry,
                    frame_attachment_rows=frame_attachment_rows,
                    generated_questions=audited_questions,
                ))

        processed += 1
        if write_frame_debug:
            scene_debug_records[scene_id] = {
                "scene_id": scene_id,
                "occlusion_backend": occlusion_backend,
                "scene_attachment_rows": scene_attachment_rows,
                "frames": scene_frame_debug_entries,
            }
        logger.info(
            "Scene %s: %d questions accumulated", scene_id, len(all_questions),
        )

    # ---- Stage 7: Benchmark quality control ----
    logger.info(
        "Running benchmark quality control on %d raw questions (viewer-only attachment filtering excluded)…",
        len(all_questions),
    )
    final_questions = full_quality_pipeline(all_questions)
    final_questions = _apply_question_post_generation_audit(
        questions=final_questions,
        data_root=Path(data_root),
        output_dir=output_dir,
    )

    by_scene: dict[str, list] = defaultdict(list)
    final_by_scene_frame: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for q in final_questions:
        by_scene[q["scene_id"]].append(q)
        final_by_scene_frame[q["scene_id"]][q["image_name"]].append(q)

    for sid, qs in by_scene.items():
        _write_json_file(questions_dir / f"{sid}.json", qs)

    if write_frame_debug:
        for scene_id, record in scene_debug_records.items():
            frame_map = final_by_scene_frame.get(scene_id, {})
            frames = record.get("frames", [])
            if isinstance(frames, list):
                total_generated = 0
                total_final = 0
                for frame_entry in frames:
                    if not isinstance(frame_entry, dict):
                        continue
                    generated_questions = frame_entry.get("generated_questions", [])
                    if isinstance(generated_questions, list):
                        total_generated += len(generated_questions)
                    final_frame_questions = list(frame_map.get(str(frame_entry.get("image_name", "")), []))
                    frame_entry["final_questions"] = final_frame_questions
                    frame_entry["final_question_count"] = len(final_frame_questions)
                    total_final += len(final_frame_questions)
                record["summary"] = {
                    "frame_count": len(frames),
                    "generated_question_count": total_generated,
                    "final_question_count": total_final,
                }
            with open(frame_debug_dir / f"{scene_id}.json", "w", encoding="utf-8") as f:
                json.dump(record, f, indent=2, ensure_ascii=False)

    benchmark = {
        "name":       "PSR-Bench",
        "version":    "1.0",
        "object_move_semantics_version": L2_OBJECT_MOVE_SEMANTICS_VERSION,
        "statistics": compute_statistics(final_questions),
        "questions":  final_questions,
    }
    benchmark_path = output_dir / "benchmark.json"
    with open(benchmark_path, "w", encoding="utf-8") as f:
        json.dump(benchmark, f, indent=2, ensure_ascii=False)

    if run_question_presence_review:
        _run_question_presence_review(
            questions=final_questions,
            data_root=data_root,
            output_dir=output_dir,
            vlm_url=vlm_url,
            vlm_model=vlm_model,
            workers=question_presence_review_workers,
            scannetpp_frame_root=scannetpp_frame_root,
        )

    logger.info(
        "Pipeline complete! %d questions saved to %s",
        len(final_questions), benchmark_path,
    )
    return final_questions


"""


def main():
    parser = argparse.ArgumentParser(
        description="PSR-Bench data generation pipeline"
    )
    parser.add_argument(
        "--data_root", type=str,
        default=os.getenv("SCANNET_PATH", "/home/lihongxing/datasets/ScanNet/data/scans"),
        help="Root directory containing scene subdirectories (ScanNet v2 or ScanNet++)",
    )
    parser.add_argument(
        "--dataset", type=str, choices=("scannet", "scannetpp"), default="scannet",
        help="Dataset to process (scannet or scannetpp)",
    )
    parser.add_argument(
        "--scannetpp_sensor", type=str, choices=("iphone", "dslr"), default="iphone",
        help="Sensor to use when dataset=scannetpp",
    )
    parser.add_argument(
        "--split", type=str, choices=("train", "val", "all"), default=None,
        help="Dataset split. ScanNet v2: filters discovered scene dirs by the matching "
        "SCANNET_METADATA_SPLIT_FILES entry. ScanNet++: selects the matching "
        "SCANNETPP_METADATA_SPLIT_FILES entry; overridden by --scannetpp_split_file. "
        "Candidate budgets per scene/type are val L1=75, L2=400, and L3=300, "
        "or train L1=300, L2=600, and L3=600; final type totals are not capped. "
        "'all' or omitted scans every scene directory under --data_root.",
    )
    parser.add_argument(
        "--scannetpp_split_file", type=str,
        default=None,
        help="Path to scene-ID list (one per line) for scannetpp; overrides --split. If both are omitted, all scenes with geometry in data_root are used",
    )
    parser.add_argument(
        "--scannetpp_frame_root", type=str,
        default=None,
        help="Root directory for extracted ScanNet++ iPhone frames; defaults to data_root/../iphone_frames",
    )
    parser.add_argument(
        "--output_dir", type=str, default="output",
        help="Output directory for generated data",
    )
    parser.add_argument(
        "--max_scenes", type=int, default=300,
        help="Maximum number of scenes to process",
    )
    parser.add_argument(
        "--max_frames", type=int, default=5,
        help="Maximum frames per scene",
    )
    parser.add_argument(
        "--auxiliary_route_method",
        type=str,
        choices=AUXILIARY_ROUTE_METHODS,
        default=AUXILIARY_ROUTE_METHOD_DEPTH_CORRIDOR_GEOMETRIC,
        help=(
            "Auxiliary-frame routing for cross-frame questions. "
            "depth_corridor_geometric uses sensor depth plus an A-to-B-aligned "
            "camera corridor, keeps camera-motion soft costs, disables hard "
            "camera-motion gates, and is the default; "
            "visual_pose_graph uses the current per-scene ORB/RANSAC pose graph; "
            "legacy_geometric uses per-question global Dijkstra search over A-to-B "
            "route projection while excluding portions already covered by the two "
            "main frames; "
            "hybrid_geometric_visual adds reciprocal ORB plus Fundamental/Homography "
            "verification and role-aware semantic gating to that per-question route."
        ),
    )
    parser.add_argument(
        "--auxiliary_max_pose_candidates",
        type=int,
        default=DEFAULT_MAX_CANDIDATE_POSES,
        help=(
            "Maximum geometrically ranked pose candidates per depth auxiliary "
            "route; 0 keeps all candidates"
        ),
    )
    parser.add_argument(
        "--scannetpp_depth_cache_size",
        type=int,
        default=DEFAULT_DEPTH_CACHE_SIZE,
        help="Maximum decoded ScanNet++ depth frames cached per active scene",
    )
    parser.add_argument(
        "--attachment_reference_cluster_radius_m",
        type=float,
        default=0.5,
        help=(
            "Deduplicate nearby attachment_move references with identical "
            "movement signatures within this XY radius; 0 disables"
        ),
    )
    parser.add_argument(
        "--no_occlusion", action="store_true",
        help="Disable depth-map occlusion (faster but no occlusion questions)",
    )
    parser.add_argument(
        "--occlusion_backend",
        type=str,
        choices=("depth", "mesh_ray"),
        default="mesh_ray",
        help="Backend for visibility/occlusion estimation",
    )
    parser.add_argument(
        "--referability_cache", type=str, default=None,
        help=(
            "Optional referability batch cache JSON path, batch glob, or scene_status.json "
            "produced by scripts/run_vlm_referability.py. May be omitted when "
            "--manual_attachment_cache is supplied"
        ),
    )
    parser.add_argument(
        "--manual_attachment_cache",
        type=Path,
        default=None,
        help=(
            "Human-authored two_hop_attachment_salvage_v1 JSON. It can be used alone; "
            "manual mode automatically generates only L3_attachment_chain. When a "
            "referability cache is also supplied, the human attachment data overrides it"
        ),
    )
    parser.add_argument(
        "--label_map", type=str, default=None,
        help="Path to scannetv2-labels.combined.tsv for raw_category→nyu40class normalization",
    )
    parser.add_argument(
        "--vlm_url", type=str, default=DEFAULT_VLM_URL,
        help="Default OpenAI-compatible VLM API base URL",
    )
    parser.add_argument(
        "--vlm_model", type=str, default=None,
        help="Default model name",
    )
    parser.add_argument(
        "--write_frame_debug",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write per-scene frame_debug/<scene_id>.json with frame/object audit data",
    )
    parser.add_argument(
        "--question_dinox_audit",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run the DINO-X-dependent post-generation audit (question_dinox_audit/question_mesh_audit/question_post_generation_review)",
    )
    parser.add_argument(
        "--question_presence_review",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="After benchmark generation, review L1 occlusion visibility plus non-self L2 attachment pairs and export flagged samples for manual review",
    )
    parser.add_argument(
        "--skip_question_vlm_check",
        "--skip_vlm_check",
        dest="skip_question_vlm_check",
        action="store_true",
        help="Skip the post-generation VLM question presence/attachment-pair review",
    )
    parser.add_argument(
        "--question_presence_review_workers",
        type=int,
        default=8,
        help="Thread pool size for post-generation question presence review",
    )
    parser.add_argument(
        "--slow_frame_warn_seconds",
        type=float,
        default=120.0,
        help="Warn when one frame exceeds this many seconds, without interrupting work",
    )
    parser.add_argument(
        "--slow_phase_warn_seconds",
        type=float,
        default=30.0,
        help="Warn when one frame phase exceeds this many seconds, without interrupting work",
    )
    parser.add_argument(
        "--generator_progress_log_seconds",
        type=float,
        default=15.0,
        help="Heartbeat interval in seconds for long-running QA generator loops",
    )
    parser.add_argument(
        "--slow_generator_warn_seconds",
        type=float,
        default=60.0,
        help="Warn once when a QA generator invocation exceeds this many seconds",
    )
    parser.add_argument(
        "--repair_referability_cache",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Repair deterministic final-field drift inside a same-version referability cache and write the repaired cache back to disk",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from output_dir/scene_status.json and _raw_questions_scene_cache instead of starting a new run",
    )
    parser.add_argument(
        "--rebuild_benchmark",
        action="store_true",
        help=(
            "With --resume, skip all pending scene generation and rebuild "
            "benchmark.json from completed _raw_questions_scene_cache entries only"
        ),
    )
    parser.add_argument(
        "--reset",
        type=int,
        default=None,
        help="Before resuming, remove the most recently completed N scene cache/status entries so they will be regenerated",
    )
    parser.add_argument(
        "--no_salvage",
        action="store_true",
        help="Skip automatic human salvage backfill from _edited.html review files",
    )
    parser.add_argument(
        "--only_question_types",
        nargs="*",
        default=None,
        help=(
            "If provided, only generate the listed question types. Valid values: "
            "L1_direction_agent, L1_occlusion, L1_distance, "
            "L1_direction_object_centric, L1_direction_allocentric, "
            "L2_object_move_agent, L2_object_move_distance, "
            "L2_object_move_occlusion, "
            "L2_object_move_object_centric, L2_object_rotate_object_centric, "
            "L2_object_move_allocentric, L2_object_remove, "
            "L3_attachment_chain, L3_attachment_move, L3_coordinate_rotation_agent, L3_coordinate_rotation_object_centric, "
            "L3_coordinate_rotation_allocentric. When omitted, all types are generated."
        ),
    )
    parser.add_argument(
        "--scene_type_cap",
        type=int,
        default=None,
        help=(
            "L1 candidate-generation budget per (dataset, scene, type). Defaults "
            "to 75 for val, 300 for train, and unlimited otherwise; use 0 to disable."
        ),
    )
    parser.add_argument(
        "--frame_type_cap",
        type=int,
        default=2,
        help=(
            "Deprecated compatibility option; ignored. L1/L2/L3 have no "
            "per-frame type-total cap."
        ),
    )
    parser.add_argument(
        "--frame_type_object_cap",
        type=int,
        default=1,
        help=(
            "Deprecated compatibility option; ignored for L1. L1 and L3 use "
            "a fixed per-frame primary-object cap of 1; L2 uses 2."
        ),
    )
    parser.add_argument(
        "--max_questions_per_scene_type",
        type=int,
        default=None,
        help=(
            "Deprecated alias for the L1 candidate-generation budget. When "
            "provided, overrides --scene_type_cap."
        ),
    )
    parser.add_argument(
        "--max_occlusion_objects",
        type=int,
        default=None,
        help=(
            "Maximum number of movement objects per frame that run expensive L2 "
            "object-move occlusion mesh-ray visibility checks. Defaults to an "
            "adaptive per-scene cap when not specified; use 0 to disable the cap "
            "entirely, or a positive integer for a fixed hard cap."
        ),
    )
    parser.add_argument(
        "--occlusion_max_references_per_query",
        type=int,
        default=64,
        help="Maximum projectively ranked reference objects searched per object-move occlusion query.",
    )
    parser.add_argument(
        "--occlusion_max_combinations_per_scene",
        type=int,
        default=2000,
        help="Scene-wide cap on expensive directed object-move occlusion query/reference combinations.",
    )
    parser.add_argument(
        "--max_move_sources",
        type=int,
        default=0,
        help="Maximum number of source objects to process in the L2 object-move outer loop. Use 0 (default) to disable the cap.",
    )
    args = parser.parse_args()
    if args.reset is not None and int(args.reset) <= 0:
        parser.error("--reset must be >= 1")
    if args.reset is not None and not args.resume:
        parser.error("--reset requires --resume")
    if args.rebuild_benchmark and not args.resume:
        parser.error("--rebuild_benchmark requires --resume")
    if args.rebuild_benchmark and args.reset is not None:
        parser.error("--rebuild_benchmark cannot be combined with --reset")
    if args.max_questions_per_scene_type is not None and int(args.max_questions_per_scene_type) < 0:
        parser.error("--max_questions_per_scene_type must be >= 0")
    if args.scene_type_cap is not None and int(args.scene_type_cap) < 0:
        parser.error("--scene_type_cap must be >= 0")
    if int(args.frame_type_cap) < 0:
        parser.error("--frame_type_cap must be >= 0")
    if int(args.frame_type_object_cap) < 0:
        parser.error("--frame_type_object_cap must be >= 0")
    if args.max_occlusion_objects is not None and int(args.max_occlusion_objects) < 0:
        parser.error("--max_occlusion_objects must be >= 0")
    if int(args.occlusion_max_references_per_query) < 0:
        parser.error("--occlusion_max_references_per_query must be >= 0")
    if int(args.occlusion_max_combinations_per_scene) < 0:
        parser.error("--occlusion_max_combinations_per_scene must be >= 0")
    if args.skip_question_vlm_check:
        args.question_presence_review = False

    _set_pipeline_random_seed()

    if args.label_map:
        load_scannet_label_map(args.label_map)

    if args.referability_cache is None and args.manual_attachment_cache is None:
        parser.error(
            "one of --referability_cache or --manual_attachment_cache is required"
        )

    referability_cache = None
    if args.referability_cache is not None:
        referability_cache = _load_referability_cache(
            args.referability_cache,
            repair_inconsistent_entries=args.repair_referability_cache,
            persist_repaired_entries=args.repair_referability_cache,
            no_salvage=args.no_salvage,
        )
        if referability_cache is None:
            raise ValueError(f"Referability cache not found: {args.referability_cache}")
    if args.manual_attachment_cache is not None:
        manual_attachment_cache = _load_single_referability_cache(
            args.manual_attachment_cache,
            no_salvage=True,
        )
        if manual_attachment_cache is None:
            raise ValueError(
                f"Manual attachment cache not found: {args.manual_attachment_cache}"
            )
        referability_cache = _merge_manual_attachment_cache(
            referability_cache
            or {"version": EXPECTED_REFERABILITY_CACHE_VERSION, "frames": {}},
            manual_attachment_cache,
        )
    elif _is_manual_attachment_cache(referability_cache):
        referability_cache = _merge_manual_attachment_cache(
            {"version": EXPECTED_REFERABILITY_CACHE_VERSION, "frames": {}},
            referability_cache,
        )
    if _has_manual_attachment_overrides(referability_cache):
        args.only_question_types = list(MANUAL_ATTACHMENT_QUESTION_TYPES)

    run_pipeline(
        data_root=Path(args.data_root),
        output_dir=Path(args.output_dir),
        dataset=args.dataset,
        scannetpp_sensor=args.scannetpp_sensor,
        split=args.split,
        scannetpp_split_file=args.scannetpp_split_file,
        scannetpp_frame_root=args.scannetpp_frame_root,
        max_scenes=args.max_scenes,
        max_frames=args.max_frames,
        use_occlusion=not args.no_occlusion,
        referability_cache=referability_cache,
        occlusion_backend=args.occlusion_backend,
        vlm_url=args.vlm_url,
        vlm_model=args.vlm_model,
        write_frame_debug=args.write_frame_debug,
        run_question_dinox_audit=args.question_dinox_audit,
        run_question_presence_review=args.question_presence_review,
        question_presence_review_workers=args.question_presence_review_workers,
        slow_frame_warn_seconds=args.slow_frame_warn_seconds,
        slow_phase_warn_seconds=args.slow_phase_warn_seconds,
        generator_progress_log_seconds=args.generator_progress_log_seconds,
        slow_generator_warn_seconds=args.slow_generator_warn_seconds,
        resume=args.resume,
        rebuild_benchmark=args.rebuild_benchmark,
        reset=args.reset,
        only_question_types=args.only_question_types,
        scene_type_cap=args.scene_type_cap,
        frame_type_cap=args.frame_type_cap,
        frame_type_object_cap=args.frame_type_object_cap,
        max_questions_per_scene_type=args.max_questions_per_scene_type,
        max_occlusion_objects=(
            MAX_OCCLUSION_OBJECTS_AUTO
            if args.max_occlusion_objects is None
            else (None if int(args.max_occlusion_objects) == 0 else int(args.max_occlusion_objects))
        ),
        occlusion_max_references_per_query=args.occlusion_max_references_per_query,
        occlusion_max_combinations_per_scene=args.occlusion_max_combinations_per_scene,
        max_move_sources=(None if int(args.max_move_sources) == 0 else int(args.max_move_sources)),
        auxiliary_route_method=args.auxiliary_route_method,
        auxiliary_max_pose_candidates=args.auxiliary_max_pose_candidates,
        scannetpp_depth_cache_size=args.scannetpp_depth_cache_size,
        attachment_reference_cluster_radius_m=(
            args.attachment_reference_cluster_radius_m
        ),
    )


if __name__ == "__main__":
    main()
