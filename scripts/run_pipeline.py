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
    _in_frame_surface_sample_subset,
    _instance_triangle_id_set,
    _mesh_visibility_stats_compat,
    _apply_attachment_surface_text_overrides,
    generate_all_questions,
)
from src.referability_checks import (
    QUESTION_MENTION_FIELDS,
    build_question_referability_audit as _shared_build_question_referability_audit,
    collect_question_mentions as _shared_collect_question_mentions,
    coerce_object_id as _shared_coerce_object_id,
    normalize_label_to_object_ids as _shared_normalize_label_to_object_ids,
)
from src.quality_control import full_quality_pipeline, compute_statistics
from src.utils.colmap_loader import (
    load_axis_alignment,
    load_scannet_depth_intrinsics,
    load_scannet_intrinsics,
    load_scannet_poses,
)
from src.utils.depth_occlusion import load_depth_image
from src.utils import RayCaster
from scripts.run_vlm_referability import (
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
PIPELINE_SCENE_STATUS_VERSION = 1
PIPELINE_RANDOM_SEED = 20240506
RAW_QUESTIONS_SCENE_CACHE_DIRNAME = "_raw_questions_scene_cache"
QUESTION_REVIEW_MAX_RETRIES = 4
QUESTION_REVIEW_RETRY_DELAY_SECONDS = 2.0
QUESTION_REVIEW_MAX_TOKENS_PER_TARGET = 128
QUESTION_REVIEW_MAX_TOKENS_CAP = 1024
VLM_API_KEY_ENV_NAMES = ("DASHSCOPE_API_KEY", "OPENAI_API_KEY")
PLACEHOLDER_VLM_API_KEY = "EMPTY"
QUESTION_REVIEW_CROP_PADDING_RATIO = 0.10
QUESTION_REVIEW_CROP_MIN_PADDING_PX = 12
QUESTION_REVIEW_CROP_MAX_PADDING_PX = 80
QUESTION_REVIEW_CROP_MIN_DIM_PX = 16
QUESTION_REVIEW_CROP_MIN_PROJECTED_AREA_PX = 800.0
QUESTION_REVIEW_CROP_MIN_IN_FRAME_RATIO = 0.35
QUESTION_MENTION_FALLBACK_FIELDS = QUESTION_MENTION_FIELDS
REFERABLE_OCCLUSION_VETO_DENSE_BASE_SAMPLE_COUNT = 512
REFERABLE_OCCLUSION_VETO_DENSE_BASE_PROJECTED_AREA_PX = 400.0
REFERABLE_OCCLUSION_VETO_DENSE_MAX_SAMPLE_COUNT = 4096
REFERABLE_OCCLUSION_VETO_MIN_VISIBLE_RATIO = 0.35
REFERABLE_OCCLUSION_VETO_DENSE_CHUNK_SIZE = 64
_GENERATE_ALL_QUESTIONS_ATTACHMENT_SURFACE_COMPAT_WARNING_EMITTED = False


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
            if "attachment_priority_pairs" in message:
                if "attachment_priority_pairs" not in compat_kwargs:
                    raise
                compat_kwargs.pop("attachment_priority_pairs", None)
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
        )
    scene_mesh_cache: dict[str, object] = {}
    scene_depth_intrinsics_cache: dict[str, object] = {}
    topology_cache: dict[tuple[str, int], dict[str, object]] = {}

    for question in questions:
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
        if isinstance(entry, dict) and not _frame_entry_has_consistent_final_fields(entry):
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
    if no_salvage:
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
            "disabled via --no_salvage" if no_salvage else "no review HTML found",
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
        "model",
        "alias_config_version",
    )
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


def _load_referability_cache(
    path_or_pattern: str | Path,
    *,
    repair_inconsistent_entries: bool = False,
    persist_repaired_entries: bool = False,
    no_salvage: bool = False,
) -> dict | None:
    paths, used_glob = _expand_referability_cache_paths(path_or_pattern)
    if len(paths) == 1 and not used_glob:
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


def _get_referability_entry(cache: dict | None, scene_id: str, image_name: str) -> dict | None:
    if not cache:
        return None
    frames = cache.get("frames", cache)
    scene_frames = frames.get(scene_id)
    if isinstance(scene_frames, dict):
        entry = scene_frames.get(image_name)
        if not isinstance(entry, dict):
            return entry
        if not _frame_entry_has_consistent_final_fields(entry):
            raise ValueError(
                f"Referability cache entry for {scene_id}/{image_name} is inconsistent with cache version "
                f"{EXPECTED_REFERABILITY_CACHE_VERSION}. Regenerate the referability cache instead of repairing it at read time."
            )
        return entry
    entry = frames.get(f"{scene_id}/{image_name}")
    if not isinstance(entry, dict):
        return entry
    if not _frame_entry_has_consistent_final_fields(entry):
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
    return (
        question_type == "attachment_chain"
        or question_type.startswith("attachment")
        or bool(question.get("attachment_remapped", False))
    )


MIXED_ATTACHMENT_OBJECT_MOVE_TYPES = {
    "object_move_agent",
    "object_move_distance",
}


def _effective_question_referability_ids(
    question: dict[str, object],
    *,
    frame_referable_ids: list[int],
    attachment_frame_referable_ids: list[int] | None = None,
) -> list[int]:
    if (
        attachment_frame_referable_ids is None
        or not _question_uses_attachment_referability(question)
    ):
        return list(frame_referable_ids)
    question_type = str(question.get("type", "")).strip().lower()
    if question_type in MIXED_ATTACHMENT_OBJECT_MOVE_TYPES:
        return sorted(
            set(int(obj_id) for obj_id in frame_referable_ids)
            | set(int(obj_id) for obj_id in attachment_frame_referable_ids)
        )
    return list(attachment_frame_referable_ids)


def _build_question_referability_audit(
    question: dict[str, object],
    *,
    objects_by_id: dict[int, dict[str, object]],
    referability_entry: dict[str, object] | None,
    frame_referable_ids: list[int],
    attachment_frame_referable_ids: list[int] | None = None,
) -> dict[str, object]:
    effective_frame_referable_ids = _effective_question_referability_ids(
        question,
        frame_referable_ids=frame_referable_ids,
        attachment_frame_referable_ids=attachment_frame_referable_ids,
    )
    return _shared_build_question_referability_audit(
        question,
        objects_by_id=objects_by_id,
        label_statuses=(referability_entry or {}).get("label_statuses"),
        label_to_object_ids=(referability_entry or {}).get("label_to_object_ids"),
        frame_referable_ids=effective_frame_referable_ids,
    )


def _apply_question_referability_filter(
    questions: list[dict[str, object]],
    *,
    objects_by_id: dict[int, dict[str, object]],
    referability_entry: dict[str, object] | None,
    frame_referable_ids: list[int],
    attachment_frame_referable_ids: list[int] | None = None,
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
            f"{len(dropped_summaries)} question(s) that should have been filtered by the generator "
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
        or projected_area_px < QUESTION_REVIEW_CROP_MIN_PROJECTED_AREA_PX
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
) -> dict[str, object]:
    from src.datasets import make_data_source
    ds_review = make_data_source(dataset, data_root / scene_id, sensor=scannetpp_sensor)
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
) -> dict[tuple[str, str], dict[str, object]]:
    frame_keys: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for question in questions:
        scene_id = str(question.get("scene_id", "")).strip()
        image_name = str(question.get("image_name", "")).strip()
        key = (scene_id, image_name)
        if key in seen:
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
    return (
        str(question.get("level", "")).strip().upper() == "L1"
        and str(question.get("type", "")).strip() == "occlusion"
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

    objects_by_id = (
        dict(frame_context.get("objects_by_id", {}))
        if isinstance(frame_context, dict) else {}
    )
    targets = _collect_question_presence_targets(question, objects_by_id)
    object_reviews: list[dict[str, object]] = []
    raw_response = ""
    valid_targets: list[dict[str, object]] = []

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
        if not isinstance(frame_context, dict):
            object_reviews.append(
                _build_presence_review_entry(
                    target,
                    status="unsure",
                    reason="missing_frame_context",
                )
            )
            continue
        if not bool(frame_context.get("image_exists", False)):
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
            not bool(frame_context.get("has_projection_context", False))
            or not str(frame_context.get("image_b64", "") or "")
        ):
            object_reviews.append(
                _build_presence_review_entry(
                    target,
                    status="unsure",
                    reason="missing_frame_context",
                )
            )
            continue

        crop_entry = frame_context.get("crop_by_obj_id", {}).get(obj_id)
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

        valid_targets.append(
            {
                **target,
                "roi_bounds_px": crop_entry.get("roi_bounds_px"),
                "crop_image_b64": crop_entry.get("image_b64"),
                "crop_mime": crop_entry.get("mime", "image/jpeg"),
            }
        )

    if valid_targets:
        try:
            vlm_review = review_fn(frame_context, question, valid_targets)
            raw_response = str(vlm_review.get("raw_response", "") or "")
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

    review = _finalize_presence_review(object_reviews, raw_response=raw_response)
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
    for grandparent_id, parent_ids in graph.items():
        if grandparent_id not in eligible_ids:
            continue
        for parent_id in parent_ids:
            parent_id = int(parent_id)
            if parent_id not in eligible_ids:
                continue
            if any(int(grandchild_id) in eligible_ids for grandchild_id in graph.get(parent_id, [])):
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


def _set_pipeline_random_seed(seed: int = PIPELINE_RANDOM_SEED) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))


def _pipeline_scene_status_path(output_dir: Path) -> Path:
    return output_dir / "scene_status.json"


def _raw_scene_questions_cache_dir(output_dir: Path) -> Path:
    return output_dir / RAW_QUESTIONS_SCENE_CACHE_DIRNAME


def _build_empty_pipeline_scene_status_doc() -> dict[str, object]:
    return {
        "version": PIPELINE_SCENE_STATUS_VERSION,
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

    completed_scenes = loaded.get("completed_scenes")
    if not isinstance(completed_scenes, dict):
        raise RuntimeError(f"Invalid scene status document at {path}: completed_scenes must be an object")

    return {
        "version": PIPELINE_SCENE_STATUS_VERSION,
        "completed_scenes": dict(completed_scenes),
    }


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
    try:
        scene_status_path.unlink()
    except FileNotFoundError:
        pass
    shutil.rmtree(raw_questions_dir, ignore_errors=True)


def _delete_raw_scene_cache_files(raw_questions_dir: Path, scene_ids: list[str]) -> None:
    for scene_id in scene_ids:
        try:
            (raw_questions_dir / f"{scene_id}.json").unlink()
        except FileNotFoundError:
            pass


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
    return {
        "name": "PSR-Bench",
        "version": "1.0",
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
_PUBLIC_TO_CANONICAL_QUESTION_TYPES = {
    "L1_direction_agent": "direction_agent",
    "L2_object_move_agent": "object_move_agent",
    "L2_object_move_distance": "object_move_distance",
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
_QUESTION_CAP_OBJECT_ID_FIELDS = [
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
]
_PAIR_KEY_FIELDS_BY_TYPE: dict[str, tuple[str, str]] = {
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


def _question_cap_object_id(question: dict) -> str:
    for field in _QUESTION_CAP_OBJECT_ID_FIELDS:
        value = question.get(field)
        if value is not None:
            return str(value)
    trace_question_id = question.get("trace_question_id")
    if trace_question_id is not None:
        return f"trace:{trace_question_id}"
    return f"question:{question.get('question', '')}"


def _question_pair_key(question: dict) -> tuple[str, str] | None:
    canonical_type = _canonical_scene_question_type(question)
    field_pair = _PAIR_KEY_FIELDS_BY_TYPE.get(canonical_type)
    if field_pair is not None:
        left = question.get(field_pair[0])
        right = question.get(field_pair[1])
        if left is not None and right is not None:
            pair = tuple(sorted((str(left), str(right))))
            if pair[0] != pair[1]:
                return pair

    unique_ids: list[str] = []
    for field in _QUESTION_CAP_OBJECT_ID_FIELDS:
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
            return pair
    return None


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
    if int(frame.get("attachment_referable_pair_count", 0) or 0) > 0:
        return True

    if referability_entry is not None:
        visible_ids = set(_normalize_object_ids(frame.get("visible_object_ids")))
        if not visible_ids:
            visible_ids = set(
                _normalize_object_ids((referability_entry or {}).get("candidate_visible_object_ids"))
            )
        for pair in (referability_entry.get("attachment_referable_pairs") or []):
            if not isinstance(pair, dict):
                continue
            parent_id = pair.get("parent_id")
            child_id = pair.get("child_id")
            try:
                parent_id = int(parent_id)
                child_id = int(child_id)
            except (TypeError, ValueError):
                continue
            if not visible_ids or (parent_id in visible_ids and child_id in visible_ids):
                return True

    graph = {
        int(parent_id): [int(child_id) for child_id in (child_ids or [])]
        for parent_id, child_ids in (attachment_graph or {}).items()
    }
    return any(child_ids for child_ids in graph.values())


def _apply_incremental_question_caps(
    questions: list[dict],
    *,
    scene_type_cap: int,
    frame_type_cap: int,
    frame_type_object_cap: int,
    scene_type_counts: Counter[str] | None = None,
    frame_type_counts: Counter[tuple[str, str]] | None = None,
    frame_type_object_counts: Counter[tuple[str, str, str]] | None = None,
    pair_counts: Counter[tuple[str, str]] | None = None,
) -> list[dict]:
    kept: list[dict] = []
    scene_counts = scene_type_counts if scene_type_counts is not None else Counter()
    frame_counts = frame_type_counts if frame_type_counts is not None else Counter()
    frame_object_counts = (
        frame_type_object_counts if frame_type_object_counts is not None else Counter()
    )
    pair_counter = pair_counts if pair_counts is not None else Counter()

    for question in questions:
        canonical_type = _canonical_scene_question_type(question)
        if not canonical_type:
            kept.append(question)
            continue
        image_name = str(question.get("image_name", "")).strip()
        object_id = _question_cap_object_id(question)
        pair_key = _question_pair_key(question)
        frame_key = (image_name, canonical_type)
        frame_object_key = (image_name, canonical_type, object_id)
        if scene_type_cap > 0 and scene_counts[canonical_type] >= scene_type_cap:
            continue
        if frame_type_cap > 0 and frame_counts[frame_key] >= frame_type_cap:
            continue
        if (
            frame_type_object_cap > 0
            and frame_object_counts[frame_object_key] >= frame_type_object_cap
        ):
            continue
        if pair_key is not None and pair_counter[pair_key] >= 1:
            continue
        kept.append(question)
        scene_counts[canonical_type] += 1
        frame_counts[frame_key] += 1
        frame_object_counts[frame_object_key] += 1
        if pair_key is not None:
            pair_counter[pair_key] += 1
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
    frame_type_counts: Counter[tuple[str, str]] | None = None,
    frame_type_object_counts: Counter[tuple[str, str, str]] | None = None,
    pair_counts: Counter[tuple[str, str]] | None = None,
) -> list[dict]:
    return _apply_incremental_question_caps(
        questions,
        scene_type_cap=scene_type_cap,
        frame_type_cap=frame_type_cap,
        frame_type_object_cap=frame_type_object_cap,
        scene_type_counts=type_counts,
        frame_type_counts=frame_type_counts,
        frame_type_object_counts=frame_type_object_counts,
        pair_counts=pair_counts,
    )


def _remaining_scene_type_budgets(
    type_counts: Counter[str],
    *,
    scene_type_cap: int,
    allowed_types: set[str] | None = None,
) -> dict[str, int] | None:
    if scene_type_cap <= 0:
        return None
    target_types = set(allowed_types) if allowed_types else set(_ALL_CANONICAL_QUESTION_TYPES)
    return {
        question_type: max(scene_type_cap - int(type_counts[question_type]), 0)
        for question_type in sorted(target_types)
    }


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
    return (q.get("scene_id"), q.get("question"), ids_sorted)


def _load_cached_scene_questions(
    raw_questions_dir: Path,
    *,
    scene_ids: list[str],
    scene_type_cap: int,
    frame_type_cap: int,
    frame_type_object_cap: int,
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
                        _attachment_human_review_surface_text_by_object_id(
                            entry.get("attachment_human_review_cards")
                        )
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
        raw_question_count += len(scene_questions)
        scene_questions = _deduplicate_scene_questions(scene_questions)
        scene_questions = _apply_scene_type_cap(
            scene_questions,
            scene_type_cap=scene_type_cap,
            frame_type_cap=frame_type_cap,
            frame_type_object_cap=frame_type_object_cap,
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
    dataset: str = "scannet",
    scannetpp_sensor: str = "iphone",
) -> list[dict]:
    all_questions, raw_question_count = _load_cached_scene_questions(
        raw_questions_dir,
        scene_ids=scene_ids,
        scene_type_cap=scene_type_cap,
        frame_type_cap=frame_type_cap,
        frame_type_object_cap=frame_type_object_cap,
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
        )
    if run_question_dinox_audit:
        final_questions = _apply_question_post_generation_audit(
            questions=final_questions,
            data_root=Path(data_root),
            output_dir=output_dir,
            frame_context_by_key=question_review_frame_contexts,
            dataset=dataset,
            scannetpp_sensor=scannetpp_sensor,
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
    reset: int | None = None,
    only_question_types: list[str] | None = None,
    scene_type_cap: int = 8,
    frame_type_cap: int = 2,
    frame_type_object_cap: int = 1,
    max_questions_per_scene_type: int | None = None,
    max_occlusion_objects: int | None = 20,
    max_move_sources: int | None = None,
):
    """Execute the full PSR-Bench data generation pipeline."""
    _set_pipeline_random_seed()
    only_question_types = _normalize_only_question_types(only_question_types)

    if referability_cache is None:
        raise ValueError(
            "run_pipeline requires a referability_cache generated by scripts/run_vlm_referability.py"
        )
    if reset is not None and int(reset) <= 0:
        raise ValueError("reset must be >= 1")
    if reset is not None and not resume:
        raise ValueError("reset requires resume=True")
    if max_questions_per_scene_type is not None:
        scene_type_cap = int(max_questions_per_scene_type)
    scene_type_cap = int(scene_type_cap)
    frame_type_cap = int(frame_type_cap)
    frame_type_object_cap = int(frame_type_object_cap)
    if scene_type_cap < 0:
        raise ValueError("scene_type_cap must be >= 0")
    if frame_type_cap < 0:
        raise ValueError("frame_type_cap must be >= 0")
    if frame_type_object_cap < 0:
        raise ValueError("frame_type_object_cap must be >= 0")
    if max_occlusion_objects is not None:
        max_occlusion_objects = int(max_occlusion_objects)
        if max_occlusion_objects < 0:
            raise ValueError("max_occlusion_objects must be >= 0 or None")
    if dataset not in ("scannet", "scannetpp"):
        raise ValueError(f"Unknown dataset: {dataset!r}. Expected 'scannet' or 'scannetpp'.")
    l3_attachment_chain_only = only_question_types == ["L3_attachment_chain"]
    l3_attachment_move_only = only_question_types == ["L3_attachment_move"]
    target_scene_question_types = _scene_question_target_types(only_question_types)
    attachment_only_l2_mode = _only_l2_attachment_types_requested(only_question_types)

    meta_dir = output_dir / "scene_metadata"
    questions_dir = output_dir / "questions"
    frame_debug_dir = output_dir / "frame_debug"
    meta_dir.mkdir(parents=True, exist_ok=True)
    questions_dir.mkdir(parents=True, exist_ok=True)
    if write_frame_debug:
        frame_debug_dir.mkdir(parents=True, exist_ok=True)

    if dataset == "scannetpp":
        from src.datasets.scannetpp import resolve_scannetpp_scene_dirs
        discovered_scene_dirs = resolve_scannetpp_scene_dirs(
            data_root,
            sensor=scannetpp_sensor,
        )
    else:
        discovered_scene_dirs = sorted(
            p for p in data_root.iterdir()
            if p.is_dir() and (p / "pose").exists()
        )
    cached_scene_ids = _get_referability_scene_ids(referability_cache)
    scene_dirs = [p for p in discovered_scene_dirs if p.name in cached_scene_ids]
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
            _write_json_file(scene_status_path, scene_status_doc)
            logger.info(
                "Reset cleared %d completed scene(s) from %s",
                len(removed_scene_ids),
                scene_status_path,
            )
        completed_scene_ids, corrupted_scene_ids, scene_status_changed = _reconcile_pipeline_completed_scenes(
            scene_status_doc,
            raw_questions_dir=raw_questions_dir,
            target_scene_ids=target_scene_ids,
        )
        if corrupted_scene_ids:
            logger.warning(
                "Resume found %d scene status record(s) with missing raw scene cache; they will be regenerated: %s",
                len(corrupted_scene_ids),
                ", ".join(corrupted_scene_ids),
            )
        if scene_status_changed:
            _write_json_file(scene_status_path, scene_status_doc)
    else:
        _clear_pipeline_resume_state(output_dir)
        scene_status_doc = _build_empty_pipeline_scene_status_doc()
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
        scene_pair_counts: Counter[tuple[str, str]] = Counter()
        logger.info(
            "=== Processing scene %s (%d/%d) ===",
            scene_id,
            scene_index,
            total_scenes,
        )

        scene_questions: list[dict] = []
        scene_frame_debug_entries: list[dict[str, object]] = []
        preloaded_geometry = None
        needs_mesh_resources = occlusion_backend in ("depth", "mesh_ray")
        if needs_mesh_resources:
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
            ds = make_data_source(dataset, scene_dir, sensor=scannetpp_sensor)

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
            if scene_type_cap > 0:
                remaining_scene_type_budgets = _remaining_scene_type_budgets(
                    scene_question_type_counts,
                    scene_type_cap=scene_type_cap,
                    allowed_types=target_scene_question_types,
                )
                if remaining_scene_type_budgets is not None and all(
                    budget <= 0 for budget in remaining_scene_type_budgets.values()
                ):
                    logger.info(
                        "Scene %s reached all active per-type caps after %d frame(s); stopping early",
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
                            _attachment_human_review_surface_text_by_object_id(
                                referability_entry.get("attachment_human_review_cards")
                            )
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

                if referability_entry is not None and not referable_ids and not _has_l1_visibility_candidates(
                    label_statuses,
                    out_of_frame_not_visible_labels,
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

                if attachment_only_l2_mode and not _frame_has_attachment_pair(
                    frame,
                    referability_entry,
                    attachment_graph,
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
                                    pipeline_skip_reason="no_attachment_pair_for_attachment_only_l2",
                                )
                            )
                    frame_status = "skipped"
                    frame_skip_reason = "no_attachment_pair_for_attachment_only_l2"
                    continue

                with _timed_frame_phase(frame_ctx, "generate_all_questions"):
                    try:
                        question_type_budgets = _remaining_scene_type_budgets(
                            scene_question_type_counts,
                            scene_type_cap=scene_type_cap,
                            allowed_types=target_scene_question_types,
                        )
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
                            only_question_types=only_question_types,
                            question_type_budgets=question_type_budgets,
                            max_occlusion_objects=max_occlusion_objects,
                            max_move_sources=max_move_sources,
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
                frame_raw_generated_count = len(questions)

                for q in questions:
                    q["scene_id"] = scene_id
                    q["image_name"] = image_name

                with _timed_frame_phase(frame_ctx, "referability_post_filter"):
                    kept_questions, audited_questions = _apply_question_referability_filter(
                        questions,
                        objects_by_id=objects_by_id,
                        referability_entry=referability_entry,
                        frame_referable_ids=referable_ids or [],
                        attachment_frame_referable_ids=attachment_referable_ids or [],
                    )
                    frame_question_type_counts: Counter[tuple[str, str]] = Counter()
                    frame_question_type_object_counts: Counter[tuple[str, str, str]] = Counter()
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
                    _attachment_human_review_surface_text_by_object_id(
                        referability_entry.get("attachment_human_review_cards")
                    )
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
        "--referability_cache", type=str, required=True,
        help="Referability cache JSON path or glob (for example output/referability_cache/flash*.json) produced by scripts/run_vlm_referability.py",
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
            "L1_direction_agent, L2_object_move_agent, L2_object_move_distance, "
            "L2_object_move_object_centric, L2_object_rotate_object_centric, "
            "L2_object_move_allocentric, L2_object_remove, "
            "L3_attachment_chain, L3_attachment_move, L3_coordinate_rotation_agent, L3_coordinate_rotation_object_centric, "
            "L3_coordinate_rotation_allocentric. When omitted, all types are generated."
        ),
    )
    parser.add_argument(
        "--scene_type_cap",
        type=int,
        default=8,
        help="Maximum kept questions per (dataset, scene, type). Use 0 to disable.",
    )
    parser.add_argument(
        "--frame_type_cap",
        type=int,
        default=2,
        help="Maximum kept questions per (dataset, scene, frame, type). Use 0 to disable.",
    )
    parser.add_argument(
        "--frame_type_object_cap",
        type=int,
        default=1,
        help="Maximum kept questions per (dataset, scene, frame, type, obj). Use 0 to disable.",
    )
    parser.add_argument(
        "--max_questions_per_scene_type",
        type=int,
        default=None,
        help="Deprecated alias for --scene_type_cap. When provided, overrides --scene_type_cap.",
    )
    parser.add_argument(
        "--max_occlusion_objects",
        type=int,
        default=20,
        help="Maximum number of movement objects per frame that run expensive L2 object-move occlusion mesh-ray visibility checks. Use 0 to disable the cap.",
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
    if args.max_questions_per_scene_type is not None and int(args.max_questions_per_scene_type) < 0:
        parser.error("--max_questions_per_scene_type must be >= 0")
    if int(args.scene_type_cap) < 0:
        parser.error("--scene_type_cap must be >= 0")
    if int(args.frame_type_cap) < 0:
        parser.error("--frame_type_cap must be >= 0")
    if int(args.frame_type_object_cap) < 0:
        parser.error("--frame_type_object_cap must be >= 0")
    if int(args.max_occlusion_objects) < 0:
        parser.error("--max_occlusion_objects must be >= 0")
    if args.skip_question_vlm_check:
        args.question_presence_review = False

    _set_pipeline_random_seed()

    if args.label_map:
        load_scannet_label_map(args.label_map)

    referability_cache = _load_referability_cache(
        args.referability_cache,
        repair_inconsistent_entries=args.repair_referability_cache,
        persist_repaired_entries=args.repair_referability_cache,
        no_salvage=args.no_salvage,
    )

    run_pipeline(
        data_root=Path(args.data_root),
        output_dir=Path(args.output_dir),
        dataset=args.dataset,
        scannetpp_sensor=args.scannetpp_sensor,
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
        reset=args.reset,
        only_question_types=args.only_question_types,
        scene_type_cap=args.scene_type_cap,
        frame_type_cap=args.frame_type_cap,
        frame_type_object_cap=args.frame_type_object_cap,
        max_questions_per_scene_type=args.max_questions_per_scene_type,
        max_occlusion_objects=(None if int(args.max_occlusion_objects) == 0 else int(args.max_occlusion_objects)),
        max_move_sources=(None if int(args.max_move_sources) == 0 else int(args.max_move_sources)),
    )


if __name__ == "__main__":
    main()
