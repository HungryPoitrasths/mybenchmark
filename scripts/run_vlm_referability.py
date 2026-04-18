#!/usr/bin/env python3
"""VLM-based frame-clarity and per-object referability prefilter.

This script runs *before* QA generation. For each selected frame it asks a VLM:
  1. whether the full frame looks clear overall to a human viewer;
  2. for each projected candidate object, whether its crop is clear, absent,
     or unsure for the expected label;
  3. for labels that survive crop review as uniquely grounded, whether the
     full frame still makes that label unique, multiple, absent, or unsure.

The output is a cache that can be consumed by scripts/run_pipeline.py via
--referability_cache.
"""

from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import inspect
import json
import logging
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.frame_selector import (
    VIEWPOINT_DIVERSITY_MIN_ANGLE,
    build_selector_visibility_audit_from_meta,
    compute_frame_object_visibility,
    refine_visible_ids_with_depth,
    select_frames,
)
from src.alias_groups import ALIAS_CONFIG_VERSION
from src.referability_checks import (
    normalize_label_to_object_ids as _shared_normalize_label_to_object_ids,
    normalize_object_ids as _shared_normalize_object_ids,
)
from src.scene_parser import InstanceMeshData, load_instance_mesh_data
from src.utils import RayCaster
from src.utils.colmap_loader import (
    CameraIntrinsics,
    CameraPose,
    load_axis_alignment,
    load_scannet_depth_intrinsics,
    load_scannet_intrinsics,
    load_scannet_poses,
)
from src.utils.coordinate_transform import project_to_image, world_to_camera
from src.utils.depth_occlusion import load_depth_image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("vlm_referability")

DEFAULT_VLM_URL = "http://183.129.178.195:60029/v1"
DEFAULT_VLM_MODEL = "Qwen2.5-VL-72B-Instruct"
EXCLUDED_LABELS: set[str] = set()
LABEL_BATCH_SIZE = 1
REFERABILITY_CACHE_VERSION = "19.0"

QUESTION_REVIEW_CROP_PADDING_RATIO = 0.10
QUESTION_REVIEW_CROP_MIN_PADDING_PX = 12
QUESTION_REVIEW_CROP_MAX_PADDING_PX = 80
QUESTION_REVIEW_CROP_MIN_DIM_PX = 16
QUESTION_REVIEW_CROP_MIN_PROJECTED_AREA_PX = 800.0
REFERABLE_BBOX_IN_FRAME_RATIO_MIN = 0.70
ATTACHMENT_REFERABLE_BBOX_IN_FRAME_RATIO_MIN = 0.50
SEGMENTATION_EXTREME_NOISE_MIN_AREA_PX = 100
SEGMENTATION_EXTREME_NOISE_MIN_SCORE = 0.10
SEGMENTATION_STRONG_MIN_SCORE = 0.50
SEGMENTATION_STRONG_MIN_AREA_RATIO = 0.0005
SEGMENTATION_MASK_DEDUP_IOU_THRESHOLD = 0.70
RENDER_DEPTH_TOLERANCE_M = 0.12
DEFAULT_DINOX_MODEL = "DINO-X-1.0"
DINOX_BBOX_THRESHOLD = 0.05
DINOX_MASK_THRESHOLD = 0.10
DINOX_IOU_THRESHOLD = 0.80
REFERABILITY_MESH_RAY_STAGE1_BASE_SAMPLE_COUNT = 64
REFERABILITY_MESH_RAY_STAGE2_BASE_SAMPLE_COUNT = 512
REFERABILITY_MESH_RAY_VISIBLE_RATIO_MIN = 0.10
NON_ATTACHMENT_FRAME_CANDIDATE_MULTIPLIER = 5
ATTACHMENT_GROUP_EARLY_STOP_CLARITY_SCORE = 70
NON_ATTACHMENT_GROUP_EARLY_STOP_CLARITY_SCORE = 70
FRAME_USABLE_BONUS = 100000
FRAME_SELECTION_FALLBACK_RANK = 1_000_000

OBJECT_STATUS_CLEAR = "clear"
OBJECT_STATUS_ABSENT = "absent"
OBJECT_STATUS_UNSURE = "unsure"
VALID_OBJECT_STATUSES = {
    OBJECT_STATUS_CLEAR,
    OBJECT_STATUS_ABSENT,
    OBJECT_STATUS_UNSURE,
}

LOCAL_OUTCOME_OUT_OF_FRAME = "out_of_frame"
LOCAL_OUTCOME_EXCLUDED = "excluded"
LOCAL_OUTCOME_REVIEWED = "reviewed"
OBJECT_REVIEW_MODE_VLM_CROP = "vlm_crop"
OBJECT_REVIEW_MODE_SELECTOR_DUPLICATE_SHORTCUT = "selector_duplicate_shortcut"
OBJECT_REVIEW_SKIP_SELECTOR_DUPLICATE_REASON = "selector_visible_label_multiple"

LABEL_STATUS_UNIQUE = "unique"
LABEL_STATUS_MULTIPLE = "multiple"
LABEL_STATUS_ABSENT = "absent"
LABEL_STATUS_UNSURE = "unsure"

_DINOX_CLIENT_CACHE: Any | None = None


def _invoke_method_with_supported_kwargs(method, **kwargs):
    signature = inspect.signature(method)
    parameters = signature.parameters
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values()):
        return method(**kwargs)

    supported = {
        name
        for name, param in parameters.items()
        if param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    filtered_kwargs = {
        key: value for key, value in kwargs.items() if key in supported
    }
    return method(**filtered_kwargs)


def _mesh_visibility_stats_compat(
    ray_caster: Any,
    **kwargs: Any,
) -> tuple[int, int]:
    return _invoke_method_with_supported_kwargs(
        ray_caster.mesh_visibility_stats,
        **kwargs,
    )


def _image_to_base64(image: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", image)
    if not ok:
        raise ValueError("Failed to encode image")
    return base64.b64encode(buf.tobytes()).decode()


def _extract_json_object(text: str) -> dict[str, Any] | None:
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


def _call_vlm_json(
    client,
    model: str,
    content: list[dict],
    default: dict[str, Any],
    max_tokens: int = 512,
) -> tuple[dict[str, Any], str]:
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": content}],
            max_tokens=max_tokens,
            temperature=0,
        )
        text = (resp.choices[0].message.content or "").strip()
        parsed = _extract_json_object(text)
        if parsed is None:
            return default, text
        return parsed, text
    except Exception as exc:
        logger.warning("VLM call failed: %s", exc)
        return default, ""


def _run_in_thread_pool(
    items: list[Any],
    fn: Callable[[Any], Any],
    *,
    max_workers: int,
) -> list[Any]:
    if not items:
        return []
    worker_count = max(1, int(max_workers))
    if worker_count <= 1 or len(items) <= 1:
        return [fn(item) for item in items]
    with ThreadPoolExecutor(max_workers=min(worker_count, len(items))) as executor:
        futures = [executor.submit(fn, item) for item in items]
        return [future.result() for future in futures]


def _frame_prompt() -> str:
    return (
        "You are judging only the perceived visual clarity of this image as a human viewer would. "
        "Look at the full image and decide whether it appears clear overall at normal viewing size. "
        "Ignore scene semantics, downstream task usefulness, and object categories. "
        "Focus only on whether the image looks visually clear or blurry overall. "
        "Slight softness is acceptable. "
        "Mark clear=true if a human would naturally consider the image clear overall. "
        "Mark clear=false if the image has obvious blur, defocus, motion blur, or strong softness that makes the scene look unclear overall. "
        "Return a clarity_score from 0 to 100, where higher means clearer and sharper. "
        'Answer with strict JSON only: {"clear": true, "clarity_score": 82, "reason": "overall clear with slight softness"}'
    )


def _object_review_prompt(label: str) -> str:
    return (
        "You are given two images: first the full scene image, then a crop for one candidate object. "
        "The expected label is "
        f"{json.dumps(str(label), ensure_ascii=False)}. "
        "Use the crop as the primary evidence and the full image only as context. "
        "Return clear only when the crop clearly shows an identifiable instance of that label. "
        "Return absent when the crop does not show an identifiable instance of that label. "
        "Return unsure when you cannot decide confidently. "
        'Answer with strict JSON only using this schema: {"status": "clear", "reason": "short reason"}'
    )


def _normalize_object_review_status(value: object) -> str | None:
    text = str(value or "").strip().lower()
    if text in {"clear", "present", "visible", "yes"}:
        return OBJECT_STATUS_CLEAR
    if text in {"absent", "missing", "not_present", "not present", "no"}:
        return OBJECT_STATUS_ABSENT
    if text in {"unsure", "uncertain", "unknown", "cannot_tell", "can't tell"}:
        return OBJECT_STATUS_UNSURE
    return None


def _normalize_full_frame_label_status(value: object, *, count: object = None) -> str | None:
    text = str(value or "").strip().lower()
    if text in {"unique", "one", "single", "exactly_one", "exactly one"}:
        return LABEL_STATUS_UNIQUE
    if text in {"multiple", "many", "more_than_one", "more than one", "two_or_more", "two or more"}:
        return LABEL_STATUS_MULTIPLE
    if text in {"absent", "none", "zero", "not_present", "not present"}:
        return LABEL_STATUS_ABSENT
    if text in {"unsure", "uncertain", "unknown", "unclear", "cannot_tell", "can't tell"}:
        return LABEL_STATUS_UNSURE
    try:
        count_int = int(count)
    except (TypeError, ValueError):
        return None
    if count_int <= 0:
        return LABEL_STATUS_ABSENT
    if count_int == 1:
        return LABEL_STATUS_UNIQUE
    return LABEL_STATUS_MULTIPLE


def _label_status_count(status: object) -> int | None:
    text = str(status or "").strip().lower()
    if text == LABEL_STATUS_ABSENT:
        return 0
    if text == LABEL_STATUS_UNIQUE:
        return 1
    if text == LABEL_STATUS_MULTIPLE:
        return 2
    return None


def _label_counts_from_statuses(label_statuses: dict[str, str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for label, status in sorted(label_statuses.items()):
        count = _label_status_count(status)
        if count is None:
            continue
        counts[str(label)] = int(count)
    return counts


def _merge_final_label_statuses(
    *,
    crop_label_statuses: dict[str, str],
    selector_visible_label_counts: dict[str, int],
    full_frame_label_statuses: dict[str, str],
) -> dict[str, str]:
    """Combine referable-instance review with earlier quantity evidence.

    Crop review answers "is there one referable instance left after filtering?"
    but benchmark uniqueness needs a stricter veto: if an earlier visibility pass
    already found two instances of the same label, that label cannot become
    unique later just because only one instance survived deeper filtering.
    """

    merged = {
        str(label): str(status).strip().lower()
        for label, status in crop_label_statuses.items()
        if str(label).strip() and str(status).strip()
    }

    for label, status in full_frame_label_statuses.items():
        label_key = str(label).strip().lower()
        if not label_key:
            continue
        normalized_status = str(status).strip().lower()
        if normalized_status in {
            LABEL_STATUS_UNIQUE,
            LABEL_STATUS_MULTIPLE,
            LABEL_STATUS_ABSENT,
            LABEL_STATUS_UNSURE,
        }:
            merged[label_key] = normalized_status

    for label, count in selector_visible_label_counts.items():
        label_key = str(label).strip().lower()
        if not label_key:
            continue
        try:
            count_int = int(count)
        except (TypeError, ValueError):
            continue
        if count_int >= 2 and merged.get(label_key) != LABEL_STATUS_ABSENT:
            merged[label_key] = LABEL_STATUS_MULTIPLE

    return dict(sorted(merged.items()))


def _final_referable_object_ids(
    *,
    label_statuses: dict[str, str],
    crop_unique_label_object_ids: dict[str, int],
    object_reviews: object = None,
    visibility_audit_by_object_id: object = None,
    bbox_in_frame_ratio_min: float = REFERABLE_BBOX_IN_FRAME_RATIO_MIN,
) -> list[int]:
    def _lookup_review(container: object, obj_id: int) -> dict[str, Any] | None:
        if isinstance(container, dict):
            review = container.get(str(obj_id))
            if not isinstance(review, dict):
                review = container.get(obj_id)
            return review if isinstance(review, dict) else None
        if isinstance(container, list):
            for item in container:
                if not isinstance(item, dict):
                    continue
                try:
                    item_obj_id = int(item.get("obj_id"))
                except (TypeError, ValueError):
                    continue
                if item_obj_id == int(obj_id):
                    return item
        return None

    def _passes_geometry_gate(obj_id: int) -> bool:
        for container in (object_reviews, visibility_audit_by_object_id):
            review = _lookup_review(container, int(obj_id))
            if review is None:
                continue
            try:
                ratio = float(review.get("bbox_in_frame_ratio"))
            except (TypeError, ValueError):
                continue
            return ratio >= float(bbox_in_frame_ratio_min)
        # Preserve compatibility for older/minimal cache entries that do not
        # carry per-object bbox ratios, while enforcing the gate for new ones.
        return True

    referable_object_ids: list[int] = []
    for label, obj_id in sorted(crop_unique_label_object_ids.items()):
        if str(label_statuses.get(label, "")).strip().lower() != LABEL_STATUS_UNIQUE:
            continue
        if not _passes_geometry_gate(int(obj_id)):
            continue
        referable_object_ids.append(int(obj_id))
    return sorted(set(referable_object_ids))


def _normalize_cached_object_ids(value: object) -> list[int]:
    return _shared_normalize_object_ids(value)


def _normalize_cached_label_counts(value: object) -> dict[str, int]:
    counts: dict[str, int] = {}
    if not isinstance(value, dict):
        return counts
    for label, count in value.items():
        label_key = str(label or "").strip().lower()
        if not label_key:
            continue
        try:
            count_int = int(count)
        except (TypeError, ValueError):
            continue
        counts[label_key] = max(0, count_int)
    return dict(sorted(counts.items()))


def _normalize_cached_label_statuses(
    value: object,
    *,
    counts: object = None,
) -> dict[str, str]:
    normalized: dict[str, str] = {}
    normalized_counts = _normalize_cached_label_counts(counts)
    if isinstance(value, dict):
        for label, status in value.items():
            label_key = str(label or "").strip().lower()
            if not label_key:
                continue
            normalized_status = _normalize_full_frame_label_status(
                status,
                count=normalized_counts.get(label_key),
            )
            if normalized_status is None:
                continue
            normalized[label_key] = normalized_status
    for label_key, count_int in normalized_counts.items():
        if label_key in normalized:
            continue
        normalized_status = _normalize_full_frame_label_status(None, count=count_int)
        if normalized_status is None:
            continue
        normalized[label_key] = normalized_status
    return dict(sorted(normalized.items()))


def _infer_crop_unique_label_object_ids(
    *,
    label_to_object_ids: dict[str, list[int]],
    crop_label_statuses: dict[str, str],
    crop_referable_object_ids: list[int],
) -> dict[str, int]:
    crop_referable_set = set(_normalize_cached_object_ids(crop_referable_object_ids))
    crop_unique_label_object_ids: dict[str, int] = {}
    for label, status in sorted(crop_label_statuses.items()):
        if str(status or "").strip().lower() != LABEL_STATUS_UNIQUE:
            continue
        label_object_ids = list(label_to_object_ids.get(str(label), []))
        candidate_ids = [
            int(obj_id)
            for obj_id in label_object_ids
            if int(obj_id) in crop_referable_set
        ]
        if len(candidate_ids) == 1:
            crop_unique_label_object_ids[str(label)] = int(candidate_ids[0])
            continue
        if not candidate_ids and len(label_object_ids) == 1:
            crop_unique_label_object_ids[str(label)] = int(label_object_ids[0])
    return dict(sorted(crop_unique_label_object_ids.items()))


def _selector_duplicate_shortcut_labels(
    selector_visible_label_counts: dict[str, int],
) -> set[str]:
    shortcut_labels: set[str] = set()
    for label, count in selector_visible_label_counts.items():
        label_key = str(label).strip().lower()
        if not label_key:
            continue
        try:
            count_int = int(count)
        except (TypeError, ValueError):
            continue
        if count_int >= 2:
            shortcut_labels.add(label_key)
    return shortcut_labels


def _derive_selector_duplicate_shortcut_crop_reviews(
    *,
    label_to_object_ids: dict[str, list[int]],
    selector_visible_label_counts: dict[str, int],
) -> tuple[dict[str, str], dict[str, int]]:
    crop_label_statuses: dict[str, str] = {}
    crop_label_counts: dict[str, int] = {}
    for label in sorted(_selector_duplicate_shortcut_labels(selector_visible_label_counts)):
        candidate_count = len({int(obj_id) for obj_id in label_to_object_ids.get(label, [])})
        crop_label_statuses[label] = (
            LABEL_STATUS_MULTIPLE
            if candidate_count > 0 else LABEL_STATUS_ABSENT
        )
        crop_label_counts[label] = int(candidate_count)
    return dict(sorted(crop_label_statuses.items())), dict(sorted(crop_label_counts.items()))


def _derive_crop_label_counts(
    *,
    label_to_object_ids: dict[str, list[int]],
    crop_label_statuses: dict[str, str],
    object_reviews: object = None,
) -> dict[str, int]:
    def _lookup_review(container: object, obj_id: int) -> dict[str, Any] | None:
        if isinstance(container, dict):
            review = container.get(str(obj_id))
            if not isinstance(review, dict):
                review = container.get(obj_id)
            return review if isinstance(review, dict) else None
        if isinstance(container, list):
            for item in container:
                if not isinstance(item, dict):
                    continue
                try:
                    item_obj_id = int(item.get("obj_id"))
                except (TypeError, ValueError):
                    continue
                if item_obj_id == int(obj_id):
                    return item
        return None

    crop_label_counts = _label_counts_from_statuses(crop_label_statuses)
    for label, obj_ids in sorted(label_to_object_ids.items()):
        clear_count = 0
        saw_review = False
        used_selector_duplicate_shortcut = False
        for obj_id in obj_ids:
            review = _lookup_review(object_reviews, int(obj_id))
            if review is None:
                continue
            saw_review = True
            review_mode = str(review.get("review_mode", "")).strip().lower()
            if review_mode == OBJECT_REVIEW_MODE_SELECTOR_DUPLICATE_SHORTCUT:
                used_selector_duplicate_shortcut = True
            if _effective_object_review_status(review) == OBJECT_STATUS_CLEAR:
                clear_count += 1
        if used_selector_duplicate_shortcut:
            crop_status = str(crop_label_statuses.get(str(label), "")).strip().lower()
            if crop_status == LABEL_STATUS_MULTIPLE:
                crop_label_counts[str(label)] = len({int(obj_id) for obj_id in obj_ids})
                continue
            if crop_status == LABEL_STATUS_ABSENT:
                crop_label_counts[str(label)] = 0
                continue
        if saw_review:
            crop_label_counts[str(label)] = int(clear_count)
    return dict(sorted(crop_label_counts.items()))


def _derive_final_referability_fields(entry: Any) -> dict[str, Any]:
    if not isinstance(entry, dict):
        return {}

    label_to_object_ids = _shared_normalize_label_to_object_ids(entry.get("label_to_object_ids"))
    selector_visible_label_counts = _normalize_cached_label_counts(
        entry.get("selector_visible_label_counts")
    )
    crop_label_statuses = _normalize_cached_label_statuses(
        entry.get("crop_label_statuses"),
        counts=entry.get("crop_label_counts"),
    )
    crop_label_counts = _derive_crop_label_counts(
        label_to_object_ids=label_to_object_ids,
        crop_label_statuses=crop_label_statuses,
        object_reviews=entry.get("object_reviews"),
    )
    crop_referable_object_ids = _normalize_cached_object_ids(entry.get("crop_referable_object_ids"))
    full_frame_label_statuses = _normalize_cached_label_statuses(
        entry.get("full_frame_label_statuses"),
        counts=entry.get("full_frame_label_counts"),
    )
    full_frame_label_counts = _label_counts_from_statuses(full_frame_label_statuses)
    crop_unique_label_object_ids = _infer_crop_unique_label_object_ids(
        label_to_object_ids=label_to_object_ids,
        crop_label_statuses=crop_label_statuses,
        crop_referable_object_ids=crop_referable_object_ids,
    )
    label_statuses = _merge_final_label_statuses(
        crop_label_statuses=crop_label_statuses,
        selector_visible_label_counts=selector_visible_label_counts,
        full_frame_label_statuses=full_frame_label_statuses,
    )
    label_counts = _label_counts_from_statuses(label_statuses)
    referable_object_ids = _final_referable_object_ids(
        label_statuses=label_statuses,
        crop_unique_label_object_ids=crop_unique_label_object_ids,
        object_reviews=entry.get("object_reviews"),
        visibility_audit_by_object_id=entry.get("visibility_audit_by_object_id"),
    )
    attachment_referable_object_ids = _final_referable_object_ids(
        label_statuses=label_statuses,
        crop_unique_label_object_ids=crop_unique_label_object_ids,
        object_reviews=entry.get("object_reviews"),
        visibility_audit_by_object_id=entry.get("visibility_audit_by_object_id"),
        bbox_in_frame_ratio_min=ATTACHMENT_REFERABLE_BBOX_IN_FRAME_RATIO_MIN,
    )

    return {
        "label_to_object_ids": label_to_object_ids,
        "selector_visible_label_counts": selector_visible_label_counts,
        "crop_label_statuses": crop_label_statuses,
        "crop_label_counts": crop_label_counts,
        "crop_referable_object_ids": crop_referable_object_ids,
        "full_frame_label_statuses": full_frame_label_statuses,
        "full_frame_label_counts": full_frame_label_counts,
        "label_statuses": label_statuses,
        "label_counts": label_counts,
        "attachment_referable_object_ids": attachment_referable_object_ids,
        "referable_object_ids": referable_object_ids,
        "vlm_unique_object_ids": list(referable_object_ids),
    }


def _repair_final_referability_fields(entry: Any) -> dict[str, Any]:
    if not isinstance(entry, dict):
        return {}

    repaired = dict(entry)
    repaired.update(_derive_final_referability_fields(entry))
    return repaired


def _frame_entry_has_consistent_final_fields(entry: Any) -> bool:
    if not isinstance(entry, dict):
        return False
    if not entry.get("frame_usable", True):
        return True
    required_keys = {
        "label_to_object_ids",
        "crop_label_statuses",
        "crop_label_counts",
        "crop_referable_object_ids",
        "full_frame_label_statuses",
        "full_frame_label_counts",
        "label_statuses",
        "label_counts",
        "referable_object_ids",
    }
    if not required_keys.issubset(entry.keys()):
        return False

    normalized_entry = {
        "label_to_object_ids": _shared_normalize_label_to_object_ids(entry.get("label_to_object_ids")),
        "crop_label_statuses": _normalize_cached_label_statuses(
            entry.get("crop_label_statuses"),
            counts=entry.get("crop_label_counts"),
        ),
        "crop_label_counts": _normalize_cached_label_counts(entry.get("crop_label_counts")),
        "crop_referable_object_ids": _normalize_cached_object_ids(entry.get("crop_referable_object_ids")),
        "full_frame_label_statuses": _normalize_cached_label_statuses(
            entry.get("full_frame_label_statuses"),
            counts=entry.get("full_frame_label_counts"),
        ),
        "full_frame_label_counts": _normalize_cached_label_counts(entry.get("full_frame_label_counts")),
        "label_statuses": _normalize_cached_label_statuses(
            entry.get("label_statuses"),
            counts=entry.get("label_counts"),
        ),
        "label_counts": _normalize_cached_label_counts(entry.get("label_counts")),
        "referable_object_ids": _normalize_cached_object_ids(entry.get("referable_object_ids")),
    }
    if "attachment_referable_object_ids" in entry:
        normalized_entry["attachment_referable_object_ids"] = _normalize_cached_object_ids(
            entry.get("attachment_referable_object_ids")
        )
    if "selector_visible_label_counts" in entry:
        normalized_entry["selector_visible_label_counts"] = _normalize_cached_label_counts(
            entry.get("selector_visible_label_counts")
        )
    if "vlm_unique_object_ids" in entry:
        normalized_entry["vlm_unique_object_ids"] = _normalize_cached_object_ids(
            entry.get("vlm_unique_object_ids")
        )
    expected_entry = _derive_final_referability_fields(entry)
    for key, actual_value in normalized_entry.items():
        if expected_entry.get(key) != actual_value:
            return False
    return True


def _coerce_bool(value: object, *, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    if text in {"true", "yes", "1"}:
        return True
    if text in {"false", "no", "0"}:
        return False
    return default


def _normalize_clarity_score(value: object, *, default: int = 60) -> int:
    try:
        score = int(round(float(value)))
    except (TypeError, ValueError):
        return default
    return max(0, min(100, score))


def _legacy_frame_clear(parsed: dict[str, Any]) -> bool:
    if isinstance(parsed.get("frame_usable"), bool):
        return bool(parsed.get("frame_usable"))
    if "usable_for_spatial_reasoning" in parsed or "severely_out_of_focus" in parsed:
        usable_for_spatial_reasoning = _coerce_bool(
            parsed.get("usable_for_spatial_reasoning"),
            default=True,
        )
        severely_out_of_focus = _coerce_bool(
            parsed.get("severely_out_of_focus"),
            default=False,
        )
        return usable_for_spatial_reasoning and not severely_out_of_focus
    return True


def _normalize_frame_review(value: dict[str, Any] | None) -> dict[str, Any]:
    parsed = value if isinstance(value, dict) else {}
    clear = _coerce_bool(parsed.get("clear"), default=_legacy_frame_clear(parsed))
    clarity_score = _normalize_clarity_score(parsed.get("clarity_score"), default=60)
    return {
        "clear": clear,
        "clarity_score": clarity_score,
        "frame_usable": clear,
        "reason": str(parsed.get("reason", "")).strip() or "frame_clarity_parse_fallback",
    }


def _frame_selection_score(selector_score: int, frame_info: dict[str, Any]) -> int:
    normalized = _normalize_frame_review(frame_info)
    usable_bonus = FRAME_USABLE_BONUS if normalized["frame_usable"] else 0
    return usable_bonus + int(selector_score)


def _selector_quality_pass_frame_info() -> dict[str, Any]:
    return {
        "clear": True,
        "clarity_score": 60,
        "frame_usable": True,
        "reason": "selector_image_quality_pass",
    }


def _camera_heading_distance_deg(pose_a: CameraPose, pose_b: CameraPose) -> float:
    fwd_a = pose_a.rotation.T[:, 2]
    fwd_b = pose_b.rotation.T[:, 2]
    cos_angle = np.clip(np.dot(fwd_a, fwd_b), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def _attachment_frame_sort_key(frame: dict[str, Any]) -> tuple[int, int, str]:
    return (
        -int(frame.get("attachment_referable_pair_count", 0) or 0),
        -int(frame.get("crop_ge_70_count", 0) or 0),
        str(frame.get("image_name", "")),
    )


def _build_attachment_view_groups(
    frames: list[dict[str, Any]],
    poses: dict[str, CameraPose],
) -> list[list[dict[str, Any]]]:
    groups: list[list[dict[str, Any]]] = []
    group_anchor_poses: list[CameraPose | None] = []
    for frame in frames:
        image_name = str(frame.get("image_name", "")).strip()
        pose = poses.get(image_name)
        assigned = False
        for idx, anchor_pose in enumerate(group_anchor_poses):
            if pose is None or anchor_pose is None:
                continue
            if _camera_heading_distance_deg(pose, anchor_pose) < VIEWPOINT_DIVERSITY_MIN_ANGLE:
                groups[idx].append(frame)
                assigned = True
                break
        if assigned:
            continue
        groups.append([frame])
        group_anchor_poses.append(pose)
    return groups


def _build_attachment_referable_pairs(
    attachment_graph: dict[int, list[int]] | None,
    attachment_referable_object_ids: list[int] | None,
) -> list[list[int]]:
    if not attachment_graph:
        return []

    referable_ids = {int(obj_id) for obj_id in (attachment_referable_object_ids or [])}
    if not referable_ids:
        return []

    pairs: list[list[int]] = []
    for parent_id, child_ids in sorted(attachment_graph.items()):
        parent_id_int = int(parent_id)
        if parent_id_int not in referable_ids:
            continue
        for child_id in sorted(int(value) for value in child_ids):
            if child_id in referable_ids:
                pairs.append([parent_id_int, child_id])
    return pairs


def _with_attachment_pair_metadata(
    frame: dict[str, Any],
    entry: dict[str, Any],
    attachment_graph: dict[int, list[int]] | None,
    *,
    attachment_view_group_id: int | None = None,
) -> dict[str, Any]:
    attachment_pairs = _build_attachment_referable_pairs(
        attachment_graph,
        entry.get("attachment_referable_object_ids"),
    )
    enriched = dict(frame)
    enriched["attachment_referable_pairs"] = list(attachment_pairs)
    enriched["attachment_referable_pair_count"] = len(attachment_pairs)
    enriched["attachment_view_group_id"] = attachment_view_group_id
    return enriched


def _compress_attachment_group_frames(
    frames: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    covered_pairs: set[tuple[int, int]] = set()
    kept: list[dict[str, Any]] = []
    for frame in sorted(frames, key=_attachment_frame_sort_key):
        pairs = {
            (int(pair[0]), int(pair[1]))
            for pair in frame.get("attachment_referable_pairs", [])
            if isinstance(pair, list) and len(pair) == 2
        }
        if not pairs:
            continue
        if pairs - covered_pairs:
            kept.append(frame)
            covered_pairs.update(pairs)
    return kept


def _review_frame_clarity(
    *,
    client,
    model_name: str,
    color_dir: Path,
    frame: dict[str, Any],
) -> dict[str, Any] | None:
    image_name = str(frame.get("image_name", "")).strip()
    if not image_name:
        return None
    image_path = color_dir / image_name
    image = cv2.imread(str(image_path))
    if image is None:
        logger.warning("Cannot read image %s", image_path)
        return None
    frame_info = _normalize_frame_review(_frame_decision(client, model_name, image))
    selector_score = int(frame.get("selector_score", frame.get("score", frame.get("n_visible", 0))) or 0)
    return {
        **frame,
        "selector_score": selector_score,
        "frame_info": frame_info,
        "frame_selection_score": _frame_selection_score(selector_score, frame_info),
    }


def _attachment_frame_group_key(
    frame: dict[str, Any],
    attachment_graph: dict[int, list[int]] | None,
) -> tuple[int, ...]:
    attachment_ids: set[int] = set()
    for parent_id, child_ids in (attachment_graph or {}).items():
        attachment_ids.add(int(parent_id))
        attachment_ids.update(int(child_id) for child_id in child_ids)
    return tuple(
        sorted(
            {
                int(obj_id)
                for obj_id in frame.get("visible_object_ids", [])
                if int(obj_id) in attachment_ids
            }
        )
    )


def _visible_object_frame_group_key(frame: dict[str, Any]) -> tuple[Any, ...] | None:
    visible_object_ids = frame.get("visible_object_ids")
    if isinstance(visible_object_ids, list):
        return tuple(sorted(int(obj_id) for obj_id in visible_object_ids))
    return None


def _select_attachment_group_representatives(
    *,
    client,
    model_name: str,
    scene_dir: Path,
    frames: list[dict[str, Any]],
    attachment_graph: dict[int, list[int]] | None,
    attachment_entry_builder: Callable[[dict[str, Any], dict[str, Any]], dict[str, Any] | None] | None = None,
    vlm_workers: int = 1,
) -> list[dict[str, Any]]:
    color_dir = scene_dir / "color"
    grouped_frames: dict[tuple[int, ...], list[dict[str, Any]]] = {}
    sampled_frames = list(frames[::3])
    for frame in sampled_frames:
        group_key = _attachment_frame_group_key(frame, attachment_graph)
        grouped_frames.setdefault(group_key, []).append(frame)

    def _select_group(item: tuple[int, list[dict[str, Any]]]) -> dict[str, Any] | None:
        group_id, group_frames = item
        ordered_frames = sorted(
            group_frames,
            key=lambda frame: len(frame.get("visible_object_ids", []) or []),
            reverse=True,
        )
        expected_attachment_ids = _attachment_frame_group_key(ordered_frames[0], attachment_graph) if ordered_frames else ()
        for frame in ordered_frames:
            reviewed_frame = _review_frame_clarity(
                client=client,
                model_name=model_name,
                color_dir=color_dir,
                frame=frame,
            )
            if reviewed_frame is None:
                continue
            reviewed_frame["attachment_view_group_id"] = group_id
            if int(reviewed_frame.get("frame_info", {}).get("clarity_score", 0) or 0) < ATTACHMENT_GROUP_EARLY_STOP_CLARITY_SCORE:
                continue
            if attachment_entry_builder is None:
                return reviewed_frame
            entry = attachment_entry_builder(frame, reviewed_frame)
            if not isinstance(entry, dict):
                continue
            combined = dict(reviewed_frame)
            combined.update(entry)
            actual_attachment_ids = tuple(
                sorted(int(obj_id) for obj_id in combined.get("attachment_referable_object_ids", []) or [])
            )
            if actual_attachment_ids != expected_attachment_ids:
                continue
            return _with_attachment_pair_metadata(
                combined,
                combined,
                attachment_graph,
                attachment_view_group_id=group_id,
            )
        return None

    selected = _run_in_thread_pool(
        list(enumerate(grouped_frames.values())),
        _select_group,
        max_workers=vlm_workers,
    )
    return [entry for entry in selected if entry is not None]


def _run_frame_clarity_reviews(
    *,
    client,
    model_name: str,
    scene_dir: Path,
    frames: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    color_dir = scene_dir / "color"
    reviewed: list[dict[str, Any]] = []
    for frame in frames:
        reviewed_frame = _review_frame_clarity(
            client=client,
            model_name=model_name,
            color_dir=color_dir,
            frame=frame,
        )
        if reviewed_frame is None:
            continue
        frame_info = reviewed_frame.get("frame_info", {})
        if not frame_info["frame_usable"]:
            continue
        reviewed.append(reviewed_frame)
    return reviewed


def _select_non_attachment_group_representatives(
    *,
    client,
    model_name: str,
    scene_dir: Path,
    frames: list[dict[str, Any]],
    vlm_workers: int = 1,
) -> list[dict[str, Any]]:
    if not frames:
        return []

    color_dir = scene_dir / "color"
    grouped_frames: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for frame in frames:
        group_key = _visible_object_frame_group_key(frame)
        if group_key is None:
            continue
        grouped_frames.setdefault(group_key, []).append(frame)

    def _select_group(item: tuple[tuple[Any, ...], list[dict[str, Any]]]) -> list[dict[str, Any]]:
        _group_key, group_frames = item
        kept: list[dict[str, Any]] = []
        for frame in group_frames:
            reviewed_frame = _review_frame_clarity(
                client=client,
                model_name=model_name,
                color_dir=color_dir,
                frame=frame,
            )
            if reviewed_frame is None:
                continue
            frame_info = reviewed_frame.get("frame_info", {})
            if frame_info.get("frame_usable", True):
                kept.append(reviewed_frame)
            if (
                frame_info.get("frame_usable", True)
                and int(frame_info.get("clarity_score", 0) or 0) >= NON_ATTACHMENT_GROUP_EARLY_STOP_CLARITY_SCORE
            ):
                break
        return kept

    selected_groups = _run_in_thread_pool(
        list(grouped_frames.items()),
        _select_group,
        max_workers=vlm_workers,
    )
    return [frame for group in selected_groups for frame in group]


def _select_attachment_frames_by_global_pair_coverage(
    frames: list[dict[str, Any]],
    *,
    max_frames: int,
) -> list[dict[str, Any]]:
    remaining = list(sorted(frames, key=_attachment_frame_sort_key))
    if max_frames <= 0:
        return []

    selected: list[dict[str, Any]] = []
    covered_pairs: set[tuple[int, int]] = set()
    while remaining and len(selected) < max_frames:
        best_idx = -1
        best_key: tuple[int, int, int] | None = None
        best_image_name = ""
        for idx, frame in enumerate(remaining):
            pairs = {
                (int(pair[0]), int(pair[1]))
                for pair in frame.get("attachment_referable_pairs", [])
                if isinstance(pair, list) and len(pair) == 2
            }
            new_pair_count = len(pairs - covered_pairs)
            key = (
                new_pair_count,
                int(frame.get("attachment_referable_pair_count", 0) or 0),
                int(frame.get("crop_ge_70_count", 0) or 0),
            )
            image_name = str(frame.get("image_name", ""))
            if (
                best_key is None
                or key > best_key
                or (key == best_key and image_name < best_image_name)
            ):
                best_idx = idx
                best_key = key
                best_image_name = image_name
        if best_idx < 0 or best_key is None or best_key[0] <= 0:
            break
        chosen = remaining.pop(best_idx)
        selected.append(chosen)
        covered_pairs.update(
            {
                (int(pair[0]), int(pair[1]))
                for pair in chosen.get("attachment_referable_pairs", [])
                if isinstance(pair, list) and len(pair) == 2
            }
        )

    for frame in remaining:
        if len(selected) >= max_frames:
            break
        selected.append(frame)
    return selected


def _apply_frame_review_to_entry(
    entry: dict[str, Any],
    frame_info: dict[str, Any],
) -> dict[str, Any]:
    normalized = _normalize_frame_review(frame_info)
    updated = dict(entry)
    selector_score = int(updated.get("selector_score", 0) or 0)
    updated["frame_usable"] = normalized["frame_usable"]
    updated["frame_reject_reason"] = None if normalized["frame_usable"] else normalized["reason"]
    updated["frame_quality_clear"] = _coerce_bool(
        normalized.get("clear"),
        default=bool(normalized.get("frame_usable", True)),
    )
    updated["frame_quality_score"] = _normalize_clarity_score(
        normalized.get("clarity_score"),
        default=60,
    )
    updated["frame_quality_reason"] = str(normalized.get("reason", "")).strip()
    updated["frame_selection_score"] = _frame_selection_score(selector_score, normalized)
    return updated


def _attach_selection_metadata(
    entry: dict[str, Any],
    attachment_graph: dict[int, list[int]] | None,
    *,
    final_selection_rank: int,
    attachment_view_group_id: int | None = None,
) -> dict[str, Any]:
    updated = dict(entry)
    updated["attachment_referable_pairs"] = _build_attachment_referable_pairs(
        attachment_graph,
        updated.get("attachment_referable_object_ids"),
    )
    updated["attachment_referable_pair_count"] = len(updated["attachment_referable_pairs"])
    updated["attachment_view_group_id"] = attachment_view_group_id
    updated["final_selection_rank"] = int(final_selection_rank)
    return updated


def _frame_decision(
    client,
    model: str,
    image: np.ndarray,
) -> dict[str, Any]:
    full_b64 = _image_to_base64(image)
    default = {
        "clear": True,
        "clarity_score": 60,
        "reason": "frame_clarity_parse_fallback",
    }
    parsed, _raw_text = _call_vlm_json(
        client,
        model,
        [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{full_b64}"}},
            {"type": "text", "text": _frame_prompt()},
        ],
        default=default,
        max_tokens=128,
    )
    return _normalize_frame_review({**default, **parsed})


def _object_review_decision(
    client,
    model: str,
    image_b64: str,
    crop_b64: str,
    label: str,
) -> tuple[str, str]:
    default = {"status": OBJECT_STATUS_UNSURE, "reason": "parse_fallback"}
    parsed, raw_text = _call_vlm_json(
        client,
        model,
        [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{crop_b64}"}},
            {"type": "text", "text": _object_review_prompt(label)},
        ],
        default=default,
        max_tokens=128,
    )
    status = _normalize_object_review_status(parsed.get("status")) or OBJECT_STATUS_UNSURE
    return status, raw_text


def _full_frame_label_dinox_review(
    *,
    client: object | None,
    image_path: Path,
    image_shape: tuple[int, ...],
    label: str,
) -> dict[str, Any]:
    normalized_label = str(label or "").strip().lower()
    review = {
        "backend": "dinox",
        "label": normalized_label,
        "status": LABEL_STATUS_UNSURE,
        "reason": "pending",
        "raw_detection_count": 0,
        "raw_detections": [],
        "raw_response": None,
    }
    if not normalized_label:
        review["reason"] = "missing_label"
        return review

    try:
        detections = _call_dinox_joint_detection(
            client=client,
            image_path=image_path,
            alias_variants=[normalized_label],
            image_shape=image_shape,
            targets=["bbox"],
        )
    except Exception as exc:
        review["reason"] = str(exc).strip() or "dinox_error"
        return review

    serialized_detections = [
        _serialize_detection(detection)
        for detection in detections
        if isinstance(detection, dict)
    ]
    raw_detection_count = len(serialized_detections)
    if raw_detection_count <= 0:
        status = LABEL_STATUS_ABSENT
        reason = "no_raw_detections"
    elif raw_detection_count == 1:
        status = LABEL_STATUS_UNIQUE
        reason = "single_raw_detection"
    else:
        status = LABEL_STATUS_MULTIPLE
        reason = "multiple_raw_detections"

    review.update(
        {
            "status": status,
            "reason": reason,
            "raw_detection_count": raw_detection_count,
            "raw_detections": serialized_detections,
        }
    )
    return review


def _build_frame_label_candidates(
    visible_object_ids: list[int],
    objects_by_id: dict[int, dict[str, Any]],
) -> tuple[list[str], dict[str, list[int]]]:
    label_to_ids: dict[str, list[int]] = defaultdict(list)
    for obj_id in visible_object_ids:
        obj = objects_by_id.get(int(obj_id))
        if obj is None:
            continue
        label = str(obj.get("label", "")).strip().lower()
        if not label or label in EXCLUDED_LABELS:
            continue
        label_to_ids[label].append(int(obj_id))
    normalized = {
        str(label): sorted(set(int(obj_id) for obj_id in obj_ids))
        for label, obj_ids in sorted(label_to_ids.items())
    }
    return sorted(normalized.keys()), normalized


def _refine_candidate_visible_object_ids(
    visible_object_ids: list[int],
    objects: list[dict[str, Any]],
    camera_pose,
    color_intrinsics: CameraIntrinsics | None,
    depth_image: np.ndarray | None,
    depth_intrinsics,
    ray_caster_getter: Callable[[], Any] | None = None,
    instance_mesh_data_getter: Callable[[int], InstanceMeshData] | None = None,
) -> tuple[list[int], str]:
    selector_ids = sorted({int(obj_id) for obj_id in visible_object_ids})
    if (
        selector_ids
        and color_intrinsics is not None
        and depth_image is not None
        and depth_intrinsics is not None
        and callable(ray_caster_getter)
        and callable(instance_mesh_data_getter)
    ):
        try:
            # Depth filtering is much cheaper than mesh-ray visibility. Run it
            # first and only spend mesh-ray work on objects that already pass
            # the depth gate. The final decision still requires both backends.
            depth_refined = refine_visible_ids_with_depth(
                visible_object_ids=selector_ids,
                objects=objects,
                pose=camera_pose,
                depth_image=depth_image,
                depth_intrinsics=depth_intrinsics,
            )
            if not depth_refined:
                return [], "mesh_ray_depth_refined"

            ray_caster = ray_caster_getter()
            if ray_caster is not None:
                stage1_instance_mesh_data = instance_mesh_data_getter(
                    REFERABILITY_MESH_RAY_STAGE1_BASE_SAMPLE_COUNT,
                )
                stage2_instance_mesh_data: InstanceMeshData | None = None
                mesh_ray_refined: list[int] = []
                for obj_id in depth_refined:
                    stage1 = _evaluate_crop_unique_mesh_ray_stage(
                        obj_id=int(obj_id),
                        camera_pose=camera_pose,
                        color_intrinsics=color_intrinsics,
                        ray_caster=ray_caster,
                        instance_mesh_data=stage1_instance_mesh_data,
                        base_sample_count=REFERABILITY_MESH_RAY_STAGE1_BASE_SAMPLE_COUNT,
                    )
                    if _ray_visibility_stage_passes(stage1):
                        mesh_ray_refined.append(int(obj_id))
                        continue
                    if stage2_instance_mesh_data is None:
                        stage2_instance_mesh_data = instance_mesh_data_getter(
                            REFERABILITY_MESH_RAY_STAGE2_BASE_SAMPLE_COUNT,
                        )
                    stage2 = _evaluate_crop_unique_mesh_ray_stage(
                        obj_id=int(obj_id),
                        camera_pose=camera_pose,
                        color_intrinsics=color_intrinsics,
                        ray_caster=ray_caster,
                        instance_mesh_data=stage2_instance_mesh_data,
                        base_sample_count=REFERABILITY_MESH_RAY_STAGE2_BASE_SAMPLE_COUNT,
                    )
                    if _ray_visibility_stage_passes(stage2):
                        mesh_ray_refined.append(int(obj_id))
                refined_intersection = sorted(
                    set(int(obj_id) for obj_id in mesh_ray_refined)
                    & set(int(obj_id) for obj_id in depth_refined)
                )
                return refined_intersection, "mesh_ray_depth_refined"
        except Exception as exc:
            logger.warning("Mesh-ray/depth joint refine failed: %s", exc)
    return selector_ids, "projection_fallback"


def _build_visibility_audit_by_object_id(
    scene_objects: list[dict[str, Any]],
    objects_by_id: dict[int, dict[str, Any]],
    visibility_by_obj_id: dict[int, dict[str, Any]],
    color_intrinsics: CameraIntrinsics,
    selector_visible_object_ids: list[int],
    candidate_visible_object_ids: list[int],
    candidate_visibility_source: str,
) -> dict[str, dict[str, Any]]:
    selector_set = {int(obj_id) for obj_id in selector_visible_object_ids}
    candidate_set = {int(obj_id) for obj_id in candidate_visible_object_ids}
    audit_by_obj_id: dict[str, dict[str, Any]] = {}

    for obj in scene_objects:
        obj_id = int(obj.get("id", -1))
        if obj_id < 0:
            continue
        resolved = objects_by_id.get(obj_id, obj)
        meta = visibility_by_obj_id.get(obj_id, {})
        selector_audit = build_selector_visibility_audit_from_meta(
            meta,
            color_intrinsics,
        )
        candidate_considered = obj_id in selector_set
        candidate_passed = obj_id in candidate_set
        candidate_rejection_reasons: list[str] = []
        if not candidate_considered:
            candidate_rejection_reasons.append("not_in_selector_pool")
        elif not candidate_passed:
            if candidate_visibility_source == "mesh_ray_depth_refined":
                candidate_rejection_reasons.append("mesh_ray_or_depth_not_visible")
            elif candidate_visibility_source == "projection_fallback":
                candidate_rejection_reasons.append("projection_not_promoted")
            else:
                candidate_rejection_reasons.append("not_applicable")

        audit_by_obj_id[str(obj_id)] = {
            "obj_id": obj_id,
            "label": str(resolved.get("label", "")).strip().lower(),
            **selector_audit,
            "candidate_considered": bool(candidate_considered),
            "candidate_passed": bool(candidate_passed),
            "candidate_rejection_reasons": candidate_rejection_reasons,
        }

    return audit_by_obj_id


def _count_labels_for_object_ids(
    object_ids: list[int],
    objects_by_id: dict[int, dict[str, Any]],
) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for obj_id in object_ids:
        obj = objects_by_id.get(int(obj_id))
        if obj is None:
            continue
        label = str(obj.get("label", "")).strip().lower()
        if not label:
            continue
        counts[label] += 1
    return dict(sorted(counts.items()))


def _safe_float(value: object, *, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _safe_int(value: object, *, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _normalize_alias_variants(values: list[str] | tuple[str, ...] | set[str] | None) -> list[str]:
    variants: list[str] = []
    seen: set[str] = set()
    for value in values or []:
        text = re.sub(r"\s+", " ", str(value or "").strip().lower())
        if not text or text in seen:
            continue
        seen.add(text)
        variants.append(text)
    return variants


def _strong_detection_min_area(image_shape: tuple[int, ...]) -> int:
    height = int(image_shape[0]) if len(image_shape) >= 1 else 0
    width = int(image_shape[1]) if len(image_shape) >= 2 else 0
    return int(max(QUESTION_REVIEW_CROP_MIN_PROJECTED_AREA_PX, round(SEGMENTATION_STRONG_MIN_AREA_RATIO * width * height)))


def _extract_sdk_field(obj: object, *names: str, default: Any = None) -> Any:
    for name in names:
        if isinstance(obj, dict) and name in obj:
            return obj.get(name)
        value = getattr(obj, name, None)
        if value is not None:
            return value
    return default


def _decode_coco_rle_mask(
    mask_data: object,
    *,
    height: int,
    width: int,
) -> np.ndarray | None:
    if isinstance(mask_data, np.ndarray):
        mask = np.asarray(mask_data)
        if mask.ndim != 2:
            return None
        return (mask > 0).astype(np.uint8)

    if not isinstance(mask_data, dict):
        return None

    size = mask_data.get("size")
    if isinstance(size, (list, tuple)) and len(size) == 2:
        height = int(size[0])
        width = int(size[1])

    counts = mask_data.get("counts")
    if counts is None or height <= 0 or width <= 0:
        return None

    try:
        from pycocotools import mask as mask_utils  # type: ignore

        decoded = mask_utils.decode(
            {
                "size": [int(height), int(width)],
                "counts": counts,
            }
        )
        if decoded.ndim == 3:
            decoded = decoded[..., 0]
        return (np.asarray(decoded) > 0).astype(np.uint8)
    except ImportError:
        if isinstance(counts, str):
            raise RuntimeError(
                "pycocotools is required to decode compressed COCO RLE masks from DINO-X",
            )

    if not isinstance(counts, (list, tuple)):
        return None

    flat = np.zeros(int(height) * int(width), dtype=np.uint8)
    index = 0
    value = 0
    for count in counts:
        run_length = max(0, _safe_int(count))
        if run_length <= 0:
            value = 1 - value
            continue
        end = min(index + run_length, flat.size)
        if value == 1 and end > index:
            flat[index:end] = 1
        index = end
        value = 1 - value
        if index >= flat.size:
            break
    return flat.reshape((int(height), int(width)), order="F")


def _bbox_to_mask(bbox: list[float], *, height: int, width: int) -> np.ndarray:
    mask = np.zeros((int(height), int(width)), dtype=np.uint8)
    if len(bbox) != 4:
        return mask
    x1, y1, x2, y2 = [_safe_float(value) for value in bbox]
    u0 = max(0, min(int(np.floor(min(x1, x2))), int(width)))
    u1 = max(0, min(int(np.ceil(max(x1, x2))), int(width)))
    v0 = max(0, min(int(np.floor(min(y1, y2))), int(height)))
    v1 = max(0, min(int(np.ceil(max(y1, y2))), int(height)))
    if u1 <= u0 or v1 <= v0:
        return mask
    mask[v0:v1, u0:u1] = 1
    return mask


def _normalize_dinox_detection(
    raw_detection: object,
    *,
    image_shape: tuple[int, ...],
) -> dict[str, Any] | None:
    height = int(image_shape[0]) if len(image_shape) >= 1 else 0
    width = int(image_shape[1]) if len(image_shape) >= 2 else 0
    bbox_raw = _extract_sdk_field(raw_detection, "bbox", "box")
    bbox = (
        [float(value) for value in bbox_raw]
        if isinstance(bbox_raw, (list, tuple)) and len(bbox_raw) == 4
        else None
    )
    mask_data = _extract_sdk_field(raw_detection, "mask", "segmentation")
    mask = _decode_coco_rle_mask(mask_data, height=height, width=width)
    if mask is None and bbox is not None:
        mask = _bbox_to_mask(bbox, height=height, width=width)
    if mask is None:
        return None

    mask_bool = np.asarray(mask > 0, dtype=bool)
    area_px = int(mask_bool.sum())
    if area_px <= 0:
        return None

    return {
        "bbox": bbox,
        "mask": mask_bool,
        "score": _safe_float(_extract_sdk_field(raw_detection, "score", "confidence"), default=0.0),
        "category": str(_extract_sdk_field(raw_detection, "category", "label", "text", default="")).strip().lower(),
        "area_px": area_px,
    }


def _resolve_dinox_api_key() -> str:
    for env_name in (
        "DDS_API_TOKEN",
        "DDS_CLOUD_API_TOKEN",
        "DEEPDATASPACE_API_TOKEN",
        "DASHSCOPE_API_KEY",
    ):
        token = os.getenv(env_name)
        if token:
            return token
    raise RuntimeError(
        "DINO-X segmentation requires DDS_API_TOKEN/DDS_CLOUD_API_TOKEN/DEEPDATASPACE_API_TOKEN",
    )


def _get_dinox_client(existing_client: object | None = None) -> Any:
    global _DINOX_CLIENT_CACHE
    if existing_client is not None and hasattr(existing_client, "run_task") and hasattr(existing_client, "upload_file"):
        return existing_client
    if _DINOX_CLIENT_CACHE is not None:
        return _DINOX_CLIENT_CACHE

    try:
        from dds_cloudapi_sdk import Client, Config  # type: ignore
    except ImportError as exc:
        raise RuntimeError("dds-cloudapi-sdk is required for DINO-X referability segmentation") from exc

    _DINOX_CLIENT_CACHE = Client(Config(_resolve_dinox_api_key()))
    return _DINOX_CLIENT_CACHE


def _call_dinox_joint_detection(
    *,
    client: object | None,
    image_path: Path,
    alias_variants: list[str],
    image_shape: tuple[int, ...],
    targets: list[str] | tuple[str, ...] | None = None,
) -> list[dict[str, Any]]:
    normalized_variants = _normalize_alias_variants(alias_variants)
    if not normalized_variants:
        return []
    normalized_targets = [
        str(target).strip().lower()
        for target in (targets if targets is not None else ["bbox", "mask"])
        if str(target).strip()
    ]
    if not normalized_targets:
        normalized_targets = ["bbox", "mask"]

    try:
        from dds_cloudapi_sdk.tasks.v2_task import create_task_with_local_image_auto_resize  # type: ignore
    except ImportError as exc:
        raise RuntimeError("dds-cloudapi-sdk is required for DINO-X referability segmentation") from exc

    prompt_text = ".".join(normalized_variants)
    cloud_client = _get_dinox_client(client)
    api_body_without_image: dict[str, Any] = {
        "model": DEFAULT_DINOX_MODEL,
        "prompt": {
            "type": "text",
            "text": prompt_text,
        },
        "targets": normalized_targets,
        "bbox_threshold": DINOX_BBOX_THRESHOLD,
        "iou_threshold": DINOX_IOU_THRESHOLD,
    }
    if "mask" in normalized_targets:
        api_body_without_image["mask_format"] = "coco_rle"
    task = create_task_with_local_image_auto_resize(
        api_path="/v2/task/dinox/detection",
        api_body_without_image=api_body_without_image,
        image_path=str(image_path),
    )
    cloud_client.run_task(task)

    result = getattr(task, "result", None)
    raw_objects = _extract_sdk_field(result, "objects", default=[])
    detections: list[dict[str, Any]] = []
    for raw_detection in raw_objects or []:
        normalized = _normalize_dinox_detection(
            raw_detection,
            image_shape=image_shape,
        )
        if normalized is not None:
            detections.append(normalized)
    return detections


def _mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = np.asarray(mask_a, dtype=bool)
    b = np.asarray(mask_b, dtype=bool)
    union = np.logical_or(a, b).sum()
    if union <= 0:
        return 0.0
    intersection = np.logical_and(a, b).sum()
    return float(intersection / union)


def _serialize_detection(detection: dict[str, Any]) -> dict[str, Any]:
    bbox = detection.get("bbox")
    return {
        "bbox": [float(value) for value in bbox] if isinstance(bbox, list) else None,
        "score": float(detection.get("score", 0.0) or 0.0),
        "area_px": int(detection.get("area_px", 0) or 0),
        "category": str(detection.get("category", "")).strip().lower(),
    }


def _dedupe_detections_by_mask_iou(
    detections: list[dict[str, Any]],
    *,
    iou_threshold: float = SEGMENTATION_MASK_DEDUP_IOU_THRESHOLD,
) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for detection in sorted(
        detections,
        key=lambda item: (
            float(item.get("score", 0.0) or 0.0),
            int(item.get("area_px", 0) or 0),
        ),
        reverse=True,
    ):
        mask = detection.get("mask")
        if not isinstance(mask, np.ndarray):
            continue
        if any(_mask_iou(mask, existing["mask"]) >= float(iou_threshold) for existing in kept):
            continue
        kept.append(detection)
    return kept


def _build_scene_alias_group_index(
    scene_objects: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    alias_groups: dict[str, dict[str, Any]] = {}
    for obj in scene_objects:
        label = str(obj.get("label", "")).strip().lower()
        alias_group = str(obj.get("alias_group", "")).strip().lower()
        if not label or not alias_group:
            continue
        entry = alias_groups.setdefault(
            alias_group,
            {
                "alias_group": alias_group,
                "object_ids": [],
                "canonical_labels": set(),
                "alias_variants": [],
                "raw_labels": [],
            },
        )
        entry["object_ids"].append(int(obj.get("id")))
        entry["canonical_labels"].add(label)
        entry["alias_variants"].extend(obj.get("alias_variants", []) or [label])
        raw_label = str(obj.get("raw_label", "")).strip().lower()
        if raw_label:
            entry["raw_labels"].append(raw_label)

    normalized: dict[str, dict[str, Any]] = {}
    for alias_group, entry in sorted(alias_groups.items()):
        normalized[alias_group] = {
            "alias_group": alias_group,
            "object_ids": sorted(set(int(obj_id) for obj_id in entry["object_ids"])),
            "canonical_labels": sorted(str(label) for label in entry["canonical_labels"]),
            "alias_variants": _normalize_alias_variants(entry["alias_variants"] + entry["raw_labels"]),
        }
    return normalized


def _build_compat_label_to_object_ids(
    scene_objects: list[dict[str, Any]],
    alias_group_index: dict[str, dict[str, Any]],
) -> dict[str, list[int]]:
    label_to_object_ids: dict[str, list[int]] = {}
    for obj in scene_objects:
        label = str(obj.get("label", "")).strip().lower()
        alias_group = str(obj.get("alias_group", "")).strip().lower()
        if not label or not alias_group:
            continue
        family_ids = alias_group_index.get(alias_group, {}).get("object_ids", [])
        label_to_object_ids[label] = [int(obj_id) for obj_id in family_ids]
    return dict(sorted(label_to_object_ids.items()))


def _compute_topology_quality_for_object(
    *,
    obj_id: int,
    instance_mesh_data: InstanceMeshData | None,
) -> dict[str, Any]:
    triangle_ids = np.asarray(
        sorted(_instance_triangle_id_set(instance_mesh_data, int(obj_id))),
        dtype=np.int64,
    )
    if instance_mesh_data is None or len(triangle_ids) == 0:
        return {
            "obj_id": int(obj_id),
            "triangle_count": int(len(triangle_ids)),
            "connected_component_count": 0,
            "largest_component_triangle_share": 0.0,
            "boundary_edge_ratio": 1.0,
            "num_boundary_loops": 0,
            "largest_boundary_loop_edge_share": 1.0,
            "status": "fail",
            "reason_codes": ["no_triangles"],
        }

    faces = np.asarray(instance_mesh_data.faces, dtype=np.int64)[triangle_ids]
    edge_to_triangles: dict[tuple[int, int], list[int]] = defaultdict(list)
    triangle_neighbors: dict[int, set[int]] = {
        int(local_idx): set()
        for local_idx in range(len(faces))
    }

    for local_idx, face in enumerate(faces):
        a, b, c = [int(value) for value in face]
        for edge in ((a, b), (b, c), (c, a)):
            normalized_edge = tuple(sorted(edge))
            edge_to_triangles[normalized_edge].append(int(local_idx))

    for triangle_indices in edge_to_triangles.values():
        if len(triangle_indices) < 2:
            continue
        for triangle_idx in triangle_indices:
            triangle_neighbors[int(triangle_idx)].update(
                other_idx for other_idx in triangle_indices if int(other_idx) != int(triangle_idx)
            )

    component_sizes: list[int] = []
    visited_triangles: set[int] = set()
    for start_idx in range(len(faces)):
        if start_idx in visited_triangles:
            continue
        stack = [int(start_idx)]
        visited_triangles.add(int(start_idx))
        component_size = 0
        while stack:
            current = stack.pop()
            component_size += 1
            for neighbor in triangle_neighbors.get(int(current), set()):
                if neighbor in visited_triangles:
                    continue
                visited_triangles.add(int(neighbor))
                stack.append(int(neighbor))
        component_sizes.append(int(component_size))

    boundary_edges = [edge for edge, owner_triangles in edge_to_triangles.items() if len(owner_triangles) == 1]
    boundary_vertices: dict[int, set[int]] = defaultdict(set)
    for v0, v1 in boundary_edges:
        boundary_vertices[int(v0)].add(int(v1))
        boundary_vertices[int(v1)].add(int(v0))

    boundary_component_edge_counts: list[int] = []
    visited_boundary_vertices: set[int] = set()
    for start_vertex in list(boundary_vertices.keys()):
        if start_vertex in visited_boundary_vertices:
            continue
        stack = [int(start_vertex)]
        component_vertices: set[int] = set()
        component_edges: set[tuple[int, int]] = set()
        visited_boundary_vertices.add(int(start_vertex))
        while stack:
            current = stack.pop()
            component_vertices.add(int(current))
            for neighbor in boundary_vertices.get(int(current), set()):
                component_edges.add(tuple(sorted((int(current), int(neighbor)))))
                if neighbor in visited_boundary_vertices:
                    continue
                visited_boundary_vertices.add(int(neighbor))
                stack.append(int(neighbor))
        if component_vertices:
            boundary_component_edge_counts.append(len(component_edges))

    triangle_count = int(len(faces))
    total_unique_edges = int(len(edge_to_triangles))
    connected_component_count = int(len(component_sizes))
    largest_component_triangle_share = (
        float(max(component_sizes) / triangle_count) if triangle_count > 0 and component_sizes else 0.0
    )
    boundary_edge_ratio = (
        float(len(boundary_edges) / total_unique_edges) if total_unique_edges > 0 else 1.0
    )
    largest_boundary_loop_edge_share = (
        float(max(boundary_component_edge_counts) / len(boundary_edges))
        if boundary_edges and boundary_component_edge_counts
        else 0.0
    )
    num_boundary_loops = int(len(boundary_component_edge_counts))

    fail_reason_codes: list[str] = []
    if triangle_count < 16:
        fail_reason_codes.append("too_few_triangles")
    if largest_component_triangle_share < 0.45:
        fail_reason_codes.append("fragmented_components")
    if connected_component_count > 6 and boundary_edge_ratio > 0.55:
        fail_reason_codes.append("many_components_with_high_boundary_ratio")
    if boundary_edge_ratio > 0.75 and largest_boundary_loop_edge_share > 0.45:
        fail_reason_codes.append("large_open_boundary")

    warn_reason_codes: list[str] = []
    if connected_component_count > 3:
        warn_reason_codes.append("component_count_warn")
    if boundary_edge_ratio > 0.45:
        warn_reason_codes.append("boundary_edge_ratio_warn")
    if largest_boundary_loop_edge_share > 0.25:
        warn_reason_codes.append("largest_boundary_loop_warn")

    if fail_reason_codes:
        status = "fail"
        reason_codes = fail_reason_codes
    elif warn_reason_codes:
        status = "warn"
        reason_codes = warn_reason_codes
    else:
        status = "pass"
        reason_codes = []

    return {
        "obj_id": int(obj_id),
        "triangle_count": triangle_count,
        "connected_component_count": connected_component_count,
        "largest_component_triangle_share": float(largest_component_triangle_share),
        "boundary_edge_ratio": float(boundary_edge_ratio),
        "num_boundary_loops": num_boundary_loops,
        "largest_boundary_loop_edge_share": float(largest_boundary_loop_edge_share),
        "status": status,
        "reason_codes": reason_codes,
    }


def _project_vertices_to_image(
    vertices: np.ndarray,
    camera_pose: CameraPose,
    intrinsics: CameraIntrinsics,
) -> tuple[np.ndarray, np.ndarray]:
    world_vertices = np.asarray(vertices, dtype=np.float64)
    camera_vertices = world_vertices @ np.asarray(camera_pose.rotation, dtype=np.float64).T + np.asarray(
        camera_pose.translation,
        dtype=np.float64,
    )
    depths = camera_vertices[:, 2]
    uv = np.full((len(world_vertices), 2), np.nan, dtype=np.float64)
    positive_depth = depths > 1e-6
    if np.any(positive_depth):
        uv[positive_depth, 0] = (
            intrinsics.fx * camera_vertices[positive_depth, 0] / depths[positive_depth]
        ) + intrinsics.cx
        uv[positive_depth, 1] = (
            intrinsics.fy * camera_vertices[positive_depth, 1] / depths[positive_depth]
        ) + intrinsics.cy
    return uv, depths


def _rasterize_instance_depth_map(
    *,
    obj_id: int,
    camera_pose: CameraPose,
    intrinsics: CameraIntrinsics,
    instance_mesh_data: InstanceMeshData | None,
) -> dict[str, Any]:
    height = int(intrinsics.height)
    width = int(intrinsics.width)
    depth_buffer = np.full((height, width), np.inf, dtype=np.float32)

    triangle_ids = sorted(_instance_triangle_id_set(instance_mesh_data, int(obj_id)))
    if instance_mesh_data is None or not triangle_ids:
        return {
            "mask": np.zeros((height, width), dtype=bool),
            "depth": depth_buffer,
            "triangle_count": 0,
        }

    vertices = np.asarray(instance_mesh_data.vertices, dtype=np.float64)
    faces = np.asarray(instance_mesh_data.faces, dtype=np.int64)
    projected_uv, projected_depths = _project_vertices_to_image(vertices, camera_pose, intrinsics)

    for triangle_id in triangle_ids:
        tri_indices = faces[int(triangle_id)]
        tri_depths = projected_depths[tri_indices]
        if np.any(tri_depths <= 1e-6):
            continue
        tri_uv = projected_uv[tri_indices]
        if np.any(np.isnan(tri_uv)):
            continue

        xs = tri_uv[:, 0]
        ys = tri_uv[:, 1]
        if float(np.max(xs)) < 0 or float(np.max(ys)) < 0:
            continue
        if float(np.min(xs)) >= width or float(np.min(ys)) >= height:
            continue

        x_min = max(int(np.floor(float(np.min(xs)))), 0)
        x_max = min(int(np.ceil(float(np.max(xs)))), width - 1)
        y_min = max(int(np.floor(float(np.min(ys)))), 0)
        y_max = min(int(np.ceil(float(np.max(ys)))), height - 1)
        if x_max < x_min or y_max < y_min:
            continue

        x0, y0 = tri_uv[0]
        x1, y1 = tri_uv[1]
        x2, y2 = tri_uv[2]
        denominator = ((y1 - y2) * (x0 - x2)) + ((x2 - x1) * (y0 - y2))
        if abs(float(denominator)) < 1e-12:
            continue

        grid_x, grid_y = np.meshgrid(
            np.arange(x_min, x_max + 1, dtype=np.float64) + 0.5,
            np.arange(y_min, y_max + 1, dtype=np.float64) + 0.5,
        )
        w0 = (((y1 - y2) * (grid_x - x2)) + ((x2 - x1) * (grid_y - y2))) / denominator
        w1 = (((y2 - y0) * (grid_x - x2)) + ((x0 - x2) * (grid_y - y2))) / denominator
        w2 = 1.0 - w0 - w1
        inside = (w0 >= -1e-6) & (w1 >= -1e-6) & (w2 >= -1e-6)
        if not np.any(inside):
            continue

        tri_depth_map = (w0 * tri_depths[0]) + (w1 * tri_depths[1]) + (w2 * tri_depths[2])
        target_slice = depth_buffer[y_min:y_max + 1, x_min:x_max + 1]
        update_mask = inside & (tri_depth_map < target_slice)
        if np.any(update_mask):
            target_slice[update_mask] = tri_depth_map[update_mask].astype(np.float32)

    return {
        "mask": np.isfinite(depth_buffer),
        "depth": depth_buffer,
        "triangle_count": len(triangle_ids),
    }


def _compute_depth_bad_ratio(
    *,
    obj_id: int,
    camera_pose: CameraPose,
    depth_image: np.ndarray | None,
    depth_intrinsics: CameraIntrinsics | None,
    instance_mesh_data: InstanceMeshData | None,
) -> float | None:
    if depth_image is None or depth_intrinsics is None:
        return None

    rendered = _rasterize_instance_depth_map(
        obj_id=int(obj_id),
        camera_pose=camera_pose,
        intrinsics=depth_intrinsics,
        instance_mesh_data=instance_mesh_data,
    )
    render_mask = np.asarray(rendered["mask"], dtype=bool)
    rendered_depth = np.asarray(rendered["depth"], dtype=np.float32)
    if depth_image.shape[:2] != render_mask.shape[:2]:
        return None

    valid = render_mask & np.isfinite(rendered_depth) & (np.asarray(depth_image) > 0)
    if not np.any(valid):
        return None

    depth_delta = np.abs(rendered_depth[valid] - np.asarray(depth_image, dtype=np.float32)[valid])
    return float(np.mean(depth_delta > RENDER_DEPTH_TOLERANCE_M))


def _mesh_quality_thresholds_for_topology_status(topology_status: str) -> dict[str, float]:
    normalized_status = str(topology_status or "").strip().lower()
    if normalized_status == "warn":
        return {
            "iou_min": 0.50,
            "under_coverage_max": 0.35,
            "over_coverage_max": 0.30,
            "area_ratio_min": 0.60,
            "area_ratio_max": 1.60,
            "depth_bad_ratio_max": 0.20,
        }
    return {
        "iou_min": 0.45,
        "under_coverage_max": 0.45,
        "over_coverage_max": 0.35,
        "area_ratio_min": 0.55,
        "area_ratio_max": 1.80,
        "depth_bad_ratio_max": 0.25,
    }


def _compute_mesh_mask_quality_for_object(
    *,
    obj_id: int,
    detection_mask: np.ndarray,
    topology_status: str,
    camera_pose: CameraPose,
    color_intrinsics: CameraIntrinsics,
    depth_image: np.ndarray | None,
    depth_intrinsics: CameraIntrinsics | None,
    instance_mesh_data: InstanceMeshData | None,
) -> dict[str, Any]:
    rendered = _rasterize_instance_depth_map(
        obj_id=int(obj_id),
        camera_pose=camera_pose,
        intrinsics=color_intrinsics,
        instance_mesh_data=instance_mesh_data,
    )
    mesh_mask = np.asarray(rendered["mask"], dtype=bool)
    img_mask = np.asarray(detection_mask, dtype=bool)
    image_mask_area_px = int(img_mask.sum())
    mesh_mask_area_px = int(mesh_mask.sum())
    intersection_px = int(np.logical_and(mesh_mask, img_mask).sum())
    union_px = int(np.logical_or(mesh_mask, img_mask).sum())

    iou = float(intersection_px / union_px) if union_px > 0 else 0.0
    under_coverage = (
        float((image_mask_area_px - intersection_px) / image_mask_area_px)
        if image_mask_area_px > 0
        else 1.0
    )
    over_coverage = (
        float((mesh_mask_area_px - intersection_px) / mesh_mask_area_px)
        if mesh_mask_area_px > 0
        else 1.0
    )
    area_ratio = (
        float(mesh_mask_area_px / image_mask_area_px)
        if image_mask_area_px > 0
        else float("inf")
    )
    depth_bad_ratio = _compute_depth_bad_ratio(
        obj_id=int(obj_id),
        camera_pose=camera_pose,
        depth_image=depth_image,
        depth_intrinsics=depth_intrinsics,
        instance_mesh_data=instance_mesh_data,
    )

    thresholds = _mesh_quality_thresholds_for_topology_status(topology_status)
    reason_codes: list[str] = []
    if mesh_mask_area_px <= 0:
        reason_codes.append("mesh_projects_out_of_frame")
    if iou < thresholds["iou_min"]:
        reason_codes.append("low_iou")
    if under_coverage > thresholds["under_coverage_max"]:
        reason_codes.append("high_under_coverage")
    if over_coverage > thresholds["over_coverage_max"]:
        reason_codes.append("high_over_coverage")
    if area_ratio < thresholds["area_ratio_min"] or area_ratio > thresholds["area_ratio_max"]:
        reason_codes.append("bad_area_ratio")
    if depth_bad_ratio is not None and depth_bad_ratio > thresholds["depth_bad_ratio_max"]:
        reason_codes.append("high_depth_bad_ratio")

    return {
        "obj_id": int(obj_id),
        "status": "pass" if not reason_codes else "fail",
        "profile": "topology_warn_strict" if str(topology_status).strip().lower() == "warn" else "topology_pass_base",
        "image_mask_area_px": image_mask_area_px,
        "mesh_mask_area_px": mesh_mask_area_px,
        "intersection_px": intersection_px,
        "union_px": union_px,
        "iou": float(iou),
        "under_coverage": float(under_coverage),
        "over_coverage": float(over_coverage),
        "area_ratio": float(area_ratio),
        "depth_bad_ratio": None if depth_bad_ratio is None else float(depth_bad_ratio),
        "reason_codes": reason_codes,
        "thresholds": thresholds,
    }


def _build_object_review_records(
    *,
    scene_objects: list[dict[str, Any]],
    visibility_by_obj_id: dict[int, dict[str, Any]],
    candidate_visible_object_ids: list[int],
    topology_quality_by_obj_id: dict[int, dict[str, Any]],
    anchor_candidate_ids_by_alias_group: dict[str, list[int]],
    mesh_mask_quality_by_obj_id: dict[int, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    candidate_set = {int(obj_id) for obj_id in candidate_visible_object_ids}
    object_reviews: dict[str, dict[str, Any]] = {}

    for obj in scene_objects:
        obj_id = int(obj.get("id", -1))
        if obj_id < 0:
            continue
        visibility_meta = visibility_by_obj_id.get(obj_id, {})
        topology_quality = topology_quality_by_obj_id.get(obj_id, {})
        mesh_quality = mesh_mask_quality_by_obj_id.get(obj_id, {})
        alias_group = str(obj.get("alias_group", "")).strip().lower()
        anchor_candidate_ids = set(anchor_candidate_ids_by_alias_group.get(alias_group, []))
        object_reviews[str(obj_id)] = {
            "obj_id": obj_id,
            "label": str(obj.get("label", "")).strip().lower(),
            "alias_group": alias_group,
            "bbox_in_frame_ratio": _safe_float(visibility_meta.get("bbox_in_frame_ratio"), default=0.0),
            "projected_area_px": _safe_float(visibility_meta.get("projected_area_px"), default=0.0),
            "roi_bounds_px": visibility_meta.get("roi_bounds_px"),
            "candidate_visible": obj_id in candidate_set,
            "topology_status": str(topology_quality.get("status", "")).strip().lower() or None,
            "topology_reason_codes": list(topology_quality.get("reason_codes", [])),
            "anchor_candidate": obj_id in anchor_candidate_ids,
            "mesh_mask_status": str(mesh_quality.get("status", "")).strip().lower() or None,
            "mesh_mask_reason_codes": list(mesh_quality.get("reason_codes", [])),
            "mesh_mask_iou": mesh_quality.get("iou"),
            "mesh_mask_under_coverage": mesh_quality.get("under_coverage"),
            "mesh_mask_over_coverage": mesh_quality.get("over_coverage"),
            "mesh_mask_area_ratio": mesh_quality.get("area_ratio"),
            "mesh_mask_depth_bad_ratio": mesh_quality.get("depth_bad_ratio"),
        }
    return object_reviews


def _build_object_review_crop(
    image: np.ndarray,
    visibility_meta: dict[str, Any] | None,
) -> dict[str, Any]:
    meta = visibility_meta or {}
    roi_bounds = meta.get("roi_bounds_px")
    projected_area_px = float(meta.get("projected_area_px", 0.0) or 0.0)
    bbox_in_frame_ratio = float(meta.get("bbox_in_frame_ratio", 0.0) or 0.0)
    edge_margin_px = float(meta.get("edge_margin_px", 0.0) or 0.0)
    zbuffer_mask_area_px = float(meta.get("zbuffer_mask_area_px", 0.0) or 0.0)
    has_zbuffer_mask_area = bool(meta.get("has_zbuffer_mask_area", False))
    result = {
        "valid": False,
        "local_outcome": LOCAL_OUTCOME_OUT_OF_FRAME,
        "reason": "missing_projection",
        "roi_bounds_px": None,
        "crop_bounds_px": None,
        "projected_area_px": projected_area_px,
        "bbox_in_frame_ratio": bbox_in_frame_ratio,
        "edge_margin_px": edge_margin_px,
        "zbuffer_mask_area_px": zbuffer_mask_area_px,
        "has_zbuffer_mask_area": has_zbuffer_mask_area,
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

    pad = int(
        round(
            max(
                QUESTION_REVIEW_CROP_MIN_PADDING_PX,
                min(
                    QUESTION_REVIEW_CROP_PADDING_RATIO * max(width, height),
                    QUESTION_REVIEW_CROP_MAX_PADDING_PX,
                ),
            )
        )
    )
    crop_u_min = max(0, u_min - pad)
    crop_u_max = min(int(image.shape[1]), u_max + pad)
    crop_v_min = max(0, v_min - pad)
    crop_v_max = min(int(image.shape[0]), v_max + pad)
    crop_width = max(0, crop_u_max - crop_u_min)
    crop_height = max(0, crop_v_max - crop_v_min)
    result["roi_bounds_px"] = [u_min, u_max, v_min, v_max]
    result["crop_bounds_px"] = [crop_u_min, crop_u_max, crop_v_min, crop_v_max]

    if crop_width < QUESTION_REVIEW_CROP_MIN_DIM_PX or crop_height < QUESTION_REVIEW_CROP_MIN_DIM_PX:
        result["local_outcome"] = LOCAL_OUTCOME_EXCLUDED
        result["reason"] = "crop_too_small"
        return result
    if projected_area_px < QUESTION_REVIEW_CROP_MIN_PROJECTED_AREA_PX:
        result["local_outcome"] = LOCAL_OUTCOME_EXCLUDED
        result["reason"] = "projected_area_too_small"
        return result
    crop_image = image[crop_v_min:crop_v_max, crop_u_min:crop_u_max]
    if crop_image.size == 0:
        return result

    result["valid"] = True
    result["local_outcome"] = LOCAL_OUTCOME_REVIEWED
    result["reason"] = ""
    result["image_b64"] = _image_to_base64(crop_image)
    return result


def _build_object_review_entry(
    *,
    obj_id: int,
    label: str,
    crop_entry: dict[str, Any],
) -> dict[str, Any]:
    return {
        "obj_id": int(obj_id),
        "label": str(label).strip().lower(),
        "review_mode": OBJECT_REVIEW_MODE_VLM_CROP,
        "review_skip_reason": None,
        "local_outcome": str(crop_entry.get("local_outcome", "")),
        "local_reason": str(crop_entry.get("reason", "")),
        "vlm_status": None,
        "raw_response": None,
        "roi_bounds_px": crop_entry.get("roi_bounds_px"),
        "crop_bounds_px": crop_entry.get("crop_bounds_px"),
        "projected_area_px": crop_entry.get("projected_area_px"),
        "bbox_in_frame_ratio": crop_entry.get("bbox_in_frame_ratio"),
        "edge_margin_px": crop_entry.get("edge_margin_px"),
        "zbuffer_mask_area_px": crop_entry.get("zbuffer_mask_area_px"),
        "has_zbuffer_mask_area": bool(crop_entry.get("has_zbuffer_mask_area", False)),
        "topology_status": None,
        "topology_reason_codes": [],
        "mesh_mask_status": None,
        "mesh_mask_reason_codes": [],
        "mesh_mask_iou": None,
        "mesh_mask_under_coverage": None,
        "mesh_mask_over_coverage": None,
        "mesh_mask_area_ratio": None,
        "mesh_mask_depth_bad_ratio": None,
        "ray_visibility_review": {
            "applied": False,
            "decision": "not_applicable",
            "reason": "not_crop_unique",
            "stage1": None,
            "stage2": None,
        },
        "mesh_quality_review": {
            "applied": False,
            "decision": "not_applicable",
            "reason": "not_crop_unique",
            "detection_prompt_variants": [],
            "raw_detection_count": 0,
            "candidate_detection_count": 0,
            "matched_detection": None,
        },
    }


def _effective_object_review_status(review: dict[str, Any]) -> str | None:
    status = _normalize_object_review_status(review.get("vlm_status"))
    ray_review = review.get("ray_visibility_review")
    if (
        status == OBJECT_STATUS_CLEAR
        and isinstance(ray_review, dict)
        and str(ray_review.get("decision", "")).strip().lower() == "drop"
    ):
        return OBJECT_STATUS_ABSENT
    return status


def _is_absent_like_review(review: dict[str, Any]) -> bool:
    local_outcome = str(review.get("local_outcome", "")).strip().lower()
    status = _effective_object_review_status(review)
    return local_outcome in {LOCAL_OUTCOME_OUT_OF_FRAME, LOCAL_OUTCOME_EXCLUDED} or status == OBJECT_STATUS_ABSENT


def _instance_triangle_id_set(
    instance_mesh_data: InstanceMeshData | None,
    obj_id: int,
) -> set[int]:
    if instance_mesh_data is None:
        return set()

    triangle_ids_by_instance = getattr(instance_mesh_data, "triangle_ids_by_instance", {}) or {}
    boundary_triangle_ids_by_instance = getattr(
        instance_mesh_data,
        "boundary_triangle_ids_by_instance",
        {},
    ) or {}
    tri_parts = [
        arr for arr in (
            triangle_ids_by_instance.get(int(obj_id)),
            boundary_triangle_ids_by_instance.get(int(obj_id)),
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
    surface_points_by_instance = getattr(instance_mesh_data, "surface_points_by_instance", {}) or {}
    samples = surface_points_by_instance.get(int(obj_id))
    if samples is None:
        return np.empty((0, 3), dtype=np.float64)
    return np.asarray(samples, dtype=np.float64)


def _instance_surface_sample_metadata(
    instance_mesh_data: InstanceMeshData | None,
    obj_id: int,
) -> tuple[np.ndarray, np.ndarray]:
    if instance_mesh_data is None:
        return (
            np.empty((0,), dtype=np.int64),
            np.empty((0, 3), dtype=np.float64),
        )
    surface_triangle_ids_by_instance = getattr(
        instance_mesh_data,
        "surface_triangle_ids_by_instance",
        {},
    ) or {}
    surface_barycentrics_by_instance = getattr(
        instance_mesh_data,
        "surface_barycentrics_by_instance",
        {},
    ) or {}
    triangle_ids = surface_triangle_ids_by_instance.get(int(obj_id))
    barycentrics = surface_barycentrics_by_instance.get(int(obj_id))
    if triangle_ids is None or barycentrics is None:
        return (
            np.empty((0,), dtype=np.int64),
            np.empty((0, 3), dtype=np.float64),
        )
    return (
        np.asarray(triangle_ids, dtype=np.int64),
        np.asarray(barycentrics, dtype=np.float64),
    )


def _in_frame_surface_sample_subset(
    sample_points: np.ndarray,
    camera_pose: CameraPose,
    color_intrinsics: CameraIntrinsics,
    sample_triangle_ids: np.ndarray | None = None,
    sample_barycentrics: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = np.asarray(sample_points, dtype=np.float64)
    if len(points) == 0:
        return (
            np.empty((0, 3), dtype=np.float64),
            np.empty((0,), dtype=np.int64),
            np.empty((0, 3), dtype=np.float64),
        )

    in_frame_indices: list[int] = []
    for idx, point in enumerate(points):
        uv, depth = project_to_image(point, camera_pose, color_intrinsics)
        if uv is None or depth <= 0:
            continue
        u = float(uv[0])
        v = float(uv[1])
        if 0 <= u < color_intrinsics.width and 0 <= v < color_intrinsics.height:
            in_frame_indices.append(int(idx))
    if not in_frame_indices:
        return (
            np.empty((0, 3), dtype=np.float64),
            np.empty((0,), dtype=np.int64),
            np.empty((0, 3), dtype=np.float64),
        )

    index_array = np.asarray(in_frame_indices, dtype=np.int64)
    return (
        points[index_array],
        (
            np.asarray(sample_triangle_ids, dtype=np.int64)[index_array]
            if sample_triangle_ids is not None and len(sample_triangle_ids) == len(points)
            else np.empty((0,), dtype=np.int64)
        ),
        (
            np.asarray(sample_barycentrics, dtype=np.float64)[index_array]
            if sample_barycentrics is not None and len(sample_barycentrics) == len(points)
            else np.empty((0, 3), dtype=np.float64)
        ),
    )


def _build_ray_visibility_stage_result(
    *,
    base_sample_count: int,
    sampled_point_count: int,
    in_frame_sample_count: int,
    visible_count: int,
    valid_count: int,
) -> dict[str, Any]:
    visible_ratio = float(visible_count / valid_count) if valid_count > 0 else 0.0
    return {
        "base_sample_count": int(base_sample_count),
        "sampled_point_count": int(sampled_point_count),
        "in_frame_sample_count": int(in_frame_sample_count),
        "visible_count": int(visible_count),
        "valid_count": int(valid_count),
        "visible_ratio": visible_ratio,
    }


def _ray_visibility_stage_passes(
    stage_result: dict[str, Any] | None,
    *,
    min_visible_ratio: float = REFERABILITY_MESH_RAY_VISIBLE_RATIO_MIN,
) -> bool:
    if not isinstance(stage_result, dict):
        return False
    valid_count = int(stage_result.get("valid_count", 0) or 0)
    if valid_count <= 0:
        return False
    visible_ratio = float(stage_result.get("visible_ratio", 0.0) or 0.0)
    return visible_ratio >= float(min_visible_ratio)


def _evaluate_crop_unique_mesh_ray_stage(
    *,
    obj_id: int,
    camera_pose: CameraPose,
    color_intrinsics: CameraIntrinsics,
    ray_caster: Any,
    instance_mesh_data: InstanceMeshData,
    base_sample_count: int,
) -> dict[str, Any]:
    sample_points = _instance_surface_samples(instance_mesh_data, int(obj_id))
    sampled_point_count = int(len(sample_points))
    sample_triangle_ids, sample_barycentrics = _instance_surface_sample_metadata(
        instance_mesh_data,
        int(obj_id),
    )
    target_tri_ids = _instance_triangle_id_set(instance_mesh_data, int(obj_id))
    in_frame_points, in_frame_triangle_ids, in_frame_barycentrics = _in_frame_surface_sample_subset(
        sample_points,
        camera_pose,
        color_intrinsics,
        sample_triangle_ids=sample_triangle_ids,
        sample_barycentrics=sample_barycentrics,
    )
    in_frame_sample_count = int(len(in_frame_points))
    visible_count = 0
    valid_count = 0
    if in_frame_sample_count > 0 and target_tri_ids:
        visible_count, valid_count = _mesh_visibility_stats_compat(
            ray_caster,
            camera_pos=np.asarray(camera_pose.position, dtype=np.float64),
            target_points=in_frame_points,
            target_tri_ids=target_tri_ids,
            sample_triangle_ids=in_frame_triangle_ids,
            sample_barycentrics=in_frame_barycentrics,
            vertices=np.asarray(instance_mesh_data.vertices, dtype=np.float64),
            faces=np.asarray(instance_mesh_data.faces, dtype=np.int64),
        )
    return _build_ray_visibility_stage_result(
        base_sample_count=base_sample_count,
        sampled_point_count=sampled_point_count,
        in_frame_sample_count=in_frame_sample_count,
        visible_count=visible_count,
        valid_count=valid_count,
    )


def _bounds_to_mask(
    bounds: object,
    *,
    image_shape: tuple[int, ...],
) -> np.ndarray | None:
    if not isinstance(bounds, (list, tuple)) or len(bounds) != 4:
        return None
    try:
        u_min, u_max, v_min, v_max = [int(value) for value in bounds]
    except (TypeError, ValueError):
        return None
    height = int(image_shape[0]) if len(image_shape) >= 1 else 0
    width = int(image_shape[1]) if len(image_shape) >= 2 else 0
    if width <= 0 or height <= 0:
        return None
    u_min = max(0, min(width, u_min))
    u_max = max(0, min(width, u_max))
    v_min = max(0, min(height, v_min))
    v_max = max(0, min(height, v_max))
    if u_max <= u_min or v_max <= v_min:
        return None
    mask = np.zeros((height, width), dtype=bool)
    mask[v_min:v_max, u_min:u_max] = True
    return mask


def _select_best_detection_for_object_review(
    *,
    detections: list[dict[str, Any]],
    review: dict[str, Any],
    image_shape: tuple[int, ...],
) -> dict[str, Any] | None:
    focus_masks = [
        mask
        for mask in (
            _bounds_to_mask(review.get("roi_bounds_px"), image_shape=image_shape),
            _bounds_to_mask(review.get("crop_bounds_px"), image_shape=image_shape),
        )
        if isinstance(mask, np.ndarray)
    ]
    best_detection: dict[str, Any] | None = None
    best_key: tuple[float, float, float, int] | None = None

    for detection in detections:
        detection_mask = detection.get("mask")
        if not isinstance(detection_mask, np.ndarray):
            continue
        detection_mask_bool = np.asarray(detection_mask, dtype=bool)
        detection_area = int(detection_mask_bool.sum())
        if detection_area <= 0:
            continue
        max_overlap_ratio = 0.0
        max_iou = 0.0
        for focus_mask in focus_masks:
            intersection = int(np.logical_and(detection_mask_bool, focus_mask).sum())
            if intersection <= 0:
                continue
            union = int(np.logical_or(detection_mask_bool, focus_mask).sum())
            max_overlap_ratio = max(max_overlap_ratio, float(intersection / detection_area))
            max_iou = max(max_iou, float(intersection / union) if union > 0 else 0.0)
        ranking_key = (
            max_overlap_ratio,
            max_iou,
            float(detection.get("score", 0.0) or 0.0),
            detection_area,
        )
        if best_key is None or ranking_key > best_key:
            best_key = ranking_key
            best_detection = detection

    if best_detection is None:
        return None
    if focus_masks and best_key is not None and best_key[0] <= 0.0 and best_key[1] <= 0.0:
        return None
    return best_detection


def _apply_crop_unique_mesh_quality_review(
    *,
    crop_unique_label_object_ids: dict[str, int],
    object_reviews: dict[int, dict[str, Any]],
    objects_by_id: dict[int, dict[str, Any]],
    image_path: Path,
    image_shape: tuple[int, ...],
    camera_pose: CameraPose,
    color_intrinsics: CameraIntrinsics,
    depth_image: np.ndarray | None,
    depth_intrinsics: CameraIntrinsics | None,
    instance_mesh_data_getter: Callable[[int], InstanceMeshData] | None,
    topology_quality_by_obj_id: dict[int, dict[str, Any]],
    mesh_mask_quality_by_obj_id: dict[int, dict[str, Any]],
    client: object | None,
) -> dict[str, str]:
    if not crop_unique_label_object_ids:
        return {}
    if not callable(instance_mesh_data_getter):
        raise RuntimeError("mesh-quality referability validation requires lazy instance mesh data loaders")

    instance_mesh_data = instance_mesh_data_getter(REFERABILITY_MESH_RAY_STAGE1_BASE_SAMPLE_COUNT)
    detection_cache: dict[tuple[str, ...], dict[str, Any]] = {}
    failed_label_reasons: dict[str, str] = {}

    for label, obj_id in sorted(crop_unique_label_object_ids.items()):
        review = object_reviews.get(int(obj_id))
        obj = objects_by_id.get(int(obj_id))
        if not isinstance(review, dict) or not isinstance(obj, dict):
            failed_label_reasons[str(label)] = "missing_object_review"
            continue

        alias_variants = _normalize_alias_variants(
            list(obj.get("alias_variants", []) or []) + [obj.get("label", label)]
        )
        review["mesh_quality_review"] = {
            "applied": True,
            "decision": "drop",
            "reason": "pending",
            "detection_prompt_variants": list(alias_variants),
            "raw_detection_count": 0,
            "candidate_detection_count": 0,
            "matched_detection": None,
        }

        topology_quality = _compute_topology_quality_for_object(
            obj_id=int(obj_id),
            instance_mesh_data=instance_mesh_data,
        )
        topology_quality_by_obj_id[int(obj_id)] = topology_quality
        review["topology_status"] = str(topology_quality.get("status", "")).strip().lower() or None
        review["topology_reason_codes"] = list(topology_quality.get("reason_codes", []))

        if str(topology_quality.get("status", "")).strip().lower() == "fail":
            review["mesh_quality_review"]["reason"] = "topology_fail"
            failed_label_reasons[str(label)] = "topology_fail"
            continue

        cache_key = tuple(alias_variants)
        cached = detection_cache.get(cache_key)
        if cached is None:
            try:
                raw_detections = _call_dinox_joint_detection(
                    client=client,
                    image_path=image_path,
                    alias_variants=alias_variants,
                    image_shape=image_shape,
                )
                candidate_detections = _dedupe_detections_by_mask_iou(
                    [
                        detection
                        for detection in raw_detections
                        if int(detection.get("area_px", 0) or 0) >= SEGMENTATION_EXTREME_NOISE_MIN_AREA_PX
                        and float(detection.get("score", 0.0) or 0.0) >= SEGMENTATION_EXTREME_NOISE_MIN_SCORE
                    ]
                )
                cached = {
                    "error": None,
                    "raw_detections": raw_detections,
                    "candidate_detections": candidate_detections,
                }
            except Exception as exc:
                logger.warning(
                    "DINO-X mesh-quality check failed for %s/%s label=%s: %s",
                    image_path.parent.name,
                    image_path.name,
                    label,
                    exc,
                )
                cached = {
                    "error": str(exc),
                    "raw_detections": [],
                    "candidate_detections": [],
                }
            detection_cache[cache_key] = cached

        raw_detections = list(cached.get("raw_detections", []))
        candidate_detections = list(cached.get("candidate_detections", []))
        review["mesh_quality_review"]["raw_detection_count"] = len(raw_detections)
        review["mesh_quality_review"]["candidate_detection_count"] = len(candidate_detections)

        if cached.get("error") is not None:
            review["mesh_quality_review"]["reason"] = "segmentation_api_failed"
            failed_label_reasons[str(label)] = "segmentation_api_failed"
            continue

        matched_detection = _select_best_detection_for_object_review(
            detections=candidate_detections,
            review=review,
            image_shape=image_shape,
        )
        if matched_detection is None:
            review["mesh_quality_review"]["reason"] = (
                "no_detection_overlap" if candidate_detections else "no_detection_mask"
            )
            failed_label_reasons[str(label)] = str(review["mesh_quality_review"]["reason"])
            continue

        review["mesh_quality_review"]["matched_detection"] = _serialize_detection(matched_detection)
        mesh_quality = _compute_mesh_mask_quality_for_object(
            obj_id=int(obj_id),
            detection_mask=np.asarray(matched_detection["mask"], dtype=bool),
            topology_status=str(topology_quality.get("status", "")),
            camera_pose=camera_pose,
            color_intrinsics=color_intrinsics,
            depth_image=depth_image,
            depth_intrinsics=depth_intrinsics,
            instance_mesh_data=instance_mesh_data,
        )
        mesh_mask_quality_by_obj_id[int(obj_id)] = mesh_quality
        review["mesh_mask_status"] = str(mesh_quality.get("status", "")).strip().lower() or None
        review["mesh_mask_reason_codes"] = list(mesh_quality.get("reason_codes", []))
        review["mesh_mask_iou"] = mesh_quality.get("iou")
        review["mesh_mask_under_coverage"] = mesh_quality.get("under_coverage")
        review["mesh_mask_over_coverage"] = mesh_quality.get("over_coverage")
        review["mesh_mask_area_ratio"] = mesh_quality.get("area_ratio")
        review["mesh_mask_depth_bad_ratio"] = mesh_quality.get("depth_bad_ratio")

        if str(mesh_quality.get("status", "")).strip().lower() == "fail":
            review["mesh_quality_review"]["reason"] = "mesh_mask_mismatch"
            failed_label_reasons[str(label)] = "mesh_mask_mismatch"
            continue

        review["mesh_quality_review"]["decision"] = "pass"
        review["mesh_quality_review"]["reason"] = "mesh_mask_match"

    return failed_label_reasons


def _resolve_scene_mesh_path(scene_dir: Path) -> Path:
    mesh_path = scene_dir / f"{scene_dir.name}_vh_clean.ply"
    if mesh_path.exists():
        return mesh_path
    fallback = scene_dir / f"{scene_dir.name}_vh_clean_2.ply"
    if fallback.exists():
        return fallback
    raise RuntimeError(f"mesh geometry not found for referability scene {scene_dir.name}")


def _make_lazy_mesh_ray_resource_getters(
    *,
    scene_dir: Path,
    scene_objects: list[dict[str, Any]],
    axis_alignment: np.ndarray | None,
) -> tuple[Callable[[], Any], Callable[[int], InstanceMeshData]]:
    object_ids = sorted(
        {
            int(obj.get("id"))
            for obj in scene_objects
            if obj.get("id") is not None
        }
    )
    resource_cache: dict[str, Any] = {}

    def _get_ray_caster() -> Any:
        if "ray_caster" not in resource_cache:
            mesh_path = _resolve_scene_mesh_path(scene_dir)
            resource_cache["ray_caster"] = RayCaster.from_ply(
                str(mesh_path),
                axis_alignment=axis_alignment,
            )
        return resource_cache["ray_caster"]

    def _get_instance_mesh_data(base_sample_count: int) -> InstanceMeshData:
        base_count = int(base_sample_count)
        cache_key = f"instance_mesh_data:{base_count}"
        if cache_key not in resource_cache:
            resource_cache[cache_key] = load_instance_mesh_data(
                scene_dir,
                instance_ids=list(object_ids),
                n_surface_samples=base_count,
            )
        return resource_cache[cache_key]

    return _get_ray_caster, _get_instance_mesh_data


def _aggregate_label_reviews(
    label_to_ids: dict[str, list[int]],
    object_reviews: dict[int, dict[str, Any]],
) -> tuple[dict[str, str], dict[str, int], list[int]]:
    label_statuses, label_counts, referable_object_ids, _unique_label_object_ids = _aggregate_crop_label_reviews(
        label_to_ids,
        object_reviews,
    )
    return label_statuses, label_counts, referable_object_ids


def _aggregate_crop_label_reviews(
    label_to_ids: dict[str, list[int]],
    object_reviews: dict[int, dict[str, Any]],
) -> tuple[dict[str, str], dict[str, int], list[int], dict[str, int]]:
    label_statuses: dict[str, str] = {}
    label_counts: dict[str, int] = {}
    referable_object_ids: list[int] = []
    unique_label_object_ids: dict[str, int] = {}

    for label, obj_ids in sorted(label_to_ids.items()):
        clear_ids: list[int] = []
        has_unsure = False
        all_absent_like = True

        for obj_id in obj_ids:
            review = object_reviews.get(int(obj_id))
            if not isinstance(review, dict):
                has_unsure = True
                all_absent_like = False
                continue
            status = _effective_object_review_status(review)
            if status == OBJECT_STATUS_CLEAR:
                clear_ids.append(int(obj_id))
                all_absent_like = False
                continue
            if status == OBJECT_STATUS_UNSURE:
                has_unsure = True
                all_absent_like = False
                continue
            if not _is_absent_like_review(review):
                has_unsure = True
                all_absent_like = False

        clear_count = len(clear_ids)
        label_counts[label] = clear_count

        if clear_count == 1 and not has_unsure:
            label_statuses[label] = LABEL_STATUS_UNIQUE
            unique_obj_id = int(clear_ids[0])
            unique_label_object_ids[label] = unique_obj_id
            referable_object_ids.append(unique_obj_id)
            continue
        if clear_count >= 2:
            label_statuses[label] = LABEL_STATUS_MULTIPLE
            continue
        if clear_count == 0 and not has_unsure and all_absent_like:
            label_statuses[label] = LABEL_STATUS_ABSENT
            continue
        label_statuses[label] = LABEL_STATUS_UNSURE

    return (
        dict(sorted(label_statuses.items())),
        dict(sorted(label_counts.items())),
        sorted(set(int(obj_id) for obj_id in referable_object_ids)),
        {str(label): int(obj_id) for label, obj_id in sorted(unique_label_object_ids.items())},
    )


def _compute_frame_referability_entry(
    *,
    client,
    model_name: str,
    scene_objects: list[dict[str, Any]],
    objects_by_id: dict[int, dict[str, Any]],
    image: np.ndarray,
    image_path: Path,
    camera_pose,
    color_intrinsics,
    depth_image: np.ndarray | None,
    depth_intrinsics,
    selector_visible_object_ids: list[int],
    selector_score: int | None = None,
    frame_info: dict[str, Any] | None = None,
    frame_selection_score: int | None = None,
    vlm_workers: int = 1,
    ray_caster_getter: Callable[[], Any] | None = None,
    instance_mesh_data_getter: Callable[[int], InstanceMeshData] | None = None,
) -> dict[str, Any]:
    selector_visible_object_ids = sorted(
        int(obj_id)
        for obj_id in selector_visible_object_ids
        if int(obj_id) in objects_by_id
    )
    selector_visible_label_counts = _count_labels_for_object_ids(
        selector_visible_object_ids,
        objects_by_id,
    )
    candidate_visible_object_ids, candidate_visibility_source = _refine_candidate_visible_object_ids(
        selector_visible_object_ids,
        scene_objects,
        camera_pose,
        color_intrinsics,
        depth_image,
        depth_intrinsics,
        ray_caster_getter=ray_caster_getter,
        instance_mesh_data_getter=instance_mesh_data_getter,
    )
    candidate_labels, label_to_object_ids = _build_frame_label_candidates(
        candidate_visible_object_ids,
        objects_by_id,
    )
    selector_duplicate_shortcut_labels = _selector_duplicate_shortcut_labels(
        selector_visible_label_counts
    )
    vlm_label_to_object_ids = {
        str(label): [int(obj_id) for obj_id in obj_ids]
        for label, obj_ids in sorted(label_to_object_ids.items())
        if str(label) not in selector_duplicate_shortcut_labels
    }

    normalized_frame_info = _normalize_frame_review(
        frame_info if isinstance(frame_info, dict) else _frame_decision(client, model_name, image)
    )
    selector_score_value = int(selector_score) if selector_score is not None else len(selector_visible_object_ids)
    selection_score_value = (
        int(frame_selection_score)
        if frame_selection_score is not None
        else _frame_selection_score(selector_score_value, normalized_frame_info)
    )
    visibility_instance_mesh_data = None
    if callable(instance_mesh_data_getter):
        visibility_instance_mesh_data = instance_mesh_data_getter(
            REFERABILITY_MESH_RAY_STAGE1_BASE_SAMPLE_COUNT
        )
    visibility_by_obj_id = compute_frame_object_visibility(
        scene_objects,
        camera_pose,
        color_intrinsics,
        image_path=image_path,
        depth_image=depth_image,
        depth_intrinsics=depth_intrinsics,
        instance_mesh_data=visibility_instance_mesh_data,
        strict_mode=False,
    )
    visibility_audit_by_object_id = _build_visibility_audit_by_object_id(
        scene_objects,
        objects_by_id,
        visibility_by_obj_id,
        color_intrinsics,
        selector_visible_object_ids,
        candidate_visible_object_ids,
        candidate_visibility_source,
    )

    object_reviews: dict[int, dict[str, Any]] = {}
    crop_label_statuses: dict[str, str] = {}
    crop_label_counts: dict[str, int] = {}
    crop_referable_object_ids: list[int] = []
    full_frame_label_reviews: list[dict[str, Any]] = []
    full_frame_label_statuses: dict[str, str] = {}
    full_frame_label_counts: dict[str, int] = {}
    label_statuses: dict[str, str] = {}
    label_counts: dict[str, int] = {}
    attachment_referable_object_ids: list[int] = []
    referable_object_ids: list[int] = []
    alias_group_statuses: dict[str, str] = {}
    referability_reason_by_alias_group: dict[str, str] = {}
    label_status_reason_by_label: dict[str, str] = {}
    frame_anchor_candidate_ids_by_alias_group: dict[str, list[int]] = {}
    frame_anchor_candidate_count_by_alias_group: dict[str, int] = {}
    alias_group_reviews: list[dict[str, Any]] = []
    topology_quality_by_obj_id: dict[int, dict[str, Any]] = {}
    mesh_mask_quality_by_obj_id: dict[int, dict[str, Any]] = {}
    visibility_probe_object_ids: list[int] = []

    if normalized_frame_info["frame_usable"]:
        image_b64: str | None = None
        pending_object_review_jobs: list[tuple[int, str, str, dict[str, Any]]] = []
        for obj_id in candidate_visible_object_ids:
            obj = objects_by_id.get(int(obj_id))
            if obj is None:
                continue
            label = str(obj.get("label", "")).strip().lower()
            crop_entry = _build_object_review_crop(image, visibility_by_obj_id.get(int(obj_id)))
            review = _build_object_review_entry(
                obj_id=int(obj_id),
                label=label,
                crop_entry=crop_entry,
            )
            if label in selector_duplicate_shortcut_labels:
                review["review_mode"] = OBJECT_REVIEW_MODE_SELECTOR_DUPLICATE_SHORTCUT
                review["review_skip_reason"] = OBJECT_REVIEW_SKIP_SELECTOR_DUPLICATE_REASON
                object_reviews[int(obj_id)] = review
                continue
            if crop_entry.get("local_outcome") == LOCAL_OUTCOME_REVIEWED:
                if image_b64 is None:
                    image_b64 = _image_to_base64(image)
                pending_object_review_jobs.append(
                    (
                        int(obj_id),
                        label,
                        str(crop_entry.get("image_b64", "") or ""),
                        dict(review),
                    )
                )
                continue
            object_reviews[int(obj_id)] = review

        def _run_object_review_job(job: tuple[int, str, str, dict[str, Any]]) -> tuple[int, dict[str, Any]]:
            obj_id, label, crop_b64, review = job
            status, raw_response = _object_review_decision(
                client,
                model_name,
                str(image_b64 or ""),
                crop_b64,
                label,
            )
            updated_review = dict(review)
            updated_review["vlm_status"] = status
            updated_review["raw_response"] = raw_response or None
            return int(obj_id), updated_review

        for obj_id, review in _run_in_thread_pool(
            pending_object_review_jobs,
            _run_object_review_job,
            max_workers=vlm_workers,
        ):
            object_reviews[int(obj_id)] = review

        crop_label_statuses, crop_label_counts, crop_referable_object_ids, crop_unique_label_object_ids = (
            _aggregate_crop_label_reviews(
                vlm_label_to_object_ids,
                object_reviews,
            )
        )
        shortcut_crop_label_statuses, shortcut_crop_label_counts = (
            _derive_selector_duplicate_shortcut_crop_reviews(
                label_to_object_ids=label_to_object_ids,
                selector_visible_label_counts=selector_visible_label_counts,
            )
        )
        crop_label_statuses.update(shortcut_crop_label_statuses)
        crop_label_counts.update(shortcut_crop_label_counts)
        crop_label_statuses = dict(sorted(crop_label_statuses.items()))
        crop_label_counts = dict(sorted(crop_label_counts.items()))
        label_status_reason_by_label = {
            str(label): "selector_duplicate_shortcut"
            for label in shortcut_crop_label_statuses
        }
        for label in crop_label_statuses:
            label_status_reason_by_label.setdefault(str(label), "derived_from_crop_vlm")
        if crop_unique_label_object_ids:
            def _run_full_frame_label_review_job(item: tuple[str, int]) -> dict[str, Any]:
                label, obj_id = item
                dinox_review = _full_frame_label_dinox_review(
                    client=client,
                    image_path=image_path,
                    image_shape=tuple(image.shape),
                    label=label,
                )
                status = (
                    _normalize_full_frame_label_status(dinox_review.get("status"))
                    or LABEL_STATUS_UNSURE
                )
                return {
                    "label": str(label),
                    "status": status,
                    "crop_status": crop_label_statuses.get(label),
                    "crop_clear_count": crop_label_counts.get(label),
                    "crop_referable_object_id": int(obj_id),
                    "backend": str(dinox_review.get("backend", "dinox") or "dinox"),
                    "reason": str(dinox_review.get("reason", "")).strip() or None,
                    "raw_detection_count": int(dinox_review.get("raw_detection_count", 0) or 0),
                    "raw_detections": [
                        dict(item)
                        for item in dinox_review.get("raw_detections", [])
                        if isinstance(item, dict)
                    ],
                    "raw_response": dinox_review.get("raw_response"),
                }

            for review_payload in _run_in_thread_pool(
                list(sorted(crop_unique_label_object_ids.items())),
                _run_full_frame_label_review_job,
                max_workers=vlm_workers,
            ):
                full_frame_label_reviews.append(review_payload)
                full_frame_label_statuses[str(review_payload["label"])] = str(review_payload["status"])

        full_frame_label_statuses = dict(sorted(full_frame_label_statuses.items()))
        full_frame_label_counts = _label_counts_from_statuses(full_frame_label_statuses)
        label_statuses = _merge_final_label_statuses(
            crop_label_statuses=crop_label_statuses,
            selector_visible_label_counts=selector_visible_label_counts,
            full_frame_label_statuses=full_frame_label_statuses,
        )
        label_counts = _label_counts_from_statuses(label_statuses)
        referable_object_ids = _final_referable_object_ids(
            label_statuses=label_statuses,
            crop_unique_label_object_ids=crop_unique_label_object_ids,
            object_reviews=object_reviews,
            visibility_audit_by_object_id=visibility_audit_by_object_id,
        )
        attachment_referable_object_ids = _final_referable_object_ids(
            label_statuses=label_statuses,
            crop_unique_label_object_ids=crop_unique_label_object_ids,
            object_reviews=object_reviews,
            visibility_audit_by_object_id=visibility_audit_by_object_id,
            bbox_in_frame_ratio_min=ATTACHMENT_REFERABLE_BBOX_IN_FRAME_RATIO_MIN,
        )

        alias_group_to_statuses: dict[str, set[str]] = defaultdict(set)
        alias_group_to_reasons: dict[str, set[str]] = defaultdict(set)
        for obj in scene_objects:
            alias_group = str(obj.get("alias_group", "")).strip().lower()
            label = str(obj.get("label", "")).strip().lower()
            if not alias_group or label not in label_statuses:
                continue
            alias_group_to_statuses[alias_group].add(label_statuses[label])
            alias_group_to_reasons[alias_group].add(
                label_status_reason_by_label.get(label, "derived_from_crop_vlm")
            )
        alias_group_statuses = {
            alias_group: (next(iter(statuses)) if len(statuses) == 1 else LABEL_STATUS_UNSURE)
            for alias_group, statuses in sorted(alias_group_to_statuses.items())
        }
        referability_reason_by_alias_group = {
            alias_group: (
                next(iter(alias_group_to_reasons.get(alias_group, {"derived_from_crop_vlm"})))
                if len(alias_group_to_reasons.get(alias_group, set())) == 1
                else "mixed_sources"
            )
            for alias_group in alias_group_statuses
        }

    return {
        "frame_usable": normalized_frame_info["frame_usable"],
        "frame_reject_reason": None if normalized_frame_info["frame_usable"] else normalized_frame_info["reason"],
        "selector_score": selector_score_value,
        "frame_quality_clear": _coerce_bool(
            normalized_frame_info.get("clear"),
            default=bool(normalized_frame_info.get("frame_usable", True)),
        ),
        "frame_quality_score": _normalize_clarity_score(normalized_frame_info.get("clarity_score"), default=60),
        "frame_quality_reason": str(normalized_frame_info.get("reason", "")).strip(),
        "frame_selection_score": selection_score_value,
        "selector_visible_object_ids": selector_visible_object_ids,
        "selector_visible_label_counts": selector_visible_label_counts,
        "candidate_visible_object_ids": candidate_visible_object_ids,
        "candidate_visibility_source": candidate_visibility_source,
        "candidate_visible_label_counts": _count_labels_for_object_ids(
            candidate_visible_object_ids,
            objects_by_id,
        ),
        "candidate_labels": list(candidate_labels),
        "label_to_object_ids": {
            str(label): [int(obj_id) for obj_id in obj_ids]
            for label, obj_ids in sorted(label_to_object_ids.items())
        },
        "alias_group_statuses": dict(sorted(alias_group_statuses.items())),
        "referability_reason_by_alias_group": dict(sorted(referability_reason_by_alias_group.items())),
        "frame_anchor_candidate_ids_by_alias_group": {
            str(alias_group): [int(obj_id) for obj_id in obj_ids]
            for alias_group, obj_ids in sorted(frame_anchor_candidate_ids_by_alias_group.items())
        },
        "frame_anchor_candidate_count_by_alias_group": {
            str(alias_group): int(count)
            for alias_group, count in sorted(frame_anchor_candidate_count_by_alias_group.items())
        },
        "alias_group_reviews": list(alias_group_reviews),
        "visibility_probe_object_ids": visibility_probe_object_ids,
        "visibility_audit_by_object_id": visibility_audit_by_object_id,
        "topology_quality_by_obj_id": {
            str(obj_id): payload
            for obj_id, payload in sorted(topology_quality_by_obj_id.items())
        },
        "mesh_mask_quality_by_obj_id": {
            str(obj_id): payload
            for obj_id, payload in sorted(mesh_mask_quality_by_obj_id.items())
        },
        "object_reviews": {
            str(obj_id): review
            for obj_id, review in sorted(object_reviews.items())
        },
        "crop_label_statuses": dict(sorted(crop_label_statuses.items())),
        "crop_label_counts": dict(sorted(crop_label_counts.items())),
        "crop_referable_object_ids": sorted(set(int(obj_id) for obj_id in crop_referable_object_ids)),
        "full_frame_label_reviews": list(full_frame_label_reviews),
        "full_frame_label_statuses": full_frame_label_statuses,
        "full_frame_label_counts": full_frame_label_counts,
        "vlm_label_reviews": list(alias_group_reviews),
        "label_statuses": dict(sorted(label_statuses.items())),
        "label_counts": dict(sorted(label_counts.items())),
        "attachment_referable_pairs": [],
        "attachment_referable_pair_count": 0,
        "attachment_view_group_id": None,
        "final_selection_rank": FRAME_SELECTION_FALLBACK_RANK,
        "attachment_referable_object_ids": sorted(
            set(int(obj_id) for obj_id in attachment_referable_object_ids)
        ),
        "referable_object_ids": sorted(set(int(obj_id) for obj_id in referable_object_ids)),
        "vlm_unique_object_ids": sorted(set(int(obj_id) for obj_id in referable_object_ids)),
    }


def _frame_entry_has_debug_fields(entry: Any) -> bool:
    if not isinstance(entry, dict):
        return False
    required_keys = {
        "frame_quality_clear",
        "frame_quality_score",
        "frame_quality_reason",
        "frame_selection_score",
        "attachment_referable_pairs",
        "attachment_referable_pair_count",
        "final_selection_rank",
        "candidate_visible_object_ids",
        "candidate_visibility_source",
        "candidate_labels",
        "label_to_object_ids",
        "selector_visible_object_ids",
        "selector_visible_label_counts",
        "visibility_audit_by_object_id",
        "object_reviews",
        "crop_label_statuses",
        "crop_label_counts",
        "crop_referable_object_ids",
        "full_frame_label_reviews",
        "full_frame_label_statuses",
        "full_frame_label_counts",
        "label_statuses",
        "label_counts",
        "referable_object_ids",
    }
    if not required_keys.issubset(entry.keys()):
        return False
    return _frame_entry_has_consistent_final_fields(entry)


def _select_and_rerank_frames(
    *,
    client,
    model_name: str,
    scene_dir: Path,
    frame_candidates: list[dict[str, Any]],
    max_frames: int,
    vlm_workers: int = 1,
) -> list[dict[str, Any]]:
    if not frame_candidates:
        return []

    reranked: list[dict[str, Any]] = []
    reviewed_frame_count = 0

    def _sort_key(entry: dict[str, Any]) -> tuple[int, str]:
        return (
            -int(entry.get("frame_info", {}).get("clarity_score", 0) or 0),
            str(entry.get("image_name", "")),
        )

    reviewed_frames = _select_non_attachment_group_representatives(
        client=client,
        model_name=model_name,
        scene_dir=scene_dir,
        frames=frame_candidates,
        vlm_workers=vlm_workers,
    )
    for reviewed_frame in reviewed_frames:
        reviewed_frame_count += 1
        reranked.append(reviewed_frame)

    reranked.sort(
        key=_sort_key,
    )
    selected = reranked[:max(0, int(max_frames))]
    if reranked:
        logger.info(
            "VLM reranked %d geometric frame candidates for %s; reviewed %d image(s), kept %d clear frames (clear candidates=%d, best clarity=%d)",
            len(frame_candidates),
            scene_dir.name,
            reviewed_frame_count,
            len(selected),
            len(reranked),
            int(reranked[0].get("frame_info", {}).get("clarity_score", 0)),
        )
    return selected


def main():
    parser = argparse.ArgumentParser(description="Precompute VLM frame/object referability cache")
    parser.add_argument(
        "--data_root", type=str,
        default=os.getenv("SCANNET_PATH", "/home/lihongxing/datasets/ScanNet/data/scans"),
        help="Root directory of ScanNet scans (contains scene subdirectories)",
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Output JSON cache path",
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
        "--label_map", type=str, default=None,
        help="Path to scannetv2-labels.combined.tsv for raw_category normalization",
    )
    parser.add_argument(
        "--vlm_url", type=str, default=DEFAULT_VLM_URL,
        help="OpenAI-compatible VLM API base URL",
    )
    parser.add_argument(
        "--vlm_model", type=str, default=None,
        help="Model name to use; if omitted, auto-detect from /v1/models",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from an existing output cache if present",
    )
    parser.add_argument(
        "--label_batch_size", type=int, default=LABEL_BATCH_SIZE,
        help="Legacy compatibility flag; per-object review now issues one VLM request per valid crop",
    )
    parser.add_argument(
        "--vlm_workers", type=int, default=4,
        help="Maximum number of concurrent independent VLM requests",
    )
    args = parser.parse_args()

    global EXCLUDED_LABELS
    from src.scene_parser import EXCLUDED_LABELS as SCENE_EXCLUDED_LABELS
    from src.scene_parser import load_scannet_label_map, parse_scene
    from src.support_graph import (
        enrich_scene_with_attachment,
        get_scene_attachment_graph,
        has_nontrivial_attachment,
    )

    EXCLUDED_LABELS = set(SCENE_EXCLUDED_LABELS)

    if args.label_map:
        load_scannet_label_map(args.label_map)
        EXCLUDED_LABELS = set(SCENE_EXCLUDED_LABELS)

    from openai import OpenAI

    api_key = (
        os.getenv("DASHSCOPE_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or "EMPTY"
    )
    client = OpenAI(api_key=api_key, base_url=args.vlm_url)
    try:
        models = client.models.list()
        available = [model.id for model in models.data]
        logger.info("VLM available models: %s", available)
    except Exception as exc:
        logger.error("Cannot reach VLM at %s: %s", args.vlm_url, exc)
        sys.exit(1)

    model_name = args.vlm_model if args.vlm_model else available[0]
    logger.info("Using model: %s", model_name)

    output_path = Path(args.output)
    cache: dict[str, Any] = {
        "version": REFERABILITY_CACHE_VERSION,
        "model": model_name,
        "alias_config_version": ALIAS_CONFIG_VERSION,
        "referability_backend": "crop_vlm_with_mesh_ray",
        "label_batch_size": 1,
        "frames": {},
    }
    if args.resume and output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        if isinstance(loaded, dict) and "frames" in loaded:
            loaded_version = str(loaded.get("version", ""))
            if loaded_version != REFERABILITY_CACHE_VERSION:
                raise RuntimeError(
                    f"Cannot resume referability cache version {loaded_version or '<missing>'}; "
                    f"expected {REFERABILITY_CACHE_VERSION}. Regenerate the cache from scratch."
                )
            cache = loaded
            logger.info("Resuming from %s", output_path)
    cache["alias_config_version"] = ALIAS_CONFIG_VERSION
    cache["referability_backend"] = "crop_vlm_with_mesh_ray"
    cache["label_batch_size"] = 1

    data_root = Path(args.data_root)
    scene_dirs = sorted(
        path for path in data_root.iterdir()
        if path.is_dir() and (path / "pose").exists()
    )
    logger.info("Found %d candidate scenes", len(scene_dirs))

    processed = 0
    for scene_dir in scene_dirs:
        if processed >= args.max_scenes:
            break

        scene_id = scene_dir.name
        logger.info(
            "=== Referability scene %s (%d/%d) ===",
            scene_id,
            processed + 1,
            args.max_scenes,
        )

        scene = parse_scene(scene_dir)
        if scene is None:
            continue

        enrich_scene_with_attachment(scene)
        attachment_graph = get_scene_attachment_graph(scene, scene_id=scene_id)
        if not has_nontrivial_attachment(attachment_graph):
            logger.info("Scene %s has no attachment relations -> skipping", scene_id)
            continue
        scene_cache = cache.setdefault("frames", {}).setdefault(scene_id, {})
        if scene_cache and all(_frame_entry_has_debug_fields(entry) for entry in scene_cache.values()):
            logger.info("Scene %s already cached -> skipping", scene_id)
            processed += 1
            continue

        non_attachment_candidate_limit = max(
            0,
            int(args.max_frames) * NON_ATTACHMENT_FRAME_CANDIDATE_MULTIPLIER,
        )
        frame_candidates = select_frames(
            scene_dir,
            scene["objects"],
            attachment_graph,
            int(args.max_frames),
            keep_all_attachment_frames=True,
            non_attachment_limit=non_attachment_candidate_limit,
        )
        if not frame_candidates:
            continue

        axis_align = load_axis_alignment(scene_dir)
        poses = load_scannet_poses(scene_dir, axis_alignment=axis_align)
        try:
            color_intrinsics = load_scannet_intrinsics(scene_dir)
        except Exception as exc:
            logger.warning("Color intrinsics load failed for %s: %s", scene_id, exc)
            continue
        try:
            depth_intrinsics = load_scannet_depth_intrinsics(scene_dir)
        except Exception as exc:
            logger.warning("Depth intrinsics load failed for %s: %s", scene_id, exc)
            depth_intrinsics = None

        objects_by_id = {int(obj["id"]): obj for obj in scene["objects"]}
        ray_caster_getter, instance_mesh_data_getter = _make_lazy_mesh_ray_resource_getters(
            scene_dir=scene_dir,
            scene_objects=scene["objects"],
            axis_alignment=axis_align,
        )
        attachment_candidate_frames = [
            frame
            for frame in frame_candidates
            if bool(frame.get("attachment_viewpoint_exempt"))
        ]
        non_attachment_candidate_frames = [
            frame
            for frame in frame_candidates
            if not bool(frame.get("attachment_viewpoint_exempt"))
        ]
        logger.info(
            "Selected %d attachment-qualified and %d non-attachment frame candidates for %s before VLM review",
            len(attachment_candidate_frames),
            len(non_attachment_candidate_frames),
            scene_id,
        )

        non_attachment_frames = _select_and_rerank_frames(
            client=client,
            model_name=model_name,
            scene_dir=scene_dir,
            frame_candidates=non_attachment_candidate_frames,
            max_frames=int(args.max_frames),
            vlm_workers=int(args.vlm_workers),
        ) if non_attachment_candidate_frames else []

        def _build_attachment_entry(
            frame: dict[str, Any],
            reviewed_frame: dict[str, Any],
        ) -> dict[str, Any] | None:
            image_name = str(frame.get("image_name", "")).strip()
            if not image_name or image_name not in poses:
                return None

            image_path = scene_dir / "color" / image_name
            image = cv2.imread(str(image_path))
            if image is None:
                logger.warning("Cannot read image %s", image_path)
                return None

            camera_pose = poses[image_name]
            selector_visible_object_ids = [
                int(obj_id)
                for obj_id in frame.get("visible_object_ids", [])
                if int(obj_id) in objects_by_id
            ]
            depth_image = None
            frame_id = Path(image_name).stem
            depth_path = scene_dir / "depth" / f"{frame_id}.png"
            if depth_intrinsics is not None and depth_path.exists():
                try:
                    depth_image = load_depth_image(depth_path)
                except Exception as exc:
                    logger.warning("Depth load failed for %s/%s: %s", scene_id, image_name, exc)

            return _compute_frame_referability_entry(
                client=client,
                model_name=model_name,
                scene_objects=scene["objects"],
                objects_by_id=objects_by_id,
                image=image,
                image_path=image_path,
                camera_pose=camera_pose,
                color_intrinsics=color_intrinsics,
                depth_image=depth_image,
                depth_intrinsics=depth_intrinsics,
                selector_visible_object_ids=selector_visible_object_ids,
                selector_score=int(
                    frame.get("selector_score", frame.get("score", len(selector_visible_object_ids))) or 0
                ),
                frame_info=reviewed_frame.get("frame_info", _selector_quality_pass_frame_info()),
                frame_selection_score=int(
                    reviewed_frame.get(
                        "frame_selection_score",
                        frame.get("selector_score", frame.get("score", 0)),
                    ) or 0
                ),
                vlm_workers=int(args.vlm_workers),
                ray_caster_getter=ray_caster_getter,
                instance_mesh_data_getter=instance_mesh_data_getter,
            )

        attachment_selected_frames = _select_attachment_group_representatives(
            client=client,
            model_name=model_name,
            scene_dir=scene_dir,
            frames=attachment_candidate_frames,
            attachment_graph=attachment_graph,
            attachment_entry_builder=_build_attachment_entry,
            vlm_workers=int(args.vlm_workers),
        ) if attachment_candidate_frames else []

        selected_attachment_frames = _select_attachment_frames_by_global_pair_coverage(
            attachment_selected_frames,
            max_frames=int(args.max_frames),
        )
        remaining_slots = max(0, int(args.max_frames) - len(selected_attachment_frames))
        selected_non_attachment_frames = non_attachment_frames[:remaining_slots]

        if not selected_attachment_frames and not selected_non_attachment_frames:
            logger.info("Scene %s has no final referability frames -> skipping", scene_id)
            continue

        logger.info(
            "Processing referability scene %s (%d/%d) with %d attachment-selected frame(s) and %d non-attachment fallback(s)",
            scene_id,
            processed + 1,
            args.max_scenes,
            len(selected_attachment_frames),
            len(selected_non_attachment_frames),
        )

        final_scene_entries: dict[str, dict[str, Any]] = {}
        final_selection_rank = 0

        for frame in selected_attachment_frames:
            image_name = str(frame.get("image_name", "")).strip()
            final_scene_entries[image_name] = _attach_selection_metadata(
                frame,
                attachment_graph,
                final_selection_rank=final_selection_rank,
                attachment_view_group_id=frame.get("attachment_view_group_id"),
            )
            final_selection_rank += 1

        for frame in selected_non_attachment_frames:
            image_name = str(frame.get("image_name", "")).strip()
            if not image_name or image_name not in poses:
                continue

            image_path = scene_dir / "color" / image_name
            image = cv2.imread(str(image_path))
            if image is None:
                logger.warning("Cannot read image %s", image_path)
                continue

            camera_pose = poses[image_name]
            selector_visible_object_ids = [
                int(obj_id)
                for obj_id in frame.get("visible_object_ids", [])
                if int(obj_id) in objects_by_id
            ]
            depth_image = None
            frame_id = Path(image_name).stem
            depth_path = scene_dir / "depth" / f"{frame_id}.png"
            if depth_intrinsics is not None and depth_path.exists():
                try:
                    depth_image = load_depth_image(depth_path)
                except Exception as exc:
                    logger.warning("Depth load failed for %s/%s: %s", scene_id, image_name, exc)

            entry = _compute_frame_referability_entry(
                client=client,
                model_name=model_name,
                scene_objects=scene["objects"],
                objects_by_id=objects_by_id,
                image=image,
                image_path=image_path,
                camera_pose=camera_pose,
                color_intrinsics=color_intrinsics,
                depth_image=depth_image,
                depth_intrinsics=depth_intrinsics,
                selector_visible_object_ids=selector_visible_object_ids,
                selector_score=int(frame.get("selector_score", frame.get("score", len(selector_visible_object_ids))) or 0),
                frame_info=dict(frame.get("frame_info", {})),
                frame_selection_score=int(frame.get("frame_selection_score", 0) or 0),
                vlm_workers=int(args.vlm_workers),
                ray_caster_getter=ray_caster_getter,
                instance_mesh_data_getter=instance_mesh_data_getter,
            )
            final_scene_entries[image_name] = _attach_selection_metadata(
                entry,
                attachment_graph,
                final_selection_rank=final_selection_rank,
            )
            final_selection_rank += 1

        if not final_scene_entries:
            logger.info("Scene %s produced no cacheable referability entries -> skipping", scene_id)
            continue

        scene_cache.clear()
        for image_name, entry in sorted(
            final_scene_entries.items(),
            key=lambda item: int(item[1].get("final_selection_rank", FRAME_SELECTION_FALLBACK_RANK)),
        ):
            scene_cache[image_name] = entry

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2, ensure_ascii=False)

        processed += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)
    logger.info("Saved referability cache to %s", output_path)


if __name__ == "__main__":
    main()
