#!/usr/bin/env python3
"""VLM-based frame-quality and per-object referability prefilter.

This script runs *before* QA generation. For each selected frame it asks a VLM:
  1. how clear the full frame is and whether it is severely out of focus;
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
import json
import logging
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.frame_selector import (
    compute_frame_object_visibility,
    refine_visible_ids_with_depth,
    select_frames,
)
from src.utils.colmap_loader import (
    load_axis_alignment,
    load_scannet_depth_intrinsics,
    load_scannet_intrinsics,
    load_scannet_poses,
)
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
REFERABILITY_CACHE_VERSION = "11.0"

QUESTION_REVIEW_CROP_PADDING_RATIO = 0.10
QUESTION_REVIEW_CROP_MIN_PADDING_PX = 12
QUESTION_REVIEW_CROP_MAX_PADDING_PX = 80
QUESTION_REVIEW_CROP_MIN_DIM_PX = 16
QUESTION_REVIEW_CROP_MIN_PROJECTED_AREA_PX = 400.0
FRAME_SELECTION_CANDIDATE_MULTIPLIER = 3
FRAME_QUALITY_PRIMARY_WEIGHT = 1000
FRAME_USABLE_BONUS = 100000

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

LABEL_STATUS_UNIQUE = "unique"
LABEL_STATUS_MULTIPLE = "multiple"
LABEL_STATUS_ABSENT = "absent"
LABEL_STATUS_UNSURE = "unsure"


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


def _frame_prompt() -> str:
    return (
        "You are given one original indoor scene image. "
        "Evaluate the full-frame image clarity first. "
        "Slight softness is acceptable; only mark severely_out_of_focus=true when the image has obvious global defocus blur, autofocus failure, or severe blur that makes object boundaries hard to read. "
        "Then decide whether the frame is still usable for spatial-reasoning questions. "
        "A frame can be usable even if it is slightly soft, as long as the scene layout and object extents remain readable. "
        "Return a clarity_score from 0 to 100, where higher means clearer and sharper. "
        "Prioritize image clarity over scene semantics when assigning the score. "
        'Answer with strict JSON only: {"clarity_score": 78, "severely_out_of_focus": false, "usable_for_spatial_reasoning": true, "reason": "clear enough with minor softness"}'
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


def _full_frame_label_review_prompt(label: str) -> str:
    return (
        "You are given one indoor scene image. "
        "Decide how many objects in the full frame would reasonably be called "
        f"{json.dumps(str(label), ensure_ascii=False)} by a human viewer. "
        "Return unique when exactly one object in the image would be called that label. "
        "Return multiple when two or more objects in the image would be called that label. "
        "Return absent when none would be called that label. "
        "Return unsure when you cannot decide confidently. "
        'Answer with strict JSON only using this schema: {"status": "unique", "reason": "short reason"}'
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


def _normalize_frame_review(value: dict[str, Any] | None) -> dict[str, Any]:
    parsed = value if isinstance(value, dict) else {}
    fallback_usable = (
        bool(parsed.get("frame_usable"))
        if "usable_for_spatial_reasoning" not in parsed and isinstance(parsed.get("frame_usable"), bool)
        else True
    )
    clarity_score = _normalize_clarity_score(parsed.get("clarity_score"), default=60)
    severely_out_of_focus = _coerce_bool(parsed.get("severely_out_of_focus"), default=False)
    usable_for_spatial_reasoning = _coerce_bool(
        parsed.get("usable_for_spatial_reasoning"),
        default=fallback_usable,
    )
    return {
        "clarity_score": clarity_score,
        "severely_out_of_focus": severely_out_of_focus,
        "usable_for_spatial_reasoning": usable_for_spatial_reasoning,
        "frame_usable": usable_for_spatial_reasoning and not severely_out_of_focus,
        "reason": str(parsed.get("reason", "")).strip() or "frame_quality_parse_fallback",
    }


def _frame_selection_score(selector_score: int, frame_info: dict[str, Any]) -> int:
    normalized = _normalize_frame_review(frame_info)
    usable_bonus = FRAME_USABLE_BONUS if normalized["frame_usable"] else 0
    clarity_score = normalized["clarity_score"]
    return usable_bonus + (clarity_score * FRAME_QUALITY_PRIMARY_WEIGHT) + int(selector_score)


def _frame_decision(
    client,
    model: str,
    image: np.ndarray,
) -> dict[str, Any]:
    full_b64 = _image_to_base64(image)
    default = {
        "clarity_score": 60,
        "severely_out_of_focus": False,
        "usable_for_spatial_reasoning": True,
        "reason": "frame_quality_parse_fallback",
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


def _full_frame_label_review_decision(
    client,
    model: str,
    image_b64: str,
    label: str,
) -> tuple[str, str]:
    default = {"status": LABEL_STATUS_UNSURE, "reason": "parse_fallback"}
    parsed, raw_text = _call_vlm_json(
        client,
        model,
        [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
            {"type": "text", "text": _full_frame_label_review_prompt(label)},
        ],
        default=default,
        max_tokens=128,
    )
    status = (
        _normalize_full_frame_label_status(
            parsed.get("status"),
            count=parsed.get("count"),
        )
        or LABEL_STATUS_UNSURE
    )
    return status, raw_text


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
    depth_image: np.ndarray | None,
    depth_intrinsics,
) -> tuple[list[int], str]:
    selector_ids = sorted({int(obj_id) for obj_id in visible_object_ids})
    if depth_image is None or depth_intrinsics is None:
        return selector_ids, "projection_fallback"
    try:
        refined = refine_visible_ids_with_depth(
            visible_object_ids=selector_ids,
            objects=objects,
            pose=camera_pose,
            depth_image=depth_image,
            depth_intrinsics=depth_intrinsics,
        )
    except Exception as exc:
        logger.warning("Depth refine failed: %s", exc)
        return selector_ids, "projection_fallback"
    return sorted({int(obj_id) for obj_id in refined}), "depth_refined"


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


def _build_object_review_crop(
    image: np.ndarray,
    visibility_meta: dict[str, Any] | None,
) -> dict[str, Any]:
    meta = visibility_meta or {}
    roi_bounds = meta.get("roi_bounds_px")
    projected_area_px = float(meta.get("projected_area_px", 0.0) or 0.0)
    bbox_in_frame_ratio = float(meta.get("bbox_in_frame_ratio", 0.0) or 0.0)
    edge_margin_px = float(meta.get("edge_margin_px", 0.0) or 0.0)
    result = {
        "valid": False,
        "local_outcome": LOCAL_OUTCOME_OUT_OF_FRAME,
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
        "local_outcome": str(crop_entry.get("local_outcome", "")),
        "local_reason": str(crop_entry.get("reason", "")),
        "vlm_status": None,
        "raw_response": None,
        "roi_bounds_px": crop_entry.get("roi_bounds_px"),
        "crop_bounds_px": crop_entry.get("crop_bounds_px"),
        "projected_area_px": crop_entry.get("projected_area_px"),
        "bbox_in_frame_ratio": crop_entry.get("bbox_in_frame_ratio"),
        "edge_margin_px": crop_entry.get("edge_margin_px"),
    }


def _is_absent_like_review(review: dict[str, Any]) -> bool:
    local_outcome = str(review.get("local_outcome", "")).strip().lower()
    status = _normalize_object_review_status(review.get("vlm_status"))
    return local_outcome in {LOCAL_OUTCOME_OUT_OF_FRAME, LOCAL_OUTCOME_EXCLUDED} or status == OBJECT_STATUS_ABSENT


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
            status = _normalize_object_review_status(review.get("vlm_status"))
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
        depth_image,
        depth_intrinsics,
    )
    candidate_labels, label_to_object_ids = _build_frame_label_candidates(
        candidate_visible_object_ids,
        objects_by_id,
    )

    normalized_frame_info = (
        _normalize_frame_review(frame_info)
        if isinstance(frame_info, dict)
        else _frame_decision(client, model_name, image)
    )
    selector_score_value = int(selector_score) if selector_score is not None else len(selector_visible_object_ids)
    selection_score_value = (
        int(frame_selection_score)
        if frame_selection_score is not None
        else _frame_selection_score(selector_score_value, normalized_frame_info)
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
    referable_object_ids: list[int] = []

    if normalized_frame_info["frame_usable"]:
        visibility_by_obj_id = compute_frame_object_visibility(
            scene_objects,
            camera_pose,
            color_intrinsics,
            image_path=image_path,
            depth_image=depth_image,
            depth_intrinsics=depth_intrinsics,
            strict_mode=False,
        )
        image_b64: str | None = None
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
            if crop_entry.get("local_outcome") == LOCAL_OUTCOME_REVIEWED:
                if image_b64 is None:
                    image_b64 = _image_to_base64(image)
                status, raw_response = _object_review_decision(
                    client,
                    model_name,
                    image_b64,
                    str(crop_entry.get("image_b64", "") or ""),
                    label,
                )
                review["vlm_status"] = status
                review["raw_response"] = raw_response or None
            object_reviews[int(obj_id)] = review
        crop_label_statuses, crop_label_counts, crop_referable_object_ids, crop_unique_label_object_ids = _aggregate_crop_label_reviews(
            label_to_object_ids,
            object_reviews,
        )
        # The cache's primary referability fields are crop-based. Full-frame
        # reviews are preserved for diagnostics but do not gate generation.
        label_statuses = dict(crop_label_statuses)

        if crop_unique_label_object_ids:
            if image_b64 is None:
                image_b64 = _image_to_base64(image)
            for label, obj_id in sorted(crop_unique_label_object_ids.items()):
                status, raw_response = _full_frame_label_review_decision(
                    client,
                    model_name,
                    str(image_b64 or ""),
                    label,
                )
                status = _normalize_full_frame_label_status(status) or LABEL_STATUS_UNSURE
                full_frame_label_reviews.append(
                    {
                        "label": str(label),
                        "status": status,
                        "crop_status": crop_label_statuses.get(label),
                        "crop_clear_count": crop_label_counts.get(label),
                        "crop_referable_object_id": int(obj_id),
                        "raw_response": raw_response or None,
                    }
                )
                full_frame_label_statuses[str(label)] = status

        full_frame_label_statuses = dict(sorted(full_frame_label_statuses.items()))
        full_frame_label_counts = _label_counts_from_statuses(full_frame_label_statuses)
        label_statuses = dict(sorted(label_statuses.items()))
        label_counts = _label_counts_from_statuses(label_statuses)
        referable_object_ids = sorted(set(int(obj_id) for obj_id in crop_referable_object_ids))

    return {
        "frame_usable": normalized_frame_info["frame_usable"],
        "frame_reject_reason": None if normalized_frame_info["frame_usable"] else normalized_frame_info["reason"],
        "selector_score": selector_score_value,
        "frame_quality_score": _normalize_clarity_score(normalized_frame_info.get("clarity_score"), default=60),
        "frame_quality_severely_out_of_focus": _coerce_bool(
            normalized_frame_info.get("severely_out_of_focus"),
            default=False,
        ),
        "frame_quality_usable_for_spatial_reasoning": _coerce_bool(
            normalized_frame_info.get("usable_for_spatial_reasoning"),
            default=True,
        ),
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
        "vlm_label_reviews": list(full_frame_label_reviews),
        "label_statuses": dict(sorted(label_statuses.items())),
        "label_counts": dict(sorted(label_counts.items())),
        "referable_object_ids": referable_object_ids,
        "vlm_unique_object_ids": list(referable_object_ids),
    }


def _frame_entry_has_debug_fields(entry: Any) -> bool:
    if not isinstance(entry, dict):
        return False
    required_keys = {
        "frame_quality_score",
        "frame_quality_severely_out_of_focus",
        "frame_quality_usable_for_spatial_reasoning",
        "frame_quality_reason",
        "frame_selection_score",
        "candidate_visible_object_ids",
        "candidate_visibility_source",
        "candidate_labels",
        "label_to_object_ids",
        "selector_visible_object_ids",
        "selector_visible_label_counts",
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
    return required_keys.issubset(entry.keys())


def _select_and_rerank_frames(
    *,
    client,
    model_name: str,
    scene_dir: Path,
    frame_candidates: list[dict[str, Any]],
    max_frames: int,
) -> list[dict[str, Any]]:
    color_dir = scene_dir / "color"
    reranked: list[dict[str, Any]] = []
    for frame in frame_candidates:
        image_name = str(frame.get("image_name", "")).strip()
        if not image_name:
            continue
        image_path = color_dir / image_name
        image = cv2.imread(str(image_path))
        if image is None:
            logger.warning("Cannot read image %s", image_path)
            continue
        frame_info = _frame_decision(client, model_name, image)
        selector_score = int(frame.get("score", frame.get("n_visible", 0)) or 0)
        reranked.append(
            {
                **frame,
                "selector_score": selector_score,
                "frame_info": frame_info,
                "frame_selection_score": _frame_selection_score(selector_score, frame_info),
            }
        )

    reranked.sort(
        key=lambda entry: (
            int(entry.get("frame_selection_score", 0)),
            int(entry.get("selector_score", 0)),
            int(entry.get("n_visible", 0)),
            str(entry.get("image_name", "")),
        ),
        reverse=True,
    )
    selected = reranked[:max(0, int(max_frames))]
    if reranked:
        logger.info(
            "VLM reranked %d geometric frame candidates for %s; kept %d (usable candidates=%d, best clarity=%d, best rerank score=%d)",
            len(reranked),
            scene_dir.name,
            len(selected),
            sum(1 for entry in reranked if entry.get("frame_info", {}).get("frame_usable", True)),
            int(reranked[0].get("frame_info", {}).get("clarity_score", 0)),
            int(reranked[0].get("frame_selection_score", 0)),
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

        frame_candidate_limit = max(
            int(args.max_frames),
            int(args.max_frames) * FRAME_SELECTION_CANDIDATE_MULTIPLIER,
        )
        frame_candidates = select_frames(
            scene_dir,
            scene["objects"],
            attachment_graph,
            frame_candidate_limit,
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
        frames = _select_and_rerank_frames(
            client=client,
            model_name=model_name,
            scene_dir=scene_dir,
            frame_candidates=frame_candidates,
            max_frames=int(args.max_frames),
        )
        if not frames:
            logger.info("Scene %s has no reranked frames -> skipping", scene_id)
            continue
        selected_names = {str(frame["image_name"]) for frame in frames}
        for stale_name in list(scene_cache.keys()):
            if stale_name not in selected_names:
                del scene_cache[stale_name]
        pending_frames = [
            frame
            for frame in frames
            if not _frame_entry_has_debug_fields(scene_cache.get(frame["image_name"]))
        ]
        if not pending_frames:
            logger.info("Scene %s selected frames already cached -> skipping", scene_id)
            processed += 1
            continue
        logger.info(
            "Processing referability scene %s (%d/%d) with %d selected frames (%d pending after resume)",
            scene_id,
            processed + 1,
            args.max_scenes,
            len(frames),
            len(pending_frames),
        )

        for frame in pending_frames:
            image_name = frame["image_name"]
            if image_name not in poses:
                continue

            image_path = scene_dir / "color" / image_name
            image = cv2.imread(str(image_path))
            if image is None:
                logger.warning("Cannot read image %s", image_path)
                continue

            camera_pose = poses[image_name]
            selector_visible_object_ids = [
                int(obj_id)
                for obj_id in frame["visible_object_ids"]
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

            scene_cache[image_name] = _compute_frame_referability_entry(
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
            )

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
