#!/usr/bin/env python3
"""One-click pipeline runner for CausalSpatial-Bench.

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
import json
import logging
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.frame_selector import compute_frame_object_visibility
from src.scene_parser import (
    EXCLUDED_LABELS,
    _load_scene_geometry,
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
from src.qa_generator import generate_all_questions
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("pipeline")
DEFAULT_VLM_URL = "http://183.129.178.195:60029/v1"
EXPECTED_REFERABILITY_CACHE_VERSION = "14.0"
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
QUESTION_REVIEW_CROP_MIN_PROJECTED_AREA_PX = 400.0
QUESTION_REVIEW_CROP_MIN_IN_FRAME_RATIO = 0.35
# L1 occlusion generation uses a stricter in-frame gate than crop review.
# Other question types now apply their own per-qtype mention thresholds.
QUESTION_MENTION_MIN_IN_FRAME_RATIO = 0.60
QUESTION_MENTION_FALLBACK_FIELDS = QUESTION_MENTION_FIELDS


def _load_referability_cache(path: Path) -> dict | None:
    if not path.exists():
        logger.warning("Referability cache not found: %s", path)
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    version = str(data.get("version", ""))
    if version != EXPECTED_REFERABILITY_CACHE_VERSION:
        raise ValueError(
            f"Referability cache version mismatch: expected {EXPECTED_REFERABILITY_CACHE_VERSION}, got {version or '<missing>'}. "
            "Regenerate the referability cache with the updated VLM prompts before running the pipeline."
        )
    logger.info("Loaded referability cache from %s", path)
    return data


def _get_referability_entry(cache: dict | None, scene_id: str, image_name: str) -> dict | None:
    if not cache:
        return None
    frames = cache.get("frames", cache)
    scene_frames = frames.get(scene_id)
    if isinstance(scene_frames, dict):
        return scene_frames.get(image_name)
    return frames.get(f"{scene_id}/{image_name}")


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


def _iter_question_referability_mentions(
    question: dict[str, object],
    objects_by_id: dict[int, dict[str, object]],
) -> list[dict[str, object]]:
    return _shared_collect_question_mentions(question, objects_by_id)


def _build_question_referability_audit(
    question: dict[str, object],
    *,
    objects_by_id: dict[int, dict[str, object]],
    referability_entry: dict[str, object] | None,
    frame_referable_ids: list[int],
) -> dict[str, object]:
    return _shared_build_question_referability_audit(
        question,
        objects_by_id=objects_by_id,
        label_statuses=(referability_entry or {}).get("label_statuses"),
        label_to_object_ids=(referability_entry or {}).get("label_to_object_ids"),
        frame_referable_ids=frame_referable_ids,
    )


def _apply_question_referability_filter(
    questions: list[dict[str, object]],
    *,
    objects_by_id: dict[int, dict[str, object]],
    referability_entry: dict[str, object] | None,
    frame_referable_ids: list[int],
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
        try:
            axis_align = load_axis_alignment(scene_dir)
            poses = load_scannet_poses(scene_dir, axis_alignment=axis_align)
        except Exception as e:
            logger.warning("Question review pose load failed for %s: %s", scene_id, e)
            errors.append("missing_pose_data")
        try:
            color_intrinsics = load_scannet_intrinsics(scene_dir)
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
) -> dict[str, object]:
    image_path = data_root / scene_id / "color" / image_name
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
                strict_mode=False,
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
        "image_path": image_path,
        "image_exists": image_exists,
        "image_b64": image_b64,
        "mime": mime,
        "objects_by_id": objects_by_id,
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
            )
        frame_contexts[(scene_id, image_name)] = _build_question_review_frame_context(
            scene_id=scene_id,
            image_name=image_name,
            data_root=data_root,
            scene_context=scene_contexts[scene_id],
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
        "and uniquely identifiable in the frame.\n"
        "You will receive the full scene image first, followed by one crop for each target instance.\n"
        "Each crop appears in the same order as the Targets list, so crop_index 1 refers to the first crop after "
        "the full image.\n"
        "Use the crop as the primary evidence and the full image only as context.\n"
        "Judge each crop_index independently.\n"
        "Return present only when the exact instance is clearly visible, belongs to the given label, and can be "
        "uniquely identified as a standalone instance in the frame.\n"
        "If the crop is too partial, blurry, heavily occluded, confusing among multiple same-label instances, or "
        "might only show a component/substructure of a larger object, return unsure instead of present.\n"
        "If the instance does not appear in the image, return absent.\n"
        "Return strict JSON only with this schema:\n"
        '{"objects":[{"crop_index":1,"status":"present","reason":"short reason"}]}\n'
        f"Question: {question_text}\n"
        f"Targets: {targets_json}"
    )


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


def _question_answer_prompt(question: dict[str, object]) -> str | None:
    question_text = str(question.get("question", "")).strip()
    options = question.get("options")
    if not question_text or not isinstance(options, list) or not options:
        return None

    prompt_lines = [question_text, ""]
    for idx, option in enumerate(options):
        prompt_lines.append(f"{chr(65 + idx)}) {option}")
    prompt_lines.append("")
    letters = [chr(65 + idx) for idx in range(len(options))]
    if len(letters) == 1:
        answer_choices = letters[0]
    elif len(letters) == 2:
        answer_choices = " or ".join(letters)
    else:
        answer_choices = ", ".join(letters[:-1]) + f", or {letters[-1]}"
    prompt_lines.append(
        f"Answer with a single letter only ({answer_choices}). Do not explain."
    )
    return "\n".join(prompt_lines)


def _parse_mcq_answer(raw: str) -> str | None:
    if not raw:
        return None
    stripped = raw.strip()
    if not stripped:
        return None

    upper = stripped.upper()
    if re.fullmatch(r"[ABCD]", upper):
        return upper

    match = re.match(r"^[\(\[]?([ABCD])(?:[\)\].:\s-]*)?$", upper)
    if match:
        return match.group(1)

    explicit_patterns = [
        r"\bANSWER(?:\s+IS)?\s*[:\-]?\s*[\(\[]?([ABCD])(?:[\)\]]|\b)",
        r"\bOPTION\s+([ABCD])\b",
        r"\bI\s+CHOOSE\s+([ABCD])\b",
        r"\bMY\s+ANSWER\s+IS\s+([ABCD])\b",
    ]
    for pattern in explicit_patterns:
        match = re.search(pattern, upper)
        if match:
            return match.group(1)

    standalone_letters = re.findall(r"\b([ABCD])\b", upper)
    unique_letters: list[str] = []
    for letter in standalone_letters:
        if letter not in unique_letters:
            unique_letters.append(letter)
    return unique_letters[0] if len(unique_letters) == 1 else None


def _question_option_for_answer(question: dict[str, object], answer: str | None) -> str | None:
    if answer not in {"A", "B", "C", "D"}:
        return None
    options = question.get("options")
    if not isinstance(options, list):
        return None
    idx = ord(answer) - ord("A")
    if idx < 0 or idx >= len(options):
        return None
    return str(options[idx])


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


def _make_question_answer_reviewer(client, model_name: str):
    logger.info("Using question answer review VLM model: %s", model_name)

    def _review(image_path: Path, question: dict[str, object]) -> dict[str, object]:
        prompt = _question_answer_prompt(question)
        gold_answer = str(question.get("answer", "")).strip().upper()
        if prompt is None:
            return {
                "decision": "manual_review",
                "predicted_answer": None,
                "gold_answer": gold_answer or None,
                "predicted_option": None,
                "gold_option": _question_option_for_answer(question, gold_answer),
                "reason": "missing question text or options",
                "raw_response": "",
            }
        if gold_answer not in {"A", "B", "C", "D"}:
            return {
                "decision": "manual_review",
                "predicted_answer": None,
                "gold_answer": gold_answer or None,
                "predicted_option": None,
                "gold_option": _question_option_for_answer(question, gold_answer),
                "reason": f"invalid gold answer: {gold_answer or '<missing>'}",
                "raw_response": "",
            }

        image_b64, mime = _image_path_to_base64(image_path)
        resp = _call_question_review_vlm(
            lambda: client.chat.completions.create(
                model=model_name,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{image_b64}"}},
                        {"type": "text", "text": prompt},
                    ],
                }],
                max_tokens=32,
                temperature=0,
            ),
            context=f"question answer review for {image_path.name}",
        )
        raw_text = (resp.choices[0].message.content or "").strip()
        predicted_answer = _parse_mcq_answer(raw_text)
        gold_option = _question_option_for_answer(question, gold_answer)
        predicted_option = _question_option_for_answer(question, predicted_answer)

        if predicted_answer is None:
            decision = "manual_review"
            reason = "could not parse model answer"
        elif predicted_answer != gold_answer:
            decision = "manual_review"
            reason = f"model answered {predicted_answer} but gold answer is {gold_answer}"
        else:
            decision = "pass"
            reason = ""

        return {
            "decision": decision,
            "predicted_answer": predicted_answer,
            "gold_answer": gold_answer or None,
            "predicted_option": predicted_option,
            "gold_option": gold_option,
            "reason": reason,
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


def _manual_review_reason_from_answer_review(review: dict[str, object]) -> str:
    predicted_answer = str(review.get("predicted_answer", "")).strip().upper()
    gold_answer = str(review.get("gold_answer", "")).strip().upper()
    predicted_option = str(review.get("predicted_option", "")).strip()
    gold_option = str(review.get("gold_option", "")).strip()
    reason = str(review.get("reason", "")).strip()

    if (
        predicted_answer in {"A", "B", "C", "D"}
        and gold_answer in {"A", "B", "C", "D"}
        and predicted_answer != gold_answer
    ):
        detail = f"VLM answered {predicted_answer}"
        if predicted_option:
            detail += f" ({predicted_option})"
        detail += f" but gold answer is {gold_answer}"
        if gold_option:
            detail += f" ({gold_option})"
        return detail
    if reason:
        return f"VLM answer review flagged this question: {reason}"
    return "VLM answer review marked this question for manual review."


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


def _should_review_question_answer(question: dict[str, object]) -> bool:
    options = question.get("options")
    return (
        isinstance(options, list)
        and len(options) >= 2
        and bool(str(question.get("question", "")).strip())
    )


def _is_answer_mismatch_review(review: dict[str, object]) -> bool:
    predicted_answer = str(review.get("predicted_answer", "")).strip().upper()
    gold_answer = str(review.get("gold_answer", "")).strip().upper()
    return (
        predicted_answer in {"A", "B", "C", "D"}
        and gold_answer in {"A", "B", "C", "D"}
        and predicted_answer != gold_answer
    )


def _review_question_object_presence(
    review_fn,
    answer_review_fn,
    *,
    question_index: int,
    question: dict[str, object],
    data_root: Path,
    frame_context_by_key: dict[tuple[str, str], dict[str, object]],
) -> dict[str, object]:
    reviewed_question = dict(question)
    reviewed_question["benchmark_index"] = int(question_index)
    review_reasons: list[str] = []

    scene_id = str(question.get("scene_id", "")).strip()
    image_name = str(question.get("image_name", "")).strip()
    frame_context = frame_context_by_key.get((scene_id, image_name))
    image_path = (
        frame_context.get("image_path")
        if isinstance(frame_context, dict) and isinstance(frame_context.get("image_path"), Path)
        else (data_root / scene_id / "color" / image_name)
    )

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

    answer_review: dict[str, object] = {"decision": "skipped"}
    if _should_review_question_answer(question):
        gold_answer = str(question.get("answer", "")).strip().upper()
        if not image_path.exists():
            answer_review = {
                "decision": "manual_review",
                "predicted_answer": None,
                "gold_answer": gold_answer or None,
                "predicted_option": None,
                "gold_option": _question_option_for_answer(question, gold_answer),
                "reason": f"image not found: {image_path.name}",
                "raw_response": "",
            }
        else:
            try:
                answer_review = answer_review_fn(image_path, question)
            except Exception as e:
                answer_review = {
                    "decision": "manual_review",
                    "predicted_answer": None,
                    "gold_answer": gold_answer or None,
                    "predicted_option": None,
                    "gold_option": _question_option_for_answer(question, gold_answer),
                    "reason": f"VLM answer review failed: {e}",
                    "raw_response": "",
                }
        if answer_review.get("decision") == "manual_review":
            review_reasons.append(_manual_review_reason_from_answer_review(answer_review))
    reviewed_question["question_answer_review"] = answer_review

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
) -> dict[str, object]:
    from scripts.make_viewer import build_viewer_html

    client, model_name = _resolve_question_review_vlm(
        vlm_url,
        vlm_model,
        purpose="question post-review",
    )
    answer_model_name = model_name
    _, review_fn = _make_question_presence_reviewer(client, model_name)
    _, answer_review_fn = _make_question_answer_reviewer(client, model_name)
    reviewed_questions: list[dict[str, object]] = []
    frame_context_by_key = _prebuild_question_review_frame_contexts(
        questions=questions,
        data_root=data_root,
        output_dir=output_dir,
    )

    if questions:
        with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
            futures = [
                pool.submit(
                    _review_question_object_presence,
                    review_fn,
                    answer_review_fn,
                    question_index=idx,
                    question=question,
                    data_root=data_root,
                    frame_context_by_key=frame_context_by_key,
                )
                for idx, question in enumerate(questions)
            ]
            for future in as_completed(futures):
                reviewed_questions.append(future.result())
    reviewed_questions.sort(key=lambda item: int(item.get("benchmark_index", -1)))

    referability_issue_count = sum(
        1
        for question in reviewed_questions
        if isinstance(question.get("question_presence_review"), dict)
        and question["question_presence_review"].get("decision") == "manual_review"
    )
    flagged_questions = [
        question for question in reviewed_questions
        if (
            isinstance(question.get("question_presence_review"), dict)
            and question["question_presence_review"].get("decision") == "manual_review"
        ) or (
            isinstance(question.get("question_answer_review"), dict)
            and question["question_answer_review"].get("decision") == "manual_review"
        )
    ]
    answer_review_question_count = sum(
        1 for question in reviewed_questions if _should_review_question_answer(question)
    )
    answer_mismatch_count = sum(
        1
        for question in reviewed_questions
        if isinstance(question.get("question_answer_review"), dict)
        and _is_answer_mismatch_review(question["question_answer_review"])
    )

    review_payload = {
        "name": "CausalSpatial-Bench question presence review",
        "model": model_name,
        "answer_review_model": answer_model_name,
        "reviewed_question_count": len(reviewed_questions),
        "manual_review_count": len(flagged_questions),
        "referability_issue_count": referability_issue_count,
        "answer_review_question_count": answer_review_question_count,
        "answer_mismatch_count": answer_mismatch_count,
        "questions": reviewed_questions,
    }
    flagged_payload = {
        "name": "CausalSpatial-Bench question presence review (flagged)",
        "model": model_name,
        "answer_review_model": answer_model_name,
        "reviewed_question_count": len(reviewed_questions),
        "manual_review_count": len(flagged_questions),
        "referability_issue_count": referability_issue_count,
        "answer_review_question_count": answer_review_question_count,
        "answer_mismatch_count": answer_mismatch_count,
        "questions": flagged_questions,
    }

    review_json_path = output_dir / "question_presence_review.json"
    flagged_json_path = output_dir / "question_presence_review_flagged.json"
    flagged_html_path = output_dir / "question_presence_review_flagged.html"

    with open(review_json_path, "w", encoding="utf-8") as f:
        json.dump(review_payload, f, indent=2, ensure_ascii=False)
    with open(flagged_json_path, "w", encoding="utf-8") as f:
        json.dump(flagged_payload, f, indent=2, ensure_ascii=False)

    flagged_html = build_viewer_html(
        flagged_questions,
        data_root,
        title="question presence manual review",
        include_referability_audit=False,
        apply_filters=False,
    )
    flagged_html_path.write_text(flagged_html, encoding="utf-8")

    logger.info(
        "Question presence review complete: %d reviewed, %d flagged. JSON: %s HTML: %s",
        len(reviewed_questions),
        len(flagged_questions),
        flagged_json_path,
        flagged_html_path,
    )
    return {
        "model": model_name,
        "answer_review_model": answer_model_name,
        "reviewed_question_count": len(reviewed_questions),
        "manual_review_count": len(flagged_questions),
        "referability_issue_count": referability_issue_count,
        "answer_review_question_count": answer_review_question_count,
        "answer_mismatch_count": answer_mismatch_count,
        "review_json_path": review_json_path,
        "flagged_json_path": flagged_json_path,
        "flagged_html_path": flagged_html_path,
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


def _has_l1_visibility_candidates(label_statuses: object) -> bool:
    if not isinstance(label_statuses, dict):
        return False
    for status in label_statuses.values():
        if str(status or "").strip().lower() == "absent":
            return True
    return False


def _frames_from_referability_cache(scene_frames: dict[str, dict]) -> list[dict[str, object]]:
    frames: list[dict[str, object]] = []
    for image_name, entry in sorted(scene_frames.items()):
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
            }
        )
    return frames


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
    object_reviews = (referability_entry or {}).get("object_reviews")
    if isinstance(object_reviews, dict):
        for obj_id in visible_ids:
            review = object_reviews.get(str(obj_id))
            if not isinstance(review, dict):
                review = object_reviews.get(obj_id)
            if not isinstance(review, dict):
                continue
            try:
                ratios_by_obj_id[int(obj_id)] = float(review.get("bbox_in_frame_ratio", 0.0) or 0.0)
            except (TypeError, ValueError):
                continue
    elif isinstance(object_reviews, list):
        for review in object_reviews:
            if not isinstance(review, dict):
                continue
            try:
                obj_id = int(review.get("obj_id"))
                ratio = float(review.get("bbox_in_frame_ratio", 0.0) or 0.0)
            except (TypeError, ValueError):
                continue
            if obj_id in visible_ids:
                ratios_by_obj_id[obj_id] = ratio

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
            strict_mode=False,
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
    """Return visible object ids whose projected bbox is sufficiently in-frame."""
    visible_ids = _normalize_object_ids(visible_object_ids)
    if not visible_ids:
        return []

    return [
        int(obj_id)
        for obj_id in visible_ids
        if float((mention_in_frame_ratio_by_obj_id or {}).get(int(obj_id), 0.0) or 0.0)
        >= QUESTION_MENTION_MIN_IN_FRAME_RATIO
    ]


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
    referability_entry: dict | None,
    frame_attachment_rows: list[dict[str, object]],
    generated_questions: list[dict] | None = None,
    pipeline_skip_reason: str | None = None,
) -> dict[str, object]:
    generated_questions = [] if generated_questions is None else [dict(q) for q in generated_questions]
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
        "referable_object_ids": _normalize_object_ids((referability_entry or {}).get("referable_object_ids")),
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
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


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


def run_pipeline(
    data_root: Path,
    output_dir: Path,
    max_scenes: int = 300,
    max_frames: int = 5,
    use_occlusion: bool = True,
    referability_cache: dict | None = None,
    occlusion_backend: str = "mesh_ray",
    vlm_url: str | None = None,
    vlm_model: str | None = None,
    write_frame_debug: bool = True,
    run_question_presence_review: bool = True,
    question_presence_review_workers: int = 8,
):
    """Execute the full CausalSpatial-Bench data generation pipeline."""
    if referability_cache is None:
        raise ValueError(
            "run_pipeline requires a referability_cache generated by scripts/run_vlm_referability.py"
        )

    meta_dir = output_dir / "scene_metadata"
    questions_dir = output_dir / "questions"
    frame_debug_dir = output_dir / "frame_debug"
    meta_dir.mkdir(parents=True, exist_ok=True)
    questions_dir.mkdir(parents=True, exist_ok=True)
    if write_frame_debug:
        frame_debug_dir.mkdir(parents=True, exist_ok=True)

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
    raw_question_count = 0
    debug_scene_ids: list[str] = []
    processed_scene_ids: list[str] = []
    final_questions: list[dict] = []

    raw_questions_dir = output_dir / "_raw_questions_tmp"
    raw_questions_dir.mkdir(parents=True, exist_ok=True)
    for stale_raw_question_path in raw_questions_dir.glob("*.json"):
        try:
            stale_raw_question_path.unlink()
        except OSError:
            pass
    raw_question_paths: list[Path] = []

    try:
        for scene_index, scene_dir in enumerate(scene_dirs, start=1):
            scene_id = scene_dir.name
            logger.info(
                "=== Processing scene %s (%d/%d) ===",
                scene_id,
                scene_index,
                total_scenes,
            )

            scene_questions: list[dict] = []
            preloaded_geometry = None
            needs_mesh_resources = occlusion_backend in ("depth", "mesh_ray")
            if needs_mesh_resources:
                try:
                    preloaded_geometry = _load_scene_geometry(scene_dir)
                except Exception as e:
                    logger.warning("Scene geometry preload failed for %s: %s", scene_id, e)

            scene = parse_scene(scene_dir, preloaded_geometry=preloaded_geometry)
            if scene is None:
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
                continue

            _write_json_file(meta_dir / f"{scene_id}.json", scene)

            scene_frames = _get_referability_scene_frames(referability_cache, scene_id)
            frames = _frames_from_referability_cache(scene_frames)
            if len(frames) > frame_limit:
                frames = frames[:frame_limit]
            if not frames:
                logger.info("No valid frames for scene %s after cache filtering; skipping", scene_id)
                continue

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

            depth_intrinsics = None
            if use_occlusion:
                try:
                    depth_intrinsics = load_scannet_depth_intrinsics(scene_dir)
                except Exception as e:
                    logger.warning("Depth intrinsics load failed for %s: %s", scene_id, e)

            try:
                color_intrinsics = load_scannet_intrinsics(scene_dir)
            except Exception as e:
                logger.warning("Color intrinsics load failed for %s: %s", scene_id, e)
                color_intrinsics = None

            scene_frame_debug_entries: list[dict[str, object]] = []

            for frame in frames:
                image_name = frame["image_name"]
                if image_name not in poses:
                    if write_frame_debug:
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
                                pipeline_visible_ids=[],
                                occlusion_eligible_object_ids=[],
                                referability_entry=_get_referability_entry(
                                    referability_cache,
                                    scene_id,
                                    image_name,
                                ),
                                frame_attachment_rows=frame_attachment_rows,
                                pipeline_skip_reason="missing_pose",
                            )
                        )
                    continue

                camera_pose = poses[image_name]
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
                label_statuses = None
                label_counts = None
                referability_entry = _get_referability_entry(
                    referability_cache,
                    scene_id,
                    image_name,
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
                    referable_ids = [
                        int(obj_id)
                        for obj_id in referability_entry.get("referable_object_ids", [])
                        if int(obj_id) in visible_id_set
                    ]
                    if not referable_ids and not _has_l1_visibility_candidates(label_statuses):
                        if write_frame_debug:
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
                                    referability_entry=referability_entry,
                                    frame_attachment_rows=frame_attachment_rows,
                                    pipeline_skip_reason="no_referable_objects_or_l1_candidates",
                                )
                            )
                        logger.debug(
                            "Frame %s/%s has no referable objects or L1 visibility candidates",
                            scene_id,
                            image_name,
                        )
                        continue

                questions = generate_all_questions(
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
                    occlusion_eligible_object_ids=occlusion_eligible_ids,
                    mention_in_frame_ratio_by_obj_id=mention_in_frame_ratio_by_obj_id,
                    label_statuses=label_statuses,
                    label_counts=label_counts,
                    label_to_object_ids=(referability_entry or {}).get("label_to_object_ids"),
                    room_bounds=scene.get("room_bounds"),
                    wall_objects=scene.get("wall_objects"),
                    attachment_edges=scene.get("attachment_edges", []),
                )

                for q in questions:
                    q["scene_id"] = scene_id
                    q["image_name"] = image_name

                kept_questions, audited_questions = _apply_question_referability_filter(
                    questions,
                    objects_by_id=objects_by_id,
                    referability_entry=referability_entry,
                    frame_referable_ids=referable_ids or [],
                )

                scene_questions.extend(kept_questions)
                frame_attachment_rows = _filter_frame_attachment_rows(
                    scene_attachment_rows,
                    set(selector_visible_ids) | set(int(obj_id) for obj_id in visible_ids),
                )
                if write_frame_debug:
                    scene_frame_debug_entries.append(
                        _build_frame_debug_entry(
                            image_name=image_name,
                            scene_objects=scene["objects"],
                            objects_by_id=objects_by_id,
                            selector_visible_ids=selector_visible_ids,
                            pipeline_visible_ids=list(visible_ids),
                            occlusion_eligible_object_ids=occlusion_eligible_ids,
                            referability_entry=referability_entry,
                            frame_attachment_rows=frame_attachment_rows,
                            generated_questions=audited_questions,
                        )
                    )

            raw_question_path = raw_questions_dir / f"{scene_id}.json"
            _write_json_file(raw_question_path, scene_questions)
            raw_question_paths.append(raw_question_path)
            raw_question_count += len(scene_questions)
            processed_scene_ids.append(scene_id)

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
                debug_scene_ids.append(scene_id)

            logger.info(
                "Scene %s: %d raw question(s); raw total %d",
                scene_id,
                len(scene_questions),
                raw_question_count,
            )

        all_questions: list[dict] = []
        for raw_question_path in raw_question_paths:
            with open(raw_question_path, "r", encoding="utf-8") as f:
                scene_questions = json.load(f)
            if isinstance(scene_questions, list):
                all_questions.extend(scene_questions)

        logger.info(
            "Running benchmark quality control on %d raw questions (viewer-only attachment filtering excluded)",
            raw_question_count,
        )
        final_questions = full_quality_pipeline(all_questions)
        all_questions = []

        by_scene: dict[str, list] = defaultdict(list)
        final_by_scene_frame: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
        for q in final_questions:
            by_scene[q["scene_id"]].append(q)
            final_by_scene_frame[q["scene_id"]][q["image_name"]].append(q)

        for scene_id in processed_scene_ids:
            _write_json_file(questions_dir / f"{scene_id}.json", by_scene.get(scene_id, []))

        if write_frame_debug:
            for scene_id in debug_scene_ids:
                _finalize_scene_debug_file(
                    frame_debug_dir / f"{scene_id}.json",
                    final_questions_by_frame=final_by_scene_frame.get(scene_id, {}),
                )
    finally:
        for raw_question_path in raw_question_paths:
            try:
                raw_question_path.unlink()
            except OSError:
                pass
        try:
            raw_questions_dir.rmdir()
        except OSError:
            pass

    benchmark = {
        "name": "CausalSpatial-Bench",
        "version": "1.0",
        "statistics": compute_statistics(final_questions),
        "questions": final_questions,
    }
    benchmark_path = output_dir / "benchmark.json"
    _write_json_file(benchmark_path, benchmark)

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
        len(final_questions),
        benchmark_path,
    )
    return final_questions
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
                depth_intrinsics = load_scannet_depth_intrinsics(scene_dir)
            except Exception as e:
                logger.warning("Depth intrinsics load failed for %s: %s", scene_id, e)

        # Load colour intrinsics for local ROI blur check
        try:
            color_intrinsics = load_scannet_intrinsics(scene_dir)
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
            label_statuses = None
            label_counts = None
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
                referable_ids = [
                    int(obj_id) for obj_id in referability_entry.get("referable_object_ids", [])
                    if int(obj_id) in visible_id_set
                ]
                if not referable_ids and not _has_l1_visibility_candidates(label_statuses):
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

            questions = generate_all_questions(
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
                occlusion_eligible_object_ids=occlusion_eligible_ids,
                mention_in_frame_ratio_by_obj_id=mention_in_frame_ratio_by_obj_id,
                label_statuses=label_statuses,
                label_counts=label_counts,
                label_to_object_ids=(referability_entry or {}).get("label_to_object_ids"),
                room_bounds=scene.get("room_bounds"),
                wall_objects=scene.get("wall_objects"),
                attachment_edges=scene.get("attachment_edges", []),
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

    by_scene: dict[str, list] = defaultdict(list)
    final_by_scene_frame: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for q in final_questions:
        by_scene[q["scene_id"]].append(q)
        final_by_scene_frame[q["scene_id"]][q["image_name"]].append(q)

    for sid, qs in by_scene.items():
        with open(questions_dir / f"{sid}.json", "w", encoding="utf-8") as f:
            json.dump(qs, f, indent=2, ensure_ascii=False)

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
        "name":       "CausalSpatial-Bench",
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
        description="CausalSpatial-Bench data generation pipeline"
    )
    parser.add_argument(
        "--data_root", type=str,
        default=os.getenv("SCANNET_PATH", "/home/lihongxing/datasets/ScanNet/data/scans"),
        help="Root directory of ScanNet scans (contains scene subdirectories)",
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
        help="JSON cache of VLM frame/object referability decisions produced by scripts/run_vlm_referability.py",
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
        "--question_presence_review",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="After benchmark generation, ask the VLM whether mentioned objects are clearly visible and uniquely identifiable, then answer each question and export flagged samples for manual review",
    )
    parser.add_argument(
        "--question_presence_review_workers",
        type=int,
        default=8,
        help="Thread pool size for post-generation question presence/answer review",
    )
    args = parser.parse_args()

    if args.label_map:
        load_scannet_label_map(args.label_map)

    referability_cache = _load_referability_cache(Path(args.referability_cache))

    run_pipeline(
        data_root=Path(args.data_root),
        output_dir=Path(args.output_dir),
        max_scenes=args.max_scenes,
        max_frames=args.max_frames,
        use_occlusion=not args.no_occlusion,
        referability_cache=referability_cache,
        occlusion_backend=args.occlusion_backend,
        vlm_url=args.vlm_url,
        vlm_model=args.vlm_model,
        write_frame_debug=args.write_frame_debug,
        run_question_presence_review=args.question_presence_review,
        question_presence_review_workers=args.question_presence_review_workers,
    )


if __name__ == "__main__":
    main()
