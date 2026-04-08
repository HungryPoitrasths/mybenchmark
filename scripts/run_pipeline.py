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
import json
import logging
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

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
EXPECTED_REFERABILITY_CACHE_VERSION = "9.0"
VLM_API_KEY_ENV_NAMES = ("DASHSCOPE_API_KEY", "OPENAI_API_KEY")
PLACEHOLDER_VLM_API_KEY = "EMPTY"
QUESTION_REVIEW_CROP_PADDING_RATIO = 0.10
QUESTION_REVIEW_CROP_MIN_PADDING_PX = 12
QUESTION_REVIEW_CROP_MAX_PADDING_PX = 80
QUESTION_REVIEW_CROP_MIN_DIM_PX = 16
QUESTION_REVIEW_CROP_MIN_PROJECTED_AREA_PX = 400.0
QUESTION_REVIEW_CROP_MIN_IN_FRAME_RATIO = 0.35
QUESTION_MENTION_FALLBACK_FIELDS = (
    ("obj_a_id", "obj_a_label", "obj_a"),
    ("obj_b_id", "obj_b_label", "obj_b"),
    ("obj_ref_id", "obj_ref_label", "obj_ref"),
    ("obj_face_id", "obj_face_label", "obj_face"),
    ("obj_target_id", "obj_target_label", "obj_target"),
    ("moved_obj_id", "moved_obj_label", "moved_obj"),
    ("query_obj_id", "query_obj_label", "query_obj"),
    ("obj_c_id", "obj_c_label", "obj_c"),
    ("removed_obj_id", "removed_obj_label", "removed_obj"),
    ("grandparent_id", "grandparent_label", "grandparent"),
    ("parent_id", "parent_label", "parent"),
    ("grandchild_id", "grandchild_label", "grandchild"),
    ("neighbor_id", "neighbor_label", "neighbor"),
)


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


def _normalize_label_to_object_ids(value: object) -> dict[str, list[int]]:
    label_to_object_ids: dict[str, list[int]] = {}
    if not isinstance(value, dict):
        return label_to_object_ids
    for key, obj_ids in value.items():
        if not isinstance(key, str):
            continue
        label = key.strip().lower()
        if not label:
            continue
        label_to_object_ids[label] = _normalize_object_ids(obj_ids)
    return dict(sorted(label_to_object_ids.items()))


def _coerce_object_id(value: object) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _iter_question_referability_mentions(
    question: dict[str, object],
    objects_by_id: dict[int, dict[str, object]],
) -> list[dict[str, object]]:
    mentions: list[dict[str, object]] = []
    seen: set[tuple[object, str, str]] = set()
    raw_mentions = question.get("mentioned_objects")

    def _append_mention(
        *,
        role: object,
        label_value: object,
        obj_id_value: object,
    ) -> None:
        obj_id = _coerce_object_id(obj_id_value)
        label = str(label_value or "").strip()
        if not label and obj_id is not None:
            label = str(objects_by_id.get(obj_id, {}).get("label", "")).strip()
        if not label and obj_id is None:
            return
        role_text = str(role or "mentioned").strip() or "mentioned"
        key = (obj_id if obj_id is not None else "", label.lower(), role_text)
        if key in seen:
            return
        seen.add(key)
        mentions.append(
            {
                "role": role_text,
                "label": label,
                "obj_id": obj_id,
                "obj_id_parse_failed": (
                    obj_id is None and obj_id_value not in (None, "")
                ),
            }
        )

    if isinstance(raw_mentions, list) and raw_mentions:
        for mention in raw_mentions:
            if not isinstance(mention, dict):
                continue
            _append_mention(
                role=mention.get("role"),
                label_value=mention.get("label"),
                obj_id_value=mention.get("obj_id"),
            )
        return mentions

    for id_key, label_key, role in QUESTION_MENTION_FALLBACK_FIELDS:
        _append_mention(
            role=role,
            label_value=question.get(label_key),
            obj_id_value=question.get(id_key),
        )
    return mentions


def _build_question_referability_audit(
    question: dict[str, object],
    *,
    objects_by_id: dict[int, dict[str, object]],
    referability_entry: dict[str, object] | None,
    frame_referable_ids: list[int],
) -> dict[str, object]:
    referable_set = set(int(obj_id) for obj_id in frame_referable_ids)
    label_statuses = _normalize_label_statuses((referability_entry or {}).get("label_statuses"))
    label_to_object_ids = _normalize_label_to_object_ids((referability_entry or {}).get("label_to_object_ids"))
    audited_mentions: list[dict[str, object]] = []
    reason_codes: list[str] = []
    seen_reasons: set[str] = set()

    def _add_reason(code: str, mention_reasons: list[str]) -> None:
        if code not in mention_reasons:
            mention_reasons.append(code)
        if code not in seen_reasons:
            seen_reasons.add(code)
            reason_codes.append(code)

    for mention in _iter_question_referability_mentions(question, objects_by_id):
        label = str(mention.get("label", "")).strip()
        label_key = label.lower()
        mention_obj_id = _coerce_object_id(mention.get("obj_id"))
        label_status = label_statuses.get(label_key)
        candidate_ids = label_to_object_ids.get(label_key, [])
        referable_label_ids = [
            obj_id for obj_id in candidate_ids
            if int(obj_id) in referable_set
        ]
        mention_reasons: list[str] = []

        if mention_obj_id is not None and mention_obj_id not in referable_set:
            _add_reason("mentioned_nonreferable_object", mention_reasons)

        if label_key:
            if label_status == "multiple":
                _add_reason("mentioned_label_not_unique", mention_reasons)
            elif label_status != "unique":
                _add_reason("mentioned_label_not_resolved", mention_reasons)

            if len(referable_label_ids) != 1:
                _add_reason("mentioned_label_not_resolved", mention_reasons)
            elif mention_obj_id is not None and mention_obj_id not in set(referable_label_ids):
                _add_reason("mentioned_nonreferable_object", mention_reasons)
        elif mention_obj_id is None:
            _add_reason("mentioned_label_not_resolved", mention_reasons)

        audited_mentions.append(
            {
                "role": str(mention.get("role", "mentioned")),
                "label": label,
                "obj_id": mention_obj_id,
                "obj_id_parse_failed": bool(mention.get("obj_id_parse_failed", False)),
                "label_status": label_status,
                "candidate_object_ids": candidate_ids,
                "referable_object_ids": referable_label_ids,
                "passes_referability_check": not mention_reasons,
                "reason_codes": mention_reasons,
            }
        )

    return {
        "decision": "drop" if reason_codes else "pass",
        "reason_codes": reason_codes,
        "mentioned_objects": audited_mentions,
        "frame_referable_object_ids": sorted(referable_set),
    }


def _apply_question_referability_filter(
    questions: list[dict[str, object]],
    *,
    objects_by_id: dict[int, dict[str, object]],
    referability_entry: dict[str, object] | None,
    frame_referable_ids: list[int],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    kept_questions: list[dict[str, object]] = []
    audited_questions: list[dict[str, object]] = []
    dropped = 0

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
        dropped += 1

    if dropped:
        logger.info(
            "Referability backstop removed %d questions for frame %s/%s",
            dropped,
            str(questions[0].get("scene_id", "<unknown>")) if questions else "<unknown>",
            str(questions[0].get("image_name", "<unknown>")) if questions else "<unknown>",
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


def _has_l1_visibility_candidates(label_counts: object) -> bool:
    if not isinstance(label_counts, dict):
        return False
    for count in label_counts.values():
        try:
            count_int = int(count)
        except (TypeError, ValueError):
            continue
        if count_int in (0, 1):
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
        "candidate_visibility_source": (referability_entry or {}).get("candidate_visibility_source"),
        "candidate_visible_label_counts": _normalize_label_counts(
            (referability_entry or {}).get("candidate_visible_label_counts")
        ),
        "vlm_label_statuses": _normalize_label_statuses((referability_entry or {}).get("label_statuses")),
        "vlm_label_counts": _normalize_label_counts((referability_entry or {}).get("label_counts")),
        "referable_object_ids": _normalize_object_ids((referability_entry or {}).get("referable_object_ids")),
        "candidate_labels": list((referability_entry or {}).get("candidate_labels", [])),
        "label_to_object_ids": {
            str(label): _normalize_object_ids(obj_ids)
            for label, obj_ids in label_to_object_ids.items()
        },
        "vlm_label_reviews": list((referability_entry or {}).get("vlm_label_reviews", [])),
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
):
    """Execute the full CausalSpatial-Bench data generation pipeline."""
    if referability_cache is None:
        raise ValueError(
            "run_pipeline requires a referability_cache generated by scripts/run_vlm_referability.py"
        )

    meta_dir      = output_dir / "scene_metadata"
    questions_dir = output_dir / "questions"
    frame_debug_dir = output_dir / "frame_debug"
    meta_dir.mkdir(parents=True, exist_ok=True)
    questions_dir.mkdir(parents=True, exist_ok=True)
    if write_frame_debug:
        frame_debug_dir.mkdir(parents=True, exist_ok=True)

    # Discover scene directories (ScanNet scenes have a pose/ subdir)
    discovered_scene_dirs = sorted(
        p for p in data_root.iterdir()
        if p.is_dir() and (p / "pose").exists()
    )
    cached_scene_ids = _get_referability_scene_ids(referability_cache)
    scene_dirs = [p for p in discovered_scene_dirs if p.name in cached_scene_ids]
    logger.info(
        "Loaded %d cached scenes from referability cache; ignoring --max_scenes/--max_frames",
        len(scene_dirs),
    )

    all_questions: list[dict] = []
    scene_debug_records: dict[str, dict[str, object]] = {}
    processed = 0
    total_scenes = len(scene_dirs)

    for scene_dir in scene_dirs:
        scene_id = scene_dir.name
        logger.info(
            "=== Processing scene %s (%d/%d) ===",
            scene_id, processed + 1, total_scenes,
        )

        # ---- Stage 1: Parse ----
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

        # ---- Stage 2: Attachment graph ----
        enrich_scene_with_attachment(scene)
        attachment_graph = get_scene_attachment_graph(scene, scene_id=scene_id)
        attached_by = get_scene_attached_by(scene, scene_id=scene_id)
        support_chain_graph = get_scene_support_chain_graph(scene, scene_id=scene_id)
        support_chain_by = get_scene_support_chain_by(scene, scene_id=scene_id)
        scene_attachment_rows = _build_scene_attachment_rows(scene)
        objects_by_id = {int(obj["id"]): obj for obj in scene["objects"]}

        if not has_nontrivial_attachment(attachment_graph):
            logger.info("Scene %s has no support relations — skipping", scene_id)
            continue

        with open(meta_dir / f"{scene_id}.json", "w", encoding="utf-8") as f:
            json.dump(scene, f, indent=2, ensure_ascii=False)

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
            if referability_entry is not None:
                label_statuses = _normalize_label_statuses(referability_entry.get("label_statuses"))
                label_counts = _normalize_label_counts(referability_entry.get("label_counts"))
                referable_ids = [
                    int(obj_id) for obj_id in referability_entry.get("referable_object_ids", [])
                    if int(obj_id) in visible_id_set
                ]
                if not referable_ids and not _has_l1_visibility_candidates(label_counts):
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
                label_statuses=label_statuses,
                label_counts=label_counts,
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

    logger.info(
        "Pipeline complete! %d questions saved to %s",
        len(final_questions), benchmark_path,
    )
    return final_questions


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
    )


if __name__ == "__main__":
    main()
