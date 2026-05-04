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

When attachment pair salvage review is enabled, the script also writes:
  - salvage/<prefix>_salvage_review.json and .html as batch-level summaries;
  - <cache-stem>_<scene_id>_edited.html as the per-scene human-edited files
    that scripts/run_pipeline.py consumes.

The batch salvage_review.html is not a pipeline input. For backward
compatibility, scripts/run_pipeline.py still falls back to a neighboring
legacy edited.html only when no per-scene edited HTML files exist.
"""

from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
import html
from html.parser import HTMLParser
import inspect
import json
import logging
import os
import re
import sys
import threading
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.frame_selector import (
    build_selector_visibility_audit_from_meta,
    compute_referability_object_visibility,
    select_frames,
)
from src.alias_groups import ALIAS_CONFIG_VERSION
from src.referability_checks import (
    normalize_label_to_object_ids as _shared_normalize_label_to_object_ids,
    normalize_object_ids as _shared_normalize_object_ids,
)
from src.scene_parser import InstanceMeshData, _load_scene_geometry, load_instance_mesh_data
from src.image_quality import BrisqueScorer, compute_brisque_score
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
LABEL_BATCH_SIZE = 4
OBJECT_REVIEW_BATCH_SIZE = 4
REFERABILITY_CACHE_VERSION = "20.0"
REFERABILITY_BACKEND = "crop_vlm_with_mesh_ray"
ATTACHMENT_REVIEW_VERSION = "1.0"
ATTACHMENT_REVIEW_NAME = "attachment_candidate_review"
ATTACHMENT_REVIEW_STAGE = "post_attachment_enrichment"
ATTACHMENT_PAIR_SALVAGE_REVIEW_VERSION = "1.0"
ATTACHMENT_PAIR_SALVAGE_REVIEW_NAME = "attachment_pair_salvage_review"
ATTACHMENT_PAIR_SALVAGE_REVIEW_STAGE = "post_attachment_referability"
FRAME_CACHE_SIDECAR_DIR_NAME = ".run_vlm_referability_frame_cache"
SCANNET_METADATA_SPLIT_FILES: dict[str, Path] = {
    "train": Path("/home/lihongxing/datasets/ScanNet/data/metadata/scannetv2_train.txt"),
    "val": Path("/home/lihongxing/datasets/ScanNet/data/metadata/scannetv2_val.txt"),
}
SCENE_STATUS_VERSION = 1
DEFAULT_BATCH_OUTPUT_PREFIX = "flash"

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
FRAME_CLARITY_BATCH_SIZE = 6
FRAME_CLARITY_MAX_TOKENS_PER_IMAGE = 128
FRAME_CLARITY_BATCH_MAX_TOKENS = 1024
DEFAULT_ATTACHMENT_CLARITY_MIN_SCORE = 70
VISIBLE_OBJECT_GROUP_MAX_VISIBLE_SYMMETRIC_DIFF = 3
ATTACHMENT_GROUP_MAX_POSE_ANGLE_DEG = 20.0
DEFAULT_NON_ATTACHMENT_CLARITY_MIN_SCORE = 70
DEFAULT_NON_ATTACHMENT_REFERABILITY_SHORTLIST = 3
NON_ATTACHMENT_GROUP_EARLY_STOP_REFERABLE_COUNT = 2
NON_ATTACHMENT_GROUP_MIN_REFERABLE_OBJECT_COUNT = 2
FRAME_USABLE_BONUS = 100000
FRAME_SELECTION_FALLBACK_RANK = 1_000_000
FRAME_BRISQUE_MAX_SIDE = 0

ATTACHMENT_PAIR_PROGRAM_DECISION_KEPT = "kept"
ATTACHMENT_PAIR_PROGRAM_DECISION_AUTO_DROP_HARD_FAIL = "auto_drop_hard_fail"
ATTACHMENT_PAIR_PROGRAM_DECISION_NEEDS_VLM_SALVAGE_REVIEW = "needs_vlm_salvage_review"
ATTACHMENT_PAIR_PROGRAM_DECISION_UNCERTAIN = "uncertain"

ATTACHMENT_PAIR_PROGRAM_STATUS_KEPT = "kept"
ATTACHMENT_PAIR_PROGRAM_STATUS_HARD_FAIL = "hard_fail"
ATTACHMENT_PAIR_PROGRAM_STATUS_SALVAGE_REVIEW = "salvage_review"
ATTACHMENT_PAIR_PROGRAM_STATUS_UNCERTAIN = "uncertain"

ATTACHMENT_PAIR_RENAME_ADVICE_STATUS_OK = "ok"
ATTACHMENT_PAIR_RENAME_ADVICE_STATUS_UNAVAILABLE = "unavailable"

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

OUT_OF_FRAME_REVIEW_STATUS_NOT_VISIBLE = "not_visible"
OUT_OF_FRAME_REVIEW_STATUS_REJECT = "reject"
OUT_OF_FRAME_REVIEW_STATUS_UNSURE = "unsure"

_DINOX_CLIENT_CACHE: Any | None = None
_VLM_CALL_FAILURE_COUNT = 0
_VLM_CALL_FAILURE_COUNT_LOCK = threading.Lock()
_VLM_REQUEST_SEMAPHORE: threading.BoundedSemaphore | None = None
_VLM_THREAD_LOCAL_CLIENTS = threading.local()


@dataclass
class SceneWorkerResult:
    scene_index: int
    scene_id: str
    split: str
    pipeline_outcome: str
    scene_skip_reason: str | None
    scene_cache: dict[str, Any] | None
    scene_grouping_summary: dict[str, Any] | None
    attachment_review_record: dict[str, Any] | None
    attachment_pair_salvage_review_record: dict[str, Any] | None
    frame_sidecar_cache: dict[str, dict[str, Any]] | None = None


class MeshRayRequiredError(RuntimeError):
    """Raised when referability candidate refinement cannot complete with mesh-ray."""


class _ThreadLocalOpenAIClientFactory:
    def __init__(self, openai_cls: Callable[..., Any], *, api_key: str, base_url: str) -> None:
        self._openai_cls = openai_cls
        self._api_key = str(api_key)
        self._base_url = str(base_url)

    def get_client(self) -> Any:
        client_cache = getattr(_VLM_THREAD_LOCAL_CLIENTS, "clients", None)
        if not isinstance(client_cache, dict):
            client_cache = {}
            _VLM_THREAD_LOCAL_CLIENTS.clients = client_cache
        cache_key = (id(self._openai_cls), self._api_key, self._base_url)
        client = client_cache.get(cache_key)
        if client is None:
            client = self._openai_cls(api_key=self._api_key, base_url=self._base_url)
            client_cache[cache_key] = client
        return client


def _reset_vlm_call_failure_count() -> None:
    global _VLM_CALL_FAILURE_COUNT
    with _VLM_CALL_FAILURE_COUNT_LOCK:
        _VLM_CALL_FAILURE_COUNT = 0


def _record_vlm_call_failure() -> None:
    global _VLM_CALL_FAILURE_COUNT
    with _VLM_CALL_FAILURE_COUNT_LOCK:
        _VLM_CALL_FAILURE_COUNT += 1


def _get_vlm_call_failure_count() -> int:
    with _VLM_CALL_FAILURE_COUNT_LOCK:
        return int(_VLM_CALL_FAILURE_COUNT)


def _configure_vlm_request_concurrency(max_workers: int) -> None:
    global _VLM_REQUEST_SEMAPHORE
    _VLM_REQUEST_SEMAPHORE = threading.BoundedSemaphore(max(1, int(max_workers)))


def _resolve_vlm_client(client: Any) -> Any:
    getter = getattr(client, "get_client", None)
    if callable(getter):
        return getter()
    return client


def _write_json_payload(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _timestamp_for_filename(now: datetime | None = None) -> str:
    current = now or _utc_now()
    return current.strftime("%Y%m%d_%H%M%S")


def _timestamp_for_status(now: datetime | None = None) -> str:
    current = now or _utc_now()
    return current.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _resolve_batch_output_dir(output_arg: Path) -> Path:
    if output_arg.suffix.lower() == ".json":
        return output_arg.parent
    return output_arg


def _resolve_batch_output_prefix(output_arg: Path) -> str:
    if output_arg.suffix.lower() == ".json":
        prefix = output_arg.stem.strip()
        return prefix or DEFAULT_BATCH_OUTPUT_PREFIX
    return DEFAULT_BATCH_OUTPUT_PREFIX


def _scene_status_output_path(output_arg: Path) -> Path:
    return _resolve_batch_output_dir(output_arg) / "scene_status.json"


def _build_batch_output_path(output_arg: Path, *, now: datetime | None = None) -> Path:
    output_dir = _resolve_batch_output_dir(output_arg)
    prefix = _resolve_batch_output_prefix(output_arg)
    timestamp = _timestamp_for_filename(now)
    candidate = output_dir / f"{prefix}_{timestamp}.json"
    if not candidate.exists():
        return candidate
    suffix = 2
    while True:
        deduped_candidate = output_dir / f"{prefix}_{timestamp}_{suffix}.json"
        if not deduped_candidate.exists():
            return deduped_candidate
        suffix += 1


def _build_empty_scene_status_doc(split: str) -> dict[str, Any]:
    return {
        "version": SCENE_STATUS_VERSION,
        "split": str(split),
        "completed_scenes": {},
    }


def _load_scene_status_doc(
    path: Path,
    *,
    split: str,
) -> dict[str, Any]:
    if not path.exists():
        return _build_empty_scene_status_doc(split)

    with open(path, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    if not isinstance(loaded, dict):
        raise RuntimeError(f"Invalid scene status document at {path}: expected JSON object")

    version = int(loaded.get("version", 0) or 0)
    if version != SCENE_STATUS_VERSION:
        raise RuntimeError(
            f"Unsupported scene status version {version or '<missing>'} at {path}; "
            f"expected {SCENE_STATUS_VERSION}."
        )

    loaded_split = str(loaded.get("split", "")).strip()
    if loaded_split != str(split):
        raise RuntimeError(
            f"scene_status split mismatch at {path}: found {loaded_split or '<missing>'}, expected {split}. "
            "Use --reset N to remove existing completed scenes before running a different split."
        )

    completed_scenes = loaded.get("completed_scenes")
    if not isinstance(completed_scenes, dict):
        raise RuntimeError(f"Invalid scene status document at {path}: completed_scenes must be an object")

    return {
        "version": SCENE_STATUS_VERSION,
        "split": loaded_split,
        "completed_scenes": dict(completed_scenes),
    }


def _parse_scene_status_updated_at(value: Any) -> datetime | None:
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


def _parse_closed_scene_range(raw_value: str, *, arg_name: str) -> tuple[int, int]:
    candidate = str(raw_value or "").strip()
    match = re.fullmatch(r"(\d+)-(\d+)", candidate)
    if match is None:
        raise ValueError(f"{arg_name} must use START-END with 0-based inclusive indexes")
    start = int(match.group(1))
    end = int(match.group(2))
    if start < 0 or end < 0:
        raise ValueError(f"{arg_name} does not allow negative indexes")
    if start > end:
        raise ValueError(f"{arg_name} must satisfy START <= END")
    return start, end


def _select_scene_entries_by_closed_range(
    scene_entries: list[tuple[str, Path]],
    *,
    start: int,
    end: int,
) -> list[tuple[str, Path]]:
    if start < 0 or end < 0:
        raise ValueError("scene range indexes must be >= 0")
    if start > end:
        raise ValueError("scene range start must be <= end")
    return list(scene_entries[start : end + 1])


def _reset_completed_scene_status(
    scene_status_doc: dict[str, Any],
    *,
    count: int,
) -> list[str]:
    if int(count) <= 0:
        raise ValueError("count must be >= 1")
    completed_scenes = scene_status_doc.get("completed_scenes")
    if not isinstance(completed_scenes, dict):
        raise RuntimeError("scene_status_doc.completed_scenes must be an object")

    ranked_items = sorted(completed_scenes.items(), key=lambda item: str(item[0]))
    ranked_items.sort(
        key=lambda item: _parse_scene_status_updated_at(
            item[1].get("updated_at") if isinstance(item[1], dict) else None
        ) or datetime.min.replace(tzinfo=timezone.utc),
        reverse=True,
    )
    removed_scene_ids = [str(scene_id) for scene_id, _ in ranked_items[: int(count)]]
    for scene_id in removed_scene_ids:
        completed_scenes.pop(scene_id, None)
    return removed_scene_ids


def _reset_completed_scene_status_for_scene_ids(
    scene_status_doc: dict[str, Any],
    *,
    scene_ids: list[str],
) -> list[str]:
    completed_scenes = scene_status_doc.get("completed_scenes")
    if not isinstance(completed_scenes, dict):
        raise RuntimeError("scene_status_doc.completed_scenes must be an object")

    removed_scene_ids: list[str] = []
    seen_scene_ids: set[str] = set()
    for raw_scene_id in scene_ids:
        scene_id = str(raw_scene_id)
        if scene_id in seen_scene_ids:
            continue
        seen_scene_ids.add(scene_id)
        if scene_id in completed_scenes:
            completed_scenes.pop(scene_id, None)
            removed_scene_ids.append(scene_id)
    return removed_scene_ids


def _batch_cache_contains_scene(cache_doc: dict[str, Any], scene_id: str) -> bool:
    for field_name in ("frames", "scene_grouping", "scene_status"):
        field_value = cache_doc.get(field_name)
        if isinstance(field_value, dict) and scene_id in field_value:
            return True
    return False


def _validate_scene_status_doc(
    scene_status_doc: dict[str, Any],
    *,
    scene_status_path: Path,
) -> None:
    completed_scenes = scene_status_doc.get("completed_scenes")
    if not isinstance(completed_scenes, dict):
        raise RuntimeError(f"Invalid scene status document at {scene_status_path}: completed_scenes must be an object")

    loaded_batch_docs: dict[str, dict[str, Any]] = {}
    for scene_id, record in completed_scenes.items():
        if not isinstance(record, dict):
            raise RuntimeError(
                f"Invalid scene status record for {scene_id} at {scene_status_path}: expected object"
            )
        batch_file = str(record.get("batch_file", "")).strip()
        if not batch_file:
            raise RuntimeError(
                f"Invalid scene status record for {scene_id} at {scene_status_path}: missing batch_file"
            )
        batch_path = scene_status_path.parent / batch_file
        if not batch_path.exists():
            raise RuntimeError(
                f"Scene status says {scene_id} completed in {batch_file}, but that batch file does not exist: {batch_path}"
            )
        batch_doc = loaded_batch_docs.get(batch_file)
        if batch_doc is None:
            with open(batch_path, "r", encoding="utf-8") as f:
                batch_doc = json.load(f)
            if not isinstance(batch_doc, dict):
                raise RuntimeError(f"Invalid referability batch cache at {batch_path}: expected JSON object")
            loaded_batch_docs[batch_file] = batch_doc
        if not _batch_cache_contains_scene(batch_doc, str(scene_id)):
            raise RuntimeError(
                f"Scene status says {scene_id} completed in {batch_file}, but that batch file does not contain the scene."
            )


def _mark_scene_completed(
    scene_status_doc: dict[str, Any],
    *,
    scene_id: str,
    batch_file: str,
    updated_at: str,
) -> None:
    completed_scenes = scene_status_doc.setdefault("completed_scenes", {})
    if not isinstance(completed_scenes, dict):
        raise RuntimeError("scene_status_doc.completed_scenes must be an object")
    completed_scenes[str(scene_id)] = {
        "status": "completed",
        "batch_file": str(batch_file),
        "updated_at": str(updated_at),
    }


def _default_review_output_prefix(output_path: Path) -> str:
    match = re.search(r"(?:^|_)(flash[^_]*)$", output_path.stem)
    if match is None:
        return output_path.stem
    return match.group(1)


def _salvage_artifact_dir(output_path: Path) -> Path:
    return output_path.parent / "salvage"


def _candidate_artifact_dir(output_path: Path) -> Path:
    return output_path.parent / "candidate"


def _attachment_review_output_path(output_path: Path) -> Path:
    prefix = _default_review_output_prefix(output_path)
    return _candidate_artifact_dir(output_path) / f"{prefix}_attachment_candidate_review.json"


def _attachment_pair_salvage_review_output_path(output_path: Path) -> Path:
    prefix = _default_review_output_prefix(output_path)
    return _salvage_artifact_dir(output_path) / f"{prefix}_salvage_review.json"


def _attachment_pair_salvage_review_html_output_path(output_path: Path) -> Path:
    prefix = _default_review_output_prefix(output_path)
    return _salvage_artifact_dir(output_path) / f"{prefix}_salvage_review.html"


def _frame_cache_artifact_dir(output_path: Path) -> Path:
    return output_path.parent / FRAME_CACHE_SIDECAR_DIR_NAME


def _frame_cache_sidecar_path(output_path: Path, scene_id: str) -> Path:
    return _frame_cache_artifact_dir(output_path) / f"{str(scene_id).strip()}.json"


def _normalize_frame_sidecar_record(record: Any) -> dict[str, Any] | None:
    if not isinstance(record, dict):
        return None
    frame_info = record.get("frame_info")
    if not isinstance(frame_info, dict):
        return None
    try:
        frame_selection_score = int(record.get("frame_selection_score"))
    except (TypeError, ValueError):
        return None
    normalized_entry = None
    if record.get("referability_entry") is not None:
        referability_entry = record.get("referability_entry")
        if not isinstance(referability_entry, dict):
            return None
        if _frame_entry_has_debug_fields(referability_entry):
            normalized_entry = dict(referability_entry)
        else:
            normalized_entry = _repair_final_referability_fields(referability_entry)
            if not _frame_entry_has_debug_fields(normalized_entry):
                return None
    return {
        "frame_info": _normalize_frame_review(frame_info),
        "frame_selection_score": frame_selection_score,
        "referability_entry": normalized_entry,
    }


def _load_frame_sidecar_scene_cache(
    *,
    output_path: Path,
    scene_id: str,
    model_name: str,
    referability_backend: str,
) -> dict[str, dict[str, Any]]:
    sidecar_path = _frame_cache_sidecar_path(output_path, scene_id)
    if not sidecar_path.exists():
        return {}
    try:
        sidecar_doc = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Ignoring unreadable frame sidecar %s: %s", sidecar_path, exc)
        return {}
    if not isinstance(sidecar_doc, dict):
        logger.warning("Ignoring malformed frame sidecar %s: expected object", sidecar_path)
        return {}
    expected_meta = {
        "scene_id": str(scene_id),
        "version": REFERABILITY_CACHE_VERSION,
        "alias_config_version": ALIAS_CONFIG_VERSION,
        "referability_backend": str(referability_backend),
        "vlm_model": str(model_name),
    }
    for key, expected_value in expected_meta.items():
        if sidecar_doc.get(key) != expected_value:
            return {}
    raw_frames = sidecar_doc.get("frames")
    if not isinstance(raw_frames, dict):
        logger.warning("Ignoring malformed frame sidecar %s: missing frames mapping", sidecar_path)
        return {}

    normalized_frames: dict[str, dict[str, Any]] = {}
    for image_name, record in raw_frames.items():
        normalized_record = _normalize_frame_sidecar_record(record)
        if normalized_record is None:
            logger.warning(
                "Ignoring malformed frame sidecar %s: invalid record for %s",
                sidecar_path,
                image_name,
            )
            return {}
        normalized_frames[str(image_name)] = normalized_record
    return normalized_frames


def _build_frame_sidecar_scene_doc(
    *,
    scene_id: str,
    model_name: str,
    referability_backend: str,
    frame_records: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    return {
        "scene_id": str(scene_id),
        "version": REFERABILITY_CACHE_VERSION,
        "alias_config_version": ALIAS_CONFIG_VERSION,
        "referability_backend": str(referability_backend),
        "vlm_model": str(model_name),
        "frames": {
            str(image_name): {
                "frame_info": dict(record["frame_info"]),
                "frame_selection_score": int(record["frame_selection_score"]),
                "referability_entry": (
                    dict(record["referability_entry"])
                    if isinstance(record.get("referability_entry"), dict)
                    else None
                ),
            }
            for image_name, record in sorted(frame_records.items())
            if isinstance(record, dict)
        },
    }


def _write_frame_sidecar_scene_cache(
    *,
    output_path: Path,
    scene_id: str,
    model_name: str,
    referability_backend: str,
    frame_records: dict[str, dict[str, Any]],
) -> None:
    sidecar_path = _frame_cache_sidecar_path(output_path, scene_id)
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json_payload(
        sidecar_path,
        _build_frame_sidecar_scene_doc(
            scene_id=scene_id,
            model_name=model_name,
            referability_backend=referability_backend,
            frame_records=frame_records,
        ),
    )


def _edited_attachment_pair_salvage_html_output_path(output_path: Path, scene_id: str) -> Path:
    return output_path.parent / f"{output_path.stem}_{str(scene_id).strip()}_edited.html"


def _edited_attachment_pair_salvage_html_output_glob(output_path: Path) -> str:
    return str(output_path.parent / f"{output_path.stem}_*_edited.html")


def _scene_object_label(obj: dict[str, Any]) -> str:
    for key in ("label", "canonical_label", "raw_label"):
        label = str(obj.get(key, "")).strip()
        if label:
            return label
    return "unknown"


def _attachment_edge_key(edge: dict[str, Any]) -> tuple[int, int, str]:
    return (
        int(edge.get("parent_id", 0) or 0),
        int(edge.get("child_id", 0) or 0),
        str(edge.get("type", "")).strip(),
    )


def _build_attachment_selector_signal_payload(
    *,
    well_cropped_pair_count: object,
    viewpoint_exempt: object,
) -> dict[str, Any]:
    pair_count = int(well_cropped_pair_count or 0)
    return {
        "well_cropped_pair_count": pair_count,
        "viewpoint_exempt": bool(viewpoint_exempt),
    }


def _build_attachment_final_referability_payload(
    *,
    attachment_referable_object_ids: object,
    attachment_pairs: object,
) -> dict[str, Any]:
    object_ids = sorted(
        {
            int(obj_id)
            for obj_id in (attachment_referable_object_ids or [])
        }
    )
    normalized_pairs = [
        [int(pair[0]), int(pair[1])]
        for pair in (attachment_pairs or [])
        if isinstance(pair, (list, tuple)) and len(pair) == 2
    ]
    return {
        "object_ids": object_ids,
        "pairs": normalized_pairs,
        "pair_count": len(normalized_pairs),
    }


def _build_attachment_final_frame_selection_payload(
    *,
    final_selection_rank: object,
) -> dict[str, Any]:
    rank = int(final_selection_rank if final_selection_rank is not None else FRAME_SELECTION_FALLBACK_RANK)
    selected_for_final_cache = rank < FRAME_SELECTION_FALLBACK_RANK
    return {
        "selected_for_final_cache": selected_for_final_cache,
        "selection_rank": rank if selected_for_final_cache else None,
    }


def _normalize_attachment_human_review_cards(value: object) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    if not isinstance(value, list):
        return normalized

    seen: set[tuple[int, int, str, str]] = set()
    for item in value:
        if not isinstance(item, dict):
            continue
        try:
            parent_id = int(item.get("parent_id"))
            child_id = int(item.get("child_id"))
        except (TypeError, ValueError):
            continue
        parent_surface_text = str(item.get("parent_surface_text", "")).strip()
        child_surface_text = str(item.get("child_surface_text", "")).strip()
        if not parent_surface_text or not child_surface_text:
            continue
        dedupe_key = (parent_id, child_id, parent_surface_text, child_surface_text)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        normalized.append(
            {
                "pair_id": str(item.get("pair_id", f"{parent_id}->{child_id}")).strip() or f"{parent_id}->{child_id}",
                "parent_id": parent_id,
                "parent_label": str(item.get("parent_label", "")).strip(),
                "parent_surface_text": parent_surface_text,
                "child_id": child_id,
                "child_label": str(item.get("child_label", "")).strip(),
                "child_surface_text": child_surface_text,
                "source": str(item.get("source", "")).strip() or "human_salvage_html",
            }
        )
    normalized.sort(
        key=lambda item: (
            int(item["parent_id"]),
            int(item["child_id"]),
            str(item["parent_surface_text"]).lower(),
            str(item["child_surface_text"]).lower(),
        )
    )
    return normalized


def _attachment_human_review_object_ids(cards: object) -> list[int]:
    object_ids: set[int] = set()
    for item in _normalize_attachment_human_review_cards(cards):
        object_ids.add(int(item["parent_id"]))
        object_ids.add(int(item["child_id"]))
    return sorted(object_ids)


def _attachment_human_review_surface_text_by_object_id(cards: object) -> dict[int, str]:
    surface_text_by_obj_id: dict[int, str] = {}
    for item in _normalize_attachment_human_review_cards(cards):
        parent_id = int(item["parent_id"])
        child_id = int(item["child_id"])
        parent_surface_text = str(item["parent_surface_text"]).strip()
        child_surface_text = str(item["child_surface_text"]).strip()
        if parent_id not in surface_text_by_obj_id and parent_surface_text:
            surface_text_by_obj_id[parent_id] = parent_surface_text
        if child_id not in surface_text_by_obj_id and child_surface_text:
            surface_text_by_obj_id[child_id] = child_surface_text
    return dict(sorted(surface_text_by_obj_id.items()))


def _apply_attachment_layer_payloads(
    entry: dict[str, Any],
    *,
    attachment_pairs: list[list[int]] | None = None,
    selector_pair_count: object | None = None,
    selector_viewpoint_exempt: object | None = None,
    final_selection_rank: object | None = None,
) -> dict[str, Any]:
    updated = dict(entry)
    pair_count = selector_pair_count
    if pair_count is None:
        pair_count = updated.get("attachment_pair_ge_50_count", 0) or 0
    viewpoint_exempt = selector_viewpoint_exempt
    if viewpoint_exempt is None:
        viewpoint_exempt = updated.get("attachment_viewpoint_exempt", False)
    if attachment_pairs is None:
        attachment_pairs = [
            [int(pair[0]), int(pair[1])]
            for pair in (updated.get("attachment_referable_pairs") or [])
            if isinstance(pair, (list, tuple)) and len(pair) == 2
        ]
    if final_selection_rank is None:
        final_selection_rank = updated.get("final_selection_rank", FRAME_SELECTION_FALLBACK_RANK)

    updated["attachment_selector_signal"] = _build_attachment_selector_signal_payload(
        well_cropped_pair_count=pair_count,
        viewpoint_exempt=viewpoint_exempt,
    )
    updated["attachment_final_referability"] = _build_attachment_final_referability_payload(
        attachment_referable_object_ids=updated.get("attachment_referable_object_ids"),
        attachment_pairs=attachment_pairs,
    )
    updated["attachment_final_frame_selection"] = _build_attachment_final_frame_selection_payload(
        final_selection_rank=final_selection_rank,
    )
    return updated


def _build_attachment_review_scene_record(
    *,
    scene_id: str,
    objects: list[dict[str, Any]],
    raw_candidates: list[dict[str, Any]],
    final_attachment_edges: list[dict[str, Any]],
    pipeline_outcome: str,
) -> dict[str, Any]:
    object_labels = {
        int(obj["id"]): _scene_object_label(obj)
        for obj in objects
        if "id" in obj
    }
    final_edge_keys = {
        _attachment_edge_key(edge)
        for edge in final_attachment_edges
    }

    candidate_rows: list[dict[str, Any]] = []
    candidate_rank_for_child: dict[int, int] = defaultdict(int)
    terminal_output_lines: list[str] = []

    summary_line = (
        f"[attachment-review] scene={scene_id} outcome={pipeline_outcome} "
        f"objects={len(objects)} raw_candidates={len(raw_candidates)} "
        f"final_attachment_edges={len(final_attachment_edges)}"
    )
    terminal_output_lines.append(summary_line)

    for edge in raw_candidates:
        parent_id = int(edge.get("parent_id", 0) or 0)
        child_id = int(edge.get("child_id", 0) or 0)
        candidate_rank_for_child[child_id] += 1
        relation_type = str(edge.get("type", "")).strip()
        selected = _attachment_edge_key(edge) in final_edge_keys
        row = {
            "parent_id": parent_id,
            "parent_label": object_labels.get(parent_id, "unknown"),
            "child_id": child_id,
            "child_label": object_labels.get(child_id, "unknown"),
            "relation_type": relation_type,
            "confidence": float(edge.get("confidence", 0.0) or 0.0),
            "candidate_rank_for_child": int(candidate_rank_for_child[child_id]),
            "selected_for_attachment_graph": bool(selected),
            "selected_for_final_attachment_graph": bool(selected),
            "evidence": edge.get("evidence") or {},
        }
        candidate_rows.append(row)
        terminal_output_lines.append(
            f"[attachment-review] scene={scene_id} parent={parent_id}:{row['parent_label']} "
            f"child={child_id}:{row['child_label']} rank={row['candidate_rank_for_child']} "
            f"selected={int(selected)} relation={relation_type} confidence={row['confidence']:.4f}"
        )

    return {
        "scene_id": scene_id,
        "object_count": len(objects),
        "pipeline_outcome": pipeline_outcome,
        "raw_candidate_edge_count": len(raw_candidates),
        "raw_attachment_candidate_edge_count": len(raw_candidates),
        "final_attachment_edge_count": len(final_attachment_edges),
        "final_attachment_graph_edge_count": len(final_attachment_edges),
        "attachment_graph_layers": {
            "raw_candidates": {
                "edge_count": len(raw_candidates),
            },
            "final_attachment_graph": {
                "edge_count": len(final_attachment_edges),
            },
        },
        "terminal_output_lines": terminal_output_lines,
        "candidate_rows": candidate_rows,
    }


def _build_attachment_review_document(
    *,
    referability_cache_output: Path,
    scenes: list[dict[str, Any]],
    terminal_output_lines: list[str],
) -> dict[str, Any]:
    raw_candidate_edge_count = sum(
        int(scene.get("raw_candidate_edge_count", 0) or 0)
        for scene in scenes
    )
    final_attachment_edge_count = sum(
        int(scene.get("final_attachment_edge_count", 0) or 0)
        for scene in scenes
    )
    return {
        "name": ATTACHMENT_REVIEW_NAME,
        "version": ATTACHMENT_REVIEW_VERSION,
        "generated_by": "scripts/run_vlm_referability.py",
        "review_stage": ATTACHMENT_REVIEW_STAGE,
        "referability_cache_output": str(referability_cache_output),
        "scene_count": len(scenes),
        "raw_candidate_edge_count": raw_candidate_edge_count,
        "raw_attachment_candidate_edge_count": raw_candidate_edge_count,
        "final_attachment_edge_count": final_attachment_edge_count,
        "final_attachment_graph_edge_count": final_attachment_edge_count,
        "attachment_graph_layers": {
            "raw_candidates": {
                "scene_count": len(scenes),
                "edge_count": raw_candidate_edge_count,
            },
            "final_attachment_graph": {
                "scene_count": len(scenes),
                "edge_count": final_attachment_edge_count,
            },
        },
        "terminal_output_lines": list(terminal_output_lines),
        "scenes": list(scenes),
    }


def _attachment_pair_id(parent_id: int, child_id: int) -> str:
    return f"{int(parent_id)}->{int(child_id)}"


def _attachment_pair_program_status(program_decision: object) -> str:
    decision = str(program_decision or "").strip().lower()
    if decision == ATTACHMENT_PAIR_PROGRAM_DECISION_KEPT:
        return ATTACHMENT_PAIR_PROGRAM_STATUS_KEPT
    if decision == ATTACHMENT_PAIR_PROGRAM_DECISION_AUTO_DROP_HARD_FAIL:
        return ATTACHMENT_PAIR_PROGRAM_STATUS_HARD_FAIL
    if decision == ATTACHMENT_PAIR_PROGRAM_DECISION_NEEDS_VLM_SALVAGE_REVIEW:
        return ATTACHMENT_PAIR_PROGRAM_STATUS_SALVAGE_REVIEW
    return ATTACHMENT_PAIR_PROGRAM_STATUS_UNCERTAIN


def _attachment_edge_relation_type_map(
    final_attachment_edges: list[dict[str, Any]],
) -> dict[tuple[int, int], list[str]]:
    relation_types: dict[tuple[int, int], set[str]] = defaultdict(set)
    for edge in final_attachment_edges:
        try:
            parent_id = int(edge.get("parent_id", 0) or 0)
            child_id = int(edge.get("child_id", 0) or 0)
        except (TypeError, ValueError):
            continue
        relation_type = str(edge.get("type", "")).strip()
        if parent_id <= 0 or child_id <= 0 or not relation_type:
            continue
        relation_types[(parent_id, child_id)].add(relation_type)
    return {
        pair_key: sorted(types)
        for pair_key, types in sorted(relation_types.items())
    }


def _attachment_pairs_for_visible_group(
    attachment_graph: dict[int, list[int]] | None,
    visible_object_ids: list[int] | tuple[int, ...],
) -> list[tuple[int, int]]:
    visible_set = {int(obj_id) for obj_id in visible_object_ids}
    pairs: list[tuple[int, int]] = []
    for parent_id, child_ids in (attachment_graph or {}).items():
        try:
            parent_id_int = int(parent_id)
        except (TypeError, ValueError):
            continue
        if parent_id_int not in visible_set:
            continue
        for child_id in child_ids or []:
            try:
                child_id_int = int(child_id)
            except (TypeError, ValueError):
                continue
            if child_id_int not in visible_set:
                continue
            pairs.append((parent_id_int, child_id_int))
    return sorted(set(pairs))


def _lookup_object_payload(container: object, obj_id: int) -> dict[str, Any] | None:
    if isinstance(container, dict):
        payload = container.get(str(obj_id))
        if payload is None:
            payload = container.get(int(obj_id))
        return payload if isinstance(payload, dict) else None
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


def _object_payload_object_ids(container: object) -> list[int]:
    object_ids: set[int] = set()
    if isinstance(container, dict):
        for key, payload in container.items():
            if not isinstance(payload, dict):
                continue
            try:
                object_ids.add(int(key))
            except (TypeError, ValueError):
                payload_obj_id = payload.get("obj_id")
                try:
                    object_ids.add(int(payload_obj_id))
                except (TypeError, ValueError):
                    continue
        return sorted(object_ids)
    if isinstance(container, list):
        for item in container:
            if not isinstance(item, dict):
                continue
            try:
                object_ids.add(int(item.get("obj_id")))
            except (TypeError, ValueError):
                continue
    return sorted(object_ids)


def _geometry_signature_object_ids(
    container: object,
    *,
    bbox_in_frame_ratio_min: float,
    projected_area_px_min: float,
) -> tuple[int, ...]:
    candidate_object_ids = _object_payload_object_ids(container)
    signature: list[int] = []
    for obj_id in candidate_object_ids:
        payload = _lookup_object_payload(container, int(obj_id))
        if not isinstance(payload, dict):
            continue
        bbox_in_frame_ratio = _safe_float(payload.get("bbox_in_frame_ratio"), default=-1.0)
        projected_area_px = _safe_float(payload.get("projected_area_px"), default=-1.0)
        if bbox_in_frame_ratio < float(bbox_in_frame_ratio_min):
            continue
        if projected_area_px < float(projected_area_px_min):
            continue
        signature.append(int(obj_id))
    return tuple(sorted(set(signature)))


def _failed_referability_object_id_signature(
    entry: object,
    *,
    bbox_in_frame_ratio_min: float,
    projected_area_px_min: float,
) -> tuple[int, ...]:
    if not isinstance(entry, dict):
        return ()

    object_reviews = entry.get("object_reviews")
    visibility_audit_by_object_id = entry.get("visibility_audit_by_object_id")
    return tuple(
        sorted(
            set(
                _geometry_signature_object_ids(
                    object_reviews,
                    bbox_in_frame_ratio_min=bbox_in_frame_ratio_min,
                    projected_area_px_min=projected_area_px_min,
                )
            )
            | set(
                _geometry_signature_object_ids(
                    visibility_audit_by_object_id,
                    bbox_in_frame_ratio_min=bbox_in_frame_ratio_min,
                    projected_area_px_min=projected_area_px_min,
                )
            )
        )
    )


def _normalize_attachment_pair_string_list(values: object) -> list[str]:
    if not isinstance(values, list):
        return []
    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        normalized.append(text)
    return normalized


def _normalize_attachment_pair_rename_advice_status(value: object) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in {
        ATTACHMENT_PAIR_RENAME_ADVICE_STATUS_OK,
        ATTACHMENT_PAIR_RENAME_ADVICE_STATUS_UNAVAILABLE,
    }:
        return normalized
    return ATTACHMENT_PAIR_RENAME_ADVICE_STATUS_UNAVAILABLE


def _normalize_attachment_pair_rename_advice_candidate(raw_candidate: object) -> dict[str, str] | None:
    if not isinstance(raw_candidate, dict):
        return None
    candidate = {
        "parent_surface_text": str(raw_candidate.get("parent_surface_text", "")).strip(),
        "child_surface_text": str(raw_candidate.get("child_surface_text", "")).strip(),
        "relation_hint_text": str(raw_candidate.get("relation_hint_text", "")).strip(),
    }
    if not any(candidate.values()):
        return None
    return candidate


def _normalize_attachment_pair_rename_advice_candidates(values: object) -> list[dict[str, str]]:
    if not isinstance(values, list):
        return []
    normalized: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for raw_candidate in values:
        candidate = _normalize_attachment_pair_rename_advice_candidate(raw_candidate)
        if candidate is None:
            continue
        key = (
            candidate["parent_surface_text"],
            candidate["child_surface_text"],
            candidate["relation_hint_text"],
        )
        if key in seen:
            continue
        seen.add(key)
        normalized.append(candidate)
        if len(normalized) >= 3:
            break
    return normalized


def _default_attachment_pair_rename_advice(*, reason: str) -> dict[str, Any]:
    return {
        "status": ATTACHMENT_PAIR_RENAME_ADVICE_STATUS_UNAVAILABLE,
        "reason": str(reason or "").strip() or "rename_advice_unavailable",
        "candidates": [],
    }


def _image_name_stem(image_name: object) -> str:
    text = str(image_name or "").strip()
    return Path(text).stem if text else ""


def _image_to_data_url(image: np.ndarray) -> str:
    return f"data:image/jpeg;base64,{_image_to_base64(image)}"


def _normalize_bounds(bounds: object) -> list[int] | None:
    if not isinstance(bounds, (list, tuple)) or len(bounds) != 4:
        return None
    try:
        values = [int(value) for value in bounds]
    except (TypeError, ValueError):
        return None
    return values


def _crop_image_from_bounds(image: np.ndarray, bounds: object) -> np.ndarray | None:
    normalized_bounds = _normalize_bounds(bounds)
    if normalized_bounds is None:
        return None
    u_min, u_max, v_min, v_max = normalized_bounds
    u_min = max(0, min(int(image.shape[1]), u_min))
    u_max = max(0, min(int(image.shape[1]), u_max))
    v_min = max(0, min(int(image.shape[0]), v_min))
    v_max = max(0, min(int(image.shape[0]), v_max))
    if u_max <= u_min or v_max <= v_min:
        return None
    crop = image[v_min:v_max, u_min:u_max]
    if crop.size == 0:
        return None
    return crop


def _attachment_pair_object_color(obj_id: int) -> tuple[int, int, int]:
    palette = [
        (236, 100, 75),
        (52, 152, 219),
        (46, 204, 113),
        (241, 196, 15),
        (155, 89, 182),
        (26, 188, 156),
        (230, 126, 34),
        (231, 76, 60),
    ]
    return palette[int(obj_id) % len(palette)]


def _annotate_attachment_pair_salvage_frame(
    image: np.ndarray,
    *,
    entry: dict[str, Any],
    visible_object_ids: list[int],
    objects_by_id: dict[int, dict[str, Any]],
) -> np.ndarray:
    annotated = image.copy()
    for obj_id in visible_object_ids:
        review = _lookup_object_payload(entry.get("object_reviews"), int(obj_id))
        visibility = _lookup_object_payload(entry.get("visibility_audit_by_object_id"), int(obj_id))
        bounds = (
            _normalize_bounds((review or {}).get("roi_bounds_px"))
            or _normalize_bounds((visibility or {}).get("roi_bounds_px"))
            or _normalize_bounds((review or {}).get("crop_bounds_px"))
        )
        if bounds is None:
            continue
        u_min, u_max, v_min, v_max = bounds
        color = _attachment_pair_object_color(int(obj_id))
        cv2.rectangle(annotated, (u_min, v_min), (u_max, v_max), color, 2)
        obj = objects_by_id.get(int(obj_id), {})
        label = str(obj.get("label", "")).strip().lower() or "object"
        text = f"{label}#{int(obj_id)}"
        text_origin = (u_min, max(14, v_min - 6))
        cv2.putText(
            annotated,
            text,
            text_origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            2,
            cv2.LINE_AA,
        )
    return annotated


def _attachment_pair_object_gate(
    *,
    entry: dict[str, Any],
    obj_id: int,
    bbox_hard_fail_min: float,
    projected_area_hard_fail_min: float,
) -> dict[str, Any]:
    if "candidate_visible_object_ids" not in entry:
        return {
            "decision": ATTACHMENT_PAIR_PROGRAM_DECISION_UNCERTAIN,
            "reason_codes": ["missing_candidate_visible_object_ids"],
        }
    candidate_visible_ids = set(_normalize_cached_object_ids(entry.get("candidate_visible_object_ids")))
    if int(obj_id) not in candidate_visible_ids:
        return {
            "decision": ATTACHMENT_PAIR_PROGRAM_DECISION_AUTO_DROP_HARD_FAIL,
            "reason_codes": ["candidate_not_visible"],
        }
    review = _lookup_object_payload(entry.get("object_reviews"), int(obj_id))
    if not isinstance(review, dict):
        return {
            "decision": ATTACHMENT_PAIR_PROGRAM_DECISION_UNCERTAIN,
            "reason_codes": ["missing_object_review"],
        }

    local_outcome = str(review.get("local_outcome", "")).strip().lower()
    if local_outcome in {
        LOCAL_OUTCOME_OUT_OF_FRAME,
        LOCAL_OUTCOME_EXCLUDED,
        "not_visible",
        "empty_or_invalid_crop",
    }:
        return {
            "decision": ATTACHMENT_PAIR_PROGRAM_DECISION_AUTO_DROP_HARD_FAIL,
            "reason_codes": [f"local_outcome_{local_outcome or 'missing'}"],
        }

    object_status = _normalize_object_review_status(review.get("vlm_status"))
    if object_status == OBJECT_STATUS_ABSENT:
        return {
            "decision": ATTACHMENT_PAIR_PROGRAM_DECISION_AUTO_DROP_HARD_FAIL,
            "reason_codes": ["vlm_status_absent"],
        }

    bbox_in_frame_ratio = _safe_float(review.get("bbox_in_frame_ratio"), default=0.0)
    if bbox_in_frame_ratio < float(bbox_hard_fail_min):
        return {
            "decision": ATTACHMENT_PAIR_PROGRAM_DECISION_AUTO_DROP_HARD_FAIL,
            "reason_codes": ["bbox_in_frame_ratio_too_small"],
        }

    projected_area_px = _safe_float(review.get("projected_area_px"), default=0.0)
    if projected_area_px < float(projected_area_hard_fail_min):
        return {
            "decision": ATTACHMENT_PAIR_PROGRAM_DECISION_AUTO_DROP_HARD_FAIL,
            "reason_codes": ["projected_area_too_small"],
        }

    return {
        "decision": "pass",
        "reason_codes": [],
    }


def _attachment_pair_coverage_for_entry(
    *,
    entry: dict[str, Any],
    parent_id: int,
    child_id: int,
    bbox_hard_fail_min: float,
    projected_area_hard_fail_min: float,
) -> dict[str, Any]:
    parent_gate = _attachment_pair_object_gate(
        entry=entry,
        obj_id=int(parent_id),
        bbox_hard_fail_min=bbox_hard_fail_min,
        projected_area_hard_fail_min=projected_area_hard_fail_min,
    )
    child_gate = _attachment_pair_object_gate(
        entry=entry,
        obj_id=int(child_id),
        bbox_hard_fail_min=bbox_hard_fail_min,
        projected_area_hard_fail_min=projected_area_hard_fail_min,
    )

    covered = (
        parent_gate["decision"] == "pass"
        and child_gate["decision"] == "pass"
    )
    uncertain = (
        parent_gate["decision"] == ATTACHMENT_PAIR_PROGRAM_DECISION_UNCERTAIN
        or child_gate["decision"] == ATTACHMENT_PAIR_PROGRAM_DECISION_UNCERTAIN
    )

    reason_codes = [
        f"parent_{reason}"
        for reason in parent_gate.get("reason_codes", [])
    ] + [
        f"child_{reason}"
        for reason in child_gate.get("reason_codes", [])
    ]
    return {
        "covered": bool(covered),
        "uncertain": bool(uncertain),
        "reason_codes": reason_codes,
        "parent_gate": parent_gate,
        "child_gate": child_gate,
    }


def _entry_has_attachment_pair(
    entry: dict[str, Any],
    *,
    parent_id: int,
    child_id: int,
) -> bool:
    for pair in entry.get("attachment_referable_pairs", []):
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            continue
        try:
            pair_parent_id = int(pair[0])
            pair_child_id = int(pair[1])
        except (TypeError, ValueError):
            continue
        if pair_parent_id == int(parent_id) and pair_child_id == int(child_id):
            return True
    return False


def _attachment_pair_failure_category_for_entry(
    *,
    entry: dict[str, Any],
    parent_id: int,
    child_id: int,
    objects_by_id: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    if _entry_has_attachment_pair(entry, parent_id=parent_id, child_id=child_id):
        return {
            "decision": ATTACHMENT_PAIR_PROGRAM_DECISION_KEPT,
            "reason_codes": ["attachment_pair_referable"],
        }

    reason_codes: list[str] = []
    salvageable_failure = False
    uncertain = False
    status_containers = {
        "crop": entry.get("crop_label_statuses"),
        "full_frame": entry.get("full_frame_label_statuses"),
        "final": entry.get("label_statuses"),
    }

    for role, obj_id in (("parent", int(parent_id)), ("child", int(child_id))):
        obj = objects_by_id.get(int(obj_id))
        if obj is None:
            uncertain = True
            reason_codes.append(f"{role}_object_missing")
            continue
        label = str(obj.get("label", "")).strip().lower()
        if not label:
            uncertain = True
            reason_codes.append(f"{role}_label_missing")
            continue

        crop_status = str((status_containers["crop"] or {}).get(label, "")).strip().lower()
        full_frame_status = str((status_containers["full_frame"] or {}).get(label, "")).strip().lower()
        final_status = str((status_containers["final"] or {}).get(label, "")).strip().lower()

        if crop_status in {LABEL_STATUS_MULTIPLE, LABEL_STATUS_UNSURE}:
            salvageable_failure = True
            reason_codes.append(f"{role}_crop_{crop_status}")
        elif crop_status not in {LABEL_STATUS_UNIQUE, LABEL_STATUS_ABSENT, ""}:
            uncertain = True
            reason_codes.append(f"{role}_crop_status_{crop_status}")

        if full_frame_status in {LABEL_STATUS_MULTIPLE, LABEL_STATUS_UNSURE}:
            salvageable_failure = True
            reason_codes.append(f"{role}_full_frame_{full_frame_status}")
        elif full_frame_status not in {LABEL_STATUS_UNIQUE, LABEL_STATUS_ABSENT, ""}:
            uncertain = True
            reason_codes.append(f"{role}_full_frame_status_{full_frame_status}")

        if final_status in {LABEL_STATUS_MULTIPLE, LABEL_STATUS_UNSURE}:
            salvageable_failure = True
            reason_codes.append(f"{role}_final_{final_status}")
        elif not final_status:
            uncertain = True
            reason_codes.append(f"{role}_final_status_missing")
        elif final_status not in {LABEL_STATUS_UNIQUE, LABEL_STATUS_ABSENT}:
            uncertain = True
            reason_codes.append(f"{role}_final_status_{final_status}")

    if salvageable_failure and not uncertain:
        return {
            "decision": ATTACHMENT_PAIR_PROGRAM_DECISION_NEEDS_VLM_SALVAGE_REVIEW,
            "reason_codes": reason_codes,
        }
    return {
        "decision": ATTACHMENT_PAIR_PROGRAM_DECISION_UNCERTAIN,
        "reason_codes": reason_codes or ["status_conflict"],
    }


def _select_attachment_pair_cover_images(
    clarity_pass_frames: list[dict[str, Any]],
    pair_rows: list[dict[str, Any]],
) -> list[str]:
    image_to_pair_ids: dict[str, set[str]] = defaultdict(set)
    frame_selection_scores: dict[str, int] = {}
    for frame in clarity_pass_frames:
        image_name = str(frame.get("image_name", "")).strip()
        if not image_name:
            continue
        frame_selection_scores[image_name] = int(frame.get("frame_selection_score", 0) or 0)
    for pair_row in pair_rows:
        pair_id = str(pair_row.get("pair_id", "")).strip()
        if not pair_id:
            continue
        for image_name in pair_row.get("cover_image_names", []):
            image_to_pair_ids[str(image_name)].add(pair_id)

    uncovered = {
        pair_id
        for pair_ids in image_to_pair_ids.values()
        for pair_id in pair_ids
    }
    selected: list[str] = []
    remaining_images = set(image_to_pair_ids.keys())
    while uncovered and remaining_images:
        best_image_name = max(
            remaining_images,
            key=lambda image_name: (
                len(image_to_pair_ids.get(image_name, set()) & uncovered),
                frame_selection_scores.get(image_name, 0),
                image_name,
            ),
        )
        covered_here = image_to_pair_ids.get(best_image_name, set()) & uncovered
        if not covered_here:
            break
        selected.append(best_image_name)
        uncovered -= covered_here
        remaining_images.remove(best_image_name)
    return selected


def _attachment_pair_rename_advice_prompt(group_id: str, pair_rows: list[dict[str, Any]]) -> str:
    pair_lines: list[str] = []
    for pair_row in pair_rows:
        pair_lines.append(
            f'- pair_id={pair_row["pair_id"]}: '
            f'parent={pair_row["parent_label"]}#{pair_row["parent_id"]}, '
            f'child={pair_row["child_label"]}#{pair_row["child_id"]}, '
            f'program_reason_codes={json.dumps(pair_row.get("program_reason_codes", []), ensure_ascii=False)}'
        )
    return (
        "You are reviewing attachment pairs in a scene. "
        "You will see one or more full-frame images with consistent object-id annotations, "
        "followed by parent/child crops for specific dropped attachment pairs. "
        "For each pair, the current naming is not sufficient to uniquely refer to the two objects. "
        "Your task is to propose up to 3 rename-advice candidates for human review. "
        "Each candidate should contain object-level renaming fields and an optional short relation hint. "
        "You may modify only the parent name, only the child name, or both. "
        "Use only visible information from the images, such as color, material, text, shape, size, or relations to clearly visible nearby objects. "
        "A relation hint may be something like 'next to the window', but do not write a full question sentence. "
        "Try to offer multiple distinct plausible options when reliable. "
        "If the imagery is insufficient or you cannot give reliable unique rename advice, return unavailable and do not invent details. "
        f"This group's id is {json.dumps(group_id)}. "
        "Return strict JSON only using this schema: "
        '{"group_id":"...",'
        '"pair_reviews":[{"pair_id":"1->2","rename_advice_status":"ok",'
        '"rename_advice_reason":"",'
        '"rename_advice_candidates":[{"parent_surface_text":"round wooden table",'
        '"child_surface_text":"blue book",'
        '"relation_hint_text":"book on top of the table"}]}]}. '
        "Review every listed pair exactly once:\n"
        + "\n".join(pair_lines)
    )


def _attachment_pair_group_rename_advice_vlm_review(
    *,
    client,
    model_name: str,
    group_id: str,
    cover_images: list[dict[str, Any]],
    pair_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    default = {
        "group_id": group_id,
        "pair_reviews": [],
    }
    content: list[dict[str, Any]] = []
    for index, image_payload in enumerate(cover_images, start=1):
        data_url = str(image_payload.get("data_url", "")).strip()
        if not data_url:
            continue
        content.append(
            {
                "type": "text",
                "text": (
                    f"Full-frame cover image {index}: image_name={image_payload.get('image_name')} "
                    "with object-id annotations."
                ),
            }
        )
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": data_url},
            }
        )

    eligible_pair_rows: list[dict[str, Any]] = []
    for pair_row in pair_rows:
        parent_crop = str(pair_row.get("parent_crop_image_data_url", "")).strip()
        child_crop = str(pair_row.get("child_crop_image_data_url", "")).strip()
        if not parent_crop or not child_crop:
            continue
        eligible_pair_rows.append(pair_row)
        content.append(
            {
                "type": "text",
                "text": (
                    f"Pair {pair_row['pair_id']}: parent={pair_row['parent_label']}#{pair_row['parent_id']}, "
                    f"child={pair_row['child_label']}#{pair_row['child_id']}, "
                    f"program_reason_codes={json.dumps(pair_row.get('program_reason_codes', []), ensure_ascii=False)}. "
                    "The current naming is insufficient to uniquely refer to this dropped pair. "
                    f"The next two images are the parent crop then the child crop from image "
                    f"{pair_row.get('first_covered_image_name') or pair_row.get('cover_image_names', ['-'])[0]}."
                ),
            }
        )
        content.append({"type": "image_url", "image_url": {"url": parent_crop}})
        content.append({"type": "image_url", "image_url": {"url": child_crop}})

    if not cover_images:
        return {
            "group_id": group_id,
            "pair_reviews": [
                {
                    "pair_id": str(pair_row.get("pair_id", "")),
                    "rename_advice": _default_attachment_pair_rename_advice(
                        reason="missing_group_cover_images"
                    ),
                }
                for pair_row in pair_rows
            ],
            "raw_response": None,
        }
    if not eligible_pair_rows:
        return {
            "group_id": group_id,
            "pair_reviews": [
                {
                    "pair_id": str(pair_row.get("pair_id", "")),
                    "rename_advice": _default_attachment_pair_rename_advice(
                        reason="missing_pair_crops"
                    ),
                }
                for pair_row in pair_rows
            ],
            "raw_response": None,
        }

    content.append(
        {
            "type": "text",
            "text": _attachment_pair_rename_advice_prompt(group_id, eligible_pair_rows),
        }
    )
    parsed, raw_text = _call_vlm_json(
        client,
        model_name,
        content,
        default=default,
        max_tokens=1024,
    )
    reviews_by_pair_id: dict[str, dict[str, Any]] = {}
    for raw_review in parsed.get("pair_reviews", []):
        if not isinstance(raw_review, dict):
            continue
        pair_id = str(raw_review.get("pair_id", "")).strip()
        if not pair_id:
            continue
        status = _normalize_attachment_pair_rename_advice_status(
            raw_review.get("rename_advice_status")
        )
        candidates = _normalize_attachment_pair_rename_advice_candidates(
            raw_review.get("rename_advice_candidates")
        )
        reason = str(raw_review.get("rename_advice_reason", "")).strip()
        if status == ATTACHMENT_PAIR_RENAME_ADVICE_STATUS_OK and not candidates:
            status = ATTACHMENT_PAIR_RENAME_ADVICE_STATUS_UNAVAILABLE
            reason = reason or "missing_rename_advice_candidates"
        if status == ATTACHMENT_PAIR_RENAME_ADVICE_STATUS_UNAVAILABLE:
            candidates = []
        reviews_by_pair_id[pair_id] = {
            "pair_id": pair_id,
            "rename_advice": {
                "status": status,
                "reason": reason if status == ATTACHMENT_PAIR_RENAME_ADVICE_STATUS_UNAVAILABLE else (reason or ""),
                "candidates": candidates,
            },
        }

    normalized_reviews: list[dict[str, Any]] = []
    for pair_row in pair_rows:
        pair_id = str(pair_row.get("pair_id", "")).strip()
        normalized_reviews.append(
            reviews_by_pair_id.get(
                pair_id,
                {
                    "pair_id": pair_id,
                    "rename_advice": _default_attachment_pair_rename_advice(
                        reason="missing_pair_review_in_vlm_response"
                    ),
                },
            )
        )
    return {
        "group_id": str(parsed.get("group_id", group_id) or group_id),
        "pair_reviews": normalized_reviews,
        "raw_response": raw_text or None,
    }

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


def _call_vlm_json_impl(
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
        _record_vlm_call_failure()
        logger.warning("VLM call failed: %s", exc)
        return default, ""


def _call_vlm_json(
    client,
    model: str,
    content: list[dict],
    default: dict[str, Any],
    max_tokens: int = 512,
) -> tuple[dict[str, Any], str]:
    resolved_client = _resolve_vlm_client(client)
    semaphore = _VLM_REQUEST_SEMAPHORE
    if semaphore is None:
        return _call_vlm_json_impl(
            resolved_client,
            model,
            content,
            default,
            max_tokens=max_tokens,
        )
    with semaphore:
        return _call_vlm_json_impl(
            resolved_client,
            model,
            content,
            default,
            max_tokens=max_tokens,
        )


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
        "You are given one original scene image. "
        "Decide whether this frame is usable for object-level visual spatial-reasoning questions. "
        "A usable frame should allow several scene objects to be recognized and referred to reliably from the image alone. "
        "Reject frames that are too blurry, too dark, too unclear, or where most candidate objects are hard to identify or distinguish. "
        'Answer with strict JSON only: {"frame_usable": true, "reason": "clear_scene"}'
    )


def _frame_batch_prompt(image_names: list[str]) -> str:
    rendered_names = ", ".join(json.dumps(str(name), ensure_ascii=False) for name in image_names)
    return (
        "You are given multiple original scene images. "
        "For each image independently, decide whether that frame is usable for object-level visual spatial-reasoning questions. "
        "A usable frame should allow several scene objects to be recognized and referred to reliably from the image alone. "
        "Reject frames that are too blurry, too dark, too unclear, or where most candidate objects are hard to identify or distinguish. "
        "You will receive image_name text before each image. "
        f"Return exactly one result for each of these image_name values: {rendered_names}. "
        'Answer with strict JSON only using this schema: {"images":[{"image_name":"000001.jpg","frame_usable":true,"reason":"clear_scene"}]}'
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


def _object_review_batch_prompt(labels: list[str]) -> str:
    items = "\n".join(
        f'{index + 1}. {json.dumps(str(label), ensure_ascii=False)}'
        for index, label in enumerate(labels)
    )
    return (
        f"You will see a full scene image followed by {len(labels)} cropped regions.\n"
        "For each numbered crop, decide if the labeled object is clearly and unambiguously visible.\n"
        f"Labels:\n{items}\n"
        'Reply with JSON only: {"results": [{"index": 1, "status": "clear|absent|unsure", "reason": "..."}]}'
    )


def _full_frame_label_count_prompt(label: str) -> str:
    return (
        "You are given one full scene image. "
        "Count how many clearly identifiable instances of the target label are visible in the full image. "
        "The target label is "
        f"{json.dumps(str(label), ensure_ascii=False)}. "
        "Count only objects that are visually present and identifiable in the image itself. "
        "Do not infer hidden objects, off-screen objects, or objects that are too ambiguous to recognize. "
        "If none are visible, use count=0 and status=absent. "
        "If exactly one is clearly visible, use count=1 and status=unique. "
        "If two or more are clearly visible, use the best exact integer count you can and status=multiple. "
        "If you cannot judge confidently, use status=unsure and count=null. "
        'Answer with strict JSON only using this schema: {"count": 1, "status": "unique", "reason": "short reason"}'
    )


def _full_frame_label_count_batch_prompt(normalized_labels: list[str]) -> str:
    labels_str = ", ".join(json.dumps(str(label), ensure_ascii=False) for label in normalized_labels)
    return (
        "You are given one full scene image. "
        "For each target label independently, count how many clearly identifiable instances are visible in the image. "
        f"Objects to count: {labels_str}. "
        "Count only objects that are visually present and identifiable in the image itself. "
        "Do not infer hidden objects, off-screen objects, or objects that are too ambiguous to recognize. "
        "If none are visible, use count=0 and status=absent. "
        "If exactly one is clearly visible, use count=1 and status=unique. "
        "If two or more are clearly visible, use the best exact integer count you can and status=multiple. "
        "If you cannot judge confidently, use status=unsure and count=null. "
        "Return exactly one result for each label above. "
        'Reply with JSON only: {"results": [{"label": "...", "count": 1, "status": "unique", "reason": "short reason"}]}'
    )


def _full_frame_out_of_frame_label_prompt(label: str) -> str:
    return (
        "You are given one full scene image. "
        "Judge only whether the target label is completely not visible anywhere in this image. "
        "The target label is "
        f"{json.dumps(str(label), ensure_ascii=False)}. "
        "Return status=not_visible only when no identifiable instance of that label can be seen at all, "
        "and the absence is consistent with the object simply being outside the image frame. "
        "Return status=reject if any identifiable instance is visible, or if the absence could be explained by another cause instead of being out of frame. "
        "Return status=unsure if you cannot decide confidently. "
        'Answer with strict JSON only using this schema: {"status": "not_visible"}'
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


def _normalize_full_frame_label_count(value: object) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        pass
    text = str(value or "").strip().lower()
    if not text:
        return None
    word_to_count = {
        "zero": 0,
        "none": 0,
        "one": 1,
        "single": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
    }
    if text in word_to_count:
        return word_to_count[text]
    match = re.search(r"\d+", text)
    if match:
        return max(0, int(match.group(0)))
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


def _normalize_out_of_frame_review_status(value: object) -> str | None:
    text = str(value or "").strip().lower()
    if text in {
        OUT_OF_FRAME_REVIEW_STATUS_NOT_VISIBLE,
        "not visible",
        "not-visible",
        "out_of_frame",
        "out of frame",
        "off_screen",
        "off screen",
        "off-frame",
    }:
        return OUT_OF_FRAME_REVIEW_STATUS_NOT_VISIBLE
    if text in {
        OUT_OF_FRAME_REVIEW_STATUS_REJECT,
        "visible",
        "present",
        "in_frame",
        "in frame",
        "no",
    }:
        return OUT_OF_FRAME_REVIEW_STATUS_REJECT
    if text in {
        OUT_OF_FRAME_REVIEW_STATUS_UNSURE,
        "uncertain",
        "unknown",
        "unclear",
        "cannot_tell",
        "can't tell",
    }:
        return OUT_OF_FRAME_REVIEW_STATUS_UNSURE
    return None


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


def _normalize_cached_out_of_frame_not_visible_labels(value: object) -> list[str]:
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


def _normalize_cached_out_of_frame_label_reviews(value: object) -> list[dict[str, Any]]:
    reviews: list[dict[str, Any]] = []
    if not isinstance(value, list):
        return reviews
    for item in value:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label", "") or "").strip().lower()
        status = _normalize_out_of_frame_review_status(item.get("status"))
        if not label or status is None:
            continue
        reviews.append(
            {
                "label": label,
                "status": status,
                "raw_response": item.get("raw_response"),
            }
        )
    return reviews


def _normalize_cached_out_of_frame_vlm_early_stop(value: object) -> bool:
    return _coerce_bool(value, default=False)


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
    attachment_human_review_cards = _normalize_attachment_human_review_cards(
        entry.get("attachment_human_review_cards")
    )
    attachment_referable_object_ids = sorted(
        {
            int(obj_id)
            for obj_id in attachment_referable_object_ids
        }
        | {
            int(obj_id)
            for obj_id in _attachment_human_review_object_ids(attachment_human_review_cards)
        }
    )

    derived = {
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
    derived = _apply_attachment_layer_payloads(
        derived,
        attachment_pairs=entry.get("attachment_referable_pairs"),
        selector_pair_count=entry.get("attachment_pair_ge_50_count", 0),
        selector_viewpoint_exempt=entry.get("attachment_viewpoint_exempt", False),
        final_selection_rank=entry.get("final_selection_rank", FRAME_SELECTION_FALLBACK_RANK),
    )
    out_of_frame_keys = {
        "out_of_frame_label_reviews",
        "out_of_frame_not_visible_labels",
        "out_of_frame_label_to_object_ids",
        "out_of_frame_vlm_early_stop",
    }
    if out_of_frame_keys.issubset(entry.keys()):
        derived.update(
            {
                "out_of_frame_label_reviews": _normalize_cached_out_of_frame_label_reviews(
                    entry.get("out_of_frame_label_reviews")
                ),
                "out_of_frame_not_visible_labels": _normalize_cached_out_of_frame_not_visible_labels(
                    entry.get("out_of_frame_not_visible_labels")
                ),
                "out_of_frame_label_to_object_ids": _shared_normalize_label_to_object_ids(
                    entry.get("out_of_frame_label_to_object_ids")
                ),
                "out_of_frame_vlm_early_stop": _normalize_cached_out_of_frame_vlm_early_stop(
                    entry.get("out_of_frame_vlm_early_stop")
                ),
            }
        )
    return derived


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
        "out_of_frame_label_reviews",
        "out_of_frame_not_visible_labels",
        "out_of_frame_label_to_object_ids",
        "out_of_frame_vlm_early_stop",
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
        "out_of_frame_label_reviews": _normalize_cached_out_of_frame_label_reviews(
            entry.get("out_of_frame_label_reviews")
        ),
        "out_of_frame_not_visible_labels": _normalize_cached_out_of_frame_not_visible_labels(
            entry.get("out_of_frame_not_visible_labels")
        ),
        "out_of_frame_label_to_object_ids": _shared_normalize_label_to_object_ids(
            entry.get("out_of_frame_label_to_object_ids")
        ),
        "out_of_frame_vlm_early_stop": _normalize_cached_out_of_frame_vlm_early_stop(
            entry.get("out_of_frame_vlm_early_stop")
        ),
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
    if "attachment_selector_signal" in entry:
        normalized_entry["attachment_selector_signal"] = _build_attachment_selector_signal_payload(
            well_cropped_pair_count=(entry.get("attachment_selector_signal") or {}).get("well_cropped_pair_count", 0),
            viewpoint_exempt=(entry.get("attachment_selector_signal") or {}).get("viewpoint_exempt", False),
        )
    if "attachment_final_referability" in entry:
        normalized_entry["attachment_final_referability"] = _build_attachment_final_referability_payload(
            attachment_referable_object_ids=(entry.get("attachment_final_referability") or {}).get("object_ids", []),
            attachment_pairs=(entry.get("attachment_final_referability") or {}).get("pairs", []),
        )
    if "attachment_final_frame_selection" in entry:
        normalized_entry["attachment_final_frame_selection"] = _build_attachment_final_frame_selection_payload(
            final_selection_rank=(entry.get("attachment_final_frame_selection") or {}).get(
                "selection_rank",
                FRAME_SELECTION_FALLBACK_RANK,
            ),
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


def _normalize_optional_int(value: object) -> int | None:
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return None


def _normalize_optional_float(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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
    frame_usable = _legacy_frame_clear(parsed)
    if "clear" in parsed and "frame_usable" not in parsed:
        frame_usable = _coerce_bool(parsed.get("clear"), default=frame_usable)
    clear = _coerce_bool(parsed.get("clear"), default=frame_usable)
    clarity_score = _normalize_clarity_score(parsed.get("clarity_score"), default=60)
    normalized = {
        "clear": clear,
        "clarity_score": clarity_score,
        "frame_usable": frame_usable,
        "reason": str(parsed.get("reason", "")).strip() or "frame_usable_parse_fallback",
    }
    brisque_score = _normalize_optional_float(parsed.get("brisque_score"))
    brisque_input_width = _normalize_optional_int(parsed.get("brisque_input_width"))
    brisque_input_height = _normalize_optional_int(parsed.get("brisque_input_height"))
    if (
        brisque_score is not None
        or brisque_input_width is not None
        or brisque_input_height is not None
    ):
        normalized["brisque_score"] = brisque_score
        normalized["brisque_input_width"] = brisque_input_width
        normalized["brisque_input_height"] = brisque_input_height
    return normalized


def _chunk_list(items: list[Any], chunk_size: int) -> list[list[Any]]:
    size = max(1, int(chunk_size))
    return [
        items[index : index + size]
        for index in range(0, len(items), size)
    ]


def _normalize_frame_batch_item(
    value: Any,
    *,
    expected_image_names: set[str],
) -> tuple[str, dict[str, Any]] | None:
    if not isinstance(value, dict):
        return None
    image_name = str(value.get("image_name", "") or "").strip()
    if not image_name or image_name not in expected_image_names:
        return None
    if (
        "frame_usable" not in value
        and "clear" not in value
        and "usable_for_spatial_reasoning" not in value
        and "severely_out_of_focus" not in value
    ):
        return None
    return image_name, _normalize_frame_review(value)


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
    return _apply_attachment_layer_payloads(
        enriched,
        attachment_pairs=attachment_pairs,
        selector_pair_count=enriched.get("attachment_pair_ge_50_count", 0),
        selector_viewpoint_exempt=enriched.get("attachment_viewpoint_exempt", False),
        final_selection_rank=enriched.get("final_selection_rank", FRAME_SELECTION_FALLBACK_RANK),
    )


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


def _frame_decision_batch(
    client,
    model: str,
    batch_items: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    if not batch_items:
        return {}

    expected_image_names = {
        str(item.get("image_name", "")).strip()
        for item in batch_items
        if str(item.get("image_name", "")).strip()
    }
    if not expected_image_names:
        return {}

    content: list[dict[str, Any]] = []
    ordered_names: list[str] = []
    for item in batch_items:
        image_name = str(item.get("image_name", "")).strip()
        if not image_name:
            continue
        ordered_names.append(image_name)
        image_b64 = str(item.get("image_b64", "") or "")
        if not image_b64:
            image = item.get("image")
            if not isinstance(image, np.ndarray):
                continue
            image_b64 = _image_to_base64(image)
        content.append({"type": "text", "text": f"image_name: {image_name}"})
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}})
    if not ordered_names:
        return {}

    default = {"images": []}
    parsed, _raw_text = _call_vlm_json(
        client,
        model,
        [
            *content,
            {"type": "text", "text": _frame_batch_prompt(ordered_names)},
        ],
        default=default,
        max_tokens=min(
            FRAME_CLARITY_BATCH_MAX_TOKENS,
            FRAME_CLARITY_MAX_TOKENS_PER_IMAGE * max(1, len(ordered_names)),
        ),
    )

    batch_results: dict[str, dict[str, Any]] = {}
    raw_items = parsed.get("images") if isinstance(parsed, dict) else None
    if isinstance(raw_items, list):
        for raw_item in raw_items:
            normalized_item = _normalize_frame_batch_item(
                raw_item,
                expected_image_names=expected_image_names,
            )
            if normalized_item is None:
                continue
            image_name, normalized_review = normalized_item
            batch_results.setdefault(image_name, normalized_review)

    for item in batch_items:
        image_name = str(item.get("image_name", "")).strip()
        if not image_name or image_name in batch_results:
            continue
        image = item.get("image")
        if not isinstance(image, np.ndarray):
            continue
        batch_results[image_name] = _frame_decision(
            client,
            model,
            image,
            image_b64=str(item.get("image_b64", "") or "") or None,
        )
    return batch_results


def _review_frame_clarity_batch(
    *,
    client,
    model_name: str,
    color_dir: Path,
    frames: list[dict[str, Any]],
    batch_size: int = FRAME_CLARITY_BATCH_SIZE,
) -> dict[str, dict[str, Any] | None]:
    reviewed_by_image_name: dict[str, dict[str, Any] | None] = {}
    pending_batch_items: list[dict[str, Any]] = []

    for frame in frames:
        image_name = str(frame.get("image_name", "")).strip()
        if not image_name or image_name in reviewed_by_image_name:
            continue
        image_path = color_dir / image_name
        image = cv2.imread(str(image_path))
        if image is None:
            logger.warning("Cannot read image %s", image_path)
            reviewed_by_image_name[image_name] = None
            continue
        pending_batch_items.append(
            {
                "image_name": image_name,
                "image": image,
            }
        )

    for batch in _chunk_list(pending_batch_items, batch_size):
        batch_reviews = _frame_decision_batch(client, model_name, batch)
        for batch_item in batch:
            image_name = str(batch_item.get("image_name", "")).strip()
            frame_info = batch_reviews.get(image_name)
            if not isinstance(frame_info, dict):
                reviewed_by_image_name.setdefault(image_name, None)
                continue
            frame = next(
                (
                    candidate
                    for candidate in frames
                    if str(candidate.get("image_name", "")).strip() == image_name
                ),
                None,
            )
            if not isinstance(frame, dict):
                reviewed_by_image_name[image_name] = None
                continue
            selector_score = int(
                frame.get("selector_score", frame.get("score", frame.get("n_visible", 0))) or 0
            )
            reviewed_by_image_name[image_name] = {
                **frame,
                "selector_score": selector_score,
                "frame_info": dict(frame_info),
                "frame_selection_score": _frame_selection_score(selector_score, frame_info),
            }
    return reviewed_by_image_name


def _resolve_group_frame_reviews(
    *,
    sampled_frames: list[dict[str, Any]],
    client,
    model_name: str,
    color_dir: Path,
    frame_review_getter: Callable[[dict[str, Any]], dict[str, Any] | None] | None = None,
    frame_review_batch_getter: Callable[[list[dict[str, Any]]], dict[str, Any] | None] | None = None,
    frame_clarity_batch_size: int = FRAME_CLARITY_BATCH_SIZE,
) -> dict[str, dict[str, Any] | None]:
    reviewed_by_image_name: dict[str, dict[str, Any] | None] = {}
    if callable(frame_review_batch_getter):
        batch_result = frame_review_batch_getter(sampled_frames)
        if isinstance(batch_result, dict):
            for image_name, reviewed_frame in batch_result.items():
                normalized_name = str(image_name or "").strip()
                if not normalized_name:
                    continue
                reviewed_by_image_name[normalized_name] = (
                        dict(reviewed_frame) if isinstance(reviewed_frame, dict) else None
                )
    if callable(frame_review_getter):
        for frame in sampled_frames:
            image_name = str(frame.get("image_name", "")).strip()
            if not image_name or image_name in reviewed_by_image_name:
                continue
            reviewed_by_image_name[image_name] = frame_review_getter(frame)
    elif not callable(frame_review_batch_getter):
        for frame in sampled_frames:
            image_name = str(frame.get("image_name", "")).strip()
            if not image_name or image_name in reviewed_by_image_name:
                continue
            reviewed_by_image_name[image_name] = _review_frame_clarity(
                client=client,
                model_name=model_name,
                color_dir=color_dir,
                frame=frame,
            )
    return reviewed_by_image_name


_BRISQUE_SCORER_LOCAL = threading.local()


def _get_brisque_scorer() -> BrisqueScorer:
    scorer = getattr(_BRISQUE_SCORER_LOCAL, "scorer", None)
    if scorer is None:
        scorer = BrisqueScorer()
        _BRISQUE_SCORER_LOCAL.scorer = scorer
    return scorer


def _load_scene_image_for_brisque(
    *,
    color_dir: Path,
    image_name: str,
    scene_image_getter: Callable[[str], np.ndarray | None] | None = None,
) -> np.ndarray | None:
    if callable(scene_image_getter):
        image = scene_image_getter(image_name)
        return image if isinstance(image, np.ndarray) else None
    image = cv2.imread(str(color_dir / image_name))
    if image is None:
        logger.warning("Cannot read image %s", color_dir / image_name)
    return image


def _score_group_frames_by_brisque(
    *,
    sampled_frames: list[dict[str, Any]],
    color_dir: Path,
    scene_image_getter: Callable[[str], np.ndarray | None] | None = None,
) -> list[dict[str, Any]]:
    if not sampled_frames:
        return []

    scorer = _get_brisque_scorer()
    scored_frames: list[dict[str, Any]] = []
    for original_index, frame in enumerate(sampled_frames):
        image_name = str(frame.get("image_name", "")).strip()
        brisque_info = {
            "brisque_score": None,
            "brisque_input_width": None,
            "brisque_input_height": None,
        }
        if image_name:
            image = _load_scene_image_for_brisque(
                color_dir=color_dir,
                image_name=image_name,
                scene_image_getter=scene_image_getter,
            )
            if image is not None:
                brisque_info = compute_brisque_score(
                    image,
                    scorer=scorer,
                    max_side=FRAME_BRISQUE_MAX_SIDE,
                )
        scored_frames.append(
            {
                "frame": frame,
                "image_name": image_name,
                "original_index": int(original_index),
                "brisque_score": brisque_info["brisque_score"],
                "brisque_input_width": brisque_info["brisque_input_width"],
                "brisque_input_height": brisque_info["brisque_input_height"],
            }
        )
    return scored_frames


def _sort_group_frames_for_clarity_review(
    scored_frames: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return sorted(
        scored_frames,
        key=lambda item: (
            item.get("brisque_score") is None,
            float(item.get("brisque_score")) if item.get("brisque_score") is not None else float("inf"),
            int(item.get("original_index", 0)),
            str(item.get("image_name", "")),
        ),
    )


def _attach_brisque_to_reviewed_frame(
    reviewed_frame: dict[str, Any] | None,
    brisque_doc: dict[str, Any],
) -> dict[str, Any] | None:
    if not isinstance(reviewed_frame, dict):
        return reviewed_frame
    updated = dict(reviewed_frame)
    frame_info = _normalize_frame_review(updated.get("frame_info"))
    frame_info["brisque_score"] = brisque_doc.get("brisque_score")
    frame_info["brisque_input_width"] = brisque_doc.get("brisque_input_width")
    frame_info["brisque_input_height"] = brisque_doc.get("brisque_input_height")
    updated["frame_info"] = frame_info
    return updated


def _review_group_frames_until_stop(
    *,
    ordered_scored_frames: list[dict[str, Any]],
    client,
    model_name: str,
    color_dir: Path,
    frame_review_getter: Callable[[dict[str, Any]], dict[str, Any] | None] | None = None,
    frame_review_batch_getter: Callable[[list[dict[str, Any]]], dict[str, Any] | None] | None = None,
    frame_clarity_batch_size: int = FRAME_CLARITY_BATCH_SIZE,
    stop_on_reviewed_frame: Callable[[dict[str, Any], dict[str, Any]], str | None] | None = None,
) -> dict[str, Any]:
    reviewed_by_image_name: dict[str, dict[str, Any] | None] = {}
    reviewed_image_names_in_order: list[str] = []
    early_stop_image_name: str | None = None
    early_stop_reason: str | None = None

    batch_size = frame_clarity_batch_size if callable(frame_review_batch_getter) else 1
    for batch in _chunk_list(ordered_scored_frames, batch_size):
        batch_frames = [
            item.get("frame")
            for item in batch
            if isinstance(item.get("frame"), dict)
        ]
        batch_reviews = _resolve_group_frame_reviews(
            sampled_frames=batch_frames,
            client=client,
            model_name=model_name,
            color_dir=color_dir,
            frame_review_getter=frame_review_getter,
            frame_review_batch_getter=frame_review_batch_getter,
            frame_clarity_batch_size=frame_clarity_batch_size,
        )
        for scored_frame in batch:
            image_name = str(scored_frame.get("image_name", "")).strip()
            reviewed_frame = _attach_brisque_to_reviewed_frame(
                batch_reviews.get(image_name),
                scored_frame,
            )
            reviewed_by_image_name[image_name] = reviewed_frame
            reviewed_image_names_in_order.append(image_name)
            if not isinstance(reviewed_frame, dict) or not callable(stop_on_reviewed_frame):
                continue
            stop_reason = stop_on_reviewed_frame(scored_frame, reviewed_frame)
            if stop_reason:
                early_stop_image_name = image_name
                early_stop_reason = str(stop_reason)
                return {
                    "reviewed_by_image_name": reviewed_by_image_name,
                    "reviewed_image_names_in_order": reviewed_image_names_in_order,
                    "early_stop_image_name": early_stop_image_name,
                    "early_stop_reason": early_stop_reason,
                }

    return {
        "reviewed_by_image_name": reviewed_by_image_name,
        "reviewed_image_names_in_order": reviewed_image_names_in_order,
        "early_stop_image_name": early_stop_image_name,
        "early_stop_reason": early_stop_reason,
    }


def _visible_object_frame_group_key(frame: dict[str, Any]) -> tuple[Any, ...] | None:
    visible_object_ids = frame.get("visible_object_ids")
    if isinstance(visible_object_ids, list):
        return tuple(sorted(int(obj_id) for obj_id in visible_object_ids))
    return None


def _frame_image_name(frame: dict[str, Any]) -> str:
    return str(frame.get("image_name", "")).strip()


def _group_frame_sampling_stride(group_frame_count: int) -> int:
    count = max(0, int(group_frame_count))
    if count <= 10:
        return 1
    if count <= 30:
        return 2
    return 3


def _sample_group_frames(frames: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    group_frame_stride = _group_frame_sampling_stride(len(frames))
    sampled_frames = list(frames[::group_frame_stride])
    if not sampled_frames and frames:
        sampled_frames = [frames[0]]
    return sampled_frames, group_frame_stride


def _attachment_pair_set_for_frame(
    frame: dict[str, Any],
    attachment_graph: dict[int, list[int]] | None,
) -> tuple[tuple[int, int], ...] | None:
    visible_object_ids = _visible_object_frame_group_key(frame)
    if visible_object_ids is None:
        return None
    return tuple(_attachment_pairs_for_visible_group(attachment_graph, visible_object_ids))


def _attachment_frame_pose_angle_deg(
    frame_a: dict[str, Any],
    frame_b: dict[str, Any],
    poses: dict[str, CameraPose] | None,
) -> float | None:
    if not poses:
        return None
    image_name_a = _frame_image_name(frame_a)
    image_name_b = _frame_image_name(frame_b)
    if not image_name_a or not image_name_b:
        return None
    pose_a = poses.get(image_name_a)
    pose_b = poses.get(image_name_b)
    if pose_a is None or pose_b is None:
        return None
    forward_a = np.asarray(pose_a.rotation, dtype=np.float64).T[:, 2]
    forward_b = np.asarray(pose_b.rotation, dtype=np.float64).T[:, 2]
    denom = float(np.linalg.norm(forward_a) * np.linalg.norm(forward_b))
    if denom <= 0.0:
        return None
    cosine = float(np.clip(np.dot(forward_a, forward_b) / denom, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _visible_object_frame_merge_metrics(
    anchor_frame: dict[str, Any],
    candidate_frame: dict[str, Any],
    poses: dict[str, CameraPose] | None,
) -> tuple[float | None, int] | None:
    anchor_visible_ids = _visible_object_frame_group_key(anchor_frame)
    candidate_visible_ids = _visible_object_frame_group_key(candidate_frame)
    if anchor_visible_ids is None or candidate_visible_ids is None:
        return None

    symmetric_diff_size = len(set(anchor_visible_ids) ^ set(candidate_visible_ids))
    if symmetric_diff_size > VISIBLE_OBJECT_GROUP_MAX_VISIBLE_SYMMETRIC_DIFF:
        return None

    angle_deg = _attachment_frame_pose_angle_deg(anchor_frame, candidate_frame, poses)
    if angle_deg is None:
        if symmetric_diff_size == 0:
            return None, symmetric_diff_size
        return None
    if angle_deg > ATTACHMENT_GROUP_MAX_POSE_ANGLE_DEG:
        return None
    return angle_deg, symmetric_diff_size


def _attachment_frame_merge_metrics(
    anchor_frame: dict[str, Any],
    candidate_frame: dict[str, Any],
    poses: dict[str, CameraPose] | None,
) -> tuple[float | None, int] | None:
    angle_deg = _attachment_frame_pose_angle_deg(anchor_frame, candidate_frame, poses)
    if angle_deg is None:
        if _frame_image_name(anchor_frame) and _frame_image_name(anchor_frame) == _frame_image_name(candidate_frame):
            return None, 0
        return None
    if angle_deg > ATTACHMENT_GROUP_MAX_POSE_ANGLE_DEG:
        return None
    return angle_deg, 0


def _build_visible_object_pose_merged_groups(
    *,
    frames: list[dict[str, Any]],
    poses: dict[str, CameraPose] | None,
    merge_metrics_getter: Callable[
        [dict[str, Any], dict[str, Any], dict[str, CameraPose] | None],
        tuple[float | None, int] | None,
    ] = _visible_object_frame_merge_metrics,
) -> list[dict[str, Any]]:
    merged_groups: list[dict[str, Any]] = []
    for frame in frames:
        frame_visible_ids = _visible_object_frame_group_key(frame)
        if frame_visible_ids is None:
            continue
        matching_groups: list[tuple[float, int, int]] = []
        for group_index, group in enumerate(merged_groups):
            metrics = merge_metrics_getter(
                group["anchor_frame"],
                frame,
                poses,
            )
            if metrics is None:
                continue
            angle_deg, merge_tiebreaker = metrics
            matching_groups.append(
                (
                    float("inf") if angle_deg is None else float(angle_deg),
                    int(merge_tiebreaker),
                    int(group_index),
                )
            )
        if matching_groups:
            _, _, best_group_index = min(matching_groups)
            best_group = merged_groups[best_group_index]
            best_group["frames"].append(frame)
            best_group["visible_object_ids"].update(frame_visible_ids)
            continue
        merged_groups.append(
            {
                "anchor_frame": frame,
                "frames": [frame],
                "visible_object_ids": set(frame_visible_ids),
            }
        )

    return [
        {
            "anchor_frame": group["anchor_frame"],
            "frames": list(group["frames"]),
            "visible_object_ids": sorted(int(obj_id) for obj_id in group["visible_object_ids"]),
        }
        for group in merged_groups
    ]


def _count_visible_object_frame_groups(
    frames: list[dict[str, Any]],
    poses: dict[str, CameraPose] | None = None,
) -> int:
    return len(
        _build_visible_object_pose_merged_groups(
            frames=frames,
            poses=poses,
        )
    )


def _build_attachment_frame_groups(
    *,
    frames: list[dict[str, Any]],
    attachment_graph: dict[int, list[int]] | None,
    poses: dict[str, CameraPose] | None,
) -> list[dict[str, Any]]:
    pair_buckets: dict[tuple[tuple[int, int], ...], list[dict[str, Any]]] = {}
    for frame in frames:
        pair_set_key = _attachment_pair_set_for_frame(frame, attachment_graph)
        if pair_set_key is None:
            continue
        pair_buckets.setdefault(pair_set_key, []).append(frame)

    merged_groups: list[dict[str, Any]] = []
    for pair_set_key, bucket_frames in pair_buckets.items():
        bucket_groups = _build_visible_object_pose_merged_groups(
            frames=bucket_frames,
            poses=poses,
            merge_metrics_getter=_attachment_frame_merge_metrics,
        )
        for group in bucket_groups:
            visible_object_ids = [
                int(obj_id) for obj_id in group.get("visible_object_ids", [])
            ]
            merged_groups.append(
                {
                    "anchor_frame": group.get("anchor_frame"),
                    "frames": list(group.get("frames", [])),
                    "visible_object_ids": visible_object_ids,
                    "group_pairs": [
                        (int(parent_id), int(child_id))
                        for parent_id, child_id in pair_set_key
                    ],
                    "pair_set_key": [list(pair) for pair in pair_set_key],
                }
            )
    return merged_groups


def _select_attachment_group_representatives(
    *,
    client,
    model_name: str,
    scene_dir: Path,
    frames: list[dict[str, Any]],
    attachment_graph: dict[int, list[int]] | None,
    poses: dict[str, CameraPose] | None = None,
    frame_review_getter: Callable[[dict[str, Any]], dict[str, Any] | None] | None = None,
    frame_review_batch_getter: Callable[[list[dict[str, Any]]], dict[str, Any] | None] | None = None,
    attachment_entry_builder: Callable[[dict[str, Any], dict[str, Any]], dict[str, Any] | None] | None = None,
    max_accepted_frame_count: int | None = None,
    vlm_workers: int = 1,
    frame_clarity_batch_size: int = FRAME_CLARITY_BATCH_SIZE,
    attachment_clarity_min_score: int = DEFAULT_ATTACHMENT_CLARITY_MIN_SCORE,
    failed_signatures_seen: set[tuple[int, ...]] | None = None,
) -> list[dict[str, Any]]:
    if not frames:
        return []

    color_dir = scene_dir / "color"
    grouped_items = list(
        enumerate(
            _build_attachment_frame_groups(
                frames=frames,
                attachment_graph=attachment_graph,
                poses=poses,
            )
        )
    )
    accepted_target: int | None = None
    if max_accepted_frame_count is not None:
        accepted_target = max(0, int(max_accepted_frame_count))
        if accepted_target <= 0:
            return []
    scene_failed_signatures_seen = (
        failed_signatures_seen if failed_signatures_seen is not None else set()
    )

    def _select_group(item: tuple[int, dict[str, Any]]) -> dict[str, Any] | None:
        group_id, group_doc = item
        group_frames = list(group_doc.get("frames", []))
        sampled_frames, _group_frame_stride = _sample_group_frames(group_frames)
        scored_frames = _score_group_frames_by_brisque(
            sampled_frames=sampled_frames,
            color_dir=color_dir,
        )
        ordered_scored_frames = _sort_group_frames_for_clarity_review(scored_frames)
        accepted_frame: dict[str, Any] | None = None

        def _stop_on_reviewed_frame(
            scored_frame: dict[str, Any],
            reviewed_frame: dict[str, Any],
        ) -> str | None:
            nonlocal accepted_frame
            frame = scored_frame.get("frame")
            if not isinstance(frame, dict):
                return None
            frame_info = reviewed_frame.get("frame_info", {})
            if not bool(frame_info.get("frame_usable", True)):
                return None
            combined = dict(reviewed_frame)
            combined["attachment_view_group_id"] = group_id
            if attachment_entry_builder is not None:
                entry = attachment_entry_builder(frame, combined)
                if not isinstance(entry, dict):
                    return None
                combined.update(entry)
            attachment_pairs = _build_attachment_referable_pairs(
                attachment_graph,
                combined.get("attachment_referable_object_ids"),
            )
            if not attachment_pairs:
                failed_signature = _failed_referability_object_id_signature(
                    combined,
                    bbox_in_frame_ratio_min=ATTACHMENT_REFERABLE_BBOX_IN_FRAME_RATIO_MIN,
                    projected_area_px_min=QUESTION_REVIEW_CROP_MIN_PROJECTED_AREA_PX,
                )
                if failed_signature:
                    if failed_signature in scene_failed_signatures_seen:
                        return None
                    scene_failed_signatures_seen.add(failed_signature)
                return None
            accepted_frame = _with_attachment_pair_metadata(
                combined,
                combined,
                attachment_graph,
                attachment_view_group_id=group_id,
            )
            return "accepted_attachment_referable_frame"

        _review_group_frames_until_stop(
            ordered_scored_frames=ordered_scored_frames,
            client=client,
            model_name=model_name,
            color_dir=color_dir,
            frame_review_getter=frame_review_getter,
            frame_review_batch_getter=frame_review_batch_getter,
            frame_clarity_batch_size=frame_clarity_batch_size,
            stop_on_reviewed_frame=_stop_on_reviewed_frame,
        )
        return accepted_frame

    selected_by_group: list[tuple[int, dict[str, Any]]] = []
    next_group_index = 0
    accepted_frame_count = 0
    while next_group_index < len(grouped_items):
        remaining_target = None
        if accepted_target is not None:
            remaining_target = accepted_target - accepted_frame_count
            if remaining_target <= 0:
                break
        batch_size = (
            max(1, remaining_target)
            if remaining_target is not None
            else len(grouped_items) - next_group_index
        )
        batch_items = grouped_items[next_group_index : next_group_index + batch_size]
        # Scene-level failed-signature reuse needs deterministic group order.
        batch_results = [_select_group(item) for item in batch_items]
        for batch_item, reviewed_selection in zip(batch_items, batch_results):
            if isinstance(reviewed_selection, dict):
                selected_by_group.append((int(batch_item[0]), reviewed_selection))
                accepted_frame_count += 1
        next_group_index += len(batch_items)
    selected_by_group.sort(key=lambda item: item[0])
    return [frame for _, frame in selected_by_group]


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


def _build_attachment_pair_salvage_pair_row(
    *,
    parent_id: int,
    child_id: int,
    relation_types: list[str],
    clarity_pass_frames: list[dict[str, Any]],
    objects_by_id: dict[int, dict[str, Any]],
    bbox_hard_fail_min: float,
    projected_area_hard_fail_min: float,
) -> dict[str, Any]:
    pair_id = _attachment_pair_id(parent_id, child_id)
    parent_label = _scene_object_label(objects_by_id.get(int(parent_id), {})).lower()
    child_label = _scene_object_label(objects_by_id.get(int(child_id), {})).lower()
    cover_image_names: list[str] = []
    kept_image_names: list[str] = []
    coverage_by_image_name: list[dict[str, Any]] = []
    covered_failure_categories: list[dict[str, Any]] = []
    saw_uncertain_coverage = False
    reason_codes: list[str] = []

    for frame in clarity_pass_frames:
        image_name = str(frame.get("image_name", "")).strip()
        entry = frame.get("entry")
        if not isinstance(entry, dict):
            coverage_by_image_name.append(
                {
                    "image_name": image_name,
                    "covered": False,
                    "uncertain": True,
                    "reason_codes": ["missing_referability_entry"],
                }
            )
            saw_uncertain_coverage = True
            reason_codes.append("missing_referability_entry")
            continue

        coverage = _attachment_pair_coverage_for_entry(
            entry=entry,
            parent_id=int(parent_id),
            child_id=int(child_id),
            bbox_hard_fail_min=bbox_hard_fail_min,
            projected_area_hard_fail_min=projected_area_hard_fail_min,
        )
        coverage_by_image_name.append(
            {
                "image_name": image_name,
                "covered": bool(coverage["covered"]),
                "uncertain": bool(coverage["uncertain"]),
                "reason_codes": list(coverage.get("reason_codes", [])),
            }
        )
        reason_codes.extend(coverage.get("reason_codes", []))
        if coverage["uncertain"]:
            saw_uncertain_coverage = True
        if not coverage["covered"]:
            continue

        cover_image_names.append(image_name)
        if _entry_has_attachment_pair(entry, parent_id=parent_id, child_id=child_id):
            kept_image_names.append(image_name)
            continue
        failure_category = _attachment_pair_failure_category_for_entry(
            entry=entry,
            parent_id=parent_id,
            child_id=child_id,
            objects_by_id=objects_by_id,
        )
        covered_failure_categories.append(failure_category)
        reason_codes.extend(failure_category.get("reason_codes", []))

    dedup_reason_codes = sorted({str(code) for code in reason_codes if str(code).strip()})
    if kept_image_names:
        program_decision = ATTACHMENT_PAIR_PROGRAM_DECISION_KEPT
    elif not clarity_pass_frames:
        program_decision = ATTACHMENT_PAIR_PROGRAM_DECISION_AUTO_DROP_HARD_FAIL
        dedup_reason_codes = dedup_reason_codes or ["no_clarity_pass_images"]
    elif not cover_image_names:
        if saw_uncertain_coverage:
            program_decision = ATTACHMENT_PAIR_PROGRAM_DECISION_UNCERTAIN
            dedup_reason_codes = dedup_reason_codes or ["coverage_uncertain"]
        else:
            program_decision = ATTACHMENT_PAIR_PROGRAM_DECISION_AUTO_DROP_HARD_FAIL
            dedup_reason_codes = dedup_reason_codes or ["no_coverable_clarity_pass_image"]
    elif covered_failure_categories and all(
        item.get("decision") == ATTACHMENT_PAIR_PROGRAM_DECISION_NEEDS_VLM_SALVAGE_REVIEW
        for item in covered_failure_categories
    ):
        program_decision = ATTACHMENT_PAIR_PROGRAM_DECISION_NEEDS_VLM_SALVAGE_REVIEW
    else:
        program_decision = ATTACHMENT_PAIR_PROGRAM_DECISION_UNCERTAIN
        dedup_reason_codes = dedup_reason_codes or ["status_conflict"]

    return {
        "pair_id": pair_id,
        "parent_id": int(parent_id),
        "parent_label": parent_label,
        "child_id": int(child_id),
        "child_label": child_label,
        "relation_types": list(relation_types),
        "program_decision": program_decision,
        "program_status": _attachment_pair_program_status(program_decision),
        "program_reason_codes": dedup_reason_codes,
        "current_attachment_referable": bool(kept_image_names),
        "cover_image_names": cover_image_names,
        "kept_image_names": kept_image_names,
        "first_covered_image_name": cover_image_names[0] if cover_image_names else None,
        "coverage_by_image_name": coverage_by_image_name,
        "rename_advice": _default_attachment_pair_rename_advice(reason="not_requested"),
        "human_decision": None,
        "human_notes": "",
    }


def _build_attachment_pair_salvage_scene_review(
    *,
    client,
    model_name: str,
    scene_id: str,
    split: str,
    scene_dir: Path,
    objects: list[dict[str, Any]],
    objects_by_id: dict[int, dict[str, Any]],
    attachment_graph: dict[int, list[int]] | None,
    attachment_edges: list[dict[str, Any]],
    frames: list[dict[str, Any]],
    poses: dict[str, CameraPose] | None,
    frame_review_getter: Callable[[dict[str, Any]], dict[str, Any] | None] | None = None,
    frame_review_batch_getter: Callable[[list[dict[str, Any]]], dict[str, Any] | None] | None = None,
    scene_image_getter: Callable[[str], np.ndarray | None] | None = None,
    attachment_entry_builder: Callable[[dict[str, Any], dict[str, Any]], dict[str, Any] | None] | None = None,
    bbox_hard_fail_min: float,
    projected_area_hard_fail_min: float,
    frame_clarity_batch_size: int = FRAME_CLARITY_BATCH_SIZE,
    attachment_clarity_min_score: int = DEFAULT_ATTACHMENT_CLARITY_MIN_SCORE,
    failed_signatures_seen: set[tuple[int, ...]] | None = None,
) -> dict[str, Any]:
    relation_type_map = _attachment_edge_relation_type_map(attachment_edges)
    color_dir = scene_dir / "color"
    image_cache: dict[str, np.ndarray | None] = {}
    terminal_output_lines: list[str] = []
    grouped_frames = _build_attachment_frame_groups(
        frames=frames,
        attachment_graph=attachment_graph,
        poses=poses,
    )
    scene_failed_signatures_seen = (
        failed_signatures_seen if failed_signatures_seen is not None else set()
    )

    def _load_scene_image(image_name: str) -> np.ndarray | None:
        if callable(scene_image_getter):
            image = scene_image_getter(image_name)
            return image if isinstance(image, np.ndarray) else None
        if image_name not in image_cache:
            image_cache[image_name] = cv2.imread(str(color_dir / image_name))
        return image_cache[image_name]

    groups: list[dict[str, Any]] = []
    for group_index, grouped_frame_doc in enumerate(grouped_frames):
        group_frames = list(grouped_frame_doc.get("frames", []))
        group_visible_object_ids = [
            int(obj_id) for obj_id in grouped_frame_doc.get("visible_object_ids", [])
        ]
        group_pairs = [
            (int(parent_id), int(child_id))
            for parent_id, child_id in grouped_frame_doc.get("group_pairs", [])
        ]
        if not group_pairs:
            continue

        sampled_frames, group_frame_stride = _sample_group_frames(group_frames)
        scored_frames = _score_group_frames_by_brisque(
            sampled_frames=sampled_frames,
            color_dir=color_dir,
            scene_image_getter=scene_image_getter,
        )
        ordered_scored_frames = _sort_group_frames_for_clarity_review(scored_frames)
        clarity_pass_frames: list[dict[str, Any]] = []
        clarity_pass_image_names: list[str] = []
        frame_records_by_image_name: dict[str, dict[str, Any]] = {}
        duplicate_failed_signature_image_names: list[str] = []
        duplicate_failed_signature_object_ids_by_image_name: dict[str, list[int]] = {}

        def _stop_on_reviewed_frame(
            scored_frame: dict[str, Any],
            reviewed_frame: dict[str, Any],
        ) -> str | None:
            frame = scored_frame.get("frame")
            if not isinstance(frame, dict):
                return None
            frame_info = reviewed_frame.get("frame_info", {})
            if not bool(frame_info.get("frame_usable", True)):
                return None
            image_name = str(scored_frame.get("image_name", "")).strip()
            entry = None
            if attachment_entry_builder is not None:
                entry = attachment_entry_builder(frame, reviewed_frame)
                if isinstance(entry, dict):
                    entry = _with_attachment_pair_metadata(
                        entry,
                        entry,
                        attachment_graph,
                        attachment_view_group_id=group_index,
                    )
            failed_signature: tuple[int, ...] = ()
            if isinstance(entry, dict) and not entry.get("attachment_referable_pairs"):
                failed_signature = _failed_referability_object_id_signature(
                    entry,
                    bbox_in_frame_ratio_min=ATTACHMENT_REFERABLE_BBOX_IN_FRAME_RATIO_MIN,
                    projected_area_px_min=QUESTION_REVIEW_CROP_MIN_PROJECTED_AREA_PX,
                )
                if failed_signature:
                    if failed_signature in scene_failed_signatures_seen:
                        duplicate_failed_signature_image_names.append(image_name)
                        duplicate_failed_signature_object_ids_by_image_name[image_name] = list(
                            failed_signature
                        )
                        return None
                    scene_failed_signatures_seen.add(failed_signature)
            frame_record = {
                "image_name": image_name,
                "image_stem": _image_name_stem(image_name),
                "clarity_score": int(frame_info.get("clarity_score", 0) or 0),
                "frame_selection_score": int(
                    reviewed_frame.get(
                        "frame_selection_score",
                        frame.get("selector_score", frame.get("score", frame.get("n_visible", 0))),
                    )
                    or 0
                ),
                "frame_info": dict(frame_info),
                "entry": dict(entry) if isinstance(entry, dict) else None,
            }
            clarity_pass_frames.append(frame_record)
            clarity_pass_image_names.append(image_name)
            frame_records_by_image_name[image_name] = frame_record
            if isinstance(entry, dict) and entry.get("attachment_referable_pairs"):
                return "accepted_attachment_referable_frame"
            return None

        review_result = _review_group_frames_until_stop(
            ordered_scored_frames=ordered_scored_frames,
            client=client,
            model_name=model_name,
            color_dir=color_dir,
            frame_review_getter=frame_review_getter,
            frame_review_batch_getter=frame_review_batch_getter,
            frame_clarity_batch_size=frame_clarity_batch_size,
            stop_on_reviewed_frame=_stop_on_reviewed_frame,
        )

        pair_rows = [
            _build_attachment_pair_salvage_pair_row(
                parent_id=parent_id,
                child_id=child_id,
                relation_types=relation_type_map.get((parent_id, child_id), []),
                clarity_pass_frames=clarity_pass_frames,
                objects_by_id=objects_by_id,
                bbox_hard_fail_min=bbox_hard_fail_min,
                projected_area_hard_fail_min=projected_area_hard_fail_min,
            )
            for parent_id, child_id in group_pairs
        ]
        selected_cover_image_names = _select_attachment_pair_cover_images(
            [
                frame
                for frame in clarity_pass_frames
                if isinstance(frame.get("entry"), dict)
            ],
            pair_rows,
        )

        selected_cover_images: list[dict[str, Any]] = []
        for image_name in selected_cover_image_names:
            frame_record = frame_records_by_image_name.get(image_name)
            if not isinstance(frame_record, dict):
                continue
            entry = frame_record.get("entry")
            image = _load_scene_image(image_name)
            if image is None or not isinstance(entry, dict):
                continue
            annotated = _annotate_attachment_pair_salvage_frame(
                image,
                entry=entry,
                visible_object_ids=group_visible_object_ids,
                objects_by_id=objects_by_id,
            )
            selected_cover_images.append(
                {
                    "image_name": image_name,
                    "image_stem": _image_name_stem(image_name),
                    "clarity_score": int(frame_record.get("clarity_score", 0) or 0),
                    "covered_pair_ids": [
                        str(pair_row.get("pair_id", ""))
                        for pair_row in pair_rows
                        if image_name in pair_row.get("cover_image_names", [])
                    ],
                    "data_url": _image_to_data_url(annotated),
                }
            )

        dropped_pairs = [
            pair_row
            for pair_row in pair_rows
            if pair_row.get("program_decision") != ATTACHMENT_PAIR_PROGRAM_DECISION_KEPT
        ]
        for pair_row in dropped_pairs:
            first_covered_image_name = str(pair_row.get("first_covered_image_name", "") or "").strip()
            if not first_covered_image_name:
                continue
            frame_record = frame_records_by_image_name.get(first_covered_image_name)
            entry = (frame_record or {}).get("entry")
            if not isinstance(entry, dict):
                continue
            image = _load_scene_image(first_covered_image_name)
            if image is None:
                continue
            parent_review = _lookup_object_payload(entry.get("object_reviews"), int(pair_row["parent_id"]))
            child_review = _lookup_object_payload(entry.get("object_reviews"), int(pair_row["child_id"]))
            parent_crop = _crop_image_from_bounds(image, (parent_review or {}).get("crop_bounds_px"))
            child_crop = _crop_image_from_bounds(image, (child_review or {}).get("crop_bounds_px"))
            if parent_crop is not None:
                pair_row["parent_crop_image_data_url"] = _image_to_data_url(parent_crop)
            if child_crop is not None:
                pair_row["child_crop_image_data_url"] = _image_to_data_url(child_crop)

        group_vlm_review = None
        if dropped_pairs:
            group_vlm_review = _attachment_pair_group_rename_advice_vlm_review(
                client=client,
                model_name=model_name,
                group_id=f"{scene_id}:group_{group_index}",
                cover_images=selected_cover_images,
                pair_rows=dropped_pairs,
            )
            pair_vlm_reviews = {
                str(review.get("pair_id", "")): review
                for review in group_vlm_review.get("pair_reviews", [])
                if isinstance(review, dict)
            }
            for pair_row in dropped_pairs:
                pair_review = pair_vlm_reviews.get(pair_row["pair_id"], {})
                pair_row["rename_advice"] = dict(
                    pair_review.get("rename_advice")
                    or _default_attachment_pair_rename_advice(
                        reason="missing_pair_review_in_vlm_response"
                    )
                )

        terminal_output_lines.append(
            (
                f"[attachment-pair-salvage] scene={scene_id} group={group_index} "
                f"pairs={len(pair_rows)} clarity_pass={len(clarity_pass_image_names)} "
                f"cover={len(selected_cover_image_names)} dropped={len(dropped_pairs)} "
                f"duplicate_failed_signature_skips={len(duplicate_failed_signature_image_names)}"
            )
        )
        groups.append(
            {
                "group_id": f"{scene_id}:group_{group_index}",
                "group_index": int(group_index),
                "visible_object_ids": group_visible_object_ids,
                "visible_object_labels": [
                    f"{_scene_object_label(objects_by_id.get(int(obj_id), {})).lower()}#{int(obj_id)}"
                    for obj_id in group_visible_object_ids
                ],
                "group_frame_image_names": [
                    str(frame.get("image_name", "")).strip()
                    for frame in group_frames
                ],
                "sampled_frame_image_names": [
                    str(frame.get("image_name", "")).strip()
                    for frame in sampled_frames
                ],
                "sampled_frames": [
                    {
                        "image_name": str(item.get("image_name", "")).strip(),
                        "brisque_score": item.get("brisque_score"),
                        "brisque_input_width": item.get("brisque_input_width"),
                        "brisque_input_height": item.get("brisque_input_height"),
                    }
                    for item in scored_frames
                ],
                "brisque_sorted_frame_image_names": [
                    str(item.get("image_name", "")).strip()
                    for item in ordered_scored_frames
                ],
                "clarity_pass_image_names": clarity_pass_image_names,
                "duplicate_failed_signature_image_names": duplicate_failed_signature_image_names,
                "duplicate_failed_signature_object_ids_by_image_name": (
                    duplicate_failed_signature_object_ids_by_image_name
                ),
                "early_stop_image_name": review_result["early_stop_image_name"],
                "early_stop_reason": review_result["early_stop_reason"],
                "selected_cover_image_names": selected_cover_image_names,
                "selected_cover_images": selected_cover_images,
                "group_frame_stride": int(group_frame_stride),
                "pair_count_total": len(pair_rows),
                "kept_pair_ids": [
                    str(pair_row.get("pair_id", ""))
                    for pair_row in pair_rows
                    if pair_row.get("program_decision") == ATTACHMENT_PAIR_PROGRAM_DECISION_KEPT
                ],
                "dropped_pair_ids": [
                    str(pair_row.get("pair_id", ""))
                    for pair_row in dropped_pairs
                ],
                "pairs": pair_rows,
                "dropped_pairs": dropped_pairs,
                "vlm_group_review": group_vlm_review,
            }
        )

    scene_record = {
        "scene_id": str(scene_id),
        "split": str(split),
        "pipeline_outcome": None,
        "object_count": len(objects),
        "group_count_total": len(groups),
        "group_count_with_clarity_pass_images": sum(
            1 for group in groups if group.get("clarity_pass_image_names")
        ),
        "group_count_with_multi_image_cover": sum(
            1
            for group in groups
            if len(group.get("selected_cover_image_names", [])) > 1
        ),
        "pair_count_total": sum(int(group.get("pair_count_total", 0) or 0) for group in groups),
        "pair_count_kept": sum(
            1
            for group in groups
            for pair_row in group.get("pairs", [])
            if pair_row.get("program_decision") == ATTACHMENT_PAIR_PROGRAM_DECISION_KEPT
        ),
        "pair_count_auto_drop_hard_fail": sum(
            1
            for group in groups
            for pair_row in group.get("pairs", [])
            if pair_row.get("program_decision") == ATTACHMENT_PAIR_PROGRAM_DECISION_AUTO_DROP_HARD_FAIL
        ),
        "pair_count_needs_vlm_salvage_review": sum(
            1
            for group in groups
            for pair_row in group.get("pairs", [])
            if pair_row.get("program_decision") == ATTACHMENT_PAIR_PROGRAM_DECISION_NEEDS_VLM_SALVAGE_REVIEW
        ),
        "pair_count_uncertain": sum(
            1
            for group in groups
            for pair_row in group.get("pairs", [])
            if pair_row.get("program_decision") == ATTACHMENT_PAIR_PROGRAM_DECISION_UNCERTAIN
        ),
        "terminal_output_lines": terminal_output_lines,
        "groups": groups,
    }
    return scene_record


def _build_attachment_pair_salvage_review_scene_record(
    *,
    scene_id: str,
    split: str,
    pipeline_outcome: str,
    scene_review: dict[str, Any] | None,
) -> dict[str, Any]:
    base = {
        "scene_id": str(scene_id),
        "split": str(split),
        "pipeline_outcome": str(pipeline_outcome),
        "object_count": 0,
        "group_count_total": 0,
        "group_count_with_clarity_pass_images": 0,
        "group_count_with_multi_image_cover": 0,
        "pair_count_total": 0,
        "pair_count_kept": 0,
        "pair_count_auto_drop_hard_fail": 0,
        "pair_count_needs_vlm_salvage_review": 0,
        "pair_count_uncertain": 0,
        "terminal_output_lines": [],
        "groups": [],
    }
    if not isinstance(scene_review, dict):
        base["terminal_output_lines"] = [
            f"[attachment-pair-salvage] scene={scene_id} outcome={pipeline_outcome} groups=0 pairs=0"
        ]
        return base
    record = dict(scene_review)
    record["scene_id"] = str(scene_id)
    record["split"] = str(split)
    record["pipeline_outcome"] = str(pipeline_outcome)
    record["terminal_output_lines"] = list(scene_review.get("terminal_output_lines", []))
    if not record["terminal_output_lines"]:
        record["terminal_output_lines"] = [
            f"[attachment-pair-salvage] scene={scene_id} outcome={pipeline_outcome} "
            f"groups={record.get('group_count_total', 0)} pairs={record.get('pair_count_total', 0)}"
        ]
    return record


def _build_attachment_pair_salvage_review_document(
    *,
    referability_cache_output: Path,
    scenes: list[dict[str, Any]],
) -> dict[str, Any]:
    scene_count = len(scenes)
    edited_html_outputs_by_scene = {
        scene_id: str(_edited_attachment_pair_salvage_html_output_path(referability_cache_output, scene_id))
        for scene_id in [
            str(scene.get("scene_id", "")).strip()
            for scene in scenes
            if str(scene.get("scene_id", "")).strip()
        ]
    }
    return {
        "name": ATTACHMENT_PAIR_SALVAGE_REVIEW_NAME,
        "version": ATTACHMENT_PAIR_SALVAGE_REVIEW_VERSION,
        "generated_by": "scripts/run_vlm_referability.py",
        "review_stage": ATTACHMENT_PAIR_SALVAGE_REVIEW_STAGE,
        "referability_cache_output": str(referability_cache_output),
        "edited_html_output_glob": _edited_attachment_pair_salvage_html_output_glob(referability_cache_output),
        "edited_html_outputs_by_scene": edited_html_outputs_by_scene,
        "scene_count": scene_count,
        "group_count_total": sum(int(scene.get("group_count_total", 0) or 0) for scene in scenes),
        "group_count_with_clarity_pass_images": sum(
            int(scene.get("group_count_with_clarity_pass_images", 0) or 0)
            for scene in scenes
        ),
        "group_count_with_multi_image_cover": sum(
            int(scene.get("group_count_with_multi_image_cover", 0) or 0)
            for scene in scenes
        ),
        "pair_count_total": sum(int(scene.get("pair_count_total", 0) or 0) for scene in scenes),
        "pair_count_kept": sum(int(scene.get("pair_count_kept", 0) or 0) for scene in scenes),
        "pair_count_auto_drop_hard_fail": sum(
            int(scene.get("pair_count_auto_drop_hard_fail", 0) or 0)
            for scene in scenes
        ),
        "pair_count_needs_vlm_salvage_review": sum(
            int(scene.get("pair_count_needs_vlm_salvage_review", 0) or 0)
            for scene in scenes
        ),
        "pair_count_uncertain": sum(
            int(scene.get("pair_count_uncertain", 0) or 0)
            for scene in scenes
        ),
        "terminal_output_lines": [
            line
            for scene in scenes
            for line in scene.get("terminal_output_lines", [])
        ],
        "scenes": scenes,
    }


def _build_attachment_pair_salvage_review_scene_document(
    *,
    review_doc: dict[str, Any],
    scene_id: str,
) -> dict[str, Any]:
    scene_key = str(scene_id).strip()
    selected_scene = None
    for scene in review_doc.get("scenes", []):
        if str(scene.get("scene_id", "")).strip() == scene_key:
            selected_scene = scene
            break
    if selected_scene is None:
        raise ValueError(f"Scene {scene_key!r} not found in attachment pair salvage review document")

    scene_doc = dict(review_doc)
    scene_doc["scene_count"] = 1
    scene_doc["group_count_total"] = int(selected_scene.get("group_count_total", 0) or 0)
    scene_doc["group_count_with_clarity_pass_images"] = int(
        selected_scene.get("group_count_with_clarity_pass_images", 0) or 0
    )
    scene_doc["group_count_with_multi_image_cover"] = int(
        selected_scene.get("group_count_with_multi_image_cover", 0) or 0
    )
    scene_doc["pair_count_total"] = int(selected_scene.get("pair_count_total", 0) or 0)
    scene_doc["pair_count_kept"] = int(selected_scene.get("pair_count_kept", 0) or 0)
    scene_doc["pair_count_auto_drop_hard_fail"] = int(
        selected_scene.get("pair_count_auto_drop_hard_fail", 0) or 0
    )
    scene_doc["pair_count_needs_vlm_salvage_review"] = int(
        selected_scene.get("pair_count_needs_vlm_salvage_review", 0) or 0
    )
    scene_doc["pair_count_uncertain"] = int(selected_scene.get("pair_count_uncertain", 0) or 0)
    scene_doc["terminal_output_lines"] = list(selected_scene.get("terminal_output_lines", []))
    scene_doc["scenes"] = [selected_scene]
    edited_outputs = review_doc.get("edited_html_outputs_by_scene", {})
    if isinstance(edited_outputs, dict):
        scene_output = edited_outputs.get(scene_key)
    else:
        scene_output = None
    scene_doc["edited_html_outputs_by_scene"] = (
        {scene_key: str(scene_output)}
        if scene_output is not None
        else {}
    )
    return scene_doc


_ATTACHMENT_PAIR_REASON_CODE_ZH = {
    "attachment_pair_referable": "该 attachment pair 已可指代",
    "coverage_uncertain": "覆盖情况不确定",
    "missing_referability_entry": "缺少 referability 条目",
    "no_clarity_pass_images": "没有清晰可用图像",
    "no_coverable_clarity_pass_image": "没有可覆盖该 attachment pair 的清晰图像",
    "status_conflict": "状态冲突，无法稳定判定",
}

_ATTACHMENT_PAIR_REASON_SUFFIX_ZH = {
    "bbox_in_frame_ratio_too_small": "框在画面中的占比过小",
    "candidate_not_visible": "在候选可见物体中不可见",
    "final_status_missing": "最终状态缺失",
    "label_missing": "标签缺失",
    "missing_candidate_visible_object_ids": "缺少候选可见物体列表",
    "missing_object_review": "缺少物体审阅结果",
    "object_missing": "对象缺失",
    "projected_area_too_small": "投影面积过小",
    "vlm_status_absent": "被判定为不存在",
}

_ATTACHMENT_PAIR_REASON_SCOPE_ZH = {
    "crop": "裁剪图中",
    "final": "最终判定中",
    "full_frame": "整图中",
}

_ATTACHMENT_PAIR_REASON_LOCAL_OUTCOME_ZH = {
    "empty_or_invalid_crop": "裁剪为空或无效",
    "excluded": "被局部规则排除",
    "missing": "局部结果缺失",
    "not_visible": "局部视图中不可见",
    "out_of_frame": "已出框",
    "reviewed": "已审阅",
}


def _attachment_pair_reason_object_ref_zh(pair_row: dict[str, Any], role: str) -> str:
    object_id = str(pair_row.get(f"{role}_id", "") or "").strip()
    label = str(pair_row.get(f"{role}_label", "") or "").strip()
    if object_id and label:
        return f"物体{object_id}({label})"
    if object_id:
        return f"物体{object_id}"
    if label:
        return f"物体({label})"
    return "该物体"


def _attachment_pair_reason_code_to_zh(reason_code: Any, pair_row: dict[str, Any] | None = None) -> str:
    code = str(reason_code or "").strip()
    if not code:
        return "-"
    mapped = _ATTACHMENT_PAIR_REASON_CODE_ZH.get(code)
    if mapped is not None:
        return mapped

    for role_prefix, role_key in (("parent_", "parent"), ("child_", "child")):
        if not code.startswith(role_prefix):
            continue
        suffix = code[len(role_prefix):]
        object_ref = (
            _attachment_pair_reason_object_ref_zh(pair_row or {}, role_key)
            if pair_row is not None
            else role_key
        )
        mapped_suffix = _ATTACHMENT_PAIR_REASON_SUFFIX_ZH.get(suffix)
        if mapped_suffix is not None:
            return f"{object_ref}{mapped_suffix}"
        if suffix.startswith("local_outcome_"):
            outcome = suffix[len("local_outcome_"):]
            return f"{object_ref}{_ATTACHMENT_PAIR_REASON_LOCAL_OUTCOME_ZH.get(outcome, suffix)}"
        for scope, scope_zh in _ATTACHMENT_PAIR_REASON_SCOPE_ZH.items():
            if suffix == f"{scope}_multiple":
                return f"{object_ref}{scope_zh}存在多个同类目标"
            if suffix == f"{scope}_unsure":
                return f"{object_ref}{scope_zh}判定不确定"
        return code

    return code


def _attachment_pair_reason_codes_to_zh(
    reason_codes: list[Any],
    pair_row: dict[str, Any] | None = None,
) -> str:
    rendered: list[str] = []
    seen: set[str] = set()
    for reason_code in reason_codes:
        text = _attachment_pair_reason_code_to_zh(reason_code, pair_row=pair_row)
        if not text or text in seen:
            continue
        seen.add(text)
        rendered.append(text)
    return "，".join(rendered) if rendered else "-"


def _attachment_pair_renderable_cover(
    group: dict[str, Any],
    pair_row: dict[str, Any],
) -> dict[str, str] | None:
    image_name = str(pair_row.get("first_covered_image_name", "") or "").strip()
    if not image_name:
        return None
    for item in group.get("selected_cover_images", []):
        if str(item.get("image_name", "")).strip() != image_name:
            continue
        data_url = str(item.get("data_url", "")).strip()
        if not data_url:
            return None
        return {
            "image_name": image_name,
            "image_stem": str(item.get("image_stem", "")).strip() or _image_name_stem(image_name),
            "data_url": data_url,
        }
    return None


class _AttachmentPairSalvageHtmlParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.cards: list[dict[str, Any]] = []
        self._current_card: dict[str, Any] | None = None
        self._card_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_map = {str(key): ("" if value is None else str(value)) for key, value in attrs}
        classes = {
            item.strip()
            for item in str(attrs_map.get("class", "")).split()
            if item.strip()
        }
        if tag == "article" and "pair-card" in classes and self._current_card is None:
            self._current_card = {
                "scene_id": str(attrs_map.get("data-scene-id", "")).strip(),
                "image_name": str(attrs_map.get("data-image-name", "")).strip(),
                "group_id": str(attrs_map.get("data-group-id", "")).strip(),
                "pair_id": str(attrs_map.get("data-pair-id", "")).strip(),
                "parent_id": str(attrs_map.get("data-parent-id", "")).strip(),
                "parent_label": str(attrs_map.get("data-parent-label", "")).strip(),
                "child_id": str(attrs_map.get("data-child-id", "")).strip(),
                "child_label": str(attrs_map.get("data-child-label", "")).strip(),
                "deleted": str(attrs_map.get("data-deleted", "")).strip().lower() == "true",
                "parent_surface_text": "",
                "child_surface_text": "",
            }
            self._card_depth = 1
            return

        if self._current_card is None:
            return

        if tag == "article":
            self._card_depth += 1
            return
        if tag != "input":
            return

        name = str(attrs_map.get("name", "")).strip().lower()
        value = html.unescape(str(attrs_map.get("value", ""))).strip()
        if name == "image_id":
            self._current_card["image_id"] = value
            self._current_card["image_name"] = _salvage_review_image_name_with_original_suffix(
                original_image_name=str(self._current_card.get("image_name", "")),
                image_id=value,
            )
        elif name == "parent_surface_text":
            self._current_card["parent_surface_text"] = value
        elif name == "child_surface_text":
            self._current_card["child_surface_text"] = value

    def handle_endtag(self, tag: str) -> None:
        if tag != "article" or self._current_card is None:
            return
        self._card_depth -= 1
        if self._card_depth > 0:
            return
        self.cards.append(dict(self._current_card))
        self._current_card = None
        self._card_depth = 0


def _parse_attachment_pair_salvage_review_html(html_text: str) -> list[dict[str, Any]]:
    parser = _AttachmentPairSalvageHtmlParser()
    parser.feed(str(html_text))
    parser.close()
    return parser.cards


def _salvage_review_image_name_with_original_suffix(*, original_image_name: str, image_id: str) -> str:
    original_name = str(original_image_name).strip()
    updated_stem = str(image_id).strip()
    if not updated_stem:
        return ""
    original_suffix = Path(original_name).suffix if original_name else ""
    return f"{updated_stem}{original_suffix}"


def _apply_attachment_pair_salvage_html_review(
    *,
    html_text: str,
    cache_doc: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(cache_doc, dict):
        raise ValueError("referability cache must be a JSON object")
    if str(cache_doc.get("version", "")).strip() != REFERABILITY_CACHE_VERSION:
        raise ValueError(
            f"Referability cache version mismatch: expected {REFERABILITY_CACHE_VERSION}, "
            f"got {str(cache_doc.get('version', '') or '<missing>').strip()}."
        )
    frames = cache_doc.get("frames")
    if not isinstance(frames, dict):
        raise ValueError("referability cache is missing a frames mapping")

    kept_by_frame: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    seen_pair_texts: dict[tuple[str, str, int, int], tuple[str, str]] = {}
    for card in _parse_attachment_pair_salvage_review_html(html_text):
        if bool(card.get("deleted", False)):
            continue
        scene_id = str(card.get("scene_id", "")).strip()
        image_name = str(card.get("image_name", "")).strip()
        if not scene_id or not image_name:
            continue
        try:
            parent_id = int(card.get("parent_id"))
            child_id = int(card.get("child_id"))
        except (TypeError, ValueError):
            continue
        parent_surface_text = str(card.get("parent_surface_text", "")).strip()
        child_surface_text = str(card.get("child_surface_text", "")).strip()
        if not parent_surface_text or not child_surface_text:
            continue

        pair_key = (scene_id, image_name, parent_id, child_id)
        pair_texts = (parent_surface_text, child_surface_text)
        existing_pair_texts = seen_pair_texts.get(pair_key)
        if existing_pair_texts is not None and existing_pair_texts != pair_texts:
            raise ValueError(
                "Conflicting human review texts for "
                f"{scene_id}/{image_name} pair {parent_id}->{child_id}: "
                f"{existing_pair_texts[0]!r}/{existing_pair_texts[1]!r} vs "
                f"{parent_surface_text!r}/{child_surface_text!r}"
            )
        seen_pair_texts[pair_key] = pair_texts
        kept_by_frame[(scene_id, image_name)].append(
            {
                "pair_id": str(card.get("pair_id", f"{parent_id}->{child_id}")).strip() or f"{parent_id}->{child_id}",
                "parent_id": parent_id,
                "parent_label": str(card.get("parent_label", "")).strip(),
                "parent_surface_text": parent_surface_text,
                "child_id": child_id,
                "child_label": str(card.get("child_label", "")).strip(),
                "child_surface_text": child_surface_text,
                "source": "human_salvage_html",
            }
        )

    updated_cache = json.loads(json.dumps(cache_doc, ensure_ascii=False))
    updated_frames = updated_cache.get("frames", {})
    for (scene_id, image_name), review_cards in kept_by_frame.items():
        scene_frames = updated_frames.get(scene_id)
        if not isinstance(scene_frames, dict):
            raise ValueError(f"Scene {scene_id} not found in referability cache")
        entry = scene_frames.get(image_name)
        if not isinstance(entry, dict):
            raise ValueError(f"Frame {scene_id}/{image_name} not found in referability cache")

        normalized_cards = _normalize_attachment_human_review_cards(review_cards)
        surface_text_by_obj_id: dict[int, str] = {}
        for item in normalized_cards:
            parent_id = int(item["parent_id"])
            child_id = int(item["child_id"])
            parent_surface_text = str(item["parent_surface_text"]).strip()
            child_surface_text = str(item["child_surface_text"]).strip()
            parent_existing = surface_text_by_obj_id.get(parent_id)
            if parent_existing is not None and parent_existing != parent_surface_text:
                raise ValueError(
                    f"Conflicting parent surface text for {scene_id}/{image_name} object {parent_id}: "
                    f"{parent_existing!r} vs {parent_surface_text!r}"
                )
            child_existing = surface_text_by_obj_id.get(child_id)
            if child_existing is not None and child_existing != child_surface_text:
                raise ValueError(
                    f"Conflicting child surface text for {scene_id}/{image_name} object {child_id}: "
                    f"{child_existing!r} vs {child_surface_text!r}"
                )
            surface_text_by_obj_id[parent_id] = parent_surface_text
            surface_text_by_obj_id[child_id] = child_surface_text

        existing_pairs = [
            [int(pair[0]), int(pair[1])]
            for pair in (entry.get("attachment_referable_pairs") or [])
            if isinstance(pair, (list, tuple)) and len(pair) == 2
        ]
        merged_pairs = sorted(
            {
                (int(pair[0]), int(pair[1]))
                for pair in existing_pairs
            }
            | {
                (int(item["parent_id"]), int(item["child_id"]))
                for item in normalized_cards
            }
        )
        merged_object_ids = sorted(
            {
                int(obj_id)
                for obj_id in (entry.get("attachment_referable_object_ids") or [])
            }
            | {
                int(obj_id)
                for obj_id in _attachment_human_review_object_ids(normalized_cards)
            }
        )

        updated_entry = dict(entry)
        updated_entry["attachment_human_review_cards"] = normalized_cards
        updated_entry["attachment_referable_pairs"] = [
            [int(parent_id), int(child_id)]
            for parent_id, child_id in merged_pairs
        ]
        updated_entry["attachment_referable_pair_count"] = len(updated_entry["attachment_referable_pairs"])
        updated_entry["attachment_referable_object_ids"] = merged_object_ids
        updated_entry["attachment_final_referability"] = _build_attachment_final_referability_payload(
            attachment_referable_object_ids=merged_object_ids,
            attachment_pairs=updated_entry["attachment_referable_pairs"],
        )
        scene_frames[image_name] = updated_entry

    return updated_cache


def _render_attachment_pair_salvage_review_html(review_doc: dict[str, Any]) -> str:
    def _render_simple_list(values: list[str]) -> str:
        if not values:
            return "-"
        return ", ".join(html.escape(str(value)) for value in values)

    def _render_rename_advice_block(pair_row: dict[str, Any]) -> str:
        rename_advice = pair_row.get("rename_advice")
        if not isinstance(rename_advice, dict):
            rename_advice = _default_attachment_pair_rename_advice(reason="rename_advice_missing")
        status = _normalize_attachment_pair_rename_advice_status(rename_advice.get("status"))
        candidates = _normalize_attachment_pair_rename_advice_candidates(rename_advice.get("candidates"))
        if status != ATTACHMENT_PAIR_RENAME_ADVICE_STATUS_OK or not candidates:
            return (
                '<div class="pair-text pair-rename-advice">'
                "<strong>VLM Rename Advice</strong>"
                '<div class="rename-advice-unavailable">'
                "VLM could not provide reliable rename advice for this pair."
                "</div>"
                "</div>"
            )
        candidate_blocks: list[str] = []
        for index, candidate in enumerate(candidates, start=1):
            lines: list[str] = []
            if candidate.get("parent_surface_text"):
                lines.append(
                    f'<div class="rename-advice-line"><strong>parent -&gt;</strong> '
                    f'{html.escape(candidate["parent_surface_text"])}</div>'
                )
            if candidate.get("child_surface_text"):
                lines.append(
                    f'<div class="rename-advice-line"><strong>child -&gt;</strong> '
                    f'{html.escape(candidate["child_surface_text"])}</div>'
                )
            if candidate.get("relation_hint_text"):
                lines.append(
                    f'<div class="rename-advice-line"><strong>relation hint -&gt;</strong> '
                    f'{html.escape(candidate["relation_hint_text"])}</div>'
                )
            candidate_blocks.append(
                f'<div class="rename-advice-candidate"><div class="rename-advice-index">Option {index}</div>'
                f'{"".join(lines)}</div>'
            )
        return (
            '<div class="pair-text pair-rename-advice">'
            "<strong>VLM Rename Advice</strong>"
            f'{"".join(candidate_blocks)}'
            "</div>"
        )

    rendered_scene_count = 0
    rendered_group_count = 0
    rendered_pair_count = 0
    included_scene_ids: list[str] = []
    seen_scene_ids: set[str] = set()
    referability_cache_output = str(review_doc.get("referability_cache_output", "")).strip()
    edited_html_output_glob = str(review_doc.get("edited_html_output_glob", "")).strip()
    raw_edited_outputs_by_scene = review_doc.get("edited_html_outputs_by_scene", {})
    edited_html_outputs_by_scene = (
        {
            str(scene_id).strip(): str(path).strip()
            for scene_id, path in raw_edited_outputs_by_scene.items()
            if str(scene_id).strip() and str(path).strip()
        }
        if isinstance(raw_edited_outputs_by_scene, dict)
        else {}
    )
    scene_ids_in_doc = [
        str(scene.get("scene_id", "")).strip()
        for scene in review_doc.get("scenes", [])
        if str(scene.get("scene_id", "")).strip()
    ]
    editable_scene_ids = [
        scene_id
        for scene_id in edited_html_outputs_by_scene
        if scene_id in set(scene_ids_in_doc)
    ]
    editable_scene_id = editable_scene_ids[0] if len(editable_scene_ids) == 1 else None
    if editable_scene_id is None and len(scene_ids_in_doc) == 1:
        editable_scene_id = scene_ids_in_doc[0]
    edited_html_output = (
        edited_html_outputs_by_scene.get(editable_scene_id, "")
        if editable_scene_id is not None
        else ""
    )
    if edited_html_output:
        edited_html_target_display = edited_html_output
        edited_html_filename = Path(edited_html_output).name
    elif edited_html_output_glob:
        edited_html_target_display = edited_html_output_glob
        edited_html_filename = Path(referability_cache_output).stem + "_edited.html"
    else:
        edited_html_target_display = "edited.html"
        edited_html_filename = "edited.html"
    per_scene_output_lines = [
        f"{scene_id} -> {Path(path).name}"
        for scene_id, path in sorted(edited_html_outputs_by_scene.items())
    ]
    scene_sections: list[str] = []
    for scene in review_doc.get("scenes", []):
        scene_id = str(scene.get("scene_id", "")).strip()
        rendered_group_ids: set[str] = set()
        pair_cards: list[str] = []
        for group in scene.get("groups", []):
            group_id = str(group.get("group_id", "")).strip()
            for pair_row in group.get("dropped_pairs", []):
                cover = _attachment_pair_renderable_cover(group, pair_row)
                if cover is None:
                    continue
                pair_id = str(pair_row.get("pair_id", "")).strip()
                if not pair_id:
                    continue
                rendered_group_ids.add(group_id)
                rendered_pair_count += 1
                reason_text = _attachment_pair_reason_codes_to_zh(
                    pair_row.get("program_reason_codes", []),
                    pair_row=pair_row,
                )
                parent_id = int(pair_row.get("parent_id", 0) or 0)
                child_id = int(pair_row.get("child_id", 0) or 0)
                parent_label = str(pair_row.get("parent_label", "")).strip()
                child_label = str(pair_row.get("child_label", "")).strip()
                rename_advice_html = _render_rename_advice_block(pair_row)
                pair_cards.append(
                    f'<article class="pair-card" data-scene-id="{html.escape(scene_id)}" '
                    f'data-image-name="{html.escape(cover["image_name"])}" '
                    f'data-group-id="{html.escape(group_id)}" '
                    f'data-pair-id="{html.escape(pair_id)}" '
                    f'data-parent-id="{parent_id}" '
                    f'data-parent-label="{html.escape(parent_label)}" '
                    f'data-child-id="{child_id}" '
                    f'data-child-label="{html.escape(child_label)}" '
                    'data-deleted="false">'
                    '<div class="pair-visual">'
                    f'<img src="{html.escape(cover["data_url"])}" alt="{html.escape(cover["image_name"])}">'
                    f'<div class="pair-image-name">{html.escape(cover["image_stem"] or cover["image_name"])}</div>'
                    "</div>"
                    '<div class="pair-copy">'
                    f'<div class="pair-text"><strong>group</strong> {html.escape(group_id or "-")}</div>'
                    f'<div class="pair-text"><strong>scene id</strong> {html.escape(scene_id or "-")}</div>'
                    f'<div class="pair-text"><strong>attachment pair</strong> {html.escape(parent_label)}#{parent_id} -> '
                    f'{html.escape(child_label)}#{child_id}</div>'
                    f'<div class="pair-text"><strong>pair id</strong> {html.escape(pair_id)}</div>'
                    f'<div class="pair-text"><strong>筛除理由</strong> {html.escape(reason_text)}</div>'
                    f"{rename_advice_html}"
                    '<div class="pair-editor">'
                    '<label class="pair-editor-field">'
                    '<span class="pair-editor-label">Image ID</span>'
                    f'<input type="text" name="image_id" class="pair-name-input pair-image-id-input" value="{html.escape(cover["image_stem"] or Path(cover["image_name"]).stem)}">'
                    "</label>"
                    '<label class="pair-editor-field">'
                    '<span class="pair-editor-label">Parent Name</span>'
                    f'<input type="text" name="parent_surface_text" class="pair-name-input pair-name-input-parent" value="{html.escape(parent_label)}">'
                    "</label>"
                    '<label class="pair-editor-field">'
                    '<span class="pair-editor-label">Child Name</span>'
                    f'<input type="text" name="child_surface_text" class="pair-name-input pair-name-input-child" value="{html.escape(child_label)}">'
                    "</label>"
                    '<div class="pair-editor-actions">'
                    '<button type="button" class="pair-delete-toggle">Delete Card</button>'
                    "</div>"
                    "</div>"
                    "</div>"
                    "</article>"
                )
        if not pair_cards:
            continue
        rendered_group_count += len(rendered_group_ids)
        rendered_scene_count += 1
        if scene_id and scene_id not in seen_scene_ids:
            seen_scene_ids.add(scene_id)
            included_scene_ids.append(scene_id)
        scene_sections.append(
            '<section class="scene-card">'
            f'<h2>{html.escape(scene_id)} [{html.escape(scene.get("pipeline_outcome", ""))}]</h2>'
            f'<div class="pair-list">{"".join(pair_cards)}</div>'
            + "</section>"
        )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Attachment Pair Salvage Review</title>
  <style>
    body{{margin:0;background:linear-gradient(180deg,#efe7dc 0%,#f7f3ee 48%,#f4efe8 100%);color:#1f2937;font:14px/1.5 Georgia, 'Times New Roman', serif;}}
    .page{{max-width:1200px;margin:0 auto;padding:28px 20px 60px;}}
    h1,h2,h3{{margin:0 0 12px;color:#111827;}}
    .summary,.scene-card,.pair-card{{background:#fff;border:1px solid #ddd6c8;border-radius:16px;box-shadow:0 10px 24px rgba(15,23,42,.06);}}
    .summary{{padding:18px 20px;margin-bottom:22px;}}
    .summary-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:8px 14px;}}
    .summary-scenes{{margin-top:16px;padding-top:14px;border-top:1px solid #e7dfd1;}}
    .summary-actions{{display:flex;gap:12px;flex-wrap:wrap;margin-top:16px;}}
    .summary-action{{appearance:none;border:1px solid #c8a875;background:#a44a3f;color:#fff;border-radius:999px;padding:10px 16px;font:600 13px/1 Georgia, 'Times New Roman', serif;cursor:pointer;}}
    .scene-card{{padding:18px 20px;margin-bottom:24px;}}
    .pair-list{{display:grid;gap:14px;margin-top:16px;}}
    .pair-card{{padding:14px 16px;display:grid;grid-template-columns:360px 1fr;gap:18px;align-items:start;}}
    .pair-card[data-deleted="true"]{{opacity:.42;filter:saturate(.4);}}
    .pair-visual{{display:grid;gap:8px;}}
    .pair-visual img{{width:100%;height:264px;object-fit:cover;display:block;border-radius:12px;background:#d1d5db;}}
    .pair-image-name{{font-size:12px;color:#6b7280;letter-spacing:.02em;}}
    .pair-copy{{display:grid;gap:10px;align-content:start;}}
    .pair-text{{padding:10px 12px;border-radius:12px;background:#f8f6f1;border:1px solid #e7dfd1;}}
    .pair-rename-advice{{display:grid;gap:10px;}}
    .rename-advice-candidate{{padding:10px 12px;border-radius:10px;background:#fffdf8;border:1px solid #eadfce;display:grid;gap:6px;}}
    .rename-advice-index{{font-size:12px;font-weight:700;color:#7b5a2d;letter-spacing:.02em;text-transform:uppercase;}}
    .rename-advice-line{{font-size:13px;}}
    .rename-advice-unavailable{{font-size:13px;color:#6b7280;}}
    .pair-editor{{display:grid;gap:10px;padding:12px;border-radius:12px;background:#f4efe7;border:1px solid #ddcfbc;}}
    .pair-editor-field{{display:grid;gap:6px;}}
    .pair-editor-label{{font-size:12px;font-weight:700;color:#6b4f2a;letter-spacing:.02em;text-transform:uppercase;}}
    .pair-name-input{{width:100%;box-sizing:border-box;border:1px solid #cab89f;border-radius:10px;padding:10px 12px;background:#fffaf2;color:#111827;font:14px/1.4 Georgia, 'Times New Roman', serif;}}
    .pair-editor-actions{{display:flex;justify-content:flex-end;gap:8px;}}
    .pair-delete-toggle{{appearance:none;border:1px solid #b85b52;background:#fff;color:#8d2f28;border-radius:999px;padding:8px 12px;font:600 12px/1 Georgia, 'Times New Roman', serif;cursor:pointer;}}
    .empty-state{{padding:14px;border:1px dashed #c7bba7;border-radius:12px;background:#faf7f2;color:#6b7280;}}
    @media (max-width: 720px){{
      .pair-card{{grid-template-columns:1fr;}}
      .pair-visual img{{height:auto;max-height:220px;}}
    }}
  </style>
</head>
<body>
  <div class="page">
    <section class="summary">
      <h1>Attachment Pair Salvage Review</h1>
      <div class="summary-grid">
        <div><strong>scene count:</strong> {rendered_scene_count}</div>
        <div><strong>group count:</strong> {rendered_group_count}</div>
        <div><strong>pair count:</strong> {rendered_pair_count}</div>
      </div>
      <div class="summary-scenes"><strong>referability cache:</strong> {html.escape(referability_cache_output or "-")}</div>
      <div class="summary-scenes"><strong>scene review files:</strong> {html.escape(edited_html_target_display)}</div>
      <div class="summary-scenes"><strong>per-scene targets:</strong> {_render_simple_list(per_scene_output_lines)}</div>
      <div class="summary-scenes"><strong>pipeline input:</strong> run_pipeline reads the per-scene edited HTML files shown above, or a legacy neighboring edited.html only when no per-scene files exist. This salvage_review.html file is a batch summary only.</div>
      <div class="summary-scenes"><strong>included scenes:</strong> {_render_simple_list(included_scene_ids)}</div>
      {('<div class="summary-actions"><button type="button" id="export-edited-html" class="summary-action">Export Edited HTML</button></div>' if editable_scene_id is not None else '')}
    </section>
    {''.join(scene_sections) or '<div class="empty-state">No attachment pair salvage scenes recorded.</div>'}
  </div>
  <script>
    (() => {{
      if (!document.getElementById('export-edited-html')) {{
        return;
      }}
      const editedHtmlTargetName = {json.dumps(edited_html_filename, ensure_ascii=False)};

      function persistCardState() {{
        document.querySelectorAll('.pair-card').forEach((card) => {{
          card.setAttribute('data-deleted', card.dataset.deleted === 'true' ? 'true' : 'false');
          const imageIdInput = card.querySelector('input[name="image_id"]');
          if (imageIdInput) {{
            const imageId = imageIdInput.value.trim();
            const originalImageName = card.getAttribute('data-image-name') || '';
            const suffixMatch = originalImageName.match(/(\.[^./\\\\]+)$/);
            const nextImageName = imageId ? `${{imageId}}${{suffixMatch ? suffixMatch[1] : ''}}` : '';
            card.dataset.imageName = nextImageName;
            card.setAttribute('data-image-name', nextImageName);
            imageIdInput.setAttribute('value', imageIdInput.value);
          }}
          card.querySelectorAll('input.pair-name-input').forEach((input) => {{
            input.setAttribute('value', input.value);
          }});
        }});
      }}

      document.querySelectorAll('.pair-delete-toggle').forEach((button) => {{
        button.addEventListener('click', () => {{
          const card = button.closest('.pair-card');
          if (!card) {{
            return;
          }}
          const deleted = card.dataset.deleted === 'true';
          card.dataset.deleted = deleted ? 'false' : 'true';
          button.textContent = deleted ? 'Delete Card' : 'Restore Card';
        }});
      }});

      const exportButton = document.getElementById('export-edited-html');
      if (exportButton) {{
        exportButton.addEventListener('click', async () => {{
          persistCardState();
          const htmlText = '<!doctype html>\\n' + document.documentElement.outerHTML;
          if (typeof window.showSaveFilePicker === 'function') {{
            try {{
              const handle = await window.showSaveFilePicker({{
                id: 'attachment-pair-salvage-edited-html',
                suggestedName: editedHtmlTargetName,
                types: [
                  {{
                    description: 'HTML file',
                    accept: {{ 'text/html': ['.html'] }},
                  }},
                ],
              }});
              const writable = await handle.createWritable();
              await writable.write(htmlText);
              await writable.close();
              return;
            }} catch (error) {{
              if (error && error.name === 'AbortError') {{
                return;
              }}
            }}
          }}
          const blob = new Blob([htmlText], {{ type: 'text/html;charset=utf-8' }});
          const anchor = document.createElement('a');
          anchor.href = URL.createObjectURL(blob);
          anchor.download = editedHtmlTargetName;
          document.body.appendChild(anchor);
          anchor.click();
          anchor.remove();
          setTimeout(() => URL.revokeObjectURL(anchor.href), 1000);
        }});
      }}
    }})();
  </script>
</body>
</html>"""


def _select_non_attachment_group_representatives(
    *,
    client,
    model_name: str,
    scene_dir: Path,
    frames: list[dict[str, Any]],
    poses: dict[str, CameraPose] | None = None,
    max_group_count: int | None = None,
    max_accepted_frame_count: int | None = None,
    vlm_workers: int = 1,
    frame_review_getter: Callable[[dict[str, Any]], dict[str, Any] | None] | None = None,
    frame_review_batch_getter: Callable[[list[dict[str, Any]]], dict[str, Any] | None] | None = None,
    referability_entry_builder: Callable[[dict[str, Any], dict[str, Any]], dict[str, Any] | None] | None = None,
    debug_groups_out: list[dict[str, Any]] | None = None,
    frame_clarity_batch_size: int = FRAME_CLARITY_BATCH_SIZE,
    non_attachment_referability_shortlist: int = DEFAULT_NON_ATTACHMENT_REFERABILITY_SHORTLIST,
    non_attachment_clarity_min_score: int = DEFAULT_NON_ATTACHMENT_CLARITY_MIN_SCORE,
) -> list[dict[str, Any]]:
    if not frames:
        return []

    color_dir = scene_dir / "color"
    grouped_items = _build_visible_object_pose_merged_groups(
        frames=frames,
        poses=poses,
    )
    if max_group_count is not None:
        grouped_items = grouped_items[:max(0, int(max_group_count))]
    accepted_target: int | None = None
    if max_accepted_frame_count is not None:
        accepted_target = max(0, int(max_accepted_frame_count))
        if accepted_target <= 0:
            if debug_groups_out is not None:
                debug_groups_out.clear()
            return []
    scene_failed_signatures_seen: set[tuple[int, ...]] = set()

    def _select_group(
        item: tuple[int, dict[str, Any]]
    ) -> dict[str, Any]:
        group_index, group_doc = item
        group_key = tuple(int(obj_id) for obj_id in group_doc.get("visible_object_ids", []))
        group_frames = list(group_doc.get("frames", []))
        accepted: list[dict[str, Any]] = []
        attempts: list[dict[str, Any]] = []
        stopped_after_image_name: str | None = None
        early_stop_image_name: str | None = None
        early_stop_reason: str | None = None
        stop_reason = "exhausted_group_frames"
        sampled_frames, group_frame_stride = _sample_group_frames(group_frames)
        scored_frames = _score_group_frames_by_brisque(
            sampled_frames=sampled_frames,
            color_dir=color_dir,
        )
        ordered_scored_frames = _sort_group_frames_for_clarity_review(scored_frames)
        referable_object_ids_by_image_name: dict[str, list[int]] = {}
        accepted_frames_by_image_name: dict[str, dict[str, Any]] = {}
        accepted_image_names: set[str] = set()
        failed_signature_object_ids_by_image_name: dict[str, list[int]] = {}
        duplicate_failed_signature_image_names: set[str] = set()
        skipped_before_clarity_duplicate_failed_signature_image_names: set[str] = set()
        ordered_scored_frames_for_review: list[dict[str, Any]] = []
        for scored_frame in ordered_scored_frames:
            frame = scored_frame.get("frame")
            if not isinstance(frame, dict):
                ordered_scored_frames_for_review.append(scored_frame)
                continue
            image_name = str(scored_frame.get("image_name", "")).strip()
            failed_signature_candidate = tuple(
                _normalize_cached_object_ids(
                    frame.get("failed_signature_candidate_object_ids")
                )
            )
            if (
                failed_signature_candidate
                and failed_signature_candidate in scene_failed_signatures_seen
            ):
                skipped_before_clarity_duplicate_failed_signature_image_names.add(image_name)
                duplicate_failed_signature_image_names.add(image_name)
                failed_signature_object_ids_by_image_name[image_name] = list(
                    failed_signature_candidate
                )
                continue
            ordered_scored_frames_for_review.append(scored_frame)

        def _stop_on_reviewed_frame(
            scored_frame: dict[str, Any],
            reviewed_frame: dict[str, Any],
        ) -> str | None:
            frame = scored_frame.get("frame")
            if not isinstance(frame, dict):
                return None
            frame_info = reviewed_frame.get("frame_info", {})
            if not bool(frame_info.get("frame_usable", True)):
                return None
            image_name = str(scored_frame.get("image_name", "")).strip()
            referable_entry = None
            referable_object_ids: list[int] = []
            if referability_entry_builder is not None:
                referable_entry = referability_entry_builder(frame, reviewed_frame)
                if isinstance(referable_entry, dict):
                    referable_object_ids = _normalize_cached_object_ids(
                        referable_entry.get("referable_object_ids")
                    )
            referable_object_ids_by_image_name[image_name] = list(referable_object_ids)
            accepted_for_group = bool(
                referability_entry_builder is None
                or len(referable_object_ids) >= NON_ATTACHMENT_GROUP_MIN_REFERABLE_OBJECT_COUNT
            )
            if not accepted_for_group:
                failed_signature = _failed_referability_object_id_signature(
                    referable_entry,
                    bbox_in_frame_ratio_min=REFERABLE_BBOX_IN_FRAME_RATIO_MIN,
                    projected_area_px_min=QUESTION_REVIEW_CROP_MIN_PROJECTED_AREA_PX,
                )
                if failed_signature:
                    failed_signature_object_ids_by_image_name[image_name] = list(failed_signature)
                    scene_failed_signatures_seen.add(failed_signature)
                return None
            accepted_image_names.add(image_name)
            accepted_frame = dict(reviewed_frame)
            if isinstance(referable_entry, dict):
                accepted_frame["_referability_entry"] = referable_entry
                accepted_frame["referable_object_ids"] = referable_object_ids
            accepted_frames_by_image_name[image_name] = accepted_frame
            return "accepted_frame_has_min_referable_objects"

        review_result = _review_group_frames_until_stop(
            ordered_scored_frames=ordered_scored_frames_for_review,
            client=client,
            model_name=model_name,
            color_dir=color_dir,
            frame_review_getter=frame_review_getter,
            frame_review_batch_getter=frame_review_batch_getter,
            frame_clarity_batch_size=frame_clarity_batch_size,
            stop_on_reviewed_frame=_stop_on_reviewed_frame,
        )
        reviewed_by_image_name = review_result["reviewed_by_image_name"]
        reviewed_name_set = set(review_result["reviewed_image_names_in_order"])
        early_stop_image_name = review_result["early_stop_image_name"]
        early_stop_reason = review_result["early_stop_reason"]
        clarity_batch_image_names = [
            str(item.get("image_name", "")).strip()
            for item in ordered_scored_frames_for_review
            if str(item.get("image_name", "")).strip()
        ]
        attempts_by_image_name: dict[str, dict[str, Any]] = {}
        for scored_frame in scored_frames:
            frame = scored_frame.get("frame")
            if not isinstance(frame, dict):
                continue
            image_name = str(scored_frame.get("image_name", "")).strip()
            selector_score = int(
                frame.get("selector_score", frame.get("score", frame.get("n_visible", 0))) or 0
            )
            attempt = {
                "image_name": image_name,
                "selector_score": selector_score,
                "review_status": "not_reviewed",
                "frame_usable": False,
                "clarity_score": None,
                "frame_quality_reason": None,
                "frame_selection_score": None,
                "referable_object_count": 0,
                "referable_object_ids": [],
                "accepted_for_group": False,
                "failed_signature_object_ids": [],
                "duplicate_failed_signature_skip": False,
                "stop_after_this_frame": False,
                "brisque_score": scored_frame.get("brisque_score"),
                "brisque_input_width": scored_frame.get("brisque_input_width"),
                "brisque_input_height": scored_frame.get("brisque_input_height"),
            }
            reviewed_frame = reviewed_by_image_name.get(image_name)
            if image_name in skipped_before_clarity_duplicate_failed_signature_image_names:
                attempt["review_status"] = (
                    "skipped_before_clarity_duplicate_failed_signature"
                )
                attempt["failed_signature_object_ids"] = list(
                    failed_signature_object_ids_by_image_name.get(image_name, [])
                )
                attempt["duplicate_failed_signature_skip"] = True
            elif image_name in reviewed_name_set:
                if not isinstance(reviewed_frame, dict):
                    attempt["review_status"] = "review_failed_or_missing_image"
                else:
                    frame_info = reviewed_frame.get("frame_info", {})
                    frame_usable = bool(frame_info.get("frame_usable", True))
                    clarity_score = int(frame_info.get("clarity_score", 0) or 0)
                    attempt["frame_usable"] = frame_usable
                    attempt["clarity_score"] = clarity_score
                    attempt["frame_quality_reason"] = str(frame_info.get("reason", "")).strip() or None
                    attempt["frame_selection_score"] = int(reviewed_frame.get("frame_selection_score", 0) or 0)
                    if not frame_usable:
                        attempt["review_status"] = "frame_not_usable"
                    else:
                        referable_object_ids = referable_object_ids_by_image_name.get(image_name, [])
                        accepted_for_group = image_name in accepted_image_names
                        attempt["referable_object_count"] = len(referable_object_ids)
                        attempt["referable_object_ids"] = list(referable_object_ids)
                        attempt["accepted_for_group"] = accepted_for_group
                        attempt["failed_signature_object_ids"] = list(
                            failed_signature_object_ids_by_image_name.get(image_name, [])
                        )
                        if image_name in duplicate_failed_signature_image_names:
                            attempt["duplicate_failed_signature_skip"] = True
                            attempt["review_status"] = (
                                "frame_usable_duplicate_failed_signature_skip"
                            )
                        else:
                            attempt["review_status"] = (
                                "accepted_for_group"
                                if accepted_for_group
                                else "frame_usable_not_referable"
                            )
                        if accepted_for_group and image_name == early_stop_image_name:
                            attempt["stop_after_this_frame"] = True
            elif early_stop_image_name:
                attempt["review_status"] = "skipped_after_early_stop"
            attempts.append(attempt)
            attempts_by_image_name[image_name] = attempt

        clarity_eligible_image_names = [
            str(item.get("image_name", "")).strip()
            for item in ordered_scored_frames_for_review
            if isinstance(reviewed_by_image_name.get(str(item.get("image_name", "")).strip()), dict)
            and bool(
                reviewed_by_image_name[str(item.get("image_name", "")).strip()]
                .get("frame_info", {})
                .get("frame_usable", True)
            )
        ]
        referability_shortlist_image_names = list(clarity_batch_image_names)

        if early_stop_image_name:
            stopped_after_image_name = early_stop_image_name
            stop_reason = str(early_stop_reason or "accepted_frame_has_min_referable_objects")
            accepted_frame = accepted_frames_by_image_name.get(early_stop_image_name)
            if isinstance(accepted_frame, dict):
                accepted.append(accepted_frame)
        elif not clarity_eligible_image_names:
            stop_reason = "no_usable_frames"
        else:
            stop_reason = "exhausted_usable_frames"
        return {
            "group_index": int(group_index),
            "group_key_visible_object_ids": [int(obj_id) for obj_id in group_key],
            "candidate_frame_image_names": [
                str(frame.get("image_name", "")).strip()
                for frame in group_frames
            ],
            "sampled_frame_image_names": [
                str(frame.get("image_name", "")).strip()
                for frame in sampled_frames
            ],
            "sampled_frames": [
                {
                    "image_name": str(item.get("image_name", "")).strip(),
                    "brisque_score": item.get("brisque_score"),
                    "brisque_input_width": item.get("brisque_input_width"),
                    "brisque_input_height": item.get("brisque_input_height"),
                }
                for item in scored_frames
            ],
            "group_frame_stride": group_frame_stride,
            "clarity_batch_image_names": clarity_batch_image_names,
            "clarity_eligible_image_names": clarity_eligible_image_names,
            "brisque_sorted_frame_image_names": clarity_batch_image_names,
            "referability_shortlist_image_names": referability_shortlist_image_names,
            "attempts": attempts,
            "stopped_after_image_name": stopped_after_image_name,
            "early_stop_image_name": early_stop_image_name,
            "early_stop_reason": early_stop_reason,
            "stop_reason": stop_reason,
            "_accepted_frames": accepted,
        }

    selected_groups: list[dict[str, Any]] = []
    next_group_index = 0
    accepted_frame_count = 0
    while next_group_index < len(grouped_items):
        remaining_target = None
        if accepted_target is not None:
            remaining_target = accepted_target - accepted_frame_count
            if remaining_target <= 0:
                break
        batch_size = (
            max(1, remaining_target)
            if remaining_target is not None
            else len(grouped_items) - next_group_index
        )
        batch_items = list(
            enumerate(
                grouped_items[next_group_index : next_group_index + batch_size],
                start=next_group_index,
            )
        )
        # Scene-level failed-signature reuse needs deterministic group order.
        batch_results = [_select_group(item) for item in batch_items]
        selected_groups.extend(
            doc for doc in batch_results
            if isinstance(doc, dict)
        )
        accepted_frame_count += sum(
            len(
                [
                    frame for frame in doc.get("_accepted_frames", [])
                    if isinstance(frame, dict)
                ]
            )
            for doc in batch_results
            if isinstance(doc, dict)
        )
        next_group_index += len(batch_items)
    selected_frames: list[dict[str, Any]] = []
    group_debug_docs = sorted(
        selected_groups,
        key=lambda doc: int(doc.get("group_index", 0)),
    )
    for doc in group_debug_docs:
        accepted_frames = [
            frame for frame in doc.pop("_accepted_frames", [])
            if isinstance(frame, dict)
        ]
        selected_frames.extend(accepted_frames)
        if debug_groups_out is None:
            continue
        accepted_image_names = [
            str(frame.get("image_name", "")).strip()
            for frame in accepted_frames
        ]
        best_accepted_clarity = max(
            [
                int(frame.get("frame_info", {}).get("clarity_score", 0) or 0)
                for frame in accepted_frames
            ],
            default=None,
        )
        any_usable_frame = any(
            bool(attempt.get("frame_usable", False))
            for attempt in doc.get("attempts", [])
        )
        debug_groups_out.append(
            {
                "group_index": int(doc.get("group_index", 0)),
                "non_attachment_bbox_in_frame_ratio_min": REFERABLE_BBOX_IN_FRAME_RATIO_MIN,
                "non_attachment_min_referable_object_count": NON_ATTACHMENT_GROUP_MIN_REFERABLE_OBJECT_COUNT,
                "group_key_visible_object_ids": list(doc.get("group_key_visible_object_ids", [])),
                "candidate_frame_image_names": list(doc.get("candidate_frame_image_names", [])),
                "sampled_frame_image_names": list(doc.get("sampled_frame_image_names", [])),
                "sampled_frames": list(doc.get("sampled_frames", [])),
                "group_frame_stride": int(doc.get("group_frame_stride", 1)),
                "clarity_batch_image_names": list(doc.get("clarity_batch_image_names", [])),
                "clarity_eligible_image_names": list(doc.get("clarity_eligible_image_names", [])),
                "brisque_sorted_frame_image_names": list(doc.get("brisque_sorted_frame_image_names", [])),
                "referability_shortlist_image_names": list(doc.get("referability_shortlist_image_names", [])),
                "attempts": list(doc.get("attempts", [])),
                "accepted_frame_image_names": accepted_image_names,
                "accepted_frame_count": len(accepted_image_names),
                "best_accepted_clarity": best_accepted_clarity,
                "stopped_after_image_name": doc.get("stopped_after_image_name"),
                "early_stop_image_name": doc.get("early_stop_image_name"),
                "early_stop_reason": doc.get("early_stop_reason"),
                "stop_reason": str(doc.get("stop_reason", "exhausted_group_frames")),
                "group_exhausted_without_usable_frame": not any_usable_frame,
                "group_exhausted_without_referable_frame": len(accepted_image_names) == 0,
            }
        )
    return selected_frames


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
    attachment_selector_pair_count: object | None = None,
    attachment_selector_viewpoint_exempt: object | None = None,
) -> dict[str, Any]:
    updated = dict(entry)
    if attachment_selector_pair_count is not None:
        updated["attachment_pair_ge_50_count"] = int(attachment_selector_pair_count or 0)
    if attachment_selector_viewpoint_exempt is not None:
        updated["attachment_viewpoint_exempt"] = bool(attachment_selector_viewpoint_exempt)
    attachment_pairs = _build_attachment_referable_pairs(
        attachment_graph,
        updated.get("attachment_referable_object_ids"),
    )
    updated["attachment_referable_pairs"] = attachment_pairs
    updated["attachment_referable_pair_count"] = len(updated["attachment_referable_pairs"])
    updated["attachment_view_group_id"] = attachment_view_group_id
    updated["final_selection_rank"] = int(final_selection_rank)
    return _apply_attachment_layer_payloads(
        updated,
        attachment_pairs=attachment_pairs,
        selector_pair_count=attachment_selector_pair_count,
        selector_viewpoint_exempt=attachment_selector_viewpoint_exempt,
        final_selection_rank=final_selection_rank,
    )


def _frame_decision(
    client,
    model: str,
    image: np.ndarray,
    image_b64: str | None = None,
) -> dict[str, Any]:
    full_b64 = str(image_b64 or "") or _image_to_base64(image)
    default = {
        "frame_usable": True,
        "reason": "frame_usable_parse_fallback",
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


def _object_review_decision_batch(
    client,
    model: str,
    image_b64: str,
    crop_b64_list: list[str],
    labels: list[str],
) -> list[tuple[str, str]]:
    """Evaluate multiple object crops in one VLM call."""
    batch_size = min(len(crop_b64_list), len(labels))
    if batch_size <= 0:
        return []
    content: list[dict[str, Any]] = [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
    ]
    for crop_b64 in crop_b64_list[:batch_size]:
        content.append(
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{crop_b64}"}}
        )
    content.append(
        {"type": "text", "text": _object_review_batch_prompt(labels[:batch_size])}
    )
    parsed, raw_text = _call_vlm_json(
        client,
        model,
        content,
        default={"results": []},
        max_tokens=min(512, 128 * max(1, batch_size)),
    )
    fallback_raw = raw_text or ""
    results: list[tuple[str, str]] = [
        (OBJECT_STATUS_UNSURE, fallback_raw)
        for _ in range(batch_size)
    ]
    raw_items = parsed.get("results") if isinstance(parsed, dict) else None
    if not isinstance(raw_items, list):
        return results
    seen_indices: set[int] = set()
    for raw_item in raw_items:
        if not isinstance(raw_item, dict):
            continue
        raw_index = raw_item.get("index")
        if isinstance(raw_index, bool) or raw_index is None:
            continue
        try:
            item_index = int(raw_index) - 1
        except (TypeError, ValueError):
            continue
        if item_index < 0 or item_index >= batch_size or item_index in seen_indices:
            continue
        seen_indices.add(item_index)
        status = _normalize_object_review_status(raw_item.get("status")) or OBJECT_STATUS_UNSURE
        results[item_index] = (status, json.dumps(raw_item, ensure_ascii=False))
    return results


def _full_frame_label_vlm_review(
    *,
    client,
    model: str,
    image_b64: str,
    label: str,
) -> dict[str, Any]:
    batch_reviews = _full_frame_label_vlm_review_batch(
        client=client,
        model=model,
        image_b64=image_b64,
        labels=[label],
    )
    if batch_reviews:
        return batch_reviews[0]
    normalized_label = str(label or "").strip().lower()
    return {
        "backend": "vlm",
        "label": normalized_label,
        "status": LABEL_STATUS_UNSURE,
        "count": None,
        "reason": "missing_label" if not normalized_label else "parse_fallback",
        "raw_response": None,
    }


def _full_frame_label_vlm_review_batch(
    *,
    client,
    model: str,
    image_b64: str,
    labels: list[str],
) -> list[dict[str, Any]]:
    """Evaluate multiple labels in a single VLM call."""

    def _default_review(normalized_label: str, *, reason: str) -> dict[str, Any]:
        return {
            "backend": "vlm",
            "label": normalized_label,
            "status": LABEL_STATUS_UNSURE,
            "count": None,
            "reason": reason,
            "raw_response": None,
        }

    normalized_labels = [str(label or "").strip().lower() for label in labels]
    reviews = [
        _default_review(normalized_label, reason="missing_label" if not normalized_label else "pending")
        for normalized_label in normalized_labels
    ]
    expected_labels = [normalized_label for normalized_label in normalized_labels if normalized_label]
    if not expected_labels:
        return reviews

    parsed, raw_text = _call_vlm_json(
        client,
        model,
        [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
            {"type": "text", "text": _full_frame_label_count_batch_prompt(expected_labels)},
        ],
        default={"results": []},
        max_tokens=128 * max(1, len(expected_labels)),
    )

    raw_items = parsed.get("results") if isinstance(parsed, dict) else None
    parsed_by_label: dict[str, list[dict[str, Any]]] = {}
    if isinstance(raw_items, list):
        for raw_item in raw_items:
            if not isinstance(raw_item, dict):
                continue
            item_label = str(raw_item.get("label", "")).strip().lower()
            if not item_label:
                continue
            parsed_by_label.setdefault(item_label, []).append(raw_item)

    consumed_by_label: dict[str, int] = {}
    for index, normalized_label in enumerate(normalized_labels):
        if not normalized_label:
            continue
        label_items = parsed_by_label.get(normalized_label, [])
        item_index = consumed_by_label.get(normalized_label, 0)
        if item_index >= len(label_items):
            reviews[index]["reason"] = "parse_fallback"
            reviews[index]["raw_response"] = raw_text or None
            continue
        consumed_by_label[normalized_label] = item_index + 1

        parsed_item = label_items[item_index]
        count = _normalize_full_frame_label_count(
            parsed_item.get("count", parsed_item.get("visible_count", parsed_item.get("label_count")))
        )
        status = (
            _normalize_full_frame_label_status(parsed_item.get("status"), count=count)
            or LABEL_STATUS_UNSURE
        )
        reason = str(parsed_item.get("reason", "")).strip() or "parse_fallback"
        reviews[index].update(
            {
                "status": status,
                "count": count,
                "reason": reason,
                "raw_response": raw_text or None,
            }
        )

    return reviews


def _out_of_frame_label_vlm_review(
    *,
    client,
    model: str,
    image_b64: str,
    label: str,
) -> dict[str, Any]:
    normalized_label = str(label or "").strip().lower()
    review = {
        "status": OUT_OF_FRAME_REVIEW_STATUS_UNSURE,
        "raw_response": None,
    }
    if not normalized_label:
        return review

    default = {
        "status": OUT_OF_FRAME_REVIEW_STATUS_UNSURE,
    }
    parsed, raw_text = _call_vlm_json(
        client,
        model,
        [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
            {"type": "text", "text": _full_frame_out_of_frame_label_prompt(normalized_label)},
        ],
        default=default,
        max_tokens=128,
    )
    status = (
        _normalize_out_of_frame_review_status(parsed.get("status"))
        or OUT_OF_FRAME_REVIEW_STATUS_UNSURE
    )
    review.update(
        {
            "status": status,
            "raw_response": raw_text or None,
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


def _object_bbox_projection_points(obj: dict[str, Any]) -> list[np.ndarray]:
    bbox_min = np.asarray(obj.get("bbox_min", []), dtype=np.float64)
    bbox_max = np.asarray(obj.get("bbox_max", []), dtype=np.float64)
    if bbox_min.shape != (3,) or bbox_max.shape != (3,):
        return []
    points: list[np.ndarray] = []
    for x in [bbox_min[0], bbox_max[0]]:
        for y in [bbox_min[1], bbox_max[1]]:
            for z in [bbox_min[2], bbox_max[2]]:
                points.append(np.array([x, y, z], dtype=np.float64))
    points.append((bbox_min + bbox_max) / 2.0)
    return points


def _projected_bbox_outside_distance_px(
    obj: dict[str, Any],
    camera_pose: CameraPose,
    color_intrinsics: CameraIntrinsics,
) -> float:
    projected: list[tuple[float, float]] = []
    for point in _object_bbox_projection_points(obj):
        uv, depth = project_to_image(point, camera_pose, color_intrinsics)
        if uv is None or depth <= 0:
            continue
        projected.append((float(uv[0]), float(uv[1])))
    if not projected:
        return float(max(color_intrinsics.width, color_intrinsics.height) * 4)

    us = [item[0] for item in projected]
    vs = [item[1] for item in projected]
    u_min = float(min(us) - 5.0)
    u_max = float(max(us) + 5.0)
    v_min = float(min(vs) - 5.0)
    v_max = float(max(vs) + 5.0)

    if u_max < 0.0:
        dx = float(-u_max)
    elif u_min > float(color_intrinsics.width):
        dx = float(u_min - float(color_intrinsics.width))
    else:
        dx = 0.0

    if v_max < 0.0:
        dy = float(-v_max)
    elif v_min > float(color_intrinsics.height):
        dy = float(v_min - float(color_intrinsics.height))
    else:
        dy = 0.0

    if dx > 0.0 and dy > 0.0:
        return float(np.hypot(dx, dy))
    return float(max(dx, dy))


def _evaluate_out_of_frame_geometry_for_object(
    *,
    obj: dict[str, Any],
    visibility_meta: dict[str, Any] | None,
    camera_pose: CameraPose | None,
    color_intrinsics: CameraIntrinsics | None,
    instance_mesh_data_getter: Callable[[int], InstanceMeshData] | None = None,
) -> dict[str, Any]:
    obj_id = int(obj.get("id", -1))
    projected_area_px = _safe_float(
        (visibility_meta or {}).get("projected_area_px"),
        default=0.0,
    )
    in_frame_ratio = _safe_float(
        (visibility_meta or {}).get("bbox_in_frame_ratio"),
        default=0.0,
    )
    sample_count_available = False
    in_frame_sample_count = 0

    if (
        obj_id >= 0
        and camera_pose is not None
        and color_intrinsics is not None
        and callable(instance_mesh_data_getter)
    ):
        try:
            instance_mesh_data = instance_mesh_data_getter(
                REFERABILITY_MESH_RAY_STAGE1_BASE_SAMPLE_COUNT
            )
        except Exception:
            instance_mesh_data = None
        sample_points = _instance_surface_samples(instance_mesh_data, obj_id)
        if len(sample_points) > 0:
            sample_count_available = True
            in_frame_points, _unused_triangles, _unused_barycentrics = _in_frame_surface_sample_subset(
                sample_points,
                camera_pose,
                color_intrinsics,
            )
            in_frame_sample_count = int(len(in_frame_points))
            in_frame_ratio = float(in_frame_sample_count / len(sample_points))

    outside_distance_px = 0.0
    if camera_pose is not None and color_intrinsics is not None:
        outside_distance_px = _projected_bbox_outside_distance_px(
            obj,
            camera_pose,
            color_intrinsics,
        )

    is_out_of_frame = (
        (sample_count_available and in_frame_sample_count == 0)
        or in_frame_ratio <= 0.0
    )
    return {
        "obj_id": obj_id,
        "label": str(obj.get("label", "")).strip().lower(),
        "projected_area_px": projected_area_px,
        "in_frame_ratio": float(in_frame_ratio),
        "in_frame_sample_count": int(in_frame_sample_count),
        "outside_distance_px": float(outside_distance_px),
        "is_out_of_frame": bool(is_out_of_frame),
    }


def _build_out_of_frame_label_candidates(
    *,
    scene_objects: list[dict[str, Any]],
    objects_by_id: dict[int, dict[str, Any]],
    visibility_by_obj_id: dict[int, dict[str, Any]],
    camera_pose: CameraPose | None,
    color_intrinsics: CameraIntrinsics | None,
    instance_mesh_data_getter: Callable[[int], InstanceMeshData] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, list[int]]]:
    alias_group_to_labels: dict[str, set[str]] = defaultdict(set)
    label_to_scene_object_ids: dict[str, list[int]] = defaultdict(list)
    label_to_alias_groups: dict[str, set[str]] = defaultdict(set)

    for raw_obj in scene_objects:
        try:
            obj_id = int(raw_obj.get("id", -1))
        except (TypeError, ValueError):
            continue
        obj = objects_by_id.get(obj_id, raw_obj)
        label = str(obj.get("label", "")).strip().lower()
        alias_group = str(obj.get("alias_group", "")).strip().lower()
        if not label or label in EXCLUDED_LABELS:
            continue
        label_to_scene_object_ids[label].append(obj_id)
        label_to_alias_groups[label].add(alias_group)
        if alias_group:
            alias_group_to_labels[alias_group].add(label)

    unique_alias_group_to_label = {
        alias_group: next(iter(labels))
        for alias_group, labels in alias_group_to_labels.items()
        if len(labels) == 1
    }

    candidates: list[dict[str, Any]] = []
    out_of_frame_label_to_object_ids: dict[str, list[int]] = {}
    for label, obj_ids in sorted(label_to_scene_object_ids.items()):
        normalized_obj_ids = sorted(set(int(obj_id) for obj_id in obj_ids))
        alias_groups = label_to_alias_groups.get(label, set())
        if not normalized_obj_ids or not alias_groups or "" in alias_groups:
            continue
        if any(unique_alias_group_to_label.get(alias_group) != label for alias_group in alias_groups):
            continue

        object_geometries: list[dict[str, Any]] = []
        all_out_of_frame = True
        for obj_id in normalized_obj_ids:
            obj = objects_by_id.get(int(obj_id))
            if obj is None:
                all_out_of_frame = False
                break
            geometry = _evaluate_out_of_frame_geometry_for_object(
                obj=obj,
                visibility_meta=visibility_by_obj_id.get(int(obj_id)),
                camera_pose=camera_pose,
                color_intrinsics=color_intrinsics,
                instance_mesh_data_getter=instance_mesh_data_getter,
            )
            object_geometries.append(geometry)
            if not bool(geometry.get("is_out_of_frame", False)):
                all_out_of_frame = False
                break
        if not all_out_of_frame or not object_geometries:
            continue

        representative = max(
            object_geometries,
            key=lambda item: (
                float(item.get("projected_area_px", 0.0) or 0.0),
                float(item.get("outside_distance_px", 0.0) or 0.0),
                -int(item.get("obj_id", 0) or 0),
            ),
        )
        candidates.append(
            {
                "label": label,
                "object_ids": normalized_obj_ids,
                "representative": representative,
            }
        )
        out_of_frame_label_to_object_ids[label] = normalized_obj_ids

    candidates.sort(
        key=lambda item: (
            -float(item["representative"].get("projected_area_px", 0.0) or 0.0),
            -float(item["representative"].get("outside_distance_px", 0.0) or 0.0),
            int(item["representative"].get("obj_id", 0) or 0),
            str(item.get("label", "")),
        )
    )
    return candidates, out_of_frame_label_to_object_ids


def _review_out_of_frame_label_candidates(
    *,
    client,
    model_name: str,
    image: np.ndarray,
    image_b64: str | None = None,
    scene_objects: list[dict[str, Any]],
    objects_by_id: dict[int, dict[str, Any]],
    visibility_by_obj_id: dict[int, dict[str, Any]],
    camera_pose: CameraPose | None,
    color_intrinsics: CameraIntrinsics | None,
    instance_mesh_data_getter: Callable[[int], InstanceMeshData] | None = None,
) -> dict[str, Any]:
    candidates, label_to_object_ids = _build_out_of_frame_label_candidates(
        scene_objects=scene_objects,
        objects_by_id=objects_by_id,
        visibility_by_obj_id=visibility_by_obj_id,
        camera_pose=camera_pose,
        color_intrinsics=color_intrinsics,
        instance_mesh_data_getter=instance_mesh_data_getter,
    )
    if not candidates:
        return {
            "out_of_frame_label_reviews": [],
            "out_of_frame_not_visible_labels": [],
            "out_of_frame_label_to_object_ids": {},
            "out_of_frame_vlm_early_stop": False,
        }

    encoded_image_b64 = str(image_b64 or "") or _image_to_base64(image)
    pending_reviews: list[dict[str, Any]] = []
    not_visible_labels: list[str] = []
    early_stop = False

    for candidate in candidates:
        label = str(candidate.get("label", "")).strip().lower()
        if not label:
            continue
        vlm_review = _out_of_frame_label_vlm_review(
            client=client,
            model=model_name,
            image_b64=encoded_image_b64,
            label=label,
        )
        status = (
            _normalize_out_of_frame_review_status(vlm_review.get("status"))
            or OUT_OF_FRAME_REVIEW_STATUS_UNSURE
        )
        pending_reviews.append(
            {
                "label": label,
                "status": status,
                "raw_response": vlm_review.get("raw_response"),
            }
        )
        if status == OUT_OF_FRAME_REVIEW_STATUS_NOT_VISIBLE:
            not_visible_labels = [label]
            early_stop = True
            return {
                "out_of_frame_label_reviews": pending_reviews,
                "out_of_frame_not_visible_labels": not_visible_labels,
                "out_of_frame_label_to_object_ids": {
                    str(candidate_label): [int(obj_id) for obj_id in obj_ids]
                    for candidate_label, obj_ids in sorted(label_to_object_ids.items())
                },
                "out_of_frame_vlm_early_stop": early_stop,
            }

    return {
        "out_of_frame_label_reviews": [],
        "out_of_frame_not_visible_labels": [],
        "out_of_frame_label_to_object_ids": {},
        "out_of_frame_vlm_early_stop": False,
    }


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
    _ = (objects, depth_image, depth_intrinsics)
    if not selector_ids:
        return [], "mesh_ray_refined"
    if color_intrinsics is None:
        raise MeshRayRequiredError(
            "mesh-ray candidate refinement requires color intrinsics"
        )
    if not callable(ray_caster_getter):
        raise MeshRayRequiredError(
            "mesh-ray candidate refinement requires ray_caster_getter"
        )
    if not callable(instance_mesh_data_getter):
        raise MeshRayRequiredError(
            "mesh-ray candidate refinement requires instance_mesh_data_getter"
        )

    try:
        ray_caster = ray_caster_getter()
    except Exception as exc:
        raise MeshRayRequiredError(
            "mesh-ray candidate refinement failed while loading ray caster"
        ) from exc
    if ray_caster is None:
        raise MeshRayRequiredError(
            "mesh-ray candidate refinement requires a non-null ray caster"
        )

    try:
        stage1_instance_mesh_data = instance_mesh_data_getter(
            REFERABILITY_MESH_RAY_STAGE1_BASE_SAMPLE_COUNT,
        )
    except Exception as exc:
        raise MeshRayRequiredError(
            "mesh-ray candidate refinement failed while loading stage1 mesh samples"
        ) from exc
    if stage1_instance_mesh_data is None:
        raise MeshRayRequiredError(
            "mesh-ray candidate refinement requires stage1 mesh samples"
        )

    stage2_instance_mesh_data: InstanceMeshData | None = None
    mesh_ray_refined: list[int] = []
    for obj_id in selector_ids:
        try:
            stage1 = _evaluate_crop_unique_mesh_ray_stage(
                obj_id=int(obj_id),
                camera_pose=camera_pose,
                color_intrinsics=color_intrinsics,
                ray_caster=ray_caster,
                instance_mesh_data=stage1_instance_mesh_data,
                base_sample_count=REFERABILITY_MESH_RAY_STAGE1_BASE_SAMPLE_COUNT,
            )
        except Exception as exc:
            raise MeshRayRequiredError(
                f"mesh-ray stage1 evaluation failed for object {int(obj_id)}"
            ) from exc
        if _ray_visibility_stage_passes(stage1):
            mesh_ray_refined.append(int(obj_id))
            continue
        if stage2_instance_mesh_data is None:
            try:
                stage2_instance_mesh_data = instance_mesh_data_getter(
                    REFERABILITY_MESH_RAY_STAGE2_BASE_SAMPLE_COUNT,
                )
            except Exception as exc:
                raise MeshRayRequiredError(
                    "mesh-ray candidate refinement failed while loading stage2 mesh samples"
                ) from exc
            if stage2_instance_mesh_data is None:
                raise MeshRayRequiredError(
                    "mesh-ray candidate refinement requires stage2 mesh samples"
                )
        try:
            stage2 = _evaluate_crop_unique_mesh_ray_stage(
                obj_id=int(obj_id),
                camera_pose=camera_pose,
                color_intrinsics=color_intrinsics,
                ray_caster=ray_caster,
                instance_mesh_data=stage2_instance_mesh_data,
                base_sample_count=REFERABILITY_MESH_RAY_STAGE2_BASE_SAMPLE_COUNT,
            )
        except Exception as exc:
            raise MeshRayRequiredError(
                f"mesh-ray stage2 evaluation failed for object {int(obj_id)}"
            ) from exc
        if _ray_visibility_stage_passes(stage2):
            mesh_ray_refined.append(int(obj_id))
    return sorted(set(int(obj_id) for obj_id in mesh_ray_refined)), "mesh_ray_refined"


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
            if candidate_visibility_source == "mesh_ray_refined":
                candidate_rejection_reasons.append("mesh_ray_not_visible")
            elif candidate_visibility_source == "mesh_ray_depth_refined":
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
    preloaded_geometry: Any | None = None,
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
                preloaded_geometry=preloaded_geometry,
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
    image_b64: str | None = None,
    visibility_by_obj_id: dict[int, dict[str, Any]] | None = None,
    out_of_frame_review: dict[str, Any] | None = None,
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
    computed_visibility_by_obj_id = visibility_by_obj_id
    if computed_visibility_by_obj_id is None:
        computed_visibility_by_obj_id = compute_referability_object_visibility(
            scene_objects,
            camera_pose,
            color_intrinsics,
        )
    visibility_audit_by_object_id = _build_visibility_audit_by_object_id(
        scene_objects,
        objects_by_id,
        computed_visibility_by_obj_id,
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
    out_of_frame_label_reviews: list[dict[str, Any]] = []
    out_of_frame_not_visible_labels: list[str] = []
    out_of_frame_label_to_object_ids: dict[str, list[int]] = {}
    out_of_frame_vlm_early_stop = False
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
            crop_entry = _build_object_review_crop(
                image,
                computed_visibility_by_obj_id.get(int(obj_id)),
            )
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

        pending_object_review_batches = _chunk_list(
            pending_object_review_jobs,
            OBJECT_REVIEW_BATCH_SIZE,
        )

        def _run_object_review_batch(
            batch: list[tuple[int, str, str, dict[str, Any]]],
        ) -> list[tuple[int, dict[str, Any]]]:
            if not batch:
                return []
            statuses = _object_review_decision_batch(
                client,
                model_name,
                str(image_b64 or ""),
                [str(crop_b64 or "") for _obj_id, _label, crop_b64, _review in batch],
                [str(label or "") for _obj_id, label, _crop_b64, _review in batch],
            )
            batch_results: list[tuple[int, dict[str, Any]]] = []
            for (obj_id, _label, _crop_b64, review), (status, raw_response) in zip(batch, statuses):
                updated_review = dict(review)
                updated_review["vlm_status"] = status
                updated_review["raw_response"] = raw_response or None
                batch_results.append((int(obj_id), updated_review))
            return batch_results

        for batch_results in _run_in_thread_pool(
            pending_object_review_batches,
            _run_object_review_batch,
            max_workers=vlm_workers,
        ):
            for obj_id, review in batch_results:
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
            if image_b64 is None:
                image_b64 = _image_to_base64(image)

            def _build_full_frame_label_review_payload(
                *,
                label: str,
                obj_id: int,
                vlm_review: dict[str, Any],
            ) -> dict[str, Any]:
                count = _normalize_full_frame_label_count(vlm_review.get("count"))
                status = (
                    _normalize_full_frame_label_status(vlm_review.get("status"), count=count)
                    or LABEL_STATUS_UNSURE
                )
                if count is None:
                    count = _label_status_count(status)
                return {
                    "label": str(label),
                    "status": status,
                    "count": count,
                    "crop_status": crop_label_statuses.get(label),
                    "crop_clear_count": crop_label_counts.get(label),
                    "crop_referable_object_id": int(obj_id),
                    "backend": str(vlm_review.get("backend", "vlm") or "vlm"),
                    "reason": str(vlm_review.get("reason", "")).strip() or None,
                    "raw_detection_count": int(vlm_review.get("raw_detection_count", 0) or 0),
                    "raw_detections": [
                        dict(item)
                        for item in vlm_review.get("raw_detections", [])
                        if isinstance(item, dict)
                    ],
                    "raw_response": vlm_review.get("raw_response"),
                }

            def _run_full_frame_label_review_batch_job(
                batch_items: list[tuple[str, int]],
            ) -> list[dict[str, Any]]:
                batch_reviews = _full_frame_label_vlm_review_batch(
                    client=client,
                    model=model_name,
                    image_b64=str(image_b64 or ""),
                    labels=[label for label, _obj_id in batch_items],
                )
                review_payloads: list[dict[str, Any]] = []
                for index, (label, obj_id) in enumerate(batch_items):
                    vlm_review = batch_reviews[index] if index < len(batch_reviews) else {}
                    review_payloads.append(
                        _build_full_frame_label_review_payload(
                            label=label,
                            obj_id=obj_id,
                            vlm_review=vlm_review,
                        )
                    )
                return review_payloads

            for batch_review_payloads in _run_in_thread_pool(
                _chunk_list(list(sorted(crop_unique_label_object_ids.items())), LABEL_BATCH_SIZE),
                _run_full_frame_label_review_batch_job,
                max_workers=vlm_workers,
            ):
                for review_payload in batch_review_payloads:
                    full_frame_label_reviews.append(review_payload)
                    full_frame_label_statuses[str(review_payload["label"])] = str(
                        review_payload["status"]
                    )

        full_frame_label_statuses = dict(sorted(full_frame_label_statuses.items()))
        full_frame_label_counts = {
            str(review_payload["label"]): int(review_payload["count"])
            for review_payload in full_frame_label_reviews
            if _normalize_full_frame_label_count(review_payload.get("count")) is not None
        }
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
        normalized_out_of_frame_review = out_of_frame_review
        if normalized_out_of_frame_review is None:
            if image_b64 is None:
                image_b64 = _image_to_base64(image)
            normalized_out_of_frame_review = _review_out_of_frame_label_candidates(
                client=client,
                model_name=model_name,
                image=image,
                image_b64=image_b64,
                scene_objects=scene_objects,
                objects_by_id=objects_by_id,
                visibility_by_obj_id=computed_visibility_by_obj_id,
                camera_pose=camera_pose,
                color_intrinsics=color_intrinsics,
                instance_mesh_data_getter=instance_mesh_data_getter,
            )
        out_of_frame_label_reviews = list(
            normalized_out_of_frame_review["out_of_frame_label_reviews"]
        )
        out_of_frame_not_visible_labels = list(
            normalized_out_of_frame_review["out_of_frame_not_visible_labels"]
        )
        out_of_frame_label_to_object_ids = {
            str(label): [int(obj_id) for obj_id in obj_ids]
            for label, obj_ids in sorted(
                normalized_out_of_frame_review["out_of_frame_label_to_object_ids"].items()
            )
        }
        out_of_frame_vlm_early_stop = bool(
            normalized_out_of_frame_review["out_of_frame_vlm_early_stop"]
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

    entry = {
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
        "out_of_frame_label_reviews": list(out_of_frame_label_reviews),
        "out_of_frame_not_visible_labels": list(out_of_frame_not_visible_labels),
        "out_of_frame_label_to_object_ids": {
            str(label): [int(obj_id) for obj_id in obj_ids]
            for label, obj_ids in sorted(out_of_frame_label_to_object_ids.items())
        },
        "out_of_frame_vlm_early_stop": bool(out_of_frame_vlm_early_stop),
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
    return _apply_attachment_layer_payloads(
        entry,
    )


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
        "out_of_frame_label_reviews",
        "out_of_frame_not_visible_labels",
        "out_of_frame_label_to_object_ids",
        "out_of_frame_vlm_early_stop",
        "referable_object_ids",
    }
    if not required_keys.issubset(entry.keys()):
        return False
    return _frame_entry_has_consistent_final_fields(entry)


def _frame_entry_has_out_of_frame_review_data(entry: Any) -> bool:
    if not isinstance(entry, dict):
        return False
    return bool(
        _normalize_cached_out_of_frame_label_reviews(entry.get("out_of_frame_label_reviews"))
        or _normalize_cached_out_of_frame_not_visible_labels(
            entry.get("out_of_frame_not_visible_labels")
        )
        or _shared_normalize_label_to_object_ids(entry.get("out_of_frame_label_to_object_ids"))
        or _normalize_cached_out_of_frame_vlm_early_stop(entry.get("out_of_frame_vlm_early_stop"))
    )


def _extract_out_of_frame_review_payload(entry: Any) -> dict[str, Any]:
    return {
        "out_of_frame_label_reviews": _normalize_cached_out_of_frame_label_reviews(
            entry.get("out_of_frame_label_reviews") if isinstance(entry, dict) else None
        ),
        "out_of_frame_not_visible_labels": _normalize_cached_out_of_frame_not_visible_labels(
            entry.get("out_of_frame_not_visible_labels") if isinstance(entry, dict) else None
        ),
        "out_of_frame_label_to_object_ids": _shared_normalize_label_to_object_ids(
            entry.get("out_of_frame_label_to_object_ids") if isinstance(entry, dict) else None
        ),
        "out_of_frame_vlm_early_stop": _normalize_cached_out_of_frame_vlm_early_stop(
            entry.get("out_of_frame_vlm_early_stop") if isinstance(entry, dict) else None
        ),
    }


def _enrich_final_scene_entries_out_of_frame(
    *,
    client,
    model_name: str,
    scene_dir: Path,
    final_scene_entries: dict[str, dict[str, Any]],
    scene_objects: list[dict[str, Any]],
    objects_by_id: dict[int, dict[str, Any]],
    poses: dict[str, CameraPose],
    color_intrinsics: CameraIntrinsics | None,
    depth_intrinsics: CameraIntrinsics | None,
    referability_entry_getter: Callable[[str], dict[str, Any] | None] | None = None,
    instance_mesh_data_getter: Callable[[int], InstanceMeshData] | None = None,
) -> dict[str, dict[str, Any]]:
    enriched_entries: dict[str, dict[str, Any]] = {}
    for image_name, entry in final_scene_entries.items():
        updated_entry = dict(entry)
        if _frame_entry_has_out_of_frame_review_data(updated_entry):
            enriched_entries[image_name] = updated_entry
            continue

        if callable(referability_entry_getter):
            cached_entry = referability_entry_getter(image_name)
            if isinstance(cached_entry, dict) and _frame_entry_has_out_of_frame_review_data(cached_entry):
                updated_entry.update(_extract_out_of_frame_review_payload(cached_entry))
                enriched_entries[image_name] = updated_entry
                continue

        camera_pose = poses.get(image_name)
        if camera_pose is None:
            enriched_entries[image_name] = updated_entry
            continue

        image_path = scene_dir / "color" / image_name
        image = cv2.imread(str(image_path))
        if image is None:
            logger.warning("Cannot read image %s for out-of-frame enrichment", image_path)
            enriched_entries[image_name] = updated_entry
            continue

        visibility_by_obj_id = compute_referability_object_visibility(
            scene_objects,
            camera_pose,
            color_intrinsics,
        )
        updated_entry.update(
            _review_out_of_frame_label_candidates(
                client=client,
                model_name=model_name,
                image=image,
                scene_objects=scene_objects,
                objects_by_id=objects_by_id,
                visibility_by_obj_id=visibility_by_obj_id,
                camera_pose=camera_pose,
                color_intrinsics=color_intrinsics,
                instance_mesh_data_getter=instance_mesh_data_getter,
            )
        )
        enriched_entries[image_name] = updated_entry

    return enriched_entries


def _build_scene_grouping_summary(scene_id: str) -> dict[str, Any]:
    return {
        "scene_id": str(scene_id),
        "pipeline_outcome": None,
        "grouping_available": True,
        "scene_skip_reason": None,
        "non_attachment_candidate_frame_count": 0,
        "non_attachment_visible_object_group_count": 0,
        "non_attachment_processed_group_count": 0,
        "accepted_frame_count_after_group_scan": 0,
        "reranked_accepted_frame_image_names": [],
        "selected_before_attachment_slots_image_names": [],
        "selected_before_attachment_slots_count": 0,
        "attachment_selected_frame_image_names": [],
        "attachment_selected_frame_count": 0,
        "remaining_slots_after_attachment_selection": None,
        "selected_after_attachment_slots_image_names": [],
        "selected_after_attachment_slots_count": 0,
        "final_cacheable_frame_image_names": [],
        "final_cacheable_frame_count": 0,
        "groups": [],
    }


def _scene_grouping_has_details(record: Any) -> bool:
    if not isinstance(record, dict):
        return False
    return any(
        key in record
        for key in (
            "non_attachment_candidate_frame_count",
            "non_attachment_visible_object_group_count",
            "non_attachment_processed_group_count",
            "reranked_accepted_frame_image_names",
            "selected_before_attachment_slots_image_names",
            "selected_after_attachment_slots_image_names",
            "attachment_selected_frame_image_names",
            "remaining_slots_after_attachment_selection",
            "final_cacheable_frame_image_names",
            "groups",
        )
    )


def _infer_default_split(data_root: Path) -> str:
    return "train"


def _is_scannet_scene_dir(path: Path) -> bool:
    return path.is_dir() and (path / "pose").exists()


def _resolve_scannet_scene_root(data_root: Path) -> Path:
    if data_root.name == "scans":
        return data_root
    nested_scans_root = data_root / "scans"
    if nested_scans_root.exists() and nested_scans_root.is_dir():
        return nested_scans_root
    has_flat_scene_dirs = data_root.exists() and data_root.is_dir() and any(
        _is_scannet_scene_dir(path)
        for path in data_root.iterdir()
    )
    if has_flat_scene_dirs:
        return data_root
    return nested_scans_root


def _read_scannet_split_scene_ids(split: str) -> list[str]:
    split_file = SCANNET_METADATA_SPLIT_FILES.get(split)
    if split_file is None:
        raise ValueError(f"Unsupported ScanNet split metadata request: {split}")
    try:
        raw_lines = split_file.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"ScanNet split metadata file for {split} does not exist: {split_file}"
        ) from exc

    scene_ids: list[str] = []
    seen: set[str] = set()
    for raw_line in raw_lines:
        scene_id = raw_line.strip()
        if not scene_id or scene_id in seen:
            continue
        seen.add(scene_id)
        scene_ids.append(scene_id)
    return scene_ids


def _resolve_scannet_scene_dirs(data_root: Path, split: str) -> list[tuple[str, Path]]:
    scene_root = _resolve_scannet_scene_root(data_root)
    split_order = ["train", "val"] if split == "all" else [split]
    scene_entries: list[tuple[str, Path]] = []
    if not scene_root.exists() or not scene_root.is_dir():
        logger.warning("ScanNet scene root does not exist: %s", scene_root)
        return scene_entries

    available_scene_dirs = {
        path.name: path
        for path in scene_root.iterdir()
        if _is_scannet_scene_dir(path)
    }
    for split_name in split_order:
        split_scene_ids = _read_scannet_split_scene_ids(split_name)
        split_scene_dirs = [
            available_scene_dirs[scene_id]
            for scene_id in sorted(scene_id for scene_id in split_scene_ids if scene_id in available_scene_dirs)
        ]
        if not split_scene_dirs:
            logger.warning(
                "Found no ScanNet scenes under %s for split=%s after metadata filtering",
                scene_root,
                split_name,
            )
        scene_entries.extend((split_name, path) for path in split_scene_dirs)
    return scene_entries


def _cache_contains_legacy_test_split(cache: dict[str, Any]) -> bool:
    for field_name in ("scene_status", "scene_grouping"):
        field_value = cache.get(field_name)
        if not isinstance(field_value, dict):
            continue
        for record in field_value.values():
            if not isinstance(record, dict):
                continue
            split_value = str(record.get("split") or "").strip().lower()
            if split_value == "test":
                return True
    return False


def _build_scene_status_record(
    scene_id: str,
    *,
    split: str | None,
    pipeline_outcome: str,
    has_cache_frames: bool,
    final_cacheable_frame_count: int,
    scene_skip_reason: str | None,
) -> dict[str, Any]:
    return {
        "scene_id": str(scene_id),
        "processed": True,
        "pipeline_outcome": str(pipeline_outcome),
        "split": None if split is None else str(split),
        "has_cache_frames": bool(has_cache_frames),
        "final_cacheable_frame_count": max(0, int(final_cacheable_frame_count)),
        "scene_skip_reason": None if scene_skip_reason is None else str(scene_skip_reason),
    }


def _prepare_scene_grouping_summary(
    scene_id: str,
    split: str,
    summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    scene_grouping_summary = (
        dict(summary)
        if isinstance(summary, dict)
        else _build_scene_grouping_summary(scene_id)
    )
    scene_grouping_summary["scene_id"] = str(scene_id)
    scene_grouping_summary["split"] = str(split)
    if "grouping_available" not in scene_grouping_summary:
        scene_grouping_summary["grouping_available"] = _scene_grouping_has_details(summary)
    return scene_grouping_summary


def _persist_scene_state(
    *,
    cache: dict[str, Any],
    scene_grouping_cache: dict[str, Any],
    scene_status_cache: dict[str, Any],
    output_path: Path,
    scene_id: str,
    split: str,
    pipeline_outcome: str,
    scene_skip_reason: str | None,
    scene_grouping_summary: dict[str, Any] | None,
    scene_cache: dict[str, Any] | None,
) -> None:
    summary = _prepare_scene_grouping_summary(scene_id, split, scene_grouping_summary)
    summary["pipeline_outcome"] = str(pipeline_outcome)
    summary["scene_skip_reason"] = None if scene_skip_reason is None else str(scene_skip_reason)
    final_cacheable_frame_count = int(
        summary.get(
            "final_cacheable_frame_count",
            len(scene_cache) if isinstance(scene_cache, dict) else 0,
        ) or 0
    )
    if "final_cacheable_frame_image_names" not in summary:
        summary["final_cacheable_frame_image_names"] = (
            [str(image_name) for image_name in scene_cache.keys()]
            if isinstance(scene_cache, dict) and scene_cache
            else []
        )
    scene_grouping_cache[scene_id] = summary
    scene_status_cache[scene_id] = _build_scene_status_record(
        scene_id,
        split=split,
        pipeline_outcome=pipeline_outcome,
        has_cache_frames=bool(scene_cache) or final_cacheable_frame_count > 0,
        final_cacheable_frame_count=final_cacheable_frame_count,
        scene_skip_reason=scene_skip_reason,
    )
    _write_json_payload(output_path, cache)


def _persist_scene_state_and_status(
    *,
    cache: dict[str, Any],
    scene_grouping_cache: dict[str, Any],
    scene_status_cache: dict[str, Any],
    output_path: Path,
    scene_id: str,
    split: str,
    pipeline_outcome: str,
    scene_skip_reason: str | None,
    scene_grouping_summary: dict[str, Any] | None,
    scene_cache: dict[str, Any] | None,
    global_scene_status_doc: dict[str, Any],
    global_scene_status_path: Path,
    batch_file_name: str,
    status_updated_at: str,
) -> None:
    _persist_scene_state(
        cache=cache,
        scene_grouping_cache=scene_grouping_cache,
        scene_status_cache=scene_status_cache,
        output_path=output_path,
        scene_id=scene_id,
        split=split,
        pipeline_outcome=pipeline_outcome,
        scene_skip_reason=scene_skip_reason,
        scene_grouping_summary=scene_grouping_summary,
        scene_cache=scene_cache,
    )
    _mark_scene_completed(
        global_scene_status_doc,
        scene_id=scene_id,
        batch_file=batch_file_name,
        updated_at=status_updated_at,
    )
    _write_json_payload(global_scene_status_path, global_scene_status_doc)


def _migrate_scene_status_cache(cache: dict[str, Any]) -> bool:
    changed = False
    if not isinstance(cache.get("frames"), dict):
        cache["frames"] = {}
        changed = True
    if not isinstance(cache.get("scene_grouping"), dict):
        cache["scene_grouping"] = {}
        changed = True
    if not isinstance(cache.get("scene_status"), dict):
        cache["scene_status"] = {}
        changed = True

    frames_cache = cache["frames"]
    scene_grouping_cache = cache["scene_grouping"]
    scene_status_cache = cache["scene_status"]

    for scene_id, frame_entries in frames_cache.items():
        if scene_id in scene_status_cache:
            continue
        frame_count = len(frame_entries) if isinstance(frame_entries, dict) else 0
        grouping_summary = scene_grouping_cache.get(scene_id)
        pipeline_outcome = "processed"
        scene_skip_reason = None
        split = None
        if isinstance(grouping_summary, dict):
            pipeline_outcome = str(grouping_summary.get("pipeline_outcome") or "processed")
            scene_skip_reason = grouping_summary.get("scene_skip_reason")
            split = grouping_summary.get("split")
        scene_status_cache[scene_id] = _build_scene_status_record(
            scene_id,
            split=None if split is None else str(split),
            pipeline_outcome=pipeline_outcome,
            has_cache_frames=frame_count > 0,
            final_cacheable_frame_count=frame_count,
            scene_skip_reason=None if scene_skip_reason is None else str(scene_skip_reason),
        )
        changed = True

    for scene_id, grouping_summary in scene_grouping_cache.items():
        if scene_id in scene_status_cache or not isinstance(grouping_summary, dict):
            continue
        frame_entries = frames_cache.get(scene_id)
        frame_count = len(frame_entries) if isinstance(frame_entries, dict) else 0
        scene_status_cache[scene_id] = _build_scene_status_record(
            scene_id,
            split=(
                None
                if grouping_summary.get("split") is None
                else str(grouping_summary.get("split"))
            ),
            pipeline_outcome=str(grouping_summary.get("pipeline_outcome") or "legacy_migrated"),
            has_cache_frames=bool(frame_count or grouping_summary.get("final_cacheable_frame_count", 0)),
            final_cacheable_frame_count=int(
                grouping_summary.get("final_cacheable_frame_count", frame_count) or 0
            ),
            scene_skip_reason=(
                None
                if grouping_summary.get("scene_skip_reason") is None
                else str(grouping_summary.get("scene_skip_reason"))
            ),
        )
        changed = True
    return changed


def _log_final_batch_banner(
    *,
    split: str,
    total_scene_count: int,
    processed_scene_count: int,
    remaining_scene_count: int,
    completed: bool = False,
) -> None:
    if completed:
        headline = f"ALL SCENES COMPLETED FOR SPLIT {split}"
        body = "ALL SCENES PROCESSED AFTER THIS RUN"
    else:
        headline = f"FINAL BATCH FOR SPLIT {split}"
        body = (
            f"Only {remaining_scene_count} unprocessed scenes remain; "
            "all remaining scenes will be processed in this run."
        )
    logger.warning("============================================================")
    logger.warning("%s", headline)
    logger.warning(
        "Total scenes: %d | Already processed: %d | Remaining unprocessed: %d",
        total_scene_count,
        processed_scene_count,
        remaining_scene_count,
    )
    logger.warning("%s", body)
    logger.warning("============================================================")


def _select_and_rerank_frames(
    *,
    client,
    model_name: str,
    scene_dir: Path,
    frame_candidates: list[dict[str, Any]],
    max_frames: int,
    max_group_count: int | None = None,
    poses: dict[str, CameraPose] | None = None,
    vlm_workers: int = 1,
    frame_review_getter: Callable[[dict[str, Any]], dict[str, Any] | None] | None = None,
    frame_review_batch_getter: Callable[[list[dict[str, Any]]], dict[str, Any] | None] | None = None,
    referability_entry_builder: Callable[[dict[str, Any], dict[str, Any]], dict[str, Any] | None] | None = None,
    stats_output: dict[str, Any] | None = None,
    debug_output: dict[str, Any] | None = None,
    frame_clarity_batch_size: int = FRAME_CLARITY_BATCH_SIZE,
    non_attachment_referability_shortlist: int = DEFAULT_NON_ATTACHMENT_REFERABILITY_SHORTLIST,
    non_attachment_clarity_min_score: int = DEFAULT_NON_ATTACHMENT_CLARITY_MIN_SCORE,
) -> list[dict[str, Any]]:
    if not frame_candidates or int(max_frames) <= 0:
        return []

    reranked: list[dict[str, Any]] = []
    accepted_frame_count = 0
    group_count = _count_visible_object_frame_groups(
        frame_candidates,
        poses=poses,
    )
    group_limit = group_count
    if max_group_count is not None:
        group_limit = max(0, min(group_count, int(max_group_count)))
    group_debug: list[dict[str, Any]] = []

    def _sort_key(entry: dict[str, Any]) -> tuple[int, str]:
        return (
            -int(entry.get("frame_selection_score", 0) or 0),
            str(entry.get("image_name", "")),
        )

    reviewed_frames = _select_non_attachment_group_representatives(
        client=client,
        model_name=model_name,
        scene_dir=scene_dir,
        frames=frame_candidates,
        poses=poses,
        max_group_count=group_limit,
        max_accepted_frame_count=int(max_frames),
        vlm_workers=vlm_workers,
        frame_review_getter=frame_review_getter,
        frame_review_batch_getter=frame_review_batch_getter,
        referability_entry_builder=referability_entry_builder,
        debug_groups_out=group_debug,
        frame_clarity_batch_size=frame_clarity_batch_size,
        non_attachment_referability_shortlist=non_attachment_referability_shortlist,
        non_attachment_clarity_min_score=non_attachment_clarity_min_score,
    )
    for reviewed_frame in reviewed_frames:
        accepted_frame_count += 1
        reranked.append(reviewed_frame)
    processed_group_count = len(group_debug)

    reranked.sort(
        key=_sort_key,
    )
    selected = reranked[:max(0, int(max_frames))]
    if stats_output is not None:
        stats_output.clear()
        stats_output.update(
            {
                "scene_id": scene_dir.name,
                "non_attachment_bbox_in_frame_ratio_min": REFERABLE_BBOX_IN_FRAME_RATIO_MIN,
                "non_attachment_min_referable_object_count": NON_ATTACHMENT_GROUP_MIN_REFERABLE_OBJECT_COUNT,
                "non_attachment_candidate_frame_count": len(frame_candidates),
                "non_attachment_visible_object_group_count": group_count,
                "non_attachment_processed_group_count": processed_group_count,
                "accepted_frame_count_after_group_scan": accepted_frame_count,
            }
        )
    if debug_output is not None:
        selected_before_attachment_slots = [
            str(frame.get("image_name", "")).strip()
            for frame in selected
        ]
        selected_before_attachment_set = set(selected_before_attachment_slots)
        reranked_image_names = [
            str(frame.get("image_name", "")).strip()
            for frame in reranked
        ]
        for group in group_debug or []:
            accepted_names = list(group.get("accepted_frame_image_names", []))
            selected_before = [
                image_name for image_name in accepted_names
                if image_name in selected_before_attachment_set
            ]
            dropped_by_group_rerank = [
                image_name for image_name in accepted_names
                if image_name not in selected_before_attachment_set
            ]
            group["selected_before_attachment_slots_image_names"] = selected_before
            group["dropped_by_group_rerank_image_names"] = dropped_by_group_rerank
            if not accepted_names:
                if bool(group.get("group_exhausted_without_usable_frame", False)):
                    group["status_before_attachment_slots"] = "no_usable_frame"
                else:
                    group["status_before_attachment_slots"] = "no_referable_frame"
            elif selected_before:
                group["status_before_attachment_slots"] = "selected_before_attachment_slots"
            else:
                group["status_before_attachment_slots"] = "dropped_by_group_rerank"
        debug_output.clear()
        debug_output.update(
            {
                "scene_id": scene_dir.name,
                "pipeline_outcome": None,
                "grouping_available": True,
                "scene_skip_reason": None,
                "non_attachment_bbox_in_frame_ratio_min": REFERABLE_BBOX_IN_FRAME_RATIO_MIN,
                "non_attachment_min_referable_object_count": NON_ATTACHMENT_GROUP_MIN_REFERABLE_OBJECT_COUNT,
                "non_attachment_candidate_frame_count": len(frame_candidates),
                "non_attachment_visible_object_group_count": group_count,
                "non_attachment_processed_group_count": processed_group_count,
                "accepted_frame_count_after_group_scan": accepted_frame_count,
                "reranked_accepted_frame_image_names": reranked_image_names,
                "selected_before_attachment_slots_image_names": selected_before_attachment_slots,
                "selected_before_attachment_slots_count": len(selected_before_attachment_slots),
                "attachment_selected_frame_image_names": [],
                "attachment_selected_frame_count": 0,
                "remaining_slots_after_attachment_selection": None,
                "selected_after_attachment_slots_image_names": [],
                "selected_after_attachment_slots_count": 0,
                "final_cacheable_frame_image_names": [],
                "final_cacheable_frame_count": 0,
                "groups": group_debug,
            }
        )
    if reranked:
        logger.info(
            "VLM non-attachment group filtering for %d geometric frame candidates after reviewing %d/%d visible-object groups in %s: %d accepted frame(s) meeting bbox_in_frame_ratio >= %.2f with at least %d referable objects (best selection score=%d)",
            len(frame_candidates),
            processed_group_count,
            group_count,
            scene_dir.name,
            accepted_frame_count,
            REFERABLE_BBOX_IN_FRAME_RATIO_MIN,
            NON_ATTACHMENT_GROUP_MIN_REFERABLE_OBJECT_COUNT,
            int(reranked[0].get("frame_selection_score", 0) or 0),
        )
    elif processed_group_count < group_count:
        logger.info(
            "VLM non-attachment group filtering capped %s to %d/%d visible-object groups before review",
            scene_dir.name,
            processed_group_count,
            group_count,
        )
    return selected


def main():
    parser = argparse.ArgumentParser(description="Precompute VLM frame/object referability cache")
    parser.add_argument(
        "--data_root", type=str,
        default=os.getenv("SCANNET_PATH", "/home/lihongxing/datasets/ScanNet/data/scans"),
        help="ScanNet scene root; supports .../data, .../data/scans, or a directory containing scene folders directly",
    )
    parser.add_argument(
        "--split",
        choices=("train", "val", "all"),
        default="train",
        help="ScanNet split to process using fixed metadata scene lists; --split all runs train first, then val",
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Batch output path or directory; each run writes one new timestamped batch JSON and stores global progress in scene_status.json beside it. Concurrent shard runs must use different --output directories.",
    )
    parser.add_argument(
        "--max_scenes", type=int, default=300,
        help="Legacy scene cap used only when --scene_number is omitted; not recommended for long resume-heavy experiments",
    )
    parser.add_argument(
        "--scene_number", type=str, default=None,
        help="Fixed scene interval START-END using 0-based inclusive indexes over the full ordered split scene list; e.g. 0-20 selects 21 scenes",
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
        help="Resume from the global scene_status.json if present; with --scene_number, only scenes inside that fixed interval are considered",
    )
    parser.add_argument(
        "--reset", type=int, default=None,
        help="Remove the most recently completed N scene entries from scene_status.json before processing; existing batch JSON files, frame sidecars, and review artifacts are kept",
    )
    parser.add_argument(
        "--label_batch_size", type=int, default=LABEL_BATCH_SIZE,
        help="Legacy compatibility flag; per-object review now issues one VLM request per valid crop",
    )
    parser.add_argument(
        "--vlm_workers", type=int, default=4,
        help="Maximum number of concurrent independent VLM requests",
    )
    parser.add_argument(
        "--frame_clarity_batch_size",
        type=int,
        default=FRAME_CLARITY_BATCH_SIZE,
        help="Maximum number of sampled group frames to send in one batch clarity VLM request",
    )
    parser.add_argument(
        "--non_attachment_referability_shortlist",
        type=int,
        default=DEFAULT_NON_ATTACHMENT_REFERABILITY_SHORTLIST,
        help="Legacy compatibility flag; non-attachment group review now follows BRISQUE order and stops at the first clarity-pass frame",
    )
    parser.add_argument(
        "--non_attachment_clarity_min_score",
        type=int,
        default=DEFAULT_NON_ATTACHMENT_CLARITY_MIN_SCORE,
        help="Legacy compatibility flag; retained for older scripts but ignored by the current frame-usable VLM review path",
    )
    parser.add_argument(
        "--attachment_clarity_min_score",
        type=int,
        default=DEFAULT_ATTACHMENT_CLARITY_MIN_SCORE,
        help="Legacy compatibility flag; retained for older scripts but ignored by the current frame-usable VLM review path",
    )
    parser.add_argument(
        "--scene_workers", type=int, default=1,
        help="Number of scenes to process concurrently",
    )
    parser.add_argument(
        "--write_attachment_review",
        dest="write_attachment_review",
        action="store_true",
        help="Write a scene-level attachment candidate review JSON alongside the referability cache",
    )
    parser.add_argument(
        "--no-write_attachment_review",
        dest="write_attachment_review",
        action="store_false",
        help="Disable the attachment candidate review JSON output",
    )
    parser.set_defaults(write_attachment_review=True)
    parser.add_argument(
        "--attachment_review_output", type=str, default=None,
        help="Optional path for the attachment candidate review JSON; defaults to candidate/<short-prefix>_attachment_candidate_review.json beside --output",
    )
    parser.add_argument(
        "--write_attachment_pair_salvage_review",
        dest="write_attachment_pair_salvage_review",
        action="store_true",
        help="Write group-level attachment pair salvage review JSON and HTML to salvage/<short-prefix>_salvage_review.{json,html} beside --output",
    )
    parser.add_argument(
        "--no-write_attachment_pair_salvage_review",
        dest="write_attachment_pair_salvage_review",
        action="store_false",
        help="Disable the group-level attachment pair salvage review JSON and HTML outputs",
    )
    parser.set_defaults(write_attachment_pair_salvage_review=True)
    parser.add_argument(
        "--attachment_pair_salvage_bbox_hard_fail_min",
        type=float,
        default=0.15,
        help="Object-level bbox_in_frame_ratio hard-fail threshold for attachment pair salvage review",
    )
    parser.add_argument(
        "--attachment_pair_salvage_projected_area_hard_fail_min",
        type=float,
        default=QUESTION_REVIEW_CROP_MIN_PROJECTED_AREA_PX,
        help="Object-level projected_area_px hard-fail threshold for attachment pair salvage review",
    )
    args = parser.parse_args()
    _reset_vlm_call_failure_count()
    if args.reset is not None and int(args.reset) <= 0:
        parser.error("--reset must be >= 1")
    try:
        scene_number_range = (
            None
            if args.scene_number is None
            else _parse_closed_scene_range(args.scene_number, arg_name="--scene_number")
        )
    except ValueError as exc:
        parser.error(str(exc))
    if scene_number_range is not None and args.split == "all":
        parser.error("--scene_number only supports --split train or --split val; --split all is ambiguous")
    if int(args.vlm_workers) <= 0:
        parser.error("--vlm_workers must be >= 1")
    if int(args.frame_clarity_batch_size) <= 0:
        parser.error("--frame_clarity_batch_size must be >= 1")
    if int(args.non_attachment_referability_shortlist) <= 0:
        parser.error("--non_attachment_referability_shortlist must be >= 1")
    if int(args.scene_workers) <= 0:
        parser.error("--scene_workers must be >= 1")

    global EXCLUDED_LABELS
    from src.scene_parser import EXCLUDED_LABELS as SCENE_EXCLUDED_LABELS
    from src.scene_parser import load_scannet_label_map, parse_scene
    from src.support_graph import (
        build_attachment_candidates,
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
    _configure_vlm_request_concurrency(int(args.vlm_workers))
    worker_client_factory = _ThreadLocalOpenAIClientFactory(
        OpenAI,
        api_key=api_key,
        base_url=args.vlm_url,
    )

    data_root = Path(args.data_root)
    selected_split = args.split or _infer_default_split(data_root)
    scene_entries = _resolve_scannet_scene_dirs(data_root, selected_split)
    logger.info(
        "Found %d candidate scenes for split=%s",
        len(scene_entries),
        selected_split,
    )
    output_arg = Path(args.output)
    batch_output_path = _build_batch_output_path(output_arg)
    scene_status_path = _scene_status_output_path(output_arg)
    if scene_number_range is not None:
        logger.warning(
            "Fixed scene shards require distinct --output directories per tmux/process. Do not share scene_status.json, attachment review JSON, salvage review JSON, salvage review HTML, or per-scene edited HTML files across concurrent runs."
        )
    scene_status_doc = _load_scene_status_doc(
        scene_status_path,
        split=selected_split,
    )
    if args.reset is not None:
        logger.info(
            "Reset requested for split=%s at %s: removing up to %d most recently completed scene(s)",
            selected_split,
            scene_status_path,
            int(args.reset),
        )
        removed_scene_ids = _reset_completed_scene_status(
            scene_status_doc,
            count=int(args.reset),
        )
        if removed_scene_ids:
            _write_json_payload(scene_status_path, scene_status_doc)
            scene_status_doc = _load_scene_status_doc(
                scene_status_path,
                split=selected_split,
            )
            logger.info(
                "Reset cleared %d completed scene(s): %s",
                len(removed_scene_ids),
                ", ".join(removed_scene_ids),
            )
        else:
            logger.info(
                "Reset cleared 0 completed scene(s) at %s because there was nothing to reset",
                scene_status_path,
            )
    _validate_scene_status_doc(
        scene_status_doc,
        scene_status_path=scene_status_path,
    )
    completed_scene_records = scene_status_doc.get("completed_scenes", {})
    completed_scene_ids = (
        set(completed_scene_records.keys())
        if isinstance(completed_scene_records, dict)
        else set()
    )
    if args.resume and completed_scene_ids:
        logger.info(
            "Resuming from %s with %d completed scene(s) for split=%s",
            scene_status_path,
            len(completed_scene_ids),
            selected_split,
        )

    output_path = batch_output_path
    attachment_review_output = (
        Path(args.attachment_review_output)
        if args.attachment_review_output else _attachment_review_output_path(output_path)
    )
    attachment_pair_salvage_review_output = _attachment_pair_salvage_review_output_path(output_path)
    attachment_pair_salvage_review_html_output = _attachment_pair_salvage_review_html_output_path(output_path)
    attachment_review_scenes: list[dict[str, Any]] = []
    attachment_review_terminal_lines: list[str] = []
    attachment_pair_salvage_review_scenes: list[dict[str, Any]] = []
    cache: dict[str, Any] = {
        "version": REFERABILITY_CACHE_VERSION,
        "model": model_name,
        "alias_config_version": ALIAS_CONFIG_VERSION,
        "referability_backend": REFERABILITY_BACKEND,
        "label_batch_size": 1,
        "frames": {},
        "scene_grouping": {},
        "scene_status": {},
    }
    scene_grouping_cache = cache["scene_grouping"]
    scene_status_cache = cache["scene_status"]

    def _write_attachment_review() -> None:
        if not args.write_attachment_review:
            return
        review_doc = _build_attachment_review_document(
            referability_cache_output=output_path,
            scenes=attachment_review_scenes,
            terminal_output_lines=attachment_review_terminal_lines,
        )
        _write_json_payload(attachment_review_output, review_doc)
        logger.info("Saved attachment candidate review to %s", attachment_review_output)

    def _finalize_attachment_review_scene(record: dict[str, Any]) -> None:
        if not args.write_attachment_review:
            return
        attachment_review_scenes.append(record)
        attachment_review_terminal_lines.extend(record.get("terminal_output_lines", []))
        for line in record.get("terminal_output_lines", []):
            logger.info("%s", line)

    def _write_attachment_pair_salvage_review() -> None:
        if not args.write_attachment_pair_salvage_review:
            return
        review_doc = _build_attachment_pair_salvage_review_document(
            referability_cache_output=output_path,
            scenes=attachment_pair_salvage_review_scenes,
        )
        _write_json_payload(attachment_pair_salvage_review_output, review_doc)
        rendered_review_html = _render_attachment_pair_salvage_review_html(review_doc)
        attachment_pair_salvage_review_html_output.parent.mkdir(parents=True, exist_ok=True)
        attachment_pair_salvage_review_html_output.write_text(rendered_review_html, encoding="utf-8")
        edited_outputs_by_scene = review_doc.get("edited_html_outputs_by_scene", {})
        if isinstance(edited_outputs_by_scene, dict):
            for scene_id in sorted(edited_outputs_by_scene):
                scene_review_path = Path(str(edited_outputs_by_scene[scene_id]))
                scene_review_doc = _build_attachment_pair_salvage_review_scene_document(
                    review_doc=review_doc,
                    scene_id=scene_id,
                )
                scene_review_html = _render_attachment_pair_salvage_review_html(scene_review_doc)
                scene_review_path.parent.mkdir(parents=True, exist_ok=True)
                scene_review_path.write_text(scene_review_html, encoding="utf-8")
        logger.info("Saved attachment pair salvage review JSON to %s", attachment_pair_salvage_review_output)
        logger.info("Saved attachment pair salvage review HTML to %s", attachment_pair_salvage_review_html_output)
        logger.warning(
            "run_pipeline does not read %s directly; it reads the per-scene edited HTML files listed below, or a legacy neighboring edited.html only when no per-scene files exist.",
            attachment_pair_salvage_review_html_output,
        )
        logger.warning("========== 人工审核 ==========")
        logger.warning(
            "请按 scene 打开并修改对应 HTML；pipeline 会读取 %s",
            review_doc.get("edited_html_output_glob", ""),
        )
        if isinstance(edited_outputs_by_scene, dict):
            for scene_id in sorted(edited_outputs_by_scene):
                logger.warning("[%s] %s", scene_id, edited_outputs_by_scene[scene_id])
        logger.warning("==============================")

    def _finalize_attachment_pair_salvage_review_scene(
        *,
        scene_id: str,
        split: str,
        pipeline_outcome: str,
        scene_review: dict[str, Any] | None,
    ) -> None:
        if not args.write_attachment_pair_salvage_review:
            return
        record = _build_attachment_pair_salvage_review_scene_record(
            scene_id=scene_id,
            split=split,
            pipeline_outcome=pipeline_outcome,
            scene_review=scene_review,
        )
        attachment_pair_salvage_review_scenes.append(record)
        for line in record.get("terminal_output_lines", []):
            logger.info("%s", line)

    def _persist_current_scene(
        *,
        scene_id: str,
        split: str,
        pipeline_outcome: str,
        scene_skip_reason: str | None,
        scene_grouping_summary: dict[str, Any] | None,
        scene_cache: dict[str, Any] | None,
    ) -> None:
        _persist_scene_state_and_status(
            cache=cache,
            scene_grouping_cache=scene_grouping_cache,
            scene_status_cache=scene_status_cache,
            output_path=output_path,
            scene_id=scene_id,
            split=split,
            pipeline_outcome=pipeline_outcome,
            scene_skip_reason=scene_skip_reason,
            scene_grouping_summary=scene_grouping_summary,
            scene_cache=scene_cache,
            global_scene_status_doc=scene_status_doc,
            global_scene_status_path=scene_status_path,
            batch_file_name=output_path.name,
            status_updated_at=_timestamp_for_status(),
        )

    def _finalize_attachment_pair_salvage_review_record(record: dict[str, Any]) -> None:
        if not args.write_attachment_pair_salvage_review:
            return
        attachment_pair_salvage_review_scenes.append(record)
        for line in record.get("terminal_output_lines", []):
            logger.info("%s", line)

    def _commit_scene_result(result: SceneWorkerResult) -> None:
        frames_cache = cache.setdefault("frames", {})
        if isinstance(result.scene_cache, dict):
            frames_cache[result.scene_id] = {
                str(image_name): dict(entry)
                for image_name, entry in result.scene_cache.items()
            }
        else:
            frames_cache.pop(result.scene_id, None)
        _persist_current_scene(
            scene_id=result.scene_id,
            split=result.split,
            pipeline_outcome=result.pipeline_outcome,
            scene_skip_reason=result.scene_skip_reason,
            scene_grouping_summary=result.scene_grouping_summary,
            scene_cache=result.scene_cache,
        )
        if isinstance(result.frame_sidecar_cache, dict):
            _write_frame_sidecar_scene_cache(
                output_path=output_path,
                scene_id=result.scene_id,
                model_name=model_name,
                referability_backend=REFERABILITY_BACKEND,
                frame_records=result.frame_sidecar_cache,
            )
        if isinstance(result.attachment_review_record, dict):
            _finalize_attachment_review_scene(result.attachment_review_record)
        if isinstance(result.attachment_pair_salvage_review_record, dict):
            _finalize_attachment_pair_salvage_review_record(
                result.attachment_pair_salvage_review_record
            )
        _write_attachment_review()
        _write_attachment_pair_salvage_review()

    def _process_scene_worker(
        scene_position: int,
        scene_split: str,
        scene_dir: Path,
    ) -> SceneWorkerResult:
        scene_id = scene_dir.name
        scene_index = scene_index_by_id.get(scene_id, scene_position + 1)
        if scene_number_range is not None:
            logger.info(
                "=== Referability scene %s [split=%s] (%d/%d; split total=%d) ===",
                scene_id,
                scene_split,
                scene_index,
                len(selected_scene_entries),
                len(scene_entries),
            )
        else:
            logger.info(
                "=== Referability scene %s [split=%s] (%d/%d) ===",
                scene_id,
                scene_split,
                scene_index,
                len(scene_entries),
            )
        scene_client = worker_client_factory

        def _build_attachment_pair_record(
            pipeline_outcome: str,
            scene_review: dict[str, Any] | None,
        ) -> dict[str, Any] | None:
            if not args.write_attachment_pair_salvage_review:
                return None
            return _build_attachment_pair_salvage_review_scene_record(
                scene_id=scene_id,
                split=scene_split,
                pipeline_outcome=pipeline_outcome,
                scene_review=scene_review,
            )

        def _build_result(
            *,
            pipeline_outcome: str,
            scene_skip_reason: str | None,
            scene_grouping_summary: dict[str, Any] | None,
            scene_cache: dict[str, Any] | None,
            attachment_review_record: dict[str, Any] | None,
            attachment_pair_salvage_review_record: dict[str, Any] | None,
            frame_sidecar_cache: dict[str, dict[str, Any]] | None = None,
        ) -> SceneWorkerResult:
            return SceneWorkerResult(
                scene_index=scene_index,
                scene_id=scene_id,
                split=scene_split,
                pipeline_outcome=pipeline_outcome,
                scene_skip_reason=scene_skip_reason,
                scene_cache=scene_cache,
                scene_grouping_summary=scene_grouping_summary,
                attachment_review_record=attachment_review_record,
                attachment_pair_salvage_review_record=attachment_pair_salvage_review_record,
                frame_sidecar_cache=frame_sidecar_cache,
            )

        preloaded_geometry = None
        try:
            preloaded_geometry = _load_scene_geometry(scene_dir)
        except Exception as exc:
            logger.warning("Scene geometry preload failed for %s: %s", scene_id, exc)
        scene = parse_scene(scene_dir, preloaded_geometry=preloaded_geometry)
        if scene is None:
            return _build_result(
                pipeline_outcome="parse_scene_failed",
                scene_skip_reason="parse_scene_failed",
                scene_grouping_summary=_prepare_scene_grouping_summary(scene_id, scene_split),
                scene_cache=None,
                attachment_review_record=None,
                attachment_pair_salvage_review_record=_build_attachment_pair_record(
                    "parse_scene_failed",
                    None,
                ),
            )

        enrich_scene_with_attachment(scene)
        attachment_graph = get_scene_attachment_graph(scene, scene_id=scene_id)
        raw_attachment_candidates = (
            build_attachment_candidates(scene["objects"])
            if args.write_attachment_review else []
        )
        final_attachment_edges = [
            dict(edge)
            for edge in scene.get("attachment_edges", [])
            if isinstance(edge, dict)
        ]

        def _make_attachment_review_record(pipeline_outcome: str) -> dict[str, Any] | None:
            if not args.write_attachment_review:
                return None
            return _build_attachment_review_scene_record(
                scene_id=scene_id,
                objects=scene["objects"],
                raw_candidates=raw_attachment_candidates,
                final_attachment_edges=final_attachment_edges,
                pipeline_outcome=pipeline_outcome,
            )

        if not has_nontrivial_attachment(attachment_graph):
            logger.info("Scene %s has no attachment relations -> skipping", scene_id)
            return _build_result(
                pipeline_outcome="no_attachment_relations",
                scene_skip_reason="no_attachment_relations",
                scene_grouping_summary=_prepare_scene_grouping_summary(scene_id, scene_split),
                scene_cache=None,
                attachment_review_record=_make_attachment_review_record(
                    "no_attachment_relations"
                ),
                attachment_pair_salvage_review_record=_build_attachment_pair_record(
                    "no_attachment_relations",
                    None,
                ),
            )

        axis_align = load_axis_alignment(scene_dir)
        poses = load_scannet_poses(scene_dir, axis_alignment=axis_align)
        try:
            color_intrinsics = load_scannet_intrinsics(scene_dir)
        except Exception as exc:
            logger.warning("Color intrinsics load failed for %s: %s", scene_id, exc)
            return _build_result(
                pipeline_outcome="color_intrinsics_load_failed",
                scene_skip_reason="color_intrinsics_load_failed",
                scene_grouping_summary=_prepare_scene_grouping_summary(scene_id, scene_split),
                scene_cache=None,
                attachment_review_record=_make_attachment_review_record(
                    "color_intrinsics_load_failed"
                ),
                attachment_pair_salvage_review_record=_build_attachment_pair_record(
                    "color_intrinsics_load_failed",
                    None,
                ),
            )
        selector_instance_mesh_data: InstanceMeshData | None = None
        try:
            selector_instance_mesh_data = load_instance_mesh_data(
                scene_dir,
                instance_ids=[
                    int(obj["id"])
                    for obj in scene["objects"]
                    if obj.get("id") is not None
                ],
                n_surface_samples=1,
                preloaded_geometry=preloaded_geometry,
            )
        except Exception as exc:
            logger.warning(
                "Instance mesh preload failed for %s; selector falls back to bbox ratio: %s",
                scene_id,
                exc,
            )
        frame_candidates = select_frames(
            scene_dir,
            scene["objects"],
            attachment_graph,
            int(args.max_frames),
            keep_all_attachment_frames=True,
            color_intrinsics=color_intrinsics,
            axis_alignment=axis_align,
            poses=poses,
            instance_mesh_data=selector_instance_mesh_data,
            preloaded_geometry=preloaded_geometry,
        )
        if not frame_candidates:
            return _build_result(
                pipeline_outcome="no_frame_candidates",
                scene_skip_reason="no_frame_candidates",
                scene_grouping_summary=_prepare_scene_grouping_summary(scene_id, scene_split),
                scene_cache=None,
                attachment_review_record=_make_attachment_review_record(
                    "no_frame_candidates"
                ),
                attachment_pair_salvage_review_record=_build_attachment_pair_record(
                    "no_frame_candidates",
                    None,
                ),
            )

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
            preloaded_geometry=preloaded_geometry,
        )
        scene_grouping_summary = _prepare_scene_grouping_summary(scene_id, scene_split)
        attachment_pair_salvage_scene_review: dict[str, Any] | None = None
        _cache_miss = object()
        scene_image_cache: dict[str, np.ndarray | None] = {}
        scene_depth_cache: dict[str, np.ndarray | None] = {}
        scene_image_b64_cache: dict[str, str | None] = {}
        scene_visibility_cache: dict[str, dict[int, dict[str, Any]] | None] = {}
        scene_out_of_frame_review_cache: dict[str, dict[str, Any] | None] = {}
        frame_clarity_cache: dict[str, dict[str, Any] | None] = {}
        referability_entry_cache: dict[str, dict[str, Any] | None] = {}
        scene_frame_sidecar_cache = _load_frame_sidecar_scene_cache(
            output_path=output_path,
            scene_id=scene_id,
            model_name=model_name,
            referability_backend=REFERABILITY_BACKEND,
        )
        sidecar_dirty = False

        def _build_mesh_ray_failure_result(exc: MeshRayRequiredError) -> SceneWorkerResult:
            logger.warning("Mesh-ray required failure for %s: %s", scene_id, exc)
            scene_grouping_summary["pipeline_outcome"] = "mesh_ray_failed"
            scene_grouping_summary["scene_skip_reason"] = "mesh_ray_failed"
            return _build_result(
                pipeline_outcome="mesh_ray_failed",
                scene_skip_reason="mesh_ray_failed",
                scene_grouping_summary=scene_grouping_summary,
                scene_cache=None,
                attachment_review_record=_make_attachment_review_record(
                    "mesh_ray_failed"
                ),
                attachment_pair_salvage_review_record=_build_attachment_pair_record(
                    "mesh_ray_failed",
                    attachment_pair_salvage_scene_review,
                ),
                frame_sidecar_cache=scene_frame_sidecar_cache if sidecar_dirty else None,
            )

        def _should_reuse_cached_referability_entry(entry: Any) -> bool:
            return isinstance(entry, dict) and (
                str(entry.get("candidate_visibility_source", "")).strip().lower()
                != "projection_fallback"
            )

        def _load_scene_image(image_name: str) -> np.ndarray | None:
            cached_image = scene_image_cache.get(image_name, _cache_miss)
            if cached_image is _cache_miss:
                image_path = scene_dir / "color" / image_name
                cached_image = cv2.imread(str(image_path))
                if cached_image is None:
                    logger.warning("Cannot read image %s", image_path)
                scene_image_cache[image_name] = cached_image
            return cached_image if isinstance(cached_image, np.ndarray) else None

        def _load_scene_depth_image(image_name: str) -> np.ndarray | None:
            cached_depth = scene_depth_cache.get(image_name, _cache_miss)
            if cached_depth is not _cache_miss:
                return cached_depth if isinstance(cached_depth, np.ndarray) else None

            depth_image = None
            frame_id = Path(image_name).stem
            depth_path = scene_dir / "depth" / f"{frame_id}.png"
            if depth_intrinsics is not None and depth_path.exists():
                try:
                    depth_image = load_depth_image(depth_path)
                except Exception as exc:
                    logger.warning("Depth load failed for %s/%s: %s", scene_id, image_name, exc)
            scene_depth_cache[image_name] = depth_image
            return depth_image

        def _get_scene_image_b64(image_name: str) -> str | None:
            cached_image_b64 = scene_image_b64_cache.get(image_name, _cache_miss)
            if cached_image_b64 is _cache_miss:
                image = _load_scene_image(image_name)
                cached_image_b64 = None if image is None else _image_to_base64(image)
                scene_image_b64_cache[image_name] = cached_image_b64
            return cached_image_b64 if isinstance(cached_image_b64, str) else None

        def _update_scene_frame_sidecar_record(
            image_name: str,
            *,
            frame_info: dict[str, Any] | None = None,
            frame_selection_score: int | None = None,
            referability_entry: dict[str, Any] | None | object = _cache_miss,
        ) -> None:
            nonlocal sidecar_dirty
            if not image_name:
                return
            current_record = scene_frame_sidecar_cache.get(image_name)
            updated_record = (
                dict(current_record)
                if isinstance(current_record, dict)
                else {
                    "frame_info": None,
                    "frame_selection_score": None,
                    "referability_entry": None,
                }
            )
            changed = False
            if isinstance(frame_info, dict):
                normalized_frame_info = _normalize_frame_review(frame_info)
                if updated_record.get("frame_info") != normalized_frame_info:
                    updated_record["frame_info"] = normalized_frame_info
                    changed = True
            if frame_selection_score is not None:
                normalized_selection_score = int(frame_selection_score)
                if updated_record.get("frame_selection_score") != normalized_selection_score:
                    updated_record["frame_selection_score"] = normalized_selection_score
                    changed = True
            if referability_entry is not _cache_miss:
                normalized_entry = (
                    dict(referability_entry)
                    if isinstance(referability_entry, dict)
                    else None
                )
                if updated_record.get("referability_entry") != normalized_entry:
                    updated_record["referability_entry"] = normalized_entry
                    changed = True
            if changed:
                scene_frame_sidecar_cache[image_name] = updated_record
                sidecar_dirty = True

        def _get_scene_visibility_by_obj_id(
            image_name: str,
        ) -> dict[int, dict[str, Any]] | None:
            cached_visibility = scene_visibility_cache.get(image_name, _cache_miss)
            if cached_visibility is not _cache_miss:
                return cached_visibility if isinstance(cached_visibility, dict) else None

            camera_pose = poses.get(image_name)
            if camera_pose is None:
                scene_visibility_cache[image_name] = None
                return None
            cached_visibility = compute_referability_object_visibility(
                scene["objects"],
                camera_pose,
                color_intrinsics,
            )
            scene_visibility_cache[image_name] = cached_visibility
            return cached_visibility if isinstance(cached_visibility, dict) else None

        def _get_scene_out_of_frame_review(image_name: str) -> dict[str, Any] | None:
            cached_review = scene_out_of_frame_review_cache.get(image_name, _cache_miss)
            if cached_review is not _cache_miss:
                return cached_review if isinstance(cached_review, dict) else None

            image = _load_scene_image(image_name)
            camera_pose = poses.get(image_name)
            visibility_by_obj_id = _get_scene_visibility_by_obj_id(image_name)
            image_b64 = _get_scene_image_b64(image_name)
            if (
                image is None
                or camera_pose is None
                or not isinstance(visibility_by_obj_id, dict)
                or not isinstance(image_b64, str)
            ):
                scene_out_of_frame_review_cache[image_name] = None
                return None
            cached_review = _review_out_of_frame_label_candidates(
                client=scene_client,
                model_name=model_name,
                image=image,
                image_b64=image_b64,
                scene_objects=scene["objects"],
                objects_by_id=objects_by_id,
                visibility_by_obj_id=visibility_by_obj_id,
                camera_pose=camera_pose,
                color_intrinsics=color_intrinsics,
                instance_mesh_data_getter=instance_mesh_data_getter,
            )
            scene_out_of_frame_review_cache[image_name] = cached_review
            return cached_review if isinstance(cached_review, dict) else None

        def _get_referability_entry_by_image_name(
            image_name: str,
            *,
            frame: dict[str, Any] | None = None,
            reviewed_frame: dict[str, Any] | None = None,
        ) -> dict[str, Any] | None:
            if not image_name or image_name not in poses:
                return None

            cached_entry = referability_entry_cache.get(image_name, _cache_miss)
            if cached_entry is _cache_miss:
                sidecar_record = scene_frame_sidecar_cache.get(image_name)
                sidecar_entry = (
                    sidecar_record.get("referability_entry")
                    if isinstance(sidecar_record, dict)
                    else None
                )
                if _should_reuse_cached_referability_entry(sidecar_entry):
                    cached_entry = dict(sidecar_entry)
                    referability_entry_cache[image_name] = cached_entry
                    scene_out_of_frame_review_cache.setdefault(
                        image_name,
                        _extract_out_of_frame_review_payload(cached_entry),
                    )
            if cached_entry is not _cache_miss:
                return dict(cached_entry) if isinstance(cached_entry, dict) else None

            if not isinstance(frame, dict):
                return None
            if not isinstance(reviewed_frame, dict):
                reviewed_frame = _get_reviewed_frame(frame)
            if not isinstance(reviewed_frame, dict):
                referability_entry_cache[image_name] = None
                return None

            image = _load_scene_image(image_name)
            if image is None:
                referability_entry_cache[image_name] = None
                return None

            selector_visible_object_ids = [
                int(obj_id)
                for obj_id in frame.get("visible_object_ids", [])
                if int(obj_id) in objects_by_id
            ]
            visibility_by_obj_id = _get_scene_visibility_by_obj_id(image_name)
            if visibility_by_obj_id is None:
                referability_entry_cache[image_name] = None
                return None
            frame_info = reviewed_frame.get("frame_info", _selector_quality_pass_frame_info())
            frame_selection_score = int(
                reviewed_frame.get(
                    "frame_selection_score",
                    frame.get("selector_score", frame.get("score", 0)),
                ) or 0
            )
            image_b64 = _get_scene_image_b64(image_name)
            entry = _compute_frame_referability_entry(
                client=scene_client,
                model_name=model_name,
                scene_objects=scene["objects"],
                objects_by_id=objects_by_id,
                image=image,
                image_path=scene_dir / "color" / image_name,
                camera_pose=poses[image_name],
                color_intrinsics=color_intrinsics,
                depth_image=_load_scene_depth_image(image_name),
                depth_intrinsics=depth_intrinsics,
                selector_visible_object_ids=selector_visible_object_ids,
                selector_score=int(
                    frame.get("selector_score", frame.get("score", len(selector_visible_object_ids))) or 0
                ),
                frame_info=frame_info,
                frame_selection_score=frame_selection_score,
                image_b64=image_b64,
                visibility_by_obj_id=visibility_by_obj_id,
                out_of_frame_review=(
                    _get_scene_out_of_frame_review(image_name)
                    if bool(frame_info.get("frame_usable", True))
                    else None
                ),
                vlm_workers=int(args.vlm_workers),
                ray_caster_getter=ray_caster_getter,
                instance_mesh_data_getter=instance_mesh_data_getter,
            )
            referability_entry_cache[image_name] = dict(entry) if isinstance(entry, dict) else None
            if isinstance(entry, dict):
                scene_out_of_frame_review_cache[image_name] = _extract_out_of_frame_review_payload(entry)
                _update_scene_frame_sidecar_record(
                    image_name,
                    frame_info=frame_info,
                    frame_selection_score=frame_selection_score,
                    referability_entry=entry,
                )
            return dict(entry) if isinstance(entry, dict) else None

        def _prime_frame_sidecar_record(
            image_name: str,
        ) -> tuple[dict[str, Any] | None, dict[str, Any] | None | object]:
            cached_frame_info = frame_clarity_cache.get(image_name, _cache_miss)
            sidecar_record = (
                scene_frame_sidecar_cache.get(image_name)
                if isinstance(scene_frame_sidecar_cache.get(image_name), dict)
                else None
            )
            if cached_frame_info is _cache_miss and isinstance(sidecar_record, dict):
                sidecar_frame_info = sidecar_record.get("frame_info")
                if isinstance(sidecar_frame_info, dict):
                    cached_frame_info = dict(sidecar_frame_info)
                    frame_clarity_cache[image_name] = cached_frame_info
                sidecar_entry = sidecar_record.get("referability_entry")
                if _should_reuse_cached_referability_entry(sidecar_entry):
                    referability_entry_cache.setdefault(image_name, dict(sidecar_entry))
                    scene_out_of_frame_review_cache.setdefault(
                        image_name,
                        _extract_out_of_frame_review_payload(sidecar_entry),
                    )
            return sidecar_record, cached_frame_info

        def _build_reviewed_frame_from_frame_info(
            frame: dict[str, Any],
            *,
            frame_info: dict[str, Any],
            sidecar_record: dict[str, Any] | None = None,
        ) -> dict[str, Any]:
            image_name = str(frame.get("image_name", "")).strip()
            selector_score = int(
                frame.get("selector_score", frame.get("score", frame.get("n_visible", 0))) or 0
            )
            frame_selection_score = (
                int(sidecar_record.get("frame_selection_score"))
                if isinstance(sidecar_record, dict)
                and sidecar_record.get("frame_selection_score") is not None
                else _frame_selection_score(selector_score, frame_info)
            )
            _update_scene_frame_sidecar_record(
                image_name,
                frame_info=frame_info,
                frame_selection_score=frame_selection_score,
            )
            return {
                **frame,
                "selector_score": selector_score,
                "frame_info": dict(frame_info),
                "frame_selection_score": frame_selection_score,
            }

        def _get_reviewed_frames(
            frames: list[dict[str, Any]],
        ) -> dict[str, dict[str, Any] | None]:
            reviewed_by_image_name: dict[str, dict[str, Any] | None] = {}
            uncached_batch_items: list[dict[str, Any]] = []
            uncached_frames_by_image_name: dict[str, dict[str, Any]] = {}

            for frame in frames:
                image_name = str(frame.get("image_name", "")).strip()
                if not image_name or image_name in reviewed_by_image_name:
                    continue

                sidecar_record, cached_frame_info = _prime_frame_sidecar_record(image_name)
                if cached_frame_info is _cache_miss:
                    image = _load_scene_image(image_name)
                    if image is None:
                        frame_clarity_cache[image_name] = None
                        reviewed_by_image_name[image_name] = None
                        continue
                    uncached_frames_by_image_name[image_name] = frame
                    uncached_batch_items.append(
                        {
                            "image_name": image_name,
                            "image": image,
                            "image_b64": _get_scene_image_b64(image_name),
                        }
                    )
                    continue
                if not isinstance(cached_frame_info, dict):
                    reviewed_by_image_name[image_name] = None
                    continue
                reviewed_by_image_name[image_name] = _build_reviewed_frame_from_frame_info(
                    frame,
                    frame_info=cached_frame_info,
                    sidecar_record=sidecar_record,
                )

            for batch in _chunk_list(uncached_batch_items, int(args.frame_clarity_batch_size)):
                batch_frame_info = _frame_decision_batch(
                    scene_client,
                    model_name,
                    batch,
                )
                for batch_item in batch:
                    image_name = str(batch_item.get("image_name", "")).strip()
                    frame = uncached_frames_by_image_name.get(image_name)
                    if not image_name or not isinstance(frame, dict):
                        continue
                    frame_info = batch_frame_info.get(image_name)
                    frame_clarity_cache[image_name] = dict(frame_info) if isinstance(frame_info, dict) else None
                    if not isinstance(frame_info, dict):
                        reviewed_by_image_name[image_name] = None
                        continue
                    reviewed_by_image_name[image_name] = _build_reviewed_frame_from_frame_info(
                        frame,
                        frame_info=frame_info,
                    )
            return reviewed_by_image_name

        def _get_reviewed_frame(frame: dict[str, Any]) -> dict[str, Any] | None:
            image_name = str(frame.get("image_name", "")).strip()
            if not image_name:
                return None
            return _get_reviewed_frames([frame]).get(image_name)

        attachment_candidate_frames = [
            frame
            for frame in frame_candidates
            if bool(frame.get("attachment_viewpoint_exempt"))
        ]
        non_attachment_candidate_frames = [
            dict(frame)
            for frame in frame_candidates
            if not bool(frame.get("attachment_viewpoint_exempt"))
        ]
        for frame in non_attachment_candidate_frames:
            image_name = str(frame.get("image_name", "")).strip()
            visibility_by_obj_id = _get_scene_visibility_by_obj_id(image_name)
            failed_signature_candidate = _geometry_signature_object_ids(
                visibility_by_obj_id,
                bbox_in_frame_ratio_min=REFERABLE_BBOX_IN_FRAME_RATIO_MIN,
                projected_area_px_min=QUESTION_REVIEW_CROP_MIN_PROJECTED_AREA_PX,
            )
            frame["failed_signature_candidate_object_ids"] = list(
                failed_signature_candidate
            )
        non_attachment_group_count = _count_visible_object_frame_groups(
            non_attachment_candidate_frames,
            poses=poses,
        )
        logger.info(
            "Selected %d attachment-qualified and %d non-attachment frame candidates across %d visible-object groups for %s before VLM review",
            len(attachment_candidate_frames),
            len(non_attachment_candidate_frames),
            non_attachment_group_count,
            scene_id,
        )

        def _get_referability_entry(
            frame: dict[str, Any],
            reviewed_frame: dict[str, Any],
        ) -> dict[str, Any] | None:
            image_name = str(frame.get("image_name", "")).strip()
            return _get_referability_entry_by_image_name(
                image_name,
                frame=frame,
                reviewed_frame=reviewed_frame,
            )

        scene_grouping_summary["non_attachment_candidate_frame_count"] = len(non_attachment_candidate_frames)
        scene_grouping_summary["non_attachment_visible_object_group_count"] = non_attachment_group_count
        try:
            non_attachment_frames = _select_and_rerank_frames(
                client=scene_client,
                model_name=model_name,
                scene_dir=scene_dir,
                frame_candidates=non_attachment_candidate_frames,
                max_frames=int(args.max_frames),
                poses=poses,
                vlm_workers=int(args.vlm_workers),
                frame_review_getter=_get_reviewed_frame,
                frame_review_batch_getter=_get_reviewed_frames,
                referability_entry_builder=_get_referability_entry,
                debug_output=scene_grouping_summary,
                frame_clarity_batch_size=int(args.frame_clarity_batch_size),
                non_attachment_referability_shortlist=int(args.non_attachment_referability_shortlist),
                non_attachment_clarity_min_score=int(args.non_attachment_clarity_min_score),
            ) if non_attachment_candidate_frames else []
        except MeshRayRequiredError as exc:
            return _build_mesh_ray_failure_result(exc)

        def _build_attachment_entry(
            frame: dict[str, Any],
            reviewed_frame: dict[str, Any],
        ) -> dict[str, Any] | None:
            return _get_referability_entry(frame, reviewed_frame)

        attachment_failed_signatures_seen: set[tuple[int, ...]] = set()
        try:
            attachment_selected_frames = _select_attachment_group_representatives(
                client=scene_client,
                model_name=model_name,
                scene_dir=scene_dir,
                frames=attachment_candidate_frames,
                attachment_graph=attachment_graph,
                poses=poses,
                frame_review_getter=_get_reviewed_frame,
                frame_review_batch_getter=_get_reviewed_frames,
                attachment_entry_builder=_build_attachment_entry,
                max_accepted_frame_count=int(args.max_frames),
                vlm_workers=int(args.vlm_workers),
                frame_clarity_batch_size=int(args.frame_clarity_batch_size),
                attachment_clarity_min_score=int(args.attachment_clarity_min_score),
                failed_signatures_seen=attachment_failed_signatures_seen,
            ) if attachment_candidate_frames else []
            if args.write_attachment_pair_salvage_review and attachment_candidate_frames:
                attachment_pair_salvage_scene_review = _build_attachment_pair_salvage_scene_review(
                    client=scene_client,
                    model_name=model_name,
                    scene_id=scene_id,
                    split=scene_split,
                    scene_dir=scene_dir,
                    objects=scene["objects"],
                    objects_by_id=objects_by_id,
                    attachment_graph=attachment_graph,
                    attachment_edges=final_attachment_edges,
                    frames=attachment_candidate_frames,
                    poses=poses,
                    frame_review_getter=_get_reviewed_frame,
                    frame_review_batch_getter=_get_reviewed_frames,
                    scene_image_getter=_load_scene_image,
                    attachment_entry_builder=_build_attachment_entry,
                    bbox_hard_fail_min=float(args.attachment_pair_salvage_bbox_hard_fail_min),
                    projected_area_hard_fail_min=float(
                        args.attachment_pair_salvage_projected_area_hard_fail_min
                    ),
                    frame_clarity_batch_size=int(args.frame_clarity_batch_size),
                    attachment_clarity_min_score=int(args.attachment_clarity_min_score),
                    failed_signatures_seen=attachment_failed_signatures_seen,
                )
        except MeshRayRequiredError as exc:
            return _build_mesh_ray_failure_result(exc)
        selected_attachment_frames = attachment_selected_frames
        remaining_slots = max(0, int(args.max_frames) - len(selected_attachment_frames))
        selected_non_attachment_frames = non_attachment_frames[:remaining_slots]
        selected_before_attachment = set(
            scene_grouping_summary.get("selected_before_attachment_slots_image_names", [])
        )
        selected_after_attachment = {
            str(frame.get("image_name", "")).strip()
            for frame in selected_non_attachment_frames
        }
        scene_grouping_summary["attachment_selected_frame_image_names"] = [
            str(frame.get("image_name", "")).strip()
            for frame in selected_attachment_frames
        ]
        scene_grouping_summary["attachment_selected_frame_count"] = len(selected_attachment_frames)
        scene_grouping_summary["remaining_slots_after_attachment_selection"] = remaining_slots
        scene_grouping_summary["selected_after_attachment_slots_image_names"] = [
            str(frame.get("image_name", "")).strip()
            for frame in selected_non_attachment_frames
        ]
        scene_grouping_summary["selected_after_attachment_slots_count"] = len(selected_non_attachment_frames)
        for group in scene_grouping_summary.get("groups", []):
            accepted_names = list(group.get("accepted_frame_image_names", []))
            selected_after = [
                image_name for image_name in accepted_names
                if image_name in selected_after_attachment
            ]
            dropped_after_attachment = [
                image_name for image_name in accepted_names
                if image_name in selected_before_attachment
                and image_name not in selected_after_attachment
            ]
            group["selected_after_attachment_slots_image_names"] = selected_after
            group["dropped_after_attachment_slots_image_names"] = dropped_after_attachment
            if not accepted_names:
                if bool(group.get("group_exhausted_without_usable_frame", False)):
                    group["status_after_attachment_slots"] = "no_usable_frame"
                else:
                    group["status_after_attachment_slots"] = "no_referable_frame"
            elif selected_after:
                group["status_after_attachment_slots"] = "final_selected"
            elif dropped_after_attachment:
                group["status_after_attachment_slots"] = "dropped_by_attachment_slot_limit"
            else:
                group["status_after_attachment_slots"] = str(
                    group.get("status_before_attachment_slots", "dropped_by_group_rerank")
                )

        if not selected_attachment_frames and not selected_non_attachment_frames:
            scene_grouping_summary["pipeline_outcome"] = "no_final_referability_frames"
            scene_grouping_summary["scene_skip_reason"] = "no_final_referability_frames"
            logger.info("Scene %s has no final referability frames -> skipping", scene_id)
            return _build_result(
                pipeline_outcome="no_final_referability_frames",
                scene_skip_reason="no_final_referability_frames",
                scene_grouping_summary=scene_grouping_summary,
                scene_cache=None,
                attachment_review_record=_make_attachment_review_record(
                    "no_final_referability_frames"
                ),
                attachment_pair_salvage_review_record=_build_attachment_pair_record(
                    "no_final_referability_frames",
                    attachment_pair_salvage_scene_review,
                ),
                frame_sidecar_cache=scene_frame_sidecar_cache if sidecar_dirty else None,
            )

        logger.info(
            "Processing referability scene %s [split=%s] with %d attachment-selected frame(s) and %d non-attachment fallback(s)",
            scene_id,
            scene_split,
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
                attachment_selector_pair_count=frame.get("attachment_pair_ge_50_count", 0),
                attachment_selector_viewpoint_exempt=frame.get("attachment_viewpoint_exempt", False),
            )
            final_selection_rank += 1

        for frame in selected_non_attachment_frames:
            image_name = str(frame.get("image_name", "")).strip()
            if not image_name or image_name not in poses:
                continue

            cached_entry = frame.get("_referability_entry")
            if isinstance(cached_entry, dict):
                entry = dict(cached_entry)
            else:
                reviewed_frame = dict(frame)
                if not isinstance(reviewed_frame.get("frame_info"), dict):
                    reviewed_frame = _get_reviewed_frame(frame)
                if reviewed_frame is None:
                    continue
                try:
                    entry = _get_referability_entry(frame, reviewed_frame)
                except MeshRayRequiredError as exc:
                    return _build_mesh_ray_failure_result(exc)
                if not isinstance(entry, dict):
                    continue
            final_scene_entries[image_name] = _attach_selection_metadata(
                entry,
                attachment_graph,
                final_selection_rank=final_selection_rank,
                attachment_selector_pair_count=frame.get("attachment_pair_ge_50_count", 0),
                attachment_selector_viewpoint_exempt=frame.get("attachment_viewpoint_exempt", False),
            )
            final_selection_rank += 1

        if not final_scene_entries:
            scene_grouping_summary["pipeline_outcome"] = "no_cacheable_referability_entries"
            scene_grouping_summary["scene_skip_reason"] = "no_cacheable_referability_entries"
            logger.info("Scene %s produced no cacheable referability entries -> skipping", scene_id)
            return _build_result(
                pipeline_outcome="no_cacheable_referability_entries",
                scene_skip_reason="no_cacheable_referability_entries",
                scene_grouping_summary=scene_grouping_summary,
                scene_cache=None,
                attachment_review_record=_make_attachment_review_record(
                    "no_cacheable_referability_entries"
                ),
                attachment_pair_salvage_review_record=_build_attachment_pair_record(
                    "no_cacheable_referability_entries",
                    attachment_pair_salvage_scene_review,
                ),
                frame_sidecar_cache=scene_frame_sidecar_cache if sidecar_dirty else None,
            )

        try:
            final_scene_entries = _enrich_final_scene_entries_out_of_frame(
                client=scene_client,
                model_name=model_name,
                scene_dir=scene_dir,
                final_scene_entries=final_scene_entries,
                scene_objects=scene["objects"],
                objects_by_id=objects_by_id,
                poses=poses,
                color_intrinsics=color_intrinsics,
                depth_intrinsics=depth_intrinsics,
                referability_entry_getter=lambda image_name: _get_referability_entry_by_image_name(image_name),
                instance_mesh_data_getter=instance_mesh_data_getter,
            )
        except MeshRayRequiredError as exc:
            return _build_mesh_ray_failure_result(exc)

        scene_cache: dict[str, dict[str, Any]] = {}
        for image_name, entry in sorted(
            final_scene_entries.items(),
            key=lambda item: int(item[1].get("final_selection_rank", FRAME_SELECTION_FALLBACK_RANK)),
        ):
            scene_cache[image_name] = entry
        scene_grouping_summary["pipeline_outcome"] = "processed"
        scene_grouping_summary["scene_skip_reason"] = None
        scene_grouping_summary["final_cacheable_frame_image_names"] = [
            str(image_name)
            for image_name in scene_cache.keys()
        ]
        scene_grouping_summary["final_cacheable_frame_count"] = len(scene_cache)
        return _build_result(
            pipeline_outcome="processed",
            scene_skip_reason=None,
            scene_grouping_summary=scene_grouping_summary,
            scene_cache=scene_cache,
            attachment_review_record=_make_attachment_review_record("processed"),
            attachment_pair_salvage_review_record=_build_attachment_pair_record(
                "processed",
                attachment_pair_salvage_scene_review,
            ),
            frame_sidecar_cache=scene_frame_sidecar_cache if sidecar_dirty else None,
        )

    if scene_number_range is not None:
        shard_start, shard_end = scene_number_range
        selected_scene_entries = _select_scene_entries_by_closed_range(
            scene_entries,
            start=shard_start,
            end=shard_end,
        )
        logger.info(
            "Fixed scene shard for split=%s interval=%d-%d resolved to %d scene(s)",
            selected_split,
            shard_start,
            shard_end,
            len(selected_scene_entries),
        )
    else:
        selected_scene_entries = list(scene_entries)
    selected_scene_ids = [scene_dir.name for _, scene_dir in selected_scene_entries]
    scene_index_by_id = {
        scene_dir.name: index
        for index, (_scene_split, scene_dir) in enumerate(selected_scene_entries, start=1)
    }
    pending_scene_entries = [
        (scene_split, scene_dir)
        for scene_split, scene_dir in selected_scene_entries
        if scene_dir.name not in completed_scene_ids
    ]

    final_batch_mode = False
    if scene_number_range is not None:
        requested_scene_count = (scene_number_range[1] - scene_number_range[0]) + 1
        remaining_unprocessed = len(pending_scene_entries)
        if 0 < remaining_unprocessed < requested_scene_count:
            final_batch_mode = True
            _log_final_batch_banner(
                split=selected_split,
                total_scene_count=len(selected_scene_entries),
                processed_scene_count=len(selected_scene_entries) - remaining_unprocessed,
                remaining_scene_count=remaining_unprocessed,
            )

    if scene_number_range is not None:
        target_scene_entries = pending_scene_entries
    else:
        target_scene_entries = pending_scene_entries[:max(0, int(args.max_scenes))]

    if not target_scene_entries:
        logger.info(
            "No unprocessed scenes remain for split=%s according to %s",
            selected_split,
            scene_status_path,
        )
    else:
        scene_worker_count = min(int(args.scene_workers), len(target_scene_entries))
        reorder_buffer: dict[int, SceneWorkerResult] = {}
        next_commit_position = 0
        next_submit_position = 0
        executor = ThreadPoolExecutor(max_workers=scene_worker_count)
        in_flight: dict[Any, int] = {}
        try:
            while next_submit_position < scene_worker_count:
                scene_split, scene_dir = target_scene_entries[next_submit_position]
                future = executor.submit(
                    _process_scene_worker,
                    next_submit_position,
                    scene_split,
                    scene_dir,
                )
                in_flight[future] = next_submit_position
                next_submit_position += 1

            while in_flight:
                completed_future = next(as_completed(list(in_flight.keys())))
                completed_position = in_flight.pop(completed_future)
                reorder_buffer[completed_position] = completed_future.result()
                while next_commit_position in reorder_buffer:
                    _commit_scene_result(reorder_buffer.pop(next_commit_position))
                    next_commit_position += 1
                if next_submit_position < len(target_scene_entries):
                    scene_split, scene_dir = target_scene_entries[next_submit_position]
                    future = executor.submit(
                        _process_scene_worker,
                        next_submit_position,
                        scene_split,
                        scene_dir,
                    )
                    in_flight[future] = next_submit_position
                    next_submit_position += 1
        except Exception:
            for future in in_flight:
                future.cancel()
            executor.shutdown(wait=False, cancel_futures=True)
            raise
        else:
            executor.shutdown(wait=True, cancel_futures=False)

    batch_scene_count = len(scene_status_cache)
    if batch_scene_count > 0:
        _write_json_payload(output_path, cache)
        _write_attachment_review()
        _write_attachment_pair_salvage_review()
    if scene_number_range is not None and final_batch_mode:
        completed_scene_ids_after_run = set(
            scene_status_doc.get("completed_scenes", {}).keys()
        )
        remaining_unprocessed = sum(
            1 for scene_id in selected_scene_ids
            if scene_id not in completed_scene_ids_after_run
        )
        _log_final_batch_banner(
            split=selected_split,
            total_scene_count=len(selected_scene_entries),
            processed_scene_count=len(selected_scene_entries) - remaining_unprocessed,
            remaining_scene_count=remaining_unprocessed,
            completed=True,
        )
    if batch_scene_count > 0:
        logger.info("Saved referability batch cache to %s", output_path)
    else:
        logger.info(
            "No new referability batch cache written for split=%s; scene_status remains at %s",
            selected_split,
            scene_status_path,
        )
    logger.info("VLM call failures: %d", _get_vlm_call_failure_count())


if __name__ == "__main__":
    main()
