#!/usr/bin/env python3
"""Sample benchmark questions by type, run a VLM, and build an HTML report.

The script scans benchmark.json files under one or more output roots, samples
up to N questions per question type, evaluates them through an OpenAI-compatible
vision API, and writes both a resumable JSON result file and a self-contained
HTML viewer.

This script is an API client only: it does not load model weights from a local
filesystem path. If you moved local model files (for example from
``/home/sujinyue/mybenchmark/models`` to ``/home/sujinyue/models``), update the
model server behind ``--base_url`` instead; this script only needs the served
``--model`` name.

Example:
    python scripts/run_sampled_type_vlm_eval.py \
        --root output/pilot \
        --root output/scannetpp_polit \
        --scannet_image_root data/scannet \
        --scannetpp_image_root output/scannetpp_iphone_frames \
        --scannetpp_sensor iphone \
        --vlm_url https://www.packyapi.com/v1 \
        --vlm_model qwen3.5-flash \
        --output_json output/type_sample_eval/results.json \
        --output_html output/type_sample_eval/viewer.html

BEV example:
    python scripts/run_sampled_type_vlm_eval.py \
        --benchmark_file output/multi_image_questions.json \
        --bev \
        --bev_dir output/multi_image_questions_bev \
        --output_json output/type_sample_eval/bev_results.json \
        --output_html output/type_sample_eval/bev_viewer.html
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import html
import json
import mimetypes
import os
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter, defaultdict
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from scripts.enrich_oracle_fields import (
        SceneCacheEntry as OracleSceneCacheEntry,
        _dataset_kind as _oracle_dataset_kind,
        _frame_kind_for_question as _oracle_frame_kind_for_question,
        _load_poses as _oracle_load_poses,
        _make_task_frame_oracle_prefix,
        _make_world_oracle_prefix,
        _scene_path as _oracle_scene_path,
    )
    from src.scene_parser import parse_scene as _parse_oracle_scene
except ImportError:  # pragma: no cover - optional when only using pre-enriched oracle fields
    OracleSceneCacheEntry = None  # type: ignore[assignment]
    _oracle_dataset_kind = None  # type: ignore[assignment]
    _oracle_frame_kind_for_question = None  # type: ignore[assignment]
    _oracle_load_poses = None  # type: ignore[assignment]
    _make_task_frame_oracle_prefix = None  # type: ignore[assignment]
    _make_world_oracle_prefix = None  # type: ignore[assignment]
    _oracle_scene_path = None  # type: ignore[assignment]
    _parse_oracle_scene = None  # type: ignore[assignment]

try:
    from PIL import Image
except ImportError:  # pragma: no cover - optional dependency
    Image = None

from scripts.make_viewer import _collect_aux_image_names


SYSTEM_PROMPT = (
    "You are a careful vision-language assistant solving multiple-choice "
    "spatial reasoning questions about an image."
)

PROMPT_SUFFIX = (
    "Work through your reasoning briefly, then give your final choice as the LAST line.\n"
    "Keep your reasoning as short as possible (a few sentences at most) so the final Answer line is not truncated.\n"
    "Return this format:\n"
    "Reasoning: <brief reasoning>\n"
    "Answer: <single letter>"
)

MULTI_SELECT_PROMPT_SUFFIX = (
    "Work through your reasoning briefly, then give your final choice as the LAST line.\n"
    "If more than one option is correct, list all letters comma-separated.\n"
    "Keep your reasoning as short as possible (a few sentences at most) so the final Answer line is not truncated.\n"
    "Return this format:\n"
    "Reasoning: <brief reasoning>\n"
    "Answer: <letter(s)>"
)

DIRECT_PROMPT_SUFFIX = "Answer with a single letter only (A, B, C, or D). Do not explain."
DIRECT_MULTI_SELECT_PROMPT_SUFFIX = (
    "Answer with the correct letter(s) only, comma-separated if more than one. Do not explain."
)

BLIND_PROMPT_INSTRUCTION = (
    "Images are intentionally unavailable for this benchmark condition.\n"
    "You must select exactly one option using only the question text and choices.\n"
    "Do not request an image and do not abstain."
)
BLIND_MULTI_SELECT_PROMPT_INSTRUCTION = (
    "Images are intentionally unavailable for this benchmark condition.\n"
    "You must select the required option(s) using only the question text and choices.\n"
    "Do not request an image and do not abstain."
)

QTYPE_ORDER = [
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
]

QTYPE_DISPLAY = {
    "direction_agent": "L1_direction_agent",
    "occlusion": "L1_occlusion",
    "distance": "L1_distance",
    "direction_object_centric": "L1_direction_object_centric",
    "direction_allocentric": "L1_direction_allocentric",
    "object_move_agent": "L2_object_move_agent",
    "object_move_distance": "L2_object_move_distance",
    "object_move_occlusion": "L2_object_move_occlusion",
    "object_move_object_centric": "L2_object_move_object_centric",
    "object_rotate_object_centric": "L2_object_rotate_object_centric",
    "object_move_allocentric": "L2_object_move_allocentric",
    "object_remove": "L2_object_remove",
    "attachment_chain": "L3_attachment_chain",
    "attachment_move": "L3_attachment_move",
    "coordinate_rotation_agent": "L3_coordinate_rotation_agent",
    "coordinate_rotation_object_centric": "L3_coordinate_rotation_object_centric",
    "coordinate_rotation_allocentric": "L3_coordinate_rotation_allocentric",
}


@dataclass(frozen=True)
class ImageResolution:
    path: Path | None
    checked_paths: tuple[str, ...]
    role: str | None = None


ROLLOUT_SCHEMA_VERSION = "predictive-spatial-rollout-v1"
ROLLOUT_MODES = ("picture", "video")
ROLLOUT_MEDIA_ROLES = {
    "source_view",
    "source_to_destination_bridge",
    "destination_environment",
    "predicted_future_view",
    "destination_to_query_bridge",
    "query_reference_view",
    "motion_reference_view",
    "predicted_video_frame",
}
ROLLOUT_ROLE_LABELS = {
    "source_view": "operation-before source view",
    "source_to_destination_bridge": "source-to-destination bridge view",
    "destination_environment": "operation-before destination environment",
    "predicted_future_view": "predicted future destination view after the operation",
    "destination_to_query_bridge": "destination-to-query bridge view",
    "query_reference_view": "static query-object reference view",
    "motion_reference_view": "operation-before motion reference view",
    "predicted_video_frame": "predicted video rollout frame",
}
BEV_SCHEMA_VERSION = "predictive-spatial-bev-v2"
BEV_DIRECTION_MODES = ("none", "task", "task_ticks")
BEV_FRAME_KINDS = ("agent", "object_centric", "allocentric")
BEV_MEDIA_ROLE = "bev_initial_layout"
BEV_MEDIA_LABEL = "operation-before bird's-eye layout"
ROLLOUT_FORBIDDEN_KEYS = {
    "answer",
    "correct_answer",
    "correct_option",
    "correct_options",
    "correct_value",
    "correct_values",
    "future_3d_coordinate",
    "future_3d_coordinates",
    "future_3d_position",
    "future_3d_positions",
    "future_bbox",
    "future_bboxes",
    "future_box",
    "future_boxes",
    "future_projection",
    "future_projection_box",
    "gt_answer",
    "gt_future_position",
    "projected_future_bbox",
    "projected_future_box",
    "prompt",
    "request_payload",
    "response_payload",
}


@dataclass(frozen=True)
class RolloutManifest:
    path: Path
    entries: dict[str, dict[str, Any]]
    entry_order: tuple[str, ...]
    sha256: str


@dataclass(frozen=True)
class BevManifest:
    path: Path
    entries: dict[str, dict[str, Any]]
    sha256: str
    image_size_px: int
    direction_mode: str


def _find_forbidden_rollout_key(value: Any, location: str = "$") -> str | None:
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = str(key).strip().lower()
            child_location = f"{location}.{key}"
            geometry_leak = (
                normalized.startswith(("future_", "gt_future_"))
                and any(
                    token in normalized
                    for token in (
                        "coordinate",
                        "position",
                        "center",
                        "bbox",
                        "box",
                        "projection",
                    )
                )
            )
            visibility_leak = normalized in {
                "future_visibility",
                "future_visibility_ratio",
                "simulated_visibility",
                "simulated_visibility_ratio",
            }
            if normalized in ROLLOUT_FORBIDDEN_KEYS or geometry_leak or visibility_leak:
                return child_location
            found = _find_forbidden_rollout_key(child, child_location)
            if found:
                return found
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found = _find_forbidden_rollout_key(child, f"{location}[{index}]")
            if found:
                return found
    return None


def load_rollout_manifest(path: Path) -> RolloutManifest:
    raw_bytes = path.read_bytes()
    try:
        payload = json.loads(raw_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid rollout manifest JSON: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Rollout manifest root must be a JSON object")
    if payload.get("schema_version") != ROLLOUT_SCHEMA_VERSION:
        raise ValueError(
            f"Rollout manifest schema_version must be {ROLLOUT_SCHEMA_VERSION!r}"
        )
    forbidden = _find_forbidden_rollout_key(payload)
    if forbidden:
        raise ValueError(
            f"Rollout manifest contains forbidden GT/answer field at {forbidden}; "
            "put geometry and answer-bearing data in the separate audit manifest"
        )
    raw_entries = payload.get("entries")
    if not isinstance(raw_entries, list):
        raise ValueError("Rollout manifest entries must be a JSON array")

    entries: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for index, raw_entry in enumerate(raw_entries):
        if not isinstance(raw_entry, dict):
            raise ValueError(f"Rollout manifest entries[{index}] must be an object")
        uid = str(raw_entry.get("question_uid") or "").strip()
        if not uid:
            raise ValueError(f"Rollout manifest entries[{index}] has no question_uid")
        if re.search(r'[\{\"]\s*(?:answer|correct_value)[\"\s:]', uid, re.IGNORECASE):
            raise ValueError(
                f"Rollout manifest entries[{index}].question_uid embeds answer-bearing JSON; "
                "use the evaluator's opaque SHA-1 question_uid instead"
            )
        if uid in entries:
            raise ValueError(f"Duplicate rollout manifest question_uid: {uid}")
        for mode in ROLLOUT_MODES:
            branch = raw_entry.get(mode)
            if branch is None:
                continue
            if not isinstance(branch, dict):
                raise ValueError(f"entries[{index}].{mode} must be an object")
            if not isinstance(branch.get("eligible"), bool):
                raise ValueError(f"entries[{index}].{mode}.eligible must be boolean")
            rejection_reasons = branch.get("rejection_reasons", [])
            if not isinstance(rejection_reasons, list) or not all(
                isinstance(reason, str) for reason in rejection_reasons
            ):
                raise ValueError(
                    f"entries[{index}].{mode}.rejection_reasons must be a string array"
                )
            media = branch.get("media", [])
            if not isinstance(media, list):
                raise ValueError(f"entries[{index}].{mode}.media must be an array")
            for media_index, item in enumerate(media):
                prefix = f"entries[{index}].{mode}.media[{media_index}]"
                if not isinstance(item, dict):
                    raise ValueError(f"{prefix} must be an object")
                if not str(item.get("path") or "").strip():
                    raise ValueError(f"{prefix}.path is required")
                role = str(item.get("role") or "").strip()
                if role not in ROLLOUT_MEDIA_ROLES:
                    raise ValueError(
                        f"{prefix}.role must be one of {sorted(ROLLOUT_MEDIA_ROLES)}"
                    )
                kind = str(item.get("kind") or "").strip()
                if kind not in {"context", "prediction"}:
                    raise ValueError(f"{prefix}.kind must be 'context' or 'prediction'")
                if role in {"predicted_future_view", "predicted_video_frame"} and kind != "prediction":
                    raise ValueError(f"{prefix} uses a prediction role but kind is not 'prediction'")
                if role not in {"predicted_future_view", "predicted_video_frame"} and kind != "context":
                    raise ValueError(f"{prefix} uses a context role but kind is not 'context'")
        entries[uid] = raw_entry
        order.append(uid)

    return RolloutManifest(
        path=path.resolve(),
        entries=entries,
        entry_order=tuple(order),
        sha256=hashlib.sha256(raw_bytes).hexdigest(),
    )


def load_bev_manifest(bev_dir: Path) -> BevManifest:
    manifest_path = bev_dir / "bev_manifest.json"
    try:
        raw_bytes = manifest_path.read_bytes()
        payload = json.loads(raw_bytes.decode("utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"BEV manifest not found: {manifest_path}") from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid BEV manifest JSON: {manifest_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("BEV manifest root must be a JSON object")
    if payload.get("schema_version") != BEV_SCHEMA_VERSION:
        raise ValueError(f"BEV manifest schema_version must be {BEV_SCHEMA_VERSION!r}")
    direction_mode = str(payload.get("direction_mode") or "").strip()
    if direction_mode not in BEV_DIRECTION_MODES:
        raise ValueError(
            f"BEV manifest direction_mode must be one of {list(BEV_DIRECTION_MODES)}"
        )
    if int(payload.get("failure_count") or 0) != 0 or payload.get("failures"):
        raise ValueError("BEV manifest reports generation failures")
    raw_entries = payload.get("entries")
    if not isinstance(raw_entries, list):
        raise ValueError("BEV manifest entries must be a JSON array")
    try:
        image_size_px = int(payload.get("image_size_px"))
    except (TypeError, ValueError) as exc:
        raise ValueError("BEV manifest image_size_px must be a positive integer") from exc
    if image_size_px <= 0:
        raise ValueError("BEV manifest image_size_px must be a positive integer")
    if int(payload.get("generated_count", len(raw_entries))) != len(raw_entries):
        raise ValueError("BEV manifest generated_count does not match entries")

    entries: dict[str, dict[str, Any]] = {}
    seen_image_paths: set[Path] = set()
    root = bev_dir.resolve()
    for index, raw_entry in enumerate(raw_entries):
        if not isinstance(raw_entry, dict):
            raise ValueError(f"BEV manifest entries[{index}] must be an object")
        uid = str(raw_entry.get("question_uid") or "").strip()
        if not uid:
            raise ValueError(f"BEV manifest entries[{index}] has no question_uid")
        if uid in entries:
            raise ValueError(f"Duplicate BEV manifest question_uid: {uid}")
        task_frame_kind = str(raw_entry.get("task_frame_kind") or "").strip()
        if task_frame_kind not in BEV_FRAME_KINDS:
            raise ValueError(
                f"BEV manifest entries[{index}].task_frame_kind must be one of "
                f"{list(BEV_FRAME_KINDS)}"
            )
        raw_image_path = str(raw_entry.get("image_path") or "").strip()
        if not raw_image_path:
            raise ValueError(f"BEV manifest entries[{index}].image_path is required")
        image_path = Path(raw_image_path)
        image_sha256 = str(raw_entry.get("image_sha256") or "").strip().lower()
        if not re.fullmatch(r"[0-9a-f]{64}", image_sha256):
            raise ValueError(
                f"BEV manifest entries[{index}].image_sha256 must be a SHA-256 hex digest"
            )
        if image_path.is_absolute():
            raise ValueError(f"BEV manifest entries[{index}].image_path must be relative")
        resolved = (root / image_path).resolve()
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise ValueError(
                f"BEV manifest entries[{index}].image_path escapes --bev_dir"
            ) from exc
        if resolved in seen_image_paths:
            raise ValueError(f"Duplicate BEV manifest image_path: {raw_image_path}")
        seen_image_paths.add(resolved)
        entries[uid] = {**raw_entry, "_resolved_image_path": str(resolved)}

    return BevManifest(
        path=manifest_path.resolve(),
        entries=entries,
        sha256=hashlib.sha256(raw_bytes).hexdigest(),
        image_size_px=image_size_px,
        direction_mode=direction_mode,
    )


def _json_key(payload: Any) -> str:
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _question_uid(question: dict[str, Any]) -> str:
    return _json_key(
        {
            "dataset": question.get("_dataset"),
            "scene_id": question.get("scene_id"),
            "image_name": question.get("image_name"),
            "type": question.get("type"),
            "question": question.get("question"),
            "options": question.get("options"),
            "answer": question.get("answer"),
        }
    )


def _question_dedupe_key(question: dict[str, Any]) -> str:
    return _json_key(
        {
            "dataset": question.get("_dataset") or question.get("dataset"),
            "scene_id": question.get("scene_id"),
            "image_name": question.get("image_name"),
            "type": question.get("type"),
            "question": question.get("question"),
            "options": question.get("options"),
            "answer": question.get("answer"),
        }
    )


def _question_cache_key(question: dict[str, Any]) -> str:
    """Identify the model-visible question while excluding its ground-truth answer."""
    return _json_key(
        {
            "dataset": question.get("_dataset") or question.get("dataset"),
            "scene_id": question.get("scene_id"),
            "image_name": question.get("image_name"),
            "type": question.get("type"),
            "question": question.get("question"),
            "options": question.get("options"),
        }
    )


def _load_benchmark(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8-sig") as f:
        data = json.load(f)
    questions = data.get("questions", data) if isinstance(data, dict) else data
    if not isinstance(questions, list):
        raise ValueError(f"Unsupported benchmark structure: {path}")
    return [q for q in questions if isinstance(q, dict)]


def _infer_question_dataset(question: dict[str, Any], *source_paths: Path) -> str:
    explicit = str(question.get("_dataset") or question.get("dataset") or "").strip().lower()
    if explicit in {"scannet", "scannetpp"}:
        return explicit

    source_text = " ".join(
        [str(question.get("_source_benchmark") or ""), *(path.as_posix() for path in source_paths)]
    ).lower()
    if "scannetpp" in source_text:
        return "scannetpp"

    # ScanNet and ScanNet++ use unambiguous scene-id formats, which lets us
    # recover when a curated subset dropped or corrupted its dataset field.
    scene_id = str(question.get("scene_id") or "").strip()
    return "scannet" if re.fullmatch(r"scene\d{4}_\d{2}", scene_id) else "scannetpp"


def _load_questions_from_roots(roots: list[Path]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    questions: list[dict[str, Any]] = []
    seen: set[str] = set()
    source_files: list[str] = []
    duplicate_count = 0

    for root in roots:
        for benchmark_path in sorted(root.rglob("benchmark.json")):
            source_files.append(str(benchmark_path))
            for q in _load_benchmark(benchmark_path):
                item = dict(q)
                item["_dataset"] = _infer_question_dataset(item, root, benchmark_path)
                item["_source_root"] = str(root)
                item["_source_benchmark"] = str(benchmark_path)
                if item.get("question_uid") is not None:
                    item["_source_question_uid"] = str(item["question_uid"])
                uid = _question_uid(item)
                item["question_uid"] = uid
                dedupe_key = _question_dedupe_key(item)
                if dedupe_key in seen:
                    duplicate_count += 1
                    continue
                seen.add(dedupe_key)
                questions.append(item)

    metadata = {
        "source_files": source_files,
        "source_file_count": len(source_files),
        "deduped_question_count": len(questions),
        "duplicate_question_count": duplicate_count,
        "dedupe_rule": "dataset + scene_id + image_name + type + question + options + answer",
        "input_mode": "roots",
    }
    return questions, metadata


def load_questions_from_subset(subset_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    questions: list[dict[str, Any]] = []
    seen: set[str] = set()
    duplicate_count = 0

    for q in _load_benchmark(subset_path):
        item = dict(q)
        item["_dataset"] = _infer_question_dataset(item, subset_path)
        item["_source_root"] = str(subset_path.parent)
        item["_source_benchmark"] = str(subset_path)
        if item.get("question_uid") is not None:
            item["_source_question_uid"] = str(item["question_uid"])
        uid = _question_uid(item)
        item["question_uid"] = uid
        dedupe_key = _question_dedupe_key(item)
        if dedupe_key in seen:
            duplicate_count += 1
            continue
        seen.add(dedupe_key)
        questions.append(item)

    metadata = {
        "source_files": [str(subset_path)],
        "source_file_count": 1,
        "deduped_question_count": len(questions),
        "duplicate_question_count": duplicate_count,
        "dedupe_rule": "dataset + scene_id + image_name + type + question + options + answer",
        "input_mode": "subset",
        "subset_path": str(subset_path),
    }
    return questions, metadata


def load_questions(
    roots: list[Path], subset_path: Path | None = None
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if subset_path is not None:
        return load_questions_from_subset(subset_path)
    return _load_questions_from_roots(roots)


def load_fixed_questions(benchmark_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    with benchmark_path.open(encoding="utf-8-sig") as f:
        data = json.load(f)

    if isinstance(data, dict):
        raw_questions = data.get("questions", [])
        metadata = dict(data.get("metadata", {}))
        sampling_stats = dict(data.get("sampling_stats", {}))
    elif isinstance(data, list):
        raw_questions = data
        metadata = {}
        sampling_stats = {}
    else:
        raise ValueError(f"Unsupported benchmark structure: {benchmark_path}")

    if not isinstance(raw_questions, list):
        raise ValueError(f"Unsupported benchmark structure: {benchmark_path}")

    questions: list[dict[str, Any]] = []
    seen: set[str] = set()
    duplicate_count = 0
    for q in raw_questions:
        if not isinstance(q, dict):
            continue
        item = dict(q)
        item["_dataset"] = _infer_question_dataset(item, benchmark_path)
        item["_source_root"] = str(benchmark_path.parent)
        item["_source_benchmark"] = str(benchmark_path)
        if item.get("question_uid") is not None:
            item["_source_question_uid"] = str(item["question_uid"])
        uid = _question_uid(item)
        item["question_uid"] = uid
        dedupe_key = _question_dedupe_key(item)
        if dedupe_key in seen:
            duplicate_count += 1
            continue
        seen.add(dedupe_key)
        questions.append(item)

    metadata.update(
        {
            "source_files": [str(benchmark_path)],
            "source_file_count": 1,
            "deduped_question_count": len(questions),
            "duplicate_question_count": duplicate_count,
            "dedupe_rule": "dataset + scene_id + image_name + type + question + options + answer",
            "input_mode": "benchmark_file",
            "benchmark_file": str(benchmark_path),
        }
    )
    return questions, metadata, sampling_stats


def _qtype_sort_key(qtype: str) -> tuple[int, str]:
    try:
        return (QTYPE_ORDER.index(qtype), qtype)
    except ValueError:
        return (len(QTYPE_ORDER), qtype)


def _qtype_display(qtype: str) -> str:
    return QTYPE_DISPLAY.get(qtype, qtype)


def sample_questions(
    questions: list[dict[str, Any]],
    *,
    per_type: int,
    scene_cap: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(seed)
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for q in questions:
        by_type[str(q.get("type") or "unknown")].append(q)

    sampled: list[dict[str, Any]] = []
    sampling_stats: dict[str, Any] = {}
    for qtype in sorted(by_type, key=_qtype_sort_key):
        group = list(by_type[qtype])
        rng.shuffle(group)

        chosen: list[dict[str, Any]] = []
        chosen_uids: set[str] = set()
        per_scene: Counter[str] = Counter()
        relaxed_added = 0
        relaxed_cap = scene_cap
        while len(chosen) < per_type:
            before = len(chosen)
            for q in group:
                if len(chosen) >= per_type:
                    break
                scene_id = str(q.get("scene_id") or "unknown")
                if per_scene[scene_id] >= relaxed_cap:
                    continue
                uid = str(q["question_uid"])
                if uid in chosen_uids:
                    continue
                chosen.append(q)
                chosen_uids.add(uid)
                per_scene[scene_id] += 1
                if relaxed_cap > scene_cap:
                    relaxed_added += 1
            if len(chosen) == before:
                break
            relaxed_cap += 1

        sampled.extend(chosen)
        sampling_stats[qtype] = {
            "available": len(group),
            "sampled": len(chosen),
            "relaxed_scene_cap_added": relaxed_added,
            "scene_count": len({str(q.get("scene_id") or "unknown") for q in chosen}),
            "initial_scene_cap": scene_cap,
            "final_scene_cap": max(scene_cap, relaxed_cap - 1),
        }

    return sampled, sampling_stats


def build_prompt(
    question: dict[str, Any],
    direct: bool = False,
    oracle: bool = False,
    blind: bool = False,
) -> str:
    parts = [str(question.get("question") or "").strip(), ""]
    if oracle and "_oracle_info" in question:
        parts.insert(0, question["_oracle_info"] + "\n")
    options = question.get("options") or []
    for idx, option in enumerate(options):
        parts.append(f"{chr(65 + idx)}) {option}")
    if blind:
        blind_instruction = (
            BLIND_MULTI_SELECT_PROMPT_INSTRUCTION
            if is_multi_select_question(question)
            else BLIND_PROMPT_INSTRUCTION
        )
        parts.extend(["", blind_instruction])
    if direct:
        suffix = DIRECT_MULTI_SELECT_PROMPT_SUFFIX if is_multi_select_question(question) else DIRECT_PROMPT_SUFFIX
    else:
        suffix = MULTI_SELECT_PROMPT_SUFFIX if is_multi_select_question(question) else PROMPT_SUFFIX
    parts.extend(["", suffix])
    return "\n".join(parts)


def allowed_letters(question: dict[str, Any]) -> str:
    n_options = len(question.get("options") or [])
    if n_options <= 0:
        return "ABCD"
    return "".join(chr(65 + idx) for idx in range(min(n_options, 26)))


def is_multi_select_question(question: dict[str, Any]) -> bool:
    return bool(question.get("multi_select")) or isinstance(question.get("answer"), list)


def _ordered_unique_letters(values: list[str], letters: str) -> list[str]:
    seen = {value.upper() for value in values if value and value.upper() in letters.upper()}
    return [letter for letter in letters.upper() if letter in seen]


def _parse_multi_candidate(candidate: str, letters: str) -> list[str]:
    allowed = re.escape(letters.upper())
    tokens = re.findall(rf"(?<![A-Z0-9])([{allowed}])(?![A-Z0-9])", candidate)
    if tokens:
        return _ordered_unique_letters(tokens, letters)
    compact = re.sub(r"[\s,;/&+|\-]+", "", candidate)
    if compact and re.fullmatch(rf"[{allowed}]+", compact):
        return _ordered_unique_letters(list(compact), letters)
    return []


def parse_answers(raw: str | None, letters: str) -> list[str]:
    if not raw:
        return []
    allowed = re.escape(letters.upper())
    upper = raw.strip().upper()

    parsed = _parse_multi_candidate(upper, letters)
    if parsed and re.fullmatch(rf"[\s{allowed},;/&+|\-]+", upper):
        return parsed

    tail = re.search(
        rf"(?:^|[\r\n.!?])\s*([\(\[]?[{allowed}]"
        rf"(?:\s*(?:[,;/&+|\-]|\bAND\b)\s*[{allowed}])*\s*[\)\]]?)\s*$",
        upper,
    )
    if tail:
        parsed = _parse_multi_candidate(tail.group(1), letters)
        if parsed:
            return parsed

    patterns = [
        r"(?:FINAL\s+)?ANSWER(?:S)?(?:\s+IS|\s+ARE)?\s*[:：-]?\s*([^\r\n]+)",
        r"(?:CHOICES?|OPTIONS?)(?:\s+IS|\s+ARE)?\s*[:：-]?\s*([^\r\n]+)",
    ]
    for pattern in patterns:
        matches = list(re.finditer(pattern, upper))
        if matches:
            parsed = _parse_multi_candidate(matches[-1].group(1), letters)
            if parsed:
                return parsed
    return []


def parse_answer(raw: str | None, letters: str) -> str | None:
    if not raw:
        return None
    allowed = re.escape(letters.upper())
    upper = raw.strip().upper()
    if re.fullmatch(rf"[{allowed}]", upper):
        return upper

    # Some direct-answer models append the final letter to the last punctuation
    # mark after emitting hidden or visible reasoning (for example, "...sense.B").
    tail = re.search(rf"(?:^|[\r\n.!?])\s*[\(\[]?([{allowed}])[\)\].!?]?\s*$", upper)
    if tail:
        return tail.group(1)

    patterns = [
        rf"(?:FINAL\s+)?ANSWER(?:\s+IS)?\s*[:：-]?\s*[\(\[]?\s*([{allowed}])\s*[\)\]]?",
        rf"(?:CHOICE|OPTION)(?:\s+IS)?\s*[:：-]?\s*[\(\[]?\s*([{allowed}])\s*[\)\]]?",
        rf"^[\(\[]?\s*([{allowed}])\s*[\)\].:：-]",
    ]
    for pattern in patterns:
        matches = list(re.finditer(pattern, upper))
        if matches:
            return matches[-1].group(1)
    return None


def normalize_answer_letters(value: Any, letters: str, *, multi_select: bool) -> list[str]:
    if isinstance(value, list):
        return _ordered_unique_letters([str(item).strip().upper() for item in value], letters)
    text = str(value or "").strip()
    if not text:
        return []
    if multi_select:
        return parse_answers(text, letters)
    parsed = parse_answer(text, letters)
    return [parsed] if parsed else []


def _mime_for_path(path: Path) -> str:
    mime, _ = mimetypes.guess_type(str(path))
    return mime or "image/jpeg"


def _encode_image(path: Path, max_px: int | None = None) -> tuple[str, str]:
    if max_px and max_px > 0 and Image is not None:
        with Image.open(path) as img:
            img = img.convert("RGB")
            img.thumbnail((max_px, max_px))
            buf = BytesIO()
            img.save(buf, format="JPEG", quality=90)
        return base64.b64encode(buf.getvalue()).decode("ascii"), "image/jpeg"

    with path.open("rb") as f:
        return base64.b64encode(f.read()).decode("ascii"), _mime_for_path(path)


def resolve_image(
    question: dict[str, Any],
    *,
    scannet_roots: list[Path],
    scannetpp_roots: list[Path],
    scannetpp_sensor: str,
) -> ImageResolution:
    explicit_image_path = str(question.get("image_path") or "").strip()
    if explicit_image_path:
        explicit = Path(explicit_image_path)
        if explicit.exists():
            return ImageResolution(explicit, (str(explicit),))

    dataset = str(question.get("_dataset") or "scannet")
    scene = str(question.get("scene_id") or "")
    image_name = str(question.get("image_name") or "")
    roots = scannetpp_roots if dataset == "scannetpp" else scannet_roots

    candidates: list[Path] = []
    if explicit_image_path:
        candidates.append(Path(explicit_image_path))
    for root in roots:
        if dataset == "scannetpp":
            if scannetpp_sensor == "iphone":
                candidates.append(root / scene / image_name)
            elif scannetpp_sensor == "dslr":
                candidates.append(root / scene / "dslr" / "resized_images" / image_name)
            else:
                raise ValueError(
                    f"scannetpp_sensor must be 'iphone' or 'dslr', got {scannetpp_sensor!r}"
                )
            candidates.extend(
                [
                    root / scene / image_name,
                    root / scene / "dslr" / "resized_images" / image_name,
                    root / scene / "iphone" / "rgb" / image_name,
                ]
            )
        else:
            candidates.extend(
                [
                    root / scene / "color" / image_name,
                    root / scene / image_name,
                ]
            )

    checked: list[str] = []
    for candidate in candidates:
        checked.append(str(candidate))
        if candidate.exists():
            return ImageResolution(candidate, tuple(checked))
    return ImageResolution(None, tuple(checked))


def resolve_question_images(
    question: dict[str, Any],
    *,
    scannet_roots: list[Path],
    scannetpp_roots: list[Path],
    scannetpp_sensor: str,
) -> list[ImageResolution]:
    """Resolve every frame a question needs, in the order its text describes
    them: the primary image_name, then any reasoning_frame_2 /
    auxiliary_image_names frames from a two-frame-split question
    (_apply_two_frame_split in run_pipeline.py; see _collect_aux_image_names)."""
    names = [str(question.get("image_name") or "")]
    names.extend(_collect_aux_image_names(question))
    resolutions = []
    for i, name in enumerate(names):
        sub_question = question if i == 0 else {**question, "image_name": name, "image_path": None}
        resolutions.append(
            resolve_image(
                sub_question,
                scannet_roots=scannet_roots,
                scannetpp_roots=scannetpp_roots,
                scannetpp_sensor=scannetpp_sensor,
            )
        )
    return resolutions


def is_multi_image_question(question: dict[str, Any]) -> bool:
    return bool(_collect_aux_image_names(question))


def filter_multi_image_questions(
    questions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [question for question in questions if is_multi_image_question(question)]


def preflight_bev_questions(
    questions: list[dict[str, Any]],
    manifest: BevManifest,
) -> None:
    if Image is None:
        raise RuntimeError("Pillow is required to validate BEV images")
    problems: list[str] = []
    for question in questions:
        uid = str(question["question_uid"])
        entry = manifest.entries.get(uid)
        if entry is None:
            problems.append(f"{uid}: missing manifest entry")
            continue
        expected_scene = str(entry.get("scene_id") or "")
        actual_scene = str(question.get("scene_id") or "")
        if expected_scene and expected_scene != actual_scene:
            problems.append(
                f"{uid}: scene mismatch manifest={expected_scene!r} benchmark={actual_scene!r}"
            )
            continue
        expected_type = str(entry.get("question_type") or "")
        actual_type = str(question.get("type") or "")
        if expected_type and expected_type != actual_type:
            problems.append(
                f"{uid}: type mismatch manifest={expected_type!r} benchmark={actual_type!r}"
            )
            continue
        image_path = Path(str(entry["_resolved_image_path"]))
        if not image_path.is_file():
            problems.append(f"{uid}: image not found: {image_path}")
            continue
        actual_sha256 = _sha256_file(image_path)
        expected_sha256 = str(entry.get("image_sha256") or "").lower()
        if actual_sha256 != expected_sha256:
            problems.append(
                f"{uid}: image hash mismatch expected={expected_sha256} actual={actual_sha256}"
            )
            continue
        try:
            with Image.open(image_path) as image:
                image.verify()
            with Image.open(image_path) as image:
                if image.format != "PNG":
                    raise ValueError(f"expected PNG, got {image.format}")
                expected_size = (manifest.image_size_px, manifest.image_size_px)
                if image.size != expected_size:
                    raise ValueError(
                        f"expected {expected_size[0]}x{expected_size[1]}, got "
                        f"{image.size[0]}x{image.size[1]}"
                    )
        except Exception as exc:
            problems.append(f"{uid}: unreadable image {image_path}: {exc}")
            continue
        question["_bev_image_path"] = str(image_path)
        question["_bev_manifest_sha256"] = manifest.sha256
        question["_bev_direction_mode"] = manifest.direction_mode
        question["_bev_task_frame_kind"] = str(entry["task_frame_kind"])
    if problems:
        preview = "\n".join(f"  - {problem}" for problem in problems[:20])
        remainder = len(problems) - min(len(problems), 20)
        suffix = f"\n  - ... and {remainder} more" if remainder else ""
        raise ValueError(f"BEV preflight failed for {len(problems)} question(s):\n{preview}{suffix}")


def resolve_bev_image(question: dict[str, Any]) -> ImageResolution:
    path_text = str(question.get("_bev_image_path") or "").strip()
    if not path_text:
        return ImageResolution(None, (), BEV_MEDIA_ROLE)
    path = Path(path_text)
    return ImageResolution(path if path.is_file() else None, (str(path),), BEV_MEDIA_ROLE)


def rollout_condition(*, picture: bool, video: bool, context_only: bool) -> str:
    if picture:
        return "picture_context_only" if context_only else "picture_rollout"
    if video:
        return "video_context_only" if context_only else "video_rollout"
    return "baseline"


def select_manifest_questions(
    questions: list[dict[str, Any]],
    manifest: RolloutManifest,
    *,
    mode: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_uid: dict[str, dict[str, Any]] = {}
    for question in questions:
        by_uid[str(question["question_uid"])] = question
        source_uid = str(question.get("_source_question_uid") or "").strip()
        if source_uid:
            previous = by_uid.get(source_uid)
            if previous is not None and previous is not question:
                raise ValueError(f"Benchmark has duplicate source question_uid: {source_uid}")
            by_uid[source_uid] = question

    selected: list[dict[str, Any]] = []
    seen_questions: set[str] = set()
    rejected = 0
    missing_branches = 0
    for manifest_uid in manifest.entry_order:
        entry = manifest.entries[manifest_uid]
        branch = entry.get(mode)
        if branch is None:
            missing_branches += 1
            continue
        if not branch["eligible"]:
            rejected += 1
            continue
        question = by_uid.get(manifest_uid)
        if question is None:
            raise ValueError(
                f"Eligible rollout question_uid is absent from the benchmark: {manifest_uid}"
            )
        canonical_uid = str(question["question_uid"])
        if canonical_uid in seen_questions:
            raise ValueError(
                f"Multiple rollout entries resolve to benchmark question_uid: {canonical_uid}"
            )
        expected_type = str(entry.get("question_type") or "").strip()
        if expected_type and expected_type != str(question.get("type") or ""):
            raise ValueError(
                f"Rollout question type mismatch for {manifest_uid}: "
                f"manifest={expected_type!r}, benchmark={question.get('type')!r}"
            )
        expected_scene = str(entry.get("scene_id") or "").strip()
        if expected_scene and expected_scene != str(question.get("scene_id") or ""):
            raise ValueError(
                f"Rollout scene mismatch for {manifest_uid}: "
                f"manifest={expected_scene!r}, benchmark={question.get('scene_id')!r}"
            )
        question["_rollout_manifest_uid"] = manifest_uid
        selected.append(question)
        seen_questions.add(canonical_uid)

    stats: dict[str, Any] = {}
    counts = Counter(str(question.get("type") or "unknown") for question in selected)
    for qtype, count in sorted(counts.items(), key=lambda item: _qtype_sort_key(item[0])):
        stats[qtype] = {
            "available": count,
            "sampled": count,
            "selection": f"eligible_{mode}_manifest_entries",
        }
    stats["_rollout_coverage"] = {
        "manifest_entries": len(manifest.entry_order),
        "eligible_selected": len(selected),
        "ineligible": rejected,
        "missing_modality_branch": missing_branches,
    }
    return selected, stats


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_rollout_images(
    question: dict[str, Any],
    manifest: RolloutManifest,
    *,
    mode: str,
    context_only: bool,
) -> tuple[list[ImageResolution], str | None]:
    manifest_uid = str(question.get("_rollout_manifest_uid") or question["question_uid"])
    entry = manifest.entries.get(manifest_uid)
    if entry is None:
        return [ImageResolution(None, (), None)], "rollout_manifest_missing"
    branch = entry.get(mode)
    if not isinstance(branch, dict):
        return [ImageResolution(None, (), None)], f"rollout_{mode}_missing"
    if not branch.get("eligible"):
        return [ImageResolution(None, (), None)], f"rollout_{mode}_ineligible"

    all_media = list(branch.get("media") or [])
    context_media = [item for item in all_media if item.get("kind") == "context"]
    prediction_media = [item for item in all_media if item.get("kind") == "prediction"]
    if mode == "picture":
        role_rank = {
            "source_view": 0,
            "motion_reference_view": 0,
            "source_to_destination_bridge": 1,
            "destination_environment": 2,
            "predicted_future_view": 3,
            "destination_to_query_bridge": 4,
            "query_reference_view": 5,
        }
        ranks = [role_rank.get(str(item.get("role")), -1) for item in all_media]
        if -1 in ranks or ranks != sorted(ranks):
            return [ImageResolution(None, (), None)], "rollout_picture_order_invalid"
        context_roles = {str(item.get("role")) for item in context_media}
        has_single_frame_canvas = "motion_reference_view" in context_roles
        has_destination_canvas = (
            "source_view" in context_roles and "destination_environment" in context_roles
        )
        if not has_single_frame_canvas and not has_destination_canvas:
            return [ImageResolution(None, (), None)], "rollout_picture_context_invalid"
        if not context_only and (
            len(prediction_media) != 1
            or prediction_media[0].get("role") != "predicted_future_view"
        ):
            return [ImageResolution(None, (), None)], "rollout_picture_prediction_invalid"
    else:
        video_role_rank = {
            "motion_reference_view": 0,
            "predicted_video_frame": 1,
            "destination_to_query_bridge": 2,
            "query_reference_view": 3,
        }
        video_ranks = [
            video_role_rank.get(str(item.get("role")), -1) for item in all_media
        ]
        if (
            -1 in video_ranks
            or video_ranks != sorted(video_ranks)
            or not all_media
            or all_media[0].get("role") != "motion_reference_view"
            or sum(
                item.get("role") == "motion_reference_view" for item in context_media
            )
            != 1
        ):
            return [ImageResolution(None, (), None)], "rollout_video_context_invalid"
        if not context_only:
            frame_indices = [item.get("frame_index") for item in prediction_media]
            if (
                len(prediction_media) != 8
                or any(item.get("role") != "predicted_video_frame" for item in prediction_media)
                or frame_indices != list(range(8))
            ):
                return [ImageResolution(None, (), None)], "rollout_video_frames_invalid"

    selected_media = context_media if context_only else all_media
    if not selected_media:
        return [ImageResolution(None, (), None)], f"rollout_{mode}_media_missing"

    resolutions: list[ImageResolution] = []
    for item in selected_media:
        raw_path = Path(str(item["path"]))
        path = raw_path if raw_path.is_absolute() else manifest.path.parent / raw_path
        checked = (str(path),)
        if not path.is_file():
            resolutions.append(ImageResolution(None, checked, str(item["role"])))
            continue
        expected_sha = str(item.get("sha256") or "").strip().lower()
        if expected_sha and (
            not re.fullmatch(r"[0-9a-f]{64}", expected_sha)
            or _sha256_file(path) != expected_sha
        ):
            return [ImageResolution(None, checked, str(item["role"]))], "rollout_media_hash_mismatch"
        role = str(item["role"])
        if role == "predicted_video_frame":
            role = f"{role}:{int(item['frame_index']) + 1}/8"
        resolutions.append(ImageResolution(path, checked, role))
    if any(resolution.path is None for resolution in resolutions):
        return resolutions, "rollout_media_not_found"
    return resolutions, None


def rollout_role_label(role: str | None) -> str | None:
    if not role:
        return None
    base_role, _, suffix = role.partition(":")
    if base_role == BEV_MEDIA_ROLE:
        return BEV_MEDIA_LABEL
    label = ROLLOUT_ROLE_LABELS.get(base_role)
    if not label:
        return None
    return f"{label} {suffix}" if suffix else label


def _resolve_scannet_geometry_root(image_roots: list[str]) -> str:
    for root_text in image_roots:
        root = Path(root_text)
        if root.name == "scans":
            return str(root)
        if (root / "scans").exists():
            return str(root / "scans")
    return image_roots[0] if image_roots else "data/scannet/scans"


def _unique_paths(paths: list[Path]) -> list[Path]:
    seen: set[str] = set()
    unique: list[Path] = []
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def _resolve_scannetpp_geometry_roots(image_roots: list[str], explicit_root: str | None) -> list[str]:
    candidates: list[Path] = []
    if explicit_root:
        candidates.append(Path(explicit_root))
    candidates.extend([Path("data/scannetpp"), Path("++data")])
    for root_text in image_roots:
        root = Path(root_text)
        candidates.extend(
            [
                root,
                root.parent / "data" / "scannetpp",
                root.parent / "++data",
                root.parent.parent / "data" / "scannetpp",
                root.parent.parent / "++data",
            ]
        )
    return [str(path) for path in _unique_paths(candidates)]


def _has_scannetpp_geometry(scene_path: Path) -> bool:
    return (
        (scene_path / "scans" / "mesh_aligned_0.05.ply").is_file()
        and (scene_path / "scans" / "segments.json").is_file()
        and (scene_path / "scans" / "segments_anno.json").is_file()
    )


def _has_scannetpp_pose_files(scene_path: Path, sensor: str) -> bool:
    if sensor == "iphone":
        return (scene_path / "iphone" / "colmap" / "images.txt").is_file()
    if sensor == "dslr":
        return (scene_path / "dslr" / "nerfstudio" / "transforms.json").is_file()
    return False


def _load_oracle_scene_cache_entry(
    scene_id: str,
    dataset: str,
    *,
    scannet_root: str,
    scannetpp_roots: list[str],
    scannetpp_sensor: str,
    need_poses: bool,
    oracle_cache_dir: str | None = None,
    scannetpp_root_option: str = "--scannetpp_geometry_root",
) -> Any:
    if (
        OracleSceneCacheEntry is None
        or _oracle_scene_path is None
        or _oracle_dataset_kind is None
        or _parse_oracle_scene is None
    ):
        raise RuntimeError("runtime oracle generation helpers are unavailable")
    dataset_kind = _oracle_dataset_kind(scene_id, dataset)

    import pickle as _pkl
    _cache_file = Path(oracle_cache_dir) / f"{scene_id}.pkl" if oracle_cache_dir else None
    if _cache_file and _cache_file.is_file():
        with open(_cache_file, "rb") as _f:
            _cached = _pkl.load(_f)
        _scene_path = Path(_cached["scene_path"])
        _poses = _cached.get("poses")
        if _poses is None and need_poses and _oracle_load_poses is not None:
            _poses = _oracle_load_poses(_scene_path, dataset_kind, scannetpp_sensor)
        return OracleSceneCacheEntry(scene_path=_scene_path, objects=_cached["objects"], poses=_poses)

    if dataset_kind == "scannet":
        scene_path = _oracle_scene_path(scene_id, dataset, scannet_root, scannetpp_roots[0] if scannetpp_roots else "")
        parsed = _parse_oracle_scene(scene_path, dataset=dataset_kind)
        objects = {int(o["id"]): o for o in (parsed or {}).get("objects", [])}
        if _cache_file and objects:
            _cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(_cache_file, "wb") as _f:
                _pkl.dump({"objects": objects, "scene_path": str(scene_path)}, _f)
        poses = None
        if need_poses:
            if _oracle_load_poses is None:
                raise RuntimeError("runtime oracle pose loader is unavailable")
            poses = _oracle_load_poses(scene_path, dataset_kind, scannetpp_sensor)
        return OracleSceneCacheEntry(scene_path=scene_path, objects=objects, poses=poses)

    errors: list[str] = []
    for root in scannetpp_roots:
        scene_path = _oracle_scene_path(scene_id, dataset, scannet_root, root)
        if not _has_scannetpp_geometry(scene_path):
            errors.append(f"{scene_path}: missing scans/mesh_aligned_0.05.ply or segment annotation files")
            continue
        if need_poses and not _has_scannetpp_pose_files(scene_path, scannetpp_sensor):
            errors.append(f"{scene_path}: missing {scannetpp_sensor} pose files")
            continue
        try:
            parsed = _parse_oracle_scene(scene_path, dataset=dataset_kind)
            objects = {int(o["id"]): o for o in (parsed or {}).get("objects", [])}
            if not objects:
                errors.append(f"{scene_path}: no parsed objects")
                continue
            if _cache_file and objects:
                _cache_file.parent.mkdir(parents=True, exist_ok=True)
                with open(_cache_file, "wb") as _f:
                    _pkl.dump({"objects": objects, "scene_path": str(scene_path)}, _f)
            poses = None
            if need_poses:
                if _oracle_load_poses is None:
                    raise RuntimeError("runtime oracle pose loader is unavailable")
                poses = _oracle_load_poses(scene_path, dataset_kind, scannetpp_sensor)
            return OracleSceneCacheEntry(scene_path=scene_path, objects=objects, poses=poses)
        except Exception as exc:
            errors.append(f"{scene_path}: {exc}")
            continue

    tried = "; ".join(errors[:6])
    if len(errors) > 6:
        tried += f"; ... {len(errors) - 6} more"
    raise RuntimeError(
        f"could not load ScanNet++ raw geometry/poses for {scene_id}; "
        f"Pass the raw ScanNet++ root with {scannetpp_root_option}. "
        f"Tried {tried}"
    )


def ensure_runtime_oracle_info(
    questions: list[dict[str, Any]],
    *,
    oracle_mode: str,
    scannet_root: str,
    scannetpp_roots: list[str],
    scannetpp_sensor: str,
    oracle_cache_dir: str | None = None,
) -> dict[str, int]:
    if oracle_mode == "none":
        return {"generated": 0, "preexisting": 0, "skipped": 0, "pose_missing": 0, "scene_errors": 0}
    if oracle_mode == "task_frame" and _make_task_frame_oracle_prefix is None:
        raise RuntimeError("runtime task-frame oracle generation is unavailable")
    if oracle_mode == "world" and _make_world_oracle_prefix is None:
        raise RuntimeError("runtime world oracle generation is unavailable")

    scene_cache: dict[str, Any] = {}
    stats = {"generated": 0, "preexisting": 0, "skipped": 0, "pose_missing": 0, "scene_errors": 0}
    for question in questions:
        if question.get("_oracle_info"):
            stats["preexisting"] += 1
            continue

        scene_id = str(question.get("scene_id") or "")
        dataset = str(question.get("_dataset") or question.get("dataset") or "")
        if not scene_id:
            stats["skipped"] += 1
            continue

        if scene_id not in scene_cache:
            need_poses = oracle_mode == "task_frame"
            try:
                scene_cache[scene_id] = _load_oracle_scene_cache_entry(
                    scene_id,
                    dataset,
                    scannet_root=scannet_root,
                    scannetpp_roots=scannetpp_roots,
                    scannetpp_sensor=scannetpp_sensor,
                    need_poses=need_poses,
                    oracle_cache_dir=oracle_cache_dir,
                )
            except Exception as exc:
                print(f"oracle warning: {scene_id}: {exc}", file=sys.stderr, flush=True)
                stats["scene_errors"] += 1
                scene_cache[scene_id] = None

        entry = scene_cache[scene_id]
        if entry is None:
            stats["skipped"] += 1
            continue

        if oracle_mode == "world":
            prefix = _make_world_oracle_prefix(question, entry.objects)
        else:
            pose = entry.poses.get(str(question.get("image_name") or "")) if entry.poses is not None else None
            if _oracle_frame_kind_for_question is not None and _oracle_frame_kind_for_question(question) in {"agent", "object_centric"} and pose is None:
                stats["pose_missing"] += 1
            prefix = _make_task_frame_oracle_prefix(question, entry.objects, pose)

        if prefix:
            question["_oracle_info"] = prefix
            question["_oracle_mode"] = oracle_mode
            stats["generated"] += 1
        else:
            stats["skipped"] += 1
    return stats


def _is_reasoning_chat_model(model: str) -> bool:
    """GPT-5 / o-series chat models reject `max_tokens` and non-default `temperature`.

    They require `max_completion_tokens` and only support the default temperature (1),
    so callers must omit `temperature` and rename the token-budget parameter.
    """
    name = (model or "").lower()
    return name.startswith(("gpt-5", "o1", "o3", "o4"))


def _should_omit_temperature(model: str) -> bool:
    """Return whether the served model/proxy rejects an explicit temperature."""
    name = (model or "").lower()
    return _is_reasoning_chat_model(model) or name.startswith(
        (
            "claude-opus-4",
            "claude-sonnet-4",
            "claude-4",
        )
    )


def _supports_qwen_thinking_control(model: str) -> bool:
    normalized = str(model).strip().lower().replace("_", "-")
    return "qwen3.5" in normalized or "qwen3-5" in normalized


def make_client(
    api_provider: str,
    base_url: str,
    api_key: str,
    timeout: float,
    *,
    credential_kind: str = "api_key",
):
    if api_provider == "anthropic":
        from anthropic import Anthropic

        credential = (
            {"auth_token": api_key}
            if credential_kind == "auth_token"
            else {"api_key": api_key}
        )
        return Anthropic(**credential, base_url=base_url, timeout=timeout)

    from openai import OpenAI

    return OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)


class ThreadLocalOpenAIClientFactory:
    def __init__(
        self,
        *,
        api_provider: str,
        base_url: str,
        api_key: str,
        timeout: float,
        credential_kind: str = "api_key",
    ) -> None:
        self.api_provider = api_provider
        self.base_url = base_url
        self.api_key = api_key
        self.timeout = timeout
        self.credential_kind = credential_kind
        self.local = threading.local()

    def get_client(self) -> Any:
        client = getattr(self.local, "client", None)
        if client is None:
            client = make_client(
                self.api_provider,
                self.base_url,
                self.api_key,
                self.timeout,
                credential_kind=self.credential_kind,
            )
            self.local.client = client
        return client


def _resolve_api_credential(args: argparse.Namespace) -> tuple[str, str]:
    if args.api_key:
        return args.api_key, "api_key"

    if args.api_key_env:
        value = os.getenv(args.api_key_env)
        if value:
            kind = (
                "auth_token"
                if args.api_provider == "anthropic"
                and args.api_key_env.upper().endswith("AUTH_TOKEN")
                else "api_key"
            )
            return value, kind

    if args.api_provider == "anthropic":
        auth_token = os.getenv("ANTHROPIC_AUTH_TOKEN")
        if auth_token:
            return auth_token, "auth_token"
        anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        if anthropic_api_key:
            return anthropic_api_key, "api_key"
    else:
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if openai_api_key:
            return openai_api_key, "api_key"

    return os.getenv("DASHSCOPE_API_KEY") or "EMPTY", "api_key"


def _content_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if text:
                    parts.append(str(text))
            else:
                text = getattr(item, "text", None) or getattr(item, "content", None)
                if text:
                    parts.append(str(text))
        return "".join(parts)
    return str(value)


def _get_field(value: Any, name: str) -> Any:
    if isinstance(value, dict):
        return value.get(name)
    return getattr(value, name, None)


def _extract_chat_choice_text(choice: Any, *, include_reasoning: bool = True) -> str:
    parts: list[str] = []
    field_names = (
        ("content", "reasoning_content", "text")
        if include_reasoning
        else ("content", "text")
    )
    for container_name in ("delta", "message"):
        container = _get_field(choice, container_name)
        if container is None:
            continue
        for field_name in field_names:
            text = _content_text(_get_field(container, field_name))
            if text:
                parts.append(text)
    text = _content_text(_get_field(choice, "text"))
    if text:
        parts.append(text)
    return "".join(parts)


def _extract_chat_response_text(response: Any, *, include_reasoning: bool = True) -> str:
    if isinstance(response, str):
        return response
    choices = _get_field(response, "choices") or []
    parts = [
        _extract_chat_choice_text(choice, include_reasoning=include_reasoning)
        for choice in choices
    ]
    text = "".join(part for part in parts if part)
    if text:
        return text
    return _content_text(_get_field(response, "content"))


def _require_response_text(text: str, *, provider: str, model: str) -> str:
    text = text.strip()
    if not text:
        raise RuntimeError(
            f"{provider} returned an empty response for model {model!r}; "
            "check the API base URL, model name, provider protocol, and streaming compatibility"
        )
    return text


def call_model(
    client: Any,
    *,
    api_provider: str,
    model: str,
    image_paths: list[Path],
    prompt: str,
    max_tokens: int,
    temperature: float,
    api_image_max_px: int,
    blind: bool = False,
    image_roles: list[str | None] | None = None,
    direct: bool = False,
) -> str:
    omit_temperature = _should_omit_temperature(model)
    encoded: list[tuple[str, str]] = []
    if not blind:
        encoded = [_encode_image(path, api_image_max_px) for path in image_paths]
    if image_roles is not None and len(image_roles) != len(image_paths):
        raise ValueError("image_roles must have the same length as image_paths")
    roles = list(image_roles or [None] * len(image_paths))
    has_roles = any(roles)
    if api_provider == "openai_responses":
        user_content: list[Any] = []
        for index, ((b64, mime), role) in enumerate(zip(encoded, roles), 1):
            if role:
                user_content.append(
                    {"type": "input_text", "text": f"Image {index} role: {role}."}
                )
            user_content.append(
                {"type": "input_image", "image_url": f"data:{mime};base64,{b64}"}
            )
        user_content.append({"type": "input_text", "text": prompt})
        response_kwargs: dict[str, Any] = {
            "model": model,
            "input": [
                {
                    "role": "system",
                    "content": [
                        {"type": "input_text", "text": SYSTEM_PROMPT},
                    ],
                },
                {"role": "user", "content": user_content},
            ],
            "max_output_tokens": max_tokens,
        }
        if not omit_temperature:
            response_kwargs["temperature"] = temperature
        response = client.responses.create(**response_kwargs)
        output_text = getattr(response, "output_text", None)
        if output_text:
            return str(output_text).strip()
        chunks: list[str] = []
        for item in getattr(response, "output", []) or []:
            for content in getattr(item, "content", []) or []:
                text = getattr(content, "text", None)
                if text:
                    chunks.append(str(text))
        return _require_response_text(
            "\n".join(chunks),
            provider=api_provider,
            model=model,
        )

    if api_provider == "anthropic":
        anthropic_user_content: list[Any] = []
        for index, ((b64, mime), role) in enumerate(zip(encoded, roles), 1):
            if role:
                anthropic_user_content.append(
                    {"type": "text", "text": f"Image {index} role: {role}."}
                )
            anthropic_user_content.append(
                {"type": "image", "source": {"type": "base64", "media_type": mime, "data": b64}}
            )
        anthropic_user_content.append({"type": "text", "text": prompt})
        message_kwargs: dict[str, Any] = {
            "model": model,
            "system": SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": anthropic_user_content}],
            "max_tokens": max_tokens,
        }
        if not omit_temperature:
            message_kwargs["temperature"] = temperature
        response = client.messages.create(**message_kwargs)
        chunks = [
            str(getattr(block, "text", ""))
            for block in getattr(response, "content", []) or []
            if getattr(block, "type", None) == "text" and getattr(block, "text", None)
        ]
        return _require_response_text(
            "\n".join(chunks),
            provider=api_provider,
            model=model,
        )

    if has_roles:
        chat_user_content: list[dict[str, Any]] = []
        for index, ((b64, mime), role) in enumerate(zip(encoded, roles), 1):
            if role:
                chat_user_content.append(
                    {"type": "text", "text": f"Image {index} role: {role}."}
                )
            chat_user_content.append(
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}
            )
        chat_user_content.append({"type": "text", "text": prompt})
    else:
        chat_user_content = [{"type": "text", "text": prompt}] + [
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}
            for b64, mime in encoded
        ]

    chat_kwargs: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": chat_user_content,
            },
        ],
    }
    if _is_reasoning_chat_model(model):
        # GPT-5 / o-series: use max_completion_tokens.
        chat_kwargs["max_completion_tokens"] = max_tokens
    else:
        chat_kwargs["max_tokens"] = max_tokens
    if not omit_temperature:
        chat_kwargs["temperature"] = temperature
    if _supports_qwen_thinking_control(model):
        chat_kwargs["extra_body"] = {"enable_thinking": not direct}

    # Stream explicitly: some OpenAI-compatible proxies always reply with SSE chunks
    # for certain models, which the SDK cannot parse in non-streaming mode (it returns
    # the raw text instead). Streaming handles both behaviours uniformly.
    chat_kwargs["stream"] = True
    stream = client.chat.completions.create(**chat_kwargs)
    parts: list[str] = []
    reasoning_fallback_parts: list[str] = []
    for chunk in stream:
        text = _extract_chat_response_text(chunk, include_reasoning=not direct)
        if text:
            parts.append(text)
        if direct:
            fallback_text = _extract_chat_response_text(chunk, include_reasoning=True)
            if fallback_text:
                reasoning_fallback_parts.append(fallback_text)
    return _require_response_text(
        "".join(parts) or "".join(reasoning_fallback_parts),
        provider=api_provider,
        model=model,
    )


def load_existing_results(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    rows = data.get("results", data) if isinstance(data, dict) else data
    if not isinstance(rows, list):
        return {}
    return {
        str(row.get("question_uid")): row
        for row in rows
        if isinstance(row, dict) and row.get("question_uid")
    }


def save_results(
    path: Path,
    *,
    metadata: dict[str, Any],
    sampling_stats: dict[str, Any],
    results: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": metadata,
        "sampling_stats": sampling_stats,
        "summary": summarize_results(results),
        "results": results,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _answer_fields(question: dict[str, Any], raw_response: str | None) -> dict[str, Any]:
    letters = allowed_letters(question)
    multi_select = is_multi_select_question(question)
    gt_answers = normalize_answer_letters(
        question.get("answer"),
        letters,
        multi_select=multi_select,
    )
    predictions = parse_answers(raw_response, letters) if multi_select else []
    prediction = ",".join(predictions) if multi_select else parse_answer(raw_response, letters)
    gt_answer = ",".join(gt_answers) if multi_select else (gt_answers[0] if gt_answers else "")
    correct = (
        bool(predictions and gt_answers and set(predictions) == set(gt_answers))
        if multi_select
        else bool(prediction and gt_answer and prediction == gt_answer)
    )
    fields: dict[str, Any] = {
        "gt_answer": gt_answer,
        "prediction": prediction,
        "correct": correct,
    }
    if multi_select:
        fields.update(
            {
                "multi_select": True,
                "gt_answers": gt_answers,
                "predictions": predictions,
            }
        )
        if question.get("correct_values") is not None:
            fields["correct_values"] = question.get("correct_values")
    return fields


def refresh_cached_result(question: dict[str, Any], cached: dict[str, Any]) -> dict[str, Any]:
    row = dict(cached)
    row.update(_answer_fields(question, cached.get("raw_response")))
    row["question_uid"] = question.get("question_uid")
    row["dataset"] = question.get("_dataset")
    return row


def result_from_question(
    question: dict[str, Any],
    *,
    image_resolutions: list[ImageResolution],
    raw_response: str | None,
    error: str | None,
) -> dict[str, Any]:
    primary_resolution = image_resolutions[0]
    aux_resolutions = image_resolutions[1:]
    has_rollout_media = any(resolution.role for resolution in image_resolutions)

    row = {
        "question_uid": question.get("question_uid"),
        "dataset": question.get("_dataset"),
        "source_benchmark": question.get("_source_benchmark"),
        "scene_id": question.get("scene_id"),
        "image_name": question.get("image_name"),
        "image_path": str(primary_resolution.path) if primary_resolution.path else None,
        "aux_image_names": (
            [resolution.path.name if resolution.path else None for resolution in aux_resolutions]
            if has_rollout_media
            else _collect_aux_image_names(question)
        ),
        "aux_image_paths": [str(r.path) if r.path else None for r in aux_resolutions],
        "checked_image_paths": [
            p for r in image_resolutions for p in r.checked_paths
        ],
        "media_roles": [r.role for r in image_resolutions],
        "evaluation_condition": question.get("_evaluation_condition", "baseline"),
        "level": question.get("level"),
        "type": question.get("type"),
        "relation_unchanged": question.get("relation_unchanged"),
        "cross_frame": bool(_collect_aux_image_names(question)),
        "has_attachment_chain": question.get("has_attachment_chain"),
        "attachment_pair_id": question.get("attachment_pair_id"),
        "attachment_parent_id": question.get("attachment_parent_id"),
        "attachment_child_id": question.get("attachment_child_id"),
        "question": question.get("question"),
        "options": question.get("options"),
        "correct_value": question.get("correct_value"),
        "raw_response": raw_response,
        "error": error,
        **_answer_fields(question, raw_response),
    }
    if question.get("_oracle_info"):
        row["oracle_info"] = question.get("_oracle_info")
        row["oracle_mode"] = question.get("_oracle_mode")
    if question.get("_rollout_manifest_sha256"):
        row["rollout_manifest_sha256"] = question["_rollout_manifest_sha256"]
        row["rollout_manifest_uid"] = question.get("_rollout_manifest_uid")
        row["rollout_generation"] = question.get("_rollout_generation")
        row["rollout_quality_audit"] = question.get("_rollout_quality_audit")
    if question.get("_bev_image_path"):
        row["bev_image_path"] = question["_bev_image_path"]
        row["bev_manifest_sha256"] = question.get("_bev_manifest_sha256")
        row["bev_direction_mode"] = question.get("_bev_direction_mode")
        row["bev_task_frame_kind"] = question.get("_bev_task_frame_kind")
    return row


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_type: dict[str, dict[str, Any]] = {}
    for row in results:
        qtype = str(row.get("type") or "unknown")
        stats = by_type.setdefault(
            qtype,
            {"total": 0, "answered": 0, "correct": 0, "errors": 0, "missing_images": 0},
        )
        stats["total"] += 1
        if row.get("prediction"):
            stats["answered"] += 1
        if row.get("correct"):
            stats["correct"] += 1
        if row.get("error"):
            stats["errors"] += 1
        if row.get("error") == "image_not_found":
            stats["missing_images"] += 1

    for stats in by_type.values():
        total = int(stats["total"])
        answered = int(stats["answered"])
        stats["accuracy"] = (float(stats["correct"]) / total) if total else None
        stats["answered_accuracy"] = (
            float(stats["correct"]) / answered if answered else None
        )

    ordered = {
        qtype: by_type[qtype]
        for qtype in sorted(by_type, key=_qtype_sort_key)
    }
    return {"by_type": ordered}


def _fmt_pct(value: float | None) -> str:
    return "-" if value is None else f"{value * 100:.1f}%"


def _option_html(row: dict[str, Any]) -> str:
    letters = "".join(chr(65 + idx) for idx, _ in enumerate(row.get("options") or []))
    gt_letters = set(
        normalize_answer_letters(
            row.get("gt_answers") if row.get("gt_answers") is not None else row.get("gt_answer"),
            letters,
            multi_select=bool(row.get("multi_select")),
        )
    )
    pred_letters = set(
        normalize_answer_letters(
            row.get("predictions") if row.get("predictions") is not None else row.get("prediction"),
            letters,
            multi_select=bool(row.get("multi_select")),
        )
    )
    chunks: list[str] = []
    for idx, option in enumerate(row.get("options") or []):
        letter = chr(65 + idx)
        classes = ["option"]
        if letter in gt_letters:
            classes.append("gold")
        if letter in pred_letters and letter not in gt_letters:
            classes.append("predicted")
        chunks.append(
            f'<div class="{" ".join(classes)}">'
            f'<span class="letter">{html.escape(letter)}</span>'
            f'<span>{html.escape(str(option))}</span>'
            "</div>"
        )
    return "\n".join(chunks)


def _single_image_html(
    path_text: str | None, html_image_max_px: int, *, missing_hint: str
) -> str:
    if not path_text:
        return (
            '<div class="missing-image">'
            "image not found"
            f"<small>{html.escape(str(missing_hint))}</small>"
            "</div>"
        )
    path = Path(str(path_text))
    if not path.exists():
        return (
            '<div class="missing-image">'
            "image path no longer exists"
            f"<small>{html.escape(str(path))}</small>"
            "</div>"
        )
    b64, mime = _encode_image(path, html_image_max_px)
    return f'<img src="data:{mime};base64,{b64}" alt="">'


def _image_html(row: dict[str, Any], html_image_max_px: int) -> str:
    checked = row.get("checked_image_paths") or []
    primary_html = _single_image_html(
        row.get("image_path"), html_image_max_px, missing_hint=checked[0] if checked else ""
    )
    aux_paths = row.get("aux_image_paths") or []
    if not aux_paths:
        return primary_html

    media_roles = list(row.get("media_roles") or [])
    total = len(aux_paths)
    primary_label = rollout_role_label(media_roles[0] if media_roles else None) or "original"
    blocks = [f'<div class="img-label">{html.escape(primary_label)}</div>{primary_html}']
    for i, aux_path_text in enumerate(aux_paths, start=1):
        role = media_roles[i] if i < len(media_roles) else None
        label = rollout_role_label(role) or (
            "auxiliary" if total == 1 else f"auxiliary {i}/{total}"
        )
        aux_html = _single_image_html(
            aux_path_text, html_image_max_px, missing_hint=f"{label} not found"
        )
        blocks.append(f'<div class="img-label">{html.escape(label)}</div>{aux_html}')
    return '<div class="multi-img">' + "".join(blocks) + '</div>'


def build_html(results: list[dict[str, Any]], *, title: str, html_image_max_px: int) -> str:
    summary = summarize_results(results)["by_type"]
    summary_rows = []
    for qtype, stats in summary.items():
        summary_rows.append(
            "<tr>"
            f'<td class="qtype-cell">{html.escape(_qtype_display(qtype))}</td>'
            f"<td>{stats['correct']} / {stats['total']}</td>"
            f"<td>{_fmt_pct(stats['accuracy'])}</td>"
            f"<td>{stats['answered']}</td>"
            f"<td>{stats['missing_images']}</td>"
            f"<td>{stats['errors']}</td>"
            "</tr>"
        )

    cards = []
    for idx, row in enumerate(sorted(results, key=lambda r: (_qtype_sort_key(str(r.get("type") or "")), str(r.get("scene_id") or ""), str(r.get("image_name") or ""))), 1):
        status = "correct" if row.get("correct") else "wrong"
        if row.get("error"):
            status = "error"
        pred = row.get("prediction") or "-"
        raw = row.get("raw_response") or row.get("error") or ""
        oracle_details = ""
        if row.get("oracle_info"):
            oracle_details = (
                "<details>"
                "<summary>Oracle prompt prefix</summary>"
                f"<pre>{html.escape(str(row.get('oracle_info') or ''))}</pre>"
                "</details>"
            )
        cards.append(
            f'<article class="card {status}">'
            '<div class="image-wrap">'
            f'{_image_html(row, html_image_max_px)}'
            "</div>"
            '<div class="content">'
            '<div class="meta">'
            f'<span>#{idx}</span>'
            f'<span class="qtype">{html.escape(_qtype_display(str(row.get("type") or "unknown")))}</span>'
            f'<span>{html.escape(str(row.get("dataset") or ""))}</span>'
            f'<span>{html.escape(str(row.get("scene_id") or ""))} / {html.escape(str(row.get("image_name") or ""))}</span>'
            f'<span class="pill">{html.escape(status)}</span>'
            "</div>"
            f'<h2>{html.escape(str(row.get("question") or ""))}</h2>'
            f'<div class="options">{_option_html(row)}</div>'
            '<div class="answer-line">'
            f'<strong>GT:</strong> {html.escape(str(row.get("gt_answer") or "-"))}'
            f'<strong>Model:</strong> {html.escape(str(pred))}'
            f'<strong>Correct value:</strong> {html.escape(str(row.get("correct_value") or "-"))}'
            "</div>"
            f"{oracle_details}"
            "<details open>"
            "<summary>Model reasoning and raw answer</summary>"
            f"<pre>{html.escape(str(raw))}</pre>"
            "</details>"
            "</div>"
            "</article>"
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>
:root {{
  --bg: #f4f1ea;
  --ink: #1c2520;
  --muted: #5f6a62;
  --line: #d7d0c3;
  --paper: #fffdf8;
  --good: #146c43;
  --bad: #a9271b;
  --warn: #9a6700;
  --accent: #245b63;
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  background:
    linear-gradient(135deg, rgba(36,91,99,0.10), transparent 34rem),
    linear-gradient(315deg, rgba(184,125,42,0.10), transparent 30rem),
    var(--bg);
  color: var(--ink);
  font-family: Georgia, "Times New Roman", serif;
}}
header {{
  padding: 32px clamp(16px, 4vw, 48px) 18px;
  border-bottom: 1px solid var(--line);
}}
h1 {{
  margin: 0 0 16px;
  font-size: clamp(28px, 4vw, 48px);
  font-weight: 700;
  letter-spacing: 0;
}}
.summary-table {{
  width: 100%;
  border-collapse: collapse;
  background: rgba(255,253,248,0.72);
}}
.summary-table th,
.summary-table td {{
  padding: 9px 10px;
  border-bottom: 1px solid var(--line);
  text-align: left;
  font-size: 14px;
}}
.summary-table th {{
  color: var(--muted);
  font-family: "Trebuchet MS", sans-serif;
  text-transform: uppercase;
  letter-spacing: 0.03em;
}}
.summary-table .qtype-cell {{
  color: var(--accent);
  font-weight: 800;
  font-family: "Trebuchet MS", sans-serif;
}}
main {{
  padding: 24px clamp(16px, 4vw, 48px) 48px;
}}
.card {{
  display: grid;
  grid-template-columns: minmax(260px, 38%) 1fr;
  gap: 22px;
  margin: 0 0 18px;
  padding: 16px;
  background: var(--paper);
  border: 1px solid var(--line);
  border-left: 6px solid var(--line);
  box-shadow: 0 12px 28px rgba(40, 34, 23, 0.08);
}}
.card.correct {{ border-left-color: var(--good); }}
.card.wrong {{ border-left-color: var(--bad); }}
.card.error {{ border-left-color: var(--warn); }}
.image-wrap {{
  min-height: 220px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: #e8e2d6;
  border: 1px solid var(--line);
  overflow: hidden;
}}
.image-wrap img {{
  width: 100%;
  height: auto;
  display: block;
}}
.missing-image {{
  padding: 18px;
  color: var(--muted);
  font-family: "Trebuchet MS", sans-serif;
  text-align: center;
}}
.missing-image small {{
  display: block;
  margin-top: 8px;
  overflow-wrap: anywhere;
}}
.multi-img {{
  display: flex;
  flex-direction: column;
  gap: 4px;
}}
.multi-img .img-label {{
  font-size: 0.75em;
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: 0.04em;
}}
.meta {{
  display: flex;
  flex-wrap: wrap;
  gap: 7px;
  margin-bottom: 10px;
  font-family: "Trebuchet MS", sans-serif;
  color: var(--muted);
  font-size: 13px;
}}
.meta span {{
  border: 1px solid var(--line);
  padding: 3px 7px;
  background: rgba(255,255,255,0.52);
}}
.meta .qtype {{
  color: #ffffff;
  background: var(--accent);
  border-color: var(--accent);
  font-weight: 800;
}}
.pill {{
  color: var(--ink);
  font-weight: 700;
}}
h2 {{
  margin: 0 0 12px;
  font-size: 19px;
  line-height: 1.35;
}}
.options {{
  display: grid;
  gap: 7px;
  margin-bottom: 12px;
}}
.option {{
  display: grid;
  grid-template-columns: 28px 1fr;
  gap: 8px;
  padding: 8px 10px;
  border: 1px solid var(--line);
  background: #fbf7ef;
}}
.option.gold {{
  border-color: rgba(20,108,67,0.55);
  background: rgba(20,108,67,0.08);
}}
.option.predicted {{
  border-color: rgba(169,39,27,0.55);
  background: rgba(169,39,27,0.08);
}}
.letter {{
  font-family: "Trebuchet MS", sans-serif;
  font-weight: 700;
}}
.answer-line {{
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
  margin: 8px 0 12px;
  font-family: "Trebuchet MS", sans-serif;
}}
details {{
  border-top: 1px solid var(--line);
  padding-top: 10px;
}}
summary {{
  cursor: pointer;
  color: var(--accent);
  font-family: "Trebuchet MS", sans-serif;
  font-weight: 700;
}}
pre {{
  white-space: pre-wrap;
  overflow-wrap: anywhere;
  margin: 10px 0 0;
  padding: 12px;
  background: #1f2924;
  color: #f2efe8;
  font-family: Consolas, "Courier New", monospace;
  font-size: 13px;
  line-height: 1.45;
}}
@media (max-width: 820px) {{
  .card {{ grid-template-columns: 1fr; }}
  .summary-table {{ display: block; overflow-x: auto; }}
}}
</style>
</head>
<body>
<header>
  <h1>{html.escape(title)}</h1>
  <table class="summary-table">
    <thead><tr><th>Type</th><th>Correct</th><th>Accuracy</th><th>Answered</th><th>Missing images</th><th>Errors</th></tr></thead>
    <tbody>
      {''.join(summary_rows)}
    </tbody>
  </table>
</header>
<main>
  {''.join(cards)}
</main>
</body>
</html>
"""


def _result_dedupe_key(row: dict[str, Any]) -> str:
    uid = row.get("question_uid")
    return str(uid) if uid else _question_cache_key(row)


def dedupe_results_by_frame_question(
    results: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    seen: set[str] = set()
    filtered: list[dict[str, Any]] = []
    dropped = 0
    for row in results:
        key = _result_dedupe_key(row)
        if key in seen:
            dropped += 1
            continue
        seen.add(key)
        filtered.append(row)
    return filtered, dropped


def _summary_rows_html(summary: dict[str, Any]) -> str:
    rows: list[str] = []
    for qtype, stats in summary.items():
        rows.append(
            "<tr>"
            f'<td class="qtype-cell">{html.escape(_qtype_display(qtype))}</td>'
            f"<td>{stats['correct']} / {stats['total']}</td>"
            f"<td>{_fmt_pct(stats['accuracy'])}</td>"
            f"<td>{stats['answered']}</td>"
            f"<td>{stats['missing_images']}</td>"
            f"<td>{stats['errors']}</td>"
            "</tr>"
        )
    return "".join(rows)


def _ensure_postprocess_css(text: str) -> str:
    if ".summary-table .qtype-cell" not in text:
        text = text.replace(
            '.summary-table th {\n'
            '  color: var(--muted);\n'
            '  font-family: "Trebuchet MS", sans-serif;\n'
            '  text-transform: uppercase;\n'
            '  letter-spacing: 0.03em;\n'
            '}\n',
            '.summary-table th {\n'
            '  color: var(--muted);\n'
            '  font-family: "Trebuchet MS", sans-serif;\n'
            '  text-transform: uppercase;\n'
            '  letter-spacing: 0.03em;\n'
            '}\n'
            '.summary-table .qtype-cell {\n'
            '  color: var(--accent);\n'
            '  font-weight: 800;\n'
            '  font-family: "Trebuchet MS", sans-serif;\n'
            '}\n',
        )
    if ".meta .qtype" not in text:
        text = text.replace(
            ".meta span {\n"
            "  border: 1px solid var(--line);\n"
            "  padding: 3px 7px;\n"
            "  background: rgba(255,255,255,0.52);\n"
            "}\n",
            ".meta span {\n"
            "  border: 1px solid var(--line);\n"
            "  padding: 3px 7px;\n"
            "  background: rgba(255,255,255,0.52);\n"
            "}\n"
            ".meta .qtype {\n"
            "  color: #ffffff;\n"
            "  background: var(--accent);\n"
            "  border-color: var(--accent);\n"
            "  font-weight: 800;\n"
            "}\n",
        )
    return text


def postprocess_existing_html(
    *,
    html_path: Path,
    json_path: Path,
) -> dict[str, int]:
    with json_path.open(encoding="utf-8") as f:
        payload = json.load(f)
    rows = payload.get("results", payload) if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError(f"Unsupported result JSON structure: {json_path}")

    filtered, json_dropped = dedupe_results_by_frame_question(rows)
    summary = summarize_results(filtered)["by_type"]

    if isinstance(payload, dict):
        payload["results"] = filtered
        payload["summary"] = {"by_type": summary}
        metadata = payload.setdefault("metadata", {})
        metadata["result_dedupe_rule"] = (
            "question_uid; fallback dataset + scene_id + image_name + type + question + options"
        )
        metadata["result_dedupe_dropped_count"] = json_dropped
        json_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    else:
        json_path.write_text(
            json.dumps(filtered, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    text = html_path.read_text(encoding="utf-8")
    article_re = re.compile(r'<article class="card ([^"]+)">.*?</article>', re.S)
    meta_re = re.compile(
        r'<div class="meta"><span>#(\d+)</span>'
        r'<span(?: class="qtype")?>([^<]+)</span>'
        r"<span>[^<]*</span>"
        r"<span>([^<]+) / ([^<]+)</span>"
        r'<span class="pill">[^<]+</span></div>'
    )
    h2_re = re.compile(r"<h2>(.*?)</h2>", re.S)

    raw_by_display = {display: raw for raw, display in QTYPE_DISPLAY.items()}
    seen_html: set[tuple[str, str, str]] = set()
    kept_articles: list[str] = []
    html_dropped = 0
    for article_match in article_re.finditer(text):
        article = article_match.group(0)
        meta_match = meta_re.search(article)
        h2_match = h2_re.search(article)
        if not meta_match or not h2_match:
            kept_articles.append(article)
            continue

        qtype_text = html.unescape(meta_match.group(2))
        raw_type = raw_by_display.get(qtype_text, qtype_text)
        scene_id = html.unescape(meta_match.group(3))
        image_name = html.unescape(meta_match.group(4))
        question = html.unescape(re.sub(r"<.*?>", "", h2_match.group(1)))
        dedupe_key = (scene_id, image_name, question)
        if dedupe_key in seen_html:
            html_dropped += 1
            continue
        seen_html.add(dedupe_key)

        display_type = _qtype_display(raw_type)
        next_idx = len(kept_articles) + 1
        article = re.sub(
            r'(<div class="meta"><span>#)\d+(</span>)'
            r'<span(?: class="qtype")?>[^<]+</span>',
            (
                rf"\g<1>{next_idx}\g<2>"
                f'<span class="qtype">{html.escape(display_type)}</span>'
            ),
            article,
            count=1,
        )
        kept_articles.append(article)

    first_article = article_re.search(text)
    last_article: re.Match[str] | None = None
    for last_article in article_re.finditer(text):
        pass
    if first_article and last_article:
        text = (
            text[: first_article.start()]
            + "\n".join(kept_articles)
            + text[last_article.end() :]
        )

    text = re.sub(
        r"<tbody>\s*.*?\s*</tbody>",
        f"<tbody>\n      {_summary_rows_html(summary)}\n    </tbody>",
        text,
        count=1,
        flags=re.S,
    )
    text = _ensure_postprocess_css(text)
    html_path.write_text(text, encoding="utf-8")

    return {
        "json_kept": len(filtered),
        "json_dropped": json_dropped,
        "html_kept": len(kept_articles),
        "html_dropped": html_dropped,
    }


def run_api_question(
    *,
    args: argparse.Namespace,
    client_factory: ThreadLocalOpenAIClientFactory,
    idx: int,
    total: int,
    question: dict[str, Any],
    resolutions: list[ImageResolution],
) -> dict[str, Any]:
    raw_response: str | None = None
    error: str | None = None
    prompt = build_prompt(
        question,
        direct=getattr(args, "direct", False),
        oracle=getattr(args, "oracle", False),
        blind=getattr(args, "blind", False),
    )
    auxiliary_count = max(0, len(resolutions) - 1)
    frame_note = f" (+{auxiliary_count} more frame(s))" if auxiliary_count else ""
    print(
        f"[{idx}/{total}] {question.get('type')} "
        f"{question.get('scene_id')}/{question.get('image_name')}{frame_note} -> API",
        flush=True,
    )
    for attempt in range(args.retries + 1):
        try:
            raw_response = call_model(
                client_factory.get_client(),
                api_provider=args.api_provider,
                model=args.model,
                image_paths=[r.path for r in resolutions],
                prompt=prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                api_image_max_px=args.api_image_max_px,
                blind=getattr(args, "blind", False),
                image_roles=[rollout_role_label(r.role) for r in resolutions],
                direct=getattr(args, "direct", False),
            )
            print(f"[{idx}/{total}] done", flush=True)
            break
        except Exception as exc:  # pragma: no cover - network/API dependent
            if attempt >= args.retries:
                error = f"api_error: {exc}"
                print(f"[{idx}/{total}] failed: {exc}", flush=True)
            else:
                is_rate_limit = "429" in str(exc) or "quota" in str(exc).lower()
                wait = (60.0 if is_rate_limit else args.retry_delay) * (2 ** attempt)
                print(
                    f"[{idx}/{total}] attempt {attempt + 1} failed: {exc}; "
                    f"retrying in {wait:.1f}s",
                    flush=True,
                )
                time.sleep(wait)

    if args.delay > 0:
        time.sleep(args.delay)

    return result_from_question(
        question,
        image_resolutions=resolutions,
        raw_response=raw_response,
        error=error,
    )


def evaluate(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    bev_enabled = bool(getattr(args, "bev", False))
    if args.benchmark_file:
        benchmark_path = Path(args.benchmark_file)
        selected, metadata, sampling_stats = load_fixed_questions(benchmark_path)
        all_questions = selected
        if bev_enabled:
            selected = filter_multi_image_questions(selected)
    else:
        roots = [Path(root) for root in args.root]
        subset_path = Path(args.subset) if args.subset else None
        all_questions, metadata = load_questions(roots, subset_path)
        sampling_pool = (
            filter_multi_image_questions(all_questions) if bev_enabled else all_questions
        )
        selected, sampling_stats = sample_questions(
            sampling_pool,
            per_type=args.per_type,
            scene_cap=args.scene_cap,
            seed=args.seed,
        )

    picture = bool(getattr(args, "picture", False))
    video = bool(getattr(args, "video", False))
    context_only = bool(getattr(args, "context_only", False))
    condition = rollout_condition(
        picture=picture,
        video=video,
        context_only=context_only,
    )
    manifest: RolloutManifest | None = None
    bev_manifest: BevManifest | None = None
    rollout_mode: str | None = "picture" if picture else ("video" if video else None)
    if rollout_mode:
        manifest = load_rollout_manifest(Path(args.rollout_manifest))
        selected, sampling_stats = select_manifest_questions(
            all_questions,
            manifest,
            mode=rollout_mode,
        )
    if bev_enabled:
        if not selected:
            raise ValueError("--bev found no multi-image questions to evaluate")
        bev_manifest = load_bev_manifest(Path(args.bev_dir))
        preflight_bev_questions(selected, bev_manifest)
        condition = f"bev_{bev_manifest.direction_mode}"
        sampling_stats["_bev_coverage"] = {
            "input_questions": len(all_questions),
            "eligible_multi_image_questions": len(
                filter_multi_image_questions(all_questions)
            ),
            "selected": len(selected),
        }
    for question in selected:
        question["_evaluation_condition"] = condition
        if manifest is not None:
            question["_rollout_manifest_sha256"] = manifest.sha256
            manifest_uid = str(question["_rollout_manifest_uid"])
            branch = manifest.entries[manifest_uid][str(rollout_mode)]
            question["_rollout_generation"] = branch.get("generation")
            question["_rollout_quality_audit"] = branch.get("quality_audit")

    oracle_stats: dict[str, int] | None = None
    if getattr(args, "oracle", False):
        scannet_root = _resolve_scannet_geometry_root(list(args.scannet_image_root or []))
        scannetpp_roots = _resolve_scannetpp_geometry_roots(
            list(args.scannetpp_image_root or []),
            getattr(args, "scannetpp_geometry_root", None),
        )
        oracle_mode = str(getattr(args, "oracle_mode", "task_frame") or "task_frame")
        oracle_stats = ensure_runtime_oracle_info(
            selected,
            oracle_mode=oracle_mode,
            scannet_root=scannet_root,
            scannetpp_roots=scannetpp_roots,
            scannetpp_sensor=args.scannetpp_sensor,
            oracle_cache_dir=getattr(args, "oracle_cache_dir", None),
        )
    else:
        oracle_mode = "none"
        scannet_root = None
        scannetpp_roots = []

    metadata.update(
        {
            "input_question_count": len(all_questions),
            "sampled_question_count": len(selected),
            "per_type": args.per_type,
            "scene_cap": args.scene_cap,
            "seed": args.seed,
            "model": args.model,
            "base_url": args.base_url,
            "api_provider": args.api_provider,
            "scannetpp_sensor": args.scannetpp_sensor,
            "vlm_workers": args.vlm_workers,
            "oracle": bool(getattr(args, "oracle", False)),
            "oracle_mode": oracle_mode,
            "oracle_scannet_root": scannet_root,
            "oracle_scannetpp_roots": list(scannetpp_roots),
            "evaluation_condition": condition,
            "rollout_mode": rollout_mode,
            "context_only": context_only,
            "rollout_manifest": str(manifest.path) if manifest else None,
            "rollout_manifest_sha256": manifest.sha256 if manifest else None,
            "bev": bev_enabled,
            "bev_dir": str(Path(args.bev_dir).resolve()) if bev_enabled else None,
            "bev_manifest": str(bev_manifest.path) if bev_manifest else None,
            "bev_manifest_sha256": bev_manifest.sha256 if bev_manifest else None,
            "bev_direction_mode": bev_manifest.direction_mode if bev_manifest else None,
        }
    )
    if oracle_stats is not None:
        metadata["oracle_stats"] = oracle_stats

    output_json = Path(args.output_json)
    existing = load_existing_results(output_json)
    existing_by_cache_key = {
        _question_cache_key(row): row
        for row in existing.values()
    }
    results_by_uid: dict[str, dict[str, Any]] = {}

    client_factory: ThreadLocalOpenAIClientFactory | None = None
    if not args.skip_api:
        api_key, credential_kind = _resolve_api_credential(args)
        client_factory = ThreadLocalOpenAIClientFactory(
            api_provider=args.api_provider,
            base_url=args.base_url,
            api_key=api_key,
            timeout=args.timeout,
            credential_kind=credential_kind,
        )

    only_types: set[str] | None = (
        {t.strip() for t in args.only_type.split(",") if t.strip()}
        if args.only_type
        else None
    )

    api_call_count = 0
    api_work: list[tuple[int, dict[str, Any], list[ImageResolution]]] = []
    for idx, question in enumerate(selected, 1):
        uid = str(question["question_uid"])
        qtype = str(question.get("type") or "")
        cached = existing.get(uid) or existing_by_cache_key.get(_question_cache_key(question))
        cached_condition = str(cached.get("evaluation_condition", "baseline")) if cached else None
        cache_matches_condition = cached_condition == condition
        if manifest is not None and cached is not None:
            cache_matches_condition = cache_matches_condition and (
                cached.get("rollout_manifest_sha256") == manifest.sha256
            )
        if bev_manifest is not None and cached is not None:
            cache_matches_condition = cache_matches_condition and (
                cached.get("bev_manifest_sha256") == bev_manifest.sha256
            )
        # Non-targeted types always use cache (ignore --force).
        # Targeted types (or all types when --only_type is absent) respect --force.
        is_targeted = only_types is None or qtype in only_types
        if (
            cached
            and cache_matches_condition
            and (not args.force or not is_targeted)
            and cached.get("raw_response") is not None
        ):
            results_by_uid[uid] = refresh_cached_result(question, cached)
            continue

        resolution_error: str | None = None
        if getattr(args, "blind", False):
            resolutions = [ImageResolution(None, ())]
        elif manifest is not None and rollout_mode is not None:
            resolutions, resolution_error = resolve_rollout_images(
                question,
                manifest,
                mode=rollout_mode,
                context_only=context_only,
            )
        else:
            resolutions = resolve_question_images(
                question,
                scannet_roots=[Path(p) for p in args.scannet_image_root],
                scannetpp_roots=[Path(p) for p in args.scannetpp_image_root],
                scannetpp_sensor=args.scannetpp_sensor,
            )
            if bev_enabled:
                resolutions.append(resolve_bev_image(question))

        raw_response: str | None = None
        error: str | None = None
        # Fail closed: a two-frame-split question's text promises a photo series
        # ("the last photo shows Y") -- if any required frame (primary or aux) is
        # missing, don't silently send fewer frames than promised.
        if resolution_error:
            error = resolution_error
        elif not getattr(args, "blind", False) and any(r.path is None for r in resolutions):
            error = "image_not_found"
        elif args.skip_api:
            error = "api_skipped"
        else:
            api_work.append((idx, question, resolutions))
            continue

        results_by_uid[uid] = result_from_question(
            question,
            image_resolutions=resolutions,
            raw_response=raw_response,
            error=error,
        )

        if idx % args.checkpoint_every == 0:
            ordered = [results_by_uid[str(q["question_uid"])] for q in selected if str(q["question_uid"]) in results_by_uid]
            save_results(
                output_json,
                metadata=metadata,
                sampling_stats=sampling_stats,
                results=ordered,
            )
            print(f"checkpoint: {len(ordered)}/{len(selected)} results saved")

    if api_work:
        if client_factory is None:
            raise RuntimeError("client_factory was not initialized")
        workers = max(1, int(args.vlm_workers))
        print(f"running {len(api_work)} VLM request(s) with --vlm_workers {workers}", flush=True)

        def _store_result(row: dict[str, Any]) -> None:
            nonlocal api_call_count
            results_by_uid[str(row["question_uid"])] = row
            api_call_count += 1
            if api_call_count % args.checkpoint_every == 0:
                ordered_rows = [
                    results_by_uid[str(q["question_uid"])]
                    for q in selected
                    if str(q["question_uid"]) in results_by_uid
                ]
                save_results(
                    output_json,
                    metadata=metadata,
                    sampling_stats=sampling_stats,
                    results=ordered_rows,
                )
                print(f"checkpoint: {len(ordered_rows)}/{len(selected)} results saved")

        if workers == 1:
            for idx, question, resolutions in api_work:
                _store_result(
                    run_api_question(
                        args=args,
                        client_factory=client_factory,
                        idx=idx,
                        total=len(selected),
                        question=question,
                        resolutions=resolutions,
                    )
                )
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [
                    executor.submit(
                        run_api_question,
                        args=args,
                        client_factory=client_factory,
                        idx=idx,
                        total=len(selected),
                        question=question,
                        resolutions=resolutions,
                    )
                    for idx, question, resolutions in api_work
                ]
                for future in as_completed(futures):
                    _store_result(future.result())

    results = [results_by_uid[str(q["question_uid"])] for q in selected]
    metadata["api_calls_made"] = api_call_count
    return results, metadata, sampling_stats


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample questions by type, run a VLM, and build an HTML report."
    )
    parser.add_argument("--root", action="append", default=None, help="Output root to scan for benchmark.json files")
    parser.add_argument("--benchmark_file", default=None, help="Fixed benchmark JSON to evaluate directly; skips runtime sampling")
    parser.add_argument("--subset", default="output/benchmark_subset.json", help="Benchmark subset JSON to sample from; overrides --root. Defaults to the curated benchmark_subset.json")
    parser.add_argument("--per_type", type=int, default=50, help="Questions sampled per type")
    parser.add_argument("--scene_cap", type=int, default=3, help="Max questions per scene within each type before relaxation")
    parser.add_argument("--seed", type=int, default=20260602, help="Random seed for sampling")
    parser.add_argument("--scannet_image_root", action="append", default=None, help="ScanNet image root; can be repeated")
    parser.add_argument("--scannetpp_image_root", action="append", default=None, help="ScanNet++ image root; can be repeated")
    parser.add_argument("--scannetpp_sensor", choices=("iphone", "dslr"), default="iphone", help="ScanNet++ image layout, matching scripts/make_viewer.py")
    parser.add_argument("--base_url", "--vlm_url", dest="base_url", default="https://www.packyapi.com/v1", help="OpenAI-compatible API base URL for the VLM server")
    parser.add_argument("--model", "--vlm_model", dest="model", default="qwen3.5-flash", help="Served model name exposed by the VLM endpoint")
    parser.add_argument(
        "--api_provider",
        choices=("openai_chat", "openai_responses", "anthropic"),
        default="openai_chat",
        help="Wire protocol to use for image+text VLM calls",
    )
    parser.add_argument("--api_key", default=None, help="API key; otherwise read from --api_key_env or provider defaults")
    parser.add_argument("--api_key_env", default=None, help="Environment variable for API key")
    parser.add_argument("--max_tokens", type=int, default=3072, help="Maximum model output tokens")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--api_image_max_px", type=int, default=1280, help="Resize longest image side for API; 0 disables")
    parser.add_argument("--html_image_max_px", type=int, default=720, help="Resize longest image side embedded in HTML; 0 disables")
    parser.add_argument("--delay", type=float, default=0.2, help="Delay between API calls")
    parser.add_argument("--timeout", type=float, default=60.0, help="Per-request API timeout in seconds")
    parser.add_argument("--retries", type=int, default=2, help="Retries per API call")
    parser.add_argument("--retry_delay", type=float, default=2.0, help="Initial retry delay in seconds")
    parser.add_argument("--vlm_workers", type=int, default=1, help="Maximum number of concurrent VLM requests")
    parser.add_argument("--checkpoint_every", type=int, default=1, help="Save JSON every N processed questions")
    parser.add_argument("--output_json", default="output/type_sample_vlm_eval/results.json", help="Resumable JSON result path")
    parser.add_argument("--output_html", default="output/type_sample_vlm_eval/viewer.html", help="HTML report path")
    parser.add_argument("--title", default="Sampled VLM Spatial QA Evaluation", help="HTML report title")
    parser.add_argument("--skip_api", action="store_true", help="Only sample and build a report skeleton; do not call the API")
    parser.add_argument("--blind", action="store_true", help="Text-only baseline: omit image from all API requests")
    parser.add_argument("--direct", action="store_true", help="Direct-answer baseline: ask for a single letter with no reasoning")
    parser.add_argument("--oracle", action="store_true", help="Generate/prepend ground-truth 3D oracle information to each prompt")
    parser.add_argument(
        "--bev",
        action="store_true",
        help="Evaluate only multi-image questions and append an initial-state BEV image",
    )
    parser.add_argument(
        "--bev_dir",
        default=None,
        help="Directory produced by scripts/generate_bev_images.py (requires --bev)",
    )
    rollout_group = parser.add_mutually_exclusive_group()
    rollout_group.add_argument(
        "--picture",
        action="store_true",
        help="Use the ordered picture-rollout media from --rollout_manifest",
    )
    rollout_group.add_argument(
        "--video",
        action="store_true",
        help="Use the motion context and eight video-rollout frames from --rollout_manifest",
    )
    parser.add_argument(
        "--context_only",
        action="store_true",
        help="Omit prediction media while retaining the selected rollout branch's real context",
    )
    parser.add_argument(
        "--rollout_manifest",
        default=None,
        help="Public, answer-free predictive-spatial-rollout-v1 evaluation manifest",
    )
    parser.add_argument("--oracle_mode", choices=("world", "task_frame"), default="task_frame", help="Oracle coordinate mode used when --oracle is set")
    parser.add_argument("--oracle_cache_dir", default=None, help="Directory for pre-computed oracle scene cache (pickle files). Pre-compute with scripts/precompute_oracle_cache.py.")
    parser.add_argument("--scannetpp_geometry_root", default=None, help="Optional ScanNet++ geometry root; defaults to auto-detecting data/scannetpp, ++data, then --scannetpp_image_root")
    parser.add_argument("--force", action="store_true", help="Re-run questions even if cached in output_json")
    parser.add_argument(
        "--only_type",
        default=None,
        help=(
            "Comma-separated question type(s) to re-run (e.g. coordinate_rotation_object_centric). "
            "Questions of other types are loaded from cache unchanged. "
            "Implies --force for the targeted type(s) only."
        ),
    )
    parser.add_argument(
        "--postprocess_existing_html",
        action="store_true",
        help="Deduplicate and relabel an existing embedded HTML report without rebuilding images",
    )
    args = parser.parse_args(argv)

    if args.root is None:
        args.root = ["output/pilot", "output/scannetpp_polit"]
    if args.scannet_image_root is None:
        args.scannet_image_root = ["data/scannet"]
    if args.scannetpp_image_root is None:
        args.scannetpp_image_root = ["output/scannetpp_iphone_frames", "++data"]
    if args.per_type <= 0:
        parser.error("--per_type must be positive")
    if args.scene_cap <= 0:
        parser.error("--scene_cap must be positive")
    if args.checkpoint_every <= 0:
        parser.error("--checkpoint_every must be positive")
    if args.vlm_workers <= 0:
        parser.error("--vlm_workers must be positive")
    rollout_enabled = bool(args.picture or args.video)
    if args.bev and not args.bev_dir:
        parser.error("--bev requires --bev_dir")
    if args.bev_dir and not args.bev:
        parser.error("--bev_dir requires --bev")
    if args.bev and (args.oracle or rollout_enabled or args.context_only or args.blind):
        parser.error(
            "--bev cannot be combined with --oracle, --picture, --video, "
            "--context_only, or --blind"
        )
    if args.bev_dir and not Path(args.bev_dir).is_dir():
        parser.error(f"--bev_dir not found: {args.bev_dir}")
    if args.context_only and not rollout_enabled:
        parser.error("--context_only requires --picture or --video")
    if rollout_enabled and not args.rollout_manifest:
        parser.error("--picture/--video requires --rollout_manifest")
    if args.rollout_manifest and not rollout_enabled:
        parser.error("--rollout_manifest requires --picture or --video")
    if rollout_enabled and args.blind:
        parser.error("--blind cannot be combined with --picture or --video")
    if rollout_enabled and args.oracle:
        parser.error("--oracle cannot be combined with --picture or --video")
    if args.rollout_manifest and not Path(args.rollout_manifest).is_file():
        parser.error(f"--rollout_manifest not found: {args.rollout_manifest}")
    if args.benchmark_file:
        benchmark_path = Path(args.benchmark_file)
        if not benchmark_path.exists():
            parser.error(f"--benchmark_file not found: {args.benchmark_file}")
    if args.subset:
        subset_path = Path(args.subset)
        if not subset_path.exists():
            parser.error(f"--subset not found: {args.subset}")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.postprocess_existing_html:
        stats = postprocess_existing_html(
            html_path=Path(args.output_html),
            json_path=Path(args.output_json),
        )
        print(f"json kept       : {stats['json_kept']}")
        print(f"json dropped    : {stats['json_dropped']}")
        print(f"html kept       : {stats['html_kept']}")
        print(f"html dropped    : {stats['html_dropped']}")
        print(f"json output     : {args.output_json}")
        print(f"html output     : {args.output_html}")
        return

    results, metadata, sampling_stats = evaluate(args)

    output_json = Path(args.output_json)
    save_results(
        output_json,
        metadata=metadata,
        sampling_stats=sampling_stats,
        results=results,
    )

    output_html = Path(args.output_html)
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(
        build_html(results, title=args.title, html_image_max_px=args.html_image_max_px),
        encoding="utf-8",
    )

    summary = summarize_results(results)["by_type"]
    print(f"loaded questions : {metadata['input_question_count']}")
    print(f"sampled questions: {len(results)}")
    print(f"json output      : {output_json}")
    print(f"html output      : {output_html}")
    for qtype, stats in summary.items():
        print(
            f"{qtype:36s} {stats['correct']:4d}/{stats['total']:<4d} "
            f"acc={_fmt_pct(stats['accuracy']):>6s} answered={stats['answered']}"
        )


if __name__ == "__main__":
    main(sys.argv[1:])
