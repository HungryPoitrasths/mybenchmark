#!/usr/bin/env python3
"""Run the full question-generation pipeline for one specified ScanNet frame."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.make_pipeline_trace_viewer import build_single_frame_trace_html
from scripts.run_pipeline import (
    DEFAULT_VLM_URL,
    _build_frame_debug_entry,
    _build_scene_attachment_rows,
    _filter_frame_attachment_rows,
    _get_referability_entry,
    _has_l1_visibility_candidates,
    _load_referability_cache,
    _normalize_label_counts,
    _normalize_label_statuses,
    _normalize_object_ids,
    _resolve_question_review_vlm,
)
from scripts.run_vlm_referability import (
    LABEL_BATCH_SIZE,
    _compute_frame_referability_entry,
    _frame_entry_has_debug_fields,
)
from src.frame_selector import get_visible_objects
from src.qa_generator import generate_all_questions
from src.quality_control import quality_filter
from src.scene_parser import (
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
from src.utils import RayCaster
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
logger = logging.getLogger("single_frame_trace")


def _json_clone(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False))


def _sanitize_question(question: dict[str, Any]) -> dict[str, Any]:
    payload = _json_clone(question)
    trace_source = payload.pop("_trace_source", None)
    if trace_source is not None and "trace_source" not in payload:
        payload["trace_source"] = trace_source
    return payload


def _sanitize_questions(questions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [_sanitize_question(question) for question in questions]


def _resolve_scene_dir(data_root: Path, scene_id: str) -> Path:
    for candidate in (data_root / scene_id, data_root / "scans" / scene_id):
        if candidate.exists():
            return candidate
    return data_root / scene_id


def _ensure_trace_question_ids(
    questions: list[dict[str, Any]],
    *,
    trace_id_prefix: str,
) -> None:
    seen_ids = {
        str(question.get("trace_question_id"))
        for question in questions
        if question.get("trace_question_id")
    }
    counter = 0
    for question in questions:
        if question.get("trace_question_id"):
            continue
        while True:
            counter += 1
            candidate = f"{trace_id_prefix}_{counter:04d}"
            if candidate not in seen_ids:
                question["trace_question_id"] = candidate
                seen_ids.add(candidate)
                break
        if "_trace_source" not in question:
            question["_trace_source"] = str(question.get("type") or "unknown")


def _record_stage(
    trace_doc: dict[str, Any],
    *,
    stage: str,
    status: str,
    started_at: float,
    details: dict[str, Any] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "stage": stage,
        "status": status,
        "elapsed_ms": round((time.perf_counter() - started_at) * 1000, 1),
    }
    if details:
        payload["details"] = details
    trace_doc.setdefault("stage_summaries", []).append(payload)


def _pose_examples(poses: dict[str, Any], *, limit: int = 20) -> list[str]:
    names = sorted(str(name) for name in poses.keys())
    return names[:limit]


def _summarize_generators(trace_events: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for event in trace_events:
        if event.get("event") != "generator_output":
            continue
        generator = str(event.get("generator", "")).strip()
        if not generator:
            continue
        counts[generator] = counts.get(generator, 0) + int(event.get("count", 0))
    return dict(sorted(counts.items()))


def _sanitize_vlm_label_reviews(
    reviews: Any,
    *,
    payload_mode: str,
) -> list[dict[str, Any]]:
    if not isinstance(reviews, list):
        return []
    mode = str(payload_mode).strip().lower()
    sanitized: list[dict[str, Any]] = []
    for review in reviews:
        if not isinstance(review, dict):
            continue
        item = _json_clone(review)
        if mode == "none":
            continue
        if mode == "summary":
            item.pop("raw_response", None)
        sanitized.append(item)
    return sanitized


def _sanitize_object_reviews(
    reviews: Any,
    *,
    payload_mode: str,
) -> dict[str, dict[str, Any]]:
    if not isinstance(reviews, dict):
        return {}
    mode = str(payload_mode).strip().lower()
    sanitized: dict[str, dict[str, Any]] = {}
    for obj_id, review in reviews.items():
        if not isinstance(review, dict):
            continue
        item = _json_clone(review)
        if mode == "none":
            item.pop("raw_response", None)
        elif mode == "summary":
            item.pop("raw_response", None)
        sanitized[str(obj_id)] = item
    return dict(sorted(sanitized.items()))


def _build_referability_audit(
    referability_entry: dict[str, Any] | None,
    *,
    referability_source: str | None,
    payload_mode: str,
) -> dict[str, Any]:
    entry = _json_clone(referability_entry or {})
    entry["referability_source"] = referability_source
    entry["vlm_label_reviews"] = _sanitize_vlm_label_reviews(
        entry.get("vlm_label_reviews"),
        payload_mode=payload_mode,
    )
    entry["object_reviews"] = _sanitize_object_reviews(
        entry.get("object_reviews"),
        payload_mode=payload_mode,
    )
    if str(payload_mode).strip().lower() == "none":
        entry.pop("vlm_label_reviews", None)
        entry.pop("object_reviews", None)
    return entry


def _safe_audit_name(name: str) -> str:
    chars = []
    for ch in str(name):
        if ch.isalnum():
            chars.append(ch.lower())
        else:
            chars.append("_")
    safe = "".join(chars).strip("_")
    while "__" in safe:
        safe = safe.replace("__", "_")
    return safe or "unknown"


def _build_generator_audits(trace_events: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    generator_docs: dict[str, dict[str, Any]] = {}
    for event in trace_events:
        generator = str(event.get("generator", "")).strip()
        if not generator:
            continue
        doc = generator_docs.setdefault(
            generator,
            {
                "generator": generator,
                "context_events": [],
                "candidate_events": [],
                "summary_events": [],
                "output_events": [],
                "cap_events": [],
                "other_events": [],
            },
        )
        event_name = str(event.get("event", "")).strip()
        if event_name == "generator_context":
            doc["context_events"].append(event)
        elif event_name == "generator_candidate":
            doc["candidate_events"].append(event)
        elif event_name == "generator_summary":
            doc["summary_events"].append(event)
        elif event_name == "generator_output":
            doc["output_events"].append(event)
        elif event_name == "generator_cap_applied":
            doc["cap_events"].append(event)
        else:
            doc["other_events"].append(event)

    for generator, doc in generator_docs.items():
        output_events = doc.get("output_events", [])
        summary_events = doc.get("summary_events", [])
        candidate_events = doc.get("candidate_events", [])
        generated_question_ids: list[str] = []
        output_count = 0
        for event in output_events:
            output_count += int(event.get("count", 0))
            generated_question_ids.extend(
                str(question_id)
                for question_id in event.get("question_ids", [])
                if str(question_id).strip()
            )
        latest_summary = summary_events[-1] if summary_events else {}
        doc["generated_question_ids"] = generated_question_ids
        doc["output_count"] = int(output_count)
        doc["candidate_count"] = int(
            latest_summary.get("candidate_count", len(candidate_events))
        )
        doc["generated_count"] = int(
            latest_summary.get("generated_count", output_count)
        )
        doc["reason_counts"] = latest_summary.get("reason_counts", {})
        doc["summary"] = latest_summary
        doc["has_candidate_audit"] = bool(candidate_events)
        doc["audit_mode"] = (
            ((latest_summary.get("details") or {}).get("audit_mode"))
            if isinstance(latest_summary.get("details"), dict)
            else None
        ) or ("full" if candidate_events else "summary_only")
        doc["audit_file_name"] = f"generator_{_safe_audit_name(generator)}.json"
    return dict(sorted(generator_docs.items()))


def _build_reason_index(
    *,
    trace_doc: dict[str, Any],
    trace_events: list[dict[str, Any]],
    question_lifecycle: list[dict[str, Any]],
    generator_audits: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    by_generator: dict[str, dict[str, int]] = {}
    global_blockers: Counter[str] = Counter()
    for generator, audit in generator_audits.items():
        reason_counts = Counter()
        for event in audit.get("candidate_events", []):
            reason_code = str(event.get("reason_code", "")).strip()
            if not reason_code:
                continue
            reason_counts[reason_code] += 1
            if reason_code != "generated":
                global_blockers[f"{generator}:{reason_code}"] += 1
        if reason_counts:
            by_generator[generator] = dict(sorted(reason_counts.items()))

    lifecycle_removals = Counter(
        str(row.get("removal_reason", "")).strip()
        for row in question_lifecycle
        if str(row.get("status")) == "removed" and str(row.get("removal_reason", "")).strip()
    )
    quality_filter_removals = Counter(
        str(event.get("reason", "")).strip()
        for event in trace_events
        if event.get("event") == "question_removed" and str(event.get("stage")) == "quality_filter"
    )
    stage_stops = [
        {
            "stage": entry.get("stage"),
            "reason": (entry.get("details") or {}).get("reason"),
            "status": entry.get("status"),
        }
        for entry in trace_doc.get("stage_summaries", [])
        if str(entry.get("status")) == "stopped"
    ]
    top_blockers = [
        {"reason": reason, "count": count}
        for reason, count in global_blockers.most_common(10)
    ]
    return {
        "status": trace_doc.get("status"),
        "stop_reason": trace_doc.get("stop_reason"),
        "question_count": len(trace_doc.get("final_questions", [])),
        "raw_question_count": len(trace_doc.get("raw_questions", [])),
        "stage_stops": stage_stops,
        "generator_reason_counts": dict(sorted(by_generator.items())),
        "quality_filter_removals": dict(sorted(quality_filter_removals.items())),
        "lifecycle_removals": dict(sorted(lifecycle_removals.items())),
        "top_blockers": top_blockers,
    }


def _build_trace_recorder(
    trace_events: list[dict[str, Any]],
    question_snapshots: dict[str, dict[str, Any]],
    question_events: dict[str, list[dict[str, Any]]],
) -> Callable[[dict[str, Any]], None]:
    def _record(event: dict[str, Any]) -> None:
        event_copy = _json_clone(event)
        if isinstance(event_copy.get("question"), dict):
            event_copy["question"] = _sanitize_question(event_copy["question"])
        if isinstance(event_copy.get("questions"), list):
            event_copy["questions"] = [
                _sanitize_question(question)
                for question in event_copy["questions"]
                if isinstance(question, dict)
            ]
        trace_events.append(event_copy)

        event_name = str(event_copy.get("event", "")).strip()
        if event_name == "generator_output":
            generator = str(event_copy.get("generator", "")).strip() or "unknown"
            for question in event_copy.get("questions", []):
                trace_question_id = str(question.get("trace_question_id", "")).strip()
                if not trace_question_id:
                    continue
                question_snapshots[trace_question_id] = question
                question_events[trace_question_id].append(
                    {
                        "event": "generated",
                        "stage": "qa_generation",
                        "generator": generator,
                    }
                )
            return

        if event_name == "generator_cap_applied":
            generator = str(event_copy.get("generator", "")).strip() or "unknown"
            for trace_question_id in event_copy.get("removed_question_ids", []):
                if not str(trace_question_id).strip():
                    continue
                question_events[str(trace_question_id)].append(
                    {
                        "event": "question_removed",
                        "stage": "qa_generation",
                        "filter": "generator_cap",
                        "reason": f"{generator}_cap",
                    }
                )
            return

        if event_name == "question_removed":
            question = event_copy.get("question")
            trace_question_id = str(event_copy.get("trace_question_id", "")).strip()
            if not trace_question_id and isinstance(question, dict):
                trace_question_id = str(question.get("trace_question_id", "")).strip()
            if not trace_question_id:
                return
            if isinstance(question, dict):
                question_snapshots.setdefault(trace_question_id, question)
            question_events[trace_question_id].append(
                {
                    "event": "question_removed",
                    "stage": str(event_copy.get("stage", "")),
                    "filter": str(event_copy.get("filter", "")),
                    "reason": str(event_copy.get("reason", "")),
                    "detail": event_copy.get("detail"),
                    "details": event_copy.get("details"),
                    "duplicate_of_trace_question_id": event_copy.get("duplicate_of_trace_question_id"),
                    "duplicate_of_question": event_copy.get("duplicate_of_question"),
                }
            )

    return _record


def _build_question_lifecycle(
    *,
    question_snapshots: dict[str, dict[str, Any]],
    question_events: dict[str, list[dict[str, Any]]],
    raw_questions: list[dict[str, Any]],
    final_questions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    raw_map = {
        str(question.get("trace_question_id")): _sanitize_question(question)
        for question in raw_questions
        if question.get("trace_question_id")
    }
    final_map = {
        str(question.get("trace_question_id")): _sanitize_question(question)
        for question in final_questions
        if question.get("trace_question_id")
    }
    all_ids = sorted(
        set(question_snapshots.keys())
        | set(raw_map.keys())
        | set(final_map.keys())
        | set(question_events.keys())
    )
    lifecycle: list[dict[str, Any]] = []
    for trace_question_id in all_ids:
        snapshot = _json_clone(
            final_map.get(trace_question_id)
            or raw_map.get(trace_question_id)
            or question_snapshots.get(trace_question_id, {})
        )
        events = _json_clone(question_events.get(trace_question_id, []))
        removal_event = next(
            (
                event
                for event in reversed(events)
                if str(event.get("event", "")) == "question_removed"
            ),
            None,
        )
        status = "kept" if trace_question_id in final_map else "removed" if removal_event else "generated"
        lifecycle.append(
            {
                "trace_question_id": trace_question_id,
                "status": status,
                "trace_source": snapshot.get("trace_source") or snapshot.get("_trace_source"),
                "level": snapshot.get("level"),
                "type": snapshot.get("type"),
                "question": snapshot.get("question"),
                "answer": snapshot.get("answer"),
                "removal_filter": None if removal_event is None else removal_event.get("filter"),
                "removal_reason": None if removal_event is None else removal_event.get("reason"),
                "removal_detail": None if removal_event is None else removal_event.get("detail"),
                "removal_details": None if removal_event is None else removal_event.get("details"),
                "duplicate_of_trace_question_id": None if removal_event is None else removal_event.get("duplicate_of_trace_question_id"),
                "duplicate_of_question": None if removal_event is None else removal_event.get("duplicate_of_question"),
                "events": events,
            }
        )
    return lifecycle


def _compute_single_frame_referability_entry(
    *,
    scene: dict[str, Any],
    scene_dir: Path,
    image_name: str,
    image_path: Path,
    camera_pose,
    color_intrinsics,
    depth_intrinsics,
    objects_by_id: dict[int, dict[str, Any]],
    vlm_url: str | None,
    vlm_model: str | None,
    label_batch_size: int,
) -> tuple[dict[str, Any], str]:
    _ = int(label_batch_size)
    if color_intrinsics is None:
        raise RuntimeError("color intrinsics are required for single-frame referability fallback")
    if not image_path.exists():
        raise RuntimeError(f"image not found: {image_path}")
    client, model_name = _resolve_question_review_vlm(
        vlm_url,
        vlm_model,
        purpose="single-frame referability",
    )

    selector_visible_object_ids = [
        int(object_entry["id"])
        for object_entry in get_visible_objects(scene["objects"], camera_pose, color_intrinsics)
        if int(object_entry["id"]) in objects_by_id
    ]

    depth_image = None
    depth_path = scene_dir / "depth" / f"{Path(image_name).stem}.png"
    if depth_intrinsics is not None and depth_path.exists():
        try:
            depth_image = load_depth_image(depth_path)
        except Exception as exc:
            logger.warning("Depth load failed for referability fallback %s/%s: %s", scene_dir.name, image_name, exc)

    import cv2

    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"cannot read image: {image_path}")
    frame_entry = _compute_frame_referability_entry(
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
    )
    frame_entry["referability_model"] = model_name
    return frame_entry, "online"


def run_single_frame_trace(
    *,
    data_root: Path,
    scene_id: str,
    image_name: str,
    output_dir: Path,
    referability_cache: dict | None = None,
    referability_cache_path: Path | None = None,
    use_occlusion: bool = True,
    occlusion_backend: str = "mesh_ray",
    vlm_url: str | None = None,
    vlm_model: str | None = None,
    label_batch_size: int = LABEL_BATCH_SIZE,
    trace_detail: str = "full",
    trace_vlm_payload: str = "summary",
) -> dict[str, Any]:
    frame_stem = Path(image_name).stem
    scene_dir = _resolve_scene_dir(data_root, scene_id)
    output_root = output_dir / "single_frame" / scene_id / frame_stem
    output_root.mkdir(parents=True, exist_ok=True)
    trace_json_path = output_root / "trace.json"
    trace_html_path = output_root / "trace.html"
    final_questions_path = output_root / "final_questions.json"
    audits_dir = output_root / "audits"
    audits_dir.mkdir(parents=True, exist_ok=True)
    image_path = scene_dir / "color" / image_name
    trace_doc: dict[str, Any] = {
        "name": "CausalSpatial-Bench single-frame trace",
        "version": "1.0",
        "status": "running",
        "stop_reason": None,
        "stop_details": {},
        "input": {
            "data_root": str(data_root),
            "scene_id": scene_id,
            "image_name": image_name,
            "scene_dir": str(scene_dir),
            "image_path": str(image_path),
            "referability_cache_path": None if referability_cache_path is None else str(referability_cache_path),
            "occlusion_backend": occlusion_backend,
            "use_occlusion": bool(use_occlusion),
            "vlm_url": vlm_url,
            "vlm_model": vlm_model,
            "trace_detail": trace_detail,
            "trace_vlm_payload": trace_vlm_payload,
        },
        "artifacts": {
            "trace_json_path": str(trace_json_path),
            "trace_html_path": str(trace_html_path),
            "final_questions_path": str(final_questions_path),
            "audits_dir": str(audits_dir),
            "audits": {},
        },
        "stage_summaries": [],
        "frame_context": {},
        "quality_control": {},
        "trace_events": [],
        "raw_questions": [],
        "final_questions": [],
        "question_lifecycle": [],
    }
    trace_events: list[dict[str, Any]] = []
    question_snapshots: dict[str, dict[str, Any]] = {}
    question_events: dict[str, list[dict[str, Any]]] = defaultdict(list)
    trace_recorder = _build_trace_recorder(trace_events, question_snapshots, question_events)
    raw_questions: list[dict[str, Any]] = []
    final_questions: list[dict[str, Any]] = []
    referability_entry: dict[str, Any] | None = None
    referability_source: str | None = None

    try:
        stage_started = time.perf_counter()
        if not scene_dir.exists():
            _record_stage(
                trace_doc,
                stage="stage_1_parse",
                status="stopped",
                started_at=stage_started,
                details={"reason": "scene_directory_not_found"},
            )
            trace_doc["status"] = "stopped"
            trace_doc["stop_reason"] = "scene_directory_not_found"
            return trace_doc

        needs_mesh_resources = occlusion_backend in ("depth", "mesh_ray")
        preloaded_geometry = None
        if needs_mesh_resources:
            try:
                preloaded_geometry = _load_scene_geometry(scene_dir)
            except Exception as exc:
                logger.warning("Scene geometry preload failed for %s: %s", scene_id, exc)

        scene = parse_scene(scene_dir, preloaded_geometry=preloaded_geometry)
        if scene is None:
            _record_stage(
                trace_doc,
                stage="stage_1_parse",
                status="stopped",
                started_at=stage_started,
                details={"reason": "parse_scene_returned_none"},
            )
            trace_doc["status"] = "stopped"
            trace_doc["stop_reason"] = "parse_scene_returned_none"
            return trace_doc
        _record_stage(
            trace_doc,
            stage="stage_1_parse",
            status="completed",
            started_at=stage_started,
            details={"object_count": len(scene.get("objects", []))},
        )

        stage_started = time.perf_counter()
        enrich_scene_with_attachment(scene)
        attachment_graph = get_scene_attachment_graph(scene, scene_id=scene_id)
        attached_by = get_scene_attached_by(scene, scene_id=scene_id)
        support_chain_graph = get_scene_support_chain_graph(scene, scene_id=scene_id)
        support_chain_by = get_scene_support_chain_by(scene, scene_id=scene_id)
        scene_attachment_rows = _build_scene_attachment_rows(scene)
        objects_by_id = {int(obj["id"]): obj for obj in scene["objects"]}
        if not has_nontrivial_attachment(attachment_graph):
            _record_stage(
                trace_doc,
                stage="stage_2_attachment",
                status="stopped",
                started_at=stage_started,
                details={"reason": "no_nontrivial_attachment", "attachment_rows": len(scene_attachment_rows)},
            )
            trace_doc["status"] = "stopped"
            trace_doc["stop_reason"] = "no_nontrivial_attachment"
            return trace_doc
        _record_stage(
            trace_doc,
            stage="stage_2_attachment",
            status="completed",
            started_at=stage_started,
            details={"attachment_rows": len(scene_attachment_rows)},
        )

        stage_started = time.perf_counter()
        axis_align = load_axis_alignment(scene_dir)
        poses = load_scannet_poses(scene_dir, axis_alignment=axis_align)
        pose_dir = scene_dir / "pose"
        preflight_context = {
            "scene_dir": str(scene_dir),
            "pose_dir": str(pose_dir),
            "image_path": str(image_path),
            "image_uri": image_path.resolve().as_uri() if image_path.exists() else None,
            "requested_image_name": image_name,
            "image_exists": image_path.exists(),
            "available_pose_count": len(poses),
            "available_pose_examples": _pose_examples(poses),
        }
        trace_doc["frame_context"] = {**trace_doc.get("frame_context", {}), **preflight_context}
        if image_name not in poses:
            _record_stage(
                trace_doc,
                stage="stage_3_forced_frame",
                status="stopped",
                started_at=stage_started,
                details={
                    "reason": "missing_pose",
                    "requested_image_name": image_name,
                    "available_pose_count": len(poses),
                    "available_pose_examples": _pose_examples(poses),
                },
            )
            trace_doc["status"] = "stopped"
            trace_doc["stop_reason"] = "missing_pose"
            trace_doc["stop_details"] = {
                "requested_image_name": image_name,
                "expected_pose_dir": str(pose_dir),
                "image_path": str(image_path),
                "image_exists": image_path.exists(),
                "available_pose_count": len(poses),
                "available_pose_examples": _pose_examples(poses),
            }
            return trace_doc
        camera_pose = poses[image_name]
        try:
            color_intrinsics = load_scannet_intrinsics(scene_dir)
        except Exception as exc:
            logger.warning("Color intrinsics load failed for %s: %s", scene_id, exc)
            color_intrinsics = None
        try:
            depth_intrinsics = load_scannet_depth_intrinsics(scene_dir) if use_occlusion else None
        except Exception as exc:
            logger.warning("Depth intrinsics load failed for %s: %s", scene_id, exc)
            depth_intrinsics = None
        _record_stage(
            trace_doc,
            stage="stage_3_forced_frame",
            status="completed" if image_path.exists() else "stopped",
            started_at=stage_started,
            details={
                "forced_frame": image_name,
                "image_exists": image_path.exists(),
                "color_intrinsics_loaded": color_intrinsics is not None,
                "depth_intrinsics_loaded": depth_intrinsics is not None,
            },
        )
        if not image_path.exists():
            trace_doc["status"] = "stopped"
            trace_doc["stop_reason"] = "missing_image"
            trace_doc["stop_details"] = {
                "requested_image_name": image_name,
                "image_path": str(image_path),
                "expected_pose_dir": str(pose_dir),
                "available_pose_count": len(poses),
            }
            return trace_doc

        stage_started = time.perf_counter()
        referability_entry = _get_referability_entry(referability_cache, scene_id, image_name)
        referability_source = "cache"
        if not _frame_entry_has_debug_fields(referability_entry):
            referability_entry, referability_source = _compute_single_frame_referability_entry(
                scene=scene,
                scene_dir=scene_dir,
                image_name=image_name,
                image_path=image_path,
                camera_pose=camera_pose,
                color_intrinsics=color_intrinsics,
                depth_intrinsics=depth_intrinsics,
                objects_by_id=objects_by_id,
                vlm_url=vlm_url,
                vlm_model=vlm_model,
                label_batch_size=label_batch_size,
            )
        referability_entry = _json_clone(referability_entry)
        trace_doc["input"]["referability_source"] = referability_source
        label_statuses = _normalize_label_statuses(referability_entry.get("label_statuses"))
        label_counts = _normalize_label_counts(referability_entry.get("label_counts"))
        selector_visible_ids = _normalize_object_ids(
            referability_entry.get("selector_visible_object_ids")
            or referability_entry.get("candidate_visible_object_ids")
        )
        visible_ids = _normalize_object_ids(
            referability_entry.get("candidate_visible_object_ids")
            or referability_entry.get("selector_visible_object_ids")
        )
        visible_id_set = set(visible_ids)
        referable_ids = [
            int(obj_id)
            for obj_id in _normalize_object_ids(referability_entry.get("referable_object_ids"))
            if int(obj_id) in visible_id_set
        ]
        _record_stage(
            trace_doc,
            stage="referability",
            status="completed" if referability_entry.get("frame_usable", True) else "stopped",
            started_at=stage_started,
            details={
                "source": referability_source,
                "frame_usable": bool(referability_entry.get("frame_usable", True)),
                "selector_visible_count": len(selector_visible_ids),
                "candidate_visible_count": len(visible_ids),
                "referable_count": len(referable_ids),
            },
        )
        frame_attachment_rows = _filter_frame_attachment_rows(
            scene_attachment_rows,
            set(selector_visible_ids) | set(visible_ids),
        )
        trace_doc["frame_context"] = {
            **trace_doc.get("frame_context", {}),
            **_build_frame_debug_entry(
            image_name=image_name,
            scene_objects=scene["objects"],
            objects_by_id=objects_by_id,
            selector_visible_ids=selector_visible_ids,
            pipeline_visible_ids=visible_ids,
            referability_entry=referability_entry,
            frame_attachment_rows=frame_attachment_rows,
            ),
        }
        trace_doc["frame_context"]["image_path"] = str(image_path)
        trace_doc["frame_context"]["image_uri"] = image_path.resolve().as_uri() if image_path.exists() else None
        trace_doc["frame_context"]["referability_source"] = referability_source
        if not referability_entry.get("frame_usable", True):
            trace_doc["status"] = "stopped"
            trace_doc["stop_reason"] = "frame_rejected_by_vlm_frame_review"
            trace_doc["stop_details"] = {
                "requested_image_name": image_name,
                "frame_reject_reason": referability_entry.get("frame_reject_reason"),
                "referability_source": referability_source,
            }
            return trace_doc
        if not referable_ids and not _has_l1_visibility_candidates(label_counts):
            trace_doc["status"] = "stopped"
            trace_doc["stop_reason"] = "no_referable_objects_or_l1_candidates"
            trace_doc["stop_details"] = {
                "requested_image_name": image_name,
                "referable_count": len(referable_ids),
                "label_counts": label_counts,
                "referability_source": referability_source,
            }
            return trace_doc

        stage_started = time.perf_counter()
        ray_caster = None
        if needs_mesh_resources:
            mesh_path = scene_dir / f"{scene_id}_vh_clean.ply"
            if not mesh_path.exists():
                mesh_path = scene_dir / f"{scene_id}_vh_clean_2.ply"
            if mesh_path.exists() and RayCaster is not None:
                ray_caster = RayCaster.from_ply(str(mesh_path), axis_alignment=axis_align)
            else:
                raise RuntimeError(
                    f"{occlusion_backend} backend requested for {scene_id}, but mesh geometry or RayCaster is unavailable",
                )
        instance_mesh_data = None
        if needs_mesh_resources:
            instance_mesh_data = load_instance_mesh_data(
                scene_dir,
                instance_ids=[int(obj["id"]) for obj in scene["objects"]],
                n_surface_samples=512,
                preloaded_geometry=preloaded_geometry,
            )
        depth_image = None
        depth_path = scene_dir / "depth" / f"{frame_stem}.png"
        if use_occlusion and depth_intrinsics is not None and depth_path.exists():
            try:
                depth_image = load_depth_image(depth_path)
            except Exception as exc:
                logger.warning("Depth load failed for %s/%s: %s", scene_id, image_name, exc)

        raw_questions = generate_all_questions(
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
            trace_recorder=trace_recorder,
            trace_id_prefix=f"{scene_id}_{frame_stem}",
            trace_detail=trace_detail,
        )
        for question in raw_questions:
            question["scene_id"] = scene_id
            question["image_name"] = image_name
        _ensure_trace_question_ids(
            raw_questions,
            trace_id_prefix=f"{scene_id}_{frame_stem}",
        )
        trace_doc["raw_questions"] = _sanitize_questions(raw_questions)
        trace_doc["frame_context"]["generated_questions"] = _sanitize_questions(raw_questions)
        _record_stage(
            trace_doc,
            stage="stage_4_to_6_generation",
            status="completed",
            started_at=stage_started,
            details={
                "raw_question_count": len(raw_questions),
                "by_type": dict(sorted(Counter(str(question.get("type", "?")) for question in raw_questions).items())),
                "generator_counts": _summarize_generators(trace_events),
            },
        )

        stage_started = time.perf_counter()
        final_questions = quality_filter(raw_questions, trace_recorder=trace_recorder)
        trace_doc["quality_control"] = {
            "mode": "quality_filter_only",
            "applied_steps": ["quality_filter"],
            "skipped_steps": [
                "cap_l1_occlusion_not_visible_ratio",
                "balance_answer_values",
                "balance_answer_distribution",
            ],
            "input_count": len(raw_questions),
            "output_count": len(final_questions),
        }
        trace_doc["final_questions"] = _sanitize_questions(final_questions)
        trace_doc["frame_context"]["final_questions"] = _sanitize_questions(final_questions)
        trace_doc["frame_context"]["final_question_count"] = len(final_questions)
        _record_stage(
            trace_doc,
            stage="stage_7_quality_control",
            status="completed",
            started_at=stage_started,
            details={
                "mode": "quality_filter_only",
                "input_count": len(raw_questions),
                "output_count": len(final_questions),
            },
        )
        trace_doc["status"] = "completed"
    except Exception as exc:
        logger.exception("Single-frame trace failed for %s/%s", scene_id, image_name)
        trace_doc["status"] = "failed"
        trace_doc["stop_reason"] = str(exc)
        trace_doc["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
    finally:
        for question in raw_questions:
            if question.get("trace_question_id"):
                question_snapshots[str(question["trace_question_id"])] = _sanitize_question(question)
        for question in final_questions:
            if question.get("trace_question_id"):
                question_snapshots[str(question["trace_question_id"])] = _sanitize_question(question)
        trace_doc["trace_events"] = trace_events
        trace_doc["question_lifecycle"] = _build_question_lifecycle(
            question_snapshots=question_snapshots,
            question_events=question_events,
            raw_questions=raw_questions,
            final_questions=final_questions,
        )
        generator_audits = _build_generator_audits(trace_events)
        object_pool_event = next(
            (
                event for event in trace_events
                if str(event.get("event")) == "object_pool_snapshot"
            ),
            {},
        )
        quality_filter_events = [
            event for event in trace_events
            if str(event.get("event")) == "question_removed"
            and str(event.get("stage")) == "quality_filter"
        ]
        referability_audit = _build_referability_audit(
            referability_entry,
            referability_source=referability_source,
            payload_mode=trace_vlm_payload,
        )
        reason_index = _build_reason_index(
            trace_doc=trace_doc,
            trace_events=trace_events,
            question_lifecycle=trace_doc["question_lifecycle"],
            generator_audits=generator_audits,
        )
        audit_docs: dict[str, Any] = {
            "frame_gate": {
                "status": trace_doc.get("status"),
                "stop_reason": trace_doc.get("stop_reason"),
                "stop_details": trace_doc.get("stop_details", {}),
                "stage_summaries": trace_doc.get("stage_summaries", []),
                "input": trace_doc.get("input", {}),
                "frame_context": trace_doc.get("frame_context", {}),
            },
            "object_pool": object_pool_event,
            "referability": referability_audit,
            "question_lifecycle": trace_doc["question_lifecycle"],
            "quality_filter": {
                "quality_control": trace_doc.get("quality_control", {}),
                "removed_questions": quality_filter_events,
            },
            "reason_index": reason_index,
        }
        for generator, audit in generator_audits.items():
            audit_docs[f"generator_{generator}"] = audit

        audit_paths: dict[str, str] = {}
        for audit_name, payload in audit_docs.items():
            if audit_name.startswith("generator_"):
                generator_name = audit_name[len("generator_"):]
                file_name = f"generator_{_safe_audit_name(generator_name)}.json"
                artifact_key = f"generator:{generator_name}"
            else:
                file_name = f"{audit_name}.json"
                artifact_key = audit_name
            audit_path = audits_dir / file_name
            audit_path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            audit_paths[artifact_key] = str(audit_path)
        trace_doc["artifacts"]["audits"] = dict(sorted(audit_paths.items()))
        final_payload = {
            "scene_id": scene_id,
            "image_name": image_name,
            "question_count": len(final_questions),
            "quality_control_mode": "quality_filter_only",
            "questions": _sanitize_questions(final_questions),
        }
        final_questions_path.write_text(
            json.dumps(final_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        trace_json_path.write_text(
            json.dumps(trace_doc, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        trace_html_path.write_text(
            build_single_frame_trace_html(trace_doc, audit_docs=audit_docs),
            encoding="utf-8",
        )

    return trace_doc


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full pipeline for one specified ScanNet frame")
    parser.add_argument(
        "--data_root",
        type=str,
        default=os.getenv("SCANNET_PATH", "/home/lihongxing/datasets/ScanNet/data/scans"),
        help="Root directory containing ScanNet scene folders",
    )
    parser.add_argument("--scene_id", type=str, required=True)
    parser.add_argument("--image_name", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument(
        "--referability_cache",
        type=str,
        default=None,
        help="Optional referability cache; if the frame is missing, VLM fallback is used",
    )
    parser.add_argument("--label_map", type=str, default=None)
    parser.add_argument("--no_occlusion", action="store_true")
    parser.add_argument(
        "--occlusion_backend",
        type=str,
        choices=("depth", "mesh_ray"),
        default="mesh_ray",
    )
    parser.add_argument("--vlm_url", type=str, default=DEFAULT_VLM_URL)
    parser.add_argument("--vlm_model", type=str, default=None)
    parser.add_argument("--label_batch_size", type=int, default=LABEL_BATCH_SIZE)
    parser.add_argument(
        "--trace_detail",
        type=str,
        choices=("light", "medium", "full"),
        default="full",
    )
    parser.add_argument(
        "--trace_vlm_payload",
        type=str,
        choices=("none", "summary", "full"),
        default="summary",
    )
    args = parser.parse_args()

    if args.label_map:
        load_scannet_label_map(args.label_map)

    referability_cache_path = Path(args.referability_cache) if args.referability_cache else None
    referability_cache = (
        _load_referability_cache(referability_cache_path)
        if referability_cache_path is not None else None
    )
    trace_doc = run_single_frame_trace(
        data_root=Path(args.data_root),
        scene_id=args.scene_id,
        image_name=args.image_name,
        output_dir=Path(args.output_dir),
        referability_cache=referability_cache,
        referability_cache_path=referability_cache_path,
        use_occlusion=not args.no_occlusion,
        occlusion_backend=args.occlusion_backend,
        vlm_url=args.vlm_url,
        vlm_model=args.vlm_model,
        label_batch_size=args.label_batch_size,
        trace_detail=args.trace_detail,
        trace_vlm_payload=args.trace_vlm_payload,
    )
    artifacts = trace_doc.get("artifacts", {})
    print(
        json.dumps(
            {
                "status": trace_doc.get("status"),
                "stop_reason": trace_doc.get("stop_reason"),
                "question_count": len(trace_doc.get("final_questions", [])),
                "trace_json_path": artifacts.get("trace_json_path"),
                "trace_html_path": artifacts.get("trace_html_path"),
                "final_questions_path": artifacts.get("final_questions_path"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
