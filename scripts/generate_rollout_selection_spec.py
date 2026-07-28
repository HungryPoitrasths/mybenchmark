#!/usr/bin/env python3
"""Automatically select leakage-safe single-frame future-rollout inputs."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
import shutil
import sys
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_sampled_type_vlm_eval import _sha256_file, load_fixed_questions  # noqa: E402
from scripts.validate_rollout_manifest import L2_ROLLOUT_TYPES  # noqa: E402
from src.datasets import make_data_source  # noqa: E402
from src.frame_selector import (  # noqa: E402
    FRAME_STRIDE_SCANNET,
    FRAME_STRIDE_SCANNETPP,
    _passes_absolute_image_quality_gate,
    _project_object_roi,
    _read_image_quality_metrics,
)
from src.qa_generator import (  # noqa: E402
    _bbox_fully_in_frame,
    _compute_mesh_ray_l1_occlusion_metrics_for_moved_target,
    _compute_mesh_ray_l1_occlusion_metrics_for_static_target,
    quick_moved_bbox_projection,
)
from src.scene_parser import load_instance_mesh_data  # noqa: E402
from src.support_graph import (  # noqa: E402
    enrich_scene_with_attachment,
    get_attachment_chain,
    get_scene_attachment_graph,
)
from src.utils.coordinate_transform import default_edge_margin_px  # noqa: E402
from src.utils.depth_occlusion import (  # noqa: E402
    FULLY_VISIBLE_RATIO_MIN,
    compute_depth_occlusion_metrics,
    min_projected_area_px,
)
from src.utils.ray_casting import RayCaster  # noqa: E402
from src.virtual_ops import apply_movement, apply_orbit_rotation  # noqa: E402


SELECTION_SCHEMA_VERSION = "predictive-spatial-selection-v1"
SELECTION_AUDIT_SCHEMA_VERSION = "predictive-spatial-selection-audit-v1"


@dataclass(frozen=True)
class SelectionPaths:
    spec: Path
    audit: Path


@dataclass
class SceneContext:
    dataset: str
    scene_id: str
    scene_dir: Path
    data_source: Any
    objects: list[dict[str, Any]]
    objects_by_id: dict[int, dict[str, Any]]
    attachment_graph: dict[int, list[int]]
    intrinsics: Any
    poses: dict[str, Any]
    ray_caster: Any
    instance_mesh_data: Any


@dataclass
class QuestionMotion:
    moved_ids: tuple[int, ...]
    source_required_ids: tuple[int, ...]
    moved_objects: list[dict[str, Any]]
    moved_objects_by_id: dict[int, dict[str, Any]]


@dataclass
class FrameCandidate:
    question: dict[str, Any]
    source_index: int
    context: SceneContext
    motion: QuestionMotion
    image_name: str
    image_path: Path
    pose: Any
    score_key: tuple[Any, ...]
    metrics: dict[str, Any]
    context_paths: list[Path]


def _candidate_is_better(candidate: FrameCandidate, best: FrameCandidate | None) -> bool:
    if best is None or candidate.score_key > best.score_key:
        return True
    return bool(
        candidate.score_key == best.score_key
        and _natural_frame_key(candidate.image_name) < _natural_frame_key(best.image_name)
    )


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def _infer_dataset(question: dict[str, Any]) -> str:
    explicit = str(question.get("_dataset") or question.get("dataset") or "").strip().lower()
    if explicit in {"scannet", "scannetpp"}:
        return explicit
    source = str(question.get("_source_benchmark") or "").lower()
    if "scannetpp" in source:
        return "scannetpp"
    scene_id = str(question.get("scene_id") or "")
    return "scannet" if scene_id.startswith("scene") else "scannetpp"


def _resolve_scene_dir(root: Path | None, dataset: str, scene_id: str) -> Path:
    if root is None:
        raise FileNotFoundError(f"{dataset} root is required for {scene_id}")
    candidates = [root] if root.name == scene_id else []
    candidates.extend((root / scene_id, root / "scans" / scene_id))
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()
    raise FileNotFoundError(f"cannot find {dataset} scene {scene_id} below {root}")


def _coerce_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be an integer")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be an integer") from exc


def _coerce_delta(value: Any) -> np.ndarray:
    delta = np.asarray(value, dtype=np.float64)
    if delta.shape != (3,) or not np.all(np.isfinite(delta)):
        raise ValueError("delta must be a finite 3-vector")
    return delta


def _moving_group_ids(
    question: dict[str, Any], attachment_graph: dict[int, list[int]]
) -> tuple[int, ...]:
    moved_id = _coerce_int(question.get("moved_obj_id"), field="moved_obj_id")
    return tuple(dict.fromkeys([moved_id, *get_attachment_chain(moved_id, attachment_graph)]))


def _apply_question_motion(
    question: dict[str, Any],
    objects: list[dict[str, Any]],
    attachment_graph: dict[int, list[int]],
) -> QuestionMotion:
    moved_ids = _moving_group_ids(question, attachment_graph)
    query_id = _coerce_int(question.get("query_obj_id"), field="query_obj_id")
    source_required_ids = tuple(dict.fromkeys((*moved_ids, query_id)))
    qtype = str(question.get("type") or "")
    moved_id = moved_ids[0]
    if qtype == "object_rotate_object_centric":
        pivot_id = _coerce_int(question.get("obj_face_id"), field="obj_face_id")
        angle = float(question.get("rotation_angle"))
        direction = str(question.get("rotation_direction") or "").strip().lower()
        if not math.isfinite(angle) or angle <= 0.0:
            raise ValueError("rotation_angle must be positive")
        if direction not in {"clockwise", "counterclockwise"}:
            raise ValueError("rotation_direction must be clockwise or counterclockwise")
        signed_angle = -angle if direction == "clockwise" else angle
        moved_objects = apply_orbit_rotation(
            objects,
            attachment_graph,
            moved_id,
            pivot_id,
            signed_angle,
        )
    else:
        moved_objects = apply_movement(
            objects,
            attachment_graph,
            moved_id,
            _coerce_delta(question.get("delta")),
        )
    moved_map = {int(obj["id"]): obj for obj in moved_objects}
    return QuestionMotion(
        moved_ids=moved_ids,
        source_required_ids=source_required_ids,
        moved_objects=moved_objects,
        moved_objects_by_id=moved_map,
    )


def _natural_frame_key(image_name: str) -> tuple[tuple[int, Any], ...]:
    parts = re.split(r"(\d+)", image_name)
    return tuple(
        (0, int(part)) if part.isdigit() else (1, part.lower())
        for part in parts
    )


def _candidate_frame_names(
    question: dict[str, Any], context: SceneContext, frame_stride: int
) -> list[str]:
    ordered = sorted(context.poses, key=_natural_frame_key)
    names = [name for index, name in enumerate(ordered) if index % max(frame_stride, 1) == 0]
    original = str(question.get("image_name") or "").strip()
    if original in context.poses and original not in names:
        names.append(original)
    return sorted(set(names), key=_natural_frame_key)


def _collect_question_context_names(question: dict[str, Any]) -> list[str]:
    names = [str(question.get("image_name") or "").strip()]
    raw_aux = question.get("auxiliary_image_names")
    if isinstance(raw_aux, list):
        names.extend(str(value).strip() for value in raw_aux if str(value).strip())
    if len(names) == 1:
        fallback = str(
            question.get("aux_image_name") or question.get("image_name_2") or ""
        ).strip()
        if fallback:
            names.append(fallback)
    final_name = str(question.get("reasoning_frame_2") or "").strip()
    if final_name and final_name not in names:
        names.append(final_name)
    return list(dict.fromkeys(name for name in names if name))


def _resolve_context_paths(question: dict[str, Any], context: SceneContext) -> list[Path]:
    paths: list[Path] = []
    for image_name in _collect_question_context_names(question):
        path = context.data_source.image_path(image_name).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"answer context image not found: {path}")
        paths.append(path)
    return paths


def _metrics_visible_ratio(metrics: Any) -> float:
    valid = int(getattr(metrics, "valid_in_frame_count", 0) or 0)
    visible = int(getattr(metrics, "visible_in_frame_count", 0) or 0)
    if valid > 0:
        return float(visible / valid)
    return max(0.0, 1.0 - float(getattr(metrics, "occlusion_ratio_in_frame", 1.0)))


def _static_visibility(
    obj: dict[str, Any],
    *,
    pose: Any,
    context: SceneContext,
    depth_frame: Any | None,
) -> tuple[bool, float, str, dict[str, Any]]:
    if depth_frame is not None:
        depth_metrics = compute_depth_occlusion_metrics(
            bbox_min=np.asarray(obj["bbox_min"], dtype=np.float64),
            bbox_max=np.asarray(obj["bbox_max"], dtype=np.float64),
            camera_pose=pose,
            intrinsics=depth_frame.intrinsics,
            depth_image=depth_frame.image_m,
        )
        valid = int(depth_metrics["valid_in_frame_count"])
        ratio = float(depth_metrics["visible_ratio_in_frame"])
        if valid > 0:
            return (
                ratio > FULLY_VISIBLE_RATIO_MIN,
                ratio,
                "depth",
                {key: value for key, value in depth_metrics.items()},
            )

    mesh_metrics = _compute_mesh_ray_l1_occlusion_metrics_for_static_target(
        obj=obj,
        camera_pose=pose,
        color_intrinsics=context.intrinsics,
        ray_caster=context.ray_caster,
        instance_mesh_data=context.instance_mesh_data,
    )
    ratio = _metrics_visible_ratio(mesh_metrics)
    valid = int(getattr(mesh_metrics, "valid_in_frame_count", 0) or 0)
    return (
        valid > 0 and ratio > FULLY_VISIBLE_RATIO_MIN,
        ratio,
        "mesh_ray",
        {
            "valid_in_frame_count": valid,
            "visible_in_frame_count": int(
                getattr(mesh_metrics, "visible_in_frame_count", 0) or 0
            ),
            "projected_area": float(getattr(mesh_metrics, "projected_area", 0.0)),
            "in_frame_ratio": float(getattr(mesh_metrics, "in_frame_ratio", 0.0)),
        },
    )


def _future_visibility(
    obj_id: int,
    *,
    pose: Any,
    context: SceneContext,
    motion: QuestionMotion,
) -> tuple[bool, float, dict[str, Any]]:
    metrics = _compute_mesh_ray_l1_occlusion_metrics_for_moved_target(
        target_obj_id=obj_id,
        original_objects=context.objects,
        moved_objects=motion.moved_objects,
        moved_ids=set(motion.moved_ids),
        camera_pose=pose,
        color_intrinsics=context.intrinsics,
        ray_caster=context.ray_caster,
        instance_mesh_data=context.instance_mesh_data,
    )
    ratio = _metrics_visible_ratio(metrics)
    valid = int(getattr(metrics, "valid_in_frame_count", 0) or 0)
    return (
        valid > 0 and ratio > FULLY_VISIBLE_RATIO_MIN,
        ratio,
        {
            "valid_in_frame_count": valid,
            "visible_in_frame_count": int(
                getattr(metrics, "visible_in_frame_count", 0) or 0
            ),
            "projected_area": float(getattr(metrics, "projected_area", 0.0)),
            "in_frame_ratio": float(getattr(metrics, "in_frame_ratio", 0.0)),
        },
    )


def _projection_gate(
    obj: dict[str, Any], pose: Any, context: SceneContext
) -> tuple[bool, dict[str, float]]:
    area, in_frame_ratio = quick_moved_bbox_projection(obj, pose, context.intrinsics)
    roi = _project_object_roi(obj, pose, context.intrinsics)
    edge_margin = float(roi.get("edge_margin_px", 0.0) or 0.0)
    minimum_area = min_projected_area_px(
        context.intrinsics.width,
        context.intrinsics.height,
    )
    minimum_margin = float(default_edge_margin_px(context.intrinsics))
    passed = bool(
        _bbox_fully_in_frame(obj, pose, context.intrinsics)
        and in_frame_ratio >= 1.0
        and area >= minimum_area
        and edge_margin >= minimum_margin
    )
    return passed, {
        "projected_area_px": float(area),
        "projected_area_ratio": float(
            area / max(context.intrinsics.width * context.intrinsics.height, 1)
        ),
        "in_frame_ratio": float(in_frame_ratio),
        "edge_margin_px": edge_margin,
        "edge_margin_ratio": float(
            edge_margin / max(min(context.intrinsics.width, context.intrinsics.height), 1)
        ),
        "minimum_projected_area_px": float(minimum_area),
        "minimum_edge_margin_px": minimum_margin,
    }


def _evaluate_frame(
    *,
    question: dict[str, Any],
    source_index: int,
    context: SceneContext,
    motion: QuestionMotion,
    image_name: str,
    context_paths: list[Path],
) -> tuple[FrameCandidate | None, str]:
    pose = context.poses[image_name]
    image_path = context.data_source.image_path(image_name).resolve()
    if not image_path.is_file():
        return None, "image_missing"
    quality = _read_image_quality_metrics(image_path)
    if not bool(quality.get("readable")):
        return None, "image_unreadable"
    laplacian = float(quality["laplacian_variance"])
    tenengrad = float(quality["tenengrad"])
    if not _passes_absolute_image_quality_gate(laplacian, tenengrad):
        return None, "image_quality_below_threshold"

    depth_frame = context.data_source.load_depth_frame(image_name)
    source_metrics: dict[str, Any] = {}
    future_metrics: dict[str, Any] = {}
    source_ratios: list[float] = []
    future_ratios: list[float] = []
    area_ratios: list[float] = []
    margin_ratios: list[float] = []

    for obj_id in motion.source_required_ids:
        obj = context.objects_by_id.get(obj_id)
        if obj is None:
            return None, "source_object_missing"
        projection_ok, projection = _projection_gate(obj, pose, context)
        if not projection_ok:
            return None, "source_projection_gate_failed"
        visible, ratio, backend, visibility = _static_visibility(
            obj,
            pose=pose,
            context=context,
            depth_frame=depth_frame,
        )
        if not visible:
            return None, "source_not_fully_visible"
        source_ratios.append(ratio)
        area_ratios.append(projection["projected_area_ratio"])
        margin_ratios.append(projection["edge_margin_ratio"])
        source_metrics[str(obj_id)] = {
            "projection": projection,
            "visibility_backend": backend,
            "visible_ratio": ratio,
            "visibility": visibility,
        }

    for obj_id in motion.moved_ids:
        moved_obj = motion.moved_objects_by_id.get(obj_id)
        if moved_obj is None:
            return None, "future_object_missing"
        projection_ok, projection = _projection_gate(moved_obj, pose, context)
        if not projection_ok:
            return None, "future_projection_gate_failed"
        visible, ratio, visibility = _future_visibility(
            obj_id,
            pose=pose,
            context=context,
            motion=motion,
        )
        if not visible:
            return None, "future_not_fully_visible"
        future_ratios.append(ratio)
        area_ratios.append(projection["projected_area_ratio"])
        margin_ratios.append(projection["edge_margin_ratio"])
        future_metrics[str(obj_id)] = {
            "projection": projection,
            "visibility_backend": "counterfactual_mesh_ray",
            "visible_ratio": ratio,
            "visibility": visibility,
        }

    original_bonus = int(image_name == str(question.get("image_name") or ""))
    score_key: tuple[Any, ...] = (
        min([*source_ratios, *future_ratios]),
        min(area_ratios),
        min(margin_ratios),
        laplacian,
        tenengrad,
        original_bonus,
    )
    metrics = {
        "source": source_metrics,
        "future": future_metrics,
        "image_quality": {
            "laplacian_variance": laplacian,
            "tenengrad": tenengrad,
        },
        "score": list(score_key),
    }
    return (
        FrameCandidate(
            question=question,
            source_index=source_index,
            context=context,
            motion=motion,
            image_name=image_name,
            image_path=image_path,
            pose=pose,
            score_key=score_key,
            metrics=metrics,
            context_paths=context_paths,
        ),
        "selected_candidate",
    )


def _best_frame_for_question(
    *,
    question: dict[str, Any],
    source_index: int,
    context: SceneContext,
    frame_stride: int,
) -> tuple[FrameCandidate | None, dict[str, int], str | None]:
    try:
        motion = _apply_question_motion(question, context.objects, context.attachment_graph)
        for obj_id in set((*motion.source_required_ids, *motion.moved_ids)):
            if obj_id not in context.objects_by_id:
                raise ValueError(f"object {obj_id} is absent from scene")
        if bool(question.get("has_attachment_chain")) and len(motion.moved_ids) <= 1:
            raise ValueError("question requires an attachment chain but none was found")
        context_paths = _resolve_context_paths(question, context)
    except Exception as exc:
        return None, {}, str(exc)

    best: FrameCandidate | None = None
    reasons: Counter[str] = Counter()
    for image_name in _candidate_frame_names(question, context, frame_stride):
        candidate, reason = _evaluate_frame(
            question=question,
            source_index=source_index,
            context=context,
            motion=motion,
            image_name=image_name,
            context_paths=context_paths,
        )
        reasons[reason] += 1
        if candidate is None:
            continue
        if _candidate_is_better(candidate, best):
            best = candidate
    return best, dict(sorted(reasons.items())), None


def _load_scene_context(
    *,
    dataset: str,
    scene_id: str,
    scannet_root: Path | None,
    scannetpp_root: Path | None,
    scannetpp_frame_root: Path | None,
    scannetpp_sensor: str,
    required_instance_ids: set[int],
) -> SceneContext:
    root = scannetpp_root if dataset == "scannetpp" else scannet_root
    scene_dir = _resolve_scene_dir(root, dataset, scene_id)
    data_source = make_data_source(
        dataset,
        scene_dir,
        sensor=scannetpp_sensor,
        frame_root=scannetpp_frame_root,
    )
    scene = data_source.load_scene()
    try:
        attachment_graph = get_scene_attachment_graph(scene)
    except KeyError:
        enrich_scene_with_attachment(scene)
        attachment_graph = get_scene_attachment_graph(scene)
    objects = [obj for obj in scene.get("objects", []) if isinstance(obj, dict)]
    objects_by_id = {int(obj["id"]): obj for obj in objects}
    for moved_id in list(required_instance_ids):
        required_instance_ids.update(get_attachment_chain(moved_id, attachment_graph))
    axis_alignment = data_source.load_axis_alignment()
    mesh_path = data_source.mesh_path()
    ray_caster = RayCaster.from_ply(str(mesh_path), axis_alignment=axis_alignment)
    instance_mesh_data = load_instance_mesh_data(
        scene_dir,
        instance_ids=sorted(required_instance_ids),
        dataset=dataset,
    )
    return SceneContext(
        dataset=dataset,
        scene_id=scene_id,
        scene_dir=scene_dir,
        data_source=data_source,
        objects=objects,
        objects_by_id=objects_by_id,
        attachment_graph=attachment_graph,
        intrinsics=data_source.load_intrinsics(),
        poses=data_source.load_poses(),
        ray_caster=ray_caster,
        instance_mesh_data=instance_mesh_data,
    )


def _copy_media(source: Path, target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() != target.resolve():
        shutil.copy2(source, target)
    return target.resolve()


def _question_required_instance_ids(question: dict[str, Any]) -> set[int]:
    required = {
        _coerce_int(question.get("moved_obj_id"), field="moved_obj_id"),
        _coerce_int(question.get("query_obj_id"), field="query_obj_id"),
    }
    if str(question.get("type") or "") == "object_rotate_object_centric":
        required.add(_coerce_int(question.get("obj_face_id"), field="obj_face_id"))
    return required


def _materialize_candidate(candidate: FrameCandidate, output_dir: Path) -> dict[str, Any]:
    uid = str(candidate.question["question_uid"])
    condition_suffix = candidate.image_path.suffix.lower() or ".jpg"
    condition_path = _copy_media(
        candidate.image_path,
        output_dir / "media" / "conditions" / uid / f"motion{condition_suffix}",
    )
    copied_context: list[Path] = []
    for index, source in enumerate(candidate.context_paths):
        if source.resolve() == candidate.image_path.resolve():
            continue
        suffix = source.suffix.lower() or ".jpg"
        copied_context.append(
            _copy_media(
                source,
                output_dir / "media" / "context" / uid / f"frame_{index:02d}{suffix}",
            )
        )
    answer_context_media: list[dict[str, Any]] = []
    for index, path in enumerate(copied_context):
        role = (
            "query_reference_view"
            if index == len(copied_context) - 1
            else "destination_to_query_bridge"
        )
        answer_context_media.append({"path": str(path), "role": role})

    moving_group = [
        {
            "obj_id": obj_id,
            "label": str(candidate.context.objects_by_id[obj_id].get("label") or "").strip(),
        }
        for obj_id in candidate.motion.moved_ids
    ]
    if any(not item["label"] for item in moving_group):
        raise ValueError(f"{uid}: moving group contains an empty label")
    entry: dict[str, Any] = {
        "question_uid": uid,
        "source_index": candidate.source_index,
        "motion_frame_path": str(condition_path),
        "camera_rotation_world_to_camera": np.asarray(
            candidate.pose.rotation, dtype=np.float64
        ).tolist(),
        "moving_group": moving_group,
        "picture_eligible": True,
        "video_eligible": True,
        "answer_context_media": answer_context_media,
    }
    if str(candidate.question.get("type") or "") == "object_rotate_object_centric":
        entry["orbit_anchor_label"] = str(
            candidate.question.get("obj_face_label") or ""
        ).strip()
    candidate.metrics["materialized"] = {
        "motion_frame_path": str(condition_path),
        "motion_frame_sha256": _sha256_file(condition_path),
        "source_motion_frame_path": str(candidate.image_path),
        "answer_context_paths": [str(path) for path in copied_context],
    }
    return entry


def generate_selection_spec(
    *,
    benchmark_path: Path,
    output_dir: Path,
    scannet_root: Path | None,
    scannetpp_root: Path | None,
    scannetpp_frame_root: Path | None,
    scannetpp_sensor: str = "iphone",
    expected_per_type: int = 50,
    frame_stride_scannet: int = FRAME_STRIDE_SCANNET,
    frame_stride_scannetpp: int = FRAME_STRIDE_SCANNETPP,
) -> SelectionPaths:
    questions, _, _ = load_fixed_questions(benchmark_path)
    indexed_questions = [
        (index, question)
        for index, question in enumerate(questions)
        if str(question.get("type") or "") in L2_ROLLOUT_TYPES
    ]
    grouped: dict[tuple[str, str], list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for source_index, question in indexed_questions:
        dataset = _infer_dataset(question)
        scene_id = str(question.get("scene_id") or "").strip()
        grouped[(dataset, scene_id)].append((source_index, question))

    eligible: dict[str, list[FrameCandidate]] = defaultdict(list)
    rejected: list[dict[str, Any]] = []
    scene_errors: dict[str, str] = {}
    for (dataset, scene_id), raw_scene_questions in sorted(grouped.items()):
        required_ids: set[int] = set()
        scene_questions: list[tuple[int, dict[str, Any]]] = []
        for source_index, question in raw_scene_questions:
            try:
                if not scene_id:
                    raise ValueError("scene_id must be non-empty")
                required_ids.update(_question_required_instance_ids(question))
            except ValueError as exc:
                rejected.append(
                    {
                        "question_uid": question.get("question_uid"),
                        "source_index": source_index,
                        "question_type": question.get("type"),
                        "scene_id": scene_id,
                        "reason": "malformed_question",
                        "detail": str(exc),
                    }
                )
                continue
            scene_questions.append((source_index, question))
        if not scene_questions:
            continue
        try:
            context = _load_scene_context(
                dataset=dataset,
                scene_id=scene_id,
                scannet_root=scannet_root,
                scannetpp_root=scannetpp_root,
                scannetpp_frame_root=scannetpp_frame_root,
                scannetpp_sensor=scannetpp_sensor,
                required_instance_ids=required_ids,
            )
        except Exception as exc:
            scene_key = f"{dataset}:{scene_id}"
            scene_errors[scene_key] = str(exc)
            for source_index, question in scene_questions:
                rejected.append(
                    {
                        "question_uid": question.get("question_uid"),
                        "source_index": source_index,
                        "question_type": question.get("type"),
                        "scene_id": scene_id,
                        "reason": "scene_resources_unavailable",
                        "detail": str(exc),
                    }
                )
            continue

        frame_stride = (
            frame_stride_scannetpp if dataset == "scannetpp" else frame_stride_scannet
        )
        for source_index, question in scene_questions:
            best, reason_counts, detail = _best_frame_for_question(
                question=question,
                source_index=source_index,
                context=context,
                frame_stride=frame_stride,
            )
            if best is None:
                rejected.append(
                    {
                        "question_uid": question.get("question_uid"),
                        "source_index": source_index,
                        "question_type": question.get("type"),
                        "scene_id": scene_id,
                        "reason": "no_strict_single_frame",
                        "detail": detail,
                        "candidate_reasons": reason_counts,
                    }
                )
                continue
            eligible[str(question["type"])].append(best)

    selected: list[FrameCandidate] = []
    counts: dict[str, dict[str, int]] = {}
    for qtype in L2_ROLLOUT_TYPES:
        candidates = eligible.get(qtype, [])
        candidates.sort(key=lambda item: str(item.question["question_uid"]))
        candidates.sort(key=lambda item: item.score_key, reverse=True)
        chosen = candidates[:expected_per_type]
        selected.extend(chosen)
        counts[qtype] = {
            "eligible": len(candidates),
            "selected": len(chosen),
            "limit": expected_per_type,
            "shortfall": max(0, expected_per_type - len(chosen)),
        }
        for candidate in candidates[expected_per_type:]:
            rejected.append(
                {
                    "question_uid": candidate.question.get("question_uid"),
                    "source_index": candidate.source_index,
                    "question_type": qtype,
                    "scene_id": candidate.question.get("scene_id"),
                    "reason": "type_quota_exceeded",
                }
            )

    selected.sort(key=lambda item: L2_ROLLOUT_TYPES.index(str(item.question["type"])))
    entries = [_materialize_candidate(candidate, output_dir) for candidate in selected]
    private_dir = output_dir / "private_jobs"
    spec_path = private_dir / "selection_spec.json"
    audit_path = private_dir / "selection_audit.json"
    spec_payload = {
        "schema_version": SELECTION_SCHEMA_VERSION,
        "metadata": {
            "benchmark_path": str(benchmark_path.resolve()),
            "benchmark_sha256": _sha256_file(benchmark_path),
            "max_per_type": expected_per_type,
            "selection_mode": "strict_single_frame",
        },
        "entries": entries,
    }
    rejection_counts = Counter(str(item["reason"]) for item in rejected)
    audit_payload = {
        "schema_version": SELECTION_AUDIT_SCHEMA_VERSION,
        "benchmark_path": str(benchmark_path.resolve()),
        "benchmark_sha256": _sha256_file(benchmark_path),
        "configuration": {
            "max_per_type": expected_per_type,
            "frame_stride_scannet": frame_stride_scannet,
            "frame_stride_scannetpp": frame_stride_scannetpp,
            "fully_visible_ratio_min_exclusive": FULLY_VISIBLE_RATIO_MIN,
            "source_visibility": "registered_depth_then_mesh_ray",
            "future_visibility": "counterfactual_mesh_ray",
            "require_full_bbox": True,
            "require_default_edge_margin": True,
        },
        "counts_by_type": counts,
        "selected": [
            {
                "question_uid": candidate.question.get("question_uid"),
                "source_index": candidate.source_index,
                "question_type": candidate.question.get("type"),
                "scene_id": candidate.question.get("scene_id"),
                "image_name": candidate.image_name,
                "metrics": candidate.metrics,
            }
            for candidate in selected
        ],
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "rejected": rejected,
        "scene_errors": scene_errors,
    }
    _atomic_write_json(spec_path, spec_payload)
    _atomic_write_json(audit_path, audit_payload)
    return SelectionPaths(spec=spec_path, audit=audit_path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark_file", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--scannet_root", type=Path, default=None)
    parser.add_argument("--scannetpp_root", type=Path, default=None)
    parser.add_argument("--scannetpp_frame_root", type=Path, default=None)
    parser.add_argument("--scannetpp_sensor", choices=("iphone", "dslr"), default="iphone")
    parser.add_argument(
        "--expected_per_type",
        type=int,
        default=50,
        help="maximum number of selected questions per supported type",
    )
    parser.add_argument("--frame_stride_scannet", type=int, default=FRAME_STRIDE_SCANNET)
    parser.add_argument("--frame_stride_scannetpp", type=int, default=FRAME_STRIDE_SCANNETPP)
    args = parser.parse_args(argv)
    if not args.benchmark_file.is_file():
        parser.error(f"--benchmark_file not found: {args.benchmark_file}")
    if args.expected_per_type <= 0:
        parser.error("--expected_per_type must be positive")
    if args.frame_stride_scannet <= 0 or args.frame_stride_scannetpp <= 0:
        parser.error("frame strides must be positive")
    for field in ("scannet_root", "scannetpp_root", "scannetpp_frame_root"):
        path = getattr(args, field)
        if path is not None and not path.is_dir():
            parser.error(f"--{field} is not a directory: {path}")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    paths = generate_selection_spec(
        benchmark_path=args.benchmark_file,
        output_dir=args.output_dir,
        scannet_root=args.scannet_root,
        scannetpp_root=args.scannetpp_root,
        scannetpp_frame_root=args.scannetpp_frame_root,
        scannetpp_sensor=args.scannetpp_sensor,
        expected_per_type=args.expected_per_type,
        frame_stride_scannet=args.frame_stride_scannet,
        frame_stride_scannetpp=args.frame_stride_scannetpp,
    )
    print(f"selection_spec : {paths.spec}")
    print(f"selection_audit: {paths.audit}")


if __name__ == "__main__":
    main(sys.argv[1:])
