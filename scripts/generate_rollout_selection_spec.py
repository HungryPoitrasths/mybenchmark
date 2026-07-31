#!/usr/bin/env python3
"""Select source-to-destination routes for future rollout generation."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import time
from typing import Any

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_sampled_type_vlm_eval import _sha256_file, load_fixed_questions  # noqa: E402
from scripts.validate_rollout_manifest import L2_ROLLOUT_TYPES  # noqa: E402
from src.datasets import make_data_source  # noqa: E402
from src.frame_selector import (  # noqa: E402
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


SELECTION_SCHEMA_VERSION = "predictive-spatial-selection-v2"
SELECTION_AUDIT_SCHEMA_VERSION = "predictive-spatial-selection-audit-v2"
SELECTION_CHECKPOINT_SCHEMA_VERSION = "predictive-spatial-selection-checkpoint-v2"
SELECTION_ALGORITHM_VERSION = "source-destination-route-v4"
MESH_RAY_SURFACE_SAMPLES = 64
MESH_RAY_BBOX_SAMPLES = 0
MESH_RAY_LOCAL_RESAMPLES = 4


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
    image_quality_cache: dict[str, dict[str, Any]] = field(default_factory=dict)
    static_visibility_cache: dict[tuple[Any, ...], tuple[bool, float, str, dict[str, Any]]] = field(
        default_factory=dict
    )
    depth_frame_cache: dict[str, Any | None] = field(default_factory=dict)
    cache_stats: Counter[str] = field(default_factory=Counter)


@dataclass
class QuestionMotion:
    moved_ids: tuple[int, ...]
    source_required_ids: tuple[int, ...]
    moved_objects: list[dict[str, Any]]
    moved_objects_by_id: dict[int, dict[str, Any]]


@dataclass
class RouteCandidate:
    question: dict[str, Any]
    source_index: int
    dataset: str
    scene_id: str
    generation_image_names: tuple[str, ...]
    generation_roles: tuple[str, ...]
    score_key: tuple[float, ...]
    metrics: dict[str, Any]


@dataclass
class SelectionProgress:
    total: int
    processed: int = 0
    eligible: int = 0
    rejected: int = 0
    started_at: float = field(default_factory=time.monotonic)
    last_report_at: float = field(default_factory=time.monotonic)
    run_start_processed: int = field(init=False)

    def __post_init__(self) -> None:
        self.run_start_processed = self.processed

    def report_resume(self, restored: int, checkpoint_path: Path) -> None:
        if restored:
            print(
                f"[selection] resumed {restored}/{self.total} completed questions from {checkpoint_path}",
                file=sys.stderr,
                flush=True,
            )

    def report_scene(
        self,
        *,
        scene_index: int,
        scene_count: int,
        dataset: str,
        scene_id: str,
        pending_questions: int,
    ) -> None:
        print(
            f"[selection] scene {scene_index}/{scene_count} {dataset}:{scene_id} "
            f"pending={pending_questions}",
            file=sys.stderr,
            flush=True,
        )

    def advance(self, *, was_eligible: bool) -> None:
        self.processed += 1
        if was_eligible:
            self.eligible += 1
        else:
            self.rejected += 1
        now = time.monotonic()
        if self.processed == self.total or self.processed % 10 == 0 or now - self.last_report_at >= 10:
            elapsed = max(now - self.started_at, 0.0)
            completed = self.processed - self.run_start_processed
            rate = completed / elapsed if elapsed else 0.0
            eta = (self.total - self.processed) / rate if rate else 0.0
            percent = 100.0 * self.processed / max(self.total, 1)
            print(
                f"[selection] {self.processed}/{self.total} ({percent:.1f}%) "
                f"eligible={self.eligible} rejected={self.rejected} "
                f"elapsed={_format_duration(elapsed)} eta={_format_duration(eta)}",
                file=sys.stderr,
                flush=True,
            )
            self.last_report_at = now


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _format_duration(seconds: float) -> str:
    total_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds_part = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m"
    if minutes:
        return f"{minutes:d}m{seconds_part:02d}s"
    return f"{seconds_part:d}s"


def _infer_dataset(question: dict[str, Any]) -> str:
    explicit = str(question.get("_dataset") or question.get("dataset") or "").strip().lower()
    if explicit in {"scannet", "scannetpp"}:
        return explicit
    source = str(question.get("_source_benchmark") or "").lower()
    if "scannetpp" in source:
        return "scannetpp"
    return "scannet" if str(question.get("scene_id") or "").startswith("scene") else "scannetpp"


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
    qtype = str(question.get("type") or "")
    if qtype == "object_rotate_object_centric":
        pivot_id = _coerce_int(question.get("obj_face_id"), field="obj_face_id")
        angle = float(question.get("rotation_angle"))
        direction = str(question.get("rotation_direction") or "").strip().lower()
        if not math.isfinite(angle) or angle <= 0:
            raise ValueError("rotation_angle must be positive")
        if direction not in {"clockwise", "counterclockwise"}:
            raise ValueError("rotation_direction must be clockwise or counterclockwise")
        moved_objects = apply_orbit_rotation(
            objects,
            attachment_graph,
            moved_ids[0],
            pivot_id,
            -angle if direction == "clockwise" else angle,
        )
    else:
        moved_objects = apply_movement(
            objects,
            attachment_graph,
            moved_ids[0],
            _coerce_delta(question.get("delta")),
        )
    moved_map = {int(obj["id"]): obj for obj in moved_objects}
    return QuestionMotion(
        moved_ids=moved_ids,
        source_required_ids=tuple(dict.fromkeys((*moved_ids, query_id))),
        moved_objects=moved_objects,
        moved_objects_by_id=moved_map,
    )


def _question_route_names(question: dict[str, Any]) -> tuple[str, list[str], str]:
    source = str(question.get("image_name") or "").strip()
    destination = str(question.get("reasoning_frame_2") or "").strip()
    if not source:
        raise ValueError("image_name is required")
    if not destination:
        raise ValueError("reasoning_frame_2 is required")
    if source == destination:
        raise ValueError("source and destination images must differ")
    raw_auxiliary = question.get("auxiliary_image_names")
    auxiliary: list[str] = []
    if isinstance(raw_auxiliary, (list, tuple)):
        auxiliary.extend(str(value).strip() for value in raw_auxiliary if str(value).strip())
    if not auxiliary:
        fallback = str(question.get("aux_image_name") or question.get("image_name_2") or "").strip()
        if fallback:
            auxiliary.append(fallback)
    auxiliary = list(dict.fromkeys(name for name in auxiliary if name not in {source, destination}))
    return source, auxiliary, destination


def _pose_position(pose: Any, *, image_name: str) -> np.ndarray:
    position = np.asarray(getattr(pose, "position", None), dtype=np.float64)
    if position.shape != (3,) or not np.all(np.isfinite(position)):
        raise ValueError(f"pose.position is invalid for {image_name}")
    return position


def _select_midpoint_bridge(
    source_name: str,
    auxiliary_names: list[str] | tuple[str, ...],
    destination_name: str,
    poses: dict[str, Any],
) -> str | None:
    """Choose the route frame nearest 50% cumulative camera travel."""
    auxiliary = [name for name in auxiliary_names if name not in {source_name, destination_name}]
    if not auxiliary:
        return None
    route = [source_name, *auxiliary, destination_name]
    positions = [_pose_position(poses[name], image_name=name) for name in route]
    cumulative = [0.0]
    for previous, current in zip(positions, positions[1:]):
        cumulative.append(cumulative[-1] + float(np.linalg.norm(current - previous)))
    midpoint = cumulative[-1] / 2.0
    bridge_index = min(
        range(1, len(route) - 1),
        key=lambda index: (abs(cumulative[index] - midpoint), index),
    )
    return route[bridge_index]


def _metrics_visible_ratio(metrics: Any) -> float:
    valid = int(getattr(metrics, "valid_in_frame_count", 0) or 0)
    visible = int(getattr(metrics, "visible_in_frame_count", 0) or 0)
    if valid:
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
        metrics = compute_depth_occlusion_metrics(
            bbox_min=np.asarray(obj["bbox_min"], dtype=np.float64),
            bbox_max=np.asarray(obj["bbox_max"], dtype=np.float64),
            camera_pose=pose,
            intrinsics=depth_frame.intrinsics,
            depth_image=depth_frame.image_m,
        )
        valid = int(metrics["valid_in_frame_count"])
        ratio = float(metrics["visible_ratio_in_frame"])
        if valid:
            return ratio > FULLY_VISIBLE_RATIO_MIN, ratio, "depth", dict(metrics)
    metrics = _compute_mesh_ray_l1_occlusion_metrics_for_static_target(
        obj=obj,
        camera_pose=pose,
        color_intrinsics=context.intrinsics,
        ray_caster=context.ray_caster,
        instance_mesh_data=context.instance_mesh_data,
        max_surface_samples=MESH_RAY_SURFACE_SAMPLES,
        bbox_probe_ray_count=MESH_RAY_BBOX_SAMPLES,
        local_resample_count=MESH_RAY_LOCAL_RESAMPLES,
    )
    ratio = _metrics_visible_ratio(metrics)
    valid = int(getattr(metrics, "valid_in_frame_count", 0) or 0)
    return valid > 0 and ratio > FULLY_VISIBLE_RATIO_MIN, ratio, "mesh_ray", {
        "valid_in_frame_count": valid,
        "visible_in_frame_count": int(getattr(metrics, "visible_in_frame_count", 0) or 0),
        "projected_area": float(getattr(metrics, "projected_area", 0.0)),
        "in_frame_ratio": float(getattr(metrics, "in_frame_ratio", 0.0)),
    }


def _cached_image_quality_metrics(
    *, context: SceneContext, image_name: str, image_path: Path
) -> dict[str, Any]:
    if image_name in context.image_quality_cache:
        context.cache_stats["image_quality_hits"] += 1
        return context.image_quality_cache[image_name]
    context.cache_stats["image_quality_misses"] += 1
    metrics = _read_image_quality_metrics(image_path)
    context.image_quality_cache[image_name] = metrics
    return metrics


def _cached_static_visibility(
    obj: dict[str, Any],
    *,
    image_name: str,
    pose: Any,
    context: SceneContext,
) -> tuple[bool, float, str, dict[str, Any]]:
    key = (image_name, int(obj["id"]))
    if key in context.static_visibility_cache:
        context.cache_stats["static_visibility_hits"] += 1
        return context.static_visibility_cache[key]
    context.cache_stats["static_visibility_misses"] += 1
    if image_name not in context.depth_frame_cache:
        context.depth_frame_cache[image_name] = context.data_source.load_depth_frame(image_name)
        context.cache_stats["depth_frame_misses"] += 1
    else:
        context.cache_stats["depth_frame_hits"] += 1
    result = _static_visibility(
        obj,
        pose=pose,
        context=context,
        depth_frame=context.depth_frame_cache[image_name],
    )
    context.static_visibility_cache[key] = result
    return result


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
        max_surface_samples=MESH_RAY_SURFACE_SAMPLES,
        bbox_probe_ray_count=MESH_RAY_BBOX_SAMPLES,
        local_resample_count=MESH_RAY_LOCAL_RESAMPLES,
    )
    ratio = _metrics_visible_ratio(metrics)
    valid = int(getattr(metrics, "valid_in_frame_count", 0) or 0)
    return valid > 0 and ratio > FULLY_VISIBLE_RATIO_MIN, ratio, {
        "valid_in_frame_count": valid,
        "visible_in_frame_count": int(getattr(metrics, "visible_in_frame_count", 0) or 0),
        "projected_area": float(getattr(metrics, "projected_area", 0.0)),
        "in_frame_ratio": float(getattr(metrics, "in_frame_ratio", 0.0)),
    }


def _projection_gate(
    obj: dict[str, Any], pose: Any, context: SceneContext
) -> tuple[bool, dict[str, float]]:
    area, in_frame_ratio = quick_moved_bbox_projection(obj, pose, context.intrinsics)
    roi = _project_object_roi(obj, pose, context.intrinsics)
    edge_margin = float(roi.get("edge_margin_px", 0.0) or 0.0)
    minimum_area = min_projected_area_px(context.intrinsics.width, context.intrinsics.height)
    minimum_margin = float(default_edge_margin_px(context.intrinsics))
    passed = bool(
        _bbox_fully_in_frame(obj, pose, context.intrinsics)
        and in_frame_ratio >= 1.0
        and area >= minimum_area
        and edge_margin >= minimum_margin
    )
    return passed, {
        "projected_area_px": float(area),
        "projected_area_ratio": float(area / max(context.intrinsics.width * context.intrinsics.height, 1)),
        "in_frame_ratio": float(in_frame_ratio),
        "edge_margin_px": edge_margin,
        "edge_margin_ratio": float(edge_margin / max(min(context.intrinsics.width, context.intrinsics.height), 1)),
        "minimum_projected_area_px": float(minimum_area),
        "minimum_edge_margin_px": minimum_margin,
    }


def _evaluate_question_route(
    *,
    question: dict[str, Any],
    source_index: int,
    context: SceneContext,
) -> tuple[RouteCandidate | None, str | None]:
    try:
        source_name, auxiliary_names, destination_name = _question_route_names(question)
        referenced_names = [source_name, *auxiliary_names, destination_name]
        for image_name in referenced_names:
            if image_name not in context.poses:
                raise ValueError(f"pose is missing for route image {image_name}")
            if not context.data_source.image_path(image_name).resolve().is_file():
                raise FileNotFoundError(f"route image not found: {image_name}")
        bridge_name = _select_midpoint_bridge(
            source_name, auxiliary_names, destination_name, context.poses
        )
        generation_names = tuple(
            [source_name, *([bridge_name] if bridge_name else []), destination_name]
        )
        generation_roles = tuple(
            [
                "source_view",
                *(["source_to_destination_bridge"] if bridge_name else []),
                "destination_environment",
            ]
        )
        motion = _apply_question_motion(question, context.objects, context.attachment_graph)
        for obj_id in set((*motion.source_required_ids, *motion.moved_ids)):
            if obj_id not in context.objects_by_id:
                raise ValueError(f"object {obj_id} is absent from scene")
        if bool(question.get("has_attachment_chain")) and len(motion.moved_ids) <= 1:
            raise ValueError("question requires an attachment chain but none was found")

        quality_metrics: dict[str, dict[str, float]] = {}
        laplacian_scores: list[float] = []
        tenengrad_scores: list[float] = []
        for image_name in generation_names:
            image_path = context.data_source.image_path(image_name).resolve()
            quality = _cached_image_quality_metrics(
                context=context, image_name=image_name, image_path=image_path
            )
            if not bool(quality.get("readable")):
                return None, f"image_unreadable:{image_name}"
            laplacian = float(quality["laplacian_variance"])
            tenengrad = float(quality["tenengrad"])
            quality_metrics[image_name] = {
                "laplacian_variance": laplacian,
                "tenengrad": tenengrad,
            }
            laplacian_scores.append(laplacian)
            tenengrad_scores.append(tenengrad)

        source_pose = context.poses[source_name]
        destination_pose = context.poses[destination_name]
        source_metrics: dict[str, Any] = {}
        source_ratios: list[float] = []
        for obj_id in motion.moved_ids:
            obj = context.objects_by_id[obj_id]
            _, projection = _projection_gate(obj, source_pose, context)
            _, ratio, backend, visibility = _cached_static_visibility(
                obj, image_name=source_name, pose=source_pose, context=context
            )
            source_ratios.append(ratio)
            source_metrics[str(obj_id)] = {
                "projection": projection,
                "visibility_backend": backend,
                "visible_ratio": ratio,
                "visibility": visibility,
            }

        destination_metrics: dict[str, Any] = {}
        destination_ratios: list[float] = []
        for obj_id in motion.moved_ids:
            moved_obj = motion.moved_objects_by_id[obj_id]
            _, projection = _projection_gate(moved_obj, destination_pose, context)
            _, ratio, visibility = _future_visibility(
                obj_id, pose=destination_pose, context=context, motion=motion
            )
            destination_ratios.append(ratio)
            destination_metrics[str(obj_id)] = {
                "projection": projection,
                "visibility_backend": "counterfactual_mesh_ray",
                "visible_ratio": ratio,
                "visibility": visibility,
            }

        query_id = _coerce_int(question.get("query_obj_id"), field="query_obj_id")
        if query_id not in motion.moved_ids:
            query_obj = context.objects_by_id[query_id]
            _, projection = _projection_gate(query_obj, destination_pose, context)
            _, ratio, backend, visibility = _cached_static_visibility(
                query_obj,
                image_name=destination_name,
                pose=destination_pose,
                context=context,
            )
            destination_metrics[f"query:{query_id}"] = {
                "projection": projection,
                "visibility_backend": backend,
                "visible_ratio": ratio,
                "visibility": visibility,
            }

        score_key = (
            min(destination_ratios),
            min(source_ratios),
            min(laplacian_scores),
            min(tenengrad_scores),
        )
        dataset = context.dataset
        return RouteCandidate(
            question=question,
            source_index=source_index,
            dataset=dataset,
            scene_id=context.scene_id,
            generation_image_names=generation_names,
            generation_roles=generation_roles,
            score_key=score_key,
            metrics={
                "route": {
                    "source": source_name,
                    "bridge": bridge_name,
                    "destination": destination_name,
                    "auxiliary_candidates": auxiliary_names,
                },
                "source": source_metrics,
                "destination": destination_metrics,
                "image_quality": quality_metrics,
                "score": list(score_key),
            },
        ), None
    except Exception as exc:
        return None, str(exc)


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
    ray_caster = RayCaster.from_ply(str(data_source.mesh_path()), axis_alignment=axis_alignment)
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


def _load_materialization_context(
    *,
    dataset: str,
    scene_id: str,
    scannet_root: Path | None,
    scannetpp_root: Path | None,
    scannetpp_frame_root: Path | None,
    scannetpp_sensor: str,
) -> SceneContext:
    """Load only the camera and object data needed to copy views and make crops."""
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
    return SceneContext(
        dataset=dataset,
        scene_id=scene_id,
        scene_dir=scene_dir,
        data_source=data_source,
        objects=objects,
        objects_by_id={int(obj["id"]): obj for obj in objects},
        attachment_graph=attachment_graph,
        intrinsics=data_source.load_intrinsics(),
        poses=data_source.load_poses(),
        ray_caster=None,
        instance_mesh_data=None,
    )


def _question_required_instance_ids(question: dict[str, Any]) -> set[int]:
    required = {
        _coerce_int(question.get("moved_obj_id"), field="moved_obj_id"),
        _coerce_int(question.get("query_obj_id"), field="query_obj_id"),
    }
    if str(question.get("type") or "") == "object_rotate_object_centric":
        required.add(_coerce_int(question.get("obj_face_id"), field="obj_face_id"))
    return required


def _copy_media(source: Path, target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() != target.resolve():
        shutil.copy2(source, target)
    return target.resolve()


def _question_frame_group_ids(question: dict[str, Any], frame_key: str) -> set[int]:
    raw_groups = question.get("object_frame_groups")
    raw_ids = raw_groups.get(frame_key) if isinstance(raw_groups, dict) else None
    if not isinstance(raw_ids, (list, tuple)):
        return set()
    resolved: set[int] = set()
    for value in raw_ids:
        try:
            resolved.add(_coerce_int(value, field=f"object_frame_groups.{frame_key}"))
        except ValueError:
            continue
    return resolved


def _clipped_object_roi(
    obj: dict[str, Any], pose: Any, context: SceneContext
) -> tuple[int, int, int, int] | None:
    raw_bounds = _project_object_roi(obj, pose, context.intrinsics).get("roi_bounds")
    if raw_bounds is None:
        return None
    intrinsic_width = max(int(context.intrinsics.width), 1)
    intrinsic_height = max(int(context.intrinsics.height), 1)
    left, right, top, bottom = (int(value) for value in raw_bounds)
    clipped = (
        max(0, min(intrinsic_width, left)),
        max(0, min(intrinsic_width, right)),
        max(0, min(intrinsic_height, top)),
        max(0, min(intrinsic_height, bottom)),
    )
    if clipped[1] <= clipped[0] or clipped[3] <= clipped[2]:
        return None
    return clipped


def _qwen_moving_group_reference_layout(
    candidate: RouteCandidate,
    context: SceneContext,
) -> tuple[dict[int, tuple[int, int, int, int]], set[int], set[int]]:
    """Assign each moved member to the Qwen input that actually shows it."""
    motion = _apply_question_motion(
        candidate.question, context.objects, context.attachment_graph
    )
    source_pose = context.poses[candidate.generation_image_names[0]]
    destination_pose = context.poses[candidate.generation_image_names[-1]]
    frame_1_ids = _question_frame_group_ids(candidate.question, "frame_1")
    frame_2_ids = _question_frame_group_ids(candidate.question, "frame_2")
    source_bounds: dict[int, tuple[int, int, int, int]] = {}
    destination_ids: set[int] = set()
    unavailable_ids: set[int] = set()

    for obj_id in motion.moved_ids:
        # Cross-frame questions deliberately reserve some objects for the last view.
        if obj_id in frame_2_ids and obj_id not in frame_1_ids:
            destination_ids.add(obj_id)
            continue
        source_roi = _clipped_object_roi(
            context.objects_by_id[obj_id], source_pose, context
        )
        if source_roi is not None:
            source_bounds[obj_id] = source_roi
            continue
        destination_roi = _clipped_object_roi(
            context.objects_by_id[obj_id], destination_pose, context
        )
        if destination_roi is not None:
            destination_ids.add(obj_id)
        else:
            unavailable_ids.add(obj_id)
    return source_bounds, destination_ids, unavailable_ids


def _materialize_qwen_moving_group_reference(
    candidate: RouteCandidate,
    context: SceneContext,
    output_dir: Path,
    source_image: Path,
) -> dict[str, Any]:
    """Crop the source-view moving group so Qwen does not receive two full scenes."""
    source_bounds, destination_ids, unavailable_ids = _qwen_moving_group_reference_layout(
        candidate, context
    )
    intrinsic_width = max(int(context.intrinsics.width), 1)
    intrinsic_height = max(int(context.intrinsics.height), 1)
    bounds = list(source_bounds.values())
    if destination_ids:
        print(
            f"[selection] {candidate.question['question_uid']}: moving-group object(s) "
            f"{sorted(destination_ids)} use the destination environment as their visual reference",
            file=sys.stderr,
            flush=True,
        )
    if unavailable_ids:
        print(
            f"[selection] {candidate.question['question_uid']}: moving-group object(s) "
            f"{sorted(unavailable_ids)} have no visual reference in either Qwen input",
            file=sys.stderr,
            flush=True,
        )

    with Image.open(source_image) as image:
        width, height = image.size
        reference = image.convert("RGB")
        if bounds:
            left = min(item[0] for item in bounds) * width // intrinsic_width
            right = max(item[1] for item in bounds) * width // intrinsic_width
            top = min(item[2] for item in bounds) * height // intrinsic_height
            bottom = max(item[3] for item in bounds) * height // intrinsic_height
            padding = max(12, round(0.15 * max(right - left, bottom - top)))
            left = max(0, left - padding)
            top = max(0, top - padding)
            right = min(width, right + padding)
            bottom = min(height, bottom + padding)
            if right - left < 2 or bottom - top < 2:
                raise ValueError(f"{candidate.question['question_uid']}: invalid moving-group crop")
            reference = reference.crop((left, top, right, bottom))

    target = output_dir / "media" / "qwen_references" / f"{candidate.question['question_uid']}.png"
    target.parent.mkdir(parents=True, exist_ok=True)
    reference.save(target, format="PNG")
    return {
        "path": str(target.resolve()),
        "role": "moving_group_reference",
        "sha256": _sha256_file(target),
        "source_obj_ids": sorted(source_bounds),
        "destination_obj_ids": sorted(destination_ids),
        "unavailable_obj_ids": sorted(unavailable_ids),
    }


def _materialize_route(
    candidate: RouteCandidate, context: SceneContext, output_dir: Path
) -> dict[str, Any]:
    uid = str(candidate.question["question_uid"])
    generation_images: list[dict[str, str]] = []
    copied_by_role: dict[str, Path] = {}
    for index, (image_name, role) in enumerate(
        zip(candidate.generation_image_names, candidate.generation_roles)
    ):
        source = context.data_source.image_path(image_name).resolve()
        suffix = source.suffix.lower() or ".jpg"
        copied = _copy_media(
            source,
            output_dir / "media" / "conditions" / uid / f"{index:02d}_{role}{suffix}",
        )
        copied_by_role[role] = copied
        generation_images.append({"path": str(copied), "role": role})
    answer_context_media: list[dict[str, str]] = []
    bridge = copied_by_role.get("source_to_destination_bridge")
    if bridge is not None:
        answer_context_media.append(
            {"path": str(bridge), "role": "destination_to_query_bridge"}
        )
    destination = copied_by_role["destination_environment"]
    answer_context_media.append({"path": str(destination), "role": "query_reference_view"})

    motion = _apply_question_motion(
        candidate.question, context.objects, context.attachment_graph
    )
    qwen_reference = _materialize_qwen_moving_group_reference(
        candidate, context, output_dir, copied_by_role["source_view"]
    )
    source_reference_ids = {
        int(value) for value in qwen_reference.get("source_obj_ids", motion.moved_ids)
    }
    destination_reference_ids = {
        int(value) for value in qwen_reference.get("destination_obj_ids", [])
    }
    unavailable_ids = {
        int(value) for value in qwen_reference.get("unavailable_obj_ids", [])
    }
    moving_group: list[dict[str, Any]] = []
    for obj_id in motion.moved_ids:
        item: dict[str, Any] = {
            "obj_id": obj_id,
            "label": str(context.objects_by_id[obj_id].get("label") or "").strip(),
        }
        if obj_id in source_reference_ids:
            item["visual_reference_role"] = "moving_group_reference"
        elif obj_id in destination_reference_ids:
            item["visual_reference_role"] = "destination_environment"
        moving_group.append(item)
    if any(not item["label"] for item in moving_group):
        raise ValueError(f"{uid}: moving group contains an empty label")
    qwen_rejection_reasons: list[str] = []
    if not source_reference_ids:
        qwen_rejection_reasons.append("no_source_moving_group_reference")
    if unavailable_ids:
        qwen_rejection_reasons.append(
            "moving_group_without_visual_reference:"
            + ",".join(str(value) for value in sorted(unavailable_ids))
        )
    qwen_picture_eligible = not qwen_rejection_reasons
    source_pose = context.poses[candidate.generation_image_names[0]]
    entry: dict[str, Any] = {
        "question_uid": uid,
        "source_index": candidate.source_index,
        "generation_images": generation_images,
        "qwen_reference_image": qwen_reference,
        "camera_rotation_world_to_camera": np.asarray(
            source_pose.rotation, dtype=np.float64
        ).tolist(),
        "moving_group": moving_group,
        "picture_eligible": True,
        "qwen_picture_eligible": qwen_picture_eligible,
        "qwen_picture_rejection_reasons": qwen_rejection_reasons,
        "video_eligible": True,
        "answer_context_media": answer_context_media,
    }
    if str(candidate.question.get("type") or "") == "object_rotate_object_centric":
        entry["orbit_anchor_label"] = str(
            candidate.question.get("obj_face_label") or ""
        ).strip()
    candidate.metrics["materialized"] = {
        "generation_images": [
            {
                "source_image_name": image_name,
                "role": role,
                "path": item["path"],
                "sha256": _sha256_file(Path(item["path"])),
            }
            for image_name, role, item in zip(
                candidate.generation_image_names,
                candidate.generation_roles,
                generation_images,
            )
        ],
        "qwen_visual_references": {
            "source_obj_ids": sorted(source_reference_ids),
            "destination_obj_ids": sorted(destination_reference_ids),
            "unavailable_obj_ids": sorted(unavailable_ids),
            "eligible": qwen_picture_eligible,
        },
    }
    return entry


def _materialize_routes(
    candidates: list[RouteCandidate],
    *,
    output_dir: Path,
    scannet_root: Path | None,
    scannetpp_root: Path | None,
    scannetpp_frame_root: Path | None,
    scannetpp_sensor: str,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[RouteCandidate]] = defaultdict(list)
    for candidate in candidates:
        grouped[(candidate.dataset, candidate.scene_id)].append(candidate)
    materialized: dict[int, dict[str, Any]] = {}
    scene_count = len(grouped)
    for scene_index, ((dataset, scene_id), scene_candidates) in enumerate(
        grouped.items(), start=1
    ):
        print(
            f"[selection] materializing {scene_index}/{scene_count} "
            f"{dataset}:{scene_id} routes={len(scene_candidates)}",
            file=sys.stderr,
            flush=True,
        )
        context = _load_materialization_context(
            dataset=dataset,
            scene_id=scene_id,
            scannet_root=scannet_root,
            scannetpp_root=scannetpp_root,
            scannetpp_frame_root=scannetpp_frame_root,
            scannetpp_sensor=scannetpp_sensor,
        )
        for candidate in scene_candidates:
            materialized[candidate.source_index] = _materialize_route(
                candidate, context, output_dir
            )
    return [materialized[candidate.source_index] for candidate in candidates]


def _checkpoint_configuration(
    *,
    benchmark_path: Path,
    scannet_root: Path | None,
    scannetpp_root: Path | None,
    scannetpp_frame_root: Path | None,
    scannetpp_sensor: str,
) -> dict[str, Any]:
    return {
        "schema_version": SELECTION_CHECKPOINT_SCHEMA_VERSION,
        "algorithm_version": SELECTION_ALGORITHM_VERSION,
        "benchmark_sha256": _sha256_file(benchmark_path),
        "scannet_root": str(scannet_root.resolve()) if scannet_root else None,
        "scannetpp_root": str(scannetpp_root.resolve()) if scannetpp_root else None,
        "scannetpp_frame_root": str(scannetpp_frame_root.resolve()) if scannetpp_frame_root else None,
        "scannetpp_sensor": scannetpp_sensor,
        "mesh_ray_surface_samples": MESH_RAY_SURFACE_SAMPLES,
        "mesh_ray_bbox_samples": MESH_RAY_BBOX_SAMPLES,
        "mesh_ray_local_resamples": MESH_RAY_LOCAL_RESAMPLES,
    }


def _checkpoint_path(output_dir: Path, configuration: dict[str, Any]) -> Path:
    encoded = json.dumps(configuration, sort_keys=True, separators=(",", ":")).encode("utf-8")
    fingerprint = hashlib.sha256(encoded).hexdigest()[:16]
    return output_dir / "private_jobs" / f"selection_checkpoint_{fingerprint}.jsonl"


def _load_or_create_checkpoint(
    path: Path, configuration: dict[str, Any]
) -> dict[int, dict[str, Any]]:
    expected_header = {"kind": "header", "configuration": configuration}
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(expected_header, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return {}
    results: dict[int, dict[str, Any]] = {}
    needs_rewrite = False
    with path.open("r", encoding="utf-8") as stream:
        try:
            header = json.loads(stream.readline())
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid selection checkpoint header: {path}") from exc
        if header != expected_header:
            raise ValueError(f"selection checkpoint configuration mismatch: {path}")
        for line_number, line in enumerate(stream, start=2):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                source_index = int(record["source_index"])
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                needs_rewrite = True
                print(
                    f"[selection] ignoring incomplete checkpoint record at {path}:{line_number}",
                    file=sys.stderr,
                    flush=True,
                )
                continue
            if record.get("kind") == "question_result":
                if source_index in results:
                    needs_rewrite = True
                results[source_index] = record
    if needs_rewrite:
        with path.with_suffix(path.suffix + ".tmp").open("w", encoding="utf-8") as stream:
            stream.write(json.dumps(expected_header, ensure_ascii=False, sort_keys=True) + "\n")
            for source_index in sorted(results):
                stream.write(json.dumps(results[source_index], ensure_ascii=False, sort_keys=True) + "\n")
        path.with_suffix(path.suffix + ".tmp").replace(path)
    return results


def _append_checkpoint_result(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()


def _checkpoint_record_matches_question(
    record: dict[str, Any], *, source_index: int, question: dict[str, Any]
) -> bool:
    return bool(
        int(record.get("source_index", -1)) == source_index
        and str(record.get("question_uid") or "") == str(question.get("question_uid") or "")
        and str(record.get("question_type") or "") == str(question.get("type") or "")
        and str(record.get("scene_id") or "") == str(question.get("scene_id") or "").strip()
    )


def _eligible_checkpoint_record(candidate: RouteCandidate) -> dict[str, Any]:
    return {
        "kind": "question_result",
        "source_index": candidate.source_index,
        "question_uid": str(candidate.question.get("question_uid") or ""),
        "question_type": str(candidate.question.get("type") or ""),
        "dataset": candidate.dataset,
        "scene_id": candidate.scene_id,
        "status": "eligible",
        "candidate": {
            "generation_image_names": list(candidate.generation_image_names),
            "generation_roles": list(candidate.generation_roles),
            "score_key": list(candidate.score_key),
            "metrics": candidate.metrics,
        },
    }


def _rejected_checkpoint_record(
    rejection: dict[str, Any], *, dataset: str
) -> dict[str, Any]:
    return {
        "kind": "question_result",
        "source_index": int(rejection["source_index"]),
        "question_uid": str(rejection.get("question_uid") or ""),
        "question_type": str(rejection.get("question_type") or ""),
        "dataset": dataset,
        "scene_id": str(rejection.get("scene_id") or ""),
        "status": "rejected",
        "rejection": rejection,
    }


def _route_from_checkpoint(
    record: dict[str, Any], question: dict[str, Any]
) -> RouteCandidate:
    raw = record.get("candidate")
    if not isinstance(raw, dict):
        raise ValueError("eligible checkpoint record is missing candidate data")
    names = raw.get("generation_image_names")
    roles = raw.get("generation_roles")
    score = raw.get("score_key")
    metrics = raw.get("metrics")
    if not isinstance(names, list) or not isinstance(roles, list) or not isinstance(score, list) or not isinstance(metrics, dict):
        raise ValueError("eligible checkpoint route data is malformed")
    return RouteCandidate(
        question=question,
        source_index=int(record["source_index"]),
        dataset=str(record["dataset"]),
        scene_id=str(record["scene_id"]),
        generation_image_names=tuple(str(value) for value in names),
        generation_roles=tuple(str(value) for value in roles),
        score_key=tuple(float(value) for value in score),
        metrics=metrics,
    )


def generate_selection_spec(
    *,
    benchmark_path: Path,
    output_dir: Path,
    scannet_root: Path | None,
    scannetpp_root: Path | None,
    scannetpp_frame_root: Path | None,
    scannetpp_sensor: str = "iphone",
    expected_per_type: int = 2,
) -> SelectionPaths:
    if expected_per_type <= 0:
        raise ValueError("expected_per_type must be positive")
    questions, _, _ = load_fixed_questions(benchmark_path)
    indexed_questions = [
        (index, question)
        for index, question in enumerate(questions)
        if str(question.get("type") or "") in L2_ROLLOUT_TYPES
    ]
    configuration = _checkpoint_configuration(
        benchmark_path=benchmark_path,
        scannet_root=scannet_root,
        scannetpp_root=scannetpp_root,
        scannetpp_frame_root=scannetpp_frame_root,
        scannetpp_sensor=scannetpp_sensor,
    )
    checkpoint_path = _checkpoint_path(output_dir, configuration)
    checkpoint_results = _load_or_create_checkpoint(checkpoint_path, configuration)
    grouped: dict[tuple[str, str], list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for source_index, question in indexed_questions:
        grouped[(_infer_dataset(question), str(question.get("scene_id") or "").strip())].append(
            (source_index, question)
        )

    eligible: dict[str, list[RouteCandidate]] = defaultdict(list)
    rejected: list[dict[str, Any]] = []
    scene_errors: dict[str, str] = {}
    processed_indices: set[int] = set()
    for source_index, question in indexed_questions:
        record = checkpoint_results.get(source_index)
        if record is None or not _checkpoint_record_matches_question(
            record, source_index=source_index, question=question
        ):
            continue
        try:
            if record.get("status") == "eligible":
                eligible[str(question["type"])].append(_route_from_checkpoint(record, question))
            elif record.get("status") == "rejected" and isinstance(record.get("rejection"), dict):
                rejected.append(record["rejection"])
            else:
                raise ValueError("checkpoint record has an unknown status")
        except (KeyError, TypeError, ValueError) as exc:
            print(
                f"[selection] ignoring invalid checkpoint result for source index {source_index}: {exc}",
                file=sys.stderr,
                flush=True,
            )
            continue
        processed_indices.add(source_index)

    progress = SelectionProgress(
        total=len(indexed_questions),
        processed=len(processed_indices),
        eligible=sum(len(items) for items in eligible.values()),
        rejected=len(rejected),
    )
    progress.report_resume(len(processed_indices), checkpoint_path)
    cache_stats: Counter[str] = Counter()
    scene_groups = sorted(grouped.items())
    for scene_index, ((dataset, scene_id), scene_questions) in enumerate(scene_groups, start=1):
        pending = [item for item in scene_questions if item[0] not in processed_indices]
        if not pending:
            continue
        progress.report_scene(
            scene_index=scene_index,
            scene_count=len(scene_groups),
            dataset=dataset,
            scene_id=scene_id,
            pending_questions=len(pending),
        )
        valid_questions: list[tuple[int, dict[str, Any]]] = []
        required_ids: set[int] = set()
        for source_index, question in pending:
            try:
                if not scene_id:
                    raise ValueError("scene_id must be non-empty")
                required_ids.update(_question_required_instance_ids(question))
                _question_route_names(question)
            except ValueError as exc:
                rejection = {
                    "question_uid": question.get("question_uid"),
                    "source_index": source_index,
                    "question_type": question.get("type"),
                    "scene_id": scene_id,
                    "reason": "malformed_question",
                    "detail": str(exc),
                }
                rejected.append(rejection)
                _append_checkpoint_result(
                    checkpoint_path, _rejected_checkpoint_record(rejection, dataset=dataset)
                )
                processed_indices.add(source_index)
                progress.advance(was_eligible=False)
                continue
            valid_questions.append((source_index, question))
        if not valid_questions:
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
            scene_errors[f"{dataset}:{scene_id}"] = str(exc)
            for source_index, question in valid_questions:
                rejection = {
                    "question_uid": question.get("question_uid"),
                    "source_index": source_index,
                    "question_type": question.get("type"),
                    "scene_id": scene_id,
                    "reason": "scene_resources_unavailable",
                    "detail": str(exc),
                }
                rejected.append(rejection)
                _append_checkpoint_result(
                    checkpoint_path, _rejected_checkpoint_record(rejection, dataset=dataset)
                )
                processed_indices.add(source_index)
                progress.advance(was_eligible=False)
            continue

        for source_index, question in valid_questions:
            candidate, detail = _evaluate_question_route(
                question=question, source_index=source_index, context=context
            )
            if candidate is None:
                rejection = {
                    "question_uid": question.get("question_uid"),
                    "source_index": source_index,
                    "question_type": question.get("type"),
                    "scene_id": scene_id,
                    "reason": "route_not_eligible",
                    "detail": detail,
                }
                rejected.append(rejection)
                _append_checkpoint_result(
                    checkpoint_path, _rejected_checkpoint_record(rejection, dataset=dataset)
                )
                progress.advance(was_eligible=False)
            else:
                eligible[str(question["type"])].append(candidate)
                _append_checkpoint_result(checkpoint_path, _eligible_checkpoint_record(candidate))
                progress.advance(was_eligible=True)
            processed_indices.add(source_index)
        cache_stats.update(context.cache_stats)

    selected: list[RouteCandidate] = []
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
                    "scene_id": candidate.scene_id,
                    "reason": "type_quota_exceeded",
                }
            )
    selected.sort(key=lambda item: L2_ROLLOUT_TYPES.index(str(item.question["type"])))
    entries = _materialize_routes(
        selected,
        output_dir=output_dir,
        scannet_root=scannet_root,
        scannetpp_root=scannetpp_root,
        scannetpp_frame_root=scannetpp_frame_root,
        scannetpp_sensor=scannetpp_sensor,
    )
    private_dir = output_dir / "private_jobs"
    spec_path = private_dir / "selection_spec.json"
    audit_path = private_dir / "selection_audit.json"
    _atomic_write_json(
        spec_path,
        {
            "schema_version": SELECTION_SCHEMA_VERSION,
            "metadata": {
                "benchmark_path": str(benchmark_path.resolve()),
                "benchmark_sha256": _sha256_file(benchmark_path),
                "max_per_type": expected_per_type,
                "selection_mode": "source_destination_route",
                "selection_algorithm_version": SELECTION_ALGORITHM_VERSION,
                "picture_input_cap": 3,
            },
            "entries": entries,
        },
    )
    rejection_counts = Counter(str(item["reason"]) for item in rejected)
    _atomic_write_json(
        audit_path,
        {
            "schema_version": SELECTION_AUDIT_SCHEMA_VERSION,
            "benchmark_path": str(benchmark_path.resolve()),
            "benchmark_sha256": _sha256_file(benchmark_path),
            "configuration": {
                "max_per_type": expected_per_type,
                "route_source": "image_name",
                "route_auxiliary": "auxiliary_image_names",
                "route_destination": "reasoning_frame_2",
                "bridge_rule": "closest_to_half_cumulative_camera_distance_earlier_tie",
                "picture_input_cap": 3,
                "source_visibility": "moved_group_registered_depth_then_mesh_ray",
                "destination_visibility": "moved_group_counterfactual_mesh_ray_and_static_query",
                "checkpoint_path": str(checkpoint_path.resolve()),
            },
            "cache_stats_current_run": dict(sorted(cache_stats.items())),
            "counts_by_type": counts,
            "selected": [
                {
                    "question_uid": candidate.question.get("question_uid"),
                    "source_index": candidate.source_index,
                    "question_type": candidate.question.get("type"),
                    "scene_id": candidate.scene_id,
                    "generation_image_names": list(candidate.generation_image_names),
                    "metrics": candidate.metrics,
                }
                for candidate in selected
            ],
            "rejection_counts": dict(sorted(rejection_counts.items())),
            "rejected": rejected,
            "scene_errors": scene_errors,
        },
    )
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
        default=2,
        help="maximum number of selected questions per supported type",
    )
    args = parser.parse_args(argv)
    if not args.benchmark_file.is_file():
        parser.error(f"--benchmark_file not found: {args.benchmark_file}")
    if args.expected_per_type <= 0:
        parser.error("--expected_per_type must be positive")
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
    )
    print(f"selection_spec : {paths.spec}")
    print(f"selection_audit: {paths.audit}")


if __name__ == "__main__":
    main(sys.argv[1:])
