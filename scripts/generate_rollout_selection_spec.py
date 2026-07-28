#!/usr/bin/env python3
"""Automatically select leakage-safe single-frame future-rollout inputs."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import sys
import time
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
SELECTION_CHECKPOINT_SCHEMA_VERSION = "predictive-spatial-selection-checkpoint-v1"
SELECTION_ALGORITHM_VERSION = "strict-single-frame-v3-two-stage-mesh-ray"
DEFAULT_MESH_RAY_SHORTLIST_SIZE = 32
DEFAULT_MESH_RAY_SURFACE_SAMPLES = 64
DEFAULT_MESH_RAY_BBOX_SAMPLES = 0
DEFAULT_MESH_RAY_LOCAL_RESAMPLES = 4


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
    static_visibility_cache: dict[
        tuple[Any, ...], tuple[bool, float, str, dict[str, Any]]
    ] = field(default_factory=dict)
    depth_frame_cache: dict[str, Any | None] = field(default_factory=dict)
    cache_stats: Counter[str] = field(default_factory=Counter)


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


@dataclass
class FramePrefilterCandidate:
    question: dict[str, Any]
    source_index: int
    context: SceneContext
    motion: QuestionMotion
    image_name: str
    image_path: Path
    pose: Any
    prefilter_score_key: tuple[Any, ...]
    source_projections: dict[int, dict[str, float]]
    future_projections: dict[int, dict[str, float]]
    image_quality: dict[str, float]
    context_paths: list[Path]
    prefilter_rank: int = 0


@dataclass
class CandidateRecord:
    question: dict[str, Any]
    source_index: int
    dataset: str
    scene_id: str
    image_name: str
    score_key: tuple[Any, ...]
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
        if restored <= 0:
            return
        print(
            f"[selection] resumed {restored}/{self.total} completed questions "
            f"from {checkpoint_path}",
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
        if (
            self.processed == self.total
            or self.processed % 10 == 0
            or now - self.last_report_at >= 10.0
        ):
            elapsed = max(now - self.started_at, 0.0)
            run_processed = self.processed - self.run_start_processed
            rate = run_processed / elapsed if elapsed > 0.0 else 0.0
            remaining = self.total - self.processed
            eta = remaining / rate if rate > 0.0 else 0.0
            percent = 100.0 * self.processed / max(self.total, 1)
            print(
                f"[selection] {self.processed}/{self.total} ({percent:.1f}%) "
                f"eligible={self.eligible} rejected={self.rejected} "
                f"elapsed={_format_duration(elapsed)} eta={_format_duration(eta)}",
                file=sys.stderr,
                flush=True,
            )
            self.last_report_at = now


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


def _format_duration(seconds: float) -> str:
    total_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds_part = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m"
    if minutes:
        return f"{minutes:d}m{seconds_part:02d}s"
    return f"{seconds_part:d}s"


def _resolved_path_text(path: Path | None) -> str | None:
    return str(path.resolve()) if path is not None else None


def _checkpoint_configuration(
    *,
    benchmark_path: Path,
    scannet_root: Path | None,
    scannetpp_root: Path | None,
    scannetpp_frame_root: Path | None,
    scannetpp_sensor: str,
    frame_stride_scannet: int,
    frame_stride_scannetpp: int,
    mesh_ray_shortlist_size: int,
    mesh_ray_surface_samples: int,
    mesh_ray_local_resamples: int,
) -> dict[str, Any]:
    return {
        "schema_version": SELECTION_CHECKPOINT_SCHEMA_VERSION,
        "algorithm_version": SELECTION_ALGORITHM_VERSION,
        "benchmark_sha256": _sha256_file(benchmark_path),
        "scannet_root": _resolved_path_text(scannet_root),
        "scannetpp_root": _resolved_path_text(scannetpp_root),
        "scannetpp_frame_root": _resolved_path_text(scannetpp_frame_root),
        "scannetpp_sensor": scannetpp_sensor,
        "frame_stride_scannet": frame_stride_scannet,
        "frame_stride_scannetpp": frame_stride_scannetpp,
        "mesh_ray_shortlist_size": mesh_ray_shortlist_size,
        "mesh_ray_surface_samples": mesh_ray_surface_samples,
        "mesh_ray_bbox_samples": DEFAULT_MESH_RAY_BBOX_SAMPLES,
        "mesh_ray_local_resamples": mesh_ray_local_resamples,
    }


def _checkpoint_path(output_dir: Path, configuration: dict[str, Any]) -> Path:
    encoded = json.dumps(
        configuration,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    fingerprint = hashlib.sha256(encoded).hexdigest()[:16]
    return output_dir / "private_jobs" / f"selection_checkpoint_{fingerprint}.jsonl"


def _load_or_create_checkpoint(
    path: Path,
    configuration: dict[str, Any],
) -> dict[int, dict[str, Any]]:
    expected_header = {"kind": "header", "configuration": configuration}
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(expected_header, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
        return {}

    results: dict[int, dict[str, Any]] = {}
    needs_rewrite = False
    with path.open("r", encoding="utf-8") as stream:
        header_line = stream.readline()
        try:
            header = json.loads(header_line)
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
                    f"[selection] ignoring incomplete checkpoint record at "
                    f"{path}:{line_number}",
                    file=sys.stderr,
                    flush=True,
                )
                continue
            if record.get("kind") == "question_result":
                if source_index in results:
                    needs_rewrite = True
                results[source_index] = record
    if needs_rewrite:
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            stream.write(json.dumps(expected_header, ensure_ascii=False, sort_keys=True) + "\n")
            for source_index in sorted(results):
                stream.write(
                    json.dumps(results[source_index], ensure_ascii=False, sort_keys=True)
                    + "\n"
                )
        temporary.replace(path)
    return results


def _append_checkpoint_result(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()


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
    mesh_ray_surface_samples: int = DEFAULT_MESH_RAY_SURFACE_SAMPLES,
    mesh_ray_bbox_samples: int = DEFAULT_MESH_RAY_BBOX_SAMPLES,
    mesh_ray_local_resamples: int = DEFAULT_MESH_RAY_LOCAL_RESAMPLES,
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
        max_surface_samples=mesh_ray_surface_samples,
        bbox_probe_ray_count=mesh_ray_bbox_samples,
        local_resample_count=mesh_ray_local_resamples,
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


def _context_cache(context: Any, name: str) -> dict[Any, Any]:
    cache = getattr(context, name, None)
    if cache is None:
        cache = {}
        setattr(context, name, cache)
    return cache


def _context_cache_stats(context: Any) -> Counter[str]:
    stats = getattr(context, "cache_stats", None)
    if stats is None:
        stats = Counter()
        setattr(context, "cache_stats", stats)
    return stats


def _cached_image_quality_metrics(
    *,
    context: SceneContext,
    image_name: str,
    image_path: Path,
) -> dict[str, Any]:
    cache = _context_cache(context, "image_quality_cache")
    stats = _context_cache_stats(context)
    if image_name in cache:
        stats["image_quality_hits"] += 1
        return cache[image_name]
    stats["image_quality_misses"] += 1
    metrics = _read_image_quality_metrics(image_path)
    cache[image_name] = metrics
    return metrics


def _cached_static_visibility(
    obj: dict[str, Any],
    *,
    image_name: str,
    pose: Any,
    context: SceneContext,
    mesh_ray_surface_samples: int = DEFAULT_MESH_RAY_SURFACE_SAMPLES,
    mesh_ray_bbox_samples: int = DEFAULT_MESH_RAY_BBOX_SAMPLES,
    mesh_ray_local_resamples: int = DEFAULT_MESH_RAY_LOCAL_RESAMPLES,
) -> tuple[bool, float, str, dict[str, Any]]:
    obj_id = int(obj["id"])
    key = (
        image_name,
        obj_id,
        mesh_ray_surface_samples,
        mesh_ray_bbox_samples,
        mesh_ray_local_resamples,
    )
    cache = _context_cache(context, "static_visibility_cache")
    stats = _context_cache_stats(context)
    cached = cache.get(key)
    if cached is not None:
        stats["static_visibility_hits"] += 1
        return cached

    stats["static_visibility_misses"] += 1
    depth_cache = _context_cache(context, "depth_frame_cache")
    if image_name not in depth_cache:
        depth_cache[image_name] = context.data_source.load_depth_frame(image_name)
        stats["depth_frame_misses"] += 1
    else:
        stats["depth_frame_hits"] += 1
    result = _static_visibility(
        obj,
        pose=pose,
        context=context,
        depth_frame=depth_cache[image_name],
        mesh_ray_surface_samples=mesh_ray_surface_samples,
        mesh_ray_bbox_samples=mesh_ray_bbox_samples,
        mesh_ray_local_resamples=mesh_ray_local_resamples,
    )
    cache[key] = result
    return result


def _future_visibility(
    obj_id: int,
    *,
    pose: Any,
    context: SceneContext,
    motion: QuestionMotion,
    mesh_ray_surface_samples: int = DEFAULT_MESH_RAY_SURFACE_SAMPLES,
    mesh_ray_bbox_samples: int = DEFAULT_MESH_RAY_BBOX_SAMPLES,
    mesh_ray_local_resamples: int = DEFAULT_MESH_RAY_LOCAL_RESAMPLES,
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
        max_surface_samples=mesh_ray_surface_samples,
        bbox_probe_ray_count=mesh_ray_bbox_samples,
        local_resample_count=mesh_ray_local_resamples,
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


def _prefilter_frame(
    *,
    question: dict[str, Any],
    source_index: int,
    context: SceneContext,
    motion: QuestionMotion,
    image_name: str,
    context_paths: list[Path],
) -> tuple[FramePrefilterCandidate | None, str]:
    pose = context.poses[image_name]
    image_path = context.data_source.image_path(image_name).resolve()
    if not image_path.is_file():
        return None, "image_missing"
    quality = _cached_image_quality_metrics(
        context=context,
        image_name=image_name,
        image_path=image_path,
    )
    if not bool(quality.get("readable")):
        return None, "image_unreadable"
    laplacian = float(quality["laplacian_variance"])
    tenengrad = float(quality["tenengrad"])
    if not _passes_absolute_image_quality_gate(laplacian, tenengrad):
        return None, "image_quality_below_threshold"

    source_projections: dict[int, dict[str, float]] = {}
    future_projections: dict[int, dict[str, float]] = {}
    area_ratios: list[float] = []
    margin_ratios: list[float] = []

    for obj_id in motion.source_required_ids:
        obj = context.objects_by_id.get(obj_id)
        if obj is None:
            return None, "source_object_missing"
        projection_ok, projection = _projection_gate(obj, pose, context)
        if not projection_ok:
            return None, "source_projection_gate_failed"
        area_ratios.append(projection["projected_area_ratio"])
        margin_ratios.append(projection["edge_margin_ratio"])
        source_projections[obj_id] = projection

    for obj_id in motion.moved_ids:
        moved_obj = motion.moved_objects_by_id.get(obj_id)
        if moved_obj is None:
            return None, "future_object_missing"
        projection_ok, projection = _projection_gate(moved_obj, pose, context)
        if not projection_ok:
            return None, "future_projection_gate_failed"
        area_ratios.append(projection["projected_area_ratio"])
        margin_ratios.append(projection["edge_margin_ratio"])
        future_projections[obj_id] = projection

    original_bonus = int(image_name == str(question.get("image_name") or ""))
    prefilter_score_key: tuple[Any, ...] = (
        min(area_ratios),
        min(margin_ratios),
        laplacian,
        tenengrad,
        original_bonus,
    )
    return (
        FramePrefilterCandidate(
            question=question,
            source_index=source_index,
            context=context,
            motion=motion,
            image_name=image_name,
            image_path=image_path,
            pose=pose,
            prefilter_score_key=prefilter_score_key,
            source_projections=source_projections,
            future_projections=future_projections,
            image_quality={
                "laplacian_variance": laplacian,
                "tenengrad": tenengrad,
            },
            context_paths=context_paths,
        ),
        "prefilter_candidate",
    )


def _shortlist_prefilter_candidates(
    candidates: list[FramePrefilterCandidate],
    *,
    original_image_name: str,
    limit: int,
) -> list[FramePrefilterCandidate]:
    if limit <= 0 or not candidates:
        return []
    ordered = sorted(candidates, key=lambda item: _natural_frame_key(item.image_name))
    ordered.sort(key=lambda item: item.prefilter_score_key, reverse=True)
    for rank, candidate in enumerate(ordered, start=1):
        candidate.prefilter_rank = rank
    shortlisted = ordered[:limit]
    original = next(
        (item for item in ordered if item.image_name == original_image_name),
        None,
    )
    if original is not None and all(
        item.image_name != original.image_name for item in shortlisted
    ):
        if len(shortlisted) >= limit:
            shortlisted[-1] = original
        else:
            shortlisted.append(original)
    return shortlisted


def _evaluate_prefiltered_frame(
    candidate: FramePrefilterCandidate,
    *,
    prefilter_rank: int,
    prefilter_candidate_count: int,
    mesh_ray_shortlist_size: int,
    mesh_ray_surface_samples: int,
    mesh_ray_bbox_samples: int,
    mesh_ray_local_resamples: int,
) -> tuple[FrameCandidate | None, str]:
    source_metrics: dict[str, Any] = {}
    future_metrics: dict[str, Any] = {}
    source_ratios: list[float] = []
    future_ratios: list[float] = []
    area_ratios: list[float] = []
    margin_ratios: list[float] = []

    for obj_id in candidate.motion.source_required_ids:
        obj = candidate.context.objects_by_id[obj_id]
        visible, ratio, backend, visibility = _cached_static_visibility(
            obj,
            image_name=candidate.image_name,
            pose=candidate.pose,
            context=candidate.context,
            mesh_ray_surface_samples=mesh_ray_surface_samples,
            mesh_ray_bbox_samples=mesh_ray_bbox_samples,
            mesh_ray_local_resamples=mesh_ray_local_resamples,
        )
        if not visible:
            return None, "source_not_fully_visible"
        projection = candidate.source_projections[obj_id]
        source_ratios.append(ratio)
        area_ratios.append(projection["projected_area_ratio"])
        margin_ratios.append(projection["edge_margin_ratio"])
        source_metrics[str(obj_id)] = {
            "projection": projection,
            "visibility_backend": backend,
            "visible_ratio": ratio,
            "visibility": visibility,
        }

    for obj_id in candidate.motion.moved_ids:
        visible, ratio, visibility = _future_visibility(
            obj_id,
            pose=candidate.pose,
            context=candidate.context,
            motion=candidate.motion,
            mesh_ray_surface_samples=mesh_ray_surface_samples,
            mesh_ray_bbox_samples=mesh_ray_bbox_samples,
            mesh_ray_local_resamples=mesh_ray_local_resamples,
        )
        if not visible:
            return None, "future_not_fully_visible"
        projection = candidate.future_projections[obj_id]
        future_ratios.append(ratio)
        area_ratios.append(projection["projected_area_ratio"])
        margin_ratios.append(projection["edge_margin_ratio"])
        future_metrics[str(obj_id)] = {
            "projection": projection,
            "visibility_backend": "counterfactual_mesh_ray",
            "visible_ratio": ratio,
            "visibility": visibility,
        }

    original_bonus = int(
        candidate.image_name == str(candidate.question.get("image_name") or "")
    )
    score_key: tuple[Any, ...] = (
        min([*source_ratios, *future_ratios]),
        min(area_ratios),
        min(margin_ratios),
        candidate.image_quality["laplacian_variance"],
        candidate.image_quality["tenengrad"],
        original_bonus,
    )
    metrics = {
        "source": source_metrics,
        "future": future_metrics,
        "image_quality": candidate.image_quality,
        "prefilter": {
            "rank": prefilter_rank,
            "candidate_count": prefilter_candidate_count,
            "shortlist_limit": mesh_ray_shortlist_size,
            "score": list(candidate.prefilter_score_key),
        },
        "mesh_ray_budget": {
            "surface_samples": mesh_ray_surface_samples,
            "bbox_samples": mesh_ray_bbox_samples,
            "local_resamples": mesh_ray_local_resamples,
        },
        "score": list(score_key),
    }
    return (
        FrameCandidate(
            question=candidate.question,
            source_index=candidate.source_index,
            context=candidate.context,
            motion=candidate.motion,
            image_name=candidate.image_name,
            image_path=candidate.image_path,
            pose=candidate.pose,
            score_key=score_key,
            metrics=metrics,
            context_paths=candidate.context_paths,
        ),
        "selected_candidate",
    )


def _evaluate_frame(
    *,
    question: dict[str, Any],
    source_index: int,
    context: SceneContext,
    motion: QuestionMotion,
    image_name: str,
    context_paths: list[Path],
    mesh_ray_surface_samples: int = DEFAULT_MESH_RAY_SURFACE_SAMPLES,
    mesh_ray_bbox_samples: int = DEFAULT_MESH_RAY_BBOX_SAMPLES,
    mesh_ray_local_resamples: int = DEFAULT_MESH_RAY_LOCAL_RESAMPLES,
) -> tuple[FrameCandidate | None, str]:
    prefiltered, reason = _prefilter_frame(
        question=question,
        source_index=source_index,
        context=context,
        motion=motion,
        image_name=image_name,
        context_paths=context_paths,
    )
    if prefiltered is None:
        return None, reason
    return _evaluate_prefiltered_frame(
        prefiltered,
        prefilter_rank=1,
        prefilter_candidate_count=1,
        mesh_ray_shortlist_size=1,
        mesh_ray_surface_samples=mesh_ray_surface_samples,
        mesh_ray_bbox_samples=mesh_ray_bbox_samples,
        mesh_ray_local_resamples=mesh_ray_local_resamples,
    )


def _best_frame_for_question(
    *,
    question: dict[str, Any],
    source_index: int,
    context: SceneContext,
    frame_stride: int,
    mesh_ray_shortlist_size: int = DEFAULT_MESH_RAY_SHORTLIST_SIZE,
    mesh_ray_surface_samples: int = DEFAULT_MESH_RAY_SURFACE_SAMPLES,
    mesh_ray_bbox_samples: int = DEFAULT_MESH_RAY_BBOX_SAMPLES,
    mesh_ray_local_resamples: int = DEFAULT_MESH_RAY_LOCAL_RESAMPLES,
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

    reasons: Counter[str] = Counter()
    prefiltered_candidates: list[FramePrefilterCandidate] = []
    for image_name in _candidate_frame_names(question, context, frame_stride):
        candidate, reason = _prefilter_frame(
            question=question,
            source_index=source_index,
            context=context,
            motion=motion,
            image_name=image_name,
            context_paths=context_paths,
        )
        reasons[reason] += 1
        if candidate is not None:
            prefiltered_candidates.append(candidate)

    shortlisted = _shortlist_prefilter_candidates(
        prefiltered_candidates,
        original_image_name=str(question.get("image_name") or "").strip(),
        limit=mesh_ray_shortlist_size,
    )
    reasons["mesh_ray_shortlisted"] += len(shortlisted)
    best: FrameCandidate | None = None
    for prefiltered in shortlisted:
        candidate, reason = _evaluate_prefiltered_frame(
            prefiltered,
            prefilter_rank=prefiltered.prefilter_rank,
            prefilter_candidate_count=len(prefiltered_candidates),
            mesh_ray_shortlist_size=mesh_ray_shortlist_size,
            mesh_ray_surface_samples=mesh_ray_surface_samples,
            mesh_ray_bbox_samples=mesh_ray_bbox_samples,
            mesh_ray_local_resamples=mesh_ray_local_resamples,
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


def _candidate_record(
    candidate: FrameCandidate,
    *,
    dataset: str,
    scene_id: str,
) -> CandidateRecord:
    return CandidateRecord(
        question=candidate.question,
        source_index=candidate.source_index,
        dataset=dataset,
        scene_id=scene_id,
        image_name=candidate.image_name,
        score_key=candidate.score_key,
        metrics=candidate.metrics,
    )


def _eligible_checkpoint_record(candidate: CandidateRecord) -> dict[str, Any]:
    return {
        "kind": "question_result",
        "source_index": candidate.source_index,
        "question_uid": str(candidate.question.get("question_uid") or ""),
        "question_type": str(candidate.question.get("type") or ""),
        "dataset": candidate.dataset,
        "scene_id": candidate.scene_id,
        "status": "eligible",
        "candidate": {
            "image_name": candidate.image_name,
            "score_key": list(candidate.score_key),
            "metrics": candidate.metrics,
        },
    }


def _rejected_checkpoint_record(
    rejection: dict[str, Any],
    *,
    dataset: str,
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


def _checkpoint_record_matches_question(
    record: dict[str, Any],
    *,
    source_index: int,
    question: dict[str, Any],
) -> bool:
    return bool(
        int(record.get("source_index", -1)) == source_index
        and str(record.get("question_uid") or "")
        == str(question.get("question_uid") or "")
        and str(record.get("question_type") or "") == str(question.get("type") or "")
        and str(record.get("scene_id") or "") == str(question.get("scene_id") or "").strip()
    )


def _candidate_record_from_checkpoint(
    record: dict[str, Any],
    question: dict[str, Any],
) -> CandidateRecord:
    raw_candidate = record.get("candidate")
    if not isinstance(raw_candidate, dict):
        raise ValueError("eligible checkpoint record is missing candidate data")
    score_key = raw_candidate.get("score_key")
    metrics = raw_candidate.get("metrics")
    if not isinstance(score_key, list) or not isinstance(metrics, dict):
        raise ValueError("eligible checkpoint candidate data is malformed")
    return CandidateRecord(
        question=question,
        source_index=int(record["source_index"]),
        dataset=str(record["dataset"]),
        scene_id=str(record["scene_id"]),
        image_name=str(raw_candidate["image_name"]),
        score_key=tuple(score_key),
        metrics=metrics,
    )


def _rehydrate_candidate(
    candidate: CandidateRecord,
    context: SceneContext,
) -> FrameCandidate:
    motion = _apply_question_motion(
        candidate.question,
        context.objects,
        context.attachment_graph,
    )
    for obj_id in set((*motion.source_required_ids, *motion.moved_ids)):
        if obj_id not in context.objects_by_id:
            raise ValueError(f"object {obj_id} is absent from scene")
    if (
        bool(candidate.question.get("has_attachment_chain"))
        and len(motion.moved_ids) <= 1
    ):
        raise ValueError("question requires an attachment chain but none was found")
    if candidate.image_name not in context.poses:
        raise ValueError(
            f"selected frame {candidate.image_name!r} is absent from {candidate.scene_id}"
        )
    image_path = context.data_source.image_path(candidate.image_name).resolve()
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    return FrameCandidate(
        question=candidate.question,
        source_index=candidate.source_index,
        context=context,
        motion=motion,
        image_name=candidate.image_name,
        image_path=image_path,
        pose=context.poses[candidate.image_name],
        score_key=candidate.score_key,
        metrics=candidate.metrics,
        context_paths=_resolve_context_paths(candidate.question, context),
    )


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


def _materialize_candidate_records(
    candidates: list[CandidateRecord],
    *,
    output_dir: Path,
    scannet_root: Path | None,
    scannetpp_root: Path | None,
    scannetpp_frame_root: Path | None,
    scannetpp_sensor: str,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[CandidateRecord]] = defaultdict(list)
    for candidate in candidates:
        grouped[(candidate.dataset, candidate.scene_id)].append(candidate)

    materialized: dict[int, dict[str, Any]] = {}
    for (dataset, scene_id), scene_candidates in grouped.items():
        required_ids: set[int] = set()
        for candidate in scene_candidates:
            required_ids.update(_question_required_instance_ids(candidate.question))
        context = _load_scene_context(
            dataset=dataset,
            scene_id=scene_id,
            scannet_root=scannet_root,
            scannetpp_root=scannetpp_root,
            scannetpp_frame_root=scannetpp_frame_root,
            scannetpp_sensor=scannetpp_sensor,
            required_instance_ids=required_ids,
        )
        for candidate in scene_candidates:
            materialized[candidate.source_index] = _materialize_candidate(
                _rehydrate_candidate(candidate, context),
                output_dir,
            )
    return [materialized[candidate.source_index] for candidate in candidates]


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
    mesh_ray_shortlist_size: int = DEFAULT_MESH_RAY_SHORTLIST_SIZE,
    mesh_ray_surface_samples: int = DEFAULT_MESH_RAY_SURFACE_SAMPLES,
    mesh_ray_local_resamples: int = DEFAULT_MESH_RAY_LOCAL_RESAMPLES,
) -> SelectionPaths:
    if mesh_ray_shortlist_size <= 0:
        raise ValueError("mesh_ray_shortlist_size must be positive")
    if mesh_ray_surface_samples <= 0:
        raise ValueError("mesh_ray_surface_samples must be positive")
    if mesh_ray_local_resamples < 0:
        raise ValueError("mesh_ray_local_resamples must be non-negative")
    questions, _, _ = load_fixed_questions(benchmark_path)
    indexed_questions = [
        (index, question)
        for index, question in enumerate(questions)
        if str(question.get("type") or "") in L2_ROLLOUT_TYPES
    ]
    checkpoint_configuration = _checkpoint_configuration(
        benchmark_path=benchmark_path,
        scannet_root=scannet_root,
        scannetpp_root=scannetpp_root,
        scannetpp_frame_root=scannetpp_frame_root,
        scannetpp_sensor=scannetpp_sensor,
        frame_stride_scannet=frame_stride_scannet,
        frame_stride_scannetpp=frame_stride_scannetpp,
        mesh_ray_shortlist_size=mesh_ray_shortlist_size,
        mesh_ray_surface_samples=mesh_ray_surface_samples,
        mesh_ray_local_resamples=mesh_ray_local_resamples,
    )
    checkpoint_path = _checkpoint_path(output_dir, checkpoint_configuration)
    checkpoint_results = _load_or_create_checkpoint(
        checkpoint_path,
        checkpoint_configuration,
    )
    grouped: dict[tuple[str, str], list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for source_index, question in indexed_questions:
        dataset = _infer_dataset(question)
        scene_id = str(question.get("scene_id") or "").strip()
        grouped[(dataset, scene_id)].append((source_index, question))

    eligible: dict[str, list[CandidateRecord]] = defaultdict(list)
    rejected: list[dict[str, Any]] = []
    scene_errors: dict[str, str] = {}
    processed_indices: set[int] = set()
    for source_index, question in indexed_questions:
        checkpoint_record = checkpoint_results.get(source_index)
        if checkpoint_record is None or not _checkpoint_record_matches_question(
            checkpoint_record,
            source_index=source_index,
            question=question,
        ):
            continue
        try:
            if checkpoint_record.get("status") == "eligible":
                candidate = _candidate_record_from_checkpoint(checkpoint_record, question)
                eligible[str(question["type"])].append(candidate)
            elif checkpoint_record.get("status") == "rejected":
                rejection = checkpoint_record.get("rejection")
                if not isinstance(rejection, dict):
                    raise ValueError("rejected checkpoint record is missing rejection data")
                rejected.append(rejection)
                if rejection.get("reason") == "scene_resources_unavailable":
                    scene_errors[
                        f"{checkpoint_record.get('dataset')}:{checkpoint_record.get('scene_id')}"
                    ] = str(rejection.get("detail") or "")
            else:
                raise ValueError("checkpoint record has an unknown status")
        except (KeyError, TypeError, ValueError) as exc:
            print(
                f"[selection] ignoring invalid checkpoint result for source index "
                f"{source_index}: {exc}",
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
    for scene_index, ((dataset, scene_id), raw_scene_questions) in enumerate(
        scene_groups,
        start=1,
    ):
        pending_scene_questions = [
            (source_index, question)
            for source_index, question in raw_scene_questions
            if source_index not in processed_indices
        ]
        if not pending_scene_questions:
            continue
        progress.report_scene(
            scene_index=scene_index,
            scene_count=len(scene_groups),
            dataset=dataset,
            scene_id=scene_id,
            pending_questions=len(pending_scene_questions),
        )
        required_ids: set[int] = set()
        scene_questions: list[tuple[int, dict[str, Any]]] = []
        for source_index, question in pending_scene_questions:
            try:
                if not scene_id:
                    raise ValueError("scene_id must be non-empty")
                required_ids.update(_question_required_instance_ids(question))
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
                    checkpoint_path,
                    _rejected_checkpoint_record(rejection, dataset=dataset),
                )
                processed_indices.add(source_index)
                progress.advance(was_eligible=False)
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
                    checkpoint_path,
                    _rejected_checkpoint_record(rejection, dataset=dataset),
                )
                processed_indices.add(source_index)
                progress.advance(was_eligible=False)
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
                mesh_ray_shortlist_size=mesh_ray_shortlist_size,
                mesh_ray_surface_samples=mesh_ray_surface_samples,
                mesh_ray_bbox_samples=DEFAULT_MESH_RAY_BBOX_SAMPLES,
                mesh_ray_local_resamples=mesh_ray_local_resamples,
            )
            if best is None:
                rejection = {
                    "question_uid": question.get("question_uid"),
                    "source_index": source_index,
                    "question_type": question.get("type"),
                    "scene_id": scene_id,
                    "reason": "no_strict_single_frame",
                    "detail": detail,
                    "candidate_reasons": reason_counts,
                }
                rejected.append(rejection)
                _append_checkpoint_result(
                    checkpoint_path,
                    _rejected_checkpoint_record(rejection, dataset=dataset),
                )
                processed_indices.add(source_index)
                progress.advance(was_eligible=False)
                continue
            candidate = _candidate_record(best, dataset=dataset, scene_id=scene_id)
            eligible[str(question["type"])].append(candidate)
            _append_checkpoint_result(
                checkpoint_path,
                _eligible_checkpoint_record(candidate),
            )
            processed_indices.add(source_index)
            progress.advance(was_eligible=True)
        cache_stats.update(context.cache_stats)
        print(
            f"[selection] scene {dataset}:{scene_id} cache "
            f"quality={context.cache_stats['image_quality_hits']}/"
            f"{context.cache_stats['image_quality_misses']} hits/misses "
            f"static={context.cache_stats['static_visibility_hits']}/"
            f"{context.cache_stats['static_visibility_misses']} hits/misses",
            file=sys.stderr,
            flush=True,
        )

    selected: list[CandidateRecord] = []
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
    entries = _materialize_candidate_records(
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
    spec_payload = {
        "schema_version": SELECTION_SCHEMA_VERSION,
        "metadata": {
            "benchmark_path": str(benchmark_path.resolve()),
            "benchmark_sha256": _sha256_file(benchmark_path),
            "max_per_type": expected_per_type,
            "selection_mode": "strict_single_frame",
            "selection_algorithm_version": SELECTION_ALGORITHM_VERSION,
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
            "selection_pipeline": "geometry_prefilter_then_mesh_ray",
            "mesh_ray_shortlist_size": mesh_ray_shortlist_size,
            "mesh_ray_surface_samples": mesh_ray_surface_samples,
            "mesh_ray_bbox_samples": DEFAULT_MESH_RAY_BBOX_SAMPLES,
            "mesh_ray_local_resamples": mesh_ray_local_resamples,
            "fully_visible_ratio_min_exclusive": FULLY_VISIBLE_RATIO_MIN,
            "source_visibility": "registered_depth_then_mesh_ray",
            "future_visibility": "counterfactual_mesh_ray",
            "require_full_bbox": True,
            "require_default_edge_margin": True,
            "checkpoint_path": str(checkpoint_path.resolve()),
            "checkpoint_granularity": "question",
        },
        "cache_stats_current_run": dict(sorted(cache_stats.items())),
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
    parser.add_argument(
        "--mesh_ray_shortlist_size",
        type=int,
        default=DEFAULT_MESH_RAY_SHORTLIST_SIZE,
    )
    parser.add_argument(
        "--mesh_ray_surface_samples",
        type=int,
        default=DEFAULT_MESH_RAY_SURFACE_SAMPLES,
    )
    parser.add_argument(
        "--mesh_ray_local_resamples",
        type=int,
        default=DEFAULT_MESH_RAY_LOCAL_RESAMPLES,
    )
    args = parser.parse_args(argv)
    if not args.benchmark_file.is_file():
        parser.error(f"--benchmark_file not found: {args.benchmark_file}")
    if args.expected_per_type <= 0:
        parser.error("--expected_per_type must be positive")
    if args.frame_stride_scannet <= 0 or args.frame_stride_scannetpp <= 0:
        parser.error("frame strides must be positive")
    if args.mesh_ray_shortlist_size <= 0:
        parser.error("--mesh_ray_shortlist_size must be positive")
    if args.mesh_ray_surface_samples <= 0:
        parser.error("--mesh_ray_surface_samples must be positive")
    if args.mesh_ray_local_resamples < 0:
        parser.error("--mesh_ray_local_resamples must be non-negative")
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
        mesh_ray_shortlist_size=args.mesh_ray_shortlist_size,
        mesh_ray_surface_samples=args.mesh_ray_surface_samples,
        mesh_ray_local_resamples=args.mesh_ray_local_resamples,
    )
    print(f"selection_spec : {paths.spec}")
    print(f"selection_audit: {paths.audit}")


if __name__ == "__main__":
    main(sys.argv[1:])
