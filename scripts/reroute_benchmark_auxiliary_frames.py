#!/usr/bin/env python3
"""Recompute cross-frame auxiliary images without camera-motion hard limits.

The first and last main images stay fixed so camera-bound question semantics and
answers remain valid. Only ``auxiliary_image_names`` and ``auxiliary_route`` are
updated for questions that can be rerouted successfully.
"""

from __future__ import annotations

import argparse
import copy
from collections import Counter, defaultdict
from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any, Callable

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.auxiliary_path import MAX_AUXILIARY_FRAMES
from src.datasets import make_data_source
from src.datasets.base import SceneDataSource
from src.datasets.scannetpp_depth import DEFAULT_DEPTH_CACHE_SIZE
from src.depth_auxiliary_path import (
    DEFAULT_MAX_CANDIDATE_POSES,
    DepthCorridorAuxiliaryRoute,
    DepthRouteGeometryCache,
    DepthVisualRedundancyEvaluator,
    find_depth_corridor_auxiliary_route,
)
from src.hybrid_auxiliary_path import HybridAuxiliaryRouter
from src.legacy_auxiliary_path import object_group_center


logger = logging.getLogger("reroute_benchmark_auxiliary_frames")
_SCANNET_SCENE_RE = re.compile(r"^scene\d{4}_\d{2}$")
_DISABLED_HARD_LIMITS = (
    "forward_angle_deg",
    "height_change_m",
    "local_perpendicular_m",
    "global_perpendicular_m",
    "degenerate_xy_translation_m",
)


@dataclass(frozen=True)
class RerouteConfig:
    scannet_root: Path | None
    scannetpp_root: Path | None
    scannetpp_frame_root: Path | None
    max_auxiliary_frames: int = MAX_AUXILIARY_FRAMES
    max_candidate_poses: int | None = DEFAULT_MAX_CANDIDATE_POSES
    depth_cache_size: int = DEFAULT_DEPTH_CACHE_SIZE


@dataclass
class SceneRoutingResources:
    data_source: SceneDataSource
    objects_by_id: dict[int, dict[str, Any]]
    poses: dict[str, Any]
    intrinsics: Any
    geometry_cache: DepthRouteGeometryCache
    visual_redundancy: DepthVisualRedundancyEvaluator


def _dataset_for_scene(scene_id: str) -> str:
    return "scannet" if _SCANNET_SCENE_RE.fullmatch(scene_id) else "scannetpp"


def _is_cross_frame_question(question: dict[str, Any]) -> bool:
    return bool(str(question.get("reasoning_frame_2") or "").strip())


def _question_groups(
    question: dict[str, Any]
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    raw_groups = question.get("object_frame_groups")
    if not isinstance(raw_groups, dict):
        raise ValueError("missing object_frame_groups")

    def normalize(name: str) -> tuple[int, ...]:
        raw_ids = raw_groups.get(name)
        if not isinstance(raw_ids, (list, tuple)) or not raw_ids:
            raise ValueError(f"object_frame_groups.{name} must be a non-empty list")
        try:
            ids = tuple(dict.fromkeys(int(value) for value in raw_ids))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"object_frame_groups.{name} contains a non-integer id"
            ) from exc
        if not ids:
            raise ValueError(f"object_frame_groups.{name} must not be empty")
        return ids

    frame_1_ids = normalize("frame_1")
    frame_2_ids = normalize("frame_2")
    if set(frame_1_ids) & set(frame_2_ids):
        raise ValueError("frame_1 and frame_2 object groups overlap")
    return frame_1_ids, frame_2_ids


def _load_scene_resources(
    scene_id: str, config: RerouteConfig
) -> SceneRoutingResources:
    dataset = _dataset_for_scene(scene_id)
    if dataset == "scannet":
        if config.scannet_root is None:
            raise ValueError("--scannet_root is required for ScanNet questions")
        scene_dir = config.scannet_root / scene_id
        data_source = make_data_source("scannet", scene_dir)
    else:
        if config.scannetpp_root is None:
            raise ValueError("--scannetpp_root is required for ScanNet++ questions")
        if config.scannetpp_frame_root is None:
            raise ValueError(
                "--scannetpp_frame_root is required for ScanNet++ RGB frames"
            )
        scene_dir = config.scannetpp_root / scene_id
        data_source = make_data_source(
            "scannetpp",
            scene_dir,
            sensor="iphone",
            frame_root=config.scannetpp_frame_root,
            depth_cache_size=config.depth_cache_size,
        )
    if not scene_dir.is_dir():
        raise FileNotFoundError(f"scene directory not found: {scene_dir}")

    scene = data_source.load_scene()
    objects_by_id = {
        int(obj["id"]): obj
        for obj in scene.get("objects", [])
        if isinstance(obj, dict) and obj.get("id") is not None
    }
    poses = data_source.load_poses()
    intrinsics = data_source.load_intrinsics()
    if not objects_by_id:
        raise ValueError(f"scene {scene_id} contains no objects")
    if not poses:
        raise ValueError(f"scene {scene_id} contains no camera poses")

    visual_router = HybridAuxiliaryRouter(
        poses=poses,
        intrinsics=intrinsics,
        image_path_for=data_source.image_path,
    )
    visual_redundancy = DepthVisualRedundancyEvaluator(
        poses=poses,
        depth_frame_for=data_source.load_depth_frame,
        rgb_evidence_for=visual_router.visual_continuity,
    )
    return SceneRoutingResources(
        data_source=data_source,
        objects_by_id=objects_by_id,
        poses=poses,
        intrinsics=intrinsics,
        geometry_cache=DepthRouteGeometryCache(),
        visual_redundancy=visual_redundancy,
    )


def _objects_for_ids(
    objects_by_id: dict[int, dict[str, Any]],
    object_ids: tuple[int, ...],
) -> list[dict[str, Any]]:
    missing = [obj_id for obj_id in object_ids if obj_id not in objects_by_id]
    if missing:
        raise ValueError(f"scene metadata is missing object ids: {missing}")
    return [objects_by_id[obj_id] for obj_id in object_ids]


def _route_metadata(route: DepthCorridorAuxiliaryRoute) -> dict[str, Any]:
    return {
        "method": "depth_corridor_geometric",
        "search_method": route.search_method,
        "edge_count": route.edge_count,
        "cost": route.cost,
        "route_sample_count": route.route_sample_count,
        "frame_a_coverage_end": route.frame_a_coverage_end,
        "frame_b_coverage_start": route.frame_b_coverage_start,
        "auxiliary_responsibility_fraction": route.auxiliary_responsibility_fraction,
        "transition_overlap_fraction": route.transition_overlap_fraction,
        "min_progress_fraction": route.min_progress_fraction,
        "min_depth_valid_fraction": route.min_depth_valid_fraction,
        "min_depth_visible_fraction": route.min_depth_visible_fraction,
        "max_local_perpendicular_m": route.max_local_perpendicular_m,
        "max_global_perpendicular_m": route.max_global_perpendicular_m,
        "max_height_change_m": route.max_height_change_m,
        "max_parallel_change_m": route.max_parallel_change_m,
        "max_forward_angle_deg": route.max_forward_angle_deg,
        "depth_sources": list(route.depth_sources),
        "pre_prune_auxiliary_count": route.pre_prune_auxiliary_count,
        "pruned_auxiliary_frame_count": route.pruned_auxiliary_frame_count,
        "visual_pruned_auxiliary_frame_count": (
            route.visual_pruned_auxiliary_frame_count
        ),
        "visual_duplicate_candidate_count": route.visual_duplicate_candidate_count,
        "visual_prune_relaxed_angle_edge_count": (
            route.visual_prune_relaxed_angle_edge_count
        ),
        "visual_redundancy_metric_version": route.visual_redundancy_metric_version,
        "semantic_rejected_frame_count": route.semantic_rejected_frame_count,
        "camera_motion_hard_limits_enabled": False,
        "disabled_camera_motion_hard_limits": list(_DISABLED_HARD_LIMITS),
    }


def _failure_record(
    *,
    question_index: int,
    question: dict[str, Any],
    reason: str,
) -> dict[str, Any]:
    return {
        "question_index": question_index,
        "question_uid": question.get("question_uid"),
        "scene_id": question.get("scene_id"),
        "type": question.get("type"),
        "image_name": question.get("image_name"),
        "reasoning_frame_2": question.get("reasoning_frame_2"),
        "reason": reason,
        "original_auxiliary_image_names": copy.deepcopy(
            question.get("auxiliary_image_names")
        ),
    }


def _increment_outcome(
    counters: dict[str, Counter[str]], dataset: str, question_type: str, outcome: str
) -> None:
    counters[f"dataset:{dataset}"][outcome] += 1
    counters[f"type:{question_type}"][outcome] += 1


def reroute_payload(
    payload: dict[str, Any],
    config: RerouteConfig,
    *,
    resource_loader: Callable[[str, RerouteConfig], SceneRoutingResources] = (
        _load_scene_resources
    ),
) -> tuple[dict[str, Any], dict[str, Any]]:
    questions = payload.get("questions")
    if not isinstance(questions, list) or not all(
        isinstance(question, dict) for question in questions
    ):
        raise ValueError("input JSON must contain a 'questions' list of objects")

    output_payload = copy.deepcopy(payload)
    output_questions: list[dict[str, Any]] = output_payload["questions"]
    questions_by_scene: dict[str, list[int]] = defaultdict(list)
    skipped_single = 0
    invalid_without_scene: list[int] = []
    for index, question in enumerate(output_questions):
        if not _is_cross_frame_question(question):
            skipped_single += 1
            continue
        scene_id = str(question.get("scene_id") or "").strip()
        if not scene_id:
            invalid_without_scene.append(index)
            continue
        questions_by_scene[scene_id].append(index)

    failures: list[dict[str, Any]] = []
    changes: list[dict[str, Any]] = []
    outcome_counts: dict[str, Counter[str]] = defaultdict(Counter)
    succeeded = 0
    changed = 0
    unchanged = 0
    route_cache_hits = 0

    for index in invalid_without_scene:
        question = output_questions[index]
        failures.append(
            _failure_record(
                question_index=index,
                question=question,
                reason="missing scene_id",
            )
        )
        _increment_outcome(
            outcome_counts,
            "unknown",
            str(question.get("type") or "unknown"),
            "failed",
        )

    for scene_id in sorted(questions_by_scene):
        indices = questions_by_scene[scene_id]
        dataset = _dataset_for_scene(scene_id)
        logger.info(
            "Loading %s scene %s for %d question(s)", dataset, scene_id, len(indices)
        )
        try:
            resources = resource_loader(scene_id, config)
        except Exception as exc:
            reason = f"scene_resource_error: {type(exc).__name__}: {exc}"
            logger.warning("%s: %s", scene_id, reason)
            for index in indices:
                question = output_questions[index]
                failures.append(
                    _failure_record(
                        question_index=index,
                        question=question,
                        reason=reason,
                    )
                )
                _increment_outcome(
                    outcome_counts,
                    dataset,
                    str(question.get("type") or "unknown"),
                    "failed",
                )
            continue

        route_cache: dict[tuple[object, ...], DepthCorridorAuxiliaryRoute | None] = {}
        route_errors: dict[tuple[object, ...], str] = {}
        for index in indices:
            question = output_questions[index]
            question_type = str(question.get("type") or "unknown")
            try:
                frame_1_name = str(question.get("image_name") or "").strip()
                frame_2_name = str(question.get("reasoning_frame_2") or "").strip()
                if not frame_1_name or not frame_2_name:
                    raise ValueError("missing first or last main image")
                if frame_1_name == frame_2_name:
                    raise ValueError("first and last main images must differ")
                if frame_1_name not in resources.poses:
                    raise ValueError(
                        f"missing pose for first main image {frame_1_name}"
                    )
                if frame_2_name not in resources.poses:
                    raise ValueError(f"missing pose for last main image {frame_2_name}")
                frame_1_ids, frame_2_ids = _question_groups(question)
                group_a = _objects_for_ids(resources.objects_by_id, frame_1_ids)
                group_b = _objects_for_ids(resources.objects_by_id, frame_2_ids)
                cache_key = (
                    frame_1_name,
                    frame_2_name,
                    frame_1_ids,
                    frame_2_ids,
                    config.max_auxiliary_frames,
                    config.max_candidate_poses,
                )
                if cache_key in route_cache:
                    route_cache_hits += 1
                    route = route_cache[cache_key]
                else:
                    try:
                        route = find_depth_corridor_auxiliary_route(
                            center_a=object_group_center(group_a),
                            center_b=object_group_center(group_b),
                            frame_a_name=frame_1_name,
                            frame_b_name=frame_2_name,
                            poses=resources.poses,
                            intrinsics=resources.intrinsics,
                            depth_frame_for=resources.data_source.load_depth_frame,
                            group_a_objects=group_a,
                            group_b_objects=group_b,
                            visual_redundancy_for=resources.visual_redundancy,
                            geometry_cache=resources.geometry_cache,
                            max_auxiliary_frames=config.max_auxiliary_frames,
                            max_candidate_poses=config.max_candidate_poses,
                            enforce_camera_motion_hard_limits=False,
                        )
                    except Exception as exc:
                        route = None
                        route_errors[cache_key] = (
                            f"route_error: {type(exc).__name__}: {exc}"
                        )
                    route_cache[cache_key] = route
                if route is None:
                    raise RuntimeError(
                        route_errors.get(cache_key, "no_relaxed_depth_corridor_route")
                    )

                old_auxiliary = copy.deepcopy(question.get("auxiliary_image_names"))
                new_auxiliary = list(route.auxiliary_image_names)
                question["auxiliary_image_names"] = new_auxiliary
                question["auxiliary_route"] = _route_metadata(route)
                selection_changed = old_auxiliary != new_auxiliary
                succeeded += 1
                changed += int(selection_changed)
                unchanged += int(not selection_changed)
                _increment_outcome(outcome_counts, dataset, question_type, "succeeded")
                changes.append(
                    {
                        "question_index": index,
                        "question_uid": question.get("question_uid"),
                        "scene_id": scene_id,
                        "type": question_type,
                        "selection_changed": selection_changed,
                        "original_auxiliary_image_names": old_auxiliary,
                        "new_auxiliary_image_names": new_auxiliary,
                        "new_route_cost": route.cost,
                    }
                )
            except Exception as exc:
                reason = str(exc) or type(exc).__name__
                failures.append(
                    _failure_record(
                        question_index=index,
                        question=question,
                        reason=reason,
                    )
                )
                _increment_outcome(outcome_counts, dataset, question_type, "failed")

    def grouped_counts(prefix: str) -> dict[str, dict[str, int]]:
        return {
            key.removeprefix(prefix): dict(sorted(counter.items()))
            for key, counter in sorted(outcome_counts.items())
            if key.startswith(prefix)
        }

    report = {
        "schema_version": 1,
        "policy": {
            "main_frames_fixed": True,
            "camera_motion_hard_limits_enabled": False,
            "disabled_camera_motion_hard_limits": list(_DISABLED_HARD_LIMITS),
            "camera_motion_soft_costs_enabled": True,
            "max_auxiliary_frames": config.max_auxiliary_frames,
            "max_candidate_poses": config.max_candidate_poses,
            "failure_policy": "keep_original_selection",
        },
        "summary": {
            "total_question_count": len(output_questions),
            "cross_frame_question_count": len(output_questions) - skipped_single,
            "single_frame_question_count": skipped_single,
            "reroute_succeeded_count": succeeded,
            "auxiliary_selection_changed_count": changed,
            "auxiliary_selection_unchanged_count": unchanged,
            "reroute_failed_count": len(failures),
            "route_cache_hit_count": route_cache_hits,
        },
        "by_dataset": grouped_counts("dataset:"),
        "by_type": grouped_counts("type:"),
        "failures": failures,
        "changes": changes,
    }
    return output_payload, report


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary_path, path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def _default_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}.relaxed_camera_edges.json")


def _default_report_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}.report.json")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recompute auxiliary images while keeping each question's main "
            "frames fixed and disabling camera-motion hard limits."
        )
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument(
        "--scannet_root",
        type=Path,
        default=None,
        help="Root containing ScanNet sceneXXXX_XX directories.",
    )
    parser.add_argument(
        "--scannetpp_root",
        type=Path,
        default=None,
        help="Root containing ScanNet++ scene-id directories.",
    )
    parser.add_argument(
        "--scannetpp_frame_root",
        type=Path,
        default=None,
        help="Root containing extracted ScanNet++ iPhone JPEG frames.",
    )
    parser.add_argument(
        "--max_auxiliary_frames", type=int, default=MAX_AUXILIARY_FRAMES
    )
    parser.add_argument(
        "--max_candidate_poses",
        type=int,
        default=DEFAULT_MAX_CANDIDATE_POSES,
        help="Maximum depth-tested pose candidates per route; 0 keeps all.",
    )
    parser.add_argument(
        "--depth_cache_size", type=int, default=DEFAULT_DEPTH_CACHE_SIZE
    )
    parser.add_argument(
        "--force", action="store_true", help="Overwrite existing output/report files."
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args(argv)
    input_path = args.input.resolve()
    output_path = (args.output or _default_output_path(input_path)).resolve()
    report_path = (args.report or _default_report_path(output_path)).resolve()
    if not input_path.is_file():
        raise SystemExit(f"Input JSON does not exist: {input_path}")
    if output_path == input_path or report_path == input_path:
        raise SystemExit("Refusing to overwrite the input JSON")
    if output_path == report_path:
        raise SystemExit("--output and --report must be different paths")
    for path in (output_path, report_path):
        if path.exists() and not args.force:
            raise SystemExit(
                f"Output already exists; pass --force to overwrite: {path}"
            )
    if args.max_auxiliary_frames < 0:
        raise SystemExit("--max_auxiliary_frames must be non-negative")
    if args.max_candidate_poses < 0:
        raise SystemExit("--max_candidate_poses must be non-negative")
    if args.depth_cache_size <= 0:
        raise SystemExit("--depth_cache_size must be positive")

    try:
        payload = json.loads(input_path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Could not read input JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit("Input JSON must be an object containing a questions list")
    questions = payload.get("questions")
    if not isinstance(questions, list):
        raise SystemExit("Input JSON must contain a questions list")

    target_datasets = {
        _dataset_for_scene(str(question.get("scene_id") or ""))
        for question in questions
        if isinstance(question, dict)
        and _is_cross_frame_question(question)
        and str(question.get("scene_id") or "").strip()
    }
    if "scannet" in target_datasets and args.scannet_root is None:
        raise SystemExit("--scannet_root is required by ScanNet questions")
    if "scannetpp" in target_datasets and args.scannetpp_root is None:
        raise SystemExit("--scannetpp_root is required by ScanNet++ questions")
    if "scannetpp" in target_datasets and args.scannetpp_frame_root is None:
        raise SystemExit("--scannetpp_frame_root is required by ScanNet++ questions")

    config = RerouteConfig(
        scannet_root=args.scannet_root.resolve() if args.scannet_root else None,
        scannetpp_root=args.scannetpp_root.resolve() if args.scannetpp_root else None,
        scannetpp_frame_root=(
            args.scannetpp_frame_root.resolve() if args.scannetpp_frame_root else None
        ),
        max_auxiliary_frames=args.max_auxiliary_frames,
        max_candidate_poses=(
            None if args.max_candidate_poses == 0 else args.max_candidate_poses
        ),
        depth_cache_size=args.depth_cache_size,
    )
    output_payload, report = reroute_payload(payload, config)
    report["input_path"] = str(input_path)
    report["output_path"] = str(output_path)
    report["report_path"] = str(report_path)
    _atomic_write_json(output_path, output_payload)
    _atomic_write_json(report_path, report)
    summary = report["summary"]
    logger.info(
        "Wrote %s: %d succeeded, %d changed, %d failed",
        output_path,
        summary["reroute_succeeded_count"],
        summary["auxiliary_selection_changed_count"],
        summary["reroute_failed_count"],
    )
    logger.info("Wrote report %s", report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
