#!/usr/bin/env python3
"""Generate initial-state bird's-eye-view images for multi-image questions.

Example:
    python scripts/generate_bev_images.py \
        --benchmark output/benchmark_subset.json \
        --out_dir output/benchmark_subset_bev \
        --direction_mode none \
        --scannet_root data/scannet/scans \
        --scannetpp_root data/scannetpp \
        --scannetpp_root ++data
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter, MultipleLocator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.make_viewer import _collect_aux_image_names
from scripts.run_sampled_type_vlm_eval import (
    _load_benchmark,
    _load_oracle_scene_cache_entry,
    _question_uid,
    _sha256_file,
)
from src.utils.coordinate_transform import get_camera_forward, get_camera_right


BEV_SCHEMA_VERSION = "predictive-spatial-bev-v2"
BEV_IMAGE_SIZE_PX = 1600
BEV_DPI = 160
BEV_DIRECTION_MODES = ("none", "task", "task_ticks")
BEV_FRAME_KINDS = ("agent", "object_centric", "allocentric")


@dataclass(frozen=True)
class TaskDirection:
    frame_kind: str
    primary_label: str
    primary_xy: np.ndarray
    secondary_label: str
    secondary_xy: np.ndarray


def is_multi_image_question(question: dict[str, Any]) -> bool:
    return bool(_collect_aux_image_names(question))


def normalize_question(question: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(question)
    normalized["_dataset"] = str(
        normalized.get("_dataset") or normalized.get("dataset") or "unknown"
    )
    normalized["question_uid"] = _question_uid(normalized)
    return normalized


def _object_footprint(obj: dict[str, Any]) -> tuple[np.ndarray, np.ndarray] | None:
    try:
        bbox_min = np.asarray(obj["bbox_min"], dtype=np.float64)
        bbox_max = np.asarray(obj["bbox_max"], dtype=np.float64)
    except (KeyError, TypeError, ValueError):
        return None
    if (
        bbox_min.shape != (3,)
        or bbox_max.shape != (3,)
        or not np.all(np.isfinite(bbox_min))
        or not np.all(np.isfinite(bbox_max))
        or np.any(bbox_max < bbox_min)
    ):
        return None
    return bbox_min, bbox_max


def _object_center(obj: dict[str, Any]) -> np.ndarray:
    try:
        center = np.asarray(obj["center"], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("object has no valid center") from exc
    if center.shape != (3,) or not np.all(np.isfinite(center)):
        raise ValueError("object has no valid center")
    return center


def _object_id(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def bev_frame_kind(question: dict[str, Any]) -> str:
    qtype = str(question.get("type") or "").strip().lower()
    text = str(question.get("question") or "").lower()
    if "distance" in qtype or "occlusion" in qtype or "remove" in qtype:
        return "agent"
    if qtype.endswith("_object_centric"):
        return "object_centric"
    if qtype.endswith("_allocentric"):
        return "allocentric"
    if qtype.endswith("_agent"):
        return "agent"
    if qtype == "attachment_chain":
        return "allocentric"
    if qtype == "attachment_move":
        if question.get("camera_cardinal") or "cardinal direction" in text or "floor plan" in text:
            return "allocentric"
        if "imagine you are" in text or "from your perspective" in text:
            return "object_centric"
        return "agent"
    raise ValueError(f"unsupported BEV task frame for question type {qtype!r}")


def _normalize_xy(value: Any, *, description: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64).reshape(-1)
    if vector.size < 2 or not np.all(np.isfinite(vector[:2])):
        raise ValueError(f"invalid {description}")
    xy = vector[:2]
    norm = float(np.linalg.norm(xy))
    if norm <= 1e-8:
        raise ValueError(f"invalid floor-plane {description}")
    return xy / norm


def _object_frame_reference_ids(question: dict[str, Any]) -> tuple[int | None, int | None]:
    qtype = str(question.get("type") or "")
    text = str(question.get("question") or "").lower()
    query_id = _object_id(question.get("query_obj_id"))
    if qtype in {"object_move_object_centric", "object_rotate_object_centric", "attachment_move"} and query_id is not None:
        return query_id, _object_id(question.get("obj_face_id"))
    ref_id = _object_id(question.get("obj_ref_id"))
    if ("imagine you are" in text or "from your perspective" in text) and query_id is not None:
        ref_id = query_id
    return ref_id, _object_id(question.get("obj_face_id"))


def task_direction_for_question(
    question: dict[str, Any],
    objects: dict[int, dict[str, Any]],
    pose: Any | None,
) -> TaskDirection:
    frame_kind = bev_frame_kind(question)
    if frame_kind == "allocentric":
        return TaskDirection(
            frame_kind=frame_kind,
            primary_label="North",
            primary_xy=np.array([0.0, 1.0]),
            secondary_label="East",
            secondary_xy=np.array([1.0, 0.0]),
        )
    if frame_kind == "agent":
        if pose is None:
            raise ValueError(
                f"camera pose not found for frame {question.get('image_name')!r}"
            )
        return TaskDirection(
            frame_kind=frame_kind,
            primary_label="Front",
            primary_xy=_normalize_xy(get_camera_forward(pose), description="camera forward direction"),
            secondary_label="Right",
            secondary_xy=_normalize_xy(get_camera_right(pose), description="camera right direction"),
        )

    ref_id, face_id = _object_frame_reference_ids(question)
    if ref_id is None or ref_id not in objects:
        raise ValueError("object-centric question has no valid reference object")
    ref_center = np.asarray(
        question.get("facing_anchor_center") or _object_center(objects[ref_id]),
        dtype=np.float64,
    )
    if question.get("facing_target_center") is not None:
        face_center = np.asarray(question["facing_target_center"], dtype=np.float64)
    elif face_id is not None and face_id in objects:
        face_center = _object_center(objects[face_id])
    elif pose is not None:
        toward_camera = -_normalize_xy(
            get_camera_forward(pose), description="camera forward direction"
        )
        face_center = ref_center.copy()
        face_center[:2] += toward_camera
    else:
        raise ValueError("object-centric question has no valid facing target or camera pose")
    forward = _normalize_xy(face_center - ref_center, description="object-centric front direction")
    right = np.array([forward[1], -forward[0]], dtype=np.float64)
    return TaskDirection(
        frame_kind=frame_kind,
        primary_label="Front",
        primary_xy=forward,
        secondary_label="Right",
        secondary_xy=right,
    )


def _mentioned_objects(question: dict[str, Any]) -> list[dict[str, Any]]:
    mentions: list[dict[str, Any]] = []
    seen: set[int] = set()
    for raw in question.get("mentioned_objects") or []:
        if not isinstance(raw, dict):
            continue
        try:
            obj_id = int(raw.get("obj_id"))
        except (TypeError, ValueError):
            continue
        if obj_id in seen:
            continue
        mentions.append({**raw, "obj_id": obj_id})
        seen.add(obj_id)
    return mentions


def _nice_tick_step(span: float) -> float:
    target = max(float(span), 1e-6) / 6.0
    magnitude = 10.0 ** np.floor(np.log10(target))
    normalized = target / magnitude
    for candidate in (1.0, 2.0, 2.5, 5.0, 10.0):
        if normalized <= candidate:
            return float(candidate * magnitude)
    return float(10.0 * magnitude)


def _format_metric_tick(value: float, _position: int) -> str:
    if abs(value) < 5e-10:
        value = 0.0
    return f"{value:.2f}".rstrip("0").rstrip(".")


def _draw_task_direction(axis: Any, direction: TaskDirection) -> None:
    compass = axis.inset_axes([0.76, 0.75, 0.20, 0.20])
    compass.set_facecolor((1.0, 1.0, 1.0, 0.88))
    for label, vector, color in (
        (direction.primary_label, direction.primary_xy, "#202124"),
        (direction.secondary_label, direction.secondary_xy, "#a33a2b"),
    ):
        compass.annotate(
            "",
            xy=vector * 0.82,
            xytext=(0.0, 0.0),
            arrowprops={"arrowstyle": "-|>", "color": color, "lw": 2.3},
        )
        compass.text(
            float(vector[0] * 1.02),
            float(vector[1] * 1.02),
            label,
            color=color,
            fontsize=9,
            fontweight="bold",
            horizontalalignment="center",
            verticalalignment="center",
        )
    compass.set_xlim(-1.25, 1.25)
    compass.set_ylim(-1.25, 1.25)
    compass.set_aspect("equal", adjustable="box")
    compass.set_axis_off()


def render_bev_image(
    *,
    question: dict[str, Any],
    objects: dict[int, dict[str, Any]],
    output_path: Path,
    direction_mode: str = "none",
    task_direction: TaskDirection | None = None,
) -> None:
    if direction_mode not in BEV_DIRECTION_MODES:
        raise ValueError(f"unsupported BEV direction mode: {direction_mode!r}")
    if direction_mode != "none" and task_direction is None:
        raise ValueError(f"{direction_mode} requires a task direction")
    mentions = _mentioned_objects(question)
    colors = plt.get_cmap("tab10")

    figure, axis = plt.subplots(
        figsize=(BEV_IMAGE_SIZE_PX / BEV_DPI, BEV_IMAGE_SIZE_PX / BEV_DPI),
        dpi=BEV_DPI,
    )
    figure.subplots_adjust(left=0.12, right=0.95, bottom=0.11, top=0.88)
    extent_points: list[np.ndarray] = []

    for index, mention in enumerate(mentions):
        obj = objects.get(int(mention["obj_id"]))
        if obj is None:
            continue
        footprint = _object_footprint(obj)
        if footprint is None:
            continue
        bbox_min, bbox_max = footprint
        width, height = bbox_max[:2] - bbox_min[:2]
        color = colors(index % 10)
        axis.add_patch(
            Rectangle(
                bbox_min[:2],
                float(width),
                float(height),
                facecolor=(*color[:3], 0.32),
                edgecolor=color,
                linewidth=2.5,
                zorder=4,
            )
        )
        extent_points.extend((bbox_min[:2], bbox_max[:2]))
        center_xy = (bbox_min[:2] + bbox_max[:2]) / 2.0
        label = str(mention.get("label") or obj.get("label") or "object")
        axis.annotate(
            label,
            xy=center_xy,
            xytext=(0, 7),
            textcoords="offset points",
            horizontalalignment="center",
            fontsize=9,
            fontweight="bold",
            color=color,
            bbox={
                "boxstyle": "round,pad=0.25",
                "fc": "white",
                "ec": color,
                "alpha": 0.88,
            },
            zorder=10,
        )

    if not extent_points:
        raise ValueError("question contains no mentioned objects with plottable footprints")
    extent = np.asarray(extent_points, dtype=np.float64)
    minimum = np.min(extent, axis=0)
    maximum = np.max(extent, axis=0)
    span = np.maximum(maximum - minimum, 1.0)
    margin = np.maximum(span * 0.08, 0.5)
    axis.set_xlim(minimum[0] - margin[0], maximum[0] + margin[0])
    axis.set_ylim(minimum[1] - margin[1], maximum[1] + margin[1])
    axis.set_aspect("equal", adjustable="box")
    axis.set_title(
        "Initial layout before the hypothetical operation\n"
        f"{question.get('scene_id', '')} · {question.get('type', '')}",
        fontsize=15,
        fontweight="bold",
    )
    if direction_mode == "task_ticks":
        plotted_span = maximum - minimum + 2.0 * margin
        tick_step = _nice_tick_step(float(np.max(plotted_span)))
        axis.xaxis.set_major_locator(MultipleLocator(tick_step))
        axis.yaxis.set_major_locator(MultipleLocator(tick_step))
        formatter = FuncFormatter(_format_metric_tick)
        axis.xaxis.set_major_formatter(formatter)
        axis.yaxis.set_major_formatter(formatter)
        axis.set_xlabel("World X (meters)")
        axis.set_ylabel("World Y (meters)")
        axis.tick_params(axis="both", which="major", labelsize=9)
        axis.grid(True, linestyle=":", linewidth=0.7, alpha=0.5)
    else:
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_xlabel("")
        axis.set_ylabel("")
        axis.grid(False)
        for spine in axis.spines.values():
            spine.set_visible(False)
    if task_direction is not None:
        _draw_task_direction(axis, task_direction)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=BEV_DPI, facecolor="white")
    plt.close(figure)


def generate_bev_images(args: argparse.Namespace) -> dict[str, Any]:
    benchmark_path = Path(args.benchmark)
    direction_mode = str(args.direction_mode)
    if direction_mode not in BEV_DIRECTION_MODES:
        raise ValueError(f"unsupported BEV direction mode: {direction_mode!r}")
    questions = [normalize_question(question) for question in _load_benchmark(benchmark_path)]
    multi_image_questions = [question for question in questions if is_multi_image_question(question)]
    if not multi_image_questions:
        raise ValueError(f"No multi-image questions found in {benchmark_path}")

    frame_kinds = {
        str(question["question_uid"]): bev_frame_kind(question)
        for question in multi_image_questions
    }
    pose_scene_keys = {
        (
            str(question.get("_dataset") or "unknown"),
            str(question.get("scene_id") or ""),
        )
        for question in multi_image_questions
        if direction_mode != "none"
        and frame_kinds[str(question["question_uid"])] != "allocentric"
    }

    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scene_cache: dict[tuple[str, str], Any] = {}
    entries: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []

    for index, question in enumerate(multi_image_questions, 1):
        uid = str(question["question_uid"])
        scene_id = str(question.get("scene_id") or "")
        dataset = str(question.get("_dataset") or "unknown")
        cache_key = (dataset, scene_id)
        try:
            if cache_key not in scene_cache:
                scene_cache[cache_key] = _load_oracle_scene_cache_entry(
                    scene_id,
                    dataset,
                    scannet_root=str(args.scannet_root),
                    scannetpp_roots=[str(root) for root in args.scannetpp_root],
                    scannetpp_sensor=args.scannetpp_sensor,
                    need_poses=cache_key in pose_scene_keys,
                    oracle_cache_dir=args.oracle_cache_dir,
                    scannetpp_root_option="--scannetpp_root",
                )
            scene = scene_cache[cache_key]
            pose = None
            poses = getattr(scene, "poses", None)
            if poses is not None:
                pose = poses.get(str(question.get("image_name") or ""))
            task_direction = (
                task_direction_for_question(question, scene.objects, pose)
                if direction_mode != "none"
                else None
            )
            image_name = f"{uid}.png"
            render_bev_image(
                question=question,
                objects=scene.objects,
                output_path=output_dir / image_name,
                direction_mode=direction_mode,
                task_direction=task_direction,
            )
            image_sha256 = _sha256_file(output_dir / image_name)
            entries.append(
                {
                    "question_uid": uid,
                    "dataset": dataset,
                    "scene_id": scene_id,
                    "question_type": question.get("type"),
                    "task_frame_kind": frame_kinds[uid],
                    "image_path": image_name,
                    "image_sha256": image_sha256,
                    "frame_names": [
                        str(question.get("image_name") or ""),
                        *_collect_aux_image_names(question),
                    ],
                }
            )
            print(f"[{index}/{len(multi_image_questions)}] {scene_id} -> {image_name}")
        except Exception as exc:
            failures.append({"question_uid": uid, "scene_id": scene_id, "error": str(exc)})
            print(f"[{index}/{len(multi_image_questions)}] ERROR {scene_id}/{uid}: {exc}", file=sys.stderr)

    manifest = {
        "schema_version": BEV_SCHEMA_VERSION,
        "source_benchmark": str(benchmark_path.resolve()),
        "source_benchmark_sha256": _sha256_file(benchmark_path),
        "coordinate_frame": "world_xy_x_east_y_north_z_up",
        "state": "before_hypothetical_operation",
        "direction_mode": direction_mode,
        "image_size_px": BEV_IMAGE_SIZE_PX,
        "input_question_count": len(questions),
        "multi_image_question_count": len(multi_image_questions),
        "generated_count": len(entries),
        "failure_count": len(failures),
        "entries": entries,
        "failures": failures,
    }
    manifest_path = output_dir / "bev_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    if failures:
        raise RuntimeError(
            f"BEV generation failed for {len(failures)} question(s); see {manifest_path}"
        )
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True, help="Question JSON to process")
    parser.add_argument("--out_dir", required=True, help="Directory for PNG files and bev_manifest.json")
    parser.add_argument(
        "--direction_mode",
        required=True,
        choices=BEV_DIRECTION_MODES,
        help="Direction annotation condition for this BEV output directory",
    )
    parser.add_argument("--scannet_root", default="data/scannet/scans", help="Raw ScanNet scans root")
    parser.add_argument(
        "--scannetpp_root",
        action="append",
        default=None,
        help="Raw ScanNet++ geometry root; repeatable (defaults: data/scannetpp and ++data)",
    )
    parser.add_argument("--scannetpp_sensor", choices=("iphone", "dslr"), default="iphone")
    parser.add_argument("--oracle_cache_dir", default=None, help="Optional precomputed scene cache directory")
    args = parser.parse_args(argv)
    if not Path(args.benchmark).is_file():
        parser.error(f"--benchmark not found: {args.benchmark}")
    if args.scannetpp_root is None:
        args.scannetpp_root = ["data/scannetpp", "++data"]
    return args


def main() -> None:
    args = parse_args()
    try:
        manifest = generate_bev_images(args)
    except Exception as exc:
        raise SystemExit(str(exc)) from exc
    print(
        f"Generated {manifest['generated_count']} BEV image(s) in {Path(args.out_dir).resolve()}"
    )


if __name__ == "__main__":
    main()
