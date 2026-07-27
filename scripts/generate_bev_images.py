#!/usr/bin/env python3
"""Generate initial-state bird's-eye-view images for multi-image questions.

Example:
    python scripts/generate_bev_images.py \
        --benchmark output/benchmark_subset.json \
        --out_dir output/benchmark_subset_bev \
        --scannet_root data/scannet/scans \
        --scannetpp_root data/scannetpp \
        --scannetpp_root ++data
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

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


BEV_SCHEMA_VERSION = "predictive-spatial-bev-v1"
BEV_IMAGE_SIZE_PX = 1600
BEV_DPI = 160


def is_multi_image_question(question: dict[str, Any]) -> bool:
    return bool(_collect_aux_image_names(question))


def normalize_question(question: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(question)
    normalized["_dataset"] = str(
        normalized.get("_dataset") or normalized.get("dataset") or "unknown"
    )
    normalized["question_uid"] = _question_uid(normalized)
    return normalized


def _object_center(obj: dict[str, Any]) -> np.ndarray | None:
    try:
        center = np.asarray(obj["center"], dtype=np.float64)
    except (KeyError, TypeError, ValueError):
        return None
    if center.shape != (3,) or not np.all(np.isfinite(center)):
        return None
    return center


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


def _camera_forward_xy(pose: Any) -> np.ndarray | None:
    try:
        forward = np.asarray(pose.rotation, dtype=np.float64).T[:, 2]
    except (AttributeError, TypeError, ValueError, IndexError):
        return None
    forward_xy = forward[:2]
    norm = float(np.linalg.norm(forward_xy))
    if not math.isfinite(norm) or norm <= 1e-9:
        return None
    return forward_xy / norm


def render_bev_image(
    *,
    question: dict[str, Any],
    objects: dict[int, dict[str, Any]],
    poses: dict[str, Any],
    output_path: Path,
) -> None:
    mentions = _mentioned_objects(question)
    mentioned_by_id = {int(mention["obj_id"]): mention for mention in mentions}
    colors = plt.get_cmap("tab10")

    figure, axis = plt.subplots(
        figsize=(BEV_IMAGE_SIZE_PX / BEV_DPI, BEV_IMAGE_SIZE_PX / BEV_DPI),
        dpi=BEV_DPI,
        constrained_layout=True,
    )
    extent_points: list[np.ndarray] = []

    for obj_id, obj in sorted(objects.items()):
        footprint = _object_footprint(obj)
        center = _object_center(obj)
        mention = mentioned_by_id.get(int(obj_id))
        if mention is not None:
            color = colors(len([key for key in mentioned_by_id if key < int(obj_id)]) % 10)
            face_color = (*color[:3], 0.32)
            edge_color = color
            line_width = 2.5
            zorder = 4
        else:
            face_color = (0.76, 0.78, 0.81, 0.18)
            edge_color = (0.50, 0.53, 0.57, 0.55)
            line_width = 0.8
            zorder = 1

        if footprint is not None:
            bbox_min, bbox_max = footprint
            width, height = bbox_max[:2] - bbox_min[:2]
            axis.add_patch(
                Rectangle(
                    bbox_min[:2],
                    float(width),
                    float(height),
                    facecolor=face_color,
                    edgecolor=edge_color,
                    linewidth=line_width,
                    zorder=zorder,
                )
            )
            extent_points.extend((bbox_min[:2], bbox_max[:2]))

        if center is None:
            continue
        axis.scatter(
            [center[0]],
            [center[1]],
            s=55 if mention is not None else 8,
            color=edge_color,
            zorder=zorder + 1,
        )
        extent_points.append(center[:2])
        if mention is not None:
            label = str(mention.get("label") or obj.get("label") or "object")
            role = str(mention.get("role") or "object")
            axis.annotate(
                f"{role}: {label} #{obj_id}\n"
                f"(x,y,z)=({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}) m",
                xy=center[:2],
                xytext=(7, 7),
                textcoords="offset points",
                fontsize=8.5,
                fontweight="bold",
                color=edge_color,
                bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": edge_color, "alpha": 0.88},
                zorder=10,
            )

    frame_names = [str(question.get("image_name") or "")]
    frame_names.extend(_collect_aux_image_names(question))
    camera_points: list[np.ndarray] = []
    for index, frame_name in enumerate(frame_names):
        pose = poses.get(frame_name)
        if pose is None:
            raise ValueError(f"camera pose not found for frame {frame_name!r}")
        position = np.asarray(pose.position, dtype=np.float64)
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            raise ValueError(f"invalid camera position for frame {frame_name!r}")
        camera_points.append(position[:2])
        extent_points.append(position[:2])

    if len(camera_points) > 1:
        route = np.asarray(camera_points)
        axis.plot(
            route[:, 0],
            route[:, 1],
            linestyle="--",
            linewidth=1.5,
            color="#6b4f9b",
            alpha=0.75,
            zorder=5,
        )

    for index, (frame_name, point) in enumerate(zip(frame_names, camera_points), 1):
        pose = poses[frame_name]
        forward = _camera_forward_xy(pose)
        if forward is None:
            raise ValueError(f"invalid camera forward direction for frame {frame_name!r}")
        is_endpoint = index in {1, len(frame_names)}
        color = "#1f4e79" if index == 1 else ("#8b1e3f" if index == len(frame_names) else "#6b4f9b")
        axis.scatter(
            [point[0]],
            [point[1]],
            marker="^",
            s=150 if is_endpoint else 80,
            color=color,
            edgecolor="white",
            linewidth=0.8,
            zorder=12,
        )
        axis.arrow(
            point[0],
            point[1],
            forward[0] * 0.65,
            forward[1] * 0.65,
            width=0.025,
            head_width=0.16,
            head_length=0.20,
            length_includes_head=True,
            color=color,
            zorder=11,
        )
        label = "first main view" if index == 1 else (
            "last main view" if index == len(frame_names) else f"bridge view {index - 1}"
        )
        axis.annotate(
            label,
            xy=point,
            xytext=(6, -15),
            textcoords="offset points",
            fontsize=8.5,
            color=color,
            fontweight="bold" if is_endpoint else "normal",
            zorder=13,
        )

    if not extent_points:
        raise ValueError("scene contains no plottable objects or cameras")
    extent = np.asarray(extent_points, dtype=np.float64)
    minimum = np.min(extent, axis=0)
    maximum = np.max(extent, axis=0)
    span = np.maximum(maximum - minimum, 1.0)
    margin = np.maximum(span * 0.08, 0.5)
    axis.set_xlim(minimum[0] - margin[0], maximum[0] + margin[0])
    axis.set_ylim(minimum[1] - margin[1], maximum[1] + margin[1])
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("World X (meters, east +)")
    axis.set_ylabel("World Y (meters, north +)")
    axis.set_title(
        "Initial layout before the hypothetical operation\n"
        f"{question.get('scene_id', '')} · {question.get('type', '')}",
        fontsize=15,
        fontweight="bold",
    )
    axis.grid(True, linestyle=":", linewidth=0.7, alpha=0.5)
    axis.legend(
        handles=[
            Line2D([0], [0], color="#7f858d", lw=6, alpha=0.45, label="scene object footprint"),
            Line2D([0], [0], color="#1f4e79", marker="^", lw=1.5, label="first main view camera"),
            Line2D([0], [0], color="#8b1e3f", marker="^", lw=1.5, label="last main view camera"),
        ],
        loc="best",
        fontsize=8,
        framealpha=0.9,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=BEV_DPI, facecolor="white")
    plt.close(figure)


def generate_bev_images(args: argparse.Namespace) -> dict[str, Any]:
    benchmark_path = Path(args.benchmark)
    questions = [normalize_question(question) for question in _load_benchmark(benchmark_path)]
    multi_image_questions = [question for question in questions if is_multi_image_question(question)]
    if not multi_image_questions:
        raise ValueError(f"No multi-image questions found in {benchmark_path}")

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
                    need_poses=True,
                    oracle_cache_dir=args.oracle_cache_dir,
                )
            scene = scene_cache[cache_key]
            if scene.poses is None:
                raise ValueError(f"camera poses unavailable for {scene_id}")
            image_name = f"{uid}.png"
            render_bev_image(
                question=question,
                objects=scene.objects,
                poses=scene.poses,
                output_path=output_dir / image_name,
            )
            image_sha256 = _sha256_file(output_dir / image_name)
            entries.append(
                {
                    "question_uid": uid,
                    "dataset": dataset,
                    "scene_id": scene_id,
                    "question_type": question.get("type"),
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
