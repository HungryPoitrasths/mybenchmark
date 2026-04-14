#!/usr/bin/env python3
"""Visualize API segmentation masks against z-buffer masks for one ScanNet frame."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_vlm_referability import (
    QUESTION_REVIEW_CROP_MIN_PROJECTED_AREA_PX,
    REFERABILITY_MESH_RAY_STAGE1_BASE_SAMPLE_COUNT,
    REFERABILITY_MESH_RAY_STAGE2_BASE_SAMPLE_COUNT,
    SEGMENTATION_EXTREME_NOISE_MIN_AREA_PX,
    SEGMENTATION_EXTREME_NOISE_MIN_SCORE,
    _build_scene_alias_group_index,
    _call_dinox_joint_detection,
    _compute_topology_quality_for_object,
    _dedupe_detections_by_mask_iou,
    _evaluate_crop_unique_mesh_ray_stage,
    _instance_triangle_id_set,
    _make_lazy_mesh_ray_resource_getters,
    _mask_iou,
    _normalize_alias_variants,
    _refine_candidate_visible_object_ids,
    _serialize_detection,
)
from src.frame_selector import compute_frame_object_visibility, get_visible_objects
from src.scene_parser import load_instance_mesh_data, load_scannet_label_map, parse_scene
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
logger = logging.getLogger("debug_instance_masks")

_API_COLORS = [
    (48, 76, 255),
    (32, 152, 255),
    (90, 200, 250),
    (0, 180, 180),
]
_ZBUFFER_COLORS = [
    (0, 196, 96),
    (0, 220, 180),
    (80, 235, 120),
    (0, 160, 255),
]


def _resolve_scene_dir(data_root: Path, scene_id: str) -> Path:
    for candidate in (data_root / scene_id, data_root / "scans" / scene_id):
        if candidate.exists():
            return candidate
    return data_root / scene_id


def _safe_name(value: str) -> str:
    chars: list[str] = []
    for ch in str(value):
        if ch.isalnum():
            chars.append(ch.lower())
        else:
            chars.append("_")
    safe = "".join(chars).strip("_")
    while "__" in safe:
        safe = safe.replace("__", "_")
    return safe or "item"


def _mask_bbox(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.where(np.asarray(mask, dtype=bool))
    if len(xs) == 0 or len(ys) == 0:
        return None
    x0 = int(xs.min())
    y0 = int(ys.min())
    x1 = int(xs.max()) + 1
    y1 = int(ys.max()) + 1
    return [x0, y0, x1, y1]


def _draw_label_box(
    image: np.ndarray,
    *,
    text: str,
    anchor_xy: tuple[int, int],
    color: tuple[int, int, int],
) -> None:
    x, y = anchor_xy
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 2
    (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
    x = max(0, x)
    y = max(text_h + baseline + 4, y)
    box_x1 = min(image.shape[1] - 1, x + text_w + 10)
    box_y0 = max(0, y - text_h - baseline - 8)
    box_y1 = min(image.shape[0] - 1, y + 2)
    cv2.rectangle(image, (x, box_y0), (box_x1, box_y1), color, thickness=-1)
    cv2.putText(
        image,
        text,
        (x + 5, y - baseline - 2),
        font,
        scale,
        (255, 255, 255),
        thickness,
        lineType=cv2.LINE_AA,
    )


def _overlay_mask_items(
    image: np.ndarray,
    items: list[dict[str, Any]],
    *,
    alpha: float = 0.35,
) -> np.ndarray:
    canvas = image.copy()
    for item in items:
        mask = np.asarray(item.get("mask"), dtype=bool)
        if mask.shape[:2] != canvas.shape[:2] or not np.any(mask):
            continue

        color = tuple(int(v) for v in item.get("color", (0, 255, 0)))
        tint = np.zeros_like(canvas, dtype=np.uint8)
        tint[mask] = color
        canvas = cv2.addWeighted(canvas, 1.0, tint, float(alpha), 0.0)

        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cv2.drawContours(canvas, contours, -1, color, thickness=2, lineType=cv2.LINE_AA)

        bbox = item.get("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            bbox = _mask_bbox(mask)
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            x0, y0, x1, y1 = [int(v) for v in bbox]
            cv2.rectangle(canvas, (x0, y0), (x1 - 1, y1 - 1), color, thickness=2)
            text = str(item.get("text", "")).strip()
            if text:
                _draw_label_box(
                    canvas,
                    text=text,
                    anchor_xy=(x0, max(12, y0)),
                    color=color,
                )
    return canvas


def _write_mask_png(mask: np.ndarray, path: Path) -> None:
    payload = np.where(np.asarray(mask, dtype=bool), 255, 0).astype(np.uint8)
    cv2.imwrite(str(path), payload)


def _flatten_requested_labels(raw_values: list[str]) -> list[str]:
    labels: list[str] = []
    seen: set[str] = set()
    for raw_value in raw_values:
        for piece in str(raw_value).split(","):
            normalized = piece.strip().lower()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            labels.append(normalized)
    return labels


def _resolve_requested_label(
    requested_label: str,
    alias_group_index: dict[str, dict[str, Any]],
    objects_by_id: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    normalized = str(requested_label).strip().lower()
    matched_groups: list[dict[str, Any]] = []
    for alias_group, entry in alias_group_index.items():
        canonical_labels = {
            str(label).strip().lower()
            for label in entry.get("canonical_labels", [])
        }
        alias_variants = {
            str(label).strip().lower()
            for label in entry.get("alias_variants", [])
        }
        if normalized == str(alias_group).strip().lower() or normalized in canonical_labels or normalized in alias_variants:
            matched_groups.append(entry)

    matched_object_ids = sorted(
        {
            int(obj_id)
            for entry in matched_groups
            for obj_id in entry.get("object_ids", [])
        }
    )
    if not matched_object_ids:
        matched_object_ids = sorted(
            int(obj_id)
            for obj_id, obj in objects_by_id.items()
            if str(obj.get("label", "")).strip().lower() == normalized
        )

    alias_variants = _normalize_alias_variants(
        [normalized]
        + [
            str(label)
            for entry in matched_groups
            for label in entry.get("alias_variants", [])
        ]
    )
    return {
        "requested_label": normalized,
        "matched_alias_groups": sorted(
            {
                str(entry.get("alias_group", "")).strip().lower()
                for entry in matched_groups
                if str(entry.get("alias_group", "")).strip()
            }
        ),
        "canonical_labels": sorted(
            {
                str(label).strip().lower()
                for entry in matched_groups
                for label in entry.get("canonical_labels", [])
                if str(label).strip()
            }
        ),
        "alias_variants": alias_variants,
        "matched_object_ids": matched_object_ids,
    }


def _build_api_visual_items(
    detections: list[dict[str, Any]],
    *,
    requested_label: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    items: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for idx, detection in enumerate(detections, start=1):
        color = _API_COLORS[(idx - 1) % len(_API_COLORS)]
        bbox = detection.get("bbox")
        item = {
            "mask": np.asarray(detection.get("mask"), dtype=bool),
            "bbox": bbox if isinstance(bbox, list) else _mask_bbox(np.asarray(detection.get("mask"), dtype=bool)),
            "text": f"API {requested_label} #{idx}",
            "color": color,
        }
        items.append(item)
        row = _serialize_detection(detection)
        row["index"] = idx
        row["mask_bbox"] = item["bbox"]
        summary_rows.append(row)
    return items, summary_rows


def _project_vertices_to_image(
    vertices: np.ndarray,
    camera_pose: Any,
    intrinsics: Any,
) -> tuple[np.ndarray, np.ndarray]:
    world_vertices = np.asarray(vertices, dtype=np.float64)
    camera_vertices = (
        world_vertices @ np.asarray(camera_pose.rotation, dtype=np.float64).T
        + np.asarray(camera_pose.translation, dtype=np.float64)
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


def _pixel_rays_world(
    us: np.ndarray,
    vs: np.ndarray,
    camera_pose: Any,
    intrinsics: Any,
) -> np.ndarray:
    ray_dirs_cam = np.stack(
        [
            (us - intrinsics.cx) / intrinsics.fx,
            (vs - intrinsics.cy) / intrinsics.fy,
            np.ones_like(us, dtype=np.float64),
        ],
        axis=1,
    )
    return ray_dirs_cam @ np.asarray(camera_pose.rotation, dtype=np.float64)


def _render_instance_meshray_mask(
    *,
    obj_id: int,
    camera_pose: Any,
    color_intrinsics: Any,
    ray_caster: Any,
    instance_mesh_data: Any,
    hit_epsilon: float = 0.05,
) -> dict[str, Any]:
    target_tri_ids = _instance_triangle_id_set(instance_mesh_data, int(obj_id))
    height = int(color_intrinsics.height)
    width = int(color_intrinsics.width)
    empty_mask = np.zeros((height, width), dtype=bool)
    if not target_tri_ids:
        return {
            "mask": empty_mask,
            "triangle_count": 0,
            "query_bbox": None,
            "query_pixel_count": 0,
            "visible_pixel_count": 0,
        }

    vertices = np.asarray(instance_mesh_data.vertices, dtype=np.float64)
    faces = np.asarray(instance_mesh_data.faces, dtype=np.int64)
    triangle_ids = sorted(int(tid) for tid in target_tri_ids)
    tri_vertex_ids = np.unique(faces[np.asarray(triangle_ids, dtype=np.int64)].reshape(-1))
    projected_uv, projected_depths = _project_vertices_to_image(
        vertices[tri_vertex_ids],
        camera_pose,
        color_intrinsics,
    )
    valid_vertex_mask = np.isfinite(projected_uv[:, 0]) & np.isfinite(projected_uv[:, 1]) & (projected_depths > 1e-6)
    if not np.any(valid_vertex_mask):
        return {
            "mask": empty_mask,
            "triangle_count": len(triangle_ids),
            "query_bbox": None,
            "query_pixel_count": 0,
            "visible_pixel_count": 0,
        }

    valid_uv = projected_uv[valid_vertex_mask]
    x0 = max(0, int(np.floor(float(np.min(valid_uv[:, 0])))))
    x1 = min(width, int(np.ceil(float(np.max(valid_uv[:, 0])))) + 1)
    y0 = max(0, int(np.floor(float(np.min(valid_uv[:, 1])))))
    y1 = min(height, int(np.ceil(float(np.max(valid_uv[:, 1])))) + 1)
    if x1 <= x0 or y1 <= y0:
        return {
            "mask": empty_mask,
            "triangle_count": len(triangle_ids),
            "query_bbox": None,
            "query_pixel_count": 0,
            "visible_pixel_count": 0,
        }

    grid_x, grid_y = np.meshgrid(
        np.arange(x0, x1, dtype=np.float64) + 0.5,
        np.arange(y0, y1, dtype=np.float64) + 0.5,
    )
    flat_us = grid_x.reshape(-1)
    flat_vs = grid_y.reshape(-1)
    directions = _pixel_rays_world(flat_us, flat_vs, camera_pose, color_intrinsics)
    origins = np.broadcast_to(
        np.asarray(camera_pose.position, dtype=np.float64),
        directions.shape,
    ).copy()
    first_target = ray_caster.first_hits_for_triangles(
        origins=origins,
        directions=directions,
        target_tri_ids=target_tri_ids,
    )

    if hasattr(ray_caster, "_first_non_ignored_hits"):
        first_any, _has_any_hit, forced_blocked = ray_caster._first_non_ignored_hits(
            origins=origins,
            directions=directions,
            ignored_tri_ids=None,
        )
    else:
        first_any = {}
        forced_blocked = np.zeros(len(origins), dtype=bool)
        for ray_idx, direction in enumerate(directions):
            hit = ray_caster.first_visible_hit(origins[ray_idx], direction)
            if hit is not None:
                _hit_point, tri_id, dist = hit
                first_any[ray_idx] = (int(tri_id), float(dist))

    mask = np.zeros((height, width), dtype=bool)
    visible_pixel_count = 0
    for ray_idx, target_hit in first_target.items():
        if forced_blocked[ray_idx]:
            continue
        any_hit = first_any.get(int(ray_idx))
        if any_hit is None:
            continue
        any_tri_id, any_dist = any_hit
        _target_point, _target_tri_id, target_dist = target_hit
        if any_tri_id not in target_tri_ids:
            continue
        if abs(float(any_dist) - float(target_dist)) > float(hit_epsilon):
            continue
        local_y = int(ray_idx) // max(1, (x1 - x0))
        local_x = int(ray_idx) % max(1, (x1 - x0))
        mask[y0 + local_y, x0 + local_x] = True
        visible_pixel_count += 1

    return {
        "mask": mask,
        "triangle_count": len(triangle_ids),
        "query_bbox": [x0, y0, x1, y1],
        "query_pixel_count": int(len(flat_us)),
        "visible_pixel_count": int(visible_pixel_count),
    }


def _evaluate_meshray_review(
    *,
    obj_id: int,
    camera_pose: Any,
    color_intrinsics: Any,
    ray_caster: Any,
    instance_mesh_data_getter: Any,
) -> dict[str, Any]:
    stage1 = _evaluate_crop_unique_mesh_ray_stage(
        obj_id=int(obj_id),
        camera_pose=camera_pose,
        color_intrinsics=color_intrinsics,
        ray_caster=ray_caster,
        instance_mesh_data=instance_mesh_data_getter(REFERABILITY_MESH_RAY_STAGE1_BASE_SAMPLE_COUNT),
        base_sample_count=REFERABILITY_MESH_RAY_STAGE1_BASE_SAMPLE_COUNT,
    )
    if int(stage1.get("visible_count", 0) or 0) > 0:
        return {
            "applied": True,
            "decision": "pass",
            "reason": "stage1_visible_evidence",
            "stage1": stage1,
            "stage2": None,
        }

    stage2 = _evaluate_crop_unique_mesh_ray_stage(
        obj_id=int(obj_id),
        camera_pose=camera_pose,
        color_intrinsics=color_intrinsics,
        ray_caster=ray_caster,
        instance_mesh_data=instance_mesh_data_getter(REFERABILITY_MESH_RAY_STAGE2_BASE_SAMPLE_COUNT),
        base_sample_count=REFERABILITY_MESH_RAY_STAGE2_BASE_SAMPLE_COUNT,
    )
    if int(stage2.get("visible_count", 0) or 0) > 0:
        decision = "pass"
        reason = "stage2_visible_evidence"
    elif int(stage2.get("valid_count", 0) or 0) <= 0:
        decision = "drop"
        reason = "no_valid_rays_after_stage2"
    else:
        decision = "drop"
        reason = "fully_occluded_after_stage2"
    return {
        "applied": True,
        "decision": decision,
        "reason": reason,
        "stage1": stage1,
        "stage2": stage2,
    }


def _build_meshray_visual_items(
    *,
    requested_label: str,
    obj_ids: list[int],
    objects_by_id: dict[int, dict[str, Any]],
    visibility_by_obj_id: dict[int, dict[str, Any]],
    topology_quality_by_obj_id: dict[int, dict[str, Any]],
    camera_pose: Any,
    color_intrinsics: Any,
    ray_caster: Any,
    instance_mesh_data_getter: Any,
    output_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[int, np.ndarray], dict[int, dict[str, Any]]]:
    items: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    rendered_masks: dict[int, np.ndarray] = {}
    visibility_reviews: dict[int, dict[str, Any]] = {}
    visible_index = 0

    for obj_id in obj_ids:
        review = _evaluate_meshray_review(
            obj_id=int(obj_id),
            camera_pose=camera_pose,
            color_intrinsics=color_intrinsics,
            ray_caster=ray_caster,
            instance_mesh_data_getter=instance_mesh_data_getter,
        )
        visibility_reviews[int(obj_id)] = review
        if str(review.get("decision", "")).strip().lower() != "pass":
            continue

        rendered = _render_instance_meshray_mask(
            obj_id=int(obj_id),
            camera_pose=camera_pose,
            color_intrinsics=color_intrinsics,
            ray_caster=ray_caster,
            instance_mesh_data=instance_mesh_data_getter(REFERABILITY_MESH_RAY_STAGE1_BASE_SAMPLE_COUNT),
        )
        mask = np.asarray(rendered["mask"], dtype=bool)
        if not np.any(mask):
            continue

        visible_index += 1
        rendered_masks[int(obj_id)] = mask
        color = _ZBUFFER_COLORS[(visible_index - 1) % len(_ZBUFFER_COLORS)]
        bbox = _mask_bbox(mask)
        items.append(
            {
                "mask": mask,
                "bbox": bbox,
                "text": f"MR {requested_label} obj#{obj_id}",
                "color": color,
            }
        )
        mask_path = output_dir / f"{_safe_name(requested_label)}_meshray_obj_{obj_id}.png"
        _write_mask_png(mask, mask_path)
        visibility_meta = visibility_by_obj_id.get(int(obj_id), {})
        summary_rows.append(
            {
                "obj_id": int(obj_id),
                "label": str(objects_by_id.get(int(obj_id), {}).get("label", "")).strip().lower(),
                "area_px": int(mask.sum()),
                "mask_bbox": bbox,
                "triangle_count": int(rendered.get("triangle_count", 0) or 0),
                "query_bbox": rendered.get("query_bbox"),
                "query_pixel_count": int(rendered.get("query_pixel_count", 0) or 0),
                "visible_pixel_count": int(rendered.get("visible_pixel_count", 0) or 0),
                "topology_status": str(
                    topology_quality_by_obj_id.get(int(obj_id), {}).get("status", "")
                ).strip().lower(),
                "projected_area_px": float(visibility_meta.get("projected_area_px", 0.0) or 0.0),
                "bbox_in_frame_ratio": float(visibility_meta.get("bbox_in_frame_ratio", 0.0) or 0.0),
                "zbuffer_mask_in_frame_ratio": float(
                    visibility_meta.get("zbuffer_mask_in_frame_ratio", 0.0) or 0.0
                ),
                "mesh_ray_visibility_review": review,
                "mask_path": str(mask_path),
            }
        )
    return items, summary_rows, rendered_masks, visibility_reviews


def _compute_iou_rows(
    detections: list[dict[str, Any]],
    rendered_masks: dict[int, np.ndarray],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for det_index, detection in enumerate(detections, start=1):
        det_mask = np.asarray(detection.get("mask"), dtype=bool)
        ious = {
            str(obj_id): float(_mask_iou(det_mask, mask))
            for obj_id, mask in sorted(rendered_masks.items())
        }
        best_obj_id = None
        best_iou = 0.0
        for obj_id, value in sorted(ious.items()):
            if float(value) > best_iou:
                best_iou = float(value)
                best_obj_id = int(obj_id)
        rows.append(
            {
                "detection_index": det_index,
                "ious_by_obj_id": ious,
                "best_match_obj_id": best_obj_id,
                "best_match_iou": float(best_iou),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize API masks and mesh-ray-filtered projected masks for one ScanNet frame")
    parser.add_argument(
        "--data_root",
        type=str,
        default="data/scannet/scans",
        help="Root containing ScanNet scene folders, or a parent directory that has scans/<scene_id>",
    )
    parser.add_argument("--scene_id", type=str, required=True)
    parser.add_argument("--image_name", type=str, required=True)
    parser.add_argument(
        "--labels",
        type=str,
        nargs="+",
        required=True,
        help="Requested labels; supports either space-separated values or one comma-separated string",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output/mask_debug",
        help="Directory for overlay images and summary JSON",
    )
    parser.add_argument("--label_map", type=str, default=None)
    args = parser.parse_args()

    if args.label_map:
        load_scannet_label_map(args.label_map)

    requested_labels = _flatten_requested_labels(list(args.labels))
    if not requested_labels:
        raise ValueError("No valid labels provided")

    data_root = Path(args.data_root)
    scene_dir = _resolve_scene_dir(data_root, str(args.scene_id))
    if not scene_dir.exists():
        raise FileNotFoundError(f"Scene directory not found: {scene_dir}")

    image_path = scene_dir / "color" / str(args.image_name)
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Image not found or unreadable: {image_path}")

    scene = parse_scene(scene_dir)
    if scene is None:
        raise RuntimeError(f"parse_scene returned None for {scene_dir}")

    axis_alignment = load_axis_alignment(scene_dir)
    poses = load_scannet_poses(scene_dir, axis_alignment=axis_alignment)
    if str(args.image_name) not in poses:
        raise KeyError(f"Pose not found for frame {args.image_name}")
    camera_pose = poses[str(args.image_name)]

    color_intrinsics = load_scannet_intrinsics(scene_dir)
    try:
        depth_intrinsics = load_scannet_depth_intrinsics(scene_dir)
    except Exception as exc:
        logger.warning("Depth intrinsics load failed for %s: %s", scene_dir.name, exc)
        depth_intrinsics = None

    depth_image = None
    depth_path = scene_dir / "depth" / f"{Path(str(args.image_name)).stem}.png"
    if depth_intrinsics is not None and depth_path.exists():
        try:
            depth_image = load_depth_image(depth_path)
        except Exception as exc:
            logger.warning("Depth image load failed for %s: %s", depth_path, exc)

    scene_objects = list(scene.get("objects", []))
    objects_by_id = {
        int(obj.get("id")): obj
        for obj in scene_objects
        if obj.get("id") is not None
    }
    alias_group_index = _build_scene_alias_group_index(scene_objects)
    label_match_info_by_requested = {
        requested_label: _resolve_requested_label(
            requested_label,
            alias_group_index,
            objects_by_id,
        )
        for requested_label in requested_labels
    }
    relevant_object_ids = sorted(
        {
            int(obj_id)
            for match_info in label_match_info_by_requested.values()
            for obj_id in match_info.get("matched_object_ids", [])
        }
    )
    instance_mesh_data = load_instance_mesh_data(
        scene_dir,
        instance_ids=relevant_object_ids,
        n_surface_samples=REFERABILITY_MESH_RAY_STAGE1_BASE_SAMPLE_COUNT,
    )
    ray_caster_getter, _unused_instance_mesh_data_getter = _make_lazy_mesh_ray_resource_getters(
        scene_dir=scene_dir,
        scene_objects=scene_objects,
        axis_alignment=axis_alignment,
    )
    ray_caster = ray_caster_getter()
    mesh_data_cache: dict[int, Any] = {
        REFERABILITY_MESH_RAY_STAGE1_BASE_SAMPLE_COUNT: instance_mesh_data,
    }

    def instance_mesh_data_getter(base_sample_count: int) -> Any:
        base_count = int(base_sample_count)
        if base_count not in mesh_data_cache:
            mesh_data_cache[base_count] = load_instance_mesh_data(
                scene_dir,
                instance_ids=relevant_object_ids,
                n_surface_samples=base_count,
            )
        return mesh_data_cache[base_count]

    output_root = Path(args.output_dir) / str(args.scene_id) / Path(str(args.image_name)).stem
    output_root.mkdir(parents=True, exist_ok=True)
    original_path = output_root / "original.jpg"
    cv2.imwrite(str(original_path), image)

    summary: dict[str, Any] = {
        "scene_id": str(args.scene_id),
        "image_name": str(args.image_name),
        "image_path": str(image_path),
        "depth_path": str(depth_path) if depth_path.exists() else None,
        "candidate_visibility_scope": "per_label",
        "original_path": str(original_path),
        "labels": {},
    }

    for requested_label in requested_labels:
        label_key = _safe_name(requested_label)
        label_output_dir = output_root / label_key
        label_output_dir.mkdir(parents=True, exist_ok=True)

        match_info = label_match_info_by_requested[requested_label]
        matched_object_ids = [int(obj_id) for obj_id in match_info["matched_object_ids"]]
        matched_objects = [
            objects_by_id[int(obj_id)]
            for obj_id in matched_object_ids
            if int(obj_id) in objects_by_id
        ]
        selector_visible_objects = get_visible_objects(
            matched_objects,
            camera_pose,
            color_intrinsics,
            instance_mesh_data=instance_mesh_data,
        )
        selector_visible_object_ids = sorted(
            int(obj.get("id"))
            for obj in selector_visible_objects
            if obj.get("id") is not None
        )
        candidate_visible_object_ids, candidate_visibility_source = _refine_candidate_visible_object_ids(
            selector_visible_object_ids,
            matched_objects,
            camera_pose,
            depth_image,
            depth_intrinsics,
        )
        candidate_visible_set = {int(obj_id) for obj_id in candidate_visible_object_ids}
        visibility_by_obj_id = compute_frame_object_visibility(
            matched_objects,
            camera_pose,
            color_intrinsics,
            image_path=image_path,
            depth_image=depth_image,
            depth_intrinsics=depth_intrinsics,
            instance_mesh_data=instance_mesh_data,
            strict_mode=False,
        )
        topology_quality_by_obj_id = {
            int(obj_id): _compute_topology_quality_for_object(
                obj_id=int(obj_id),
                instance_mesh_data=instance_mesh_data,
            )
            for obj_id in matched_object_ids
        }
        visible_matching_object_ids = [
            int(obj_id)
            for obj_id in matched_object_ids
            if int(obj_id) in candidate_visible_set
        ]
        anchor_candidate_object_ids = [
            int(obj_id)
            for obj_id in visible_matching_object_ids
            if float(visibility_by_obj_id.get(int(obj_id), {}).get("projected_area_px", 0.0) or 0.0)
            >= QUESTION_REVIEW_CROP_MIN_PROJECTED_AREA_PX
            and str(topology_quality_by_obj_id.get(int(obj_id), {}).get("status", "")).strip().lower() != "fail"
        ]

        segmentation_error = None
        raw_detections: list[dict[str, Any]] = []
        filtered_detections: list[dict[str, Any]] = []
        deduped_detections: list[dict[str, Any]] = []
        try:
            raw_detections = _call_dinox_joint_detection(
                client=None,
                image_path=image_path,
                alias_variants=list(match_info["alias_variants"]),
                image_shape=tuple(image.shape),
            )
            filtered_detections = [
                detection
                for detection in raw_detections
                if int(detection.get("area_px", 0) or 0) >= SEGMENTATION_EXTREME_NOISE_MIN_AREA_PX
                and float(detection.get("score", 0.0) or 0.0) >= SEGMENTATION_EXTREME_NOISE_MIN_SCORE
            ]
            deduped_detections = _dedupe_detections_by_mask_iou(filtered_detections)
        except Exception as exc:
            segmentation_error = str(exc)
            logger.warning(
                "Segmentation failed for %s/%s label=%s: %s",
                scene_dir.name,
                args.image_name,
                requested_label,
                exc,
            )

        api_items, api_rows = _build_api_visual_items(
            deduped_detections,
            requested_label=requested_label,
        )
        for idx, detection in enumerate(deduped_detections, start=1):
            mask_path = label_output_dir / f"{label_key}_api_mask_{idx:02d}.png"
            _write_mask_png(np.asarray(detection.get("mask"), dtype=bool), mask_path)
            api_rows[idx - 1]["mask_path"] = str(mask_path)

        meshray_items, meshray_rows, rendered_masks, meshray_reviews = _build_meshray_visual_items(
            requested_label=requested_label,
            obj_ids=visible_matching_object_ids,
            objects_by_id=objects_by_id,
            visibility_by_obj_id=visibility_by_obj_id,
            topology_quality_by_obj_id=topology_quality_by_obj_id,
            camera_pose=camera_pose,
            color_intrinsics=color_intrinsics,
            ray_caster=ray_caster,
            instance_mesh_data_getter=instance_mesh_data_getter,
            output_dir=label_output_dir,
        )

        api_overlay = _overlay_mask_items(image, api_items)
        meshray_overlay = _overlay_mask_items(image, meshray_items)
        combined_overlay = _overlay_mask_items(api_overlay, meshray_items, alpha=0.28)

        api_overlay_path = label_output_dir / f"{label_key}_api_overlay.jpg"
        meshray_overlay_path = label_output_dir / f"{label_key}_meshray_overlay.jpg"
        combined_overlay_path = label_output_dir / f"{label_key}_combined_overlay.jpg"
        cv2.imwrite(str(api_overlay_path), api_overlay)
        cv2.imwrite(str(meshray_overlay_path), meshray_overlay)
        cv2.imwrite(str(combined_overlay_path), combined_overlay)

        iou_rows = _compute_iou_rows(deduped_detections, rendered_masks)
        summary["labels"][requested_label] = {
            "requested_label": requested_label,
            "matched_alias_groups": match_info["matched_alias_groups"],
            "canonical_labels": match_info["canonical_labels"],
            "alias_variants": match_info["alias_variants"],
            "matched_object_ids": matched_object_ids,
            "selector_visible_object_ids": selector_visible_object_ids,
            "candidate_visibility_source": candidate_visibility_source,
            "candidate_visible_object_ids": candidate_visible_object_ids,
            "visible_matching_object_ids": visible_matching_object_ids,
            "anchor_candidate_object_ids": anchor_candidate_object_ids,
            "segmentation_error": segmentation_error,
            "raw_detection_count": len(raw_detections),
            "filtered_detection_count": len(filtered_detections),
            "deduped_detection_count": len(deduped_detections),
            "api_detections": api_rows,
            "meshray_candidate_object_ids": visible_matching_object_ids,
            "meshray_visibility_reviews_by_obj_id": {
                str(obj_id): review
                for obj_id, review in sorted(meshray_reviews.items())
            },
            "meshray_objects": meshray_rows,
            "detection_to_meshray_iou": iou_rows,
            "api_overlay_path": str(api_overlay_path),
            "meshray_overlay_path": str(meshray_overlay_path),
            "combined_overlay_path": str(combined_overlay_path),
        }

    summary_path = output_root / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_root),
                "summary_path": str(summary_path),
                "labels": requested_labels,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
