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
    SEGMENTATION_EXTREME_NOISE_MIN_AREA_PX,
    SEGMENTATION_EXTREME_NOISE_MIN_SCORE,
    _build_scene_alias_group_index,
    _call_dinox_joint_detection,
    _compute_topology_quality_for_object,
    _dedupe_detections_by_mask_iou,
    _mask_iou,
    _normalize_alias_variants,
    _rasterize_instance_depth_map,
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


def _build_zbuffer_visual_items(
    *,
    requested_label: str,
    obj_ids: list[int],
    objects_by_id: dict[int, dict[str, Any]],
    visibility_by_obj_id: dict[int, dict[str, Any]],
    topology_quality_by_obj_id: dict[int, dict[str, Any]],
    camera_pose: Any,
    color_intrinsics: Any,
    instance_mesh_data: Any,
    output_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[int, np.ndarray]]:
    items: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    rendered_masks: dict[int, np.ndarray] = {}
    for idx, obj_id in enumerate(obj_ids, start=1):
        rendered = _rasterize_instance_depth_map(
            obj_id=int(obj_id),
            camera_pose=camera_pose,
            intrinsics=color_intrinsics,
            instance_mesh_data=instance_mesh_data,
        )
        mask = np.asarray(rendered["mask"], dtype=bool)
        if not np.any(mask):
            continue

        rendered_masks[int(obj_id)] = mask
        color = _ZBUFFER_COLORS[(idx - 1) % len(_ZBUFFER_COLORS)]
        bbox = _mask_bbox(mask)
        items.append(
            {
                "mask": mask,
                "bbox": bbox,
                "text": f"Z {requested_label} obj#{obj_id}",
                "color": color,
            }
        )
        mask_path = output_dir / f"{_safe_name(requested_label)}_zbuffer_obj_{obj_id}.png"
        _write_mask_png(mask, mask_path)
        visibility_meta = visibility_by_obj_id.get(int(obj_id), {})
        summary_rows.append(
            {
                "obj_id": int(obj_id),
                "label": str(objects_by_id.get(int(obj_id), {}).get("label", "")).strip().lower(),
                "area_px": int(mask.sum()),
                "mask_bbox": bbox,
                "triangle_count": int(rendered.get("triangle_count", 0) or 0),
                "topology_status": str(
                    topology_quality_by_obj_id.get(int(obj_id), {}).get("status", "")
                ).strip().lower(),
                "projected_area_px": float(visibility_meta.get("projected_area_px", 0.0) or 0.0),
                "bbox_in_frame_ratio": float(visibility_meta.get("bbox_in_frame_ratio", 0.0) or 0.0),
                "mask_path": str(mask_path),
            }
        )
    return items, summary_rows, rendered_masks


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
    parser = argparse.ArgumentParser(description="Visualize API masks and z-buffer masks for one ScanNet frame")
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

    selector_visible_objects = get_visible_objects(
        scene_objects,
        camera_pose,
        color_intrinsics,
    )
    selector_visible_object_ids = sorted(
        int(obj.get("id"))
        for obj in selector_visible_objects
        if obj.get("id") is not None
    )
    candidate_visible_object_ids, candidate_visibility_source = _refine_candidate_visible_object_ids(
        selector_visible_object_ids,
        scene_objects,
        camera_pose,
        depth_image,
        depth_intrinsics,
    )
    candidate_visible_set = {int(obj_id) for obj_id in candidate_visible_object_ids}

    visibility_by_obj_id = compute_frame_object_visibility(
        scene_objects,
        camera_pose,
        color_intrinsics,
        image_path=image_path,
        depth_image=depth_image,
        depth_intrinsics=depth_intrinsics,
        strict_mode=False,
    )

    instance_mesh_data = load_instance_mesh_data(
        scene_dir,
        instance_ids=list(objects_by_id.keys()),
        n_surface_samples=REFERABILITY_MESH_RAY_STAGE1_BASE_SAMPLE_COUNT,
    )
    topology_quality_by_obj_id = {
        int(obj_id): _compute_topology_quality_for_object(
            obj_id=int(obj_id),
            instance_mesh_data=instance_mesh_data,
        )
        for obj_id in sorted(objects_by_id)
    }

    output_root = Path(args.output_dir) / str(args.scene_id) / Path(str(args.image_name)).stem
    output_root.mkdir(parents=True, exist_ok=True)
    original_path = output_root / "original.jpg"
    cv2.imwrite(str(original_path), image)

    summary: dict[str, Any] = {
        "scene_id": str(args.scene_id),
        "image_name": str(args.image_name),
        "image_path": str(image_path),
        "depth_path": str(depth_path) if depth_path.exists() else None,
        "candidate_visibility_source": candidate_visibility_source,
        "selector_visible_object_ids": selector_visible_object_ids,
        "candidate_visible_object_ids": candidate_visible_object_ids,
        "original_path": str(original_path),
        "labels": {},
    }

    for requested_label in requested_labels:
        label_key = _safe_name(requested_label)
        label_output_dir = output_root / label_key
        label_output_dir.mkdir(parents=True, exist_ok=True)

        match_info = _resolve_requested_label(
            requested_label,
            alias_group_index,
            objects_by_id,
        )
        matched_object_ids = [int(obj_id) for obj_id in match_info["matched_object_ids"]]
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

        zbuffer_items, zbuffer_rows, rendered_masks = _build_zbuffer_visual_items(
            requested_label=requested_label,
            obj_ids=visible_matching_object_ids,
            objects_by_id=objects_by_id,
            visibility_by_obj_id=visibility_by_obj_id,
            topology_quality_by_obj_id=topology_quality_by_obj_id,
            camera_pose=camera_pose,
            color_intrinsics=color_intrinsics,
            instance_mesh_data=instance_mesh_data,
            output_dir=label_output_dir,
        )

        api_overlay = _overlay_mask_items(image, api_items)
        zbuffer_overlay = _overlay_mask_items(image, zbuffer_items)
        combined_overlay = _overlay_mask_items(api_overlay, zbuffer_items, alpha=0.28)

        api_overlay_path = label_output_dir / f"{label_key}_api_overlay.jpg"
        zbuffer_overlay_path = label_output_dir / f"{label_key}_zbuffer_overlay.jpg"
        combined_overlay_path = label_output_dir / f"{label_key}_combined_overlay.jpg"
        cv2.imwrite(str(api_overlay_path), api_overlay)
        cv2.imwrite(str(zbuffer_overlay_path), zbuffer_overlay)
        cv2.imwrite(str(combined_overlay_path), combined_overlay)

        iou_rows = _compute_iou_rows(deduped_detections, rendered_masks)
        summary["labels"][requested_label] = {
            "requested_label": requested_label,
            "matched_alias_groups": match_info["matched_alias_groups"],
            "canonical_labels": match_info["canonical_labels"],
            "alias_variants": match_info["alias_variants"],
            "matched_object_ids": matched_object_ids,
            "visible_matching_object_ids": visible_matching_object_ids,
            "anchor_candidate_object_ids": anchor_candidate_object_ids,
            "segmentation_error": segmentation_error,
            "raw_detection_count": len(raw_detections),
            "filtered_detection_count": len(filtered_detections),
            "deduped_detection_count": len(deduped_detections),
            "api_detections": api_rows,
            "zbuffer_objects": zbuffer_rows,
            "detection_to_zbuffer_iou": iou_rows,
            "api_overlay_path": str(api_overlay_path),
            "zbuffer_overlay_path": str(zbuffer_overlay_path),
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
