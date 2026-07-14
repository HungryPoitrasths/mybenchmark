"""Stage 3: Representative frame selection.

For each scene, select 3-5 frames that collectively cover as many objects
(and support relationships) as possible while preserving strong referable
attachment candidates.

ScanNet stores one colour image per frame in ``color/<frame_id>.jpg`` and the
corresponding camera pose in ``pose/<frame_id>.txt``.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .scene_parser import InstanceMeshData
from .utils.colmap_loader import (
    CameraIntrinsics,
    CameraPose,
    load_axis_alignment,
    load_scannet_intrinsics,
    load_scannet_poses,
)
from .utils.coordinate_transform import default_edge_margin_px, is_in_image

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  Image quality gate
# ---------------------------------------------------------------------------

# Coarse prefilter thresholds for the geometric selector. Detailed image-
# quality screening still happens later in the VLM rerank stage.
QUALITY_STAGE1_DROP_FRACTION = 0.30
QUALITY_STAGE2_LAPLACIAN_MIN = 80.0
QUALITY_STAGE2_TENENGRAD_MIN = 12.0


def _compute_laplacian_variance(gray_image: np.ndarray) -> float:
    laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
    return float(laplacian.var())


def _compute_tenengrad(gray_image: np.ndarray) -> float:
    grad_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt((grad_x * grad_x) + (grad_y * grad_y))
    return float(magnitude.mean())


def _read_image_quality_metrics(image_path: Path) -> dict[str, Any]:
    """Read an image once and return the selector's coarse focus metrics."""
    img = cv2.imread(str(image_path))
    if img is None:
        logger.debug("Cannot read image %s; excluding from quality filter", image_path)
        return {
            "readable": False,
            "laplacian_variance": None,
            "tenengrad": None,
        }

    # Downsample to 1/4 size for speed. These thresholds are calibrated for the
    # reduced-resolution image and are not resolution-invariant.
    small = cv2.resize(img, (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    denoised = cv2.GaussianBlur(gray, (3, 3), 0)
    return {
        "readable": True,
        "laplacian_variance": _compute_laplacian_variance(denoised),
        "tenengrad": _compute_tenengrad(denoised),
    }


def _passes_absolute_image_quality_gate(
    laplacian_variance: float,
    tenengrad: float,
) -> bool:
    return (
        laplacian_variance >= QUALITY_STAGE2_LAPLACIAN_MIN
        and tenengrad >= QUALITY_STAGE2_TENENGRAD_MIN
    )


def _assign_descending_ordinal_ranks(
    frame_entries: list[dict[str, Any]],
    *,
    metric_key: str,
    rank_key: str,
) -> None:
    ordered = sorted(
        frame_entries,
        key=lambda entry: (
            -float(entry[metric_key]),
            int(entry["sampled_index"]),
        ),
    )
    for rank, entry in enumerate(ordered, start=1):
        entry[rank_key] = rank


def _filter_sampled_frames_by_quality(
    sampled_frames: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    readable_frames: list[dict[str, Any]] = []
    unreadable_count = 0
    for entry in sampled_frames:
        metrics = _read_image_quality_metrics(entry["image_path"])
        if not bool(metrics.get("readable")):
            unreadable_count += 1
            continue
        enriched_entry = dict(entry)
        enriched_entry["laplacian_variance"] = float(metrics["laplacian_variance"])
        enriched_entry["tenengrad"] = float(metrics["tenengrad"])
        readable_frames.append(enriched_entry)

    if not readable_frames:
        return [], {
            "unreadable_count": unreadable_count,
            "stage1_rejected_count": 0,
            "stage2_rejected_count": 0,
            "final_quality_kept_count": 0,
        }

    _assign_descending_ordinal_ranks(
        readable_frames,
        metric_key="laplacian_variance",
        rank_key="laplacian_rank",
    )
    _assign_descending_ordinal_ranks(
        readable_frames,
        metric_key="tenengrad",
        rank_key="tenengrad_rank",
    )
    ranked_frames = sorted(
        readable_frames,
        key=lambda entry: (
            int(entry["laplacian_rank"]) + int(entry["tenengrad_rank"]),
            -float(entry["laplacian_variance"]),
            -float(entry["tenengrad"]),
            int(entry["sampled_index"]),
        ),
    )

    stage1_rejected_count = math.floor(
        QUALITY_STAGE1_DROP_FRACTION * len(readable_frames)
    )
    stage1_survivors = ranked_frames[: len(ranked_frames) - stage1_rejected_count]
    stage1_survivors.sort(key=lambda entry: int(entry["sampled_index"]))

    # Stage 2 (absolute Laplacian/Tenengrad thresholds) removed: those
    # constants were tuned for ScanNet v2's 640x480 sensor and rejected the
    # majority of ScanNet++ iPhone frames (82.8% below QUALITY_STAGE2_LAPLACIAN_MIN
    # on scene c50d2d1d42) despite the relative Stage 1 cut already discarding
    # the worst 30%.
    return stage1_survivors, {
        "unreadable_count": unreadable_count,
        "stage1_rejected_count": stage1_rejected_count,
        "stage2_rejected_count": 0,
        "final_quality_kept_count": len(stage1_survivors),
    }


def passes_image_quality(image_path: Path) -> bool:
    """Return True if an image passes the selector's Stage-2 absolute gate."""
    metrics = _read_image_quality_metrics(image_path)
    if not bool(metrics.get("readable")):
        return False

    laplacian_var = float(metrics["laplacian_variance"])
    if laplacian_var < QUALITY_STAGE2_LAPLACIAN_MIN:
        logger.debug(
            "Image %s too blurry (Laplacian var=%.1f < %.1f)",
            image_path.name, laplacian_var, QUALITY_STAGE2_LAPLACIAN_MIN,
        )
        return False

    tenengrad = float(metrics["tenengrad"])
    if tenengrad < QUALITY_STAGE2_TENENGRAD_MIN:
        logger.debug(
            "Image %s too blurry (Tenengrad=%.1f < %.1f)",
            image_path.name, tenengrad, QUALITY_STAGE2_TENENGRAD_MIN,
        )
        return False

    return _passes_absolute_image_quality_gate(laplacian_var, tenengrad)

    img = cv2.imread(str(image_path))
    if img is None:
        logger.debug("Cannot read image %s — failing quality check", image_path)
        return False

    # Downsample to 1/4 size for speed. These thresholds are calibrated for the
    # reduced-resolution image and are not resolution-invariant.
    small = cv2.resize(img, (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    # Use a conservative first-pass blur gate: combine two focus metrics so we
    # trim obviously weak candidates before the later VLM review.
    denoised = cv2.GaussianBlur(gray, (3, 3), 0)
    laplacian_var = _compute_laplacian_variance(denoised)
    if laplacian_var < SHARPNESS_MIN:
        logger.debug(
            "Image %s too blurry (Laplacian var=%.1f < %.1f)",
            image_path.name, laplacian_var, SHARPNESS_MIN,
        )
        return False

    tenengrad = _compute_tenengrad(denoised)
    if tenengrad < TENENGRAD_MIN:
        logger.debug(
            "Image %s too blurry (Tenengrad=%.1f < %.1f)",
            image_path.name, tenengrad, TENENGRAD_MIN,
        )
        return False

    return True


def _project_object_roi(
    obj: dict,
    pose: CameraPose,
    intrinsics: CameraIntrinsics,
) -> dict[str, Any]:
    """Project an object's bbox to image space and summarise the footprint.

    Uses :func:`project_camera_points_to_image` so that distortion models
    (e.g. ``OPENCV_FISHEYE``) are respected when present in *intrinsics*.
    """
    from .utils.coordinate_transform import project_camera_points_to_image

    bbox_min = np.array(obj["bbox_min"], dtype=np.float64)
    bbox_max = np.array(obj["bbox_max"], dtype=np.float64)
    mid = (bbox_min + bbox_max) / 2.0

    sample_points_world = []
    for x in [bbox_min[0], bbox_max[0]]:
        for y in [bbox_min[1], bbox_max[1]]:
            for z in [bbox_min[2], bbox_max[2]]:
                sample_points_world.append(np.array([x, y, z], dtype=np.float64))
    sample_points_world.append(mid)
    points_world = np.stack(sample_points_world, axis=0)  # (9, 3)

    # Batch world→camera
    points_cam = (pose.rotation @ points_world.T + pose.translation.reshape(3, 1)).T
    uv_all, depths = project_camera_points_to_image(points_cam, intrinsics)

    in_frame = 0
    valid = 0
    projected: list[tuple[float, float]] = []
    for i in range(len(uv_all)):
        if depths[i] <= 0 or not np.isfinite(uv_all[i]).all():
            continue
        valid += 1
        u, v = float(uv_all[i, 0]), float(uv_all[i, 1])
        projected.append((u, v))
        if 0 <= u < intrinsics.width and 0 <= v < intrinsics.height:
            in_frame += 1

    if len(projected) < 2:
        return {
            "valid_projection_count": valid,
            "bbox_in_frame_ratio": 0.0 if valid == 0 else in_frame / valid,
            "projected_area_px": 0.0,
            "edge_margin_px": 0.0,
            "roi_bounds": None,
        }

    us = [p[0] for p in projected]
    vs = [p[1] for p in projected]
    u_min = max(0, int(np.floor(min(us) - 5)))
    u_max = min(intrinsics.width, int(np.ceil(max(us) + 5)))
    v_min = max(0, int(np.floor(min(vs) - 5)))
    v_max = min(intrinsics.height, int(np.ceil(max(vs) + 5)))
    width = max(0, u_max - u_min)
    height = max(0, v_max - v_min)
    area = float(width * height)
    edge_margin = float(
        min(
            u_min,
            v_min,
            intrinsics.width - u_max,
            intrinsics.height - v_max,
        )
    )
    return {
        "valid_projection_count": valid,
        "bbox_in_frame_ratio": 0.0 if valid == 0 else in_frame / valid,
        "projected_area_px": area,
        "edge_margin_px": edge_margin,
        "roi_bounds": (u_min, u_max, v_min, v_max),
    }


def _compute_roi_sharpness(
    gray_image,
    roi_bounds: tuple[int, int, int, int] | None,
) -> float | None:
    """Return Laplacian variance inside a projected ROI, or None if invalid."""
    if gray_image is None or roi_bounds is None:
        return None
    u_min, u_max, v_min, v_max = roi_bounds
    if u_max - u_min < 10 or v_max - v_min < 10:
        return None
    roi = gray_image[v_min:v_max, u_min:u_max]
    if roi.size == 0:
        return None
    return float(cv2.Laplacian(roi, cv2.CV_64F).var())


def _build_projection_visibility_meta(
    obj: dict[str, Any],
    pose: CameraPose,
    intrinsics: CameraIntrinsics,
    *,
    roi_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from .utils.coordinate_transform import project_camera_points_to_image

    center = np.array(obj["center"], dtype=np.float64)
    point_cam = pose.world_to_camera_point(center)
    uv_all, depths = project_camera_points_to_image(point_cam.reshape(1, 3), intrinsics)
    depth = float(depths[0])
    uv = uv_all[0] if depth > 0 and np.isfinite(uv_all[0]).all() else None

    resolved_roi_info = roi_info or _project_object_roi(obj, pose, intrinsics)
    return {
        "center_uv_px": [float(uv[0]), float(uv[1])] if uv is not None else None,
        "depth_m": depth,
        "bbox_in_frame_ratio": float(resolved_roi_info["bbox_in_frame_ratio"]),
        "projected_area_px": float(resolved_roi_info["projected_area_px"]),
        "edge_margin_px": float(resolved_roi_info["edge_margin_px"]),
        "roi_bounds_px": (
            [int(value) for value in resolved_roi_info["roi_bounds"]]
            if resolved_roi_info["roi_bounds"] is not None else None
        ),
    }


def compute_referability_object_visibility(
    objects: list[dict],
    pose: CameraPose,
    color_intrinsics: CameraIntrinsics,
) -> dict[int, dict[str, Any]]:
    """Compute projection-only visibility metadata for referability review."""
    visibility: dict[int, dict[str, Any]] = {}
    for obj in objects:
        visibility[int(obj["id"])] = _build_projection_visibility_meta(
            obj,
            pose,
            color_intrinsics,
        )
    return visibility


def compute_frame_object_visibility(
    objects: list[dict],
    pose: CameraPose,
    color_intrinsics: CameraIntrinsics,
    image_path: Path | None = None,
    depth_image=None,
    depth_intrinsics: CameraIntrinsics | None = None,
    instance_mesh_data: InstanceMeshData | None = None,
    margin: int | None = None,
    min_depth: float = 0.3,
    max_depth: float = 8.0,
) -> dict[int, dict[str, Any]]:
    """Compute per-object visibility metadata for a single frame."""
    from .utils.depth_occlusion import compute_depth_occlusion

    if margin is None:
        margin = default_edge_margin_px(color_intrinsics)

    gray = None
    if image_path is not None and image_path.exists():
        img = cv2.imread(str(image_path))
        if img is not None:
            gray = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (3, 3), 0)

    visibility: dict[int, dict[str, Any]] = {}
    for obj in objects:
        roi_info = _project_object_roi(obj, pose, color_intrinsics)
        projection_meta = _build_projection_visibility_meta(
            obj,
            pose,
            color_intrinsics,
            roi_info=roi_info,
        )
        center_uv_px = projection_meta["center_uv_px"]
        uv = (
            np.array(center_uv_px, dtype=np.float64)
            if isinstance(center_uv_px, list) and len(center_uv_px) == 2
            else None
        )
        depth = float(projection_meta["depth_m"])
        center_in_frame = (
            min_depth < depth <= max_depth
            and is_in_image(uv, color_intrinsics, margin=margin)
        )

        roi_bounds_px = projection_meta["roi_bounds_px"]
        roi_sharpness = _compute_roi_sharpness(
            gray,
            tuple(int(value) for value in roi_bounds_px)
            if isinstance(roi_bounds_px, list) and len(roi_bounds_px) == 4
            else None,
        )

        occlusion_status = "unknown"
        visible_ratio = 0.0
        if depth_image is not None and depth_intrinsics is not None:
            occlusion_status, visible_ratio = compute_depth_occlusion(
                bbox_min=np.array(obj["bbox_min"], dtype=np.float64),
                bbox_max=np.array(obj["bbox_max"], dtype=np.float64),
                camera_pose=pose,
                intrinsics=depth_intrinsics,
                depth_image=depth_image,
            )

        visibility[obj["id"]] = {
            "obj_id": obj["id"],
            "label": obj.get("label", ""),
            "center_in_frame": center_in_frame,
            "center_visible": (
                occlusion_status != "not visible"
                if occlusion_status != "unknown" else center_in_frame
            ),
            **projection_meta,
            "occlusion_status": occlusion_status,
            "visible_ratio": float(visible_ratio),
            "valid_projection_count": int(roi_info["valid_projection_count"]),
            "roi_sharpness": roi_sharpness,
            "label_unique_in_frame": True,
            "eligible_as_reference": center_in_frame,
            "eligible_as_target": center_in_frame,
            "rejection_reasons": [],
        }

    label_counts: dict[str, int] = {}
    for meta in visibility.values():
        if meta["center_in_frame"]:
            label = str(meta["label"]).lower()
            label_counts[label] = label_counts.get(label, 0) + 1

    for meta in visibility.values():
        label = str(meta["label"]).lower()
        meta["label_unique_in_frame"] = label_counts.get(label, 0) <= 1
        reasons: list[str] = []
        if not meta["center_in_frame"]:
            reasons.append("center_out_of_frame")

        meta["rejection_reasons"] = reasons
        meta["eligible_as_reference"] = len(reasons) == 0
        meta["eligible_as_target"] = len(reasons) == 0

    return visibility


DEFAULT_MAX_FRAMES = 5


def refine_visible_ids_with_raycasting(
    visible_object_ids: list[int],
    objects: list[dict],
    pose: CameraPose,
    ray_caster,
) -> list[int]:
    """Remove fully-occluded objects from *visible_object_ids*.

    Called AFTER projection-based frame selection so that ray casting only
    runs on the 3-5 selected frames rather than every frame in the scene.

    An object is dropped when ``multi_ray_occlusion`` (8 sample rays toward
    the bbox) classifies it as ``"not visible"``.  Partially-occluded
    objects are kept — they are still meaningful as question subjects.
    """
    obj_map = {o["id"]: o for o in objects}
    refined: list[int] = []
    for obj_id in visible_object_ids:
        obj = obj_map.get(obj_id)
        if obj is None:
            continue
        status = ray_caster.multi_ray_occlusion(
            camera_pos=pose.position,
            target_bbox_min=np.array(obj["bbox_min"]),
            target_bbox_max=np.array(obj["bbox_max"]),
        )
        if status != "fully_occluded":
            refined.append(obj_id)
    return refined


def refine_visible_ids_with_depth(
    visible_object_ids: list[int],
    objects: list[dict],
    pose: CameraPose,
    depth_image,
    depth_intrinsics,
) -> list[int]:
    """Remove fully-occluded objects using depth-map occlusion.

    Similar to ``refine_visible_ids_with_raycasting`` but uses the depth map
    instead of trimesh ray casting — much faster and does not require pyembree.

    Partially-occluded objects are kept; only fully-occluded ones are dropped.
    """
    from .utils.depth_occlusion import compute_depth_occlusion
    import numpy as np

    obj_map = {o["id"]: o for o in objects}
    refined: list[int] = []
    for obj_id in visible_object_ids:
        obj = obj_map.get(obj_id)
        if obj is None:
            continue
        status, _ratio = compute_depth_occlusion(
            bbox_min=np.array(obj["bbox_min"]),
            bbox_max=np.array(obj["bbox_max"]),
            camera_pose=pose,
            intrinsics=depth_intrinsics,
            depth_image=depth_image,
        )
        if status != "not visible":
            refined.append(obj_id)
    return refined


MIN_VISIBLE_OBJECTS = 3
VIEWPOINT_DIVERSITY_MIN_ANGLE = 20  # degrees
# Denser sampling gives the later VLM rerank a broader candidate pool.
FRAME_STRIDE = 3
VISIBLE_BBOX_IN_FRAME_RATIO_MIN = 0.35
# Calibrated against ScanNet v2's 640x480 sensor; scaled per-frame by
# visible_projected_area_px() so higher-resolution sensors (e.g. ScanNet++
# iPhone at 1920x1440) get a proportionally larger floor.
VISIBLE_PROJECTED_AREA_RATIO = 800.0 / (640 * 480)
FRAME_CROP_BONUS_IN_FRAME_RATIO_MIN = 0.70
NON_ATTACHMENT_VISIBLE_BBOX_IN_FRAME_RATIO_MIN = 0.70
NON_ATTACHMENT_MIN_VISIBLE_BBOX_COUNT = 2
ATTACHMENT_PAIR_BBOX_IN_FRAME_RATIO_MIN = 0.50
ATTACHMENT_PAIR_BONUS_WEIGHT = 15


def visible_projected_area_px(width: int, height: int) -> float:
    return VISIBLE_PROJECTED_AREA_RATIO * float(width) * float(height)


def build_selector_visibility_audit_from_meta(
    meta: dict[str, Any],
    intrinsics: CameraIntrinsics,
    *,
    margin: int | None = None,
    min_depth: float = 0.3,
    max_depth: float = 8.0,
) -> dict[str, Any]:
    """Explain whether an object passes selector visibility gating."""
    if margin is None:
        margin = default_edge_margin_px(intrinsics)
    center_uv_px = meta.get("center_uv_px")
    uv = None
    if isinstance(center_uv_px, (list, tuple)) and len(center_uv_px) == 2:
        try:
            uv = np.array([float(center_uv_px[0]), float(center_uv_px[1])], dtype=np.float64)
        except (TypeError, ValueError):
            uv = None

    depth_m = float(meta.get("depth_m", 0.0) or 0.0)
    depth_in_range = bool(min_depth < depth_m <= max_depth)
    center_in_image_margin = is_in_image(uv, intrinsics, margin=margin)
    bbox_in_frame_ratio = float(meta.get("bbox_in_frame_ratio", 0.0) or 0.0)
    projected_area_px = float(meta.get("projected_area_px", 0.0) or 0.0)

    decision = "rejected"
    selector_passed = False
    rejection_reasons: list[str] = []

    if not depth_in_range:
        rejection_reasons.append("depth_out_of_range")
    elif center_in_image_margin:
        decision = "selected_center"
        selector_passed = True
    elif (
        bbox_in_frame_ratio >= VISIBLE_BBOX_IN_FRAME_RATIO_MIN
        and projected_area_px >= visible_projected_area_px(intrinsics.width, intrinsics.height)
    ):
        decision = "selected_roi_fallback"
        selector_passed = True
    else:
        rejection_reasons.append("center_not_in_margin")
        if bbox_in_frame_ratio < VISIBLE_BBOX_IN_FRAME_RATIO_MIN:
            rejection_reasons.append("bbox_in_frame_ratio_below_threshold")
        if projected_area_px < visible_projected_area_px(intrinsics.width, intrinsics.height):
            rejection_reasons.append("projected_area_below_threshold")

    return {
        "center_depth_m": depth_m,
        "center_uv_px": [float(uv[0]), float(uv[1])] if uv is not None else None,
        "center_in_image_margin80": bool(center_in_image_margin),
        "bbox_in_frame_ratio": bbox_in_frame_ratio,
        "selector_roi_ratio_source": "bbox_projection",
        "projected_area_px": projected_area_px,
        "selector_passed": selector_passed,
        "selector_decision": decision,
        "selector_rejection_reasons": rejection_reasons,
    }


def _build_selector_visibility_meta(
    obj: dict[str, Any],
    pose: CameraPose,
    intrinsics: CameraIntrinsics,
    *,
    instance_mesh_data: InstanceMeshData | None = None,
    margin: int | None = None,
    min_depth: float = 0.3,
    max_depth: float = 8.0,
    include_roi_metrics: bool = False,
) -> dict[str, Any]:
    from .utils.coordinate_transform import project_camera_points_to_image

    if margin is None:
        margin = default_edge_margin_px(intrinsics)
    center = np.array(obj["center"], dtype=np.float64)
    point_cam = pose.world_to_camera_point(center)
    uv_all, depths = project_camera_points_to_image(point_cam.reshape(1, 3), intrinsics)
    depth = float(depths[0])
    uv = uv_all[0] if depth > 0 and np.isfinite(uv_all[0]).all() else None

    meta: dict[str, Any] = {
        "center_uv_px": [float(uv[0]), float(uv[1])] if uv is not None else None,
        "depth_m": depth,
        "bbox_in_frame_ratio": 0.0,
        "projected_area_px": 0.0,
    }
    depth_in_range = bool(min_depth < depth <= max_depth)
    center_in_image_margin = is_in_image(uv, intrinsics, margin=margin)
    if depth_in_range and (include_roi_metrics or not center_in_image_margin):
        roi_info = _project_object_roi(obj, pose, intrinsics)
        meta["bbox_in_frame_ratio"] = float(roi_info["bbox_in_frame_ratio"])
        meta["projected_area_px"] = float(roi_info["projected_area_px"])
    return meta


def build_selector_visibility_audit(
    obj: dict[str, Any],
    pose: CameraPose,
    intrinsics: CameraIntrinsics,
    *,
    instance_mesh_data: InstanceMeshData | None = None,
    margin: int | None = None,
    min_depth: float = 0.3,
    max_depth: float = 8.0,
    include_roi_metrics: bool = False,
) -> dict[str, Any]:
    """Project one object and explain selector visibility gating."""
    meta = _build_selector_visibility_meta(
        obj,
        pose,
        intrinsics,
        instance_mesh_data=instance_mesh_data,
        margin=margin,
        min_depth=min_depth,
        max_depth=max_depth,
        include_roi_metrics=include_roi_metrics,
    )
    return build_selector_visibility_audit_from_meta(
        meta,
        intrinsics,
        margin=margin,
        min_depth=min_depth,
        max_depth=max_depth,
    )


def get_visible_objects(
    objects: list[dict],
    pose: CameraPose,
    intrinsics: CameraIntrinsics,
    instance_mesh_data: InstanceMeshData | None = None,
    margin: int | None = None,
    min_depth: float = 0.3,
    max_depth: float = 8.0,
    return_audits: bool = False,
) -> list[dict] | tuple[list[dict], dict[int, dict[str, Any]]]:
    """Return objects whose projected footprint is meaningfully visible.

    The old centre-only rule was too strict for attachment/support relations:
    a large parent object can be clearly visible while its 3D centre lies just
    outside the image crop. Keep those cases when enough of the projected bbox
    remains inside the frame.

    margin:    pixels from image edge (larger = more conservative)
    min_depth: minimum distance from camera in metres (filters objects
               clipped right against the camera plane)
    max_depth: maximum distance from camera in metres. Objects beyond this
               threshold are filtered out even if their centre projects into
               the frame — this prevents objects in adjacent rooms (visible
               through a doorway as a tiny projection) from appearing in
               questions. Typical ScanNet indoor scenes are ≤8 m across;
               6 m is conservative enough to keep far-wall objects while
               excluding the next room.
    """
    visible = []
    visibility_audits_by_obj_id: dict[int, dict[str, Any]] = {}
    for obj in objects:
        audit = build_selector_visibility_audit(
            obj,
            pose,
            intrinsics,
            instance_mesh_data=instance_mesh_data,
            margin=margin,
            min_depth=min_depth,
            max_depth=max_depth,
            include_roi_metrics=return_audits,
        )
        if audit["selector_passed"]:
            visible.append(obj)
            if return_audits:
                visibility_audits_by_obj_id[int(obj["id"])] = audit
    if return_audits:
        return visible, visibility_audits_by_obj_id
    return visible


def _angular_distance(pose_a: CameraPose, pose_b: CameraPose) -> float:
    """Approximate angular difference between two viewing directions (degrees)."""
    fwd_a = pose_a.rotation.T[:, 2]
    fwd_b = pose_b.rotation.T[:, 2]
    cos_angle = np.clip(np.dot(fwd_a, fwd_b), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def _count_attachment_objects(visible: list[dict], attachment_ids: set[int]) -> int:
    return sum(1 for o in visible if o["id"] in attachment_ids)


def _count_visible_objects_with_min_bbox_in_frame_ratio(
    visible: list[dict],
    pose: CameraPose | None = None,
    intrinsics: CameraIntrinsics | None = None,
    *,
    bbox_in_frame_ratio_min: float,
    visibility_audits_by_obj_id: dict[int, dict[str, Any]] | None = None,
) -> int:
    count = 0
    for obj in visible:
        audit = (visibility_audits_by_obj_id or {}).get(int(obj["id"]))
        if audit is not None:
            bbox_in_frame_ratio = float(audit.get("bbox_in_frame_ratio", 0.0) or 0.0)
        else:
            if pose is None or intrinsics is None:
                raise ValueError(
                    "_count_well_cropped_visible_objects requires either precomputed audits "
                    "or both pose and intrinsics"
                )
            roi_info = _project_object_roi(obj, pose, intrinsics)
            bbox_in_frame_ratio = float(roi_info["bbox_in_frame_ratio"])
        if bbox_in_frame_ratio >= float(bbox_in_frame_ratio_min):
            count += 1
    return count


def _count_well_cropped_visible_objects(
    visible: list[dict],
    pose: CameraPose | None = None,
    intrinsics: CameraIntrinsics | None = None,
    *,
    visibility_audits_by_obj_id: dict[int, dict[str, Any]] | None = None,
) -> int:
    return _count_visible_objects_with_min_bbox_in_frame_ratio(
        visible,
        pose,
        intrinsics,
        bbox_in_frame_ratio_min=FRAME_CROP_BONUS_IN_FRAME_RATIO_MIN,
        visibility_audits_by_obj_id=visibility_audits_by_obj_id,
    )


def _count_well_cropped_attachment_pairs(
    visible: list[dict],
    attachment_graph: dict[int, list[int]] | None,
    pose: CameraPose | None = None,
    intrinsics: CameraIntrinsics | None = None,
    *,
    visibility_audits_by_obj_id: dict[int, dict[str, Any]] | None = None,
) -> int:
    if not attachment_graph:
        return 0

    visible_ids = {int(obj["id"]) for obj in visible}
    ratio_by_obj_id: dict[int, float] = {}
    for obj in visible:
        obj_id = int(obj["id"])
        audit = (visibility_audits_by_obj_id or {}).get(obj_id)
        if audit is not None:
            ratio_by_obj_id[obj_id] = float(audit.get("bbox_in_frame_ratio", 0.0) or 0.0)
            continue
        if pose is None or intrinsics is None:
            raise ValueError(
                "_count_well_cropped_attachment_pairs requires either precomputed audits "
                "or both pose and intrinsics"
            )
        roi_info = _project_object_roi(obj, pose, intrinsics)
        ratio_by_obj_id[obj_id] = float(roi_info.get("bbox_in_frame_ratio", 0.0) or 0.0)

    pair_count = 0
    for parent_id, child_ids in attachment_graph.items():
        parent_id_int = int(parent_id)
        if parent_id_int not in visible_ids:
            continue
        parent_ratio = float(ratio_by_obj_id.get(parent_id_int, 0.0) or 0.0)
        if parent_ratio < ATTACHMENT_PAIR_BBOX_IN_FRAME_RATIO_MIN:
            continue
        for child_id in child_ids:
            child_id_int = int(child_id)
            if child_id_int not in visible_ids:
                continue
            child_ratio = float(ratio_by_obj_id.get(child_id_int, 0.0) or 0.0)
            if child_ratio >= ATTACHMENT_PAIR_BBOX_IN_FRAME_RATIO_MIN:
                pair_count += 1
    return pair_count


def _frame_candidate_score(
    *,
    n_visible: int,
    n_attachment: int,
    crop_ge_70_count: int,
    attachment_pair_ge_50_count: int,
) -> tuple[int, int]:
    # Visibility count remains a hard gate upstream, but it no longer boosts
    # the candidate ranking itself.
    _ = int(n_visible)
    # Well-cropped objects still define the preferred candidate pool, but they
    # no longer add score once a frame is in that pool.
    _ = int(crop_ge_70_count)
    base_score = int(n_attachment)
    attachment_pair_bonus = int(attachment_pair_ge_50_count) * ATTACHMENT_PAIR_BONUS_WEIGHT
    return base_score, base_score + attachment_pair_bonus


def _legacy_select_frames(
    scene_path: str | Path,
    objects: list[dict],
    attachment_graph: dict[int, list[int]] | None = None,
    max_frames: int = DEFAULT_MAX_FRAMES,
    *,
    keep_all_attachment_frames: bool = False,
    non_attachment_limit: int | None = None,
    color_intrinsics: CameraIntrinsics | None = None,
    axis_alignment: np.ndarray | None = None,
    poses: dict[str, CameraPose] | None = None,
    instance_mesh_data: InstanceMeshData | None = None,
    preloaded_geometry: Any | None = None,
) -> list[dict[str, Any]]:
    """Select representative frames for a ScanNet scene.

    Algorithm:
        1. Coarsely reject only obviously blurry frames.
        2. If any frame has at least one well-cropped visible object
           (bbox_in_frame_ratio >= 0.7), restrict selection to that subset.
           Within the active candidate pool, rank frames by
           #attachment_objects + 15 × #well-cropped-attachment-pairs first,
           then by the number of well-cropped visible objects.
        3. Frames with at least one well-cropped attachment pair bypass the
           viewpoint-diversity pruning so the later referability stage can
           compare nearby views. Remaining frames are greedily filtered to keep
           only viewpoint-diverse candidates.
        4. Callers that need a larger referability review pool can keep all
           attachment-qualified frames while separately capping the number of
           non-attachment candidates.

    Returns:
        List of dicts: ``{image_name, camera_position, visible_object_ids,
        n_visible, score}``.
    """
    scene_path = Path(scene_path)

    intrinsics = color_intrinsics
    loaded_poses = poses
    axis_align = axis_alignment
    if intrinsics is None or loaded_poses is None:
        # Support both intrinsic/intrinsic_color.txt and intrinsic_color.txt (at root)
        intr_path = scene_path / "intrinsic" / "intrinsic_color.txt"
        if not intr_path.exists():
            intr_path = scene_path / "intrinsic_color.txt"
        pose_dir = scene_path / "pose"
        if (intrinsics is None and not intr_path.exists()) or (loaded_poses is None and not pose_dir.exists()):
            logger.warning("Intrinsic or pose directory missing for %s", scene_path.name)
            return []
    if intrinsics is None:
        intrinsics = load_scannet_intrinsics(scene_path)
    if axis_align is None:
        axis_align = load_axis_alignment(scene_path)
    if loaded_poses is None:
        loaded_poses = load_scannet_poses(scene_path, axis_alignment=axis_align)

    if not loaded_poses:
        logger.warning("No valid poses found for %s", scene_path.name)
        return []

    # Collect attachment-participating object IDs.
    attachment_ids: set[int] = set()
    if attachment_graph:
        for parent_id, children in attachment_graph.items():
            attachment_ids.add(int(parent_id))
            attachment_ids.update(children)

    # Score every frame — but stride to avoid processing thousands of frames.
    # ScanNet captures ~30 fps; stride=30 samples every ~1s which is still
    # dense enough to find diverse viewpoints while cutting runtime by ~30×.
    color_dir = scene_path / "color"
    n_quality_rejected = 0
    n_missing_images = 0
    frame_entries: list[dict[str, Any]] = []
    for i, (image_name, pose) in enumerate(loaded_poses.items()):
        if i % FRAME_STRIDE != 0:
            continue

        # Coarse image-quality gate — only reject obviously blurry frames.
        image_path = color_dir / image_name
        if not image_path.exists():
            n_missing_images += 1
            logger.debug("Missing color frame %s in %s", image_name, scene_path.name)
            continue
        if not passes_image_quality(image_path):
            n_quality_rejected += 1
            continue

        visible_result = get_visible_objects(
            objects,
            pose,
            intrinsics,
            return_audits=True,
        )
        if (
            isinstance(visible_result, tuple)
            and len(visible_result) == 2
            and isinstance(visible_result[1], dict)
        ):
            visible, visibility_audits_by_obj_id = visible_result
        else:
            visible = visible_result
            visibility_audits_by_obj_id = {}
        if len(visible) < MIN_VISIBLE_OBJECTS:
            continue

        n_attachment = _count_attachment_objects(visible, attachment_ids)
        crop_ge_70_count = _count_well_cropped_visible_objects(
            visible,
            visibility_audits_by_obj_id=visibility_audits_by_obj_id,
        )
        visible_bbox_ge_70_count = _count_visible_objects_with_min_bbox_in_frame_ratio(
            visible,
            pose,
            intrinsics,
            bbox_in_frame_ratio_min=NON_ATTACHMENT_VISIBLE_BBOX_IN_FRAME_RATIO_MIN,
            visibility_audits_by_obj_id=visibility_audits_by_obj_id,
        )
        attachment_pair_ge_50_count = _count_well_cropped_attachment_pairs(
            visible,
            attachment_graph,
            visibility_audits_by_obj_id=visibility_audits_by_obj_id,
        )
        base_score, score = _frame_candidate_score(
            n_visible=len(visible),
            n_attachment=n_attachment,
            crop_ge_70_count=crop_ge_70_count,
            attachment_pair_ge_50_count=attachment_pair_ge_50_count,
        )
        frame_entries.append(
            {
                "image_name":        image_name,
                "pose":              pose,
                "visible_object_ids": [o["id"] for o in visible],
                "n_visible":         len(visible),
                "base_score":        base_score,
                "crop_ge_70_count":  crop_ge_70_count,
                "visible_bbox_ge_70_count": visible_bbox_ge_70_count,
                "attachment_pair_ge_50_count": attachment_pair_ge_50_count,
                "score":             score,
            }
        )

    if n_quality_rejected:
        logger.info(
            "Rejected %d frames for low image quality in %s",
            n_quality_rejected, scene_path.name,
        )
    if n_missing_images:
        logger.warning(
            "Skipped %d frames with missing color images in %s",
            n_missing_images, scene_path.name,
        )

    if not frame_entries:
        logger.warning(
            "No frames with >= %d visible objects in %s",
            MIN_VISIBLE_OBJECTS, scene_path.name,
        )
        return []

    attachment_entries = [
        entry
        for entry in frame_entries
        if int(entry.get("attachment_pair_ge_50_count", 0) or 0) > 0
    ]
    non_attachment_candidates = [
        entry
        for entry in frame_entries
        if int(entry.get("attachment_pair_ge_50_count", 0) or 0) <= 0
    ]
    preferred_non_attachment_entries = [
        entry
        for entry in non_attachment_candidates
        if int(entry.get("visible_bbox_ge_70_count", 0) or 0)
        >= NON_ATTACHMENT_MIN_VISIBLE_BBOX_COUNT
    ]
    non_attachment_entries = (
        preferred_non_attachment_entries
        if preferred_non_attachment_entries
        else non_attachment_candidates
    )
    if preferred_non_attachment_entries:
        logger.info(
            "Restricting non-attachment frame selection for %s to %d/%d candidates with at least %d visible objects at bbox_in_frame_ratio >= %.2f",
            scene_path.name,
            len(preferred_non_attachment_entries),
            len(non_attachment_candidates),
            NON_ATTACHMENT_MIN_VISIBLE_BBOX_COUNT,
            NON_ATTACHMENT_VISIBLE_BBOX_IN_FRAME_RATIO_MIN,
        )

    attachment_entries.sort(
        key=lambda e: (
            int(e.get("score", 0) or 0),
            int(e.get("crop_ge_70_count", 0) or 0),
        ),
        reverse=True,
    )
    non_attachment_entries.sort(
        key=lambda e: (
            int(e.get("score", 0) or 0),
            int(e.get("crop_ge_70_count", 0) or 0),
        ),
        reverse=True,
    )
    selection_pool = attachment_entries + non_attachment_entries
    selection_limit = max(1, int(max_frames))

    selected: list[dict[str, Any]]
    if keep_all_attachment_frames:
        selected = list(attachment_entries)
    else:
        selected = attachment_entries[:selection_limit]

    if non_attachment_limit is None:
        if keep_all_attachment_frames:
            non_attachment_target = len(non_attachment_entries)
        else:
            non_attachment_target = max(0, selection_limit - len(selected))
    else:
        non_attachment_target = max(0, int(non_attachment_limit))

    selected_non_attachment = non_attachment_entries[:non_attachment_target]
    selected.extend(selected_non_attachment)

    if len(selected) < len(selection_pool):
        logger.info(
            "Selected %d/%d frame candidates for %s after attachment-aware viewpoint filtering",
            len(selected),
            len(selection_pool),
            scene_path.name,
        )

    results = []
    for s in selected:
        results.append(
            {
                "image_name":        s["image_name"],
                "camera_position":   s["pose"].position.tolist(),
                "visible_object_ids": s["visible_object_ids"],
                "n_visible":         s["n_visible"],
                "base_score":        s["base_score"],
                "crop_ge_70_count":  s["crop_ge_70_count"],
                "visible_bbox_ge_70_count": s["visible_bbox_ge_70_count"],
                "attachment_pair_ge_50_count": s["attachment_pair_ge_50_count"],
                "attachment_viewpoint_exempt": bool(
                    int(s.get("attachment_pair_ge_50_count", 0) or 0) > 0
                ),
                "score":             s["score"],
            }
        )

    logger.info(
        "Selected %d frames for %s (top score=%d)",
        len(results), scene_path.name,
        results[0]["score"] if results else 0,
    )
    return results


def select_frames(
    scene_path: str | Path,
    objects: list[dict],
    attachment_graph: dict[int, list[int]] | None = None,
    max_frames: int = DEFAULT_MAX_FRAMES,
    *,
    keep_all_attachment_frames: bool = False,
    non_attachment_limit: int | None = None,
    color_intrinsics: CameraIntrinsics | None = None,
    axis_alignment: np.ndarray | None = None,
    poses: dict[str, CameraPose] | None = None,
    instance_mesh_data: InstanceMeshData | None = None,
    preloaded_geometry: Any | None = None,
    data_source: Any | None = None,
) -> list[dict[str, Any]]:
    """Select representative frames for a scene.

    When *data_source* is provided, camera data and image paths are resolved
    through it (supports ScanNet++).  When ``None``, the existing ScanNet v2
    file-based loading is used (backward compatible).
    """
    scene_path = Path(scene_path)

    intrinsics = color_intrinsics
    loaded_poses = poses
    axis_align = axis_alignment
    color_dir: Path | None = None
    _resolve_image_path = None  # callable or None

    if data_source is not None:
        if intrinsics is None:
            intrinsics = data_source.load_intrinsics()
        if axis_align is None:
            axis_align = data_source.load_axis_alignment()
        if loaded_poses is None:
            loaded_poses = data_source.load_poses()
        # image path resolution via data_source
        def _resolve_image_path(name: str) -> Path:
            return data_source.image_path(name)
    else:
        # ScanNet v2 legacy path
        if intrinsics is None or loaded_poses is None:
            intr_path = scene_path / "intrinsic" / "intrinsic_color.txt"
            if not intr_path.exists():
                intr_path = scene_path / "intrinsic_color.txt"
            pose_dir = scene_path / "pose"
            if (intrinsics is None and not intr_path.exists()) or (loaded_poses is None and not pose_dir.exists()):
                logger.warning("Intrinsic or pose directory missing for %s", scene_path.name)
                return []
        if intrinsics is None:
            intrinsics = load_scannet_intrinsics(scene_path)
        if axis_align is None:
            axis_align = load_axis_alignment(scene_path)
        if loaded_poses is None:
            loaded_poses = load_scannet_poses(scene_path, axis_alignment=axis_align)
        color_dir = scene_path / "color"
        def _resolve_image_path(name: str) -> Path:
            return color_dir / name

    if not loaded_poses:
        logger.warning("No valid poses found for %s", scene_path.name)
        return []

    attachment_ids: set[int] = set()
    if attachment_graph:
        for parent_id, children in attachment_graph.items():
            attachment_ids.add(int(parent_id))
            attachment_ids.update(children)

    n_sampled_frames = 0
    n_missing_images = 0
    sampled_frames: list[dict[str, Any]] = []
    for i, (image_name, pose) in enumerate(loaded_poses.items()):
        if i % FRAME_STRIDE != 0:
            continue

        n_sampled_frames += 1
        image_path = _resolve_image_path(image_name)
        if not image_path.exists():
            n_missing_images += 1
            logger.debug("Missing color frame %s in %s", image_name, scene_path.name)
            continue
        sampled_frames.append(
            {
                "image_name": image_name,
                "pose": pose,
                "image_path": image_path,
                "sampled_index": n_sampled_frames - 1,
            }
        )

    quality_kept_frames, quality_stats = _filter_sampled_frames_by_quality(sampled_frames)
    logger.info(
        "Frame quality prefilter for %s: sampled=%d missing=%d unreadable=%d stage1_rejected=%d stage2_rejected=%d kept=%d",
        scene_path.name,
        n_sampled_frames,
        n_missing_images,
        int(quality_stats["unreadable_count"]),
        int(quality_stats["stage1_rejected_count"]),
        int(quality_stats["stage2_rejected_count"]),
        int(quality_stats["final_quality_kept_count"]),
    )

    if n_missing_images:
        logger.warning(
            "Skipped %d frames with missing color images in %s",
            n_missing_images, scene_path.name,
        )

    if not quality_kept_frames:
        logger.warning("No quality-kept sampled frames in %s", scene_path.name)
        return []

    frame_entries: list[dict[str, Any]] = []
    for quality_entry in quality_kept_frames:
        image_name = str(quality_entry["image_name"])
        pose = quality_entry["pose"]
        visible_result = get_visible_objects(
            objects,
            pose,
            intrinsics,
            return_audits=True,
        )
        if (
            isinstance(visible_result, tuple)
            and len(visible_result) == 2
            and isinstance(visible_result[1], dict)
        ):
            visible, visibility_audits_by_obj_id = visible_result
        else:
            visible = visible_result
            visibility_audits_by_obj_id = {}
        if len(visible) < MIN_VISIBLE_OBJECTS:
            continue

        n_attachment = _count_attachment_objects(visible, attachment_ids)
        crop_ge_70_count = _count_well_cropped_visible_objects(
            visible,
            visibility_audits_by_obj_id=visibility_audits_by_obj_id,
        )
        visible_bbox_ge_70_count = _count_visible_objects_with_min_bbox_in_frame_ratio(
            visible,
            pose,
            intrinsics,
            bbox_in_frame_ratio_min=NON_ATTACHMENT_VISIBLE_BBOX_IN_FRAME_RATIO_MIN,
            visibility_audits_by_obj_id=visibility_audits_by_obj_id,
        )
        attachment_pair_ge_50_count = _count_well_cropped_attachment_pairs(
            visible,
            attachment_graph,
            visibility_audits_by_obj_id=visibility_audits_by_obj_id,
        )
        base_score, score = _frame_candidate_score(
            n_visible=len(visible),
            n_attachment=n_attachment,
            crop_ge_70_count=crop_ge_70_count,
            attachment_pair_ge_50_count=attachment_pair_ge_50_count,
        )
        frame_entries.append(
            {
                "image_name": image_name,
                "pose": pose,
                "visible_object_ids": [o["id"] for o in visible],
                "n_visible": len(visible),
                "base_score": base_score,
                "crop_ge_70_count": crop_ge_70_count,
                "visible_bbox_ge_70_count": visible_bbox_ge_70_count,
                "attachment_pair_ge_50_count": attachment_pair_ge_50_count,
                "score": score,
            }
        )

    if not frame_entries:
        logger.warning(
            "No frames with >= %d visible objects in %s",
            MIN_VISIBLE_OBJECTS, scene_path.name,
        )
        return []

    attachment_entries = [
        entry
        for entry in frame_entries
        if int(entry.get("attachment_pair_ge_50_count", 0) or 0) > 0
    ]
    non_attachment_candidates = [
        entry
        for entry in frame_entries
        if int(entry.get("attachment_pair_ge_50_count", 0) or 0) <= 0
    ]
    preferred_non_attachment_entries = [
        entry
        for entry in non_attachment_candidates
        if int(entry.get("visible_bbox_ge_70_count", 0) or 0)
        >= NON_ATTACHMENT_MIN_VISIBLE_BBOX_COUNT
    ]
    non_attachment_entries = (
        preferred_non_attachment_entries
        if preferred_non_attachment_entries
        else non_attachment_candidates
    )
    if preferred_non_attachment_entries:
        logger.info(
            "Restricting non-attachment frame selection for %s to %d/%d candidates with at least %d visible objects at bbox_in_frame_ratio >= %.2f",
            scene_path.name,
            len(preferred_non_attachment_entries),
            len(non_attachment_candidates),
            NON_ATTACHMENT_MIN_VISIBLE_BBOX_COUNT,
            NON_ATTACHMENT_VISIBLE_BBOX_IN_FRAME_RATIO_MIN,
        )

    attachment_entries.sort(
        key=lambda e: (
            int(e.get("score", 0) or 0),
            int(e.get("crop_ge_70_count", 0) or 0),
        ),
        reverse=True,
    )
    non_attachment_entries.sort(
        key=lambda e: (
            int(e.get("score", 0) or 0),
            int(e.get("crop_ge_70_count", 0) or 0),
        ),
        reverse=True,
    )
    selection_pool = attachment_entries + non_attachment_entries
    selection_limit = max(1, int(max_frames))

    selected: list[dict[str, Any]]
    if keep_all_attachment_frames:
        selected = list(attachment_entries)
    else:
        selected = attachment_entries[:selection_limit]

    if non_attachment_limit is None:
        if keep_all_attachment_frames:
            non_attachment_target = len(non_attachment_entries)
        else:
            non_attachment_target = max(0, selection_limit - len(selected))
    else:
        non_attachment_target = max(0, int(non_attachment_limit))

    selected_non_attachment = non_attachment_entries[:non_attachment_target]
    selected.extend(selected_non_attachment)

    if len(selected) < len(selection_pool):
        logger.info(
            "Selected %d/%d frame candidates for %s after attachment-aware viewpoint filtering",
            len(selected),
            len(selection_pool),
            scene_path.name,
        )

    results = []
    for selected_entry in selected:
        results.append(
            {
                "image_name": selected_entry["image_name"],
                "camera_position": selected_entry["pose"].position.tolist(),
                "visible_object_ids": selected_entry["visible_object_ids"],
                "n_visible": selected_entry["n_visible"],
                "base_score": selected_entry["base_score"],
                "crop_ge_70_count": selected_entry["crop_ge_70_count"],
                "visible_bbox_ge_70_count": selected_entry["visible_bbox_ge_70_count"],
                "attachment_pair_ge_50_count": selected_entry["attachment_pair_ge_50_count"],
                "attachment_viewpoint_exempt": bool(
                    int(selected_entry.get("attachment_pair_ge_50_count", 0) or 0) > 0
                ),
                "score": selected_entry["score"],
            }
        )

    logger.info(
        "Selected %d frames for %s (top score=%d)",
        len(results), scene_path.name,
        results[0]["score"] if results else 0,
    )
    return results
