"""Stage 3: Representative frame selection.

For each scene, select 3-5 frames that collectively cover as many objects
(and support relationships) as possible while maintaining viewpoint diversity.

ScanNet stores one colour image per frame in ``color/<frame_id>.jpg`` and the
corresponding camera pose in ``pose/<frame_id>.txt``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .utils.colmap_loader import (
    CameraIntrinsics,
    CameraPose,
    load_axis_alignment,
    load_scannet_intrinsics,
    load_scannet_poses,
)
from .utils.coordinate_transform import is_in_image, project_to_image

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  Image quality gate
# ---------------------------------------------------------------------------

# Thresholds (tuned for ScanNet 640×480 colour frames)
SHARPNESS_MIN = 50.0       # Laplacian variance; below → motion blur / out-of-focus
BRIGHTNESS_MIN = 30.0      # mean grayscale; below → too dark
BRIGHTNESS_MAX = 235.0     # mean grayscale; above → overexposed
CONTRAST_MIN = 25.0        # grayscale stddev; below → hazy / washed-out


def passes_image_quality(image_path: Path) -> bool:
    """Return True if the image at *image_path* passes quality checks.

    Checks:
        1. Sharpness — Laplacian variance on Gaussian-denoised image.
        2. Brightness — grayscale mean (filters underexposed / overexposed).
        3. Contrast — grayscale stddev (filters hazy / low-contrast frames).

    Reads at 1/4 resolution to reduce I/O and compute cost.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        logger.debug("Cannot read image %s — failing quality check", image_path)
        return False

    # Downsample to 1/4 size for speed. These thresholds are calibrated for the
    # reduced-resolution image and are not resolution-invariant.
    small = cv2.resize(img, (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    # Sharpness: Gaussian blur first to suppress sensor noise, then Laplacian
    denoised = cv2.GaussianBlur(gray, (3, 3), 0)
    laplacian_var = cv2.Laplacian(denoised, cv2.CV_64F).var()
    if laplacian_var < SHARPNESS_MIN:
        logger.debug(
            "Image %s too blurry (Laplacian var=%.1f < %.1f)",
            image_path.name, laplacian_var, SHARPNESS_MIN,
        )
        return False

    # Brightness: grayscale mean
    mean_brightness = float(gray.mean())
    if mean_brightness < BRIGHTNESS_MIN:
        logger.debug(
            "Image %s too dark (mean=%.1f < %.1f)",
            image_path.name, mean_brightness, BRIGHTNESS_MIN,
        )
        return False
    if mean_brightness > BRIGHTNESS_MAX:
        logger.debug(
            "Image %s overexposed (mean=%.1f > %.1f)",
            image_path.name, mean_brightness, BRIGHTNESS_MAX,
        )
        return False

    # Contrast: grayscale standard deviation
    contrast = float(gray.std())
    if contrast < CONTRAST_MIN:
        logger.debug(
            "Image %s low contrast (std=%.1f < %.1f)",
            image_path.name, contrast, CONTRAST_MIN,
        )
        return False

    return True


# ---------------------------------------------------------------------------
#  Strict visibility constants (used by compute_frame_object_visibility)
# ---------------------------------------------------------------------------

STRICT_VISIBLE_RATIO_MIN = 0.6
STRICT_PROJECTED_AREA_MIN = 800.0
STRICT_IN_FRAME_RATIO_MIN = 0.6
STRICT_EDGE_MARGIN_MIN = 12.0
STRICT_LOCAL_SHARPNESS_MIN = 45.0


def _project_object_roi(
    obj: dict,
    pose: CameraPose,
    intrinsics: CameraIntrinsics,
) -> dict[str, Any]:
    """Project an object's bbox to image space and summarise the footprint."""
    bbox_min = np.array(obj["bbox_min"], dtype=np.float64)
    bbox_max = np.array(obj["bbox_max"], dtype=np.float64)
    mid = (bbox_min + bbox_max) / 2.0

    sample_points = []
    for x in [bbox_min[0], bbox_max[0]]:
        for y in [bbox_min[1], bbox_max[1]]:
            for z in [bbox_min[2], bbox_max[2]]:
                sample_points.append(np.array([x, y, z], dtype=np.float64))
    sample_points.append(mid)

    projected: list[tuple[float, float]] = []
    in_frame = 0
    valid = 0
    for pt in sample_points:
        uv, depth = project_to_image(pt, pose, intrinsics)
        if uv is None or depth <= 0:
            continue
        valid += 1
        u, v = float(uv[0]), float(uv[1])
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


def compute_frame_object_visibility(
    objects: list[dict],
    pose: CameraPose,
    color_intrinsics: CameraIntrinsics,
    image_path: Path | None = None,
    depth_image=None,
    depth_intrinsics: CameraIntrinsics | None = None,
    margin: int = 80,
    min_depth: float = 0.3,
    max_depth: float = 6.0,
    strict_mode: bool = False,
) -> dict[int, dict[str, Any]]:
    """Compute per-object visibility metadata for a single frame."""
    from .utils.depth_occlusion import compute_depth_occlusion

    gray = None
    if image_path is not None and image_path.exists():
        img = cv2.imread(str(image_path))
        if img is not None:
            gray = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (3, 3), 0)

    visibility: dict[int, dict[str, Any]] = {}
    for obj in objects:
        center = np.array(obj["center"], dtype=np.float64)
        uv, depth = project_to_image(center, pose, color_intrinsics)
        center_in_frame = (
            min_depth < depth <= max_depth
            and is_in_image(uv, color_intrinsics, margin=margin)
        )

        roi_info = _project_object_roi(obj, pose, color_intrinsics)
        roi_sharpness = _compute_roi_sharpness(gray, roi_info["roi_bounds"])

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
            "center_uv_px": (
                [float(uv[0]), float(uv[1])]
                if uv is not None else None
            ),
            "depth_m": float(depth),
            "occlusion_status": occlusion_status,
            "visible_ratio": float(visible_ratio),
            "valid_projection_count": int(roi_info["valid_projection_count"]),
            "projected_area_px": float(roi_info["projected_area_px"]),
            "bbox_in_frame_ratio": float(roi_info["bbox_in_frame_ratio"]),
            "edge_margin_px": float(roi_info["edge_margin_px"]),
            "roi_bounds_px": (
                [int(v) for v in roi_info["roi_bounds"]]
                if roi_info["roi_bounds"] is not None else None
            ),
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
        if strict_mode:
            if depth_image is None or depth_intrinsics is None:
                reasons.append("missing_depth")
            elif meta["occlusion_status"] == "not visible":
                reasons.append("depth_occluded")
            elif meta["visible_ratio"] < STRICT_VISIBLE_RATIO_MIN:
                reasons.append("low_visible_ratio")
            if meta["projected_area_px"] < STRICT_PROJECTED_AREA_MIN:
                reasons.append("small_projection")
            if meta["bbox_in_frame_ratio"] < STRICT_IN_FRAME_RATIO_MIN:
                reasons.append("bbox_cut_off")
            if meta["edge_margin_px"] < STRICT_EDGE_MARGIN_MIN:
                reasons.append("too_close_to_edge")
            if meta["roi_sharpness"] is None:
                reasons.append("missing_roi_sharpness")
            elif meta["roi_sharpness"] < STRICT_LOCAL_SHARPNESS_MIN:
                reasons.append("blurry_roi")
            if not meta["label_unique_in_frame"]:
                reasons.append("duplicate_label")

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
VISIBLE_BBOX_IN_FRAME_RATIO_MIN = 0.35
VISIBLE_PROJECTED_AREA_MIN = 400.0
FRAME_CROP_BONUS_IN_FRAME_RATIO_MIN = 0.60
FRAME_CROP_BONUS_WEIGHT = 10


def build_selector_visibility_audit_from_meta(
    meta: dict[str, Any],
    intrinsics: CameraIntrinsics,
    *,
    margin: int = 80,
    min_depth: float = 0.3,
    max_depth: float = 6.0,
) -> dict[str, Any]:
    """Explain whether an object passes selector visibility gating."""
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
        and projected_area_px >= VISIBLE_PROJECTED_AREA_MIN
    ):
        decision = "selected_roi_fallback"
        selector_passed = True
    else:
        rejection_reasons.append("center_not_in_margin")
        if bbox_in_frame_ratio < VISIBLE_BBOX_IN_FRAME_RATIO_MIN:
            rejection_reasons.append("bbox_in_frame_ratio_below_threshold")
        if projected_area_px < VISIBLE_PROJECTED_AREA_MIN:
            rejection_reasons.append("projected_area_below_threshold")

    return {
        "center_depth_m": depth_m,
        "center_uv_px": [float(uv[0]), float(uv[1])] if uv is not None else None,
        "center_in_image_margin80": bool(center_in_image_margin),
        "bbox_in_frame_ratio": bbox_in_frame_ratio,
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
    margin: int = 80,
    min_depth: float = 0.3,
    max_depth: float = 6.0,
    include_roi_metrics: bool = False,
) -> dict[str, Any]:
    center = np.array(obj["center"], dtype=np.float64)
    uv, depth = project_to_image(center, pose, intrinsics)

    meta: dict[str, Any] = {
        "center_uv_px": [float(uv[0]), float(uv[1])] if uv is not None else None,
        "depth_m": float(depth),
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
    margin: int = 80,
    min_depth: float = 0.3,
    max_depth: float = 6.0,
    include_roi_metrics: bool = False,
) -> dict[str, Any]:
    """Project one object and explain selector visibility gating."""
    meta = _build_selector_visibility_meta(
        obj,
        pose,
        intrinsics,
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
    margin: int = 80,
    min_depth: float = 0.3,
    max_depth: float = 6.0,
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
    """Approximate angular difference between two viewing directions (degrees).

    This ignores camera translation and only measures heading diversity.
    """
    fwd_a = pose_a.rotation.T[:, 2]
    fwd_b = pose_b.rotation.T[:, 2]
    cos_angle = np.clip(np.dot(fwd_a, fwd_b), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def _count_attachment_objects(visible: list[dict], attachment_ids: set[int]) -> int:
    return sum(1 for o in visible if o["id"] in attachment_ids)


def _count_well_cropped_visible_objects(
    visible: list[dict],
    pose: CameraPose | None = None,
    intrinsics: CameraIntrinsics | None = None,
    *,
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
        if bbox_in_frame_ratio >= FRAME_CROP_BONUS_IN_FRAME_RATIO_MIN:
            count += 1
    return count


def _frame_candidate_score(
    *,
    n_visible: int,
    n_attachment: int,
    crop_ge_60_count: int,
) -> tuple[int, int]:
    base_score = int(n_visible) * (1 + int(n_attachment))
    crop_bonus = int(crop_ge_60_count) * FRAME_CROP_BONUS_WEIGHT
    return base_score, base_score + crop_bonus


def select_frames(
    scene_path: str | Path,
    objects: list[dict],
    attachment_graph: dict[int, list[int]] | None = None,
    max_frames: int = DEFAULT_MAX_FRAMES,
) -> list[dict[str, Any]]:
    """Select representative frames for a ScanNet scene.

    Algorithm:
        1. Score each frame = #visible_objects × (1 + #support_objects).
        2. Greedy selection: pick the highest-scoring frame, then iteratively
           pick the next that is at least VIEWPOINT_DIVERSITY_MIN_ANGLE away
           from all already-selected frames.

    Returns:
        List of dicts: ``{image_name, camera_position, visible_object_ids,
        n_visible, score}``.
    """
    scene_path = Path(scene_path)

    # Support both intrinsic/intrinsic_color.txt and intrinsic_color.txt (at root)
    intr_path = scene_path / "intrinsic" / "intrinsic_color.txt"
    if not intr_path.exists():
        intr_path = scene_path / "intrinsic_color.txt"
    pose_dir  = scene_path / "pose"

    if not intr_path.exists() or not pose_dir.exists():
        logger.warning("Intrinsic or pose directory missing for %s", scene_path.name)
        return []

    intrinsics = load_scannet_intrinsics(scene_path)
    axis_align = load_axis_alignment(scene_path)
    poses      = load_scannet_poses(scene_path, axis_alignment=axis_align)

    if not poses:
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
    FRAME_STRIDE = 30
    color_dir = scene_path / "color"
    n_quality_rejected = 0
    n_missing_images = 0
    frame_entries: list[dict[str, Any]] = []
    for i, (image_name, pose) in enumerate(poses.items()):
        if i % FRAME_STRIDE != 0:
            continue

        # Image quality gate — reject blurry / dark / overexposed frames
        image_path = color_dir / image_name
        if not image_path.exists():
            n_missing_images += 1
            logger.debug("Missing color frame %s in %s", image_name, scene_path.name)
            continue
        if not passes_image_quality(image_path):
            n_quality_rejected += 1
            continue

        visible, visible_audits = get_visible_objects(
            objects,
            pose,
            intrinsics,
            return_audits=True,
        )
        if len(visible) < MIN_VISIBLE_OBJECTS:
            continue

        n_attachment = _count_attachment_objects(visible, attachment_ids)
        crop_ge_60_count = _count_well_cropped_visible_objects(
            visible,
            visibility_audits_by_obj_id=visible_audits,
        )
        base_score, score = _frame_candidate_score(
            n_visible=len(visible),
            n_attachment=n_attachment,
            crop_ge_60_count=crop_ge_60_count,
        )
        frame_entries.append(
            {
                "image_name":        image_name,
                "pose":              pose,
                "visible_object_ids": [o["id"] for o in visible],
                "n_visible":         len(visible),
                "base_score":        base_score,
                "crop_ge_60_count":  crop_ge_60_count,
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

    # Greedy diverse selection
    frame_entries.sort(key=lambda e: e["score"], reverse=True)
    selected: list[dict[str, Any]] = [frame_entries[0]]

    for entry in frame_entries[1:]:
        if len(selected) >= max_frames:
            break
        too_close = any(
            _angular_distance(entry["pose"], sel["pose"]) < VIEWPOINT_DIVERSITY_MIN_ANGLE
            for sel in selected
        )
        if not too_close:
            selected.append(entry)

    if len(selected) < min(max_frames, len(frame_entries)):
        logger.info(
            "Selected %d/%d frames for %s after viewpoint-diversity filtering",
            len(selected),
            min(max_frames, len(frame_entries)),
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
                "crop_ge_60_count":  s["crop_ge_60_count"],
                "score":             s["score"],
            }
        )

    logger.info(
        "Selected %d frames for %s (top score=%d)",
        len(results), scene_path.name,
        results[0]["score"] if results else 0,
    )
    return results
