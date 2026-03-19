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
        1. Sharpness — Laplacian variance on Gaussian-denoised image
           (Gaussian pre-filter kills sensor noise / compression artifacts
           that would otherwise inflate the variance on blurry frames).
        2. Brightness — grayscale mean (filters underexposed / overexposed).
        3. Contrast — grayscale stddev (filters hazy / low-contrast frames).
    """
    img = cv2.imread(str(image_path))
    if img is None:
        logger.debug("Cannot read image %s — failing quality check", image_path)
        return False

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Sharpness: Gaussian blur first to suppress sensor noise / compression
    # artifacts, then compute Laplacian variance on the cleaned image.
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
#  Per-object local sharpness check (ROI blur filter)
# ---------------------------------------------------------------------------

LOCAL_SHARPNESS_MIN = 30.0  # Laplacian variance within object ROI


def filter_blurry_objects(
    visible_object_ids: list[int],
    objects: list[dict],
    pose: CameraPose,
    intrinsics: CameraIntrinsics,
    image_path: Path,
) -> list[int]:
    """Remove objects whose local image region is too blurry.

    For each visible object, projects its 3D bbox onto the colour image to
    get a 2D ROI, then computes the Laplacian variance within that ROI.
    Objects below LOCAL_SHARPNESS_MIN are dropped — even if the global image
    is sharp, these objects may be out of focus or motion-blurred locally.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        return visible_object_ids
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.GaussianBlur(gray, (3, 3), 0)
    h, w = gray.shape[:2]

    obj_map = {o["id"]: o for o in objects}
    kept: list[int] = []

    for obj_id in visible_object_ids:
        obj = obj_map.get(obj_id)
        if obj is None:
            continue

        # Project bbox corners + centre to 2D to find the ROI
        bbox_min = np.array(obj["bbox_min"])
        bbox_max = np.array(obj["bbox_max"])
        mid = (bbox_min + bbox_max) / 2.0
        corners_3d = []
        for x in [bbox_min[0], bbox_max[0]]:
            for y in [bbox_min[1], bbox_max[1]]:
                for z in [bbox_min[2], bbox_max[2]]:
                    corners_3d.append([x, y, z])
        corners_3d.append(mid.tolist())

        us, vs = [], []
        for pt in corners_3d:
            uv, depth = project_to_image(np.array(pt), pose, intrinsics)
            if uv is not None and depth > 0:
                us.append(int(round(uv[0])))
                vs.append(int(round(uv[1])))

        if len(us) < 2:
            kept.append(obj_id)  # can't compute ROI — keep as-is
            continue

        # Clamp to image bounds with a small padding
        u_min = max(0, min(us) - 5)
        u_max = min(w, max(us) + 5)
        v_min = max(0, min(vs) - 5)
        v_max = min(h, max(vs) + 5)

        if u_max - u_min < 10 or v_max - v_min < 10:
            kept.append(obj_id)  # ROI too small to measure
            continue

        roi = denoised[v_min:v_max, u_min:u_max]
        roi_var = cv2.Laplacian(roi, cv2.CV_64F).var()

        if roi_var >= LOCAL_SHARPNESS_MIN:
            kept.append(obj_id)
        else:
            logger.debug(
                "Object %d (%s) locally blurry (ROI Laplacian var=%.1f < %.1f)",
                obj_id, obj.get("label", "?"), roi_var, LOCAL_SHARPNESS_MIN,
            )

    return kept


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


def get_visible_objects(
    objects: list[dict],
    pose: CameraPose,
    intrinsics: CameraIntrinsics,
    margin: int = 80,
    min_depth: float = 0.3,
    max_depth: float = 6.0,
) -> list[dict]:
    """Return objects whose centre projects into the image frame.

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
    for obj in objects:
        center = np.array(obj["center"])
        uv, depth = project_to_image(center, pose, intrinsics)
        if min_depth < depth <= max_depth and is_in_image(uv, intrinsics, margin=margin):
            visible.append(obj)
    return visible


def _angular_distance(pose_a: CameraPose, pose_b: CameraPose) -> float:
    """Approximate angular difference between two viewing directions (degrees)."""
    fwd_a = pose_a.rotation.T[:, 2]
    fwd_b = pose_b.rotation.T[:, 2]
    cos_angle = np.clip(np.dot(fwd_a, fwd_b), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def _count_support_objects(visible: list[dict], support_ids: set[int]) -> int:
    return sum(1 for o in visible if o["id"] in support_ids)


def select_frames(
    scene_path: str | Path,
    objects: list[dict],
    support_graph: dict[int, list[int]] | None = None,
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

    # Collect support-participating object IDs
    support_ids: set[int] = set()
    if support_graph:
        for parent_id, children in support_graph.items():
            support_ids.add(int(parent_id))
            support_ids.update(children)

    # Score every frame
    color_dir = scene_path / "color"
    n_quality_rejected = 0
    frame_entries: list[dict[str, Any]] = []
    for image_name, pose in poses.items():
        visible = get_visible_objects(objects, pose, intrinsics)
        if len(visible) < MIN_VISIBLE_OBJECTS:
            continue

        # Image quality gate — reject blurry / dark / overexposed frames
        image_path = color_dir / image_name
        if image_path.exists() and not passes_image_quality(image_path):
            n_quality_rejected += 1
            continue

        n_support = _count_support_objects(visible, support_ids)
        score = len(visible) * (1 + n_support)
        frame_entries.append(
            {
                "image_name":        image_name,
                "pose":              pose,
                "visible_object_ids": [o["id"] for o in visible],
                "n_visible":         len(visible),
                "score":             score,
            }
        )

    if n_quality_rejected:
        logger.info(
            "Rejected %d frames for low image quality in %s",
            n_quality_rejected, scene_path.name,
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

    results = []
    for s in selected:
        results.append(
            {
                "image_name":        s["image_name"],
                "camera_position":   s["pose"].position.tolist(),
                "visible_object_ids": s["visible_object_ids"],
                "n_visible":         s["n_visible"],
                "score":             s["score"],
            }
        )

    logger.info(
        "Selected %d frames for %s (top score=%d)",
        len(results), scene_path.name,
        results[0]["score"] if results else 0,
    )
    return results
