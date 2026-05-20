#!/usr/bin/env python3
"""Diagnose ScanNet++ DSLR camera projection without running VLM inference.

Projects 3D object bounding-boxes into sample DSLR frames and reports
statistics that help catch coordinate-system mismatches before entering
the full frame-selection pipeline.

Usage::

    python scripts/diagnose_scannetpp_dslr_camera.py \
        --data_root ++data --scene 0d2ee665be --max_frames 5

The script uses the native ScanNet++ DSLR camera model.  OPENCV_FISHEYE
frames are projected with cv2.fisheye.projectPoints so debug overlays match
the curved resized images.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _aabb_corners(bb_min: np.ndarray, bb_max: np.ndarray) -> np.ndarray:
    """Return 8 AABB corner points (shape 8×3)."""
    return np.array([
        [bb_min[0], bb_min[1], bb_min[2]],
        [bb_min[0], bb_min[1], bb_max[2]],
        [bb_min[0], bb_max[1], bb_min[2]],
        [bb_min[0], bb_max[1], bb_max[2]],
        [bb_max[0], bb_min[1], bb_min[2]],
        [bb_max[0], bb_min[1], bb_max[2]],
        [bb_max[0], bb_max[1], bb_min[2]],
        [bb_max[0], bb_max[1], bb_max[2]],
    ], dtype=np.float64)


def _project_corners(
    corners: np.ndarray,
    pose,
    camera,
) -> tuple[list, list[tuple[float, float] | None]]:
    """Project 8 AABB corners.  Returns (uv_list, depth_list)."""
    from src.datasets.scannetpp import project_scannetpp_dslr_point

    uv_list: list[tuple[float, float] | None] = []
    depth_list: list[float] = []
    for corner in corners:
        uv, depth = project_scannetpp_dslr_point(corner, pose, camera)
        uv_list.append(uv)
        depth_list.append(depth)
    return uv_list, depth_list


def _write_image_unicode_safe(path: Path, image: np.ndarray) -> None:
    """Write an image robustly on Windows paths that may contain Unicode."""
    ext = path.suffix or ".jpg"
    ok, encoded = cv2.imencode(ext, image)
    if not ok:
        raise RuntimeError(f"cv2.imencode failed for {path}")
    path.write_bytes(encoded.tobytes())


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Diagnose ScanNet++ DSLR camera projection")
    p.add_argument("--data_root", default="/home/sujinyue/datasets/scannetcpp",
                   help="ScanNet++ data root (default: %(default)s)")
    p.add_argument("--scene", default="0d2ee665be",
                   help="Scene ID (default: %(default)s)")
    p.add_argument("--max_frames", type=int, default=5,
                   help="Sample at most N frames (default: %(default)s)")
    p.add_argument("--max_objects", type=int, default=20,
                   help="Sample at most N objects per frame (default: %(default)s)")
    p.add_argument("--output_dir", default=None,
                   help="Debug image output dir (default: output/scannetpp_debug/<scene>)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed (default: %(default)s)")
    p.add_argument("--no_images", action="store_true",
                   help="Skip exporting debug images")
    p.add_argument("--write_undistorted", action="store_true",
                   help="Also export an undistorted preview image for visual inspection")
    p.add_argument("--z_offset", type=str, default="0",
                   help="Z offset in world units, or 'auto' to compute from "
                        "segments_anno.json (default: 0, no offset)")
    return p.parse_args()


def _undistort_fisheye_preview(image: np.ndarray, camera) -> np.ndarray:
    """Return a fisheye-undistorted preview using the original image size."""
    if not getattr(camera, "is_fisheye", False):
        return image
    K = camera.camera_matrix()
    D = camera.distortion_coeffs()
    size = (int(camera.width), int(camera.height))
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K,
        D,
        size,
        np.eye(3, dtype=np.float64),
        balance=0.0,
    )
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K,
        D,
        np.eye(3, dtype=np.float64),
        new_K,
        size,
        cv2.CV_16SC2,
    )
    return cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    data_root = Path(args.data_root)
    scene_dir = data_root / args.scene

    if not scene_dir.is_dir():
        print(f"ERROR: scene directory not found: {scene_dir}")
        sys.exit(1)

    # ---- 1. Load objects from segments_anno.json ----
    anno_path = scene_dir / "scans" / "segments_anno.json"
    if not anno_path.is_file():
        print(f"ERROR: missing {anno_path}")
        sys.exit(1)
    with open(anno_path, encoding="utf-8") as f:
        anno_data = json.load(f)
    anno_list = anno_data if isinstance(anno_data, list) else anno_data.get("segGroups", [])

    # ---- 2. Filter objects by label ----
    from src.scene_parser import EXCLUDED_LABELS, normalize_label

    objects: list[dict] = []
    for anno in anno_list:
        raw_label = str(anno.get("label", ""))
        label = normalize_label(raw_label)
        if label in EXCLUDED_LABELS:
            continue
        obb = anno.get("obb")
        if not obb:
            continue
        objects.append({
            "id": anno.get("id", anno.get("objectId")),
            "label": label,
            "raw_label": raw_label,
            "centroid": np.array(obb["centroid"], dtype=np.float64),
            "bb_min": np.array(obb["min"], dtype=np.float64),
            "bb_max": np.array(obb["max"], dtype=np.float64),
        })

    print(f"Objects after label filter: {len(objects)} (from {len(anno_list)} total)")

    if len(objects) == 0:
        print("ERROR: no objects remain after label filtering")
        sys.exit(1)

    # ---- 3. Load camera data ----
    from src.datasets.scannetpp import (
        load_scannetpp_dslr_camera,
        load_scannetpp_dslr_intrinsics,
        load_scannetpp_dslr_poses,
        get_scannetpp_dslr_pose_stats,
    )

    camera = load_scannetpp_dslr_camera(scene_dir)
    intrinsics = load_scannetpp_dslr_intrinsics(scene_dir)
    poses = load_scannetpp_dslr_poses(scene_dir)
    stats = get_scannetpp_dslr_pose_stats(scene_dir)

    print(f"\nIntrinsics: {intrinsics.width}x{intrinsics.height} "
          f"fx={intrinsics.fx:.2f} fy={intrinsics.fy:.2f} "
          f"cx={intrinsics.cx:.2f} cy={intrinsics.cy:.2f}")
    print(f"Poses: {len(poses)} loaded")
    print(f"Skip stats: {stats}")
    if camera.is_fisheye:
        print(
            f"\nprojection_model = {camera.camera_model} "
            f"(k1={camera.k1:.6g}, k2={camera.k2:.6g}, "
            f"k3={camera.k3:.6g}, k4={camera.k4:.6g})"
        )
    else:
        print(f"\nprojection_model = {camera.camera_model}")

    if len(poses) == 0:
        print("ERROR: no poses loaded — nothing to diagnose")
        sys.exit(1)

    # ---- 4. Resolve Z-offset ----
    from src.datasets.scannetpp import project_scannetpp_dslr_point
    from src.utils.coordinate_transform import is_in_image

    all_centroids = np.array([o["centroid"] for o in objects])

    if args.z_offset == "auto":
        obj_z_center = float(all_centroids[:, 2].mean())
        cam_z_mean = float(np.mean([p.position[2] for p in poses.values()]))
        z_offset = obj_z_center - cam_z_mean
        print(f"\nAuto Z-offset from segments_anno.json:")
        print(f"  Object Z center:  {obj_z_center:.2f}")
        print(f"  Camera Z mean:    {cam_z_mean:.2f}")
        print(f"  Applied Z offset: {z_offset:.2f}")
    else:
        try:
            z_offset = float(args.z_offset)
        except ValueError:
            print(f"ERROR: invalid --z_offset value: {args.z_offset!r} "
                  f"(use a number or 'auto')")
            sys.exit(1)
        if z_offset != 0.0:
            print(f"\nUsing explicit Z offset: {z_offset:.3f}")

    # Apply Z-offset to poses if non-zero
    if z_offset != 0.0:
        class _ShiftedPose:
            def __init__(self, pose, dz):
                self._pose = pose
                self._dz = dz
            @property
            def rotation(self):
                return self._pose.rotation
            @property
            def translation(self):
                return self._pose.translation - self._pose.rotation @ np.array([0, 0, self._dz])
            @property
            def position(self):
                return self._pose.position + np.array([0, 0, self._dz])
            @property
            def image_name(self):
                return self._pose.image_name

        proj_poses = {name: _ShiftedPose(p, z_offset) for name, p in poses.items()}
    else:
        proj_poses = poses
        print("\nNo Z-offset applied (raw transforms.json poses).")

    # ---- 5. Score frames by objects-in-front and select top-k ----
    scored: list[tuple[int, str]] = []
    for image_name, pose in proj_poses.items():
        cam_fwd = pose.rotation.T[:, 2]
        depths = np.dot(all_centroids - pose.position, cam_fwd)
        n_front = int((depths > 0.3).sum())
        scored.append((n_front, image_name))
    scored.sort(key=lambda x: x[0], reverse=True)

    # Diversify: pick best, worst-among-valid, and a mid-quartile frame
    best = scored[:1]
    valid = [(s, n) for s, n in scored if s > 0]
    if len(valid) > 1:
        mid_idx = len(valid) // 2
        rest = [valid[1]]  # second-best
        if mid_idx > 1:
            rest.append(valid[mid_idx])
        if len(valid) > 2:
            rest.append(valid[-1])  # worst-among-valid
    else:
        rest = []
    selected_names = [n for _, n in best] + [n for _, n in rest[:args.max_frames - 1]]
    n_frames = min(args.max_frames, len(selected_names))
    selected_names = selected_names[:n_frames]
    sampled = [(name, proj_poses[name]) for name in selected_names]

    print(f"\nFrame score range: {scored[0][0]}-{scored[-1][0]} objects in front "
          f"(out of {len(objects)} total)")
    print(f"Selected {n_frames} frames for diagnosis:")
    for name in selected_names:
        n_front = next(s for s, n in scored if n == name)
        print(f"  {name}: {n_front}/{len(objects)} objects in front")

    # ---- 6. Per-frame projection audit (with Z-aligned poses) ----
    all_warnings: list[str] = []
    frame_reports: list[dict] = []

    for image_name, pose in sampled:
        obj_sample = random.sample(objects, min(args.max_objects, len(objects)))

        center_depths: list[float] = []
        center_in_image = 0
        behind_centers = 0
        valid_bboxes = 0
        bboxes_in_image = 0
        bbox_areas: list[float] = []
        obj_details: list[dict] = []

        for obj in obj_sample:
            # Centroid
            uv_c, depth_c = project_scannetpp_dslr_point(obj["centroid"], pose, camera)
            if depth_c <= 0:
                behind_centers += 1
            elif uv_c is not None:
                center_depths.append(depth_c)
                if is_in_image(uv_c, intrinsics, margin=0):
                    center_in_image += 1

            # AABB corners
            corners = _aabb_corners(obj["bb_min"], obj["bb_max"])
            uv_list, depth_list = _project_corners(corners, pose, camera)

            valid_uvs: list[np.ndarray] = []
            for uv, d in zip(uv_list, depth_list):
                if uv is not None and d > 0 and np.isfinite(uv).all():
                    valid_uvs.append(uv)

            bbox_2d = None
            bbox_area = 0.0
            if len(valid_uvs) >= 2:
                uvs = np.array(valid_uvs)
                u_min, v_min = uvs.min(axis=0)
                u_max, v_max = uvs.max(axis=0)
                bbox_2d = (float(u_min), float(v_min), float(u_max), float(v_max))
                bbox_area = (u_max - u_min) * (v_max - v_min)
                valid_bboxes += 1
                bbox_areas.append(bbox_area)
                # Check if bbox overlaps the image
                if (u_min < intrinsics.width and u_max >= 0
                        and v_min < intrinsics.height and v_max >= 0):
                    bboxes_in_image += 1

            obj_details.append({
                "label": obj["label"],
                "center_uv": (float(uv_c[0]), float(uv_c[1])) if uv_c is not None else None,
                "center_depth": float(depth_c),
                "bbox_2d": bbox_2d,
                "bbox_area": float(bbox_area),
                "n_valid_corners": len(valid_uvs),
            })

        avg_depth = float(np.mean(center_depths)) if center_depths else 0.0
        avg_bbox_area = float(np.mean(bbox_areas)) if bbox_areas else 0.0
        img_area = intrinsics.width * intrinsics.height

        frame_reports.append({
            "image_name": image_name,
            "n_objects": len(obj_sample),
            "center_in_image": center_in_image,
            "behind_centers": behind_centers,
            "avg_center_depth": float(avg_depth),
            "valid_bboxes": valid_bboxes,
            "bboxes_in_image": bboxes_in_image,
            "avg_bbox_area": float(avg_bbox_area),
            "objects": obj_details,
        })

        # ---- Warnings ----
        if behind_centers == len(obj_sample):
            all_warnings.append(f"[WARN] {image_name}: ALL {behind_centers} object centers behind camera (depth <= 0)")

        if center_in_image == 0 and behind_centers < len(obj_sample):
            all_warnings.append(
                f"[WARN] {image_name}: 0/{len(obj_sample)} centers in image "
                f"({behind_centers} behind)"
            )

        huge_bboxes = [a for a in bbox_areas if a > img_area * 0.5]
        if huge_bboxes:
            all_warnings.append(
                f"[WARN] {image_name}: {len(huge_bboxes)} bboxes > 50% image area "
                f"(max={max(huge_bboxes):.0f} px)"
            )

        nan_inf_count = sum(
            1 for d in obj_details
            if d["center_uv"] and (not np.isfinite(d["center_uv"]).all())
        )
        if nan_inf_count:
            all_warnings.append(
                f"[WARN] {image_name}: {nan_inf_count} objects have NaN/inf projected coords"
            )

    # ---- 6. Print report ----
    print(f"\n{'='*60}")
    print(f"Per-frame summary ({n_frames} frames)")
    print(f"{'='*60}")
    for r in frame_reports:
        print(
            f"  {r['image_name']:20s}  "
            f"centers={r['center_in_image']:2d}/{r['n_objects']:<2d}  "
            f"bboxes_in={r['bboxes_in_image']:2d}/{r['valid_bboxes']:<2d}  "
            f"depth_avg={r['avg_center_depth']:.2f}m  "
            f"area_avg={r['avg_bbox_area']:.0f}px"
        )

    if all_warnings:
        print(f"\n{'='*60}")
        print("WARNINGS")
        print(f"{'='*60}")
        for w in all_warnings:
            print(f"  {w}")
    else:
        print("\nNo warnings.")

    # Overall sanity
    total_objs = sum(r["n_objects"] for r in frame_reports)
    total_centers_in = sum(r["center_in_image"] for r in frame_reports)
    total_behind = sum(r["behind_centers"] for r in frame_reports)
    total_bboxes = sum(r["valid_bboxes"] for r in frame_reports)
    total_bboxes_in = sum(r["bboxes_in_image"] for r in frame_reports)
    print(f"\nOverall: {total_centers_in}/{total_objs} centers in image, "
          f"{total_bboxes_in}/{total_bboxes} bboxes overlap image, "
          f"{total_behind} centers behind camera")

    if total_bboxes_in == 0 and total_objs > 0:
        print("\n*** CRITICAL: No bboxes overlap any image. "
              "Coordinate system may be wrong. ***")
        print("Do NOT proceed to 板块 5 until this is resolved.")
    elif total_centers_in == 0 and total_bboxes_in > 0:
        print("\nNote: centers outside image but some bboxes overlap — "
              "typical for wide-FOV lens with objects near image edges. "
              "This is acceptable for 板块 3 diagnostic.")

    # ---- 7. Debug images ----
    if args.no_images:
        print("\n--no_images set, skipping debug export.")
        return

    output_dir = Path(args.output_dir) if args.output_dir else (
        PROJECT_ROOT / "output" / "scannetpp_debug" / args.scene
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    for r in frame_reports:
        image_name = r["image_name"]
        image_path = scene_dir / "dslr" / "resized_images" / image_name
        if not image_path.is_file():
            print(f"  SKIP {image_name}: image not found")
            continue

        img = cv2.imread(str(image_path))
        if img is None:
            print(f"  SKIP {image_name}: cv2 read failed")
            continue

        for obj_d in r["objects"]:
            # Draw bbox
            if obj_d["bbox_2d"] is not None:
                u1, v1, u2, v2 = obj_d["bbox_2d"]
                p1 = (int(round(u1)), int(round(v1)))
                p2 = (int(round(u2)), int(round(v2)))
                cv2.rectangle(img, p1, p2, (255, 0, 0), 2)  # blue bbox

            # Draw center
            if obj_d["center_uv"] is not None and np.isfinite(obj_d["center_uv"]).all():
                cu, cv_ = obj_d["center_uv"]
                cp = (int(round(cu)), int(round(cv_)))
                cv2.circle(img, cp, 5, (0, 0, 255), -1)  # red dot

                # Label text
                cv2.putText(
                    img, obj_d["label"], (cp[0] + 8, cp[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
                )

        out_path = output_dir / f"{Path(image_name).stem}_projection.jpg"
        _write_image_unicode_safe(out_path, img)
        print(f"  Wrote {out_path}")

        if args.write_undistorted:
            undistorted = _undistort_fisheye_preview(img, camera)
            undistorted_path = output_dir / f"{Path(image_name).stem}_undistorted_preview.jpg"
            _write_image_unicode_safe(undistorted_path, undistorted)
            print(f"  Wrote {undistorted_path}")

    print(f"\nDebug images saved to: {output_dir}")


if __name__ == "__main__":
    main()
