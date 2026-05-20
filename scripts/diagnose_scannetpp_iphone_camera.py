#!/usr/bin/env python3
"""Diagnose ScanNet++ iPhone camera projection without running VLM inference.

Projects 3D object bounding-boxes into sample iPhone frames and reports
statistics that help catch coordinate-system mismatches before entering
the full frame-selection pipeline.

Usage::

    python scripts/diagnose_scannetpp_iphone_camera.py \
        --data_root ++data --scene 0d2ee665be \
        --frame_root output/scannetpp_iphone_frames \
        --max_frames 5

The script uses the native ScanNet++ iPhone camera model.  OPENCV
distortion is applied via cv2.projectPoints so debug overlays match
the iPhone video frames.
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
    """Return 8 AABB corner points (shape 8x3)."""
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


def _project_corners(corners, pose, intrinsics):
    """Project 8 AABB corners via unified distortion-aware projection."""
    from src.utils.coordinate_transform import (
        world_to_camera,
        project_camera_points_to_image,
    )

    points_cam = np.stack([
        world_to_camera(c, pose) for c in corners
    ], axis=0)
    uv_all, depths = project_camera_points_to_image(points_cam, intrinsics)

    uv_list = []
    depth_list = []
    for i in range(len(uv_all)):
        if depths[i] > 0 and np.isfinite(uv_all[i]).all():
            uv_list.append((float(uv_all[i, 0]), float(uv_all[i, 1])))
        else:
            uv_list.append(None)
        depth_list.append(float(depths[i]))
    return uv_list, depth_list


def _write_image_unicode_safe(path, image):
    ext = path.suffix or ".jpg"
    ok, encoded = cv2.imencode(ext, image)
    if not ok:
        raise RuntimeError(f"cv2.imencode failed for {path}")
    path.write_bytes(encoded.tobytes())


def _parse_args():
    p = argparse.ArgumentParser(
        description="Diagnose ScanNet++ iPhone camera projection"
    )
    p.add_argument(
        "--data_root", default="/home/sujinyue/datasets/scannetcpp",
        help="ScanNet++ data root (default: %(default)s)",
    )
    p.add_argument(
        "--scene", default="0d2ee665be",
        help="Scene ID (default: %(default)s)",
    )
    p.add_argument(
        "--frame_root", default="output/scannetpp_iphone_frames",
        help="Root for extracted iPhone frames (default: %(default)s)",
    )
    p.add_argument(
        "--max_frames", type=int, default=5,
        help="Sample at most N frames (default: %(default)s)",
    )
    p.add_argument(
        "--max_objects", type=int, default=20,
        help="Sample at most N objects per frame (default: %(default)s)",
    )
    p.add_argument(
        "--output_dir", default=None,
        help="Debug image output dir (default: output/scannetpp_iphone_debug/<scene>)",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: %(default)s)",
    )
    p.add_argument("--no_images", action="store_true",
                   help="Skip exporting debug images")
    p.add_argument(
        "--z_offset", type=str, default="0",
        help="Z offset in world units, or 'auto' (default: 0, no offset)",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = _parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    scene_dir = Path(args.data_root) / args.scene
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
    anno_list = (
        anno_data if isinstance(anno_data, list)
        else anno_data.get("segGroups", [])
    )

    # ---- 2. Filter objects by label ----
    from src.scene_parser import EXCLUDED_LABELS, normalize_label

    objects = []
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
            "centroid": np.array(obb["centroid"], dtype=np.float64),
            "bb_min": np.array(obb["min"], dtype=np.float64),
            "bb_max": np.array(obb["max"], dtype=np.float64),
        })

    print(f"Objects after label filter: {len(objects)} "
          f"(from {len(anno_list)} total)")

    # ---- 3. Load iPhone camera data via DataSource ----
    from src.datasets.scannetpp import ScanNetPPDataSource

    ds = ScanNetPPDataSource(
        scene_dir, sensor="iphone",
        frame_root=Path(args.frame_root),
    )
    intrinsics = ds.load_intrinsics()
    poses = ds.load_poses()

    print(f"\nIntrinsics: {intrinsics.width}x{intrinsics.height} "
          f"fx={intrinsics.fx:.2f} fy={intrinsics.fy:.2f} "
          f"cx={intrinsics.cx:.2f} cy={intrinsics.cy:.2f}")
    print(f"Distortion model: {intrinsics.distortion_model or 'PINHOLE'}")
    if intrinsics.distortion_params is not None:
        print(f"Distortion params: {intrinsics.distortion_params}")
    print(f"Poses: {len(poses)} loaded")

    if len(poses) == 0:
        print("ERROR: no poses loaded — nothing to diagnose")
        sys.exit(1)

    # ---- 4. Resolve Z-offset ----
    all_centroids = np.array([o["centroid"] for o in objects])

    if args.z_offset == "auto":
        obj_z_center = float(all_centroids[:, 2].mean())
        cam_z_mean = float(np.mean(
            [p.position[2] for p in poses.values()]
        ))
        z_offset = obj_z_center - cam_z_mean
        print(f"\nAuto Z-offset:")
        print(f"  Object Z center:  {obj_z_center:.2f}")
        print(f"  Camera Z mean:    {cam_z_mean:.2f}")
        print(f"  Applied Z offset: {z_offset:.2f}")
    else:
        try:
            z_offset = float(args.z_offset)
        except ValueError:
            print(f"ERROR: invalid --z_offset: {args.z_offset!r}")
            sys.exit(1)
        if z_offset != 0.0:
            print(f"\nUsing explicit Z offset: {z_offset:.3f}")
        else:
            print("\nNo Z-offset applied (default).")

    # Apply Z-offset
    if z_offset != 0.0:
        class _ShiftedPose:
            def __init__(self, p, dz):
                self._pose = p
                self._dz = dz
                self.image_name = p.image_name
            @property
            def rotation(self):
                return self._pose.rotation
            @property
            def translation(self):
                return (self._pose.translation
                        - self._pose.rotation @ np.array([0, 0, self._dz]))
            @property
            def position(self):
                return self._pose.position + np.array([0, 0, self._dz])
        proj_poses = {n: _ShiftedPose(p, z_offset) for n, p in poses.items()}
    else:
        proj_poses = poses

    # ---- 5. Score frames by objects-in-front and select top-k ----
    from src.utils.coordinate_transform import is_in_image

    scored = []
    for image_name, pose in proj_poses.items():
        cam_fwd = pose.rotation.T[:, 2]
        depths = np.dot(all_centroids - pose.position, cam_fwd)
        n_front = int((depths > 0.3).sum())
        scored.append((n_front, image_name))
    scored.sort(key=lambda x: x[0], reverse=True)

    # Diversify
    best = scored[:1]
    valid = [(s, n) for s, n in scored if s > 0]
    if len(valid) > 1:
        mid_idx = len(valid) // 2
        rest = [valid[1]]
        if mid_idx > 1:
            rest.append(valid[mid_idx])
        if len(valid) > 2:
            rest.append(valid[-1])
    else:
        rest = []
    selected_names = (
        [n for _, n in best]
        + [n for _, n in rest[:args.max_frames - 1]]
    )
    n_frames = min(args.max_frames, len(selected_names))
    selected_names = selected_names[:n_frames]
    sampled = [(name, proj_poses[name]) for name in selected_names]

    print(f"\nFrame score range: {scored[0][0]}-{scored[-1][0]} "
          f"objects in front (out of {len(objects)} total)")
    print(f"Selected {n_frames} frames for diagnosis:")
    for name in selected_names:
        n_front = next(s for s, n in scored if n == name)
        print(f"  {name}: {n_front}/{len(objects)} objects in front")

    # ---- 6. Per-frame projection audit ----
    all_warnings = []
    frame_reports = []

    for image_name, pose in sampled:
        image_path = ds.image_path(image_name)
        if not image_path.is_file():
            print(f"\n  SKIP {image_name}: image not found at {image_path}")
            all_warnings.append(
                f"[WARN] {image_name}: image not found. "
                f"Run scripts/extract_scannetpp_iphone_frames.py first."
            )
            continue

        obj_sample = random.sample(objects, min(args.max_objects, len(objects)))

        center_depths = []
        center_in_image = 0
        behind_centers = 0
        valid_bboxes = 0
        bboxes_in_image = 0
        bbox_areas = []
        obj_details = []

        from src.utils.coordinate_transform import (
            world_to_camera,
            project_camera_points_to_image,
        )

        for obj in obj_sample:
            # Centroid projection (distortion-aware)
            point_cam = world_to_camera(obj["centroid"], pose)
            uv_all, depths = project_camera_points_to_image(
                point_cam.reshape(1, 3), intrinsics
            )
            depth_c = float(depths[0])
            uv_c = None
            if depth_c > 0 and np.isfinite(uv_all[0]).all():
                uv_c = uv_all[0]

            if depth_c <= 0:
                behind_centers += 1
            elif uv_c is not None:
                center_depths.append(depth_c)
                if is_in_image(uv_c, intrinsics, margin=0):
                    center_in_image += 1

            # AABB corners (distortion-aware batch)
            corners = _aabb_corners(obj["bb_min"], obj["bb_max"])
            uv_list, depth_list = _project_corners(corners, pose, intrinsics)

            valid_uvs = []
            for uv, d in zip(uv_list, depth_list):
                if uv is not None and d > 0:
                    valid_uvs.append(np.array(uv))

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
                if (u_min < intrinsics.width and u_max >= 0
                        and v_min < intrinsics.height and v_max >= 0):
                    bboxes_in_image += 1

            obj_details.append({
                "label": obj["label"],
                "center_uv": (
                    (float(uv_c[0]), float(uv_c[1]))
                    if uv_c is not None else None
                ),
                "center_depth": float(depth_c),
                "bbox_2d": bbox_2d,
                "bbox_area": float(bbox_area),
                "n_valid_corners": len(valid_uvs),
            })

        avg_depth = float(np.mean(center_depths)) if center_depths else 0.0
        avg_bbox_area = float(np.mean(bbox_areas)) if bbox_areas else 0.0

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

        if behind_centers == len(obj_sample):
            all_warnings.append(
                f"[WARN] {image_name}: ALL objects behind camera"
            )
        if center_in_image == 0 and behind_centers < len(obj_sample):
            all_warnings.append(
                f"[WARN] {image_name}: 0 centers in image "
                f"({behind_centers} behind)"
            )

    # ---- 7. Print report ----
    print(f"\n{'='*60}")
    print(f"Per-frame summary ({len(frame_reports)} frames)")
    print(f"{'='*60}")
    for r in frame_reports:
        print(
            f"  {r['image_name']:30s}  "
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
        print("\n*** CRITICAL: No bboxes overlap any image. ***")
        print("Coordinate system may be wrong. "
              "Do NOT proceed until resolved.")
    elif total_centers_in == 0 and total_bboxes_in > 0:
        print("\nNote: centers outside but some bboxes overlap — "
              "acceptable for wide-FOV lens.")

    # ---- 8. Debug overlay images ----
    if args.no_images:
        print("\n--no_images set, skipping debug export.")
        return

    output_dir = Path(args.output_dir) if args.output_dir else (
        PROJECT_ROOT / "output" / "scannetpp_iphone_debug" / args.scene
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    for r in frame_reports:
        image_name = r["image_name"]
        image_path = ds.image_path(image_name)
        if not image_path.is_file():
            print(f"  SKIP {image_name}: image not found")
            continue

        img = cv2.imread(str(image_path))
        if img is None:
            print(f"  SKIP {image_name}: cv2 read failed")
            continue

        for obj_d in r["objects"]:
            if obj_d["bbox_2d"] is not None:
                u1, v1, u2, v2 = obj_d["bbox_2d"]
                cv2.rectangle(
                    img,
                    (int(round(u1)), int(round(v1))),
                    (int(round(u2)), int(round(v2))),
                    (255, 0, 0), 2,
                )
            if obj_d["center_uv"] is not None:
                cu, cv_ = obj_d["center_uv"]
                cp = (int(round(cu)), int(round(cv_)))
                cv2.circle(img, cp, 5, (0, 0, 255), -1)
                cv2.putText(
                    img, obj_d["label"],
                    (cp[0] + 8, cp[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
                )

        out_name = f"{Path(image_name).stem}_projection.jpg"
        out_path = output_dir / out_name
        _write_image_unicode_safe(out_path, img)
        print(f"  Wrote {out_path}")

    print(f"\nDebug images saved to: {output_dir}")


if __name__ == "__main__":
    main()
