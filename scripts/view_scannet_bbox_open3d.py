#!/usr/bin/env python3
from __future__ import annotations

import argparse
import colorsys
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import open3d as o3d


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.scene_parser import load_scannet_label_map, parse_scene
from src.utils.colmap_loader import load_axis_alignment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize ScanNet 3D bounding boxes in an Open3D window."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--scene_metadata",
        type=Path,
        help="Path to a scene metadata JSON with objects/bbox_min/bbox_max.",
    )
    source.add_argument(
        "--scene_dir",
        type=Path,
        help="Path to a raw ScanNet scene directory such as scans/scene0000_00.",
    )
    parser.add_argument(
        "--label_map",
        type=Path,
        default=None,
        help=(
            "Optional path to scannetv2-labels.combined.tsv so raw-scene label "
            "normalization matches the main pipeline."
        ),
    )

    parser.add_argument(
        "--mesh_path",
        type=Path,
        default=None,
        help="Optional explicit mesh path (.ply). If omitted, infer from scene directory.",
    )
    parser.add_argument(
        "--render_mode",
        choices=("points", "mesh"),
        default="points",
        help="Render the scene as a point cloud or full mesh.",
    )
    parser.add_argument(
        "--voxel_size",
        type=float,
        default=0.03,
        help="Voxel downsampling size for point rendering. Set <= 0 to disable.",
    )
    parser.add_argument(
        "--label",
        action="append",
        default=[],
        help="Keep only exact labels. Can be passed multiple times.",
    )
    parser.add_argument(
        "--label_contains",
        action="append",
        default=[],
        help="Keep only labels containing this substring. Can be passed multiple times.",
    )
    parser.add_argument(
        "--object_ids",
        type=str,
        default="",
        help="Comma-separated object ids to keep, for example 3,8,17.",
    )
    parser.add_argument(
        "--max_boxes",
        type=int,
        default=0,
        help="Limit how many boxes are displayed after filtering. 0 means no limit.",
    )
    parser.add_argument(
        "--box_thickness_hint",
        type=float,
        default=1.0,
        help="Printed only as a reminder; Open3D line width support depends on backend.",
    )
    parser.add_argument(
        "--show_centers",
        action="store_true",
        help="Render object centers as colored points.",
    )
    parser.add_argument(
        "--hide_scene",
        action="store_true",
        help="Show bbox only, without mesh/point cloud.",
    )
    parser.add_argument(
        "--show_frame",
        action="store_true",
        help="Show a coordinate frame at the origin.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_axis_alignment_for_scene(base_dir: Path, scene_id: str) -> np.ndarray:
    """Read axisAlignment from <base_dir>/<scene_id>.txt if present."""
    meta_file = base_dir / f"{scene_id}.txt"
    if not meta_file.exists():
        return np.eye(4, dtype=np.float64)
    with meta_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            if "axisAlignment" in line:
                values = [float(x) for x in line.split("=", 1)[1].strip().split()]
                return np.asarray(values, dtype=np.float64).reshape(4, 4)
    return np.eye(4, dtype=np.float64)


def infer_mesh_path(base_dir: Path, scene_id: str, explicit: Path | None) -> Path:
    if explicit is not None:
        if not explicit.exists():
            raise FileNotFoundError(f"Mesh file not found: {explicit}")
        return explicit

    candidates = [
        base_dir / f"{scene_id}_vh_clean.ply",
        base_dir / f"{scene_id}_vh_clean_2.ply",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not infer mesh for {scene_id} under {base_dir}. "
        "Pass --mesh_path explicitly."
    )


def parse_object_ids(raw: str) -> set[int]:
    ids: set[int] = set()
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        ids.add(int(chunk))
    return ids


def label_color(label: str) -> list[float]:
    hue = (sum(ord(ch) for ch in label) % 360) / 360.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.7, 0.95)
    return [float(r), float(g), float(b)]


def filter_objects(
    objects: list[dict],
    exact_labels: set[str],
    contains_labels: list[str],
    object_ids: set[int],
    max_boxes: int,
) -> list[dict]:
    kept: list[dict] = []
    for obj in objects:
        obj_id = int(obj["id"])
        label = str(obj.get("label", "")).lower()
        if exact_labels and label not in exact_labels:
            continue
        if contains_labels and not any(token in label for token in contains_labels):
            continue
        if object_ids and obj_id not in object_ids:
            continue
        kept.append(obj)
    kept.sort(key=lambda item: (str(item.get("label", "")), int(item["id"])))
    if max_boxes > 0:
        kept = kept[:max_boxes]
    return kept


def bbox_corners(bbox_min: Iterable[float], bbox_max: Iterable[float]) -> np.ndarray:
    mn = np.asarray(list(bbox_min), dtype=np.float64)
    mx = np.asarray(list(bbox_max), dtype=np.float64)
    return np.asarray(
        [
            [mn[0], mn[1], mn[2]],
            [mx[0], mn[1], mn[2]],
            [mx[0], mx[1], mn[2]],
            [mn[0], mx[1], mn[2]],
            [mn[0], mn[1], mx[2]],
            [mx[0], mn[1], mx[2]],
            [mx[0], mx[1], mx[2]],
            [mn[0], mx[1], mx[2]],
        ],
        dtype=np.float64,
    )


def build_bbox_lineset(objects: list[dict]) -> o3d.geometry.LineSet:
    edge_template = np.asarray(
        [
            [0, 1], [1, 2], [2, 3], [3, 0],
            [4, 5], [5, 6], [6, 7], [7, 4],
            [0, 4], [1, 5], [2, 6], [3, 7],
        ],
        dtype=np.int32,
    )
    points: list[list[float]] = []
    lines: list[list[int]] = []
    colors: list[list[float]] = []
    offset = 0

    for obj in objects:
        corners = bbox_corners(obj["bbox_min"], obj["bbox_max"])
        color = label_color(str(obj.get("label", "")))
        points.extend(corners.tolist())
        lines.extend((edge_template + offset).tolist())
        colors.extend([color] * len(edge_template))
        offset += 8

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    line_set.lines = o3d.utility.Vector2iVector(np.asarray(lines, dtype=np.int32))
    line_set.colors = o3d.utility.Vector3dVector(np.asarray(colors, dtype=np.float64))
    return line_set


def build_center_cloud(objects: list[dict]) -> o3d.geometry.PointCloud:
    centers = []
    colors = []
    for obj in objects:
        centers.append(np.asarray(obj["center"], dtype=np.float64))
        colors.append(label_color(str(obj.get("label", ""))))

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(np.asarray(centers, dtype=np.float64))
    cloud.colors = o3d.utility.Vector3dVector(np.asarray(colors, dtype=np.float64))
    return cloud


def load_scene_geometry(
    mesh_path: Path,
    axis_alignment: np.ndarray,
    render_mode: str,
    voxel_size: float,
) -> o3d.geometry.Geometry:
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if mesh.is_empty():
        raise ValueError(f"Open3D could not read mesh: {mesh_path}")

    if not np.allclose(axis_alignment, np.eye(4)):
        mesh.transform(axis_alignment)

    if render_mode == "mesh":
        mesh.compute_vertex_normals()
        if not mesh.has_vertex_colors():
            mesh.paint_uniform_color([0.78, 0.80, 0.82])
        return mesh

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(np.asarray(mesh.vertices))
    if mesh.has_vertex_colors():
        cloud.colors = o3d.utility.Vector3dVector(np.asarray(mesh.vertex_colors))
    else:
        cloud.paint_uniform_color([0.65, 0.68, 0.72])
    if voxel_size > 0:
        cloud = cloud.voxel_down_sample(voxel_size=float(voxel_size))
    return cloud


def resolve_scene_from_metadata(
    metadata_path: Path,
    mesh_path: Path | None,
) -> tuple[str, list[dict], Path, np.ndarray]:
    scene = load_json(metadata_path)
    scene_id = str(scene.get("scene_id") or metadata_path.stem)
    objects = list(scene.get("objects") or [])
    if not objects:
        raise ValueError(f"No objects found in {metadata_path}")
    base_dir = metadata_path.parent
    resolved_mesh = infer_mesh_path(base_dir, scene_id, mesh_path)
    axis_alignment = load_axis_alignment_for_scene(base_dir, scene_id)
    return scene_id, objects, resolved_mesh, axis_alignment


def resolve_scene_from_raw_dir(
    scene_dir: Path,
    mesh_path: Path | None,
) -> tuple[str, list[dict], Path, np.ndarray]:
    scene = parse_scene(scene_dir)
    if scene is None:
        raise ValueError(
            f"parse_scene() returned no usable objects for {scene_dir}. "
            "Try generating scene metadata first."
        )
    scene_id = str(scene["scene_id"])
    objects = list(scene.get("objects") or [])
    resolved_mesh = infer_mesh_path(scene_dir, scene_id, mesh_path)
    axis_alignment = load_axis_alignment(scene_dir)
    return scene_id, objects, resolved_mesh, axis_alignment


def main() -> None:
    args = parse_args()

    if args.label_map is not None:
        load_scannet_label_map(args.label_map)

    if args.scene_metadata is not None:
        scene_id, objects, mesh_path, axis_alignment = resolve_scene_from_metadata(
            args.scene_metadata,
            args.mesh_path,
        )
    else:
        scene_id, objects, mesh_path, axis_alignment = resolve_scene_from_raw_dir(
            args.scene_dir,
            args.mesh_path,
        )

    exact_labels = {label.strip().lower() for label in args.label if label.strip()}
    contains_labels = [label.strip().lower() for label in args.label_contains if label.strip()]
    object_ids = parse_object_ids(args.object_ids)
    objects = filter_objects(
        objects=objects,
        exact_labels=exact_labels,
        contains_labels=contains_labels,
        object_ids=object_ids,
        max_boxes=int(args.max_boxes),
    )
    if not objects:
        raise ValueError("No objects matched the requested filters.")

    geoms: list[o3d.geometry.Geometry] = []
    if not args.hide_scene:
        geoms.append(
            load_scene_geometry(
                mesh_path=mesh_path,
                axis_alignment=axis_alignment,
                render_mode=args.render_mode,
                voxel_size=float(args.voxel_size),
            )
        )

    geoms.append(build_bbox_lineset(objects))
    if args.show_centers:
        geoms.append(build_center_cloud(objects))
    if args.show_frame:
        geoms.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5))

    labels = sorted({str(obj.get("label", "")) for obj in objects})
    print(f"scene: {scene_id}")
    print(f"mesh:  {mesh_path}")
    print(f"boxes: {len(objects)}")
    print(f"labels: {', '.join(labels[:12])}{' ...' if len(labels) > 12 else ''}")
    print(
        "Open3D controls: left drag rotate, right drag pan, wheel zoom, "
        "R reset view, Ctrl/Cmd+C close terminal window if needed."
    )
    print(
        f"box thickness hint requested: {args.box_thickness_hint} "
        "(actual line width depends on the Open3D backend)"
    )

    o3d.visualization.draw_geometries(
        geoms,
        window_name=f"ScanNet BBox Viewer - {scene_id}",
        width=1600,
        height=960,
        mesh_show_back_face=True,
    )


if __name__ == "__main__":
    main()
