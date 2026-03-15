"""Stage 1: ScanNet scene parsing.

Reads PLY meshes and instance annotations to produce per-scene object lists
with 3D axis-aligned bounding boxes.

ScanNet file layout (inside each scene directory)::

    <scene_id>_vh_clean_2.ply                   # reconstructed mesh
    <scene_id>_vh_clean_2.0.010000.segs.json    # vertex → segment mapping
    <scene_id>_vh_clean.aggregation.json        # segment → instance + label
    <scene_id>.txt                              # metadata (axisAlignment, …)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d

from .utils.colmap_loader import load_axis_alignment

logger = logging.getLogger(__name__)

# Minimum requirements for a scene to be useful
MIN_OBJECTS = 5


def _apply_axis_alignment(vertices: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Apply a 4×4 axis-alignment matrix to an (N, 3) vertex array."""
    R = M[:3, :3]
    t = M[:3, 3]
    return (R @ vertices.T).T + t


def parse_scene(scene_path: str | Path) -> dict[str, Any] | None:
    """Parse a single ScanNet scene.

    Args:
        scene_path: Path to the scene directory (e.g. ``scans/scene0000_00/``).

    Returns:
        Scene dict with keys ``scene_id`` and ``objects``.
        Returns ``None`` if the scene does not meet filtering criteria.
    """
    scene_path = Path(scene_path)
    scene_id   = scene_path.name

    mesh_file = scene_path / f"{scene_id}_vh_clean_2.ply"
    seg_file  = scene_path / f"{scene_id}_vh_clean_2.0.010000.segs.json"
    # Try both common aggregation file names
    anno_file = scene_path / f"{scene_id}_vh_clean.aggregation.json"
    if not anno_file.exists():
        anno_file = scene_path / f"{scene_id}.aggregation.json"

    for required in (mesh_file, seg_file, anno_file):
        if not required.exists():
            logger.warning("Missing %s — skipping scene %s", required.name, scene_id)
            return None

    # 1. Load mesh vertices and apply axis alignment
    mesh     = o3d.io.read_triangle_mesh(str(mesh_file))
    vertices = np.asarray(mesh.vertices)  # (N, 3)

    M = load_axis_alignment(scene_path)
    if not np.allclose(M, np.eye(4)):
        vertices = _apply_axis_alignment(vertices, M)

    # 2. Load segmentation index (vertex → segment_id)
    with open(seg_file, "r", encoding="utf-8") as f:
        seg_data = json.load(f)
    seg_indices = np.array(seg_data["segIndices"], dtype=np.int64)

    # 3. Load instance annotations (segment → instance + label)
    with open(anno_file, "r", encoding="utf-8") as f:
        annotations = json.load(f)

    if isinstance(annotations, dict):
        anno_list = annotations.get("segGroups", annotations.get("annotations", []))
    else:
        anno_list = annotations

    # 4. Build per-instance vertex sets and compute AABBs
    objects: list[dict[str, Any]] = []
    for anno in anno_list:
        instance_id = anno.get("id", anno.get("objectId"))
        label       = anno.get("label", "unknown")
        seg_ids     = set(anno.get("segments", []))
        if not seg_ids:
            continue

        mask        = np.isin(seg_indices, list(seg_ids))
        obj_vertices = vertices[mask]
        if len(obj_vertices) == 0:
            continue

        bbox_min   = obj_vertices.min(axis=0)
        bbox_max   = obj_vertices.max(axis=0)
        center     = (bbox_min + bbox_max) / 2.0
        dimensions = bbox_max - bbox_min

        objects.append(
            {
                "id":           int(instance_id),
                "label":        str(label),
                "center":       center.tolist(),
                "bbox_min":     bbox_min.tolist(),
                "bbox_max":     bbox_max.tolist(),
                "dimensions":   dimensions.tolist(),
                "vertex_count": int(obj_vertices.shape[0]),
            }
        )

    if len(objects) < MIN_OBJECTS:
        logger.info(
            "Scene %s has only %d objects (< %d) — skipping",
            scene_id, len(objects), MIN_OBJECTS,
        )
        return None

    return {"scene_id": scene_id, "objects": objects}


def parse_all_scenes(
    dataset_root: str | Path,
    output_dir: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Parse every scene under *dataset_root* and optionally save JSONs.

    Args:
        dataset_root: Root of the ScanNet scans directory (contains scene dirs).
        output_dir:   If given, write one JSON per scene to this directory.

    Returns:
        List of scene dicts that passed filtering.
    """
    dataset_root = Path(dataset_root)
    scenes: list[dict[str, Any]] = []

    # ScanNet scene dirs contain a pose/ subdirectory
    scene_dirs = sorted(
        p for p in dataset_root.iterdir()
        if p.is_dir() and (p / "pose").exists()
    )
    logger.info("Found %d candidate scene directories", len(scene_dirs))

    for scene_dir in scene_dirs:
        result = parse_scene(scene_dir)
        if result is not None:
            scenes.append(result)
            if output_dir is not None:
                out_path = Path(output_dir) / f"{result['scene_id']}.json"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)

    logger.info("Parsed %d / %d scenes successfully", len(scenes), len(scene_dirs))
    return scenes
