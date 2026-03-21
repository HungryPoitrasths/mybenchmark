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
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d

from .utils.colmap_loader import load_axis_alignment

logger = logging.getLogger(__name__)

# Minimum requirements for a scene to be useful
MIN_OBJECTS = 3

# ---------------------------------------------------------------------------
#  Label normalisation — applied before dedup / blacklist filtering.
#  Maps plural forms, synonyms, and sub-categories to canonical labels so
#  that the unique-label filter catches true duplicates.
# ---------------------------------------------------------------------------
LABEL_NORMALIZE: dict[str, str] = {
    # Plural → singular
    "books": "book",
    "doors": "door",
    "curtains": "curtain",
    "shoes": "shoe",
    "clothes": "clothing",
    "cabinets": "cabinet",
    "papers": "paper",
    "pipes": "pipe",
    "mailboxes": "mailbox",
    "cloth": "clothing",
    # Sub-category → canonical
    "kitchen cabinet": "cabinet",
    "kitchen cabinets": "cabinet",
    "bathroom cabinet": "cabinet",
    "file cabinet": "cabinet",
    "kitchen counter": "counter",
    "storage box": "storage container",
    "storage bin": "storage container",
    "trash bin": "trash can",
    "recycling bin": "trash can",
    "sofa chair": "chair",
    "armchair": "chair",
    "office chair": "chair",
    "dining chair": "chair",
    "folding chair": "chair",
    "coffee table": "table",
    "dining table": "table",
    "mini fridge": "refrigerator",
    "shower wall": "wall",
    "ceiling fan": "fan",
}

# Runtime label map loaded from scannetv2-labels.combined.tsv (raw_category → nyu40class).
# Populated by load_scannet_label_map(); empty dict = fall back to LABEL_NORMALIZE only.
_SCANNET_LABEL_MAP: dict[str, str] = {}


def load_scannet_label_map(tsv_path: str | Path) -> None:
    """Load raw_category → nyu40class mapping from scannetv2-labels.combined.tsv.

    Populates the module-level _SCANNET_LABEL_MAP used by normalize_label().
    Call this once at startup before parsing any scenes.
    """
    import csv
    global _SCANNET_LABEL_MAP
    tsv_path = Path(tsv_path)
    if not tsv_path.exists():
        logger.warning("ScanNet label map not found: %s — using built-in rules only", tsv_path)
        return
    mapping: dict[str, str] = {}
    with open(tsv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            raw = row.get("raw_category", "").strip().lower()
            nyu = row.get("nyu40class", "").strip().lower()
            if raw and nyu:
                mapping[raw] = nyu
    _SCANNET_LABEL_MAP = mapping
    logger.info("Loaded %d label mappings from %s", len(mapping), tsv_path.name)


def normalize_label(label: str) -> str:
    """Return the canonical form of *label* (lowercase, mapped).

    Priority:
      1. scannetv2-labels.combined.tsv (raw_category → nyu40class) if loaded
      2. Built-in LABEL_NORMALIZE rules
      3. Lowercase of the original label
    """
    low = label.strip().lower()
    if _SCANNET_LABEL_MAP:
        return _SCANNET_LABEL_MAP.get(low, LABEL_NORMALIZE.get(low, low))
    return LABEL_NORMALIZE.get(low, low)


# Labels to exclude — structural elements, uninformative categories,
# reflective/transparent surfaces, and ambiguous objects.
# Shared with qa_generator.py (duplicated for simplicity).
EXCLUDED_LABELS = {
    # Structural / architectural
    "floor", "wall", "ceiling", "room", "ground",
    "door", "window", "stairs", "pillar", "column",
    "doorframe", "windowsill", "hand rail", "shower",
    "shower curtain rod", "bathroom stall", "bathroom stall door",
    "ledge", "structure", "closet", "breakfast bar", "shower curtain",
    # Generic / uninformative
    "object", "otherfurniture", "otherprop", "otherstructure",
    "unknown", "misc", "stuff",
    # Reflective / transparent — depth sensor unreliable
    "mirror", "glass", "monitor", "tv",
    # Ambiguous / vague
    "case", "tube", "board", "sign", "frame", "paper", "lotion",
    # Boundary-unclear / large amorphous / unreliable 3D annotation
    "counter", "couch", "clothing", "clothes", "cloth", "blanket", "rug",
    "shelf", "bookshelf", "shelves", "rack", "storage shelf",
    # Too small to reliably identify in images
    "power outlet", "light switch", "fire alarm", "controller",
    "power strip", "soda can", "starbucks cup", "battery disposal jar",
    "can", "water bottle", "paper cutter",
}


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

    # Support both _vh_clean.ply and _vh_clean_2.ply naming conventions
    mesh_file = scene_path / f"{scene_id}_vh_clean.ply"
    if not mesh_file.exists():
        mesh_file = scene_path / f"{scene_id}_vh_clean_2.ply"

    # Support both segs file naming conventions
    seg_file = scene_path / f"{scene_id}_vh_clean.segs.json"
    if not seg_file.exists():
        seg_file = scene_path / f"{scene_id}_vh_clean_2.0.010000.segs.json"

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
        label       = normalize_label(anno.get("label", "unknown"))
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

    # Extract room boundaries from wall/floor objects BEFORE filtering them out.
    # These structural elements define the physical room extent.
    STRUCTURAL_LABELS = {"floor", "wall", "ground", "ceiling"}
    structural_objects = [
        o for o in objects if o["label"].lower() in STRUCTURAL_LABELS
    ]
    room_bounds = None
    if structural_objects:
        all_mins = np.array([o["bbox_min"] for o in structural_objects])
        all_maxs = np.array([o["bbox_max"] for o in structural_objects])
        room_bounds = {
            "bbox_min": all_mins.min(axis=0).tolist(),
            "bbox_max": all_maxs.max(axis=0).tolist(),
        }

    # Per-scene uniqueness: drop objects whose label appears more than once
    # (e.g. 5 "chair"s → all removed) and excluded structural labels.
    # This is more aggressive than per-frame filtering but eliminates
    # ambiguity at the source ("the chair" is always unambiguous).
    n_before = len(objects)
    label_counts = Counter(o["label"] for o in objects)
    objects = [
        o for o in objects
        if label_counts[o["label"]] == 1
        and o["label"].lower() not in EXCLUDED_LABELS
    ]
    if n_before != len(objects):
        logger.debug(
            "Scene %s: %d → %d objects after unique-label + excluded filter",
            scene_id, n_before, len(objects),
        )

    if len(objects) < MIN_OBJECTS:
        logger.info(
            "Scene %s has only %d objects (< %d) — skipping",
            scene_id, len(objects), MIN_OBJECTS,
        )
        return None

    return {"scene_id": scene_id, "objects": objects, "room_bounds": room_bounds}


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
