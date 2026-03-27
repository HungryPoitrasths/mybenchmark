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

from dataclasses import dataclass
import json
import logging
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
    # When the ScanNet TSV is loaded it remains the source of truth; built-in
    # normalization only fills gaps for categories absent from the TSV.
    if _SCANNET_LABEL_MAP:
        return _SCANNET_LABEL_MAP.get(low, LABEL_NORMALIZE.get(low, low))
    return LABEL_NORMALIZE.get(low, low)


# Labels to exclude — structural elements, uninformative categories,
# reflective/transparent surfaces, and ambiguous objects.
# Shared with qa_generator.py (duplicated for simplicity).
ALWAYS_EXCLUDED = {
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
    "person", "people", "human", "man", "woman", "boy", "girl", "child", "children",
    # Too small to reliably identify in images
    "power outlet", "light switch", "fire alarm", "controller",
    "power strip", "soda can", "starbucks cup", "battery disposal jar",
    "can", "water bottle", "paper cutter",
}

QUESTION_ONLY_EXCLUDED = {
    "counter", "couch", "clothing", "clothes", "cloth", "blanket", "rug",
    "cabinet",
    "shelf", "bookshelf", "shelves", "rack", "storage shelf",
    "refrigerator", "refridgerator",
}

EXCLUDED_LABELS = ALWAYS_EXCLUDED | QUESTION_ONLY_EXCLUDED

SceneGeometry = tuple[str, np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]


@dataclass
class InstanceMeshData:
    """Per-instance triangle ownership and cached surface samples."""

    vertices: np.ndarray
    faces: np.ndarray
    triangle_ids_by_instance: dict[int, np.ndarray]
    boundary_triangle_ids_by_instance: dict[int, np.ndarray]
    surface_points_by_instance: dict[int, np.ndarray]


def _apply_axis_alignment(vertices: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Apply a 4×4 axis-alignment matrix to an (N, 3) vertex array."""
    R = M[:3, :3]
    t = M[:3, 3]
    return (R @ vertices.T).T + t


def _resolve_scene_files(scene_path: Path) -> tuple[str, Path, Path, Path]:
    """Return scene id plus mesh / seg / aggregation paths."""
    scene_id = scene_path.name

    mesh_file = scene_path / f"{scene_id}_vh_clean.ply"
    if not mesh_file.exists():
        mesh_file = scene_path / f"{scene_id}_vh_clean_2.ply"

    seg_file = scene_path / f"{scene_id}_vh_clean.segs.json"
    if not seg_file.exists():
        seg_file = scene_path / f"{scene_id}_vh_clean_2.0.010000.segs.json"

    anno_file = scene_path / f"{scene_id}_vh_clean.aggregation.json"
    if not anno_file.exists():
        anno_file = scene_path / f"{scene_id}.aggregation.json"

    return scene_id, mesh_file, seg_file, anno_file


def _load_scene_geometry(
    scene_path: str | Path,
) -> SceneGeometry:
    """Load one scene's aligned vertices, faces, segment ids, and annotations."""
    scene_path = Path(scene_path)
    scene_id, mesh_file, seg_file, anno_file = _resolve_scene_files(scene_path)

    for required in (mesh_file, seg_file, anno_file):
        if not required.exists():
            raise FileNotFoundError(f"Missing {required.name} for scene {scene_id}")

    mesh = o3d.io.read_triangle_mesh(str(mesh_file))
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles, dtype=np.int64)

    M = load_axis_alignment(scene_path)
    if not np.allclose(M, np.eye(4)):
        vertices = _apply_axis_alignment(vertices, M)

    with open(seg_file, "r", encoding="utf-8") as f:
        seg_data = json.load(f)
    seg_indices = np.array(seg_data["segIndices"], dtype=np.int64)

    with open(anno_file, "r", encoding="utf-8") as f:
        annotations = json.load(f)

    if isinstance(annotations, dict):
        if "segGroups" in annotations:
            anno_list = annotations["segGroups"]
        elif "annotations" in annotations:
            anno_list = annotations["annotations"]
        else:
            logger.warning(
                "Scene %s annotation JSON has no 'segGroups' or 'annotations' field; available keys: %s",
                scene_id,
                sorted(annotations.keys()),
            )
            anno_list = []
    else:
        anno_list = annotations

    return scene_id, vertices, faces, seg_indices, anno_list


def _sample_surface_points_from_triangles(
    vertices: np.ndarray,
    faces: np.ndarray,
    triangle_ids: np.ndarray,
    n_samples: int,
    rng: np.random.RandomState,
) -> np.ndarray:
    """Area-weighted barycentric surface sampling from selected triangles."""
    if len(triangle_ids) == 0 or n_samples <= 0:
        return np.empty((0, 3), dtype=np.float64)

    tri_vertices = vertices[faces[triangle_ids]]
    cross = np.cross(
        tri_vertices[:, 1] - tri_vertices[:, 0],
        tri_vertices[:, 2] - tri_vertices[:, 0],
    )
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    positive_mask = areas > 1e-10
    if not np.any(positive_mask):
        return tri_vertices.mean(axis=1)

    tri_vertices = tri_vertices[positive_mask]
    areas = areas[positive_mask]
    probs = areas / areas.sum()
    chosen = rng.choice(len(tri_vertices), size=n_samples, replace=True, p=probs)
    chosen_triangles = tri_vertices[chosen]

    u = rng.rand(n_samples, 1)
    v = rng.rand(n_samples, 1)
    sqrt_u = np.sqrt(u)
    bary_a = 1.0 - sqrt_u
    bary_b = sqrt_u * (1.0 - v)
    bary_c = sqrt_u * v
    return (
        bary_a * chosen_triangles[:, 0]
        + bary_b * chosen_triangles[:, 1]
        + bary_c * chosen_triangles[:, 2]
    ).astype(np.float64)


def load_instance_mesh_data(
    scene_path: str | Path,
    instance_ids: list[int] | None = None,
    n_surface_samples: int = 128,
    preloaded_geometry: SceneGeometry | None = None,
) -> InstanceMeshData:
    """Return instance triangle ownership and cached sampled surface points."""
    if preloaded_geometry is None:
        _scene_id, vertices, faces, seg_indices, anno_list = _load_scene_geometry(scene_path)
    else:
        _scene_id, vertices, faces, seg_indices, anno_list = preloaded_geometry
    requested_ids = None if instance_ids is None else {int(x) for x in instance_ids}

    segment_to_instance: dict[int, int] = {}
    kept_instances: set[int] = set()
    for anno in anno_list:
        instance_id = anno.get("id", anno.get("objectId"))
        if instance_id is None:
            continue
        instance_id = int(instance_id)
        if requested_ids is not None and instance_id not in requested_ids:
            continue

        label = normalize_label(anno.get("label", "unknown"))
        if label.lower() in ALWAYS_EXCLUDED:
            continue

        seg_ids = set(anno.get("segments", []))
        if not seg_ids:
            continue

        kept_instances.add(instance_id)
        for seg_id in seg_ids:
            segment_to_instance[int(seg_id)] = instance_id

    triangle_ids_by_instance: dict[int, list[int]] = {}
    boundary_triangle_ids_by_instance: dict[int, list[int]] = {}

    for tri_id, face in enumerate(faces):
        tri_seg_ids = seg_indices[face]
        tri_instance_ids = [
            segment_to_instance.get(int(seg_id), -1)
            for seg_id in tri_seg_ids
        ]
        valid_ids = [inst_id for inst_id in tri_instance_ids if inst_id >= 0]
        if not valid_ids:
            continue

        if tri_instance_ids[0] == tri_instance_ids[1] == tri_instance_ids[2] and tri_instance_ids[0] >= 0:
            triangle_ids_by_instance.setdefault(tri_instance_ids[0], []).append(int(tri_id))
            continue

        for inst_id in set(valid_ids):
            boundary_triangle_ids_by_instance.setdefault(inst_id, []).append(int(tri_id))

    triangle_arrays = {
        inst_id: np.array(tri_ids, dtype=np.int64)
        for inst_id, tri_ids in triangle_ids_by_instance.items()
        if inst_id in kept_instances and tri_ids
    }
    boundary_arrays = {
        inst_id: np.array(sorted(set(tri_ids)), dtype=np.int64)
        for inst_id, tri_ids in boundary_triangle_ids_by_instance.items()
        if inst_id in kept_instances and tri_ids
    }

    surface_points_by_instance: dict[int, np.ndarray] = {}
    for inst_id in kept_instances:
        tri_ids = triangle_arrays.get(inst_id)
        if tri_ids is None or len(tri_ids) == 0:
            continue
        rng = np.random.RandomState(inst_id % (2 ** 32))
        surface_points_by_instance[inst_id] = _sample_surface_points_from_triangles(
            vertices=vertices,
            faces=faces,
            triangle_ids=tri_ids,
            n_samples=n_surface_samples,
            rng=rng,
        )

    return InstanceMeshData(
        vertices=np.asarray(vertices, dtype=np.float64),
        faces=np.asarray(faces, dtype=np.int64),
        triangle_ids_by_instance=triangle_arrays,
        boundary_triangle_ids_by_instance=boundary_arrays,
        surface_points_by_instance=surface_points_by_instance,
    )


def _cross_2d(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    return float((a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]))


def _convex_hull_2d(points_xy: np.ndarray) -> np.ndarray:
    """Return the 2D convex hull of *points_xy* using the monotonic chain."""
    if len(points_xy) == 0:
        return np.empty((0, 2), dtype=float)

    pts = np.unique(np.asarray(points_xy, dtype=float), axis=0)
    if len(pts) <= 2:
        return pts

    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]
    lower: list[np.ndarray] = []
    for p in pts:
        while len(lower) >= 2 and _cross_2d(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper: list[np.ndarray] = []
    for p in pts[::-1]:
        while len(upper) >= 2 and _cross_2d(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    return np.vstack((lower[:-1], upper[:-1]))


def _polygon_area(poly: np.ndarray) -> float:
    if len(poly) < 3:
        return 0.0
    x = poly[:, 0]
    y = poly[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _rectangle_from_bbox_xy(bbox_min: np.ndarray, bbox_max: np.ndarray) -> np.ndarray:
    return np.array([
        [bbox_min[0], bbox_min[1]],
        [bbox_max[0], bbox_min[1]],
        [bbox_max[0], bbox_max[1]],
        [bbox_min[0], bbox_max[1]],
    ], dtype=float)


def _slice_tolerance(obj_height: float) -> float:
    return float(np.clip(0.02 * max(obj_height, 0.0), 0.01, 0.03))


def _top_surface_candidates(
    obj_vertices: np.ndarray,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    rect_xy: np.ndarray,
    slice_tol: float,
) -> list[dict[str, float | list[list[float]]]]:
    """Approximate multiple plausible upper contact plateaus from vertices."""
    rect_area = max(_polygon_area(rect_xy), 1e-8)
    obj_height = float(max(bbox_max[2] - bbox_min[2], 0.0))
    if obj_height <= 1e-8:
        return [{
            "z": float(bbox_max[2]),
            "hull_xy": rect_xy.tolist(),
            "area": float(rect_area),
            "score": 1.0,
        }]

    z_values = np.asarray(obj_vertices[:, 2], dtype=float)
    lower_bound = float(bbox_min[2] + 0.35 * obj_height)
    upper_z = z_values[z_values >= lower_bound]
    if len(upper_z) < 6:
        upper_z = z_values

    quantiles = np.array([0.55, 0.65, 0.75, 0.85, 0.93, 0.98, 1.0], dtype=float)
    band_tol = float(max(slice_tol, min(0.06, 0.08 * obj_height)))
    candidate_zs = np.quantile(upper_z, quantiles)

    dedup: list[float] = []
    for z in sorted(float(v) for v in candidate_zs):
        if not dedup or abs(z - dedup[-1]) > (0.5 * band_tol):
            dedup.append(z)

    top_candidates: list[dict[str, float | list[list[float]]]] = []
    for z_center in dedup:
        mask = np.abs(z_values - z_center) <= band_tol
        band_xy = obj_vertices[mask][:, :2]
        if len(band_xy) < 3:
            continue
        hull_xy = _convex_hull_2d(band_xy)
        area = _polygon_area(hull_xy)
        if area <= 1e-8:
            continue
        height_score = float(np.clip((z_center - bbox_min[2]) / max(obj_height, 1e-8), 0.0, 1.0))
        area_score = float(np.clip(area / rect_area, 0.0, 1.0))
        density_score = float(np.clip(len(np.unique(band_xy, axis=0)) / max(len(obj_vertices), 1), 0.0, 1.0))
        top_candidates.append({
            "z": float(z_center),
            "hull_xy": hull_xy.tolist(),
            "area": float(area),
            "score": float(0.55 * area_score + 0.25 * density_score + 0.20 * height_score),
        })

    if not top_candidates:
        return [{
            "z": float(bbox_max[2]),
            "hull_xy": rect_xy.tolist(),
            "area": float(rect_area),
            "score": 0.25,
        }]

    top_candidates.sort(
        key=lambda item: (
            float(item["score"]),
            float(item["area"]),
            float(item["z"]),
        ),
        reverse=True,
    )
    kept = top_candidates[:6]

    highest = max(top_candidates, key=lambda item: float(item["z"]))
    if all(abs(float(c["z"]) - float(highest["z"])) > 1e-6 for c in kept):
        kept.append(highest)

    kept.sort(key=lambda item: float(item["z"]))
    return kept


def _build_support_geom(
    obj_vertices: np.ndarray,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
) -> dict[str, Any]:
    """Build lightweight support geometry for one instance."""
    obj_height = float(max(bbox_max[2] - bbox_min[2], 0.0))
    slice_tol = _slice_tolerance(obj_height)

    bottom_mask = np.abs(obj_vertices[:, 2] - bbox_min[2]) <= slice_tol

    bottom_xy = obj_vertices[bottom_mask][:, :2]

    bottom_hull = (
        _convex_hull_2d(bottom_xy)
        if len(bottom_xy) >= 3 else np.empty((0, 2), dtype=float)
    )

    rect_xy = _rectangle_from_bbox_xy(bbox_min, bbox_max)
    if len(bottom_hull) < 3:
        bottom_hull = rect_xy.copy()
    top_candidates = _top_surface_candidates(
        obj_vertices, bbox_min, bbox_max, rect_xy, slice_tol,
    )
    top_hull = np.asarray(
        max(top_candidates, key=lambda item: (float(item["area"]), float(item["score"])))["hull_xy"],
        dtype=float,
    )

    return {
        "bottom_hull_xy": bottom_hull.tolist(),
        "top_hull_xy": top_hull.tolist(),
        "top_surface_candidates": top_candidates,
    }


def parse_scene(
    scene_path: str | Path,
    preloaded_geometry: SceneGeometry | None = None,
) -> dict[str, Any] | None:
    """Parse a single ScanNet scene.

    Args:
        scene_path: Path to the scene directory (e.g. ``scans/scene0000_00/``).

    Returns:
        Scene dict with keys ``scene_id`` and ``objects``.
        Returns ``None`` if the scene does not meet filtering criteria.
    """
    scene_path = Path(scene_path)
    scene_id = scene_path.name

    try:
        if preloaded_geometry is None:
            scene_id, vertices, _faces, seg_indices, anno_list = _load_scene_geometry(scene_path)
        else:
            scene_id, vertices, _faces, seg_indices, anno_list = preloaded_geometry
    except FileNotFoundError as exc:
        logger.warning("Missing scene geometry for %s: %s", scene_id, exc)
        return None

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
        support_geom = _build_support_geom(obj_vertices, bbox_min, bbox_max)

        objects.append(
            {
                "id":           int(instance_id),
                "label":        str(label),
                "center":       center.tolist(),
                "bbox_min":     bbox_min.tolist(),
                "bbox_max":     bbox_max.tolist(),
                "dimensions":   dimensions.tolist(),
                "vertex_count": int(obj_vertices.shape[0]),
                "support_geom": support_geom,
            }
        )

    # Extract room boundaries from wall/floor objects BEFORE filtering them out.
    # These structural elements define the physical room extent.
    STRUCTURAL_LABELS = {"floor", "wall", "ground", "ceiling"}
    structural_objects = [
        o for o in objects if o["label"].lower() in STRUCTURAL_LABELS
    ]
    # Preserve walls before excluded-label filtering; they are later used as
    # allocentric/reference anchors even though they are never question subjects.
    wall_objects = [
        o for o in objects if o["label"].lower() == "wall"
    ]
    room_bounds = None
    if structural_objects:
        all_mins = np.array([o["bbox_min"] for o in structural_objects])
        all_maxs = np.array([o["bbox_max"] for o in structural_objects])
        room_bounds = {
            "bbox_min": all_mins.min(axis=0).tolist(),
            "bbox_max": all_maxs.max(axis=0).tolist(),
        }

    # Keep question-only excluded labels so they can remain as attachment
    # parents. Ordinary question-subject filtering happens downstream.
    n_before = len(objects)
    objects = [o for o in objects if o["label"].lower() not in ALWAYS_EXCLUDED]
    if n_before != len(objects):
        logger.debug(
            "Scene %s: %d -> %d objects after excluded-label filter",
            scene_id, n_before, len(objects),
        )

    if len(objects) < MIN_OBJECTS:
        logger.info(
            "Scene %s has only %d objects (< %d) — skipping",
            scene_id, len(objects), MIN_OBJECTS,
        )
        return None

    return {
        "scene_id": scene_id,
        "objects": objects,
        "room_bounds": room_bounds,
        "wall_objects": wall_objects,
    }


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
