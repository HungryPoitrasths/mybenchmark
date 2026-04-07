"""Calibrate distance bins from raw ScanNet scenes.

This script requires the original ScanNet mesh assets under ``--data_root`` so
that ``load_instance_mesh_data()`` can build per-instance surface samples. If
instance meshes are missing, scenes are skipped and the resulting calibration
is incomplete.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from src.qa_generator import enrich_objects_with_distance_geometry
from src.relation_engine import DISTANCE_BIN_SPECS, compute_distance_details
from src.scene_parser import load_instance_mesh_data, load_scannet_label_map, parse_scene

CONTACT_DISTANCE_M = 0.05
MIN_THRESHOLD_GAP_M = 0.4
ROUND_TO_M = 0.1
SUGGESTED_MARGIN_CANDIDATES = [0.05, 0.075, 0.1, 0.125, 0.15, 0.2]

logger = logging.getLogger(__name__)


def _scene_dirs(data_root: Path) -> list[Path]:
    return sorted(
        path for path in data_root.iterdir()
        if path.is_dir() and (path / "pose").exists()
    )


def _default_thresholds() -> list[float]:
    return [
        float(spec["upper_bound_m"])
        for spec in DISTANCE_BIN_SPECS
        if np.isfinite(float(spec["upper_bound_m"]))
    ]


def _round_to_step(value: float, step: float = ROUND_TO_M) -> float:
    return round(float(value) / step) * step


def _enforce_min_threshold_gap(
    thresholds: list[float],
    *,
    min_gap: float = MIN_THRESHOLD_GAP_M,
) -> list[float]:
    if not thresholds:
        return []
    ordered = [float(thresholds[0])]
    for threshold in thresholds[1:]:
        ordered.append(max(float(threshold), ordered[-1] + float(min_gap)))
    return [round(value, 1) for value in ordered]


def _calibrate_thresholds(positive_distances: list[float]) -> list[float]:
    if not positive_distances:
        return [round(value, 1) for value in _default_thresholds()]
    raw = np.quantile(np.asarray(positive_distances, dtype=np.float64), [0.25, 0.5, 0.75])
    rounded = [_round_to_step(float(value)) for value in raw.tolist()]
    return _enforce_min_threshold_gap(rounded)


def _build_labels(thresholds: list[float]) -> list[str]:
    t1, t2, t3 = [round(float(value), 1) for value in thresholds]
    return [
        f"very close (<{t1:.1f}m)",
        f"close ({t1:.1f}-{t2:.1f}m)",
        f"moderate ({t2:.1f}-{t3:.1f}m)",
        f"far (>{t3:.1f}m)",
    ]


def _bin_id_for_distance(distance_m: float, thresholds: list[float]) -> str:
    if distance_m < thresholds[0]:
        return "very_close"
    if distance_m < thresholds[1]:
        return "close"
    if distance_m < thresholds[2]:
        return "moderate"
    return "far"


def _near_boundary_ratio(
    distances: list[float],
    thresholds: list[float],
    margin: float,
) -> float:
    if not distances:
        return 0.0
    finite_thresholds = [float(value) for value in thresholds]
    flagged = sum(
        1 for distance in distances
        if any(abs(float(distance) - boundary) < float(margin) for boundary in finite_thresholds)
    )
    return float(flagged / max(len(distances), 1))


def _suggest_boundary_margin(
    distances: list[float],
    thresholds: list[float],
) -> tuple[float, float]:
    if not distances:
        return 0.1, 0.0
    scored = [
        (
            margin,
            _near_boundary_ratio(distances, thresholds, margin),
        )
        for margin in SUGGESTED_MARGIN_CANDIDATES
    ]
    best_margin, best_ratio = min(scored, key=lambda item: abs(item[1] - 0.1))
    return float(best_margin), float(best_ratio)


def calibrate_distance_bins(
    data_root: Path,
    *,
    max_scenes: int | None = None,
    n_surface_samples: int = 512,
) -> dict[str, Any]:
    """Fit distance bins from scenes with loadable instance meshes.

    Scenes whose per-instance mesh data cannot be loaded are skipped with a
    warning. Calibration quality therefore depends on ``--data_root``
    containing the raw ScanNet meshes required by ``load_instance_mesh_data``.
    """
    all_distances: list[float] = []
    scene_count = 0
    pair_count = 0
    skipped_missing_mesh = 0
    scene_dirs = _scene_dirs(data_root)
    if max_scenes is not None:
        scene_dirs = scene_dirs[: max(0, int(max_scenes))]

    for scene_dir in scene_dirs:
        scene = parse_scene(scene_dir)
        if scene is None:
            continue
        objects = list(scene.get("objects", []))
        if len(objects) < 2:
            continue
        try:
            instance_mesh_data = load_instance_mesh_data(
                scene_dir,
                instance_ids=[int(obj["id"]) for obj in objects],
                n_surface_samples=int(n_surface_samples),
            )
        except Exception as exc:
            skipped_missing_mesh += 1
            logger.warning(
                "Skipping %s: instance mesh data could not be loaded. "
                "Distance calibration requires the raw ScanNet meshes and per-instance geometry "
                "under --data_root. Error: %s",
                scene_dir.name,
                exc,
            )
            continue
        enrich_objects_with_distance_geometry(objects, instance_mesh_data)
        scene_count += 1
        for idx, obj_a in enumerate(objects):
            for obj_b in objects[idx + 1:]:
                if str(obj_a.get("label", "")) == str(obj_b.get("label", "")):
                    continue
                details = compute_distance_details(obj_a, obj_b)
                all_distances.append(float(details["distance_m"]))
                pair_count += 1

    positive_distances = [distance for distance in all_distances if distance > CONTACT_DISTANCE_M]
    thresholds = _calibrate_thresholds(positive_distances)
    labels = _build_labels(thresholds)
    suggested_margin, near_ratio = _suggest_boundary_margin(all_distances, thresholds)
    if skipped_missing_mesh > 0:
        logger.warning(
            "Skipped %d scene(s) because instance mesh data was unavailable. "
            "Without raw ScanNet meshes, threshold recalibration will be partial.",
            skipped_missing_mesh,
        )
    if scene_count == 0 and scene_dirs:
        logger.warning(
            "No usable scenes were calibrated. Check that --data_root contains raw ScanNet meshes "
            "and instance geometry files required for surface sampling."
        )
    counts_per_bin = {
        bin_id: 0
        for bin_id in ("very_close", "close", "moderate", "far")
    }
    for distance in all_distances:
        counts_per_bin[_bin_id_for_distance(float(distance), thresholds)] += 1

    bins = [
        {
            "bin_id": "very_close",
            "upper_bound_m": thresholds[0],
            "display_label": labels[0],
        },
        {
            "bin_id": "close",
            "upper_bound_m": thresholds[1],
            "display_label": labels[1],
        },
        {
            "bin_id": "moderate",
            "upper_bound_m": thresholds[2],
            "display_label": labels[2],
        },
        {
            "bin_id": "far",
            "upper_bound_m": None,
            "display_label": labels[3],
        },
    ]
    return {
        "scene_count": int(scene_count),
        "skipped_scene_count_missing_mesh": int(skipped_missing_mesh),
        "pair_count": int(pair_count),
        "positive_pair_count": int(len(positive_distances)),
        "contact_or_overlap_ratio": (
            float(sum(distance <= CONTACT_DISTANCE_M for distance in all_distances) / max(len(all_distances), 1))
            if all_distances else 0.0
        ),
        "thresholds_m": thresholds,
        "labels": labels,
        "counts_per_bin": counts_per_bin,
        "near_boundary_ratio": near_ratio,
        "suggested_boundary_margin_m": suggested_margin,
        "bins": bins,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Calibrate closest-point distance bins from raw ScanNet scenes. "
            "This requires the original ScanNet meshes and per-instance geometry "
            "under --data_root so surface samples can be loaded."
        ),
    )
    parser.add_argument(
        "--data_root",
        type=Path,
        required=True,
        help=(
            "Root directory containing raw ScanNet scene folders, meshes, and instance geometry "
            "needed to build per-instance surface samples."
        ),
    )
    parser.add_argument("--output", type=Path, help="Optional JSON output path.")
    parser.add_argument("--max_scenes", type=int, help="Optional cap on scenes processed.")
    parser.add_argument(
        "--n_surface_samples",
        type=int,
        default=512,
        help="Surface samples per instance mesh; requires loadable instance meshes under --data_root.",
    )
    parser.add_argument("--label_map", type=Path, help="Optional ScanNet TSV label map.")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    if not data_root.exists():
        raise SystemExit(f"--data_root does not exist: {data_root}")
    if args.label_map:
        load_scannet_label_map(args.label_map)

    report = calibrate_distance_bins(
        data_root,
        max_scenes=args.max_scenes,
        n_surface_samples=args.n_surface_samples,
    )
    output_text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.write_text(output_text, encoding="utf-8")
    else:
        print(output_text)


if __name__ == "__main__":
    main()
