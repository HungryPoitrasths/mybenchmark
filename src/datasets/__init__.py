"""DataSource factory and public API."""

from __future__ import annotations

from pathlib import Path

from .base import SceneDataSource
from .scannet import ScanNetDataSource
from .scannetpp import ScanNetPPDataSource


def make_data_source(
    dataset: str,
    scene_dir: str | Path,
    *,
    sensor: str = "iphone",
    frame_root: Path | None = None,
) -> SceneDataSource:
    """Create a :class:`SceneDataSource` for a scene.

    Args:
        dataset: ``"scannet"`` or ``"scannetpp"``.
        scene_dir: Path to the scene directory.
        sensor: ``"iphone"`` or ``"dslr"`` (ScanNet++ only, default ``"iphone"``).
        frame_root: Root for extracted iPhone frames (ScanNet++ only).

    Returns:
        A :class:`SceneDataSource` instance appropriate for *dataset*.
    """
    if dataset == "scannet":
        return ScanNetDataSource(scene_dir)
    if dataset == "scannetpp":
        return ScanNetPPDataSource(
            scene_dir, sensor=sensor, frame_root=frame_root,
        )
    raise ValueError(
        f"Unknown dataset: {dataset!r}. Expected 'scannet' or 'scannetpp'."
    )
