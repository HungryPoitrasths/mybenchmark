"""Abstract DataSource interface for scene data access.

All scene loaders (ScanNet, ScanNet++) implement this interface so that
downstream pipeline code never needs to know which dataset produced a
particular scene.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class DepthFrame:
    """A depth image registered to one camera frame."""

    image_m: np.ndarray
    intrinsics: "CameraIntrinsics"
    valid_ratio: float
    source: str


class SceneDataSource(abc.ABC):
    """Uniform access to scene geometry, camera intrinsics, poses, and images.

    Subclasses implement the loader logic for a specific dataset / sensor
    combination.
    """

    def __init__(self, scene_dir: str | Path) -> None:
        self.scene_dir = Path(scene_dir)
        self.scene_id = self.scene_dir.name

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def load_scene(self) -> dict[str, Any]:
        """Return a scene dict with ``scene_id`` and ``objects``.

        Equivalent to :func:`scene_parser.parse_scene`.
        """
        ...

    @abc.abstractmethod
    def load_intrinsics(self) -> "CameraIntrinsics":
        """Return the intrinsic parameters of the active sensor.

        Must return a :class:`CameraIntrinsics` regardless of whether the
        underlying camera model is pinhole, fisheye, or OPENCV.
        """
        ...

    @abc.abstractmethod
    def load_poses(self) -> "dict[str, CameraPose]":
        """Return per-frame camera poses (world-to-camera convention).

        Returns:
            ``dict`` mapping ``image_name`` → :class:`CameraPose`.
        """
        ...

    @abc.abstractmethod
    def image_path(self, image_name: str) -> Path:
        """Return the absolute path to the image for *image_name*."""
        ...

    @abc.abstractmethod
    def validate(self) -> dict[str, Any]:
        """Run basic integrity checks and return a diagnostic dict.

        Subclasses should include at least:
        ``dataset``, ``scene_id``, ``poses``, ``intrinsics``.
        """
        ...

    # ------------------------------------------------------------------
    # Optional capabilities (subclasses may override)
    # ------------------------------------------------------------------

    def load_axis_alignment(self) -> np.ndarray:
        """Return 4x4 axis-alignment matrix. Default is identity."""
        return np.eye(4, dtype=np.float64)

    def mesh_path(self) -> Path:
        """Return the path to the scene mesh (used by RayCaster).

        Subclasses that support mesh-based occlusion **must** override this.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement mesh_path()"
        )

    def load_depth_intrinsics(self) -> "CameraIntrinsics | None":
        """Return depth sensor intrinsics, or ``None`` if depth is unavailable."""
        return None

    def depth_image_path(self, image_name: str) -> Path | None:
        """Return path to per-frame depth image, or ``None`` if unavailable."""
        return None

    def load_depth_frame(self, image_name: str) -> DepthFrame | None:
        """Load a metric depth image and its matching intrinsics."""
        depth_path = self.depth_image_path(image_name)
        if depth_path is None or not depth_path.is_file():
            return None
        from ..utils.depth_occlusion import load_depth_image

        try:
            intrinsics = self.load_depth_intrinsics()
            image_m = load_depth_image(depth_path)
        except (FileNotFoundError, OSError, ValueError):
            return None
        if intrinsics is None:
            return None
        valid = np.isfinite(image_m) & (image_m > 0.0)
        return DepthFrame(
            image_m=image_m,
            intrinsics=intrinsics,
            valid_ratio=float(np.count_nonzero(valid) / max(image_m.size, 1)),
            source="sensor_png",
        )
