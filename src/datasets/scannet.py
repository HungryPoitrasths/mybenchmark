"""ScanNet v2 DataSource implementation.

Wraps the existing ScanNet loader functions behind the
:class:`SceneDataSource` interface so that pipeline code
never imports ``colmap_loader`` or ``scene_parser`` directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .base import SceneDataSource


class ScanNetDataSource(SceneDataSource):
    """DataSource for a ScanNet v2 scene.

    Uses the existing ``load_scannet_intrinsics``, ``load_scannet_poses``,
    and ``parse_scene(dataset="scannet")`` internally.
    """

    def load_scene(self) -> dict[str, Any]:
        from ..scene_parser import parse_scene

        scene = parse_scene(self.scene_dir, dataset="scannet")
        if scene is None:
            raise RuntimeError(
                f"Failed to parse ScanNet scene: {self.scene_dir}"
            )
        return scene

    def load_intrinsics(self) -> "CameraIntrinsics":
        from ..utils.colmap_loader import load_scannet_intrinsics

        return load_scannet_intrinsics(self.scene_dir)

    def load_poses(self) -> "dict[str, CameraPose]":
        from ..utils.colmap_loader import (
            load_axis_alignment,
            load_scannet_poses,
        )

        axis_align = load_axis_alignment(self.scene_dir)
        return load_scannet_poses(self.scene_dir, axis_alignment=axis_align)

    def image_path(self, image_name: str) -> Path:
        return self.scene_dir / "color" / image_name

    # ------------------------------------------------------------------
    # Optional capabilities
    # ------------------------------------------------------------------

    def load_axis_alignment(self) -> np.ndarray:
        from ..utils.colmap_loader import load_axis_alignment

        return load_axis_alignment(self.scene_dir)

    def mesh_path(self) -> Path:
        for name in (f"{self.scene_id}_vh_clean.ply", f"{self.scene_id}_vh_clean_2.ply"):
            candidate = self.scene_dir / name
            if candidate.is_file():
                return candidate
        return self.scene_dir / f"{self.scene_id}_vh_clean.ply"

    def load_depth_intrinsics(self) -> "CameraIntrinsics | None":
        from ..utils.colmap_loader import load_scannet_depth_intrinsics

        return load_scannet_depth_intrinsics(self.scene_dir)

    def depth_image_path(self, image_name: str) -> Path | None:
        import os

        frame_id, _ = os.path.splitext(image_name)
        return self.scene_dir / "depth" / f"{frame_id}.png"

    def validate(self) -> dict[str, Any]:
        from ..utils.colmap_loader import load_scannet_intrinsics

        issues: list[str] = []
        pose_count = 0
        missing_images = 0

        # Intrinsics
        try:
            intr = load_scannet_intrinsics(self.scene_dir)
            intr_str = f"{intr.width}x{intr.height}"
        except Exception as exc:
            intr_str = "error"
            issues.append(f"intrinsics: {exc}")

        # Poses
        poses = {}
        try:
            poses = self.load_poses()
            pose_count = len(poses)
        except Exception as exc:
            issues.append(f"poses: {exc}")

        # Images
        for name in poses:
            if not self.image_path(name).is_file():
                missing_images += 1
        if missing_images:
            issues.append(f"missing_images: {missing_images}")

        return {
            "dataset": "scannet",
            "scene_id": self.scene_id,
            "intrinsics": intr_str,
            "poses": pose_count,
            "missing_images": missing_images,
            "issues": issues,
        }
