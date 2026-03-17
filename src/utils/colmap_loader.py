"""ScanNet camera pose and intrinsics loader.

ScanNet stores:
  - intrinsic/intrinsic_color.txt  — 4×4 intrinsic matrix (shared across all frames)
  - pose/<frame_id>.txt            — 4×4 camera-to-world matrix per frame
  - <scene_id>.txt                 — scene metadata (colorWidth/Height, axisAlignment)

The axisAlignment matrix rotates/translates the raw scan into an upright,
gravity-aligned coordinate frame.  It must be applied to both mesh vertices
(in scene_parser) and to camera poses (here).

Public API (keeps the same names used by the rest of the codebase):
  CameraIntrinsics         — unchanged
  CameraPose               — unchanged
  load_scannet_intrinsics  — replaces load_colmap_intrinsics
  load_scannet_poses       — replaces load_colmap_poses
  load_axis_alignment      — new helper, consumed by scene_parser & frame_selector
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class CameraIntrinsics:
    """Pinhole camera intrinsic parameters."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    def to_matrix(self) -> np.ndarray:
        """Return 3×3 intrinsic matrix K."""
        return np.array(
            [[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]],
            dtype=np.float64,
        )


@dataclass
class CameraPose:
    """Camera extrinsic parameters (world-to-camera convention)."""

    image_name: str
    rotation: np.ndarray    # 3×3  R  (world → camera)
    translation: np.ndarray  # 3-vec t  (world → camera)

    @property
    def position(self) -> np.ndarray:
        """Camera centre in world coordinates: C = -R^T @ t."""
        return -self.rotation.T @ self.translation

    @position.setter
    def position(self, new_pos: np.ndarray) -> None:
        """Move camera to *new_pos* (world coords) keeping orientation fixed."""
        self.translation = -self.rotation @ new_pos

    def world_to_camera_point(self, point_world: np.ndarray) -> np.ndarray:
        """Transform a world-space point into camera coordinates."""
        return self.rotation @ point_world + self.translation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_axis_alignment(scene_dir: Path) -> np.ndarray:
    """Read the 4×4 axisAlignment matrix from <scene_id>.txt.

    Returns identity if the file or key is absent.
    """
    meta_file = scene_dir / f"{scene_dir.name}.txt"
    if not meta_file.exists():
        return np.eye(4, dtype=np.float64)
    with open(meta_file, encoding="utf-8") as f:
        for line in f:
            if "axisAlignment" in line:
                values = [float(x) for x in line.split("=")[1].strip().split()]
                return np.array(values, dtype=np.float64).reshape(4, 4)
    return np.eye(4, dtype=np.float64)


def _read_scene_meta(scene_dir: Path) -> dict[str, str]:
    """Parse key = value lines from <scene_id>.txt into a dict."""
    meta: dict[str, str] = {}
    meta_file = scene_dir / f"{scene_dir.name}.txt"
    if not meta_file.exists():
        return meta
    with open(meta_file, encoding="utf-8") as f:
        for line in f:
            if "=" in line:
                key, _, val = line.partition("=")
                meta[key.strip()] = val.strip()
    return meta


# ---------------------------------------------------------------------------
# Public loaders
# ---------------------------------------------------------------------------

def load_scannet_intrinsics(scene_dir: Path) -> CameraIntrinsics:
    """Load colour-camera intrinsics for a ScanNet scene.

    Reads the 4×4 matrix from ``intrinsic/intrinsic_color.txt`` and the image
    dimensions from ``<scene_id>.txt`` (falls back to 1296 × 968).
    """
    # Support both intrinsic/intrinsic_color.txt and intrinsic_color.txt (at root)
    intr_file = scene_dir / "intrinsic" / "intrinsic_color.txt"
    if not intr_file.exists():
        intr_file = scene_dir / "intrinsic_color.txt"
    M = np.loadtxt(str(intr_file))   # 4×4
    fx, fy = float(M[0, 0]), float(M[1, 1])
    cx, cy = float(M[0, 2]), float(M[1, 2])

    meta = _read_scene_meta(scene_dir)
    width  = int(meta.get("colorWidth",  1296))
    height = int(meta.get("colorHeight",  968))

    return CameraIntrinsics(width=width, height=height, fx=fx, fy=fy, cx=cx, cy=cy)


def load_scannet_depth_intrinsics(scene_dir: Path) -> CameraIntrinsics:
    """Load depth-camera intrinsics for a ScanNet scene.

    Reads the 4×4 matrix from ``intrinsic/intrinsic_depth.txt`` and the depth
    image dimensions from ``<scene_id>.txt`` (falls back to 640 × 480).
    """
    intr_file = scene_dir / "intrinsic" / "intrinsic_depth.txt"
    if not intr_file.exists():
        intr_file = scene_dir / "intrinsic_depth.txt"
    M = np.loadtxt(str(intr_file))  # 4×4
    fx, fy = float(M[0, 0]), float(M[1, 1])
    cx, cy = float(M[0, 2]), float(M[1, 2])

    meta = _read_scene_meta(scene_dir)
    width  = int(meta.get("depthWidth",  640))
    height = int(meta.get("depthHeight", 480))

    return CameraIntrinsics(width=width, height=height, fx=fx, fy=fy, cx=cx, cy=cy)


def load_scannet_poses(
    scene_dir: Path,
    axis_alignment: np.ndarray | None = None,
) -> dict[str, CameraPose]:
    """Load all valid per-frame camera poses for a ScanNet scene.

    Each ``pose/<frame_id>.txt`` contains a 4×4 **camera-to-world** matrix.
    This function converts them to the world-to-camera convention used by
    ``CameraPose`` and optionally applies the axis-alignment transform so that
    poses live in the same coordinate frame as the aligned mesh.

    Args:
        scene_dir:       Root directory of the ScanNet scene.
        axis_alignment:  4×4 axis-alignment matrix M.  If None, identity is used.

    Returns:
        dict mapping ``"<frame_id>.jpg"`` → ``CameraPose``.
    """
    pose_dir  = scene_dir / "pose"
    color_dir = scene_dir / "color"
    if not pose_dir.exists():
        return {}

    M = axis_alignment if axis_alignment is not None else np.eye(4, dtype=np.float64)

    poses: dict[str, CameraPose] = {}
    for pose_file in sorted(pose_dir.glob("*.txt"), key=lambda p: int(p.stem)):
        frame_id   = pose_file.stem          # "0", "1", …
        image_name = f"{frame_id}.jpg"

        # Skip frames whose colour image is missing
        if not (color_dir / image_name).exists():
            continue

        T_c2w = np.loadtxt(str(pose_file))   # 4×4 camera-to-world

        # ScanNet marks tracking failures with ±inf / nan — skip them
        if not np.isfinite(T_c2w).all():
            continue

        # Apply axis alignment: T_aligned = M @ T_original
        T_c2w_aligned = M @ T_c2w

        R_c2w = T_c2w_aligned[:3, :3]
        t_c2w = T_c2w_aligned[:3, 3]

        # Convert to world-to-camera
        R_w2c = R_c2w.T
        t_w2c = -R_c2w.T @ t_c2w

        poses[image_name] = CameraPose(
            image_name=image_name,
            rotation=R_w2c.astype(np.float64),
            translation=t_w2c.astype(np.float64),
        )

    return poses


# ---------------------------------------------------------------------------
# Legacy aliases (kept so that any code still importing the old names works)
# ---------------------------------------------------------------------------

def load_colmap_intrinsics(cameras_txt: str | Path) -> dict:  # type: ignore[return]
    raise NotImplementedError(
        "load_colmap_intrinsics is not available for ScanNet data. "
        "Use load_scannet_intrinsics(scene_dir) instead."
    )


def load_colmap_poses(images_txt: str | Path) -> dict:  # type: ignore[return]
    raise NotImplementedError(
        "load_colmap_poses is not available for ScanNet data. "
        "Use load_scannet_poses(scene_dir, axis_alignment) instead."
    )
