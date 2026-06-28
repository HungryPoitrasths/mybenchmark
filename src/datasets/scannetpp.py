"""ScanNet++ native scene discovery and geometry loading.

This module covers only the 3D inputs needed by scene parsing. DSLR camera
poses, iPhone streams, VLM review, and pipeline integration are handled by
later adapter layers.
"""

from __future__ import annotations

import gzip
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

SceneGeometry = tuple[str, np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]
"""(scene_id, vertices, faces, seg_indices, anno_list)."""


@dataclass(frozen=True)
class ScanNetPPDSLRIntrinsics:
    """ScanNet++ DSLR camera parameters from nerfstudio/transforms.json."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    k1: float = 0.0
    k2: float = 0.0
    k3: float = 0.0
    k4: float = 0.0
    camera_model: str = "PINHOLE"

    @property
    def is_fisheye(self) -> bool:
        return self.camera_model.upper() == "OPENCV_FISHEYE"

    def to_pinhole(self) -> "CameraIntrinsics":
        from ..utils.colmap_loader import CameraIntrinsics

        return CameraIntrinsics(
            width=self.width,
            height=self.height,
            fx=self.fx,
            fy=self.fy,
            cx=self.cx,
            cy=self.cy,
        )

    def camera_matrix(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    def distortion_coeffs(self) -> np.ndarray:
        return np.array([self.k1, self.k2, self.k3, self.k4], dtype=np.float64)


def has_scannetpp_geometry(path: Path) -> bool:
    """Return True when path has the ScanNet++ 3D annotation files."""
    path = Path(path)
    return (
        (path / "scans" / "mesh_aligned_0.05.ply").is_file()
        and (path / "scans" / "segments.json").is_file()
        and (path / "scans" / "segments_anno.json").is_file()
    )


def has_scannetpp_iphone(path: Path) -> bool:
    """Return True when path has 3D files plus iPhone COLMAP/video assets."""
    path = Path(path)
    return (
        has_scannetpp_geometry(path)
        and (path / "iphone" / "rgb.mkv").is_file()
        and (path / "iphone" / "colmap" / "cameras.txt").is_file()
        and (path / "iphone" / "colmap" / "images.txt").is_file()
    )


def has_scannetpp_dslr(path: Path) -> bool:
    """Return True when path has 3D files plus DSLR assets."""
    path = Path(path)
    return (
        has_scannetpp_geometry(path)
        and (path / "dslr" / "nerfstudio" / "transforms.json").is_file()
        and (path / "dslr" / "resized_images").is_dir()
    )


def is_scannetpp_scene_dir(path: Path) -> bool:
    """Return True when path has 3D files plus the DSLR files needed later."""
    return has_scannetpp_dslr(path)


def resolve_scannetpp_scene_dirs(data_root: Path, sensor: str = "iphone") -> list[Path]:
    """Discover ScanNet++ scene directories for the requested sensor.

    Supports either a dataset root containing scene subdirectories or one scene
    directory directly. Use ``has_scannetpp_geometry`` for 3D-only checks.
    """
    data_root = Path(data_root).resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"Not a directory: {data_root}")

    if sensor == "iphone":
        predicate = has_scannetpp_iphone
    elif sensor == "dslr":
        predicate = has_scannetpp_dslr
    else:
        raise ValueError(f"sensor must be 'iphone' or 'dslr', got {sensor!r}")

    if predicate(data_root):
        return [data_root]

    return sorted(
        child for child in data_root.iterdir()
        if child.is_dir() and predicate(child)
    )


def load_scannetpp_scene_geometry(scene_path: Path) -> SceneGeometry:
    """Load native ScanNet++ mesh, segments, and annotations.

    Returns a SceneGeometry tuple compatible with scene_parser:
        (scene_id, vertices, faces, seg_indices, anno_list)

    Raises:
        ValueError: the mesh is empty, segIndices do not match vertices, or
        annotations are missing.
    """
    import open3d as o3d

    scene_path = Path(scene_path).resolve()
    scene_id = scene_path.name

    mesh_file = scene_path / "scans" / "mesh_aligned_0.05.ply"
    seg_file = scene_path / "scans" / "segments.json"
    anno_file = scene_path / "scans" / "segments_anno.json"

    if not mesh_file.is_file():
        raise FileNotFoundError(f"Missing mesh: {mesh_file}")
    mesh = o3d.io.read_triangle_mesh(str(mesh_file))
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles, dtype=np.int64)

    if len(vertices) == 0:
        raise ValueError(f"Empty mesh in scene {scene_id}")
    if len(faces) == 0:
        raise ValueError(f"Mesh has no faces in scene {scene_id}")

    if not seg_file.is_file():
        raise FileNotFoundError(f"Missing segments: {seg_file}")
    with open(seg_file, "r", encoding="utf-8") as f:
        seg_data = json.load(f)
    seg_indices = np.asarray(seg_data["segIndices"], dtype=np.int64)

    if len(seg_indices) != len(vertices):
        raise ValueError(
            f"segIndices length {len(seg_indices)} != "
            f"mesh vertices {len(vertices)} in scene {scene_id}"
        )

    if not anno_file.is_file():
        raise FileNotFoundError(f"Missing annotations: {anno_file}")
    with open(anno_file, "r", encoding="utf-8") as f:
        anno_data = json.load(f)

    if isinstance(anno_data, dict):
        if "segGroups" not in anno_data:
            raise ValueError(
                f"segments_anno.json has no 'segGroups' key in scene {scene_id}; "
                f"available keys: {sorted(anno_data.keys())}"
            )
        anno_list = anno_data["segGroups"]
    else:
        anno_list = anno_data

    if len(anno_list) == 0:
        raise ValueError(f"No annotations found in scene {scene_id}")

    # Lightweight annotation sanity check. Some datasets may contain a few
    # stale segment ids, so this warns without blocking parsing.
    available_segments: set[int] = set(np.unique(seg_indices).tolist())
    missing: list[tuple[int, int]] = []
    for anno in anno_list:
        obj_id = anno.get("id", anno.get("objectId"))
        for seg_id in anno.get("segments", []):
            if int(seg_id) not in available_segments:
                missing.append((int(obj_id), int(seg_id)))
    if missing:
        logger.warning(
            "Scene %s: %d annotation segment(s) not found in segIndices "
            "(first 10: %s)",
            scene_id,
            len(missing),
            missing[:10],
        )

    logger.info(
        "Loaded ScanNet++ scene %s: %d vertices, %d faces, %d objects",
        scene_id,
        len(vertices),
        len(faces),
        len(anno_list),
    )
    return scene_id, vertices, faces, seg_indices, anno_list


# ===========================================================================
# 板块 3: DSLR 相机 intrinsics / poses 读取
# ===========================================================================

def load_scannetpp_dslr_camera(scene_dir: str | Path) -> ScanNetPPDSLRIntrinsics:
    """Load the native ScanNet++ DSLR camera model."""
    transforms = _read_transforms(Path(scene_dir))

    return ScanNetPPDSLRIntrinsics(
        width=int(transforms["w"]),
        height=int(transforms["h"]),
        fx=float(transforms["fl_x"]),
        fy=float(transforms["fl_y"]),
        cx=float(transforms["cx"]),
        cy=float(transforms["cy"]),
        k1=float(transforms.get("k1", 0.0)),
        k2=float(transforms.get("k2", 0.0)),
        k3=float(transforms.get("k3", 0.0)),
        k4=float(transforms.get("k4", 0.0)),
        camera_model=str(transforms.get("camera_model", "PINHOLE")),
    )


def load_scannetpp_dslr_intrinsics(scene_dir: str | Path) -> "CameraIntrinsics":
    """Load DSLR intrinsics with distortion parameters preserved.

    Returns :class:`CameraIntrinsics` with ``distortion_model`` and
    ``distortion_params`` populated from transforms.json when the camera
    model is ``OPENCV_FISHEYE``.  Downstream projection uses
    :func:`~..utils.coordinate_transform.project_camera_points_to_image`
    which automatically applies the correct distortion model.
    """
    from ..utils.colmap_loader import CameraIntrinsics

    cam = load_scannetpp_dslr_camera(scene_dir)
    if cam.is_fisheye:
        return CameraIntrinsics(
            width=cam.width,
            height=cam.height,
            fx=cam.fx,
            fy=cam.fy,
            cx=cam.cx,
            cy=cam.cy,
            distortion_model=cam.camera_model,
            distortion_params=cam.distortion_coeffs(),
        )
    return cam.to_pinhole()


def project_scannetpp_dslr_point(
    point_world: np.ndarray,
    pose: "CameraPose",
    camera: ScanNetPPDSLRIntrinsics,
) -> tuple[np.ndarray | None, float]:
    """Project a world point into the native ScanNet++ DSLR image."""
    from ..utils.coordinate_transform import world_to_camera

    point_cam = world_to_camera(np.asarray(point_world, dtype=np.float64), pose)
    depth = float(point_cam[2])
    if depth <= 0:
        return None, depth

    if camera.is_fisheye:
        import cv2

        uv, _ = cv2.fisheye.projectPoints(
            point_cam.reshape(1, 1, 3),
            np.zeros((3, 1), dtype=np.float64),
            np.zeros((3, 1), dtype=np.float64),
            camera.camera_matrix(),
            camera.distortion_coeffs(),
        )
        return uv.reshape(2).astype(np.float64), depth

    u = camera.fx * point_cam[0] / depth + camera.cx
    v = camera.fy * point_cam[1] / depth + camera.cy
    return np.array([u, v], dtype=np.float64), depth


def load_scannetpp_dslr_poses(
    scene_dir: str | Path,
    z_offset: float = 0.0,
) -> "dict[str, CameraPose]":
    """Load per-frame DSLR camera poses from *transforms.json*.

    Filters out frames where ``is_bad`` is ``True``, whose JPEG is
    missing, or whose JPEG is unreadable by OpenCV.  Converts
    camera-to-world matrices to the world-to-camera convention used by
    :class:`CameraPose` (OpenCV / COLMAP axes).

    Args:
        scene_dir: Path to the ScanNet++ scene directory.
        z_offset: Optional Z-axis translation (in world units) applied to
            every camera position.  Use :func:`compute_scannetpp_dslr_z_alignment`
            to estimate this from object annotations.

    Returns:
        ``dict`` mapping ``image_name`` (e.g. ``"DSC00001.JPG"``) to
        :class:`CameraPose`.
    """
    import cv2

    from ..utils.colmap_loader import CameraPose

    scene_dir = Path(scene_dir)
    transforms = _read_transforms(scene_dir)
    image_dir = scene_dir / "dslr" / "resized_images"

    poses: dict[str, CameraPose] = {}
    for frame in transforms["frames"]:
        image_name = frame["file_path"]

        if frame.get("is_bad") is True:
            continue

        image_path = image_dir / image_name
        if not image_path.is_file():
            continue
        if _read_image_unicode_safe(image_path) is None:
            continue

        T_c2w = np.asarray(frame["transform_matrix"], dtype=np.float64)
        if not np.isfinite(T_c2w).all():
            continue

        R_c2w = T_c2w[:3, :3]
        t_c2w = T_c2w[:3, 3]

        R_w2c = R_c2w.T
        t_w2c = -R_w2c @ t_c2w

        if z_offset != 0.0:
            t_w2c = t_w2c - R_w2c @ np.array([0.0, 0.0, z_offset], dtype=np.float64)

        poses[image_name] = CameraPose(
            image_name=image_name,
            rotation=R_w2c,
            translation=t_w2c,
        )

    return poses


def compute_scannetpp_dslr_z_alignment(scene_dir: str | Path) -> float:
    """Estimate Z-offset between camera and object coordinate frames.

    The ScanNet++ ``mesh_aligned_0.05.ply`` and the Nerfstudio camera poses
    often use coordinate frames with different Z origins.  This function
    returns ``obj_z_mean - cam_z_mean`` so that calling
    ``load_scannetpp_dslr_poses(scene_dir, z_offset=...)`` brings cameras
    to roughly the same height as the scene objects.

    Returns 0.0 when no ``segments_anno.json`` is available.
    """
    scene_dir = Path(scene_dir)
    anno_path = scene_dir / "scans" / "segments_anno.json"
    if not anno_path.is_file():
        return 0.0

    with open(anno_path, encoding="utf-8") as f:
        anno_data = json.load(f)
    anno_list = anno_data if isinstance(anno_data, list) else anno_data.get("segGroups", [])
    if not anno_list:
        return 0.0

    obj_zs = []
    for anno in anno_list:
        obb = anno.get("obb")
        if obb and "centroid" in obb:
            obj_zs.append(float(obb["centroid"][2]))
    if not obj_zs:
        return 0.0
    obj_z_mean = float(np.mean(obj_zs))

    transforms = _read_transforms(scene_dir)
    cam_zs = []
    for frame in transforms["frames"]:
        if frame.get("is_bad") is True:
            continue
        T_c2w = np.asarray(frame["transform_matrix"], dtype=np.float64)
        if not np.isfinite(T_c2w).all():
            continue
        cam_zs.append(float(T_c2w[2, 3]))
    if not cam_zs:
        return 0.0
    cam_z_mean = float(np.mean(cam_zs))

    return obj_z_mean - cam_z_mean


def get_scannetpp_dslr_pose_stats(scene_dir: str | Path) -> dict[str, int]:
    """Return skip-reason counts for the DSLR frames in *transforms.json*.

    Useful for diagnosing why certain frames were excluded by
    :func:`load_scannetpp_dslr_poses`.

    Return keys: ``total``, ``is_bad``, ``missing_image``,
    ``unreadable_image``, ``loaded``.
    """
    import cv2

    scene_dir = Path(scene_dir)
    transforms = _read_transforms(scene_dir)
    image_dir = scene_dir / "dslr" / "resized_images"

    total = len(transforms["frames"])
    is_bad = 0
    missing_image = 0
    unreadable_image = 0
    loaded = 0

    for frame in transforms["frames"]:
        image_name = frame["file_path"]

        if frame.get("is_bad") is True:
            is_bad += 1
            continue

        image_path = image_dir / image_name
        if not image_path.is_file():
            missing_image += 1
            continue

        if _read_image_unicode_safe(image_path) is None:
            unreadable_image += 1
            continue

        T_c2w = np.asarray(frame["transform_matrix"], dtype=np.float64)
        if not np.isfinite(T_c2w).all():
            unreadable_image += 1
            continue

        loaded += 1

    return {
        "total": total,
        "is_bad": is_bad,
        "missing_image": missing_image,
        "unreadable_image": unreadable_image,
        "loaded": loaded,
    }


def scannetpp_dslr_image_path(scene_dir: str | Path, image_name: str) -> Path:
    """Return the absolute path to a DSLR resized image."""
    return Path(scene_dir) / "dslr" / "resized_images" / image_name


def _read_image_unicode_safe(path: Path) -> np.ndarray | None:
    """Read an image on Windows paths that may contain non-ASCII chars."""
    import cv2

    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


# ===========================================================================
# 板块 3: iPhone COLMAP 相机 intrinsics / poses 读取
# ===========================================================================

def load_scannetpp_iphone_intrinsics(scene_dir: str | Path) -> "CameraIntrinsics":
    """Load iPhone camera intrinsics from ``iphone/colmap/cameras.txt``.

    Preserves OPENCV radial-tangential distortion (k1, k2, p1, p2) so that
    downstream projection via :func:`project_camera_points_to_image` uses
    :func:`cv2.projectPoints` instead of a lossy pinhole approximation.

    COLMAP camera models:
      - OPENCV       → ``distortion_model="OPENCV"``, params=[k1,k2,p1,p2]
      - PINHOLE      → ``distortion_model=""``
      - SIMPLE_RADIAL, RADIAL → converted as OPENCV (warned)
    """
    from ..utils.colmap_loader import CameraIntrinsics

    scene_dir = Path(scene_dir)
    cam_path = scene_dir / "iphone" / "colmap" / "cameras.txt"
    cameras = _parse_colmap_cameras(cam_path)
    if not cameras:
        raise FileNotFoundError(f"No camera entries found in {cam_path}")

    cam = cameras[0]
    model = cam.get("model", "PINHOLE").upper()
    params = cam.get("params", [])

    if model in ("OPENCV", "OPENCV_FISHEYE"):
        if len(params) < 8:
            raise ValueError(
                f"COLMAP camera model {model!r} in {cam_path} needs at least "
                f"8 params (fx, fy, cx, cy, distortion...), got {len(params)}"
            )
        k1, k2, p1, p2 = params[4], params[5], params[6], params[7]
        return CameraIntrinsics(
            width=cam["width"],
            height=cam["height"],
            fx=cam["fx"],
            fy=cam["fy"],
            cx=cam["cx"],
            cy=cam["cy"],
            distortion_model=model,
            distortion_params=np.array([k1, k2, p1, p2], dtype=np.float64),
        )
    elif model in ("PINHOLE", ""):
        return CameraIntrinsics(
            width=cam["width"],
            height=cam["height"],
            fx=cam["fx"],
            fy=cam["fy"],
            cx=cam["cx"],
            cy=cam["cy"],
        )
    elif model in ("SIMPLE_RADIAL", "RADIAL"):
        logger.warning(
            "iPhone camera model %s not fully supported; "
            "using OPENCV approximation with k1,k2 from params",
            model,
        )
        k1, k2 = params[0], params[1] if len(params) > 1 else 0.0
        return CameraIntrinsics(
            width=cam["width"],
            height=cam["height"],
            fx=cam["fx"],
            fy=cam["fy"],
            cx=cam["cx"],
            cy=cam["cy"],
            distortion_model="OPENCV",
            distortion_params=np.array([k1, k2, 0.0, 0.0], dtype=np.float64),
        )
    else:
        raise ValueError(
            f"Unsupported COLMAP camera model {model!r} in {cam_path}. "
            f"Supported: PINHOLE, OPENCV, OPENCV_FISHEYE, SIMPLE_RADIAL, RADIAL"
        )


def load_scannetpp_iphone_poses(scene_dir: str | Path) -> "dict[str, CameraPose]":
    """Load iPhone camera poses from ``iphone/colmap/images.txt``.

    COLMAP stores **world-to-camera** poses natively, so no c2w→w2c
    conversion is needed.  The rotation is derived directly from the
    quaternion ``(QW, QX, QY, QZ)``.

    Returns:
        ``dict`` mapping ``image_name`` (e.g. ``"frame_000010.jpg"``) to
        :class:`CameraPose`.
    """
    from ..utils.colmap_loader import CameraPose

    scene_dir = Path(scene_dir)
    img_path = scene_dir / "iphone" / "colmap" / "images.txt"
    images = _parse_colmap_images(img_path)

    poses: dict[str, CameraPose] = {}
    for entry in images:
        image_name = entry["name"]
        qw, qx, qy, qz = entry["qw"], entry["qx"], entry["qy"], entry["qz"]
        tx, ty, tz = entry["tx"], entry["ty"], entry["tz"]

        R_w2c = _colmap_qvec_to_rotmat(qw, qx, qy, qz)
        t_w2c = np.array([tx, ty, tz], dtype=np.float64)

        poses[image_name] = CameraPose(
            image_name=image_name,
            rotation=R_w2c,
            translation=t_w2c,
        )

    return poses


def scannetpp_iphone_image_path(
    scene_dir: str | Path,
    image_name: str,
    frame_root: Path | None = None,
) -> Path:
    """Return the path to an extracted iPhone frame.

    If *frame_root* is not given, defaults to::

        output/scannetpp_iphone_frames/<scene_id>/<image_name>
    """
    scene_dir = Path(scene_dir)
    if frame_root is None:
        # Default relative to project root when used from scripts
        frame_root = Path("output") / "scannetpp_iphone_frames"
    return frame_root / scene_dir.name / image_name


# ---------------------------------------------------------------------------
# Internal helpers (DSLR)
# ---------------------------------------------------------------------------

def _read_transforms(scene_dir: Path) -> dict:
    """Read and return the parsed ``transforms.json``."""
    transforms_path = scene_dir / "dslr" / "nerfstudio" / "transforms.json"
    with open(transforms_path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Internal helpers (iPhone / COLMAP)
# ---------------------------------------------------------------------------

def _colmap_qvec_to_rotmat(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    """Convert a COLMAP quaternion (Hamilton convention) to a 3×3 rotation matrix."""
    return np.array(
        [
            [1 - 2 * qy * qy - 2 * qz * qz, 2 * qx * qy - 2 * qw * qz, 2 * qx * qz + 2 * qw * qy],
            [2 * qx * qy + 2 * qw * qz, 1 - 2 * qx * qx - 2 * qz * qz, 2 * qy * qz - 2 * qw * qx],
            [2 * qx * qz - 2 * qw * qy, 2 * qy * qz + 2 * qw * qx, 1 - 2 * qx * qx - 2 * qy * qy],
        ],
        dtype=np.float64,
    )


def _colmap_open(path: Path):
    """Open a COLMAP text file, transparently handling .gz compression."""
    if not path.exists() and path.suffix != ".gz":
        gz = path.with_suffix(path.suffix + ".gz")
        if gz.exists():
            return gzip.open(gz, "rt", encoding="utf-8")
    return open(path, encoding="utf-8")


def _parse_colmap_cameras(path: Path) -> list[dict[str, Any]]:
    """Parse a COLMAP ``cameras.txt`` into a list of camera dicts."""
    with _colmap_open(path) as f:
        lines = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]

    cameras: list[dict[str, Any]] = []
    for line in lines:
        parts = line.split()
        if len(parts) < 8:
            continue
        camera_id = int(parts[0])
        model = parts[1]
        width = int(parts[2])
        height = int(parts[3])
        params = [float(p) for p in parts[4:]]

        fx, fy, cx, cy = params[0], params[1], params[2], params[3]
        cameras.append(
            {
                "camera_id": camera_id,
                "model": model,
                "width": width,
                "height": height,
                "fx": fx,
                "fy": fy,
                "cx": cx,
                "cy": cy,
                "params": params,
            }
        )
    return cameras


def _parse_colmap_images(path: Path) -> list[dict[str, Any]]:
    """Parse a COLMAP ``images.txt`` into a list of image entry dicts.

    Each image spans two lines (metadata + points2D).  The points2D line
    may be empty.  We read line-by-line (including blank lines) so that
    the two-lines-per-image structure is preserved.
    """
    with _colmap_open(path) as f:
        raw_lines = f.read().splitlines()

    # Skip comment lines
    lines: list[str] = []
    for ln in raw_lines:
        stripped = ln.strip()
        if stripped and not stripped.startswith("#"):
            lines.append(stripped)
        elif not stripped:
            # Preserve blank lines to maintain the alternating 2-line structure
            lines.append("")

    images: list[dict[str, Any]] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line:
            i += 1
            continue

        parts = line.split()
        if len(parts) < 9:
            i += 1
            continue

        image_id = int(parts[0])
        qw, qx, qy, qz = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        tx, ty, tz = float(parts[5]), float(parts[6]), float(parts[7])
        camera_id = int(parts[8])
        name = parts[9]

        images.append(
            {
                "image_id": image_id,
                "qw": qw,
                "qx": qx,
                "qy": qy,
                "qz": qz,
                "tx": tx,
                "ty": ty,
                "tz": tz,
                "camera_id": camera_id,
                "name": name,
            }
        )
        # The next line is the points2D line — skip it
        i += 2

    return images


# ===========================================================================
# 板块 4: DataSource 实现
# ===========================================================================

from .base import SceneDataSource


class ScanNetPPDataSource(SceneDataSource):
    """DataSource for a ScanNet++ scene (iPhone or DSLR).

    Parameters:
        scene_dir:  Path to the scene directory (e.g. ``++data/0d2ee665be``).
        sensor:     ``"iphone"`` (default) or ``"dslr"``.
        frame_root: Root directory for extracted iPhone frames.
                    Only used when ``sensor="iphone"``.
    """

    def __init__(
        self,
        scene_dir: str | Path,
        sensor: str = "iphone",
        frame_root: Path | None = None,
    ) -> None:
        if sensor not in ("iphone", "dslr"):
            raise ValueError(
                f"sensor must be 'iphone' or 'dslr', got {sensor!r}"
            )

        super().__init__(scene_dir)
        self.sensor = sensor
        self.frame_root = (
            Path(frame_root) if frame_root
            else Path("output") / "scannetpp_iphone_frames"
        )

    # ------------------------------------------------------------------
    # Public interface (mirrors SceneDataSource)
    # ------------------------------------------------------------------

    def load_scene(self) -> "dict[str, Any]":
        from ..scene_parser import parse_scene

        scene = parse_scene(self.scene_dir, dataset="scannetpp")
        if scene is None:
            raise RuntimeError(
                f"Failed to parse ScanNet++ scene: {self.scene_dir}"
            )
        return scene

    def load_intrinsics(self) -> "CameraIntrinsics":
        if self.sensor == "iphone":
            return load_scannetpp_iphone_intrinsics(self.scene_dir)
        else:
            return load_scannetpp_dslr_intrinsics(self.scene_dir)

    def load_poses(self) -> "dict[str, CameraPose]":
        if self.sensor == "iphone":
            return load_scannetpp_iphone_poses(self.scene_dir)
        else:
            return load_scannetpp_dslr_poses(self.scene_dir)

    def image_path(self, image_name: str) -> Path:
        if self.sensor == "iphone":
            return scannetpp_iphone_image_path(
                self.scene_dir, image_name, self.frame_root
            )
        else:
            return scannetpp_dslr_image_path(self.scene_dir, image_name)

    # ---- Geometry paths (used by frame_selector / referability) ----

    def mesh_path(self) -> Path:
        return self.scene_dir / "scans" / "mesh_aligned_0.05.ply"

    def seg_path(self) -> Path:
        return self.scene_dir / "scans" / "segments.json"

    def anno_path(self) -> Path:
        return self.scene_dir / "scans" / "segments_anno.json"

    # ---- Camera extras (used when data_source replaces v2 loaders) ----

    def load_axis_alignment(self) -> "np.ndarray":
        return np.eye(4, dtype=np.float64)

    def load_depth_intrinsics(self) -> "CameraIntrinsics | None":
        return None

    def depth_image_path(self, image_name: str) -> Path | None:
        return None

    def validate(self) -> dict[str, Any]:
        issues: list[str] = []

        # Intrinsics
        try:
            intr = self.load_intrinsics()
            intr_str = f"{intr.width}x{intr.height}"
        except Exception as exc:
            intr_str = "error"
            issues.append(f"intrinsics: {exc}")

        # Poses
        pose_count = 0
        try:
            poses = self.load_poses()
            pose_count = len(poses)
        except Exception as exc:
            issues.append(f"poses: {exc}")
            poses = {}

        if pose_count == 0:
            issues.append("no poses loaded")

        # Images
        missing_images = 0
        unreadable_images = 0
        import cv2

        if self.sensor == "iphone":
            # Check rgb.mkv + colmap files exist
            mkv = self.scene_dir / "iphone" / "rgb.mkv"
            if not mkv.is_file():
                issues.append("missing iphone/rgb.mkv")

            img_txt = self.scene_dir / "iphone" / "colmap" / "images.txt"
            if not img_txt.is_file():
                issues.append("missing iphone/colmap/images.txt")

            # Check extracted frames
            for name in poses:
                ip = self.image_path(name)
                if not ip.is_file():
                    missing_images += 1
                elif _read_image_unicode_safe(ip) is None:
                    unreadable_images += 1

            if missing_images:
                issues.append(
                    f"missing extracted iPhone frames: {missing_images}. "
                    f"Run scripts/extract_scannetpp_iphone_frames.py first."
                )
            if unreadable_images:
                issues.append(f"unreadable_images: {unreadable_images}")
        else:
            # DSLR: check transforms.json + resized_images
            transforms = self.scene_dir / "dslr" / "nerfstudio" / "transforms.json"
            if not transforms.is_file():
                issues.append("missing dslr/nerfstudio/transforms.json")

            for name in poses:
                ip = self.image_path(name)
                if not ip.is_file():
                    missing_images += 1
                elif _read_image_unicode_safe(ip) is None:
                    unreadable_images += 1

            if missing_images:
                issues.append(f"missing_images: {missing_images}")
            if unreadable_images:
                issues.append(f"unreadable_images: {unreadable_images}")

        return {
            "dataset": "scannetpp",
            "sensor": self.sensor,
            "scene_id": self.scene_id,
            "intrinsics": intr_str,
            "poses": pose_count,
            "missing_images": missing_images,
            "unreadable_images": unreadable_images,
            "issues": issues,
        }
