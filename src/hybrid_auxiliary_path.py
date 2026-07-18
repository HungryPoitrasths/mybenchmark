"""Question-specific geometric routing with RGB continuity verification."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import heapq
import json
import math
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import cv2
import numpy as np

from .auxiliary_path import MAX_AUXILIARY_FRAMES
from .legacy_auxiliary_path import object_group_center, route_visibility_mask, sample_route_points
from .utils.colmap_loader import CameraIntrinsics, CameraPose
from .utils.coordinate_transform import project_camera_points_to_image


HYBRID_VISUAL_CACHE_VERSION = 1
HYBRID_ALGORITHM_VERSION = 1


@dataclass(frozen=True)
class HybridRoutingConfig:
    max_processed_edge_px: int = 960
    orb_nfeatures: int = 2000
    orb_fast_threshold: int = 10
    orb_ratio_test: float = 0.75
    clahe_clip_limit: float = 2.0
    ransac_confidence: float = 0.99
    ransac_diagonal_threshold_ratio: float = 0.0015
    min_ransac_threshold_px: float = 1.0
    min_model_inliers: int = 8
    min_model_inlier_ratio: float = 0.30
    grid_size: int = 4
    min_grid_fraction: float = 0.25
    semantic_min_depth_m: float = 0.3
    semantic_max_depth_m: float = 8.0
    semantic_min_bbox_in_frame_ratio: float = 0.35
    semantic_min_projected_area_ratio: float = 800.0 / (640.0 * 480.0)
    min_transition_overlap_samples: int = 1
    max_visual_edge_checks_per_question: int = 256

    def __post_init__(self) -> None:
        if self.max_processed_edge_px <= 0:
            raise ValueError("max_processed_edge_px must be positive")
        if self.orb_nfeatures <= 0:
            raise ValueError("orb_nfeatures must be positive")
        if not 0.0 < self.orb_ratio_test < 1.0:
            raise ValueError("orb_ratio_test must be in (0, 1)")
        if self.min_model_inliers < 8:
            raise ValueError("min_model_inliers must be at least 8")
        if not 0.0 <= self.min_model_inlier_ratio <= 1.0:
            raise ValueError("min_model_inlier_ratio must be in [0, 1]")
        if self.grid_size <= 0:
            raise ValueError("grid_size must be positive")
        if not 0.0 <= self.min_grid_fraction <= 1.0:
            raise ValueError("min_grid_fraction must be in [0, 1]")
        if self.min_transition_overlap_samples <= 0:
            raise ValueError("min_transition_overlap_samples must be positive")
        if self.max_visual_edge_checks_per_question <= 0:
            raise ValueError("max_visual_edge_checks_per_question must be positive")


@dataclass(frozen=True)
class VisualContinuityEvidence:
    passed: bool
    model: str | None
    mutual_matches: int
    inliers: int
    inlier_ratio: float
    grid_fraction_left: float
    grid_fraction_right: float
    reason: str

    @property
    def min_grid_fraction(self) -> float:
        return min(self.grid_fraction_left, self.grid_fraction_right)


@dataclass(frozen=True)
class HybridAuxiliaryRoute:
    auxiliary_image_names: tuple[str, ...]
    cost: float
    edge_count: int
    route_sample_count: int
    frame_a_coverage_end: float
    frame_b_coverage_start: float
    auxiliary_responsibility_fraction: float
    transition_overlap_fraction: float
    min_mutual_matches: int
    min_inliers: int
    min_inlier_ratio: float
    min_grid_fraction: float
    visual_models: tuple[str, ...]
    semantic_rejected_frames: int


@dataclass(frozen=True)
class _CoverageRun:
    start: int
    end: int


@dataclass(frozen=True)
class _SearchState:
    last_name: str
    last_run: _CoverageRun
    frontier: int
    path: tuple[str, ...]
    runs: tuple[_CoverageRun, ...]
    evidence: tuple[VisualContinuityEvidence, ...]


def _read_image(path: Path) -> np.ndarray | None:
    try:
        payload = np.fromfile(path, dtype=np.uint8)
    except OSError:
        return None
    if payload.size == 0:
        return None
    return cv2.imdecode(payload, cv2.IMREAD_COLOR)


def _coverage_runs(mask: np.ndarray) -> list[_CoverageRun]:
    runs: list[_CoverageRun] = []
    start: int | None = None
    for index, visible in enumerate(np.asarray(mask, dtype=bool)):
        if visible and start is None:
            start = index
        elif not visible and start is not None:
            runs.append(_CoverageRun(start, index - 1))
            start = None
    if start is not None:
        runs.append(_CoverageRun(start, len(mask) - 1))
    return runs


def _overlap_samples(left: _CoverageRun, right: _CoverageRun) -> int:
    return max(0, min(left.end, right.end) - max(left.start, right.start) + 1)


def _object_projection_metrics(
    obj: Mapping[str, Any],
    pose: CameraPose,
    intrinsics: CameraIntrinsics,
) -> tuple[float, float, float]:
    bbox_min = np.asarray(obj.get("bbox_min"), dtype=np.float64)
    bbox_max = np.asarray(obj.get("bbox_max"), dtype=np.float64)
    if bbox_min.shape != (3,) or bbox_max.shape != (3,):
        return 0.0, 0.0, 0.0
    center = np.asarray(obj.get("center", 0.5 * (bbox_min + bbox_max)), dtype=np.float64)
    if center.shape != (3,):
        center = 0.5 * (bbox_min + bbox_max)

    corners = np.asarray(
        [
            [x, y, z]
            for x in (bbox_min[0], bbox_max[0])
            for y in (bbox_min[1], bbox_max[1])
            for z in (bbox_min[2], bbox_max[2])
        ],
        dtype=np.float64,
    )
    points_camera = (pose.rotation @ corners.T + pose.translation[:, None]).T
    uv, depths = project_camera_points_to_image(points_camera, intrinsics)
    valid = (depths > 0.0) & np.isfinite(uv).all(axis=1)
    if not valid.any():
        return float(pose.world_to_camera_point(center)[2]), 0.0, 0.0
    valid_uv = uv[valid]
    in_frame = (
        (valid_uv[:, 0] >= 0.0)
        & (valid_uv[:, 0] < intrinsics.width)
        & (valid_uv[:, 1] >= 0.0)
        & (valid_uv[:, 1] < intrinsics.height)
    )
    bbox_ratio = float(in_frame.sum() / len(valid_uv))
    left = float(np.clip(valid_uv[:, 0].min(), 0.0, intrinsics.width))
    right = float(np.clip(valid_uv[:, 0].max(), 0.0, intrinsics.width))
    top = float(np.clip(valid_uv[:, 1].min(), 0.0, intrinsics.height))
    bottom = float(np.clip(valid_uv[:, 1].max(), 0.0, intrinsics.height))
    area_ratio = max(0.0, right - left) * max(0.0, bottom - top) / max(
        float(intrinsics.width * intrinsics.height), 1.0
    )
    center_depth = float(pose.world_to_camera_point(center)[2])
    return center_depth, bbox_ratio, area_ratio


class HybridAuxiliaryRouter:
    """Route through the uncovered A-to-B interval and verify every RGB handoff."""

    def __init__(
        self,
        *,
        poses: dict[str, CameraPose],
        intrinsics: CameraIntrinsics,
        image_path_for: Callable[[str], Path],
        config: HybridRoutingConfig | None = None,
    ) -> None:
        self.poses = dict(poses)
        self.intrinsics = intrinsics
        self.image_path_for = image_path_for
        self.config = config or HybridRoutingConfig()
        self._gray_cache: dict[str, np.ndarray | None] = {}
        self._feature_cache: dict[str, tuple[list[cv2.KeyPoint], np.ndarray | None]] = {}
        self._edge_cache: dict[tuple[str, str], VisualContinuityEvidence] = {}
        self._visual_counts: Counter[str] = Counter()
        self._route_rejection_counts: Counter[str] = Counter()
        self._route_count = 0

    def _cache_signature(self) -> str:
        pose_rows: list[dict[str, object]] = []
        for image_name in sorted(self.poses):
            pose = self.poses[image_name]
            image_path = self.image_path_for(image_name)
            try:
                stat = image_path.stat()
                image_state: dict[str, int | bool] = {
                    "exists": True,
                    "size": int(stat.st_size),
                    "mtime_ns": int(stat.st_mtime_ns),
                }
            except OSError:
                image_state = {"exists": False}
            pose_rows.append(
                {
                    "image_name": image_name,
                    "rotation": np.asarray(pose.rotation, dtype=np.float64).tolist(),
                    "translation": np.asarray(pose.translation, dtype=np.float64).tolist(),
                    "image": image_state,
                }
            )
        payload = {
            "algorithm_version": HYBRID_ALGORITHM_VERSION,
            "config": asdict(self.config),
            "intrinsics": {
                "width": self.intrinsics.width,
                "height": self.intrinsics.height,
                "fx": self.intrinsics.fx,
                "fy": self.intrinsics.fy,
                "cx": self.intrinsics.cx,
                "cy": self.intrinsics.cy,
                "distortion_model": self.intrinsics.distortion_model,
                "distortion_params": (
                    np.asarray(self.intrinsics.distortion_params, dtype=np.float64).tolist()
                    if self.intrinsics.distortion_params is not None
                    else None
                ),
            },
            "poses": pose_rows,
        }
        encoded = json.dumps(
            payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def load_cache(self, cache_path: Path) -> bool:
        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return False
        if not isinstance(payload, dict):
            return False
        if payload.get("cache_version") != HYBRID_VISUAL_CACHE_VERSION:
            return False
        if payload.get("signature") != self._cache_signature():
            return False
        rows = payload.get("edges")
        if not isinstance(rows, list):
            return False
        loaded: dict[tuple[str, str], VisualContinuityEvidence] = {}
        try:
            for row in rows:
                if not isinstance(row, dict):
                    return False
                left = str(row["left"])
                right = str(row["right"])
                if left not in self.poses or right not in self.poses or left >= right:
                    return False
                raw = row["evidence"]
                loaded[(left, right)] = VisualContinuityEvidence(
                    passed=bool(raw["passed"]),
                    model=str(raw["model"]) if raw.get("model") is not None else None,
                    mutual_matches=int(raw["mutual_matches"]),
                    inliers=int(raw["inliers"]),
                    inlier_ratio=float(raw["inlier_ratio"]),
                    grid_fraction_left=float(raw["grid_fraction_left"]),
                    grid_fraction_right=float(raw["grid_fraction_right"]),
                    reason=str(raw["reason"]),
                )
        except (KeyError, TypeError, ValueError):
            return False
        self._edge_cache = loaded
        return True

    def save_cache(self, cache_path: Path) -> None:
        payload = {
            "cache_version": HYBRID_VISUAL_CACHE_VERSION,
            "signature": self._cache_signature(),
            "edges": [
                {
                    "left": left,
                    "right": right,
                    "evidence": asdict(evidence),
                }
                for (left, right), evidence in sorted(self._edge_cache.items())
            ],
        }
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = cache_path.with_suffix(f"{cache_path.suffix}.tmp")
        temporary_path.write_text(
            json.dumps(payload, ensure_ascii=True, sort_keys=True), encoding="utf-8"
        )
        temporary_path.replace(cache_path)

    def _processed_gray(self, image_name: str) -> np.ndarray | None:
        if image_name in self._gray_cache:
            return self._gray_cache[image_name]
        image = _read_image(self.image_path_for(image_name))
        if image is None:
            self._gray_cache[image_name] = None
            return None
        height, width = image.shape[:2]
        scale = min(1.0, self.config.max_processed_edge_px / max(width, height))
        if scale < 1.0:
            image = cv2.resize(
                image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
            )
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(
            clipLimit=self.config.clahe_clip_limit, tileGridSize=(8, 8)
        )
        result = clahe.apply(gray)
        self._gray_cache[image_name] = result
        return result

    def _features(self, image_name: str) -> tuple[list[cv2.KeyPoint], np.ndarray | None]:
        cached = self._feature_cache.get(image_name)
        if cached is not None:
            return cached
        gray = self._processed_gray(image_name)
        if gray is None:
            result: tuple[list[cv2.KeyPoint], np.ndarray | None] = ([], None)
        else:
            orb = cv2.ORB_create(
                nfeatures=self.config.orb_nfeatures,
                fastThreshold=self.config.orb_fast_threshold,
            )
            keypoints, descriptors = orb.detectAndCompute(gray, None)
            result = (list(keypoints or []), descriptors)
        self._feature_cache[image_name] = result
        return result

    def _ratio_matches(
        self, query: np.ndarray, train: np.ndarray
    ) -> dict[int, int]:
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        pairs = matcher.knnMatch(query, train, k=2)
        return {
            int(pair[0].queryIdx): int(pair[0].trainIdx)
            for pair in pairs
            if len(pair) == 2
            and pair[0].distance < self.config.orb_ratio_test * pair[1].distance
        }

    def _grid_fraction(
        self, points: np.ndarray, mask: np.ndarray, shape: tuple[int, int]
    ) -> float:
        height, width = shape
        cells = {
            (
                max(
                    0,
                    min(
                        int(point[0] * self.config.grid_size / max(width, 1)),
                        self.config.grid_size - 1,
                    ),
                ),
                max(
                    0,
                    min(
                        int(point[1] * self.config.grid_size / max(height, 1)),
                        self.config.grid_size - 1,
                    ),
                ),
            )
            for point in points[mask]
        }
        return len(cells) / float(self.config.grid_size * self.config.grid_size)

    def _undistort_points(
        self,
        points: np.ndarray,
        shape: tuple[int, int],
    ) -> np.ndarray | None:
        if not self.intrinsics.is_distorted:
            return points
        height, width = shape
        scale_x = width / max(float(self.intrinsics.width), 1.0)
        scale_y = height / max(float(self.intrinsics.height), 1.0)
        camera_matrix = np.array(
            [
                [self.intrinsics.fx * scale_x, 0.0, self.intrinsics.cx * scale_x],
                [0.0, self.intrinsics.fy * scale_y, self.intrinsics.cy * scale_y],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        distortion = np.asarray(
            self.intrinsics.distortion_params, dtype=np.float64
        ).reshape(-1)
        try:
            if self.intrinsics.distortion_model == "OPENCV":
                corrected = cv2.undistortPoints(
                    points.reshape(-1, 1, 2),
                    camera_matrix,
                    distortion,
                    P=camera_matrix,
                )
            elif self.intrinsics.distortion_model == "OPENCV_FISHEYE":
                corrected = cv2.fisheye.undistortPoints(
                    points.reshape(-1, 1, 2),
                    camera_matrix,
                    distortion.reshape(-1, 1),
                    P=camera_matrix,
                )
            else:
                return None
        except cv2.error:
            return None
        return np.asarray(corrected, dtype=np.float32).reshape(-1, 2)

    def _model_evidence(
        self,
        *,
        model: str,
        points_left: np.ndarray,
        points_right: np.ndarray,
        threshold_px: float,
        left_shape: tuple[int, int],
        right_shape: tuple[int, int],
    ) -> VisualContinuityEvidence:
        mask = None
        try:
            if model == "fundamental":
                _matrix, mask = cv2.findFundamentalMat(
                    points_left,
                    points_right,
                    cv2.FM_RANSAC,
                    threshold_px,
                    self.config.ransac_confidence,
                )
            elif model == "homography":
                _matrix, mask = cv2.findHomography(
                    points_left,
                    points_right,
                    cv2.RANSAC,
                    threshold_px,
                    None,
                    2000,
                    self.config.ransac_confidence,
                )
            else:
                raise ValueError(f"unknown visual model: {model}")
        except cv2.error:
            mask = None
        mutual_matches = len(points_left)
        if mask is None or len(mask) != mutual_matches:
            return VisualContinuityEvidence(
                False, model, mutual_matches, 0, 0.0, 0.0, 0.0, "model_estimation_failed"
            )
        inlier_mask = np.asarray(mask).reshape(-1).astype(bool)
        inliers = int(inlier_mask.sum())
        inlier_ratio = inliers / max(mutual_matches, 1)
        left_grid = self._grid_fraction(points_left, inlier_mask, left_shape)
        right_grid = self._grid_fraction(points_right, inlier_mask, right_shape)
        passed = (
            inliers >= self.config.min_model_inliers
            and inlier_ratio >= self.config.min_model_inlier_ratio
            and min(left_grid, right_grid) >= self.config.min_grid_fraction
        )
        return VisualContinuityEvidence(
            passed=passed,
            model=model,
            mutual_matches=mutual_matches,
            inliers=inliers,
            inlier_ratio=inlier_ratio,
            grid_fraction_left=left_grid,
            grid_fraction_right=right_grid,
            reason="passed" if passed else "model_thresholds_failed",
        )

    def _compute_visual_continuity(
        self, left: str, right: str
    ) -> VisualContinuityEvidence:
        left_kp, left_desc = self._features(left)
        right_kp, right_desc = self._features(right)
        if left_desc is None or right_desc is None:
            return VisualContinuityEvidence(
                False, None, 0, 0, 0.0, 0.0, 0.0, "missing_descriptors"
            )
        if len(left_kp) < self.config.min_model_inliers or len(right_kp) < self.config.min_model_inliers:
            return VisualContinuityEvidence(
                False, None, 0, 0, 0.0, 0.0, 0.0, "insufficient_keypoints"
            )
        forward = self._ratio_matches(left_desc, right_desc)
        reverse = self._ratio_matches(right_desc, left_desc)
        mutual = sorted(
            (left_index, right_index)
            for left_index, right_index in forward.items()
            if reverse.get(right_index) == left_index
        )
        if len(mutual) < self.config.min_model_inliers:
            return VisualContinuityEvidence(
                False,
                None,
                len(mutual),
                0,
                0.0,
                0.0,
                0.0,
                "insufficient_mutual_matches",
            )
        points_left = np.float32([left_kp[index].pt for index, _ in mutual])
        points_right = np.float32([right_kp[index].pt for _, index in mutual])
        gray_left = self._gray_cache[left]
        gray_right = self._gray_cache[right]
        assert gray_left is not None and gray_right is not None
        left_shape = (int(gray_left.shape[0]), int(gray_left.shape[1]))
        right_shape = (int(gray_right.shape[0]), int(gray_right.shape[1]))
        model_points_left = self._undistort_points(points_left, left_shape)
        model_points_right = self._undistort_points(points_right, right_shape)
        if model_points_left is None or model_points_right is None:
            return VisualContinuityEvidence(
                False,
                None,
                len(mutual),
                0,
                0.0,
                0.0,
                0.0,
                "point_undistortion_failed",
            )
        diagonal = max(math.hypot(*left_shape), math.hypot(*right_shape))
        threshold_px = max(
            self.config.min_ransac_threshold_px,
            self.config.ransac_diagonal_threshold_ratio * diagonal,
        )
        candidates = [
            self._model_evidence(
                model=model,
                points_left=model_points_left,
                points_right=model_points_right,
                threshold_px=threshold_px,
                left_shape=left_shape,
                right_shape=right_shape,
            )
            for model in ("fundamental", "homography")
        ]
        candidates.sort(
            key=lambda item: (
                item.passed,
                item.inlier_ratio,
                item.min_grid_fraction,
                item.inliers,
                item.model == "fundamental",
            ),
            reverse=True,
        )
        best = candidates[0]
        if best.passed:
            return best
        return VisualContinuityEvidence(
            passed=False,
            model=best.model,
            mutual_matches=best.mutual_matches,
            inliers=best.inliers,
            inlier_ratio=best.inlier_ratio,
            grid_fraction_left=best.grid_fraction_left,
            grid_fraction_right=best.grid_fraction_right,
            reason="no_ransac_model_passed",
        )

    def visual_continuity(self, left: str, right: str) -> VisualContinuityEvidence:
        if left == right:
            return VisualContinuityEvidence(
                True, "identity", 0, 0, 1.0, 1.0, 1.0, "same_frame"
            )
        key = tuple(sorted((left, right)))
        cached = self._edge_cache.get(key)
        if cached is not None:
            self._visual_counts["cache_hit"] += 1
            return cached
        if left not in self.poses or right not in self.poses:
            evidence = VisualContinuityEvidence(
                False, None, 0, 0, 0.0, 0.0, 0.0, "missing_pose"
            )
        else:
            evidence = self._compute_visual_continuity(key[0], key[1])
        self._edge_cache[key] = evidence
        self._visual_counts["computed"] += 1
        self._visual_counts["passed" if evidence.passed else "rejected"] += 1
        self._visual_counts[f"reason:{evidence.reason}"] += 1
        return evidence

    def _meaningfully_projected(
        self, objects: Iterable[Mapping[str, Any]], pose: CameraPose
    ) -> bool:
        for obj in objects:
            depth, bbox_ratio, area_ratio = _object_projection_metrics(
                obj, pose, self.intrinsics
            )
            if (
                self.config.semantic_min_depth_m < depth <= self.config.semantic_max_depth_m
                and bbox_ratio >= self.config.semantic_min_bbox_in_frame_ratio
                and area_ratio >= self.config.semantic_min_projected_area_ratio
            ):
                return True
        return False

    def _semantic_conflict(
        self,
        pose: CameraPose,
        group_a_objects: Iterable[Mapping[str, Any]],
        group_b_objects: Iterable[Mapping[str, Any]],
    ) -> bool:
        return self._meaningfully_projected(
            group_a_objects, pose
        ) and self._meaningfully_projected(group_b_objects, pose)

    def _route_result(
        self,
        *,
        path: tuple[str, ...],
        runs: tuple[_CoverageRun, ...],
        evidence: tuple[VisualContinuityEvidence, ...],
        sample_count: int,
        frame_a_end: int,
        frame_b_start: int,
        semantic_rejected_frames: int,
    ) -> HybridAuxiliaryRoute:
        denomin = max(sample_count - 1, 1)
        overlaps = sum(
            _overlap_samples(left, right) for left, right in zip(runs, runs[1:])
        )
        edge_count = len(evidence)
        mean_visual_penalty = sum(1.0 - item.inlier_ratio for item in evidence) / max(
            edge_count, 1
        )
        mean_grid_penalty = sum(1.0 - item.min_grid_fraction for item in evidence) / max(
            edge_count, 1
        )
        cost = edge_count + 0.1 * mean_visual_penalty + 0.1 * mean_grid_penalty
        return HybridAuxiliaryRoute(
            auxiliary_image_names=path,
            cost=cost,
            edge_count=edge_count,
            route_sample_count=sample_count,
            frame_a_coverage_end=frame_a_end / denomin,
            frame_b_coverage_start=frame_b_start / denomin,
            auxiliary_responsibility_fraction=max(
                0.0, (frame_b_start - frame_a_end) / denomin
            ),
            transition_overlap_fraction=overlaps / denomin,
            min_mutual_matches=min(item.mutual_matches for item in evidence),
            min_inliers=min(item.inliers for item in evidence),
            min_inlier_ratio=min(item.inlier_ratio for item in evidence),
            min_grid_fraction=min(item.min_grid_fraction for item in evidence),
            visual_models=tuple(str(item.model) for item in evidence),
            semantic_rejected_frames=semantic_rejected_frames,
        )

    def find_route(
        self,
        *,
        frame_a_name: str,
        frame_b_name: str,
        group_a_objects: list[Mapping[str, Any]],
        group_b_objects: list[Mapping[str, Any]],
        max_auxiliary_frames: int = MAX_AUXILIARY_FRAMES,
    ) -> HybridAuxiliaryRoute | None:
        self._route_count += 1
        if max_auxiliary_frames < 0:
            raise ValueError("max_auxiliary_frames must be non-negative")
        if (
            frame_a_name == frame_b_name
            or frame_a_name not in self.poses
            or frame_b_name not in self.poses
            or not group_a_objects
            or not group_b_objects
        ):
            self._route_rejection_counts["invalid_input"] += 1
            return None

        center_a = object_group_center(group_a_objects)
        center_b = object_group_center(group_b_objects)
        _ts, route_points = sample_route_points(center_a, center_b)
        sample_count = len(route_points)
        runs_by_frame: dict[str, list[_CoverageRun]] = {}
        for image_name, pose in self.poses.items():
            mask = route_visibility_mask(route_points, pose, self.intrinsics)
            runs = _coverage_runs(mask)
            if runs:
                runs_by_frame[image_name] = runs

        frame_a_runs = runs_by_frame.get(frame_a_name, [])
        frame_b_runs = runs_by_frame.get(frame_b_name, [])
        if not frame_a_runs or frame_a_runs[0].start != 0:
            self._route_rejection_counts["frame_a_missing_route_start"] += 1
            return None
        if not frame_b_runs or frame_b_runs[-1].end != sample_count - 1:
            self._route_rejection_counts["frame_b_missing_route_end"] += 1
            return None
        start_run = frame_a_runs[0]
        end_run = frame_b_runs[-1]
        frame_a_end = start_run.end
        frame_b_start = end_run.start

        visual_checks = 0
        if frame_a_end >= frame_b_start:
            visual_checks += 1
            direct = self.visual_continuity(frame_a_name, frame_b_name)
            if not direct.passed:
                self._route_rejection_counts["direct_rgb_discontinuity"] += 1
                return None
            return self._route_result(
                path=(),
                runs=(start_run, end_run),
                evidence=(direct,),
                sample_count=sample_count,
                frame_a_end=frame_a_end,
                frame_b_start=frame_b_start,
                semantic_rejected_frames=0,
            )
        if max_auxiliary_frames == 0:
            self._route_rejection_counts["auxiliary_limit"] += 1
            return None

        semantic_cache: dict[str, bool] = {}
        semantic_rejected_names: set[str] = set()
        candidate_runs: dict[str, list[_CoverageRun]] = {}
        for image_name, runs in runs_by_frame.items():
            if image_name in {frame_a_name, frame_b_name}:
                continue
            useful = [
                run
                for run in runs
                if run.end > frame_a_end
                and run.start <= frame_b_start
                and min(run.end, frame_b_start) > frame_a_end
            ]
            if useful:
                candidate_runs[image_name] = useful

        queue: list[tuple[int, float, int, int, _SearchState]] = []
        serial = 0
        initial = _SearchState(
            last_name=frame_a_name,
            last_run=start_run,
            frontier=frame_a_end,
            path=(),
            runs=(start_run,),
            evidence=(),
        )
        heapq.heappush(queue, (0, 0.0, -frame_a_end, serial, initial))
        best: dict[tuple[str, int, int, int], float] = {
            (frame_a_name, start_run.start, start_run.end, 0): 0.0
        }

        while queue and visual_checks < self.config.max_visual_edge_checks_per_question:
            _hops, visual_penalty, _negative_frontier, _serial, state = heapq.heappop(queue)
            if (
                state.frontier >= frame_b_start
                and _overlap_samples(state.last_run, end_run)
                >= self.config.min_transition_overlap_samples
            ):
                visual_checks += 1
                finish = self.visual_continuity(state.last_name, frame_b_name)
                if finish.passed:
                    return self._route_result(
                        path=state.path,
                        runs=state.runs + (end_run,),
                        evidence=state.evidence + (finish,),
                        sample_count=sample_count,
                        frame_a_end=frame_a_end,
                        frame_b_start=frame_b_start,
                        semantic_rejected_frames=len(semantic_rejected_names),
                    )
            if len(state.path) >= max_auxiliary_frames:
                continue

            feasible: list[tuple[int, int, str, _CoverageRun]] = []
            for image_name, runs in candidate_runs.items():
                if image_name in state.path:
                    continue
                for run in runs:
                    overlap = _overlap_samples(state.last_run, run)
                    if overlap < self.config.min_transition_overlap_samples:
                        continue
                    next_frontier = min(run.end, frame_b_start)
                    if next_frontier <= state.frontier:
                        continue
                    feasible.append((-next_frontier, -overlap, image_name, run))
            feasible.sort(key=lambda row: (row[0], row[1], row[2], row[3].start))

            for negative_frontier, _negative_overlap, image_name, run in feasible:
                if visual_checks >= self.config.max_visual_edge_checks_per_question:
                    break
                conflict = semantic_cache.get(image_name)
                if conflict is None:
                    conflict = self._semantic_conflict(
                        self.poses[image_name], group_a_objects, group_b_objects
                    )
                    semantic_cache[image_name] = conflict
                if conflict:
                    semantic_rejected_names.add(image_name)
                    continue
                visual_checks += 1
                edge = self.visual_continuity(state.last_name, image_name)
                if not edge.passed:
                    continue
                next_frontier = -negative_frontier
                next_penalty = visual_penalty + (1.0 - edge.inlier_ratio) + (
                    1.0 - edge.min_grid_fraction
                )
                next_path = state.path + (image_name,)
                key = (image_name, run.start, run.end, len(next_path))
                if next_penalty >= best.get(key, float("inf")):
                    continue
                best[key] = next_penalty
                serial += 1
                next_state = _SearchState(
                    last_name=image_name,
                    last_run=run,
                    frontier=next_frontier,
                    path=next_path,
                    runs=state.runs + (run,),
                    evidence=state.evidence + (edge,),
                )
                heapq.heappush(
                    queue,
                    (len(next_path), next_penalty, -next_frontier, serial, next_state),
                )

        reason = (
            "visual_check_budget"
            if visual_checks >= self.config.max_visual_edge_checks_per_question
            else "no_hybrid_path"
        )
        self._route_rejection_counts[reason] += 1
        if semantic_rejected_names:
            self._route_rejection_counts["semantic_conflict_seen"] += 1
        return None

    def diagnostics(self) -> dict[str, object]:
        return {
            "pose_count": len(self.poses),
            "route_count": self._route_count,
            "visual_edge_cache_count": len(self._edge_cache),
            "feature_frame_count": len(self._feature_cache),
            "visual_counts": dict(sorted(self._visual_counts.items())),
            "route_rejection_counts": dict(sorted(self._route_rejection_counts.items())),
        }
