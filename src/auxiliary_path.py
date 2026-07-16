"""Visual-pose routing between two flash-reviewed reasoning frames."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import heapq
import json
import math
from pathlib import Path
import re
from typing import Callable

import cv2
import numpy as np

from .utils.colmap_loader import CameraPose
from .utils.coordinate_transform import get_camera_forward


TEMPORAL_TRANSLATION_MAX_M = 0.75
TEMPORAL_ANGLE_MAX_DEG = 45.0
NEAR_POSE_TRANSLATION_MAX_M = 1.0
NEAR_POSE_ANGLE_MAX_DEG = 60.0
TEMPORAL_LOOKAHEAD = 20
NEAR_POSE_NEIGHBORS = 12
ORB_RATIO_TEST = 0.75
MIN_RANSAC_INLIERS = 24
MIN_RANSAC_INLIER_RATIO = 0.25
MIN_INLIER_GRID_CELLS = 4
MAX_AUXILIARY_FRAMES = 6
VISUAL_POSE_GRAPH_CACHE_VERSION = 1
VISUAL_POSE_GRAPH_ALGORITHM_VERSION = 1


def _frame_sort_key(image_name: str) -> tuple[int, str]:
    match = re.search(r"(\d+)(?!.*\d)", Path(image_name).stem)
    return (int(match.group(1)) if match else 2**31 - 1, image_name)


def _read_image(path: Path) -> np.ndarray | None:
    try:
        payload = np.fromfile(path, dtype=np.uint8)
    except OSError:
        return None
    if payload.size == 0:
        return None
    return cv2.imdecode(payload, cv2.IMREAD_COLOR)


def _pose_angle_deg(a: CameraPose, b: CameraPose) -> float:
    forward_a = np.asarray(get_camera_forward(a), dtype=np.float64)
    forward_b = np.asarray(get_camera_forward(b), dtype=np.float64)
    norm = float(np.linalg.norm(forward_a) * np.linalg.norm(forward_b))
    if norm <= 1e-9:
        return 180.0
    cosine = float(np.clip(np.dot(forward_a, forward_b) / norm, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


@dataclass(frozen=True)
class VisualPoseEdge:
    target: str
    translation_m: float
    rotation_deg: float
    inliers: int
    inlier_ratio: float
    cost: float


@dataclass(frozen=True)
class AuxiliaryRoute:
    auxiliary_image_names: tuple[str, ...]
    cost: float
    edge_count: int
    min_inliers: int
    min_inlier_ratio: float


class VisualPoseGraph:
    """A per-scene graph whose every edge has pose and RGB-overlap evidence."""

    def __init__(
        self,
        *,
        poses: dict[str, CameraPose],
        image_path_for: Callable[[str], Path],
        flash_frame_names: set[str] | None = None,
    ) -> None:
        self.poses = dict(poses)
        self.image_path_for = image_path_for
        self.flash_frame_names = set(flash_frame_names or set())
        self.edges: dict[str, list[VisualPoseEdge]] = {}
        self.node_metrics: dict[str, dict[str, float | bool]] = {}
        self.rejected_edge_counts: dict[str, int] = {}
        self._gray_cache: dict[str, np.ndarray] = {}
        self._feature_cache: dict[str, tuple[list[cv2.KeyPoint], np.ndarray | None]] = {}

    def _cache_signature(self) -> str:
        pose_rows: list[dict[str, object]] = []
        for image_name in sorted(self.poses, key=_frame_sort_key):
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
            pose_rows.append({
                "image_name": image_name,
                "rotation": np.asarray(pose.rotation, dtype=np.float64).tolist(),
                "translation": np.asarray(pose.translation, dtype=np.float64).tolist(),
                "image": image_state,
            })
        signature_payload = {
            "algorithm_version": VISUAL_POSE_GRAPH_ALGORITHM_VERSION,
            "parameters": {
                "temporal_translation_max_m": TEMPORAL_TRANSLATION_MAX_M,
                "temporal_angle_max_deg": TEMPORAL_ANGLE_MAX_DEG,
                "near_pose_translation_max_m": NEAR_POSE_TRANSLATION_MAX_M,
                "near_pose_angle_max_deg": NEAR_POSE_ANGLE_MAX_DEG,
                "temporal_lookahead": TEMPORAL_LOOKAHEAD,
                "near_pose_neighbors": NEAR_POSE_NEIGHBORS,
                "orb_ratio_test": ORB_RATIO_TEST,
                "min_ransac_inliers": MIN_RANSAC_INLIERS,
                "min_ransac_inlier_ratio": MIN_RANSAC_INLIER_RATIO,
                "min_inlier_grid_cells": MIN_INLIER_GRID_CELLS,
            },
            "flash_frame_names": sorted(self.flash_frame_names, key=_frame_sort_key),
            "poses": pose_rows,
        }
        encoded = json.dumps(
            signature_payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def load_cache(self, cache_path: Path) -> bool:
        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return False
        if not isinstance(payload, dict):
            return False
        try:
            cache_version = int(payload.get("cache_version", -1))
        except (TypeError, ValueError):
            return False
        if cache_version != VISUAL_POSE_GRAPH_CACHE_VERSION:
            return False
        if payload.get("signature") != self._cache_signature():
            return False
        raw_edges = payload.get("edges")
        if not isinstance(raw_edges, dict):
            return False
        loaded_edges: dict[str, list[VisualPoseEdge]] = {}
        try:
            for source, raw_edge_list in raw_edges.items():
                if source not in self.poses or not isinstance(raw_edge_list, list):
                    return False
                edge_list: list[VisualPoseEdge] = []
                for raw_edge in raw_edge_list:
                    target = str(raw_edge["target"])
                    if target not in self.poses:
                        return False
                    edge_list.append(VisualPoseEdge(
                        target=target,
                        translation_m=float(raw_edge["translation_m"]),
                        rotation_deg=float(raw_edge["rotation_deg"]),
                        inliers=int(raw_edge["inliers"]),
                        inlier_ratio=float(raw_edge["inlier_ratio"]),
                        cost=float(raw_edge["cost"]),
                    ))
                loaded_edges[str(source)] = edge_list
        except (KeyError, TypeError, ValueError):
            return False
        raw_node_metrics = payload.get("node_metrics", {})
        raw_rejected_counts = payload.get("rejected_edge_counts", {})
        if not isinstance(raw_node_metrics, dict) or not isinstance(raw_rejected_counts, dict):
            return False
        self.edges = loaded_edges
        self.node_metrics = {
            str(image_name): dict(metrics)
            for image_name, metrics in raw_node_metrics.items()
            if isinstance(metrics, dict)
        }
        self.rejected_edge_counts = {
            str(reason): int(count)
            for reason, count in raw_rejected_counts.items()
        }
        return True

    def save_cache(self, cache_path: Path) -> None:
        payload = {
            "cache_version": VISUAL_POSE_GRAPH_CACHE_VERSION,
            "signature": self._cache_signature(),
            "edges": {
                source: [
                    {
                        "target": edge.target,
                        "translation_m": edge.translation_m,
                        "rotation_deg": edge.rotation_deg,
                        "inliers": edge.inliers,
                        "inlier_ratio": edge.inlier_ratio,
                        "cost": edge.cost,
                    }
                    for edge in edge_list
                ]
                for source, edge_list in self.edges.items()
            },
            "node_metrics": self.node_metrics,
            "rejected_edge_counts": self.rejected_edge_counts,
        }
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = cache_path.with_suffix(f"{cache_path.suffix}.tmp")
        temporary_path.write_text(
            json.dumps(payload, ensure_ascii=True, sort_keys=True),
            encoding="utf-8",
        )
        temporary_path.replace(cache_path)

    def _load_gray_and_quality(self, image_name: str) -> tuple[np.ndarray, float, float] | None:
        cached = self._gray_cache.get(image_name)
        if cached is not None:
            metrics = self.node_metrics[image_name]
            return cached, float(metrics["laplacian"]), float(metrics["tenengrad"])
        image = _read_image(self.image_path_for(image_name))
        if image is None:
            return None
        height, width = image.shape[:2]
        scale = min(1.0, 960.0 / max(width, height))
        if scale < 1.0:
            image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        denoised = cv2.GaussianBlur(gray, (3, 3), 0)
        laplacian = float(cv2.Laplacian(denoised, cv2.CV_64F).var())
        grad_x = cv2.Sobel(denoised, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(denoised, cv2.CV_32F, 0, 1, ksize=3)
        tenengrad = float(cv2.magnitude(grad_x, grad_y).mean())
        self._gray_cache[image_name] = gray
        self.node_metrics[image_name] = {
            "readable": True,
            "laplacian": laplacian,
            "tenengrad": tenengrad,
        }
        return gray, laplacian, tenengrad

    def _features(self, image_name: str) -> tuple[list[cv2.KeyPoint], np.ndarray | None]:
        cached = self._feature_cache.get(image_name)
        if cached is not None:
            return cached
        loaded = self._load_gray_and_quality(image_name)
        if loaded is None:
            result: tuple[list[cv2.KeyPoint], np.ndarray | None] = ([], None)
        else:
            orb = cv2.ORB_create(nfeatures=2000, fastThreshold=10)
            keypoints, descriptors = orb.detectAndCompute(loaded[0], None)
            result = (list(keypoints or []), descriptors)
        self._feature_cache[image_name] = result
        return result

    def _visual_overlap(self, left: str, right: str) -> tuple[int, float, int] | None:
        left_kp, left_desc = self._features(left)
        right_kp, right_desc = self._features(right)
        if left_desc is None or right_desc is None or len(left_kp) < 8 or len(right_kp) < 8:
            return None
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        pairs = matcher.knnMatch(left_desc, right_desc, k=2)
        good = [pair[0] for pair in pairs if len(pair) == 2 and pair[0].distance < ORB_RATIO_TEST * pair[1].distance]
        if len(good) < 8:
            return None
        points_left = np.float32([left_kp[m.queryIdx].pt for m in good])
        points_right = np.float32([right_kp[m.trainIdx].pt for m in good])
        _matrix, mask = cv2.findFundamentalMat(
            points_left,
            points_right,
            cv2.FM_RANSAC,
            1.5,
            0.99,
        )
        if mask is None:
            return None
        inlier_mask = mask.ravel().astype(bool)
        inliers = int(inlier_mask.sum())
        ratio = inliers / max(len(good), 1)
        if inliers <= 0:
            return inliers, ratio, 0
        gray = self._gray_cache[left]
        height, width = gray.shape[:2]
        cells = {
            (
                min(int(point[0] * 4 / max(width, 1)), 3),
                min(int(point[1] * 4 / max(height, 1)), 3),
            )
            for point in points_left[inlier_mask]
        }
        return inliers, ratio, len(cells)

    def _try_add_edge(
        self,
        left: str,
        right: str,
        *,
        max_translation: float,
        max_rotation: float,
    ) -> None:
        left_pose = self.poses[left]
        right_pose = self.poses[right]
        translation = float(np.linalg.norm(np.asarray(left_pose.position) - np.asarray(right_pose.position)))
        if translation > max_translation:
            self.rejected_edge_counts["translation"] = self.rejected_edge_counts.get("translation", 0) + 1
            return
        rotation = _pose_angle_deg(left_pose, right_pose)
        if rotation > max_rotation:
            self.rejected_edge_counts["rotation"] = self.rejected_edge_counts.get("rotation", 0) + 1
            return
        overlap = self._visual_overlap(left, right)
        if overlap is None:
            self.rejected_edge_counts["feature_match"] = self.rejected_edge_counts.get("feature_match", 0) + 1
            return
        inliers, ratio, cells = overlap
        if inliers < MIN_RANSAC_INLIERS or ratio < MIN_RANSAC_INLIER_RATIO or cells < MIN_INLIER_GRID_CELLS:
            self.rejected_edge_counts["visual_overlap"] = self.rejected_edge_counts.get("visual_overlap", 0) + 1
            return
        cost = 1.0 + translation + (rotation / 60.0) + (MIN_RANSAC_INLIERS / max(inliers, 1))
        edge_lr = VisualPoseEdge(right, translation, rotation, inliers, ratio, cost)
        edge_rl = VisualPoseEdge(left, translation, rotation, inliers, ratio, cost)
        self.edges.setdefault(left, []).append(edge_lr)
        self.edges.setdefault(right, []).append(edge_rl)

    def build(self) -> None:
        readable: list[tuple[str, float, float]] = []
        for image_name in sorted(self.poses, key=_frame_sort_key):
            loaded = self._load_gray_and_quality(image_name)
            if loaded is not None:
                readable.append((image_name, loaded[1], loaded[2]))
        if not readable:
            return
        quality_scores = sorted((lap + 4.0 * ten for _, lap, ten in readable))
        cutoff = quality_scores[max(0, int(math.floor(0.30 * len(quality_scores))) - 1)]
        eligible = {
            name for name, lap, ten in readable
            if lap + 4.0 * ten >= cutoff or name in self.flash_frame_names
        }
        ordered = [name for name in sorted(eligible, key=_frame_sort_key) if name in self.poses]
        pair_modes: dict[tuple[str, str], tuple[float, float]] = {}
        for index, left in enumerate(ordered):
            for right in ordered[index + 1:index + 1 + TEMPORAL_LOOKAHEAD]:
                pair_modes[(left, right)] = (TEMPORAL_TRANSLATION_MAX_M, TEMPORAL_ANGLE_MAX_DEG)

        positions = {name: np.asarray(self.poses[name].position, dtype=np.float64) for name in ordered}
        for left in ordered:
            neighbors = sorted(
                (
                    (float(np.linalg.norm(positions[left] - positions[right])), right)
                    for right in ordered if right != left
                ),
                key=lambda item: item[0],
            )[:NEAR_POSE_NEIGHBORS]
            for distance, right in neighbors:
                if distance > NEAR_POSE_TRANSLATION_MAX_M:
                    continue
                pair = tuple(sorted((left, right), key=_frame_sort_key))
                pair_modes[pair] = (NEAR_POSE_TRANSLATION_MAX_M, NEAR_POSE_ANGLE_MAX_DEG)

        for (left, right), thresholds in sorted(pair_modes.items()):
            self._try_add_edge(
                left,
                right,
                max_translation=thresholds[0],
                max_rotation=thresholds[1],
            )
        for edge_list in self.edges.values():
            edge_list.sort(key=lambda edge: (edge.cost, _frame_sort_key(edge.target)))

    def find_route(
        self,
        start: str,
        end: str,
        *,
        max_auxiliary_frames: int = MAX_AUXILIARY_FRAMES,
    ) -> AuxiliaryRoute | None:
        if start == end or start not in self.poses or end not in self.poses:
            return None
        max_edges = max(1, int(max_auxiliary_frames) + 1)
        queue: list[tuple[float, int, str, tuple[str, ...], int, float]] = [
            (0.0, 0, start, (start,), 2**31 - 1, 1.0)
        ]
        best: dict[tuple[str, int], float] = {(start, 0): 0.0}
        while queue:
            cost, hops, current, path, min_inliers, min_ratio = heapq.heappop(queue)
            if current == end:
                auxiliary = path[1:-1]
                return AuxiliaryRoute(auxiliary, cost, hops, min_inliers, min_ratio)
            if hops >= max_edges:
                continue
            for edge in self.edges.get(current, []):
                if edge.target in path:
                    continue
                next_hops = hops + 1
                next_cost = cost + edge.cost
                state = (edge.target, next_hops)
                if next_cost >= best.get(state, float("inf")):
                    continue
                best[state] = next_cost
                heapq.heappush(
                    queue,
                    (
                        next_cost,
                        next_hops,
                        edge.target,
                        path + (edge.target,),
                        min(min_inliers, edge.inliers),
                        min(min_ratio, edge.inlier_ratio),
                    ),
                )
        return None

    def diagnostics(self) -> dict[str, object]:
        undirected_edges = sum(len(values) for values in self.edges.values()) // 2
        return {
            "pose_count": len(self.poses),
            "readable_count": sum(bool(row.get("readable")) for row in self.node_metrics.values()),
            "graph_node_count": len(self.edges),
            "graph_edge_count": undirected_edges,
            "rejected_edge_counts": dict(sorted(self.rejected_edge_counts.items())),
        }
