from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from src.hybrid_auxiliary_path import (
    HybridAuxiliaryRouter,
    VisualContinuityEvidence,
)
from src.utils.colmap_loader import CameraIntrinsics, CameraPose


def make_intrinsics(width: int = 320, height: int = 240) -> CameraIntrinsics:
    return CameraIntrinsics(
        width=width,
        height=height,
        fx=200.0 * width / 320.0,
        fy=200.0 * height / 240.0,
        cx=width / 2.0,
        cy=height / 2.0,
    )


def make_pose(name: str, x: float = 0.0) -> CameraPose:
    return CameraPose(
        image_name=name,
        rotation=np.eye(3, dtype=np.float64),
        translation=np.array([-x, 0.0, 0.0], dtype=np.float64),
    )


def make_object(obj_id: int, x: float, z: float = 2.0) -> dict:
    return {
        "id": obj_id,
        "label": f"object-{obj_id}",
        "center": [x, 0.0, z],
        "bbox_min": [x - 0.2, -0.2, z - 0.2],
        "bbox_max": [x + 0.2, 0.2, z + 0.2],
    }


def passing_evidence(model: str = "fundamental") -> VisualContinuityEvidence:
    return VisualContinuityEvidence(
        passed=True,
        model=model,
        mutual_matches=30,
        inliers=20,
        inlier_ratio=2.0 / 3.0,
        grid_fraction_left=0.5,
        grid_fraction_right=0.5,
        reason="passed",
    )


def test_route_advances_only_gap_and_rejects_semantic_conflict(monkeypatch) -> None:
    names = ["main_a.jpg", "repeat.jpg", "bridge.jpg", "conflict.jpg", "main_b.jpg"]
    poses = {name: make_pose(name) for name in names}
    masks = {
        "main_a.jpg": np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0], dtype=bool),
        "repeat.jpg": np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0], dtype=bool),
        "bridge.jpg": np.array([0, 0, 1, 1, 1, 1, 1, 1, 0, 0], dtype=bool),
        "conflict.jpg": np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 0], dtype=bool),
        "main_b.jpg": np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1], dtype=bool),
    }
    monkeypatch.setattr(
        "src.hybrid_auxiliary_path.route_visibility_mask",
        lambda _points, pose, _intrinsics: masks[pose.image_name],
    )
    router = HybridAuxiliaryRouter(
        poses=poses,
        intrinsics=make_intrinsics(),
        image_path_for=lambda name: Path(name),
    )
    monkeypatch.setattr(router, "visual_continuity", lambda _left, _right: passing_evidence())
    monkeypatch.setattr(
        router,
        "_semantic_conflict",
        lambda pose, _group_a, _group_b: pose.image_name == "conflict.jpg",
    )

    route = router.find_route(
        frame_a_name="main_a.jpg",
        frame_b_name="main_b.jpg",
        group_a_objects=[make_object(1, 0.0)],
        group_b_objects=[make_object(2, 0.9)],
    )

    assert route is not None
    assert route.auxiliary_image_names == ("bridge.jpg",)
    assert "repeat.jpg" not in route.auxiliary_image_names
    assert route.frame_a_coverage_end == pytest.approx(1.0 / 3.0)
    assert route.frame_b_coverage_start == pytest.approx(2.0 / 3.0)
    assert route.auxiliary_responsibility_fraction == pytest.approx(1.0 / 3.0)
    assert route.semantic_rejected_frames == 1


def test_direct_geometric_connection_still_requires_rgb(monkeypatch) -> None:
    names = ["main_a.jpg", "unused.jpg", "main_b.jpg"]
    poses = {name: make_pose(name) for name in names}
    masks = {
        "main_a.jpg": np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 0], dtype=bool),
        "unused.jpg": np.ones(10, dtype=bool),
        "main_b.jpg": np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1], dtype=bool),
    }
    monkeypatch.setattr(
        "src.hybrid_auxiliary_path.route_visibility_mask",
        lambda _points, pose, _intrinsics: masks[pose.image_name],
    )
    router = HybridAuxiliaryRouter(
        poses=poses,
        intrinsics=make_intrinsics(),
        image_path_for=lambda name: Path(name),
    )
    failed = VisualContinuityEvidence(False, None, 3, 0, 0.0, 0.0, 0.0, "test")
    monkeypatch.setattr(router, "visual_continuity", lambda _left, _right: failed)

    route = router.find_route(
        frame_a_name="main_a.jpg",
        frame_b_name="main_b.jpg",
        group_a_objects=[make_object(1, 0.0)],
        group_b_objects=[make_object(2, 0.9)],
    )

    assert route is None
    assert router.diagnostics()["route_rejection_counts"] == {"direct_rgb_discontinuity": 1}


def test_semantic_conflict_requires_meaningful_projection_from_both_groups() -> None:
    pose = make_pose("frame.jpg")
    router = HybridAuxiliaryRouter(
        poses={pose.image_name: pose},
        intrinsics=make_intrinsics(),
        image_path_for=lambda name: Path(name),
    )

    assert router._semantic_conflict(
        pose,
        [make_object(1, -0.3)],
        [make_object(2, 0.3)],
    )
    assert not router._semantic_conflict(
        pose,
        [make_object(1, -0.3)],
        [make_object(2, 100.0)],
    )


def test_semantic_projection_gate_scales_with_resolution() -> None:
    pose = make_pose("frame.jpg")
    group_a = [make_object(1, -0.3)]
    group_b = [make_object(2, 0.3)]
    decisions = []
    for width, height in ((320, 240), (960, 720)):
        router = HybridAuxiliaryRouter(
            poses={pose.image_name: pose},
            intrinsics=make_intrinsics(width, height),
            image_path_for=lambda name: Path(name),
        )
        decisions.append(router._semantic_conflict(pose, group_a, group_b))

    assert decisions == [True, True]


def test_visual_points_are_undistorted_with_processed_intrinsics() -> None:
    pose = make_pose("frame.jpg")
    intrinsics = make_intrinsics()
    intrinsics.distortion_model = "OPENCV"
    intrinsics.distortion_params = np.zeros(4, dtype=np.float64)
    router = HybridAuxiliaryRouter(
        poses={pose.image_name: pose},
        intrinsics=intrinsics,
        image_path_for=lambda name: Path(name),
    )
    points = np.float32([[20.0, 30.0], [200.0, 150.0]])

    corrected = router._undistort_points(points, (120, 160))

    assert corrected is not None
    assert corrected == pytest.approx(points)


def _keypoints() -> list[cv2.KeyPoint]:
    return [
        cv2.KeyPoint(float(x), float(y), 8.0)
        for x, y in (
            (20, 20),
            (100, 20),
            (180, 20),
            (260, 20),
            (20, 100),
            (100, 100),
            (180, 100),
            (260, 100),
        )
    ]


def test_visual_verifier_accepts_homography_when_fundamental_fails(monkeypatch) -> None:
    names = ["left.jpg", "right.jpg"]
    router = HybridAuxiliaryRouter(
        poses={name: make_pose(name) for name in names},
        intrinsics=make_intrinsics(),
        image_path_for=lambda name: Path(name),
    )
    descriptors = np.zeros((8, 32), dtype=np.uint8)
    router._gray_cache = {name: np.zeros((240, 320), dtype=np.uint8) for name in names}
    router._feature_cache = {name: (_keypoints(), descriptors) for name in names}
    calls = iter([{index: index for index in range(8)}, {index: index for index in range(8)}])
    monkeypatch.setattr(router, "_ratio_matches", lambda _query, _train: next(calls))
    monkeypatch.setattr(cv2, "findFundamentalMat", lambda *_args, **_kwargs: (None, None))
    monkeypatch.setattr(
        cv2,
        "findHomography",
        lambda *_args, **_kwargs: (np.eye(3), np.ones((8, 1), dtype=np.uint8)),
    )

    evidence = router.visual_continuity(*names)

    assert evidence.passed
    assert evidence.model == "homography"
    assert evidence.mutual_matches == 8
    assert evidence.grid_fraction_left >= 0.25


def test_visual_verifier_requires_reciprocal_matches(monkeypatch) -> None:
    names = ["left.jpg", "right.jpg"]
    router = HybridAuxiliaryRouter(
        poses={name: make_pose(name) for name in names},
        intrinsics=make_intrinsics(),
        image_path_for=lambda name: Path(name),
    )
    descriptors = np.zeros((8, 32), dtype=np.uint8)
    router._gray_cache = {name: np.zeros((240, 320), dtype=np.uint8) for name in names}
    router._feature_cache = {name: (_keypoints(), descriptors) for name in names}
    calls = iter(
        [
            {index: index for index in range(8)},
            {index: index for index in range(7)},
        ]
    )
    monkeypatch.setattr(router, "_ratio_matches", lambda _query, _train: next(calls))

    evidence = router.visual_continuity(*names)

    assert not evidence.passed
    assert evidence.mutual_matches == 7
    assert evidence.reason == "insufficient_mutual_matches"


def test_visual_verifier_rejects_spatially_concentrated_inliers(monkeypatch) -> None:
    names = ["left.jpg", "right.jpg"]
    router = HybridAuxiliaryRouter(
        poses={name: make_pose(name) for name in names},
        intrinsics=make_intrinsics(),
        image_path_for=lambda name: Path(name),
    )
    keypoints = [cv2.KeyPoint(float(10 + index), float(10 + index), 8.0) for index in range(8)]
    descriptors = np.zeros((8, 32), dtype=np.uint8)
    router._gray_cache = {name: np.zeros((240, 320), dtype=np.uint8) for name in names}
    router._feature_cache = {name: (keypoints, descriptors) for name in names}
    calls = iter([{index: index for index in range(8)}, {index: index for index in range(8)}])
    monkeypatch.setattr(router, "_ratio_matches", lambda _query, _train: next(calls))
    all_inliers = lambda *_args, **_kwargs: (
        np.eye(3),
        np.ones((8, 1), dtype=np.uint8),
    )
    monkeypatch.setattr(cv2, "findFundamentalMat", all_inliers)
    monkeypatch.setattr(cv2, "findHomography", all_inliers)

    evidence = router.visual_continuity(*names)

    assert not evidence.passed
    assert evidence.inlier_ratio == 1.0
    assert evidence.min_grid_fraction < 0.25
    assert evidence.reason == "no_ransac_model_passed"


def test_visual_cache_round_trip_and_image_invalidation(tmp_path) -> None:
    names = ["left.jpg", "right.jpg"]
    for name in names:
        (tmp_path / name).write_bytes(b"image-state")
    poses = {name: make_pose(name) for name in names}
    router = HybridAuxiliaryRouter(
        poses=poses,
        intrinsics=make_intrinsics(),
        image_path_for=lambda name: tmp_path / name,
    )
    router._edge_cache[(names[0], names[1])] = passing_evidence("homography")
    cache_path = tmp_path / "hybrid.json"
    router.save_cache(cache_path)

    restored = HybridAuxiliaryRouter(
        poses=poses,
        intrinsics=make_intrinsics(),
        image_path_for=lambda name: tmp_path / name,
    )
    assert restored.load_cache(cache_path)
    assert restored.visual_continuity(*names).model == "homography"

    (tmp_path / names[1]).write_bytes(b"changed-image-state")
    invalidated = HybridAuxiliaryRouter(
        poses=poses,
        intrinsics=make_intrinsics(),
        image_path_for=lambda name: tmp_path / name,
    )
    assert not invalidated.load_cache(cache_path)
