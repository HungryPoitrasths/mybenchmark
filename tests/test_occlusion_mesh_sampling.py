import inspect
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from src.qa_generator import (
    L1_OCCLUSION_SAMPLE_COUNT,
    _COUNTERFACTUAL_OTHER_TRI_ID,
    _COUNTERFACTUAL_TARGET_TRI_ID,
    _ModifiedSceneContext,
    _compute_counterfactual_target_visibility,
    _compute_l1_occlusion_metrics,
    _compute_movement_visibility_status_per_object,
    _compute_visibility_status_per_object,
    _counterfactual_hit_path,
    _get_instance_intersector,
    _instance_surface_sample_metadata,
    _make_l1_occlusion_metrics,
    generate_l1_occlusion_questions,
)
from src.scene_parser import (
    InstanceMeshData,
    _adaptive_instance_surface_sample_counts,
    _sample_surface_points_from_triangles,
    load_instance_mesh_data,
)
from src.utils.colmap_loader import CameraIntrinsics, CameraPose
from src.utils.depth_occlusion import compute_mesh_depth_occlusion_metrics
from src.utils.ray_casting import (
    RayCaster,
    _classify_hit_path,
    _compress_hit_path,
    _local_triangle_resamples,
)


def make_camera_pose() -> CameraPose:
    return CameraPose(
        image_name="test.jpg",
        rotation=np.eye(3, dtype=np.float64),
        translation=np.zeros(3, dtype=np.float64),
    )


def make_camera_intrinsics() -> CameraIntrinsics:
    return CameraIntrinsics(
        width=200,
        height=200,
        fx=100.0,
        fy=100.0,
        cx=100.0,
        cy=100.0,
    )


def make_instance_mesh_data(
    surface_points: np.ndarray,
    *,
    vertices: np.ndarray | None = None,
    faces: np.ndarray | None = None,
    triangle_ids_by_instance: dict[int, np.ndarray] | None = None,
    boundary_triangle_ids_by_instance: dict[int, np.ndarray] | None = None,
    surface_triangle_ids_by_instance: dict[int, np.ndarray] | None = None,
    surface_barycentrics_by_instance: dict[int, np.ndarray] | None = None,
) -> InstanceMeshData:
    points = np.asarray(surface_points, dtype=np.float64)
    vertex_array = (
        np.asarray(vertices, dtype=np.float64)
        if vertices is not None
        else np.empty((0, 3), dtype=np.float64)
    )
    face_array = (
        np.asarray(faces, dtype=np.int64)
        if faces is not None
        else np.empty((0, 3), dtype=np.int64)
    )
    triangle_map = (
        triangle_ids_by_instance
        if triangle_ids_by_instance is not None
        else {1: np.array([5], dtype=np.int64)}
    )
    boundary_map = (
        boundary_triangle_ids_by_instance
        if boundary_triangle_ids_by_instance is not None
        else {}
    )
    surface_tri_map = (
        surface_triangle_ids_by_instance
        if surface_triangle_ids_by_instance is not None
        else {1: np.full(len(points), 5, dtype=np.int64)}
    )
    surface_bary_map = (
        surface_barycentrics_by_instance
        if surface_barycentrics_by_instance is not None
        else {
            1: np.tile(
                np.array([[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]], dtype=np.float64),
                (len(points), 1),
            )
        }
    )
    return InstanceMeshData(
        vertices=vertex_array,
        faces=face_array,
        triangle_ids_by_instance=triangle_map,
        boundary_triangle_ids_by_instance=boundary_map,
        surface_points_by_instance={1: points},
        surface_triangle_ids_by_instance=surface_tri_map,
        surface_barycentrics_by_instance=surface_bary_map,
    )


class _PlaneHitCaster:
    def __init__(self, z: float, tri_id: int, half_extent: float = 0.5) -> None:
        self.z = float(z)
        self.tri_id = int(tri_id)
        self.half_extent = float(half_extent)

    def cast_ray(self, origin, direction):
        origin = np.asarray(origin, dtype=np.float64)
        direction = np.asarray(direction, dtype=np.float64)
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-12:
            return []
        unit_dir = direction / norm
        if abs(float(unit_dir[2])) <= 1e-12:
            return []
        dist = (self.z - float(origin[2])) / float(unit_dir[2])
        if dist <= 1e-12:
            return []
        hit_point = origin + unit_dir * dist
        if (
            abs(float(hit_point[0])) > self.half_extent
            or abs(float(hit_point[1])) > self.half_extent
        ):
            return []
        return [(hit_point, self.tri_id, float(dist))]


class _FakeMeshVisibilityCaster:
    mesh_visibility_stats = RayCaster.mesh_visibility_stats
    mesh_visibility_ratio = RayCaster.mesh_visibility_ratio

    def __init__(self) -> None:
        self.has_embree = False
        self._warned_slow_mesh_visibility = False

    def _first_non_ignored_hits(self, origins, directions, ignored_tri_ids=None):
        first_hits = {
            0: (10, 1.0),
            1: (99, 0.5),
        }
        has_any_hit = np.array([True, True, False], dtype=bool)
        forced_blocked = np.zeros(3, dtype=bool)
        return first_hits, has_any_hit, forced_blocked

    def first_hits_for_triangles(self, origins, directions, target_tri_ids, ignored_tri_ids=None):
        result: dict[int, tuple[np.ndarray, int, float]] = {}
        if 10 in target_tri_ids:
            result[0] = (np.array([1.0, 0.0, 0.0], dtype=np.float64), 10, 1.0)
        return result


class _FakeDepthRayCaster:
    def first_hits_for_triangles(self, origins, directions, target_tri_ids, ignored_tri_ids=None):
        return {
            0: (np.array([0.0, 0.0, 1.0], dtype=np.float64), 5, 1.0),
        }


class _FakeInsufficientEvidenceCaster:
    def mesh_visibility_stats(
        self,
        camera_pos,
        target_points,
        target_tri_ids,
        ignored_tri_ids=None,
        hit_epsilon=0.05,
        sample_triangle_ids=None,
        sample_barycentrics=None,
        vertices=None,
        faces=None,
        local_resample_count=12,
    ):
        return 20, 20


class _FixedVisibilityStatsCaster:
    def __init__(self, visible_count: int, valid_count: int) -> None:
        self.visible_count = int(visible_count)
        self.valid_count = int(valid_count)

    def mesh_visibility_stats(
        self,
        camera_pos,
        target_points,
        target_tri_ids,
        ignored_tri_ids=None,
        hit_epsilon=0.05,
        sample_triangle_ids=None,
        sample_barycentrics=None,
        vertices=None,
        faces=None,
        local_resample_count=12,
    ):
        return self.visible_count, self.valid_count


class _AlwaysVisibleBBoxCaster(_FixedVisibilityStatsCaster):
    def cast_ray(self, origin, direction):
        return []


class _AlwaysBlockedBBoxCaster(_FixedVisibilityStatsCaster):
    def cast_ray(self, origin, direction):
        origin = np.asarray(origin, dtype=np.float64)
        direction = np.asarray(direction, dtype=np.float64)
        sample_dist = float(np.linalg.norm(direction))
        if sample_dist <= 1e-12:
            return []
        unit_dir = direction / sample_dist
        hit_dist = max(0.05, sample_dist * 0.5)
        return [(origin + unit_dir * hit_dist, 99, hit_dist)]


class OcclusionHelperTests(unittest.TestCase):
    def test_classify_hit_path_all_classifications(self) -> None:
        target_tri_ids = {10}
        self.assertEqual(
            _classify_hit_path([(10, 2.0)], 2.0, target_tri_ids, 0.05),
            "visible",
        )
        self.assertEqual(
            _classify_hit_path([(99, 1.0), (10, 2.0)], 2.0, target_tri_ids, 0.05),
            "externally_occluded",
        )
        self.assertEqual(
            _classify_hit_path([(10, 1.0), (10, 2.0)], 2.0, target_tri_ids, 0.05),
            "self_occluded",
        )
        self.assertEqual(
            _classify_hit_path(
                [(10, 0.5), (99, 1.0), (10, 2.0)],
                2.0,
                target_tri_ids,
                0.05,
            ),
            "mixed_boundary",
        )
        self.assertEqual(
            _classify_hit_path([(99, 1.0)], 2.0, target_tri_ids, 0.05),
            "invalid",
        )

    def test_compress_hit_path_merges_close_same_type_only(self) -> None:
        self.assertEqual(
            _compress_hit_path([(10, 1.0), (10, 1.0005), (99, 2.0)], {10}),
            [(True, 1.0), (False, 2.0)],
        )
        self.assertEqual(
            _compress_hit_path([(10, 1.0), (99, 1.0005)], {10}),
            [(True, 1.0), (False, 1.0005)],
        )

    def test_local_triangle_resamples_are_deterministic_and_normalized(self) -> None:
        triangle_vertices = np.array(
            [
                [0.0, 0.0, 2.0],
                [1.0, 0.0, 2.0],
                [0.0, 1.0, 2.0],
            ],
            dtype=np.float64,
        )
        barycentric = np.array([0.2, 0.3, 0.5], dtype=np.float64)

        points_a, barys_a = _local_triangle_resamples(
            triangle_vertices=triangle_vertices,
            barycentric=barycentric,
            triangle_id=7,
            n_samples=8,
        )
        points_b, barys_b = _local_triangle_resamples(
            triangle_vertices=triangle_vertices,
            barycentric=barycentric,
            triangle_id=7,
            n_samples=8,
        )

        self.assertEqual(points_a.shape, (8, 3))
        self.assertEqual(barys_a.shape, (8, 3))
        self.assertTrue(np.allclose(points_a, points_b))
        self.assertTrue(np.allclose(barys_a, barys_b))
        self.assertTrue(np.allclose(barys_a.sum(axis=1), 1.0))
        self.assertTrue(np.all(barys_a >= 0.0))

    def test_local_triangle_resamples_reject_degenerate_inputs(self) -> None:
        degenerate_triangle = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
        empty_points, empty_barys = _local_triangle_resamples(
            triangle_vertices=degenerate_triangle,
            barycentric=np.array([0.2, 0.3, 0.5], dtype=np.float64),
            triangle_id=5,
            n_samples=4,
        )
        self.assertEqual(empty_points.shape, (0, 3))
        self.assertEqual(empty_barys.shape, (0, 3))

        valid_triangle = np.array(
            [
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 1.0],
            ],
            dtype=np.float64,
        )
        zero_points, zero_barys = _local_triangle_resamples(
            triangle_vertices=valid_triangle,
            barycentric=np.zeros(3, dtype=np.float64),
            triangle_id=6,
            n_samples=4,
        )
        self.assertEqual(zero_points.shape, (0, 3))
        self.assertEqual(zero_barys.shape, (0, 3))

    def test_get_instance_intersector_caches_result_and_missing_instance(self) -> None:
        instance_mesh_data = InstanceMeshData(
            vertices=np.array(
                [
                    [0.0, 0.0, 1.0],
                    [1.0, 0.0, 1.0],
                    [1.0, 1.0, 1.0],
                    [0.0, 1.0, 1.0],
                ],
                dtype=np.float64,
            ),
            faces=np.array(
                [
                    [0, 1, 2],
                    [0, 2, 3],
                ],
                dtype=np.int64,
            ),
            triangle_ids_by_instance={1: np.array([0, 1], dtype=np.int64)},
            boundary_triangle_ids_by_instance={},
            surface_points_by_instance={1: np.array([[0.5, 0.5, 1.0]], dtype=np.float64)},
        )

        class _FakeTrimeshMesh:
            def __init__(self, vertices, faces, process=False) -> None:
                self.vertices = np.asarray(vertices, dtype=np.float64)
                self.faces = np.asarray(faces, dtype=np.int64)
                self.process = bool(process)

        class _FakeRayCaster:
            def __init__(self, mesh) -> None:
                self.mesh = mesh

        fake_trimesh = SimpleNamespace(Trimesh=_FakeTrimeshMesh)
        with (
            patch.dict(sys.modules, {"trimesh": fake_trimesh}),
            patch("src.utils.ray_casting.RayCaster", _FakeRayCaster),
        ):
            first = _get_instance_intersector(instance_mesh_data, 1)
            second = _get_instance_intersector(instance_mesh_data, 1)
            missing_first = _get_instance_intersector(instance_mesh_data, 2)
            missing_second = _get_instance_intersector(instance_mesh_data, 2)

        self.assertIs(first, second)
        self.assertIsNone(missing_first)
        self.assertIs(missing_first, missing_second)

        cache = instance_mesh_data.__dict__["_intersector_cache"]
        self.assertIs(cache[1], first)
        self.assertIsNone(cache[2])

    def test_counterfactual_hit_path_uses_shifted_origin_for_moved_blocker(self) -> None:
        hit_path = _counterfactual_hit_path(
            modified_scene=_ModifiedSceneContext(
                ray_caster=object(),
                ignored_tri_ids=frozenset(),
            ),
            camera_pos=np.zeros(3, dtype=np.float64),
            direction=np.array([0.0, 0.0, 3.0], dtype=np.float64),
            max_distance=3.05,
            target_triangle_ids={10},
            target_caster=_PlaneHitCaster(z=3.0, tri_id=0),
            target_delta=np.zeros(3, dtype=np.float64),
            blocker_casters={2: _PlaneHitCaster(z=1.0, tri_id=0)},
            blocker_deltas={2: np.array([0.0, 0.0, 1.0], dtype=np.float64)},
        )

        self.assertEqual(
            hit_path,
            [
                (_COUNTERFACTUAL_OTHER_TRI_ID, 2.0),
                (_COUNTERFACTUAL_TARGET_TRI_ID, 3.0),
            ],
        )
        self.assertEqual(
            _classify_hit_path(
                hit_path,
                expected_dist=3.0,
                target_tri_ids={_COUNTERFACTUAL_TARGET_TRI_ID},
                hit_epsilon=0.05,
            ),
            "externally_occluded",
        )

    def test_counterfactual_visibility_falls_back_to_static_target_hits_without_instance_mesh(self) -> None:
        status, visible_ratio = _compute_counterfactual_target_visibility(
            modified_scene=_ModifiedSceneContext(
                ray_caster=_PlaneHitCaster(z=2.0, tri_id=10),
                ignored_tri_ids=frozenset(),
            ),
            target_surface_points=np.array([[0.0, 0.0, 2.0]], dtype=np.float64),
            target_triangle_ids={10},
            camera_pos=np.zeros(3, dtype=np.float64),
            instance_mesh_data=None,
            target_obj_id=None,
        )

        self.assertEqual(status, "fully visible")
        self.assertAlmostEqual(visible_ratio, 1.0)


class MeshSamplingTests(unittest.TestCase):
    def test_instance_mesh_loader_default_surface_samples_is_512(self) -> None:
        default_value = inspect.signature(load_instance_mesh_data).parameters[
            "n_surface_samples"
        ].default
        self.assertEqual(default_value, 512)

    def test_adaptive_surface_sample_counts_grow_with_surface_area(self) -> None:
        vertices = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [2.0, 0.0, 1.0],
                [2.0, 2.0, 1.0],
                [0.0, 2.0, 1.0],
            ],
            dtype=np.float64,
        )
        faces = np.array(
            [
                [0, 1, 2],
                [0, 2, 3],
                [4, 5, 6],
                [4, 6, 7],
            ],
            dtype=np.int64,
        )

        counts = _adaptive_instance_surface_sample_counts(
            vertices=vertices,
            faces=faces,
            triangle_ids_by_instance={
                1: np.array([0, 1], dtype=np.int64),
                2: np.array([2, 3], dtype=np.int64),
            },
            base_n_samples=512,
        )

        self.assertEqual(counts[1], 512)
        self.assertEqual(counts[2], 1024)

    def test_surface_sampling_is_deterministic_and_exact_count(self) -> None:
        vertices = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        )
        faces = np.array(
            [
                [0, 1, 2],
                [0, 2, 3],
            ],
            dtype=np.int64,
        )
        triangle_ids = np.array([0, 1], dtype=np.int64)

        samples_a = _sample_surface_points_from_triangles(
            vertices=vertices,
            faces=faces,
            triangle_ids=triangle_ids,
            n_samples=32,
            rng=np.random.RandomState(7),
        )
        samples_b = _sample_surface_points_from_triangles(
            vertices=vertices,
            faces=faces,
            triangle_ids=triangle_ids,
            n_samples=32,
            rng=np.random.RandomState(7),
        )

        self.assertEqual(samples_a.shape, (32, 3))
        self.assertTrue(np.allclose(samples_a, samples_b))
        self.assertGreaterEqual(len(np.unique(samples_a.round(6), axis=0)), 24)

    def test_mesh_visibility_ratio_excludes_no_hit_rays(self) -> None:
        caster = _FakeMeshVisibilityCaster()
        visible, valid = caster.mesh_visibility_stats(
            camera_pos=np.zeros(3, dtype=np.float64),
            target_points=np.array(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            ),
            target_tri_ids={10},
        )
        ratio = caster.mesh_visibility_ratio(
            camera_pos=np.zeros(3, dtype=np.float64),
            target_points=np.array(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            ),
            target_tri_ids={10},
        )

        self.assertEqual(visible, 1)
        self.assertEqual(valid, 1)
        self.assertAlmostEqual(ratio, 1.0)

    def test_mesh_visibility_excludes_back_face_samples(self) -> None:
        """Back-face samples on thick objects must not inflate the denominator.

        Scenario (camera at origin, 4 sample points):
          ray 0 – front-face, visible: target hit at 2.0, no blocker.
          ray 1 – back-face: target front surface at 2.0, sample at 3.0.
          ray 2 – front-face, occluded: target at 2.0 behind blocker at 1.5.
          ray 3 – no hit at all.

        Expected: valid=2 (rays 0 & 2), visible=1 (ray 0 only).
        """

        class _ThickObjectCaster:
            mesh_visibility_stats = RayCaster.mesh_visibility_stats
            _hits_up_to_distance = RayCaster._hits_up_to_distance
            has_embree = False
            _warned_slow_mesh_visibility = False

            def _first_non_ignored_hits(self, origins, directions, ignored_tri_ids=None):
                first_hits = {
                    0: (10, 2.0),   # target tri at expected distance
                    1: (10, 2.0),   # target front at 2.0 (sample expected 3.0)
                    2: (99, 1.5),   # blocker in front of target
                    # ray 3: no hit
                }
                has_any_hit = np.array([True, True, True, False], dtype=bool)
                forced_blocked = np.zeros(4, dtype=bool)
                return first_hits, has_any_hit, forced_blocked

            def first_hits_for_triangles(self, origins, directions, target_tri_ids, ignored_tri_ids=None):
                return {
                    0: (np.array([0, 0, 2.0]), 10, 2.0),
                    1: (np.array([0, 0, 2.0]), 10, 2.0),   # front surface at 2.0
                    2: (np.array([0, 2.0, 0]), 11, 2.0),
                    # ray 3: no target hit
                }

            def cast_ray(self, origin, direction):
                direction = np.asarray(direction, dtype=np.float64)
                norm = float(np.linalg.norm(direction))
                if norm <= 1e-12:
                    return []
                direction = direction / norm
                if np.allclose(direction, np.array([0.0, 0.0, 1.0], dtype=np.float64)):
                    return [
                        (direction * 2.0, 10, 2.0),
                        (direction * 3.0, 10, 3.0),
                    ]
                return []

        caster = _ThickObjectCaster()
        # Points along axes so expected_dist equals the coordinate value.
        target_points = np.array([
            [0.0, 0.0, 2.0],  # ray 0: front-face, visible
            [0.0, 0.0, 3.0],  # ray 1: back-face (front surface at 2.0)
            [0.0, 2.0, 0.0],  # ray 2: front-face, occluded by tri 99
            [2.0, 0.0, 0.0],  # ray 3: no hit
        ], dtype=np.float64)

        visible, valid = caster.mesh_visibility_stats(
            camera_pos=np.zeros(3, dtype=np.float64),
            target_points=target_points,
            target_tri_ids={10, 11},
        )
        self.assertEqual(valid, 2, "only front-face samples should count as valid")
        self.assertEqual(visible, 1, "only the unoccluded front-face sample is visible")

    def test_mesh_visibility_refines_mixed_boundary_samples(self) -> None:
        class _MixedBoundaryCaster:
            mesh_visibility_stats = RayCaster.mesh_visibility_stats
            _hits_up_to_distance = RayCaster._hits_up_to_distance
            has_embree = False
            _warned_slow_mesh_visibility = False

            def _first_non_ignored_hits(self, origins, directions, ignored_tri_ids=None):
                first_hits = {
                    0: (10, 1.0),
                }
                has_any_hit = np.array([True], dtype=bool)
                forced_blocked = np.zeros(1, dtype=bool)
                return first_hits, has_any_hit, forced_blocked

            def first_hits_for_triangles(self, origins, directions, target_tri_ids, ignored_tri_ids=None):
                return {
                    0: (np.array([0.0, 0.0, 1.0], dtype=np.float64), 10, 1.0),
                }

            def cast_ray(self, origin, direction):
                direction = np.asarray(direction, dtype=np.float64)
                norm = float(np.linalg.norm(direction))
                if norm <= 1e-12:
                    return []
                direction = direction / norm
                if np.allclose(direction, np.array([0.0, 0.0, 1.0], dtype=np.float64), atol=1e-6):
                    return [
                        (direction * 1.0, 10, 1.0),
                        (direction * 1.5, 99, 1.5),
                        (direction * 2.0, 10, 2.0),
                    ]
                if direction[0] > 0.2:
                    sample_dist = float(norm)
                    return [
                        (direction * 0.9, 99, 0.9),
                        (direction * sample_dist, 10, sample_dist),
                    ]
                if direction[0] < -0.2:
                    sample_dist = float(norm)
                    return [
                        (direction * sample_dist, 10, sample_dist),
                    ]
                sample_dist = float(norm)
                return [
                    (direction * 0.8, 10, 0.8),
                    (direction * sample_dist, 10, sample_dist),
                ]

        caster = _MixedBoundaryCaster()
        target_points = np.array([[0.0, 0.0, 2.0]], dtype=np.float64)
        vertices = np.array(
            [
                [-1.0, -1.0, 2.0],
                [1.0, -1.0, 2.0],
                [0.0, 1.0, 2.0],
            ],
            dtype=np.float64,
        )
        faces = np.array([[0, 1, 2]], dtype=np.int64)

        with patch(
            "src.utils.ray_casting._local_triangle_resamples",
            return_value=(
                np.array(
                    [
                        [1.0, 0.0, 2.0],   # externally occluded
                        [-1.0, 0.0, 2.0],  # visible
                        [0.0, 0.0, 2.0],   # self-occluded
                    ],
                    dtype=np.float64,
                ),
                np.empty((0, 3), dtype=np.float64),
            ),
        ):
            visible, valid = caster.mesh_visibility_stats(
                camera_pos=np.zeros(3, dtype=np.float64),
                target_points=target_points,
                target_tri_ids={10},
                sample_triangle_ids=np.array([0], dtype=np.int64),
                sample_barycentrics=np.array([[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]], dtype=np.float64),
                vertices=vertices,
                faces=faces,
            )

        self.assertEqual(valid, 2, "mixed boundary rays should contribute refined front-surface neighbors")
        self.assertEqual(visible, 1, "refined local samples should preserve only truly visible neighbors")


class CounterfactualMeshVisibilityTests(unittest.TestCase):
    def test_l2_mesh_visibility_uses_only_in_frame_samples(self) -> None:
        class _RecordingRatioCaster:
            def __init__(self) -> None:
                self.calls: list[tuple[int, int, int]] = []

            def mesh_visibility_ratio(
                self,
                camera_pos,
                target_points,
                target_tri_ids,
                ignored_tri_ids=None,
                hit_epsilon=0.05,
                sample_triangle_ids=None,
                sample_barycentrics=None,
                vertices=None,
                faces=None,
                local_resample_count=12,
            ):
                tri_count = len(sample_triangle_ids) if sample_triangle_ids is not None else 0
                bary_count = len(sample_barycentrics) if sample_barycentrics is not None else 0
                self.calls.append((len(target_points), tri_count, bary_count))
                return 1.0 if len(target_points) == 4 else 0.8

        sample_points = np.array(
            [
                [-1.0, -1.0, 2.0],
                [1.0, -1.0, 2.0],
                [1.0, 1.0, 2.0],
                [-1.0, 1.0, 2.0],
                [3.0, 0.0, 2.0],
            ],
            dtype=np.float64,
        )
        caster = _RecordingRatioCaster()
        visibility = _compute_visibility_status_per_object(
            objects=[{"id": 1, "label": "panel"}],
            camera_pose=make_camera_pose(),
            color_intrinsics=make_camera_intrinsics(),
            occlusion_backend="mesh_ray",
            ray_caster=None,
            instance_mesh_data=make_instance_mesh_data(sample_points),
            modified_scene=_ModifiedSceneContext(
                ray_caster=caster,
                ignored_tri_ids=frozenset(),
            ),
        )

        self.assertEqual(caster.calls, [(4, 4, 4)])
        self.assertEqual(visibility[1], ("fully visible", 1.0))

    def test_counterfactual_movement_visibility_uses_only_in_frame_samples(self) -> None:
        sample_points = np.array(
            [
                [-1.0, -1.0, 2.0],
                [1.0, -1.0, 2.0],
                [1.0, 1.0, 2.0],
                [-1.0, 1.0, 2.0],
                [3.0, 0.0, 2.0],
            ],
            dtype=np.float64,
        )
        instance_mesh_data = make_instance_mesh_data(sample_points)
        objects = [
            {
                "id": 1,
                "label": "panel",
                "center": [0.0, 0.0, 2.0],
            }
        ]

        with patch(
            "src.qa_generator._compute_counterfactual_target_visibility",
            return_value=("fully visible", 1.0),
        ) as mocked_counterfactual:
            visibility = _compute_movement_visibility_status_per_object(
                original_objects=objects,
                moved_objects=objects,
                moved_ids=set(),
                camera_pose=make_camera_pose(),
                color_intrinsics=make_camera_intrinsics(),
                ray_caster=object(),
                instance_mesh_data=instance_mesh_data,
            )

        self.assertEqual(visibility[1], ("fully visible", 1.0))
        _, kwargs = mocked_counterfactual.call_args
        self.assertEqual(len(kwargs["target_surface_points"]), 4)
        self.assertEqual(len(kwargs["sample_triangle_ids"]), 4)
        self.assertEqual(len(kwargs["sample_barycentrics"]), 4)

    def test_counterfactual_mesh_visibility_refines_mixed_boundary_samples(self) -> None:
        class _EmptySceneCaster:
            def cast_ray(self, origin, direction):
                return []

        class _TargetCaster:
            def cast_ray(self, origin, direction):
                direction = np.asarray(direction, dtype=np.float64)
                sample_dist = float(np.linalg.norm(direction))
                if sample_dist <= 1e-12:
                    return []
                direction = direction / sample_dist
                if np.allclose(direction, np.array([0.0, 0.0, 1.0], dtype=np.float64), atol=1e-6):
                    return [
                        (direction * 1.0, 0, 1.0),
                        (direction * sample_dist, 0, sample_dist),
                    ]
                if direction[0] < -0.2:
                    return [
                        (direction * sample_dist, 0, sample_dist),
                    ]
                if abs(direction[0]) <= 0.2:
                    return [
                        (direction * 0.8, 0, 0.8),
                        (direction * sample_dist, 0, sample_dist),
                    ]
                return [
                    (direction * sample_dist, 0, sample_dist),
                ]

        class _BlockerCaster:
            def cast_ray(self, origin, direction):
                direction = np.asarray(direction, dtype=np.float64)
                sample_dist = float(np.linalg.norm(direction))
                if sample_dist <= 1e-12:
                    return []
                direction = direction / sample_dist
                if np.allclose(direction, np.array([0.0, 0.0, 1.0], dtype=np.float64), atol=1e-6):
                    return [
                        (direction * 1.5, 0, 1.5),
                    ]
                if direction[0] > 0.2:
                    return [
                        (direction * 0.9, 0, 0.9),
                    ]
                return []

        target_points = np.array([[0.0, 0.0, 2.0]], dtype=np.float64)
        vertices = np.array(
            [
                [-1.0, -1.0, 2.0],
                [1.0, -1.0, 2.0],
                [0.0, 1.0, 2.0],
            ],
            dtype=np.float64,
        )
        faces = np.array([[0, 1, 2]], dtype=np.int64)
        instance_mesh_data = InstanceMeshData(
            vertices=vertices,
            faces=faces,
            triangle_ids_by_instance={
                1: np.array([10], dtype=np.int64),
                2: np.array([99], dtype=np.int64),
            },
            boundary_triangle_ids_by_instance={},
            surface_points_by_instance={1: target_points},
            surface_triangle_ids_by_instance={1: np.array([0], dtype=np.int64)},
            surface_barycentrics_by_instance={
                1: np.array([[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]], dtype=np.float64),
            },
        )

        def _fake_get_instance_intersector(instance_data, obj_id):
            if int(obj_id) == 1:
                return _TargetCaster()
            if int(obj_id) == 2:
                return _BlockerCaster()
            return None

        with (
            patch(
                "src.qa_generator._get_instance_intersector",
                side_effect=_fake_get_instance_intersector,
            ),
            patch(
                "src.qa_generator._local_triangle_resamples",
                return_value=(
                    np.array(
                        [
                            [1.0, 0.0, 2.0],
                            [-1.0, 0.0, 2.0],
                            [0.0, 0.0, 2.0],
                        ],
                        dtype=np.float64,
                    ),
                    np.empty((0, 3), dtype=np.float64),
                ),
            ),
        ):
            status, visible_ratio = _compute_counterfactual_target_visibility(
                modified_scene=_ModifiedSceneContext(
                    ray_caster=_EmptySceneCaster(),
                    ignored_tri_ids=frozenset({99}),
                ),
                target_surface_points=target_points,
                target_triangle_ids={10},
                camera_pos=np.zeros(3, dtype=np.float64),
                instance_mesh_data=instance_mesh_data,
                target_obj_id=1,
                moved_blocker_deltas={2: np.zeros(3, dtype=np.float64)},
                sample_triangle_ids=np.array([0], dtype=np.int64),
                sample_barycentrics=np.array([[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]], dtype=np.float64),
                vertices=vertices,
                faces=faces,
            )

        self.assertEqual(status, "partially occluded")
        self.assertAlmostEqual(visible_ratio, 0.5)


class MeshDepthOcclusionTests(unittest.TestCase):
    def test_mesh_depth_metrics_exclude_no_hit_and_missing_depth(self) -> None:
        intrinsics = CameraIntrinsics(
            width=100,
            height=100,
            fx=100.0,
            fy=100.0,
            cx=50.0,
            cy=50.0,
        )
        depth_image = np.zeros((100, 100), dtype=np.float32)
        depth_image[50, 50] = 1.0
        depth_image[50, 60] = 1.0
        depth_image[50, 40] = 0.0

        metrics = compute_mesh_depth_occlusion_metrics(
            target_points=np.array(
                [
                    [0.0, 0.0, 1.0],
                    [0.1, 0.0, 1.0],
                    [-0.1, 0.0, 1.0],
                ],
                dtype=np.float64,
            ),
            target_tri_ids={5},
            camera_pose=make_camera_pose(),
            intrinsics=intrinsics,
            depth_image=depth_image,
            ray_caster=_FakeDepthRayCaster(),
        )

        self.assertEqual(int(metrics["in_frame_sample_count"]), 3)
        self.assertEqual(int(metrics["valid_in_frame_count"]), 1)
        self.assertEqual(int(metrics["visible_in_frame_count"]), 1)
        self.assertAlmostEqual(float(metrics["visible_ratio_in_frame"]), 1.0)
        self.assertAlmostEqual(float(metrics["occlusion_ratio_in_frame"]), 0.0)


class L1OcclusionQuestionTests(unittest.TestCase):
    def test_instance_surface_sample_metadata_tolerates_legacy_instance_mesh_data(self) -> None:
        legacy_mesh_data = SimpleNamespace(
            vertices=np.empty((0, 3), dtype=np.float64),
            faces=np.empty((0, 3), dtype=np.int64),
            triangle_ids_by_instance={1: np.array([0], dtype=np.int64)},
            boundary_triangle_ids_by_instance={},
            surface_points_by_instance={1: np.array([[0.0, 0.0, 1.0]], dtype=np.float64)},
        )

        triangle_ids, barycentrics = _instance_surface_sample_metadata(legacy_mesh_data, 1)

        self.assertEqual(triangle_ids.shape, (0,))
        self.assertEqual(barycentrics.shape, (0, 3))

    def test_l1_depth_metrics_keep_color_in_frame_denominator(self) -> None:
        surface_points = np.array(
            [
                [0.0, 0.0, 1.0],
                [0.3, -0.2, 1.0],
                [0.6, 0.2, 1.0],
            ],
            dtype=np.float64,
        )
        metrics = _compute_l1_occlusion_metrics(
            obj={
                "id": 1,
                "label": "cup",
                "center": [0.0, 0.0, 1.0],
                "bbox_min": [-0.1, -0.1, 0.9],
                "bbox_max": [0.1, 0.1, 1.1],
            },
            camera_pose=make_camera_pose(),
            color_intrinsics=make_camera_intrinsics(),
            depth_image=np.pad(
                np.ones((1, 1), dtype=np.float32),
                ((100, 99), (20, 19)),
                mode="constant",
            ),
            depth_intrinsics=CameraIntrinsics(
                width=40,
                height=200,
                fx=100.0,
                fy=100.0,
                cx=20.0,
                cy=100.0,
            ),
            occlusion_backend="depth",
            ray_caster=_FakeDepthRayCaster(),
            instance_mesh_data=make_instance_mesh_data(surface_points),
        )

        self.assertEqual(metrics.in_frame_sample_count, 3)
        self.assertEqual(metrics.valid_in_frame_count, 1)
        self.assertAlmostEqual(metrics.effective_ratio, 1.0 / 3.0)

    def test_l1_depth_metrics_raise_when_depth_inputs_lack_ray_caster(self) -> None:
        surface_points = np.array(
            [
                [0.0, 0.0, 1.0],
                [0.3, -0.2, 1.0],
                [0.6, 0.2, 1.0],
            ],
            dtype=np.float64,
        )

        with self.assertRaisesRegex(RuntimeError, "ray_caster"):
            _compute_l1_occlusion_metrics(
                obj={
                    "id": 1,
                    "label": "cup",
                    "center": [0.0, 0.0, 1.0],
                    "bbox_min": [-0.1, -0.1, 0.9],
                    "bbox_max": [0.1, 0.1, 1.1],
                },
                camera_pose=make_camera_pose(),
                color_intrinsics=make_camera_intrinsics(),
                depth_image=np.ones((200, 40), dtype=np.float32),
                depth_intrinsics=CameraIntrinsics(
                    width=40,
                    height=200,
                    fx=100.0,
                    fy=100.0,
                    cx=20.0,
                    cy=100.0,
                ),
                occlusion_backend="depth",
                ray_caster=None,
                instance_mesh_data=make_instance_mesh_data(surface_points),
            )

    def test_l1_mesh_metrics_use_stricter_binary_thresholds(self) -> None:
        xs = np.linspace(-0.5, 0.5, 32, dtype=np.float64)
        ys = np.linspace(-0.25, 0.25, 16, dtype=np.float64)
        grid = np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1, 2)
        surface_points = np.column_stack(
            [grid[:, 0], grid[:, 1], np.full(len(grid), 2.0, dtype=np.float64)]
        )
        instance_mesh_data = make_instance_mesh_data(surface_points)
        obj = {
            "id": 1,
            "label": "cup",
            "center": [0.0, 0.0, 2.0],
            "bbox_min": [-0.5, -0.25, 1.9],
            "bbox_max": [0.5, 0.25, 2.1],
        }

        not_occluded_metrics = _compute_l1_occlusion_metrics(
            obj=obj,
            camera_pose=make_camera_pose(),
            color_intrinsics=make_camera_intrinsics(),
            depth_image=None,
            depth_intrinsics=None,
            occlusion_backend="mesh_ray",
            ray_caster=_FixedVisibilityStatsCaster(visible_count=510, valid_count=512),
            instance_mesh_data=instance_mesh_data,
        )
        occluded_metrics = _compute_l1_occlusion_metrics(
            obj=obj,
            camera_pose=make_camera_pose(),
            color_intrinsics=make_camera_intrinsics(),
            depth_image=None,
            depth_intrinsics=None,
            occlusion_backend="mesh_ray",
            ray_caster=_FixedVisibilityStatsCaster(visible_count=460, valid_count=512),
            instance_mesh_data=instance_mesh_data,
        )
        not_visible_metrics = _compute_l1_occlusion_metrics(
            obj=obj,
            camera_pose=make_camera_pose(),
            color_intrinsics=make_camera_intrinsics(),
            depth_image=None,
            depth_intrinsics=None,
            occlusion_backend="mesh_ray",
            ray_caster=_AlwaysBlockedBBoxCaster(visible_count=0, valid_count=512),
            instance_mesh_data=instance_mesh_data,
        )

        self.assertEqual(not_occluded_metrics.decision, "not occluded")
        self.assertLess(not_occluded_metrics.occlusion_ratio_in_frame, 0.005)
        self.assertEqual(not_occluded_metrics.occluded_in_frame_count, 2)
        self.assertEqual(occluded_metrics.decision, "occluded")
        self.assertGreater(occluded_metrics.occlusion_ratio_in_frame, 0.10)
        self.assertEqual(occluded_metrics.occluded_in_frame_count, 52)
        self.assertEqual(not_visible_metrics.decision, "not visible")
        self.assertEqual(not_visible_metrics.not_visible_probe_visible_count, 0)

    def test_l1_occlusion_requires_strictly_less_than_half_percent_for_not_occluded(self) -> None:
        not_occluded_metrics = _make_l1_occlusion_metrics(
            projected_area=500.0,
            in_frame_ratio=1.0,
            occlusion_ratio_in_frame=0.004,
            valid_in_frame_count=512,
            sampled_point_count=512,
            in_frame_sample_count=512,
            backend="mesh_ray",
        )
        boundary_metrics = _make_l1_occlusion_metrics(
            projected_area=500.0,
            in_frame_ratio=1.0,
            occlusion_ratio_in_frame=0.005,
            valid_in_frame_count=512,
            sampled_point_count=512,
            in_frame_sample_count=512,
            backend="mesh_ray",
        )

        self.assertEqual(not_occluded_metrics.decision, "not occluded")
        self.assertEqual(boundary_metrics.decision, "grayzone")

    def test_l1_occlusion_demotes_high_ratio_with_too_few_occluded_samples_to_grayzone(self) -> None:
        insufficient_count_metrics = _make_l1_occlusion_metrics(
            projected_area=500.0,
            in_frame_ratio=1.0,
            occlusion_ratio_in_frame=0.125,
            valid_in_frame_count=64,
            sampled_point_count=512,
            in_frame_sample_count=64,
            backend="mesh_ray",
            visible_in_frame_count=56,
        )
        sufficient_count_metrics = _make_l1_occlusion_metrics(
            projected_area=500.0,
            in_frame_ratio=1.0,
            occlusion_ratio_in_frame=0.25,
            valid_in_frame_count=64,
            sampled_point_count=512,
            in_frame_sample_count=64,
            backend="mesh_ray",
            visible_in_frame_count=48,
        )

        self.assertGreater(insufficient_count_metrics.occlusion_ratio_in_frame, 0.10)
        self.assertEqual(insufficient_count_metrics.occluded_in_frame_count, 8)
        self.assertEqual(insufficient_count_metrics.decision, "grayzone")
        self.assertEqual(sufficient_count_metrics.occluded_in_frame_count, 16)
        self.assertEqual(sufficient_count_metrics.decision, "occluded")

    def test_l1_occlusion_absent_vlm_status_uses_strict_mesh_ray_review_not_direct_absent_label(self) -> None:
        strict_review = {
            "obj_id": 1,
            "strict_not_visible": True,
            "reason": "all_mesh_rays_blocked",
            "strict_ray_budget": 512,
            "strict_ray_valid_count": 512,
            "strict_ray_visible_count": 0,
        }

        with patch(
            "src.qa_generator._evaluate_absent_label_strict_not_visible_candidate",
            return_value=strict_review,
        ) as strict_review_mock:
            questions = generate_l1_occlusion_questions(
                objects=[
                    {
                        "id": 1,
                        "label": "cup",
                        "center": [0.0, 0.0, 2.0],
                        "bbox_min": [-0.5, -0.25, 1.9],
                        "bbox_max": [0.5, 0.25, 2.1],
                    }
                ],
                camera_pose=make_camera_pose(),
                color_intrinsics=make_camera_intrinsics(),
                depth_image=None,
                depth_intrinsics=None,
                occlusion_backend="mesh_ray",
                ray_caster=_AlwaysBlockedBBoxCaster(visible_count=0, valid_count=512),
                instance_mesh_data=InstanceMeshData(
                    vertices=np.empty((0, 3), dtype=np.float64),
                    faces=np.empty((0, 3), dtype=np.int64),
                    triangle_ids_by_instance={1: np.array([0], dtype=np.int64)},
                    boundary_triangle_ids_by_instance={},
                    surface_points_by_instance={1: np.empty((0, 3), dtype=np.float64)},
                ),
                templates={},
                label_statuses={"cup": "absent"},
                label_counts={"cup": 0},
            )

        self.assertEqual(strict_review_mock.call_count, 1)
        self.assertEqual(len(questions), 1)
        self.assertEqual(questions[0]["correct_value"], "not visible")
        self.assertEqual(
            questions[0]["occlusion_decision_source"],
            "strict_mesh_ray_review_from_vlm_absent",
        )
        self.assertEqual(questions[0]["vlm_label_status"], "absent")

    def test_l1_occlusion_skips_grayzone_ratio_between_1_and_10_percent(self) -> None:
        xs = np.linspace(-0.5, 0.5, 32, dtype=np.float64)
        ys = np.linspace(-0.25, 0.25, 16, dtype=np.float64)
        grid = np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1, 2)
        surface_points = np.column_stack(
            [grid[:, 0], grid[:, 1], np.full(len(grid), 2.0, dtype=np.float64)]
        )

        questions = generate_l1_occlusion_questions(
            objects=[
                {
                    "id": 1,
                    "label": "cup",
                    "center": [0.0, 0.0, 2.0],
                    "bbox_min": [-0.5, -0.25, 1.9],
                    "bbox_max": [0.5, 0.25, 2.1],
                },
            ],
            camera_pose=make_camera_pose(),
            color_intrinsics=make_camera_intrinsics(),
            depth_image=None,
            depth_intrinsics=None,
            occlusion_backend="mesh_ray",
            ray_caster=_FixedVisibilityStatsCaster(visible_count=486, valid_count=512),
            instance_mesh_data=InstanceMeshData(
                vertices=np.empty((0, 3), dtype=np.float64),
                faces=np.empty((0, 3), dtype=np.int64),
                triangle_ids_by_instance={1: np.array([0], dtype=np.int64)},
                boundary_triangle_ids_by_instance={},
                surface_points_by_instance={1: surface_points},
            ),
            templates={},
            label_counts={"cup": 1},
        )

        self.assertEqual(questions, [])

    def test_l1_occlusion_skips_question_when_effective_evidence_is_insufficient(self) -> None:
        xs = np.linspace(-0.5, 0.5, 32, dtype=np.float64)
        ys = np.linspace(-0.25, 0.25, 16, dtype=np.float64)
        grid = np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1, 2)
        surface_points = np.column_stack(
            [grid[:, 0], grid[:, 1], np.full(len(grid), 2.0, dtype=np.float64)]
        )
        self.assertEqual(len(surface_points), L1_OCCLUSION_SAMPLE_COUNT)

        questions = generate_l1_occlusion_questions(
            objects=[
                {
                    "id": 1,
                    "label": "cup",
                    "center": [0.0, 0.0, 2.0],
                    "bbox_min": [-0.5, -0.25, 1.9],
                    "bbox_max": [0.5, 0.25, 2.1],
                },
            ],
            camera_pose=make_camera_pose(),
            color_intrinsics=make_camera_intrinsics(),
            depth_image=None,
            depth_intrinsics=None,
            occlusion_backend="mesh_ray",
            ray_caster=_FakeInsufficientEvidenceCaster(),
            instance_mesh_data=InstanceMeshData(
                vertices=np.empty((0, 3), dtype=np.float64),
                faces=np.empty((0, 3), dtype=np.int64),
                triangle_ids_by_instance={1: np.array([0], dtype=np.int64)},
                boundary_triangle_ids_by_instance={},
                surface_points_by_instance={1: surface_points},
            ),
            templates={},
            label_counts={"cup": 1},
        )

        self.assertEqual(questions, [])


if __name__ == "__main__":
    unittest.main()
