import inspect
import unittest

import numpy as np

from src.qa_generator import (
    L1_OCCLUSION_SAMPLE_COUNT,
    _compute_l1_occlusion_metrics,
    generate_l1_occlusion_questions,
)
from src.scene_parser import (
    InstanceMeshData,
    _sample_surface_points_from_triangles,
    load_instance_mesh_data,
)
from src.utils.colmap_loader import CameraIntrinsics, CameraPose
from src.utils.depth_occlusion import compute_mesh_depth_occlusion_metrics
from src.utils.ray_casting import RayCaster


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


def make_instance_mesh_data(surface_points: np.ndarray) -> InstanceMeshData:
    return InstanceMeshData(
        vertices=np.empty((0, 3), dtype=np.float64),
        faces=np.empty((0, 3), dtype=np.int64),
        triangle_ids_by_instance={1: np.array([5], dtype=np.int64)},
        boundary_triangle_ids_by_instance={},
        surface_points_by_instance={1: np.asarray(surface_points, dtype=np.float64)},
    )


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
    ):
        return 20, 20


class MeshSamplingTests(unittest.TestCase):
    def test_instance_mesh_loader_default_surface_samples_is_512(self) -> None:
        default_value = inspect.signature(load_instance_mesh_data).parameters[
            "n_surface_samples"
        ].default
        self.assertEqual(default_value, 512)

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
        self.assertEqual(valid, 2)
        self.assertAlmostEqual(ratio, 0.5)


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
