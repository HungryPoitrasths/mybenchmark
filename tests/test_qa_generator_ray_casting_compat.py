import unittest

import numpy as np

from src.qa_generator import (
    _ModifiedSceneContext,
    _compute_l1_occlusion_metrics,
    _compute_target_visibility,
    _visibility_status_from_ratio,
)
from src.scene_parser import InstanceMeshData
from src.utils.colmap_loader import CameraIntrinsics, CameraPose


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


def make_instance_mesh_data() -> InstanceMeshData:
    surface_points = np.array(
        [
            [-0.6, -0.4, 2.0],
            [0.6, -0.4, 2.0],
            [0.6, 0.4, 2.0],
            [-0.6, 0.4, 2.0],
        ],
        dtype=np.float64,
    )
    return InstanceMeshData(
        vertices=np.array(
            [
                [-0.6, -0.4, 2.0],
                [0.6, -0.4, 2.0],
                [0.6, 0.4, 2.0],
            ],
            dtype=np.float64,
        ),
        faces=np.array([[0, 1, 2]], dtype=np.int64),
        triangle_ids_by_instance={1: np.array([0], dtype=np.int64)},
        boundary_triangle_ids_by_instance={},
        surface_points_by_instance={1: surface_points},
        surface_triangle_ids_by_instance={1: np.zeros(len(surface_points), dtype=np.int64)},
        surface_barycentrics_by_instance={
            1: np.tile(
                np.array([[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]], dtype=np.float64),
                (len(surface_points), 1),
            )
        },
    )


class _LegacyRayCaster:
    def __init__(self) -> None:
        self.stats_called = False
        self.ratio_called = False

    def mesh_visibility_stats(
        self,
        camera_pos,
        target_points,
        target_tri_ids,
        ignored_tri_ids=None,
        hit_epsilon=0.05,
    ):
        self.stats_called = True
        return len(target_points), len(target_points)

    def mesh_visibility_ratio(
        self,
        camera_pos,
        target_points,
        target_tri_ids,
        ignored_tri_ids=None,
        hit_epsilon=0.05,
    ):
        self.ratio_called = True
        return 0.5


class RayCastingCompatTests(unittest.TestCase):
    def test_compute_l1_occlusion_metrics_accepts_legacy_mesh_visibility_stats_signature(self) -> None:
        ray_caster = _LegacyRayCaster()
        metrics = _compute_l1_occlusion_metrics(
            obj={
                "id": 1,
                "label": "chair",
                "center": [0.0, 0.0, 2.0],
                "bbox_min": [-0.6, -0.4, 1.9],
                "bbox_max": [0.6, 0.4, 2.1],
            },
            camera_pose=make_camera_pose(),
            color_intrinsics=make_camera_intrinsics(),
            depth_image=None,
            depth_intrinsics=None,
            occlusion_backend="mesh_ray",
            ray_caster=ray_caster,
            instance_mesh_data=make_instance_mesh_data(),
        )

        self.assertTrue(ray_caster.stats_called)
        self.assertGreater(metrics.valid_in_frame_count, 0)
        self.assertAlmostEqual(metrics.occlusion_ratio_in_frame, 0.0)

    def test_compute_target_visibility_accepts_legacy_mesh_visibility_ratio_signature(self) -> None:
        ray_caster = _LegacyRayCaster()
        status, visible_ratio = _compute_target_visibility(
            modified_scene=_ModifiedSceneContext(
                ray_caster=ray_caster,
                ignored_tri_ids=frozenset(),
            ),
            target_surface_points=np.array(
                [
                    [-0.3, -0.2, 2.0],
                    [0.3, 0.2, 2.0],
                ],
                dtype=np.float64,
            ),
            target_triangle_ids={0},
            camera_pos=np.zeros(3, dtype=np.float64),
            target_sample_triangle_ids=np.array([0, 0], dtype=np.int64),
            target_sample_barycentrics=np.array(
                [
                    [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
                    [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
                ],
                dtype=np.float64,
            ),
            mesh_vertices=np.array(
                [
                    [-0.6, -0.4, 2.0],
                    [0.6, -0.4, 2.0],
                    [0.6, 0.4, 2.0],
                ],
                dtype=np.float64,
            ),
            mesh_faces=np.array([[0, 1, 2]], dtype=np.int64),
        )

        self.assertTrue(ray_caster.ratio_called)
        self.assertAlmostEqual(visible_ratio, 0.5)
        self.assertEqual(status, _visibility_status_from_ratio(0.5))


if __name__ == "__main__":
    unittest.main()
