from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from src.qa_generator import (
    _default_templates,
    _direction_with_camera_hint,
    _l1_occlusion_question,
    _make_l1_occlusion_metrics,
    _normalize_template_aliases,
    _removed_object_occludes_target_mesh,
    generate_l2_object_remove,
    generate_l2_viewpoint_move,
)
from src.utils.colmap_loader import CameraIntrinsics, CameraPose


EXPECTED_L2_OBJECT_CENTRIC_ORBIT_TEMPLATE = (
    "Imagine you are {obj_query} and facing toward {obj_face}. "
    "If {obj_move_source} were moved along a {angle}-degree {rotation_direction} "
    "(viewed from above) orbit around the center of {obj_face} in the horizontal "
    "plane, without changing its own facing direction, from your perspective, in "
    "which direction would {obj_ref} be? (For horizontal directions, compare "
    "the objects' 3D bounding-box centers projected onto the floor plane; "
    "above/below use the vertical spatial rule.)"
)


def make_camera_pose() -> CameraPose:
    return CameraPose(
        image_name="test.jpg",
        rotation=np.eye(3, dtype=np.float64),
        translation=np.zeros(3, dtype=np.float64),
    )


def make_camera_intrinsics() -> CameraIntrinsics:
    return CameraIntrinsics(
        width=320,
        height=240,
        fx=200.0,
        fy=200.0,
        cx=160.0,
        cy=120.0,
    )


def make_l1_metrics(status: str):
    if status == "not occluded":
        return _make_l1_occlusion_metrics(
            projected_area=500.0,
            in_frame_ratio=1.0,
            occlusion_ratio_in_frame=0.0,
            valid_in_frame_count=512,
            sampled_point_count=512,
            in_frame_sample_count=512,
            backend="mesh_ray",
        )
    if status == "occluded":
        return _make_l1_occlusion_metrics(
            projected_area=500.0,
            in_frame_ratio=1.0,
            occlusion_ratio_in_frame=0.2,
            valid_in_frame_count=512,
            sampled_point_count=512,
            in_frame_sample_count=512,
            backend="mesh_ray",
        )
    if status == "not visible":
        return _make_l1_occlusion_metrics(
            projected_area=0.0,
            in_frame_ratio=0.0,
            occlusion_ratio_in_frame=1.0,
            valid_in_frame_count=0,
            sampled_point_count=512,
            in_frame_sample_count=0,
            backend="mesh_ray",
        )
    raise ValueError(f"Unsupported test status: {status}")


def make_grayzone_metrics():
    return _make_l1_occlusion_metrics(
        projected_area=500.0,
        in_frame_ratio=1.0,
        occlusion_ratio_in_frame=0.05,
        valid_in_frame_count=512,
        sampled_point_count=512,
        in_frame_sample_count=512,
        backend="mesh_ray",
    )


def make_removed_object_occlusion_probe_metrics(
    *,
    blocking_hit_count: int = 32,
    valid_probe_count: int = 512,
    passes_threshold: bool | None = None,
    reason_code: str | None = None,
):
    valid_probe_count = max(int(valid_probe_count), 0)
    blocking_hit_count = max(0, min(int(blocking_hit_count), valid_probe_count))
    blocking_hit_ratio = (
        float(blocking_hit_count / valid_probe_count)
        if valid_probe_count > 0 else 0.0
    )
    if passes_threshold is None:
        passes_threshold = valid_probe_count > 0 and blocking_hit_ratio > 0.05
    if reason_code is None:
        reason_code = (
            "blocking_ratio_threshold_met"
            if passes_threshold
            else "blocking_ratio_below_threshold"
        )
    return {
        "removed_obj_id": 1,
        "target_obj_id": 2,
        "probe_sample_budget": 512,
        "threshold_ratio": 0.05,
        "threshold_operator": ">",
        "projected_area": 500.0,
        "in_frame_ratio": 1.0,
        "in_frame_sample_count": valid_probe_count,
        "selected_probe_sample_count": valid_probe_count,
        "valid_probe_count": valid_probe_count,
        "blocking_hit_count": blocking_hit_count,
        "blocking_hit_ratio": blocking_hit_ratio,
        "removed_obj_triangle_count": 24,
        "passes_threshold": bool(passes_threshold),
        "reason_code": str(reason_code),
    }


_TARGET_TRI_ID = 0
_REMOVED_TRI_ID = 99
_OTHER_TRI_ID = 50


def make_object_remove_probe_instance_mesh_data(target_points: np.ndarray) -> SimpleNamespace:
    points = np.asarray(target_points, dtype=np.float64)
    barycentrics = np.tile(
        np.asarray([[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]], dtype=np.float64),
        (len(points), 1),
    )
    return SimpleNamespace(
        vertices=np.asarray(
            [
                [-1.0, -1.0, 2.0],
                [1.0, -1.0, 2.0],
                [0.0, 1.0, 2.0],
            ],
            dtype=np.float64,
        ),
        faces=np.asarray([[0, 1, 2]], dtype=np.int64),
        triangle_ids_by_instance={
            1: np.asarray([_REMOVED_TRI_ID], dtype=np.int64),
            2: np.asarray([_TARGET_TRI_ID], dtype=np.int64),
        },
        boundary_triangle_ids_by_instance={},
        surface_points_by_instance={2: points},
        surface_triangle_ids_by_instance={
            2: np.full(len(points), _TARGET_TRI_ID, dtype=np.int64),
        },
        surface_barycentrics_by_instance={2: barycentrics},
    )


def make_object_remove_target_points(count: int = 20) -> np.ndarray:
    return np.asarray(
        [[float(x), 0.0, 2.0] for x in np.linspace(-0.2, 0.2, int(count))],
        dtype=np.float64,
    )


def _visible_target_path(sample_dist: float) -> list[tuple[int, float]]:
    return [(_TARGET_TRI_ID, float(sample_dist))]


def _removed_blocking_path(sample_dist: float) -> list[tuple[int, float]]:
    return [
        (_REMOVED_TRI_ID, max(0.1, float(sample_dist) - 0.2)),
        (_TARGET_TRI_ID, float(sample_dist)),
    ]


def _other_object_path(sample_dist: float) -> list[tuple[int, float]]:
    return [
        (_OTHER_TRI_ID, max(0.1, float(sample_dist) - 0.2)),
        (_TARGET_TRI_ID, float(sample_dist)),
    ]


def _mixed_boundary_path(sample_dist: float) -> list[tuple[int, float]]:
    return [
        (_TARGET_TRI_ID, max(0.1, float(sample_dist) - 0.4)),
        (_REMOVED_TRI_ID, max(0.2, float(sample_dist) - 0.2)),
        (_TARGET_TRI_ID, float(sample_dist)),
    ]


def _local_refine_points(count: int = 12) -> np.ndarray:
    return np.asarray(
        [[-0.05 + 0.01 * idx, 0.0, 2.0] for idx in range(int(count))],
        dtype=np.float64,
    )


class _SequenceHitPathCaster:
    def __init__(
        self,
        hit_tri_ids: list[int],
        blocker_tri_id: int,
        target_tri_id: int = _TARGET_TRI_ID,
    ) -> None:
        self._hit_tri_ids = list(hit_tri_ids)
        self._blocker_tri_id = int(blocker_tri_id)
        self._target_tri_id = int(target_tri_id)
        self._index = 0

    def _hits_up_to_distance(self, origin, direction, max_distance, ignored_tri_ids=None):
        tri_id = self._hit_tri_ids[self._index]
        self._index += 1
        direction_arr = np.asarray(direction, dtype=np.float64)
        dist = float(np.linalg.norm(direction_arr))
        if int(tri_id) == self._blocker_tri_id:
            return [
                (self._blocker_tri_id, max(0.1, dist - 0.1)),
                (self._target_tri_id, dist),
            ]
        return [(self._target_tri_id, dist)]


class _ScriptedHitPathCaster:
    def __init__(self, path_builders) -> None:
        self._path_builders = list(path_builders)
        self._index = 0

    def _hits_up_to_distance(self, origin, direction, max_distance, ignored_tri_ids=None):
        direction_arr = np.asarray(direction, dtype=np.float64)
        sample_dist = float(np.linalg.norm(direction_arr))
        if not np.isfinite(sample_dist) or sample_dist <= 1e-12:
            return []
        if self._index >= len(self._path_builders):
            builder = _visible_target_path
        else:
            builder = self._path_builders[self._index]
        self._index += 1
        ignored = set(ignored_tri_ids or set())
        return [
            (int(tri_id), float(dist))
            for tri_id, dist in builder(sample_dist)
            if float(dist) <= float(max_distance) and int(tri_id) not in ignored
        ]


class QuestionTemplateTests(unittest.TestCase):
    def test_default_object_centric_rotation_templates_use_explicit_orbit_wording(self) -> None:
        templates = _default_templates()

        self.assertEqual(
            templates["L2_object_rotate_object_centric"],
            [EXPECTED_L2_OBJECT_CENTRIC_ORBIT_TEMPLATE],
        )
        self.assertEqual(
            templates["L2_object_move_object_centric"],
            [EXPECTED_L2_OBJECT_CENTRIC_ORBIT_TEMPLATE],
        )

    def test_normalize_template_aliases_backfills_missing_sibling_key(self) -> None:
        with self.subTest("canonical_present_alias_missing"):
            normalized = _normalize_template_aliases(
                {
                    "L2_object_rotate_object_centric": [
                        EXPECTED_L2_OBJECT_CENTRIC_ORBIT_TEMPLATE
                    ]
                }
            )
            self.assertEqual(
                normalized["L2_object_move_object_centric"],
                [EXPECTED_L2_OBJECT_CENTRIC_ORBIT_TEMPLATE],
            )

        with self.subTest("alias_present_canonical_missing"):
            normalized = _normalize_template_aliases(
                {
                    "L2_object_move_object_centric": [
                        EXPECTED_L2_OBJECT_CENTRIC_ORBIT_TEMPLATE
                    ]
                }
            )
            self.assertEqual(
                normalized["L2_object_rotate_object_centric"],
                [EXPECTED_L2_OBJECT_CENTRIC_ORBIT_TEMPLATE],
            )

    def test_direction_with_camera_hint_uses_subject_specific_forward_backward_wording(self) -> None:
        self.assertEqual(
            _direction_with_camera_hint("forward"),
            "forward (away from the camera)",
        )
        self.assertEqual(
            _direction_with_camera_hint("backward"),
            "backward (toward the camera)",
        )
        self.assertEqual(
            _direction_with_camera_hint("forward", moving_subject="camera"),
            "forward (along its viewing direction)",
        )
        self.assertEqual(
            _direction_with_camera_hint("backward", moving_subject="camera"),
            "backward (opposite its viewing direction)",
        )

    def test_l1_occlusion_question_appends_definition_for_custom_templates(self) -> None:
        question = _l1_occlusion_question(
            label="lamp",
            correct="occluded",
            templates={"L1_occlusion": ["Is {obj_a} occluded?"]},
        )

        self.assertIn("blocked by another object", question["question"])
        self.assertIn("does not count as occlusion", question["question"])

    def test_viewpoint_move_questions_use_camera_specific_backward_wording(self) -> None:
        camera_pose = make_camera_pose()
        intrinsics = make_camera_intrinsics()

        with (
            patch("src.qa_generator._counterfactual_occlusion_backend", return_value="mesh_ray"),
            patch("src.qa_generator._build_modified_scene", return_value=None),
            patch("src.qa_generator.apply_viewpoint_change", return_value=camera_pose),
            patch(
                "src.qa_generator._compute_l1_style_visibility_metrics_for_static_target",
                side_effect=[
                    (make_l1_metrics("not occluded"), "mesh_ray"),
                    *[(make_l1_metrics("not occluded"), "mesh_ray")] * 9,
                    (make_l1_metrics("not visible"), "mesh_ray"),
                    *[(make_l1_metrics("not occluded"), "mesh_ray")] * 2,
                ],
            ),
        ):
            questions = generate_l2_viewpoint_move(
                objects=[{"id": 1, "label": "curtain"}],
                camera_pose=camera_pose,
                color_intrinsics=intrinsics,
                depth_image=None,
                depth_intrinsics=None,
                occlusion_backend="mesh_ray",
                ray_caster=object(),
                instance_mesh_data=object(),
                templates={
                    "L2_viewpoint_move": [
                        "If the camera translates {direction_with_camera_hint} by {distance}, what happens to {obj_a}?"
                    ]
                },
            )

        self.assertEqual(len(questions), 1)
        self.assertIn(
            "backward (opposite its viewing direction)",
            questions[0]["question"],
        )
        self.assertNotIn("toward the camera", questions[0]["question"])
        self.assertEqual(questions[0]["correct_value"], "not visible")
        self.assertEqual(
            set(questions[0]["options"]),
            {"not occluded", "occluded", "not visible"},
        )
        self.assertEqual(len(questions[0]["options"]), 3)
        self.assertIn("blocked by another object", questions[0]["question"])
        self.assertIn("does not count as occlusion", questions[0]["question"])

    def test_object_remove_questions_use_l1_style_occlusion_options(self) -> None:
        camera_pose = make_camera_pose()
        intrinsics = make_camera_intrinsics()
        objects = [
            {"id": 1, "label": "chair", "center": [0.0, 0.0, 2.0]},
            {"id": 2, "label": "lamp", "center": [1.0, 0.0, 2.0]},
            {"id": 3, "label": "cabinet", "center": [2.0, 0.0, 2.0]},
        ]

        with (
            patch("src.qa_generator._counterfactual_occlusion_backend", return_value="mesh_ray"),
            patch("src.qa_generator._build_modified_scene", return_value=None),
            patch(
                "src.qa_generator._removed_object_occludes_target_mesh",
                return_value=make_removed_object_occlusion_probe_metrics(),
            ),
            patch(
                "src.qa_generator._compute_l1_style_visibility_metrics_for_static_target",
                side_effect=[
                    (make_l1_metrics("not occluded"), "mesh_ray"),
                    (make_l1_metrics("not occluded"), "mesh_ray"),
                    (make_l1_metrics("not occluded"), "mesh_ray"),
                    (make_l1_metrics("occluded"), "mesh_ray"),
                    (make_l1_metrics("not occluded"), "mesh_ray"),
                    (make_l1_metrics("not occluded"), "mesh_ray"),
                    (make_l1_metrics("not occluded"), "mesh_ray"),
                    (make_l1_metrics("not occluded"), "mesh_ray"),
                    (make_l1_metrics("not occluded"), "mesh_ray"),
                ],
            ),
        ):
            questions = generate_l2_object_remove(
                objects=objects,
                attachment_graph={},
                camera_pose=camera_pose,
                templates={
                    "L2_object_remove": [
                        "If {obj_a} were removed, what is the occlusion status of {obj_b}?"
                    ]
                },
                color_intrinsics=intrinsics,
                depth_image=None,
                depth_intrinsics=None,
                occlusion_backend="mesh_ray",
                ray_caster=object(),
                instance_mesh_data=object(),
            )

        self.assertEqual(len(questions), 2)
        self.assertIn("removed_object_occlusion_probe_metrics", questions[0])
        for question in questions:
            self.assertEqual(
                set(question["options"]),
                {"not occluded", "occluded", "not visible"},
            )
            self.assertEqual(len(question["options"]), 3)
            self.assertIn("blocked by another object", question["question"])
            self.assertIn("does not count as occlusion", question["question"])

        changed_question = next(
            question
            for question in questions
            if question["removed_obj_label"] == "chair"
            and question["obj_b_label"] == "lamp"
        )
        self.assertEqual(changed_question["correct_value"], "occluded")
        self.assertFalse(changed_question["relation_unchanged"])
        self.assertTrue(
            any(question["relation_unchanged"] for question in questions),
        )
        self.assertEqual(
            sum(1 for question in questions if question["relation_unchanged"]),
            1,
        )

    def test_viewpoint_move_skips_grayzone_counterfactual_state(self) -> None:
        camera_pose = make_camera_pose()
        intrinsics = make_camera_intrinsics()

        with (
            patch("src.qa_generator._counterfactual_occlusion_backend", return_value="mesh_ray"),
            patch("src.qa_generator._build_modified_scene", return_value=None),
            patch("src.qa_generator.apply_viewpoint_change", return_value=camera_pose),
            patch(
                "src.qa_generator._compute_l1_style_visibility_metrics_for_static_target",
                side_effect=[
                    (make_l1_metrics("not occluded"), "mesh_ray"),
                    *[(make_l1_metrics("not occluded"), "mesh_ray")] * 11,
                    (make_grayzone_metrics(), "mesh_ray"),
                ],
            ),
        ):
            questions = generate_l2_viewpoint_move(
                objects=[{"id": 1, "label": "curtain"}],
                camera_pose=camera_pose,
                color_intrinsics=intrinsics,
                depth_image=None,
                depth_intrinsics=None,
                occlusion_backend="mesh_ray",
                ray_caster=object(),
                instance_mesh_data=object(),
                templates={
                    "L2_viewpoint_move": [
                        "If the camera translates {direction_with_camera_hint} by {distance}, what happens to {obj_a}?"
                    ]
                },
            )

        self.assertEqual(len(questions), 0)

    def test_viewpoint_move_allows_visible_to_occluded_transition(self) -> None:
        camera_pose = make_camera_pose()
        intrinsics = make_camera_intrinsics()

        with (
            patch("src.qa_generator._counterfactual_occlusion_backend", return_value="mesh_ray"),
            patch("src.qa_generator._build_modified_scene", return_value=None),
            patch("src.qa_generator.apply_viewpoint_change", return_value=camera_pose),
            patch(
                "src.qa_generator._compute_l1_style_visibility_metrics_for_static_target",
                side_effect=[
                    (make_l1_metrics("not occluded"), "mesh_ray"),
                    (make_l1_metrics("occluded"), "mesh_ray"),
                    *[(make_l1_metrics("not occluded"), "mesh_ray")] * 11,
                ],
            ),
        ):
            questions = generate_l2_viewpoint_move(
                objects=[{"id": 1, "label": "curtain"}],
                camera_pose=camera_pose,
                color_intrinsics=intrinsics,
                depth_image=None,
                depth_intrinsics=None,
                occlusion_backend="mesh_ray",
                ray_caster=object(),
                instance_mesh_data=object(),
                templates={
                    "L2_viewpoint_move": [
                        "If the camera translates {direction_with_camera_hint} by {distance}, what happens to {obj_a}?"
                    ]
                },
            )

        self.assertEqual(len(questions), 1)
        self.assertEqual(questions[0]["correct_value"], "occluded")

    def test_viewpoint_move_skips_not_visible_origin_transition(self) -> None:
        camera_pose = make_camera_pose()
        intrinsics = make_camera_intrinsics()

        with (
            patch("src.qa_generator._counterfactual_occlusion_backend", return_value="mesh_ray"),
            patch("src.qa_generator._build_modified_scene", return_value=None),
            patch("src.qa_generator.apply_viewpoint_change", return_value=camera_pose),
            patch(
                "src.qa_generator._compute_l1_style_visibility_metrics_for_static_target",
                side_effect=[
                    (make_l1_metrics("not visible"), "mesh_ray"),
                    *[(make_l1_metrics("occluded"), "mesh_ray")] * 12,
                ],
            ),
        ):
            questions = generate_l2_viewpoint_move(
                objects=[{"id": 1, "label": "curtain"}],
                camera_pose=camera_pose,
                color_intrinsics=intrinsics,
                depth_image=None,
                depth_intrinsics=None,
                occlusion_backend="mesh_ray",
                ray_caster=object(),
                instance_mesh_data=object(),
                templates={
                    "L2_viewpoint_move": [
                        "If the camera translates {direction_with_camera_hint} by {distance}, what happens to {obj_a}?"
                    ]
                },
            )

        self.assertEqual(len(questions), 0)

    def test_object_remove_skips_only_grayzone_counterfactual_state(self) -> None:
        camera_pose = make_camera_pose()
        intrinsics = make_camera_intrinsics()
        objects = [
            {"id": 1, "label": "chair", "center": [0.0, 0.0, 2.0]},
            {"id": 2, "label": "lamp", "center": [1.0, 0.0, 2.0]},
            {"id": 3, "label": "cabinet", "center": [2.0, 0.0, 2.0]},
        ]

        with (
            patch("src.qa_generator._counterfactual_occlusion_backend", return_value="mesh_ray"),
            patch("src.qa_generator._build_modified_scene", return_value=None),
            patch(
                "src.qa_generator._removed_object_occludes_target_mesh",
                return_value=make_removed_object_occlusion_probe_metrics(),
            ),
            patch(
                "src.qa_generator._compute_l1_style_visibility_metrics_for_static_target",
                side_effect=[
                    (make_l1_metrics("not occluded"), "mesh_ray"),
                    (make_l1_metrics("not occluded"), "mesh_ray"),
                    (make_l1_metrics("not occluded"), "mesh_ray"),
                    (make_grayzone_metrics(), "mesh_ray"),
                    (make_l1_metrics("not occluded"), "mesh_ray"),
                    (make_l1_metrics("not occluded"), "mesh_ray"),
                    (make_l1_metrics("not occluded"), "mesh_ray"),
                    (make_l1_metrics("not occluded"), "mesh_ray"),
                    (make_l1_metrics("not occluded"), "mesh_ray"),
                ],
            ),
        ):
            questions = generate_l2_object_remove(
                objects=objects,
                attachment_graph={},
                camera_pose=camera_pose,
                templates={
                    "L2_object_remove": [
                        "If {obj_a} were removed, what is the occlusion status of {obj_b}?"
                    ]
                },
                color_intrinsics=intrinsics,
                depth_image=None,
                depth_intrinsics=None,
                occlusion_backend="mesh_ray",
                ray_caster=object(),
                instance_mesh_data=object(),
            )

        self.assertEqual(len(questions), 1)
        self.assertTrue(
            all(question["correct_value"] == "not occluded" for question in questions),
        )
        self.assertTrue(
            all(question["relation_unchanged"] for question in questions),
        )

    def test_object_remove_does_not_backfill_unchanged_once_changed_floor_is_met(self) -> None:
        camera_pose = make_camera_pose()
        intrinsics = make_camera_intrinsics()
        objects = [
            {"id": 1, "label": "chair", "center": [0.0, 0.0, 2.0]},
            {"id": 2, "label": "lamp", "center": [1.0, 0.0, 2.0]},
            {"id": 3, "label": "cabinet", "center": [2.0, 0.0, 2.0]},
        ]

        with (
            patch("src.qa_generator._counterfactual_occlusion_backend", return_value="mesh_ray"),
            patch("src.qa_generator._build_modified_scene", return_value=None),
            patch(
                "src.qa_generator._removed_object_occludes_target_mesh",
                return_value=make_removed_object_occlusion_probe_metrics(),
            ),
            patch(
                "src.qa_generator._compute_l1_style_visibility_metrics_for_static_target",
                side_effect=[
                    (make_l1_metrics("not occluded"), "mesh_ray"),
                    (make_l1_metrics("not occluded"), "mesh_ray"),
                    (make_l1_metrics("not occluded"), "mesh_ray"),
                    (make_l1_metrics("occluded"), "mesh_ray"),
                    (make_l1_metrics("not visible"), "mesh_ray"),
                    (make_l1_metrics("not occluded"), "mesh_ray"),
                    (make_l1_metrics("not occluded"), "mesh_ray"),
                    (make_l1_metrics("not occluded"), "mesh_ray"),
                    (make_l1_metrics("not occluded"), "mesh_ray"),
                ],
            ),
        ):
            questions = generate_l2_object_remove(
                objects=objects,
                attachment_graph={},
                camera_pose=camera_pose,
                templates={
                    "L2_object_remove": [
                        "If {obj_a} were removed, what is the occlusion status of {obj_b}?"
                    ]
                },
                color_intrinsics=intrinsics,
                depth_image=None,
                depth_intrinsics=None,
                occlusion_backend="mesh_ray",
                ray_caster=object(),
                instance_mesh_data=object(),
            )

        self.assertEqual(len(questions), 2)
        self.assertTrue(
            all(not question["relation_unchanged"] for question in questions),
        )
        self.assertEqual(
            {question["correct_value"] for question in questions},
            {"occluded", "not visible"},
        )

    def test_removed_object_occludes_target_mesh_requires_strictly_more_than_five_percent_hits(self) -> None:
        camera_pose = make_camera_pose()
        intrinsics = make_camera_intrinsics()
        target_points = make_object_remove_target_points()
        instance_mesh_data = make_object_remove_probe_instance_mesh_data(target_points)
        removed_obj = {"id": 1, "label": "chair"}
        target_obj = {
            "id": 2,
            "label": "lamp",
            "bbox_min": [-0.2, -0.1, 1.9],
            "bbox_max": [0.2, 0.1, 2.1],
        }

        equal_metrics = _removed_object_occludes_target_mesh(
            removed_obj=removed_obj,
            target_obj=target_obj,
            camera_pose=camera_pose,
            color_intrinsics=intrinsics,
            ray_caster=_SequenceHitPathCaster(
                [_REMOVED_TRI_ID] + [_TARGET_TRI_ID] * 19,
                blocker_tri_id=_REMOVED_TRI_ID,
            ),
            instance_mesh_data=instance_mesh_data,
        )
        above_metrics = _removed_object_occludes_target_mesh(
            removed_obj=removed_obj,
            target_obj=target_obj,
            camera_pose=camera_pose,
            color_intrinsics=intrinsics,
            ray_caster=_SequenceHitPathCaster(
                [_REMOVED_TRI_ID, _REMOVED_TRI_ID] + [_TARGET_TRI_ID] * 18,
                blocker_tri_id=_REMOVED_TRI_ID,
            ),
            instance_mesh_data=instance_mesh_data,
        )

        self.assertEqual(equal_metrics["valid_probe_count"], 20)
        self.assertEqual(equal_metrics["blocking_hit_count"], 1)
        self.assertAlmostEqual(equal_metrics["blocking_hit_ratio"], 0.05)
        self.assertFalse(equal_metrics["passes_threshold"])
        self.assertEqual(equal_metrics["reason_code"], "blocking_ratio_below_threshold")

        self.assertEqual(above_metrics["blocking_hit_count"], 2)
        self.assertGreater(above_metrics["blocking_hit_ratio"], 0.05)
        self.assertTrue(above_metrics["passes_threshold"])
        self.assertEqual(above_metrics["reason_code"], "blocking_ratio_threshold_met")

    def test_removed_object_occludes_target_mesh_refines_mixed_boundary_to_blocking(self) -> None:
        camera_pose = make_camera_pose()
        intrinsics = make_camera_intrinsics()
        target_points = make_object_remove_target_points()
        instance_mesh_data = make_object_remove_probe_instance_mesh_data(target_points)
        removed_obj = {"id": 1, "label": "chair"}
        target_obj = {
            "id": 2,
            "label": "lamp",
            "bbox_min": [-0.2, -0.1, 1.9],
            "bbox_max": [0.2, 0.1, 2.1],
        }
        ray_caster = _ScriptedHitPathCaster(
            [_mixed_boundary_path] + [_visible_target_path] * 19
            + [_removed_blocking_path] * 12
        )

        with patch(
            "src.qa_generator._local_triangle_resamples",
            return_value=(_local_refine_points(), np.empty((0, 3), dtype=np.float64)),
        ) as resample_mock:
            metrics = _removed_object_occludes_target_mesh(
                removed_obj=removed_obj,
                target_obj=target_obj,
                camera_pose=camera_pose,
                color_intrinsics=intrinsics,
                ray_caster=ray_caster,
                instance_mesh_data=instance_mesh_data,
            )

        self.assertEqual(resample_mock.call_count, 1)
        self.assertEqual(metrics["valid_probe_count"], 31)
        self.assertEqual(metrics["blocking_hit_count"], 12)
        self.assertGreater(metrics["blocking_hit_ratio"], 0.05)
        self.assertTrue(metrics["passes_threshold"])

    def test_removed_object_occludes_target_mesh_refines_mixed_boundary_below_threshold(self) -> None:
        camera_pose = make_camera_pose()
        intrinsics = make_camera_intrinsics()
        target_points = make_object_remove_target_points()
        instance_mesh_data = make_object_remove_probe_instance_mesh_data(target_points)
        removed_obj = {"id": 1, "label": "chair"}
        target_obj = {
            "id": 2,
            "label": "lamp",
            "bbox_min": [-0.2, -0.1, 1.9],
            "bbox_max": [0.2, 0.1, 2.1],
        }
        ray_caster = _ScriptedHitPathCaster(
            [_mixed_boundary_path] + [_visible_target_path] * 19
            + [_removed_blocking_path] + [_visible_target_path] * 11
        )

        with patch(
            "src.qa_generator._local_triangle_resamples",
            return_value=(_local_refine_points(), np.empty((0, 3), dtype=np.float64)),
        ) as resample_mock:
            metrics = _removed_object_occludes_target_mesh(
                removed_obj=removed_obj,
                target_obj=target_obj,
                camera_pose=camera_pose,
                color_intrinsics=intrinsics,
                ray_caster=ray_caster,
                instance_mesh_data=instance_mesh_data,
            )

        self.assertEqual(resample_mock.call_count, 1)
        self.assertEqual(metrics["valid_probe_count"], 31)
        self.assertEqual(metrics["blocking_hit_count"], 1)
        self.assertLess(metrics["blocking_hit_ratio"], 0.05)
        self.assertFalse(metrics["passes_threshold"])
        self.assertEqual(metrics["reason_code"], "blocking_ratio_below_threshold")

    def test_object_remove_generation_keeps_candidate_when_refined_precheck_passes(self) -> None:
        camera_pose = make_camera_pose()
        intrinsics = make_camera_intrinsics()
        target_points = make_object_remove_target_points()
        instance_mesh_data = make_object_remove_probe_instance_mesh_data(target_points)
        objects = [
            {"id": 1, "label": "chair", "center": [0.0, 0.0, 2.0]},
            {
                "id": 2,
                "label": "lamp",
                "center": [1.0, 0.0, 2.0],
                "bbox_min": [-0.2, -0.1, 1.9],
                "bbox_max": [0.2, 0.1, 2.1],
            },
            {"id": 3, "label": "cabinet", "center": [2.0, 0.0, 2.0]},
        ]
        ray_caster = _ScriptedHitPathCaster(
            [_mixed_boundary_path] + [_visible_target_path] * 19
            + [_removed_blocking_path] * 12
        )

        with (
            patch("src.qa_generator._counterfactual_occlusion_backend", return_value="mesh_ray"),
            patch(
                "src.qa_generator._compute_l1_style_visibility_metrics_for_static_target",
                side_effect=[
                    (make_l1_metrics("not occluded"), "mesh_ray"),
                    (make_l1_metrics("not occluded"), "mesh_ray"),
                    (make_l1_metrics("not occluded"), "mesh_ray"),
                    (make_l1_metrics("occluded"), "mesh_ray"),
                ],
            ) as visibility_mock,
            patch(
                "src.qa_generator._local_triangle_resamples",
                return_value=(_local_refine_points(), np.empty((0, 3), dtype=np.float64)),
            ),
        ):
            questions = generate_l2_object_remove(
                objects=objects,
                attachment_graph={},
                camera_pose=camera_pose,
                templates={
                    "L2_object_remove": [
                        "If {obj_a} were removed, what is the occlusion status of {obj_b}?"
                    ]
                },
                color_intrinsics=intrinsics,
                depth_image=None,
                depth_intrinsics=None,
                occlusion_backend="mesh_ray",
                ray_caster=ray_caster,
                instance_mesh_data=instance_mesh_data,
            )

        self.assertEqual(visibility_mock.call_count, 4)
        self.assertEqual(len(questions), 1)
        self.assertEqual(questions[0]["removed_obj_label"], "chair")
        self.assertEqual(questions[0]["obj_b_label"], "lamp")
        self.assertEqual(questions[0]["correct_value"], "occluded")
        self.assertTrue(
            questions[0]["removed_object_occlusion_probe_metrics"]["passes_threshold"],
        )

    def test_object_remove_generation_skips_candidate_when_refined_precheck_stays_below_threshold(self) -> None:
        camera_pose = make_camera_pose()
        intrinsics = make_camera_intrinsics()
        target_points = make_object_remove_target_points()
        instance_mesh_data = make_object_remove_probe_instance_mesh_data(target_points)
        objects = [
            {"id": 1, "label": "chair", "center": [0.0, 0.0, 2.0]},
            {
                "id": 2,
                "label": "lamp",
                "center": [1.0, 0.0, 2.0],
                "bbox_min": [-0.2, -0.1, 1.9],
                "bbox_max": [0.2, 0.1, 2.1],
            },
            {"id": 3, "label": "cabinet", "center": [2.0, 0.0, 2.0]},
        ]
        ray_caster = _ScriptedHitPathCaster(
            [_mixed_boundary_path] + [_visible_target_path] * 19
            + [_other_object_path] * 12
        )

        with (
            patch("src.qa_generator._counterfactual_occlusion_backend", return_value="mesh_ray"),
            patch(
                "src.qa_generator._compute_l1_style_visibility_metrics_for_static_target",
                side_effect=[
                    (make_l1_metrics("not occluded"), "mesh_ray"),
                    (make_l1_metrics("not occluded"), "mesh_ray"),
                    (make_l1_metrics("not occluded"), "mesh_ray"),
                ],
            ) as visibility_mock,
            patch(
                "src.qa_generator._local_triangle_resamples",
                return_value=(_local_refine_points(), np.empty((0, 3), dtype=np.float64)),
            ),
        ):
            questions = generate_l2_object_remove(
                objects=objects,
                attachment_graph={},
                camera_pose=camera_pose,
                templates={
                    "L2_object_remove": [
                        "If {obj_a} were removed, what is the occlusion status of {obj_b}?"
                    ]
                },
                color_intrinsics=intrinsics,
                depth_image=None,
                depth_intrinsics=None,
                occlusion_backend="mesh_ray",
                ray_caster=ray_caster,
                instance_mesh_data=instance_mesh_data,
            )

        self.assertEqual(visibility_mock.call_count, 3)
        self.assertEqual(questions, [])

    def test_object_remove_skips_candidates_when_removed_object_not_occluding_target_mesh(self) -> None:
        camera_pose = make_camera_pose()
        intrinsics = make_camera_intrinsics()
        objects = [
            {"id": 1, "label": "chair", "center": [0.0, 0.0, 2.0]},
            {"id": 2, "label": "lamp", "center": [1.0, 0.0, 2.0]},
            {"id": 3, "label": "cabinet", "center": [2.0, 0.0, 2.0]},
        ]

        with (
            patch("src.qa_generator._counterfactual_occlusion_backend", return_value="mesh_ray"),
            patch("src.qa_generator._build_modified_scene", return_value=None),
            patch(
                "src.qa_generator._removed_object_occludes_target_mesh",
                return_value=make_removed_object_occlusion_probe_metrics(
                    blocking_hit_count=25,
                    valid_probe_count=512,
                    passes_threshold=False,
                ),
            ),
            patch(
                "src.qa_generator._compute_l1_style_visibility_metrics_for_static_target",
                side_effect=[
                    (make_l1_metrics("not occluded"), "mesh_ray"),
                    (make_l1_metrics("not occluded"), "mesh_ray"),
                    (make_l1_metrics("not occluded"), "mesh_ray"),
                ],
            ) as visibility_mock,
        ):
            questions = generate_l2_object_remove(
                objects=objects,
                attachment_graph={},
                camera_pose=camera_pose,
                templates={
                    "L2_object_remove": [
                        "If {obj_a} were removed, what is the occlusion status of {obj_b}?"
                    ]
                },
                color_intrinsics=intrinsics,
                depth_image=None,
                depth_intrinsics=None,
                occlusion_backend="mesh_ray",
                ray_caster=object(),
                instance_mesh_data=object(),
            )

        self.assertEqual(questions, [])
        self.assertEqual(visibility_mock.call_count, 3)


if __name__ == "__main__":
    unittest.main()
