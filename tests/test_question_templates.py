import unittest
from unittest.mock import patch

import numpy as np

from src.qa_generator import (
    _default_templates,
    _direction_with_camera_hint,
    _l1_occlusion_question,
    _make_l1_occlusion_metrics,
    _normalize_template_aliases,
    generate_l2_object_remove,
    generate_l2_viewpoint_move,
)
from src.utils.colmap_loader import CameraIntrinsics, CameraPose


EXPECTED_L2_OBJECT_CENTRIC_ORBIT_TEMPLATE = (
    "Imagine you are {obj_query} and facing toward {obj_face}. "
    "If {obj_move_source} were moved along a {angle}-degree {rotation_direction} "
    "(viewed from above) orbit around the center of {obj_face}, without changing "
    "its own facing direction, from your perspective, in which direction would "
    "{obj_ref} be?"
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

    def test_viewpoint_move_skips_visible_to_occluded_transition(self) -> None:
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


if __name__ == "__main__":
    unittest.main()
