import unittest
from unittest.mock import patch

import numpy as np

from src.qa_generator import (
    _ensure_question_mentions,
    _cap_question_groups,
    generate_all_questions,
    generate_l1_occlusion_questions,
    generate_l2_object_move,
    generate_l2_object_move_allocentric,
    generate_l2_object_rotate_object_centric,
)
from src.utils.colmap_loader import CameraPose


def make_camera_pose() -> CameraPose:
    return CameraPose(
        image_name="000000.jpg",
        rotation=np.eye(3, dtype=np.float64),
        translation=np.zeros(3, dtype=np.float64),
    )


def make_object(obj_id: int, label: str) -> dict:
    return {
        "id": obj_id,
        "label": label,
        "center": [float(obj_id), 0.0, 1.0],
        "bbox_min": [float(obj_id), 0.0, 0.5],
        "bbox_max": [float(obj_id) + 0.2, 0.2, 1.5],
    }


def make_l2_object_move_question(
    qtype: str,
    *,
    attached: bool,
    text: str,
) -> dict:
    return {
        "level": "L2",
        "type": qtype,
        "question": text,
        "options": ["A", "B", "C", "D"],
        "answer": "A",
        "attachment_remapped": attached,
    }


class QaGeneratorReferabilityTests(unittest.TestCase):
    def test_ensure_question_mentions_includes_obj_c_id(self) -> None:
        question = {
            "question": "If the table moves, where is the cup relative to the lamp?",
            "obj_a_id": 1,
            "obj_a_label": "cup",
            "obj_b_id": 2,
            "obj_b_label": "table",
            "obj_c_id": 3,
            "obj_c_label": "lamp",
            "mentioned_objects": [
                {"role": "query_object", "label": "cup", "obj_id": 1},
                {"role": "moved_object", "label": "table", "obj_id": 2},
            ],
        }

        normalized = _ensure_question_mentions(
            question,
            {
                1: make_object(1, "cup"),
                2: make_object(2, "table"),
                3: make_object(3, "lamp"),
            },
        )

        mentions = normalized["mentioned_objects"]
        self.assertTrue(
            any(
                mention.get("role") == "obj_c"
                and mention.get("obj_id") == 3
                and mention.get("label") == "lamp"
                for mention in mentions
            )
        )

    def test_cap_question_groups_keeps_attached_questions_even_when_group_exceeds_cap(self) -> None:
        attached_questions = [
            make_l2_object_move_question("object_move_agent", attached=True, text="attached 1"),
            make_l2_object_move_question("object_move_agent", attached=True, text="attached 2"),
        ]
        unattached_questions = [
            make_l2_object_move_question("object_move_agent", attached=False, text="free 1"),
            make_l2_object_move_question("object_move_agent", attached=False, text="free 2"),
            make_l2_object_move_question("object_move_agent", attached=False, text="free 3"),
        ]

        kept = _cap_question_groups(
            {1: attached_questions + unattached_questions},
            max_per_group=3,
        )

        kept_text = {q["question"] for q in kept}
        self.assertIn("attached 1", kept_text)
        self.assertIn("attached 2", kept_text)
        self.assertEqual(sum(1 for q in kept if q.get("attachment_remapped", False)), 2)

    def test_l3_support_chain_only_sees_fully_referable_subgraph(self) -> None:
        captured: dict[str, object] = {}

        def capture_l3(objects, attachment_graph, attached_by, camera_pose, templates):
            captured["object_ids"] = [int(o["id"]) for o in objects]
            captured["attachment_graph"] = attachment_graph
            captured["attached_by"] = attached_by
            return []

        objects = [
            make_object(1, "table"),
            make_object(2, "box"),
            make_object(3, "cup"),
        ]

        with (
            patch("src.qa_generator.compute_all_relations", return_value=[]),
            patch("src.qa_generator.generate_l1_occlusion_questions", return_value=[]),
            patch("src.qa_generator.generate_l1_direction_object_centric", return_value=[]),
            patch("src.qa_generator.generate_l1_direction_allocentric", return_value=[]),
            patch("src.qa_generator.generate_l2_object_move", return_value=[]),
            patch("src.qa_generator.generate_l2_viewpoint_move", return_value=[]),
            patch("src.qa_generator.generate_l2_object_remove", return_value=[]),
            patch("src.qa_generator.generate_l2_object_rotate_object_centric", return_value=[]),
            patch("src.qa_generator.generate_l2_object_move_allocentric", return_value=[]),
            patch("src.qa_generator.generate_l3_attachment_chain", side_effect=capture_l3),
            patch("src.qa_generator.generate_l3_coordinate_rotation", return_value=[]),
            patch("src.qa_generator.generate_l3_coordinate_rotation_object_centric", return_value=[]),
            patch("src.qa_generator.generate_l3_coordinate_rotation_allocentric", return_value=[]),
        ):
            generate_all_questions(
                objects=objects,
                attachment_graph={1: [2], 2: [3]},
                attached_by={2: 1, 3: 2},
                support_chain_graph={1: [2], 2: [3]},
                support_chain_by={2: 1, 3: 2},
                camera_pose=make_camera_pose(),
                templates={},
                visible_object_ids=[1, 2, 3],
                referable_object_ids=[1, 3],
                label_counts=None,
                attachment_edges=[
                    {"parent_id": 1, "child_id": 2, "type": "supported_by"},
                    {"parent_id": 2, "child_id": 3, "type": "supported_by"},
                ],
            )

        self.assertEqual(captured["object_ids"], [1, 3])
        self.assertEqual(captured["attachment_graph"], {})
        self.assertEqual(captured["attached_by"], {})

    def test_l2_object_move_attachment_counts_are_left_unchanged_during_generation(self) -> None:
        objects = [
            make_object(1, "table"),
            make_object(2, "box"),
            make_object(3, "cup"),
        ]

        l2_move_questions = [
            make_l2_object_move_question("object_move_agent", attached=True, text="agent attached 1"),
            make_l2_object_move_question("object_move_agent", attached=True, text="agent attached 2"),
            make_l2_object_move_question("object_move_agent", attached=False, text="agent free 1"),
            make_l2_object_move_question("object_move_agent", attached=False, text="agent free 2"),
            make_l2_object_move_question("object_move_agent", attached=False, text="agent free 3"),
            make_l2_object_move_question("object_move_distance", attached=True, text="distance attached 1"),
            make_l2_object_move_question("object_move_distance", attached=False, text="distance free 1"),
            make_l2_object_move_question("object_move_distance", attached=False, text="distance free 2"),
        ]
        l2_object_centric_questions = [
            make_l2_object_move_question("object_rotate_object_centric", attached=True, text="oc attached 1"),
            make_l2_object_move_question("object_rotate_object_centric", attached=False, text="oc free 1"),
            make_l2_object_move_question("object_rotate_object_centric", attached=False, text="oc free 2"),
        ]
        l2_allocentric_questions = [
            make_l2_object_move_question("object_move_allocentric", attached=False, text="allo free 1"),
            make_l2_object_move_question("object_move_allocentric", attached=False, text="allo free 2"),
            make_l2_object_move_question("object_move_allocentric", attached=False, text="allo free 3"),
            make_l2_object_move_question("object_move_allocentric", attached=False, text="allo free 4"),
        ]
        viewpoint_questions = [
            {
                "level": "L2",
                "type": "viewpoint_move",
                "question": "viewpoint",
                "options": ["A", "B", "C", "D"],
                "answer": "A",
            },
        ]

        with (
            patch("src.qa_generator.compute_all_relations", return_value=[]),
            patch("src.qa_generator.generate_l1_occlusion_questions", return_value=[]),
            patch("src.qa_generator.generate_l1_direction_object_centric", return_value=[]),
            patch("src.qa_generator.generate_l1_direction_allocentric", return_value=[]),
            patch("src.qa_generator.generate_l2_object_move", return_value=l2_move_questions),
            patch("src.qa_generator.generate_l2_viewpoint_move", return_value=viewpoint_questions),
            patch("src.qa_generator.generate_l2_object_remove", return_value=[]),
            patch("src.qa_generator.generate_l2_object_rotate_object_centric", return_value=l2_object_centric_questions),
            patch("src.qa_generator.generate_l2_object_move_allocentric", return_value=l2_allocentric_questions),
            patch("src.qa_generator.generate_l3_attachment_chain", return_value=[]),
            patch("src.qa_generator.generate_l3_coordinate_rotation", return_value=[]),
            patch("src.qa_generator.generate_l3_coordinate_rotation_object_centric", return_value=[]),
            patch("src.qa_generator.generate_l3_coordinate_rotation_allocentric", return_value=[]),
            patch("src.qa_generator._ensure_question_mentions", side_effect=lambda q, *_: q),
            patch("src.qa_generator._enforce_stable_facing_references", side_effect=lambda qs, *_: qs),
        ):
            questions = generate_all_questions(
                objects=objects,
                attachment_graph={},
                attached_by={},
                support_chain_graph={},
                support_chain_by={},
                camera_pose=make_camera_pose(),
                templates={},
                visible_object_ids=[1, 2, 3],
                referable_object_ids=[1, 2, 3],
                label_counts=None,
                attachment_edges=[],
            )

        counts: dict[str, tuple[int, int]] = {}
        for q in questions:
            qtype = q.get("type")
            if not (
                str(qtype).startswith("object_move_")
                or str(qtype) == "object_rotate_object_centric"
            ):
                continue
            attached, unattached = counts.get(qtype, (0, 0))
            if q.get("attachment_remapped", False):
                attached += 1
            else:
                unattached += 1
            counts[qtype] = (attached, unattached)

        self.assertEqual(counts["object_move_agent"], (2, 3))
        self.assertEqual(counts["object_move_distance"], (1, 2))
        self.assertEqual(counts["object_rotate_object_centric"], (1, 2))
        self.assertEqual(counts.get("object_move_allocentric", (0, 0)), (0, 4))
        self.assertEqual(sum(1 for q in questions if q.get("type") == "viewpoint_move"), 1)

    def test_generate_all_questions_drops_questions_with_nonreferable_mentions(self) -> None:
        objects = [
            make_object(1, "cup"),
            make_object(2, "table"),
            make_object(3, "lamp"),
        ]

        leaked_question = {
            "level": "L2",
            "type": "object_move_agent",
            "question": "If the table moves, where is the cup relative to the lamp?",
            "options": ["A", "B", "C", "D"],
            "answer": "A",
            "correct_value": "left",
            "moved_obj_id": 2,
            "moved_obj_label": "table",
            "mentioned_objects": [
                {"role": "moved_object", "obj_id": 2, "label": "table"},
                {"role": "query_object", "obj_id": 1, "label": "cup"},
                {"role": "reference_object", "obj_id": 3, "label": "lamp"},
            ],
        }

        with (
            patch("src.qa_generator.compute_all_relations", return_value=[]),
            patch("src.qa_generator.generate_l1_occlusion_questions", return_value=[]),
            patch("src.qa_generator.generate_l1_direction_object_centric", return_value=[]),
            patch("src.qa_generator.generate_l1_direction_allocentric", return_value=[]),
            patch("src.qa_generator.generate_l2_object_move", return_value=[leaked_question]),
            patch("src.qa_generator.generate_l2_viewpoint_move", return_value=[]),
            patch("src.qa_generator.generate_l2_object_remove", return_value=[]),
            patch("src.qa_generator.generate_l2_object_rotate_object_centric", return_value=[]),
            patch("src.qa_generator.generate_l2_object_move_allocentric", return_value=[]),
            patch("src.qa_generator.generate_l3_attachment_chain", return_value=[]),
            patch("src.qa_generator.generate_l3_coordinate_rotation", return_value=[]),
            patch("src.qa_generator.generate_l3_coordinate_rotation_object_centric", return_value=[]),
            patch("src.qa_generator.generate_l3_coordinate_rotation_allocentric", return_value=[]),
        ):
            questions = generate_all_questions(
                objects=objects,
                attachment_graph={2: [1]},
                attached_by={1: 2},
                support_chain_graph={},
                support_chain_by={},
                camera_pose=make_camera_pose(),
                templates={},
                visible_object_ids=[1, 2, 3],
                referable_object_ids=[1, 3],
                attachment_edges=[
                    {"parent_id": 2, "child_id": 1, "type": "supported_by"},
                ],
            )

        self.assertEqual(questions, [])

    def test_generate_all_questions_drops_questions_via_obj_c_fallback(self) -> None:
        objects = [
            make_object(1, "cup"),
            make_object(2, "table"),
            make_object(3, "lamp"),
        ]

        leaked_question = {
            "level": "L2",
            "type": "object_move_agent",
            "question": "If the table moves, where is the cup relative to the lamp?",
            "options": ["A", "B", "C", "D"],
            "answer": "A",
            "correct_value": "left",
            "moved_obj_id": 2,
            "moved_obj_label": "table",
            "query_obj_id": 1,
            "query_obj_label": "cup",
            "obj_c_id": 3,
            "obj_c_label": "lamp",
            "mentioned_objects": [
                {"role": "moved_object", "obj_id": 2, "label": "table"},
                {"role": "query_object", "obj_id": 1, "label": "cup"},
            ],
        }

        with (
            patch("src.qa_generator.compute_all_relations", return_value=[]),
            patch("src.qa_generator.generate_l1_occlusion_questions", return_value=[]),
            patch("src.qa_generator.generate_l1_direction_object_centric", return_value=[]),
            patch("src.qa_generator.generate_l1_direction_allocentric", return_value=[]),
            patch("src.qa_generator.generate_l2_object_move", return_value=[leaked_question]),
            patch("src.qa_generator.generate_l2_viewpoint_move", return_value=[]),
            patch("src.qa_generator.generate_l2_object_remove", return_value=[]),
            patch("src.qa_generator.generate_l2_object_rotate_object_centric", return_value=[]),
            patch("src.qa_generator.generate_l2_object_move_allocentric", return_value=[]),
            patch("src.qa_generator.generate_l3_attachment_chain", return_value=[]),
            patch("src.qa_generator.generate_l3_coordinate_rotation", return_value=[]),
            patch("src.qa_generator.generate_l3_coordinate_rotation_object_centric", return_value=[]),
            patch("src.qa_generator.generate_l3_coordinate_rotation_allocentric", return_value=[]),
        ):
            questions = generate_all_questions(
                objects=objects,
                attachment_graph={2: [1]},
                attached_by={1: 2},
                support_chain_graph={},
                support_chain_by={},
                camera_pose=make_camera_pose(),
                templates={},
                visible_object_ids=[1, 2, 3],
                referable_object_ids=[1, 2],
                label_statuses={"cup": "unique", "table": "unique", "lamp": "absent"},
                label_to_object_ids={"cup": [1], "table": [2], "lamp": [3]},
                attachment_edges=[
                    {"parent_id": 2, "child_id": 1, "type": "supported_by"},
                ],
            )

        self.assertEqual(questions, [])

    def test_generate_all_questions_drops_same_object_multi_role_from_fallback_fields(self) -> None:
        objects = [make_object(1, "chair")]

        leaked_question = {
            "level": "L1",
            "type": "direction_agent",
            "question": "Where is the chair relative to itself?",
            "options": ["left", "right", "front", "back"],
            "answer": "A",
            "correct_value": "left",
            "obj_a_id": 1,
            "obj_a_label": "chair",
            "obj_b_id": 1,
            "obj_b_label": "chair",
        }

        with (
            patch("src.qa_generator.compute_all_relations", return_value=[]),
            patch("src.qa_generator.generate_l1_occlusion_questions", return_value=[]),
            patch("src.qa_generator.generate_l1_direction_object_centric", return_value=[]),
            patch("src.qa_generator.generate_l1_direction_allocentric", return_value=[]),
            patch("src.qa_generator.generate_l2_object_move", return_value=[]),
            patch("src.qa_generator.generate_l2_viewpoint_move", return_value=[]),
            patch("src.qa_generator.generate_l2_object_remove", return_value=[]),
            patch("src.qa_generator.generate_l2_object_rotate_object_centric", return_value=[]),
            patch("src.qa_generator.generate_l2_object_move_allocentric", return_value=[]),
            patch("src.qa_generator.generate_l3_attachment_chain", return_value=[]),
            patch("src.qa_generator.generate_l3_coordinate_rotation", return_value=[]),
            patch("src.qa_generator.generate_l3_coordinate_rotation_object_centric", return_value=[]),
            patch("src.qa_generator.generate_l3_coordinate_rotation_allocentric", return_value=[leaked_question]),
        ):
            questions = generate_all_questions(
                objects=objects,
                attachment_graph={},
                attached_by={},
                support_chain_graph={},
                support_chain_by={},
                camera_pose=make_camera_pose(),
                templates={},
                visible_object_ids=[1],
                referable_object_ids=[1],
                label_statuses={"chair": "unique"},
                label_to_object_ids={"chair": [1]},
                attachment_edges=[],
            )

        self.assertEqual(questions, [])

    def test_generate_all_questions_keeps_static_occlusion_not_visible(self) -> None:
        objects = [make_object(3, "lamp")]
        occlusion_question = {
            "level": "L1",
            "type": "occlusion",
            "question": "Is the lamp visible?",
            "options": ["fully visible", "partially occluded", "not visible"],
            "answer": "C",
            "correct_value": "not visible",
            "obj_a_id": None,
            "obj_a_label": "lamp",
            "mentioned_objects": [
                {"role": "target", "label": "lamp", "obj_id": None},
            ],
        }

        with (
            patch("src.qa_generator.compute_all_relations", return_value=[]),
            patch("src.qa_generator.generate_l1_occlusion_questions", return_value=[occlusion_question]),
            patch("src.qa_generator.generate_l1_direction_object_centric", return_value=[]),
            patch("src.qa_generator.generate_l1_direction_allocentric", return_value=[]),
            patch("src.qa_generator.generate_l2_object_move", return_value=[]),
            patch("src.qa_generator.generate_l2_viewpoint_move", return_value=[]),
            patch("src.qa_generator.generate_l2_object_remove", return_value=[]),
            patch("src.qa_generator.generate_l2_object_rotate_object_centric", return_value=[]),
            patch("src.qa_generator.generate_l2_object_move_allocentric", return_value=[]),
            patch("src.qa_generator.generate_l3_attachment_chain", return_value=[]),
            patch("src.qa_generator.generate_l3_coordinate_rotation", return_value=[]),
            patch("src.qa_generator.generate_l3_coordinate_rotation_object_centric", return_value=[]),
            patch("src.qa_generator.generate_l3_coordinate_rotation_allocentric", return_value=[]),
        ):
            questions = generate_all_questions(
                objects=objects,
                attachment_graph={},
                attached_by={},
                support_chain_graph={},
                support_chain_by={},
                camera_pose=make_camera_pose(),
                templates={},
                visible_object_ids=[3],
                referable_object_ids=[],
                occlusion_eligible_object_ids=[],
                label_statuses={"lamp": "absent"},
                label_to_object_ids={"lamp": [3]},
                attachment_edges=[],
            )

        self.assertEqual(len(questions), 1)
        self.assertEqual(questions[0]["type"], "occlusion")
        self.assertEqual(questions[0]["correct_value"], "not visible")
        self.assertEqual(
            questions[0]["question_referability_audit"]["decision"],
            "pass",
        )
        self.assertTrue(
            questions[0]["question_referability_audit"]["mentioned_objects"][0]["exempt"]
        )

    def test_generate_all_questions_drops_static_occlusion_target_below_in_frame_threshold(self) -> None:
        objects = [make_object(3, "lamp")]
        occlusion_question = {
            "level": "L1",
            "type": "occlusion",
            "question": "Is the lamp occluded?",
            "options": ["not occluded", "occluded", "not visible"],
            "answer": "B",
            "correct_value": "occluded",
            "obj_a_id": 3,
            "obj_a_label": "lamp",
            "mentioned_objects": [
                {"role": "target", "label": "lamp", "obj_id": 3},
            ],
        }

        with (
            patch("src.qa_generator.compute_all_relations", return_value=[]),
            patch("src.qa_generator.generate_l1_occlusion_questions", return_value=[occlusion_question]),
            patch("src.qa_generator.generate_l1_direction_object_centric", return_value=[]),
            patch("src.qa_generator.generate_l1_direction_allocentric", return_value=[]),
            patch("src.qa_generator.generate_l2_object_move", return_value=[]),
            patch("src.qa_generator.generate_l2_viewpoint_move", return_value=[]),
            patch("src.qa_generator.generate_l2_object_remove", return_value=[]),
            patch("src.qa_generator.generate_l2_object_rotate_object_centric", return_value=[]),
            patch("src.qa_generator.generate_l2_object_move_allocentric", return_value=[]),
            patch("src.qa_generator.generate_l3_attachment_chain", return_value=[]),
            patch("src.qa_generator.generate_l3_coordinate_rotation", return_value=[]),
            patch("src.qa_generator.generate_l3_coordinate_rotation_object_centric", return_value=[]),
            patch("src.qa_generator.generate_l3_coordinate_rotation_allocentric", return_value=[]),
        ):
            questions = generate_all_questions(
                objects=objects,
                attachment_graph={},
                attached_by={},
                support_chain_graph={},
                support_chain_by={},
                camera_pose=make_camera_pose(),
                templates={},
                visible_object_ids=[3],
                referable_object_ids=[3],
                occlusion_eligible_object_ids=[],
                label_statuses={"lamp": "unique"},
                label_to_object_ids={"lamp": [3]},
                attachment_edges=[],
            )

        self.assertEqual(questions, [])

    def test_generate_all_questions_does_not_flag_explicit_role_with_matching_legacy_alias(self) -> None:
        objects = [make_object(1, "cup")]
        question = {
            "level": "L2",
            "type": "object_move_agent",
            "question": "If the table moves, where is the cup?",
            "options": ["A", "B", "C", "D"],
            "answer": "A",
            "correct_value": "left",
            "query_obj_id": 1,
            "query_obj_label": "cup",
            "mentioned_objects": [
                {"role": "query_object", "obj_id": 1, "label": "cup"},
            ],
        }

        with (
            patch("src.qa_generator.compute_all_relations", return_value=[]),
            patch("src.qa_generator.generate_l1_occlusion_questions", return_value=[]),
            patch("src.qa_generator.generate_l1_direction_object_centric", return_value=[]),
            patch("src.qa_generator.generate_l1_direction_allocentric", return_value=[]),
            patch("src.qa_generator.generate_l2_object_move", return_value=[question]),
            patch("src.qa_generator.generate_l2_viewpoint_move", return_value=[]),
            patch("src.qa_generator.generate_l2_object_remove", return_value=[]),
            patch("src.qa_generator.generate_l2_object_rotate_object_centric", return_value=[]),
            patch("src.qa_generator.generate_l2_object_move_allocentric", return_value=[]),
            patch("src.qa_generator.generate_l3_attachment_chain", return_value=[]),
            patch("src.qa_generator.generate_l3_coordinate_rotation", return_value=[]),
            patch("src.qa_generator.generate_l3_coordinate_rotation_object_centric", return_value=[]),
            patch("src.qa_generator.generate_l3_coordinate_rotation_allocentric", return_value=[]),
        ):
            questions = generate_all_questions(
                objects=objects,
                attachment_graph={},
                attached_by={},
                support_chain_graph={},
                support_chain_by={},
                camera_pose=make_camera_pose(),
                templates={},
                visible_object_ids=[1],
                referable_object_ids=[1],
                label_statuses={"cup": "unique"},
                label_to_object_ids={"cup": [1]},
                attachment_edges=[],
            )

        self.assertEqual(len(questions), 1)
        audit = questions[0]["question_referability_audit"]
        self.assertEqual(audit["decision"], "pass")
        self.assertEqual(audit["reason_codes"], [])
        self.assertEqual(
            audit["mentioned_objects"][0]["explicit_roles"],
            ["query_object"],
        )
        self.assertEqual(
            audit["mentioned_objects"][0]["fallback_roles"],
            ["query_obj"],
        )

    def test_generate_all_questions_drops_object_move_occlusion_nonreferable_target(self) -> None:
        objects = [
            make_object(1, "cup"),
            make_object(2, "table"),
            make_object(3, "lamp"),
        ]
        leaked_question = {
            "level": "L2",
            "type": "object_move_occlusion",
            "question": "If the table moves, is the lamp visible from the camera?",
            "options": ["fully visible", "partially occluded", "not visible", "unknown"],
            "answer": "C",
            "correct_value": "not visible",
            "moved_obj_id": 2,
            "moved_obj_label": "table",
            "target_obj_id": 3,
            "target_obj_label": "lamp",
            "mentioned_objects": [
                {"role": "moved_object", "obj_id": 2, "label": "table"},
                {"role": "target_object", "obj_id": 3, "label": "lamp"},
            ],
        }

        with (
            patch("src.qa_generator.compute_all_relations", return_value=[]),
            patch("src.qa_generator.generate_l1_occlusion_questions", return_value=[]),
            patch("src.qa_generator.generate_l1_direction_object_centric", return_value=[]),
            patch("src.qa_generator.generate_l1_direction_allocentric", return_value=[]),
            patch("src.qa_generator.generate_l2_object_move", return_value=[leaked_question]),
            patch("src.qa_generator.generate_l2_viewpoint_move", return_value=[]),
            patch("src.qa_generator.generate_l2_object_remove", return_value=[]),
            patch("src.qa_generator.generate_l2_object_rotate_object_centric", return_value=[]),
            patch("src.qa_generator.generate_l2_object_move_allocentric", return_value=[]),
            patch("src.qa_generator.generate_l3_attachment_chain", return_value=[]),
            patch("src.qa_generator.generate_l3_coordinate_rotation", return_value=[]),
            patch("src.qa_generator.generate_l3_coordinate_rotation_object_centric", return_value=[]),
            patch("src.qa_generator.generate_l3_coordinate_rotation_allocentric", return_value=[]),
        ):
            questions = generate_all_questions(
                objects=objects,
                attachment_graph={2: [1]},
                attached_by={1: 2},
                support_chain_graph={},
                support_chain_by={},
                camera_pose=make_camera_pose(),
                templates={},
                visible_object_ids=[1, 2, 3],
                referable_object_ids=[1, 2],
                label_statuses={"cup": "unique", "table": "unique", "lamp": "absent"},
                label_to_object_ids={"cup": [1], "table": [2], "lamp": [3]},
                attachment_edges=[
                    {"parent_id": 2, "child_id": 1, "type": "supported_by"},
                ],
            )

        self.assertEqual(questions, [])

    def test_generate_all_questions_drops_object_move_occlusion_when_mention_not_in_frame_eligible(self) -> None:
        objects = [
            make_object(1, "cup"),
            make_object(2, "table"),
            make_object(3, "lamp"),
        ]
        leaked_question = {
            "level": "L2",
            "type": "object_move_occlusion",
            "question": "If the table moves, is the lamp visible from the camera?",
            "options": ["not occluded", "occluded", "not visible"],
            "answer": "B",
            "correct_value": "occluded",
            "moved_obj_id": 2,
            "moved_obj_label": "table",
            "target_obj_id": 3,
            "target_obj_label": "lamp",
            "mentioned_objects": [
                {"role": "moved_object", "obj_id": 2, "label": "table"},
                {"role": "target_object", "obj_id": 3, "label": "lamp"},
            ],
        }

        with (
            patch("src.qa_generator.compute_all_relations", return_value=[]),
            patch("src.qa_generator.generate_l1_occlusion_questions", return_value=[]),
            patch("src.qa_generator.generate_l1_direction_object_centric", return_value=[]),
            patch("src.qa_generator.generate_l1_direction_allocentric", return_value=[]),
            patch("src.qa_generator.generate_l2_object_move", return_value=[leaked_question]),
            patch("src.qa_generator.generate_l2_viewpoint_move", return_value=[]),
            patch("src.qa_generator.generate_l2_object_remove", return_value=[]),
            patch("src.qa_generator.generate_l2_object_rotate_object_centric", return_value=[]),
            patch("src.qa_generator.generate_l2_object_move_allocentric", return_value=[]),
            patch("src.qa_generator.generate_l3_attachment_chain", return_value=[]),
            patch("src.qa_generator.generate_l3_coordinate_rotation", return_value=[]),
            patch("src.qa_generator.generate_l3_coordinate_rotation_object_centric", return_value=[]),
            patch("src.qa_generator.generate_l3_coordinate_rotation_allocentric", return_value=[]),
        ):
            questions = generate_all_questions(
                objects=objects,
                attachment_graph={},
                attached_by={},
                support_chain_graph={},
                support_chain_by={},
                camera_pose=make_camera_pose(),
                templates={},
                visible_object_ids=[1, 2, 3],
                referable_object_ids=[2, 3],
                occlusion_eligible_object_ids=[3],
                label_statuses={"table": "unique", "lamp": "unique"},
                label_to_object_ids={"table": [2], "lamp": [3]},
                attachment_edges=[],
            )

        self.assertEqual(questions, [])

    def test_generate_all_questions_drops_viewpoint_move_target_below_in_frame_threshold(self) -> None:
        objects = [make_object(1, "lamp")]
        viewpoint_question = {
            "level": "L2",
            "type": "viewpoint_move",
            "question": "If the camera moves right, what is the occlusion status of the lamp?",
            "options": ["not occluded", "occluded", "not visible"],
            "answer": "A",
            "correct_value": "not occluded",
            "obj_a_id": 1,
            "obj_a_label": "lamp",
            "mentioned_objects": [
                {"role": "target", "label": "lamp", "obj_id": 1},
            ],
        }

        with (
            patch("src.qa_generator.compute_all_relations", return_value=[]),
            patch("src.qa_generator.generate_l1_occlusion_questions", return_value=[]),
            patch("src.qa_generator.generate_l1_direction_object_centric", return_value=[]),
            patch("src.qa_generator.generate_l1_direction_allocentric", return_value=[]),
            patch("src.qa_generator.generate_l2_object_move", return_value=[]),
            patch("src.qa_generator.generate_l2_viewpoint_move", return_value=[viewpoint_question]),
            patch("src.qa_generator.generate_l2_object_remove", return_value=[]),
            patch("src.qa_generator.generate_l2_object_rotate_object_centric", return_value=[]),
            patch("src.qa_generator.generate_l2_object_move_allocentric", return_value=[]),
            patch("src.qa_generator.generate_l3_attachment_chain", return_value=[]),
            patch("src.qa_generator.generate_l3_coordinate_rotation", return_value=[]),
            patch("src.qa_generator.generate_l3_coordinate_rotation_object_centric", return_value=[]),
            patch("src.qa_generator.generate_l3_coordinate_rotation_allocentric", return_value=[]),
        ):
            questions = generate_all_questions(
                objects=objects,
                attachment_graph={},
                attached_by={},
                support_chain_graph={},
                support_chain_by={},
                camera_pose=make_camera_pose(),
                templates={},
                visible_object_ids=[1],
                referable_object_ids=[1],
                occlusion_eligible_object_ids=[],
                label_statuses={"lamp": "unique"},
                label_to_object_ids={"lamp": [1]},
                attachment_edges=[],
            )

        self.assertEqual(questions, [])

    def test_generate_all_questions_drops_object_remove_pair_below_in_frame_threshold(self) -> None:
        objects = [
            make_object(1, "lamp"),
            make_object(2, "table"),
        ]
        remove_question = {
            "level": "L2",
            "type": "object_remove",
            "question": "If the table were removed, what would be the occlusion status of the lamp?",
            "options": ["not occluded", "occluded", "not visible"],
            "answer": "A",
            "correct_value": "not occluded",
            "removed_obj_id": 2,
            "removed_obj_label": "table",
            "obj_b_id": 1,
            "obj_b_label": "lamp",
            "mentioned_objects": [
                {"role": "removed_object", "label": "table", "obj_id": 2},
                {"role": "remaining_object", "label": "lamp", "obj_id": 1},
            ],
        }

        with (
            patch("src.qa_generator.compute_all_relations", return_value=[]),
            patch("src.qa_generator.generate_l1_occlusion_questions", return_value=[]),
            patch("src.qa_generator.generate_l1_direction_object_centric", return_value=[]),
            patch("src.qa_generator.generate_l1_direction_allocentric", return_value=[]),
            patch("src.qa_generator.generate_l2_object_move", return_value=[]),
            patch("src.qa_generator.generate_l2_viewpoint_move", return_value=[]),
            patch("src.qa_generator.generate_l2_object_remove", return_value=[remove_question]),
            patch("src.qa_generator.generate_l2_object_rotate_object_centric", return_value=[]),
            patch("src.qa_generator.generate_l2_object_move_allocentric", return_value=[]),
            patch("src.qa_generator.generate_l3_attachment_chain", return_value=[]),
            patch("src.qa_generator.generate_l3_coordinate_rotation", return_value=[]),
            patch("src.qa_generator.generate_l3_coordinate_rotation_object_centric", return_value=[]),
            patch("src.qa_generator.generate_l3_coordinate_rotation_allocentric", return_value=[]),
        ):
            questions = generate_all_questions(
                objects=objects,
                attachment_graph={},
                attached_by={},
                support_chain_graph={},
                support_chain_by={},
                camera_pose=make_camera_pose(),
                templates={},
                visible_object_ids=[1, 2],
                referable_object_ids=[1, 2],
                occlusion_eligible_object_ids=[1],
                label_statuses={"lamp": "unique", "table": "unique"},
                label_to_object_ids={"lamp": [1], "table": [2]},
                attachment_edges=[],
            )

        self.assertEqual(questions, [])

    def test_l1_occlusion_skips_multiple_status_without_unique_instance(self) -> None:
        questions = generate_l1_occlusion_questions(
            objects=[make_object(1, "cup")],
            camera_pose=make_camera_pose(),
            color_intrinsics=None,
            depth_image=None,
            depth_intrinsics=None,
            occlusion_backend="depth",
            ray_caster=None,
            instance_mesh_data=None,
            templates={},
            label_statuses={"cup": "multiple"},
            referable_object_ids=[1],
        )

        self.assertEqual(questions, [])

    def test_l2_generators_skip_attachment_remapped_nonreferable_move_source(self) -> None:
        referable_child = make_object(1, "cup")
        hidden_parent = make_object(2, "table")
        attached_by = {1: 2}
        attachment_graph = {2: [1]}
        movement_objects = [referable_child, hidden_parent]
        object_map = {1: referable_child, 2: hidden_parent}

        move_questions = generate_l2_object_move(
            objects=[referable_child],
            attachment_graph=attachment_graph,
            attached_by=attached_by,
            camera_pose=make_camera_pose(),
            templates={},
            movement_objects=movement_objects,
            object_map=object_map,
        )
        rotate_questions = generate_l2_object_rotate_object_centric(
            objects=[referable_child],
            attachment_graph=attachment_graph,
            attached_by=attached_by,
            camera_pose=make_camera_pose(),
            templates={},
            movement_objects=movement_objects,
            object_map=object_map,
        )
        allocentric_questions = generate_l2_object_move_allocentric(
            objects=[referable_child],
            attachment_graph=attachment_graph,
            attached_by=attached_by,
            camera_pose=make_camera_pose(),
            templates={},
            movement_objects=movement_objects,
            object_map=object_map,
        )

        self.assertEqual(move_questions, [])
        self.assertEqual(rotate_questions, [])
        self.assertEqual(allocentric_questions, [])

    def test_attachment_remapped_rotate_questions_can_be_kept_when_answer_is_unchanged(self) -> None:
        child = make_object(1, "cup")
        parent = make_object(2, "table")
        face = make_object(3, "lamp")
        ref = make_object(4, "chair")
        objects = [child, parent, face, ref]
        rotated_objects = [make_object(1, "cup"), make_object(2, "table"), face, ref]

        with (
            patch("src.qa_generator._has_stable_object_centric_facing", return_value=True),
            patch(
                "src.qa_generator.find_meaningful_orbit_rotation",
                return_value=[{
                    "angle": 90,
                    "rotation_direction": "clockwise",
                    "signed_angle": -90,
                    "objects": rotated_objects,
                }],
            ),
            patch(
                "src.qa_generator.primary_direction_object_centric",
                return_value=("left", 0.1),
            ),
        ):
            questions = generate_l2_object_rotate_object_centric(
                objects=[child, parent, face, ref],
                attachment_graph={2: [1]},
                attached_by={1: 2},
                camera_pose=make_camera_pose(),
                templates={
                    "L2_object_rotate_object_centric": [
                        "rotate {obj_move_source}: where is {obj_ref} from {obj_query} while facing {obj_face}?"
                    ]
                },
                movement_objects=objects,
                object_map={obj["id"]: obj for obj in objects},
            )

        self.assertTrue(questions)
        question = next(q for q in questions if q.get("attachment_remapped"))
        self.assertEqual(question["type"], "object_rotate_object_centric")
        self.assertTrue(question["attachment_remapped"])
        self.assertTrue(question["relation_unchanged"])
        self.assertEqual(question["correct_value"], "left")
        self.assertEqual(question["old_correct_value"], "left")
        self.assertEqual(question["new_correct_value"], "left")
        self.assertTrue(question["has_attachment_chain"])

    def test_full_quality_pipeline_leaves_attachment_counts_untouched(self) -> None:
        from src.quality_control import full_quality_pipeline

        questions = [
            make_l2_object_move_question("object_move_agent", attached=True, text="agent attached 1"),
            make_l2_object_move_question("object_move_agent", attached=False, text="agent free 1"),
            make_l2_object_move_question("object_move_agent", attached=False, text="agent free 2"),
            make_l2_object_move_question("object_move_distance", attached=False, text="distance free 1"),
            make_l2_object_move_question("object_move_distance", attached=False, text="distance free 2"),
            make_l2_object_move_question("object_rotate_object_centric", attached=False, text="rotate free 1"),
            make_l2_object_move_question("object_rotate_object_centric", attached=False, text="rotate free 2"),
            make_l2_object_move_question("object_rotate_object_centric", attached=False, text="rotate free 3"),
            make_l2_object_move_question("object_rotate_object_centric", attached=False, text="rotate free 4"),
            {
                "level": "L2",
                "type": "viewpoint_move",
                "question": "viewpoint",
                "options": ["fully visible", "partially occluded", "not visible", "unknown"],
                "answer": "A",
                "correct_value": "fully visible",
            },
        ]
        for q in questions:
            q.setdefault("correct_value", "A")
        for idx, q in enumerate(questions):
            if str(q.get("type", "")).startswith("object_move_") or q.get("type") == "object_rotate_object_centric":
                q["moved_obj_id"] = idx + 1

        filtered = full_quality_pipeline(questions)

        counts: dict[str, tuple[int, int]] = {}
        for q in filtered:
            qtype = str(q.get("type", ""))
            if not (qtype.startswith("object_move_") or qtype == "object_rotate_object_centric"):
                continue
            attached, unattached = counts.get(qtype, (0, 0))
            if q.get("attachment_remapped", False):
                attached += 1
            else:
                unattached += 1
            counts[qtype] = (attached, unattached)

        self.assertEqual(counts.get("object_move_agent", (0, 0)), (1, 2))
        self.assertEqual(counts.get("object_move_distance", (0, 0)), (0, 2))
        self.assertEqual(counts.get("object_rotate_object_centric", (0, 0)), (0, 4))
        self.assertEqual(sum(1 for q in filtered if q.get("type") == "viewpoint_move"), 1)


if __name__ == "__main__":
    unittest.main()
