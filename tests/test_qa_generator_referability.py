import unittest
from unittest.mock import patch

import numpy as np

from src.qa_generator import generate_all_questions
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
            patch("src.qa_generator.generate_l2_object_move_object_centric", return_value=[]),
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

    def test_l2_object_move_without_attachment_is_capped_per_type(self) -> None:
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
            make_l2_object_move_question("object_move_object_centric", attached=True, text="oc attached 1"),
            make_l2_object_move_question("object_move_object_centric", attached=False, text="oc free 1"),
            make_l2_object_move_question("object_move_object_centric", attached=False, text="oc free 2"),
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
            patch("src.qa_generator.generate_l2_object_move_object_centric", return_value=l2_object_centric_questions),
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
            if not str(qtype).startswith("object_move_"):
                continue
            attached, unattached = counts.get(qtype, (0, 0))
            if q.get("attachment_remapped", False):
                attached += 1
            else:
                unattached += 1
            counts[qtype] = (attached, unattached)

        self.assertEqual(counts["object_move_agent"], (2, 2))
        self.assertEqual(counts["object_move_distance"], (1, 1))
        self.assertEqual(counts["object_move_object_centric"], (1, 1))
        self.assertEqual(counts.get("object_move_allocentric", (0, 0)), (0, 3))
        self.assertEqual(sum(1 for q in questions if q.get("type") == "viewpoint_move"), 1)


if __name__ == "__main__":
    unittest.main()
