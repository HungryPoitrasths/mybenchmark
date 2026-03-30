import unittest

from scripts.make_viewer import (
    build_task_summary_v2,
    filter_viewer_questions,
    select_viewer_source_questions,
)


def make_object_move_question(
    *,
    qtype: str,
    scene_id: str = "scene0000_00",
    image_name: str = "000.jpg",
    attached: bool,
    text: str,
) -> dict:
    return {
        "type": qtype,
        "scene_id": scene_id,
        "image_name": image_name,
        "attachment_remapped": attached,
        "question": text,
    }


class MakeViewerTests(unittest.TestCase):
    def test_attachment_only_keeps_attachment_chain_and_attached_object_moves(self) -> None:
        questions = [
            {"type": "attachment_chain"},
            {"type": "object_move_agent", "attachment_remapped": True},
            {"type": "object_move_distance", "attachment_remapped": False},
            {"type": "object_move_object_centric", "attachment_remapped": True},
            {"type": "object_move_allocentric", "attachment_remapped": True},
            {"type": "viewpoint_move"},
            {"type": "attachment_type"},
        ]

        filtered = select_viewer_source_questions(questions, attachment_only=True)

        self.assertEqual(
            [q["type"] for q in filtered],
            [
                "attachment_chain",
                "object_move_agent",
                "object_move_object_centric",
                "object_move_allocentric",
            ],
        )

    def test_qtypes_filter_still_works_without_attachment_only(self) -> None:
        questions = [
            {"type": "object_move_agent", "attachment_remapped": True},
            {"type": "viewpoint_move"},
            {"type": "attachment_type"},
        ]

        filtered = select_viewer_source_questions(
            questions,
            requested_qtypes={"viewpoint_move"},
        )

        self.assertEqual(filtered, [{"type": "viewpoint_move"}])

    def test_qtypes_filter_accepts_canonical_object_rotate_label_for_legacy_input(self) -> None:
        questions = [
            {"type": "object_move_object_centric", "attachment_remapped": True},
            {"type": "object_move_agent", "attachment_remapped": True},
        ]

        filtered = select_viewer_source_questions(
            questions,
            requested_qtypes={"object_rotate_object_centric"},
        )

        self.assertEqual(filtered, [{"type": "object_move_object_centric", "attachment_remapped": True}])

    def test_viewer_attachment_filter_drops_unattached_when_no_attached_globally(self) -> None:
        questions = [
            make_object_move_question(qtype="object_move_distance", attached=False, text="keep 1"),
            make_object_move_question(qtype="object_move_distance", attached=False, text="drop 2"),
            make_object_move_question(qtype="object_move_distance", attached=False, text="drop 3"),
            make_object_move_question(
                qtype="object_move_distance",
                scene_id="scene0000_00",
                image_name="001.jpg",
                attached=False,
                text="keep other frame",
            ),
        ]

        filtered = filter_viewer_questions(questions)

        self.assertEqual(filtered, [])

    def test_viewer_attachment_filter_keeps_two_to_one_ratio_per_frame(self) -> None:
        questions = [
            make_object_move_question(qtype="object_rotate_object_centric", attached=True, text="attached"),
            make_object_move_question(qtype="object_rotate_object_centric", attached=False, text="free 1"),
            make_object_move_question(qtype="object_rotate_object_centric", attached=False, text="free 2"),
            make_object_move_question(qtype="object_rotate_object_centric", attached=False, text="drop 3"),
            make_object_move_question(qtype="object_rotate_object_centric", attached=False, text="drop 4"),
        ]

        filtered = filter_viewer_questions(questions)

        self.assertEqual(
            [q["question"] for q in filtered],
            ["attached", "free 1", "free 2"],
        )

    def test_viewer_attachment_filter_caps_unattached_globally_per_qtype(self) -> None:
        questions = [
            make_object_move_question(qtype="object_move_agent", attached=True, text="attached"),
            make_object_move_question(qtype="object_move_agent", attached=False, text="free 1"),
            make_object_move_question(qtype="object_move_agent", attached=False, text="free 2"),
            make_object_move_question(
                qtype="object_move_agent",
                image_name="001.jpg",
                attached=False,
                text="drop 3",
            ),
            make_object_move_question(
                qtype="object_move_agent",
                image_name="002.jpg",
                attached=False,
                text="drop 4",
            ),
        ]

        filtered = filter_viewer_questions(questions)

        self.assertEqual(
            [q["question"] for q in filtered],
            ["attached", "free 1", "free 2"],
        )

    def test_task_summary_v2_canonicalizes_legacy_object_centric_type(self) -> None:
        questions = [
            {"type": "object_move_object_centric", "attachment_remapped": True},
        ]

        summary = build_task_summary_v2(questions)

        self.assertIn(
            "L2_object_rotate_object_centric: with_attachment=1, without_attachment=0",
            summary,
        )
        self.assertIn(
            "L2_object_move_all: with_attachment=1, without_attachment=0",
            summary,
        )
        self.assertNotIn("Viewer Slice", summary)
        self.assertNotIn("raw=", summary)
        self.assertNotIn("shown=", summary)
        self.assertNotIn("hidden=", summary)
        self.assertNotIn("total=", summary)
        self.assertNotIn("with_attachment_changed=", summary)
        self.assertNotIn("with_attachment_unchanged=", summary)
        self.assertNotIn("object_move_object_centric", summary)

    def test_task_summary_v2_does_not_list_object_rotate_in_other_types(self) -> None:
        questions = [
            {"type": "object_rotate_object_centric", "attachment_remapped": True},
            {"type": "custom_unknown"},
            {"type": "custom_unknown"},
        ]

        summary = build_task_summary_v2(questions)

        self.assertIn(
            "L2_object_rotate_object_centric: with_attachment=1, without_attachment=0",
            summary,
        )
        self.assertIn("Other Types:</strong> custom_unknown=2", summary)
        self.assertNotIn("Other Types:</strong> object_rotate_object_centric=1", summary)


if __name__ == "__main__":
    unittest.main()
