import unittest

from scripts.make_viewer import filter_viewer_questions


class MakeViewerTests(unittest.TestCase):
    def test_attachment_only_keeps_attachment_chain_and_attached_object_moves(self) -> None:
        questions = [
            {"type": "attachment_chain"},
            {"type": "object_move_agent", "attachment_remapped": True},
            {"type": "object_move_distance", "attachment_remapped": False},
            {"type": "object_move_allocentric", "attachment_remapped": True},
            {"type": "viewpoint_move"},
            {"type": "attachment_type"},
        ]

        filtered = filter_viewer_questions(questions, attachment_only=True)

        self.assertEqual(
            [q["type"] for q in filtered],
            ["attachment_chain", "object_move_agent", "object_move_allocentric"],
        )

    def test_qtypes_filter_still_works_without_attachment_only(self) -> None:
        questions = [
            {"type": "object_move_agent", "attachment_remapped": True},
            {"type": "viewpoint_move"},
            {"type": "attachment_type"},
        ]

        filtered = filter_viewer_questions(
            questions,
            requested_qtypes={"viewpoint_move"},
            attachment_only=False,
        )

        self.assertEqual(filtered, [{"type": "viewpoint_move"}])


if __name__ == "__main__":
    unittest.main()
