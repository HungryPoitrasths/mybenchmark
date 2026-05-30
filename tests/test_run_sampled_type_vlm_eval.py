import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.extract_l3_attachment_chain_multiselect import (
    MULTI_SELECT_NOTE,
    convert_attachment_chain_question,
)
from scripts.run_sampled_type_vlm_eval import (
    ImageResolution,
    _option_html,
    build_prompt,
    load_questions,
    parse_answers,
    result_from_question,
)


class RunSampledTypeVlmEvalMultiselectTests(unittest.TestCase):
    def test_convert_attachment_chain_question_removes_both_option(self) -> None:
        question = convert_attachment_chain_question(
            {
                "type": "attachment_chain",
                "dataset": "scannetpp",
                "scene_id": "scene0000_00",
                "image_name": "000.jpg",
                "question": "Which objects move?",
                "options": [
                    "the oven",
                    "the clothes dryer",
                    "Both the oven and the cereal box",
                    "the cereal box",
                ],
                "correct_value": "Both the oven and the cereal box",
                "gt_answer": "C",
                "prediction": "C",
                "raw_response": "Answer: C",
                "correct": True,
            }
        )

        self.assertEqual(
            question["options"],
            ["the oven", "the clothes dryer", "the cereal box"],
        )
        self.assertTrue(question["multi_select"])
        self.assertEqual(question["answer"], ["A", "C"])
        self.assertEqual(question["_dataset"], "scannetpp")
        self.assertEqual(question["correct_values"], ["the oven", "the cereal box"])
        self.assertEqual(question["correct_value"], "the oven; the cereal box")
        self.assertIn(MULTI_SELECT_NOTE, question["question"])
        self.assertNotIn("raw_response", question)
        self.assertNotIn("prediction", question)

    def test_build_prompt_uses_multi_select_suffix(self) -> None:
        prompt = build_prompt(
            {
                "question": "Which objects move?",
                "options": ["the oven", "the cereal box"],
                "multi_select": True,
                "answer": ["A", "B"],
            }
        )

        self.assertIn("Choose all correct option letters", prompt)
        self.assertIn("Answer: <letter(s)>", prompt)

    def test_parse_answers_accepts_common_multi_select_formats(self) -> None:
        self.assertEqual(parse_answers("Answer: A,C\nReasoning: ...", "ABC"), ["A", "C"])
        self.assertEqual(parse_answers("Answer: C and A", "ABC"), ["A", "C"])
        self.assertEqual(parse_answers("AC", "ABC"), ["A", "C"])

    def test_result_from_question_scores_multi_select_as_set(self) -> None:
        question = {
            "question_uid": "q1",
            "type": "attachment_chain",
            "question": "Which objects move?",
            "options": ["the oven", "the clothes dryer", "the cereal box"],
            "answer": ["A", "C"],
            "correct_value": "the oven; the cereal box",
            "multi_select": True,
        }

        row = result_from_question(
            question,
            image_resolution=ImageResolution(path=None, checked_paths=()),
            raw_response="Answer: C,A\nReasoning: both move.",
            error=None,
        )

        self.assertEqual(row["gt_answer"], "A,C")
        self.assertEqual(row["prediction"], "A,C")
        self.assertEqual(row["gt_answers"], ["A", "C"])
        self.assertEqual(row["predictions"], ["A", "C"])
        self.assertTrue(row["correct"])

    def test_result_from_question_rejects_partial_multi_select_answer(self) -> None:
        question = {
            "question_uid": "q1",
            "type": "attachment_chain",
            "question": "Which objects move?",
            "options": ["the oven", "the clothes dryer", "the cereal box"],
            "answer": ["A", "C"],
            "correct_value": "the oven; the cereal box",
            "multi_select": True,
        }

        row = result_from_question(
            question,
            image_resolution=ImageResolution(path=None, checked_paths=()),
            raw_response="Answer: A\nReasoning: only one.",
            error=None,
        )

        self.assertEqual(row["prediction"], "A")
        self.assertFalse(row["correct"])

    def test_option_html_marks_multiple_gold_and_wrong_prediction(self) -> None:
        html = _option_html(
            {
                "options": ["the oven", "the clothes dryer", "the cereal box"],
                "multi_select": True,
                "gt_answers": ["A", "C"],
                "predictions": ["A", "B"],
            }
        )

        self.assertIn('<span class="letter">A</span>', html)
        self.assertIn('<div class="option gold"><span class="letter">A</span>', html)
        self.assertIn('<div class="option predicted"><span class="letter">B</span>', html)
        self.assertIn('<div class="option gold"><span class="letter">C</span>', html)

    def test_load_questions_prefers_explicit_dataset_field(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "mixed_multiselect"
            root.mkdir()
            benchmark = root / "benchmark.json"
            benchmark.write_text(
                """
{
  "questions": [
    {
      "dataset": "scannetpp",
      "scene_id": "abc",
      "image_name": "frame.jpg",
      "type": "attachment_chain",
      "question": "Which objects move?",
      "options": ["the oven", "the cereal box"],
      "answer": ["A", "B"],
      "multi_select": true
    }
  ]
}
""".strip(),
                encoding="utf-8",
            )

            questions, _metadata = load_questions([root])

        self.assertEqual(questions[0]["_dataset"], "scannetpp")


if __name__ == "__main__":
    unittest.main()
