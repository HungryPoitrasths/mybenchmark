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
    _resolve_scannetpp_geometry_roots,
    _should_omit_temperature,
    build_prompt,
    call_model,
    load_questions,
    parse_answers,
    resolve_question_images,
    result_from_question,
)


class RunSampledTypeVlmEvalMultiselectTests(unittest.TestCase):
    def test_call_model_reads_proxy_stream_reasoning_content(self) -> None:
        class FakeCompletions:
            def create(self, **_kwargs):
                return iter(
                    [
                        {"choices": [{"delta": {"reasoning_content": "Reasoning: visible. "}}]},
                        {"choices": [{"delta": {"content": "Answer: B"}}]},
                    ]
                )

        class FakeClient:
            chat = type("Chat", (), {"completions": FakeCompletions()})()

        response = call_model(
            FakeClient(),
            api_provider="openai_chat",
            model="claude-sonnet-4-6",
            image_paths=[Path("unused.jpg")],
            prompt="Question?",
            max_tokens=16,
            temperature=0.0,
            api_image_max_px=0,
            blind=True,
        )

        self.assertEqual(response, "Reasoning: visible. Answer: B")

    def test_call_model_rejects_empty_stream_response(self) -> None:
        class FakeCompletions:
            def create(self, **_kwargs):
                return iter([{"choices": [{"delta": {"content": ""}}]}])

        class FakeClient:
            chat = type("Chat", (), {"completions": FakeCompletions()})()

        with self.assertRaisesRegex(RuntimeError, "empty response"):
            call_model(
                FakeClient(),
                api_provider="openai_chat",
                model="claude-sonnet-4-6",
                image_paths=[Path("unused.jpg")],
                prompt="Question?",
                max_tokens=16,
                temperature=0.0,
                api_image_max_px=0,
                blind=True,
            )

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

        self.assertIn("list all letters comma-separated", prompt)
        self.assertIn("Answer: <letter(s)>", prompt)

    def test_claude_opus_4_omits_temperature_for_proxy_compatibility(self) -> None:
        self.assertTrue(_should_omit_temperature("claude-opus-4-7"))
        self.assertTrue(_should_omit_temperature("claude-sonnet-4-5"))

    def test_scannetpp_geometry_roots_include_frame_root_siblings(self) -> None:
        roots = _resolve_scannetpp_geometry_roots(
            ["/home/sujinyue/mybenchmark/output/scannetpp_iphone_frames"],
            None,
        )
        normalized = {root.replace("\\", "/") for root in roots}

        self.assertIn("/home/sujinyue/mybenchmark/data/scannetpp", normalized)
        self.assertIn("/home/sujinyue/mybenchmark/++data", normalized)
        self.assertFalse(_should_omit_temperature("qwen3.5-flash"))

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
            image_resolutions=[ImageResolution(path=None, checked_paths=())],
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
            image_resolutions=[ImageResolution(path=None, checked_paths=())],
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

    def test_resolve_question_images_returns_single_resolution_for_ordinary_question(
        self,
    ) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "scene0000_00" / "color").mkdir(parents=True)
            (root / "scene0000_00" / "color" / "000.jpg").write_bytes(b"fake")

            resolutions = resolve_question_images(
                {"scene_id": "scene0000_00", "image_name": "000.jpg"},
                scannet_roots=[root],
                scannetpp_roots=[],
                scannetpp_sensor="iphone",
            )

        self.assertEqual(len(resolutions), 1)
        self.assertEqual(resolutions[0].path.name, "000.jpg")

    def test_resolve_question_images_appends_reasoning_frame_2(self) -> None:
        # Two-frame-split questions (_apply_two_frame_split in run_pipeline.py)
        # store the destination frame in reasoning_frame_2 -- the model must
        # actually receive it, not just the primary image_name.
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "scene0000_00" / "color").mkdir(parents=True)
            (root / "scene0000_00" / "color" / "frame_a.jpg").write_bytes(b"fake")
            (root / "scene0000_00" / "color" / "frame_b.jpg").write_bytes(b"fake")

            resolutions = resolve_question_images(
                {
                    "scene_id": "scene0000_00",
                    "image_name": "frame_a.jpg",
                    "reasoning_frame_2": "frame_b.jpg",
                    "auxiliary_image_names": [],
                },
                scannet_roots=[root],
                scannetpp_roots=[],
                scannetpp_sensor="iphone",
            )

        self.assertEqual(len(resolutions), 2)
        self.assertEqual(resolutions[0].path.name, "frame_a.jpg")
        self.assertEqual(resolutions[1].path.name, "frame_b.jpg")

    def test_call_model_openai_chat_sends_one_image_block_per_path(self) -> None:
        captured_kwargs = {}

        class FakeCompletions:
            def create(self, **kwargs):
                captured_kwargs.update(kwargs)
                return iter([{"choices": [{"delta": {"content": "Answer: A"}}]}])

        class FakeClient:
            chat = type("Chat", (), {"completions": FakeCompletions()})()

        with TemporaryDirectory() as tmp:
            path_a = Path(tmp) / "a.jpg"
            path_b = Path(tmp) / "b.jpg"
            path_a.write_bytes(b"fake-a")
            path_b.write_bytes(b"fake-b")

            call_model(
                FakeClient(),
                api_provider="openai_chat",
                model="claude-sonnet-4-6",
                image_paths=[path_a, path_b],
                prompt="Question?",
                max_tokens=16,
                temperature=0.0,
                api_image_max_px=0,
            )

        user_content = captured_kwargs["messages"][1]["content"]
        image_blocks = [b for b in user_content if b["type"] == "image_url"]
        self.assertEqual(len(image_blocks), 2)

    def test_result_from_question_populates_aux_image_fields(self) -> None:
        question = {
            "question_uid": "q1",
            "type": "object_move_agent",
            "question": "Where is the microwave?",
            "options": ["a", "b"],
            "answer": "A",
            "correct_value": "a",
            "image_name": "frame_a.jpg",
            "reasoning_frame_2": "frame_b.jpg",
            "auxiliary_image_names": [],
        }

        row = result_from_question(
            question,
            image_resolutions=[
                ImageResolution(path=Path("/tmp/frame_a.jpg"), checked_paths=("/tmp/frame_a.jpg",)),
                ImageResolution(path=Path("/tmp/frame_b.jpg"), checked_paths=("/tmp/frame_b.jpg",)),
            ],
            raw_response="Answer: A",
            error=None,
        )

        self.assertEqual(row["image_path"], str(Path("/tmp/frame_a.jpg")))
        self.assertEqual(row["aux_image_names"], ["frame_b.jpg"])
        self.assertEqual(row["aux_image_paths"], [str(Path("/tmp/frame_b.jpg"))])
        self.assertEqual(
            row["checked_image_paths"], ["/tmp/frame_a.jpg", "/tmp/frame_b.jpg"]
        )


if __name__ == "__main__":
    unittest.main()
