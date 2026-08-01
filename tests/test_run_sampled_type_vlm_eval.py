import json
import os
import sys
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from scripts.extract_l3_attachment_chain_multiselect import (
    MULTI_SELECT_NOTE,
    convert_attachment_chain_question,
)
from scripts.run_sampled_type_vlm_eval import (
    ImageResolution,
    _option_html,
    _question_cache_key,
    _resolve_scannetpp_geometry_roots,
    _resolve_api_credential,
    _should_omit_temperature,
    build_prompt,
    call_model,
    load_fixed_questions,
    load_questions,
    load_rollout_manifest,
    make_client,
    parse_args,
    parse_answer,
    parse_answers,
    refresh_cached_result,
    resolve_rollout_images,
    resolve_question_images,
    result_from_question,
    sample_questions,
)
from scripts.validate_rollout_manifest import validate_manifest


class RunSampledTypeVlmEvalMultiselectTests(unittest.TestCase):
    def test_anthropic_auth_token_uses_bearer_credential(self) -> None:
        args = types.SimpleNamespace(
            api_key=None,
            api_key_env="ANTHROPIC_AUTH_TOKEN",
            api_provider="anthropic",
        )
        with patch.dict(os.environ, {"ANTHROPIC_AUTH_TOKEN": "token-value"}, clear=True):
            credential, kind = _resolve_api_credential(args)

        self.assertEqual(credential, "token-value")
        self.assertEqual(kind, "auth_token")

    def test_anthropic_api_key_keeps_x_api_key_credential(self) -> None:
        args = types.SimpleNamespace(
            api_key=None,
            api_key_env=None,
            api_provider="anthropic",
        )
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "key-value"}, clear=True):
            credential, kind = _resolve_api_credential(args)

        self.assertEqual(credential, "key-value")
        self.assertEqual(kind, "api_key")

    def test_make_client_passes_anthropic_auth_token(self) -> None:
        captured_kwargs = {}

        def fake_anthropic(**kwargs):
            captured_kwargs.update(kwargs)
            return object()

        fake_module = types.ModuleType("anthropic")
        fake_module.Anthropic = fake_anthropic
        with patch.dict(sys.modules, {"anthropic": fake_module}):
            make_client(
                "anthropic",
                "https://example.test",
                "token-value",
                30,
                credential_kind="auth_token",
            )

        self.assertEqual(captured_kwargs["auth_token"], "token-value")
        self.assertNotIn("api_key", captured_kwargs)

    def test_subset_keeps_same_stem_with_distinct_options(self) -> None:
        first = {
            "dataset": "scannetpp",
            "scene_id": "scene-1",
            "image_name": "frame.jpg",
            "type": "attachment_chain",
            "question": "What moves with the table?",
            "options": ["the lamp", "the chair"],
            "answer": "A",
        }
        second = {
            **first,
            "options": ["the monitor", "the keyboard"],
            "answer": "B",
        }
        with TemporaryDirectory() as temporary_directory:
            subset_path = Path(temporary_directory) / "benchmark50.json"
            subset_path.write_text(
                json.dumps({"questions": [first, second, dict(first)]}),
                encoding="utf-8",
            )

            questions, metadata = load_questions([], subset_path)

        self.assertEqual(len(questions), 2)
        self.assertEqual(metadata["deduped_question_count"], 2)
        self.assertEqual(metadata["duplicate_question_count"], 1)
        self.assertNotEqual(
            _question_cache_key(questions[0]),
            _question_cache_key(questions[1]),
        )

    def test_load_fixed_questions_accepts_utf8_bom(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            benchmark_path = Path(temporary_directory) / "benchmark.json"
            benchmark_path.write_text(
                json.dumps(
                    {
                        "questions": [{"question_uid": "question-1", "question": "Where?"}],
                        "metadata": {"source": "test"},
                        "sampling_stats": {"count": 1},
                    }
                ),
                encoding="utf-8-sig",
            )

            questions, metadata, sampling_stats = load_fixed_questions(benchmark_path)

        self.assertEqual(len(questions), 1)
        self.assertEqual(questions[0]["question"], "Where?")
        self.assertEqual(questions[0]["_source_question_uid"], "question-1")
        self.assertEqual(metadata["source"], "test")
        self.assertEqual(metadata["deduped_question_count"], 1)
        self.assertEqual(sampling_stats, {"count": 1})

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

    def test_call_model_direct_prefers_final_content_over_reasoning(self) -> None:
        class FakeCompletions:
            def create(self, **_kwargs):
                return iter(
                    [
                        {"choices": [{"delta": {"reasoning_content": "Long analysis."}}]},
                        {"choices": [{"delta": {"content": "B"}}]},
                    ]
                )

        class FakeClient:
            chat = type("Chat", (), {"completions": FakeCompletions()})()

        response = call_model(
            FakeClient(),
            api_provider="openai_chat",
            model="ep-test",
            image_paths=[Path("unused.jpg")],
            prompt="Question?",
            max_tokens=16,
            temperature=0.0,
            api_image_max_px=0,
            blind=True,
            direct=True,
        )

        self.assertEqual(response, "B")

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

    def test_blind_prompt_forces_a_single_answer_on_the_last_line(self) -> None:
        prompt = build_prompt(
            {
                "question": "Where is the chair?",
                "options": ["left", "right"],
            },
            blind=True,
        )

        self.assertIn("Images are intentionally unavailable", prompt)
        self.assertIn("Do not request an image and do not abstain.", prompt)
        self.assertTrue(prompt.endswith("Answer: <single letter>"))

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

    def test_parse_answer_accepts_direct_letter_appended_after_reasoning(self) -> None:
        response = (
            "The cabinet is to the left. So answer B? "
            "After checking the camera frame, that makes sense.B"
        )

        self.assertEqual(parse_answer(response, "ABCD"), "B")

    def test_parse_answers_does_not_collect_option_letters_from_reasoning(self) -> None:
        response = "I considered option B but rejected it. Final answer: D"

        self.assertEqual(parse_answers(response, "ABCD"), ["D"])

    def test_sample_questions_runs_when_scene_cap_exceeds_each_scene_count(self) -> None:
        questions = [
            {
                "question_uid": f"q{index}",
                "type": "occlusion",
                "scene_id": f"scene{index:04d}_00",
            }
            for index in range(5)
        ]

        sampled, stats = sample_questions(
            questions,
            per_type=5,
            scene_cap=3,
            seed=1,
        )

        self.assertEqual(len(sampled), 5)
        self.assertEqual(stats["occlusion"]["sampled"], 5)

    def test_refresh_cached_result_reparses_without_new_api_response(self) -> None:
        question = {
            "question_uid": "new-uid",
            "answer": "B",
            "options": ["left", "right", "front", "back"],
            "_dataset": "scannetpp",
        }
        cached = {
            "question_uid": "old-uid",
            "raw_response": "Reasoning that ends with the direct choice.B",
            "prediction": None,
            "correct": False,
            "dataset": "unknown",
        }

        refreshed = refresh_cached_result(question, cached)

        self.assertEqual(refreshed["prediction"], "B")
        self.assertTrue(refreshed["correct"])
        self.assertEqual(refreshed["dataset"], "scannetpp")
        self.assertEqual(refreshed["question_uid"], "new-uid")

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

    def test_load_questions_recovers_dataset_from_scene_id(self) -> None:
        with TemporaryDirectory() as tmp:
            subset = Path(tmp) / "benchmark50.json"
            subset.write_text(
                json.dumps(
                    {
                        "questions": [
                            {
                                "_dataset": "0-9",
                                "scene_id": "c49a8c6cff",
                                "image_name": "frame.jpg",
                                "type": "object_move_agent",
                                "question": "Where?",
                            },
                            {
                                "scene_id": "scene0633_00",
                                "image_name": "3.jpg",
                                "type": "direction_agent",
                                "question": "Where now?",
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )

            questions, _metadata = load_questions([], subset)

        self.assertEqual(
            [question["_dataset"] for question in questions],
            ["scannetpp", "scannet"],
        )

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

    def test_call_model_interleaves_fixed_rollout_role_labels(self) -> None:
        captured_kwargs = {}

        class FakeCompletions:
            def create(self, **kwargs):
                captured_kwargs.update(kwargs)
                return iter([{"choices": [{"delta": {"content": "Answer: A"}}]}])

        class FakeClient:
            chat = type("Chat", (), {"completions": FakeCompletions()})()

        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "future.jpg"
            path.write_bytes(b"fake")
            call_model(
                FakeClient(),
                api_provider="openai_chat",
                model="qwen3.5-flash",
                image_paths=[path],
                image_roles=["predicted future destination view after the operation"],
                prompt="Question?",
                max_tokens=16,
                temperature=0.0,
                api_image_max_px=0,
            )

        content = captured_kwargs["messages"][1]["content"]
        self.assertEqual(content[0]["type"], "text")
        self.assertEqual(
            content[0]["text"],
            "Image 1 role: predicted future destination view after the operation.",
        )
        self.assertEqual(content[1]["type"], "image_url")
        self.assertEqual(content[2], {"type": "text", "text": "Question?"})
        self.assertEqual(
            captured_kwargs["extra_body"],
            {"enable_thinking": True},
        )

    def test_call_model_disables_qwen_thinking_for_direct_answers(self) -> None:
        captured_kwargs = {}

        class FakeCompletions:
            def create(self, **kwargs):
                captured_kwargs.update(kwargs)
                return iter([{"choices": [{"delta": {"content": "A"}}]}])

        class FakeClient:
            chat = type("Chat", (), {"completions": FakeCompletions()})()

        call_model(
            FakeClient(),
            api_provider="openai_chat",
            model="qwen3.5-flash",
            image_paths=[Path("unused.jpg")],
            prompt="Question?",
            max_tokens=16,
            temperature=0.0,
            api_image_max_px=0,
            blind=True,
            direct=True,
        )

        self.assertEqual(
            captured_kwargs["extra_body"],
            {"enable_thinking": False},
        )

    def test_rollout_manifest_rejects_answer_leakage(self) -> None:
        with TemporaryDirectory() as tmp:
            manifest_path = Path(tmp) / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "schema_version": "predictive-spatial-rollout-v1",
                        "entries": [{"question_uid": "q1", "correct_value": "left"}],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "forbidden GT/answer field"):
                load_rollout_manifest(manifest_path)

    def test_rollout_manifest_rejects_answer_bearing_legacy_uid(self) -> None:
        with TemporaryDirectory() as tmp:
            manifest_path = Path(tmp) / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "schema_version": "predictive-spatial-rollout-v1",
                        "entries": [
                            {"question_uid": '{"scene_id":"s1","answer":"B"}'}
                        ],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "embeds answer-bearing JSON"):
                load_rollout_manifest(manifest_path)

    def test_picture_rollout_context_only_removes_prediction_in_place(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("source.jpg", "destination.jpg", "future.jpg", "query.jpg"):
                (root / name).write_bytes(name.encode("ascii"))
            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "schema_version": "predictive-spatial-rollout-v1",
                        "entries": [
                            {
                                "question_uid": "q1",
                                "question_type": "object_move_agent",
                                "scene_id": "scene0000_00",
                                "picture": {
                                    "eligible": True,
                                    "rejection_reasons": [],
                                    "media": [
                                        {"path": "source.jpg", "role": "source_view", "kind": "context"},
                                        {"path": "destination.jpg", "role": "destination_environment", "kind": "context"},
                                        {"path": "future.jpg", "role": "predicted_future_view", "kind": "prediction"},
                                        {"path": "query.jpg", "role": "query_reference_view", "kind": "context"},
                                    ],
                                },
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            manifest = load_rollout_manifest(manifest_path)
            question = {"question_uid": "q1", "_rollout_manifest_uid": "q1"}
            full, full_error = resolve_rollout_images(
                question, manifest, mode="picture", context_only=False
            )
            context, context_error = resolve_rollout_images(
                question, manifest, mode="picture", context_only=True
            )

        self.assertIsNone(full_error)
        self.assertEqual([item.path.name for item in full], ["source.jpg", "destination.jpg", "future.jpg", "query.jpg"])
        self.assertIsNone(context_error)
        self.assertEqual([item.path.name for item in context], ["source.jpg", "destination.jpg", "query.jpg"])
        self.assertEqual([item.role for item in context], ["source_view", "destination_environment", "query_reference_view"])

    def test_video_rollout_requires_exactly_eight_ordered_frames(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "motion.jpg").write_bytes(b"motion")
            media = [
                {"path": "motion.jpg", "role": "motion_reference_view", "kind": "context"}
            ]
            for index in range(8):
                name = f"frame-{index}.jpg"
                (root / name).write_bytes(name.encode("ascii"))
                media.append(
                    {
                        "path": name,
                        "role": "predicted_video_frame",
                        "kind": "prediction",
                        "frame_index": index,
                    }
                )
            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "schema_version": "predictive-spatial-rollout-v1",
                        "entries": [
                            {
                                "question_uid": "q1",
                                "question_type": "object_move_agent",
                                "scene_id": "scene0000_00",
                                "video": {
                                    "eligible": True,
                                    "rejection_reasons": [],
                                    "media": media,
                                },
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            manifest = load_rollout_manifest(manifest_path)
            resolutions, error = resolve_rollout_images(
                {"question_uid": "q1"}, manifest, mode="video", context_only=False
            )
            report = validate_manifest(
                manifest,
                mode="video",
                expected_per_type=50,
                strict_provenance=False,
            )

        self.assertIsNone(error)
        self.assertEqual(len(resolutions), 9)
        self.assertEqual(resolutions[1].role, "predicted_video_frame:1/8")
        self.assertEqual(resolutions[-1].role, "predicted_video_frame:8/8")
        self.assertTrue(report["valid"], report["errors"])

    def test_rollout_cli_enforces_mode_combinations(self) -> None:
        with self.assertRaises(SystemExit):
            parse_args(["--context_only"])
        with self.assertRaises(SystemExit):
            parse_args(["--picture", "--video", "--rollout_manifest", "missing.json"])
        with self.assertRaises(SystemExit):
            parse_args(["--picture", "--blind", "--rollout_manifest", "missing.json"])

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
