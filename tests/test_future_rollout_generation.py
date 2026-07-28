import json
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image

from scripts.finalize_cosmos_rollouts import eight_frame_indices, finalize_jobs
import scripts.generate_rollout_selection_spec as selection_generator
import scripts.prepare_future_rollout_jobs as rollout_preparer
from scripts.generate_rollout_selection_spec import (
    FrameCandidate,
    QuestionMotion,
    SelectionPaths,
    _apply_question_motion,
    _candidate_is_better,
    _evaluate_frame,
    _materialize_candidate,
    _moving_group_ids,
    _static_visibility,
    generate_selection_spec,
)
from scripts.prepare_future_rollout_jobs import (
    action_duration_seconds,
    build_cosmos_prompt,
    build_picture_prompt,
    build_safe_action,
    prepare_jobs,
    world_delta_to_agent_components,
)
from scripts.run_future_picture_generation import (
    RetryableGenerationError,
    _is_retryable,
    run_jobs,
)
from scripts.run_sampled_type_vlm_eval import (
    load_rollout_manifest,
    resolve_rollout_images,
)


class FutureRolloutGenerationTests(unittest.TestCase):
    CAMERA_ROTATION_WORLD_TO_CAMERA = [
        [1.0, 0.0, 0.0],
        [0.0, 0.9396926208, -0.3420201433],
        [0.0, 0.3420201433, 0.9396926208],
    ]

    @staticmethod
    def _object(obj_id, center, label):
        center_array = np.asarray(center, dtype=float)
        return {
            "id": obj_id,
            "label": label,
            "center": center_array.tolist(),
            "bbox_min": (center_array - 0.25).tolist(),
            "bbox_max": (center_array + 0.25).tolist(),
        }

    def test_moving_group_and_motion_endpoints_include_attachment_chain(self) -> None:
        objects = [
            self._object(1, [1.0, 0.0, 0.5], "table"),
            self._object(2, [2.0, 0.0, 1.0], "book"),
            self._object(3, [0.0, 0.0, 0.5], "chair"),
        ]
        graph = {1: [2]}
        translation = {
            "type": "object_move_agent",
            "moved_obj_id": 1,
            "query_obj_id": 3,
            "delta": [0.5, 1.0, 0.0],
        }
        self.assertEqual(_moving_group_ids(translation, graph), (1, 2))
        translated = _apply_question_motion(translation, objects, graph)
        self.assertEqual(translated.source_required_ids, (1, 2, 3))
        np.testing.assert_allclose(translated.moved_objects_by_id[1]["center"], [1.5, 1.0, 0.5])
        np.testing.assert_allclose(translated.moved_objects_by_id[2]["center"], [2.5, 1.0, 1.0])

        orbit = {
            "type": "object_rotate_object_centric",
            "moved_obj_id": 1,
            "query_obj_id": 3,
            "obj_face_id": 3,
            "rotation_angle": 90,
            "rotation_direction": "counterclockwise",
        }
        rotated = _apply_question_motion(orbit, objects, graph)
        np.testing.assert_allclose(rotated.moved_objects_by_id[1]["center"], [0.0, 1.0, 0.5], atol=1e-7)
        np.testing.assert_allclose(rotated.moved_objects_by_id[2]["center"], [0.0, 2.0, 1.0], atol=1e-7)

    def test_candidate_ranking_prefers_score_then_natural_frame_name(self) -> None:
        def candidate(name, score):
            return FrameCandidate(
                question={}, source_index=0, context=None, motion=None,
                image_name=name, image_path=Path(name), pose=None,
                score_key=score, metrics={}, context_paths=[],
            )

        current = candidate("frame_10.jpg", (0.9, 0.2))
        self.assertTrue(_candidate_is_better(candidate("frame_2.jpg", (0.9, 0.2)), current))
        self.assertTrue(_candidate_is_better(candidate("frame_20.jpg", (0.95, 0.1)), current))
        self.assertFalse(_candidate_is_better(candidate("frame_1.jpg", (0.8, 0.9)), current))

    def test_frame_evaluation_rejects_failed_strict_projection_gate(self) -> None:
        with TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "frame.jpg"
            Image.new("RGB", (32, 24), color=(10, 20, 30)).save(image_path)
            obj = self._object(1, [0.0, 0.0, 1.0], "table")
            context = SimpleNamespace(
                poses={"frame.jpg": object()},
                data_source=SimpleNamespace(
                    image_path=lambda _name: image_path,
                    load_depth_frame=lambda _name: None,
                ),
                objects_by_id={1: obj},
            )
            motion = QuestionMotion(
                moved_ids=(1,), source_required_ids=(1,),
                moved_objects=[obj], moved_objects_by_id={1: obj},
            )
            with patch.object(
                selection_generator,
                "_read_image_quality_metrics",
                return_value={"readable": True, "laplacian_variance": 100.0, "tenengrad": 100.0},
            ), patch.object(
                selection_generator, "_passes_absolute_image_quality_gate", return_value=True
            ), patch.object(
                selection_generator, "_projection_gate", return_value=(False, {})
            ):
                selected, reason = _evaluate_frame(
                    question={"image_name": "frame.jpg"},
                    source_index=0,
                    context=context,
                    motion=motion,
                    image_name="frame.jpg",
                    context_paths=[],
                )
            self.assertIsNone(selected)
            self.assertEqual(reason, "source_projection_gate_failed")

    def test_frame_evaluation_reuses_scene_quality_and_static_visibility_caches(self) -> None:
        with TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "frame.jpg"
            Image.new("RGB", (32, 24), color=(10, 20, 30)).save(image_path)
            obj = self._object(1, [0.0, 0.0, 1.0], "table")
            depth_loads = []
            context = SimpleNamespace(
                poses={"frame.jpg": object()},
                data_source=SimpleNamespace(
                    image_path=lambda _name: image_path,
                    load_depth_frame=lambda name: depth_loads.append(name),
                ),
                objects_by_id={1: obj},
            )
            motion = QuestionMotion(
                moved_ids=(1,), source_required_ids=(1,),
                moved_objects=[obj], moved_objects_by_id={1: obj},
            )
            projection = {
                "projected_area_ratio": 0.5,
                "edge_margin_ratio": 0.25,
            }
            with patch.object(
                selection_generator,
                "_read_image_quality_metrics",
                return_value={
                    "readable": True,
                    "laplacian_variance": 100.0,
                    "tenengrad": 100.0,
                },
            ) as read_quality, patch.object(
                selection_generator, "_passes_absolute_image_quality_gate", return_value=True
            ), patch.object(
                selection_generator, "_projection_gate", return_value=(True, projection)
            ), patch.object(
                selection_generator,
                "_static_visibility",
                return_value=(True, 1.0, "depth", {"visible_ratio": 1.0}),
            ) as static_visibility, patch.object(
                selection_generator,
                "_future_visibility",
                return_value=(True, 1.0, {"visible_ratio": 1.0}),
            ):
                for source_index in (0, 1):
                    candidate, reason = _evaluate_frame(
                        question={"image_name": "frame.jpg"},
                        source_index=source_index,
                        context=context,
                        motion=motion,
                        image_name="frame.jpg",
                        context_paths=[],
                    )
                    self.assertIsNotNone(candidate)
                    self.assertEqual(reason, "selected_candidate")

            self.assertEqual(read_quality.call_count, 1)
            self.assertEqual(static_visibility.call_count, 1)
            self.assertEqual(depth_loads, ["frame.jpg"])
            self.assertEqual(context.cache_stats["image_quality_hits"], 1)
            self.assertEqual(context.cache_stats["static_visibility_hits"], 1)

    def test_static_visibility_uses_registered_depth_before_mesh_fallback(self) -> None:
        obj = self._object(1, [0.0, 0.0, 1.0], "table")
        context = SimpleNamespace(
            intrinsics=object(), ray_caster=object(), instance_mesh_data=object()
        )
        depth_frame = SimpleNamespace(intrinsics=object(), image_m=np.ones((2, 2)))
        with patch.object(
            selection_generator,
            "compute_depth_occlusion_metrics",
            return_value={
                "valid_in_frame_count": 4,
                "visible_in_frame_count": 1,
                "visible_ratio_in_frame": 0.25,
            },
        ), patch.object(
            selection_generator,
            "_compute_mesh_ray_l1_occlusion_metrics_for_static_target",
        ) as mesh_visibility:
            visible, ratio, backend, _ = _static_visibility(
                obj, pose=object(), context=context, depth_frame=depth_frame
            )
        self.assertFalse(visible)
        self.assertEqual(ratio, 0.25)
        self.assertEqual(backend, "depth")
        mesh_visibility.assert_not_called()

        with patch.object(
            selection_generator,
            "compute_depth_occlusion_metrics",
            return_value={
                "valid_in_frame_count": 0,
                "visible_in_frame_count": 0,
                "visible_ratio_in_frame": 0.0,
            },
        ), patch.object(
            selection_generator,
            "_compute_mesh_ray_l1_occlusion_metrics_for_static_target",
            return_value=SimpleNamespace(
                valid_in_frame_count=10,
                visible_in_frame_count=10,
                projected_area=100.0,
                in_frame_ratio=1.0,
            ),
        ):
            visible, ratio, backend, _ = _static_visibility(
                obj, pose=object(), context=context, depth_frame=depth_frame
            )
        self.assertTrue(visible)
        self.assertEqual(ratio, 1.0)
        self.assertEqual(backend, "mesh_ray")

    def test_materialization_copies_condition_and_orders_context_roles(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            images = []
            for index in range(3):
                path = root / f"source_{index}.jpg"
                Image.new("RGB", (16, 12), color=(index * 20, 10, 30)).save(path)
                images.append(path)
            obj = self._object(1, [0.0, 0.0, 0.5], "table")
            candidate = FrameCandidate(
                question={"question_uid": "uid-1", "type": "object_move_agent"},
                source_index=4,
                context=SimpleNamespace(objects_by_id={1: obj}),
                motion=QuestionMotion((1,), (1,), [obj], {1: obj}),
                image_name=images[0].name,
                image_path=images[0],
                pose=SimpleNamespace(rotation=np.eye(3)),
                score_key=(1.0,),
                metrics={},
                context_paths=images,
            )
            entry = _materialize_candidate(candidate, root / "rollout")
            self.assertTrue(Path(entry["motion_frame_path"]).is_file())
            self.assertEqual(
                [item["role"] for item in entry["answer_context_media"]],
                ["destination_to_query_bridge", "query_reference_view"],
            )
            self.assertTrue(all(Path(item["path"]).is_file() for item in entry["answer_context_media"]))

    def test_candidate_records_rehydrate_only_selected_scene_for_materialization(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "frame.jpg"
            Image.new("RGB", (16, 12), color=(10, 20, 30)).save(image_path)
            moved = self._object(1, [0.0, 0.0, 0.5], "table")
            query = self._object(2, [1.0, 0.0, 0.5], "chair")
            question = {
                "question_uid": "uid-1",
                "type": "object_move_agent",
                "scene_id": "scene0000_00",
                "image_name": "frame.jpg",
                "moved_obj_id": 1,
                "query_obj_id": 2,
                "delta": [0.5, 0.0, 0.0],
                "has_attachment_chain": False,
            }
            context = SimpleNamespace(
                objects=[moved, query],
                objects_by_id={1: moved, 2: query},
                attachment_graph={},
                poses={"frame.jpg": SimpleNamespace(rotation=np.eye(3))},
                data_source=SimpleNamespace(image_path=lambda _name: image_path),
            )
            candidate = selection_generator.CandidateRecord(
                question=question,
                source_index=0,
                dataset="scannet",
                scene_id="scene0000_00",
                image_name="frame.jpg",
                score_key=(1.0,),
                metrics={},
            )
            with patch.object(
                selection_generator,
                "_load_scene_context",
                return_value=context,
            ) as load_context:
                entries = selection_generator._materialize_candidate_records(
                    [candidate],
                    output_dir=root / "rollout",
                    scannet_root=None,
                    scannetpp_root=None,
                    scannetpp_frame_root=None,
                    scannetpp_sensor="iphone",
                )

            load_context.assert_called_once()
            self.assertEqual(entries[0]["question_uid"], "uid-1")
            self.assertTrue(Path(entries[0]["motion_frame_path"]).is_file())

    def test_selection_shortage_writes_spec_and_audit_without_failing(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            benchmark = root / "benchmark.json"
            benchmark.write_text('{"questions": []}', encoding="utf-8")
            with patch.object(selection_generator, "load_fixed_questions", return_value=([], {}, {})):
                paths = generate_selection_spec(
                    benchmark_path=benchmark,
                    output_dir=root / "rollout",
                    scannet_root=None,
                    scannetpp_root=None,
                    scannetpp_frame_root=None,
                    expected_per_type=1,
                )
            spec_path = root / "rollout" / "private_jobs" / "selection_spec.json"
            audit_path = root / "rollout" / "private_jobs" / "selection_audit.json"
            self.assertEqual(paths, SelectionPaths(spec=spec_path, audit=audit_path))
            self.assertTrue(spec_path.is_file())
            self.assertTrue(audit_path.is_file())
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            self.assertTrue(all(item["selected"] == 0 for item in audit["counts_by_type"].values()))
            self.assertTrue(all(item["shortfall"] == 1 for item in audit["counts_by_type"].values()))

    def test_selection_checkpoint_resumes_after_interrupted_question(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            benchmark = root / "benchmark.json"
            benchmark.write_text('{"questions": []}', encoding="utf-8")
            questions = [
                {
                    "question_uid": f"uid-{index}",
                    "type": "object_move_agent",
                    "scene_id": "scene0000_00",
                    "_dataset": "scannet",
                    "moved_obj_id": 1,
                    "query_obj_id": 2,
                }
                for index in range(2)
            ]
            fake_context = SimpleNamespace(cache_stats=selection_generator.Counter())

            def make_candidate(source_index, question):
                return FrameCandidate(
                    question=question,
                    source_index=source_index,
                    context=fake_context,
                    motion=None,
                    image_name=f"frame-{source_index}.jpg",
                    image_path=root / f"frame-{source_index}.jpg",
                    pose=None,
                    score_key=(1.0 - source_index * 0.1,),
                    metrics={"score": [1.0 - source_index * 0.1]},
                    context_paths=[],
                )

            first_run_calls = []

            def interrupt_second_question(**kwargs):
                source_index = kwargs["source_index"]
                first_run_calls.append(source_index)
                if source_index == 1:
                    raise KeyboardInterrupt
                return make_candidate(source_index, kwargs["question"]), {}, None

            common_patches = (
                patch.object(
                    selection_generator,
                    "load_fixed_questions",
                    return_value=(questions, {}, {}),
                ),
                patch.object(selection_generator, "_load_scene_context", return_value=fake_context),
                patch.object(
                    selection_generator,
                    "_materialize_candidate_records",
                    return_value=[{"question_uid": "uid-0"}, {"question_uid": "uid-1"}],
                ),
            )
            with common_patches[0], common_patches[1], common_patches[2], patch.object(
                selection_generator,
                "_best_frame_for_question",
                side_effect=interrupt_second_question,
            ):
                with self.assertRaises(KeyboardInterrupt):
                    generate_selection_spec(
                        benchmark_path=benchmark,
                        output_dir=root / "rollout",
                        scannet_root=None,
                        scannetpp_root=None,
                        scannetpp_frame_root=None,
                        expected_per_type=2,
                    )
            self.assertEqual(first_run_calls, [0, 1])
            checkpoint_path = next(
                (root / "rollout" / "private_jobs").glob("selection_checkpoint_*.jsonl")
            )
            with checkpoint_path.open("a", encoding="utf-8") as stream:
                stream.write('{"kind":')

            second_run_calls = []

            def finish_pending_question(**kwargs):
                source_index = kwargs["source_index"]
                second_run_calls.append(source_index)
                return make_candidate(source_index, kwargs["question"]), {}, None

            stderr = StringIO()
            with patch.object(
                selection_generator,
                "load_fixed_questions",
                return_value=(questions, {}, {}),
            ), patch.object(
                selection_generator, "_load_scene_context", return_value=fake_context
            ), patch.object(
                selection_generator,
                "_materialize_candidate_records",
                return_value=[{"question_uid": "uid-0"}, {"question_uid": "uid-1"}],
            ), patch.object(
                selection_generator,
                "_best_frame_for_question",
                side_effect=finish_pending_question,
            ), patch("sys.stderr", stderr):
                paths = generate_selection_spec(
                    benchmark_path=benchmark,
                    output_dir=root / "rollout",
                    scannet_root=None,
                    scannetpp_root=None,
                    scannetpp_frame_root=None,
                    expected_per_type=2,
                )

            self.assertEqual(second_run_calls, [1])
            self.assertIn("resumed 1/2 completed questions", stderr.getvalue())
            self.assertTrue(paths.spec.is_file())
            self.assertEqual(
                len(list((root / "rollout" / "private_jobs").glob("selection_checkpoint_*.jsonl"))),
                1,
            )
            for line in checkpoint_path.read_text(encoding="utf-8").splitlines():
                json.loads(line)

    def test_prepare_cli_is_automatic_and_invokes_selection_first(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            benchmark = root / "benchmark.json"
            benchmark.write_text('{"questions": []}', encoding="utf-8")
            args = rollout_preparer.parse_args(
                ["--benchmark_file", str(benchmark), "--output_dir", str(root / "rollout")]
            )
            self.assertFalse(hasattr(args, "selection_spec"))
            with self.assertRaises(SystemExit):
                rollout_preparer.parse_args(
                    [
                        "--benchmark_file", str(benchmark),
                        "--output_dir", str(root / "rollout"),
                        "--selection_spec", str(root / "manual.json"),
                    ]
                )

            paths = SelectionPaths(
                spec=root / "rollout" / "private_jobs" / "selection_spec.json",
                audit=root / "rollout" / "private_jobs" / "selection_audit.json",
            )
            with patch.object(rollout_preparer, "generate_selection_spec", return_value=paths) as generate, patch.object(
                rollout_preparer, "prepare_jobs", return_value={"gpt_jobs": root / "gpt.json"}
            ) as prepare:
                rollout_preparer.main(
                    [
                        "--benchmark_file", str(benchmark),
                        "--output_dir", str(root / "rollout"),
                        "--expected_picture_per_type", "1",
                    ]
                )
            generate.assert_called_once()
            self.assertEqual(prepare.call_args.kwargs["selection_spec_path"], paths.spec)
            self.assertEqual(prepare.call_args.kwargs["expected_picture_per_type"], 1)

    def test_world_delta_is_converted_to_camera_agent_components(self) -> None:
        components = world_delta_to_agent_components(
            [1.0, 0.0, 2.0],
            self.CAMERA_ROTATION_WORLD_TO_CAMERA,
        )

        self.assertAlmostEqual(components["right_m"], 1.0)
        self.assertAlmostEqual(components["away_m"], 0.0)
        self.assertAlmostEqual(components["up_m"], 2.0)

    def test_prompts_include_only_action_not_question_or_occlusion_result(self) -> None:
        question = {
            "type": "object_move_occlusion",
            "question": "Will the secret target be occluded?",
            "options": ["occluded", "not occluded"],
            "answer": "A",
            "correct_value": "occluded",
            "moved_obj_label": "cabinet",
            "query_obj_label": "secret target",
            "delta": [1.0, 2.0, 0.0],
            "has_attachment_chain": False,
        }
        spec = {
            "camera_rotation_world_to_camera": self.CAMERA_ROTATION_WORLD_TO_CAMERA
        }

        action = build_safe_action(question, spec)
        picture_prompt = build_picture_prompt(action)
        video_prompt = build_cosmos_prompt(action)

        for prompt in (picture_prompt, video_prompt):
            self.assertIn('"cabinet"', prompt)
            self.assertNotIn("secret target", prompt)
            self.assertNotIn("occluded", prompt.lower())
            self.assertNotIn("Will the", prompt)

    def test_attachment_questions_require_complete_moving_group(self) -> None:
        question = {
            "type": "object_move_agent",
            "moved_obj_id": 10,
            "moved_obj_label": "desk",
            "attachment_parent_id": 10,
            "attachment_child_id": 11,
            "delta": [1.0, 0.0, 0.0],
            "has_attachment_chain": True,
        }
        spec = {
            "camera_rotation_world_to_camera": self.CAMERA_ROTATION_WORLD_TO_CAMERA
        }
        with self.assertRaisesRegex(ValueError, "full transitive chain"):
            build_safe_action(question, spec)

        spec["moving_group"] = [
            {"obj_id": 10, "label": "desk"},
            {"obj_id": 11, "label": "book"},
            {"obj_id": 12, "label": "pencil"},
        ]
        action = build_safe_action(question, spec)
        self.assertEqual(action["moving_group_labels"], ["desk", "book", "pencil"])

    def test_duration_is_monotonic_and_clamped(self) -> None:
        durations = [action_duration_seconds(distance) for distance in (0.1, 1.0, 2.0, 10.0)]
        self.assertEqual(durations[0], 2.0)
        self.assertEqual(durations[-1], 5.0)
        self.assertEqual(durations, sorted(durations))

    def test_only_transient_or_empty_media_errors_are_retryable(self) -> None:
        self.assertTrue(_is_retryable(RuntimeError("request timed out")))
        self.assertTrue(_is_retryable(RetryableGenerationError("empty response")))
        self.assertFalse(_is_retryable(RuntimeError("invalid API key")))

    def test_prepare_and_fake_picture_generation_end_to_end(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "motion.png"
            Image.new("RGB", (32, 24), color=(10, 20, 30)).save(image_path)
            benchmark_path = root / "benchmark.json"
            benchmark_path.write_text(
                json.dumps(
                    {
                        "questions": [
                            {
                                "level": "L2",
                                "type": "object_move_agent",
                                "scene_id": "scene0000_00",
                                "image_name": "motion.png",
                                "_dataset": "scannet",
                                "question": "After moving the cabinet, where is the chair?",
                                "options": ["left", "right"],
                                "answer": "A",
                                "correct_value": "left",
                                "moved_obj_id": 1,
                                "moved_obj_label": "cabinet",
                                "delta": [1.0, 2.0, 0.0],
                                "has_attachment_chain": False,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            selection_path = root / "private_selection.json"
            selection_path.write_text(
                json.dumps(
                    {
                        "entries": [
                            {
                                "source_index": 0,
                                "motion_frame_path": str(image_path),
                                "camera_rotation_world_to_camera": self.CAMERA_ROTATION_WORLD_TO_CAMERA,
                                "picture_eligible": True,
                                "video_eligible": True,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            outputs = prepare_jobs(
                benchmark_path=benchmark_path,
                selection_spec_path=selection_path,
                output_dir=root / "rollouts",
                seed=7,
                expected_picture_per_type=50,
            )

            gpt_jobs = json.loads(outputs["gpt_jobs"].read_text(encoding="utf-8"))
            qwen_jobs = json.loads(outputs["qwen_jobs"].read_text(encoding="utf-8"))
            self.assertEqual(gpt_jobs["entries"][0]["prompt"], qwen_jobs["entries"][0]["prompt"])
            self.assertNotIn("where is the chair", gpt_jobs["entries"][0]["prompt"].lower())
            public_text = outputs["gpt_manifest"].read_text(encoding="utf-8")
            self.assertNotIn('"prompt"', public_text)
            self.assertNotIn('"answer"', public_text)
            manifest = load_rollout_manifest(outputs["gpt_manifest"])
            uid = manifest.entry_order[0]
            context, error = resolve_rollout_images(
                {"question_uid": uid}, manifest, mode="picture", context_only=True
            )
            self.assertIsNone(error)
            self.assertEqual(len(context), 1)

            calls = []

            def fake_generate(job):
                calls.append(job["question_uid"])
                return Image.new("RGB", (16, 16), color=(200, 10, 20)), "fake-response"

            stats = run_jobs(
                jobs_path=outputs["gpt_jobs"],
                manifest_path=outputs["gpt_manifest"],
                generate=fake_generate,
                retries=0,
                retry_delay=0,
            )
            self.assertEqual(stats["generated"], 1)
            self.assertEqual(len(calls), 1)

            pending_manifest = json.loads(
                outputs["gpt_manifest"].read_text(encoding="utf-8")
            )
            pending_manifest["entries"][0]["picture"]["generation"]["status"] = "pending"
            outputs["gpt_manifest"].write_text(
                json.dumps(pending_manifest), encoding="utf-8"
            )
            stale_stats = run_jobs(
                jobs_path=outputs["gpt_jobs"],
                manifest_path=outputs["gpt_manifest"],
                generate=fake_generate,
                retries=0,
                retry_delay=0,
            )
            self.assertEqual(stale_stats, {"generated": 1, "cached": 0, "failed": 0})
            self.assertEqual(len(calls), 2)

            cached_stats = run_jobs(
                jobs_path=outputs["gpt_jobs"],
                manifest_path=outputs["gpt_manifest"],
                generate=fake_generate,
                retries=0,
                retry_delay=0,
            )
            self.assertEqual(cached_stats, {"generated": 0, "cached": 1, "failed": 0})
            self.assertEqual(len(calls), 2)

            finalized = load_rollout_manifest(outputs["gpt_manifest"])
            full, error = resolve_rollout_images(
                {"question_uid": uid}, finalized, mode="picture", context_only=False
            )
            self.assertIsNone(error)
            self.assertEqual(len(full), 2)
            with Image.open(full[1].path) as generated:
                self.assertEqual(generated.size, (32, 24))

            cosmos_input = json.loads(
                next(outputs["cosmos_inputs"].glob("*.json")).read_text(encoding="utf-8")
            )
            self.assertEqual(cosmos_input["inference_type"], "image2world")
            self.assertEqual(set(cosmos_input), {"inference_type", "name", "prompt", "input_path"})

            cosmos_jobs = json.loads(outputs["cosmos_jobs"].read_text(encoding="utf-8"))
            video_path = Path(cosmos_jobs["entries"][0]["output_path"])
            video_path.parent.mkdir(parents=True, exist_ok=True)
            writer = cv2.VideoWriter(
                str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 8.0, (32, 24)
            )
            if not writer.isOpened():
                self.skipTest("OpenCV MP4 writer is unavailable")
            try:
                for frame_index in range(16):
                    frame = np.full((24, 32, 3), frame_index * 12, dtype=np.uint8)
                    writer.write(frame)
            finally:
                writer.release()

            video_stats = finalize_jobs(
                jobs_path=outputs["cosmos_jobs"],
                manifest_path=outputs["cosmos_manifest"],
                video_root=None,
            )
            self.assertEqual(video_stats, {"finalized": 1, "failed": 0})
            video_manifest = load_rollout_manifest(outputs["cosmos_manifest"])
            video_media, error = resolve_rollout_images(
                {"question_uid": uid},
                video_manifest,
                mode="video",
                context_only=False,
            )
            self.assertIsNone(error)
            self.assertEqual(len(video_media), 9)
            self.assertEqual(video_media[1].role, "predicted_video_frame:1/8")
            self.assertEqual(video_media[-1].role, "predicted_video_frame:8/8")

    def test_eight_frame_sampling_contains_unique_endpoints(self) -> None:
        indices = eight_frame_indices(120)
        self.assertEqual(len(indices), 8)
        self.assertEqual(indices[0], 0)
        self.assertEqual(indices[-1], 119)
        self.assertEqual(len(set(indices)), 8)


if __name__ == "__main__":
    unittest.main()
