import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image

from scripts.finalize_cosmos_rollouts import eight_frame_indices, finalize_jobs
import scripts.generate_rollout_selection_spec as selection_generator
import scripts.prepare_future_rollout_jobs as rollout_preparer
from scripts.generate_rollout_selection_spec import (
    QuestionMotion,
    RouteCandidate,
    SelectionPaths,
    _apply_question_motion,
    _materialize_route,
    _moving_group_ids,
    _select_midpoint_bridge,
    _static_visibility,
    generate_selection_spec,
)
from scripts.prepare_future_rollout_jobs import (
    action_duration_seconds,
    build_cosmos_prompt,
    build_picture_prompt,
    build_qwen_picture_prompt,
    build_safe_action,
    prepare_jobs,
    world_delta_to_agent_components,
)
from scripts.run_future_picture_generation import (
    RetryableGenerationError,
    _is_retryable,
    parse_args as parse_picture_generation_args,
    run_jobs,
)
from scripts.run_sampled_type_vlm_eval import load_rollout_manifest, resolve_rollout_images


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

    @staticmethod
    def _pose(position):
        return SimpleNamespace(position=np.asarray(position), rotation=np.eye(3))

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

    def test_midpoint_bridge_uses_cumulative_distance_and_earlier_tie(self) -> None:
        poses = {
            "source.jpg": self._pose([0.0, 0.0, 0.0]),
            "early.jpg": self._pose([2.0, 0.0, 0.0]),
            "late.jpg": self._pose([8.0, 0.0, 0.0]),
            "destination.jpg": self._pose([10.0, 0.0, 0.0]),
        }
        bridge = _select_midpoint_bridge(
            "source.jpg", ["early.jpg", "late.jpg"], "destination.jpg", poses
        )
        self.assertEqual(bridge, "early.jpg")
        self.assertIsNone(
            _select_midpoint_bridge("source.jpg", [], "destination.jpg", poses)
        )

    def test_route_evaluation_builds_two_or_three_image_inputs(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("source.jpg", "bridge.jpg", "destination.jpg"):
                Image.new("RGB", (32, 24), color=(30, 40, 50)).save(root / name)
            moved = self._object(1, [0.0, 0.0, 0.5], "cabinet")
            query = self._object(2, [1.0, 0.0, 0.5], "chair")
            context = SimpleNamespace(
                dataset="scannet",
                scene_id="scene0000_00",
                objects=[moved, query],
                objects_by_id={1: moved, 2: query},
                attachment_graph={},
                poses={
                    "source.jpg": self._pose([0, 0, 0]),
                    "bridge.jpg": self._pose([1, 0, 0]),
                    "destination.jpg": self._pose([2, 0, 0]),
                },
                data_source=SimpleNamespace(image_path=lambda name: root / name),
                image_quality_cache={},
                static_visibility_cache={},
                depth_frame_cache={},
                cache_stats=selection_generator.Counter(),
            )
            base_question = {
                "question_uid": "uid-1",
                "type": "object_move_agent",
                "scene_id": "scene0000_00",
                "image_name": "source.jpg",
                "reasoning_frame_2": "destination.jpg",
                "moved_obj_id": 1,
                "query_obj_id": 2,
                "delta": [0.5, 0.0, 0.0],
                "has_attachment_chain": False,
            }
            quality = {"readable": True, "laplacian_variance": 1.0, "tenengrad": 1.0}
            projection = {"projected_area_ratio": 0.5, "edge_margin_ratio": 0.25}
            with patch.object(
                selection_generator, "_cached_image_quality_metrics", return_value=quality
            ), patch.object(
                selection_generator,
                "_projection_gate",
                return_value=(False, projection),
            ), patch.object(
                selection_generator,
                "_cached_static_visibility",
                return_value=(False, 0.0, "depth", {}),
            ), patch.object(
                selection_generator, "_future_visibility", return_value=(False, 0.0, {})
            ):
                two, error = selection_generator._evaluate_question_route(
                    question=base_question, source_index=0, context=context
                )
                three, bridge_error = selection_generator._evaluate_question_route(
                    question={**base_question, "auxiliary_image_names": ["bridge.jpg"]},
                    source_index=0,
                    context=context,
                )
            self.assertIsNone(error)
            self.assertEqual(two.generation_roles, ("source_view", "destination_environment"))
            self.assertIsNone(bridge_error)
            self.assertEqual(
                three.generation_roles,
                (
                    "source_view",
                    "source_to_destination_bridge",
                    "destination_environment",
                ),
            )

    def test_materialization_copies_ordered_route_and_cosmos_context(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("source.jpg", "bridge.jpg", "destination.jpg"):
                Image.new("RGB", (16, 12), color=(20, 30, 40)).save(root / name)
            moved = self._object(1, [0.0, 0.0, 0.5], "table")
            query = self._object(2, [1.0, 0.0, 0.5], "chair")
            question = {
                "question_uid": "uid-1",
                "type": "object_move_agent",
                "moved_obj_id": 1,
                "query_obj_id": 2,
                "delta": [0.5, 0.0, 0.0],
            }
            candidate = RouteCandidate(
                question=question,
                source_index=4,
                dataset="scannet",
                scene_id="scene0000_00",
                generation_image_names=("source.jpg", "bridge.jpg", "destination.jpg"),
                generation_roles=(
                    "source_view",
                    "source_to_destination_bridge",
                    "destination_environment",
                ),
                score_key=(1.0, 1.0, 20.0),
                metrics={},
            )
            context = SimpleNamespace(
                objects=[moved, query],
                objects_by_id={1: moved, 2: query},
                attachment_graph={},
                poses={"source.jpg": self._pose([0, 0, 0])},
                data_source=SimpleNamespace(image_path=lambda name: root / name),
            )
            qwen_reference = {
                "path": str(root / "qwen_reference.png"),
                "role": "moving_group_reference",
                "sha256": "a" * 64,
            }
            with patch.object(
                selection_generator,
                "_materialize_qwen_moving_group_reference",
                return_value=qwen_reference,
            ):
                entry = _materialize_route(candidate, context, root / "rollout")
            self.assertEqual(
                [item["role"] for item in entry["generation_images"]],
                ["source_view", "source_to_destination_bridge", "destination_environment"],
            )
            self.assertEqual(
                [item["role"] for item in entry["answer_context_media"]],
                ["destination_to_query_bridge", "query_reference_view"],
            )
            self.assertEqual(entry["qwen_reference_image"], qwen_reference)
            self.assertTrue(all(Path(item["path"]).is_file() for item in entry["generation_images"]))

    def test_qwen_reference_is_a_padded_moving_group_crop(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.png"
            Image.new("RGB", (100, 80), color=(20, 30, 40)).save(source)
            moved = self._object(1, [0.0, 0.0, 0.5], "table")
            candidate = RouteCandidate(
                question={
                    "question_uid": "uid-1",
                    "type": "object_move_agent",
                    "moved_obj_id": 1,
                    "query_obj_id": 1,
                    "delta": [0.5, 0.0, 0.0],
                },
                source_index=0,
                dataset="scannet",
                scene_id="scene0000_00",
                generation_image_names=("source.png", "destination.png"),
                generation_roles=("source_view", "destination_environment"),
                score_key=(1.0,),
                metrics={},
            )
            context = SimpleNamespace(
                objects=[moved],
                objects_by_id={1: moved},
                attachment_graph={},
                poses={
                    "source.png": self._pose([0, 0, 0]),
                    "destination.png": self._pose([1, 0, 0]),
                },
                intrinsics=SimpleNamespace(width=100, height=80),
            )
            with patch.object(
                selection_generator,
                "_project_object_roi",
                return_value={"roi_bounds": (40, 60, 20, 40)},
            ):
                reference = selection_generator._materialize_qwen_moving_group_reference(
                    candidate, context, root / "rollout", source
                )
            self.assertEqual(reference["role"], "moving_group_reference")
            with Image.open(reference["path"]) as image:
                self.assertEqual(image.size, (44, 44))

    def test_qwen_reference_uses_destination_for_frame_2_group_member(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.png"
            Image.new("RGB", (100, 80), color=(20, 30, 40)).save(source)
            moved = self._object(1, [0.0, 0.0, 0.5], "table")
            attached = self._object(2, [0.0, 0.0, 1.0], "book")
            candidate = RouteCandidate(
                question={
                    "question_uid": "uid-1",
                    "type": "object_move_agent",
                    "moved_obj_id": 1,
                    "query_obj_id": 1,
                    "delta": [0.5, 0.0, 0.0],
                    "object_frame_groups": {"frame_1": [1], "frame_2": [2]},
                },
                source_index=0,
                dataset="scannet",
                scene_id="scene0000_00",
                generation_image_names=("source.png", "destination.png"),
                generation_roles=("source_view", "destination_environment"),
                score_key=(1.0,),
                metrics={},
            )
            context = SimpleNamespace(
                objects=[moved, attached],
                objects_by_id={1: moved, 2: attached},
                attachment_graph={1: [2]},
                poses={
                    "source.png": self._pose([0, 0, 0]),
                    "destination.png": self._pose([1, 0, 0]),
                },
                intrinsics=SimpleNamespace(width=100, height=80),
            )

            def project(obj, *_args):
                return {"roi_bounds": (40, 60, 20, 40) if obj["id"] == 1 else None}

            with patch.object(selection_generator, "_project_object_roi", side_effect=project):
                reference = selection_generator._materialize_qwen_moving_group_reference(
                    candidate, context, root / "rollout", source
                )
            self.assertEqual(reference["source_obj_ids"], [1])
            self.assertEqual(reference["destination_obj_ids"], [2])
            self.assertEqual(reference["unavailable_obj_ids"], [])
            with Image.open(reference["path"]) as image:
                self.assertEqual(image.size, (44, 44))

    def test_qwen_reference_falls_back_to_source_when_group_has_no_roi(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.png"
            Image.new("RGB", (100, 80), color=(20, 30, 40)).save(source)
            moved = self._object(1, [0.0, 0.0, 0.5], "table")
            candidate = RouteCandidate(
                question={
                    "question_uid": "uid-1",
                    "type": "object_move_agent",
                    "moved_obj_id": 1,
                    "query_obj_id": 1,
                    "delta": [0.5, 0.0, 0.0],
                },
                source_index=0,
                dataset="scannet",
                scene_id="scene0000_00",
                generation_image_names=("source.png", "destination.png"),
                generation_roles=("source_view", "destination_environment"),
                score_key=(1.0,),
                metrics={},
            )
            context = SimpleNamespace(
                objects=[moved],
                objects_by_id={1: moved},
                attachment_graph={},
                poses={
                    "source.png": self._pose([0, 0, 0]),
                    "destination.png": self._pose([1, 0, 0]),
                },
                intrinsics=SimpleNamespace(width=100, height=80),
            )
            with patch.object(
                selection_generator, "_project_object_roi", return_value={"roi_bounds": None}
            ):
                reference = selection_generator._materialize_qwen_moving_group_reference(
                    candidate, context, root / "rollout", source
                )
            self.assertEqual(reference["source_obj_ids"], [])
            self.assertEqual(reference["destination_obj_ids"], [])
            self.assertEqual(reference["unavailable_obj_ids"], [1])
            with Image.open(reference["path"]) as image:
                self.assertEqual(image.size, (100, 80))

    def test_materialization_records_cross_frame_visual_reference_roles(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("source.png", "destination.png"):
                Image.new("RGB", (100, 80), color=(20, 30, 40)).save(root / name)
            moved = self._object(1, [0.0, 0.0, 0.5], "counter")
            attached = self._object(2, [0.0, 0.0, 1.0], "pot cover")
            candidate = RouteCandidate(
                question={
                    "question_uid": "uid-cross-frame",
                    "type": "object_move_object_centric",
                    "moved_obj_id": 1,
                    "query_obj_id": 1,
                    "delta": [0.5, 0.0, 0.0],
                    "has_attachment_chain": True,
                    "object_frame_groups": {"frame_1": [1], "frame_2": [2]},
                },
                source_index=0,
                dataset="scannetpp",
                scene_id="scene-id",
                generation_image_names=("source.png", "destination.png"),
                generation_roles=("source_view", "destination_environment"),
                score_key=(1.0,),
                metrics={},
            )
            context = SimpleNamespace(
                objects=[moved, attached],
                objects_by_id={1: moved, 2: attached},
                attachment_graph={1: [2]},
                poses={
                    "source.png": self._pose([0, 0, 0]),
                    "destination.png": self._pose([1, 0, 0]),
                },
                intrinsics=SimpleNamespace(width=100, height=80),
                data_source=SimpleNamespace(image_path=lambda name: root / name),
            )
            with patch.object(
                selection_generator,
                "_project_object_roi",
                return_value={"roi_bounds": (40, 60, 20, 40)},
            ):
                entry = _materialize_route(candidate, context, root / "rollout")
            self.assertEqual(
                [item["visual_reference_role"] for item in entry["moving_group"]],
                ["moving_group_reference", "destination_environment"],
            )
            self.assertEqual(entry["qwen_reference_image"]["source_obj_ids"], [1])
            self.assertEqual(entry["qwen_reference_image"]["destination_obj_ids"], [2])
            self.assertTrue(entry["qwen_picture_eligible"])
            self.assertEqual(entry["qwen_picture_rejection_reasons"], [])

    def test_qwen_prompt_identifies_cross_frame_group_members(self) -> None:
        question = {
            "type": "object_move_agent",
            "moved_obj_id": 1,
            "moved_obj_label": "counter",
            "query_obj_id": 1,
            "delta": [0.5, 0.0, 0.0],
            "has_attachment_chain": True,
            "attachment_parent_id": 1,
            "attachment_child_id": 2,
        }
        spec = {
            "camera_rotation_world_to_camera": self.CAMERA_ROTATION_WORLD_TO_CAMERA,
            "moving_group": [
                {
                    "obj_id": 1,
                    "label": "counter",
                    "visual_reference_role": "moving_group_reference",
                },
                {
                    "obj_id": 2,
                    "label": "pot cover",
                    "visual_reference_role": "destination_environment",
                },
            ],
        }
        action = build_safe_action(question, spec)
        prompt = build_qwen_picture_prompt(action)
        self.assertIn('Picture 1 visually references these group members: "counter".', prompt)
        self.assertIn(
            'Picture 2 itself visually references these additional group members at their original locations: "pot cover".',
            prompt,
        )
        self.assertIn("including members referenced only in Picture 2", prompt)

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
            selection_generator, "_compute_mesh_ray_l1_occlusion_metrics_for_static_target"
        ) as mesh_visibility:
            visible, ratio, backend, _ = _static_visibility(
                obj, pose=object(), context=context, depth_frame=depth_frame
            )
        self.assertFalse(visible)
        self.assertEqual(ratio, 0.25)
        self.assertEqual(backend, "depth")
        mesh_visibility.assert_not_called()

    def test_selection_shortage_writes_v2_spec_without_loading_scenes(self) -> None:
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
                )
            payload = json.loads(paths.spec.read_text(encoding="utf-8"))
            self.assertEqual(payload["schema_version"], "predictive-spatial-selection-v2")
            self.assertEqual(payload["metadata"]["selection_mode"], "source_destination_route")

    def test_old_selector_checkpoint_configuration_is_not_reused(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            benchmark = root / "benchmark.json"
            benchmark.write_text('{"questions": []}', encoding="utf-8")
            old_configuration = {
                "schema_version": "predictive-spatial-selection-checkpoint-v1",
                "algorithm_version": "old-single-frame",
                "benchmark_sha256": selection_generator._sha256_file(benchmark),
            }
            old_path = selection_generator._checkpoint_path(root / "rollout", old_configuration)
            old_path.parent.mkdir(parents=True)
            old_path.write_text(json.dumps({"kind": "header", "configuration": old_configuration}) + "\n")
            with patch.object(selection_generator, "load_fixed_questions", return_value=([], {}, {})):
                generate_selection_spec(
                    benchmark_path=benchmark,
                    output_dir=root / "rollout",
                    scannet_root=None,
                    scannetpp_root=None,
                    scannetpp_frame_root=None,
                )
            checkpoints = list(old_path.parent.glob("selection_checkpoint_*.jsonl"))
            self.assertEqual(len(checkpoints), 2)

    def test_checkpoint_match_tolerates_derived_uid_changes(self) -> None:
        record = {
            "source_index": 7,
            "question_uid": "old-derived-uid",
            "question_type": "object_move_distance",
            "scene_id": "scene0000_00",
        }
        question = {
            "question_uid": "new-derived-uid",
            "type": "object_move_distance",
            "scene_id": "scene0000_00",
        }
        self.assertTrue(
            selection_generator._checkpoint_record_matches_question(
                record, source_index=7, question=question
            )
        )
        self.assertFalse(
            selection_generator._checkpoint_record_matches_question(
                record,
                source_index=7,
                question={**question, "scene_id": "scene0001_00"},
            )
        )

    def test_prepare_cli_removes_selector_tuning_and_invokes_route_selection(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            benchmark = root / "benchmark.json"
            benchmark.write_text('{"questions": []}', encoding="utf-8")
            args = rollout_preparer.parse_args(
                ["--benchmark_file", str(benchmark), "--output_dir", str(root / "rollout")]
            )
            self.assertEqual(args.expected_picture_per_type, 2)
            with self.assertRaises(SystemExit):
                rollout_preparer.parse_args(
                    [
                        "--benchmark_file",
                        str(benchmark),
                        "--output_dir",
                        str(root / "rollout"),
                        "--frame_stride_scannet",
                        "10",
                    ]
                )
            paths = SelectionPaths(
                spec=root / "selection.json", audit=root / "audit.json"
            )
            with patch.object(
                rollout_preparer, "generate_selection_spec", return_value=paths
            ) as generate, patch.object(
                rollout_preparer, "prepare_jobs", return_value={"qwen_jobs": root / "qwen.json"}
            ):
                rollout_preparer.main(
                    [
                        "--benchmark_file",
                        str(benchmark),
                        "--output_dir",
                        str(root / "rollout"),
                    ]
                )
            self.assertNotIn("frame_stride_scannet", generate.call_args.kwargs)
            self.assertEqual(generate.call_args.kwargs["expected_per_type"], 2)

    def test_world_delta_is_converted_to_camera_agent_components(self) -> None:
        components = world_delta_to_agent_components(
            [1.0, 0.0, 2.0], self.CAMERA_ROTATION_WORLD_TO_CAMERA
        )
        self.assertAlmostEqual(components["right_m"], 1.0)
        self.assertAlmostEqual(components["away_m"], 0.0)
        self.assertAlmostEqual(components["up_m"], 2.0)

    def test_prompts_include_route_roles_but_not_question_answer(self) -> None:
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
        spec = {"camera_rotation_world_to_camera": self.CAMERA_ROTATION_WORLD_TO_CAMERA}
        action = build_safe_action(question, spec)
        picture_prompt = build_picture_prompt(
            action,
            ["source_view", "source_to_destination_bridge", "destination_environment"],
        )
        video_prompt = build_cosmos_prompt(action)
        self.assertIn("Image 2 is a midpoint bridge", picture_prompt)
        self.assertIn("Image 3 is the destination environment", picture_prompt)
        for prompt in (picture_prompt, video_prompt):
            self.assertIn('"cabinet"', prompt)
            self.assertNotIn("secret target", prompt)
            self.assertNotIn("occluded", prompt.lower())

    def test_duration_and_retry_policy(self) -> None:
        durations = [action_duration_seconds(distance) for distance in (0.1, 1.0, 2.0, 10.0)]
        self.assertEqual(durations[0], 2.0)
        self.assertEqual(durations[-1], 5.0)
        self.assertEqual(durations, sorted(durations))
        self.assertTrue(_is_retryable(RuntimeError("request timed out")))
        self.assertTrue(_is_retryable(RetryableGenerationError("empty response")))
        self.assertFalse(_is_retryable(RuntimeError("invalid API key")))

    def test_qwen_performance_options_default_to_full_gpu_mode(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            jobs = root / "jobs.json"
            manifest = root / "manifest.json"
            jobs.write_text("{}", encoding="utf-8")
            manifest.write_text("{}", encoding="utf-8")
            defaults = parse_picture_generation_args(
                ["--backend", "qwen", "--jobs", str(jobs), "--manifest", str(manifest)]
            )
            tuned = parse_picture_generation_args(
                [
                    "--backend",
                    "qwen",
                    "--jobs",
                    str(jobs),
                    "--manifest",
                    str(manifest),
                    "--qwen_num_inference_steps",
                    "28",
                    "--qwen_vae_tiling",
                ]
            )
        self.assertFalse(defaults.qwen_cpu_offload)
        self.assertFalse(defaults.qwen_vae_tiling)
        self.assertEqual(defaults.qwen_num_inference_steps, 40)
        self.assertEqual(tuned.qwen_num_inference_steps, 28)
        self.assertTrue(tuned.qwen_vae_tiling)

    def test_prepare_and_fake_picture_generation_end_to_end(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.png"
            bridge = root / "bridge.png"
            destination = root / "destination.png"
            qwen_reference = root / "qwen_reference.png"
            Image.new("RGB", (32, 24), color=(10, 20, 30)).save(source)
            Image.new("RGB", (40, 30), color=(20, 30, 40)).save(bridge)
            Image.new("RGB", (48, 36), color=(30, 40, 50)).save(destination)
            Image.new("RGB", (12, 10), color=(25, 35, 45)).save(qwen_reference)
            benchmark_path = root / "benchmark.json"
            benchmark_path.write_text(
                json.dumps(
                    {
                        "questions": [
                            {
                                "level": "L2",
                                "type": "object_move_agent",
                                "scene_id": "scene0000_00",
                                "image_name": "source.png",
                                "reasoning_frame_2": "destination.png",
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
                                "generation_images": [
                                    {"path": str(source), "role": "source_view"},
                                    {
                                        "path": str(bridge),
                                        "role": "source_to_destination_bridge",
                                    },
                                    {
                                        "path": str(destination),
                                        "role": "destination_environment",
                                    },
                                ],
                                "qwen_reference_image": {
                                    "path": str(qwen_reference),
                                    "role": "moving_group_reference",
                                    "sha256": selection_generator._sha256_file(qwen_reference),
                                },
                                "camera_rotation_world_to_camera": self.CAMERA_ROTATION_WORLD_TO_CAMERA,
                                "picture_eligible": True,
                                "video_eligible": True,
                                "answer_context_media": [
                                    {
                                        "path": str(bridge),
                                        "role": "destination_to_query_bridge",
                                    },
                                    {
                                        "path": str(destination),
                                        "role": "query_reference_view",
                                    },
                                ],
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
            gpt_job = gpt_jobs["entries"][0]
            qwen_job = qwen_jobs["entries"][0]
            self.assertEqual(
                [item["role"] for item in qwen_job["input_images"]],
                ["moving_group_reference", "destination_environment"],
            )
            self.assertEqual(qwen_job["input_images"][-1], gpt_job["input_images"][-1])
            self.assertIn("Never make a collage", qwen_job["prompt"])
            self.assertNotEqual(gpt_job["prompt"], qwen_job["prompt"])
            cosmos_jobs_before = outputs["cosmos_jobs"].read_bytes()
            cosmos_manifest_before = outputs["cosmos_manifest"].read_bytes()
            qwen_only_outputs = prepare_jobs(
                benchmark_path=benchmark_path,
                selection_spec_path=selection_path,
                output_dir=root / "rollouts",
                seed=7,
                expected_picture_per_type=50,
                generation_backends=frozenset({"qwen"}),
            )
            self.assertEqual(set(qwen_only_outputs), {"qwen_jobs", "qwen_manifest"})
            self.assertEqual(outputs["cosmos_jobs"].read_bytes(), cosmos_jobs_before)
            self.assertEqual(outputs["cosmos_manifest"].read_bytes(), cosmos_manifest_before)
            self.assertNotIn("input_image_path", gpt_job)
            self.assertNotIn("where is the chair", gpt_job["prompt"].lower())

            ineligible_selection = root / "private_selection_qwen_ineligible.json"
            ineligible_payload = json.loads(selection_path.read_text(encoding="utf-8"))
            ineligible_entry = ineligible_payload["entries"][0]
            ineligible_entry.pop("qwen_reference_image")
            ineligible_entry["qwen_picture_eligible"] = False
            ineligible_entry["qwen_picture_rejection_reasons"] = [
                "moving_group_without_visual_reference:2"
            ]
            ineligible_selection.write_text(
                json.dumps(ineligible_payload), encoding="utf-8"
            )
            ineligible_outputs = prepare_jobs(
                benchmark_path=benchmark_path,
                selection_spec_path=ineligible_selection,
                output_dir=root / "rollouts_qwen_ineligible",
                seed=7,
                expected_picture_per_type=50,
                generation_backends=frozenset({"qwen", "gpt"}),
            )
            ineligible_jobs = json.loads(
                ineligible_outputs["qwen_jobs"].read_text(encoding="utf-8")
            )
            eligible_gpt_jobs = json.loads(
                ineligible_outputs["gpt_jobs"].read_text(encoding="utf-8")
            )
            ineligible_manifest = json.loads(
                ineligible_outputs["qwen_manifest"].read_text(encoding="utf-8")
            )
            self.assertEqual(ineligible_jobs["entries"], [])
            self.assertEqual(len(eligible_gpt_jobs["entries"]), 1)
            self.assertFalse(ineligible_manifest["entries"][0]["picture"]["eligible"])
            self.assertEqual(
                ineligible_manifest["entries"][0]["picture"]["rejection_reasons"],
                ["moving_group_without_visual_reference:2"],
            )

            manifest = load_rollout_manifest(outputs["gpt_manifest"])
            uid = manifest.entry_order[0]
            context, error = resolve_rollout_images(
                {"question_uid": uid}, manifest, mode="picture", context_only=True
            )
            self.assertIsNone(error)
            self.assertEqual(
                [item.role for item in context],
                ["source_view", "source_to_destination_bridge", "destination_environment"],
            )

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
            self.assertEqual(stats, {"generated": 1, "cached": 0, "failed": 0})
            finalized = load_rollout_manifest(outputs["gpt_manifest"])
            full, error = resolve_rollout_images(
                {"question_uid": uid}, finalized, mode="picture", context_only=False
            )
            self.assertIsNone(error)
            self.assertEqual(len(full), 4)
            with Image.open(full[-1].path) as generated:
                self.assertEqual(generated.size, (48, 36))

            qwen_stats = run_jobs(
                jobs_path=outputs["qwen_jobs"],
                manifest_path=outputs["qwen_manifest"],
                generate=fake_generate,
                retries=0,
                retry_delay=0,
            )
            self.assertEqual(qwen_stats, {"generated": 1, "cached": 0, "failed": 0})

            cached = run_jobs(
                jobs_path=outputs["gpt_jobs"],
                manifest_path=outputs["gpt_manifest"],
                generate=fake_generate,
                retries=0,
                retry_delay=0,
            )
            self.assertEqual(cached, {"generated": 0, "cached": 1, "failed": 0})

            prepared_again = prepare_jobs(
                benchmark_path=benchmark_path,
                selection_spec_path=selection_path,
                output_dir=root / "rollouts",
                seed=7,
                expected_picture_per_type=50,
            )
            cached_after_prepare = run_jobs(
                jobs_path=prepared_again["gpt_jobs"],
                manifest_path=prepared_again["gpt_manifest"],
                generate=fake_generate,
                retries=0,
                retry_delay=0,
            )
            self.assertEqual(
                cached_after_prepare, {"generated": 0, "cached": 1, "failed": 0}
            )
            self.assertEqual(calls.count(uid), 2)

            cosmos_input = json.loads(
                next(outputs["cosmos_inputs"].glob("*.json")).read_text(encoding="utf-8")
            )
            self.assertEqual(set(cosmos_input), {"inference_type", "name", "prompt", "input_path"})
            self.assertEqual(Path(cosmos_input["input_path"]), source.resolve())
            cosmos_jobs = json.loads(outputs["cosmos_jobs"].read_text(encoding="utf-8"))
            cosmos_job = cosmos_jobs["entries"][0]
            self.assertEqual(Path(cosmos_job["input_image_path"]), source.resolve())
            self.assertNotIn("input_images", cosmos_job)

            video_path = Path(cosmos_job["output_path"])
            video_path.parent.mkdir(parents=True, exist_ok=True)
            writer = cv2.VideoWriter(
                str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 8.0, (32, 24)
            )
            if not writer.isOpened():
                self.skipTest("OpenCV MP4 writer is unavailable")
            try:
                for frame_index in range(16):
                    writer.write(np.full((24, 32, 3), frame_index * 12, dtype=np.uint8))
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
                {"question_uid": uid}, video_manifest, mode="video", context_only=False
            )
            self.assertIsNone(error)
            self.assertEqual(len(video_media), 11)

    def test_eight_frame_sampling_contains_unique_endpoints(self) -> None:
        indices = eight_frame_indices(120)
        self.assertEqual(len(indices), 8)
        self.assertEqual(indices[0], 0)
        self.assertEqual(indices[-1], 119)
        self.assertEqual(len(set(indices)), 8)


if __name__ == "__main__":
    unittest.main()
