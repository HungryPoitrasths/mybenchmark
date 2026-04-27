import json
import shutil
import sys
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import call, patch
import uuid

import numpy as np

import scripts.run_vlm_referability as referability_module
from src.utils.colmap_loader import CameraIntrinsics, CameraPose


def make_camera_pose() -> CameraPose:
    return CameraPose(
        image_name="000000.jpg",
        rotation=np.eye(3, dtype=np.float64),
        translation=np.zeros(3, dtype=np.float64),
    )


def make_camera_intrinsics(width: int = 120, height: int = 120) -> CameraIntrinsics:
    return CameraIntrinsics(
        width=width,
        height=height,
        fx=100.0,
        fy=100.0,
        cx=width / 2.0,
        cy=height / 2.0,
    )


def make_object(
    obj_id: int,
    label: str,
    *,
    alias_group: str | None = None,
    alias_variants: list[str] | None = None,
) -> dict:
    alias_group_name = alias_group or f"{label.replace(' ', '_')}_family"
    variants = alias_variants or [label]
    base = float(obj_id)
    return {
        "id": obj_id,
        "label": label,
        "raw_label": label,
        "canonical_label": label,
        "alias_group": alias_group_name,
        "alias_variants": list(variants),
        "center": [base, 0.0, 2.0],
        "bbox_min": [base - 0.2, -0.2, 1.8],
        "bbox_max": [base + 0.2, 0.2, 2.2],
    }


def make_visibility_meta(
    *,
    projected_area_px: float = 900.0,
    bbox_in_frame_ratio: float = 0.9,
    zbuffer_mask_area_px: float | None = None,
    has_zbuffer_mask_area: bool | None = None,
) -> dict:
    zbuffer_area = 0.0 if zbuffer_mask_area_px is None else float(zbuffer_mask_area_px)
    has_zbuffer = (
        bool(zbuffer_mask_area_px is not None)
        if has_zbuffer_mask_area is None
        else bool(has_zbuffer_mask_area)
    )
    return {
        "roi_bounds_px": [20, 60, 20, 60],
        "projected_area_px": projected_area_px,
        "bbox_in_frame_ratio": bbox_in_frame_ratio,
        "edge_margin_px": 10.0,
        "zbuffer_mask_area_px": zbuffer_area,
        "has_zbuffer_mask_area": has_zbuffer,
    }


def make_detection(
    *,
    bbox: tuple[int, int, int, int],
    score: float,
    image_shape: tuple[int, int] = (120, 120),
    category: str = "object",
) -> dict:
    height, width = image_shape
    x0, y0, x1, y1 = bbox
    mask = np.zeros((height, width), dtype=bool)
    mask[max(0, y0):min(height, y1), max(0, x0):min(width, x1)] = True
    return {
        "bbox": [float(x0), float(y0), float(x1), float(y1)],
        "mask": mask,
        "score": float(score),
        "area_px": int(mask.sum()),
        "category": category,
    }


def make_strip_instance_mesh_data(*, obj_id: int = 1, cells: int = 10) -> SimpleNamespace:
    vertices: list[list[float]] = []
    for x in range(cells + 1):
        for y in range(2):
            vertices.append([float(x), float(y), 2.0])

    def vertex_id(x: int, y: int) -> int:
        return x * 2 + y

    faces: list[list[int]] = []
    for x in range(cells):
        v00 = vertex_id(x, 0)
        v01 = vertex_id(x, 1)
        v10 = vertex_id(x + 1, 0)
        v11 = vertex_id(x + 1, 1)
        faces.append([v00, v01, v11])
        faces.append([v00, v11, v10])

    face_array = np.asarray(faces, dtype=np.int64)
    return SimpleNamespace(
        vertices=np.asarray(vertices, dtype=np.float64),
        faces=face_array,
        triangle_ids_by_instance={int(obj_id): np.arange(len(face_array), dtype=np.int64)},
        boundary_triangle_ids_by_instance={},
        surface_points_by_instance={},
        surface_triangle_ids_by_instance={},
        surface_barycentrics_by_instance={},
    )


def make_topology_quality(obj_id: int, status: str = "pass") -> dict:
    return {
        "obj_id": int(obj_id),
        "triangle_count": 32,
        "connected_component_count": 1,
        "largest_component_triangle_share": 1.0,
        "boundary_edge_ratio": 0.20,
        "num_boundary_loops": 1,
        "largest_boundary_loop_edge_share": 0.20,
        "status": status,
        "reason_codes": [] if status == "pass" else ["warn_flag"],
    }


def make_mesh_quality(obj_id: int, status: str = "pass", *, reason_codes: list[str] | None = None) -> dict:
    return {
        "obj_id": int(obj_id),
        "status": status,
        "profile": "topology_pass_base",
        "image_mask_area_px": 900,
        "mesh_mask_area_px": 880,
        "intersection_px": 800,
        "union_px": 980,
        "iou": 0.82 if status == "pass" else 0.20,
        "under_coverage": 0.11 if status == "pass" else 0.60,
        "over_coverage": 0.09 if status == "pass" else 0.32,
        "area_ratio": 0.98 if status == "pass" else 2.10,
        "depth_bad_ratio": 0.05 if status == "pass" else 0.40,
        "reason_codes": list(reason_codes or ([] if status == "pass" else ["low_iou"])),
        "thresholds": referability_module._mesh_quality_thresholds_for_topology_status("pass"),
    }


def make_instance_mesh_data(
    *,
    obj_id: int,
    sample_count: int = 8,
    point: np.ndarray | None = None,
):
    base_point = (
        np.asarray(point, dtype=np.float64)
        if point is not None
        else np.array([0.0, 0.0, 2.0], dtype=np.float64)
    )
    surface_points = np.tile(base_point[None, :], (int(sample_count), 1))
    triangle_ids = np.zeros(int(sample_count), dtype=np.int64)
    barycentrics = np.tile(
        np.array([[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]], dtype=np.float64),
        (int(sample_count), 1),
    )
    return SimpleNamespace(
        vertices=np.array(
            [
                [0.0, 0.0, 2.0],
                [0.5, 0.0, 2.0],
                [0.0, 0.5, 2.0],
            ],
            dtype=np.float64,
        ),
        faces=np.array([[0, 1, 2]], dtype=np.int64),
        triangle_ids_by_instance={int(obj_id): np.array([0], dtype=np.int64)},
        boundary_triangle_ids_by_instance={},
        surface_points_by_instance={int(obj_id): surface_points},
        surface_triangle_ids_by_instance={int(obj_id): triangle_ids},
        surface_barycentrics_by_instance={int(obj_id): barycentrics},
    )


def make_debug_cache_entry() -> dict:
    return {
        "frame_usable": True,
        "frame_quality_clear": True,
        "frame_quality_score": 82,
        "frame_quality_reason": "clear enough",
        "frame_selection_score": 100082,
        "attachment_selector_signal": {
            "well_cropped_pair_count": 0,
            "viewpoint_exempt": False,
        },
        "attachment_referable_pairs": [],
        "attachment_referable_pair_count": 0,
        "attachment_final_referability": {
            "object_ids": [],
            "pairs": [],
            "pair_count": 0,
        },
        "final_selection_rank": 0,
        "attachment_final_frame_selection": {
            "selected_for_final_cache": True,
            "selection_rank": 0,
        },
        "candidate_visible_object_ids": [],
        "candidate_visibility_source": "selector_visible_object_ids",
        "candidate_labels": [],
        "label_to_object_ids": {},
        "selector_visible_object_ids": [],
        "selector_visible_label_counts": {},
        "visibility_audit_by_object_id": {},
        "object_reviews": {},
        "crop_label_statuses": {},
        "crop_label_counts": {},
        "crop_referable_object_ids": [],
        "full_frame_label_reviews": [],
        "full_frame_label_statuses": {},
        "full_frame_label_counts": {},
        "label_statuses": {},
        "label_counts": {},
        "out_of_frame_label_reviews": [],
        "out_of_frame_not_visible_labels": [],
        "out_of_frame_label_to_object_ids": {},
        "out_of_frame_vlm_early_stop": False,
        "referable_object_ids": [],
    }


def make_scene_dir(root: Path, relative_path: str) -> Path:
    scene_dir = root / relative_path
    (scene_dir / "pose").mkdir(parents=True, exist_ok=True)
    return scene_dir


def make_fake_openai_module(model_id: str = "fake-vlm") -> SimpleNamespace:
    fake_client = SimpleNamespace(
        models=SimpleNamespace(
            list=lambda: SimpleNamespace(data=[SimpleNamespace(id=model_id)])
        )
    )
    return SimpleNamespace(
        OpenAI=lambda api_key, base_url: fake_client
    )


class _SequenceVisibilityCaster:
    def __init__(self, responses: list[tuple[int, int]]) -> None:
        self._responses = [tuple((int(visible), int(valid))) for visible, valid in responses]

    def mesh_visibility_stats(
        self,
        camera_pos,
        target_points,
        target_tri_ids,
        **kwargs,
    ):
        _ = np.asarray(camera_pos, dtype=np.float64)
        _ = np.asarray(target_points, dtype=np.float64)
        _ = set(int(tri_id) for tri_id in target_tri_ids)
        if not self._responses:
            raise AssertionError("mesh_visibility_stats called more times than expected")
        return self._responses.pop(0)


class RunVlmReferabilityTests(unittest.TestCase):
    def test_resolve_scannet_scene_dirs_reads_train_from_data_root(self) -> None:
        root = Path(__file__).resolve().parent / "_tmp" / f"resolve_train_{uuid.uuid4().hex}"
        data_root = root / "data"
        make_scene_dir(data_root, "scans/scene0002_00")
        make_scene_dir(data_root, "scans/scene0001_00")
        make_scene_dir(data_root, "scans_test/scene1000_00")
        self.addCleanup(shutil.rmtree, root, True)

        entries = referability_module._resolve_scannet_scene_dirs(data_root, "train")

        self.assertEqual(
            [(split, path.name) for split, path in entries],
            [("train", "scene0001_00"), ("train", "scene0002_00")],
        )

    def test_resolve_scannet_scene_dirs_reads_test_from_scans_root(self) -> None:
        root = Path(__file__).resolve().parent / "_tmp" / f"resolve_test_{uuid.uuid4().hex}"
        data_root = root / "data"
        scans_root = data_root / "scans"
        make_scene_dir(data_root, "scans/scene0001_00")
        make_scene_dir(data_root, "scans_test/scene1001_00")
        self.addCleanup(shutil.rmtree, root, True)

        entries = referability_module._resolve_scannet_scene_dirs(scans_root, "test")

        self.assertEqual(
            [(split, path.name) for split, path in entries],
            [("test", "scene1001_00")],
        )

    def test_resolve_scannet_scene_dirs_reads_all_in_train_then_test_order(self) -> None:
        root = Path(__file__).resolve().parent / "_tmp" / f"resolve_all_{uuid.uuid4().hex}"
        data_root = root / "data"
        scans_test_root = data_root / "scans_test"
        make_scene_dir(data_root, "scans/scene0002_00")
        make_scene_dir(data_root, "scans/scene0001_00")
        make_scene_dir(data_root, "scans_test/scene1001_00")
        make_scene_dir(data_root, "scans_test/scene1000_00")
        self.addCleanup(shutil.rmtree, root, True)

        entries = referability_module._resolve_scannet_scene_dirs(scans_test_root, "all")

        self.assertEqual(
            [(split, path.name) for split, path in entries],
            [
                ("train", "scene0001_00"),
                ("train", "scene0002_00"),
                ("test", "scene1000_00"),
                ("test", "scene1001_00"),
            ],
        )

    def test_call_vlm_json_tracks_failure_count(self) -> None:
        referability_module._reset_vlm_call_failure_count()
        client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(
                    create=lambda **kwargs: (_ for _ in ()).throw(RuntimeError("boom"))
                )
            )
        )

        parsed, raw_text = referability_module._call_vlm_json(
            client,
            "fake-vlm",
            [],
            {"status": "unsure"},
        )

        self.assertEqual(parsed, {"status": "unsure"})
        self.assertEqual(raw_text, "")
        self.assertEqual(referability_module._get_vlm_call_failure_count(), 1)
        referability_module._reset_vlm_call_failure_count()

    def test_run_in_thread_pool_preserves_input_order(self) -> None:
        def work_item(value: int) -> int:
            time.sleep(0.01 * (4 - value))
            return value * 10

        results = referability_module._run_in_thread_pool(
            [1, 2, 3],
            work_item,
            max_workers=3,
        )

        self.assertEqual(results, [10, 20, 30])

    def test_topology_warn_triggers_when_any_warn_condition_is_met(self) -> None:
        quality = referability_module._compute_topology_quality_for_object(
            obj_id=1,
            instance_mesh_data=make_strip_instance_mesh_data(obj_id=1, cells=10),
        )

        self.assertEqual(quality["status"], "warn")
        self.assertIn("boundary_edge_ratio_warn", quality["reason_codes"])
        self.assertNotIn("component_count_warn", quality["reason_codes"])

    def test_mesh_mask_quality_requires_iou_and_under_coverage(self) -> None:
        detection_mask = np.zeros((20, 20), dtype=bool)
        detection_mask[0:10, 0:10] = True
        rendered_mask = np.zeros((20, 20), dtype=bool)
        rendered_mask[8:18, 8:18] = True
        rendered_depth = np.where(rendered_mask, 2.0, np.inf).astype(np.float32)

        with (
            patch.object(
                referability_module,
                "_rasterize_instance_depth_map",
                return_value={"mask": rendered_mask, "depth": rendered_depth, "triangle_count": 20},
            ),
            patch.object(
                referability_module,
                "_compute_depth_bad_ratio",
                return_value=0.0,
            ),
        ):
            quality = referability_module._compute_mesh_mask_quality_for_object(
                obj_id=1,
                detection_mask=detection_mask,
                topology_status="pass",
                camera_pose=make_camera_pose(),
                color_intrinsics=make_camera_intrinsics(20, 20),
                depth_image=None,
                depth_intrinsics=None,
                instance_mesh_data=None,
            )

        self.assertEqual(quality["status"], "fail")
        self.assertIn("low_iou", quality["reason_codes"])
        self.assertIn("high_under_coverage", quality["reason_codes"])

    def test_build_object_review_crop_requires_projected_area_of_at_least_800px(self) -> None:
        tiny_crop = referability_module._build_object_review_crop(
            np.zeros((120, 120, 3), dtype=np.uint8),
            make_visibility_meta(projected_area_px=799.0, bbox_in_frame_ratio=0.1),
        )
        valid_crop = referability_module._build_object_review_crop(
            np.zeros((120, 120, 3), dtype=np.uint8),
            make_visibility_meta(projected_area_px=800.0, bbox_in_frame_ratio=0.1),
        )

        self.assertEqual(tiny_crop["local_outcome"], "excluded")
        self.assertEqual(tiny_crop["reason"], "projected_area_too_small")
        self.assertTrue(valid_crop["valid"])
        self.assertEqual(valid_crop["local_outcome"], "reviewed")

    def test_build_object_review_crop_does_not_gate_on_zbuffer_mask_area(self) -> None:
        crop = referability_module._build_object_review_crop(
            np.zeros((120, 120, 3), dtype=np.uint8),
            make_visibility_meta(projected_area_px=900.0, zbuffer_mask_area_px=1.0),
        )

        self.assertTrue(crop["valid"])
        self.assertEqual(crop["local_outcome"], "reviewed")

    def test_refine_candidate_visible_object_ids_uses_mesh_ray_without_depth(self) -> None:
        candidate_ids, source = referability_module._refine_candidate_visible_object_ids(
            [1],
            [make_object(1, "chair")],
            make_camera_pose(),
            make_camera_intrinsics(),
            None,
            None,
            ray_caster_getter=lambda: _SequenceVisibilityCaster([(1, 4)]),
            instance_mesh_data_getter=lambda _base: make_instance_mesh_data(obj_id=1, sample_count=8),
        )
        self.assertEqual(candidate_ids, [1])
        self.assertEqual(source, "mesh_ray_refined")

    def test_refine_candidate_visible_object_ids_drops_when_mesh_ray_rejects(self) -> None:
        candidate_ids, source = referability_module._refine_candidate_visible_object_ids(
            [1],
            [make_object(1, "chair")],
            make_camera_pose(),
            make_camera_intrinsics(),
            np.ones((4, 4), dtype=np.float32),
            make_camera_intrinsics(),
            ray_caster_getter=lambda: _SequenceVisibilityCaster([(0, 4), (0, 4)]),
            instance_mesh_data_getter=lambda _base: make_instance_mesh_data(obj_id=1, sample_count=8),
        )
        self.assertEqual(candidate_ids, [])
        self.assertEqual(source, "mesh_ray_refined")

    def test_refine_candidate_visible_object_ids_uses_stage2_when_stage1_ratio_is_too_low(self) -> None:
        caster = _SequenceVisibilityCaster([(1, 20), (2, 8)])
        candidate_ids, source = referability_module._refine_candidate_visible_object_ids(
            [1],
            [make_object(1, "chair")],
            make_camera_pose(),
            make_camera_intrinsics(),
            np.ones((4, 4), dtype=np.float32),
            make_camera_intrinsics(),
            ray_caster_getter=lambda: caster,
            instance_mesh_data_getter=lambda _base: make_instance_mesh_data(obj_id=1, sample_count=8),
        )
        self.assertEqual(candidate_ids, [1])
        self.assertEqual(source, "mesh_ray_refined")
        self.assertEqual(caster._responses, [])

    def test_refine_candidate_visible_object_ids_drops_when_stage2_ratio_is_too_low(self) -> None:
        caster = _SequenceVisibilityCaster([(1, 20), (1, 20)])
        candidate_ids, source = referability_module._refine_candidate_visible_object_ids(
            [1],
            [make_object(1, "chair")],
            make_camera_pose(),
            make_camera_intrinsics(),
            np.ones((4, 4), dtype=np.float32),
            make_camera_intrinsics(),
            ray_caster_getter=lambda: caster,
            instance_mesh_data_getter=lambda _base: make_instance_mesh_data(obj_id=1, sample_count=8),
        )
        self.assertEqual(candidate_ids, [])
        self.assertEqual(source, "mesh_ray_refined")
        self.assertEqual(caster._responses, [])

    def test_refine_candidate_visible_object_ids_falls_back_to_projection_when_mesh_ray_fails(self) -> None:
        candidate_ids, source = referability_module._refine_candidate_visible_object_ids(
            [1],
            [make_object(1, "chair")],
            make_camera_pose(),
            make_camera_intrinsics(),
            np.ones((4, 4), dtype=np.float32),
            make_camera_intrinsics(),
            ray_caster_getter=lambda: (_ for _ in ()).throw(RuntimeError("ray failed")),
            instance_mesh_data_getter=lambda _base: make_instance_mesh_data(obj_id=1, sample_count=8),
        )
        self.assertEqual(candidate_ids, [1])
        self.assertEqual(source, "projection_fallback")

    def test_aggregate_label_reviews_uses_strict_policy(self) -> None:
        label_to_ids = {
            "chair": [1, 2],
            "lamp": [3, 4],
            "plant": [5, 6],
            "table": [7],
            "sofa": [8, 9],
        }
        object_reviews = {
            1: {"obj_id": 1, "local_outcome": "reviewed", "vlm_status": "clear"},
            2: {"obj_id": 2, "local_outcome": "out_of_frame", "vlm_status": None},
            3: {"obj_id": 3, "local_outcome": "reviewed", "vlm_status": "clear"},
            4: {"obj_id": 4, "local_outcome": "reviewed", "vlm_status": "clear"},
            5: {"obj_id": 5, "local_outcome": "reviewed", "vlm_status": "absent"},
            6: {"obj_id": 6, "local_outcome": "excluded", "vlm_status": None},
            7: {"obj_id": 7, "local_outcome": "reviewed", "vlm_status": "unsure"},
            8: {"obj_id": 8, "local_outcome": "reviewed", "vlm_status": "clear"},
            9: {"obj_id": 9, "local_outcome": "reviewed", "vlm_status": "unsure"},
        }

        label_statuses, label_counts, referable_ids = referability_module._aggregate_label_reviews(
            label_to_ids,
            object_reviews,
        )

        self.assertEqual(
            label_statuses,
            {
                "chair": "unique",
                "lamp": "multiple",
                "plant": "absent",
                "sofa": "unsure",
                "table": "unsure",
            },
        )
        self.assertEqual(
            label_counts,
            {
                "chair": 1,
                "lamp": 2,
                "plant": 0,
                "sofa": 1,
                "table": 0,
            },
        )
        self.assertEqual(referable_ids, [1])

    def test_normalize_frame_review_uses_clear_output_for_frame_gate(self) -> None:
        normalized = referability_module._normalize_frame_review(
            {
                "clear": False,
                "clarity_score": 28,
                "reason": "obviously blurry overall",
            }
        )

        self.assertEqual(
            normalized,
            {
                "clear": False,
                "clarity_score": 28,
                "frame_usable": False,
                "reason": "obviously blurry overall",
            },
        )

    def test_normalize_frame_review_accepts_legacy_frame_quality_fields(self) -> None:
        normalized = referability_module._normalize_frame_review(
            {
                "clarity_score": 82,
                "severely_out_of_focus": False,
                "usable_for_spatial_reasoning": True,
                "reason": "clear enough",
            }
        )

        self.assertEqual(normalized["clear"], True)
        self.assertEqual(normalized["frame_usable"], True)
        self.assertEqual(normalized["clarity_score"], 82)

    def test_full_frame_label_vlm_review_maps_count_to_label_status(self) -> None:
        cases = [
            ({"count": 0, "status": "absent", "reason": "no visible chair"}, "absent", 0, "no visible chair"),
            ({"count": 1, "status": "unique", "reason": "exactly one chair"}, "unique", 1, "exactly one chair"),
            ({"count": 3, "status": "multiple", "reason": "three chairs visible"}, "multiple", 3, "three chairs visible"),
        ]

        for parsed, expected_status, expected_count, expected_reason in cases:
            with self.subTest(expected_status=expected_status):
                with patch.object(
                    referability_module,
                    "_call_vlm_json",
                    return_value=(parsed, json.dumps(parsed)),
                ) as vlm_mock:
                    review = referability_module._full_frame_label_vlm_review(
                        client=object(),
                        model="fake-vlm",
                        image_b64="abcd",
                        label="Chair",
                    )

                self.assertEqual(review["backend"], "vlm")
                self.assertEqual(review["label"], "chair")
                self.assertEqual(review["status"], expected_status)
                self.assertEqual(review["count"], expected_count)
                self.assertEqual(review["reason"], expected_reason)
                self.assertEqual(review["raw_response"], json.dumps(parsed))
                self.assertEqual(vlm_mock.call_args.args[1], "fake-vlm")
                self.assertEqual(
                    vlm_mock.call_args.args[2][0]["image_url"]["url"],
                    "data:image/jpeg;base64,abcd",
                )
                self.assertIn('"chair"', vlm_mock.call_args.args[2][1]["text"])

    def test_full_frame_label_vlm_review_returns_unsure_on_parse_fallback(self) -> None:
        with patch.object(
            referability_module,
            "_call_vlm_json",
            return_value=(
                {"status": "mystery", "count": "unknown", "reason": ""},
                '{"status":"mystery","count":"unknown"}',
            ),
        ):
            review = referability_module._full_frame_label_vlm_review(
                client=object(),
                model="fake-vlm",
                image_b64="abcd",
                label="chair",
            )

        self.assertEqual(review["backend"], "vlm")
        self.assertEqual(review["status"], "unsure")
        self.assertIsNone(review["count"])
        self.assertEqual(review["reason"], "parse_fallback")
        self.assertEqual(review["raw_response"], '{"status":"mystery","count":"unknown"}')

    def test_compute_frame_referability_entry_builds_crop_vlm_reviews(self) -> None:
        scene_objects = [
            make_object(1, "chair"),
            make_object(2, "chair"),
            make_object(3, "lamp"),
        ]
        objects_by_id = {int(obj["id"]): obj for obj in scene_objects}
        visibility = {
            1: make_visibility_meta(projected_area_px=900.0),
            2: {"projected_area_px": 0.0},
            3: make_visibility_meta(projected_area_px=900.0),
        }

        with (
            patch.object(
                referability_module,
                "_frame_decision",
                return_value={
                    "clear": True,
                    "clarity_score": 82,
                    "frame_usable": True,
                    "reason": "clear enough",
                },
            ),
            patch.object(
                referability_module,
                "_refine_candidate_visible_object_ids",
                return_value=([1, 2, 3], "mesh_ray_refined"),
            ),
            patch.object(
                referability_module,
                "compute_frame_object_visibility",
                return_value=visibility,
            ),
            patch.object(
                referability_module,
                "_object_review_decision",
                side_effect=[("absent", '{"status":"absent"}')],
            ) as review_mock,
            patch.object(
                referability_module,
                "_full_frame_label_vlm_review",
                return_value={
                    "backend": "vlm",
                    "count": 1,
                    "status": "unique",
                    "reason": "exactly one lamp is visible",
                    "raw_response": None,
                },
            ) as full_frame_mock,
            patch.object(
                referability_module,
                "_apply_crop_unique_mesh_quality_review",
                return_value={},
            ),
        ):
            frame_entry = referability_module._compute_frame_referability_entry(
                client=object(),
                model_name="fake-vlm",
                scene_objects=scene_objects,
                objects_by_id=objects_by_id,
                image=np.zeros((120, 120, 3), dtype=np.uint8),
                image_path=Path("image.jpg"),
                camera_pose=make_camera_pose(),
                color_intrinsics=make_camera_intrinsics(),
                depth_image=None,
                depth_intrinsics=None,
                selector_visible_object_ids=[1, 2, 3],
            )

        self.assertEqual(frame_entry["frame_usable"], True)
        self.assertEqual(frame_entry["frame_quality_clear"], True)
        self.assertEqual(frame_entry["frame_quality_score"], 82)
        self.assertEqual(frame_entry["candidate_visibility_source"], "mesh_ray_refined")
        self.assertEqual(frame_entry["crop_label_statuses"], {"chair": "multiple", "lamp": "absent"})
        self.assertEqual(frame_entry["crop_label_counts"], {"chair": 2, "lamp": 0})
        self.assertEqual(frame_entry["crop_referable_object_ids"], [])
        self.assertEqual(frame_entry["full_frame_label_statuses"], {})
        self.assertEqual(frame_entry["full_frame_label_counts"], {})
        self.assertEqual(frame_entry["label_statuses"], {"chair": "multiple", "lamp": "absent"})
        self.assertEqual(frame_entry["label_counts"], {"chair": 2, "lamp": 0})
        self.assertEqual(frame_entry["out_of_frame_label_reviews"], [])
        self.assertEqual(frame_entry["out_of_frame_not_visible_labels"], [])
        self.assertEqual(frame_entry["out_of_frame_label_to_object_ids"], {})
        self.assertFalse(frame_entry["out_of_frame_vlm_early_stop"])
        self.assertEqual(frame_entry["referable_object_ids"], [])
        self.assertEqual(frame_entry["full_frame_label_reviews"], [])
        self.assertEqual(frame_entry["object_reviews"]["1"]["review_mode"], "selector_duplicate_shortcut")
        self.assertEqual(frame_entry["object_reviews"]["1"]["review_skip_reason"], "selector_visible_label_multiple")
        self.assertIsNone(frame_entry["object_reviews"]["1"]["vlm_status"])
        self.assertEqual(frame_entry["object_reviews"]["2"]["local_outcome"], "out_of_frame")
        self.assertEqual(frame_entry["object_reviews"]["3"]["vlm_status"], "absent")
        self.assertEqual(frame_entry["referability_reason_by_alias_group"]["chair_family"], "selector_duplicate_shortcut")
        self.assertEqual(review_mock.call_count, 1)
        full_frame_mock.assert_not_called()

    def test_compute_frame_referability_entry_uses_earlier_quantity_veto(self) -> None:
        scene_objects = [
            make_object(1, "chair"),
            make_object(2, "chair"),
        ]
        objects_by_id = {int(obj["id"]): obj for obj in scene_objects}
        visibility = {
            1: make_visibility_meta(projected_area_px=900.0),
            2: {"projected_area_px": 0.0},
        }

        with (
            patch.object(
                referability_module,
                "_frame_decision",
                return_value={
                    "clear": True,
                    "clarity_score": 82,
                    "frame_usable": True,
                    "reason": "clear enough",
                },
            ),
            patch.object(
                referability_module,
                "_refine_candidate_visible_object_ids",
                return_value=([1, 2], "mesh_ray_refined"),
            ),
            patch.object(
                referability_module,
                "compute_frame_object_visibility",
                return_value=visibility,
            ),
            patch.object(
                referability_module,
                "_object_review_decision",
                return_value=("clear", '{"status":"clear"}'),
            ),
            patch.object(
                referability_module,
                "_full_frame_label_vlm_review",
                return_value={
                    "backend": "vlm",
                    "count": 2,
                    "status": "multiple",
                    "reason": "two chairs are visible",
                    "raw_response": None,
                },
            ),
            patch.object(
                referability_module,
                "_apply_crop_unique_mesh_quality_review",
                return_value={},
            ),
        ):
            frame_entry = referability_module._compute_frame_referability_entry(
                client=object(),
                model_name="fake-vlm",
                scene_objects=scene_objects,
                objects_by_id=objects_by_id,
                image=np.zeros((120, 120, 3), dtype=np.uint8),
                image_path=Path("image.jpg"),
                camera_pose=make_camera_pose(),
                color_intrinsics=make_camera_intrinsics(),
                depth_image=None,
                depth_intrinsics=None,
                selector_visible_object_ids=[1, 2],
            )

        self.assertEqual(frame_entry["crop_label_statuses"], {"chair": "multiple"})
        self.assertEqual(frame_entry["crop_label_counts"], {"chair": 2})
        self.assertEqual(frame_entry["full_frame_label_statuses"], {})
        self.assertEqual(frame_entry["label_statuses"], {"chair": "multiple"})
        self.assertEqual(frame_entry["label_counts"], {"chair": 2})
        self.assertEqual(frame_entry["referable_object_ids"], [])

    def test_compute_frame_referability_entry_duplicate_selector_label_without_candidates_becomes_absent(self) -> None:
        scene_objects = [
            make_object(1, "chair"),
            make_object(2, "chair"),
        ]
        objects_by_id = {int(obj["id"]): obj for obj in scene_objects}
        visibility = {
            1: make_visibility_meta(projected_area_px=900.0),
            2: make_visibility_meta(projected_area_px=900.0),
        }

        with (
            patch.object(
                referability_module,
                "_frame_decision",
                return_value={
                    "clear": True,
                    "clarity_score": 82,
                    "frame_usable": True,
                    "reason": "clear enough",
                },
            ),
            patch.object(
                referability_module,
                "_refine_candidate_visible_object_ids",
                return_value=([], "mesh_ray_refined"),
            ),
            patch.object(
                referability_module,
                "compute_frame_object_visibility",
                return_value=visibility,
            ),
            patch.object(
                referability_module,
                "_object_review_decision",
            ) as review_mock,
            patch.object(
                referability_module,
                "_full_frame_label_vlm_review",
            ) as full_frame_mock,
            patch.object(
                referability_module,
                "_apply_crop_unique_mesh_quality_review",
                return_value={},
            ),
        ):
            frame_entry = referability_module._compute_frame_referability_entry(
                client=object(),
                model_name="fake-vlm",
                scene_objects=scene_objects,
                objects_by_id=objects_by_id,
                image=np.zeros((120, 120, 3), dtype=np.uint8),
                image_path=Path("image.jpg"),
                camera_pose=make_camera_pose(),
                color_intrinsics=make_camera_intrinsics(),
                depth_image=None,
                depth_intrinsics=None,
                selector_visible_object_ids=[1, 2],
            )

        self.assertEqual(frame_entry["candidate_visible_object_ids"], [])
        self.assertEqual(frame_entry["crop_label_statuses"], {"chair": "absent"})
        self.assertEqual(frame_entry["crop_label_counts"], {"chair": 0})
        self.assertEqual(frame_entry["full_frame_label_statuses"], {})
        self.assertEqual(frame_entry["label_statuses"], {"chair": "absent"})
        self.assertEqual(frame_entry["label_counts"], {"chair": 0})
        self.assertEqual(frame_entry["referable_object_ids"], [])
        review_mock.assert_not_called()
        full_frame_mock.assert_not_called()

    def test_compute_frame_referability_entry_full_frame_absent_vetoes_crop_unique(self) -> None:
        scene_objects = [make_object(1, "shelves")]
        objects_by_id = {int(obj["id"]): obj for obj in scene_objects}
        visibility = {
            1: make_visibility_meta(projected_area_px=900.0),
        }

        with (
            patch.object(
                referability_module,
                "_frame_decision",
                return_value={
                    "clear": True,
                    "clarity_score": 82,
                    "frame_usable": True,
                    "reason": "clear enough",
                },
            ),
            patch.object(
                referability_module,
                "_refine_candidate_visible_object_ids",
                return_value=([1], "mesh_ray_refined"),
            ),
            patch.object(
                referability_module,
                "compute_frame_object_visibility",
                return_value=visibility,
            ),
            patch.object(
                referability_module,
                "_object_review_decision",
                return_value=("clear", '{"status":"clear"}'),
            ),
            patch.object(
                referability_module,
                "_full_frame_label_vlm_review",
                return_value={
                    "backend": "vlm",
                    "count": 0,
                    "status": "absent",
                    "reason": "no visible shelves",
                    "raw_response": None,
                },
            ),
            patch.object(
                referability_module,
                "_apply_crop_unique_mesh_quality_review",
                return_value={},
            ),
        ):
            frame_entry = referability_module._compute_frame_referability_entry(
                client=object(),
                model_name="fake-vlm",
                scene_objects=scene_objects,
                objects_by_id=objects_by_id,
                image=np.zeros((120, 120, 3), dtype=np.uint8),
                image_path=Path("image.jpg"),
                camera_pose=make_camera_pose(),
                color_intrinsics=make_camera_intrinsics(),
                depth_image=None,
                depth_intrinsics=None,
                selector_visible_object_ids=[1],
            )

        self.assertEqual(frame_entry["crop_label_statuses"], {"shelves": "unique"})
        self.assertEqual(frame_entry["crop_referable_object_ids"], [1])
        self.assertEqual(frame_entry["full_frame_label_statuses"], {"shelves": "absent"})
        self.assertEqual(frame_entry["full_frame_label_reviews"][0]["backend"], "vlm")
        self.assertEqual(frame_entry["full_frame_label_reviews"][0]["raw_detection_count"], 0)
        self.assertEqual(frame_entry["full_frame_label_reviews"][0]["reason"], "no visible shelves")
        self.assertIsNone(frame_entry["full_frame_label_reviews"][0]["raw_response"])
        self.assertEqual(frame_entry["label_statuses"], {"shelves": "absent"})
        self.assertEqual(frame_entry["label_counts"], {"shelves": 0})
        self.assertEqual(frame_entry["referable_object_ids"], [])

    def test_compute_frame_referability_entry_excludes_unique_object_below_projected_area_threshold(self) -> None:
        scene_objects = [make_object(1, "chair")]
        objects_by_id = {int(obj["id"]): obj for obj in scene_objects}
        visibility = {
            1: make_visibility_meta(projected_area_px=799.0, zbuffer_mask_area_px=1.0),
        }

        with (
            patch.object(
                referability_module,
                "_frame_decision",
                return_value={
                    "clear": True,
                    "clarity_score": 82,
                    "frame_usable": True,
                    "reason": "clear enough",
                },
            ),
            patch.object(
                referability_module,
                "_refine_candidate_visible_object_ids",
                return_value=([1], "mesh_ray_refined"),
            ),
            patch.object(
                referability_module,
                "compute_frame_object_visibility",
                return_value=visibility,
            ),
            patch.object(
                referability_module,
                "_object_review_decision",
            ) as review_mock,
            patch.object(
                referability_module,
                "_full_frame_label_vlm_review",
            ) as full_frame_mock,
            patch.object(
                referability_module,
                "_apply_crop_unique_mesh_quality_review",
                return_value={},
            ),
        ):
            frame_entry = referability_module._compute_frame_referability_entry(
                client=object(),
                model_name="fake-vlm",
                scene_objects=scene_objects,
                objects_by_id=objects_by_id,
                image=np.zeros((120, 120, 3), dtype=np.uint8),
                image_path=Path("image.jpg"),
                camera_pose=make_camera_pose(),
                color_intrinsics=make_camera_intrinsics(),
                depth_image=None,
                depth_intrinsics=None,
                selector_visible_object_ids=[1],
            )

        self.assertEqual(frame_entry["crop_label_statuses"], {"chair": "absent"})
        self.assertEqual(frame_entry["crop_label_counts"], {"chair": 0})
        self.assertEqual(frame_entry["crop_referable_object_ids"], [])
        self.assertEqual(frame_entry["referable_object_ids"], [])
        self.assertEqual(frame_entry["object_reviews"]["1"]["local_outcome"], "excluded")
        self.assertEqual(frame_entry["object_reviews"]["1"]["local_reason"], "projected_area_too_small")
        review_mock.assert_not_called()
        full_frame_mock.assert_not_called()

    def test_build_out_of_frame_label_candidates_sorts_by_representative_geometry(self) -> None:
        scene_objects = [
            make_object(2, "chair", alias_group="chair_family"),
            make_object(3, "sofa", alias_group="sofa_family"),
            make_object(5, "lamp", alias_group="lamp_family"),
            make_object(6, "lamp", alias_group="lamp_family"),
        ]
        objects_by_id = {int(obj["id"]): obj for obj in scene_objects}
        fake_geometry = {
            2: {
                "obj_id": 2,
                "label": "chair",
                "projected_area_px": 100.0,
                "in_frame_ratio": 0.0,
                "in_frame_sample_count": 0,
                "outside_distance_px": 4.0,
                "is_out_of_frame": True,
            },
            3: {
                "obj_id": 3,
                "label": "sofa",
                "projected_area_px": 80.0,
                "in_frame_ratio": 0.0,
                "in_frame_sample_count": 0,
                "outside_distance_px": 100.0,
                "is_out_of_frame": True,
            },
            5: {
                "obj_id": 5,
                "label": "lamp",
                "projected_area_px": 100.0,
                "in_frame_ratio": 0.0,
                "in_frame_sample_count": 0,
                "outside_distance_px": 3.0,
                "is_out_of_frame": True,
            },
            6: {
                "obj_id": 6,
                "label": "lamp",
                "projected_area_px": 100.0,
                "in_frame_ratio": 0.0,
                "in_frame_sample_count": 0,
                "outside_distance_px": 5.0,
                "is_out_of_frame": True,
            },
        }

        with patch.object(
            referability_module,
            "_evaluate_out_of_frame_geometry_for_object",
            side_effect=lambda **kwargs: dict(fake_geometry[int(kwargs["obj"]["id"])]),
        ):
            candidates, label_to_ids = referability_module._build_out_of_frame_label_candidates(
                scene_objects=scene_objects,
                objects_by_id=objects_by_id,
                visibility_by_obj_id={},
                camera_pose=make_camera_pose(),
                color_intrinsics=make_camera_intrinsics(),
            )

        self.assertEqual([item["label"] for item in candidates], ["lamp", "chair", "sofa"])
        self.assertEqual(label_to_ids, {"chair": [2], "lamp": [5, 6], "sofa": [3]})
        self.assertEqual(candidates[0]["representative"]["obj_id"], 6)

    def test_compute_frame_referability_entry_reviews_out_of_frame_candidates_until_not_visible(self) -> None:
        scene_objects = [
            make_object(1, "lamp", alias_group="lamp_family"),
            make_object(2, "lamp", alias_group="lamp_family"),
            make_object(3, "chair", alias_group="chair_family"),
        ]
        objects_by_id = {int(obj["id"]): obj for obj in scene_objects}
        visibility = {
            1: make_visibility_meta(projected_area_px=1200.0, bbox_in_frame_ratio=0.0),
            2: make_visibility_meta(projected_area_px=900.0, bbox_in_frame_ratio=0.0),
            3: make_visibility_meta(projected_area_px=800.0, bbox_in_frame_ratio=0.0),
        }

        with (
            patch.object(
                referability_module,
                "_frame_decision",
                return_value={
                    "clear": True,
                    "clarity_score": 82,
                    "frame_usable": True,
                    "reason": "clear enough",
                },
            ),
            patch.object(
                referability_module,
                "_refine_candidate_visible_object_ids",
                return_value=([], "mesh_ray_refined"),
            ),
            patch.object(
                referability_module,
                "compute_frame_object_visibility",
                return_value=visibility,
            ),
            patch.object(
                referability_module,
                "_out_of_frame_label_vlm_review",
                side_effect=[
                    {
                        "status": "reject",
                        "raw_response": '{"status":"reject"}',
                    },
                    {
                        "status": "not_visible",
                        "raw_response": '{"status":"not_visible"}',
                    },
                ],
            ) as out_of_frame_mock,
            patch.object(
                referability_module,
                "_apply_crop_unique_mesh_quality_review",
                return_value={},
            ),
        ):
            frame_entry = referability_module._compute_frame_referability_entry(
                client=object(),
                model_name="fake-vlm",
                scene_objects=scene_objects,
                objects_by_id=objects_by_id,
                image=np.zeros((120, 120, 3), dtype=np.uint8),
                image_path=Path("image.jpg"),
                camera_pose=make_camera_pose(),
                color_intrinsics=make_camera_intrinsics(),
                depth_image=None,
                depth_intrinsics=None,
                selector_visible_object_ids=[],
            )

        self.assertEqual(out_of_frame_mock.call_count, 2)
        self.assertEqual(
            [call.kwargs["label"] for call in out_of_frame_mock.call_args_list],
            ["lamp", "chair"],
        )
        self.assertEqual(
            frame_entry["out_of_frame_label_to_object_ids"],
            {"chair": [3], "lamp": [1, 2]},
        )
        self.assertEqual(
            frame_entry["out_of_frame_label_reviews"],
            [
                {"label": "lamp", "status": "reject", "raw_response": '{"status":"reject"}'},
                {"label": "chair", "status": "not_visible", "raw_response": '{"status":"not_visible"}'},
            ],
        )
        self.assertEqual(frame_entry["out_of_frame_not_visible_labels"], ["chair"])
        self.assertTrue(frame_entry["out_of_frame_vlm_early_stop"])

    def test_review_out_of_frame_label_candidates_keeps_fields_empty_without_not_visible(self) -> None:
        scene_objects = [
            make_object(1, "lamp", alias_group="lamp_family"),
            make_object(2, "chair", alias_group="chair_family"),
        ]
        objects_by_id = {int(obj["id"]): obj for obj in scene_objects}
        visibility = {
            1: make_visibility_meta(projected_area_px=1200.0, bbox_in_frame_ratio=0.0),
            2: make_visibility_meta(projected_area_px=900.0, bbox_in_frame_ratio=0.0),
        }

        with patch.object(
            referability_module,
            "_out_of_frame_label_vlm_review",
            side_effect=[
                {"status": "reject", "raw_response": '{"status":"reject"}'},
                {"status": "unsure", "raw_response": '{"status":"unsure"}'},
            ],
        ):
            review = referability_module._review_out_of_frame_label_candidates(
                client=object(),
                model_name="fake-vlm",
                image=np.zeros((120, 120, 3), dtype=np.uint8),
                scene_objects=scene_objects,
                objects_by_id=objects_by_id,
                visibility_by_obj_id=visibility,
                camera_pose=make_camera_pose(),
                color_intrinsics=make_camera_intrinsics(),
            )

        self.assertEqual(review["out_of_frame_label_reviews"], [])
        self.assertEqual(review["out_of_frame_not_visible_labels"], [])
        self.assertEqual(review["out_of_frame_label_to_object_ids"], {})
        self.assertFalse(review["out_of_frame_vlm_early_stop"])

    def test_enrich_final_scene_entries_out_of_frame_populates_selected_frame(self) -> None:
        root = Path(__file__).resolve().parent / "_tmp" / f"final_scene_out_of_frame_{uuid.uuid4().hex}"
        scene_dir = root / "scene0001_00"
        (scene_dir / "color").mkdir(parents=True, exist_ok=True)
        self.addCleanup(shutil.rmtree, root, True)

        scene_objects = [
            make_object(1, "lamp", alias_group="lamp_family"),
            make_object(2, "lamp", alias_group="lamp_family"),
        ]
        entry = make_debug_cache_entry()
        entry["final_selection_rank"] = 0

        with (
            patch.object(
                referability_module.cv2,
                "imread",
                return_value=np.zeros((120, 120, 3), dtype=np.uint8),
            ),
            patch.object(
                referability_module,
                "compute_frame_object_visibility",
                return_value={
                    1: make_visibility_meta(projected_area_px=1200.0, bbox_in_frame_ratio=0.0),
                    2: make_visibility_meta(projected_area_px=900.0, bbox_in_frame_ratio=0.0),
                },
            ),
            patch.object(
                referability_module,
                "_out_of_frame_label_vlm_review",
                return_value={
                    "status": "not_visible",
                    "raw_response": '{"status":"not_visible"}',
                },
            ),
        ):
            enriched = referability_module._enrich_final_scene_entries_out_of_frame(
                client=object(),
                model_name="fake-vlm",
                scene_dir=scene_dir,
                final_scene_entries={"000001.jpg": entry},
                scene_objects=scene_objects,
                objects_by_id={int(obj["id"]): obj for obj in scene_objects},
                poses={"000001.jpg": make_camera_pose()},
                color_intrinsics=make_camera_intrinsics(),
                depth_intrinsics=None,
            )

        frame_entry = enriched["000001.jpg"]
        self.assertEqual(
            frame_entry["out_of_frame_label_reviews"],
            [{"label": "lamp", "status": "not_visible", "raw_response": '{"status":"not_visible"}'}],
        )
        self.assertEqual(frame_entry["out_of_frame_not_visible_labels"], ["lamp"])
        self.assertEqual(frame_entry["out_of_frame_label_to_object_ids"], {"lamp": [1, 2]})
        self.assertTrue(frame_entry["out_of_frame_vlm_early_stop"])

    def test_enrich_final_scene_entries_out_of_frame_preserves_existing_review_data(self) -> None:
        entry = make_debug_cache_entry()
        entry["out_of_frame_label_reviews"] = [
            {"label": "lamp", "status": "not_visible", "raw_response": '{"status":"not_visible"}'}
        ]
        entry["out_of_frame_not_visible_labels"] = ["lamp"]
        entry["out_of_frame_label_to_object_ids"] = {"lamp": [1]}
        entry["out_of_frame_vlm_early_stop"] = True

        with (
            patch.object(
                referability_module.cv2,
                "imread",
                side_effect=AssertionError("existing out-of-frame review data should skip enrichment"),
            ),
            patch.object(
                referability_module,
                "compute_frame_object_visibility",
                side_effect=AssertionError("existing out-of-frame review data should skip enrichment"),
            ),
        ):
            enriched = referability_module._enrich_final_scene_entries_out_of_frame(
                client=object(),
                model_name="fake-vlm",
                scene_dir=Path("."),
                final_scene_entries={"000001.jpg": entry},
                scene_objects=[make_object(1, "lamp", alias_group="lamp_family")],
                objects_by_id={1: make_object(1, "lamp", alias_group="lamp_family")},
                poses={"000001.jpg": make_camera_pose()},
                color_intrinsics=make_camera_intrinsics(),
                depth_intrinsics=None,
            )

        self.assertEqual(enriched["000001.jpg"]["out_of_frame_not_visible_labels"], ["lamp"])

    def test_compute_frame_referability_entry_skips_out_of_frame_review_for_ambiguous_alias_group(self) -> None:
        scene_objects = [
            make_object(1, "lamp", alias_group="shared_family"),
            make_object(2, "chair", alias_group="shared_family"),
        ]
        objects_by_id = {int(obj["id"]): obj for obj in scene_objects}
        visibility = {
            1: make_visibility_meta(projected_area_px=1200.0, bbox_in_frame_ratio=0.0),
            2: make_visibility_meta(projected_area_px=900.0, bbox_in_frame_ratio=0.0),
        }

        with (
            patch.object(
                referability_module,
                "_frame_decision",
                return_value={
                    "clear": True,
                    "clarity_score": 82,
                    "frame_usable": True,
                    "reason": "clear enough",
                },
            ),
            patch.object(
                referability_module,
                "_refine_candidate_visible_object_ids",
                return_value=([], "mesh_ray_refined"),
            ),
            patch.object(
                referability_module,
                "compute_frame_object_visibility",
                return_value=visibility,
            ),
            patch.object(
                referability_module,
                "_out_of_frame_label_vlm_review",
            ) as out_of_frame_mock,
            patch.object(
                referability_module,
                "_apply_crop_unique_mesh_quality_review",
                return_value={},
            ),
        ):
            frame_entry = referability_module._compute_frame_referability_entry(
                client=object(),
                model_name="fake-vlm",
                scene_objects=scene_objects,
                objects_by_id=objects_by_id,
                image=np.zeros((120, 120, 3), dtype=np.uint8),
                image_path=Path("image.jpg"),
                camera_pose=make_camera_pose(),
                color_intrinsics=make_camera_intrinsics(),
                depth_image=None,
                depth_intrinsics=None,
                selector_visible_object_ids=[],
            )

        out_of_frame_mock.assert_not_called()
        self.assertEqual(frame_entry["out_of_frame_label_reviews"], [])
        self.assertEqual(frame_entry["out_of_frame_not_visible_labels"], [])
        self.assertEqual(frame_entry["out_of_frame_label_to_object_ids"], {})
        self.assertFalse(frame_entry["out_of_frame_vlm_early_stop"])

    def test_compute_frame_referability_entry_passes_instance_mesh_data_to_visibility(self) -> None:
        scene_objects = [make_object(1, "chair")]
        objects_by_id = {int(obj["id"]): obj for obj in scene_objects}
        captured: dict[str, object] = {}
        visibility_meta = make_visibility_meta(
            projected_area_px=900.0,
            zbuffer_mask_area_px=900.0,
        )
        sentinel_instance_mesh_data = object()

        def fake_compute_frame_object_visibility(*args, **kwargs):
            captured["instance_mesh_data"] = kwargs.get("instance_mesh_data")
            return {1: visibility_meta}

        with (
            patch.object(
                referability_module,
                "_frame_decision",
                return_value={
                    "clear": True,
                    "clarity_score": 82,
                    "frame_usable": True,
                    "reason": "clear enough",
                },
            ),
            patch.object(
                referability_module,
                "_refine_candidate_visible_object_ids",
                return_value=([1], "mesh_ray_refined"),
            ),
            patch.object(
                referability_module,
                "compute_frame_object_visibility",
                side_effect=fake_compute_frame_object_visibility,
            ),
            patch.object(
                referability_module,
                "_object_review_decision",
                return_value=("clear", '{"status":"clear"}'),
            ),
            patch.object(
                referability_module,
                "_full_frame_label_vlm_review",
                return_value={
                    "backend": "vlm",
                    "count": 1,
                    "status": "unique",
                    "reason": "exactly one chair is visible",
                    "raw_response": None,
                },
            ),
            patch.object(
                referability_module,
                "_apply_crop_unique_mesh_quality_review",
                return_value={},
            ),
        ):
            frame_entry = referability_module._compute_frame_referability_entry(
                client=object(),
                model_name="fake-vlm",
                scene_objects=scene_objects,
                objects_by_id=objects_by_id,
                image=np.zeros((120, 120, 3), dtype=np.uint8),
                image_path=Path("image.jpg"),
                camera_pose=make_camera_pose(),
                color_intrinsics=make_camera_intrinsics(),
                depth_image=None,
                depth_intrinsics=None,
                selector_visible_object_ids=[1],
                instance_mesh_data_getter=lambda _base: sentinel_instance_mesh_data,
            )

        self.assertIs(captured["instance_mesh_data"], sentinel_instance_mesh_data)
        self.assertEqual(frame_entry["referable_object_ids"], [1])

    def test_compute_frame_referability_entry_applies_70_percent_bbox_ratio_gate_to_final_referable_ids(self) -> None:
        scene_objects = [make_object(1, "chair")]
        objects_by_id = {int(obj["id"]): obj for obj in scene_objects}
        visibility = {
            1: make_visibility_meta(projected_area_px=900.0, bbox_in_frame_ratio=0.69),
        }

        with (
            patch.object(
                referability_module,
                "_frame_decision",
                return_value={
                    "clear": True,
                    "clarity_score": 82,
                    "frame_usable": True,
                    "reason": "clear enough",
                },
            ),
            patch.object(
                referability_module,
                "_refine_candidate_visible_object_ids",
                return_value=([1], "mesh_ray_refined"),
            ),
            patch.object(
                referability_module,
                "compute_frame_object_visibility",
                return_value=visibility,
            ),
            patch.object(
                referability_module,
                "_object_review_decision",
                return_value=("clear", '{"status":"clear"}'),
            ),
            patch.object(
                referability_module,
                "_full_frame_label_vlm_review",
                return_value={
                    "backend": "vlm",
                    "count": 1,
                    "status": "unique",
                    "reason": "exactly one chair is visible",
                    "raw_response": None,
                },
            ),
            patch.object(
                referability_module,
                "_apply_crop_unique_mesh_quality_review",
                return_value={},
            ),
        ):
            frame_entry = referability_module._compute_frame_referability_entry(
                client=object(),
                model_name="fake-vlm",
                scene_objects=scene_objects,
                objects_by_id=objects_by_id,
                image=np.zeros((120, 120, 3), dtype=np.uint8),
                image_path=Path("image.jpg"),
                camera_pose=make_camera_pose(),
                color_intrinsics=make_camera_intrinsics(),
                depth_image=None,
                depth_intrinsics=None,
                selector_visible_object_ids=[1],
            )

        self.assertEqual(frame_entry["crop_label_statuses"], {"chair": "unique"})
        self.assertEqual(frame_entry["crop_referable_object_ids"], [1])
        self.assertEqual(frame_entry["full_frame_label_statuses"], {"chair": "unique"})
        self.assertEqual(frame_entry["label_statuses"], {"chair": "unique"})
        self.assertEqual(frame_entry["attachment_referable_object_ids"], [1])
        self.assertEqual(frame_entry["referable_object_ids"], [])

    def test_compute_frame_referability_entry_keeps_final_referable_ids_at_70_percent_boundary(self) -> None:
        scene_objects = [make_object(1, "chair")]
        objects_by_id = {int(obj["id"]): obj for obj in scene_objects}
        visibility = {
            1: make_visibility_meta(projected_area_px=900.0, bbox_in_frame_ratio=0.70),
        }

        with (
            patch.object(
                referability_module,
                "_frame_decision",
                return_value={
                    "clear": True,
                    "clarity_score": 82,
                    "frame_usable": True,
                    "reason": "clear enough",
                },
            ),
            patch.object(
                referability_module,
                "_refine_candidate_visible_object_ids",
                return_value=([1], "mesh_ray_refined"),
            ),
            patch.object(
                referability_module,
                "compute_frame_object_visibility",
                return_value=visibility,
            ),
            patch.object(
                referability_module,
                "_object_review_decision",
                return_value=("clear", '{"status":"clear"}'),
            ),
            patch.object(
                referability_module,
                "_full_frame_label_vlm_review",
                return_value={
                    "backend": "vlm",
                    "count": 1,
                    "status": "unique",
                    "reason": "exactly one chair is visible",
                    "raw_response": None,
                },
            ),
            patch.object(
                referability_module,
                "_apply_crop_unique_mesh_quality_review",
                return_value={},
            ),
        ):
            frame_entry = referability_module._compute_frame_referability_entry(
                client=object(),
                model_name="fake-vlm",
                scene_objects=scene_objects,
                objects_by_id=objects_by_id,
                image=np.zeros((120, 120, 3), dtype=np.uint8),
                image_path=Path("image.jpg"),
                camera_pose=make_camera_pose(),
                color_intrinsics=make_camera_intrinsics(),
                depth_image=None,
                depth_intrinsics=None,
                selector_visible_object_ids=[1],
            )

        self.assertEqual(frame_entry["crop_label_statuses"], {"chair": "unique"})
        self.assertEqual(frame_entry["crop_referable_object_ids"], [1])
        self.assertEqual(frame_entry["full_frame_label_statuses"], {"chair": "unique"})
        self.assertEqual(frame_entry["label_statuses"], {"chair": "unique"})
        self.assertEqual(frame_entry["attachment_referable_object_ids"], [1])
        self.assertEqual(frame_entry["referable_object_ids"], [1])

    def test_apply_crop_unique_mesh_quality_review_drops_unique_object_on_mesh_mismatch(self) -> None:
        crop_entry = {
            "local_outcome": "reviewed",
            "reason": "",
            "roi_bounds_px": [20, 60, 20, 60],
            "crop_bounds_px": [16, 64, 16, 64],
            "projected_area_px": 900.0,
            "bbox_in_frame_ratio": 0.9,
            "edge_margin_px": 10.0,
        }
        object_reviews = {
            1: referability_module._build_object_review_entry(
                obj_id=1,
                label="chair",
                crop_entry=crop_entry,
            )
        }
        objects_by_id = {1: make_object(1, "chair")}
        topology_quality_by_obj_id: dict[int, dict] = {}
        mesh_mask_quality_by_obj_id: dict[int, dict] = {}

        with (
            patch.object(
                referability_module,
                "_compute_topology_quality_for_object",
                return_value=make_topology_quality(1, "pass"),
            ),
            patch.object(
                referability_module,
                "_call_dinox_joint_detection",
                return_value=[make_detection(bbox=(22, 22, 58, 58), score=0.95, category="chair")],
            ),
            patch.object(
                referability_module,
                "_compute_mesh_mask_quality_for_object",
                return_value=make_mesh_quality(1, "fail", reason_codes=["low_iou", "high_under_coverage"]),
            ),
        ):
            failed = referability_module._apply_crop_unique_mesh_quality_review(
                crop_unique_label_object_ids={"chair": 1},
                object_reviews=object_reviews,
                objects_by_id=objects_by_id,
                image_path=Path("image.jpg"),
                image_shape=(120, 120, 3),
                camera_pose=make_camera_pose(),
                color_intrinsics=make_camera_intrinsics(),
                depth_image=None,
                depth_intrinsics=None,
                instance_mesh_data_getter=lambda _base: make_instance_mesh_data(obj_id=1),
                topology_quality_by_obj_id=topology_quality_by_obj_id,
                mesh_mask_quality_by_obj_id=mesh_mask_quality_by_obj_id,
                client=object(),
            )

        self.assertEqual(failed, {"chair": "mesh_mask_mismatch"})
        self.assertEqual(topology_quality_by_obj_id[1]["status"], "pass")
        self.assertEqual(mesh_mask_quality_by_obj_id[1]["status"], "fail")
        review = object_reviews[1]
        self.assertEqual(review["topology_status"], "pass")
        self.assertEqual(review["mesh_mask_status"], "fail")
        self.assertEqual(review["mesh_quality_review"]["decision"], "drop")
        self.assertEqual(review["mesh_quality_review"]["reason"], "mesh_mask_mismatch")
        self.assertIsNotNone(review["mesh_quality_review"]["matched_detection"])

    def test_repair_final_referability_fields_recomputes_stale_final_fields(self) -> None:
        stale_entry = {
            "frame_usable": True,
            "frame_quality_clear": True,
            "frame_quality_score": 82,
            "frame_quality_reason": "clear enough",
            "frame_selection_score": 82001,
            "attachment_referable_pairs": [],
            "attachment_referable_pair_count": 0,
            "final_selection_rank": 0,
            "candidate_visible_object_ids": [1],
            "candidate_visibility_source": "mesh_ray_refined",
            "candidate_labels": ["lamp"],
            "label_to_object_ids": {"lamp": [1]},
            "selector_visible_object_ids": [1],
            "selector_visible_label_counts": {"lamp": 1},
            "visibility_audit_by_object_id": {
                "1": {
                    "obj_id": 1,
                    "label": "lamp",
                    "candidate_considered": True,
                    "candidate_passed": True,
                    "candidate_rejection_reasons": [],
                }
            },
            "object_reviews": {
                "1": {
                    "obj_id": 1,
                    "label": "lamp",
                    "local_outcome": "reviewed",
                    "vlm_status": "clear",
                }
            },
            "crop_label_statuses": {"lamp": "unique"},
            "crop_label_counts": {"lamp": 1},
            "crop_referable_object_ids": [1],
            "full_frame_label_reviews": [{"label": "lamp", "status": "absent"}],
            "full_frame_label_statuses": {"lamp": "absent"},
            "full_frame_label_counts": {"lamp": 0},
            "label_statuses": {"lamp": "unique"},
            "label_counts": {"lamp": 1},
            "out_of_frame_label_reviews": [],
            "out_of_frame_not_visible_labels": [],
            "out_of_frame_label_to_object_ids": {},
            "out_of_frame_vlm_early_stop": False,
            "referable_object_ids": [1],
        }

        repaired = referability_module._repair_final_referability_fields(stale_entry)

        self.assertEqual(repaired["crop_label_statuses"], {"lamp": "unique"})
        self.assertEqual(repaired["full_frame_label_statuses"], {"lamp": "absent"})
        self.assertEqual(repaired["label_statuses"], {"lamp": "absent"})
        self.assertEqual(repaired["label_counts"], {"lamp": 0})
        self.assertEqual(repaired["attachment_referable_object_ids"], [])
        self.assertEqual(repaired["referable_object_ids"], [])
        self.assertEqual(repaired["vlm_unique_object_ids"], [])

    def test_derive_final_referability_fields_recovers_relaxed_attachment_ids_from_legacy_entry(self) -> None:
        legacy_entry = {
            "label_to_object_ids": {
                "table": [1],
                "cup": [2],
            },
            "crop_label_statuses": {
                "table": "unique",
                "cup": "unique",
            },
            "crop_label_counts": {
                "table": 1,
                "cup": 1,
            },
            "crop_referable_object_ids": [1, 2],
            "full_frame_label_statuses": {
                "table": "unique",
                "cup": "unique",
            },
            "full_frame_label_counts": {
                "table": 1,
                "cup": 1,
            },
            "object_reviews": {
                "1": {
                    "obj_id": 1,
                    "label": "table",
                    "local_outcome": "reviewed",
                    "vlm_status": "clear",
                    "bbox_in_frame_ratio": 0.55,
                },
                "2": {
                    "obj_id": 2,
                    "label": "cup",
                    "local_outcome": "reviewed",
                    "vlm_status": "clear",
                    "bbox_in_frame_ratio": 0.95,
                },
            },
            "referable_object_ids": [2],
        }

        derived = referability_module._derive_final_referability_fields(legacy_entry)

        self.assertEqual(derived["label_statuses"], {"cup": "unique", "table": "unique"})
        self.assertEqual(derived["referable_object_ids"], [2])
        self.assertEqual(derived["attachment_referable_object_ids"], [1, 2])

    def test_frame_entry_has_debug_fields_rejects_stale_final_field_mismatch(self) -> None:
        stale_entry = {
            "frame_usable": True,
            "frame_quality_clear": True,
            "frame_quality_score": 82,
            "frame_quality_reason": "clear enough",
            "frame_selection_score": 82001,
            "attachment_referable_pairs": [],
            "attachment_referable_pair_count": 0,
            "final_selection_rank": 0,
            "candidate_visible_object_ids": [1],
            "candidate_visibility_source": "mesh_ray_refined",
            "candidate_labels": ["lamp"],
            "label_to_object_ids": {"lamp": [1]},
            "selector_visible_object_ids": [1],
            "selector_visible_label_counts": {"lamp": 1},
            "visibility_audit_by_object_id": {
                "1": {
                    "obj_id": 1,
                    "label": "lamp",
                    "candidate_considered": True,
                    "candidate_passed": True,
                    "candidate_rejection_reasons": [],
                }
            },
            "object_reviews": {
                "1": {
                    "obj_id": 1,
                    "label": "lamp",
                    "local_outcome": "reviewed",
                    "vlm_status": "clear",
                }
            },
            "crop_label_statuses": {"lamp": "unique"},
            "crop_label_counts": {"lamp": 1},
            "crop_referable_object_ids": [1],
            "full_frame_label_reviews": [{"label": "lamp", "status": "absent"}],
            "full_frame_label_statuses": {"lamp": "absent"},
            "full_frame_label_counts": {"lamp": 0},
            "label_statuses": {"lamp": "unique"},
            "label_counts": {"lamp": 1},
            "out_of_frame_label_reviews": [],
            "out_of_frame_not_visible_labels": [],
            "out_of_frame_label_to_object_ids": {},
            "out_of_frame_vlm_early_stop": False,
            "referable_object_ids": [1],
        }

        consistent_entry = dict(stale_entry)
        consistent_entry["label_statuses"] = {"lamp": "absent"}
        consistent_entry["label_counts"] = {"lamp": 0}
        consistent_entry["referable_object_ids"] = []
        consistent_entry["vlm_unique_object_ids"] = []

        self.assertFalse(referability_module._frame_entry_has_debug_fields(stale_entry))
        self.assertTrue(referability_module._frame_entry_has_debug_fields(consistent_entry))

    def test_frame_entry_has_debug_fields_accepts_exact_crop_label_counts_above_two(self) -> None:
        entry = {
            "frame_usable": True,
            "frame_quality_clear": True,
            "frame_quality_score": 82,
            "frame_quality_reason": "clear enough",
            "frame_selection_score": 82001,
            "attachment_referable_pairs": [],
            "attachment_referable_pair_count": 0,
            "final_selection_rank": 0,
            "candidate_visible_object_ids": [1, 2, 3, 4],
            "candidate_visibility_source": "mesh_ray_refined",
            "candidate_labels": ["stool"],
            "label_to_object_ids": {"stool": [1, 2, 3, 4]},
            "selector_visible_object_ids": [1, 2, 3, 4],
            "selector_visible_label_counts": {"stool": 4},
            "visibility_audit_by_object_id": {
                str(obj_id): {
                    "obj_id": obj_id,
                    "label": "stool",
                    "candidate_considered": True,
                    "candidate_passed": True,
                    "candidate_rejection_reasons": [],
                    "bbox_in_frame_ratio": 0.9,
                }
                for obj_id in (1, 2, 3, 4)
            },
            "object_reviews": {
                str(obj_id): {
                    "obj_id": obj_id,
                    "label": "stool",
                    "local_outcome": "reviewed",
                    "vlm_status": "clear",
                    "bbox_in_frame_ratio": 0.9,
                }
                for obj_id in (1, 2, 3, 4)
            },
            "crop_label_statuses": {"stool": "multiple"},
            "crop_label_counts": {"stool": 4},
            "crop_referable_object_ids": [],
            "full_frame_label_reviews": [],
            "full_frame_label_statuses": {},
            "full_frame_label_counts": {},
            "label_statuses": {"stool": "multiple"},
            "label_counts": {"stool": 2},
            "out_of_frame_label_reviews": [],
            "out_of_frame_not_visible_labels": [],
            "out_of_frame_label_to_object_ids": {},
            "out_of_frame_vlm_early_stop": False,
            "referable_object_ids": [],
            "vlm_unique_object_ids": [],
        }

        self.assertTrue(referability_module._frame_entry_has_debug_fields(entry))
        repaired = referability_module._repair_final_referability_fields(entry)
        self.assertEqual(repaired["crop_label_counts"], {"stool": 4})

    def test_frame_entry_has_debug_fields_accepts_selector_duplicate_shortcut_counts(self) -> None:
        entry = {
            "frame_usable": True,
            "frame_quality_clear": True,
            "frame_quality_score": 82,
            "frame_quality_reason": "clear enough",
            "frame_selection_score": 82001,
            "attachment_referable_pairs": [],
            "attachment_referable_pair_count": 0,
            "final_selection_rank": 0,
            "candidate_visible_object_ids": [1],
            "candidate_visibility_source": "mesh_ray_refined",
            "candidate_labels": ["chair"],
            "label_to_object_ids": {"chair": [1]},
            "selector_visible_object_ids": [1, 2],
            "selector_visible_label_counts": {"chair": 2},
            "visibility_audit_by_object_id": {
                "1": {
                    "obj_id": 1,
                    "label": "chair",
                    "candidate_considered": True,
                    "candidate_passed": True,
                    "candidate_rejection_reasons": [],
                    "bbox_in_frame_ratio": 0.9,
                }
            },
            "object_reviews": {
                "1": {
                    "obj_id": 1,
                    "label": "chair",
                    "review_mode": "selector_duplicate_shortcut",
                    "review_skip_reason": "selector_visible_label_multiple",
                    "local_outcome": "reviewed",
                    "vlm_status": None,
                    "bbox_in_frame_ratio": 0.9,
                }
            },
            "crop_label_statuses": {"chair": "multiple"},
            "crop_label_counts": {"chair": 1},
            "crop_referable_object_ids": [],
            "full_frame_label_reviews": [],
            "full_frame_label_statuses": {},
            "full_frame_label_counts": {},
            "label_statuses": {"chair": "multiple"},
            "label_counts": {"chair": 2},
            "out_of_frame_label_reviews": [],
            "out_of_frame_not_visible_labels": [],
            "out_of_frame_label_to_object_ids": {},
            "out_of_frame_vlm_early_stop": False,
            "referable_object_ids": [],
            "vlm_unique_object_ids": [],
        }

        self.assertTrue(referability_module._frame_entry_has_debug_fields(entry))
        repaired = referability_module._repair_final_referability_fields(entry)
        self.assertEqual(repaired["crop_label_counts"], {"chair": 1})
        self.assertEqual(
            repaired["attachment_selector_signal"],
            {"well_cropped_pair_count": 0, "viewpoint_exempt": False},
        )
        self.assertEqual(
            repaired["attachment_final_referability"],
            {"object_ids": [], "pairs": [], "pair_count": 0},
        )
        self.assertEqual(
            repaired["attachment_final_frame_selection"],
            {"selected_for_final_cache": True, "selection_rank": 0},
        )

    def test_compress_attachment_group_frames_keeps_distinct_referable_pairs(self) -> None:
        frames = [
            {
                "image_name": "000003.jpg",
                "crop_ge_70_count": 3,
                "attachment_referable_pairs": [[2, 1]],
                "attachment_referable_pair_count": 1,
            },
            {
                "image_name": "000006.jpg",
                "crop_ge_70_count": 2,
                "attachment_referable_pairs": [[3, 1]],
                "attachment_referable_pair_count": 1,
            },
            {
                "image_name": "000009.jpg",
                "crop_ge_70_count": 1,
                "attachment_referable_pairs": [[2, 1]],
                "attachment_referable_pair_count": 1,
            },
        ]

        kept = referability_module._compress_attachment_group_frames(frames)

        self.assertEqual(
            [frame["image_name"] for frame in kept],
            ["000003.jpg", "000006.jpg"],
        )

    def test_select_attachment_frames_by_global_pair_coverage_prefers_new_pairs(self) -> None:
        frames = [
            {
                "image_name": "000003.jpg",
                "crop_ge_70_count": 3,
                "attachment_referable_pairs": [[2, 1], [3, 1]],
                "attachment_referable_pair_count": 2,
            },
            {
                "image_name": "000006.jpg",
                "crop_ge_70_count": 2,
                "attachment_referable_pairs": [[2, 1]],
                "attachment_referable_pair_count": 1,
            },
            {
                "image_name": "000009.jpg",
                "crop_ge_70_count": 1,
                "attachment_referable_pairs": [[4, 1]],
                "attachment_referable_pair_count": 1,
            },
        ]

        selected = referability_module._select_attachment_frames_by_global_pair_coverage(
            frames,
            max_frames=2,
        )

        self.assertEqual(
            [frame["image_name"] for frame in selected],
            ["000003.jpg", "000009.jpg"],
        )

    def test_select_and_rerank_frames_filters_unusable_frames_then_prefers_clarity_score(self) -> None:
        frame_candidates = [
            {"image_name": "000000.jpg", "score": 20, "n_visible": 5, "visible_object_ids": [1, 2, 3]},
            {"image_name": "000030.jpg", "score": 9, "n_visible": 3, "visible_object_ids": [4, 5]},
            {"image_name": "000060.jpg", "score": 7, "n_visible": 4, "visible_object_ids": [6, 7]},
        ]
        frame_decisions = [
            {
                "clear": False,
                "clarity_score": 99,
                "frame_usable": False,
                "reason": "overall blurry",
            },
            {
                "clear": True,
                "clarity_score": 10,
                "frame_usable": True,
                "reason": "barely clear but acceptable",
            },
            {
                "clear": True,
                "clarity_score": 95,
                "frame_usable": True,
                "reason": "sharp",
            },
        ]

        root = Path(__file__).resolve().parent / "_tmp" / f"rerank_{uuid.uuid4().hex}"
        root.mkdir(parents=True, exist_ok=False)
        self.addCleanup(shutil.rmtree, root, True)
        scene_dir = root / "scene0000_00"

        with (
            patch.object(
                referability_module.cv2,
                "imread",
                return_value=np.zeros((32, 32, 3), dtype=np.uint8),
            ),
            patch.object(
                referability_module,
                "_frame_decision",
                side_effect=frame_decisions,
            ),
        ):
            selected = referability_module._select_and_rerank_frames(
                client=object(),
                model_name="fake-vlm",
                scene_dir=scene_dir,
                frame_candidates=frame_candidates,
                max_frames=2,
            )

        self.assertEqual([entry["image_name"] for entry in selected], ["000060.jpg", "000030.jpg"])
        self.assertTrue(all(entry["frame_info"]["frame_usable"] for entry in selected))
        self.assertEqual([entry["frame_info"]["clarity_score"] for entry in selected], [95, 10])
        self.assertEqual(selected[0]["frame_selection_score"], 100007)
        self.assertEqual(selected[1]["frame_selection_score"], 100009)

    def test_select_and_rerank_frames_keeps_group_input_order_before_clarity_review(self) -> None:
        frame_candidates = [
            {"image_name": "000000.jpg", "score": 1, "n_visible": 2, "visible_object_ids": [1, 2]},
            {"image_name": "000030.jpg", "score": 99, "n_visible": 5, "visible_object_ids": [2, 1]},
        ]
        frame_decisions = [
            {
                "clear": True,
                "clarity_score": 72,
                "frame_usable": True,
                "reason": "clear enough",
            },
        ]

        root = Path(__file__).resolve().parent / "_tmp" / f"rerank_group_input_order_{uuid.uuid4().hex}"
        root.mkdir(parents=True, exist_ok=False)
        self.addCleanup(shutil.rmtree, root, True)
        scene_dir = root / "scene0000_00"

        with (
            patch.object(
                referability_module.cv2,
                "imread",
                return_value=np.zeros((32, 32, 3), dtype=np.uint8),
            ),
            patch.object(
                referability_module,
                "_frame_decision",
                side_effect=frame_decisions,
            ) as frame_decision_mock,
        ):
            selected = referability_module._select_and_rerank_frames(
                client=object(),
                model_name="fake-vlm",
                scene_dir=scene_dir,
                frame_candidates=frame_candidates,
                max_frames=1,
            )

        self.assertEqual(frame_decision_mock.call_count, 1)
        self.assertEqual([entry["image_name"] for entry in selected], ["000000.jpg"])
        self.assertEqual([entry["frame_info"]["clarity_score"] for entry in selected], [72])

    def test_select_and_rerank_frames_stops_reviewing_group_after_first_high_quality_hit(self) -> None:
        frame_candidates = [
            {"image_name": "000000.jpg", "score": 20, "n_visible": 5, "visible_object_ids": [1, 2]},
            {"image_name": "000030.jpg", "score": 19, "n_visible": 4, "visible_object_ids": [2, 1]},
            {"image_name": "000060.jpg", "score": 18, "n_visible": 4, "visible_object_ids": [3, 4]},
        ]
        frame_decisions = [
            {
                "clear": True,
                "clarity_score": 81,
                "frame_usable": True,
                "reason": "sharp enough",
            },
            {
                "clear": True,
                "clarity_score": 70,
                "frame_usable": True,
                "reason": "clear",
            },
        ]

        root = Path(__file__).resolve().parent / "_tmp" / f"rerank_group_stop_{uuid.uuid4().hex}"
        root.mkdir(parents=True, exist_ok=False)
        self.addCleanup(shutil.rmtree, root, True)
        scene_dir = root / "scene0000_00"

        with (
            patch.object(
                referability_module.cv2,
                "imread",
                return_value=np.zeros((32, 32, 3), dtype=np.uint8),
            ),
            patch.object(
                referability_module,
                "_frame_decision",
                side_effect=frame_decisions,
            ) as frame_decision_mock,
        ):
            selected = referability_module._select_and_rerank_frames(
                client=object(),
                model_name="fake-vlm",
                scene_dir=scene_dir,
                frame_candidates=frame_candidates,
                max_frames=2,
            )

        self.assertEqual(frame_decision_mock.call_count, 2)
        self.assertEqual([entry["image_name"] for entry in selected], ["000000.jpg", "000060.jpg"])
        self.assertEqual([entry["frame_info"]["clarity_score"] for entry in selected], [81, 70])

    def test_select_and_rerank_frames_discards_candidates_without_visible_object_ids(self) -> None:
        frame_candidates = [
            {"image_name": "000000.jpg", "score": 20, "n_visible": 5},
            {"image_name": "000030.jpg", "score": 19, "n_visible": 4, "visible_object_ids": [1, 2]},
        ]
        frame_decisions = [
            {
                "clear": True,
                "clarity_score": 72,
                "frame_usable": True,
                "reason": "clear",
            },
        ]

        root = Path(__file__).resolve().parent / "_tmp" / f"rerank_missing_visible_ids_{uuid.uuid4().hex}"
        root.mkdir(parents=True, exist_ok=False)
        self.addCleanup(shutil.rmtree, root, True)
        scene_dir = root / "scene0000_00"

        with (
            patch.object(
                referability_module.cv2,
                "imread",
                return_value=np.zeros((32, 32, 3), dtype=np.uint8),
            ),
            patch.object(
                referability_module,
                "_frame_decision",
                side_effect=frame_decisions,
            ) as frame_decision_mock,
        ):
            selected = referability_module._select_and_rerank_frames(
                client=object(),
                model_name="fake-vlm",
                scene_dir=scene_dir,
                frame_candidates=frame_candidates,
                max_frames=2,
            )

        self.assertEqual(frame_decision_mock.call_count, 1)
        self.assertEqual([entry["image_name"] for entry in selected], ["000030.jpg"])

    def test_select_and_rerank_frames_limits_non_attachment_group_count(self) -> None:
        frame_candidates = [
            {"image_name": "000000.jpg", "score": 20, "n_visible": 5, "visible_object_ids": [1, 2]},
            {"image_name": "000030.jpg", "score": 19, "n_visible": 4, "visible_object_ids": [3, 4]},
            {"image_name": "000060.jpg", "score": 18, "n_visible": 3, "visible_object_ids": [5, 6]},
        ]
        frame_decisions = [
            {
                "clear": True,
                "clarity_score": 81,
                "frame_usable": True,
                "reason": "sharp enough",
            },
            {
                "clear": True,
                "clarity_score": 75,
                "frame_usable": True,
                "reason": "clear",
            },
        ]
        debug_output: dict[str, Any] = {}

        root = Path(__file__).resolve().parent / "_tmp" / f"rerank_group_limit_{uuid.uuid4().hex}"
        root.mkdir(parents=True, exist_ok=False)
        self.addCleanup(shutil.rmtree, root, True)
        scene_dir = root / "scene0000_00"

        with (
            patch.object(
                referability_module.cv2,
                "imread",
                return_value=np.zeros((32, 32, 3), dtype=np.uint8),
            ),
            patch.object(
                referability_module,
                "_frame_decision",
                side_effect=frame_decisions,
            ) as frame_decision_mock,
        ):
            selected = referability_module._select_and_rerank_frames(
                client=object(),
                model_name="fake-vlm",
                scene_dir=scene_dir,
                frame_candidates=frame_candidates,
                max_frames=3,
                max_group_count=2,
                debug_output=debug_output,
            )

        self.assertEqual(frame_decision_mock.call_count, 2)
        self.assertEqual([entry["image_name"] for entry in selected], ["000000.jpg", "000030.jpg"])
        self.assertEqual(debug_output["non_attachment_visible_object_group_count"], 3)
        self.assertEqual(debug_output["non_attachment_processed_group_count"], 2)
        self.assertEqual(len(debug_output["groups"]), 2)

    def test_select_and_rerank_frames_non_attachment_group_requires_two_referables_for_early_stop(self) -> None:
        frame_candidates = [
            {"image_name": "000000.jpg", "score": 20, "n_visible": 5, "visible_object_ids": [1, 2]},
            {"image_name": "000030.jpg", "score": 19, "n_visible": 4, "visible_object_ids": [2, 1]},
            {"image_name": "000060.jpg", "score": 18, "n_visible": 3, "visible_object_ids": [5, 6]},
        ]
        frame_decisions = [
            {
                "clear": True,
                "clarity_score": 81,
                "frame_usable": True,
                "reason": "sharp enough",
            },
            {
                "clear": True,
                "clarity_score": 79,
                "frame_usable": True,
                "reason": "also sharp",
            },
            {
                "clear": True,
                "clarity_score": 75,
                "frame_usable": True,
                "reason": "clear",
            },
        ]

        root = Path(__file__).resolve().parent / "_tmp" / f"rerank_two_referables_{uuid.uuid4().hex}"
        root.mkdir(parents=True, exist_ok=False)
        self.addCleanup(shutil.rmtree, root, True)
        scene_dir = root / "scene0000_00"

        build_calls: list[str] = []

        def build_entry(frame: dict, reviewed_frame: dict) -> dict:
            build_calls.append(frame["image_name"])
            if frame["image_name"] == "000000.jpg":
                return {"referable_object_ids": [1]}
            if frame["image_name"] == "000030.jpg":
                return {"referable_object_ids": [1, 2]}
            return {"referable_object_ids": [5, 6]}

        with (
            patch.object(
                referability_module.cv2,
                "imread",
                return_value=np.zeros((32, 32, 3), dtype=np.uint8),
            ),
            patch.object(
                referability_module,
                "_frame_decision",
                side_effect=frame_decisions,
            ) as frame_decision_mock,
        ):
            selected = referability_module._select_and_rerank_frames(
                client=object(),
                model_name="fake-vlm",
                scene_dir=scene_dir,
                frame_candidates=frame_candidates,
                max_frames=2,
                referability_entry_builder=build_entry,
            )

        self.assertEqual(frame_decision_mock.call_count, 3)
        self.assertEqual(build_calls, ["000000.jpg", "000030.jpg", "000060.jpg"])
        self.assertEqual([entry["image_name"] for entry in selected], ["000030.jpg", "000060.jpg"])

    def test_select_and_rerank_frames_non_attachment_group_falls_back_to_single_referable_frame(self) -> None:
        frame_candidates = [
            {"image_name": "000000.jpg", "score": 20, "n_visible": 5, "visible_object_ids": [1, 2]},
            {"image_name": "000030.jpg", "score": 19, "n_visible": 4, "visible_object_ids": [2, 1]},
        ]
        frame_decisions = [
            {
                "clear": True,
                "clarity_score": 81,
                "frame_usable": True,
                "reason": "sharp enough",
            },
            {
                "clear": True,
                "clarity_score": 79,
                "frame_usable": True,
                "reason": "also sharp",
            },
        ]

        root = Path(__file__).resolve().parent / "_tmp" / f"rerank_single_referable_fallback_{uuid.uuid4().hex}"
        root.mkdir(parents=True, exist_ok=False)
        self.addCleanup(shutil.rmtree, root, True)
        scene_dir = root / "scene0000_00"

        build_calls: list[str] = []

        def build_entry(frame: dict, reviewed_frame: dict) -> dict:
            build_calls.append(frame["image_name"])
            if frame["image_name"] == "000000.jpg":
                return {"referable_object_ids": [1]}
            return {"referable_object_ids": []}

        with (
            patch.object(
                referability_module.cv2,
                "imread",
                return_value=np.zeros((32, 32, 3), dtype=np.uint8),
            ),
            patch.object(
                referability_module,
                "_frame_decision",
                side_effect=frame_decisions,
            ) as frame_decision_mock,
        ):
            selected = referability_module._select_and_rerank_frames(
                client=object(),
                model_name="fake-vlm",
                scene_dir=scene_dir,
                frame_candidates=frame_candidates,
                max_frames=1,
                referability_entry_builder=build_entry,
            )

        self.assertEqual(frame_decision_mock.call_count, 2)
        self.assertEqual(build_calls, ["000000.jpg", "000030.jpg"])
        self.assertEqual([entry["image_name"] for entry in selected], ["000000.jpg"])

    def test_select_and_rerank_frames_stats_report_only_successful_group_count(self) -> None:
        frame_candidates = [
            {"image_name": "000000.jpg", "score": 20, "n_visible": 5, "visible_object_ids": [1, 2]},
            {"image_name": "000030.jpg", "score": 19, "n_visible": 4, "visible_object_ids": [3, 4]},
        ]
        frame_decisions = [
            {
                "clear": True,
                "clarity_score": 81,
                "frame_usable": True,
                "reason": "sharp enough",
            },
            {
                "clear": True,
                "clarity_score": 79,
                "frame_usable": True,
                "reason": "also sharp",
            },
        ]
        stats_output: dict[str, Any] = {}

        root = Path(__file__).resolve().parent / "_tmp" / f"rerank_stats_success_count_{uuid.uuid4().hex}"
        root.mkdir(parents=True, exist_ok=False)
        self.addCleanup(shutil.rmtree, root, True)
        scene_dir = root / "scene0000_00"

        def build_entry(frame: dict, reviewed_frame: dict) -> dict:
            if frame["image_name"] == "000000.jpg":
                return {"referable_object_ids": []}
            return {"referable_object_ids": [3, 4]}

        with (
            patch.object(
                referability_module.cv2,
                "imread",
                return_value=np.zeros((32, 32, 3), dtype=np.uint8),
            ),
            patch.object(
                referability_module,
                "_frame_decision",
                side_effect=frame_decisions,
            ),
        ):
            selected = referability_module._select_and_rerank_frames(
                client=object(),
                model_name="fake-vlm",
                scene_dir=scene_dir,
                frame_candidates=frame_candidates,
                max_frames=2,
                referability_entry_builder=build_entry,
                stats_output=stats_output,
            )

        self.assertEqual([entry["image_name"] for entry in selected], ["000030.jpg"])
        self.assertEqual(stats_output["non_attachment_visible_object_group_count"], 2)
        self.assertEqual(stats_output["non_attachment_processed_group_count"], 2)
        self.assertEqual(stats_output["accepted_frame_count_after_group_scan"], 1)

    def test_select_and_rerank_frames_stops_scanning_groups_after_collecting_max_frames(self) -> None:
        frame_candidates = [
            {"image_name": "000000.jpg", "score": 20, "n_visible": 5, "visible_object_ids": [1, 2]},
            {"image_name": "000030.jpg", "score": 19, "n_visible": 4, "visible_object_ids": [3, 4]},
            {"image_name": "000060.jpg", "score": 18, "n_visible": 3, "visible_object_ids": [5, 6]},
            {"image_name": "000090.jpg", "score": 17, "n_visible": 3, "visible_object_ids": [7, 8]},
        ]
        frame_decisions = [
            {
                "clear": True,
                "clarity_score": 81,
                "frame_usable": True,
                "reason": "sharp enough",
            },
            {
                "clear": True,
                "clarity_score": 79,
                "frame_usable": True,
                "reason": "also sharp",
            },
        ]
        stats_output: dict[str, Any] = {}

        root = Path(__file__).resolve().parent / "_tmp" / f"rerank_group_target_stop_{uuid.uuid4().hex}"
        root.mkdir(parents=True, exist_ok=False)
        self.addCleanup(shutil.rmtree, root, True)
        scene_dir = root / "scene0000_00"

        build_calls: list[str] = []

        def build_entry(frame: dict, reviewed_frame: dict) -> dict:
            build_calls.append(frame["image_name"])
            return {"referable_object_ids": [1]}

        with (
            patch.object(
                referability_module.cv2,
                "imread",
                return_value=np.zeros((32, 32, 3), dtype=np.uint8),
            ),
            patch.object(
                referability_module,
                "_frame_decision",
                side_effect=frame_decisions,
            ) as frame_decision_mock,
        ):
            selected = referability_module._select_and_rerank_frames(
                client=object(),
                model_name="fake-vlm",
                scene_dir=scene_dir,
                frame_candidates=frame_candidates,
                max_frames=2,
                referability_entry_builder=build_entry,
                stats_output=stats_output,
            )

        self.assertEqual(frame_decision_mock.call_count, 2)
        self.assertEqual(build_calls, ["000000.jpg", "000030.jpg"])
        self.assertEqual([entry["image_name"] for entry in selected], ["000000.jpg", "000030.jpg"])
        self.assertEqual(stats_output["non_attachment_visible_object_group_count"], 4)
        self.assertEqual(stats_output["non_attachment_processed_group_count"], 2)
        self.assertEqual(stats_output["accepted_frame_count_after_group_scan"], 2)

    def test_select_attachment_group_representatives_groups_by_visible_attachment_pairs_and_orders_by_visible_object_count(self) -> None:
        frames = [
            {"image_name": "000000.jpg", "score": 20, "n_visible": 3, "visible_object_ids": [1, 2, 9]},
            {"image_name": "000010.jpg", "score": 19, "n_visible": 6, "visible_object_ids": [1, 2, 3, 9, 10, 11]},
            {"image_name": "000020.jpg", "score": 18, "n_visible": 4, "visible_object_ids": [1, 2, 3, 4]},
        ]
        frame_decisions = [
            {
                "clear": True,
                "clarity_score": 74,
                "frame_usable": True,
                "reason": "clear",
            },
            {
                "clear": True,
                "clarity_score": 72,
                "frame_usable": True,
                "reason": "clear enough",
            },
        ]

        root = Path(__file__).resolve().parent / "_tmp" / f"attachment_group_stop_{uuid.uuid4().hex}"
        root.mkdir(parents=True, exist_ok=False)
        self.addCleanup(shutil.rmtree, root, True)
        scene_dir = root / "scene0000_00"

        with (
            patch.object(
                referability_module.cv2,
                "imread",
                return_value=np.zeros((32, 32, 3), dtype=np.uint8),
            ),
            patch.object(
                referability_module,
                "_frame_decision",
                side_effect=frame_decisions,
            ) as frame_decision_mock,
        ):
            build_calls: list[str] = []

            def build_entry(frame: dict, reviewed_frame: dict) -> dict:
                build_calls.append(frame["image_name"])
                if frame["image_name"] == "000010.jpg":
                    return {"attachment_referable_object_ids": [1, 2]}
                if frame["image_name"] == "000020.jpg":
                    return {"attachment_referable_object_ids": [1, 2, 3, 4]}
                raise AssertionError(f"unexpected frame {frame['image_name']}")

            selected = referability_module._select_attachment_group_representatives(
                client=object(),
                model_name="fake-vlm",
                scene_dir=scene_dir,
                frames=frames,
                attachment_graph={1: [2], 3: [4]},
                attachment_entry_builder=build_entry,
            )

        self.assertEqual(frame_decision_mock.call_count, 2)
        self.assertEqual(build_calls, ["000010.jpg", "000020.jpg"])
        self.assertEqual([entry["image_name"] for entry in selected], ["000010.jpg", "000020.jpg"])

    def test_select_attachment_group_representatives_continues_until_multi_pair_group_covers_other_pair(self) -> None:
        frames = [
            {"image_name": "000010.jpg", "score": 20, "n_visible": 6, "visible_object_ids": [1, 2, 3, 4, 9, 10]},
            {"image_name": "000020.jpg", "score": 19, "n_visible": 4, "visible_object_ids": [1, 2, 3, 4]},
            {"image_name": "000030.jpg", "score": 18, "n_visible": 3, "visible_object_ids": [1, 2, 3, 4]},
        ]
        frame_decisions = [
            {
                "clear": True,
                "clarity_score": 74,
                "frame_usable": True,
                "reason": "clear",
            },
            {
                "clear": True,
                "clarity_score": 73,
                "frame_usable": True,
                "reason": "also clear",
            },
        ]

        root = Path(__file__).resolve().parent / "_tmp" / f"attachment_group_any_pair_{uuid.uuid4().hex}"
        root.mkdir(parents=True, exist_ok=False)
        self.addCleanup(shutil.rmtree, root, True)
        scene_dir = root / "scene0000_00"

        with (
            patch.object(
                referability_module.cv2,
                "imread",
                return_value=np.zeros((32, 32, 3), dtype=np.uint8),
            ),
            patch.object(
                referability_module,
                "_frame_decision",
                side_effect=frame_decisions,
            ) as frame_decision_mock,
        ):
            build_calls: list[str] = []

            def build_entry(frame: dict, reviewed_frame: dict) -> dict:
                build_calls.append(frame["image_name"])
                if frame["image_name"] == "000010.jpg":
                    return {"attachment_referable_object_ids": [1, 2]}
                if frame["image_name"] == "000020.jpg":
                    return {"attachment_referable_object_ids": [3, 4]}
                raise AssertionError(f"unexpected frame {frame['image_name']}")

            selected = referability_module._select_attachment_group_representatives(
                client=object(),
                model_name="fake-vlm",
                scene_dir=scene_dir,
                frames=frames,
                attachment_graph={1: [2], 3: [4]},
                attachment_entry_builder=build_entry,
            )

        self.assertEqual(frame_decision_mock.call_count, 2)
        self.assertEqual(build_calls, ["000010.jpg", "000020.jpg"])
        self.assertEqual([entry["image_name"] for entry in selected], ["000010.jpg", "000020.jpg"])
        self.assertEqual(
            [entry["attachment_referable_pairs"] for entry in selected],
            [[[1, 2]], [[3, 4]]],
        )
        self.assertEqual(
            [entry["attachment_referable_pair_count"] for entry in selected],
            [1, 1],
        )

    def test_select_attachment_group_representatives_checks_all_frames_by_visible_object_count_until_pair_match(self) -> None:
        frames = [
            {"image_name": "000000.jpg", "score": 20, "n_visible": 3, "visible_object_ids": [1, 2, 9]},
            {"image_name": "000010.jpg", "score": 19, "n_visible": 6, "visible_object_ids": [1, 2, 9, 10, 11, 12]},
            {"image_name": "000020.jpg", "score": 18, "n_visible": 2, "visible_object_ids": [1, 2]},
        ]
        frame_decisions = [
            {
                "clear": True,
                "clarity_score": 74,
                "frame_usable": True,
                "reason": "clear",
            },
            {
                "clear": True,
                "clarity_score": 72,
                "frame_usable": True,
                "reason": "clear enough",
            },
        ]

        root = Path(__file__).resolve().parent / "_tmp" / f"attachment_group_stop_{uuid.uuid4().hex}"
        root.mkdir(parents=True, exist_ok=False)
        self.addCleanup(shutil.rmtree, root, True)
        scene_dir = root / "scene0000_00"

        with (
            patch.object(
                referability_module.cv2,
                "imread",
                return_value=np.zeros((32, 32, 3), dtype=np.uint8),
            ),
            patch.object(
                referability_module,
                "_frame_decision",
                side_effect=frame_decisions,
            ) as frame_decision_mock,
        ):
            build_calls: list[str] = []

            def build_entry(frame: dict, reviewed_frame: dict) -> dict:
                build_calls.append(frame["image_name"])
                if frame["image_name"] == "000010.jpg":
                    return {"attachment_referable_object_ids": [1]}
                if frame["image_name"] == "000000.jpg":
                    return {"attachment_referable_object_ids": [1, 2]}
                raise AssertionError(f"unexpected frame {frame['image_name']}")

            selected = referability_module._select_attachment_group_representatives(
                client=object(),
                model_name="fake-vlm",
                scene_dir=scene_dir,
                frames=frames,
                attachment_graph={1: [2]},
                attachment_entry_builder=build_entry,
            )

        self.assertEqual(frame_decision_mock.call_count, 2)
        self.assertEqual(build_calls, ["000010.jpg", "000000.jpg"])
        self.assertEqual([entry["image_name"] for entry in selected], ["000000.jpg"])
        self.assertEqual([entry["frame_info"]["clarity_score"] for entry in selected], [72])

    def test_select_attachment_group_representatives_skips_group_when_no_frame_reaches_70(self) -> None:
        frames = [
            {"image_name": "000000.jpg", "score": 20, "n_visible": 5, "visible_object_ids": [1, 2]},
            {"image_name": "000010.jpg", "score": 19, "n_visible": 4, "visible_object_ids": [1, 2, 9]},
            {"image_name": "000020.jpg", "score": 18, "n_visible": 3, "visible_object_ids": [1, 2, 9, 10]},
            {"image_name": "000030.jpg", "score": 17, "n_visible": 2, "visible_object_ids": [1, 2]},
        ]
        frame_decisions = [
            {
                "clear": True,
                "clarity_score": 68,
                "frame_usable": True,
                "reason": "softish",
            },
            {
                "clear": True,
                "clarity_score": 69,
                "frame_usable": True,
                "reason": "still below threshold",
            },
            {
                "clear": True,
                "clarity_score": 67,
                "frame_usable": True,
                "reason": "still soft",
            },
            {
                "clear": True,
                "clarity_score": 66,
                "frame_usable": True,
                "reason": "too soft",
            },
        ]

        root = Path(__file__).resolve().parent / "_tmp" / f"attachment_group_drop_{uuid.uuid4().hex}"
        root.mkdir(parents=True, exist_ok=False)
        self.addCleanup(shutil.rmtree, root, True)
        scene_dir = root / "scene0000_00"

        with (
            patch.object(
                referability_module.cv2,
                "imread",
                return_value=np.zeros((32, 32, 3), dtype=np.uint8),
            ),
            patch.object(
                referability_module,
                "_frame_decision",
                side_effect=frame_decisions,
            ) as frame_decision_mock,
        ):
            selected = referability_module._select_attachment_group_representatives(
                client=object(),
                model_name="fake-vlm",
                scene_dir=scene_dir,
                frames=frames,
                attachment_graph={1: [2]},
                attachment_entry_builder=lambda frame, reviewed_frame: (_ for _ in ()).throw(
                    AssertionError("referability VLM should not run below clarity threshold")
                ),
            )

        self.assertEqual(frame_decision_mock.call_count, 4)
        self.assertEqual(selected, [])

    def test_main_persists_scene_grouping_summary_in_cache_and_debug_json(self) -> None:
        root = Path(__file__).resolve().parent / "_tmp" / f"scene_grouping_summary_{uuid.uuid4().hex}"
        data_root = root / "data"
        scene_dir = data_root / "scene0001_00"
        (scene_dir / "pose").mkdir(parents=True, exist_ok=True)
        output_path = root / "output" / "referability_cache.json"
        debug_dir = root / "output" / "group_debug"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        scene = {
            "objects": [
                make_object(1, "table"),
                make_object(2, "book"),
                make_object(3, "lamp"),
            ],
        }

        def fake_enrich(scene_dict: dict) -> dict:
            scene_dict["attachment_graph"] = {"1": [2]}
            scene_dict["attached_by"] = {"2": 1}
            scene_dict["attachment_edges"] = [{"parent_id": 1, "child_id": 2, "type": "supported_by"}]
            scene_dict["support_chain_graph"] = {"1": [2]}
            scene_dict["support_chain_by"] = {"2": 1}
            return scene_dict

        def make_selected_frame(image_name: str, clarity_score: int, visible_object_ids: list[int]) -> dict:
            entry = make_debug_cache_entry()
            entry["selector_visible_object_ids"] = list(visible_object_ids)
            entry["candidate_visible_object_ids"] = list(visible_object_ids)
            entry["attachment_referable_object_ids"] = []
            return {
                "image_name": image_name,
                "visible_object_ids": list(visible_object_ids),
                "frame_info": {
                    "clear": True,
                    "clarity_score": clarity_score,
                    "frame_usable": True,
                    "reason": "clear",
                },
                "frame_selection_score": 100000 + clarity_score,
                "_referability_entry": entry,
            }

        non_attachment_frames = [
            make_selected_frame("000101.jpg", 92, [3]),
            make_selected_frame("000102.jpg", 86, [3]),
        ]
        attachment_frame = make_debug_cache_entry()
        attachment_frame["image_name"] = "000001.jpg"
        attachment_frame["attachment_referable_object_ids"] = [1, 2]
        attachment_frame["attachment_view_group_id"] = 7

        def fake_select_and_rerank_frames(**kwargs):
            debug_output = kwargs["debug_output"]
            debug_output.clear()
            debug_output.update(
                {
                    "scene_id": "scene0001_00",
                    "pipeline_outcome": None,
                    "grouping_available": True,
                    "scene_skip_reason": None,
                    "non_attachment_candidate_frame_count": 2,
                    "non_attachment_visible_object_group_count": 2,
                    "non_attachment_processed_group_count": 2,
                    "accepted_frame_count_after_group_scan": 2,
                    "reranked_accepted_frame_image_names": ["000101.jpg", "000102.jpg"],
                    "selected_before_attachment_slots_image_names": ["000101.jpg", "000102.jpg"],
                    "selected_before_attachment_slots_count": 2,
                    "attachment_selected_frame_image_names": [],
                    "attachment_selected_frame_count": 0,
                    "remaining_slots_after_attachment_selection": None,
                    "selected_after_attachment_slots_image_names": [],
                    "selected_after_attachment_slots_count": 0,
                    "final_cacheable_frame_image_names": [],
                    "final_cacheable_frame_count": 0,
                    "groups": [
                        {
                            "group_index": 0,
                            "group_key_visible_object_ids": [3],
                            "candidate_frame_image_names": ["000101.jpg"],
                            "sampled_frame_image_names": ["000101.jpg"],
                            "accepted_frame_image_names": ["000101.jpg"],
                            "selected_before_attachment_slots_image_names": ["000101.jpg"],
                            "selected_after_attachment_slots_image_names": [],
                            "dropped_by_group_rerank_image_names": [],
                            "dropped_after_attachment_slots_image_names": [],
                            "group_frame_stride": 1,
                            "stopped_after_image_name": "000101.jpg",
                            "stop_reason": "accepted_frame_has_min_referable_objects",
                            "status_before_attachment_slots": "selected_before_attachment_slots",
                            "status_after_attachment_slots": None,
                            "group_exhausted_without_usable_frame": False,
                            "group_exhausted_without_referable_frame": False,
                        },
                        {
                            "group_index": 1,
                            "group_key_visible_object_ids": [3, 9],
                            "candidate_frame_image_names": ["000102.jpg"],
                            "sampled_frame_image_names": ["000102.jpg"],
                            "accepted_frame_image_names": ["000102.jpg"],
                            "selected_before_attachment_slots_image_names": ["000102.jpg"],
                            "selected_after_attachment_slots_image_names": [],
                            "dropped_by_group_rerank_image_names": [],
                            "dropped_after_attachment_slots_image_names": [],
                            "group_frame_stride": 1,
                            "stopped_after_image_name": "000102.jpg",
                            "stop_reason": "accepted_frame_has_min_referable_objects",
                            "status_before_attachment_slots": "selected_before_attachment_slots",
                            "status_after_attachment_slots": None,
                            "group_exhausted_without_usable_frame": False,
                            "group_exhausted_without_referable_frame": False,
                        },
                    ],
                }
            )
            return list(non_attachment_frames)

        self.addCleanup(shutil.rmtree, root, True)
        with (
            patch.dict(sys.modules, {"openai": make_fake_openai_module()}),
            patch("src.scene_parser.parse_scene", return_value=scene),
            patch("src.support_graph.enrich_scene_with_attachment", side_effect=fake_enrich),
            patch("src.support_graph.build_attachment_candidates", return_value=[]),
            patch.object(
                referability_module,
                "select_frames",
                return_value=[
                    {
                        "image_name": "000001.jpg",
                        "visible_object_ids": [1, 2],
                        "score": 10,
                        "attachment_viewpoint_exempt": True,
                    },
                    {
                        "image_name": "000101.jpg",
                        "visible_object_ids": [3],
                        "score": 9,
                        "attachment_viewpoint_exempt": False,
                    },
                    {
                        "image_name": "000102.jpg",
                        "visible_object_ids": [3, 9],
                        "score": 8,
                        "attachment_viewpoint_exempt": False,
                    },
                ],
            ),
            patch.object(referability_module, "load_axis_alignment", return_value=np.eye(4, dtype=np.float64)),
            patch.object(
                referability_module,
                "load_scannet_poses",
                return_value={
                    "000001.jpg": make_camera_pose(),
                    "000101.jpg": make_camera_pose(),
                    "000102.jpg": make_camera_pose(),
                },
            ),
            patch.object(referability_module, "load_scannet_intrinsics", return_value=make_camera_intrinsics()),
            patch.object(referability_module, "load_scannet_depth_intrinsics", return_value=None),
            patch.object(
                referability_module.cv2,
                "imread",
                return_value=np.zeros((32, 32, 3), dtype=np.uint8),
            ),
            patch.object(referability_module, "_select_and_rerank_frames", side_effect=fake_select_and_rerank_frames),
            patch.object(
                referability_module,
                "_select_attachment_group_representatives",
                return_value=[dict(attachment_frame)],
            ),
            patch.object(
                referability_module,
                "_select_attachment_frames_by_global_pair_coverage",
                side_effect=lambda frames, max_frames: list(frames),
            ),
            patch.object(sys, "argv", [
                "run_vlm_referability.py",
                "--data_root",
                str(data_root),
                "--output",
                str(output_path),
                "--max_scenes",
                "1",
                "--max_frames",
                "2",
                "--non_attachment_group_debug_dir",
                str(debug_dir),
                "--no-write_attachment_review",
            ]),
        ):
            referability_module.main()

        cache_doc = json.loads(output_path.read_text(encoding="utf-8"))
        scene_grouping = cache_doc["scene_grouping"]["scene0001_00"]
        scene_status = cache_doc["scene_status"]["scene0001_00"]
        self.assertEqual(scene_grouping["pipeline_outcome"], "processed")
        self.assertIsNone(scene_grouping["scene_skip_reason"])
        self.assertEqual(scene_grouping["reranked_accepted_frame_image_names"], ["000101.jpg", "000102.jpg"])
        self.assertEqual(scene_grouping["selected_before_attachment_slots_image_names"], ["000101.jpg", "000102.jpg"])
        self.assertEqual(scene_grouping["attachment_selected_frame_image_names"], ["000001.jpg"])
        self.assertEqual(scene_grouping["remaining_slots_after_attachment_selection"], 1)
        self.assertEqual(scene_grouping["selected_after_attachment_slots_image_names"], ["000101.jpg"])
        self.assertEqual(scene_grouping["final_cacheable_frame_image_names"], ["000001.jpg", "000101.jpg"])
        self.assertEqual(scene_grouping["groups"][0]["status_after_attachment_slots"], "final_selected")
        self.assertEqual(scene_grouping["groups"][1]["selected_after_attachment_slots_image_names"], [])
        self.assertEqual(scene_grouping["groups"][1]["dropped_after_attachment_slots_image_names"], ["000102.jpg"])
        self.assertEqual(
            scene_grouping["groups"][1]["status_after_attachment_slots"],
            "dropped_by_attachment_slot_limit",
        )
        self.assertEqual(
            list(cache_doc["frames"]["scene0001_00"].keys()),
            ["000001.jpg", "000101.jpg"],
        )
        self.assertEqual(scene_status["pipeline_outcome"], "processed")
        self.assertEqual(scene_status["split"], "train")
        self.assertTrue(scene_status["has_cache_frames"])
        self.assertEqual(scene_status["final_cacheable_frame_count"], 2)

        debug_doc = json.loads((debug_dir / "scene0001_00.json").read_text(encoding="utf-8"))
        self.assertEqual(debug_doc, scene_grouping)

    def test_main_writes_empty_scene_grouping_summary_when_no_non_attachment_candidates(self) -> None:
        root = Path(__file__).resolve().parent / "_tmp" / f"scene_grouping_empty_{uuid.uuid4().hex}"
        data_root = root / "data"
        scene_dir = data_root / "scene0001_00"
        (scene_dir / "pose").mkdir(parents=True, exist_ok=True)
        output_path = root / "output" / "referability_cache.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        scene = {
            "objects": [
                make_object(1, "table"),
                make_object(2, "book"),
            ],
        }

        def fake_enrich(scene_dict: dict) -> dict:
            scene_dict["attachment_graph"] = {"1": [2]}
            scene_dict["attached_by"] = {"2": 1}
            scene_dict["attachment_edges"] = [{"parent_id": 1, "child_id": 2, "type": "supported_by"}]
            scene_dict["support_chain_graph"] = {"1": [2]}
            scene_dict["support_chain_by"] = {"2": 1}
            return scene_dict

        attachment_entry = make_debug_cache_entry()
        attachment_entry["attachment_referable_object_ids"] = [1, 2]
        attachment_frame = {
            "image_name": "000001.jpg",
            "visible_object_ids": [1, 2],
            "attachment_referable_object_ids": [1, 2],
            "attachment_view_group_id": 3,
            **attachment_entry,
        }

        self.addCleanup(shutil.rmtree, root, True)
        with (
            patch.dict(sys.modules, {"openai": make_fake_openai_module()}),
            patch("src.scene_parser.parse_scene", return_value=scene),
            patch("src.support_graph.enrich_scene_with_attachment", side_effect=fake_enrich),
            patch("src.support_graph.build_attachment_candidates", return_value=[]),
            patch.object(
                referability_module,
                "select_frames",
                return_value=[
                    {
                        "image_name": "000001.jpg",
                        "visible_object_ids": [1, 2],
                        "score": 10,
                        "attachment_viewpoint_exempt": True,
                    }
                ],
            ),
            patch.object(referability_module, "load_axis_alignment", return_value=np.eye(4, dtype=np.float64)),
            patch.object(
                referability_module,
                "load_scannet_poses",
                return_value={"000001.jpg": make_camera_pose()},
            ),
            patch.object(referability_module, "load_scannet_intrinsics", return_value=make_camera_intrinsics()),
            patch.object(referability_module, "load_scannet_depth_intrinsics", return_value=None),
            patch.object(
                referability_module,
                "_select_and_rerank_frames",
                side_effect=AssertionError("_select_and_rerank_frames should not run without non-attachment candidates"),
            ),
            patch.object(
                referability_module,
                "_select_attachment_group_representatives",
                return_value=[attachment_frame],
            ),
            patch.object(
                referability_module,
                "_select_attachment_frames_by_global_pair_coverage",
                side_effect=lambda frames, max_frames: list(frames),
            ),
            patch.object(sys, "argv", [
                "run_vlm_referability.py",
                "--data_root",
                str(data_root),
                "--output",
                str(output_path),
                "--max_scenes",
                "1",
                "--max_frames",
                "2",
                "--no-write_attachment_review",
            ]),
        ):
            referability_module.main()

        cache_doc = json.loads(output_path.read_text(encoding="utf-8"))
        scene_grouping = cache_doc["scene_grouping"]["scene0001_00"]
        scene_status = cache_doc["scene_status"]["scene0001_00"]
        self.assertEqual(scene_grouping["pipeline_outcome"], "processed")
        self.assertEqual(scene_grouping["non_attachment_candidate_frame_count"], 0)
        self.assertEqual(scene_grouping["non_attachment_visible_object_group_count"], 0)
        self.assertEqual(scene_grouping["non_attachment_processed_group_count"], 0)
        self.assertEqual(scene_grouping["groups"], [])
        self.assertEqual(scene_grouping["selected_after_attachment_slots_image_names"], [])
        self.assertEqual(scene_grouping["attachment_selected_frame_image_names"], ["000001.jpg"])
        self.assertEqual(scene_grouping["final_cacheable_frame_image_names"], ["000001.jpg"])
        self.assertEqual(scene_status["pipeline_outcome"], "processed")
        self.assertTrue(scene_status["has_cache_frames"])
        self.assertEqual(scene_status["final_cacheable_frame_count"], 1)

    def test_main_migrates_legacy_cached_scene_to_scene_status_and_skips_processing(self) -> None:
        root = Path(__file__).resolve().parent / "_tmp" / f"attachment_review_cached_{uuid.uuid4().hex}"
        data_root = root / "data"
        scene_dir = data_root / "scene0001_00"
        (scene_dir / "pose").mkdir(parents=True, exist_ok=True)
        output_path = root / "output" / "referability_cache.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        review_path = output_path.parent / f"{output_path.stem}_attachment_candidate_review.json"
        output_path.write_text(
            json.dumps(
                {
                    "version": referability_module.REFERABILITY_CACHE_VERSION,
                    "model": "fake-vlm",
                    "alias_config_version": "test",
                    "referability_backend": "crop_vlm_with_mesh_ray",
                    "label_batch_size": 1,
                    "frames": {
                        "scene0001_00": {
                            "000001.jpg": make_debug_cache_entry(),
                        }
                    },
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        self.addCleanup(shutil.rmtree, root, True)
        with (
            patch.dict(sys.modules, {"openai": make_fake_openai_module()}),
            patch("src.scene_parser.parse_scene", side_effect=AssertionError("scene_status should skip legacy cached scene")),
            patch("src.support_graph.enrich_scene_with_attachment", side_effect=AssertionError("scene_status should skip legacy cached scene")),
            patch("src.support_graph.build_attachment_candidates", side_effect=AssertionError("scene_status should skip legacy cached scene")),
            patch.object(referability_module, "select_frames", side_effect=AssertionError("select_frames should not run for cached scenes")),
            patch.object(referability_module, "load_axis_alignment", side_effect=AssertionError("load_axis_alignment should not run for cached scenes")),
            patch.object(referability_module, "load_scannet_poses", side_effect=AssertionError("load_scannet_poses should not run for cached scenes")),
            patch.object(referability_module, "load_scannet_intrinsics", side_effect=AssertionError("load_scannet_intrinsics should not run for cached scenes")),
            patch.object(referability_module, "load_scannet_depth_intrinsics", side_effect=AssertionError("load_scannet_depth_intrinsics should not run for cached scenes")),
            patch.object(sys, "argv", [
                "run_vlm_referability.py",
                "--data_root",
                str(data_root),
                "--output",
                str(output_path),
                "--max_scenes",
                "1",
                "--max_frames",
                "5",
                "--resume",
            ]),
        ):
            referability_module.main()

        self.assertTrue(review_path.exists())
        review_doc = json.loads(review_path.read_text(encoding="utf-8"))
        cache_doc = json.loads(output_path.read_text(encoding="utf-8"))
        self.assertEqual(review_doc["scene_count"], 0)
        self.assertEqual(review_doc["raw_candidate_edge_count"], 0)
        self.assertEqual(review_doc["raw_attachment_candidate_edge_count"], 0)
        self.assertEqual(review_doc["final_attachment_edge_count"], 0)
        self.assertEqual(review_doc["final_attachment_graph_edge_count"], 0)
        self.assertEqual(review_doc["attachment_graph_layers"]["raw_candidates"]["edge_count"], 0)
        self.assertEqual(review_doc["attachment_graph_layers"]["final_attachment_graph"]["edge_count"], 0)
        self.assertEqual(review_doc["terminal_output_lines"], [])
        self.assertEqual(cache_doc["scene_status"]["scene0001_00"]["pipeline_outcome"], "processed")
        self.assertTrue(cache_doc["scene_status"]["scene0001_00"]["has_cache_frames"])
        self.assertEqual(cache_doc["scene_status"]["scene0001_00"]["final_cacheable_frame_count"], 1)

    def test_main_writes_attachment_review_json_for_scene_without_attachment_relations(self) -> None:
        root = Path(__file__).resolve().parent / "_tmp" / f"attachment_review_empty_{uuid.uuid4().hex}"
        data_root = root / "data"
        scene_dir = data_root / "scene0001_00"
        (scene_dir / "pose").mkdir(parents=True, exist_ok=True)
        output_path = root / "output" / "referability_cache.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        review_path = output_path.parent / f"{output_path.stem}_attachment_candidate_review.json"
        scene = {
            "objects": [
                make_object(1, "table"),
                make_object(2, "book"),
            ],
        }

        def fake_enrich(scene_dict: dict) -> dict:
            scene_dict["attachment_graph"] = {}
            scene_dict["attached_by"] = {}
            scene_dict["attachment_edges"] = []
            scene_dict["support_chain_graph"] = {}
            scene_dict["support_chain_by"] = {}
            return scene_dict

        self.addCleanup(shutil.rmtree, root, True)
        with (
            patch.dict(sys.modules, {"openai": make_fake_openai_module()}),
            patch("src.scene_parser.parse_scene", return_value=scene),
            patch("src.support_graph.enrich_scene_with_attachment", side_effect=fake_enrich),
            patch("src.support_graph.build_attachment_candidates", return_value=[]),
            patch.object(referability_module, "select_frames", side_effect=AssertionError("select_frames should not run without attachment relations")),
            patch.object(sys, "argv", [
                "run_vlm_referability.py",
                "--data_root",
                str(data_root),
                "--output",
                str(output_path),
                "--max_scenes",
                "1",
                "--max_frames",
                "5",
            ]),
        ):
            referability_module.main()

        self.assertTrue(review_path.exists())
        review_doc = json.loads(review_path.read_text(encoding="utf-8"))
        scene_review = review_doc["scenes"][0]
        self.assertEqual(review_doc["scene_count"], 1)
        self.assertEqual(review_doc["raw_candidate_edge_count"], 0)
        self.assertEqual(review_doc["raw_attachment_candidate_edge_count"], 0)
        self.assertEqual(review_doc["final_attachment_edge_count"], 0)
        self.assertEqual(review_doc["final_attachment_graph_edge_count"], 0)
        self.assertEqual(scene_review["pipeline_outcome"], "no_attachment_relations")
        self.assertEqual(scene_review["raw_attachment_candidate_edge_count"], 0)
        self.assertEqual(scene_review["final_attachment_graph_edge_count"], 0)
        self.assertEqual(scene_review["candidate_rows"], [])
        self.assertIn("no_attachment_relations", review_doc["terminal_output_lines"][0])
        self.assertEqual(
            json.loads(output_path.read_text(encoding="utf-8"))["scene_status"]["scene0001_00"]["pipeline_outcome"],
            "no_attachment_relations",
        )

    def test_main_writes_scene_status_for_no_frame_candidates(self) -> None:
        root = Path(__file__).resolve().parent / "_tmp" / f"scene_status_no_frames_{uuid.uuid4().hex}"
        data_root = root / "data"
        scene_dir = data_root / "scans" / "scene0001_00"
        (scene_dir / "pose").mkdir(parents=True, exist_ok=True)
        output_path = root / "output" / "referability_cache.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        scene = {"objects": [make_object(1, "table"), make_object(2, "book")]}

        def fake_enrich(scene_dict: dict) -> dict:
            scene_dict["attachment_graph"] = {"1": [2]}
            scene_dict["attached_by"] = {"2": 1}
            scene_dict["attachment_edges"] = [{"parent_id": 1, "child_id": 2, "type": "supported_by"}]
            scene_dict["support_chain_graph"] = {"1": [2]}
            scene_dict["support_chain_by"] = {"2": 1}
            return scene_dict

        self.addCleanup(shutil.rmtree, root, True)
        with (
            patch.dict(sys.modules, {"openai": make_fake_openai_module()}),
            patch("src.scene_parser.parse_scene", return_value=scene),
            patch("src.support_graph.enrich_scene_with_attachment", side_effect=fake_enrich),
            patch("src.support_graph.build_attachment_candidates", return_value=[]),
            patch.object(referability_module, "select_frames", return_value=[]),
            patch.object(sys, "argv", [
                "run_vlm_referability.py",
                "--data_root",
                str(data_root),
                "--split",
                "train",
                "--output",
                str(output_path),
                "--max_scenes",
                "1",
                "--no-write_attachment_review",
            ]),
        ):
            referability_module.main()

        cache_doc = json.loads(output_path.read_text(encoding="utf-8"))
        scene_status = cache_doc["scene_status"]["scene0001_00"]
        self.assertEqual(scene_status["pipeline_outcome"], "no_frame_candidates")
        self.assertEqual(scene_status["scene_skip_reason"], "no_frame_candidates")
        self.assertFalse(scene_status["has_cache_frames"])
        self.assertEqual(scene_status["final_cacheable_frame_count"], 0)

    def test_no_final_referability_scene_status_prevents_repeat_processing(self) -> None:
        root = Path(__file__).resolve().parent / "_tmp" / f"scene_status_no_final_{uuid.uuid4().hex}"
        data_root = root / "data"
        scene_dir = data_root / "scans" / "scene0001_00"
        (scene_dir / "pose").mkdir(parents=True, exist_ok=True)
        output_path = root / "output" / "referability_cache.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        scene = {"objects": [make_object(1, "table"), make_object(2, "book")]}

        def fake_enrich(scene_dict: dict) -> dict:
            scene_dict["attachment_graph"] = {"1": [2]}
            scene_dict["attached_by"] = {"2": 1}
            scene_dict["attachment_edges"] = [{"parent_id": 1, "child_id": 2, "type": "supported_by"}]
            scene_dict["support_chain_graph"] = {"1": [2]}
            scene_dict["support_chain_by"] = {"2": 1}
            return scene_dict

        select_frames_return = [
            {
                "image_name": "000001.jpg",
                "visible_object_ids": [1, 2],
                "score": 10,
                "attachment_viewpoint_exempt": True,
            }
        ]

        self.addCleanup(shutil.rmtree, root, True)
        with (
            patch.dict(sys.modules, {"openai": make_fake_openai_module()}),
            patch("src.scene_parser.parse_scene", return_value=scene),
            patch("src.support_graph.enrich_scene_with_attachment", side_effect=fake_enrich),
            patch("src.support_graph.build_attachment_candidates", return_value=[]),
            patch.object(referability_module, "select_frames", return_value=select_frames_return),
            patch.object(referability_module, "load_axis_alignment", return_value=np.eye(4, dtype=np.float64)),
            patch.object(
                referability_module,
                "load_scannet_poses",
                return_value={"000001.jpg": make_camera_pose()},
            ),
            patch.object(referability_module, "load_scannet_intrinsics", return_value=make_camera_intrinsics()),
            patch.object(referability_module, "load_scannet_depth_intrinsics", return_value=None),
            patch.object(referability_module, "_select_attachment_group_representatives", return_value=[]),
            patch.object(
                referability_module,
                "_select_attachment_frames_by_global_pair_coverage",
                side_effect=lambda frames, max_frames: list(frames),
            ),
            patch.object(sys, "argv", [
                "run_vlm_referability.py",
                "--data_root",
                str(data_root),
                "--split",
                "train",
                "--output",
                str(output_path),
                "--scene_batch_size",
                "1",
                "--no-write_attachment_review",
            ]),
        ):
            referability_module.main()

        first_cache_doc = json.loads(output_path.read_text(encoding="utf-8"))
        self.assertEqual(
            first_cache_doc["scene_status"]["scene0001_00"]["pipeline_outcome"],
            "no_final_referability_frames",
        )

        with (
            patch.dict(sys.modules, {"openai": make_fake_openai_module()}),
            patch("src.scene_parser.parse_scene", side_effect=AssertionError("scene_status should prevent repeat processing")),
            patch.object(referability_module, "select_frames", side_effect=AssertionError("scene_status should prevent repeat processing")),
            patch.object(sys, "argv", [
                "run_vlm_referability.py",
                "--data_root",
                str(data_root),
                "--split",
                "train",
                "--output",
                str(output_path),
                "--scene_batch_size",
                "1",
                "--resume",
                "--no-write_attachment_review",
            ]),
        ):
            referability_module.main()

    def test_scene_batch_size_resume_migrates_legacy_cache_and_skips_to_next_unprocessed_scene(self) -> None:
        root = Path(__file__).resolve().parent / "_tmp" / f"scene_batch_resume_{uuid.uuid4().hex}"
        data_root = root / "data"
        scans_root = data_root / "scans"
        for scene_id in ("scene0001_00", "scene0002_00", "scene0003_00"):
            make_scene_dir(scans_root, scene_id)
        output_path = root / "output" / "referability_cache.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(
                {
                    "version": referability_module.REFERABILITY_CACHE_VERSION,
                    "model": "fake-vlm",
                    "alias_config_version": "test",
                    "referability_backend": "crop_vlm_with_mesh_ray",
                    "label_batch_size": 1,
                    "frames": {
                        "scene0001_00": {
                            "000001.jpg": make_debug_cache_entry(),
                        }
                    },
                    "scene_grouping": {
                        "scene0002_00": {
                            "scene_id": "scene0002_00",
                            "pipeline_outcome": "no_final_referability_frames",
                            "scene_skip_reason": "no_final_referability_frames",
                            "final_cacheable_frame_count": 0,
                        }
                    },
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        scene = {"objects": [make_object(1, "table"), make_object(2, "book")]}
        select_calls: list[str] = []

        def fake_enrich(scene_dict: dict) -> dict:
            scene_dict["attachment_graph"] = {"1": [2]}
            scene_dict["attached_by"] = {"2": 1}
            scene_dict["attachment_edges"] = [{"parent_id": 1, "child_id": 2, "type": "supported_by"}]
            scene_dict["support_chain_graph"] = {"1": [2]}
            scene_dict["support_chain_by"] = {"2": 1}
            return scene_dict

        def fake_select_frames(scene_dir: Path, *args, **kwargs):
            select_calls.append(scene_dir.name)
            return [
                {
                    "image_name": "000001.jpg",
                    "visible_object_ids": [1, 2],
                    "score": 10,
                    "attachment_viewpoint_exempt": True,
                }
            ]

        attachment_entry = make_debug_cache_entry()
        attachment_entry["image_name"] = "000001.jpg"
        attachment_entry["attachment_referable_object_ids"] = [1, 2]
        attachment_entry["attachment_view_group_id"] = 1

        self.addCleanup(shutil.rmtree, root, True)
        with (
            patch.dict(sys.modules, {"openai": make_fake_openai_module()}),
            patch("src.scene_parser.parse_scene", return_value=scene),
            patch("src.support_graph.enrich_scene_with_attachment", side_effect=fake_enrich),
            patch("src.support_graph.build_attachment_candidates", return_value=[]),
            patch.object(referability_module, "select_frames", side_effect=fake_select_frames),
            patch.object(referability_module, "load_axis_alignment", return_value=np.eye(4, dtype=np.float64)),
            patch.object(
                referability_module,
                "load_scannet_poses",
                return_value={"000001.jpg": make_camera_pose()},
            ),
            patch.object(referability_module, "load_scannet_intrinsics", return_value=make_camera_intrinsics()),
            patch.object(referability_module, "load_scannet_depth_intrinsics", return_value=None),
            patch.object(
                referability_module,
                "_select_attachment_group_representatives",
                return_value=[attachment_entry],
            ),
            patch.object(
                referability_module,
                "_select_attachment_frames_by_global_pair_coverage",
                side_effect=lambda frames, max_frames: list(frames),
            ),
            patch.object(sys, "argv", [
                "run_vlm_referability.py",
                "--data_root",
                str(data_root),
                "--split",
                "train",
                "--output",
                str(output_path),
                "--scene_batch_size",
                "1",
                "--resume",
                "--no-write_attachment_review",
            ]),
        ):
            referability_module.main()

        cache_doc = json.loads(output_path.read_text(encoding="utf-8"))
        self.assertEqual(select_calls, ["scene0003_00"])
        self.assertEqual(cache_doc["scene_status"]["scene0001_00"]["pipeline_outcome"], "processed")
        self.assertEqual(
            cache_doc["scene_status"]["scene0002_00"]["pipeline_outcome"],
            "no_final_referability_frames",
        )
        self.assertEqual(cache_doc["scene_status"]["scene0003_00"]["pipeline_outcome"], "processed")

    def test_final_scene_batch_logs_banner_and_processes_all_remaining_scenes(self) -> None:
        root = Path(__file__).resolve().parent / "_tmp" / f"final_batch_banner_{uuid.uuid4().hex}"
        data_root = root / "data"
        scans_root = data_root / "scans"
        for scene_id in ("scene0001_00", "scene0002_00"):
            make_scene_dir(scans_root, scene_id)
        output_path = root / "output" / "referability_cache.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        scene = {"objects": [make_object(1, "table"), make_object(2, "book")]}

        def fake_enrich(scene_dict: dict) -> dict:
            scene_dict["attachment_graph"] = {"1": [2]}
            scene_dict["attached_by"] = {"2": 1}
            scene_dict["attachment_edges"] = [{"parent_id": 1, "child_id": 2, "type": "supported_by"}]
            scene_dict["support_chain_graph"] = {"1": [2]}
            scene_dict["support_chain_by"] = {"2": 1}
            return scene_dict

        def fake_select_frames(scene_dir: Path, *args, **kwargs):
            return [
                {
                    "image_name": "000001.jpg",
                    "visible_object_ids": [1, 2],
                    "score": 10,
                    "attachment_viewpoint_exempt": True,
                }
            ]

        attachment_entry = make_debug_cache_entry()
        attachment_entry["image_name"] = "000001.jpg"
        attachment_entry["attachment_referable_object_ids"] = [1, 2]
        attachment_entry["attachment_view_group_id"] = 1

        self.addCleanup(shutil.rmtree, root, True)
        with (
            self.assertLogs(referability_module.logger.name, level="WARNING") as logs,
            patch.dict(sys.modules, {"openai": make_fake_openai_module()}),
            patch("src.scene_parser.parse_scene", return_value=scene),
            patch("src.support_graph.enrich_scene_with_attachment", side_effect=fake_enrich),
            patch("src.support_graph.build_attachment_candidates", return_value=[]),
            patch.object(referability_module, "select_frames", side_effect=fake_select_frames),
            patch.object(referability_module, "load_axis_alignment", return_value=np.eye(4, dtype=np.float64)),
            patch.object(
                referability_module,
                "load_scannet_poses",
                return_value={"000001.jpg": make_camera_pose()},
            ),
            patch.object(referability_module, "load_scannet_intrinsics", return_value=make_camera_intrinsics()),
            patch.object(referability_module, "load_scannet_depth_intrinsics", return_value=None),
            patch.object(
                referability_module,
                "_select_attachment_group_representatives",
                return_value=[attachment_entry],
            ),
            patch.object(
                referability_module,
                "_select_attachment_frames_by_global_pair_coverage",
                side_effect=lambda frames, max_frames: list(frames),
            ),
            patch.object(sys, "argv", [
                "run_vlm_referability.py",
                "--data_root",
                str(data_root),
                "--split",
                "train",
                "--output",
                str(output_path),
                "--scene_batch_size",
                "5",
                "--no-write_attachment_review",
            ]),
        ):
            referability_module.main()

        log_text = "\n".join(logs.output)
        cache_doc = json.loads(output_path.read_text(encoding="utf-8"))
        self.assertIn("FINAL BATCH FOR SPLIT train", log_text)
        self.assertIn("ALL SCENES PROCESSED AFTER THIS RUN", log_text)
        self.assertEqual(sorted(cache_doc["scene_status"].keys()), ["scene0001_00", "scene0002_00"])

    def test_max_scenes_without_scene_batch_size_keeps_legacy_limit_behavior(self) -> None:
        root = Path(__file__).resolve().parent / "_tmp" / f"legacy_max_scenes_{uuid.uuid4().hex}"
        data_root = root / "data"
        scans_root = data_root / "scans"
        for scene_id in ("scene0001_00", "scene0002_00"):
            make_scene_dir(scans_root, scene_id)
        output_path = root / "output" / "referability_cache.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        scene = {"objects": [make_object(1, "table"), make_object(2, "book")]}
        select_calls: list[str] = []

        def fake_enrich(scene_dict: dict) -> dict:
            scene_dict["attachment_graph"] = {"1": [2]}
            scene_dict["attached_by"] = {"2": 1}
            scene_dict["attachment_edges"] = [{"parent_id": 1, "child_id": 2, "type": "supported_by"}]
            scene_dict["support_chain_graph"] = {"1": [2]}
            scene_dict["support_chain_by"] = {"2": 1}
            return scene_dict

        def fake_select_frames(scene_dir: Path, *args, **kwargs):
            select_calls.append(scene_dir.name)
            return [
                {
                    "image_name": "000001.jpg",
                    "visible_object_ids": [1, 2],
                    "score": 10,
                    "attachment_viewpoint_exempt": True,
                }
            ]

        attachment_entry = make_debug_cache_entry()
        attachment_entry["image_name"] = "000001.jpg"
        attachment_entry["attachment_referable_object_ids"] = [1, 2]
        attachment_entry["attachment_view_group_id"] = 1

        self.addCleanup(shutil.rmtree, root, True)
        with (
            patch.dict(sys.modules, {"openai": make_fake_openai_module()}),
            patch("src.scene_parser.parse_scene", return_value=scene),
            patch("src.support_graph.enrich_scene_with_attachment", side_effect=fake_enrich),
            patch("src.support_graph.build_attachment_candidates", return_value=[]),
            patch.object(referability_module, "select_frames", side_effect=fake_select_frames),
            patch.object(referability_module, "load_axis_alignment", return_value=np.eye(4, dtype=np.float64)),
            patch.object(
                referability_module,
                "load_scannet_poses",
                return_value={"000001.jpg": make_camera_pose()},
            ),
            patch.object(referability_module, "load_scannet_intrinsics", return_value=make_camera_intrinsics()),
            patch.object(referability_module, "load_scannet_depth_intrinsics", return_value=None),
            patch.object(
                referability_module,
                "_select_attachment_group_representatives",
                return_value=[attachment_entry],
            ),
            patch.object(
                referability_module,
                "_select_attachment_frames_by_global_pair_coverage",
                side_effect=lambda frames, max_frames: list(frames),
            ),
            patch.object(sys, "argv", [
                "run_vlm_referability.py",
                "--data_root",
                str(data_root),
                "--split",
                "train",
                "--output",
                str(output_path),
                "--max_scenes",
                "1",
                "--no-write_attachment_review",
            ]),
        ):
            referability_module.main()

        self.assertEqual(select_calls, ["scene0001_00"])

    def test_main_logs_final_vlm_failure_count(self) -> None:
        root = Path(__file__).resolve().parent / "_tmp" / f"vlm_failure_count_{uuid.uuid4().hex}"
        data_root = root / "data"
        scene_dir = data_root / "scene0001_00"
        (scene_dir / "pose").mkdir(parents=True, exist_ok=True)
        output_path = root / "output" / "referability_cache.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        scene = {
            "objects": [
                make_object(1, "table"),
                make_object(2, "book"),
            ],
        }

        def fake_enrich(scene_dict: dict) -> dict:
            scene_dict["attachment_graph"] = {}
            scene_dict["attached_by"] = {}
            scene_dict["attachment_edges"] = []
            scene_dict["support_chain_graph"] = {}
            scene_dict["support_chain_by"] = {}
            return scene_dict

        self.addCleanup(shutil.rmtree, root, True)
        with (
            patch.dict(sys.modules, {"openai": make_fake_openai_module()}),
            patch("src.scene_parser.parse_scene", return_value=scene),
            patch("src.support_graph.enrich_scene_with_attachment", side_effect=fake_enrich),
            patch("src.support_graph.build_attachment_candidates", return_value=[]),
            patch.object(referability_module, "select_frames", side_effect=AssertionError("select_frames should not run without attachment relations")),
            patch.object(referability_module.logger, "info") as info_mock,
            patch.object(sys, "argv", [
                "run_vlm_referability.py",
                "--data_root",
                str(data_root),
                "--output",
                str(output_path),
                "--max_scenes",
                "1",
                "--max_frames",
                "5",
            ]),
        ):
            referability_module.main()

        self.assertIn(call("VLM call failures: %d", 0), info_mock.call_args_list)


if __name__ == "__main__":
    unittest.main()
