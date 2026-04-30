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


def make_camera_pose(
    *,
    image_name: str = "000000.jpg",
    yaw_deg: float = 0.0,
) -> CameraPose:
    yaw_rad = np.deg2rad(float(yaw_deg))
    cos_yaw = float(np.cos(yaw_rad))
    sin_yaw = float(np.sin(yaw_rad))
    rotation = np.array(
        [
            [cos_yaw, 0.0, sin_yaw],
            [0.0, 1.0, 0.0],
            [-sin_yaw, 0.0, cos_yaw],
        ],
        dtype=np.float64,
    )
    return CameraPose(
        image_name=image_name,
        rotation=rotation,
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


def make_attachment_pair_salvage_entry(
    *,
    candidate_visible_object_ids: list[int],
    object_reviews: dict[int, dict] | None = None,
    crop_label_statuses: dict[str, str] | None = None,
    full_frame_label_statuses: dict[str, str] | None = None,
    label_statuses: dict[str, str] | None = None,
    attachment_referable_pairs: list[list[int]] | None = None,
) -> dict:
    entry = make_debug_cache_entry()
    entry["candidate_visible_object_ids"] = list(candidate_visible_object_ids)
    normalized_reviews: dict[str, dict] = {}
    review_overrides = object_reviews or {}
    for obj_id in sorted({*candidate_visible_object_ids, *review_overrides.keys()}):
        review = {
            "obj_id": int(obj_id),
            "local_outcome": "reviewed",
            "vlm_status": None,
            "bbox_in_frame_ratio": 0.9,
            "projected_area_px": 900.0,
            "roi_bounds_px": [20, 60, 20, 60],
            "crop_bounds_px": [16, 64, 16, 64],
        }
        review.update(review_overrides.get(int(obj_id), {}))
        normalized_reviews[str(int(obj_id))] = review
    entry["object_reviews"] = normalized_reviews
    entry["crop_label_statuses"] = dict(crop_label_statuses or {})
    entry["full_frame_label_statuses"] = dict(full_frame_label_statuses or {})
    entry["label_statuses"] = dict(label_statuses or {})
    entry["crop_label_counts"] = {
        label: (0 if status == "absent" else 2 if status == "multiple" else 1)
        for label, status in entry["crop_label_statuses"].items()
    }
    entry["full_frame_label_counts"] = {
        label: (0 if status == "absent" else 2 if status == "multiple" else 1)
        for label, status in entry["full_frame_label_statuses"].items()
    }
    entry["label_counts"] = {
        label: (0 if status == "absent" else 2 if status == "multiple" else 1)
        for label, status in entry["label_statuses"].items()
    }
    entry["attachment_referable_pairs"] = [list(pair) for pair in (attachment_referable_pairs or [])]
    entry["attachment_referable_pair_count"] = len(entry["attachment_referable_pairs"])
    entry["attachment_referable_object_ids"] = sorted(
        {
            int(obj_id)
            for pair in entry["attachment_referable_pairs"]
            for obj_id in pair
        }
    )
    entry["attachment_final_referability"] = {
        "object_ids": list(entry["attachment_referable_object_ids"]),
        "pairs": [list(pair) for pair in entry["attachment_referable_pairs"]],
        "pair_count": len(entry["attachment_referable_pairs"]),
    }
    return entry


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


def resolve_output_dir(output_arg: Path) -> Path:
    return output_arg.parent if output_arg.suffix.lower() == ".json" else output_arg


def scene_status_path_for_output(output_arg: Path) -> Path:
    return resolve_output_dir(output_arg) / "scene_status.json"


def load_scene_status_doc_for_output(output_arg: Path) -> dict:
    return json.loads(scene_status_path_for_output(output_arg).read_text(encoding="utf-8"))


def list_batch_cache_paths(output_arg: Path) -> list[Path]:
    output_dir = resolve_output_dir(output_arg)
    return sorted(
        path
        for path in output_dir.glob("*.json")
        if path.name != "scene_status.json"
    )


def load_single_batch_cache_for_output(output_arg: Path) -> tuple[Path, dict]:
    batch_paths = list_batch_cache_paths(output_arg)
    if len(batch_paths) != 1:
        raise AssertionError(f"Expected exactly one batch cache for {output_arg}, found {batch_paths}")
    batch_path = batch_paths[0]
    return batch_path, json.loads(batch_path.read_text(encoding="utf-8"))


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
    def setUp(self) -> None:
        super().setUp()
        self._split_metadata_root = (
            Path(__file__).resolve().parent / "_tmp" / f"split_metadata_{uuid.uuid4().hex}"
        )
        self._split_metadata_root.mkdir(parents=True, exist_ok=True)
        self._train_split_path = self._split_metadata_root / "scannetv2_train.txt"
        self._val_split_path = self._split_metadata_root / "scannetv2_val.txt"
        self.write_split_file("train", ["scene0001_00", "scene0002_00", "scene0003_00"])
        self.write_split_file("val", ["scene1000_00", "scene1001_00"])
        self._split_patch = patch.dict(
            referability_module.SCANNET_METADATA_SPLIT_FILES,
            {
                "train": self._train_split_path,
                "val": self._val_split_path,
            },
            clear=True,
        )
        self._split_patch.start()
        self.addCleanup(self._split_patch.stop)
        self.addCleanup(shutil.rmtree, self._split_metadata_root, True)

    def write_split_file(self, split: str, scene_ids: list[str]) -> None:
        split_path = {
            "train": self._train_split_path,
            "val": self._val_split_path,
        }[split]
        split_path.write_text("\n".join(scene_ids) + "\n", encoding="utf-8")

    def test_default_review_output_prefix_uses_trailing_flash_suffix(self) -> None:
        output_path = Path("output/pilot_referability_cache_qwen3_vl_flash12.json")

        prefix = referability_module._default_review_output_prefix(output_path)

        self.assertEqual(prefix, "flash12")

    def test_default_review_output_prefix_falls_back_to_full_stem(self) -> None:
        output_path = Path("output/referability_cache.json")

        prefix = referability_module._default_review_output_prefix(output_path)

        self.assertEqual(prefix, "referability_cache")

    def test_resolve_scannet_scene_dirs_reads_train_from_data_root(self) -> None:
        root = Path(__file__).resolve().parent / "_tmp" / f"resolve_train_{uuid.uuid4().hex}"
        data_root = root / "data"
        make_scene_dir(data_root, "scans/scene0002_00")
        make_scene_dir(data_root, "scans/scene0001_00")
        self.addCleanup(shutil.rmtree, root, True)

        entries = referability_module._resolve_scannet_scene_dirs(data_root, "train")

        self.assertEqual(
            [(split, path.name) for split, path in entries],
            [("train", "scene0001_00"), ("train", "scene0002_00")],
        )

    def test_resolve_scannet_scene_dirs_reads_val_from_scans_root(self) -> None:
        root = Path(__file__).resolve().parent / "_tmp" / f"resolve_val_{uuid.uuid4().hex}"
        data_root = root / "data"
        scans_root = data_root / "scans"
        make_scene_dir(data_root, "scans/scene1001_00")
        make_scene_dir(data_root, "scans/scene0001_00")
        self.addCleanup(shutil.rmtree, root, True)

        entries = referability_module._resolve_scannet_scene_dirs(scans_root, "val")

        self.assertEqual(
            [(split, path.name) for split, path in entries],
            [("val", "scene1001_00")],
        )

    def test_resolve_scannet_scene_dirs_reads_all_in_train_then_val_order(self) -> None:
        root = Path(__file__).resolve().parent / "_tmp" / f"resolve_all_{uuid.uuid4().hex}"
        data_root = root / "data"
        make_scene_dir(data_root, "scans/scene0002_00")
        make_scene_dir(data_root, "scans/scene0001_00")
        make_scene_dir(data_root, "scans/scene1001_00")
        make_scene_dir(data_root, "scans/scene1000_00")
        self.addCleanup(shutil.rmtree, root, True)

        entries = referability_module._resolve_scannet_scene_dirs(data_root, "all")

        self.assertEqual(
            [(split, path.name) for split, path in entries],
            [
                ("train", "scene0001_00"),
                ("train", "scene0002_00"),
                ("val", "scene1000_00"),
                ("val", "scene1001_00"),
            ],
        )

    def test_read_scannet_split_scene_ids_ignores_blank_lines(self) -> None:
        self._train_split_path.write_text("scene0002_00\n\nscene0001_00\n", encoding="utf-8")

        scene_ids = referability_module._read_scannet_split_scene_ids("train")

        self.assertEqual(scene_ids, ["scene0002_00", "scene0001_00"])

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
            {"image_name": "000030.jpg", "score": 19, "n_visible": 4, "visible_object_ids": [1, 2, 9]},
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
                poses={
                    "000000.jpg": make_camera_pose(image_name="000000.jpg", yaw_deg=0.0),
                    "000030.jpg": make_camera_pose(image_name="000030.jpg", yaw_deg=10.0),
                    "000060.jpg": make_camera_pose(image_name="000060.jpg", yaw_deg=45.0),
                },
                debug_output=debug_output,
            )

        self.assertEqual(frame_decision_mock.call_count, 2)
        self.assertEqual([entry["image_name"] for entry in selected], ["000000.jpg", "000060.jpg"])
        self.assertEqual(debug_output["non_attachment_visible_object_group_count"], 2)
        self.assertEqual(debug_output["non_attachment_processed_group_count"], 2)
        self.assertEqual(len(debug_output["groups"]), 2)
        self.assertEqual(debug_output["groups"][0]["group_key_visible_object_ids"], [1, 2, 9])
        self.assertEqual(
            debug_output["groups"][0]["candidate_frame_image_names"],
            ["000000.jpg", "000030.jpg"],
        )
        self.assertEqual(
            debug_output["groups"][0]["sampled_frame_image_names"],
            ["000000.jpg", "000030.jpg"],
        )
        self.assertEqual(debug_output["groups"][0]["accepted_frame_image_names"], ["000000.jpg"])

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

    def test_build_visible_object_pose_merged_groups_merges_when_visible_diff_and_pose_match(self) -> None:
        frames = [
            {"image_name": "000000.jpg", "visible_object_ids": [1, 2]},
            {"image_name": "000010.jpg", "visible_object_ids": [1, 2, 9]},
            {"image_name": "000020.jpg", "visible_object_ids": [9, 2, 1]},
        ]

        groups = referability_module._build_visible_object_pose_merged_groups(
            frames=frames,
            poses={
                "000000.jpg": make_camera_pose(image_name="000000.jpg", yaw_deg=0.0),
                "000010.jpg": make_camera_pose(image_name="000010.jpg", yaw_deg=10.0),
                "000020.jpg": make_camera_pose(image_name="000020.jpg", yaw_deg=12.0),
            },
        )

        self.assertEqual(len(groups), 1)
        self.assertEqual(
            [frame["image_name"] for frame in groups[0]["frames"]],
            ["000000.jpg", "000010.jpg", "000020.jpg"],
        )
        self.assertEqual(groups[0]["visible_object_ids"], [1, 2, 9])
        self.assertEqual(groups[0]["anchor_frame"]["image_name"], "000000.jpg")

    def test_build_visible_object_pose_merged_groups_keeps_separate_groups_when_pose_angle_exceeds_threshold(self) -> None:
        frames = [
            {"image_name": "000000.jpg", "visible_object_ids": [1, 2]},
            {"image_name": "000010.jpg", "visible_object_ids": [1, 2, 9]},
        ]

        groups = referability_module._build_visible_object_pose_merged_groups(
            frames=frames,
            poses={
                "000000.jpg": make_camera_pose(image_name="000000.jpg", yaw_deg=0.0),
                "000010.jpg": make_camera_pose(image_name="000010.jpg", yaw_deg=25.0),
            },
        )

        self.assertEqual(len(groups), 2)
        self.assertEqual(groups[0]["visible_object_ids"], [1, 2])
        self.assertEqual(groups[1]["visible_object_ids"], [1, 2, 9])

    def test_build_visible_object_pose_merged_groups_keeps_separate_groups_when_visible_symmetric_diff_is_too_large(self) -> None:
        frames = [
            {"image_name": "000000.jpg", "visible_object_ids": [1, 2]},
            {"image_name": "000010.jpg", "visible_object_ids": [1, 2, 9, 10, 11, 12]},
        ]

        groups = referability_module._build_visible_object_pose_merged_groups(
            frames=frames,
            poses={
                "000000.jpg": make_camera_pose(image_name="000000.jpg", yaw_deg=0.0),
                "000010.jpg": make_camera_pose(image_name="000010.jpg", yaw_deg=10.0),
            },
        )

        self.assertEqual(len(groups), 2)

    def test_build_visible_object_pose_merged_groups_missing_pose_only_allows_exact_visible_match(self) -> None:
        mergeable_frames = [
            {"image_name": "000000.jpg", "visible_object_ids": [1, 2]},
            {"image_name": "000010.jpg", "visible_object_ids": [2, 1]},
        ]
        non_mergeable_frames = [
            {"image_name": "000000.jpg", "visible_object_ids": [1, 2]},
            {"image_name": "000020.jpg", "visible_object_ids": [1, 2, 9]},
        ]

        merged_groups = referability_module._build_visible_object_pose_merged_groups(
            frames=mergeable_frames,
            poses={"000000.jpg": make_camera_pose(image_name="000000.jpg")},
        )
        split_groups = referability_module._build_visible_object_pose_merged_groups(
            frames=non_mergeable_frames,
            poses={"000000.jpg": make_camera_pose(image_name="000000.jpg")},
        )

        self.assertEqual(len(merged_groups), 1)
        self.assertEqual(merged_groups[0]["visible_object_ids"], [1, 2])
        self.assertEqual(len(split_groups), 2)

    def test_select_and_rerank_frames_stats_use_merged_non_attachment_groups(self) -> None:
        frame_candidates = [
            {"image_name": "000000.jpg", "score": 20, "n_visible": 5, "visible_object_ids": [1, 2]},
            {"image_name": "000030.jpg", "score": 19, "n_visible": 4, "visible_object_ids": [1, 2, 9]},
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
        ]
        stats_output: dict[str, Any] = {}

        root = Path(__file__).resolve().parent / "_tmp" / f"rerank_stats_merged_groups_{uuid.uuid4().hex}"
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
                max_frames=3,
                poses={
                    "000000.jpg": make_camera_pose(image_name="000000.jpg", yaw_deg=0.0),
                    "000030.jpg": make_camera_pose(image_name="000030.jpg", yaw_deg=10.0),
                    "000060.jpg": make_camera_pose(image_name="000060.jpg", yaw_deg=45.0),
                },
                stats_output=stats_output,
            )

        self.assertEqual([entry["image_name"] for entry in selected], ["000000.jpg", "000060.jpg"])
        self.assertEqual(stats_output["non_attachment_visible_object_group_count"], 2)
        self.assertEqual(stats_output["non_attachment_processed_group_count"], 2)
        self.assertEqual(stats_output["accepted_frame_count_after_group_scan"], 2)

    def test_build_attachment_frame_groups_merges_frames_with_same_pair_set_small_visible_diff_and_pose_angle(self) -> None:
        frames = [
            {"image_name": "000000.jpg", "visible_object_ids": [1, 2]},
            {"image_name": "000010.jpg", "visible_object_ids": [1, 2, 9]},
            {"image_name": "000020.jpg", "visible_object_ids": [9, 2, 1]},
        ]

        groups = referability_module._build_attachment_frame_groups(
            frames=frames,
            attachment_graph={1: [2]},
            poses={
                "000000.jpg": make_camera_pose(image_name="000000.jpg", yaw_deg=0.0),
                "000010.jpg": make_camera_pose(image_name="000010.jpg", yaw_deg=10.0),
                "000020.jpg": make_camera_pose(image_name="000020.jpg", yaw_deg=12.0),
            },
        )

        self.assertEqual(len(groups), 1)
        self.assertEqual(
            [frame["image_name"] for frame in groups[0]["frames"]],
            ["000000.jpg", "000010.jpg", "000020.jpg"],
        )
        self.assertEqual(groups[0]["visible_object_ids"], [1, 2, 9])
        self.assertEqual(groups[0]["group_pairs"], [(1, 2)])

    def test_build_attachment_frame_groups_keeps_separate_groups_when_pose_angle_exceeds_threshold(self) -> None:
        frames = [
            {"image_name": "000000.jpg", "visible_object_ids": [1, 2]},
            {"image_name": "000010.jpg", "visible_object_ids": [1, 2, 9]},
        ]

        groups = referability_module._build_attachment_frame_groups(
            frames=frames,
            attachment_graph={1: [2]},
            poses={
                "000000.jpg": make_camera_pose(image_name="000000.jpg", yaw_deg=0.0),
                "000010.jpg": make_camera_pose(image_name="000010.jpg", yaw_deg=25.0),
            },
        )

        self.assertEqual(len(groups), 2)
        self.assertEqual(groups[0]["visible_object_ids"], [1, 2])
        self.assertEqual(groups[1]["visible_object_ids"], [1, 2, 9])

    def test_build_attachment_frame_groups_keeps_separate_groups_when_visible_symmetric_diff_is_too_large(self) -> None:
        frames = [
            {"image_name": "000000.jpg", "visible_object_ids": [1, 2]},
            {"image_name": "000010.jpg", "visible_object_ids": [1, 2, 9, 10, 11, 12]},
        ]

        groups = referability_module._build_attachment_frame_groups(
            frames=frames,
            attachment_graph={1: [2]},
            poses={
                "000000.jpg": make_camera_pose(image_name="000000.jpg", yaw_deg=0.0),
                "000010.jpg": make_camera_pose(image_name="000010.jpg", yaw_deg=10.0),
            },
        )

        self.assertEqual(len(groups), 2)

    def test_build_attachment_frame_groups_never_merges_different_pair_sets(self) -> None:
        frames = [
            {"image_name": "000000.jpg", "visible_object_ids": [1, 2]},
            {"image_name": "000010.jpg", "visible_object_ids": [1, 2, 3]},
        ]

        groups = referability_module._build_attachment_frame_groups(
            frames=frames,
            attachment_graph={1: [2, 3]},
            poses={
                "000000.jpg": make_camera_pose(image_name="000000.jpg", yaw_deg=0.0),
                "000010.jpg": make_camera_pose(image_name="000010.jpg", yaw_deg=10.0),
            },
        )

        self.assertEqual(len(groups), 2)
        self.assertEqual(groups[0]["group_pairs"], [(1, 2)])
        self.assertEqual(groups[1]["group_pairs"], [(1, 2), (1, 3)])

    def test_build_attachment_frame_groups_missing_pose_only_allows_exact_visible_match(self) -> None:
        mergeable_frames = [
            {"image_name": "000000.jpg", "visible_object_ids": [1, 2]},
            {"image_name": "000010.jpg", "visible_object_ids": [2, 1]},
        ]
        non_mergeable_frames = [
            {"image_name": "000000.jpg", "visible_object_ids": [1, 2]},
            {"image_name": "000020.jpg", "visible_object_ids": [1, 2, 9]},
        ]

        merged_groups = referability_module._build_attachment_frame_groups(
            frames=mergeable_frames,
            attachment_graph={1: [2]},
            poses={"000000.jpg": make_camera_pose(image_name="000000.jpg")},
        )
        split_groups = referability_module._build_attachment_frame_groups(
            frames=non_mergeable_frames,
            attachment_graph={1: [2]},
            poses={"000000.jpg": make_camera_pose(image_name="000000.jpg")},
        )

        self.assertEqual(len(merged_groups), 1)
        self.assertEqual(merged_groups[0]["visible_object_ids"], [1, 2])
        self.assertEqual(len(split_groups), 2)

    def test_select_attachment_group_representatives_merges_near_duplicate_visible_groups_when_pair_set_and_pose_match(self) -> None:
        frames = [
            {"image_name": "000000.jpg", "score": 20, "n_visible": 2, "visible_object_ids": [1, 2]},
            {"image_name": "000010.jpg", "score": 19, "n_visible": 3, "visible_object_ids": [1, 2, 9]},
            {"image_name": "000020.jpg", "score": 18, "n_visible": 3, "visible_object_ids": [9, 2, 1]},
        ]
        frame_decisions = [{
            "clear": True,
            "clarity_score": 72,
            "frame_usable": True,
            "reason": "clear",
        }]

        root = Path(__file__).resolve().parent / "_tmp" / f"attachment_group_visible_ids_{uuid.uuid4().hex}"
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
                if frame["image_name"] == "000000.jpg":
                    return {"attachment_referable_object_ids": [1, 2]}
                raise AssertionError(f"unexpected frame {frame['image_name']}")

            selected = referability_module._select_attachment_group_representatives(
                client=object(),
                model_name="fake-vlm",
                scene_dir=scene_dir,
                frames=frames,
                attachment_graph={1: [2]},
                poses={
                    "000000.jpg": make_camera_pose(image_name="000000.jpg", yaw_deg=0.0),
                    "000010.jpg": make_camera_pose(image_name="000010.jpg", yaw_deg=10.0),
                    "000020.jpg": make_camera_pose(image_name="000020.jpg", yaw_deg=12.0),
                },
                attachment_entry_builder=build_entry,
            )

        self.assertEqual(frame_decision_mock.call_count, 1)
        self.assertEqual(build_calls, ["000000.jpg"])
        self.assertEqual([entry["image_name"] for entry in selected], ["000000.jpg"])
        self.assertEqual([entry["attachment_view_group_id"] for entry in selected], [0])

    def test_select_attachment_group_representatives_checks_group_frames_until_first_clear_pair_frame(self) -> None:
        frames = [
            {"image_name": "000000.jpg", "score": 20, "n_visible": 3, "visible_object_ids": [1, 2, 9]},
            {"image_name": "000010.jpg", "score": 19, "n_visible": 3, "visible_object_ids": [9, 1, 2]},
            {"image_name": "000020.jpg", "score": 18, "n_visible": 3, "visible_object_ids": [2, 9, 1]},
        ]
        frame_decisions = [
            {
                "clear": True,
                "clarity_score": 69,
                "frame_usable": True,
                "reason": "below threshold",
            },
            {
                "clear": True,
                "clarity_score": 74,
                "frame_usable": True,
                "reason": "clear",
            },
            {
                "clear": True,
                "clarity_score": 75,
                "frame_usable": True,
                "reason": "clear and referable",
            },
        ]

        root = Path(__file__).resolve().parent / "_tmp" / f"attachment_group_pair_search_{uuid.uuid4().hex}"
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
                if frame["image_name"] == "000020.jpg":
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

        self.assertEqual(frame_decision_mock.call_count, 3)
        self.assertEqual(build_calls, ["000010.jpg", "000020.jpg"])
        self.assertEqual([entry["image_name"] for entry in selected], ["000020.jpg"])
        self.assertEqual(selected[0]["attachment_referable_pairs"], [[1, 2]])
        self.assertEqual(selected[0]["attachment_referable_pair_count"], 1)
        self.assertEqual(selected[0]["attachment_view_group_id"], 0)

    def test_select_attachment_group_representatives_skips_groups_without_clear_pair_frame(self) -> None:
        frames = [
            {"image_name": "000000.jpg", "score": 20, "n_visible": 2, "visible_object_ids": [1, 2]},
            {"image_name": "000010.jpg", "score": 19, "n_visible": 2, "visible_object_ids": [2, 1]},
            {"image_name": "000020.jpg", "score": 18, "n_visible": 2, "visible_object_ids": [3, 4]},
            {"image_name": "000030.jpg", "score": 17, "n_visible": 2, "visible_object_ids": [4, 3]},
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
                "clarity_score": 74,
                "frame_usable": True,
                "reason": "clear but no pair",
            },
            {
                "clear": True,
                "clarity_score": 73,
                "frame_usable": True,
                "reason": "still no pair",
            },
        ]

        root = Path(__file__).resolve().parent / "_tmp" / f"attachment_group_skip_{uuid.uuid4().hex}"
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
                return {"attachment_referable_object_ids": [3]}

            selected = referability_module._select_attachment_group_representatives(
                client=object(),
                model_name="fake-vlm",
                scene_dir=scene_dir,
                frames=frames,
                attachment_graph={1: [2], 3: [4]},
                attachment_entry_builder=build_entry,
            )

        self.assertEqual(frame_decision_mock.call_count, 4)
        self.assertEqual(build_calls, ["000020.jpg", "000030.jpg"])
        self.assertEqual(selected, [])

    def test_select_attachment_group_representatives_continues_later_groups_until_target_met(self) -> None:
        frames = [
            {"image_name": "000000.jpg", "score": 20, "n_visible": 2, "visible_object_ids": [1, 2]},
            {"image_name": "000010.jpg", "score": 19, "n_visible": 2, "visible_object_ids": [3, 4]},
            {"image_name": "000020.jpg", "score": 18, "n_visible": 2, "visible_object_ids": [5, 6]},
            {"image_name": "000030.jpg", "score": 17, "n_visible": 2, "visible_object_ids": [7, 8]},
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
                "clarity_score": 75,
                "frame_usable": True,
                "reason": "clear",
            },
            {
                "clear": True,
                "clarity_score": 76,
                "frame_usable": True,
                "reason": "clear",
            },
        ]

        root = Path(__file__).resolve().parent / "_tmp" / f"attachment_group_target_{uuid.uuid4().hex}"
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
                if frame["image_name"] == "000000.jpg":
                    return {"attachment_referable_object_ids": [1]}
                if frame["image_name"] == "000010.jpg":
                    return {"attachment_referable_object_ids": [3, 4]}
                if frame["image_name"] == "000020.jpg":
                    return {"attachment_referable_object_ids": [5, 6]}
                raise AssertionError(f"unexpected frame {frame['image_name']}")

            selected = referability_module._select_attachment_group_representatives(
                client=object(),
                model_name="fake-vlm",
                scene_dir=scene_dir,
                frames=frames,
                attachment_graph={1: [2], 3: [4], 5: [6], 7: [8]},
                attachment_entry_builder=build_entry,
                max_accepted_frame_count=2,
            )

        self.assertEqual(frame_decision_mock.call_count, 3)
        self.assertEqual(build_calls, ["000000.jpg", "000010.jpg", "000020.jpg"])
        self.assertEqual([entry["image_name"] for entry in selected], ["000010.jpg", "000020.jpg"])
        self.assertEqual([entry["attachment_view_group_id"] for entry in selected], [1, 2])

    def test_build_attachment_pair_salvage_scene_review_uses_single_cover_image_when_one_image_covers_pair(self) -> None:
        objects = [make_object(1, "table"), make_object(2, "book")]
        frames = [
            {
                "image_name": "000001.jpg",
                "visible_object_ids": [1, 2],
                "score": 10,
                "attachment_viewpoint_exempt": True,
            }
        ]
        entry = make_attachment_pair_salvage_entry(
            candidate_visible_object_ids=[1, 2],
            crop_label_statuses={"table": "unique", "book": "unique"},
            full_frame_label_statuses={"table": "unique", "book": "unique"},
            label_statuses={"table": "unique", "book": "unique"},
            attachment_referable_pairs=[[1, 2]],
        )
        root = Path(__file__).resolve().parent / "_tmp" / f"salvage_single_cover_{uuid.uuid4().hex}"
        scene_dir = make_scene_dir(root, "scene0001_00")
        self.addCleanup(shutil.rmtree, root, True)

        with (
            patch.object(
                referability_module,
                "_review_frame_clarity",
                return_value={
                    "image_name": "000001.jpg",
                    "frame_info": {"clear": True, "clarity_score": 84, "frame_usable": True, "reason": "clear"},
                },
            ),
            patch.object(
                referability_module.cv2,
                "imread",
                return_value=np.zeros((120, 120, 3), dtype=np.uint8),
            ),
        ):
            scene_review = referability_module._build_attachment_pair_salvage_scene_review(
                client=object(),
                model_name="fake-vlm",
                scene_id="scene0001_00",
                split="train",
                scene_dir=scene_dir,
                objects=objects,
                objects_by_id={1: objects[0], 2: objects[1]},
                attachment_graph={1: [2]},
                attachment_edges=[{"parent_id": 1, "child_id": 2, "type": "supported_by"}],
                frames=frames,
                poses={"000001.jpg": make_camera_pose(image_name="000001.jpg")},
                attachment_entry_builder=lambda frame, reviewed_frame: dict(entry),
                bbox_hard_fail_min=0.15,
                projected_area_hard_fail_min=800.0,
            )

        group = scene_review["groups"][0]
        self.assertEqual(group["selected_cover_image_names"], ["000001.jpg"])
        self.assertEqual(group["clarity_pass_image_names"], ["000001.jpg"])
        self.assertEqual(scene_review["pair_count_kept"], 1)

    def test_build_attachment_pair_salvage_scene_review_merges_attachment_groups_and_uses_visible_id_union(self) -> None:
        objects = [make_object(1, "table"), make_object(2, "book"), make_object(9, "lamp")]
        frames = [
            {"image_name": "000001.jpg", "visible_object_ids": [1, 2], "score": 10, "attachment_viewpoint_exempt": True},
            {"image_name": "000002.jpg", "visible_object_ids": [1, 2, 9], "score": 9, "attachment_viewpoint_exempt": True},
        ]
        entries = {
            "000001.jpg": make_attachment_pair_salvage_entry(
                candidate_visible_object_ids=[1, 2],
                crop_label_statuses={"table": "unique", "book": "unique"},
                full_frame_label_statuses={"table": "unique", "book": "unique"},
                label_statuses={"table": "unique", "book": "unique"},
                attachment_referable_pairs=[[1, 2]],
            ),
            "000002.jpg": make_attachment_pair_salvage_entry(
                candidate_visible_object_ids=[1, 2, 9],
                crop_label_statuses={"table": "unique", "book": "unique", "lamp": "unique"},
                full_frame_label_statuses={"table": "unique", "book": "unique", "lamp": "unique"},
                label_statuses={"table": "unique", "book": "unique", "lamp": "unique"},
                attachment_referable_pairs=[[1, 2]],
            ),
        }
        root = Path(__file__).resolve().parent / "_tmp" / f"salvage_merged_groups_{uuid.uuid4().hex}"
        scene_dir = make_scene_dir(root, "scene0001_00")
        self.addCleanup(shutil.rmtree, root, True)

        with (
            patch.object(
                referability_module,
                "_review_frame_clarity",
                side_effect=[
                    {"image_name": "000001.jpg", "frame_info": {"clear": True, "clarity_score": 84, "frame_usable": True, "reason": "clear"}},
                    {"image_name": "000002.jpg", "frame_info": {"clear": True, "clarity_score": 83, "frame_usable": True, "reason": "clear"}},
                ],
            ),
            patch.object(
                referability_module.cv2,
                "imread",
                return_value=np.zeros((120, 120, 3), dtype=np.uint8),
            ),
        ):
            scene_review = referability_module._build_attachment_pair_salvage_scene_review(
                client=object(),
                model_name="fake-vlm",
                scene_id="scene0001_00",
                split="train",
                scene_dir=scene_dir,
                objects=objects,
                objects_by_id={obj["id"]: obj for obj in objects},
                attachment_graph={1: [2]},
                attachment_edges=[{"parent_id": 1, "child_id": 2, "type": "supported_by"}],
                frames=frames,
                poses={
                    "000001.jpg": make_camera_pose(image_name="000001.jpg", yaw_deg=0.0),
                    "000002.jpg": make_camera_pose(image_name="000002.jpg", yaw_deg=10.0),
                },
                attachment_entry_builder=lambda frame, reviewed_frame: dict(entries[frame["image_name"]]),
                bbox_hard_fail_min=0.15,
                projected_area_hard_fail_min=800.0,
            )

        self.assertEqual(scene_review["group_count_total"], 1)
        self.assertEqual(scene_review["pair_count_total"], 1)
        group = scene_review["groups"][0]
        self.assertEqual(group["visible_object_ids"], [1, 2, 9])
        self.assertEqual(group["group_frame_image_names"], ["000001.jpg", "000002.jpg"])
        self.assertEqual(group["pair_count_total"], 1)
        self.assertEqual([pair["pair_id"] for pair in group["pairs"]], ["1->2"])

    def test_build_attachment_pair_salvage_scene_review_uses_multi_image_cover_when_needed(self) -> None:
        objects = [make_object(1, "table"), make_object(2, "book"), make_object(3, "lamp")]
        frames = [
            {"image_name": "000001.jpg", "visible_object_ids": [1, 2, 3], "score": 10, "attachment_viewpoint_exempt": True},
            {"image_name": "000002.jpg", "visible_object_ids": [1, 2, 3], "score": 9, "attachment_viewpoint_exempt": True},
        ]
        entries = {
            "000001.jpg": make_attachment_pair_salvage_entry(
                candidate_visible_object_ids=[1, 2],
                crop_label_statuses={"table": "unique", "book": "multiple"},
                label_statuses={"table": "unique", "book": "multiple"},
            ),
            "000002.jpg": make_attachment_pair_salvage_entry(
                candidate_visible_object_ids=[1, 3],
                crop_label_statuses={"table": "unique", "lamp": "multiple"},
                label_statuses={"table": "unique", "lamp": "multiple"},
            ),
        }
        root = Path(__file__).resolve().parent / "_tmp" / f"salvage_multi_cover_{uuid.uuid4().hex}"
        scene_dir = make_scene_dir(root, "scene0001_00")
        self.addCleanup(shutil.rmtree, root, True)

        with (
            patch.object(
                referability_module,
                "_review_frame_clarity",
                side_effect=[
                    {"image_name": "000001.jpg", "frame_info": {"clear": True, "clarity_score": 81, "frame_usable": True, "reason": "clear"}},
                    {"image_name": "000002.jpg", "frame_info": {"clear": True, "clarity_score": 80, "frame_usable": True, "reason": "clear"}},
                ],
            ),
            patch.object(
                referability_module.cv2,
                "imread",
                return_value=np.zeros((120, 120, 3), dtype=np.uint8),
            ),
        ):
            scene_review = referability_module._build_attachment_pair_salvage_scene_review(
                client=object(),
                model_name="fake-vlm",
                scene_id="scene0001_00",
                split="train",
                scene_dir=scene_dir,
                objects=objects,
                objects_by_id={obj["id"]: obj for obj in objects},
                attachment_graph={1: [2, 3]},
                attachment_edges=[
                    {"parent_id": 1, "child_id": 2, "type": "supported_by"},
                    {"parent_id": 1, "child_id": 3, "type": "next_to"},
                ],
                frames=frames,
                poses={
                    "000001.jpg": make_camera_pose(image_name="000001.jpg"),
                    "000002.jpg": make_camera_pose(image_name="000002.jpg"),
                },
                attachment_entry_builder=lambda frame, reviewed_frame: dict(entries[frame["image_name"]]),
                bbox_hard_fail_min=0.15,
                projected_area_hard_fail_min=800.0,
            )

        group = scene_review["groups"][0]
        self.assertEqual(group["selected_cover_image_names"], ["000001.jpg", "000002.jpg"])
        self.assertEqual(scene_review["group_count_with_multi_image_cover"], 1)

    def test_attachment_pair_salvage_pair_row_does_not_mark_hard_fail_when_any_clarity_image_covers_pair(self) -> None:
        objects = {1: make_object(1, "table"), 2: make_object(2, "book")}
        clarity_pass_frames = [
            {
                "image_name": "000001.jpg",
                "entry": make_attachment_pair_salvage_entry(
                    candidate_visible_object_ids=[1],
                    object_reviews={1: {"bbox_in_frame_ratio": 0.9}},
                    crop_label_statuses={"table": "unique"},
                    label_statuses={"table": "unique"},
                ),
            },
            {
                "image_name": "000002.jpg",
                "entry": make_attachment_pair_salvage_entry(
                    candidate_visible_object_ids=[1, 2],
                    crop_label_statuses={"table": "unique", "book": "multiple"},
                    label_statuses={"table": "unique", "book": "multiple"},
                ),
            },
        ]

        pair_row = referability_module._build_attachment_pair_salvage_pair_row(
            parent_id=1,
            child_id=2,
            relation_types=["supported_by"],
            clarity_pass_frames=clarity_pass_frames,
            objects_by_id=objects,
            bbox_hard_fail_min=0.15,
            projected_area_hard_fail_min=800.0,
        )

        self.assertNotEqual(
            pair_row["program_decision"],
            referability_module.ATTACHMENT_PAIR_PROGRAM_DECISION_AUTO_DROP_HARD_FAIL,
        )
        self.assertEqual(
            pair_row["program_decision"],
            referability_module.ATTACHMENT_PAIR_PROGRAM_DECISION_NEEDS_VLM_SALVAGE_REVIEW,
        )

    def test_attachment_pair_salvage_pair_row_marks_hard_fail_when_all_clarity_images_fail_object_gate(self) -> None:
        objects = {1: make_object(1, "table"), 2: make_object(2, "book")}
        clarity_pass_frames = [
            {
                "image_name": "000001.jpg",
                "entry": make_attachment_pair_salvage_entry(
                    candidate_visible_object_ids=[1, 2],
                    object_reviews={2: {"vlm_status": "absent"}},
                    crop_label_statuses={"table": "unique", "book": "absent"},
                    label_statuses={"table": "unique", "book": "absent"},
                ),
            },
            {
                "image_name": "000002.jpg",
                "entry": make_attachment_pair_salvage_entry(
                    candidate_visible_object_ids=[1, 2],
                    object_reviews={2: {"bbox_in_frame_ratio": 0.10}},
                    crop_label_statuses={"table": "unique", "book": "unique"},
                    label_statuses={"table": "unique", "book": "unique"},
                ),
            },
        ]

        pair_row = referability_module._build_attachment_pair_salvage_pair_row(
            parent_id=1,
            child_id=2,
            relation_types=["supported_by"],
            clarity_pass_frames=clarity_pass_frames,
            objects_by_id=objects,
            bbox_hard_fail_min=0.15,
            projected_area_hard_fail_min=800.0,
        )

        self.assertEqual(
            pair_row["program_decision"],
            referability_module.ATTACHMENT_PAIR_PROGRAM_DECISION_AUTO_DROP_HARD_FAIL,
        )

    def test_attachment_pair_salvage_pair_row_marks_crop_unique_full_frame_multiple_as_salvage_review(self) -> None:
        objects = {1: make_object(1, "table"), 2: make_object(2, "book")}
        clarity_pass_frames = [
            {
                "image_name": "000001.jpg",
                "entry": make_attachment_pair_salvage_entry(
                    candidate_visible_object_ids=[1, 2],
                    crop_label_statuses={"table": "unique", "book": "unique"},
                    full_frame_label_statuses={"table": "multiple", "book": "unique"},
                    label_statuses={"table": "multiple", "book": "unique"},
                ),
            },
        ]

        pair_row = referability_module._build_attachment_pair_salvage_pair_row(
            parent_id=1,
            child_id=2,
            relation_types=["supported_by"],
            clarity_pass_frames=clarity_pass_frames,
            objects_by_id=objects,
            bbox_hard_fail_min=0.15,
            projected_area_hard_fail_min=800.0,
        )

        self.assertEqual(
            pair_row["program_decision"],
            referability_module.ATTACHMENT_PAIR_PROGRAM_DECISION_NEEDS_VLM_SALVAGE_REVIEW,
        )

    def test_attachment_pair_salvage_group_vlm_review_normalizes_multiple_pair_reviews(self) -> None:
        parsed = {
            "group_id": "scene0001_00:group_0",
            "pair_reviews": [
                {
                    "pair_id": "1->2",
                    "decision": "salvageable",
                    "parent_visibility": "visible",
                    "child_visibility": "partial",
                    "pair_unique_with_modifiers": "yes",
                    "parent_modifier_candidates": ["round"],
                    "child_modifier_candidates": ["wooden"],
                    "pair_reference_phrase_candidates": ["the round item on the wooden table"],
                    "reason": "distinct modifiers",
                },
                {
                    "pair_id": "1->3",
                    "decision": "not_salvageable",
                    "parent_visibility": "visible",
                    "child_visibility": "visible",
                    "pair_unique_with_modifiers": "no",
                    "parent_modifier_candidates": [],
                    "child_modifier_candidates": [],
                    "pair_reference_phrase_candidates": [],
                    "reason": "still ambiguous",
                },
            ],
        }
        with patch.object(
            referability_module,
            "_call_vlm_json",
            return_value=(parsed, '{"ok":true}'),
        ):
            review = referability_module._attachment_pair_salvage_group_vlm_review(
                client=object(),
                model_name="fake-vlm",
                group_id="scene0001_00:group_0",
                cover_images=[{"image_name": "000001.jpg", "data_url": "data:image/jpeg;base64,ZmFrZQ=="}],
                pair_rows=[
                    {
                        "pair_id": "1->2",
                        "parent_id": 1,
                        "parent_label": "table",
                        "child_id": 2,
                        "child_label": "book",
                        "first_covered_image_name": "000001.jpg",
                        "parent_crop_image_data_url": "data:image/jpeg;base64,ZmFrZQ==",
                        "child_crop_image_data_url": "data:image/jpeg;base64,ZmFrZQ==",
                    },
                    {
                        "pair_id": "1->3",
                        "parent_id": 1,
                        "parent_label": "table",
                        "child_id": 3,
                        "child_label": "lamp",
                        "first_covered_image_name": "000001.jpg",
                        "parent_crop_image_data_url": "data:image/jpeg;base64,ZmFrZQ==",
                        "child_crop_image_data_url": "data:image/jpeg;base64,ZmFrZQ==",
                    },
                ],
            )

        self.assertEqual(review["group_id"], "scene0001_00:group_0")
        self.assertEqual([item["pair_id"] for item in review["pair_reviews"]], ["1->2", "1->3"])
        self.assertEqual(review["pair_reviews"][0]["decision"], "salvageable")
        self.assertEqual(review["pair_reviews"][1]["decision"], "not_salvageable")

    def test_render_attachment_pair_salvage_review_html_lists_included_scenes_in_deduped_order(self) -> None:
        review_doc = {
            "scenes": [
                {
                    "scene_id": "scene0002_00",
                    "pipeline_outcome": "processed",
                    "groups": [
                        {
                            "group_id": "scene0002_00:group_0",
                            "selected_cover_images": [
                                {
                                    "image_name": "000201.jpg",
                                    "image_stem": "000201",
                                    "data_url": "data:image/jpeg;base64,c2NlbmUy",
                                }
                            ],
                            "dropped_pairs": [
                                {
                                    "pair_id": "2->3",
                                    "parent_id": 2,
                                    "parent_label": "chair",
                                    "child_id": 3,
                                    "child_label": "bag",
                                    "first_covered_image_name": "000201.jpg",
                                    "program_reason_codes": ["coverage_uncertain"],
                                }
                            ],
                        }
                    ],
                },
                {
                    "scene_id": "scene0001_00",
                    "pipeline_outcome": "processed",
                    "groups": [
                        {
                            "group_id": "scene0001_00:group_0",
                            "selected_cover_images": [
                                {
                                    "image_name": "000101.jpg",
                                    "image_stem": "000101",
                                    "data_url": "data:image/jpeg;base64,c2NlbmUx",
                                }
                            ],
                            "dropped_pairs": [
                                {
                                    "pair_id": "1->2",
                                    "parent_id": 1,
                                    "parent_label": "table",
                                    "child_id": 2,
                                    "child_label": "book",
                                    "first_covered_image_name": "000101.jpg",
                                    "program_reason_codes": ["status_conflict"],
                                }
                            ],
                        }
                    ],
                },
                {
                    "scene_id": "scene0002_00",
                    "pipeline_outcome": "processed",
                    "groups": [
                        {
                            "group_id": "scene0002_00:group_1",
                            "selected_cover_images": [],
                            "dropped_pairs": [
                                {
                                    "pair_id": "9->10",
                                    "parent_id": 9,
                                    "parent_label": "desk",
                                    "child_id": 10,
                                    "child_label": "monitor",
                                    "first_covered_image_name": "000299.jpg",
                                    "program_reason_codes": ["coverage_uncertain"],
                                }
                            ],
                        }
                    ],
                },
            ],
        }

        html_text = referability_module._render_attachment_pair_salvage_review_html(review_doc)

        self.assertIn(
            "included scenes:</strong> scene0002_00, scene0001_00",
            html_text,
        )
        self.assertIn("scene count:</strong> 2", html_text)
        self.assertIn("group count:</strong> 2", html_text)
        self.assertIn("pair count:</strong> 2", html_text)

    def test_render_attachment_pair_salvage_review_html_filters_unrenderable_pairs_and_renders_chinese_reasons(
        self,
    ) -> None:
        review_doc = {
            "scenes": [
                {
                    "scene_id": "scene0001_00",
                    "pipeline_outcome": "processed",
                    "groups": [
                        {
                            "group_id": "scene0001_00:group_keep",
                            "selected_cover_images": [
                                {
                                    "image_name": "000001.jpg",
                                    "image_stem": "000001",
                                    "data_url": "data:image/jpeg;base64,cover_kept",
                                }
                            ],
                            "dropped_pairs": [
                                {
                                    "pair_id": "1->2",
                                    "parent_id": 1,
                                    "parent_label": "table",
                                    "child_id": 2,
                                    "child_label": "book",
                                    "first_covered_image_name": "000001.jpg",
                                    "program_reason_codes": [
                                        "no_coverable_clarity_pass_image",
                                        "child_final_multiple",
                                        "mystery_reason_code",
                                    ],
                                    "parent_crop_image_data_url": "data:image/jpeg;base64,parent_crop_should_not_render",
                                    "child_crop_image_data_url": "data:image/jpeg;base64,child_crop_should_not_render",
                                },
                                {
                                    "pair_id": "1->3",
                                    "parent_id": 1,
                                    "parent_label": "table",
                                    "child_id": 3,
                                    "child_label": "lamp",
                                    "first_covered_image_name": "000099.jpg",
                                    "program_reason_codes": ["missing_referability_entry"],
                                },
                            ],
                        },
                        {
                            "group_id": "scene0001_00:group_drop",
                            "selected_cover_images": [],
                            "dropped_pairs": [
                                {
                                    "pair_id": "4->5",
                                    "parent_id": 4,
                                    "parent_label": "sofa",
                                    "child_id": 5,
                                    "child_label": "pillow",
                                    "first_covered_image_name": "000004.jpg",
                                    "program_reason_codes": ["coverage_uncertain"],
                                }
                            ],
                        },
                    ],
                },
                {
                    "scene_id": "scene0002_00",
                    "pipeline_outcome": "processed",
                    "groups": [
                        {
                            "group_id": "scene0002_00:group_drop",
                            "selected_cover_images": [],
                            "dropped_pairs": [
                                {
                                    "pair_id": "7->8",
                                    "parent_id": 7,
                                    "parent_label": "cabinet",
                                    "child_id": 8,
                                    "child_label": "cup",
                                    "first_covered_image_name": "000007.jpg",
                                    "program_reason_codes": ["status_conflict"],
                                }
                            ],
                        }
                    ],
                },
            ],
        }

        html_text = referability_module._render_attachment_pair_salvage_review_html(review_doc)

        self.assertIn("scene0001_00", html_text)
        self.assertNotIn("scene0002_00", html_text)
        self.assertIn("scene0001_00:group_keep", html_text)
        self.assertNotIn("scene0001_00:group_drop", html_text)
        self.assertIn("data:image/jpeg;base64,cover_kept", html_text)
        self.assertNotIn("parent_crop_should_not_render", html_text)
        self.assertNotIn("child_crop_should_not_render", html_text)
        self.assertIn("pair id</strong> 1-&gt;2", html_text)
        self.assertNotIn("1-&gt;3", html_text)
        self.assertIn(
            "筛除理由</strong> 没有可覆盖该 attachment pair 的清晰图像，子物体最终判定中存在多个同类目标，mystery_reason_code",
            html_text,
        )

    def test_render_attachment_pair_salvage_review_html_keeps_same_pair_as_separate_cards(
        self,
    ) -> None:
        review_doc = {
            "scenes": [
                {
                    "scene_id": "scene0003_00",
                    "pipeline_outcome": "processed",
                    "groups": [
                        {
                            "group_id": "scene0003_00:group_0",
                            "selected_cover_images": [
                                {
                                    "image_name": "000301.jpg",
                                    "image_stem": "000301",
                                    "data_url": "data:image/jpeg;base64,cover_a",
                                }
                            ],
                            "dropped_pairs": [
                                {
                                    "pair_id": "10->11",
                                    "parent_id": 10,
                                    "parent_label": "table",
                                    "child_id": 11,
                                    "child_label": "book",
                                    "first_covered_image_name": "000301.jpg",
                                    "program_reason_codes": ["coverage_uncertain"],
                                }
                            ],
                        },
                        {
                            "group_id": "scene0003_00:group_1",
                            "selected_cover_images": [
                                {
                                    "image_name": "000302.jpg",
                                    "image_stem": "000302",
                                    "data_url": "data:image/jpeg;base64,cover_b",
                                }
                            ],
                            "dropped_pairs": [
                                {
                                    "pair_id": "10->11",
                                    "parent_id": 10,
                                    "parent_label": "table",
                                    "child_id": 11,
                                    "child_label": "book",
                                    "first_covered_image_name": "000302.jpg",
                                    "program_reason_codes": ["status_conflict"],
                                }
                            ],
                        },
                    ],
                }
            ],
        }

        html_text = referability_module._render_attachment_pair_salvage_review_html(review_doc)

        self.assertNotIn('<details class="pair-bucket">', html_text)
        self.assertIn("pair count:</strong> 2", html_text)
        self.assertIn("group count:</strong> 2", html_text)
        self.assertEqual(html_text.count("pair id</strong> 10-&gt;11"), 2)
        self.assertIn("group</strong> scene0003_00:group_0", html_text)
        self.assertIn("group</strong> scene0003_00:group_1", html_text)
        self.assertIn('id="export-edited-html"', html_text)
        self.assertIn("edited html target:</strong> edited.html", html_text)
        self.assertIn("suggestedName: editedHtmlTargetName", html_text)
        self.assertIn('name="parent_surface_text"', html_text)
        self.assertIn('name="child_surface_text"', html_text)
        self.assertIn('class="pair-delete-toggle"', html_text)

    def test_parse_attachment_pair_salvage_review_html_reads_edited_inputs_and_deleted_state(self) -> None:
        html_text = """<!doctype html>
<html lang="en">
<body>
  <article class="pair-card" data-scene-id="scene0001_00" data-image-name="000001.jpg" data-group-id="scene0001_00:group_0" data-pair-id="1-&gt;2" data-parent-id="1" data-parent-label="table" data-child-id="2" data-child-label="book" data-deleted="false">
    <input type="text" name="parent_surface_text" value="wooden table">
    <input type="text" name="child_surface_text" value="blue book">
  </article>
  <article class="pair-card" data-scene-id="scene0001_00" data-image-name="000002.jpg" data-group-id="scene0001_00:group_1" data-pair-id="3-&gt;4" data-parent-id="3" data-parent-label="desk" data-child-id="4" data-child-label="lamp" data-deleted="true">
    <input type="text" name="parent_surface_text" value="desk">
    <input type="text" name="child_surface_text" value="lamp">
  </article>
</body>
</html>"""

        cards = referability_module._parse_attachment_pair_salvage_review_html(html_text)

        self.assertEqual(len(cards), 2)
        self.assertEqual(cards[0]["scene_id"], "scene0001_00")
        self.assertEqual(cards[0]["image_name"], "000001.jpg")
        self.assertEqual(cards[0]["parent_surface_text"], "wooden table")
        self.assertEqual(cards[0]["child_surface_text"], "blue book")
        self.assertFalse(cards[0]["deleted"])
        self.assertTrue(cards[1]["deleted"])

    def test_apply_attachment_pair_salvage_html_review_updates_cache_with_kept_cards_only(self) -> None:
        cache_doc = {
            "version": referability_module.REFERABILITY_CACHE_VERSION,
            "frames": {
                "scene0001_00": {
                    "000001.jpg": {
                        **make_debug_cache_entry(),
                        "attachment_referable_pairs": [],
                        "attachment_referable_object_ids": [],
                    },
                    "000002.jpg": {
                        **make_debug_cache_entry(),
                        "attachment_referable_pairs": [],
                        "attachment_referable_object_ids": [],
                    },
                }
            },
        }
        html_text = """<!doctype html>
<html lang="en">
<body>
  <article class="pair-card" data-scene-id="scene0001_00" data-image-name="000001.jpg" data-group-id="scene0001_00:group_0" data-pair-id="1-&gt;2" data-parent-id="1" data-parent-label="table" data-child-id="2" data-child-label="book" data-deleted="false">
    <input type="text" name="parent_surface_text" value="wooden table">
    <input type="text" name="child_surface_text" value="blue book">
  </article>
  <article class="pair-card" data-scene-id="scene0001_00" data-image-name="000002.jpg" data-group-id="scene0001_00:group_1" data-pair-id="3-&gt;4" data-parent-id="3" data-parent-label="desk" data-child-id="4" data-child-label="lamp" data-deleted="true">
    <input type="text" name="parent_surface_text" value="desk">
    <input type="text" name="child_surface_text" value="lamp">
  </article>
</body>
</html>"""

        updated = referability_module._apply_attachment_pair_salvage_html_review(
            html_text=html_text,
            cache_doc=cache_doc,
        )

        kept_entry = updated["frames"]["scene0001_00"]["000001.jpg"]
        dropped_entry = updated["frames"]["scene0001_00"]["000002.jpg"]
        self.assertEqual(kept_entry["attachment_referable_pairs"], [[1, 2]])
        self.assertEqual(kept_entry["attachment_referable_object_ids"], [1, 2])
        self.assertTrue(referability_module._frame_entry_has_consistent_final_fields(kept_entry))
        self.assertEqual(
            kept_entry["attachment_human_review_cards"],
            [
                {
                    "pair_id": "1->2",
                    "parent_id": 1,
                    "parent_label": "table",
                    "parent_surface_text": "wooden table",
                    "child_id": 2,
                    "child_label": "book",
                    "child_surface_text": "blue book",
                    "source": "human_salvage_html",
                }
            ],
        )
        self.assertEqual(dropped_entry["attachment_referable_pairs"], [])
        self.assertEqual(dropped_entry["attachment_referable_object_ids"], [])

    def test_main_persists_scene_grouping_summary_in_cache(self) -> None:
        root = Path(__file__).resolve().parent / "_tmp" / f"scene_grouping_summary_{uuid.uuid4().hex}"
        data_root = root / "data"
        scene_dir = data_root / "scene0001_00"
        (scene_dir / "pose").mkdir(parents=True, exist_ok=True)
        output_path = root / "output" / "referability_cache.json"
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

        _batch_path, cache_doc = load_single_batch_cache_for_output(output_path)
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
        global_scene_status = load_scene_status_doc_for_output(output_path)
        self.assertEqual(global_scene_status["split"], "train")
        self.assertEqual(
            global_scene_status["completed_scenes"]["scene0001_00"]["batch_file"],
            _batch_path.name,
        )

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

        _batch_path, cache_doc = load_single_batch_cache_for_output(output_path)
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
        global_scene_status = load_scene_status_doc_for_output(output_path)
        self.assertEqual(global_scene_status["completed_scenes"]["scene0001_00"]["batch_file"], _batch_path.name)

    def test_reset_scene_status_clears_global_state_but_keeps_existing_batches(self) -> None:
        root = Path(__file__).resolve().parent / "_tmp" / f"scene_status_reset_{uuid.uuid4().hex}"
        data_root = root / "data"
        scene_dir = data_root / "scene0001_00"
        (scene_dir / "pose").mkdir(parents=True, exist_ok=True)
        output_path = root / "output" / "referability_cache.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        first_batch_path = output_path.parent / "flash_legacy.json"
        first_batch_doc = {
            "version": referability_module.REFERABILITY_CACHE_VERSION,
            "model": "fake-vlm",
            "alias_config_version": referability_module.ALIAS_CONFIG_VERSION,
            "referability_backend": "crop_vlm_with_mesh_ray",
            "label_batch_size": 1,
            "frames": {
                "scene0001_00": {
                    "000001.jpg": make_debug_cache_entry(),
                }
            },
            "scene_grouping": {
                "scene0001_00": {
                    "scene_id": "scene0001_00",
                    "split": "train",
                    "pipeline_outcome": "processed",
                    "scene_skip_reason": None,
                    "final_cacheable_frame_count": 1,
                    "final_cacheable_frame_image_names": ["000001.jpg"],
                }
            },
            "scene_status": {
                "scene0001_00": {
                    "scene_id": "scene0001_00",
                    "processed": True,
                    "pipeline_outcome": "processed",
                    "split": "train",
                    "has_cache_frames": True,
                    "final_cacheable_frame_count": 1,
                    "scene_skip_reason": None,
                }
            },
        }
        first_batch_path.write_text(json.dumps(first_batch_doc, ensure_ascii=False), encoding="utf-8")
        scene_status_path_for_output(output_path).write_text(
            json.dumps(
                {
                    "version": referability_module.SCENE_STATUS_VERSION,
                    "split": "train",
                    "completed_scenes": {
                        "scene0001_00": {
                            "status": "completed",
                            "batch_file": first_batch_path.name,
                            "updated_at": "2026-04-30T12:00:00Z",
                        }
                    },
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        scene = {"objects": [make_object(1, "table"), make_object(2, "book")]}

        def fake_enrich(scene_dict: dict) -> dict:
            scene_dict["attachment_graph"] = {"1": [2]}
            scene_dict["attached_by"] = {"2": 1}
            scene_dict["attachment_edges"] = [{"parent_id": 1, "child_id": 2, "type": "supported_by"}]
            scene_dict["support_chain_graph"] = {"1": [2]}
            scene_dict["support_chain_by"] = {"2": 1}
            return scene_dict

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
                "_select_attachment_group_representatives",
                return_value=[attachment_entry],
            ),
            patch.object(sys, "argv", [
                "run_vlm_referability.py",
                "--data_root",
                str(data_root),
                "--output",
                str(output_path),
                "--reset_scene_status",
                "--scene_batch_size",
                "1",
                "--no-write_attachment_review",
            ]),
        ):
            referability_module.main()

        batch_paths = list_batch_cache_paths(output_path)
        self.assertEqual(len(batch_paths), 2)
        self.assertIn(first_batch_path, batch_paths)
        second_batch_path = next(path for path in batch_paths if path != first_batch_path)
        second_batch_doc = json.loads(second_batch_path.read_text(encoding="utf-8"))
        self.assertEqual(second_batch_doc["scene_status"]["scene0001_00"]["pipeline_outcome"], "processed")
        global_scene_status = load_scene_status_doc_for_output(output_path)
        self.assertEqual(global_scene_status["completed_scenes"]["scene0001_00"]["batch_file"], second_batch_path.name)

    def test_main_writes_attachment_review_json_for_scene_without_attachment_relations(self) -> None:
        root = Path(__file__).resolve().parent / "_tmp" / f"attachment_review_empty_{uuid.uuid4().hex}"
        data_root = root / "data"
        scene_dir = data_root / "scene0001_00"
        (scene_dir / "pose").mkdir(parents=True, exist_ok=True)
        output_path = root / "output" / "referability_cache.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        review_path = output_path.parent / "candidate" / "referability_cache_attachment_candidate_review.json"
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

        batch_path, _cache_doc = load_single_batch_cache_for_output(output_path)
        review_path = referability_module._attachment_review_output_path(batch_path)
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
        _batch_path, cache_doc = load_single_batch_cache_for_output(output_path)
        self.assertEqual(
            cache_doc["scene_status"]["scene0001_00"]["pipeline_outcome"],
            "no_attachment_relations",
        )

    def test_main_respects_attachment_review_output_override(self) -> None:
        root = Path(__file__).resolve().parent / "_tmp" / f"attachment_review_override_{uuid.uuid4().hex}"
        data_root = root / "data"
        scene_dir = data_root / "scene0001_00"
        (scene_dir / "pose").mkdir(parents=True, exist_ok=True)
        output_path = root / "output" / "referability_cache.json"
        custom_review_path = root / "custom" / "review.json"
        default_review_path = output_path.parent / "candidate" / "referability_cache_attachment_candidate_review.json"
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
            patch.object(sys, "argv", [
                "run_vlm_referability.py",
                "--data_root",
                str(data_root),
                "--output",
                str(output_path),
                "--attachment_review_output",
                str(custom_review_path),
                "--max_scenes",
                "1",
                "--max_frames",
                "5",
            ]),
        ):
            referability_module.main()

        batch_path, _cache_doc = load_single_batch_cache_for_output(output_path)
        default_review_path = referability_module._attachment_review_output_path(batch_path)
        self.assertTrue(custom_review_path.exists())
        self.assertFalse(default_review_path.exists())
        review_doc = json.loads(custom_review_path.read_text(encoding="utf-8"))
        self.assertEqual(review_doc["scene_count"], 1)
        _batch_path, cache_doc = load_single_batch_cache_for_output(output_path)
        self.assertEqual(
            cache_doc["scene_status"]["scene0001_00"]["pipeline_outcome"],
            "no_attachment_relations",
        )

    def test_main_writes_attachment_pair_salvage_review_json_and_html_by_default(self) -> None:
        root = Path(__file__).resolve().parent / "_tmp" / f"attachment_pair_salvage_main_{uuid.uuid4().hex}"
        data_root = root / "data"
        scene_dir = data_root / "scene0001_00"
        (scene_dir / "pose").mkdir(parents=True, exist_ok=True)
        output_path = root / "output" / "referability_cache.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        edited_html_path = output_path.parent / "edited.html"

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

        attachment_frame = make_debug_cache_entry()
        attachment_frame["image_name"] = "000001.jpg"
        attachment_frame["attachment_referable_object_ids"] = [1, 2]
        attachment_frame["attachment_view_group_id"] = 0
        salvage_pair = {
            "pair_id": "1->2",
            "parent_id": 1,
            "parent_label": "table",
            "child_id": 2,
            "child_label": "book",
            "relation_types": ["supported_by"],
            "program_decision": "needs_vlm_salvage_review",
            "program_status": "salvage_review",
            "program_reason_codes": ["child_final_multiple"],
            "current_attachment_referable": False,
            "cover_image_names": ["000001.jpg"],
            "kept_image_names": [],
            "first_covered_image_name": "000001.jpg",
            "coverage_by_image_name": [{"image_name": "000001.jpg", "covered": True, "uncertain": False, "reason_codes": []}],
            "vlm_review": {
                "pair_id": "1->2",
                "decision": "salvageable",
                "parent_visibility": "visible",
                "child_visibility": "visible",
                "pair_unique_with_modifiers": "yes",
                "parent_modifier_candidates": ["round"],
                "child_modifier_candidates": ["blue"],
                "pair_reference_phrase_candidates": ["the round table with the blue book"],
                "reason": "clear modifiers",
            },
            "human_decision": None,
            "human_notes": "",
        }
        salvage_scene_review = {
            "scene_id": "scene0001_00",
            "split": "train",
            "pipeline_outcome": None,
            "object_count": 2,
            "group_count_total": 1,
            "group_count_with_clarity_pass_images": 1,
            "group_count_with_multi_image_cover": 0,
            "pair_count_total": 1,
            "pair_count_kept": 0,
            "pair_count_auto_drop_hard_fail": 0,
            "pair_count_needs_vlm_salvage_review": 1,
            "pair_count_uncertain": 0,
            "pair_count_vlm_salvageable": 1,
            "pair_count_vlm_not_salvageable": 0,
            "pair_count_vlm_uncertain": 0,
            "terminal_output_lines": ["[attachment-pair-salvage] scene=scene0001_00 group=0 pairs=1"],
            "groups": [
                {
                    "group_id": "scene0001_00:group_0",
                    "group_index": 0,
                    "visible_object_ids": [1, 2],
                    "visible_object_labels": ["table#1", "book#2"],
                    "group_frame_image_names": ["000001.jpg", "000002.jpg"],
                    "sampled_frame_image_names": ["000001.jpg"],
                    "clarity_pass_image_names": ["000001.jpg"],
                    "selected_cover_image_names": ["000001.jpg"],
                    "selected_cover_images": [
                        {
                            "image_name": "000001.jpg",
                            "image_stem": "000001",
                            "covered_pair_ids": ["1->2"],
                            "data_url": "data:image/jpeg;base64,cover_for_html",
                        }
                    ],
                    "group_frame_stride": 1,
                    "pair_count_total": 1,
                    "kept_pair_ids": [],
                    "dropped_pair_ids": ["1->2"],
                    "pairs": [salvage_pair],
                    "dropped_pairs": [salvage_pair],
                    "vlm_group_review": None,
                }
            ],
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
            patch.object(referability_module, "load_scannet_poses", return_value={"000001.jpg": make_camera_pose()}),
            patch.object(referability_module, "load_scannet_intrinsics", return_value=make_camera_intrinsics()),
            patch.object(referability_module, "load_scannet_depth_intrinsics", return_value=None),
            patch.object(
                referability_module,
                "_select_attachment_group_representatives",
                return_value=[attachment_frame],
            ),
            patch.object(
                referability_module,
                "_build_attachment_pair_salvage_scene_review",
                return_value=salvage_scene_review,
            ),
            patch.object(sys, "argv", [
                "run_vlm_referability.py",
                "--data_root",
                str(data_root),
                "--output",
                str(output_path),
                "--max_scenes",
                "1",
                "--no-write_attachment_review",
            ]),
        ):
            referability_module.main()

        batch_path, _cache_doc = load_single_batch_cache_for_output(output_path)
        review_path = referability_module._attachment_pair_salvage_review_output_path(batch_path)
        review_html_path = referability_module._attachment_pair_salvage_review_html_output_path(batch_path)
        self.assertTrue(review_path.exists())
        self.assertTrue(review_html_path.exists())
        self.assertTrue(edited_html_path.exists())
        review_doc = json.loads(review_path.read_text(encoding="utf-8"))
        html_text = review_html_path.read_text(encoding="utf-8")
        self.assertEqual(review_doc["name"], referability_module.ATTACHMENT_PAIR_SALVAGE_REVIEW_NAME)
        self.assertEqual(review_doc["edited_html_output"], str(edited_html_path))
        self.assertEqual(review_doc["pair_count_needs_vlm_salvage_review"], 1)
        self.assertIn("Attachment Pair Salvage Review", html_text)
        self.assertIn(f"edited html target:</strong> {edited_html_path}", html_text)
        self.assertIn("included scenes:</strong> scene0001_00", html_text)
        self.assertIn("scene0001_00:group_0", html_text)
        self.assertIn("000001", html_text)
        self.assertIn("1-&gt;2", html_text)
        self.assertEqual(edited_html_path.read_text(encoding="utf-8"), html_text)
        self.assertTrue(batch_path.exists())

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

        _batch_path, cache_doc = load_single_batch_cache_for_output(output_path)
        scene_status = cache_doc["scene_status"]["scene0001_00"]
        self.assertEqual(scene_status["pipeline_outcome"], "no_frame_candidates")
        self.assertEqual(scene_status["scene_skip_reason"], "no_frame_candidates")
        self.assertFalse(scene_status["has_cache_frames"])
        self.assertEqual(scene_status["final_cacheable_frame_count"], 0)
        global_scene_status = load_scene_status_doc_for_output(output_path)
        self.assertEqual(global_scene_status["completed_scenes"]["scene0001_00"]["batch_file"], _batch_path.name)

    def test_main_marks_scene_status_split_as_val(self) -> None:
        root = Path(__file__).resolve().parent / "_tmp" / f"scene_status_val_{uuid.uuid4().hex}"
        data_root = root / "data"
        scene_dir = data_root / "scans" / "scene1001_00"
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
                "val",
                "--output",
                str(output_path),
                "--max_scenes",
                "1",
                "--no-write_attachment_review",
            ]),
        ):
            referability_module.main()

        _batch_path, cache_doc = load_single_batch_cache_for_output(output_path)
        scene_status = cache_doc["scene_status"]["scene1001_00"]
        self.assertEqual(scene_status["split"], "val")
        self.assertEqual(scene_status["pipeline_outcome"], "no_frame_candidates")
        global_scene_status = load_scene_status_doc_for_output(output_path)
        self.assertEqual(global_scene_status["split"], "val")
        self.assertEqual(global_scene_status["completed_scenes"]["scene1001_00"]["batch_file"], _batch_path.name)

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

        _first_batch_path, first_cache_doc = load_single_batch_cache_for_output(output_path)
        self.assertEqual(
            first_cache_doc["scene_status"]["scene0001_00"]["pipeline_outcome"],
            "no_final_referability_frames",
        )
        self.assertEqual(
            load_scene_status_doc_for_output(output_path)["completed_scenes"]["scene0001_00"]["batch_file"],
            _first_batch_path.name,
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

        self.assertEqual(len(list_batch_cache_paths(output_path)), 1)

    def test_scene_batch_size_uses_scene_status_to_skip_completed_scenes(self) -> None:
        root = Path(__file__).resolve().parent / "_tmp" / f"scene_batch_resume_{uuid.uuid4().hex}"
        data_root = root / "data"
        scans_root = data_root / "scans"
        for scene_id in ("scene0001_00", "scene0002_00", "scene0003_00"):
            make_scene_dir(scans_root, scene_id)
        output_path = root / "output" / "referability_cache.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        legacy_batch_a = output_path.parent / "flash_batch_a.json"
        legacy_batch_b = output_path.parent / "flash_batch_b.json"
        legacy_batch_a.write_text(
            json.dumps(
                {
                    "version": referability_module.REFERABILITY_CACHE_VERSION,
                    "model": "fake-vlm",
                    "alias_config_version": referability_module.ALIAS_CONFIG_VERSION,
                    "referability_backend": "crop_vlm_with_mesh_ray",
                    "label_batch_size": 1,
                    "frames": {
                        "scene0001_00": {
                            "000001.jpg": make_debug_cache_entry(),
                        }
                    },
                    "scene_grouping": {
                        "scene0001_00": {
                            "scene_id": "scene0001_00",
                            "split": "train",
                            "pipeline_outcome": "processed",
                            "scene_skip_reason": None,
                            "final_cacheable_frame_count": 1,
                        }
                    },
                    "scene_status": {
                        "scene0001_00": {
                            "scene_id": "scene0001_00",
                            "processed": True,
                            "pipeline_outcome": "processed",
                            "split": "train",
                            "has_cache_frames": True,
                            "final_cacheable_frame_count": 1,
                            "scene_skip_reason": None,
                        }
                    },
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        legacy_batch_b.write_text(
            json.dumps(
                {
                    "version": referability_module.REFERABILITY_CACHE_VERSION,
                    "model": "fake-vlm",
                    "alias_config_version": referability_module.ALIAS_CONFIG_VERSION,
                    "referability_backend": "crop_vlm_with_mesh_ray",
                    "label_batch_size": 1,
                    "frames": {},
                    "scene_grouping": {
                        "scene0002_00": {
                            "scene_id": "scene0002_00",
                            "split": "train",
                            "pipeline_outcome": "no_final_referability_frames",
                            "scene_skip_reason": "no_final_referability_frames",
                            "final_cacheable_frame_count": 0,
                        }
                    },
                    "scene_status": {
                        "scene0002_00": {
                            "scene_id": "scene0002_00",
                            "processed": True,
                            "pipeline_outcome": "no_final_referability_frames",
                            "split": "train",
                            "has_cache_frames": False,
                            "final_cacheable_frame_count": 0,
                            "scene_skip_reason": "no_final_referability_frames",
                        }
                    },
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        scene_status_path_for_output(output_path).write_text(
            json.dumps(
                {
                    "version": referability_module.SCENE_STATUS_VERSION,
                    "split": "train",
                    "completed_scenes": {
                        "scene0001_00": {
                            "status": "completed",
                            "batch_file": legacy_batch_a.name,
                            "updated_at": "2026-04-30T12:00:00Z",
                        },
                        "scene0002_00": {
                            "status": "completed",
                            "batch_file": legacy_batch_b.name,
                            "updated_at": "2026-04-30T12:05:00Z",
                        },
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

        batch_paths = list_batch_cache_paths(output_path)
        self.assertEqual(len(batch_paths), 3)
        new_batch_path = next(
            path for path in batch_paths
            if path not in {legacy_batch_a, legacy_batch_b}
        )
        cache_doc = json.loads(new_batch_path.read_text(encoding="utf-8"))
        self.assertEqual(select_calls, ["scene0003_00"])
        self.assertEqual(cache_doc["scene_status"]["scene0003_00"]["pipeline_outcome"], "processed")
        global_scene_status = load_scene_status_doc_for_output(output_path)
        self.assertEqual(global_scene_status["completed_scenes"]["scene0003_00"]["batch_file"], new_batch_path.name)

    def test_resume_rejects_scene_status_split_mismatch_for_val(self) -> None:
        root = Path(__file__).resolve().parent / "_tmp" / f"scene_status_split_guard_val_{uuid.uuid4().hex}"
        data_root = root / "data"
        make_scene_dir(data_root / "scans", "scene1001_00")
        output_path = root / "output" / "referability_cache.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        scene_status_path_for_output(output_path).write_text(
            json.dumps(
                {
                    "version": referability_module.SCENE_STATUS_VERSION,
                    "split": "train",
                    "completed_scenes": {},
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        self.addCleanup(shutil.rmtree, root, True)
        with (
            patch.dict(sys.modules, {"openai": make_fake_openai_module()}),
            patch.object(sys, "argv", [
                "run_vlm_referability.py",
                "--data_root",
                str(data_root),
                "--split",
                "val",
                "--output",
                str(output_path),
                "--resume",
                "--no-write_attachment_review",
            ]),
        ):
            with self.assertRaisesRegex(RuntimeError, "scene_status split mismatch"):
                referability_module.main()

    def test_resume_rejects_scene_status_split_mismatch_for_all(self) -> None:
        root = Path(__file__).resolve().parent / "_tmp" / f"scene_status_split_guard_all_{uuid.uuid4().hex}"
        data_root = root / "data"
        make_scene_dir(data_root / "scans", "scene0001_00")
        output_path = root / "output" / "referability_cache.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        scene_status_path_for_output(output_path).write_text(
            json.dumps(
                {
                    "version": referability_module.SCENE_STATUS_VERSION,
                    "split": "train",
                    "completed_scenes": {},
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        self.addCleanup(shutil.rmtree, root, True)
        with (
            patch.dict(sys.modules, {"openai": make_fake_openai_module()}),
            patch.object(sys, "argv", [
                "run_vlm_referability.py",
                "--data_root",
                str(data_root),
                "--split",
                "all",
                "--output",
                str(output_path),
                "--resume",
                "--no-write_attachment_review",
            ]),
        ):
            with self.assertRaisesRegex(RuntimeError, "scene_status split mismatch"):
                referability_module.main()

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
        _batch_path, cache_doc = load_single_batch_cache_for_output(output_path)
        self.assertIn("FINAL BATCH FOR SPLIT train", log_text)
        self.assertIn("ALL SCENES PROCESSED AFTER THIS RUN", log_text)
        self.assertEqual(sorted(cache_doc["scene_status"].keys()), ["scene0001_00", "scene0002_00"])
        global_scene_status = load_scene_status_doc_for_output(output_path)
        self.assertEqual(sorted(global_scene_status["completed_scenes"].keys()), ["scene0001_00", "scene0002_00"])

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
