from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pytest

from scripts.run_pipeline import (
    _attachment_surface_text_by_object_id,
    _find_inconsistent_referability_entry,
    _frames_from_referability_cache,
    _load_single_referability_cache,
    _manual_attachment_graph_for_scene,
    _manual_attachment_roles_for_frame,
    _merge_manual_attachment_cache,
    _prioritize_manual_attachment_scene_dirs,
)
from scripts.two_hop_attachment_salvage import (
    build_cache,
    parse_image_spec,
    project_object_box,
    render_review_html,
)
from src.qa_generator import generate_l3_attachment_chain
from src.utils.colmap_loader import CameraIntrinsics, CameraPose


def _camera() -> tuple[CameraPose, CameraIntrinsics]:
    return (
        CameraPose("frame_000010.jpg", np.eye(3), np.zeros(3)),
        CameraIntrinsics(640, 480, 500.0, 500.0, 320.0, 240.0),
    )


def _objects() -> list[dict]:
    labels = ["table", "laptop", "mouse", "chair"]
    return [
        {
            "id": index,
            "label": label,
            "center": [float(index), 0.0, 3.0],
            "bbox_min": [float(index) - 0.1, -0.1, 2.9],
            "bbox_max": [float(index) + 0.1, 0.1, 3.1],
        }
        for index, label in enumerate(labels, start=1)
    ]


def _frame(
    scene_id: str = "0d2ee665be",
    image_name: str = "frame_001510.jpg",
) -> dict:
    objects = [
        {"id": obj["id"], "label": obj["label"], "box": [10, 10, 50, 50]}
        for obj in _objects()
    ]
    return {
        "scene_id": scene_id,
        "image_name": image_name,
        "visible_object_ids": [1, 2, 3, 4],
        "objects": objects,
        "label_statuses": {obj["label"]: "unique" for obj in objects},
        "label_counts": {obj["label"]: 1 for obj in objects},
        "label_to_object_ids": {obj["label"]: [obj["id"]] for obj in objects},
        "image_width": 640,
        "image_height": 480,
        "image_data_url": "data:image/jpeg;base64,AA==",
    }


def test_parse_image_spec_expands_numeric_frame_ids() -> None:
    scene_id, image_names = parse_image_spec(r"0d2ee665be:001510\002740\001110")
    assert scene_id == "0d2ee665be"
    assert image_names == ["frame_001510.jpg", "frame_002740.jpg", "frame_001110.jpg"]


def test_project_object_box_returns_clipped_rectangle() -> None:
    pose, intrinsics = _camera()
    box = project_object_box(_objects()[0], pose, intrinsics, 640, 480)
    assert box is not None
    assert 0 <= box[0] < box[2] <= 640
    assert 0 <= box[1] < box[3] <= 480


def test_build_cache_produces_manual_graph_and_pipeline_frame() -> None:
    frame = _frame()
    selections = [{
        "scene_id": frame["scene_id"],
        "image_name": frame["image_name"],
        "roles": {
            "moved": {"id": 1, "label": "wooden desk"},
            "child": {"id": 2, "label": "open laptop"},
            "grandchild": {"id": 3, "label": "wireless mouse"},
            "contrast": {"id": 4, "label": "office chair"},
        },
    }]
    cache = build_cache([frame], selections)
    entry = cache["frames"][frame["scene_id"]][frame["image_name"]]
    assert cache["manual_attachment_graph"][frame["scene_id"]] == {"1": [2], "2": [3]}
    assert entry["attachment_referable_pairs"] == [[1, 2], [2, 3]]
    assert _manual_attachment_roles_for_frame(entry) == {
        "moved": 1,
        "child": 2,
        "grandchild": 3,
        "contrast": 4,
    }
    assert entry["manual_attachment_roles"]["child"]["label"] == "open laptop"
    assert entry["attachment_referable_pair_count"] == 2
    assert _find_inconsistent_referability_entry(cache) is None


def test_pipeline_normalizes_manual_graph() -> None:
    cache = {"manual_attachment_graph": {"scene": {"1": [2], "2": [3]}}}
    assert _manual_attachment_graph_for_scene(cache, "scene") == {1: [2], 2: [3]}


def test_pipeline_loads_manual_cache_without_vlm_final_fields(tmp_path) -> None:
    frame = _frame()
    cache = build_cache([frame], [{
        "scene_id": frame["scene_id"],
        "image_name": frame["image_name"],
        "roles": {
            "moved": {"id": 1, "label": "desk"},
            "child": {"id": 2, "label": "laptop"},
            "grandchild": {"id": 3, "label": "mouse"},
            "contrast": {"id": 4, "label": "chair"},
        },
    }])
    cache_path = tmp_path / "manual.json"
    cache_path.write_text(json.dumps(cache), encoding="utf-8")
    loaded = _load_single_referability_cache(cache_path, no_salvage=True)
    assert loaded is not None
    assert loaded["manual_attachment_graph"] == cache["manual_attachment_graph"]


def test_manual_roles_force_attachment_chain_contrast() -> None:
    pose, _ = _camera()
    questions = generate_l3_attachment_chain(
        _objects(),
        {1: [2], 2: [3]},
        {2: 1, 3: 2},
        pose,
        {"L3_attachment_chain": ["If we move {obj_a}, which objects move?"]},
        ordinary_reference_objects=_objects(),
        role_override={"moved": 1, "child": 2, "grandchild": 3, "contrast": 4},
        role_label_override_by_id={
            1: "work desk",
            2: "open computer",
            3: "wireless pointer",
            4: "visitor chair",
        },
    )
    assert len(questions) == 1
    question = questions[0]
    assert question["grandparent_id"] == 1
    assert question["parent_id"] == 2
    assert question["grandchild_id"] == 3
    assert question["neighbor_id"] == 4
    assert question["grandparent_label"] == "work desk"
    assert question["parent_label"] == "open computer"
    assert question["grandchild_label"] == "wireless pointer"
    assert question["neighbor_label"] == "visitor chair"


def test_review_html_contains_four_role_editors_and_cache_export() -> None:
    text = render_review_html([_frame()], "manual.json")
    for role in ("moved", "child", "grandchild", "contrast"):
        assert f'name="{role}_id"' in text
        assert f'name="{role}_label"' in text
    assert "<select" not in text
    assert "manual_attachment_graph" in text
    assert "manual.json" in text


def _manual_selection(scene_id: str, image_name: str, *, moved_id: int = 1) -> dict:
    return {
        "scene_id": scene_id,
        "image_name": image_name,
        "roles": {
            "moved": {"id": moved_id, "label": "manual moved"},
            "child": {"id": 2, "label": "manual child"},
            "grandchild": {"id": 3, "label": "manual grandchild"},
            "contrast": {"id": 4, "label": "manual contrast"},
        },
    }


def test_build_cache_rejects_unprojected_id_and_empty_label() -> None:
    frame = _frame()
    bad_id = _manual_selection(frame["scene_id"], frame["image_name"])
    bad_id["roles"]["contrast"]["id"] = 99
    with pytest.raises(ValueError, match="unavailable object id 99"):
        build_cache([frame], [bad_id])

    empty_label = _manual_selection(frame["scene_id"], frame["image_name"])
    empty_label["roles"]["child"]["label"] = "  "
    with pytest.raises(ValueError, match="empty label"):
        build_cache([frame], [empty_label])


def test_manual_cache_overrides_attachment_but_preserves_regular_referability() -> None:
    frame = _frame()
    manual = build_cache(
        [frame],
        [_manual_selection(frame["scene_id"], frame["image_name"])],
    )
    base = {
        "version": "20.0",
        "frames": {
            frame["scene_id"]: {
                frame["image_name"]: {
                    "frame_usable": False,
                    "candidate_visible_object_ids": [1, 5],
                    "referable_object_ids": [5],
                    "attachment_referable_object_ids": [1, 5],
                    "attachment_referable_pairs": [[5, 1]],
                    "final_selection_rank": 7,
                    "flash_only_field": "preserved",
                }
            }
        },
    }

    merged = _merge_manual_attachment_cache(base, manual)
    entry = merged["frames"][frame["scene_id"]][frame["image_name"]]

    assert entry["referable_object_ids"] == [5]
    assert entry["attachment_referable_object_ids"] == [1, 2, 3, 4]
    assert entry["attachment_referable_pairs"] == [[1, 2], [2, 3]]
    assert entry["candidate_visible_object_ids"] == [1, 2, 3, 4, 5]
    assert entry["frame_usable"] is True
    assert entry["final_selection_rank"] == -1
    assert entry["flash_only_field"] == "preserved"
    assert _attachment_surface_text_by_object_id(entry)[3] == "manual grandchild"


def test_manual_cache_adds_scene_and_keeps_only_roles_referable() -> None:
    frame = _frame("manual_scene", "frame_000020.jpg")
    frame["objects"].append({"id": 5, "label": "shelf", "box": [60, 10, 100, 50]})
    frame["visible_object_ids"].append(5)
    manual = build_cache(
        [frame],
        [_manual_selection(frame["scene_id"], frame["image_name"])],
    )
    base = {
        "version": "20.0",
        "frames": {
            "flash_scene": {
                "frame_000001.jpg": {
                    "frame_usable": True,
                    "candidate_visible_object_ids": [9],
                    "referable_object_ids": [9],
                    "attachment_referable_object_ids": [9],
                    "final_selection_rank": 0,
                }
            }
        },
    }

    merged = _merge_manual_attachment_cache(base, manual)
    entry = merged["frames"]["manual_scene"]["frame_000020.jpg"]

    assert merged["manual_attachment_scene_ids"] == ["manual_scene"]
    assert entry["referable_object_ids"] == [1, 2, 3, 4]
    assert entry["attachment_referable_object_ids"] == [1, 2, 3, 4]
    assert entry["candidate_visible_object_ids"] == [1, 2, 3, 4, 5]
    assert merged["manual_attachment_graph"]["manual_scene"] == {
        "1": [2],
        "2": [3],
    }


def test_manual_frames_sort_before_flash_frames() -> None:
    frame = _frame()
    manual = build_cache(
        [frame],
        [_manual_selection(frame["scene_id"], frame["image_name"])],
    )
    base = {
        "version": "20.0",
        "frames": {
            frame["scene_id"]: {
                "frame_000001.jpg": {
                    "frame_usable": True,
                    "candidate_visible_object_ids": [1],
                    "referable_object_ids": [1],
                    "attachment_referable_object_ids": [1],
                    "final_selection_rank": 0,
                }
            }
        },
    }

    merged = _merge_manual_attachment_cache(base, manual)
    frames = _frames_from_referability_cache(merged["frames"][frame["scene_id"]])
    assert frames[0]["image_name"] == frame["image_name"]


def test_manual_scenes_sort_before_flash_scenes() -> None:
    ordered = _prioritize_manual_attachment_scene_dirs(
        [Path("flash_scene"), Path("manual_scene"), Path("other_scene")],
        {"manual_attachment_scene_ids": ["manual_scene"]},
    )
    assert [path.name for path in ordered] == [
        "manual_scene",
        "flash_scene",
        "other_scene",
    ]


def test_manual_cache_rejects_conflicting_parents() -> None:
    first = _frame("scene", "frame_000001.jpg")
    second = _frame("scene", "frame_000002.jpg")
    manual = build_cache(
        [first, second],
        [
            _manual_selection("scene", "frame_000001.jpg"),
            {
                "scene_id": "scene",
                "image_name": "frame_000002.jpg",
                "roles": {
                    "moved": {"id": 4, "label": "other moved"},
                    "child": {"id": 2, "label": "manual child"},
                    "grandchild": {"id": 3, "label": "manual grandchild"},
                    "contrast": {"id": 1, "label": "manual contrast"},
                },
            },
        ],
    )
    with pytest.raises(ValueError, match="conflicting parents"):
        _merge_manual_attachment_cache({"version": "20.0", "frames": {}}, manual)


def test_manual_cache_rejects_wrong_schema() -> None:
    with pytest.raises(ValueError, match="two_hop_attachment_salvage_v1"):
        _merge_manual_attachment_cache(
            {"version": "20.0", "frames": {}},
            {"version": "20.0", "schema": "wrong", "frames": {"scene": {}}},
        )
