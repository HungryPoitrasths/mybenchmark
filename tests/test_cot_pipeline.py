from __future__ import annotations

import json
import importlib.util
from pathlib import Path

import pytest

from src.cot.facts import build_fact_record
from src.cot.images import collect_image_names
from src.cot.models import FactExtractionError
from src.cot.pipeline import build_dataset
from src.cot.render import render_response
from src.cot.templates import load_template_library, templates_for_signature
from src.cot.validators import validate_answer_mapping, validate_response


def _load_template_script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "generate_cot_templates.py"
    spec = importlib.util.spec_from_file_location("generate_cot_templates", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _question(qtype: str, value: str, **fields: object) -> dict[str, object]:
    question = {
        "level": "L1" if qtype in {"direction_agent", "occlusion", "distance", "direction_object_centric", "direction_allocentric"} else "L2",
        "type": qtype,
        "question": fields.pop("question", f"Synthetic {qtype} question"),
        "options": [value, "distractor one", "distractor two", "distractor three"],
        "answer": "A",
        "correct_value": value,
        "scene_id": "scene0000_00",
        "image_name": "frame.jpg",
    }
    question.update(fields)
    return question


def _supported_questions() -> list[dict[str, object]]:
    movement = {
        "question": "If the table is moved left by 1.0m, what changes?",
        "old_correct_value": "front",
        "moved_obj_id": 1,
        "moved_obj_label": "table",
        "query_obj_id": 2,
        "query_obj_label": "cup",
        "attachment_remapped": True,
        "delta": [-1.0, 0.0, 0.0],
        "relation_unchanged": False,
        "trace_reason": "attachment_relation_change",
    }
    return [
        _question("direction_agent", "front-left", obj_a_label="cup", obj_b_label="chair"),
        _question("occlusion", "not occluded", obj_a_label="cup", geometry_in_frame_ratio=0.7),
        _question("distance", "close (1.0-2.0m)", obj_a_label="cup", obj_b_label="chair", distance_bin_id="close"),
        _question("direction_object_centric", "right", obj_ref_label="chair", obj_face_label="table", obj_target_label="cup"),
        _question("direction_allocentric", "northeast", obj_a_label="cup", obj_b_label="chair", camera_cardinal="south"),
        _question("object_move_agent", "front-left", **movement, new_correct_value="front-left", obj_c_id=3, obj_c_label="chair"),
        _question(
            "object_move_distance",
            "close (1.0-2.0m)",
            **movement,
            new_correct_value="close (1.0-2.0m)",
            obj_c_id=3,
            obj_c_label="chair",
            old_distance_m=3.0,
            new_distance_m=1.5,
        ),
        _question(
            "object_move_occlusion",
            "the cup is occluded by the chair",
            **movement,
            new_correct_value="the cup is occluded by the chair",
            occlusion_semantics_version="pairwise_v1",
            new_pairwise_occlusion_relation="query_occluded_by_reference",
            obj_ref_id=3,
            obj_ref_label="chair",
        ),
        _question(
            "object_rotate_object_centric",
            "front-right",
            **movement,
            new_correct_value="front-right",
            obj_ref_id=3,
            obj_ref_label="lamp",
            obj_face_id=4,
            obj_face_label="chair",
            rotation_angle=90,
            rotation_direction="clockwise",
        ),
        _question("object_move_object_centric", "front-left", **movement, new_correct_value="front-left", obj_ref_id=3, obj_ref_label="chair"),
        _question("object_move_allocentric", "northeast", **movement, new_correct_value="northeast", obj_ref_id=3, obj_ref_label="chair", camera_cardinal="south"),
        _question(
            "object_remove",
            "not occluded",
            removed_obj_label="table",
            obj_b_label="cup",
            old_visibility="occluded",
            new_visibility="not occluded",
        ),
        {
            **_question("attachment_chain", "unused", grandparent_label="table", parent_label="tray", grandchild_label="cup", neighbor_label="chair"),
            "level": "L3",
            "multi_select": True,
            "options": ["the tray", "the cup", "the chair"],
            "answer": "A B",
            "correct_values": ["the tray", "the cup"],
            "correct_value": "the tray; the cup",
        },
        _question(
            "coordinate_rotation_agent",
            "right",
            level="L3",
            rotation_angle=90,
            old_direction="front",
            new_direction="right",
            obj_a_label="cup",
            obj_b_label="chair",
        ),
        _question(
            "coordinate_rotation_object_centric",
            "right",
            level="L3",
            rotation_angle=90,
            old_direction="front",
            new_direction="right",
            obj_ref_label="chair",
            obj_face_label="table",
            obj_target_label="cup",
            cross_frame_layout="ref_in_frame_1",
        ),
        _question(
            "coordinate_rotation_allocentric",
            "east",
            level="L3",
            rotation_angle=90,
            old_direction="north",
            new_direction="east",
            obj_a_label="cup",
            obj_b_label="chair",
            camera_cardinal="south",
        ),
    ]


@pytest.mark.parametrize("question", _supported_questions(), ids=lambda value: str(value["type"]))
def test_all_supported_types_extract_and_render(question: dict[str, object]) -> None:
    record = build_fact_record(question)
    validate_answer_mapping(question, record)
    response_a, template_a = render_response(record, seed=42)
    response_b, template_b = render_response(record, seed=42)
    validate_response(response_a, record)
    assert (response_a, template_a) == (response_b, template_b)
    assert response_a.splitlines()[-1] == "Answer: " + " ".join(record.answer_letters)
    assert response_a.count("Answer:") == 1


def test_vertical_direction_is_rejected() -> None:
    question = _question("direction_agent", "above", obj_a_label="cup", obj_b_label="chair")
    with pytest.raises(FactExtractionError, match="eight horizontal"):
        build_fact_record(question)


def test_legacy_unary_move_occlusion_is_rejected() -> None:
    question = _question(
        "object_move_occlusion",
        "not occluded",
        question="If the table is moved left by 1.0m, what is the status?",
        moved_obj_id=1,
        moved_obj_label="table",
        query_obj_id=2,
        query_obj_label="cup",
        target_obj_label="cup",
        delta=[-1.0, 0.0, 0.0],
    )
    with pytest.raises(FactExtractionError, match="pairwise"):
        build_fact_record(question)


def test_image_order_primary_auxiliary_destination() -> None:
    question = {
        "image_name": "first.jpg",
        "auxiliary_image_names": ["bridge1.jpg", "bridge2.jpg"],
        "reasoning_frame_2": "last.jpg",
    }
    assert collect_image_names(question) == [
        "first.jpg",
        "bridge1.jpg",
        "bridge2.jpg",
        "last.jpg",
    ]


def test_pipeline_exports_ms_swift_with_matching_placeholders(tmp_path: Path) -> None:
    image_path = tmp_path / "frame.jpg"
    image_path.write_bytes(b"not decoded by the exporter")
    question = _question(
        "direction_agent",
        "left",
        obj_a_label="cup",
        obj_b_label="chair",
        image_path=str(image_path),
    )
    result = build_dataset([question], require_images=True)
    assert result["report"]["accepted_count"] == 1
    item = result["sft"][0]
    assert item["messages"][0]["content"].count("<image>") == 1
    assert item["images"] == [str(image_path.resolve())]
    assert result["sidecar"][0]["option_count"] == 4
    assert result["sidecar"][0]["multi_select"] is False
    json.dumps(item, ensure_ascii=False)


def test_offline_templates_match_the_merge_contract(tmp_path: Path) -> None:
    script = _load_template_script()
    benchmark = tmp_path / "benchmark.json"
    benchmark.write_text(json.dumps(_supported_questions()[:2]), encoding="utf-8")
    rows = script.build_offline_responses(benchmark, script.DEFAULT_TEMPLATE_PATH)
    assert len(rows) == 2
    assert rows[0]["templates"] != rows[1]["templates"]
    for row in rows:
        templates = row["templates"]
        assert len(templates) == 12
        assert len(set(templates)) == 12
        for template in templates:
            assert all(slot in template for slot in ("{observation}", "{transformation}", "{conclusion}"))
            assert "signature" not in template.lower()

    rendered = [render_response(build_fact_record(question), seed=42)[0] for question in _supported_questions()]
    assert all(" signature " not in response.lower() for response in rendered)

    response_path = tmp_path / "template_responses.jsonl"
    output_path = tmp_path / "signature_templates.json"
    script.write_jsonl(response_path, rows)
    script.merge_responses(response_path, output_path, script.DEFAULT_TEMPLATE_PATH)
    library = load_template_library(output_path)
    overrides = library["signature_templates"]
    assert set(overrides) == {row["signature_id"] for row in rows}
    assert templates_for_signature(library, rows[0]["signature_id"]) != templates_for_signature(
        library, rows[1]["signature_id"]
    )


def test_template_library_rejects_embedded_answer_line() -> None:
    signature_id = "L1_direction_agent.single_axis.left"
    library = load_template_library()
    library["signature_templates"] = {
        signature_id: [
            {
                "id": f"invalid_{index:02d}",
                "template": "{observation}. {transformation}. Therefore, {conclusion}. Answer: A",
            }
            for index in range(12)
        ]
    }
    with pytest.raises(ValueError, match="must not contain an answer line"):
        templates_for_signature(library, signature_id)
