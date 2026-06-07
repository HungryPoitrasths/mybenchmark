#!/usr/bin/env python3
"""Recompute child-query answers for invalid attachment-remapped L2 object-move questions."""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import zlib
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets import make_data_source
from src.qa_generator import (
    ALL_DIRECTIONS,
    ALL_DISTANCES,
    _delta_to_description,
    _direction_values_for_query_object,
    _direction_with_camera_hint,
    _has_stable_object_centric_facing,
    _load_templates,
    _mention,
    _object_bottom_hull_xy,
    _the,
    generate_options,
)
from src.relation_engine import compute_all_relations, compute_distance_details, primary_direction_object_centric
from src.scene_parser import EXCLUDED_LABELS
from src.support_graph import enrich_scene_with_attachment, get_scene_attachment_graph
from src.virtual_ops import apply_movement


DEFAULT_SCANNET_ROOT = Path("/home/lihongxing/datasets/ScanNet/data/scans")
DEFAULT_SCANNETPP_ROOT = Path("/home/sujinyue/datasets/scannetpp")

SUPPORTED_QTYPES = {
    "object_move_agent",
    "object_move_distance",
    "object_move_object_centric",
}

SKIP_NOT_TARGET = "not_target_question"
SKIP_UNSUPPORTED_QTYPE = "unsupported_qtype"
SKIP_MISSING_CHILD_ID = "missing_attachment_child_id"
SKIP_MISSING_SCENE_ROOT = "missing_scene_root"
SKIP_SCENE_NOT_FOUND = "scene_dir_not_found"
SKIP_SCENE_LOAD_FAILED = "scene_load_failed"
SKIP_POSE_MISSING = "pose_missing_for_frame"
SKIP_OBJECT_MISSING = "moved_or_child_or_reference_missing"
SKIP_RELATION_MISSING = "relation_missing"
SKIP_DIRECTION_UNRESOLVED = "direction_unresolved"
SKIP_OBJECT_CENTRIC_FACING = "unstable_object_centric_facing"


def _json_key(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _stable_rank(seed: int, key: str) -> int:
    text = f"{seed}|{key}"
    return zlib.crc32(text.encode("utf-8")) & 0xFFFFFFFF


def _question_uid(question: dict[str, Any]) -> str:
    return _json_key(
        {
            "dataset": question.get("_dataset"),
            "scene_id": question.get("scene_id"),
            "image_name": question.get("image_name"),
            "level": question.get("level"),
            "type": question.get("type"),
            "question": question.get("question"),
            "options": question.get("options"),
            "answer": question.get("answer"),
        }
    )


def _coerce_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_seed(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _benchmark_questions(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        questions = payload.get("questions", [])
    else:
        questions = payload
    if not isinstance(questions, list):
        raise ValueError("Unsupported benchmark structure")
    return [question for question in questions if isinstance(question, dict)]


def _is_target_question(question: dict[str, Any]) -> bool:
    if str(question.get("level", "")).strip() != "L2":
        return False
    qtype = str(question.get("type", "")).strip()
    if qtype not in SUPPORTED_QTYPES:
        return False
    if not bool(question.get("attachment_remapped", False)):
        return False
    moved_obj_id = _coerce_int(question.get("moved_obj_id"))
    query_obj_id = _coerce_int(question.get("query_obj_id"))
    return moved_obj_id is not None and query_obj_id is not None and moved_obj_id == query_obj_id


def _resume_source_key(question: dict[str, Any]) -> str:
    return _json_key(
        {
            "trace_question_id": question.get("trace_question_id"),
            "scene_id": question.get("scene_id"),
            "image_name": question.get("image_name"),
            "level": question.get("level"),
            "type": question.get("type"),
            "moved_obj_id": question.get("moved_obj_id"),
            "invalid_query_obj_id": question.get("original_invalid_query_obj_id", question.get("query_obj_id")),
            "attachment_child_id": question.get("attachment_child_id"),
            "obj_c_id": question.get("obj_c_id"),
            "obj_ref_id": question.get("obj_ref_id"),
            "question": question.get("original_invalid_question", question.get("question")),
        }
    )


def _infer_dataset(question: dict[str, Any]) -> str | None:
    source_text = str(question.get("_source_benchmark", "")).lower()
    scene_id = str(question.get("scene_id", "")).strip().lower()
    if "scannetpp" in source_text:
        return "scannetpp"
    if "pilot" in source_text or scene_id.startswith("scene"):
        return "scannet"
    if scene_id and not scene_id.startswith("scene"):
        return "scannetpp"
    return None


def _resolve_scene_dir(root: Path, dataset: str, scene_id: str) -> Path:
    candidates = []
    if root.name == scene_id and root.is_dir():
        candidates.append(root)
    if dataset == "scannet":
        candidates.extend([root / scene_id, root / "scans" / scene_id])
    else:
        candidates.extend([root / scene_id, root / "scans" / scene_id])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(scene_id)


def _load_scene_context(
    *,
    dataset: str,
    scene_id: str,
    scannet_root: Path | None,
    scannetpp_root: Path | None,
    scannetpp_sensor: str,
) -> dict[str, Any]:
    root = scannetpp_root if dataset == "scannetpp" else scannet_root
    if root is None:
        raise FileNotFoundError(SKIP_MISSING_SCENE_ROOT)
    scene_dir = _resolve_scene_dir(root, dataset, scene_id)
    data_source = make_data_source(dataset, scene_dir, sensor=scannetpp_sensor)
    scene = data_source.load_scene()
    if scene is None:
        raise RuntimeError(f"Failed to parse scene {scene_id}")
    enrich_scene_with_attachment(scene)
    objects = [obj for obj in scene.get("objects", []) if isinstance(obj, dict)]
    obj_map = {int(obj["id"]): obj for obj in objects if _coerce_int(obj.get("id")) is not None}
    poses = data_source.load_poses()
    return {
        "dataset": dataset,
        "scene_dir": scene_dir,
        "scene": scene,
        "objects": objects,
        "obj_map": obj_map,
        "attachment_graph": get_scene_attachment_graph(scene),
        "poses": poses,
    }


def _relation_for_pair(relations: list[dict[str, Any]], obj_a_id: int, obj_b_id: int) -> dict[str, Any] | None:
    for relation in relations:
        ids = {_coerce_int(relation.get("obj_a_id")), _coerce_int(relation.get("obj_b_id"))}
        if ids == {obj_a_id, obj_b_id}:
            return relation
    return None


def _answer_pool_for_qtype(qtype: str) -> list[str]:
    if qtype in {"object_move_agent", "object_move_object_centric"}:
        return ALL_DIRECTIONS
    if qtype == "object_move_distance":
        return ALL_DISTANCES
    raise ValueError(f"Unsupported question type: {qtype}")


def _seeded_generate_options(correct_value: str, answer_pool: list[str], seed_key: str) -> tuple[list[str], str]:
    state = random.getstate()
    random.seed(zlib.crc32(seed_key.encode("utf-8")) & 0xFFFFFFFF)
    try:
        return generate_options(correct_value, answer_pool)
    finally:
        random.setstate(state)


def _update_options_and_answer(
    *,
    question: dict[str, Any],
    correct_value: str,
    answer_pool: list[str],
) -> tuple[list[str], str]:
    original_options = question.get("options")
    if isinstance(original_options, list) and all(isinstance(item, str) for item in original_options):
        options = [str(item) for item in original_options]
        if correct_value in options:
            answer_idx = options.index(correct_value)
            return options, chr(65 + answer_idx)
        old_correct = str(question.get("correct_value", "")).strip()
        if old_correct in options and correct_value not in options:
            replaced = [correct_value if item == old_correct else item for item in options]
            if len(set(replaced)) == len(replaced):
                answer_idx = replaced.index(correct_value)
                return replaced, chr(65 + answer_idx)
    return _seeded_generate_options(correct_value, answer_pool, str(question.get("question_uid") or question.get("trace_question_id") or question.get("question")))


def _render_question_text(
    *,
    templates: dict[str, Any],
    template_key: str,
    original_question: dict[str, Any],
    render_kwargs: dict[str, Any],
) -> str:
    tpl_list = list(templates.get(template_key, []))
    if not tpl_list:
        raise ValueError(f"Missing templates for {template_key}")
    for template in tpl_list:
        try:
            rendered = template.format(**render_kwargs["original"])
        except Exception:
            continue
        if rendered == str(original_question.get("question", "")):
            return template.format(**render_kwargs["updated"])
    return tpl_list[0].format(**render_kwargs["updated"])


def _base_render_kwargs(question: dict[str, Any], move_source_label: str) -> dict[str, Any]:
    delta = np.asarray(question.get("delta") or [0.0, 0.0, 0.0], dtype=np.float64)
    direction_desc = _delta_to_description(delta, question.get("_camera_pose"))
    return {
        "obj_a": _the(move_source_label),
        "obj_move_source": _the(move_source_label),
        "direction": direction_desc,
        "direction_with_camera_hint": _direction_with_camera_hint(direction_desc),
        "distance": f"{np.linalg.norm(delta):.1f}m",
    }


def _copy_common_updates(
    *,
    original: dict[str, Any],
    updated: dict[str, Any],
    child_id: int,
    child_label: str,
    seed: int | None,
) -> dict[str, Any]:
    result = copy.deepcopy(original)
    result.update(updated)
    result["query_obj_id"] = child_id
    result["query_obj_label"] = child_label
    result["attachment_remapped"] = True
    result["has_attachment_chain"] = True
    moved_obj_id = _coerce_int(original.get("moved_obj_id"))
    if moved_obj_id is not None:
        result["attachment_parent_id"] = moved_obj_id
        result["attachment_pair_id"] = f"{moved_obj_id}->{child_id}"
    result["attachment_child_id"] = child_id
    result["child_query_recomputed"] = True
    result["original_invalid_query_obj_id"] = original.get("query_obj_id")
    result["original_invalid_question"] = original.get("question")
    result["resume_source_key"] = _resume_source_key(original)
    result.pop("question_referability_audit", None)
    result["question_uid"] = _question_uid(result)
    if seed is not None:
        result["_rank"] = _stable_rank(seed, result["question_uid"])
    return result


def _recompute_agent_question(
    *,
    question: dict[str, Any],
    ctx: dict[str, Any],
    child_obj: dict[str, Any],
    move_source: dict[str, Any],
    moved_map: dict[int, dict[str, Any]],
    moved_objects: list[dict[str, Any]],
    templates: dict[str, Any],
    seed: int | None,
) -> tuple[dict[str, Any] | None, str | None]:
    other_id = _coerce_int(question.get("obj_c_id"))
    if other_id is None:
        return None, SKIP_OBJECT_MISSING
    other_obj = ctx["obj_map"].get(other_id)
    moved_child = moved_map.get(int(child_obj["id"]))
    moved_other = moved_map.get(other_id, other_obj)
    if other_obj is None or moved_child is None or moved_other is None:
        return None, SKIP_OBJECT_MISSING
    camera_pose = question["_camera_pose"]
    base_relations = compute_all_relations(ctx["objects"], camera_pose, None, None)
    new_relations = compute_all_relations(moved_objects, camera_pose, None, None)
    old_relation = _relation_for_pair(base_relations, int(child_obj["id"]), other_id)
    new_relation = _relation_for_pair(new_relations, int(child_obj["id"]), other_id)
    if old_relation is None or new_relation is None:
        return None, SKIP_RELATION_MISSING
    direction_values = _direction_values_for_query_object(int(child_obj["id"]), old_relation, new_relation)
    if direction_values is None:
        return None, SKIP_DIRECTION_UNRESOLVED
    old_value, new_value = direction_values
    base_kwargs = _base_render_kwargs(question, move_source["label"])
    text = _render_question_text(
        templates=templates,
        template_key="L2_object_move_agent",
        original_question=question,
        render_kwargs={
            "original": {
                **base_kwargs,
                "obj_b": _the(str(question.get("query_obj_label") or move_source["label"])),
                "obj_c": _the(str(question.get("obj_c_label") or other_obj["label"])),
            },
            "updated": {
                **base_kwargs,
                "obj_b": _the(child_obj["label"]),
                "obj_c": _the(other_obj["label"]),
            },
        },
    )
    options, answer = _update_options_and_answer(question=question, correct_value=new_value, answer_pool=ALL_DIRECTIONS)
    return _copy_common_updates(
        original=question,
        updated={
            "question": text,
            "options": options,
            "answer": answer,
            "correct_value": new_value,
            "old_correct_value": old_value,
            "new_correct_value": new_value,
            "obj_b_id": int(child_obj["id"]),
            "obj_b_label": child_obj["label"],
            "obj_c_id": other_id,
            "obj_c_label": other_obj["label"],
            "mentioned_objects": [
                _mention("moved_object", move_source["label"], int(move_source["id"])),
                _mention("query_object", child_obj["label"], int(child_obj["id"])),
                _mention("relation_obj_b", child_obj["label"], int(child_obj["id"])),
                _mention("relation_obj_c", other_obj["label"], other_id),
            ],
            "relation_unchanged": old_value == new_value,
            "trace_reason": "attachment_child_query_recomputed_agent",
        },
        child_id=int(child_obj["id"]),
        child_label=child_obj["label"],
        seed=seed,
    ), None


def _recompute_distance_question(
    *,
    question: dict[str, Any],
    ctx: dict[str, Any],
    child_obj: dict[str, Any],
    move_source: dict[str, Any],
    moved_map: dict[int, dict[str, Any]],
    templates: dict[str, Any],
    seed: int | None,
) -> tuple[dict[str, Any] | None, str | None]:
    other_id = _coerce_int(question.get("obj_c_id"))
    if other_id is None:
        return None, SKIP_OBJECT_MISSING
    other_obj = ctx["obj_map"].get(other_id)
    moved_child = moved_map.get(int(child_obj["id"]))
    moved_other = moved_map.get(other_id, other_obj)
    if other_obj is None or moved_child is None or moved_other is None:
        return None, SKIP_OBJECT_MISSING

    old_distance = compute_distance_details(child_obj, other_obj)
    new_distance = compute_distance_details(moved_child, moved_other)
    old_value = str(old_distance.get("distance_bin", "")).strip()
    new_value = str(new_distance.get("distance_bin", "")).strip()
    if not old_value or not new_value:
        return None, SKIP_DIRECTION_UNRESOLVED

    base_kwargs = _base_render_kwargs(question, move_source["label"])
    text = _render_question_text(
        templates=templates,
        template_key="L2_object_move_distance",
        original_question=question,
        render_kwargs={
            "original": {
                **base_kwargs,
                "obj_b": _the(str(question.get("query_obj_label") or move_source["label"])),
                "obj_c": _the(str(question.get("obj_c_label") or other_obj["label"])),
            },
            "updated": {
                **base_kwargs,
                "obj_b": _the(child_obj["label"]),
                "obj_c": _the(other_obj["label"]),
            },
        },
    )
    options, answer = _update_options_and_answer(question=question, correct_value=new_value, answer_pool=ALL_DISTANCES)
    return _copy_common_updates(
        original=question,
        updated={
            "question": text,
            "options": options,
            "answer": answer,
            "correct_value": new_value,
            "old_correct_value": old_value,
            "new_correct_value": new_value,
            "old_distance_m": float(old_distance.get("distance_m", 0.0) or 0.0),
            "new_distance_m": float(new_distance.get("distance_m", 0.0) or 0.0),
            "old_distance_bin_id": old_distance.get("distance_bin_id"),
            "new_distance_bin_id": new_distance.get("distance_bin_id"),
            "distance_definition": new_distance.get("distance_definition"),
            "old_distance_definition": old_distance.get("distance_definition"),
            "new_distance_definition": new_distance.get("distance_definition"),
            "obj_b_id": int(child_obj["id"]),
            "obj_b_label": child_obj["label"],
            "obj_c_id": other_id,
            "obj_c_label": other_obj["label"],
            "mentioned_objects": [
                _mention("moved_object", move_source["label"], int(move_source["id"])),
                _mention("query_object", child_obj["label"], int(child_obj["id"])),
                _mention("relation_obj_b", child_obj["label"], int(child_obj["id"])),
                _mention("relation_obj_c", other_obj["label"], other_id),
            ],
            "relation_unchanged": old_value == new_value,
            "trace_reason": "attachment_child_query_recomputed_distance",
        },
        child_id=int(child_obj["id"]),
        child_label=child_obj["label"],
        seed=seed,
    ), None


def _recompute_object_centric_question(
    *,
    question: dict[str, Any],
    ctx: dict[str, Any],
    child_obj: dict[str, Any],
    move_source: dict[str, Any],
    moved_map: dict[int, dict[str, Any]],
    templates: dict[str, Any],
    seed: int | None,
) -> tuple[dict[str, Any] | None, str | None]:
    ref_id = _coerce_int(question.get("obj_ref_id"))
    if ref_id is None:
        return None, SKIP_OBJECT_MISSING
    ref_obj = ctx["obj_map"].get(ref_id)
    moved_child = moved_map.get(int(child_obj["id"]))
    moved_ref = moved_map.get(ref_id, ref_obj)
    if ref_obj is None or moved_child is None or moved_ref is None:
        return None, SKIP_OBJECT_MISSING

    camera_center = np.asarray(question["_camera_pose"].position, dtype=float)
    query_center = np.asarray(child_obj["center"], dtype=float)
    if not _has_stable_object_centric_facing(query_center, camera_center):
        return None, SKIP_OBJECT_CENTRIC_FACING
    facing_offset = camera_center - query_center
    old_dir, old_amb = primary_direction_object_centric(
        query_center,
        query_center + facing_offset,
        np.asarray(ref_obj["center"], dtype=float),
        horizontal_only=True,
        anchor_hull_xy=_object_bottom_hull_xy(child_obj),
        target_hull_xy=_object_bottom_hull_xy(ref_obj),
        anchor_bbox_min=np.asarray(child_obj["bbox_min"], dtype=float),
        anchor_bbox_max=np.asarray(child_obj["bbox_max"], dtype=float),
        target_bbox_min=np.asarray(ref_obj["bbox_min"], dtype=float),
        target_bbox_max=np.asarray(ref_obj["bbox_max"], dtype=float),
    )
    new_query_center = np.asarray(moved_child["center"], dtype=float)
    new_dir, new_amb = primary_direction_object_centric(
        new_query_center,
        new_query_center + facing_offset,
        np.asarray(moved_ref["center"], dtype=float),
        horizontal_only=True,
        anchor_hull_xy=_object_bottom_hull_xy(moved_child),
        target_hull_xy=_object_bottom_hull_xy(moved_ref),
        anchor_bbox_min=np.asarray(moved_child["bbox_min"], dtype=float),
        anchor_bbox_max=np.asarray(moved_child["bbox_max"], dtype=float),
        target_bbox_min=np.asarray(moved_ref["bbox_min"], dtype=float),
        target_bbox_max=np.asarray(moved_ref["bbox_max"], dtype=float),
    )
    if old_dir not in ALL_DIRECTIONS or new_dir not in ALL_DIRECTIONS or max(float(old_amb), float(new_amb)) > 0.7:
        return None, SKIP_DIRECTION_UNRESOLVED

    base_kwargs = _base_render_kwargs(question, move_source["label"])
    text = _render_question_text(
        templates=templates,
        template_key="L2_object_move_object_centric",
        original_question=question,
        render_kwargs={
            "original": {
                **base_kwargs,
                "obj_query": _the(str(question.get("query_obj_label") or move_source["label"])),
                "obj_ref": _the(str(question.get("obj_ref_label") or ref_obj["label"])),
            },
            "updated": {
                **base_kwargs,
                "obj_query": _the(child_obj["label"]),
                "obj_ref": _the(ref_obj["label"]),
            },
        },
    )
    options, answer = _update_options_and_answer(question=question, correct_value=new_dir, answer_pool=ALL_DIRECTIONS)
    return _copy_common_updates(
        original=question,
        updated={
            "question": text,
            "options": options,
            "answer": answer,
            "correct_value": new_dir,
            "old_correct_value": old_dir,
            "new_correct_value": new_dir,
            "obj_ref_id": ref_id,
            "obj_ref_label": ref_obj["label"],
            "mentioned_objects": [
                _mention("moved_object", move_source["label"], int(move_source["id"])),
                _mention("query_object", child_obj["label"], int(child_obj["id"])),
                _mention("reference_object", ref_obj["label"], ref_id),
            ],
            "relation_unchanged": old_dir == new_dir,
            "trace_reason": "attachment_child_query_recomputed_object_centric",
        },
        child_id=int(child_obj["id"]),
        child_label=child_obj["label"],
        seed=seed,
    ), None


def _recompute_question_for_child_query(
    *,
    question: dict[str, Any],
    ctx: dict[str, Any],
    templates: dict[str, Any],
    seed: int | None,
) -> tuple[dict[str, Any] | None, str | None]:
    qtype = str(question.get("type", "")).strip()
    if qtype not in SUPPORTED_QTYPES:
        return None, SKIP_UNSUPPORTED_QTYPE
    child_id = _coerce_int(question.get("attachment_child_id"))
    moved_obj_id = _coerce_int(question.get("moved_obj_id"))
    if child_id is None:
        return None, SKIP_MISSING_CHILD_ID
    if moved_obj_id is None:
        return None, SKIP_OBJECT_MISSING

    move_source = ctx["obj_map"].get(moved_obj_id)
    child_obj = ctx["obj_map"].get(child_id)
    if move_source is None or child_obj is None:
        return None, SKIP_OBJECT_MISSING
    if str(move_source.get("label", "")).lower() in EXCLUDED_LABELS:
        return None, SKIP_OBJECT_MISSING

    delta = np.asarray(question.get("delta") or [0.0, 0.0, 0.0], dtype=np.float64)
    moved_objects = apply_movement(ctx["objects"], ctx["attachment_graph"], moved_obj_id, delta)
    moved_map = {int(obj["id"]): obj for obj in moved_objects}

    if qtype == "object_move_agent":
        return _recompute_agent_question(
            question=question,
            ctx=ctx,
            child_obj=child_obj,
            move_source=move_source,
            moved_map=moved_map,
            moved_objects=moved_objects,
            templates=templates,
            seed=seed,
        )
    if qtype == "object_move_distance":
        return _recompute_distance_question(
            question=question,
            ctx=ctx,
            child_obj=child_obj,
            move_source=move_source,
            moved_map=moved_map,
            templates=templates,
            seed=seed,
        )
    if qtype == "object_move_object_centric":
        return _recompute_object_centric_question(
            question=question,
            ctx=ctx,
            child_obj=child_obj,
            move_source=move_source,
            moved_map=moved_map,
            templates=templates,
            seed=seed,
        )
    return None, SKIP_UNSUPPORTED_QTYPE


def _skip_entry(question: dict[str, Any], reason: str) -> dict[str, Any]:
    return {
        "scene_id": question.get("scene_id"),
        "image_name": question.get("image_name"),
        "type": question.get("type"),
        "trace_question_id": question.get("trace_question_id"),
        "moved_obj_id": question.get("moved_obj_id"),
        "query_obj_id": question.get("query_obj_id"),
        "attachment_child_id": question.get("attachment_child_id"),
        "skip_reason": reason,
        "resume_source_key": _resume_source_key(question),
    }


def _make_processed_entry(
    question: dict[str, Any],
    *,
    status: str,
    skip_reason: str | None = None,
) -> dict[str, Any]:
    entry = {
        "scene_id": question.get("scene_id"),
        "image_name": question.get("image_name"),
        "type": question.get("type"),
        "trace_question_id": question.get("trace_question_id"),
        "moved_obj_id": question.get("moved_obj_id"),
        "query_obj_id": question.get("query_obj_id"),
        "attachment_child_id": question.get("attachment_child_id"),
        "resume_source_key": _resume_source_key(question),
        "status": status,
    }
    if skip_reason is not None:
        entry["skip_reason"] = skip_reason
    return entry


def _build_resume_maps(
    resume_output_payload: dict[str, Any] | None,
    resume_report: dict[str, Any] | None,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    resumed_question_by_key: dict[str, dict[str, Any]] = {}
    resumed_processed_by_key: dict[str, dict[str, Any]] = {}

    if resume_output_payload is not None:
        for question in _benchmark_questions(resume_output_payload):
            key = str(question.get("resume_source_key") or _resume_source_key(question))
            resumed_question_by_key[key] = question
            if bool(question.get("child_query_recomputed")):
                resumed_processed_by_key.setdefault(
                    key,
                    _make_processed_entry(question, status="fixed"),
                )

    if isinstance(resume_report, dict):
        processed = resume_report.get("processed", [])
        if isinstance(processed, list):
            for entry in processed:
                if not isinstance(entry, dict):
                    continue
                key = str(entry.get("resume_source_key") or "")
                if not key:
                    continue
                resumed_processed_by_key[key] = dict(entry)
    return resumed_question_by_key, resumed_processed_by_key


def _summarize_target_scenes(questions: list[dict[str, Any]]) -> list[tuple[str, str]]:
    seen: set[tuple[str, str]] = set()
    ordered: list[tuple[str, str]] = []
    for question in questions:
        if not _is_target_question(question):
            continue
        dataset = _infer_dataset(question)
        scene_id = str(question.get("scene_id", "")).strip()
        if dataset is None or not scene_id:
            continue
        key = (dataset, scene_id)
        if key not in seen:
            seen.add(key)
            ordered.append(key)
    return ordered


def _build_output_payload(
    payload: dict[str, Any],
    output_questions: list[dict[str, Any]],
    remaining_questions: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    output_payload = dict(payload)
    if remaining_questions:
        output_payload["questions"] = list(output_questions) + [copy.deepcopy(question) for question in remaining_questions]
    else:
        output_payload["questions"] = list(output_questions)
    return output_payload


def _build_report(
    *,
    target_count: int,
    fixed_count: int,
    skipped: list[dict[str, Any]],
    processed: list[dict[str, Any]],
    restored_count: int,
) -> dict[str, Any]:
    return {
        "target_count": target_count,
        "fixed_count": fixed_count,
        "skipped_count": len(skipped),
        "processed_count": len(processed),
        "restored_count": restored_count,
        "remaining_count": max(target_count - len(processed), 0),
        "skipped": skipped,
        "processed": processed,
    }


def _finalize_metadata(
    fixed_payload: dict[str, Any],
    report: dict[str, Any],
    *,
    input_path: Path,
    output_path: Path,
    report_path: Path,
    scannet_root: Path | None,
    scannetpp_root: Path | None,
    scannetpp_sensor: str,
    resume: bool,
    checkpoint_every: int,
    progress_every: int,
) -> dict[str, Any]:
    metadata = dict(fixed_payload.get("metadata", {}))
    postprocess = dict(metadata.get("postprocess", {})) if isinstance(metadata.get("postprocess"), dict) else {}
    postprocess["self_attachment_child_recompute"] = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "report_path": str(report_path),
        "scannet_root": str(scannet_root) if scannet_root is not None else None,
        "scannetpp_root": str(scannetpp_root) if scannetpp_root is not None else None,
        "scannetpp_sensor": scannetpp_sensor,
        "resume": resume,
        "checkpoint_every": checkpoint_every,
        "progress_every": progress_every,
        "target_count": report["target_count"],
        "fixed_count": report["fixed_count"],
        "skipped_count": report["skipped_count"],
        "processed_count": report["processed_count"],
        "restored_count": report["restored_count"],
        "remaining_count": report["remaining_count"],
    }
    metadata["postprocess"] = postprocess
    fixed_payload["metadata"] = metadata
    return fixed_payload


def _write_checkpoint_files(
    *,
    payload: dict[str, Any],
    report: dict[str, Any],
    input_path: Path,
    output_path: Path,
    report_path: Path,
    scannet_root: Path | None,
    scannetpp_root: Path | None,
    scannetpp_sensor: str,
    resume: bool,
    checkpoint_every: int,
    progress_every: int,
) -> None:
    output_payload = _finalize_metadata(
        copy.deepcopy(payload),
        copy.deepcopy(report),
        input_path=input_path,
        output_path=output_path,
        report_path=report_path,
        scannet_root=scannet_root,
        scannetpp_root=scannetpp_root,
        scannetpp_sensor=scannetpp_sensor,
        resume=resume,
        checkpoint_every=checkpoint_every,
        progress_every=progress_every,
    )
    _write_json(output_path, output_payload)
    _write_json(report_path, report)


def _maybe_log_and_checkpoint(
    *,
    payload: dict[str, Any],
    questions: list[dict[str, Any]],
    output_questions: list[dict[str, Any]],
    index: int,
    total_targets: int,
    fixed_count: int,
    skipped: list[dict[str, Any]],
    processed: list[dict[str, Any]],
    restored_count: int,
    progress_every: int,
    checkpoint_every: int,
    log: Callable[[str], None] | None,
    checkpoint_callback: Callable[[dict[str, Any], dict[str, Any]], None] | None,
    log_message: str,
) -> None:
    processed_count = len(processed)
    if log is not None and (processed_count % max(progress_every, 1) == 0 or processed_count == total_targets):
        log(log_message)
    if checkpoint_callback is not None and checkpoint_every > 0 and processed_count % checkpoint_every == 0:
        checkpoint_callback(
            _build_output_payload(payload, output_questions, questions[index + 1 :]),
            _build_report(
                target_count=total_targets,
                fixed_count=fixed_count,
                skipped=skipped,
                processed=processed,
                restored_count=restored_count,
            ),
        )


def recompute_attachment_child_queries(
    payload: dict[str, Any],
    *,
    scannet_root: Path | None,
    scannetpp_root: Path | None,
    scannetpp_sensor: str = "iphone",
    resume_output_payload: dict[str, Any] | None = None,
    resume_report: dict[str, Any] | None = None,
    progress_every: int = 10,
    checkpoint_every: int = 0,
    log: Callable[[str], None] | None = None,
    checkpoint_callback: Callable[[dict[str, Any], dict[str, Any]], None] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    questions = _benchmark_questions(payload)
    templates = _load_templates()
    seed = _coerce_seed(payload.get("metadata", {}).get("seed"))
    scene_cache: dict[tuple[str, str], dict[str, Any]] = {}
    resumed_question_by_key, resumed_processed_by_key = _build_resume_maps(resume_output_payload, resume_report)
    target_scene_keys = _summarize_target_scenes(questions)
    scene_index_lookup = {key: idx + 1 for idx, key in enumerate(target_scene_keys)}
    total_targets = sum(1 for question in questions if _is_target_question(question))

    output_questions: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    processed: list[dict[str, Any]] = []
    target_count = 0
    fixed_count = 0
    restored_count = 0

    if log is not None:
        log(
            f"[start] total_questions={len(questions)} target_questions={total_targets} "
            f"target_scenes={len(target_scene_keys)} resume_loaded={len(resumed_processed_by_key)}"
        )

    for index, question in enumerate(questions):
        if not _is_target_question(question):
            output_questions.append(question)
            continue

        target_count += 1
        processed_count = len(processed)
        resume_key = _resume_source_key(question)
        resumed_entry = resumed_processed_by_key.get(resume_key)
        if resumed_entry is not None:
            resumed_question = resumed_question_by_key.get(resume_key, question)
            output_questions.append(copy.deepcopy(resumed_question))
            processed.append(copy.deepcopy(resumed_entry))
            if str(resumed_entry.get("status")) == "fixed":
                fixed_count += 1
            else:
                skipped.append(_skip_entry(question, str(resumed_entry.get("skip_reason") or SKIP_SCENE_LOAD_FAILED)))
            restored_count += 1
            _maybe_log_and_checkpoint(
                payload=payload,
                questions=questions,
                output_questions=output_questions,
                index=index,
                total_targets=total_targets,
                fixed_count=fixed_count,
                skipped=skipped,
                processed=processed,
                restored_count=restored_count,
                progress_every=progress_every,
                checkpoint_every=checkpoint_every,
                log=log,
                checkpoint_callback=checkpoint_callback,
                log_message=(
                    f"[progress] {len(processed)}/{total_targets} targets done "
                    f"(fixed={fixed_count}, skipped={len(skipped)}, restored={restored_count}) "
                    f"resume scene={question.get('scene_id')} image={question.get('image_name')} type={question.get('type')}"
                ),
            )
            continue

        dataset = _infer_dataset(question)
        if dataset is None:
            output_questions.append(question)
            skip_item = _skip_entry(question, SKIP_MISSING_SCENE_ROOT)
            skipped.append(skip_item)
            processed.append(_make_processed_entry(question, status="skipped", skip_reason=SKIP_MISSING_SCENE_ROOT))
            _maybe_log_and_checkpoint(
                payload=payload,
                questions=questions,
                output_questions=output_questions,
                index=index,
                total_targets=total_targets,
                fixed_count=fixed_count,
                skipped=skipped,
                processed=processed,
                restored_count=restored_count,
                progress_every=progress_every,
                checkpoint_every=checkpoint_every,
                log=log,
                checkpoint_callback=checkpoint_callback,
                log_message=(
                    f"[progress] {len(processed)}/{total_targets} targets done "
                    f"(fixed={fixed_count}, skipped={len(skipped)}, restored={restored_count}) "
                    f"scene={question.get('scene_id')} image={question.get('image_name')} type={question.get('type')}"
                ),
            )
            continue
        scene_id = str(question.get("scene_id", "")).strip()
        image_name = str(question.get("image_name", "")).strip()
        cache_key = (dataset, scene_id)
        if cache_key not in scene_cache:
            if log is not None:
                scene_idx = scene_index_lookup.get(cache_key, len(scene_cache) + 1)
                log(f"[scene {scene_idx}/{len(target_scene_keys)}] loading {dataset}:{scene_id}")
            try:
                scene_cache[cache_key] = _load_scene_context(
                    dataset=dataset,
                    scene_id=scene_id,
                    scannet_root=scannet_root,
                    scannetpp_root=scannetpp_root,
                    scannetpp_sensor=scannetpp_sensor,
                )
                if log is not None:
                    ctx = scene_cache[cache_key]
                    log(
                        f"[scene {scene_idx}/{len(target_scene_keys)}] loaded {dataset}:{scene_id} "
                        f"objects={len(ctx['objects'])} poses={len(ctx['poses'])}"
                    )
            except FileNotFoundError as exc:
                reason = str(exc)
                if reason == SKIP_MISSING_SCENE_ROOT:
                    output_questions.append(question)
                    skipped.append(_skip_entry(question, SKIP_MISSING_SCENE_ROOT))
                    processed.append(_make_processed_entry(question, status="skipped", skip_reason=SKIP_MISSING_SCENE_ROOT))
                    _maybe_log_and_checkpoint(
                        payload=payload,
                        questions=questions,
                        output_questions=output_questions,
                        index=index,
                        total_targets=total_targets,
                        fixed_count=fixed_count,
                        skipped=skipped,
                        processed=processed,
                        restored_count=restored_count,
                        progress_every=progress_every,
                        checkpoint_every=checkpoint_every,
                        log=log,
                        checkpoint_callback=checkpoint_callback,
                        log_message=(
                            f"[progress] {len(processed)}/{total_targets} targets done "
                            f"(fixed={fixed_count}, skipped={len(skipped)}, restored={restored_count}) "
                            f"scene={scene_id} image={image_name} type={question.get('type')}"
                        ),
                    )
                    continue
                output_questions.append(question)
                skipped.append(_skip_entry(question, SKIP_SCENE_NOT_FOUND))
                processed.append(_make_processed_entry(question, status="skipped", skip_reason=SKIP_SCENE_NOT_FOUND))
                if log is not None:
                    log(f"[scene {scene_idx}/{len(target_scene_keys)}] missing {dataset}:{scene_id}")
                _maybe_log_and_checkpoint(
                    payload=payload,
                    questions=questions,
                    output_questions=output_questions,
                    index=index,
                    total_targets=total_targets,
                    fixed_count=fixed_count,
                    skipped=skipped,
                    processed=processed,
                    restored_count=restored_count,
                    progress_every=progress_every,
                    checkpoint_every=checkpoint_every,
                    log=log,
                    checkpoint_callback=checkpoint_callback,
                    log_message=(
                        f"[progress] {len(processed)}/{total_targets} targets done "
                        f"(fixed={fixed_count}, skipped={len(skipped)}, restored={restored_count}) "
                        f"scene={scene_id} image={image_name} type={question.get('type')}"
                    ),
                )
                continue
            except Exception:
                output_questions.append(question)
                skipped.append(_skip_entry(question, SKIP_SCENE_LOAD_FAILED))
                processed.append(_make_processed_entry(question, status="skipped", skip_reason=SKIP_SCENE_LOAD_FAILED))
                if log is not None:
                    log(f"[scene {scene_idx}/{len(target_scene_keys)}] failed {dataset}:{scene_id}")
                _maybe_log_and_checkpoint(
                    payload=payload,
                    questions=questions,
                    output_questions=output_questions,
                    index=index,
                    total_targets=total_targets,
                    fixed_count=fixed_count,
                    skipped=skipped,
                    processed=processed,
                    restored_count=restored_count,
                    progress_every=progress_every,
                    checkpoint_every=checkpoint_every,
                    log=log,
                    checkpoint_callback=checkpoint_callback,
                    log_message=(
                        f"[progress] {len(processed)}/{total_targets} targets done "
                        f"(fixed={fixed_count}, skipped={len(skipped)}, restored={restored_count}) "
                        f"scene={scene_id} image={image_name} type={question.get('type')}"
                    ),
                )
                continue
        ctx = scene_cache[cache_key]
        camera_pose = ctx["poses"].get(image_name)
        if camera_pose is None:
            output_questions.append(question)
            skipped.append(_skip_entry(question, SKIP_POSE_MISSING))
            processed.append(_make_processed_entry(question, status="skipped", skip_reason=SKIP_POSE_MISSING))
        else:
            question_with_pose = dict(question)
            question_with_pose["_camera_pose"] = camera_pose
            recomputed, reason = _recompute_question_for_child_query(
                question=question_with_pose,
                ctx=ctx,
                templates=templates,
                seed=seed,
            )
            if recomputed is None:
                output_questions.append(question)
                skip_reason = reason or SKIP_SCENE_LOAD_FAILED
                skipped.append(_skip_entry(question, skip_reason))
                processed.append(_make_processed_entry(question, status="skipped", skip_reason=skip_reason))
            else:
                recomputed.pop("_camera_pose", None)
                output_questions.append(recomputed)
                fixed_count += 1
                processed.append(_make_processed_entry(question, status="fixed"))

        _maybe_log_and_checkpoint(
            payload=payload,
            questions=questions,
            output_questions=output_questions,
            index=index,
            total_targets=total_targets,
            fixed_count=fixed_count,
            skipped=skipped,
            processed=processed,
            restored_count=restored_count,
            progress_every=progress_every,
            checkpoint_every=checkpoint_every,
            log=log,
            checkpoint_callback=checkpoint_callback,
            log_message=(
                f"[progress] {len(processed)}/{total_targets} targets done "
                f"(fixed={fixed_count}, skipped={len(skipped)}, restored={restored_count}) "
                f"scene={scene_id} image={image_name} type={question.get('type')}"
            ),
        )

    output_payload = _build_output_payload(payload, output_questions)
    report = _build_report(
        target_count=target_count,
        fixed_count=fixed_count,
        skipped=skipped,
        processed=processed,
        restored_count=restored_count,
    )
    return output_payload, report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recompute child-query answers for invalid attachment-remapped L2 object_move questions.",
    )
    parser.add_argument("--input", default="output/benchmark_subset.json", help="Input benchmark JSON")
    parser.add_argument(
        "--output",
        default="output/benchmark_subset.attachment_child_recomputed.json",
        help="Output benchmark JSON",
    )
    parser.add_argument(
        "--report",
        default="output/benchmark_subset.attachment_child_recomputed_report.json",
        help="Output JSON report",
    )
    parser.add_argument("--scannet-root", type=Path, default=DEFAULT_SCANNET_ROOT)
    parser.add_argument("--scannetpp-root", type=Path, default=DEFAULT_SCANNETPP_ROOT)
    parser.add_argument("--scannetpp-sensor", choices=("iphone", "dslr"), default="iphone")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output/report if present")
    parser.add_argument("--progress-every", type=int, default=10, help="Print progress every N target questions")
    parser.add_argument("--checkpoint-every", type=int, default=10, help="Write output/report every N processed target questions; 0 disables mid-run checkpoints")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload_data = _read_json(Path(args.input))
    payload = payload_data if isinstance(payload_data, dict) else {"questions": _benchmark_questions(payload_data)}
    output_path = Path(args.output)
    report_path = Path(args.report)

    resume_output_payload: dict[str, Any] | None = None
    resume_report: dict[str, Any] | None = None
    if args.resume:
        if output_path.exists():
            loaded_output = _read_json(output_path)
            if isinstance(loaded_output, dict):
                resume_output_payload = loaded_output
        if report_path.exists():
            loaded_report = _read_json(report_path)
            if isinstance(loaded_report, dict):
                resume_report = loaded_report

    def _log(message: str) -> None:
        print(message, flush=True)

    def _checkpoint(output_payload: dict[str, Any], report: dict[str, Any]) -> None:
        _write_checkpoint_files(
            payload=output_payload,
            report=report,
            input_path=Path(args.input),
            output_path=output_path,
            report_path=report_path,
            scannet_root=args.scannet_root,
            scannetpp_root=args.scannetpp_root,
            scannetpp_sensor=args.scannetpp_sensor,
            resume=bool(args.resume),
            checkpoint_every=max(args.checkpoint_every, 0),
            progress_every=max(args.progress_every, 1),
        )
        _log(
            f"[checkpoint] processed={report['processed_count']}/{report['target_count']} "
            f"fixed={report['fixed_count']} skipped={report['skipped_count']} "
            f"remaining={report['remaining_count']}"
        )

    fixed_payload, report = recompute_attachment_child_queries(
        payload,
        scannet_root=args.scannet_root,
        scannetpp_root=args.scannetpp_root,
        scannetpp_sensor=args.scannetpp_sensor,
        resume_output_payload=resume_output_payload,
        resume_report=resume_report,
        progress_every=max(args.progress_every, 1),
        checkpoint_every=max(args.checkpoint_every, 0),
        log=_log,
        checkpoint_callback=_checkpoint,
    )
    final_payload = _finalize_metadata(
        fixed_payload,
        report,
        input_path=Path(args.input),
        output_path=output_path,
        report_path=report_path,
        scannet_root=args.scannet_root,
        scannetpp_root=args.scannetpp_root,
        scannetpp_sensor=args.scannetpp_sensor,
        resume=bool(args.resume),
        checkpoint_every=max(args.checkpoint_every, 0),
        progress_every=max(args.progress_every, 1),
    )
    _write_json(output_path, final_payload)
    _write_json(report_path, report)

    print(f"target questions: {report['target_count']}")
    print(f"fixed questions : {report['fixed_count']}")
    print(f"skipped questions: {report['skipped_count']}")
    print(f"restored questions: {report['restored_count']}")
    print(f"output json     : {output_path}")
    print(f"report json     : {report_path}")


if __name__ == "__main__":
    main()
