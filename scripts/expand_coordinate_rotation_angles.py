#!/usr/bin/env python3
"""Expand each L3 coordinate-rotation question into 90/180/270 variants.

Existing benchmarks contain almost exclusively 90-degree coordinate-rotation
questions: the near-duplicate filter (keyed on scene/frame/type/object-ids,
without the rotation angle) collapsed the 90/180/270 variants of each
(scene, frame, object-pair) down to one. The surviving 90-degree-only set has a
heavily skewed answer distribution (the new direction is a single fixed
permutation of the old one).

This post-processor takes the existing questions and, for every coordinate
rotation question, regenerates the 90/180/270 variants with EXACT ground truth
by reloading the original scene geometry (same code path as the generator) and
re-running apply_coordinate_rotation + the primary_direction_* functions. It
does NOT relabel via the discretised old->new permutation (that is lossy at bin
boundaries); it recomputes from continuous geometry, so the answers are correct.

Only the coordinate-rotation questions are touched. Every other question is
passed through unchanged. The three variants per source question reuse the
source question's template wording and metadata; only angle, answer, options,
correct_value and the direction bookkeeping change.
"""

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
    ALL_DIRECTIONS_ALLOCENTRIC,
    _direction_suppression_reason,
    _has_stable_object_centric_facing,
    _invert_direction,
    _load_templates,
    _mention,
    _object_bottom_hull_xy,
    _the,
    generate_options,
)
from src.relation_engine import (
    camera_cardinal_direction,
    compute_all_relations,
    primary_direction_allocentric,
    primary_direction_object_centric,
)
from src.scene_parser import EXCLUDED_LABELS
from src.support_graph import enrich_scene_with_attachment
from src.virtual_ops import apply_coordinate_rotation

DEFAULT_SCANNET_ROOT = Path("/home/lihongxing/datasets/ScanNet/data/scans")
DEFAULT_SCANNETPP_ROOT = Path("/home/sujinyue/datasets/scannetpp")

ROTATION_TYPES = {
    "coordinate_rotation_agent",
    "coordinate_rotation_object_centric",
    "coordinate_rotation_allocentric",
}
ROTATION_ANGLES = (90, 180, 270)

# Skip reasons (mirrors recompute_attachment_child_queries.py vocabulary).
SKIP_UNSUPPORTED_QTYPE = "unsupported_qtype"
SKIP_MISSING_SCENE_ROOT = "missing_scene_root"
SKIP_SCENE_NOT_FOUND = "scene_dir_not_found"
SKIP_SCENE_LOAD_FAILED = "scene_load_failed"
SKIP_POSE_MISSING = "pose_missing_for_frame"
SKIP_OBJECT_MISSING = "required_object_missing"
SKIP_RELATION_MISSING = "relation_missing"
SKIP_DIRECTION_SUPPRESSED = "direction_suppressed"
SKIP_DIRECTION_AMBIGUOUS = "direction_ambiguous"
SKIP_UNSTABLE_FACING = "unstable_object_centric_facing"
def _json_key(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _stable_rank(seed: int, key: str) -> int:
    return zlib.crc32(f"{seed}|{key}".encode("utf-8")) & 0xFFFFFFFF


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
    return [q for q in questions if isinstance(q, dict)]


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


def _resolve_scene_dir(root: Path, scene_id: str) -> Path:
    candidates = []
    if root.name == scene_id and root.is_dir():
        candidates.append(root)
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
    scene_dir = _resolve_scene_dir(root, scene_id)
    data_source = make_data_source(dataset, scene_dir, sensor=scannetpp_sensor)
    scene = data_source.load_scene()
    if scene is None:
        raise RuntimeError(f"Failed to parse scene {scene_id}")
    enrich_scene_with_attachment(scene)
    objects = [o for o in scene.get("objects", []) if isinstance(o, dict)]
    obj_map = {int(o["id"]): o for o in objects if _coerce_int(o.get("id")) is not None}
    poses = data_source.load_poses()
    return {"objects": objects, "obj_map": obj_map, "poses": poses}
def _relation_for_ordered_pair(
    relations: list[dict[str, Any]], obj_a_id: int, obj_b_id: int
) -> dict[str, Any] | None:
    for rel in relations:
        if _coerce_int(rel.get("obj_a_id")) == obj_a_id and _coerce_int(rel.get("obj_b_id")) == obj_b_id:
            return rel
    return None


_TEMPLATE_KEY_BY_TYPE = {
    "coordinate_rotation_agent": "L3_coordinate_rotation_agent",
    "coordinate_rotation_object_centric": "L3_coordinate_rotation_object_centric",
    "coordinate_rotation_allocentric": "L3_coordinate_rotation_allocentric",
}


def _render_kwargs(source: dict[str, Any], angle: int) -> dict[str, Any]:
    qtype = str(source.get("type", "")).strip()
    if qtype == "coordinate_rotation_agent":
        return {
            "angle": angle,
            "obj_a": _the(str(source.get("obj_a_label", "object"))),
            "obj_b": _the(str(source.get("obj_b_label", "object"))),
        }
    if qtype == "coordinate_rotation_object_centric":
        return {
            "angle": angle,
            "obj_ref": _the(str(source.get("obj_ref_label", "object"))),
            "obj_face": _the(str(source.get("obj_face_label", "object"))),
            "obj_target": _the(str(source.get("obj_target_label", "object"))),
        }
    if qtype == "coordinate_rotation_allocentric":
        return {
            "angle": angle,
            "camera_cardinal": str(source.get("camera_cardinal", "north")),
            "obj_a": _the(str(source.get("obj_a_label", "object"))),
            "obj_b": _the(str(source.get("obj_b_label", "object"))),
        }
    return {"angle": angle}


def _retemplate(
    source_question: dict[str, Any],
    angle: int,
    templates: dict[str, Any] | None = None,
) -> str:
    """Re-render the source question's text for a new angle.

    Preferred path: re-render with the CURRENT template family so expanded
    variants pick up any wording changes (e.g. "about the camera" instead of the
    legacy "around the room center"). The generator chooses a template at random
    per variant, so we deterministically use the first current template for the
    type -- wording is uniform across the family, only the angle differs.

    Fallback: if templates/labels are unavailable, substitute the angle token in
    the source text in place (preserves whatever wording the source already had).
    """
    qtype = str(source_question.get("type", "")).strip()
    if templates is not None:
        key = _TEMPLATE_KEY_BY_TYPE.get(qtype)
        tpl_list = list(templates.get(key, [])) if key else []
        if tpl_list:
            try:
                return tpl_list[0].format(**_render_kwargs(source_question, angle))
            except Exception:
                pass
    old_angle = _coerce_int(source_question.get("rotation_angle"))
    text = str(source_question.get("question", ""))
    if old_angle is None:
        return text
    needle = f"{old_angle} degrees"
    if needle in text:
        return text.replace(needle, f"{angle} degrees", 1)
    return text


def _build_variant(
    *,
    source: dict[str, Any],
    angle: int,
    new_dir: str,
    old_dir: str,
    answer_pool: list[str],
    seed: int | None,
    templates: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    variant = copy.deepcopy(source)
    options, answer = generate_options(new_dir, answer_pool)
    variant["question"] = _retemplate(source, angle, templates)
    variant["rotation_angle"] = angle
    variant["options"] = options
    variant["answer"] = answer
    variant["correct_value"] = new_dir
    variant["old_direction"] = old_dir
    variant["new_direction"] = new_dir
    variant["relation_unchanged"] = old_dir == new_dir
    variant["angle_expanded"] = True
    if extra:
        variant.update(extra)
    variant.pop("question_referability_audit", None)
    variant["question_uid"] = _question_uid(variant)
    if seed is not None:
        variant["_rank"] = _stable_rank(seed, variant["question_uid"])
    return variant
def _expand_agent(
    *, source: dict[str, Any], ctx: dict[str, Any], camera_pose: Any, seed: int | None,
    templates: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], str | None]:
    obj_a_id = _coerce_int(source.get("obj_a_id"))
    obj_b_id = _coerce_int(source.get("obj_b_id"))
    if obj_a_id is None or obj_b_id is None:
        return [], SKIP_OBJECT_MISSING
    obj_map = ctx["obj_map"]
    if obj_a_id not in obj_map or obj_b_id not in obj_map:
        return [], SKIP_OBJECT_MISSING

    objects = ctx["objects"]
    base_relations = compute_all_relations(objects, camera_pose, None, None)
    base_rel = _relation_for_ordered_pair(base_relations, obj_a_id, obj_b_id)
    if base_rel is None:
        return [], SKIP_RELATION_MISSING
    old_dir = _invert_direction(base_rel["direction_b_rel_a"])

    variants: list[dict[str, Any]] = []
    for angle in ROTATION_ANGLES:
        rotated = apply_coordinate_rotation(objects, float(-angle))
        rot_map = {int(o["id"]): o for o in rotated}
        new_relations = compute_all_relations(rotated, camera_pose, None, None)
        new_rel = _relation_for_ordered_pair(new_relations, obj_a_id, obj_b_id)
        if new_rel is None:
            continue
        new_dir = _invert_direction(new_rel["direction_b_rel_a"])
        obj_a_rot = rot_map.get(obj_a_id)
        obj_b_rot = rot_map.get(obj_b_id)
        if obj_a_rot is None or obj_b_rot is None:
            continue
        if _direction_suppression_reason(obj_a_rot, obj_b_rot, new_dir, None) is not None:
            continue
        variants.append(
            _build_variant(
                source=source,
                angle=angle,
                new_dir=new_dir,
                old_dir=old_dir,
                answer_pool=ALL_DIRECTIONS,
                seed=seed,
                templates=templates,
            )
        )
    if not variants:
        return [], SKIP_DIRECTION_SUPPRESSED
    return variants, None
def _expand_object_centric(
    *, source: dict[str, Any], ctx: dict[str, Any], seed: int | None,
    templates: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], str | None]:
    ref_id = _coerce_int(source.get("obj_ref_id"))
    face_id = _coerce_int(source.get("obj_face_id"))
    target_id = _coerce_int(source.get("obj_target_id"))
    if ref_id is None or face_id is None or target_id is None:
        return [], SKIP_OBJECT_MISSING
    obj_map = ctx["obj_map"]
    ref = obj_map.get(ref_id)
    face = obj_map.get(face_id)
    target = obj_map.get(target_id)
    if ref is None or face is None or target is None:
        return [], SKIP_OBJECT_MISSING

    ref_c = np.array(ref["center"], dtype=float)
    face_c = np.array(face["center"], dtype=float)
    target_c = np.array(target["center"], dtype=float)
    original_heading = face_c - ref_c
    original_heading[2] = 0.0
    if float(np.linalg.norm(original_heading[:2])) < 1e-6:
        return [], SKIP_UNSTABLE_FACING
    if not _has_stable_object_centric_facing(ref_c, face_c):
        return [], SKIP_UNSTABLE_FACING

    old_dir, _ = primary_direction_object_centric(
        ref_c,
        face_c,
        target_c,
        anchor_hull_xy=_object_bottom_hull_xy(ref),
        target_hull_xy=_object_bottom_hull_xy(target),
        anchor_bbox_min=np.array(ref["bbox_min"], dtype=float),
        anchor_bbox_max=np.array(ref["bbox_max"], dtype=float),
        target_bbox_min=np.array(target["bbox_min"], dtype=float),
        target_bbox_max=np.array(target["bbox_max"], dtype=float),
    )

    objects = ctx["objects"]
    variants: list[dict[str, Any]] = []
    for angle in ROTATION_ANGLES:
        rotated = apply_coordinate_rotation(objects, float(-angle))
        rot_map = {int(o["id"]): o for o in rotated}
        ref_rot = rot_map.get(ref_id)
        target_rot = rot_map.get(target_id)
        if ref_rot is None or target_rot is None:
            continue
        ref_rot_c = np.array(ref_rot["center"], dtype=float)
        fixed_facing_c = ref_rot_c + original_heading
        target_rot_c = np.array(target_rot["center"], dtype=float)
        if not _has_stable_object_centric_facing(ref_rot_c, fixed_facing_c):
            continue
        new_dir, amb = primary_direction_object_centric(
            ref_rot_c,
            fixed_facing_c,
            target_rot_c,
            anchor_hull_xy=_object_bottom_hull_xy(ref_rot),
            target_hull_xy=_object_bottom_hull_xy(target_rot),
            anchor_bbox_min=np.array(ref_rot["bbox_min"], dtype=float),
            anchor_bbox_max=np.array(ref_rot["bbox_max"], dtype=float),
            target_bbox_min=np.array(target_rot["bbox_min"], dtype=float),
            target_bbox_max=np.array(target_rot["bbox_max"], dtype=float),
        )
        if amb > 0.7:
            continue
        if _direction_suppression_reason(ref_rot, target_rot, new_dir, None) is not None:
            continue
        variants.append(
            _build_variant(
                source=source,
                angle=angle,
                new_dir=new_dir,
                old_dir=old_dir,
                answer_pool=ALL_DIRECTIONS,
                seed=seed,
                templates=templates,
                extra={
                    "facing_anchor_center": ref_rot_c.tolist(),
                    "facing_target_center": fixed_facing_c.tolist(),
                },
            )
        )
    if not variants:
        return [], SKIP_DIRECTION_AMBIGUOUS
    return variants, None
def _expand_allocentric(
    *, source: dict[str, Any], ctx: dict[str, Any], seed: int | None,
    templates: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], str | None]:
    a_id = _coerce_int(source.get("obj_a_id"))
    b_id = _coerce_int(source.get("obj_b_id"))
    if a_id is None or b_id is None:
        return [], SKIP_OBJECT_MISSING
    obj_map = ctx["obj_map"]
    a = obj_map.get(a_id)
    b = obj_map.get(b_id)
    if a is None or b is None:
        return [], SKIP_OBJECT_MISSING

    old_dir, _ = primary_direction_allocentric(
        np.array(a["center"]),
        np.array(b["center"]),
        obj_a_hull_xy=_object_bottom_hull_xy(a),
        obj_b_hull_xy=_object_bottom_hull_xy(b),
        obj_a_bbox_min=np.array(a["bbox_min"], dtype=float),
        obj_a_bbox_max=np.array(a["bbox_max"], dtype=float),
        obj_b_bbox_min=np.array(b["bbox_min"], dtype=float),
        obj_b_bbox_max=np.array(b["bbox_max"], dtype=float),
    )

    objects = ctx["objects"]
    variants: list[dict[str, Any]] = []
    for angle in ROTATION_ANGLES:
        rotated = apply_coordinate_rotation(objects, float(-angle))
        rot_map = {int(o["id"]): o for o in rotated}
        a_rot = rot_map.get(a_id)
        b_rot = rot_map.get(b_id)
        if a_rot is None or b_rot is None:
            continue
        new_dir, amb = primary_direction_allocentric(
            np.array(a_rot["center"]),
            np.array(b_rot["center"]),
            obj_a_hull_xy=_object_bottom_hull_xy(a_rot),
            obj_b_hull_xy=_object_bottom_hull_xy(b_rot),
            obj_a_bbox_min=np.array(a_rot["bbox_min"], dtype=float),
            obj_a_bbox_max=np.array(a_rot["bbox_max"], dtype=float),
            obj_b_bbox_min=np.array(b_rot["bbox_min"], dtype=float),
            obj_b_bbox_max=np.array(b_rot["bbox_max"], dtype=float),
        )
        if amb > 0.7:
            continue
        if new_dir not in ALL_DIRECTIONS_ALLOCENTRIC:
            continue
        if _direction_suppression_reason(a_rot, b_rot, new_dir, None) is not None:
            continue
        variants.append(
            _build_variant(
                source=source,
                angle=angle,
                new_dir=new_dir,
                old_dir=old_dir,
                answer_pool=ALL_DIRECTIONS_ALLOCENTRIC,
                seed=seed,
                templates=templates,
            )
        )
    if not variants:
        return [], SKIP_DIRECTION_AMBIGUOUS
    return variants, None
def _expand_one(
    *, source: dict[str, Any], ctx: dict[str, Any], camera_pose: Any, seed: int | None,
    templates: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], str | None]:
    qtype = str(source.get("type", "")).strip()
    if qtype == "coordinate_rotation_agent":
        return _expand_agent(source=source, ctx=ctx, camera_pose=camera_pose, seed=seed, templates=templates)
    if qtype == "coordinate_rotation_object_centric":
        return _expand_object_centric(source=source, ctx=ctx, seed=seed, templates=templates)
    if qtype == "coordinate_rotation_allocentric":
        return _expand_allocentric(source=source, ctx=ctx, seed=seed, templates=templates)
    return [], SKIP_UNSUPPORTED_QTYPE


def expand_coordinate_rotation_angles(
    payload: dict[str, Any],
    *,
    scannet_root: Path | None,
    scannetpp_root: Path | None,
    scannetpp_sensor: str = "iphone",
    drop_unexpanded: bool = True,
    log: Callable[[str], None] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    questions = _benchmark_questions(payload)
    seed = _coerce_seed(payload.get("metadata", {}).get("seed"))
    templates = _load_templates()
    scene_cache: dict[tuple[str, str], dict[str, Any]] = {}

    output_questions: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    target_count = 0
    expanded_source_count = 0
    emitted_count = 0
    dropped_count = 0
    angle_counter: dict[int, int] = {a: 0 for a in ROTATION_ANGLES}

    def _handle_skip(question: dict[str, Any], reason: str) -> None:
        nonlocal dropped_count
        skipped.append({**_skip_stub(question), "skip_reason": reason})
        if drop_unexpanded:
            dropped_count += 1
        else:
            output_questions.append(question)

    total_targets = sum(1 for q in questions if str(q.get("type", "")) in ROTATION_TYPES)
    if log is not None:
        log(
            f"[start] total_questions={len(questions)} target_rotation_questions={total_targets} "
            f"drop_unexpanded={drop_unexpanded}"
        )

    for question in questions:
        qtype = str(question.get("type", "")).strip()
        if qtype not in ROTATION_TYPES:
            output_questions.append(question)
            continue

        target_count += 1
        dataset = _infer_dataset(question)
        scene_id = str(question.get("scene_id", "")).strip()
        image_name = str(question.get("image_name", "")).strip()
        if dataset is None or not scene_id:
            _handle_skip(question, SKIP_MISSING_SCENE_ROOT)
            continue

        cache_key = (dataset, scene_id)
        if cache_key not in scene_cache:
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
                    log(f"[scene] loaded {dataset}:{scene_id} objects={len(ctx['objects'])} poses={len(ctx['poses'])}")
            except FileNotFoundError as exc:
                reason = SKIP_MISSING_SCENE_ROOT if str(exc) == SKIP_MISSING_SCENE_ROOT else SKIP_SCENE_NOT_FOUND
                scene_cache[cache_key] = {}
                if log is not None:
                    log(f"[scene] {reason} {dataset}:{scene_id}")
            except Exception:
                scene_cache[cache_key] = {}
                if log is not None:
                    log(f"[scene] {SKIP_SCENE_LOAD_FAILED} {dataset}:{scene_id}")

        ctx = scene_cache[cache_key]
        if not ctx:
            _handle_skip(question, SKIP_SCENE_NOT_FOUND)
            continue
        camera_pose = ctx["poses"].get(image_name)
        if camera_pose is None:
            _handle_skip(question, SKIP_POSE_MISSING)
            continue

        variants, reason = _expand_one(source=question, ctx=ctx, camera_pose=camera_pose, seed=seed, templates=templates)
        if not variants:
            _handle_skip(question, reason or SKIP_SCENE_LOAD_FAILED)
            continue

        output_questions.extend(variants)
        expanded_source_count += 1
        emitted_count += len(variants)
        for v in variants:
            angle_counter[int(v["rotation_angle"])] += 1

    output_payload = dict(payload)
    output_payload["questions"] = output_questions
    report = {
        "target_count": target_count,
        "expanded_source_count": expanded_source_count,
        "emitted_variant_count": emitted_count,
        "skipped_count": len(skipped),
        "dropped_count": dropped_count,
        "drop_unexpanded": drop_unexpanded,
        "angle_distribution": angle_counter,
        "skipped": skipped,
    }
    if log is not None:
        log(
            f"[done] targets={target_count} expanded={expanded_source_count} "
            f"emitted={emitted_count} skipped={len(skipped)} dropped={dropped_count} angles={angle_counter}"
        )
    return output_payload, report


def _skip_stub(question: dict[str, Any]) -> dict[str, Any]:
    return {
        "scene_id": question.get("scene_id"),
        "image_name": question.get("image_name"),
        "type": question.get("type"),
        "trace_question_id": question.get("trace_question_id"),
        "obj_a_id": question.get("obj_a_id"),
        "obj_b_id": question.get("obj_b_id"),
        "obj_ref_id": question.get("obj_ref_id"),
        "obj_target_id": question.get("obj_target_id"),
    }
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Expand each L3 coordinate-rotation question into 90/180/270 variants with exact GT.",
    )
    parser.add_argument("--input", default="output/benchmark_subset.json", help="Input benchmark JSON")
    parser.add_argument(
        "--output",
        default="output/benchmark_subset.rotation_expanded.json",
        help="Output benchmark JSON",
    )
    parser.add_argument(
        "--report",
        default="output/benchmark_subset.rotation_expanded_report.json",
        help="Output JSON report",
    )
    parser.add_argument("--scannet-root", type=Path, default=DEFAULT_SCANNET_ROOT)
    parser.add_argument("--scannetpp-root", type=Path, default=DEFAULT_SCANNETPP_ROOT)
    parser.add_argument("--scannetpp-sensor", choices=("iphone", "dslr"), default="iphone")
    parser.add_argument(
        "--keep-unexpanded",
        action="store_true",
        help="Keep source rotation questions that could not be expanded into all angles "
        "(default: drop them so the output distribution stays clean).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload_data = _read_json(Path(args.input))
    payload = payload_data if isinstance(payload_data, dict) else {"questions": _benchmark_questions(payload_data)}

    def _log(message: str) -> None:
        print(message, flush=True)

    output_payload, report = expand_coordinate_rotation_angles(
        payload,
        scannet_root=args.scannet_root,
        scannetpp_root=args.scannetpp_root,
        scannetpp_sensor=args.scannetpp_sensor,
        drop_unexpanded=not args.keep_unexpanded,
        log=_log,
    )

    metadata = dict(output_payload.get("metadata", {}))
    postprocess = dict(metadata.get("postprocess", {})) if isinstance(metadata.get("postprocess"), dict) else {}
    postprocess["coordinate_rotation_angle_expansion"] = {
        "input_path": str(args.input),
        "output_path": str(args.output),
        "report_path": str(args.report),
        "target_count": report["target_count"],
        "expanded_source_count": report["expanded_source_count"],
        "emitted_variant_count": report["emitted_variant_count"],
        "skipped_count": report["skipped_count"],
        "dropped_count": report["dropped_count"],
        "drop_unexpanded": report["drop_unexpanded"],
        "angle_distribution": report["angle_distribution"],
    }
    metadata["postprocess"] = postprocess
    output_payload["metadata"] = metadata

    _write_json(Path(args.output), output_payload)
    _write_json(Path(args.report), report)

    print(f"target rotation questions: {report['target_count']}")
    print(f"expanded source questions: {report['expanded_source_count']}")
    print(f"emitted variant questions: {report['emitted_variant_count']}")
    print(f"skipped questions        : {report['skipped_count']}")
    print(f"dropped questions        : {report['dropped_count']}")
    print(f"angle distribution       : {report['angle_distribution']}")
    print(f"output json              : {args.output}")
    print(f"report json              : {args.report}")


if __name__ == "__main__":
    main()

