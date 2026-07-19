#!/usr/bin/env python3
"""Filter L2 object_move_occlusion questions by original/counterfactual in-frame corner rules."""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets import make_data_source
from src.qa_generator import _bbox_fully_in_frame, _bbox_in_frame_corner_count
from src.support_graph import enrich_scene_with_attachment, get_scene_attachment_graph
from src.virtual_ops import apply_movement


DEFAULT_SCANNET_ROOT = Path("/home/lihongxing/datasets/ScanNet/data/scans")
DEFAULT_SCANNETPP_ROOT = Path("/home/sujinyue/datasets/scannetpp")
DEFAULT_INPUT = Path("output/benchmark.json")
DEFAULT_OUTPUT = Path("output/benchmark.object_move_occlusion_inframe_filtered.json")
DEFAULT_REPORT = Path("output/benchmark.object_move_occlusion_inframe_filtered_report.json")

SKIP_NOT_OCCLUSION = "not_object_move_occlusion"
DROP_INVALID_TARGET_ID = "invalid_target_obj_id"
DROP_INVALID_REFERENCE_ID = "invalid_obj_ref_id"
DROP_INVALID_MOVED_ID = "invalid_moved_obj_id"
DROP_INVALID_DELTA = "invalid_delta"
DROP_SCENE_NOT_FOUND = "scene_dir_not_found"
DROP_SCENE_LOAD_FAILED = "scene_load_failed"
DROP_POSE_MISSING = "pose_missing_for_frame"
DROP_INTRINSICS_MISSING = "intrinsics_missing"
DROP_OBJECT_MISSING = "target_or_moved_object_missing"
DROP_ORIGINAL_NOT_FULLY_IN_FRAME = "original_target_not_fully_in_frame"
DROP_REFERENCE_NOT_ENOUGH_IN_FRAME = "original_reference_not_enough_in_frame"
DROP_COUNTERFACTUAL_NOT_ENOUGH_IN_FRAME = "counterfactual_target_not_enough_in_frame"


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _benchmark_questions(payload: Any) -> list[dict[str, Any]]:
    questions = payload.get("questions", payload) if isinstance(payload, dict) else payload
    if not isinstance(questions, list):
        raise ValueError("Unsupported benchmark structure")
    return [question for question in questions if isinstance(question, dict)]


def _coerce_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_delta(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    try:
        return [float(value[0]), float(value[1]), float(value[2])]
    except (TypeError, ValueError):
        return None


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
    scannet_root: Path,
    scannetpp_root: Path,
    scannetpp_sensor: str,
) -> dict[str, Any]:
    root = scannetpp_root if dataset == "scannetpp" else scannet_root
    scene_dir = _resolve_scene_dir(root, dataset, scene_id)
    data_source = make_data_source(dataset, scene_dir, sensor=scannetpp_sensor)
    scene = data_source.load_scene()
    if scene is None:
        raise RuntimeError(f"Failed to parse scene {scene_id}")
    enrich_scene_with_attachment(scene)
    objects = [obj for obj in scene.get("objects", []) if isinstance(obj, dict)]
    obj_map = {int(obj["id"]): obj for obj in objects if _coerce_int(obj.get("id")) is not None}
    return {
        "dataset": dataset,
        "scene_dir": scene_dir,
        "objects": objects,
        "obj_map": obj_map,
        "attachment_graph": get_scene_attachment_graph(scene),
        "intrinsics": data_source.load_intrinsics(),
        "poses": data_source.load_poses(),
    }


def _drop_entry(
    question: dict[str, Any],
    *,
    reason: str,
    visible_corner_count: int | None = None,
    required_corner_count: int | None = None,
) -> dict[str, Any]:
    item = {
        "trace_question_id": question.get("trace_question_id"),
        "scene_id": question.get("scene_id"),
        "image_name": question.get("image_name"),
        "type": question.get("type"),
        "target_obj_id": question.get("target_obj_id"),
        "moved_obj_id": question.get("moved_obj_id"),
        "query_obj_id": question.get("query_obj_id"),
        "obj_ref_id": question.get("obj_ref_id"),
        "drop_reason": reason,
    }
    if visible_corner_count is not None:
        item["visible_corner_count"] = int(visible_corner_count)
    if required_corner_count is not None:
        item["required_corner_count"] = int(required_corner_count)
    return item


def _compute_statistics(questions: list[dict[str, Any]]) -> dict[str, Any]:
    by_level = Counter(str(question.get("level") or "unknown") for question in questions)
    by_type = Counter(str(question.get("type") or "unknown") for question in questions)

    statistics: dict[str, Any] = {
        "total": len(questions),
        "by_level": dict(sorted(by_level.items())),
        "by_type": dict(sorted(by_type.items())),
    }
    for level in sorted(by_level):
        answer_counter = Counter(str(question.get("answer") or "") for question in questions if str(question.get("level")) == level)
        answer_counter = Counter({key: value for key, value in answer_counter.items() if key})
        if not answer_counter:
            continue
        total = sum(answer_counter.values())
        statistics[f"{level}_answer_dist"] = {
            answer: round(count / total, 3)
            for answer, count in sorted(answer_counter.items())
        }
    return statistics


def filter_object_move_occlusion_questions(
    payload: dict[str, Any],
    *,
    scannet_root: Path,
    scannetpp_root: Path,
    scannetpp_sensor: str = "iphone",
    progress_every: int = 50,
) -> tuple[dict[str, Any], dict[str, Any]]:
    questions = _benchmark_questions(payload)
    scene_cache: dict[tuple[str, str], dict[str, Any]] = {}
    output_questions: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    processed_occlusion = 0
    kept_occlusion = 0
    error_count = 0
    started_at = time.perf_counter()

    occlusion_questions = [q for q in questions if str(q.get("type", "")).strip() == "object_move_occlusion"]
    total_occlusion = len(occlusion_questions)
    print(f"[start] object_move_occlusion={total_occlusion}, progress_every={max(progress_every, 1)}", flush=True)

    for question in questions:
        if str(question.get("type", "")).strip() != "object_move_occlusion":
            output_questions.append(question)
            continue

        processed_occlusion += 1
        scene_id = str(question.get("scene_id", "")).strip()
        image_name = str(question.get("image_name", "")).strip()
        semantics_v2 = str(question.get("occlusion_semantics_version", "")).strip() == "2"
        target_obj_id = _coerce_int(question.get("query_obj_id"))
        if target_obj_id is None:
            target_obj_id = _coerce_int(question.get("target_obj_id"))
        ref_obj_id = _coerce_int(question.get("obj_ref_id"))
        moved_obj_id = _coerce_int(question.get("moved_obj_id"))
        delta = _coerce_delta(question.get("delta"))

        if target_obj_id is None:
            dropped.append(_drop_entry(question, reason=DROP_INVALID_TARGET_ID))
            error_count += 1
            continue
        if semantics_v2 and ref_obj_id is None:
            dropped.append(_drop_entry(question, reason=DROP_INVALID_REFERENCE_ID))
            error_count += 1
            continue
        if moved_obj_id is None:
            dropped.append(_drop_entry(question, reason=DROP_INVALID_MOVED_ID))
            error_count += 1
            continue
        if delta is None:
            dropped.append(_drop_entry(question, reason=DROP_INVALID_DELTA))
            error_count += 1
            continue

        dataset = _infer_dataset(question)
        if dataset is None:
            dropped.append(_drop_entry(question, reason=DROP_SCENE_NOT_FOUND))
            error_count += 1
            continue
        cache_key = (dataset, scene_id)
        if cache_key not in scene_cache:
            try:
                print(f"[load] scene {scene_id} ({dataset})", flush=True)
                scene_cache[cache_key] = _load_scene_context(
                    dataset=dataset,
                    scene_id=scene_id,
                    scannet_root=scannet_root,
                    scannetpp_root=scannetpp_root,
                    scannetpp_sensor=scannetpp_sensor,
                )
            except FileNotFoundError:
                dropped.append(_drop_entry(question, reason=DROP_SCENE_NOT_FOUND))
                error_count += 1
                continue
            except Exception:
                dropped.append(_drop_entry(question, reason=DROP_SCENE_LOAD_FAILED))
                error_count += 1
                continue

        ctx = scene_cache[cache_key]
        intrinsics = ctx.get("intrinsics")
        camera_pose = ctx["poses"].get(image_name)
        target_obj = ctx["obj_map"].get(target_obj_id)
        ref_obj = ctx["obj_map"].get(ref_obj_id) if ref_obj_id is not None else None
        if intrinsics is None:
            dropped.append(_drop_entry(question, reason=DROP_INTRINSICS_MISSING))
            error_count += 1
            continue
        if camera_pose is None:
            dropped.append(_drop_entry(question, reason=DROP_POSE_MISSING))
            error_count += 1
            continue
        if target_obj is None or moved_obj_id not in ctx["obj_map"] or (semantics_v2 and ref_obj is None):
            dropped.append(_drop_entry(question, reason=DROP_OBJECT_MISSING))
            error_count += 1
            continue

        original_visible, original_total = _bbox_in_frame_corner_count(target_obj, camera_pose, intrinsics)
        target_original_ok = (
            _bbox_fully_in_frame(target_obj, camera_pose, intrinsics)
            if not semantics_v2
            else original_visible >= 6
        )
        if not target_original_ok:
            dropped.append(
                _drop_entry(
                    question,
                    reason=DROP_ORIGINAL_NOT_FULLY_IN_FRAME,
                    visible_corner_count=original_visible,
                    required_corner_count=6 if semantics_v2 else 8,
                )
            )
            continue

        if semantics_v2:
            ref_visible, ref_total = _bbox_in_frame_corner_count(ref_obj, camera_pose, intrinsics)
            if ref_visible < 6:
                dropped.append(
                    _drop_entry(
                        question,
                        reason=DROP_REFERENCE_NOT_ENOUGH_IN_FRAME,
                        visible_corner_count=ref_visible,
                        required_corner_count=6,
                    )
                )
                continue

        moved_objects = apply_movement(ctx["objects"], ctx["attachment_graph"], moved_obj_id, delta)
        moved_map = {int(obj["id"]): obj for obj in moved_objects}
        moved_target = moved_map.get(target_obj_id)
        if moved_target is None:
            dropped.append(_drop_entry(question, reason=DROP_OBJECT_MISSING))
            error_count += 1
            continue

        visible_corners, total_corners = _bbox_in_frame_corner_count(moved_target, camera_pose, intrinsics)
        if visible_corners < 6:
            dropped.append(
                _drop_entry(
                    question,
                    reason=DROP_COUNTERFACTUAL_NOT_ENOUGH_IN_FRAME,
                    visible_corner_count=visible_corners,
                    required_corner_count=6,
                )
            )
            continue

        output_questions.append(copy.deepcopy(question))
        kept_occlusion += 1

        if processed_occlusion % max(progress_every, 1) == 0 or processed_occlusion == total_occlusion:
            elapsed = time.perf_counter() - started_at
            print(
                f"[progress] {processed_occlusion}/{total_occlusion} done, "
                f"kept={kept_occlusion}, dropped={len(dropped)}, errors={error_count}, elapsed={elapsed:.1f}s",
                flush=True,
            )

    output_payload = dict(payload)
    output_payload["questions"] = output_questions
    output_payload["statistics"] = _compute_statistics(output_questions)
    report = {
        "total_occlusion_questions": total_occlusion,
        "kept_count": kept_occlusion,
        "dropped_count": len(dropped),
        "error_count": error_count,
        "drop_reason_counts": {
            reason: sum(1 for item in dropped if item["drop_reason"] == reason)
            for reason in sorted({item["drop_reason"] for item in dropped})
        },
        "dropped": dropped,
    }
    return output_payload, report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter object_move_occlusion questions that fail in-frame target rules.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--scannet-root", type=Path, default=DEFAULT_SCANNET_ROOT)
    parser.add_argument("--scannetpp-root", type=Path, default=DEFAULT_SCANNETPP_ROOT)
    parser.add_argument("--scannetpp-sensor", choices=("iphone", "dslr"), default="iphone")
    parser.add_argument("--progress-every", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload_data = _read_json(args.input)
    payload = payload_data if isinstance(payload_data, dict) else {"questions": _benchmark_questions(payload_data)}
    filtered_payload, report = filter_object_move_occlusion_questions(
        payload,
        scannet_root=args.scannet_root,
        scannetpp_root=args.scannetpp_root,
        scannetpp_sensor=args.scannetpp_sensor,
        progress_every=max(args.progress_every, 1),
    )

    metadata = dict(filtered_payload.get("metadata", {}))
    postprocess = dict(metadata.get("postprocess", {})) if isinstance(metadata.get("postprocess"), dict) else {}
    postprocess["object_move_occlusion_inframe_filter"] = {
        "input_path": str(args.input),
        "output_path": str(args.output),
        "report_path": str(args.report),
        "scannet_root": str(args.scannet_root),
        "scannetpp_root": str(args.scannetpp_root),
        "scannetpp_sensor": args.scannetpp_sensor,
        "original_required_corner_count": 8,
        "counterfactual_required_corner_count": 6,
        "kept_count": report["kept_count"],
        "dropped_count": report["dropped_count"],
        "error_count": report["error_count"],
    }
    metadata["postprocess"] = postprocess
    filtered_payload["metadata"] = metadata

    _write_json(args.output, filtered_payload)
    _write_json(args.report, report)

    print(f"total occlusion : {report['total_occlusion_questions']}")
    print(f"kept questions  : {report['kept_count']}")
    print(f"dropped questions: {report['dropped_count']}")
    print(f"errors          : {report['error_count']}")
    print(f"output json     : {args.output}")
    print(f"report json     : {args.report}")


if __name__ == "__main__":
    main()
