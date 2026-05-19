#!/usr/bin/env python3
"""Diagnose where edited-pair L2 candidates disappear after possibility checks.

Inputs:
  * edited_pair_generation_debug.json from diagnose_edited_pair_generation.py
  * edited_pair_l2_skip_reasons.json from diagnose_edited_pair_l2_reasons.py

This script does not rerun geometry.  It compares L2 "candidate_possible" flags
with the questions actually emitted into frame_debug.generated_questions, the
scene raw cache, and benchmark.json.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


OBJECT_ID_FIELDS = (
    "obj_a_id",
    "obj_b_id",
    "obj_c_id",
    "obj_ref_id",
    "obj_face_id",
    "obj_target_id",
    "target_obj_id",
    "query_obj_id",
    "moved_obj_id",
    "removed_obj_id",
    "parent_id",
    "child_id",
    "grandparent_id",
    "grandchild_id",
    "neighbor_id",
    "attachment_parent_id",
    "attachment_child_id",
)


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _question_object_ids(question: dict[str, Any]) -> set[int]:
    ids: set[int] = set()
    for field in OBJECT_ID_FIELDS:
        obj_id = _as_int(question.get(field))
        if obj_id is not None:
            ids.add(obj_id)
    for mention in question.get("mentioned_objects", []) or []:
        if isinstance(mention, dict):
            obj_id = _as_int(mention.get("object_id"))
            if obj_id is None:
                obj_id = _as_int(mention.get("id"))
            if obj_id is None:
                obj_id = _as_int(mention.get("obj_id"))
            if obj_id is not None:
                ids.add(obj_id)
    return ids


def _question_uses_attachment_referability(question: dict[str, Any]) -> bool:
    qtype = str(question.get("type", "")).strip().lower()
    return (
        qtype == "attachment_chain"
        or qtype.startswith("attachment")
        or bool(question.get("attachment_remapped", False))
    )


def _load_frame_debug(pilot_root: Path) -> dict[tuple[str, str], dict[str, Any]]:
    out: dict[tuple[str, str], dict[str, Any]] = {}
    frame_debug_dir = pilot_root / "frame_debug"
    if not frame_debug_dir.exists():
        return out
    for path in sorted(frame_debug_dir.glob("*.json")):
        try:
            doc = _read_json(path)
        except Exception:
            continue
        if not isinstance(doc, dict):
            continue
        scene_id = str(doc.get("scene_id") or path.stem)
        for frame in doc.get("frames", []) or []:
            if not isinstance(frame, dict):
                continue
            image_name = str(frame.get("image_name", "")).strip()
            if image_name:
                out[(scene_id, image_name)] = frame
    return out


def _load_raw_questions(pilot_root: Path) -> dict[str, list[dict[str, Any]]]:
    raw_dir = pilot_root / "_raw_questions_scene_cache"
    out: dict[str, list[dict[str, Any]]] = {}
    if not raw_dir.exists():
        return out
    for path in sorted(raw_dir.glob("*.json")):
        try:
            data = _read_json(path)
        except Exception:
            continue
        if isinstance(data, list):
            out[path.stem] = [item for item in data if isinstance(item, dict)]
    return out


def _load_benchmark_questions(pilot_root: Path) -> list[dict[str, Any]]:
    path = pilot_root / "benchmark.json"
    if not path.exists():
        return []
    data = _read_json(path)
    if isinstance(data, dict) and isinstance(data.get("questions"), list):
        return [item for item in data["questions"] if isinstance(item, dict)]
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    return []


def _filter_frame_questions(
    questions: list[dict[str, Any]],
    *,
    scene_id: str,
    image_name: str,
) -> list[dict[str, Any]]:
    return [
        q for q in questions
        if str(q.get("scene_id", scene_id)) == scene_id
        and str(q.get("image_name", image_name)) == image_name
    ]


def _summarize_questions(
    questions: list[dict[str, Any]],
    *,
    parent_id: int,
    child_id: int,
    limit: int,
) -> dict[str, Any]:
    parent_hits: list[dict[str, Any]] = []
    child_hits: list[dict[str, Any]] = []
    pair_hits: list[dict[str, Any]] = []
    attachment_hits: list[dict[str, Any]] = []
    type_counts: Counter[str] = Counter()
    child_type_counts: Counter[str] = Counter()
    for question in questions:
        ids = _question_object_ids(question)
        qtype = str(question.get("type", ""))
        type_counts[qtype] += 1
        if parent_id in ids:
            parent_hits.append(question)
        if child_id in ids:
            child_hits.append(question)
            child_type_counts[qtype] += 1
        if parent_id in ids and child_id in ids:
            pair_hits.append(question)
        if _question_uses_attachment_referability(question) and (parent_id in ids or child_id in ids):
            attachment_hits.append(question)

    def _preview(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        previews: list[dict[str, Any]] = []
        for question in items[:limit]:
            previews.append({
                "type": question.get("type"),
                "question": question.get("question"),
                "object_ids": sorted(_question_object_ids(question)),
                "attachment_remapped": bool(question.get("attachment_remapped", False)),
                "trace_source": question.get("_trace_source"),
                "trace_reason": question.get("trace_reason"),
            })
        return previews

    return {
        "total": len(questions),
        "type_counts": dict(sorted(type_counts.items())),
        "parent_hits": len(parent_hits),
        "child_hits": len(child_hits),
        "pair_hits": len(pair_hits),
        "attachment_related_parent_or_child_hits": len(attachment_hits),
        "child_type_counts": dict(sorted(child_type_counts.items())),
        "child_examples": _preview(child_hits),
        "pair_examples": _preview(pair_hits),
        "attachment_examples": _preview(attachment_hits),
    }


def _candidate_flags(l2_item: dict[str, Any] | None) -> dict[str, bool]:
    if not isinstance(l2_item, dict):
        return {"direction": False, "distance": False, "occlusion": False}
    return {
        "direction": bool((l2_item.get("direction") or {}).get("candidate_possible"))
        if isinstance(l2_item.get("direction"), dict) else False,
        "distance": bool((l2_item.get("distance") or {}).get("candidate_possible"))
        if isinstance(l2_item.get("distance"), dict) else False,
        "occlusion": bool((l2_item.get("occlusion") or {}).get("candidate_possible"))
        if isinstance(l2_item.get("occlusion"), dict) else False,
    }


def _classify_drop(
    *,
    candidate_flags: dict[str, bool],
    frame_generated: dict[str, Any],
    raw_summary: dict[str, Any],
    benchmark_summary: dict[str, Any],
    raw_scene_type_counts: Counter[str],
    max_questions_per_scene_type: int,
) -> str:
    if not any(candidate_flags.values()):
        return "no_l2_candidate_possible"
    if frame_generated["child_hits"] <= 0:
        if candidate_flags.get("occlusion") and not candidate_flags.get("direction") and not candidate_flags.get("distance"):
            return "candidate_possible_but_generator_emitted_no_child_question_possible_occlusion_selector_or_input_mismatch"
        return "candidate_possible_but_generator_emitted_no_child_question"
    if raw_summary["child_hits"] <= 0:
        generated_child_types = set(frame_generated.get("child_type_counts", {}).keys())
        capped_types = [
            qtype for qtype in generated_child_types
            if raw_scene_type_counts.get(qtype, 0) >= max_questions_per_scene_type
        ]
        if capped_types:
            return "frame_generated_child_question_dropped_by_scene_type_cap"
        return "frame_generated_child_question_dropped_before_raw_cache_dedup_or_unknown"
    if benchmark_summary["child_hits"] <= 0:
        return "raw_child_question_dropped_after_raw_cache_quality_or_viewer_filter"
    return "child_question_reaches_benchmark"


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    pair_report = _read_json(args.pair_report)
    l2_report = _read_json(args.l2_report)
    pilot_root = Path(args.pilot_root or pair_report.get("pilot_root") or args.pair_report.parent)
    frame_debug_by_key = _load_frame_debug(pilot_root)
    raw_by_scene = _load_raw_questions(pilot_root)
    benchmark_questions = _load_benchmark_questions(pilot_root)

    l2_by_key: dict[tuple[str, str, int, int], dict[str, Any]] = {}
    for item in l2_report.get("pairs", []) if isinstance(l2_report, dict) else []:
        if not isinstance(item, dict):
            continue
        pair = item.get("pair")
        if not isinstance(pair, list) or len(pair) < 2:
            continue
        key = (
            str(item.get("scene_id", "")),
            str(item.get("image_name", "")),
            int(pair[0]),
            int(pair[1]),
        )
        l2_by_key[key] = item

    reason_counts: Counter[str] = Counter()
    results: list[dict[str, Any]] = []
    for item in pair_report.get("pairs", []) if isinstance(pair_report, dict) else []:
        if not isinstance(item, dict):
            continue
        diagnosis = str(item.get("diagnosis", ""))
        if args.diagnosis and diagnosis not in set(args.diagnosis):
            continue
        pair = item.get("pair")
        if not isinstance(pair, list) or len(pair) < 2:
            continue
        scene_id = str(item.get("scene_id", ""))
        image_name = str(item.get("image_name", ""))
        parent_id = int(pair[0])
        child_id = int(pair[1])
        key = (scene_id, image_name, parent_id, child_id)
        l2_item = l2_by_key.get(key)
        flags = _candidate_flags(l2_item)

        frame_debug = frame_debug_by_key.get((scene_id, image_name), {})
        generated_questions = [
            q for q in frame_debug.get("generated_questions", []) or []
            if isinstance(q, dict)
        ] if isinstance(frame_debug, dict) else []
        final_questions = [
            q for q in frame_debug.get("final_questions", []) or []
            if isinstance(q, dict)
        ] if isinstance(frame_debug, dict) else []
        raw_scene_questions = raw_by_scene.get(scene_id, [])
        raw_frame_questions = _filter_frame_questions(
            raw_scene_questions,
            scene_id=scene_id,
            image_name=image_name,
        )
        benchmark_frame_questions = _filter_frame_questions(
            benchmark_questions,
            scene_id=scene_id,
            image_name=image_name,
        )
        raw_scene_type_counts = Counter(str(q.get("type", "")) for q in raw_scene_questions)

        frame_generated_summary = _summarize_questions(
            generated_questions,
            parent_id=parent_id,
            child_id=child_id,
            limit=args.example_limit,
        )
        raw_summary = _summarize_questions(
            raw_frame_questions,
            parent_id=parent_id,
            child_id=child_id,
            limit=args.example_limit,
        )
        benchmark_summary = _summarize_questions(
            benchmark_frame_questions,
            parent_id=parent_id,
            child_id=child_id,
            limit=args.example_limit,
        )
        final_summary = _summarize_questions(
            final_questions,
            parent_id=parent_id,
            child_id=child_id,
            limit=args.example_limit,
        )
        drop_reason = _classify_drop(
            candidate_flags=flags,
            frame_generated=frame_generated_summary,
            raw_summary=raw_summary,
            benchmark_summary=benchmark_summary,
            raw_scene_type_counts=raw_scene_type_counts,
            max_questions_per_scene_type=int(args.max_questions_per_scene_type),
        )
        reason_counts[drop_reason] += 1

        results.append({
            "scene_id": scene_id,
            "image_name": image_name,
            "pair": [parent_id, child_id],
            "parent_surface_text": item.get("parent_surface_text"),
            "child_surface_text": item.get("child_surface_text"),
            "previous_pair_diagnosis": diagnosis,
            "l2_candidate_possible": flags,
            "drop_reason": drop_reason,
            "raw_scene_type_counts": dict(sorted(raw_scene_type_counts.items())),
            "max_questions_per_scene_type": int(args.max_questions_per_scene_type),
            "frame_generated_questions": frame_generated_summary,
            "raw_frame_questions": raw_summary,
            "frame_final_questions": final_summary,
            "benchmark_questions": benchmark_summary,
        })

    return {
        "pair_report": str(args.pair_report),
        "l2_report": str(args.l2_report),
        "pilot_root": str(pilot_root),
        "pair_count": len(results),
        "reason_counts": dict(sorted(reason_counts.items())),
        "pairs": results,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Locate where edited-pair child candidates disappear between L2 possibility and raw/benchmark outputs."
    )
    parser.add_argument("--pair_report", type=Path, required=True)
    parser.add_argument("--l2_report", type=Path, required=True)
    parser.add_argument("--pilot_root", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--diagnosis", action="append", default=[])
    parser.add_argument("--max_questions_per_scene_type", type=int, default=5)
    parser.add_argument("--example_limit", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_report(args)
    output = args.output or args.pair_report.with_name("edited_pair_raw_drop_debug.json")
    _write_json(output, report)
    print(f"Wrote {output}")
    print(f"diagnosed pairs: {report['pair_count']}")
    print("reason counts:")
    for reason, count in report["reason_counts"].items():
        print(f"  {reason}: {count}")


if __name__ == "__main__":
    main()
