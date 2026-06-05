#!/usr/bin/env python3
"""Build a balanced benchmark subset from the full benchmark pool."""

from __future__ import annotations

import argparse
import json
import zlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


LEVEL_ORDER = ["L1", "L2", "L3"]
QTYPE_ORDER = [
    "direction_agent",
    "occlusion",
    "distance",
    "direction_object_centric",
    "direction_allocentric",
    "object_move_agent",
    "object_move_distance",
    "object_move_occlusion",
    "object_move_object_centric",
    "object_rotate_object_centric",
    "object_move_allocentric",
    "object_remove",
    "attachment_chain",
    "attachment_move",
    "coordinate_rotation_agent",
    "coordinate_rotation_object_centric",
    "coordinate_rotation_allocentric",
]
ATTACHMENT_REQUIRED_TYPES = {
    "object_move_agent",
    "object_move_distance",
    "object_move_occlusion",
    "object_move_object_centric",
    "object_rotate_object_centric",
    "object_move_allocentric",
}
OBJECT_KEY_FIELDS = [
    "query_obj_id",
    "obj_a_id",
    "target_obj_id",
    "obj_target_id",
    "removed_obj_id",
    "obj_ref_id",
    "obj_face_id",
    "moved_obj_id",
    "parent_id",
    "root_id",
    "grandchild_id",
    "grandparent_id",
    "neighbor_id",
    "obj_b_id",
]


def _json_key(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _qtype_sort_key(qtype: str) -> tuple[int, str]:
    try:
        return (QTYPE_ORDER.index(qtype), qtype)
    except ValueError:
        return (len(QTYPE_ORDER), qtype)


def _level_sort_key(level: str) -> tuple[int, str]:
    try:
        return (LEVEL_ORDER.index(level), level)
    except ValueError:
        return (len(LEVEL_ORDER), level)


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


def _infer_dataset(source_path: Path) -> str:
    try:
        rel = source_path.relative_to(Path("output"))
        return rel.parts[0] if rel.parts else "unknown"
    except ValueError:
        return source_path.parent.name or "unknown"


def _load_benchmark(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    questions = data.get("questions", data) if isinstance(data, dict) else data
    if not isinstance(questions, list):
        raise ValueError(f"Unsupported benchmark structure: {path}")
    return [q for q in questions if isinstance(q, dict)]


def _normalize_question(question: dict[str, Any], source_path: Path, seed: int) -> dict[str, Any]:
    item = dict(question)
    item["_dataset"] = str(item.get("_dataset") or item.get("dataset") or _infer_dataset(source_path))
    item["_source_benchmark"] = str(source_path)
    item["scene_id"] = str(item.get("scene_id") or "unknown")
    item["image_name"] = str(item.get("image_name") or "unknown")
    item["level"] = str(item.get("level") or "unknown")
    item["type"] = str(item.get("type") or "unknown")
    attachment_flag = bool(item.get("has_attachment_chain")) or bool(item.get("attachment_remapped"))
    if attachment_flag:
        item["has_attachment_chain"] = True
        item["attachment_remapped"] = True
    item["question_uid"] = _question_uid(item)
    item["_rank"] = _stable_rank(seed, item["question_uid"])
    return item


def load_questions(paths: list[Path], *, seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    questions: list[dict[str, Any]] = []
    seen_uids: set[str] = set()
    duplicate_count = 0

    for path in paths:
        for raw in _load_benchmark(path):
            item = _normalize_question(raw, path, seed)
            uid = str(item["question_uid"])
            if uid in seen_uids:
                duplicate_count += 1
                continue
            seen_uids.add(uid)
            questions.append(item)

    metadata = {
        "source_files": [str(path) for path in paths],
        "source_file_count": len(paths),
        "input_question_count": len(questions),
        "duplicate_question_count": duplicate_count,
        "dedupe_rule": "dataset + scene_id + image_name + level + type + question + options + answer",
    }
    return questions, metadata


def _question_object_id(question: dict[str, Any]) -> str:
    for field in OBJECT_KEY_FIELDS:
        value = question.get(field)
        if value is not None:
            return str(value)
    return f"uid:{question['question_uid']}"


def _attachment_eligible(question: dict[str, Any]) -> bool:
    return bool(question.get("has_attachment_chain")) and bool(question.get("attachment_remapped"))


def _object_key(question: dict[str, Any]) -> str:
    return _json_key(
        {
            "dataset": question["_dataset"],
            "scene_id": question["scene_id"],
            "image_name": question["image_name"],
            "type": question["type"],
            "obj": _question_object_id(question),
        }
    )


def _frame_key(question: dict[str, Any]) -> str:
    return _json_key(
        {
            "dataset": question["_dataset"],
            "scene_id": question["scene_id"],
            "image_name": question["image_name"],
            "type": question["type"],
        }
    )


def _scene_key(question: dict[str, Any]) -> str:
    return _json_key(
        {
            "dataset": question["_dataset"],
            "scene_id": question["scene_id"],
            "type": question["type"],
        }
    )


def _select_scene_candidates(questions: list[dict[str, Any]], *, frame_cap: int, scene_cap: int) -> list[dict[str, Any]]:
    by_frame: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for question in sorted(questions, key=lambda q: (int(q["_rank"]), str(q["question_uid"]))):
        by_frame[_frame_key(question)].append(question)

    frame_keys = sorted(
        by_frame,
        key=lambda key: min(int(q["_rank"]) for q in by_frame[key]),
    )
    frame_positions = {key: 0 for key in frame_keys}
    frame_counts: Counter[str] = Counter()
    selected: list[dict[str, Any]] = []

    while len(selected) < scene_cap:
        progressed = False
        for frame_key in frame_keys:
            if frame_counts[frame_key] >= frame_cap:
                continue
            position = frame_positions[frame_key]
            frame_questions = by_frame[frame_key]
            if position >= len(frame_questions):
                continue
            selected.append(frame_questions[position])
            frame_positions[frame_key] += 1
            frame_counts[frame_key] += 1
            progressed = True
            if len(selected) >= scene_cap:
                break
        if not progressed:
            break

    return selected


def _build_type_pool(
    questions: list[dict[str, Any]],
    *,
    frame_cap: int,
    scene_cap: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_object: dict[str, dict[str, Any]] = {}
    for question in sorted(questions, key=lambda q: (int(q["_rank"]), str(q["question_uid"]))):
        by_object.setdefault(_object_key(question), question)
    object_unique = list(by_object.values())

    by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for question in object_unique:
        by_scene[_scene_key(question)].append(question)

    per_scene_selected: dict[str, list[dict[str, Any]]] = {}
    for scene_key, scene_questions in by_scene.items():
        per_scene_selected[scene_key] = _select_scene_candidates(
            scene_questions,
            frame_cap=frame_cap,
            scene_cap=scene_cap,
        )

    scene_keys = sorted(
        per_scene_selected,
        key=lambda key: min(int(q["_rank"]) for q in per_scene_selected[key]),
    )
    scene_positions = {key: 0 for key in scene_keys}
    ordered_pool: list[dict[str, Any]] = []
    while True:
        progressed = False
        for scene_key in scene_keys:
            position = scene_positions[scene_key]
            scene_questions = per_scene_selected[scene_key]
            if position >= len(scene_questions):
                continue
            ordered_pool.append(scene_questions[position])
            scene_positions[scene_key] += 1
            progressed = True
        if not progressed:
            break

    stats = {
        "available_after_filters": len(questions),
        "object_unique_count": len(object_unique),
        "feasible_after_caps": len(ordered_pool),
        "scene_count": len(per_scene_selected),
    }
    return ordered_pool, stats


def _round_robin_sample(
    pools_by_type: dict[str, list[dict[str, Any]]],
    *,
    target_total: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    positions = {qtype: 0 for qtype in pools_by_type}
    sampled: list[dict[str, Any]] = []
    sampled_counts: Counter[str] = Counter()
    qtypes = sorted(pools_by_type, key=_qtype_sort_key)

    while len(sampled) < target_total:
        progressed = False
        for qtype in qtypes:
            position = positions[qtype]
            pool = pools_by_type[qtype]
            if position >= len(pool):
                continue
            sampled.append(pool[position])
            positions[qtype] += 1
            sampled_counts[qtype] += 1
            progressed = True
            if len(sampled) >= target_total:
                break
        if not progressed:
            break

    return sampled, dict(sampled_counts)


def sample_questions(
    questions: list[dict[str, Any]],
    *,
    target_per_level: int,
    frame_cap: int,
    scene_cap: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_level_type: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    prefilter_counts: Counter[tuple[str, str]] = Counter()
    postfilter_counts: Counter[tuple[str, str]] = Counter()

    for question in questions:
        level = str(question["level"])
        qtype = str(question["type"])
        prefilter_counts[(level, qtype)] += 1
        if qtype in ATTACHMENT_REQUIRED_TYPES and not _attachment_eligible(question):
            continue
        postfilter_counts[(level, qtype)] += 1
        by_level_type[level][qtype].append(question)

    sampled: list[dict[str, Any]] = []
    sampling_stats: dict[str, Any] = {
        "target_per_level": target_per_level,
        "frame_cap_per_type": frame_cap,
        "scene_cap_per_type": scene_cap,
        "attachment_required_types": sorted(ATTACHMENT_REQUIRED_TYPES),
        "levels": {},
    }

    for level in sorted(by_level_type, key=_level_sort_key):
        pools_by_type: dict[str, list[dict[str, Any]]] = {}
        type_stats: dict[str, Any] = {}
        for qtype in sorted(by_level_type[level], key=_qtype_sort_key):
            pool, pool_stats = _build_type_pool(
                by_level_type[level][qtype],
                frame_cap=frame_cap,
                scene_cap=scene_cap,
            )
            pools_by_type[qtype] = pool
            type_stats[qtype] = {
                "available_raw": prefilter_counts[(level, qtype)],
                "available_after_attachment_filter": postfilter_counts[(level, qtype)],
                **pool_stats,
            }

        level_target = min(target_per_level, sum(len(pool) for pool in pools_by_type.values()))
        level_sampled, sampled_counts = _round_robin_sample(pools_by_type, target_total=level_target)
        sampled.extend(level_sampled)

        for qtype, count in sampled_counts.items():
            type_stats[qtype]["sampled"] = count
        for qtype in type_stats:
            type_stats[qtype].setdefault("sampled", 0)

        sampling_stats["levels"][level] = {
            "target": level_target,
            "sampled": len(level_sampled),
            "types": type_stats,
        }

    return sampled, sampling_stats


def resolve_input_paths(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        if any(ch in pattern for ch in "*?[]"):
            matched = sorted(Path().glob(pattern))
        else:
            candidate = Path(pattern)
            matched = [candidate] if candidate.exists() else []
        paths.extend(path for path in matched if path.is_file())
    unique_paths = sorted({path.resolve() for path in paths})
    return [Path(path) for path in unique_paths]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a balanced benchmark subset.")
    parser.add_argument(
        "--input",
        action="append",
        default=None,
        help="Input benchmark path or glob. Repeat to add more sources. Default: output/**/benchmark.json",
    )
    parser.add_argument("--output", default="output/benchmark_subset.json", help="Output benchmark JSON")
    parser.add_argument("--target_per_level", type=int, default=1166, help="Target questions per level")
    parser.add_argument("--frame_cap", type=int, default=2, help="Max questions per (dataset, scene, frame, type)")
    parser.add_argument("--scene_cap", type=int, default=8, help="Max questions per (dataset, scene, type)")
    parser.add_argument("--seed", type=int, default=20260605, help="Deterministic seed for tie-breaking")
    args = parser.parse_args()
    if args.target_per_level <= 0:
        parser.error("--target_per_level must be positive")
    if args.frame_cap <= 0:
        parser.error("--frame_cap must be positive")
    if args.scene_cap <= 0:
        parser.error("--scene_cap must be positive")
    return args


def main() -> None:
    args = parse_args()
    input_patterns = args.input or ["output/**/benchmark.json"]
    input_paths = resolve_input_paths(input_patterns)
    if not input_paths:
        raise SystemExit("No benchmark files matched the input pattern(s).")

    output_path = Path(args.output)
    questions, metadata = load_questions(input_paths, seed=args.seed)
    selected, sampling_stats = sample_questions(
        questions,
        target_per_level=args.target_per_level,
        frame_cap=args.frame_cap,
        scene_cap=args.scene_cap,
    )

    payload = {
        "metadata": {
            **metadata,
            "output_mode": "balanced_benchmark_subset",
            "input_patterns": input_patterns,
            "target_per_level": args.target_per_level,
            "frame_cap": args.frame_cap,
            "scene_cap": args.scene_cap,
            "seed": args.seed,
            "sampled_question_count": len(selected),
        },
        "sampling_stats": sampling_stats,
        "questions": selected,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    level_counts = Counter(str(q["level"]) for q in selected)
    print(f"input questions : {len(questions)}")
    print(f"sampled questions: {len(selected)}")
    print(f"level counts    : {dict(sorted(level_counts.items(), key=lambda item: _level_sort_key(item[0])))}")
    print(f"output json     : {output_path}")


if __name__ == "__main__":
    main()
