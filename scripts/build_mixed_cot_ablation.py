#!/usr/bin/env python3
"""Build the matched three-target curriculum dataset for CoT ablations."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cot.evaluation import parse_strict_answer
from src.cot.curriculum import (
    FRESH_12K_PROFILE,
    FRESH_12K_PROFILE_ID,
    LEGACY_PROFILE,
    LEGACY_PROFILE_ID,
    PROFILES,
    CurriculumProfile,
    get_curriculum_profile,
    l2_rarity_group,
    sqrt_largest_remainder_allocation,
    stable_hash,
)
from src.cot.facts import build_fact_record, question_uid
from src.cot.images import resolve_image_paths
from src.cot.models import FactExtractionError
from src.cot.pipeline import format_user_prompt
from src.cot.render import render_response
from src.cot.sampling import TYPES_BY_LEVEL, select_stratified
from src.cot.templates import load_template_library
from src.cot.validators import (
    validate_answer_mapping,
    validate_fact_consistency,
    validate_reasoning_consistency,
    validate_response,
    validate_sft_item,
)


SCHEMA_VERSION = LEGACY_PROFILE.schema_version
CURRICULUM_STAGE_1_EXPOSURES = LEGACY_PROFILE.stage1_exposures
CURRICULUM_STAGE_2_EXPOSURES = LEGACY_PROFILE.stage2_exposures
CURRICULUM_TOTAL_EXPOSURES = LEGACY_PROFILE.total_exposures
GLOBAL_BATCH = LEGACY_PROFILE.global_batch
STAGE_2_BATCH_PATTERN = LEGACY_PROFILE.stage2_pattern
TARGET_EXPOSURES_BY_LEVEL = LEGACY_PROFILE.target_exposures_by_level
STAGE_2_EXPOSURES_BY_LEVEL = LEGACY_PROFILE.stage2_exposures_by_level
VARIANTS = ("answer_only", "fixed_template_cot", "teacher_cot")
EXPECTED_FRESH_BENCHMARK_SHA256 = (
    "bab2eedd451d399132d40abdf1ed1819b67d02db4d69d78f63fb6116332c699c"
)
ROUTE_AWARE_OCCLUSION_QUOTA = 212


def fresh_candidate_type_quotas(
    available_by_type: dict[str, int],
) -> dict[str, int]:
    """Keep reviewed quotas, include both route-distinct occlusion records, and use all strict object-centric rows."""
    quotas = dict(FRESH_12K_PROFILE.candidate_quotas_by_type or {})
    for question_type, quota in quotas.items():
        if question_type == "object_move_object_centric":
            continue
        available = int(available_by_type.get(question_type, 0))
        required = int(quota)
        if (
            question_type == "object_move_occlusion"
            and available >= ROUTE_AWARE_OCCLUSION_QUOTA
        ):
            required = ROUTE_AWARE_OCCLUSION_QUOTA
        if available < required:
            raise ValueError(
                f"{question_type} has {available} unique validated rows, fewer than the "
                f"non-object-centric quota {required}"
            )
        quotas[question_type] = required
    object_centric_available = int(
        available_by_type.get("object_move_object_centric", 0)
    )
    if object_centric_available <= 0:
        raise ValueError("no repaired object_move_object_centric questions are available")
    quotas["object_move_object_centric"] = object_centric_available
    return quotas


def dynamic_l2_max_uid_exposures(
    profile: CurriculumProfile,
    candidate_type_quotas: dict[str, int],
) -> int:
    object_centric_count = int(candidate_type_quotas["object_move_object_centric"])
    object_centric_target = int(
        (profile.l2_exposures_by_type or {})["object_move_object_centric"]
    )
    return max(
        int(profile.l2_max_uid_exposures or 1),
        math.ceil(object_centric_target / object_centric_count),
    )


def stable_digest(*values: object) -> str:
    joined = "|".join(str(value) for value in values)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8-sig") as handle:
        return json.load(handle)


def load_questions(path: Path) -> list[dict[str, Any]]:
    payload = load_json(path)
    questions = payload.get("questions", payload) if isinstance(payload, dict) else payload
    if not isinstance(questions, list) or not all(
        isinstance(question, dict) for question in questions
    ):
        raise ValueError(f"{path}: expected a list or an object containing questions")
    return [dict(question) for question in questions]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(row)
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def counts_by(rows: Iterable[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get(key) or "missing") for row in rows).items()))


def select_fresh_candidates(args: argparse.Namespace) -> dict[str, Any]:
    benchmark_sha256 = file_sha256(args.benchmark)
    if args.expected_benchmark_sha256 and benchmark_sha256 != args.expected_benchmark_sha256:
        raise ValueError(
            "benchmark SHA256 changed: "
            f"expected {args.expected_benchmark_sha256}, got {benchmark_sha256}"
        )

    questions = load_questions(args.benchmark)
    validated_questions: list[dict[str, Any]] = []
    sampling_rows: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for index, raw_question in enumerate(questions):
        question = dict(raw_question)
        uid = question_uid(question)
        question["question_uid"] = uid
        try:
            record = build_fact_record(question)
            validate_fact_consistency(record)
            validate_answer_mapping(question, record)
        except (FactExtractionError, ValueError, TypeError, KeyError) as exc:
            rejected.append(
                {
                    "benchmark_index": index,
                    "question_uid": uid,
                    "question_type": str(question.get("type") or "missing"),
                    "code": getattr(exc, "code", "validation_error"),
                    "message": str(exc),
                }
            )
            continue
        validated_questions.append(question)
        sampling_rows.append(record.to_dict())

    unique_available: Counter[str] = Counter()
    seen_uids: set[str] = set()
    for question, row in zip(validated_questions, sampling_rows):
        uid = str(row.get("question_uid") or "")
        if uid in seen_uids:
            continue
        seen_uids.add(uid)
        unique_available[str(question.get("type") or "missing")] += 1
    candidate_type_quotas = fresh_candidate_type_quotas(dict(unique_available))
    candidate_level_quotas = {
        level: sum(candidate_type_quotas[question_type] for question_type in question_types)
        for level, question_types in TYPES_BY_LEVEL.items()
    }

    selection = select_stratified(
        sampling_rows,
        type_quotas=candidate_type_quotas,
        seed=args.seed,
    )
    selected = [dict(validated_questions[index]) for index in selection.indices]
    for question in selected:
        question["_curriculum_source"] = "benchmark_fresh_12k"

    selected_by_level = counts_by(selected, "level")
    selected_by_type = counts_by(selected, "type")
    if selected_by_level != candidate_level_quotas:
        raise AssertionError(f"fresh candidate level quotas differ: {selected_by_level}")
    if selected_by_type != dict(sorted(candidate_type_quotas.items())):
        raise AssertionError(f"fresh candidate type quotas differ: {selected_by_type}")
    selected_uids = [str(question["question_uid"]) for question in selected]
    expected_total = sum(candidate_type_quotas.values())
    if len(selected_uids) != expected_total or len(set(selected_uids)) != len(selected_uids):
        raise AssertionError(
            f"fresh candidate pool must contain {expected_total} unique UIDs"
        )

    rejection_counts = dict(sorted(Counter(row["code"] for row in rejected).items()))
    statistics = {
        "raw_count": len(questions),
        "validated_count": len(validated_questions),
        "rejected_count": len(rejected),
        "rejection_counts": rejection_counts,
        "duplicate_uid_count": selection.report["duplicate_uid_count"],
        "unique_validated_count": selection.report["unique_supported_count"],
        "selected_count": len(selected),
        "selected_unique_uid_count": len(set(selected_uids)),
        "selected_by_level": selected_by_level,
        "selected_by_type": selected_by_type,
        "dynamic_candidate_quotas_by_level": candidate_level_quotas,
        "dynamic_candidate_quotas_by_type": dict(sorted(candidate_type_quotas.items())),
        "selected_by_signature": selection.report["selected_by_signature"],
        "selected_by_scene": counts_by(selected, "scene_id"),
    }
    payload = {
        "schema_version": "predictive-spatial-cot-candidate-pool-v2",
        "name": "PSR-Bench strict camera-facing CoT candidate pool",
        "profile_id": FRESH_12K_PROFILE_ID,
        "seed": args.seed,
        "source": {
            "benchmark": str(args.benchmark.resolve()),
            "benchmark_sha256": benchmark_sha256,
        },
        "statistics": statistics,
        "questions": selected,
    }
    report = {
        "schema_version": "predictive-spatial-cot-selection-report-v2",
        "profile_id": FRESH_12K_PROFILE_ID,
        "seed": args.seed,
        "source": payload["source"],
        "statistics": statistics,
        "sampling": selection.report,
        "rejected": rejected,
    }
    write_json(args.output, payload)
    if args.mirror_output is not None:
        write_json(args.mirror_output, payload)
    write_json(args.report, report)
    return {
        "output": str(args.output.resolve()),
        "mirror_output": (
            str(args.mirror_output.resolve()) if args.mirror_output is not None else None
        ),
        "report": str(args.report.resolve()),
        **statistics,
    }


def merge_datasets(args: argparse.Namespace) -> dict[str, Any]:
    benchmark_questions = load_questions(args.benchmark)
    benchmark_by_uid: dict[str, dict[str, Any]] = {}
    for question in benchmark_questions:
        uid = question_uid(question)
        if uid in benchmark_by_uid:
            raise ValueError(f"duplicate UID in benchmark: {uid}")
        benchmark_by_uid[uid] = question

    old_sidecar = load_jsonl(args.old_sidecar)
    old_uids = [str(row.get("question_uid") or "").strip() for row in old_sidecar]
    if len(old_uids) != 8_000 or len(set(old_uids)) != 8_000 or "" in old_uids:
        raise ValueError("old sidecar must contain exactly 8,000 unique non-empty UIDs")
    missing = sorted(set(old_uids) - set(benchmark_by_uid))
    if missing:
        raise ValueError(f"could not recover {len(missing)} old-training UIDs from benchmark")

    merged: list[dict[str, Any]] = []
    for uid in old_uids:
        question = dict(benchmark_by_uid[uid])
        question["question_uid"] = uid
        question["_curriculum_source"] = "pilot_train_8k"
        merged.append(question)

    stage2 = load_questions(args.stage2)
    stage2_uids: set[str] = set()
    for raw_question in stage2:
        question = dict(raw_question)
        uid = question_uid(question)
        if uid in stage2_uids:
            raise ValueError(f"duplicate UID in stage-two data: {uid}")
        stage2_uids.add(uid)
        question["question_uid"] = uid
        question["_curriculum_source"] = "stage2_2k"
        merged.append(question)

    overlap = set(old_uids) & stage2_uids
    merged_uids = {str(question["question_uid"]) for question in merged}
    if len(stage2) != 2_000 or overlap or len(merged) != 10_000 or len(merged_uids) != 10_000:
        raise ValueError(
            "merge invariants failed: expected 8k + 2k unique questions with zero overlap"
        )
    merged.sort(key=lambda row: stable_digest(args.seed, "merged", row["question_uid"]))

    statistics = {
        "total": len(merged),
        "unique_question_uid_count": len(merged_uids),
        "old_count": len(old_uids),
        "stage2_count": len(stage2),
        "old_stage2_overlap_count": len(overlap),
        "by_level": counts_by(merged, "level"),
        "by_type": counts_by(merged, "type"),
        "by_source": counts_by(merged, "_curriculum_source"),
    }
    payload = {
        "schema_version": "predictive-spatial-cot-merged-10k-v1",
        "name": "PSR-Bench merged 8k+2k teacher candidate pool",
        "seed": args.seed,
        "statistics": statistics,
        "sources": {
            "benchmark": str(args.benchmark.resolve()),
            "benchmark_sha256": file_sha256(args.benchmark),
            "old_sidecar": str(args.old_sidecar.resolve()),
            "old_sidecar_sha256": file_sha256(args.old_sidecar),
            "stage2": str(args.stage2.resolve()),
            "stage2_sha256": file_sha256(args.stage2),
        },
        "questions": merged,
    }
    write_json(args.output, payload)
    return {"output": str(args.output.resolve()), **statistics}


def expected_answer(question: dict[str, Any]) -> str:
    record = build_fact_record(question)
    return " ".join(record.answer_letters)


def validate_teacher_cot(question: dict[str, Any], response: str) -> None:
    options = question.get("options") or []
    parsed = parse_strict_answer(
        response,
        option_count=len(options),
        multi_select=bool(question.get("multi_select")),
    )
    if parsed != expected_answer(question):
        raise ValueError("teacher_cot does not have the strict gold final answer")
    words = response.rsplit("\nAnswer:", 1)[0].split()
    if not 15 <= len(words) <= 180:
        raise ValueError("teacher_cot reasoning must contain 15-180 words")


def build_adjusted_pool(
    questions: list[dict[str, Any]], *, level: str, type_floor: int, seed: int
) -> list[dict[str, Any]]:
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for question in questions:
        if str(question.get("level") or "").upper() == level:
            by_type[str(question.get("type") or "")].append(question)

    expected_types = tuple(TYPES_BY_LEVEL[level])
    missing_types = [question_type for question_type in expected_types if not by_type[question_type]]
    if missing_types:
        raise ValueError(f"{level} has no teacher-success rows for: {', '.join(missing_types)}")

    adjusted: list[dict[str, Any]] = []
    for question_type in expected_types:
        rows = sorted(
            by_type[question_type],
            key=lambda row: stable_digest(seed, "type", question_type, row["question_uid"]),
        )
        target = max(len(rows), type_floor)
        for pool_index in range(target):
            source = rows[pool_index % len(rows)]
            adjusted.append(
                {
                    "question": source,
                    "pool_replica_index": pool_index // len(rows),
                }
            )
    adjusted.sort(
        key=lambda item: stable_digest(
            seed,
            "adjusted-pool",
            level,
            item["question"]["question_uid"],
            item["pool_replica_index"],
        )
    )
    return adjusted


def make_level_stream(
    pool: list[dict[str, Any]], *, count: int, seed: int, scope: str
) -> list[dict[str, Any]]:
    if not pool:
        raise ValueError(f"cannot build {scope} from an empty pool")
    result: list[dict[str, Any]] = []
    cycle = 0
    while len(result) < count:
        ordered = sorted(
            pool,
            key=lambda item: stable_digest(
                seed,
                scope,
                cycle,
                item["question"]["question_uid"],
                item["pool_replica_index"],
            ),
        )
        result.extend(ordered[: count - len(result)])
        cycle += 1
    return result


def build_l2_stream(
    questions: list[dict[str, Any]],
    *,
    target_by_type: dict[str, int],
    max_uid_exposures: int,
    max_uid_exposures_by_type: dict[str, int] | None = None,
    seed: int,
    enforce_sqrt_allocation: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    per_type_caps = {
        question_type: int((max_uid_exposures_by_type or {}).get(question_type, max_uid_exposures))
        for question_type in target_by_type
    }
    invalid_caps = {
        question_type: cap for question_type, cap in per_type_caps.items() if cap < 1
    }
    if invalid_caps:
        raise ValueError(f"L2 UID exposure caps must be positive: {invalid_caps}")
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for question in questions:
        by_type[str(question.get("type") or "")].append(question)
    capacities = {question_type: len(by_type[question_type]) for question_type in target_by_type}
    calculated_targets = sqrt_largest_remainder_allocation(sum(target_by_type.values()), capacities)
    if enforce_sqrt_allocation and calculated_targets != target_by_type:
        raise AssertionError(
            f"configured L2 exposure quotas differ from square-root allocation: {calculated_targets}"
        )

    occurrence_by_uid: Counter[str] = Counter()
    base: list[dict[str, Any]] = []
    extras: list[dict[str, Any]] = []
    group_statistics: dict[str, dict[str, int]] = {}
    for question_type, target in target_by_type.items():
        rows = by_type[question_type]
        type_cap = per_type_caps[question_type]
        if not rows:
            raise ValueError(f"cannot build L2 stream without {question_type} candidates")
        if target > len(rows) * type_cap:
            raise ValueError(
                f"{question_type} target {target} exceeds the {type_cap}x UID cap"
            )
        for question in rows:
            uid = str(question["question_uid"])
            occurrence_by_uid[uid] = 1
            base.append({"question": question, "pool_replica_index": 0})

        groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
        for question in rows:
            groups[l2_rarity_group(question)].append(question)
        extra_needed = target - len(rows)
        extra_by_group: Counter[tuple[str, ...]] = Counter()
        for _ in range(extra_needed):
            eligible_groups = [
                (group_key, group_rows)
                for group_key, group_rows in groups.items()
                if any(
                    occurrence_by_uid[str(question["question_uid"])] < type_cap
                    for question in group_rows
                )
            ]
            if not eligible_groups:
                raise AssertionError(f"L2 repetition capacity exhausted for {question_type}")
            group_key, group_rows = min(
                eligible_groups,
                key=lambda item: (
                    len(item[1]),
                    extra_by_group[item[0]] / len(item[1]),
                    stable_hash(seed, "l2-group", *item[0]),
                ),
            )
            eligible_rows = [
                question
                for question in group_rows
                if occurrence_by_uid[str(question["question_uid"])] < type_cap
            ]
            question = min(
                eligible_rows,
                key=lambda row: (
                    occurrence_by_uid[str(row["question_uid"])],
                    stable_hash(seed, "l2-uid", row["question_uid"]),
                ),
            )
            uid = str(question["question_uid"])
            replica_index = occurrence_by_uid[uid]
            occurrence_by_uid[uid] += 1
            extra_by_group[group_key] += 1
            extras.append(
                {"question": question, "pool_replica_index": replica_index}
            )
        group_statistics[question_type] = {
            "group_count": len(groups),
            "minimum_group_size": min(len(group) for group in groups.values()),
            "maximum_group_size": max(len(group) for group in groups.values()),
            "groups_receiving_repeats": len(extra_by_group),
        }

    base.sort(
        key=lambda item: stable_hash(seed, "l2-base", item["question"]["question_uid"])
    )
    extras.sort(
        key=lambda item: stable_hash(
            seed,
            "l2-extra",
            item["pool_replica_index"],
            item["question"]["question_uid"],
        )
    )
    stream = base + extras
    exposure_by_type = Counter(item["question"]["type"] for item in stream)
    if dict(exposure_by_type) != target_by_type:
        raise AssertionError(f"unexpected L2 exposure counts: {dict(exposure_by_type)}")
    multiplicities = Counter(occurrence_by_uid.values())
    multiplicities_by_type: dict[str, dict[int, int]] = {}
    for question_type, rows in by_type.items():
        counts = Counter(
            occurrence_by_uid[str(question["question_uid"])] for question in rows
        )
        multiplicities_by_type[question_type] = dict(sorted(counts.items()))
        if counts and (min(counts) < 1 or max(counts) > per_type_caps[question_type]):
            raise AssertionError(
                f"{question_type} UID multiplicities are outside the configured range"
            )
    return stream, {
        "allocation": "sqrt_largest_remainder",
        "target_by_type": target_by_type,
        "max_uid_exposures": max_uid_exposures,
        "max_uid_exposures_by_type": dict(sorted((max_uid_exposures_by_type or {}).items())),
        "multiplicity_counts": dict(sorted(multiplicities.items())),
        "multiplicity_counts_by_type": dict(sorted(multiplicities_by_type.items())),
        "rarity_groups_by_type": group_statistics,
    }


def interleave_stage2(
    streams: dict[str, list[dict[str, Any]]],
    *,
    profile: CurriculumProfile = LEGACY_PROFILE,
) -> list[tuple[str, str, dict[str, Any]]]:
    offsets = {level: 0 for level in streams}
    scheduled: list[tuple[str, str, dict[str, Any]]] = []
    for _ in range(profile.stage2_pattern_repetitions):
        for batch_name, composition in profile.stage2_pattern:
            batch: list[tuple[str, str, dict[str, Any]]] = []
            for level in ("L1", "L2", "L3"):
                start = offsets[level]
                end = start + composition[level]
                batch.extend((batch_name, level, item) for item in streams[level][start:end])
                offsets[level] = end
            batch.sort(
                key=lambda row: stable_digest(
                    "stage2-batch",
                    len(scheduled) // profile.global_batch,
                    row[1],
                    row[2]["question"]["question_uid"],
                    row[2]["pool_replica_index"],
                )
            )
            if len(batch) != profile.global_batch:
                raise AssertionError("every stage-two global batch must contain 32 rows")
            scheduled.extend(batch)
    if offsets != profile.stage2_exposures_by_level:
        raise AssertionError(f"stage-two level offsets do not match targets: {offsets}")
    return scheduled


def build_fresh_schedule(
    questions: list[dict[str, Any]], *, profile: CurriculumProfile, seed: int
) -> tuple[list[tuple[str, str, dict[str, Any]]], dict[str, Any]]:
    by_level = {
        level: [question for question in questions if str(question.get("level") or "").upper() == level]
        for level in ("L1", "L2", "L3")
    }
    candidate_by_level = {level: len(rows) for level, rows in by_level.items()}
    candidate_by_type = dict(sorted(Counter(question["type"] for question in questions).items()))
    if candidate_by_level != profile.candidate_quotas_by_level:
        raise ValueError(f"fresh candidate level quotas differ: {candidate_by_level}")
    if candidate_by_type != dict(sorted((profile.candidate_quotas_by_type or {}).items())):
        raise ValueError(f"fresh candidate type quotas differ: {candidate_by_type}")

    pools = {
        level: [{"question": question, "pool_replica_index": 0} for question in rows]
        for level, rows in by_level.items()
    }
    stage1 = make_level_stream(
        pools["L1"], count=profile.stage1_exposures, seed=seed, scope="fresh-stage1-L1"
    )
    l2_stream, l2_report = build_l2_stream(
        by_level["L2"],
        target_by_type=dict(profile.l2_exposures_by_type or {}),
        max_uid_exposures=int(profile.l2_max_uid_exposures or 1),
        max_uid_exposures_by_type=profile.l2_max_uid_exposures_by_type,
        seed=seed,
        enforce_sqrt_allocation=(
            profile.candidate_quotas_by_type
            == FRESH_12K_PROFILE.candidate_quotas_by_type
        ),
    )
    stage2_counts = profile.stage2_exposures_by_level
    stage2_streams = {
        "L1": make_level_stream(
            pools["L1"], count=stage2_counts["L1"], seed=seed, scope="fresh-stage2-L1"
        ),
        "L2": l2_stream,
        "L3": make_level_stream(
            pools["L3"], count=stage2_counts["L3"], seed=seed, scope="fresh-stage2-L3"
        ),
    }
    raw_schedule: list[tuple[str, str, dict[str, Any]]] = [
        ("L1", "L1", item) for item in stage1
    ]
    raw_schedule.extend(interleave_stage2(stage2_streams, profile=profile))
    scheduled_uids = {
        str(pool_item["question"]["question_uid"])
        for _batch_name, _level, pool_item in raw_schedule
    }
    candidate_uids = {str(question["question_uid"]) for question in questions}
    if scheduled_uids != candidate_uids:
        raise AssertionError(
            f"fresh schedule omitted {len(candidate_uids - scheduled_uids)} candidate UIDs"
        )
    return raw_schedule, {"l2_repetition_policy": l2_report}


def artifact_path(prefix: Path, variant: str, suffix: str) -> Path:
    return prefix.parent / f"{prefix.name}.{variant}.{suffix}"


def finalize_dataset(args: argparse.Namespace) -> dict[str, Any]:
    profile = get_curriculum_profile(args.profile)
    variants = tuple(variant for variant in VARIANTS if variant in args.variants)
    if not variants:
        raise ValueError("at least one supervision variant is required")
    requires_teacher = "teacher_cot" in variants
    requires_fixed_template = "fixed_template_cot" in variants
    payload = load_json(args.teacher_benchmark)
    questions = load_questions(args.teacher_benchmark)
    seen: set[str] = set()
    accepted: list[dict[str, Any]] = []
    records: dict[str, Any] = {}
    images_by_uid: dict[str, tuple[list[str], list[dict[str, Any]]]] = {}
    templates = load_template_library(args.template_path) if requires_fixed_template else None

    for raw_question in questions:
        question = dict(raw_question)
        uid = question_uid(question)
        if uid in seen:
            raise ValueError(f"teacher benchmark contains duplicate UID: {uid}")
        seen.add(uid)
        question["question_uid"] = uid
        teacher_cot = str(question.get("teacher_cot") or "").strip()
        if requires_teacher and not teacher_cot:
            raise ValueError(f"teacher-success question is missing teacher_cot: {uid}")
        record = build_fact_record(question)
        validate_fact_consistency(record)
        validate_answer_mapping(question, record)
        if requires_teacher:
            validate_teacher_cot(question, teacher_cot)
        images_by_uid[uid] = resolve_image_paths(
            question,
            benchmark_path=args.teacher_benchmark,
            scannet_roots=[path.resolve() for path in args.scannet_image_root],
            scannetpp_roots=[path.resolve() for path in args.scannetpp_image_root],
            scannetpp_sensor=args.scannetpp_sensor,
            require_exists=not args.allow_missing_images,
        )
        accepted.append(question)
        records[uid] = record

    if profile.profile_id == FRESH_12K_PROFILE_ID:
        accepted_by_type = dict(Counter(str(question["type"]) for question in accepted))
        dynamic_type_quotas = fresh_candidate_type_quotas(accepted_by_type)
        if accepted_by_type != dynamic_type_quotas:
            raise ValueError(
                f"fresh candidate contents differ from dynamic quotas: {accepted_by_type}"
            )
        dynamic_level_quotas = {
            level: sum(dynamic_type_quotas[question_type] for question_type in question_types)
            for level, question_types in TYPES_BY_LEVEL.items()
        }
        profile = replace(
            profile,
            candidate_quotas_by_level=dynamic_level_quotas,
            candidate_quotas_by_type=dynamic_type_quotas,
            l2_max_uid_exposures=dynamic_l2_max_uid_exposures(
                profile, dynamic_type_quotas
            ),
        )

    schedule_metadata: dict[str, Any] = {}
    pools: dict[str, list[dict[str, Any]]] | None = None
    if profile.profile_id == FRESH_12K_PROFILE_ID:
        raw_schedule, schedule_metadata = build_fresh_schedule(
            accepted, profile=profile, seed=args.seed
        )
    else:
        pools = {
            level: build_adjusted_pool(
                accepted,
                level=level,
                type_floor=args.type_floor,
                seed=args.seed,
            )
            for level in ("L1", "L2", "L3")
        }
        stage_capacity = {
            "L1": profile.stage1_exposures,
            "L2": profile.stage2_exposures_by_level["L2"],
            "L3": profile.stage2_exposures_by_level["L3"],
        }
        for level, pool in pools.items():
            if len(pool) > stage_capacity[level]:
                raise ValueError(
                    f"{level} adjusted pool has {len(pool)} rows but its first-pass "
                    f"curriculum capacity is only {stage_capacity[level]}; cannot preserve "
                    "every source question and the per-type floor"
                )
        unique_by_level = Counter(
            str(question.get("level") or "").upper() for question in accepted
        )
        for level, count in unique_by_level.items():
            if count > profile.target_exposures_by_level.get(level, 0):
                raise ValueError(
                    f"{level} has {count} unique rows but only "
                    f"{profile.target_exposures_by_level.get(level, 0)} curriculum exposures"
                )

        stage1 = make_level_stream(
            pools["L1"],
            count=profile.stage1_exposures,
            seed=args.seed,
            scope="stage1-L1",
        )
        stage2_streams = {
            level: make_level_stream(
                pools[level],
                count=profile.stage2_exposures_by_level[level],
                seed=args.seed,
                scope=f"stage2-{level}",
            )
            for level in ("L1", "L2", "L3")
        }
        raw_schedule = [("L1", "L1", item) for item in stage1]
        raw_schedule.extend(interleave_stage2(stage2_streams, profile=profile))

    if len(raw_schedule) != profile.total_exposures:
        raise AssertionError(
            f"curriculum schedule must contain {profile.total_exposures:,} exposures"
        )
    if any(level != "L1" for _, level, _ in raw_schedule[: profile.stage1_exposures]):
        raise AssertionError("the first 6,144 curriculum exposures must all be L1")
    stage2_raw = raw_schedule[profile.stage1_exposures :]
    macro_size = len(profile.stage2_pattern) * profile.global_batch
    for offset in range(0, len(stage2_raw), macro_size):
        macro_counts = Counter(level for _, level, _ in stage2_raw[offset : offset + macro_size])
        expected_macro_counts = {
            level: count // profile.stage2_pattern_repetitions
            for level, count in profile.stage2_exposures_by_level.items()
        }
        if dict(macro_counts) != expected_macro_counts:
            raise AssertionError(
                f"stage-two macro-cycle {offset // macro_size} has invalid counts: "
                f"{dict(macro_counts)}"
            )

    occurrence_by_uid: Counter[str] = Counter()
    master_samples: list[dict[str, Any]] = []
    sft_by_variant: dict[str, list[dict[str, Any]]] = {variant: [] for variant in variants}
    sidecar_by_variant: dict[str, list[dict[str, Any]]] = {variant: [] for variant in variants}
    schedule_hasher = hashlib.sha256()

    for zero_index, (batch_name, level, pool_item) in enumerate(raw_schedule):
        exposure_index = zero_index + 1
        question = pool_item["question"]
        uid = str(question["question_uid"])
        occurrence = occurrence_by_uid[uid]
        occurrence_by_uid[uid] += 1
        sample_uid = stable_digest(args.seed, "sample", uid, occurrence)
        record = records[uid]
        images, diagnostics = images_by_uid[uid]
        template_id = None
        targets: dict[str, str] = {}
        if "answer_only" in variants:
            targets["answer_only"] = f"Answer: {' '.join(record.answer_letters)}"
        if requires_fixed_template:
            fixed_seed = int(stable_digest(args.seed, sample_uid)[:8], 16)
            fixed_cot, template_id = render_response(
                record,
                seed=fixed_seed,
                template_library=templates,
            )
            validate_response(fixed_cot, record)
            validate_reasoning_consistency(fixed_cot, record)
            targets["fixed_template_cot"] = fixed_cot
        if requires_teacher:
            targets["teacher_cot"] = str(question["teacher_cot"]).strip()
        stage = "stage1_l1" if exposure_index <= profile.stage1_exposures else "stage2_reasoning"
        global_batch_index = zero_index // profile.global_batch
        sample = {
            "sample_uid": sample_uid,
            "question_uid": uid,
            "replica_index": occurrence,
            "pool_replica_index": pool_item["pool_replica_index"],
            "exposure_index": exposure_index,
            "global_batch_index": global_batch_index,
            "stage": stage,
            "stage2_batch_pattern": batch_name if stage == "stage2_reasoning" else None,
            "level": level,
            "question_type": record.question_type,
            "signature_id": record.signature_id,
            "source": question.get("_curriculum_source"),
            "user_content": format_user_prompt(question, len(images)),
            "images": images,
            "image_resolution": diagnostics,
            "template_id": template_id,
            "targets": targets,
        }
        master_samples.append(sample)
        schedule_hasher.update(f"{exposure_index}|{sample_uid}|{level}\n".encode("utf-8"))

        for variant, response in targets.items():
            item = {
                "messages": [
                    {"role": "user", "content": sample["user_content"]},
                    {"role": "assistant", "content": response},
                ],
                "images": images,
                "question_uid": uid,
                "sample_uid": sample_uid,
                "question_type": record.question_type,
                "signature_id": record.signature_id,
                "curriculum_stage": stage,
                "curriculum_exposure": exposure_index,
            }
            validate_sft_item(item)
            sft_by_variant[variant].append(item)
            sidecar = record.to_dict()
            sidecar.update(
                sample_uid=sample_uid,
                response=response,
                supervision_variant=variant,
                option_count=len(question.get("options") or []),
                multi_select=bool(question.get("multi_select")),
                images=images,
                curriculum_stage=stage,
                curriculum_exposure=exposure_index,
            )
            sidecar_by_variant[variant].append(sidecar)

    by_level = Counter(sample["level"] for sample in master_samples)
    if dict(by_level) != profile.target_exposures_by_level:
        raise AssertionError(f"unexpected final curriculum level counts: {dict(by_level)}")
    scheduled_by_type = Counter(sample["question_type"] for sample in master_samples)
    if profile.profile_id == LEGACY_PROFILE_ID:
        underfilled_types = {
            question_type: count
            for question_type, count in scheduled_by_type.items()
            if count < args.type_floor
        }
        if underfilled_types:
            raise AssertionError(f"scheduled types below the configured floor: {underfilled_types}")
    if len({sample["sample_uid"] for sample in master_samples}) != len(master_samples):
        raise AssertionError("sample_uid values must be unique")
    scheduled_uids = set(occurrence_by_uid)
    accepted_uids = {str(question["question_uid"]) for question in accepted}
    missing_uids = sorted(accepted_uids - scheduled_uids)
    if missing_uids:
        raise AssertionError(
            f"curriculum omitted {len(missing_uids)} source questions"
        )

    reference_order = [row["sample_uid"] for row in sft_by_variant[variants[0]]]
    for variant in variants[1:]:
        if [row["sample_uid"] for row in sft_by_variant[variant]] != reference_order:
            raise AssertionError(f"{variant} does not share the canonical sample order")
        for reference, candidate in zip(
            sft_by_variant[variants[0]], sft_by_variant[variant]
        ):
            if (
                reference["messages"][0] != candidate["messages"][0]
                or reference["images"] != candidate["images"]
            ):
                raise AssertionError(
                    f"{variant} does not share canonical prompts and images"
                )

    pool_by_type: Counter[str] = Counter()
    if pools is not None:
        for pool in pools.values():
            pool_by_type.update(item["question"]["type"] for item in pool)
        if any(
            pool_by_type[question_type] < args.type_floor
            for level in TYPES_BY_LEVEL.values()
            for question_type in level
        ):
            raise AssertionError("every supported type must meet the configured pool floor")
    else:
        pool_by_type.update(question["type"] for question in accepted)

    statistics = {
        "source_unique_count": len(accepted),
        "source_scheduled_count": len(scheduled_uids),
        "teacher_success_unique_count": len(accepted) if requires_teacher else None,
        "teacher_success_scheduled_count": len(scheduled_uids) if requires_teacher else None,
        "teacher_input_count": (
            payload.get("statistics", {}).get("input_count")
            if isinstance(payload, dict)
            else None
        ),
        "profile_id": profile.profile_id,
        "type_floor": args.type_floor if pools is not None else None,
        "adjusted_pool_count": (
            sum(len(pool) for pool in pools.values()) if pools is not None else len(accepted)
        ),
        "adjusted_pool_by_level": (
            {level: len(pool) for level, pool in pools.items()}
            if pools is not None
            else dict(sorted(Counter(question["level"] for question in accepted).items()))
        ),
        "adjusted_pool_by_type": dict(sorted(pool_by_type.items())),
        "total_exposures": len(master_samples),
        "global_batch": profile.global_batch,
        "optimizer_steps": len(master_samples) // profile.global_batch,
        "stage1_exposures": profile.stage1_exposures,
        "stage2_exposures": profile.stage2_exposures,
        "exposures_by_level": dict(sorted(by_level.items())),
        "exposures_by_type": dict(sorted(scheduled_by_type.items())),
        "schedule_sha256": schedule_hasher.hexdigest(),
        "supervision_variants": list(variants),
        **schedule_metadata,
    }
    master = {
        "schema_version": profile.schema_version,
        "name": "PSR-Bench matched two-stage curriculum",
        "seed": args.seed,
        "curriculum": {
            "profile_id": profile.profile_id,
            "candidate_quotas_by_level": profile.candidate_quotas_by_level,
            "candidate_quotas_by_type": profile.candidate_quotas_by_type,
            "stage1": {
                "exposures": profile.stage1_exposures,
                "composition": {"L1": profile.stage1_exposures},
            },
            "stage2": {
                "exposures": profile.stage2_exposures,
                "composition": profile.stage2_exposures_by_level,
                "batch_pattern": [
                    {"name": name, "composition": composition}
                    for name, composition in profile.stage2_pattern
                ],
                "macro_cycle_repetitions": profile.stage2_pattern_repetitions,
            },
            "l2_exposures_by_type": profile.l2_exposures_by_type,
            "l2_max_uid_exposures": profile.l2_max_uid_exposures,
            "l2_max_uid_exposures_by_type": profile.l2_max_uid_exposures_by_type,
            "dataset_shuffle": False,
            "train_dataloader_shuffle": False,
        },
        "statistics": statistics,
        "source": {
            "input_dataset": str(args.teacher_benchmark.resolve()),
            "input_dataset_sha256": file_sha256(args.teacher_benchmark),
        },
        "samples": master_samples,
    }
    if profile.schema_version != LEGACY_PROFILE.schema_version:
        master["profile_id"] = profile.profile_id
    write_json(args.output, master)
    variant_reports: dict[str, dict[str, Any]] = {}
    for variant in variants:
        dataset_path = artifact_path(args.output_prefix, variant, "ms_swift.jsonl")
        sidecar_path = artifact_path(args.output_prefix, variant, "sidecar.jsonl")
        write_jsonl(
            dataset_path,
            sft_by_variant[variant],
        )
        write_jsonl(
            sidecar_path,
            sidecar_by_variant[variant],
        )
        variant_report = {
            "schema_version": "predictive-spatial-cot-export-report-v2",
            "profile_id": profile.profile_id,
            "variant": variant,
            "row_count": len(sft_by_variant[variant]),
            "schedule_sha256": statistics["schedule_sha256"],
            "dataset": str(dataset_path.resolve()),
            "dataset_sha256": file_sha256(dataset_path),
            "sidecar": str(sidecar_path.resolve()),
            "sidecar_sha256": file_sha256(sidecar_path),
            "images_verified": not args.allow_missing_images,
        }
        write_json(artifact_path(args.output_prefix, variant, "report.json"), variant_report)
        variant_reports[variant] = variant_report
    report = {
        **statistics,
        "master_output": str(args.output.resolve()),
        "master_output_sha256": file_sha256(args.output),
        "output_prefix": str(args.output_prefix.resolve()),
        "images_verified": not args.allow_missing_images,
        "variants": variant_reports,
    }
    write_json(args.output_prefix.parent / f"{args.output_prefix.name}.report.json", report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    select = subparsers.add_parser(
        "select", help="Validate and select the fresh 12k candidate pool"
    )
    select.add_argument(
        "--benchmark", type=Path, default=Path("output_train/benchmark.json")
    )
    select.add_argument(
        "--output", type=Path, default=Path("output_train/mixed_train_12k.json")
    )
    select.add_argument(
        "--mirror-output", type=Path, default=Path("cot/train/mixed_train_12k.json")
    )
    select.add_argument(
        "--report",
        type=Path,
        default=Path("output_train/mixed_train_12k.selection_report.json"),
    )
    select.add_argument("--seed", type=int, default=42)
    select.add_argument(
        "--expected-benchmark-sha256",
        default=EXPECTED_FRESH_BENCHMARK_SHA256,
        help="Abort if the benchmark no longer matches the reviewed source; empty disables.",
    )

    merge = subparsers.add_parser("merge", help="Merge the exact old 8k and stage-two 2k")
    merge.add_argument("--benchmark", type=Path, default=Path("output_train/benchmark.json"))
    merge.add_argument(
        "--old-sidecar",
        type=Path,
        default=Path("cot/train/pilot_train_8k.sidecar.jsonl"),
    )
    merge.add_argument(
        "--stage2", type=Path, default=Path("output_train/stage2_train_2k.json")
    )
    merge.add_argument(
        "--output", type=Path, default=Path("cot/train/mixed_train_10k.json")
    )
    merge.add_argument("--seed", type=int, default=42)

    finalize = subparsers.add_parser(
        "finalize", help="Build the matched curriculum and three MS-SWIFT exports"
    )
    finalize.add_argument("teacher_benchmark", type=Path)
    finalize.add_argument(
        "--variants",
        nargs="+",
        choices=VARIANTS,
        default=list(VARIANTS),
        help="Supervision exports to build; answer_only does not require teacher_cot.",
    )
    finalize.add_argument(
        "--profile",
        choices=tuple(PROFILES),
        default=LEGACY_PROFILE_ID,
        help="Curriculum schedule and validation profile.",
    )
    finalize.add_argument(
        "--output",
        type=Path,
        default=Path("cot/train/mixed_train_curriculum_three_targets.json"),
    )
    finalize.add_argument(
        "--output-prefix", type=Path, default=Path("cot/train/mixed_train_curriculum")
    )
    finalize.add_argument("--type-floor", type=int, default=300)
    finalize.add_argument("--seed", type=int, default=42)
    finalize.add_argument("--template-path", type=Path)
    finalize.add_argument("--scannet-image-root", action="append", type=Path, default=[])
    finalize.add_argument("--scannetpp-image-root", action="append", type=Path, default=[])
    finalize.add_argument("--scannetpp-sensor", choices=("iphone", "dslr"), default="iphone")
    finalize.add_argument("--allow-missing-images", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "select":
        report = select_fresh_candidates(args)
    elif args.command == "merge":
        report = merge_datasets(args)
    else:
        if args.type_floor <= 0:
            raise ValueError("--type-floor must be positive")
        report = finalize_dataset(args)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
