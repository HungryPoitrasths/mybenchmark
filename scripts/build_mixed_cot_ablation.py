#!/usr/bin/env python3
"""Build the matched three-target curriculum dataset for CoT ablations."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cot.evaluation import parse_strict_answer
from src.cot.facts import build_fact_record, question_uid
from src.cot.images import resolve_image_paths
from src.cot.pipeline import format_user_prompt
from src.cot.render import render_response
from src.cot.sampling import TYPES_BY_LEVEL
from src.cot.templates import load_template_library
from src.cot.validators import (
    validate_answer_mapping,
    validate_fact_consistency,
    validate_reasoning_consistency,
    validate_response,
    validate_sft_item,
)


SCHEMA_VERSION = "predictive-spatial-cot-curriculum-v1"
CURRICULUM_STAGE_1_EXPOSURES = 6_144
CURRICULUM_STAGE_2_EXPOSURES = 14_336
CURRICULUM_TOTAL_EXPOSURES = 20_480
GLOBAL_BATCH = 32
STAGE_2_BATCH_PATTERN = (
    ("A", {"L1": 4, "L2": 14, "L3": 14}),
    ("B", {"L1": 5, "L2": 14, "L3": 13}),
    ("A", {"L1": 4, "L2": 14, "L3": 14}),
    ("C", {"L1": 5, "L2": 13, "L3": 14}),
    ("A", {"L1": 4, "L2": 14, "L3": 14}),
    ("B", {"L1": 5, "L2": 14, "L3": 13}),
    ("C", {"L1": 5, "L2": 13, "L3": 14}),
)
TARGET_EXPOSURES_BY_LEVEL = {"L1": 8_192, "L2": 6_144, "L3": 6_144}
STAGE_2_EXPOSURES_BY_LEVEL = {"L1": 2_048, "L2": 6_144, "L3": 6_144}
VARIANTS = ("answer_only", "fixed_template_cot", "teacher_cot")


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


def interleave_stage2(
    streams: dict[str, list[dict[str, Any]]]
) -> list[tuple[str, str, dict[str, Any]]]:
    offsets = {level: 0 for level in streams}
    scheduled: list[tuple[str, str, dict[str, Any]]] = []
    macro_cycles = CURRICULUM_STAGE_2_EXPOSURES // (len(STAGE_2_BATCH_PATTERN) * GLOBAL_BATCH)
    if macro_cycles != 64:
        raise AssertionError("stage-two schedule must contain 64 macro cycles")
    for _ in range(macro_cycles):
        for batch_name, composition in STAGE_2_BATCH_PATTERN:
            batch: list[tuple[str, str, dict[str, Any]]] = []
            for level in ("L1", "L2", "L3"):
                start = offsets[level]
                end = start + composition[level]
                batch.extend((batch_name, level, item) for item in streams[level][start:end])
                offsets[level] = end
            batch.sort(
                key=lambda row: stable_digest(
                    "stage2-batch",
                    len(scheduled) // GLOBAL_BATCH,
                    row[1],
                    row[2]["question"]["question_uid"],
                    row[2]["pool_replica_index"],
                )
            )
            if len(batch) != GLOBAL_BATCH:
                raise AssertionError("every stage-two global batch must contain 32 rows")
            scheduled.extend(batch)
    if offsets != STAGE_2_EXPOSURES_BY_LEVEL:
        raise AssertionError(f"stage-two level offsets do not match targets: {offsets}")
    return scheduled


def artifact_path(prefix: Path, variant: str, suffix: str) -> Path:
    return prefix.parent / f"{prefix.name}.{variant}.{suffix}"


def finalize_dataset(args: argparse.Namespace) -> dict[str, Any]:
    payload = load_json(args.teacher_benchmark)
    questions = load_questions(args.teacher_benchmark)
    seen: set[str] = set()
    accepted: list[dict[str, Any]] = []
    records: dict[str, Any] = {}
    images_by_uid: dict[str, tuple[list[str], list[dict[str, Any]]]] = {}
    templates = load_template_library(args.template_path)

    for raw_question in questions:
        question = dict(raw_question)
        uid = question_uid(question)
        if uid in seen:
            raise ValueError(f"teacher benchmark contains duplicate UID: {uid}")
        seen.add(uid)
        question["question_uid"] = uid
        teacher_cot = str(question.get("teacher_cot") or "").strip()
        if not teacher_cot:
            raise ValueError(f"teacher-success question is missing teacher_cot: {uid}")
        record = build_fact_record(question)
        validate_fact_consistency(record)
        validate_answer_mapping(question, record)
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
        "L1": CURRICULUM_STAGE_1_EXPOSURES,
        "L2": STAGE_2_EXPOSURES_BY_LEVEL["L2"],
        "L3": STAGE_2_EXPOSURES_BY_LEVEL["L3"],
    }
    for level, pool in pools.items():
        if len(pool) > stage_capacity[level]:
            raise ValueError(
                f"{level} adjusted pool has {len(pool)} rows but its first-pass "
                f"curriculum capacity is only {stage_capacity[level]}; cannot preserve "
                "every teacher-success question and the per-type floor"
            )
    unique_by_level = Counter(str(question.get("level") or "").upper() for question in accepted)
    for level, count in unique_by_level.items():
        if count > TARGET_EXPOSURES_BY_LEVEL.get(level, 0):
            raise ValueError(
                f"{level} has {count} unique rows but only "
                f"{TARGET_EXPOSURES_BY_LEVEL.get(level, 0)} curriculum exposures"
            )

    stage1 = make_level_stream(
        pools["L1"], count=CURRICULUM_STAGE_1_EXPOSURES, seed=args.seed, scope="stage1-L1"
    )
    stage2_streams = {
        level: make_level_stream(
            pools[level],
            count=STAGE_2_EXPOSURES_BY_LEVEL[level],
            seed=args.seed,
            scope=f"stage2-{level}",
        )
        for level in ("L1", "L2", "L3")
    }
    raw_schedule: list[tuple[str, str, dict[str, Any]]] = [
        ("L1", "L1", item) for item in stage1
    ]
    raw_schedule.extend(interleave_stage2(stage2_streams))
    if len(raw_schedule) != CURRICULUM_TOTAL_EXPOSURES:
        raise AssertionError("curriculum schedule must contain 20,480 exposures")
    if any(level != "L1" for _, level, _ in raw_schedule[:CURRICULUM_STAGE_1_EXPOSURES]):
        raise AssertionError("the first 6,144 curriculum exposures must all be L1")
    stage2_raw = raw_schedule[CURRICULUM_STAGE_1_EXPOSURES:]
    macro_size = len(STAGE_2_BATCH_PATTERN) * GLOBAL_BATCH
    for offset in range(0, len(stage2_raw), macro_size):
        macro_counts = Counter(level for _, level, _ in stage2_raw[offset : offset + macro_size])
        if dict(macro_counts) != {"L1": 32, "L2": 96, "L3": 96}:
            raise AssertionError(
                f"stage-two macro-cycle {offset // macro_size} has invalid counts: "
                f"{dict(macro_counts)}"
            )

    occurrence_by_uid: Counter[str] = Counter()
    master_samples: list[dict[str, Any]] = []
    sft_by_variant: dict[str, list[dict[str, Any]]] = {variant: [] for variant in VARIANTS}
    sidecar_by_variant: dict[str, list[dict[str, Any]]] = {variant: [] for variant in VARIANTS}
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
        fixed_seed = int(stable_digest(args.seed, sample_uid)[:8], 16)
        fixed_cot, template_id = render_response(
            record,
            seed=fixed_seed,
            template_library=templates,
        )
        validate_response(fixed_cot, record)
        validate_reasoning_consistency(fixed_cot, record)
        answer_only = f"Answer: {' '.join(record.answer_letters)}"
        teacher_cot = str(question["teacher_cot"]).strip()
        targets = {
            "answer_only": answer_only,
            "fixed_template_cot": fixed_cot,
            "teacher_cot": teacher_cot,
        }
        stage = "stage1_l1" if exposure_index <= CURRICULUM_STAGE_1_EXPOSURES else "stage2_reasoning"
        global_batch_index = zero_index // GLOBAL_BATCH
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
    if dict(by_level) != TARGET_EXPOSURES_BY_LEVEL:
        raise AssertionError(f"unexpected final curriculum level counts: {dict(by_level)}")
    scheduled_by_type = Counter(sample["question_type"] for sample in master_samples)
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
            f"curriculum omitted {len(missing_uids)} teacher-success questions"
        )

    reference_order = [row["sample_uid"] for row in sft_by_variant[VARIANTS[0]]]
    for variant in VARIANTS[1:]:
        if [row["sample_uid"] for row in sft_by_variant[variant]] != reference_order:
            raise AssertionError(f"{variant} does not share the canonical sample order")
        for reference, candidate in zip(
            sft_by_variant[VARIANTS[0]], sft_by_variant[variant]
        ):
            if (
                reference["messages"][0] != candidate["messages"][0]
                or reference["images"] != candidate["images"]
            ):
                raise AssertionError(
                    f"{variant} does not share canonical prompts and images"
                )

    pool_by_type: Counter[str] = Counter()
    for pool in pools.values():
        pool_by_type.update(item["question"]["type"] for item in pool)
    if any(pool_by_type[question_type] < args.type_floor for level in TYPES_BY_LEVEL.values() for question_type in level):
        raise AssertionError("every supported type must meet the configured pool floor")

    statistics = {
        "teacher_success_unique_count": len(accepted),
        "teacher_success_scheduled_count": len(scheduled_uids),
        "teacher_input_count": (
            payload.get("statistics", {}).get("input_count")
            if isinstance(payload, dict)
            else None
        ),
        "type_floor": args.type_floor,
        "adjusted_pool_count": sum(len(pool) for pool in pools.values()),
        "adjusted_pool_by_level": {level: len(pool) for level, pool in pools.items()},
        "adjusted_pool_by_type": dict(sorted(pool_by_type.items())),
        "total_exposures": len(master_samples),
        "global_batch": GLOBAL_BATCH,
        "optimizer_steps": len(master_samples) // GLOBAL_BATCH,
        "stage1_exposures": CURRICULUM_STAGE_1_EXPOSURES,
        "stage2_exposures": CURRICULUM_STAGE_2_EXPOSURES,
        "exposures_by_level": dict(sorted(by_level.items())),
        "exposures_by_type": dict(sorted(scheduled_by_type.items())),
        "schedule_sha256": schedule_hasher.hexdigest(),
    }
    master = {
        "schema_version": SCHEMA_VERSION,
        "name": "PSR-Bench matched two-stage curriculum with three supervision targets",
        "seed": args.seed,
        "curriculum": {
            "stage1": {"exposures": 6_144, "composition": {"L1": 6_144}},
            "stage2": {
                "exposures": 14_336,
                "composition": STAGE_2_EXPOSURES_BY_LEVEL,
                "batch_pattern": [
                    {"name": name, "composition": composition}
                    for name, composition in STAGE_2_BATCH_PATTERN
                ],
                "macro_cycle_repetitions": 64,
            },
            "dataset_shuffle": False,
            "train_dataloader_shuffle": False,
        },
        "statistics": statistics,
        "source": {
            "teacher_benchmark": str(args.teacher_benchmark.resolve()),
            "teacher_benchmark_sha256": file_sha256(args.teacher_benchmark),
        },
        "samples": master_samples,
    }
    write_json(args.output, master)
    for variant in VARIANTS:
        write_jsonl(
            artifact_path(args.output_prefix, variant, "ms_swift.jsonl"),
            sft_by_variant[variant],
        )
        write_jsonl(
            artifact_path(args.output_prefix, variant, "sidecar.jsonl"),
            sidecar_by_variant[variant],
        )
    report = {
        **statistics,
        "master_output": str(args.output.resolve()),
        "output_prefix": str(args.output_prefix.resolve()),
        "images_verified": not args.allow_missing_images,
    }
    write_json(args.output_prefix.parent / f"{args.output_prefix.name}.report.json", report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

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
    if args.command == "merge":
        report = merge_datasets(args)
    else:
        if args.type_floor <= 0:
            raise ValueError("--type-floor must be positive")
        report = finalize_dataset(args)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
