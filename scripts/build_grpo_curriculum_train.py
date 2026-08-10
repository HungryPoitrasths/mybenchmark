#!/usr/bin/env python3
"""Build deterministic two-stage curriculum datasets for native GRPO."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_grpo_balanced_train import (
    _diverse_order,
    allocate_fair_quotas,
    select_balanced_questions,
)
from src.cot.facts import question_uid
from src.cot.images import collect_image_names
from src.cot.pipeline import format_user_prompt


SFT_CURRICULUM_SCHEMA = "predictive-spatial-cot-curriculum-v2"
STAGE1_NAME = "stage1_l1"
STAGE2_NAME = "stage2_reasoning"


def _load(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    with path.open(encoding="utf-8-sig") as handle:
        payload = json.load(handle)
    questions = payload.get("questions", payload) if isinstance(payload, dict) else payload
    if not isinstance(questions, list) or not all(
        isinstance(row, dict) for row in questions
    ):
        raise ValueError(f"{path}: expected a list of question objects")
    return payload if isinstance(payload, dict) else {}, questions


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_object(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8-sig") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return payload


def _normalized_image_name(value: object) -> str:
    return str(value or "").strip().replace("\\", "/").lstrip("./")


def _image_sequence_matches(
    question_names: list[str], manifest_paths: object
) -> bool:
    if not isinstance(manifest_paths, list) or not all(
        isinstance(value, str) for value in manifest_paths
    ):
        return False
    if len(question_names) != len(manifest_paths):
        return False
    for name, path in zip(question_names, manifest_paths):
        normalized_name = _normalized_image_name(name)
        normalized_path = _normalized_image_name(path)
        if normalized_path != normalized_name and not normalized_path.endswith(
            f"/{normalized_name}"
        ):
            return False
    return True


def _expected_level_counts(manifest: dict[str, Any], stage: str) -> dict[str, int]:
    curriculum = manifest.get("curriculum")
    if not isinstance(curriculum, dict):
        raise ValueError("curriculum manifest is missing curriculum metadata")
    key = "stage1" if stage == STAGE1_NAME else "stage2"
    stage_metadata = curriculum.get(key)
    if not isinstance(stage_metadata, dict):
        raise ValueError(f"curriculum manifest is missing {key} metadata")
    composition = stage_metadata.get("composition")
    if not isinstance(composition, dict) or not composition:
        raise ValueError(f"curriculum manifest {key} has no composition")
    return {
        str(level).upper(): int(count)
        for level, count in composition.items()
    }


def _stage_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    sources = Counter(str(row["source_question_uid"]) for row in rows)
    return {
        "exposures": len(rows),
        "selected_by_level": _counts(rows, "level"),
        "selected_by_type": _counts(rows, "type"),
        "unique_source_question_count": len(sources),
        "repeated_instance_count": len(rows) - len(sources),
        "max_source_instances": max(sources.values()),
    }


def build_matched_curriculum_questions(
    questions: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Materialize the exact SFT exposure schedule as raw GRPO questions."""
    if manifest.get("schema_version") != SFT_CURRICULUM_SCHEMA:
        raise ValueError(
            f"curriculum manifest must use schema_version {SFT_CURRICULUM_SCHEMA!r}"
        )
    seed = int(manifest.get("seed") or 0)
    profile_id = str(manifest.get("profile_id") or "").strip()
    if not profile_id:
        raise ValueError("curriculum manifest is missing profile_id")

    by_uid: dict[str, dict[str, Any]] = {}
    for index, question in enumerate(questions, start=1):
        uid = str(question.get("question_uid") or "").strip()
        if not uid:
            raise ValueError(f"candidate pool row {index} is missing question_uid")
        if uid in by_uid:
            raise ValueError(f"candidate pool contains duplicate question_uid {uid!r}")
        by_uid[uid] = question

    samples = manifest.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("curriculum manifest must contain a non-empty samples list")

    stage1: list[dict[str, Any]] = []
    stage2: list[dict[str, Any]] = []
    seen_sample_uids: set[str] = set()
    source_occurrences: Counter[str] = Counter()
    schedule_hasher = hashlib.sha256()
    curriculum = manifest["curriculum"]
    expected_stage1_count = int(curriculum["stage1"]["exposures"])
    global_batch = int(manifest.get("statistics", {}).get("global_batch") or 0)
    if expected_stage1_count <= 0 or global_batch <= 0:
        raise ValueError("curriculum manifest has invalid exposure or batch metadata")

    for expected_exposure, sample in enumerate(samples, start=1):
        if not isinstance(sample, dict):
            raise ValueError(f"curriculum sample {expected_exposure} is not an object")
        exposure_index = int(sample.get("exposure_index") or 0)
        if exposure_index != expected_exposure:
            raise ValueError(
                f"curriculum exposure order breaks at row {expected_exposure}"
            )
        sample_uid = str(sample.get("sample_uid") or "").strip()
        source_uid = str(sample.get("question_uid") or "").strip()
        if not sample_uid or not source_uid:
            raise ValueError(
                f"curriculum sample {expected_exposure} has incomplete identity metadata"
            )
        if sample_uid in seen_sample_uids:
            raise ValueError(f"duplicate curriculum sample_uid {sample_uid!r}")
        seen_sample_uids.add(sample_uid)
        question = by_uid.get(source_uid)
        if question is None:
            raise ValueError(
                f"curriculum sample {expected_exposure} references missing question_uid "
                f"{source_uid!r}"
            )

        level = str(question.get("level") or "").upper()
        question_type = str(question.get("type") or "").strip()
        if level != str(sample.get("level") or "").upper():
            raise ValueError(f"curriculum level differs from source for {source_uid!r}")
        if question_type != str(sample.get("question_type") or "").strip():
            raise ValueError(f"curriculum type differs from source for {source_uid!r}")

        image_names = collect_image_names(question)
        if not _image_sequence_matches(image_names, sample.get("images")):
            raise ValueError(f"curriculum images differ from source for {source_uid!r}")
        if format_user_prompt(question, len(image_names)) != str(
            sample.get("user_content") or ""
        ):
            raise ValueError(f"curriculum prompt differs from source for {source_uid!r}")

        occurrence = source_occurrences[source_uid]
        if int(sample.get("replica_index") or 0) != occurrence:
            raise ValueError(f"curriculum replica index differs for {source_uid!r}")
        expected_sample_uid = hashlib.sha256(
            f"{seed}|sample|{source_uid}|{occurrence}".encode("utf-8")
        ).hexdigest()
        if sample_uid != expected_sample_uid:
            raise ValueError(f"curriculum sample_uid differs for {source_uid!r}")
        source_occurrences[source_uid] += 1

        expected_stage = (
            STAGE1_NAME if exposure_index <= expected_stage1_count else STAGE2_NAME
        )
        if str(sample.get("stage") or "") != expected_stage:
            raise ValueError(f"curriculum stage differs at exposure {exposure_index}")
        if int(sample.get("global_batch_index") or 0) != (
            exposure_index - 1
        ) // global_batch:
            raise ValueError(
                f"curriculum global batch differs at exposure {exposure_index}"
            )

        instance = {
            **question,
            "question_uid": sample_uid,
            "source_question_uid": source_uid,
            "sampling_repeat_index": occurrence,
            "curriculum_exposure_index": exposure_index,
            "curriculum_global_batch_index": int(sample["global_batch_index"]),
            "curriculum_stage": expected_stage,
            "curriculum_stage2_batch_pattern": sample.get("stage2_batch_pattern"),
        }
        (stage1 if expected_stage == STAGE1_NAME else stage2).append(instance)
        schedule_hasher.update(
            f"{exposure_index}|{sample_uid}|{level}\n".encode("utf-8")
        )

    expected_schedule_hash = str(
        manifest.get("statistics", {}).get("schedule_sha256") or ""
    )
    actual_schedule_hash = schedule_hasher.hexdigest()
    if not expected_schedule_hash or actual_schedule_hash != expected_schedule_hash:
        raise ValueError("curriculum schedule SHA256 does not match its samples")

    for stage_name, rows in ((STAGE1_NAME, stage1), (STAGE2_NAME, stage2)):
        expected_counts = _expected_level_counts(manifest, stage_name)
        actual_counts = _counts(rows, "level")
        if actual_counts != dict(sorted(expected_counts.items())):
            raise ValueError(
                f"{stage_name} level composition differs: {actual_counts}"
            )
    expected_stage2_count = int(curriculum["stage2"]["exposures"])
    if len(stage1) != expected_stage1_count or len(stage2) != expected_stage2_count:
        raise ValueError("curriculum stage exposure counts do not match metadata")

    scheduled_sources = set(source_occurrences)
    if scheduled_sources != set(by_uid):
        raise ValueError(
            "curriculum source coverage differs from candidate pool: "
            f"missing={len(set(by_uid) - scheduled_sources)}, "
            f"extra={len(scheduled_sources - set(by_uid))}"
        )

    stage1_sources = {str(row["source_question_uid"]) for row in stage1}
    stage2_sources = {str(row["source_question_uid"]) for row in stage2}
    report = {
        "schema_version": "predictive-spatial-grpo-curriculum-v2",
        "strategy": "matched_sft_two_stage_schedule",
        "profile_id": profile_id,
        "seed": seed,
        "schedule_sha256": actual_schedule_hash,
        "source_unique_question_count": len(by_uid),
        "scheduled_unique_question_count": len(scheduled_sources),
        "stage_source_overlap_count": len(stage1_sources & stage2_sources),
        "stage1": _stage_report(stage1),
        "stage2": _stage_report(stage2),
    }
    return stage1, stage2, report


def _stable_rank(seed: int, scope: str, question: dict[str, Any]) -> str:
    value = f"{seed}|{scope}|{question_uid(question)}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _level_rows(
    questions: list[dict[str, Any]], level: str
) -> list[dict[str, Any]]:
    return [
        question
        for question in questions
        if str(question.get("level") or "").upper() == level
    ]


def _counts(items: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(str(item.get(key) or "missing") for item in items).items()))


def _source_uid(question: dict[str, Any]) -> str:
    return str(question.get("source_question_uid") or question_uid(question))


def select_capped_balanced_questions(
    questions: list[dict[str, Any]],
    *,
    target: int,
    max_repeat: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if max_repeat <= 0:
        raise ValueError("max repeat must be positive")
    by_type: dict[str, list[dict[str, Any]]] = {}
    seen_uids: set[str] = set()
    for question in questions:
        question_type = str(question.get("type") or "").strip()
        if not question_type:
            raise ValueError("benchmark contains a question without type")
        uid = question_uid(question)
        if uid in seen_uids:
            raise ValueError(f"duplicate source question_uid {uid!r}")
        seen_uids.add(uid)
        by_type.setdefault(question_type, []).append(question)

    unique_capacities = {
        question_type: len(rows) for question_type, rows in sorted(by_type.items())
    }
    effective_capacities = {
        question_type: count * max_repeat
        for question_type, count in unique_capacities.items()
    }
    quotas = allocate_fair_quotas(target, effective_capacities)
    selected: list[dict[str, Any]] = []
    repeat_counts: Counter[str] = Counter()
    for question_type, rows in sorted(by_type.items()):
        remaining = quotas[question_type]
        for repeat_index in range(max_repeat):
            if remaining == 0:
                break
            ordered = _diverse_order(
                list(rows),
                question_type=f"{question_type}|repeat-{repeat_index}",
                seed=seed + repeat_index,
            )
            for question in ordered[:remaining]:
                source_uid = question_uid(question)
                instance = dict(question)
                instance["source_question_uid"] = source_uid
                instance["sampling_repeat_index"] = repeat_index
                instance["question_uid"] = (
                    source_uid
                    if repeat_index == 0
                    else f"{source_uid}::repeat-{repeat_index}"
                )
                selected.append(instance)
                repeat_counts[source_uid] += 1
            remaining -= min(remaining, len(ordered))
        if remaining:
            raise AssertionError(f"could not fill quota for {question_type}")

    selected.sort(key=lambda row: _stable_rank(seed, "capped-global", row))
    selected_by_type = Counter(str(row["type"]) for row in selected)
    if len(selected) != target or dict(sorted(selected_by_type.items())) != quotas:
        raise AssertionError("capped balanced selection did not meet its quotas")
    return selected, {
        "strategy": "max_min_type_balance_with_capped_deterministic_replay",
        "seed": seed,
        "target": target,
        "max_repeat": max_repeat,
        "unique_available_count": len(seen_uids),
        "available_by_type": unique_capacities,
        "effective_capacity_by_type": effective_capacities,
        "target_by_type": quotas,
        "selected_by_type": dict(sorted(selected_by_type.items())),
        "unique_selected_source_count": len(repeat_counts),
        "repeated_instance_count": len(selected) - len(repeat_counts),
        "max_observed_repeat": max(repeat_counts.values()),
    }


def build_curriculum_questions(
    questions: list[dict[str, Any]],
    *,
    stage1_target: int,
    stage2_target: int,
    replay_target: int,
    max_repeat: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if min(stage1_target, stage2_target, replay_target) <= 0:
        raise ValueError("stage targets and replay target must be positive")
    if replay_target >= stage2_target:
        raise ValueError("replay target must be smaller than the stage-two target")

    by_level = {level: _level_rows(questions, level) for level in ("L1", "L2", "L3")}
    stage1, stage1_report = select_balanced_questions(
        by_level["L1"], target=stage1_target, seed=seed
    )
    replay, replay_report = select_balanced_questions(
        stage1, target=replay_target, seed=seed + 1
    )

    reasoning_target = stage2_target - replay_target
    reasoning, reasoning_report = select_capped_balanced_questions(
        [*by_level["L2"], *by_level["L3"]],
        target=reasoning_target,
        max_repeat=max_repeat,
        seed=seed + 2,
    )
    replay = [
        {
            **question,
            "source_question_uid": question_uid(question),
            "sampling_repeat_index": 0,
            "question_uid": question_uid(question),
        }
        for question in replay
    ]
    stage2 = [*replay, *reasoning]
    stage2.sort(key=lambda row: _stable_rank(seed, "stage2-global", row))

    stage1_uids = {question_uid(question) for question in stage1}
    stage2_instance_uids = {question_uid(question) for question in stage2}
    overlap = stage1_uids & {_source_uid(question) for question in stage2}
    if len(stage2) != stage2_target or len(stage2_instance_uids) != stage2_target:
        raise AssertionError("stage two must contain exactly the requested unique questions")
    if len(overlap) != replay_target:
        raise AssertionError("stage overlap must consist exactly of the L1 replay subset")
    if any(str(row.get("level") or "").upper() != "L1" for row in replay):
        raise AssertionError("stage-two replay contains a non-L1 question")

    report = {
        "schema_version": "predictive-spatial-grpo-curriculum-v1",
        "seed": seed,
        "available_by_level": {
            level: len(rows) for level, rows in by_level.items()
        },
        "stage1": {
            "role": "spatial_perception",
            "target": stage1_target,
            "selected_by_level": _counts(stage1, "level"),
            "selected_by_type": _counts(stage1, "type"),
            "selection": stage1_report,
        },
        "stage2": {
            "role": "reasoning_with_perception_replay",
            "target": stage2_target,
            "l1_replay_target": replay_target,
            "reasoning_target": reasoning_target,
            "max_repeat": max_repeat,
            "selected_by_level": _counts(stage2, "level"),
            "selected_by_type": _counts(stage2, "type"),
            "stage1_overlap_count": len(overlap),
            "replay_selection": replay_report,
            "reasoning_selection": reasoning_report,
        },
    }
    return stage1, stage2, report


def _write_dataset(
    path: Path,
    *,
    name: str,
    source: Path,
    curriculum_manifest: Path,
    version: str,
    curriculum: dict[str, Any],
    stage: str,
    questions: list[dict[str, Any]],
) -> None:
    payload = {
        "name": name,
        "version": version,
        "source": str(source),
        "curriculum_manifest": str(curriculum_manifest),
        "curriculum_stage": stage,
        "curriculum": curriculum,
        "questions": questions,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("output_train/mixed_train_12k.json"),
        help="The exact 12k candidate pool shared with SFT.",
    )
    parser.add_argument(
        "--curriculum-manifest",
        type=Path,
        default=Path("cot/train/mixed_train_curriculum_12k.json"),
        help="Canonical SFT exposure schedule to reproduce for GRPO.",
    )
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=Path("output_train/benchmark.json"),
        help="Original benchmark used to verify the candidate-pool provenance hash.",
    )
    parser.add_argument(
        "--stage1-output",
        type=Path,
        default=Path("output_train/grpo_curriculum_stage1_l1_6144.json"),
    )
    parser.add_argument(
        "--stage2-output",
        type=Path,
        default=Path("output_train/grpo_curriculum_stage2_reasoning_18432.json"),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    input_path = args.input.resolve()
    curriculum_manifest_path = args.curriculum_manifest.resolve()
    benchmark_path = args.benchmark.resolve()
    stage1_path = args.stage1_output.resolve()
    stage2_path = args.stage2_output.resolve()
    metadata, questions = _load(input_path)
    manifest = _load_object(curriculum_manifest_path)

    input_sha256 = _sha256(input_path)
    expected_input_sha256 = str(
        manifest.get("source", {}).get("input_dataset_sha256") or ""
    )
    if input_sha256 != expected_input_sha256:
        raise ValueError(
            "candidate-pool SHA256 differs from the SFT curriculum manifest: "
            f"{input_sha256} != {expected_input_sha256}"
        )
    benchmark_sha256 = _sha256(benchmark_path)
    expected_benchmark_sha256 = str(
        metadata.get("source", {}).get("benchmark_sha256") or ""
    )
    if benchmark_sha256 != expected_benchmark_sha256:
        raise ValueError(
            "benchmark SHA256 differs from the candidate-pool provenance: "
            f"{benchmark_sha256} != {expected_benchmark_sha256}"
        )
    profile_id = str(metadata.get("profile_id") or "")
    if profile_id != str(manifest.get("profile_id") or ""):
        raise ValueError("candidate pool and curriculum manifest profile_id values differ")

    stage1, stage2, report = build_matched_curriculum_questions(questions, manifest)
    report["source"] = {
        "benchmark": str(args.benchmark),
        "benchmark_sha256": benchmark_sha256,
        "candidate_pool": str(args.input),
        "candidate_pool_sha256": input_sha256,
        "sft_curriculum_manifest": str(args.curriculum_manifest),
        "sft_curriculum_manifest_sha256": _sha256(curriculum_manifest_path),
    }
    version = str(metadata.get("version") or "1.0")
    _write_dataset(
        stage1_path,
        name=f"PSR-Bench GRPO curriculum stage 1 L1 {len(stage1)}",
        source=args.input,
        curriculum_manifest=args.curriculum_manifest,
        version=version,
        curriculum=report,
        stage=STAGE1_NAME,
        questions=stage1,
    )
    _write_dataset(
        stage2_path,
        name=f"PSR-Bench GRPO curriculum stage 2 reasoning {len(stage2)}",
        source=args.input,
        curriculum_manifest=args.curriculum_manifest,
        version=version,
        curriculum=report,
        stage=STAGE2_NAME,
        questions=stage2,
    )
    print(
        json.dumps(
            {
                "stage1_output": str(stage1_path),
                "stage2_output": str(stage2_path),
                **report,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
