from __future__ import annotations

from argparse import Namespace
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

from src.cot.curriculum import (
    FRESH_12K_L2_EXPOSURES_BY_TYPE,
    FRESH_12K_PROFILE,
    PROFILE_SCHEMA_VERSION,
    sqrt_largest_remainder_allocation,
)


def _load_script(filename: str):
    path = Path(__file__).resolve().parents[1] / "scripts" / filename
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_stage_two_schedule_has_exact_batch_pattern_and_balanced_macrocycles() -> None:
    script = _load_script("build_mixed_cot_ablation.py")
    streams = {
        level: [
            {
                "question": {"question_uid": f"{level}-{index}"},
                "pool_replica_index": 0,
            }
            for index in range(count)
        ]
        for level, count in script.STAGE_2_EXPOSURES_BY_LEVEL.items()
    }

    schedule = script.interleave_stage2(streams)

    assert len(schedule) == 14_336
    expected = [
        ("A", {"L1": 4, "L2": 14, "L3": 14}),
        ("B", {"L1": 5, "L2": 14, "L3": 13}),
        ("A", {"L1": 4, "L2": 14, "L3": 14}),
        ("C", {"L1": 5, "L2": 13, "L3": 14}),
        ("A", {"L1": 4, "L2": 14, "L3": 14}),
        ("B", {"L1": 5, "L2": 14, "L3": 13}),
        ("C", {"L1": 5, "L2": 13, "L3": 14}),
    ]
    for batch_index in range(len(schedule) // 32):
        batch = schedule[batch_index * 32 : (batch_index + 1) * 32]
        name, counts = expected[batch_index % 7]
        assert {row[0] for row in batch} == {name}
        assert dict(Counter(row[1] for row in batch)) == counts
    for macro_index in range(64):
        macro = schedule[macro_index * 224 : (macro_index + 1) * 224]
        assert dict(Counter(row[1] for row in macro)) == {
            "L1": 32,
            "L2": 96,
            "L3": 96,
        }


def test_fresh_stage_two_schedule_has_exact_abc_pattern() -> None:
    script = _load_script("build_mixed_cot_ablation.py")
    streams = {
        level: [
            {"question": {"question_uid": f"{level}-{index}"}, "pool_replica_index": 0}
            for index in range(count)
        ]
        for level, count in FRESH_12K_PROFILE.stage2_exposures_by_level.items()
    }

    schedule = script.interleave_stage2(streams, profile=FRESH_12K_PROFILE)

    assert len(schedule) == 18_432
    for batch_index in range(576):
        batch = schedule[batch_index * 32 : (batch_index + 1) * 32]
        name, counts = FRESH_12K_PROFILE.stage2_pattern[batch_index % 3]
        assert {row[0] for row in batch} == {name}
        assert dict(Counter(row[1] for row in batch)) == counts


def test_fresh_l2_sqrt_allocation_and_uid_cap() -> None:
    script = _load_script("build_mixed_cot_ablation.py")
    candidate_counts = {
        question_type: FRESH_12K_PROFILE.candidate_quotas_by_type[question_type]
        for question_type in FRESH_12K_L2_EXPOSURES_BY_TYPE
    }
    assert sqrt_largest_remainder_allocation(7_680, candidate_counts) == (
        FRESH_12K_L2_EXPOSURES_BY_TYPE
    )
    questions = []
    for question_type, count in candidate_counts.items():
        for index in range(count):
            questions.append(
                {
                    "question_uid": f"{question_type}-{index}",
                    "type": question_type,
                    "scene_id": f"scene-{index}",
                    "moved_obj_id": index,
                    "query_obj_id": index + 10_000,
                    "obj_c_id": index + 20_000,
                    "obj_ref_id": index + 30_000,
                    "removed_obj_id": index + 40_000,
                    "obj_b_id": index + 50_000,
                }
            )

    stream, report = script.build_l2_stream(
        questions,
        target_by_type=FRESH_12K_L2_EXPOSURES_BY_TYPE,
        max_uid_exposures=4,
        seed=42,
    )

    counts = Counter(item["question"]["question_uid"] for item in stream)
    assert len(stream) == 7_680
    assert dict(Counter(item["question"]["type"] for item in stream)) == (
        FRESH_12K_L2_EXPOSURES_BY_TYPE
    )
    assert len(counts) == 2_934
    assert min(counts.values()) == 1
    assert max(counts.values()) <= 4
    assert report["max_uid_exposures"] == 4
    assert report["max_uid_exposures_by_type"] == {}


def test_l2_per_type_uid_cap_overrides_global_cap() -> None:
    script = _load_script("build_mixed_cot_ablation.py")
    questions = [
        {
            "question_uid": f"object-centric-{index}",
            "type": "object_move_object_centric",
            "scene_id": f"scene-{index}",
            "moved_obj_id": index,
            "query_obj_id": index + 100,
            "obj_ref_id": index + 200,
        }
        for index in range(3)
    ]

    stream, report = script.build_l2_stream(
        questions,
        target_by_type={"object_move_object_centric": 6},
        max_uid_exposures=4,
        max_uid_exposures_by_type={"object_move_object_centric": 2},
        seed=42,
        enforce_sqrt_allocation=False,
    )

    assert Counter(item["question"]["question_uid"] for item in stream) == {
        "object-centric-0": 2,
        "object-centric-1": 2,
        "object-centric-2": 2,
    }
    assert report["multiplicity_counts_by_type"] == {
        "object_move_object_centric": {2: 3}
    }


def test_l2_repetitions_prioritize_low_frequency_object_groups() -> None:
    script = _load_script("build_mixed_cot_ablation.py")
    questions = [
        {
            "question_uid": "rare",
            "type": "object_move_agent",
            "scene_id": "rare-scene",
            "moved_obj_id": 1,
            "query_obj_id": 2,
            "obj_c_id": 3,
        }
    ] + [
        {
            "question_uid": f"common-{index}",
            "type": "object_move_agent",
            "scene_id": "common-scene",
            "moved_obj_id": 10,
            "query_obj_id": 20,
            "obj_c_id": 30,
        }
        for index in range(2)
    ]

    stream, _report = script.build_l2_stream(
        questions,
        target_by_type={"object_move_agent": 8},
        max_uid_exposures=3,
        seed=42,
    )

    counts = Counter(item["question"]["question_uid"] for item in stream)
    assert counts["rare"] == 3
    assert sum(counts[uid] - 1 for uid in ("common-0", "common-1")) == 3


def test_milestones_always_include_final_exposure() -> None:
    script = _load_script("run_cot_sft_pilot.py")

    milestones = script.milestone_steps(
        train_count=10,
        global_batch=2,
        epochs=1,
        interval=6,
    )

    assert milestones == {3: 6, 5: 10}


def test_answer_only_export_does_not_require_teacher_variant() -> None:
    script = _load_script("build_mixed_cot_ablation.py")

    args = script.build_parser().parse_args(
        ["finalize", "mixed_train_10k.json", "--variants", "answer_only"]
    )

    assert args.variants == ["answer_only"]


def test_fresh_candidate_selection_is_exact_deduplicated_and_deterministic(
    tmp_path: Path, monkeypatch
) -> None:
    script = _load_script("build_mixed_cot_ablation.py")
    questions: list[dict] = []
    for level, question_types in script.TYPES_BY_LEVEL.items():
        for question_type in question_types:
            count = FRESH_12K_PROFILE.candidate_quotas_by_type[question_type]
            for index in range(count):
                questions.append(
                    {
                        "question_uid": f"{question_type}-{index}",
                        "level": level,
                        "type": question_type,
                        "scene_id": f"scene-{index % 10}",
                    }
                )
    questions.extend(dict(question) for question in questions[:7])
    questions.extend(
        {
            "question_uid": f"invalid-{index}",
            "level": "L1",
            "type": "direction_agent",
            "scene_id": "invalid-scene",
            "reject": True,
        }
        for index in range(78)
    )
    benchmark = tmp_path / "benchmark.json"
    benchmark.write_text(json.dumps({"questions": questions}), encoding="utf-8")

    def fake_build_fact_record(question: dict) -> SimpleNamespace:
        if question.get("reject"):
            raise script.FactExtractionError("unsupported_direction", "synthetic")
        return SimpleNamespace(
            to_dict=lambda: {
                "question_uid": question["question_uid"],
                "question_type": question["type"],
                "signature_id": f"signature-{question['type']}",
                "answer_letters": ["A"],
                "facts": {
                    "level": question["level"],
                    "scene_id": question["scene_id"],
                },
            }
        )

    monkeypatch.setattr(script, "build_fact_record", fake_build_fact_record)
    monkeypatch.setattr(script, "validate_fact_consistency", lambda _record: None)
    monkeypatch.setattr(script, "validate_answer_mapping", lambda _question, _record: None)

    def run_selection(suffix: str) -> tuple[dict, Path]:
        output = tmp_path / f"selected-{suffix}.json"
        result = script.select_fresh_candidates(
            Namespace(
                benchmark=benchmark,
                expected_benchmark_sha256="",
                seed=42,
                output=output,
                mirror_output=None,
                report=tmp_path / f"report-{suffix}.json",
            )
        )
        return result, output

    first, first_output = run_selection("first")
    second, second_output = run_selection("second")

    assert first["raw_count"] == 12_085
    assert first["validated_count"] == 12_007
    assert first["rejected_count"] == 78
    assert first["rejection_counts"] == {"unsupported_direction": 78}
    assert first["duplicate_uid_count"] == 7
    assert first["selected_count"] == 12_000
    assert first["selected_by_level"] == {"L1": 4_533, "L2": 2_934, "L3": 4_533}
    assert first["selected_by_type"] == dict(
        sorted(FRESH_12K_PROFILE.candidate_quotas_by_type.items())
    )
    assert second["selected_by_type"] == first["selected_by_type"]
    assert first_output.read_bytes() == second_output.read_bytes()


def test_strict_object_centric_shortfall_keeps_other_quotas_and_raises_only_repeat_cap() -> None:
    script = _load_script("build_mixed_cot_ablation.py")
    available = dict(FRESH_12K_PROFILE.candidate_quotas_by_type or {})
    available["object_move_occlusion"] = 212
    available["object_move_object_centric"] = 197

    quotas = script.fresh_candidate_type_quotas(available)

    assert quotas["object_move_object_centric"] == 197
    assert quotas["object_move_occlusion"] == 212
    for question_type, original in FRESH_12K_PROFILE.candidate_quotas_by_type.items():
        if question_type not in {"object_move_object_centric", "object_move_occlusion"}:
            assert quotas[question_type] == original
    assert script.dynamic_l2_max_uid_exposures(FRESH_12K_PROFILE, quotas) == 6


def _curriculum_rows() -> tuple[list[dict], list[dict], list[dict]]:
    samples: list[dict] = []
    train: list[dict] = []
    sidecar: list[dict] = []
    levels = ["L1"] * 6_144
    pattern = [
        ("A", {"L1": 4, "L2": 14, "L3": 14}),
        ("B", {"L1": 5, "L2": 14, "L3": 13}),
        ("A", {"L1": 4, "L2": 14, "L3": 14}),
        ("C", {"L1": 5, "L2": 13, "L3": 14}),
        ("A", {"L1": 4, "L2": 14, "L3": 14}),
        ("B", {"L1": 5, "L2": 14, "L3": 13}),
        ("C", {"L1": 5, "L2": 13, "L3": 14}),
    ]
    stage2_names: list[str] = []
    for _ in range(64):
        for name, composition in pattern:
            batch_levels = [
                level
                for level in ("L1", "L2", "L3")
                for _ in range(composition[level])
            ]
            levels.extend(batch_levels)
            stage2_names.extend([name] * 32)

    for index, level in enumerate(levels, start=1):
        sample_uid = f"sample-{index:05d}"
        question_uid = f"question-{index % 10_000:05d}"
        stage2_index = index - 6_145
        sample = {
            "sample_uid": sample_uid,
            "question_uid": question_uid,
            "exposure_index": index,
            "level": level,
            "stage2_batch_pattern": stage2_names[stage2_index] if stage2_index >= 0 else None,
            "user_content": "<image>\nSynthetic prompt",
            "images": ["image.jpg"],
        }
        samples.append(sample)
        train.append(
            {
                "sample_uid": sample_uid,
                "question_uid": question_uid,
                "curriculum_exposure": index,
                "messages": [
                    {"role": "user", "content": sample["user_content"]},
                    {"role": "assistant", "content": "Answer: A"},
                ],
                "images": sample["images"],
            }
        )
        sidecar.append(
            {
                "sample_uid": sample_uid,
                "question_uid": question_uid,
                "curriculum_exposure": index,
            }
        )
    return samples, train, sidecar


def test_curriculum_manifest_verifies_order_hash_and_640_steps(tmp_path: Path) -> None:
    script = _load_script("run_cot_sft_pilot.py")
    samples, train, sidecar = _curriculum_rows()
    digest = hashlib.sha256()
    for sample in samples:
        digest.update(
            f"{sample['exposure_index']}|{sample['sample_uid']}|{sample['level']}\n".encode()
        )
    manifest = tmp_path / "curriculum.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": script.CURRICULUM_SCHEMA_VERSION,
                "statistics": {"schedule_sha256": digest.hexdigest()},
                "samples": samples,
            }
        ),
        encoding="utf-8",
    )

    report = script.validate_curriculum_manifest(
        manifest,
        train_rows=train,
        train_sidecar=sidecar,
    )

    assert report["optimizer_steps"] == 640
    assert report["level_counts"] == {"L1": 8_192, "L2": 6_144, "L3": 6_144}
    assert report["dataset_shuffle"] is False
    assert report["train_dataloader_shuffle"] is False


def test_fresh_curriculum_manifest_verifies_24576_rows_and_768_steps(
    tmp_path: Path,
) -> None:
    script = _load_script("run_cot_sft_pilot.py")
    levels = ["L1"] * FRESH_12K_PROFILE.stage1_exposures
    stage2_names: list[str] = []
    for _ in range(FRESH_12K_PROFILE.stage2_pattern_repetitions):
        for name, composition in FRESH_12K_PROFILE.stage2_pattern:
            batch_levels = [
                level
                for level in ("L1", "L2", "L3")
                for _ in range(composition[level])
            ]
            levels.extend(batch_levels)
            stage2_names.extend([name] * FRESH_12K_PROFILE.global_batch)

    samples: list[dict] = []
    train: list[dict] = []
    sidecar: list[dict] = []
    digest = hashlib.sha256()
    for index, level in enumerate(levels, start=1):
        sample_uid = f"fresh-sample-{index:05d}"
        question_uid = f"fresh-question-{index % 12_000:05d}"
        stage2_index = index - FRESH_12K_PROFILE.stage1_exposures - 1
        sample = {
            "sample_uid": sample_uid,
            "question_uid": question_uid,
            "exposure_index": index,
            "level": level,
            "stage2_batch_pattern": (
                stage2_names[stage2_index] if stage2_index >= 0 else None
            ),
            "user_content": "<image>\nSynthetic prompt",
            "images": ["image.jpg"],
        }
        samples.append(sample)
        train.append(
            {
                "sample_uid": sample_uid,
                "question_uid": question_uid,
                "curriculum_exposure": index,
                "messages": [
                    {"role": "user", "content": sample["user_content"]},
                    {"role": "assistant", "content": "Answer: A"},
                ],
                "images": sample["images"],
            }
        )
        sidecar.append(
            {
                "sample_uid": sample_uid,
                "question_uid": question_uid,
                "curriculum_exposure": index,
            }
        )
        digest.update(f"{index}|{sample_uid}|{level}\n".encode())

    manifest = tmp_path / "fresh_curriculum.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": PROFILE_SCHEMA_VERSION,
                "profile_id": FRESH_12K_PROFILE.profile_id,
                "statistics": {"schedule_sha256": digest.hexdigest()},
                "samples": samples,
            }
        ),
        encoding="utf-8",
    )

    report = script.validate_curriculum_manifest(
        manifest,
        train_rows=train,
        train_sidecar=sidecar,
    )

    assert report["profile_id"] == FRESH_12K_PROFILE.profile_id
    assert report["optimizer_steps"] == 768
    assert report["level_counts"] == {"L1": 9_216, "L2": 7_680, "L3": 7_680}


def test_fresh_curriculum_command_has_no_adapter_and_disables_shuffle(tmp_path: Path) -> None:
    script = _load_script("run_cot_sft_pilot.py")
    args = Namespace(
        swift_bin="swift",
        model="Qwen/Qwen3-VL-4B-Instruct",
        train_dataset=tmp_path / "train.jsonl",
        monitor_dataset=tmp_path / "val.jsonl",
        epochs=1,
        per_device_batch_size=2,
        gradient_accumulation_steps=8,
        optimizer="adamw_torch",
        learning_rate=1e-4,
        aligner_learning_rate=1e-5,
        lora_rank=32,
        lora_alpha=64,
        lora_dropout=0.05,
        weight_decay=0.01,
        warmup_ratio=0.03,
        max_grad_norm=1.0,
        max_length=8192,
        max_pixels=786432,
        attn_impl="sdpa",
        dataloader_workers=4,
        dataset_workers=4,
        seed=42,
        output_dir=tmp_path / "output",
        resume_from_checkpoint=None,
        initial_adapter=None,
        smoke_test_steps=None,
        curriculum_manifest=tmp_path / "curriculum.json",
    )

    command = script.build_sft_command(args)

    assert "--adapters" not in command
    assert "--resume_from_checkpoint" not in command
    assert command[command.index("--dataset_shuffle") + 1] == "false"
    assert command[command.index("--train_dataloader_shuffle") + 1] == "false"
    assert command[command.index("--group_by_length") + 1] == "false"
    assert command[command.index("--learning_rate") + 1] == "0.0001"
    assert command[command.index("--aligner_lr") + 1] == "1e-05"
