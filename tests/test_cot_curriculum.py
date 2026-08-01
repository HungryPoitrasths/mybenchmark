from __future__ import annotations

from argparse import Namespace
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path


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


def test_milestones_always_include_final_exposure() -> None:
    script = _load_script("run_cot_sft_pilot.py")

    milestones = script.milestone_steps(
        train_count=10,
        global_batch=2,
        epochs=1,
        interval=6,
    )

    assert milestones == {3: 6, 5: 10}


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
