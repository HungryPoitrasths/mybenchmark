from __future__ import annotations

import hashlib
import importlib.util
import json
from collections import Counter
from pathlib import Path

import pytest

from src.cot.evaluation import parse_relaxed_answer, parse_strict_answer
from src.cot.grpo_training import (
    _cast_adam_moments,
    answer_rewards,
    format_rewards,
    load_jsonl,
    prepare_balanced_benchmark_grpo_dataset,
    prepare_grpo_dataset,
)


def test_cast_adam_moments_restores_checkpoint_dtype() -> None:
    import torch

    parameter = torch.nn.Parameter(torch.ones(2, dtype=torch.float32))
    optimizer = torch.optim.AdamW([parameter])
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()

    assert _cast_adam_moments(optimizer, torch.bfloat16) == 2
    state = optimizer.state[parameter]
    assert state["exp_avg"].dtype == torch.bfloat16
    assert state["exp_avg_sq"].dtype == torch.bfloat16


def _load_launcher():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_grpo_qwen3vl_4b.py"
    spec = importlib.util.spec_from_file_location("run_grpo_qwen3vl_4b", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_balanced_builder():
    path = Path(__file__).resolve().parents[1] / "scripts" / "build_grpo_balanced_train.py"
    spec = importlib.util.spec_from_file_location("build_grpo_balanced_train", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_curriculum_builder():
    path = Path(__file__).resolve().parents[1] / "scripts" / "build_grpo_curriculum_train.py"
    spec = importlib.util.spec_from_file_location("build_grpo_curriculum_train", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_curriculum_launcher():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_grpo_curriculum_qwen3vl_4b.py"
    spec = importlib.util.spec_from_file_location("run_grpo_curriculum_qwen3vl_4b", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_grpo_external_plugin():
    path = Path(__file__).resolve().parents[1] / "src" / "cot" / "grpo_training.py"
    spec = importlib.util.spec_from_file_location("grpo_training_external_plugin", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row))
            handle.write("\n")


def _dataset_row(image: Path, *, answer: str = "B") -> dict:
    return {
        "messages": [
            {"role": "user", "content": "<image>\nQuestion?\nA. left\nB. right"},
            {"role": "assistant", "content": f"Reasoning.\nAnswer: {answer}"},
        ],
        "images": [str(image)],
        "question_uid": "question-1",
    }


def _sidecar_row(*, answer: str = "B") -> dict:
    return {
        "question_uid": "question-1",
        "question_type": "direction_agent",
        "signature_id": "L1_direction_agent.single_axis.right",
        "answer_letters": answer.split(),
        "option_count": 2,
        "multi_select": False,
    }


def _benchmark_question(
    image: Path,
    *,
    question_type: str,
    level: str,
    index: int,
) -> dict:
    return {
        "level": level,
        "type": question_type,
        "question": f"Question {question_type} {index}?",
        "options": ["left", "right"],
        "answer": "A" if index % 2 == 0 else "B",
        "multi_select": False,
        "scene_id": f"scene{index:04d}_00",
        "image_name": image.name,
        "image_path": str(image),
    }


def _matched_manifest(builder, questions: list[dict], source_order: list[str]) -> dict:
    by_uid = {str(question["question_uid"]): question for question in questions}
    occurrences: Counter[str] = Counter()
    samples = []
    schedule_hasher = hashlib.sha256()
    stage1_exposures = 2
    for exposure_index, source_uid in enumerate(source_order, start=1):
        question = by_uid[source_uid]
        occurrence = occurrences[source_uid]
        occurrences[source_uid] += 1
        sample_uid = hashlib.sha256(
            f"42|sample|{source_uid}|{occurrence}".encode("utf-8")
        ).hexdigest()
        stage = "stage1_l1" if exposure_index <= stage1_exposures else "stage2_reasoning"
        level = str(question["level"])
        samples.append(
            {
                "sample_uid": sample_uid,
                "question_uid": source_uid,
                "replica_index": occurrence,
                "pool_replica_index": 0,
                "exposure_index": exposure_index,
                "global_batch_index": exposure_index - 1,
                "stage": stage,
                "stage2_batch_pattern": None if stage == "stage1_l1" else "A",
                "level": level,
                "question_type": question["type"],
                "source": "benchmark_fresh_12k",
                "user_content": builder.format_user_prompt(question, 1),
                "images": [question["image_name"]],
                "signature_id": question["type"],
                "targets": {},
            }
        )
        schedule_hasher.update(
            f"{exposure_index}|{sample_uid}|{level}\n".encode("utf-8")
        )
    return {
        "schema_version": "predictive-spatial-cot-curriculum-v2",
        "profile_id": "test-profile",
        "seed": 42,
        "curriculum": {
            "profile_id": "test-profile",
            "stage1": {"exposures": 2, "composition": {"L1": 2}},
            "stage2": {
                "exposures": 3,
                "composition": {"L1": 1, "L2": 1, "L3": 1},
            },
        },
        "statistics": {
            "global_batch": 1,
            "schedule_sha256": schedule_hasher.hexdigest(),
        },
        "samples": samples,
    }


def test_prepare_grpo_dataset_removes_assistant_and_uses_sidecar_answer(
    tmp_path: Path,
) -> None:
    image = tmp_path / "image.jpg"
    image.write_bytes(b"image")
    dataset = tmp_path / "train.jsonl"
    sidecar = tmp_path / "sidecar.jsonl"
    output = tmp_path / "prepared.jsonl"
    _write_jsonl(dataset, [_dataset_row(image)])
    _write_jsonl(sidecar, [_sidecar_row()])

    report = prepare_grpo_dataset(dataset, sidecar, output)

    rows = load_jsonl(output)
    assert report["row_count"] == 1
    assert report["max_images"] == 1
    assert rows[0]["solution"] == "B"
    assert rows[0]["option_count"] == 2
    assert [message["role"] for message in rows[0]["messages"]] == ["user"]
    assert "Reasoning" not in json.dumps(rows[0])


def test_prepare_grpo_dataset_rejects_answer_or_image_mismatch(tmp_path: Path) -> None:
    image = tmp_path / "image.jpg"
    image.write_bytes(b"image")
    dataset = tmp_path / "train.jsonl"
    sidecar = tmp_path / "sidecar.jsonl"
    output = tmp_path / "prepared.jsonl"
    _write_jsonl(dataset, [_dataset_row(image, answer="A")])
    _write_jsonl(sidecar, [_sidecar_row(answer="B")])

    with pytest.raises(ValueError, match="assistant answer does not match"):
        prepare_grpo_dataset(dataset, sidecar, output)

    row = _dataset_row(image)
    row["messages"][0]["content"] = "Question without an image token"
    _write_jsonl(dataset, [row])
    with pytest.raises(ValueError, match="image placeholders"):
        prepare_grpo_dataset(dataset, sidecar, output)


def test_benchmark_builder_uses_every_preselected_question(tmp_path: Path) -> None:
    image = tmp_path / "image.jpg"
    image.write_bytes(b"image")
    benchmark = tmp_path / "benchmark.json"
    output = tmp_path / "prepared.jsonl"
    questions = [
        *[
            _benchmark_question(
                image,
                question_type="direction_agent",
                level="L1",
                index=index,
            )
            for index in range(3)
        ],
        *[
            _benchmark_question(
                image,
                question_type="distance",
                level="L1",
                index=index + 10,
            )
            for index in range(2)
        ],
        _benchmark_question(
            image,
            question_type="attachment_move",
            level="L3",
            index=20,
        ),
    ]
    benchmark.write_text(json.dumps({"questions": questions}), encoding="utf-8")

    report = prepare_balanced_benchmark_grpo_dataset(
        benchmark,
        output,
        samples_per_type=None,
        seed=42,
    )

    rows = load_jsonl(output)
    assert report["sampling_mode"] == "all_questions"
    assert report["samples_per_type"] is None
    assert report["selected_by_type"] == {
        "attachment_move": 1,
        "direction_agent": 3,
        "distance": 2,
    }
    assert report["selected_count"] == 6
    assert all(len(row["messages"]) == 1 for row in rows)
    assert all(row["messages"][0]["content"].count("<image>") == 1 for row in rows)


def test_grpo_subset_builder_allocates_exact_max_min_quotas() -> None:
    builder = _load_balanced_builder()

    quotas = builder.allocate_fair_quotas(
        10,
        {"rare": 1, "medium": 4, "large_a": 20, "large_b": 20},
    )

    assert quotas == {"large_a": 3, "large_b": 3, "medium": 3, "rare": 1}


def test_curriculum_builder_replays_stage1_l1_and_mixes_l2_l3(tmp_path: Path) -> None:
    builder = _load_curriculum_builder()
    image = tmp_path / "image.jpg"
    image.write_bytes(b"image")
    questions = []
    index = 0
    type_specs = {
        "L1": {"direction_agent": 8, "distance": 8},
        "L2": {"object_move_agent": 3, "object_move_distance": 3},
        "L3": {"attachment_chain": 2, "coordinate_rotation_agent": 8},
    }
    for level, counts in type_specs.items():
        for question_type, count in counts.items():
            for _ in range(count):
                questions.append(
                    _benchmark_question(
                        image,
                        question_type=question_type,
                        level=level,
                        index=index,
                    )
                )
                index += 1

    stage1, stage2, report = builder.build_curriculum_questions(
        questions,
        stage1_target=12,
        stage2_target=12,
        replay_target=2,
        max_repeat=2,
        seed=42,
    )

    stage1_uids = {builder.question_uid(row) for row in stage1}
    replay_rows = [row for row in stage2 if row["source_question_uid"] in stage1_uids]
    assert report["stage1"]["selected_by_level"] == {"L1": 12}
    assert report["stage2"]["selected_by_level"] == {"L1": 2, "L2": 4, "L3": 6}
    assert report["stage2"]["reasoning_selection"]["max_observed_repeat"] == 2
    assert report["stage2"]["reasoning_selection"]["repeated_instance_count"] == 1
    assert len(replay_rows) == 2
    assert all(row["level"] == "L1" for row in replay_rows)
    assert len({builder.question_uid(row) for row in stage2}) == len(stage2)


def test_matched_curriculum_builder_reproduces_sft_exposure_order(
    tmp_path: Path,
) -> None:
    builder = _load_curriculum_builder()
    image = tmp_path / "image.jpg"
    image.write_bytes(b"image")
    questions = [
        {
            **_benchmark_question(
                image, question_type="direction_agent", level="L1", index=1
            ),
            "question_uid": "source-l1",
        },
        {
            **_benchmark_question(
                image, question_type="object_move_agent", level="L2", index=2
            ),
            "question_uid": "source-l2",
        },
        {
            **_benchmark_question(
                image, question_type="attachment_chain", level="L3", index=3
            ),
            "question_uid": "source-l3",
        },
    ]
    source_order = ["source-l1", "source-l1", "source-l1", "source-l2", "source-l3"]
    manifest = _matched_manifest(builder, questions, source_order)

    stage1, stage2, report = builder.build_matched_curriculum_questions(
        questions, manifest
    )

    assert [row["source_question_uid"] for row in stage1] == source_order[:2]
    assert [row["source_question_uid"] for row in stage2] == source_order[2:]
    assert [row["question_uid"] for row in [*stage1, *stage2]] == [
        sample["sample_uid"] for sample in manifest["samples"]
    ]
    assert [row["sampling_repeat_index"] for row in [*stage1, *stage2]][:3] == [
        0,
        1,
        2,
    ]
    assert report["stage1"]["selected_by_level"] == {"L1": 2}
    assert report["stage2"]["selected_by_level"] == {"L1": 1, "L2": 1, "L3": 1}
    assert report["scheduled_unique_question_count"] == 3
    assert report["schedule_sha256"] == manifest["statistics"]["schedule_sha256"]

    broken = json.loads(json.dumps(manifest))
    broken["samples"][3]["images"] = ["different.jpg"]
    with pytest.raises(ValueError, match="curriculum images differ"):
        builder.build_matched_curriculum_questions(questions, broken)


def test_curriculum_real_stage2_quotas_cap_rare_types_at_four_repeats() -> None:
    builder = _load_curriculum_builder()
    capacities = {
        "object_move_agent": 205,
        "object_move_allocentric": 98,
        "object_move_distance": 111,
        "object_move_object_centric": 127,
        "object_rotate_object_centric": 120,
        "attachment_chain": 50,
        "attachment_move": 30,
        "coordinate_rotation_agent": 5991,
        "coordinate_rotation_allocentric": 3524,
        "coordinate_rotation_object_centric": 10280,
    }

    quotas = builder.allocate_fair_quotas(
        3200,
        {question_type: count * 4 for question_type, count in capacities.items()},
    )

    assert quotas == {
        "attachment_chain": 200,
        "attachment_move": 120,
        "coordinate_rotation_agent": 360,
        "coordinate_rotation_allocentric": 360,
        "coordinate_rotation_object_centric": 360,
        "object_move_agent": 360,
        "object_move_allocentric": 360,
        "object_move_distance": 360,
        "object_move_object_centric": 360,
        "object_rotate_object_centric": 360,
    }


def test_benchmark_preparer_preserves_materialized_repeat_audit_fields(
    tmp_path: Path,
) -> None:
    image = tmp_path / "image.jpg"
    image.write_bytes(b"image")
    source = _benchmark_question(
        image,
        question_type="attachment_move",
        level="L3",
        index=1,
    )
    source_uid = "source-question"
    first = {
        **source,
        "question_uid": source_uid,
        "source_question_uid": source_uid,
        "sampling_repeat_index": 0,
    }
    repeated = {
        **source,
        "question_uid": f"{source_uid}::repeat-1",
        "source_question_uid": source_uid,
        "sampling_repeat_index": 1,
    }
    benchmark = tmp_path / "benchmark.json"
    output = tmp_path / "prepared.jsonl"
    benchmark.write_text(json.dumps({"questions": [first, repeated]}), encoding="utf-8")

    report = prepare_balanced_benchmark_grpo_dataset(
        benchmark,
        output,
        samples_per_type=None,
        seed=42,
    )

    rows = load_jsonl(output)
    assert [row["source_question_uid"] for row in rows] == [source_uid, source_uid]
    assert sorted(row["sampling_repeat_index"] for row in rows) == [0, 1]
    assert report["unique_source_question_count"] == 1
    assert report["repeated_instance_count"] == 1
    assert report["max_source_instances"] == 2


def test_benchmark_preparer_can_preserve_explicit_curriculum_order(
    tmp_path: Path,
) -> None:
    image = tmp_path / "image.jpg"
    image.write_bytes(b"image")
    questions = []
    for uid, question_type, level, index in (
        ("uid-z", "direction_agent", "L1", 1),
        ("uid-a", "object_move_agent", "L2", 2),
        ("uid-m", "attachment_chain", "L3", 3),
    ):
        questions.append(
            {
                **_benchmark_question(
                    image,
                    question_type=question_type,
                    level=level,
                    index=index,
                ),
                "question_uid": uid,
                "source_question_uid": f"source-{uid}",
            }
        )
    benchmark = tmp_path / "curriculum.json"
    output = tmp_path / "prepared.jsonl"
    benchmark.write_text(json.dumps({"questions": questions}), encoding="utf-8")

    report = prepare_balanced_benchmark_grpo_dataset(
        benchmark,
        output,
        samples_per_type=None,
        seed=42,
        preserve_input_order=True,
    )

    assert [row["question_uid"] for row in load_jsonl(output)] == [
        "uid-z",
        "uid-a",
        "uid-m",
    ]
    assert report["sampling_mode"] == "all_questions_in_source_order"
    assert report["input_order_preserved"] is True
    with pytest.raises(ValueError, match="cannot be combined"):
        prepare_balanced_benchmark_grpo_dataset(
            benchmark,
            output,
            samples_per_type=1,
            seed=42,
            preserve_input_order=True,
        )


def test_grpo_plugin_supports_ms_swift_top_level_import() -> None:
    plugin = _load_grpo_external_plugin()

    assert plugin.orms["psr_answer"] is plugin.PSRAnswerReward
    assert plugin.orms["psr_format"] is plugin.PSRFormatReward


def test_answer_and_format_rewards_are_independent() -> None:
    completions = [
        "<think>Look right.</think><answer>B</answer>",
        "Reasoning without XML.\nAnswer: B",
        "<think>Look left.</think><answer>A</answer>",
        "<think></think><answer>B</answer>",
    ]
    solutions = ["B"] * 4
    option_counts = [2] * 4
    multi_select = [False] * 4

    assert answer_rewards(completions, solutions, option_counts, multi_select) == [
        1.0,
        1.0,
        0.0,
        1.0,
    ]
    assert format_rewards(completions, option_counts, multi_select) == [
        1.0,
        0.0,
        1.0,
        0.0,
    ]


def test_rewards_validate_multiselect_and_option_range() -> None:
    completions = [
        "<think>Both apply.</think><answer>A C</answer>",
        "<think>Duplicate.</think><answer>A A</answer>",
        "<think>Out of range.</think><answer>D</answer>",
    ]
    assert answer_rewards(
        completions,
        ["A C", "A C", "A C"],
        [3, 3, 3],
        [True, True, True],
    ) == [1.0, 0.0, 0.0]
    assert format_rewards(completions, [3, 3, 3], [True, True, True]) == [
        1.0,
        0.0,
        0.0,
    ]


def test_evaluation_accepts_legacy_and_strict_r1_formats() -> None:
    assert parse_strict_answer(
        "Reasoning.\nAnswer: B", option_count=4, multi_select=False
    ) == "B"
    assert parse_strict_answer(
        "<think>Reasoning.</think><answer>B</answer>",
        option_count=4,
        multi_select=False,
    ) == "B"
    assert parse_strict_answer(
        "prefix<think>Reasoning.</think><answer>B</answer>",
        option_count=4,
        multi_select=False,
    ) is None
    assert parse_relaxed_answer(
        "prefix <answer>b</answer>", option_count=4, multi_select=False
    ) == "B"


def test_launcher_builds_native_two_gpu_lora_grpo_command(tmp_path: Path) -> None:
    launcher = _load_launcher()
    args = launcher.parse_args(
        [
            "--output-dir",
            str(tmp_path / "output"),
            "--max-steps",
            "2",
        ]
    )
    devices = launcher.validate_args(args)
    command = launcher.build_swift_command(
        args,
        prepared_dataset=tmp_path / "prepared.jsonl",
        resume_checkpoint=None,
    )

    assert devices == ["0", "1"]
    assert args.benchmark.name == "grpo_balanced_2k.json"
    assert command[:3] == ["swift", "rlhf", "--rlhf_type"]
    assert command[command.index("--model") + 1] == "Qwen/Qwen3-VL-4B-Instruct"
    assert command[command.index("--optim") + 1] == "adamw_torch_fused"
    assert command[command.index("--reward_funcs") + 1 : command.index("--reward_weights")] == [
        "psr_answer",
        "psr_format",
    ]
    assert command[command.index("--freeze_vit") + 1] == "true"
    assert command[command.index("--freeze_aligner") + 1] == "true"
    assert command[command.index("--lora_dtype") + 1] == "bfloat16"
    assert command[command.index("--vllm_enable_lora") + 1] == "true"
    assert command[command.index("--vllm_mm_processor_cache_gb") + 1] == "0.0"
    assert command[command.index("--max_steps") + 1] == "2"
    assert "--deepspeed" not in command
    args.preserve_dataset_order = True
    ordered_command = launcher.build_swift_command(
        args,
        prepared_dataset=tmp_path / "prepared.jsonl",
        resume_checkpoint=None,
    )
    assert ordered_command[ordered_command.index("--dataset_shuffle") + 1] == "false"
    assert (
        ordered_command[ordered_command.index("--train_dataloader_shuffle") + 1]
        == "false"
    )
    assert ordered_command[ordered_command.index("--group_by_length") + 1] == "false"

    args.optim = "adamw_torch"
    args.resume_optimizer_state_dtype = "bfloat16"
    environment = launcher.build_environment(args, devices)
    assert environment["PSR_RESUME_ADAM_STATE_DTYPE"] == "bfloat16"

    args.deepspeed = "zero2"
    deepspeed_command = launcher.build_swift_command(
        args,
        prepared_dataset=tmp_path / "prepared.jsonl",
        resume_checkpoint=None,
    )
    assert deepspeed_command[deepspeed_command.index("--deepspeed") + 1] == "zero2"

    adapter = tmp_path / "stage1" / "checkpoint-10"
    adapter.mkdir(parents=True)
    args.adapter = adapter
    args.reference_adapter = adapter
    curriculum_command = launcher.build_swift_command(
        args,
        prepared_dataset=tmp_path / "prepared.jsonl",
        resume_checkpoint=None,
    )
    assert curriculum_command[curriculum_command.index("--adapters") + 1] == str(
        adapter.resolve()
    )
    assert curriculum_command[curriculum_command.index("--ref_adapters") + 1] == str(
        adapter.resolve()
    )

    resumed_command = launcher.build_swift_command(
        args,
        prepared_dataset=tmp_path / "prepared.jsonl",
        resume_checkpoint=tmp_path / "rlhf" / "checkpoint-1000",
    )
    assert "--adapters" not in resumed_command
    assert resumed_command[resumed_command.index("--ref_adapters") + 1] == str(
        adapter.resolve()
    )


def test_launcher_accepts_gpu_uuids_for_mixed_gpu_hosts(tmp_path: Path) -> None:
    launcher = _load_launcher()
    gpu0 = "GPU-0bf9db65-9cb7-2634-c3b4-46f57db82753"
    gpu1 = "GPU-f60c87d6-dc9d-315f-12ff-c2d5fccfeade"
    args = launcher.parse_args(
        [
            "--output-dir",
            str(tmp_path / "output"),
            "--devices",
            f"{gpu0},{gpu1}",
        ]
    )

    devices = launcher.validate_args(args)
    environment = launcher.build_environment(args, devices)

    assert devices == [gpu0, gpu1]
    assert environment["CUDA_VISIBLE_DEVICES"] == f"{gpu0},{gpu1}"


def test_latest_checkpoint_finds_ms_swift_version_subdirectory(tmp_path: Path) -> None:
    launcher = _load_launcher()
    (tmp_path / "v0-run" / "checkpoint-2").mkdir(parents=True)
    expected = tmp_path / "v1-run" / "checkpoint-7"
    expected.mkdir(parents=True)

    assert launcher._latest_checkpoint(tmp_path) == expected.resolve()


def test_curriculum_launcher_resets_optimizer_and_loads_stage1_adapter(
    tmp_path: Path,
) -> None:
    launcher = _load_curriculum_launcher()
    stage1_data = tmp_path / "stage1.json"
    stage2_data = tmp_path / "stage2.json"
    adapter = tmp_path / "output" / "stage1_l1_perception" / "v0" / "checkpoint-10"
    adapter.mkdir(parents=True)
    args = launcher.parse_args(
        [
            "--stage1-benchmark",
            str(stage1_data),
            "--stage2-benchmark",
            str(stage2_data),
            "--output-root",
            str(tmp_path / "output"),
            "--devices",
            "5,6",
        ]
    )

    stage1_command = launcher.build_stage_command(args, stage=1)
    stage2_command = launcher.build_stage_command(args, stage=2, adapter=adapter)

    assert "--adapter" not in stage1_command
    assert stage1_command[stage1_command.index("--learning-rate") + 1] == "1e-05"
    assert "--preserve-dataset-order" in stage1_command
    assert stage1_command[stage1_command.index("--prepared-dataset") + 1].endswith(
        "stage1_l1_6144.grpo.jsonl"
    )
    assert stage2_command[stage2_command.index("--learning-rate") + 1] == "5e-06"
    assert "--preserve-dataset-order" in stage2_command
    assert stage2_command[stage2_command.index("--prepared-dataset") + 1].endswith(
        "stage2_reasoning_18432.grpo.jsonl"
    )
    assert stage2_command[stage2_command.index("--lora-dtype") + 1] == "bfloat16"
    assert stage2_command[stage2_command.index("--adapter") + 1] == str(adapter.resolve())
    assert stage2_command[stage2_command.index("--reference-adapter") + 1] == str(
        adapter.resolve()
    )
    assert stage2_command[stage2_command.index("--save-strategy") + 1] == "steps"
    assert stage2_command[stage2_command.index("--save-steps") + 1] == "250"
    assert stage2_command[
        stage2_command.index("--vllm-mm-processor-cache-gb") + 1
    ] == "0.0"
    assert "--resume-from-checkpoint" not in stage2_command

    args.stage1_resume_from_checkpoint = Path("latest")
    resumed_stage1_command = launcher.build_stage_command(args, stage=1)
    assert resumed_stage1_command[
        resumed_stage1_command.index("--resume-from-checkpoint") + 1
    ] == "latest"
    resumed_stage2_command = launcher.build_stage_command(
        args, stage=2, adapter=adapter
    )
    assert "--resume-from-checkpoint" not in resumed_stage2_command


def test_launcher_dry_run_writes_manifest_without_swift(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    launcher = _load_launcher()
    image = tmp_path / "image.jpg"
    image.write_bytes(b"image")
    dataset = tmp_path / "train.jsonl"
    sidecar = tmp_path / "sidecar.jsonl"
    output_dir = tmp_path / "output"
    _write_jsonl(dataset, [_dataset_row(image)])
    _write_jsonl(sidecar, [_sidecar_row()])

    result = launcher.main(
        [
            "--train-dataset",
            str(dataset),
            "--train-sidecar",
            str(sidecar),
            "--output-dir",
            str(output_dir),
            "--dry-run",
        ]
    )

    assert result == 0
    manifest = json.loads((output_dir / "grpo_manifest.json").read_text(encoding="utf-8"))
    assert manifest["base_variant"] == "instruct"
    assert manifest["global_completion_batch"] == 8
    assert manifest["rollout_prompts_per_batch"] == 1
    assert manifest["dataset"]["row_count"] == 1
    assert "swift rlhf" in capsys.readouterr().out


def test_launcher_can_reuse_verified_prepared_dataset(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    launcher = _load_launcher()
    prepared = tmp_path / "prepared.jsonl"
    output_dir = tmp_path / "output"
    _write_jsonl(
        prepared,
        [
            {
                "messages": [{"role": "user", "content": "<image>\nQuestion?"}],
                "images": ["/verified/image.jpg"],
                "solution": "A",
                "option_count": 2,
                "multi_select": False,
                "question_uid": "instance-1",
                "source_question_uid": "source-1",
                "question_type": "direction_agent",
            }
        ],
    )

    result = launcher.main(
        [
            "--prepared-dataset",
            str(prepared),
            "--reuse-prepared-dataset",
            "--output-dir",
            str(output_dir),
            "--dry-run",
        ]
    )

    manifest = json.loads((output_dir / "grpo_manifest.json").read_text(encoding="utf-8"))
    assert result == 0
    assert manifest["dataset"]["selected_count"] == 1
    assert manifest["dataset"]["checked_images"] == "verified_before_reuse"
    assert "swift rlhf" in capsys.readouterr().out
