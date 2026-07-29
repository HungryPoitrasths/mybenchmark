from __future__ import annotations

import importlib.util
from argparse import Namespace
from pathlib import Path

import pytest

from src.cot.evaluation import evaluate_predictions, parse_strict_answer
from src.cot.sampling import (
    PILOT_TRAIN_8K_LEVEL_QUOTAS,
    PILOT_TRAIN_LEVEL_QUOTAS,
    SUPPORTED_TYPE_ORDER,
    TYPES_BY_LEVEL,
    SamplingError,
    select_monitor_validation,
    select_pilot_train,
    select_pilot_train_8k,
)


def _load_pilot_script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_cot_sft_pilot.py"
    spec = importlib.util.spec_from_file_location("run_cot_sft_pilot", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _level_for_type(question_type: str) -> str:
    return next(level for level, values in TYPES_BY_LEVEL.items() if question_type in values)


def _row(question_type: str, index: int, *, signature: str | None = None) -> dict:
    return {
        "question_uid": f"{question_type}-{index}",
        "question_type": question_type,
        "signature_id": signature or f"{question_type}.signature-{index % 3}",
        "facts": {
            "level": _level_for_type(question_type),
            "scene_id": f"scene-{index % 17}",
        },
        "answer_letters": [chr(ord("A") + index % 4)],
        "option_count": 4,
        "multi_select": False,
    }


def test_monitor_selection_contains_twenty_per_supported_type() -> None:
    rows = [
        _row(question_type, index)
        for question_type in SUPPORTED_TYPE_ORDER
        for index in range(25)
    ]
    result = select_monitor_validation(rows, seed=7)
    assert len(result.indices) == 320
    assert set(result.report["selected_by_type"].values()) == {20}
    assert len(set(result.indices)) == 320


def test_monitor_selection_rejects_underfilled_type() -> None:
    rows = [
        _row(question_type, index)
        for question_type in SUPPORTED_TYPE_ORDER
        for index in range(19 if question_type == "object_remove" else 20)
    ]
    with pytest.raises(SamplingError, match="object_remove"):
        select_monitor_validation(rows)


def test_pilot_selection_redistributes_short_type_without_duplicates() -> None:
    assert PILOT_TRAIN_LEVEL_QUOTAS == {"L1": 4669, "L2": 661, "L3": 4670}
    rows: list[dict] = []
    for level, question_types in TYPES_BY_LEVEL.items():
        level_quota = PILOT_TRAIN_LEVEL_QUOTAS[level]
        default_count = level_quota
        for question_type in question_types:
            count = 20 if question_type == "object_remove" else default_count
            rows.extend(_row(question_type, index) for index in range(count))
    rows.append(dict(rows[0]))

    result = select_pilot_train(rows, seed=11)
    assert len(result.indices) == 10_000
    assert result.report["selected_by_level"] == PILOT_TRAIN_LEVEL_QUOTAS
    assert result.report["selected_by_type"]["object_remove"] == 20
    assert result.report["duplicate_uid_count"] == 1
    selected_uids = [rows[index]["question_uid"] for index in result.indices]
    assert len(selected_uids) == len(set(selected_uids))


def test_8k_pilot_uses_every_available_l2_record_without_duplicates() -> None:
    assert PILOT_TRAIN_8K_LEVEL_QUOTAS == {"L1": 3669, "L2": 661, "L3": 3670}
    rows: list[dict] = []
    l2_counts = {
        "object_move_agent": 205,
        "object_move_allocentric": 98,
        "object_move_distance": 111,
        "object_move_object_centric": 127,
        "object_rotate_object_centric": 120,
    }
    for level, question_types in TYPES_BY_LEVEL.items():
        for question_type in question_types:
            if level == "L2":
                count = l2_counts.get(question_type, 0)
            else:
                count = PILOT_TRAIN_8K_LEVEL_QUOTAS[level]
            rows.extend(_row(question_type, index) for index in range(count))

    result = select_pilot_train_8k(rows, seed=11)

    assert len(result.indices) == 8_000
    assert result.report["selected_by_level"] == PILOT_TRAIN_8K_LEVEL_QUOTAS
    assert sum(
        result.report["selected_by_type"].get(question_type, 0)
        for question_type in TYPES_BY_LEVEL["L2"]
    ) == 661
    selected_uids = [rows[index]["question_uid"] for index in result.indices]
    assert len(selected_uids) == len(set(selected_uids))


def test_strict_answer_parser_separates_format_from_correctness() -> None:
    assert parse_strict_answer(
        "Reasoning text.\nAnswer: B", option_count=4, multi_select=False
    ) == "B"
    assert parse_strict_answer(
        "Reasoning text.\nThe answer is B", option_count=4, multi_select=False
    ) is None
    assert parse_strict_answer(
        "Reasoning text.\nAnswer: D", option_count=3, multi_select=False
    ) is None
    assert parse_strict_answer(
        "Reasoning text.\nAnswer: A B", option_count=4, multi_select=True
    ) == "A B"


def test_evaluation_reports_macro_and_format_rates() -> None:
    sidecar = [
        _row("direction_agent", 0),
        _row("direction_agent", 1),
        _row("occlusion", 0),
        _row("occlusion", 1),
    ]
    predictions = [
        {"response": "Reasoning.\nAnswer: A"},
        {"response": "Reasoning.\nAnswer: C"},
        {"response": "Reasoning.\nAnswer: A"},
        {"response": "The answer is B"},
    ]
    report = evaluate_predictions(sidecar, predictions)
    assert report["overall"]["strict_accuracy"] == pytest.approx(0.5)
    assert report["overall"]["macro_accuracy"] == pytest.approx(0.5)
    assert report["overall"]["format_success_rate"] == pytest.approx(0.75)
    assert report["overall"]["relaxed_accuracy"] == pytest.approx(0.75)


def test_checkpoint_sample_count_handles_epoch_boundary(tmp_path: Path) -> None:
    script = _load_pilot_script()
    schedule = script.milestone_steps(
        train_count=10_000,
        global_batch=32,
        epochs=2,
        interval=1_000,
    )
    assert list(schedule) == [
        32,
        63,
        94,
        125,
        157,
        188,
        219,
        250,
        282,
        313,
        345,
        376,
        407,
        438,
        470,
        501,
        532,
        563,
        595,
        626,
    ]
    for step in schedule:
        (tmp_path / f"checkpoint-{step}").mkdir()
    checkpoints = script.discover_checkpoints(
        tmp_path,
        train_count=10_000,
        global_batch=32,
    )
    by_step = {row["global_step"]: row["samples_seen"] for row in checkpoints}
    assert by_step[313] == 10_000
    assert by_step[345] == 11_024
    assert by_step[626] == 20_000
    milestones = script.map_milestones(
        checkpoints,
        total_exposures=20_000,
        interval=1_000,
    )
    assert len(milestones) == 20
    assert milestones[0]["milestone"] == 1_000
    assert milestones[-1]["milestone"] == 20_000
    assert milestones[-1]["global_step"] == 626


def test_8k_checkpoint_schedule_has_thirty_two_milestones() -> None:
    script = _load_pilot_script()
    schedule = script.milestone_steps(
        train_count=8_000,
        global_batch=32,
        epochs=2,
        interval=500,
    )

    assert len(schedule) == 32
    assert schedule[16] == 500
    assert schedule[250] == 8_000
    assert schedule[500] == 16_000
    assert list(schedule.values()) == list(range(500, 16_001, 500))


def test_8k_evaluation_schedule_has_eight_milestones() -> None:
    script = _load_pilot_script()
    schedule = script.milestone_steps(
        train_count=8_000,
        global_batch=32,
        epochs=2,
        interval=2_000,
    )

    assert len(schedule) == 8
    assert list(schedule.values()) == list(range(2_000, 16_001, 2_000))


def test_8k_training_loss_schedule_has_160_points() -> None:
    script = _load_pilot_script()
    schedule = script.milestone_steps(
        train_count=8_000,
        global_batch=32,
        epochs=2,
        interval=100,
    )

    assert len(schedule) == 160
    assert schedule[4] == 100
    assert schedule[250] == 8_000
    assert schedule[500] == 16_000


def test_sft_command_registers_exact_milestone_callback(tmp_path: Path) -> None:
    script = _load_pilot_script()
    args = Namespace(
        swift_bin="swift",
        model="Qwen/Qwen3-VL-4B-Instruct",
        train_dataset=tmp_path / "train.jsonl",
        monitor_dataset=tmp_path / "val.jsonl",
        epochs=2,
        per_device_batch_size=1,
        gradient_accumulation_steps=16,
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
        attn_impl="sdpa",
        dataloader_workers=4,
        dataset_workers=4,
        seed=42,
        output_dir=tmp_path / "output",
        resume_from_checkpoint=None,
    )
    command = script.build_sft_command(args)
    assert "--external_plugins" in command
    assert "cot_sft_milestone_plugin.py" in command[command.index("--external_plugins") + 1]
    assert command[command.index("--callbacks") + 1] == "cot_sample_milestones"
    assert command[command.index("--tuner_type") + 1] == "lora"
    assert "--train_type" not in command
    assert command[command.index("--optim") + 1] == "adamw_torch"
    assert command[command.index("--attn_impl") + 1] == "sdpa"
    assert command[command.index("--eval_strategy") + 1] == "no"
    assert command[command.index("--save_strategy") + 1] == "no"
    assert command[command.index("--logging_strategy") + 1] == "no"
    assert "--save_steps" not in command
    assert "--logging_steps" not in command


def test_sft_command_includes_resume_checkpoint(tmp_path: Path) -> None:
    script = _load_pilot_script()
    args = Namespace(
        swift_bin="swift",
        model="Qwen/Qwen3-VL-4B-Instruct",
        train_dataset=tmp_path / "train.jsonl",
        monitor_dataset=tmp_path / "val.jsonl",
        epochs=2,
        per_device_batch_size=1,
        gradient_accumulation_steps=16,
        optimizer="adamw_torch",
        learning_rate=1e-4,
        aligner_learning_rate=1e-5,
        lora_rank=32,
        lora_alpha=64,
        lora_dropout=0.05,
        weight_decay=0.01,
        warmup_ratio=0.03,
        max_grad_norm=1.0,
        max_length=4096,
        attn_impl="sdpa",
        dataloader_workers=4,
        dataset_workers=4,
        seed=42,
        output_dir=tmp_path / "output",
        resume_from_checkpoint=tmp_path / "output" / "checkpoint-16",
    )

    command = script.build_sft_command(args)

    assert command[command.index("--resume_from_checkpoint") + 1].endswith(
        "checkpoint-16"
    )


def test_latest_resume_skips_incomplete_checkpoint(tmp_path: Path) -> None:
    script = _load_pilot_script()
    complete = tmp_path / "checkpoint-16"
    complete.mkdir()
    (complete / "trainer_state.json").write_text("{}\n", encoding="utf-8")
    (complete / "adapter_model.safetensors").write_bytes(b"weights")
    incomplete = tmp_path / "checkpoint-32"
    incomplete.mkdir()
    (incomplete / "eval_metrics.json").write_text("{}\n", encoding="utf-8")

    selected = script.resolve_resume_checkpoint("latest", tmp_path)

    assert selected == complete


def test_completed_predictions_require_every_monitor_uid() -> None:
    script = _load_pilot_script()
    sidecar = [
        {"question_uid": "q1"},
        {"question_uid": "q2"},
    ]

    assert not script.predictions_are_complete(
        sidecar,
        [{"question_uid": "q1", "response": "Answer: A"}],
    )
    assert script.predictions_are_complete(
        sidecar,
        [
            {"question_uid": "q1", "response": "Answer: A"},
            {"question_uid": "q2", "response": "Answer: B"},
        ],
    )


def test_checkpoint_discovery_includes_teacher_forced_eval_loss(tmp_path: Path) -> None:
    script = _load_pilot_script()
    checkpoint = tmp_path / "checkpoint-32"
    checkpoint.mkdir()
    (checkpoint / "eval_metrics.json").write_text(
        '{"milestone":1000,"eval_loss":1.25}\n', encoding="utf-8"
    )

    rows = script.discover_checkpoints(
        tmp_path,
        train_count=8_000,
        global_batch=32,
    )

    assert rows[0]["teacher_forced_eval"]["eval_loss"] == pytest.approx(1.25)


def test_launcher_rejects_scene_overlap() -> None:
    script = _load_pilot_script()
    train = [_row("direction_agent", 0)]
    monitor = [_row("occlusion", 0)]
    monitor[0]["question_uid"] = "different"
    with pytest.raises(ValueError, match="scene_id"):
        script.validate_disjoint_sidecars(train, monitor)
