from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from src.cot.evaluation import parse_relaxed_answer, parse_strict_answer
from src.cot.grpo_training import (
    answer_rewards,
    format_rewards,
    load_jsonl,
    prepare_balanced_benchmark_grpo_dataset,
    prepare_grpo_dataset,
)


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
    assert command[command.index("--reward_funcs") + 1 : command.index("--reward_weights")] == [
        "psr_answer",
        "psr_format",
    ]
    assert command[command.index("--freeze_vit") + 1] == "true"
    assert command[command.index("--freeze_aligner") + 1] == "true"
    assert command[command.index("--vllm_enable_lora") + 1] == "true"
    assert command[command.index("--max_steps") + 1] == "2"
    assert "--deepspeed" not in command

    args.deepspeed = "zero2"
    deepspeed_command = launcher.build_swift_command(
        args,
        prepared_dataset=tmp_path / "prepared.jsonl",
        resume_checkpoint=None,
    )
    assert deepspeed_command[deepspeed_command.index("--deepspeed") + 1] == "zero2"


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
