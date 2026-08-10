#!/usr/bin/env python3
"""Launch pilot or curriculum CoT SFT with sample-based logging and evaluation."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cot.evaluation import evaluate_predictions
from src.cot.curriculum import (
    LEGACY_PROFILE,
    LEGACY_SCHEMA_VERSION,
    profile_for_manifest,
)


CHECKPOINT_RE = re.compile(r"^checkpoint-(\d+)$")
STANDARD_TRAIN_COUNT = 2_000
STANDARD_MONITOR_COUNT = 320
CUDA_DEVICE_ORDER = "PCI_BUS_ID"
CURRICULUM_SCHEMA_VERSION = LEGACY_SCHEMA_VERSION
CURRICULUM_TRAIN_COUNT = LEGACY_PROFILE.total_exposures
CURRICULUM_LEVEL_COUNTS = LEGACY_PROFILE.target_exposures_by_level
CURRICULUM_STAGE1_COUNT = LEGACY_PROFILE.stage1_exposures
CURRICULUM_GLOBAL_BATCH = LEGACY_PROFILE.global_batch
CURRICULUM_MODEL = "Qwen/Qwen3-VL-4B-Instruct"


def build_cuda_environment(devices: str) -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_DEVICE_ORDER"] = CUDA_DEVICE_ORDER
    env["CUDA_VISIBLE_DEVICES"] = devices
    return env


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _milestone_values(total_exposures: int, interval: int) -> list[int]:
    values = list(range(interval, total_exposures + 1, interval))
    if not values or values[-1] != total_exposures:
        values.append(total_exposures)
    return values


def validate_disjoint_sidecars(
    train_sidecar: list[dict[str, Any]],
    monitor_sidecar: list[dict[str, Any]],
) -> dict[str, int]:
    train_uids = {str(row.get("question_uid") or "") for row in train_sidecar}
    monitor_uids = {str(row.get("question_uid") or "") for row in monitor_sidecar}
    overlapping_uids = sorted((train_uids & monitor_uids) - {""})
    if overlapping_uids:
        raise ValueError(
            f"train and monitor sets share {len(overlapping_uids)} question_uid values"
        )
    train_scenes = {
        str((row.get("facts") or {}).get("scene_id") or "")
        for row in train_sidecar
        if isinstance(row.get("facts"), dict)
    }
    monitor_scenes = {
        str((row.get("facts") or {}).get("scene_id") or "")
        for row in monitor_sidecar
        if isinstance(row.get("facts"), dict)
    }
    overlapping_scenes = sorted((train_scenes & monitor_scenes) - {""})
    if overlapping_scenes:
        raise ValueError(
            f"train and monitor sets share {len(overlapping_scenes)} scene_id values: "
            f"{overlapping_scenes[:5]}"
        )
    return {
        "train_uid_count": len(train_uids - {""}),
        "monitor_uid_count": len(monitor_uids - {""}),
        "train_scene_count": len(train_scenes - {""}),
        "monitor_scene_count": len(monitor_scenes - {""}),
    }


def samples_seen_for_step(step: int, *, train_count: int, global_batch: int) -> int:
    steps_per_epoch = math.ceil(train_count / global_batch)
    completed_epochs, steps_in_epoch = divmod(step, steps_per_epoch)
    return completed_epochs * train_count + min(steps_in_epoch * global_batch, train_count)


def milestone_steps(
    *,
    train_count: int,
    global_batch: int,
    epochs: int,
    interval: int,
) -> dict[int, int]:
    steps_per_epoch = math.ceil(train_count / global_batch)
    result: dict[int, int] = {}
    for milestone in _milestone_values(train_count * epochs, interval):
        epoch_index, within_epoch = divmod(milestone - 1, train_count)
        exposure_in_epoch = within_epoch + 1
        step = epoch_index * steps_per_epoch + math.ceil(exposure_in_epoch / global_batch)
        result[step] = milestone
    return result


def validate_curriculum_manifest(
    path: Path,
    *,
    train_rows: list[dict[str, Any]],
    train_sidecar: list[dict[str, Any]],
) -> dict[str, Any]:
    payload = load_json(path)
    if not isinstance(payload, dict):
        raise ValueError("curriculum manifest must be a JSON object")
    profile = profile_for_manifest(payload)
    samples = payload.get("samples")
    if not isinstance(samples, list) or len(samples) != profile.total_exposures:
        raise ValueError(
            f"curriculum manifest must contain exactly {profile.total_exposures:,} samples"
        )
    if len(train_rows) != len(samples) or len(train_sidecar) != len(samples):
        raise ValueError("curriculum manifest, train dataset, and sidecar counts differ")

    levels: list[str] = []
    schedule_hasher = hashlib.sha256()
    for index, (sample, train_row, sidecar) in enumerate(
        zip(samples, train_rows, train_sidecar), start=1
    ):
        if not isinstance(sample, dict):
            raise ValueError(f"curriculum sample {index} is not an object")
        sample_uid = str(sample.get("sample_uid") or "")
        question_uid_value = str(sample.get("question_uid") or "")
        level = str(sample.get("level") or "").upper()
        if not sample_uid or not question_uid_value or level not in profile.target_exposures_by_level:
            raise ValueError(f"curriculum sample {index} has incomplete identity metadata")
        if int(sample.get("exposure_index") or 0) != index:
            raise ValueError(f"curriculum exposure order breaks at row {index}")
        if str(train_row.get("sample_uid") or "") != sample_uid:
            raise ValueError(f"train dataset order differs from curriculum at row {index}")
        if str(sidecar.get("sample_uid") or "") != sample_uid:
            raise ValueError(f"train sidecar order differs from curriculum at row {index}")
        if str(train_row.get("question_uid") or "") != question_uid_value:
            raise ValueError(f"train dataset question_uid differs at row {index}")
        if str(sidecar.get("question_uid") or "") != question_uid_value:
            raise ValueError(f"train sidecar question_uid differs at row {index}")
        if int(train_row.get("curriculum_exposure") or 0) != index:
            raise ValueError(f"train dataset exposure index differs at row {index}")
        if int(sidecar.get("curriculum_exposure") or 0) != index:
            raise ValueError(f"train sidecar exposure index differs at row {index}")
        messages = train_row.get("messages") or []
        if (
            len(messages) < 1
            or messages[0] != {"role": "user", "content": sample.get("user_content")}
            or train_row.get("images") != sample.get("images")
        ):
            raise ValueError(f"train prompt/images differ from curriculum at row {index}")
        levels.append(level)
        schedule_hasher.update(f"{index}|{sample_uid}|{level}\n".encode("utf-8"))

    if any(level != "L1" for level in levels[: profile.stage1_exposures]):
        raise ValueError(
            f"the first {profile.stage1_exposures:,} curriculum samples must all be L1"
        )
    if dict(Counter(levels)) != profile.target_exposures_by_level:
        raise ValueError(f"invalid curriculum level counts: {dict(Counter(levels))}")

    stage2 = samples[profile.stage1_exposures :]
    for batch_index in range(0, len(stage2), profile.global_batch):
        batch = stage2[batch_index : batch_index + profile.global_batch]
        expected_name, expected_counts = profile.stage2_pattern[
            (batch_index // profile.global_batch) % len(profile.stage2_pattern)
        ]
        names = {str(sample.get("stage2_batch_pattern") or "") for sample in batch}
        counts = Counter(str(sample.get("level") or "").upper() for sample in batch)
        if names != {expected_name} or dict(counts) != expected_counts:
            raise ValueError(
                f"invalid stage-two global batch {batch_index // CURRICULUM_GLOBAL_BATCH}: "
                f"pattern={sorted(names)}, counts={dict(counts)}"
            )

    schedule_sha256 = schedule_hasher.hexdigest()
    declared_sha256 = str((payload.get("statistics") or {}).get("schedule_sha256") or "")
    if declared_sha256 != schedule_sha256:
        raise ValueError("curriculum schedule SHA256 does not match its samples")
    return {
        "path": str(path.resolve()),
        "file_sha256": file_sha256(path),
        "profile_id": profile.profile_id,
        "schema_version": profile.schema_version,
        "schedule_sha256": schedule_sha256,
        "stage1_end_exposure": profile.stage1_exposures,
        "level_counts": profile.target_exposures_by_level,
        "global_batch": profile.global_batch,
        "optimizer_steps": profile.total_exposures // profile.global_batch,
        "dataset_shuffle": False,
        "train_dataloader_shuffle": False,
    }


def is_resumable_checkpoint(path: Path) -> bool:
    if not path.is_dir() or CHECKPOINT_RE.fullmatch(path.name) is None:
        return False
    if not (path / "trainer_state.json").is_file():
        return False
    weight_names = (
        "adapter_model.safetensors",
        "adapter_model.bin",
        "model.safetensors",
        "pytorch_model.bin",
    )
    return any((path / name).is_file() for name in weight_names)


def checkpoint_directories(output_dir: Path) -> list[Path]:
    if not output_dir.is_dir():
        return []
    groups: dict[Path, list[Path]] = {}
    for path in output_dir.rglob("checkpoint-*"):
        if path.is_dir() and CHECKPOINT_RE.fullmatch(path.name) is not None:
            groups.setdefault(path.parent, []).append(path)
    if not groups:
        return []

    complete_groups = {
        parent: paths
        for parent, paths in groups.items()
        if any(is_resumable_checkpoint(path) for path in paths)
    }
    candidates = complete_groups or groups
    _, selected = max(
        candidates.items(),
        key=lambda item: (item[0].stat().st_mtime_ns, str(item[0])),
    )
    return sorted(
        selected,
        key=lambda path: int(CHECKPOINT_RE.fullmatch(path.name).group(1)),
    )


def resolve_resume_checkpoint(value: str | None, output_dir: Path) -> Path | None:
    if value is None:
        return None
    if value.strip().lower() == "latest":
        candidates = [
            path
            for path in checkpoint_directories(output_dir)
            if is_resumable_checkpoint(path)
        ]
        if not candidates:
            raise ValueError(f"no resumable checkpoint found in {output_dir}")
        return max(candidates, key=lambda path: int(CHECKPOINT_RE.fullmatch(path.name).group(1)))
    checkpoint = Path(value).expanduser().resolve()
    if not is_resumable_checkpoint(checkpoint):
        raise ValueError(
            f"resume checkpoint is incomplete or invalid: {checkpoint}"
        )
    return checkpoint


def discover_checkpoints(
    output_dir: Path,
    *,
    train_count: int,
    global_batch: int,
) -> list[dict[str, Any]]:
    checkpoints: list[dict[str, Any]] = []
    paths = checkpoint_directories(output_dir)
    if any(is_resumable_checkpoint(path) for path in paths):
        paths = [path for path in paths if is_resumable_checkpoint(path)]
    for path in paths:
        match = CHECKPOINT_RE.fullmatch(path.name)
        step = int(match.group(1))
        checkpoint = {
            "path": str(path.resolve()),
            "global_step": step,
            "samples_seen": samples_seen_for_step(
                step,
                train_count=train_count,
                global_batch=global_batch,
            ),
        }
        eval_metrics_path = path / "eval_metrics.json"
        if eval_metrics_path.is_file():
            with eval_metrics_path.open(encoding="utf-8") as handle:
                checkpoint["teacher_forced_eval"] = json.load(handle)
        checkpoints.append(checkpoint)
    return sorted(checkpoints, key=lambda row: int(row["global_step"]))


def verify_stage2_smoke_test(
    output_dir: Path,
    *,
    initial_adapter: Path,
    expected_steps: int,
) -> dict[str, Any]:
    checkpoints = [
        path for path in checkpoint_directories(output_dir) if is_resumable_checkpoint(path)
    ]
    if not checkpoints:
        raise RuntimeError("smoke test did not produce a resumable checkpoint")
    checkpoint = max(
        checkpoints,
        key=lambda path: int(CHECKPOINT_RE.fullmatch(path.name).group(1)),
    )
    checkpoint_step = int(CHECKPOINT_RE.fullmatch(checkpoint.name).group(1))
    if checkpoint_step != expected_steps:
        raise RuntimeError(
            f"smoke checkpoint step mismatch: expected {expected_steps}, got {checkpoint_step}"
        )

    trainer_state = load_json(checkpoint / "trainer_state.json")
    if int(trainer_state.get("global_step") or 0) != expected_steps:
        raise RuntimeError("smoke trainer_state.json has an unexpected global_step")

    args_path = checkpoint / "args.json"
    if not args_path.is_file():
        args_path = checkpoint.parent / "args.json"
    if not args_path.is_file():
        raise RuntimeError("smoke test output is missing MS-SWIFT args.json")
    swift_args = load_json(args_path)
    adapters = swift_args.get("adapters") or []
    if isinstance(adapters, str):
        adapters = [adapters]
    expected_adapter = initial_adapter.resolve()
    resolved_adapters = {Path(str(path)).expanduser().resolve() for path in adapters}
    if expected_adapter not in resolved_adapters:
        raise RuntimeError(
            f"MS-SWIFT did not record the requested initial adapter: {expected_adapter}"
        )
    if swift_args.get("resume_from_checkpoint") not in (None, ""):
        raise RuntimeError("smoke test unexpectedly resumed optimizer/trainer state")

    if not any(
        (checkpoint / name).is_file()
        for name in ("adapter_model.safetensors", "adapter_model.bin")
    ):
        raise RuntimeError("smoke checkpoint is missing adapter model weights")
    required_files = ("optimizer.pt", "scheduler.pt", "eval_metrics.json")
    missing = [name for name in required_files if not (checkpoint / name).is_file()]
    if missing:
        raise RuntimeError(f"smoke checkpoint is missing required files: {missing}")

    eval_metrics = load_json(checkpoint / "eval_metrics.json")
    report = {
        "schema_version": "predictive-spatial-cot-stage2-smoke-v1",
        "passed": True,
        "initial_adapter": str(expected_adapter),
        "checkpoint": str(checkpoint.resolve()),
        "global_step": checkpoint_step,
        "eval_loss": eval_metrics.get("eval_loss"),
        "optimizer_state_created": True,
        "scheduler_state_created": True,
        "resume_from_checkpoint": None,
    }
    write_json(output_dir / "stage2_smoke_test_report.json", report)
    return report


def map_milestones(
    checkpoints: list[dict[str, Any]],
    *,
    total_exposures: int,
    interval: int,
) -> list[dict[str, Any]]:
    if not checkpoints:
        return []
    mapped: list[dict[str, Any]] = []
    used_paths: set[str] = set()
    for milestone in _milestone_values(total_exposures, interval):
        candidate = min(
            checkpoints,
            key=lambda row: (
                abs(int(row["samples_seen"]) - milestone),
                int(row["global_step"]),
            ),
        )
        path = str(candidate["path"])
        if path in used_paths:
            continue
        used_paths.add(path)
        mapped.append(
            {
                **candidate,
                "milestone": milestone,
                "milestone_error": int(candidate["samples_seen"]) - milestone,
                "milestone_name": f"samples_seen_{milestone:05d}",
            }
        )
    return mapped


def build_sft_command(args: argparse.Namespace) -> list[str]:
    plugin_path = Path(__file__).with_name("cot_sft_milestone_plugin.py").resolve()
    command = [
        args.swift_bin,
        "sft",
        "--model",
        args.model,
        "--dataset",
        str(args.train_dataset.resolve()),
        "--val_dataset",
        str(args.monitor_dataset.resolve()),
        "--tuner_type",
        "lora",
        "--torch_dtype",
        "bfloat16",
        "--num_train_epochs",
        str(args.epochs),
        "--per_device_train_batch_size",
        str(args.per_device_batch_size),
        "--per_device_eval_batch_size",
        "1",
        "--gradient_accumulation_steps",
        str(args.gradient_accumulation_steps),
        "--optim",
        args.optimizer,
        "--learning_rate",
        str(args.learning_rate),
        "--aligner_lr",
        str(args.aligner_learning_rate),
        "--lora_rank",
        str(args.lora_rank),
        "--lora_alpha",
        str(args.lora_alpha),
        "--lora_dropout",
        str(args.lora_dropout),
        "--target_modules",
        "all-linear",
        "--freeze_vit",
        "true",
        "--freeze_aligner",
        "false",
        "--weight_decay",
        str(args.weight_decay),
        "--lr_scheduler_type",
        "cosine",
        "--warmup_ratio",
        str(args.warmup_ratio),
        "--max_grad_norm",
        str(args.max_grad_norm),
        "--max_length",
        str(args.max_length),
        "--max_pixels",
        str(args.max_pixels),
        "--gradient_checkpointing",
        "true",
        "--attn_impl",
        args.attn_impl,
        "--external_plugins",
        str(plugin_path),
        "--callbacks",
        "cot_sample_milestones",
        "--eval_strategy",
        "no",
        "--save_strategy",
        "no",
        "--save_total_limit",
        "100",
        "--logging_strategy",
        "no",
        "--report_to",
        "none",
        "--dataloader_num_workers",
        str(args.dataloader_workers),
        "--dataset_num_proc",
        str(args.dataset_workers),
        "--seed",
        str(args.seed),
        "--data_seed",
        str(args.seed),
        "--output_dir",
        str(args.output_dir.resolve()),
    ]
    if getattr(args, "curriculum_manifest", None) is not None:
        command.extend(
            [
                "--dataset_shuffle",
                "false",
                "--train_dataloader_shuffle",
                "false",
                "--group_by_length",
                "false",
            ]
        )
    resume_checkpoint = getattr(args, "resume_from_checkpoint", None)
    initial_adapter = getattr(args, "initial_adapter", None)
    if resume_checkpoint is not None:
        command.extend(["--resume_from_checkpoint", str(Path(resume_checkpoint).resolve())])
    elif initial_adapter is not None:
        command.extend(["--adapters", str(Path(initial_adapter).resolve())])
    smoke_test_steps = getattr(args, "smoke_test_steps", None)
    if smoke_test_steps is not None:
        command.extend(["--max_steps", str(smoke_test_steps)])
    return command


def build_infer_command(
    args: argparse.Namespace,
    *,
    result_path: Path,
    checkpoint: Path | None,
) -> list[str]:
    command = [args.swift_bin, "infer"]
    if checkpoint is None:
        initial_adapter = getattr(args, "initial_adapter", None)
        if initial_adapter is None:
            command.extend(["--model", args.model])
        else:
            command.extend(["--adapters", str(Path(initial_adapter).resolve())])
    else:
        command.extend(["--adapters", str(checkpoint.resolve())])
    command.extend(
        [
            "--val_dataset",
            str(args.monitor_dataset.resolve()),
            "--temperature",
            "0",
            "--max_new_tokens",
            str(args.max_new_tokens),
            "--max_pixels",
            str(args.max_pixels),
            "--stream",
            "false",
            "--result_path",
            str(result_path.resolve()),
        ]
    )
    return command


def run_command(command: list[str], *, env: dict[str, str], dry_run: bool) -> None:
    print(subprocess.list2cmdline(command), flush=True)
    if not dry_run:
        subprocess.run(command, check=True, env=env)


def evaluate_result(
    *,
    sidecar: list[dict[str, Any]],
    result_path: Path,
    report_path: Path,
) -> dict[str, Any]:
    predictions = load_jsonl(result_path)
    report = evaluate_predictions(sidecar, predictions)
    details = report.pop("evaluated")
    write_json(report_path, report)
    write_json(report_path.with_name(report_path.stem + ".details.json"), details)
    return report


def predictions_are_complete(
    sidecar: list[dict[str, Any]], predictions: list[dict[str, Any]]
) -> bool:
    expected_uids = {
        str(row.get("question_uid"))
        for row in sidecar
        if str(row.get("question_uid") or "").strip()
    }
    predicted_uids = {
        str(row.get("question_uid"))
        for row in predictions
        if str(row.get("question_uid") or "").strip()
    }
    if predicted_uids:
        return expected_uids <= predicted_uids
    return len(predictions) >= len(sidecar)


def run_or_reuse_evaluation(
    args: argparse.Namespace,
    *,
    sidecar: list[dict[str, Any]],
    result_path: Path,
    report_path: Path,
    checkpoint: Path | None,
    env: dict[str, str],
) -> dict[str, Any]:
    if args.skip_completed_evals and report_path.is_file():
        report = load_json(report_path)
        if int(report.get("missing_prediction_count") or 0) == 0:
            print(f"Reusing completed evaluation: {report_path}", flush=True)
            return report
        print(f"Ignoring incomplete evaluation report: {report_path}", flush=True)
    reuse_predictions = False
    if args.skip_completed_evals and result_path.is_file():
        try:
            predictions = load_jsonl(result_path)
        except (json.JSONDecodeError, OSError):
            predictions = []
        reuse_predictions = predictions_are_complete(sidecar, predictions)
        if not reuse_predictions:
            print(f"Ignoring incomplete predictions: {result_path}", flush=True)
    if not reuse_predictions:
        run_command(
            build_infer_command(
                args,
                result_path=result_path,
                checkpoint=checkpoint,
            ),
            env=env,
            dry_run=False,
        )
    else:
        print(f"Reusing predictions: {result_path}", flush=True)
    return evaluate_result(
        sidecar=sidecar,
        result_path=result_path,
        report_path=report_path,
    )


def evaluate_checkpoint_milestones(
    args: argparse.Namespace,
    *,
    sidecar: list[dict[str, Any]],
    milestones: list[dict[str, Any]],
    monitor_dir: Path,
    devices: list[str],
) -> list[dict[str, Any]]:
    if not milestones:
        return []
    if not devices:
        raise ValueError("checkpoint evaluation requires at least one GPU")

    queues = [milestones[index::len(devices)] for index in range(len(devices))]

    def evaluate_queue(
        device: str, queue: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        env = build_cuda_environment(device)
        names = [str(milestone["milestone_name"]) for milestone in queue]
        print(f"GPU {device} checkpoint evaluation queue: {names}", flush=True)
        summaries: list[dict[str, Any]] = []
        for milestone in queue:
            name = str(milestone["milestone_name"])
            checkpoint = Path(str(milestone["path"]))
            report = run_or_reuse_evaluation(
                args,
                sidecar=sidecar,
                result_path=monitor_dir / f"{name}.predictions.jsonl",
                report_path=monitor_dir / f"{name}.report.json",
                checkpoint=checkpoint,
                env=env,
            )
            summaries.append(
                {"name": name, "checkpoint": milestone, "report": report}
            )
        return summaries

    active_queues = [
        (device, queue)
        for device, queue in zip(devices, queues)
        if queue
    ]
    if len(active_queues) == 1:
        return evaluate_queue(*active_queues[0])

    summaries: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=len(active_queues)) as executor:
        futures = [
            executor.submit(evaluate_queue, device, queue)
            for device, queue in active_queues
        ]
        for future in futures:
            summaries.extend(future.result())
    return sorted(
        summaries,
        key=lambda row: int(row["checkpoint"]["milestone"]),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dataset", type=Path, required=True)
    parser.add_argument("--train-sidecar", type=Path, required=True)
    parser.add_argument("--monitor-dataset", type=Path, required=True)
    parser.add_argument("--monitor-sidecar", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument(
        "--curriculum-manifest",
        type=Path,
        help="Canonical profile-aware curriculum JSON produced by build_mixed_cot_ablation.py.",
    )
    initialization = parser.add_mutually_exclusive_group(required=True)
    initialization.add_argument(
        "--initial-adapter",
        type=Path,
        help=(
            "LoRA checkpoint that initializes a continuation stage. The adapter weights are "
            "loaded while optimizer and scheduler state are reset."
        ),
    )
    initialization.add_argument(
        "--allow-base-model-start",
        action="store_true",
        help="Explicitly allow a fresh run from --model without an initial adapter.",
    )
    parser.add_argument("--swift-bin", default="swift")
    parser.add_argument("--devices", default="0,1")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--per-device-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--optimizer", default="adamw_torch")
    parser.add_argument("--save-every-samples", type=int, default=250)
    parser.add_argument("--eval-every-samples", type=int, default=500)
    parser.add_argument("--log-every-samples", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--aligner-learning-rate", type=float, default=5e-6)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument(
        "--max-pixels",
        type=int,
        default=786432,
        help=(
            "Maximum pixels per image; 786432 is 1024x768 and approximately "
            "768 merged visual tokens for Qwen3-VL."
        ),
    )
    parser.add_argument(
        "--attn-impl",
        choices=("sdpa", "flash_attn", "eager"),
        default="sdpa",
    )
    parser.add_argument("--max-new-tokens", type=int, default=320)
    parser.add_argument("--dataloader-workers", type=int, default=4)
    parser.add_argument("--dataset-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-base-eval", action="store_true")
    parser.add_argument("--skip-checkpoint-eval", action="store_true")
    parser.add_argument(
        "--resume-from-checkpoint",
        help="Checkpoint path to resume, or 'latest' to select the newest complete checkpoint.",
    )
    parser.add_argument(
        "--skip-completed-evals",
        action="store_true",
        help="Reuse completed reports or predictions when resuming monitoring.",
    )
    parser.add_argument("--allow-nonstandard-counts", action="store_true")
    parser.add_argument(
        "--smoke-test-steps",
        type=int,
        help=(
            "Run a fresh adapter-continuation smoke test for this many optimizer "
            "steps, save/evaluate at the final step, verify MS-SWIFT artifacts, and exit."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.initial_adapter is not None:
        args.initial_adapter = args.initial_adapter.resolve()
        adapter_config = args.initial_adapter / "adapter_config.json"
        if not adapter_config.is_file():
            raise ValueError(
                f"initial adapter is missing adapter_config.json: {args.initial_adapter}"
            )
    train_rows = load_jsonl(args.train_dataset)
    train_sidecar = load_jsonl(args.train_sidecar)
    monitor_rows = load_jsonl(args.monitor_dataset)
    monitor_sidecar = load_jsonl(args.monitor_sidecar)
    curriculum_report = None
    if args.curriculum_manifest is not None:
        curriculum_report = validate_curriculum_manifest(
            args.curriculum_manifest,
            train_rows=train_rows,
            train_sidecar=train_sidecar,
        )
    if (
        args.curriculum_manifest is None
        and not args.allow_nonstandard_counts
        and len(train_rows) != STANDARD_TRAIN_COUNT
    ):
        raise ValueError(
            f"pilot train dataset must contain {STANDARD_TRAIN_COUNT} rows, "
            f"got {len(train_rows)}"
        )
    if not args.allow_nonstandard_counts and len(monitor_rows) != STANDARD_MONITOR_COUNT:
        raise ValueError(
            f"monitor dataset must contain {STANDARD_MONITOR_COUNT} rows, "
            f"got {len(monitor_rows)}"
        )
    if len(train_rows) != len(train_sidecar):
        raise ValueError("train dataset and sidecar row counts do not match")
    if len(monitor_rows) != len(monitor_sidecar):
        raise ValueError("monitor dataset and sidecar row counts do not match")
    sample_intervals = {
        "save_every_samples": args.save_every_samples,
        "eval_every_samples": args.eval_every_samples,
        "log_every_samples": args.log_every_samples,
    }
    for name, value in sample_intervals.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    if args.max_length <= 0:
        raise ValueError("max_length must be positive")
    if args.max_pixels <= 0:
        raise ValueError("max_pixels must be positive")
    if args.smoke_test_steps is not None:
        if args.smoke_test_steps <= 0:
            raise ValueError("smoke_test_steps must be positive")
        if args.initial_adapter is None:
            raise ValueError("a stage-two smoke test requires --initial-adapter")
        if args.resume_from_checkpoint is not None:
            raise ValueError("a stage-two smoke test must start fresh, not resume")
    if args.eval_every_samples % args.save_every_samples != 0:
        raise ValueError(
            "eval_every_samples must be an integer multiple of save_every_samples"
        )

    isolation_report = validate_disjoint_sidecars(train_sidecar, monitor_sidecar)

    world_size = len([value for value in args.devices.split(",") if value.strip()])
    if world_size != 2 and not args.allow_nonstandard_counts:
        raise ValueError(f"the pilot plan requires two visible GPUs, got {world_size}")
    global_batch = (
        world_size * args.per_device_batch_size * args.gradient_accumulation_steps
    )
    if args.curriculum_manifest is not None:
        curriculum_settings = {
            "model": (args.model, CURRICULUM_MODEL),
            "epochs": (args.epochs, 1),
            "per_device_batch_size": (args.per_device_batch_size, 2),
            "gradient_accumulation_steps": (args.gradient_accumulation_steps, 8),
            "global_batch": (global_batch, curriculum_report["global_batch"]),
            "learning_rate": (args.learning_rate, 1e-4),
            "aligner_learning_rate": (args.aligner_learning_rate, 1e-5),
            "optimizer": (args.optimizer, "adamw_torch"),
            "lora_rank": (args.lora_rank, 32),
            "lora_alpha": (args.lora_alpha, 64),
            "lora_dropout": (args.lora_dropout, 0.05),
            "weight_decay": (args.weight_decay, 0.01),
            "warmup_ratio": (args.warmup_ratio, 0.03),
            "max_grad_norm": (args.max_grad_norm, 1.0),
            "max_length": (args.max_length, 8_192),
            "max_pixels": (args.max_pixels, 786_432),
            "save_every_samples": (args.save_every_samples, 2_048),
            "eval_every_samples": (args.eval_every_samples, 2_048),
            "log_every_samples": (args.log_every_samples, 256),
        }
        mismatches = [
            f"{name}={actual!r} (expected {expected!r})"
            for name, (actual, expected) in curriculum_settings.items()
            if actual != expected
        ]
        if mismatches:
            raise ValueError("invalid curriculum settings: " + "; ".join(mismatches))
        if not args.allow_base_model_start or args.initial_adapter is not None:
            raise ValueError("curriculum training must use --allow-base-model-start")
        if args.resume_from_checkpoint is not None:
            raise ValueError("a fresh curriculum run cannot use --resume-from-checkpoint")
    smoke_test_exposures = (
        samples_seen_for_step(
            args.smoke_test_steps,
            train_count=len(train_rows),
            global_batch=global_batch,
        )
        if args.smoke_test_steps is not None
        else None
    )
    total_exposures = smoke_test_exposures or len(train_rows) * args.epochs
    save_every_samples = smoke_test_exposures or args.save_every_samples
    eval_every_samples = smoke_test_exposures or args.eval_every_samples
    log_every_samples = global_batch if smoke_test_exposures else args.log_every_samples
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.resume_from_checkpoint = resolve_resume_checkpoint(
        args.resume_from_checkpoint,
        args.output_dir,
    )
    existing_checkpoint_dirs = checkpoint_directories(args.output_dir)
    if (
        not args.skip_train
        and args.resume_from_checkpoint is None
        and existing_checkpoint_dirs
    ):
        raise ValueError(
            f"output directory already contains {len(existing_checkpoint_dirs)} checkpoint(s); "
            "use --resume-from-checkpoint latest or choose a new --output-dir"
        )

    training_command = build_sft_command(args)
    manifest = {
        "schema_version": "predictive-spatial-cot-sft-pilot-v1",
        "model": args.model,
        "initial_adapter": (
            str(args.initial_adapter) if args.initial_adapter is not None else None
        ),
        "train_count": len(train_rows),
        "train_sidecar": str(args.train_sidecar.resolve()),
        "train_dataset_sha256": file_sha256(args.train_dataset),
        "train_sidecar_sha256": file_sha256(args.train_sidecar),
        "curriculum": curriculum_report,
        "dataset_shuffle": False if curriculum_report is not None else None,
        "train_dataloader_shuffle": False if curriculum_report is not None else None,
        "isolation": isolation_report,
        "monitor_count": len(monitor_rows),
        "epochs": args.epochs,
        "smoke_test_steps": args.smoke_test_steps,
        "world_size": world_size,
        "global_batch": global_batch,
        "max_length": args.max_length,
        "max_pixels": args.max_pixels,
        "save_every_samples": save_every_samples,
        "eval_every_samples": eval_every_samples,
        "log_every_samples": log_every_samples,
        "log_steps": {
            str(step): milestone
            for step, milestone in milestone_steps(
                train_count=len(train_rows),
                global_batch=global_batch,
                epochs=args.epochs,
                interval=log_every_samples,
            ).items()
            if args.smoke_test_steps is None or step <= args.smoke_test_steps
        },
        "checkpoint_steps": {
            str(step): milestone
            for step, milestone in milestone_steps(
                train_count=len(train_rows),
                global_batch=global_batch,
                epochs=args.epochs,
                interval=save_every_samples,
            ).items()
            if args.smoke_test_steps is None or step <= args.smoke_test_steps
        },
        "evaluation_steps": {
            str(step): milestone
            for step, milestone in milestone_steps(
                train_count=len(train_rows),
                global_batch=global_batch,
                epochs=args.epochs,
                interval=eval_every_samples,
            ).items()
            if args.smoke_test_steps is None or step <= args.smoke_test_steps
        },
        "total_exposures": total_exposures,
        "resume_from_checkpoint": (
            str(args.resume_from_checkpoint) if args.resume_from_checkpoint else None
        ),
        "training_command": training_command,
    }
    write_json(args.output_dir / "pilot_manifest.json", manifest)

    train_env = build_cuda_environment(args.devices)
    train_env["NPROC_PER_NODE"] = str(world_size)
    train_env["COT_SFT_TRAIN_COUNT"] = str(len(train_rows))
    train_env["COT_SFT_GLOBAL_BATCH"] = str(global_batch)
    train_env["COT_SFT_EPOCHS"] = str(args.epochs)
    train_env["COT_SFT_SAVE_EVERY_SAMPLES"] = str(save_every_samples)
    train_env["COT_SFT_EVAL_EVERY_SAMPLES"] = str(eval_every_samples)
    train_env["COT_SFT_LOG_EVERY_SAMPLES"] = str(log_every_samples)
    if not args.skip_train:
        run_command(training_command, env=train_env, dry_run=args.dry_run)

    checkpoints = discover_checkpoints(
        args.output_dir,
        train_count=len(train_rows),
        global_batch=global_batch,
    )
    milestones = map_milestones(
        checkpoints,
        total_exposures=total_exposures,
        interval=eval_every_samples,
    )
    write_json(
        args.output_dir / "checkpoint_index.json",
        {"checkpoints": checkpoints, "milestones": milestones},
    )
    if args.smoke_test_steps is not None:
        if args.dry_run:
            print("Smoke-test dry run: artifact verification skipped.", flush=True)
            return 0
        smoke_report = verify_stage2_smoke_test(
            args.output_dir,
            initial_adapter=args.initial_adapter,
            expected_steps=args.smoke_test_steps,
        )
        print(json.dumps({"cot_stage2_smoke_test": smoke_report}, ensure_ascii=False))
        return 0
    if args.dry_run:
        print(
            subprocess.list2cmdline(
                build_infer_command(
                    args,
                    result_path=args.output_dir / "monitor" / "base.predictions.jsonl",
                    checkpoint=None,
                )
            )
        )
        return 0

    eval_devices = [value.strip() for value in args.devices.split(",") if value.strip()]
    eval_env = build_cuda_environment(eval_devices[0])
    monitor_dir = args.output_dir / "monitor"
    summaries: list[dict[str, Any]] = []

    if not args.skip_base_eval:
        result_path = monitor_dir / "base.predictions.jsonl"
        report = run_or_reuse_evaluation(
            args,
            sidecar=monitor_sidecar,
            result_path=result_path,
            report_path=monitor_dir / "base.report.json",
            checkpoint=None,
            env=eval_env,
        )
        summaries.append({"name": "base", "report": report})

    if not args.skip_checkpoint_eval:
        summaries.extend(
            evaluate_checkpoint_milestones(
                args,
                sidecar=monitor_sidecar,
                milestones=milestones,
                monitor_dir=monitor_dir,
                devices=eval_devices,
            )
        )
    write_json(monitor_dir / "learning_curve.json", summaries)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
