#!/usr/bin/env python3
"""Launch an 8k or 10k CoT SFT pilot with sample-based logging and evaluation."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cot.evaluation import evaluate_predictions


CHECKPOINT_RE = re.compile(r"^checkpoint-(\d+)$")
STANDARD_TRAIN_COUNTS = {8_000, 10_000}


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
    for milestone in range(interval, train_count * epochs + 1, interval):
        epoch_index, within_epoch = divmod(milestone - 1, train_count)
        exposure_in_epoch = within_epoch + 1
        step = epoch_index * steps_per_epoch + math.ceil(exposure_in_epoch / global_batch)
        result[step] = milestone
    return result


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


def resolve_resume_checkpoint(value: str | None, output_dir: Path) -> Path | None:
    if value is None:
        return None
    if value.strip().lower() == "latest":
        candidates = [
            path
            for path in output_dir.iterdir()
            if is_resumable_checkpoint(path)
        ] if output_dir.is_dir() else []
        if not candidates:
            raise ValueError(f"no resumable checkpoint found in {output_dir}")
        return max(candidates, key=lambda path: int(CHECKPOINT_RE.fullmatch(path.name).group(1)))
    checkpoint = Path(value).expanduser().resolve()
    if not is_resumable_checkpoint(checkpoint):
        raise ValueError(
            f"resume checkpoint is incomplete or invalid: {checkpoint}"
        )
    return checkpoint


def checkpoint_directories(output_dir: Path) -> list[Path]:
    if not output_dir.is_dir():
        return []
    return sorted(
        (
            path
            for path in output_dir.iterdir()
            if path.is_dir() and CHECKPOINT_RE.fullmatch(path.name) is not None
        ),
        key=lambda path: int(CHECKPOINT_RE.fullmatch(path.name).group(1)),
    )


def discover_checkpoints(
    output_dir: Path,
    *,
    train_count: int,
    global_batch: int,
) -> list[dict[str, Any]]:
    checkpoints: list[dict[str, Any]] = []
    if not output_dir.is_dir():
        return checkpoints
    for path in output_dir.iterdir():
        match = CHECKPOINT_RE.fullmatch(path.name)
        if match is None or not path.is_dir():
            continue
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
    for milestone in range(interval, total_exposures + 1, interval):
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
    resume_checkpoint = getattr(args, "resume_from_checkpoint", None)
    if resume_checkpoint is not None:
        command.extend(["--resume_from_checkpoint", str(Path(resume_checkpoint).resolve())])
    return command


def build_infer_command(
    args: argparse.Namespace,
    *,
    result_path: Path,
    checkpoint: Path | None,
) -> list[str]:
    command = [args.swift_bin, "infer"]
    if checkpoint is None:
        command.extend(["--model", args.model])
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dataset", type=Path, required=True)
    parser.add_argument("--train-sidecar", type=Path, required=True)
    parser.add_argument("--monitor-dataset", type=Path, required=True)
    parser.add_argument("--monitor-sidecar", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--swift-bin", default="swift")
    parser.add_argument("--devices", default="0,1")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--per-device-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--optimizer", default="adamw_torch")
    parser.add_argument("--save-every-samples", type=int, default=500)
    parser.add_argument("--eval-every-samples", type=int, default=2000)
    parser.add_argument("--log-every-samples", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--aligner-learning-rate", type=float, default=1e-5)
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
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    train_rows = load_jsonl(args.train_dataset)
    train_sidecar = load_jsonl(args.train_sidecar)
    monitor_rows = load_jsonl(args.monitor_dataset)
    monitor_sidecar = load_jsonl(args.monitor_sidecar)
    if not args.allow_nonstandard_counts and len(train_rows) not in STANDARD_TRAIN_COUNTS:
        expected = " or ".join(str(value) for value in sorted(STANDARD_TRAIN_COUNTS))
        raise ValueError(
            f"pilot train dataset must contain {expected} rows, got {len(train_rows)}"
        )
    if not args.allow_nonstandard_counts and len(monitor_rows) != 320:
        raise ValueError(f"monitor dataset must contain 320 rows, got {len(monitor_rows)}")
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
    total_exposures = len(train_rows) * args.epochs
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
        "train_count": len(train_rows),
        "train_sidecar": str(args.train_sidecar.resolve()),
        "isolation": isolation_report,
        "monitor_count": len(monitor_rows),
        "epochs": args.epochs,
        "world_size": world_size,
        "global_batch": global_batch,
        "max_length": args.max_length,
        "max_pixels": args.max_pixels,
        "save_every_samples": args.save_every_samples,
        "eval_every_samples": args.eval_every_samples,
        "log_every_samples": args.log_every_samples,
        "log_steps": {
            str(step): milestone
            for step, milestone in milestone_steps(
                train_count=len(train_rows),
                global_batch=global_batch,
                epochs=args.epochs,
                interval=args.log_every_samples,
            ).items()
        },
        "checkpoint_steps": {
            str(step): milestone
            for step, milestone in milestone_steps(
                train_count=len(train_rows),
                global_batch=global_batch,
                epochs=args.epochs,
                interval=args.save_every_samples,
            ).items()
        },
        "evaluation_steps": {
            str(step): milestone
            for step, milestone in milestone_steps(
                train_count=len(train_rows),
                global_batch=global_batch,
                epochs=args.epochs,
                interval=args.eval_every_samples,
            ).items()
        },
        "total_exposures": total_exposures,
        "resume_from_checkpoint": (
            str(args.resume_from_checkpoint) if args.resume_from_checkpoint else None
        ),
        "training_command": training_command,
    }
    write_json(args.output_dir / "pilot_manifest.json", manifest)

    train_env = os.environ.copy()
    train_env["CUDA_VISIBLE_DEVICES"] = args.devices
    train_env["NPROC_PER_NODE"] = str(world_size)
    train_env["COT_SFT_TRAIN_COUNT"] = str(len(train_rows))
    train_env["COT_SFT_GLOBAL_BATCH"] = str(global_batch)
    train_env["COT_SFT_EPOCHS"] = str(args.epochs)
    train_env["COT_SFT_SAVE_EVERY_SAMPLES"] = str(args.save_every_samples)
    train_env["COT_SFT_EVAL_EVERY_SAMPLES"] = str(args.eval_every_samples)
    train_env["COT_SFT_LOG_EVERY_SAMPLES"] = str(args.log_every_samples)
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
        interval=args.eval_every_samples,
    )
    write_json(
        args.output_dir / "checkpoint_index.json",
        {"checkpoints": checkpoints, "milestones": milestones},
    )
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

    eval_env = os.environ.copy()
    eval_env["CUDA_VISIBLE_DEVICES"] = args.devices.split(",")[0].strip()
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
        for milestone in milestones:
            name = str(milestone["milestone_name"])
            checkpoint = Path(str(milestone["path"]))
            result_path = monitor_dir / f"{name}.predictions.jsonl"
            report = run_or_reuse_evaluation(
                args,
                sidecar=monitor_sidecar,
                result_path=result_path,
                report_path=monitor_dir / f"{name}.report.json",
                checkpoint=checkpoint,
                env=eval_env,
            )
            summaries.append({"name": name, "checkpoint": milestone, "report": report})
    write_json(monitor_dir / "learning_curve.json", summaries)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
