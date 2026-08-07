#!/usr/bin/env python3
"""Prepare PSR-Bench data and launch native MS-SWIFT GRPO for Qwen3-VL-4B."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cot.grpo_training import (
    load_jsonl,
    prepare_balanced_benchmark_grpo_dataset,
    prepare_grpo_dataset,
)


SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a multiple-choice "
    "spatial reasoning question, and the Assistant solves it. The Assistant first "
    "shows its reasoning inside <think> and </think>, then gives only the final option "
    "letter inside <answer> and </answer>. Use exactly this form: "
    "<think>reasoning process here</think><answer>A</answer>. For multiple-selection "
    "questions, put the uppercase option letters in the requested canonical order "
    "separated by single spaces, for example <answer>A C</answer>. Do not write any "
    "text outside these tags."
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=Path("output_train/grpo_balanced_2k.json"),
        help=(
            "Raw or preselected benchmark JSON. Used unless both legacy dataset "
            "arguments are set."
        ),
    )
    parser.add_argument(
        "--samples-per-type",
        type=int,
        help=(
            "Optionally resample an equal quota per type; by default every benchmark "
            "row is used."
        ),
    )
    parser.add_argument(
        "--preserve-dataset-order",
        action="store_true",
        help=(
            "Keep benchmark row order through preparation and disable every MS-SWIFT "
            "training shuffle. Required for an explicit curriculum schedule."
        ),
    )
    parser.add_argument(
        "--train-dataset",
        type=Path,
        help="Existing MS-SWIFT JSONL. Requires --train-sidecar and bypasses --benchmark.",
    )
    parser.add_argument(
        "--train-sidecar",
        type=Path,
        help="Existing CoT sidecar JSONL. Requires --train-dataset and bypasses --benchmark.",
    )
    parser.add_argument("--scannet-image-root", action="append", type=Path, default=[])
    parser.add_argument("--scannetpp-image-root", action="append", type=Path, default=[])
    parser.add_argument("--scannetpp-sensor", choices=("iphone", "dslr"), default="iphone")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/grpo_qwen3vl_4b_instruct"),
    )
    parser.add_argument("--prepared-dataset", type=Path)
    parser.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument(
        "--adapter",
        type=Path,
        help="Initialize policy LoRA weights from this checkpoint without optimizer state.",
    )
    parser.add_argument(
        "--reference-adapter",
        type=Path,
        help="Initialize the GRPO reference model from this LoRA checkpoint.",
    )
    parser.add_argument("--swift-bin", default="swift")
    parser.add_argument("--devices", default="0,1")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--per-device-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument(
        "--optim",
        choices=("adamw_torch_fused", "adamw_torch"),
        default="adamw_torch_fused",
        help="AdamW implementation; use adamw_torch for cross-version optimizer resumes.",
    )
    parser.add_argument(
        "--resume-optimizer-state-dtype",
        choices=("none", "bfloat16", "float32"),
        default="none",
        help=(
            "Restore Adam moment tensors to this dtype after loading a cross-version "
            "checkpoint. Use with --optim adamw_torch."
        ),
    )
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
        help=(
            "LoRA parameter dtype. Keep this explicit so resumed optimizer states "
            "use the same dtype across MS-SWIFT/PEFT versions."
        ),
    )
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--max-completion-length", type=int, default=1024)
    parser.add_argument("--max-pixels", type=int, default=786432)
    parser.add_argument("--vllm-max-model-len", type=int, default=10240)
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.45)
    parser.add_argument("--vllm-tensor-parallel-size", type=int, default=1)
    parser.add_argument(
        "--vllm-mm-processor-cache-gb",
        type=float,
        default=0.0,
        help=(
            "vLLM multimodal processor cache size. Zero disables the cache and avoids "
            "receiver-cache inconsistencies in multi-rank colocate GRPO."
        ),
    )
    parser.add_argument(
        "--deepspeed",
        choices=("none", "zero2", "zero3"),
        default="none",
        help="Optional DeepSpeed strategy; 4B LoRA on 96GB GPUs does not require it.",
    )
    parser.add_argument("--num-generations", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.001)
    parser.add_argument("--answer-reward-weight", type=float, default=1.0)
    parser.add_argument("--format-reward-weight", type=float, default=0.1)
    parser.add_argument("--save-steps", type=int, default=250)
    parser.add_argument(
        "--save-strategy",
        choices=("steps", "epoch"),
        default="steps",
    )
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--dataloader-workers", type=int, default=4)
    parser.add_argument("--dataset-workers", type=int, default=4)
    parser.add_argument("--attn-impl", default="sdpa")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument(
        "--resume-from-checkpoint",
        type=Path,
        help="Checkpoint directory or the literal value 'latest'.",
    )
    parser.add_argument("--skip-image-check", action="store_true")
    parser.add_argument(
        "--reuse-prepared-dataset",
        action="store_true",
        help="Use an existing verified --prepared-dataset without resolving images again.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def _device_ids(value: str) -> list[str]:
    devices = [item.strip() for item in value.split(",") if item.strip()]
    if not devices:
        raise ValueError("--devices must select at least one CUDA device")
    if len(set(devices)) != len(devices):
        raise ValueError("--devices cannot contain duplicates")
    valid_uuid = re.compile(r"GPU-[0-9a-fA-F-]+\Z")
    if any(not (device.isdigit() or valid_uuid.fullmatch(device)) for device in devices):
        raise ValueError(
            "--devices must contain comma-separated non-negative integers or GPU UUIDs"
        )
    return devices


def validate_args(args: argparse.Namespace) -> list[str]:
    devices = _device_ids(args.devices)
    positive_ints = {
        "per_device_batch_size": args.per_device_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "max_length": args.max_length,
        "max_completion_length": args.max_completion_length,
        "max_pixels": args.max_pixels,
        "vllm_max_model_len": args.vllm_max_model_len,
        "vllm_tensor_parallel_size": args.vllm_tensor_parallel_size,
        "num_generations": args.num_generations,
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
    }
    invalid = [name for name, value in positive_ints.items() if value <= 0]
    if invalid:
        raise ValueError(f"these arguments must be positive: {', '.join(invalid)}")
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    if args.samples_per_type is not None and args.samples_per_type <= 0:
        raise ValueError("--samples-per-type must be positive")
    if args.preserve_dataset_order and args.samples_per_type is not None:
        raise ValueError(
            "--preserve-dataset-order cannot be combined with --samples-per-type"
        )
    if (args.train_dataset is None) != (args.train_sidecar is None):
        raise ValueError("provide both --train-dataset and --train-sidecar, or neither")
    if (args.adapter is None) != (args.reference_adapter is None):
        raise ValueError(
            "provide both --adapter and --reference-adapter so policy and KL reference "
            "start from the same curriculum checkpoint"
        )
    if args.max_steps is not None and args.max_steps <= 0:
        raise ValueError("--max-steps must be positive")
    if args.resume_optimizer_state_dtype != "none":
        if args.resume_from_checkpoint is None:
            raise ValueError("--resume-optimizer-state-dtype requires a resume checkpoint")
        if args.optim != "adamw_torch":
            raise ValueError("optimizer state dtype repair requires --optim adamw_torch")
    if not 0.0 < args.vllm_gpu_memory_utilization < 1.0:
        raise ValueError("--vllm-gpu-memory-utilization must be between 0 and 1")
    if args.vllm_mm_processor_cache_gb < 0:
        raise ValueError("--vllm-mm-processor-cache-gb must be non-negative")
    if args.vllm_tensor_parallel_size > len(devices):
        raise ValueError("vLLM tensor parallel size cannot exceed the visible GPU count")
    if args.vllm_max_model_len < args.max_length + args.max_completion_length:
        raise ValueError(
            "--vllm-max-model-len must cover --max-length plus --max-completion-length"
        )
    global_completion_batch = (
        len(devices)
        * args.per_device_batch_size
        * args.gradient_accumulation_steps
    )
    if global_completion_batch % args.num_generations != 0:
        raise ValueError(
            f"global completion batch {global_completion_batch} must be divisible by "
            f"--num-generations {args.num_generations}"
        )
    return devices


def _latest_checkpoint(output_dir: Path) -> Path:
    candidates: list[tuple[int, Path]] = []
    for path in output_dir.rglob("checkpoint-*"):
        if not path.is_dir():
            continue
        try:
            step = int(path.name.removeprefix("checkpoint-"))
        except ValueError:
            continue
        candidates.append((step, path))
    if not candidates:
        raise ValueError(f"no checkpoint-* directories found in {output_dir}")
    return max(candidates)[1].resolve()


def resolve_resume_checkpoint(args: argparse.Namespace) -> Path | None:
    value = args.resume_from_checkpoint
    if value is None:
        return None
    if str(value).lower() == "latest":
        return _latest_checkpoint(args.output_dir.resolve())
    path = value.resolve()
    if not path.is_dir():
        raise ValueError(f"resume checkpoint is not a directory: {path}")
    return path


def build_swift_command(
    args: argparse.Namespace,
    *,
    prepared_dataset: Path,
    resume_checkpoint: Path | None,
) -> list[str]:
    plugin_path = ROOT / "src" / "cot" / "grpo_training.py"
    command = [
        args.swift_bin,
        "rlhf",
        "--rlhf_type",
        "grpo",
        "--model",
        args.model,
        "--dataset",
        str(prepared_dataset.resolve()),
        "--external_plugins",
        str(plugin_path.resolve()),
        "--reward_funcs",
        "psr_answer",
        "psr_format",
        "--reward_weights",
        str(args.answer_reward_weight),
        str(args.format_reward_weight),
        "--tuner_type",
        "lora",
        "--target_modules",
        "all-linear",
        "--freeze_vit",
        "true",
        "--freeze_aligner",
        "true",
        "--lora_rank",
        str(args.lora_rank),
        "--lora_alpha",
        str(args.lora_alpha),
        "--lora_dropout",
        str(args.lora_dropout),
        "--lora_dtype",
        args.lora_dtype,
        "--torch_dtype",
        "bfloat16",
        "--enable_thinking",
        "false",
        "--use_vllm",
        "true",
        "--vllm_mode",
        "colocate",
        "--vllm_gpu_memory_utilization",
        str(args.vllm_gpu_memory_utilization),
        "--vllm_tensor_parallel_size",
        str(args.vllm_tensor_parallel_size),
        "--vllm_max_model_len",
        str(args.vllm_max_model_len),
        "--vllm_mm_processor_cache_gb",
        str(args.vllm_mm_processor_cache_gb),
        "--vllm_enable_lora",
        "true",
        "--vllm_max_lora_rank",
        str(args.lora_rank),
        "--sleep_level",
        "1",
        "--gradient_checkpointing",
        "true",
        "--attn_impl",
        args.attn_impl,
        "--num_train_epochs",
        str(args.epochs),
        "--per_device_train_batch_size",
        str(args.per_device_batch_size),
        "--gradient_accumulation_steps",
        str(args.gradient_accumulation_steps),
        "--learning_rate",
        str(args.learning_rate),
        "--optim",
        args.optim,
        "--lr_scheduler_type",
        "cosine",
        "--warmup_ratio",
        str(args.warmup_ratio),
        "--max_grad_norm",
        str(args.max_grad_norm),
        "--max_length",
        str(args.max_length),
        "--max_completion_length",
        str(args.max_completion_length),
        "--max_pixels",
        str(args.max_pixels),
        "--num_generations",
        str(args.num_generations),
        "--num_iterations",
        "1",
        "--loss_type",
        "grpo",
        "--scale_rewards",
        "group",
        "--epsilon",
        "0.2",
        "--epsilon_high",
        "0.2",
        "--beta",
        str(args.beta),
        "--temperature",
        str(args.temperature),
        "--top_p",
        str(args.top_p),
        "--save_strategy",
        args.save_strategy,
        "--save_steps",
        str(args.save_steps),
        "--save_total_limit",
        str(args.save_total_limit),
        "--logging_steps",
        "1",
        "--log_completions",
        "true",
        "--report_to",
        "none",
        "--dataloader_num_workers",
        str(args.dataloader_workers),
        "--dataset_num_proc",
        str(args.dataset_workers),
        "--load_from_cache_file",
        "true",
        "--seed",
        str(args.seed),
        "--data_seed",
        str(args.seed),
        "--system",
        SYSTEM_PROMPT,
        "--output_dir",
        str(args.output_dir.resolve()),
    ]
    if args.deepspeed != "none":
        command.extend(["--deepspeed", args.deepspeed])
    if args.preserve_dataset_order:
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
    # On RLHF resume, Trainer restores the policy adapter from the checkpoint.
    # MS-SWIFT expects only the original reference adapter to be preloaded.
    if args.adapter is not None and resume_checkpoint is None:
        command.extend(["--adapters", str(args.adapter.resolve())])
    if args.reference_adapter is not None:
        command.extend(["--ref_adapters", str(args.reference_adapter.resolve())])
    if args.max_steps is not None:
        command.extend(["--max_steps", str(args.max_steps)])
    if resume_checkpoint is not None:
        command.extend(["--resume_from_checkpoint", str(resume_checkpoint)])
    return command


def build_environment(args: argparse.Namespace, devices: list[str]) -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["CUDA_VISIBLE_DEVICES"] = ",".join(devices)
    env["NPROC_PER_NODE"] = str(len(devices))
    env["MAX_PIXELS"] = str(args.max_pixels)
    if args.resume_optimizer_state_dtype != "none":
        env["PSR_RESUME_ADAM_STATE_DTYPE"] = args.resume_optimizer_state_dtype
    return env


def _write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def summarize_prepared_dataset(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"prepared dataset does not exist: {path}")
    rows = load_jsonl(path)
    if not rows:
        raise ValueError(f"prepared dataset is empty: {path}")
    image_counts: list[int] = []
    source_uids: set[str] = set()
    question_types: dict[str, int] = {}
    for index, row in enumerate(rows, start=1):
        images = row.get("images")
        if not isinstance(images, list) or not all(isinstance(item, str) for item in images):
            raise ValueError(f"{path}:{index}: images must be a list of paths")
        required = ("messages", "solution", "option_count", "question_uid")
        missing = [key for key in required if row.get(key) in (None, "")]
        if missing:
            raise ValueError(f"{path}:{index}: missing required fields: {missing}")
        image_counts.append(len(images))
        source_uids.add(str(row.get("source_question_uid") or row["question_uid"]))
        question_type = str(row.get("question_type") or "missing")
        question_types[question_type] = question_types.get(question_type, 0) + 1
    return {
        "schema_version": "predictive-spatial-grpo-reused-prepared-v1",
        "prepared_dataset": str(path),
        "selected_count": len(rows),
        "selected_by_type": dict(sorted(question_types.items())),
        "unique_source_question_count": len(source_uids),
        "repeated_instance_count": len(rows) - len(source_uids),
        "min_images": min(image_counts),
        "max_images": max(image_counts),
        "checked_images": "verified_before_reuse",
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    devices = validate_args(args)
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    resume_checkpoint = resolve_resume_checkpoint(args)
    for label, path in (
        ("adapter", args.adapter),
        ("reference adapter", args.reference_adapter),
    ):
        if path is not None and not path.resolve().is_dir():
            raise ValueError(f"{label} checkpoint is not a directory: {path.resolve()}")
    existing = [path for path in args.output_dir.glob("checkpoint-*") if path.is_dir()]
    if existing and resume_checkpoint is None and not args.dry_run:
        raise ValueError(
            f"output directory contains {len(existing)} checkpoint(s); use "
            "--resume-from-checkpoint latest or choose another --output-dir"
        )

    prepared_dataset = (
        args.prepared_dataset.resolve()
        if args.prepared_dataset is not None
        else args.output_dir / "prepared" / "train.grpo.jsonl"
    )
    if args.reuse_prepared_dataset:
        dataset_report = summarize_prepared_dataset(prepared_dataset)
    elif args.train_dataset is not None:
        dataset_report = prepare_grpo_dataset(
            args.train_dataset,
            args.train_sidecar,
            prepared_dataset,
            check_images=not args.skip_image_check,
        )
    else:
        dataset_report = prepare_balanced_benchmark_grpo_dataset(
            args.benchmark,
            prepared_dataset,
            samples_per_type=args.samples_per_type,
            seed=args.seed,
            preserve_input_order=args.preserve_dataset_order,
            scannet_roots=[path.resolve() for path in args.scannet_image_root],
            scannetpp_roots=[path.resolve() for path in args.scannetpp_image_root],
            scannetpp_sensor=args.scannetpp_sensor,
            check_images=not args.skip_image_check,
        )
    command = build_swift_command(
        args,
        prepared_dataset=prepared_dataset,
        resume_checkpoint=resume_checkpoint,
    )
    environment = build_environment(args, devices)
    global_completion_batch = (
        len(devices)
        * args.per_device_batch_size
        * args.gradient_accumulation_steps
    )
    manifest = {
        "schema_version": "predictive-spatial-grpo-run-v1",
        "model": args.model,
        "base_variant": "instruct",
        "data_mode": (
            "legacy_ms_swift"
            if args.train_dataset is not None
            else "ordered_benchmark"
            if args.preserve_dataset_order
            else "balanced_benchmark"
        ),
        "tuner": "lora_llm_only",
        "rewards": [
            {"name": "psr_answer", "weight": args.answer_reward_weight},
            {"name": "psr_format", "weight": args.format_reward_weight},
        ],
        "dataset": dataset_report,
        "devices": devices,
        "global_completion_batch": global_completion_batch,
        "rollout_prompts_per_batch": global_completion_batch // args.num_generations,
        "num_generations": args.num_generations,
        "dataset_order_preserved": args.preserve_dataset_order,
        "resume_from_checkpoint": str(resume_checkpoint) if resume_checkpoint else None,
        "adapter": str(args.adapter.resolve()) if args.adapter else None,
        "reference_adapter": (
            str(args.reference_adapter.resolve()) if args.reference_adapter else None
        ),
        "environment": {
            "CUDA_DEVICE_ORDER": environment["CUDA_DEVICE_ORDER"],
            "CUDA_VISIBLE_DEVICES": environment["CUDA_VISIBLE_DEVICES"],
            "NPROC_PER_NODE": environment["NPROC_PER_NODE"],
            "MAX_PIXELS": environment["MAX_PIXELS"],
            "PSR_RESUME_ADAM_STATE_DTYPE": environment.get(
                "PSR_RESUME_ADAM_STATE_DTYPE"
            ),
        },
        "command": command,
        "shell_command": shlex.join(command),
    }
    _write_manifest(args.output_dir / "grpo_manifest.json", manifest)

    print(json.dumps(dataset_report, ensure_ascii=False, indent=2), flush=True)
    print(shlex.join(command), flush=True)
    if args.dry_run:
        return 0
    if shutil.which(args.swift_bin) is None:
        raise RuntimeError(
            f"MS-SWIFT executable {args.swift_bin!r} was not found; install a release "
            "with Qwen3-VL GRPO and vLLM LoRA support"
        )
    subprocess.run(command, check=True, env=environment)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
