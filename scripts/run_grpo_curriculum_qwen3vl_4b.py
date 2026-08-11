#!/usr/bin/env python3
"""Run the two-stage Qwen3-VL-4B GRPO curriculum sequentially."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_grpo_qwen3vl_4b as grpo_runner


RUNNER = ROOT / "scripts" / "run_grpo_qwen3vl_4b.py"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    defaults = grpo_runner.parse_args([])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage1-benchmark",
        type=Path,
        default=Path("output_train/grpo_curriculum_stage1_l1_6144.json"),
    )
    parser.add_argument(
        "--stage2-benchmark",
        type=Path,
        default=Path("output_train/grpo_curriculum_stage2_reasoning_18432.json"),
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--prepared-root", type=Path)
    parser.add_argument("--model", default=defaults.model)
    parser.add_argument("--swift-bin", default=defaults.swift_bin)
    parser.add_argument("--scannet-image-root", action="append", type=Path, default=[])
    parser.add_argument("--scannetpp-image-root", action="append", type=Path, default=[])
    parser.add_argument("--scannetpp-sensor", choices=("iphone", "dslr"), default="iphone")
    parser.add_argument("--devices", default=defaults.devices)
    parser.add_argument("--stage1-epochs", type=float, default=1.0)
    parser.add_argument("--stage2-epochs", type=float, default=1.0)
    parser.add_argument("--stage1-learning-rate", type=float, default=1e-5)
    parser.add_argument("--stage2-learning-rate", type=float, default=5e-6)
    parser.add_argument("--optim", choices=("adamw_torch_fused", "adamw_torch"), default=defaults.optim)
    parser.add_argument(
        "--resume-optimizer-state-dtype",
        choices=("none", "bfloat16", "float32"),
        default=defaults.resume_optimizer_state_dtype,
    )
    parser.add_argument("--per-device-batch-size", type=int, default=defaults.per_device_batch_size)
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=defaults.gradient_accumulation_steps,
    )
    parser.add_argument("--num-generations", type=int, default=defaults.num_generations)
    parser.add_argument("--lora-rank", type=int, default=defaults.lora_rank)
    parser.add_argument("--lora-alpha", type=int, default=defaults.lora_alpha)
    parser.add_argument("--lora-dropout", type=float, default=defaults.lora_dropout)
    parser.add_argument("--lora-dtype", choices=("bfloat16", "float16", "float32"), default=defaults.lora_dtype)
    parser.add_argument("--vllm-tensor-parallel-size", type=int, default=1)
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.55)
    parser.add_argument("--vllm-max-model-len", type=int, default=defaults.vllm_max_model_len)
    parser.add_argument(
        "--vllm-mm-processor-cache-gb",
        type=float,
        default=defaults.vllm_mm_processor_cache_gb,
    )
    parser.add_argument("--max-length", type=int, default=defaults.max_length)
    parser.add_argument("--max-completion-length", type=int, default=defaults.max_completion_length)
    parser.add_argument("--max-pixels", type=int, default=defaults.max_pixels)
    parser.add_argument("--warmup-ratio", type=float, default=defaults.warmup_ratio)
    parser.add_argument("--max-grad-norm", type=float, default=defaults.max_grad_norm)
    parser.add_argument("--temperature", type=float, default=defaults.temperature)
    parser.add_argument("--top-p", type=float, default=defaults.top_p)
    parser.add_argument("--beta", type=float, default=defaults.beta)
    parser.add_argument(
        "--answer-reward-weight", type=float, default=defaults.answer_reward_weight
    )
    parser.add_argument(
        "--format-reward-weight", type=float, default=defaults.format_reward_weight
    )
    parser.add_argument("--deepspeed", choices=("none", "zero2", "zero3"), default="none")
    parser.add_argument("--save-steps", type=int, default=defaults.save_steps)
    parser.add_argument("--save-total-limit", type=int, default=4)
    parser.add_argument("--dataloader-workers", type=int, default=defaults.dataloader_workers)
    parser.add_argument("--dataset-workers", type=int, default=defaults.dataset_workers)
    parser.add_argument("--attn-impl", default=defaults.attn_impl)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--skip-image-check", action="store_true")
    parser.add_argument("--reuse-prepared-dataset", action="store_true")
    parser.add_argument(
        "--start-stage",
        choices=(1, 2),
        type=int,
        default=1,
        help="Start from stage 2 using the latest stage-1 checkpoint after a restart.",
    )
    parser.add_argument(
        "--stage1-resume-from-checkpoint",
        type=Path,
        help=(
            "Resume stage 1 from this Trainer checkpoint. The literal value "
            "'latest' selects the highest checkpoint below the stage-1 output."
        ),
    )
    parser.add_argument(
        "--stage2-resume-from-checkpoint",
        type=Path,
        help=(
            "Resume stage 2 from this Trainer checkpoint. The literal value "
            "'latest' selects the highest checkpoint below the stage-2 output."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Prepare and print each selected stage command without launching MS-SWIFT.",
    )
    return parser.parse_args(argv)


def _common_args(args: argparse.Namespace) -> list[str]:
    command = [
        "--model",
        args.model,
        "--swift-bin",
        args.swift_bin,
        "--scannetpp-sensor",
        args.scannetpp_sensor,
        "--devices",
        args.devices,
        "--per-device-batch-size",
        str(args.per_device_batch_size),
        "--gradient-accumulation-steps",
        str(args.gradient_accumulation_steps),
        "--optim",
        args.optim,
        "--num-generations",
        str(args.num_generations),
        "--lora-rank",
        str(args.lora_rank),
        "--lora-alpha",
        str(args.lora_alpha),
        "--lora-dropout",
        str(args.lora_dropout),
        "--lora-dtype",
        args.lora_dtype,
        "--vllm-tensor-parallel-size",
        str(args.vllm_tensor_parallel_size),
        "--vllm-gpu-memory-utilization",
        str(args.vllm_gpu_memory_utilization),
        "--vllm-max-model-len",
        str(args.vllm_max_model_len),
        "--vllm-mm-processor-cache-gb",
        str(args.vllm_mm_processor_cache_gb),
        "--max-length",
        str(args.max_length),
        "--max-completion-length",
        str(args.max_completion_length),
        "--max-pixels",
        str(args.max_pixels),
        "--warmup-ratio",
        str(args.warmup_ratio),
        "--max-grad-norm",
        str(args.max_grad_norm),
        "--temperature",
        str(args.temperature),
        "--top-p",
        str(args.top_p),
        "--beta",
        str(args.beta),
        "--answer-reward-weight",
        str(args.answer_reward_weight),
        "--format-reward-weight",
        str(args.format_reward_weight),
        "--deepspeed",
        args.deepspeed,
        "--save-strategy",
        "steps",
        "--save-steps",
        str(args.save_steps),
        "--save-total-limit",
        str(args.save_total_limit),
        "--dataloader-workers",
        str(args.dataloader_workers),
        "--dataset-workers",
        str(args.dataset_workers),
        "--attn-impl",
        args.attn_impl,
        "--seed",
        str(args.seed),
        "--preserve-dataset-order",
    ]
    for path in args.scannet_image_root:
        command.extend(["--scannet-image-root", str(path.resolve())])
    for path in args.scannetpp_image_root:
        command.extend(["--scannetpp-image-root", str(path.resolve())])
    if args.skip_image_check:
        command.append("--skip-image-check")
    if args.reuse_prepared_dataset:
        command.append("--reuse-prepared-dataset")
    if args.dry_run:
        command.append("--dry-run")
    return command


def build_stage_command(
    args: argparse.Namespace,
    *,
    stage: int,
    adapter: Path | None = None,
) -> list[str]:
    output_root = args.output_root.resolve()
    prepared_root = (
        args.prepared_root.resolve()
        if args.prepared_root is not None
        else output_root / "prepared"
    )
    if stage == 1:
        benchmark = args.stage1_benchmark.resolve()
        output_dir = output_root / "stage1_l1_perception"
        prepared = prepared_root / "stage1_l1_6144.grpo.jsonl"
        epochs = args.stage1_epochs
        learning_rate = args.stage1_learning_rate
    elif stage == 2:
        benchmark = args.stage2_benchmark.resolve()
        output_dir = output_root / "stage2_reasoning_replay"
        prepared = prepared_root / "stage2_reasoning_18432.grpo.jsonl"
        epochs = args.stage2_epochs
        learning_rate = args.stage2_learning_rate
        if adapter is None:
            raise ValueError("stage two requires the final stage-one adapter")
    else:
        raise ValueError(f"unsupported curriculum stage: {stage}")

    command = [
        sys.executable,
        str(RUNNER),
        "--benchmark",
        str(benchmark),
        "--output-dir",
        str(output_dir),
        "--prepared-dataset",
        str(prepared),
        "--epochs",
        str(epochs),
        "--learning-rate",
        str(learning_rate),
        *_common_args(args),
    ]
    if stage == 1 and args.stage1_resume_from_checkpoint is not None:
        command.extend(
            [
                "--resume-from-checkpoint",
                str(args.stage1_resume_from_checkpoint),
                "--resume-optimizer-state-dtype",
                args.resume_optimizer_state_dtype,
            ]
        )
    if stage == 2 and args.stage2_resume_from_checkpoint is not None:
        command.extend(
            [
                "--resume-from-checkpoint",
                str(args.stage2_resume_from_checkpoint),
            ]
        )
    if adapter is not None:
        command.extend(
            [
                "--adapter",
                str(adapter.resolve()),
                "--reference-adapter",
                str(adapter.resolve()),
            ]
        )
    return command


def _write_plan(args: argparse.Namespace) -> None:
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "schema_version": "predictive-spatial-grpo-curriculum-run-v1",
        "strategy": "matched_sft_stage1_l1_then_stage2_reasoning",
        "dataset_order_preserved": True,
        "stage1": {
            "benchmark": str(args.stage1_benchmark.resolve()),
            "epochs": args.stage1_epochs,
            "learning_rate": args.stage1_learning_rate,
            "resume_from_checkpoint": (
                str(args.stage1_resume_from_checkpoint)
                if args.stage1_resume_from_checkpoint is not None
                else None
            ),
        },
        "stage2": {
            "benchmark": str(args.stage2_benchmark.resolve()),
            "epochs": args.stage2_epochs,
            "learning_rate": args.stage2_learning_rate,
            "initialization": "stage1_final_lora_for_policy_and_reference",
            "resume_from_checkpoint": (
                str(args.stage2_resume_from_checkpoint)
                if args.stage2_resume_from_checkpoint is not None
                else None
            ),
            "optimizer_state": (
                "restored_from_trainer_checkpoint"
                if args.stage2_resume_from_checkpoint is not None
                else "reset"
            ),
        },
        "devices": args.devices,
    }
    path = output_root / "curriculum_plan.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    _write_plan(args)
    stage1_output = args.output_root.resolve() / "stage1_l1_perception"
    if args.start_stage == 1:
        subprocess.run(build_stage_command(args, stage=1), check=True)
    adapter = grpo_runner._latest_checkpoint(stage1_output)
    print(f"Starting stage 2 from adapter: {adapter}", flush=True)
    subprocess.run(build_stage_command(args, stage=2, adapter=adapter), check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
