import argparse
import os
import time
from dataclasses import dataclass

import torch


DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def add_sft_placeholder_arguments(parser: argparse.ArgumentParser) -> None:
    """Accept common SFT options without applying them to the GPU workload."""
    group = parser.add_argument_group(
        "SFT compatibility options",
        "Accepted for SFT-style launch commands but intentionally unused.",
    )
    group.add_argument("--model-name-or-path", default="Qwen/Qwen3-VL-4B-Instruct")
    group.add_argument("--dataset-path", default="data/train.jsonl")
    group.add_argument("--output-dir", default="output/qwen3-vl-4b-sft")
    group.add_argument("--num-train-epochs", type=float, default=3.0)
    group.add_argument("--per-device-train-batch-size", type=int, default=1)
    group.add_argument("--gradient-accumulation-steps", type=int, default=8)
    group.add_argument("--learning-rate", type=float, default=2e-5)
    group.add_argument("--weight-decay", type=float, default=0.01)
    group.add_argument("--warmup-ratio", type=float, default=0.03)
    group.add_argument("--lr-scheduler-type", default="cosine")
    group.add_argument("--optim", default="adamw_torch_fused")
    group.add_argument("--max-seq-length", type=int, default=4096)
    group.add_argument("--logging-steps", type=int, default=10)
    group.add_argument("--save-steps", type=int, default=500)
    group.add_argument("--save-total-limit", type=int, default=2)
    group.add_argument("--lora-r", type=int, default=64)
    group.add_argument("--lora-alpha", type=int, default=128)
    group.add_argument("--lora-dropout", type=float, default=0.05)
    group.add_argument("--seed", type=int, default=42)
    group.add_argument("--bf16", action="store_true")
    group.add_argument("--gradient-checkpointing", action="store_true")
    group.add_argument("--deepspeed")
    group.add_argument("--report-to", default="none")


@dataclass
class GpuWorkload:
    device: torch.device
    reserved: torch.Tensor | None
    left: torch.Tensor
    right: torch.Tensor
    result: torch.Tensor | None = None


def parse_gpu_ids(value: str, device_count: int) -> list[int]:
    if value.strip().lower() == "all":
        return list(range(device_count))

    try:
        device_ids = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError("--gpus must be 'all' or comma-separated CUDA indices") from error

    if not device_ids:
        raise argparse.ArgumentTypeError("--gpus must select at least one GPU")
    if len(set(device_ids)) != len(device_ids):
        raise argparse.ArgumentTypeError("--gpus cannot contain duplicate GPU indices")
    if any(device_id < 0 or device_id >= device_count for device_id in device_ids):
        raise argparse.ArgumentTypeError(
            f"--gpus indices must be between 0 and {device_count - 1} after CUDA_VISIBLE_DEVICES filtering"
        )
    return device_ids


def reserve_memory(device: torch.device, memory_gb: float) -> torch.Tensor | None:
    if memory_gb == 0:
        return None

    free_bytes, _ = torch.cuda.mem_get_info(device)
    requested_bytes = int(memory_gb * 1024**3)
    if requested_bytes >= free_bytes:
        raise RuntimeError(
            f"cuda:{device.index} has only {free_bytes / 1024**3:.1f} GB free; "
            f"cannot reserve {memory_gb:.1f} GB. Lower --memory-gb."
        )
    reserved = torch.empty(requested_bytes, dtype=torch.uint8, device=device)
    reserved.fill_(1)
    return reserved


def create_workloads(
    device_ids: list[int], memory_gb: float, compute_size: int, dtype: torch.dtype
) -> list[GpuWorkload]:
    workloads = []
    for device_id in device_ids:
        device = torch.device(f"cuda:{device_id}")
        reserved = reserve_memory(device, memory_gb)
        left = torch.randn((compute_size, compute_size), dtype=dtype, device=device)
        right = torch.randn((compute_size, compute_size), dtype=dtype, device=device)
        workloads.append(GpuWorkload(device=device, reserved=reserved, left=left, right=right))
    return workloads


def synchronize(workloads: list[GpuWorkload]) -> None:
    for workload in workloads:
        torch.cuda.synchronize(workload.device)


def run_workload(
    workloads: list[GpuWorkload], utilization: float, cycle_sec: float, duration_sec: float
) -> None:
    active_sec = cycle_sec * utilization / 100.0
    start_time = time.monotonic()
    print("Workload started. Stop with Ctrl+C.")

    while duration_sec == 0 or time.monotonic() - start_time < duration_sec:
        cycle_start = time.monotonic()
        active_until = cycle_start + active_sec

        while time.monotonic() < active_until:
            for workload in workloads:
                workload.result = workload.left @ workload.right
                if workload.reserved is not None:
                    workload.reserved[0] = workload.result[0, 0].to(torch.uint8)
            synchronize(workloads)

        remaining_sec = cycle_sec - (time.monotonic() - cycle_start)
        if remaining_sec > 0:
            time.sleep(remaining_sec)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a controlled matrix-multiplication workload on one or more CUDA GPUs."
    )
    parser.add_argument(
        "--gpus",
        default="all",
        help="CUDA indices to use after CUDA_VISIBLE_DEVICES filtering, e.g. 0,1. Default: all.",
    )
    parser.add_argument(
        "--utilization-percent",
        type=float,
        default=80.0,
        help="Approximate compute duty cycle per GPU (0-100). Default: 80.",
    )
    parser.add_argument(
        "--cycle-sec",
        type=float,
        default=1.0,
        help="Length of each active/idle control cycle in seconds. Default: 1.",
    )
    parser.add_argument(
        "--duration-sec",
        type=float,
        default=0.0,
        help="Total runtime in seconds; 0 runs until interrupted. Default: 0.",
    )
    parser.add_argument(
        "--memory-gb",
        type=float,
        default=0.0,
        help="Memory to reserve on each selected GPU in GB; 0 disables reservation. Default: 0.",
    )
    parser.add_argument(
        "--compute-size",
        type=int,
        default=4096,
        help="Square matrix dimension for each GPU. Larger values raise compute load. Default: 4096.",
    )
    parser.add_argument(
        "--dtype",
        choices=DTYPES,
        default="float16",
        help="Matrix dtype. float16 normally drives tensor cores most effectively. Default: float16.",
    )
    add_sft_placeholder_arguments(parser)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Check PyTorch/CUDA installation.")
    if not 0 <= args.utilization_percent <= 100:
        parser.error("--utilization-percent must be between 0 and 100")
    if args.cycle_sec <= 0:
        parser.error("--cycle-sec must be positive")
    if args.duration_sec < 0:
        parser.error("--duration-sec cannot be negative")
    if args.memory_gb < 0:
        parser.error("--memory-gb cannot be negative")
    if args.compute_size <= 0:
        parser.error("--compute-size must be positive")

    device_ids = parse_gpu_ids(args.gpus, torch.cuda.device_count())
    dtype = DTYPES[args.dtype]
    torch.backends.cuda.matmul.allow_tf32 = True

    print("SFT GPU workload started")
    print(f"PID: {os.getpid()}")
    print(f"GPUs: {', '.join(f'cuda:{device_id}' for device_id in device_ids)}")
    print(f"Target compute duty cycle: {args.utilization_percent:.0f}%")
    print(f"Matrix: {args.compute_size}x{args.compute_size} ({args.dtype})")
    print(f"Memory reserved per GPU: {args.memory_gb:.1f} GB")
    for device_id in device_ids:
        properties = torch.cuda.get_device_properties(device_id)
        print(f"cuda:{device_id}: {properties.name} ({properties.total_memory / 1024**3:.1f} GB)")

    workloads: list[GpuWorkload] = []
    try:
        workloads = create_workloads(device_ids, args.memory_gb, args.compute_size, dtype)
        run_workload(workloads, args.utilization_percent, args.cycle_sec, args.duration_sec)
    except KeyboardInterrupt:
        print("\nWorkload stopped.")
    finally:
        workloads.clear()
        for device_id in device_ids:
            with torch.cuda.device(device_id):
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
