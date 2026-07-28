import argparse
import os
import time

import torch


def reserve_gpu(memory_gb: float, interval_sec: float) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Check PyTorch/CUDA installation.")

    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(device)
    total_gb = torch.cuda.get_device_properties(device).total_memory / 1024**3

    print(f"SFT GPU reserve started")
    print(f"PID: {os.getpid()}")
    print(f"GPU: {gpu_name}")
    print(f"Total memory: {total_gb:.1f} GB")
    print(f"Requested memory: {memory_gb:.1f} GB")
    print("Stop with Ctrl+C")

    # Allocate one large tensor to reserve GPU memory.
    num_float32 = int(memory_gb * 1024**3 / 4)
    reserved = torch.empty(num_float32, dtype=torch.float32, device=device)
    reserved.fill_(1.0)

    # Keep a tiny workload active so schedulers/tools show GPU utilization.
    a = torch.randn((4096, 4096), device=device)
    b = torch.randn((4096, 4096), device=device)

    while True:
        c = a @ b
        torch.cuda.synchronize()
        # Touch the reserved tensor so it remains live and cannot be optimized away.
        reserved[0] = c[0, 0]
        time.sleep(interval_sec)


def main() -> None:
    parser = argparse.ArgumentParser(description="Simple SFT-related GPU reservation script.")
    parser.add_argument(
        "--memory-gb",
        type=float,
        default=32.0,
        help="GPU memory to reserve in GB. Use a value below the card capacity.",
    )
    parser.add_argument(
        "--interval-sec",
        type=float,
        default=1.0,
        help="Sleep interval between small compute steps.",
    )
    args = parser.parse_args()

    reserve_gpu(memory_gb=args.memory_gb, interval_sec=args.interval_sec)


if __name__ == "__main__":
    main()
