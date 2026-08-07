#!/usr/bin/env python3
"""Minimal two-rank NCCL collective smoke test."""

from __future__ import annotations

import os

import torch
import torch.distributed as dist


def main() -> None:
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    value = torch.tensor([float(dist.get_rank() + 1)], device="cuda")
    dist.all_reduce(value)
    torch.cuda.synchronize()
    if dist.get_rank() == 0:
        expected = dist.get_world_size() * (dist.get_world_size() + 1) / 2
        if value.item() != expected:
            raise RuntimeError(f"unexpected NCCL all-reduce result: {value.item()}")
        print(f"NCCL smoke passed on {dist.get_world_size()} ranks: {value.item()}", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
