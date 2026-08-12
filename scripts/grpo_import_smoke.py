from __future__ import annotations

import os

import setuptools
import vllm
from torch.utils.cpp_extension import load_inline


print(
    "GRPO imports passed:",
    f"rank={os.environ.get('RANK', 'single')}",
    f"setuptools={setuptools.__version__}",
    f"vllm={vllm.__version__}",
    f"cpp_extension={load_inline.__module__}",
    flush=True,
)
