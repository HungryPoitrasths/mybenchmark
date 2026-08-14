#!/usr/bin/env python3
"""Run Base, SFT, and GRPO checkpoints on a frozen spatial manifest."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.classic_spatial_eval.inference import main


if __name__ == "__main__":
    raise SystemExit(main())
