#!/usr/bin/env python3
"""Prepare a frozen, media-validated classic spatial benchmark manifest."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.classic_spatial_eval.preparation import main


if __name__ == "__main__":
    raise SystemExit(main())
