#!/usr/bin/env python3
"""Compatibility wrapper for the ablation-grid CLI."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    package_root = Path(__file__).resolve().parents[1]
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))
    from freegp.cli.run_ablation_grid import main
else:
    from .cli.run_ablation_grid import main


if __name__ == "__main__":
    main()
