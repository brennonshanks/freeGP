#!/usr/bin/env python3
"""Compatibility CLI shim for the main HMC-NUTS runner."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    package_root = Path(__file__).resolve().parents[2]
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))
    from freegp.run_gprhd_hmc import main
else:
    from ..run_gprhd_hmc import main


if __name__ == "__main__":
    main()
