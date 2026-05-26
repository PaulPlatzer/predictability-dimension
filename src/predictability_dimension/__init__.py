"""Tools for local dimension and predictability experiments."""

from __future__ import annotations

import os
from pathlib import Path


# Some dependencies import matplotlib during numerical workflows.  On shared
# systems the default user cache may be read-only, so keep the cache local.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_MPLCONFIGDIR = _PROJECT_ROOT / ".cache" / "matplotlib"
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))

__version__ = "0.1.0"
