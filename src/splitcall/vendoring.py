"""Put the vendored vLLM and SGLang trees on sys.path."""

from __future__ import annotations

import sys
from pathlib import Path

_VENDOR = Path(__file__).resolve().parent / "vendor_path"
_ACTIVE = False


def vendor_root() -> Path:
    return _VENDOR


def activate() -> Path:
    """Insert the vendor root at the front of sys.path once."""
    global _ACTIVE
    root = str(_VENDOR)
    if root not in sys.path:
        sys.path.insert(0, root)
    _ACTIVE = True
    return _VENDOR
