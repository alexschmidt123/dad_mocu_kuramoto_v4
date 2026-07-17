"""Deprecated compatibility shim.

``run_keyed_history`` lives in ``src.control.terminal_rule``.
"""

from __future__ import annotations

from src.control.terminal_rule import run_keyed_history

__all__ = ["run_keyed_history"]
