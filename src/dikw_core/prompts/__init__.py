"""Prompt templates packaged alongside the engine.

Each prompt lives as a ``*.md`` file in this package and is loaded via
``importlib.resources`` so it works in source checkouts, wheels, and zipapps
alike. Prompts may contain ``{placeholder}`` markers that callers fill with
``str.format`` (the simplest possible templating we can get away with).
"""

from __future__ import annotations

from functools import cache
from importlib import resources


@cache
def load(name: str) -> str:
    """Return the raw prompt text for ``name`` (no extension).

    Cached: prompts are immutable for the life of the process and
    callers (synth fan-out, lint fixers) hit them in tight loops —
    avoids re-reading the same packaged resource every iteration.
    """
    path = resources.files(__package__).joinpath(f"{name}.md")
    return path.read_text(encoding="utf-8")
