"""Prompt templates packaged alongside the engine.

Each prompt lives as a ``*.md`` file in this package and is loaded via
``importlib.resources`` so it works in source checkouts, wheels, and zipapps
alike. Prompts may contain ``{placeholder}`` markers that callers fill with
``str.format`` (the simplest possible templating we can get away with).
"""

from __future__ import annotations

from functools import cache
from importlib import resources
from pathlib import Path

from ._contract import contract_for, placeholders_in


class PromptOverrideError(ValueError):
    """A configured per-base prompt override is missing, escapes the base, or
    violates the prompt's placeholder / output-format contract."""


@cache
def load(name: str) -> str:
    """Return the raw packaged prompt text for ``name`` (no extension).

    Cached: packaged prompts are immutable for the life of the process and
    callers (synth fan-out, lint fixers) hit them in tight loops —
    avoids re-reading the same packaged resource every iteration.
    """
    path = resources.files(__package__).joinpath(f"{name}.md")
    return path.read_text(encoding="utf-8")


def resolve(
    name: str, *, override_path: str | None = None, base_root: Path | None = None
) -> str:
    """Return the prompt text for ``name``, honouring a per-base override.

    With no ``override_path`` set this is just :func:`load` (the packaged
    default). When a base configures an override (``synth.prompt_path`` /
    ``lint.fixer_prompts`` in ``dikw.yml``), the file is resolved against
    ``base_root``, required to stay **inside** the base, and validated against
    the prompt's :mod:`._contract` (required ``{placeholders}`` present, no
    stray ones, output-format markers intact) before its text is returned.

    The override file is read fresh each call (callers hoist the load out of
    tight loops, so there is no caching need) — a long-lived ``dikw serve``
    therefore picks up edits without a restart. Raises
    :class:`PromptOverrideError` on any failure so misconfig fails fast.
    """
    if not override_path:
        return load(name)
    if base_root is None:
        raise PromptOverrideError(
            f"prompt override for {name!r} requires a base_root to resolve {override_path!r}"
        )
    base = base_root.resolve()
    abs_path = (base / override_path).resolve()
    try:
        abs_path.relative_to(base)
    except ValueError as exc:
        raise PromptOverrideError(
            f"prompt override {override_path!r} for {name!r} resolves outside the base {base}"
        ) from exc
    if not abs_path.is_file():
        raise PromptOverrideError(
            f"prompt override {override_path!r} for {name!r} not found at {abs_path}"
        )
    text = abs_path.read_text(encoding="utf-8")
    _validate_override(name, text, override_path)
    return text


def _validate_override(name: str, text: str, override_path: str) -> None:
    contract = contract_for(name)
    if contract is None:  # not an overridable prompt — defensive, config gates this
        raise PromptOverrideError(f"prompt {name!r} is not overridable")
    # ``placeholders_in`` runs ``string.Formatter().parse`` which raises a
    # *bare* ``ValueError`` on a syntactically malformed token (a lone ``{``,
    # an unclosed ``{field``). Surface it as the documented ``PromptOverrideError``
    # so synth/lint and ``dikw client check`` get a typed, catchable failure
    # instead of a raw crash (``PromptOverrideError`` subclasses ``ValueError``,
    # so a bare ``ValueError`` would slip past callers' ``except`` clauses).
    try:
        present = placeholders_in(text)
    except ValueError as exc:
        raise PromptOverrideError(
            f"prompt override {override_path!r} for {name!r} is not a valid "
            f"template: {exc}"
        ) from exc
    problems: list[str] = []
    if missing := contract.placeholders - present:
        problems.append(
            "missing required placeholder(s): "
            + ", ".join(sorted("{" + m + "}" for m in missing))
        )
    if extra := present - contract.placeholders:
        problems.append(
            "unknown placeholder(s) the engine will not fill: "
            + ", ".join(sorted("{" + e + "}" for e in extra))
        )
    problems.extend(
        f"missing required output-format marker {marker!r}"
        for marker in contract.markers
        if marker not in text
    )
    if problems:
        raise PromptOverrideError(
            f"prompt override {override_path!r} for {name!r} is invalid: "
            + "; ".join(problems)
        )
