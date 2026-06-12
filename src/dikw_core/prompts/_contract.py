"""Per-prompt override contracts.

A base may override an engine prompt with its own markdown (``synth.prompt_path``
/ ``lint.fixer_prompts`` in ``dikw.yml``). An override is only safe if it still
carries the ``{placeholders}`` the engine fills via ``str.format`` and the
output-format markers the parser depends on — otherwise synth/lint silently
break (a missing ``{source_body}`` starves the LLM of content; a stray
placeholder raises ``KeyError`` at format time; a dropped ``<page`` marker
makes every response unparseable).

This module declares, per overridable prompt, the exact set of placeholders the
engine provides (an override must use exactly that set — no more, no less) and
the literal markers the output must instruct. ``prompts.resolve`` validates an
override against the matching contract at load and surfaces failures via
``dikw client check``.
"""

from __future__ import annotations

import string
from dataclasses import dataclass


@dataclass(frozen=True)
class PromptContract:
    placeholders: frozenset[str]
    markers: tuple[str, ...]


# The ``markers`` are the ``<page category="..." slug="...">`` output-format
# tokens the synth parser relies on (see ``domains/knowledge/synthesize.py``).
_PAGE_MARKERS = ("<page", "category=", "slug=", "</page>")

_CONTRACTS: dict[str, PromptContract] = {
    "synthesize": PromptContract(
        placeholders=frozenset(
            {
                "categories",
                "existing_pages_section",
                "source_path",
                "source_body",
                "group_outline",
                "group_index",
                "group_total",
                "max_pages",
            }
        ),
        # Beyond the output markers, the synth renderer fills
        # ``{existing_pages_section}`` with H3 sub-sections that assume a
        # ``## Knowledge-base context`` H2 container in the template — an
        # override missing it would silently nest those sections under
        # whatever heading precedes the placeholder, so require it here and
        # let ``dikw client check`` fail loudly instead. The leading ``\n``
        # line-anchors the marker (``### …`` contains the H2 string as a
        # substring; see the marker check in ``prompts.resolve``).
        markers=(*_PAGE_MARKERS, "\n## Knowledge-base context"),
    ),
    "lint_fix_broken_wikilink_grounded": PromptContract(
        placeholders=frozenset(
            {
                "broken_target",
                "source_path",
                "source_context",
                "evidence_block",
                "categories",
            }
        ),
        markers=_PAGE_MARKERS,
    ),
    "lint_fix_orphan_merge": PromptContract(
        placeholders=frozenset(
            {
                "target_path",
                "target_category",
                "target_slug",
                "target_title",
                "target_body",
                "orphan_path",
                "orphan_body",
                "score_reason",
            }
        ),
        markers=_PAGE_MARKERS,
    ),
}


def contract_for(name: str) -> PromptContract | None:
    """Return the override contract for prompt ``name``, or ``None`` if the
    prompt is not user-overridable."""
    return _CONTRACTS.get(name)


def placeholders_in(text: str) -> set[str]:
    """Return the top-level ``{field}`` names referenced by a ``str.format`` template."""
    out: set[str] = set()
    for _literal, field, _spec, _conv in string.Formatter().parse(text):
        if field:
            out.add(field.split(".")[0].split("[")[0])
    return out
