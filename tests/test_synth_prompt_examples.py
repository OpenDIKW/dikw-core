"""Guard the worked examples embedded in ``synthesize.md``.

PR 1b (Phase 1 prompt quality) adds an ``## Example`` section with two
worked ``<page>`` blocks (one English, one Chinese) so the synthesis LLM
sees the target shape — atomic body, inline ``[[wikilinks]]``, source-
language fidelity — rather than only an abstract template. A malformed
example teaches the model bad structure, so these tests parse the *shipped*
prompt and assert every example block round-trips through the real parser
and clears the atomicity heuristic. They also pin the enriched system
prompt's intent without over-constraining its wording.
"""

from __future__ import annotations

import re

from dikw_core import prompts
from dikw_core.domains.knowledge.lint import check_atomicity
from dikw_core.domains.knowledge.synthesize import (
    DEFAULT_ALLOWED_CATEGORIES,
    DEFAULT_SYNTH_SYSTEM,
    parse_synthesis_response,
)

# CJK Unified Ideographs — tells a Chinese example body from an English one.
# (Plain non-ASCII would misfire on an em dash in otherwise-English prose.)
_CJK = re.compile(r"[一-鿿]")


def _section(prompt: str, heading: str) -> str:
    """Return the body of a ``## <heading>`` section up to the next ``## ``."""
    pattern = re.compile(
        rf"^##\s+{re.escape(heading)}\s*$(.*?)(?=^##\s|\Z)",
        flags=re.DOTALL | re.MULTILINE,
    )
    m = pattern.search(prompt)
    assert m is not None, f"synthesize.md has no '## {heading}' section"
    return m.group(1)


def test_synthesize_prompt_has_worked_example_section() -> None:
    raw = prompts.load("synthesize")
    # The example section must sit before the output-format template so the
    # model reads concrete examples before the abstract block spec.
    assert "## Example" in raw
    assert raw.index("## Example") < raw.index("## Output format")


def test_worked_examples_parse_as_atomic_pages() -> None:
    section = _section(prompts.load("synthesize"), "Example")
    pages = parse_synthesis_response(section, source_path="prompt-example")
    # Two worked examples — one English, one Chinese — per the plan.
    assert len(pages) >= 2
    for page in pages:
        verdict = check_atomicity(body=page.body, tags=page.tags)
        assert verdict.atomic, (
            f"worked example {page.title!r} is non-atomic: {verdict.violations}"
        )
        # Each worked example stays a stub-free but compact page.
        assert page.body.strip(), f"example {page.title!r} has an empty body"


def test_worked_examples_cover_both_languages() -> None:
    section = _section(prompts.load("synthesize"), "Example")
    pages = parse_synthesis_response(section, source_path="prompt-example")
    has_cjk = any(_CJK.search(p.body) for p in pages)
    has_non_cjk = any(not _CJK.search(p.body) for p in pages)
    assert has_cjk, "expected a Chinese (CJK) worked example"
    assert has_non_cjk, "expected an English (non-CJK) worked example"


def test_worked_examples_carry_category_attribute() -> None:
    """Examples that omit ``category=`` teach the model to omit it, landing
    real pages in the fallback bucket (``fallback_ratio_max`` 0.31-0.47 on the
    MiniMax baselines). Each worked example must model the attribute with a
    value from the default taxonomy so parsing files it OUT of the fallback."""
    section = _section(prompts.load("synthesize"), "Example")
    pages = parse_synthesis_response(section, source_path="prompt-example")
    assert len(pages) >= 2
    for page in pages:
        assert page.category in DEFAULT_ALLOWED_CATEGORIES, (
            f"worked example {page.title!r} must carry a category= attribute "
            f"from the default taxonomy; parsed category={page.category!r}"
        )


def test_template_nests_dynamic_sections_under_context_heading() -> None:
    """The dynamic ``{existing_pages_section}`` carries both *existing-page*
    lists and the *priority-create* directive — semantically distinct things
    that must not sit under a heading claiming they all "exist". The template
    introduces them with a neutral H2; the rendered sub-sections are H3
    (see ``_render_existing_section`` / ``_render_priority_targets``)."""
    raw = prompts.load("synthesize")
    # Anchored at line start so a demotion to H3 ("### Knowledge-base context",
    # which still contains the H2 string) cannot sneak past a substring check.
    # ``\s*$`` tolerates trailing spaces and a CR if a reader ever bypasses
    # universal-newline translation on a CRLF checkout.
    assert re.search(r"^## Knowledge-base context\s*$", raw, flags=re.MULTILINE)
    assert "## Existing pages" not in raw


def test_worked_examples_use_inline_wikilinks() -> None:
    section = _section(prompts.load("synthesize"), "Example")
    pages = parse_synthesis_response(section, source_path="prompt-example")
    # The whole point of a worked example here is dense inline linking; each
    # example must demonstrate at least one [[wikilink]] in its body.
    for page in pages:
        assert "[[" in page.body, (
            f"worked example {page.title!r} shows no inline [[wikilink]]"
        )


def test_enriched_system_prompt_is_concise_but_substantive() -> None:
    # Enriched from one sentence to a few; keep it under ~100 words so the
    # model doesn't echo a verbose preamble. Lower bound guards against a
    # future edit silently reverting to the terse original.
    words = DEFAULT_SYNTH_SYSTEM.split()
    assert 20 <= len(words) <= 100, f"system prompt is {len(words)} words"
    # Still must carry the language-fidelity instruction that the original
    # one-liner existed to deliver.
    assert "language" in DEFAULT_SYNTH_SYSTEM.lower()
