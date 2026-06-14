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
from dikw_core.prompts._contract import contract_for

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
    # The SP now carries the full standing policy as named invariants (the
    # operational numbers + the output format live in the UP). Keep it under a
    # few hundred words so it stays the cached, scannable policy spine rather
    # than a verbose preamble. Lower bound guards against a future edit silently
    # reverting to the terse one-liner original.
    words = DEFAULT_SYNTH_SYSTEM.split()
    assert 20 <= len(words) <= 350, f"system prompt is {len(words)} words"
    # Still must carry the language-fidelity instruction that the original
    # one-liner existed to deliver.
    assert "language" in DEFAULT_SYNTH_SYSTEM.lower()


def test_system_prompt_carries_no_density_pressure() -> None:
    """The UP frames link density as a *ceiling* (manufactured links dilute
    the graph) and atomicity as "complete, then concise". The system prompt —
    the cached channel the fan-out leg and the non-atomic-page splitter
    share — must not push the opposite posture, or providers that weight
    the system prompt heavily keep manufacturing links and thin pages. The
    word "dense" is banned outright: "densely-linked" reads as a floor, not the
    ceiling the Honest-linking invariant intends."""
    lowered = DEFAULT_SYNTH_SYSTEM.lower()
    assert "dense" not in lowered, "SP must not ask for dense linking"
    assert "densely" not in lowered, "SP must not call pages 'densely-linked'"
    assert "favour many" not in lowered, (
        "SP must not push many-small-pages over complete atomic pages"
    )


def test_system_prompt_states_named_invariants() -> None:
    """The SP is a structured standing-policy spine: an ``## Invariants``
    section naming the load-bearing rules. Pinning the names (not their exact
    wording) keeps the rewrite from silently collapsing back to an unstructured
    paragraph that buries a rule the eval gate depends on."""
    lowered = DEFAULT_SYNTH_SYSTEM.lower()
    assert "invariant" in lowered
    for anchor in ("atomicity", "faithfulness", "reuse", "taxonomy", "linking"):
        assert anchor in lowered, f"SP dropped the {anchor!r} invariant"


def test_system_prompt_preserves_source_language_rule() -> None:
    """The dropped-then-restored rule (review must-fix): the SP must carry the
    *don't translate* instruction, not merely the slug-is-ASCII note — losing it
    regresses ``language_fidelity`` on the Chinese-primary corpus this engine
    targets."""
    lowered = DEFAULT_SYNTH_SYSTEM.lower()
    assert "language" in lowered
    assert "translate" in lowered, "SP must forbid translating source-language terms"


def test_system_prompt_carries_faithfulness_precision_rule() -> None:
    """Faithfulness in the SP must guard against *added precision* ("recent
    growth" -> "grew 40% in 2023"), not just outright fabrication — the
    anti-hallucination nuance the review flagged as dropped."""
    assert "precision" in DEFAULT_SYNTH_SYSTEM.lower()


def test_system_prompt_acknowledges_category_taxonomy() -> None:
    """The role line must acknowledge the closed category taxonomy (the 0.5.0
    folder tree IS the catalogue) rather than claim structure comes from
    wikilinks "not from folders" — a line that contradicts the Closed-taxonomy
    invariant four lines later."""
    lowered = DEFAULT_SYNTH_SYSTEM.lower()
    assert "not from folders" not in lowered
    assert "taxonomy" in lowered or "category" in lowered


def test_system_prompt_is_byte_stable() -> None:
    """The SP is the prompt-cache channel (anthropic ``cache_control``); it
    must carry no ``str.format`` placeholders or other per-call variance —
    a templated SP would silently shift bytes per call and bust the cache."""
    assert "{" not in DEFAULT_SYNTH_SYSTEM and "}" not in DEFAULT_SYNTH_SYSTEM


def test_template_intro_carries_no_density_pressure() -> None:
    raw = prompts.load("synthesize")
    assert "densely-linked" not in raw, (
        "UP intro must match the ceiling framing of the links rules"
    )


def test_placeholders_render_after_static_instructions() -> None:
    """Cache contract: every ``{placeholder}`` lives in the dynamic Task zone
    at the template tail, AFTER all static instruction sections (anchored on
    ``## Output format``, the last static section). The static prefix is then
    byte-stable across calls, so OpenAI-compatible providers' automatic
    prefix caching (and codex) cover ~the whole instruction body; only the
    tail diverges per call."""
    raw = prompts.load("synthesize")
    anchor = raw.index("## Output format")
    contract = contract_for("synthesize")
    assert contract is not None
    for name in sorted(contract.placeholders):
        pos = raw.index("{" + name + "}")
        assert pos > anchor, (
            f"placeholder {{{name}}} renders at {pos}, before the static "
            f"anchor at {anchor} — it busts the shared prompt-cache prefix"
        )


def test_category_omission_is_last_resort_everywhere() -> None:
    """Category omission must read as a last resort on both tiers: the SP's
    Closed-taxonomy invariant (standing policy) and the UP's Output-format
    bullet (the last thing the model reads before emitting). The ``## Category``
    prose section is gone — its principle moved to the SP — so the old
    cross-reference must not survive."""
    raw = prompts.load("synthesize")
    assert "omit the attribute entirely if none fits" not in raw
    # The principle now lives in the SP invariant, not a UP "## Category" section.
    assert "last resort" in DEFAULT_SYNTH_SYSTEM.lower()
    assert "described under Category" not in raw, "stale cross-reference to a deleted section"
    # The Output-format bullet still frames omission as a last resort inline.
    assert "last resort" in raw.lower()


def test_output_format_forbids_engine_owned_frontmatter_keys() -> None:
    """C8: the engine owns ``sources`` (authoritative provenance) and ``lint``
    (the leaf-acknowledgement block) — an LLM that emits either in front-matter
    overwrites engine state (``page.py`` applies ``meta['sources']`` then
    ``meta.update(extras)``). The Output-format forbidden-key list must name
    both, alongside the originals."""
    raw = prompts.load("synthesize").lower()
    for key in ("title", "id", "category", "sources", "created", "updated", "lint"):
        assert f"`{key}`" in raw, f"output-format must forbid emitting `{key}`"


def test_template_prose_references_current_section_names() -> None:
    """Stale references to pre-PR1 section titles ("the existing-pages
    section above") must not survive — the heading is now ``## Knowledge-base
    context`` with H3 sub-sections."""
    raw = prompts.load("synthesize")
    assert "existing-pages section above" not in raw


def test_duplicate_rule_scoped_to_existing_page_lists() -> None:
    """"Scan the lists above" textually swept in the priority-targets list —
    pages that do NOT exist — so a literal-minded model could suppress
    creating the very page the priority directive asks for. The duplicate
    rule must scope itself to the two existing-page lists and explicitly
    exempt priority targets."""
    raw = prompts.load("synthesize")
    assert "scan the lists above" not in raw
    assert "do **not** exist yet" in raw
