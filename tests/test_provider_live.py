"""Opt-in LIVE provider smoke tests — exercise the REAL backends, not fakes.

These catch backend-specific quirks the SDK-fake suite cannot. Issue #160 is
the canonical example: the ChatGPT codex backend ships a terminal
``response.completed`` whose ``output`` is an empty list while valid
``response.output_text.delta`` events already streamed the real answer, and
the provider discarded it — every fake stream we wrote modelled only what we
already knew, so nothing caught it until a live call did. The live call is the
ground truth (cf. ``feedback_provider_backend_invariants``: an SDK fake passing
is not the same as the backend passing).

Skipped by default (CI has no provider credentials). To run locally:

    # codex — needs a base with imported OAuth (`dikw auth import openai-codex`)
    $env:DIKW_TEST_CODEX_BASE = "C:\\Users\\...\\dikw-demo"
    # optional: $env:DIKW_TEST_CODEX_MODEL = "gpt-5.5"

    # minimax — anthropic-compatible; ANTHROPIC_API_KEY is the MiniMax key
    $env:DIKW_TEST_MINIMAX_MODEL = "<minimax-model>"
    # optional: $env:DIKW_TEST_MINIMAX_BASE_URL = "https://api.minimaxi.com/anthropic"

    uv run --env-file .env pytest tests/test_provider_live.py -v -s
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

# A minimal synth-shaped prompt: a real content section that MUST yield at
# least one ``<page>`` block from a working LLM. An empty completion here is
# the #160 failure mode.
_SYNTH_SYS = (
    "You synthesise K-layer knowledge pages for dikw-core. "
    "Emit one <page> block per distinct entity."
)
_SYNTH_USER = (
    "Section 1 of 1. Emit <page> blocks for the entities below.\n"
    'Format: <page category="entity" slug="...">\n# Title\n\nbody\n</page>\n\n'
    "SOURCE:\nElon Musk founded SpaceX and co-founded Tesla."
)

_CODEX_BASE = os.environ.get("DIKW_TEST_CODEX_BASE")
_CODEX_MODEL = os.environ.get("DIKW_TEST_CODEX_MODEL", "gpt-5.5")
_MINIMAX_MODEL = os.environ.get("DIKW_TEST_MINIMAX_MODEL")
_MINIMAX_BASE_URL = os.environ.get(
    "DIKW_TEST_MINIMAX_BASE_URL", "https://api.minimaxi.com/anthropic"
)


@pytest.mark.asyncio
@pytest.mark.skipif(
    not _CODEX_BASE,
    reason="set DIKW_TEST_CODEX_BASE to a base dir with imported codex OAuth",
)
async def test_live_codex_synth_returns_pages() -> None:
    """Live regression for #160: the codex provider must surface the streamed
    completion even when the SDK final response carries ``output=[]``."""
    from dikw_core.providers.codex_auth import DEFAULT_CODEX_BASE_URL
    from dikw_core.providers.openai_codex import OpenAICodexLLM

    llm = OpenAICodexLLM(
        base_url=DEFAULT_CODEX_BASE_URL, base_root=Path(_CODEX_BASE)
    )
    resp = await llm.complete(
        system=_SYNTH_SYS, user=_SYNTH_USER, model=_CODEX_MODEL, max_tokens=512
    )
    assert resp.text.strip(), (
        "codex returned an EMPTY completion for a content-bearing synth prompt "
        "— issue #160 regression (final.output=[] discarding streamed deltas)"
    )
    assert "<page" in resp.text, (
        f"expected <page> blocks from synth prompt, got: {resp.text[:200]!r}"
    )


@pytest.mark.asyncio
@pytest.mark.skipif(
    not _MINIMAX_MODEL,
    reason="set DIKW_TEST_MINIMAX_MODEL (+ ANTHROPIC_API_KEY) to run the minimax leg",
)
async def test_live_minimax_synth_returns_pages() -> None:
    """Live smoke for the other configured LLM provider (MiniMax via the
    anthropic-compatible endpoint). MiniMax's synth path uses the
    non-streaming ``messages.create`` leg, which does not share the codex
    final-response-precedence bug — this pins that it returns real text."""
    from dikw_core.providers.anthropic_compat import AnthropicCompatLLM

    llm = AnthropicCompatLLM(
        api_key_env="ANTHROPIC_API_KEY", base_url=_MINIMAX_BASE_URL, timeout_seconds=120.0
    )
    resp = await llm.complete(
        system=_SYNTH_SYS, user=_SYNTH_USER, model=_MINIMAX_MODEL, max_tokens=512
    )
    assert resp.text.strip(), "minimax returned an empty completion"
    assert "<page" in resp.text, (
        f"expected <page> blocks from synth prompt, got: {resp.text[:200]!r}"
    )
