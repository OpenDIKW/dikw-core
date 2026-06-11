"""Engine-side unit tests for ``api.health()`` helpers.

The integration tests live in ``tests/server/test_health_route.py`` (HTTP
wire shape + secret-leak grep). This file covers the small pure helpers
that are easy to regress without a real failure surfacing on the wire —
notably the ``base_url`` sanitizer that strips embedded credentials.
"""

from __future__ import annotations

import pytest

from dikw_core.api import _sanitize_base_url
from dikw_core.api_health import _probe_embed


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # Plain URL → unchanged shape.
        ("https://api.openai.com/v1", "https://api.openai.com/v1"),
        # Userinfo (``user:token@host``) is the most common credential
        # leak vector — must be stripped.
        (
            "https://user:s3cret@api.example.com/v1",
            "https://api.example.com/v1",
        ),
        # Bare token before ``@`` (no ``user:``) — also userinfo.
        (
            "https://sk-leak-token@api.example.com/v1",
            "https://api.example.com/v1",
        ),
        # Query string can carry an api_key — strip it.
        (
            "https://api.example.com/v1?api_key=sk-leak",
            "https://api.example.com/v1",
        ),
        # Fragment is not a known leak surface but also not an API
        # endpoint identifier — strip for consistency.
        (
            "https://api.example.com/v1#frag",
            "https://api.example.com/v1",
        ),
        # Custom port preserved.
        (
            "http://localhost:11434/v1",
            "http://localhost:11434/v1",
        ),
        # IPv6 literal — must keep brackets so the URL is still parseable
        # by httpx / the OpenAI SDK after round-tripping.
        ("http://[::1]:8080/v1", "http://[::1]:8080/v1"),
        ("http://[::1]/v1", "http://[::1]/v1"),
        # Empty path is OK (just scheme + host).
        ("https://api.example.com", "https://api.example.com"),
    ],
)
def test_sanitize_base_url_strips_credentials_keeps_endpoint(
    raw: str, expected: str
) -> None:
    assert _sanitize_base_url(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        None,
        "",
        # No scheme → can't safely round-trip; drop rather than expose
        # something the caller didn't intend.
        "api.example.com/v1",
        # Garbage scheme-less string.
        "not a url",
        # Out-of-range port — ``urlsplit().port`` raises ``ValueError``;
        # we must catch and drop rather than 5xx the health probe.
        "http://api.example.com:99999/v1",
        # Non-numeric port — same surface; verify the helper survives.
        "http://api.example.com:abc/v1",
    ],
)
def test_sanitize_base_url_drops_unparseable_or_empty(raw: str | None) -> None:
    assert _sanitize_base_url(raw) is None


# ---- _probe_embed: degenerate responses must fail the probe ---------------


class _CannedEmbedder:
    """Returns a fixed vector list — simulates degenerate provider output."""

    def __init__(self, vectors: list[list[float]]) -> None:
        self._vectors = vectors

    async def embed(self, texts: list[str], *, model: str) -> list[list[float]]:
        _ = (texts, model)
        return self._vectors


async def test_probe_embed_no_vectors_fails() -> None:
    """A call that returns without raising but yields NO vectors is not a
    healthy provider — same contract as ``_probe_llm``'s empty-completion
    check and ``_probe_multimodal``'s shape check. ``ok=True`` here let a
    green ``dikw client check`` precede an ingest whose every embed call
    produced nothing."""
    res = await _probe_embed(_CannedEmbedder([]), "m", "embedding target")
    assert res.ok is False
    assert "EMPTY" in res.detail


async def test_probe_embed_zero_dim_vector_fails() -> None:
    res = await _probe_embed(_CannedEmbedder([[]]), "m", "embedding target")
    assert res.ok is False
    assert "EMPTY" in res.detail


async def test_probe_embed_real_vector_passes() -> None:
    res = await _probe_embed(_CannedEmbedder([[0.1, 0.2]]), "m", "embedding target")
    assert res.ok is True
    assert "dim=2" in res.detail


async def test_probe_multimodal_zero_dim_vectors_fail() -> None:
    """Two zero-dim vectors pass the count check AND the dim-equality check
    (0 == 0) — without an explicit empty guard a silently-dead multimodal
    gateway probes green. Same contract as the text-embed probe above."""
    from dikw_core.api_health import _probe_multimodal

    res = await _probe_multimodal(_CannedEmbedder([[], []]), "m", "mm target")
    assert res.ok is False
    assert "EMPTY" in res.detail


async def test_probe_multimodal_real_vectors_pass() -> None:
    from dikw_core.api_health import _probe_multimodal

    res = await _probe_multimodal(
        _CannedEmbedder([[0.1, 0.2], [0.3, 0.4]]), "m", "mm target"
    )
    assert res.ok is True
    assert "dim=2" in res.detail
