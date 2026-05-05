"""Engine-side unit tests for ``api.health()`` helpers.

The integration tests live in ``tests/server/test_health_route.py`` (HTTP
wire shape + secret-leak grep). This file covers the small pure helpers
that are easy to regress without a real failure surfacing on the wire —
notably the ``base_url`` sanitizer that strips embedded credentials.
"""

from __future__ import annotations

import pytest

from dikw_core.api import _sanitize_base_url


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
