"""HTTP-level tests for ``GET /v1/health``.

The health endpoint is the agent-bootstrap probe: it must surface
enough resolved config to confirm "I'm attached to the right server"
without ever leaking secrets. Tests here lock both the wire shape and
the security boundary.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from dikw_core import __version__
from dikw_core import api as api_module


@pytest.mark.asyncio
async def test_health_returns_well_formed_report(
    server_client: httpx.AsyncClient,
    wiki_root: Path,
) -> None:
    resp = await server_client.get("/v1/health")
    assert resp.status_code == 200
    body = resp.json()

    assert body["status"] == "ok"
    assert body["version"] == __version__
    assert Path(body["base_root"]) == wiki_root.resolve()
    assert body["storage_engine"] in ("sqlite", "postgres")

    # Layer counts present + integer-typed; a freshly init'd wiki has
    # zero documents in every layer.
    counts = body["layer_counts"]
    for key in ("sources", "wiki_pages", "wisdom_items", "chunks"):
        assert key in counts and isinstance(counts[key], int)
    assert counts["sources"] == 0
    assert counts["chunks"] == 0


@pytest.mark.asyncio
async def test_health_layer_counts_track_ingest(
    server_client: httpx.AsyncClient,
    ingested_wiki: Path,
) -> None:
    """After ingesting fixture markdown, ``layer_counts`` must reflect
    the new documents — proves health reads live storage state, not a
    cached snapshot from server start."""
    resp = await server_client.get("/v1/health")
    body = resp.json()
    counts = body["layer_counts"]
    # ``ingested_wiki`` (conftest) ingests every ``tests/fixtures/notes/*.md``;
    # the count reflects whatever lives in that fixture set today.
    assert counts["sources"] > 0
    assert counts["chunks"] > 0


@pytest.mark.asyncio
async def test_health_providers_block_exposes_resolved_config(
    server_client: httpx.AsyncClient,
) -> None:
    resp = await server_client.get("/v1/health")
    body = resp.json()
    providers = body["providers"]

    llm = providers["llm"]
    assert llm["provider"] in (
        "anthropic_compat",
        "openai_compat",
        "openai_codex",
    )
    assert isinstance(llm["model"], str) and llm["model"]
    assert "base_url" in llm  # nullable but field must be present
    assert isinstance(llm["max_retries"], int)
    assert isinstance(llm["api_key_present"], bool)

    embedding = providers["embedding"]
    assert embedding["provider"] == "openai_compat"
    assert isinstance(embedding["dim"], int) and embedding["dim"] > 0
    assert embedding["distance"] in ("cosine", "l2", "dot")
    assert isinstance(embedding["normalize"], bool)
    assert isinstance(embedding["api_key_present"], bool)
    # Multimodal nests under embedding (per design); test wiki has no
    # multimodal config so it's null.
    assert embedding["multimodal"] is None


@pytest.mark.asyncio
async def test_health_never_leaks_secrets(
    server_client: httpx.AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Critical security boundary: even when API keys are in the env,
    the health response must not echo any of them.

    Sentinels deliberately mix URL-unsafe (``!@#$``) and URL-safe
    characters so a buggy serializer can't mask a leak via URL-encoding;
    a 16-char prefix substring is also checked, catching the case where
    a serializer truncates / pretty-prints + breaks the full string.
    """
    sentinels = {
        "OPENAI_API_KEY": "sk-leak-OPENAI-AAA-BBB-CCC-DDD!@#$",
        "ANTHROPIC_API_KEY": "sk-ant-LEAK-XYZ-987-AAA-BBB-CCC",
        "DIKW_EMBEDDING_API_KEY": "sk-embed-LEAK-321-AAA-BBB-CCC",
    }
    for k, v in sentinels.items():
        monkeypatch.setenv(k, v)

    resp = await server_client.get("/v1/health")
    raw = resp.text
    body = resp.json()

    for k, v in sentinels.items():
        assert v not in raw, (
            f"secret {k} leaked into /v1/health response:\n{raw}"
        )
        # Substring check catches a partial leak (e.g. a future bug
        # truncating the value at 16 chars before logging).
        assert v[:16] not in raw, (
            f"prefix of secret {k} leaked into /v1/health response:\n{raw}"
        )

    # Positive check: at least one ``api_key_present`` flag is True
    # (we just set the env var) so the health endpoint is actually
    # reading env, not silently returning False.
    assert body["providers"]["embedding"]["api_key_present"] is True


@pytest.mark.asyncio
async def test_health_storage_engine_omits_dsn_and_path(
    server_client: httpx.AsyncClient,
    wiki_root: Path,
) -> None:
    """``storage_engine`` is the *type* (``sqlite``/``postgres``) only.
    DSN, SQLite path, schema name — none of those belong on /v1/health.
    """
    resp = await server_client.get("/v1/health")
    raw = resp.text
    body = resp.json()

    assert body["storage_engine"] in ("sqlite", "postgres")
    # Cover both ``postgres://`` and ``postgresql://`` URI schemes; psycopg
    # accepts either. ``index.sqlite`` is the default SQLite path under
    # ``.dikw/`` — neither the path nor the bare ``.dikw`` directory name
    # should appear in the response. ``base_root`` is the *only* place
    # the wiki root directory itself is exposed; check we didn't leak the
    # full SQLite file path beyond that.
    assert "postgresql://" not in raw
    assert "postgres://" not in raw
    assert ".dikw/" not in raw
    assert ".dikw\\" not in raw  # Windows path separator
    assert "index.sqlite" not in raw
    # base_root is exposed by design; everything else under wiki_root must
    # not appear (so the .sqlite path under wiki_root/.dikw/ is excluded).
    sqlite_path = wiki_root / ".dikw" / "index.sqlite"
    assert str(sqlite_path) not in raw


@pytest.mark.asyncio
async def test_health_strips_credentials_from_base_url(
    server_client: httpx.AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Defence-in-depth: even if a user encodes a token directly in
    ``provider.llm_base_url`` (``https://user:token@host/`` or
    ``?api_key=…``), the health response must never echo the credential.
    Sanitisation lives in ``api._sanitize_base_url`` — this guards the
    seam by tweaking the resolved config and asserting on the wire.
    """
    original_with_storage = api_module._with_storage
    sentinel_token = "sk-leak-IN-URL-AAA-BBB"

    async def _patched_with_storage(path: object) -> object:
        cfg, root, storage = await original_with_storage(path)  # type: ignore[arg-type]
        cfg.provider.llm_base_url = (
            f"https://user:{sentinel_token}@api.example.com/v1"
        )
        cfg.provider.embedding_base_url = (
            f"https://api.example.com/v1?api_key={sentinel_token}"
        )
        return cfg, root, storage

    monkeypatch.setattr(api_module, "_with_storage", _patched_with_storage)

    resp = await server_client.get("/v1/health")
    raw = resp.text
    body = resp.json()

    assert sentinel_token not in raw
    assert "user:" not in raw  # userinfo separator gone too
    assert "api_key=" not in raw  # query param gone

    # Positive check: a sanitised host still surfaces (so we didn't just
    # drop the field altogether and call that "secure").
    assert body["providers"]["llm"]["base_url"] == "https://api.example.com/v1"
    assert (
        body["providers"]["embedding"]["base_url"]
        == "https://api.example.com/v1"
    )
