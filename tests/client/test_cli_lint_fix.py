"""``dikw client lint propose|proposals|apply`` CLI tests.

End-to-end against the in-memory ASGI server: the CLI uses
``patch_transport_factory`` to route through the same ASGI client that
serves the lint propose / apply tasks, so the assertions cover the
full submit → events → result render path.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from dikw_core import api as api_module
from dikw_core.cli import app
from dikw_core.schemas import DocumentRecord, Layer
from dikw_core.server.runtime import ServerRuntime


@pytest.fixture()
async def seeded_wiki(client_wiki: Path) -> Path:
    """``client_wiki`` with two pages seeded — one target + one source
    that links to a broken alias. Lives as an async fixture so the
    sync CliRunner-based tests don't have to drive asyncio themselves
    (CliRunner.invoke calls ``asyncio.run`` internally and conflicts
    with a test-level event loop)."""
    await _seed(client_wiki)
    return client_wiki


def _run(args: list[str]) -> Any:
    return CliRunner().invoke(app, args)


def _wiki_doc_id(path: str) -> str:
    from dikw_core.domains.data.path_norm import normalize_path

    return f"wiki:{normalize_path(path)}"


async def _seed(wiki_root: Path) -> None:
    target = wiki_root / "wiki/concepts/foo-bar.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        "---\nid: K-foobar\ntype: concept\ntitle: Foo Bar\n"
        "created: 2026-05-09T00:00:00+00:00\n"
        "updated: 2026-05-09T00:00:00+00:00\n---\n\n"
        "# Foo Bar\n\nbody\n",
        encoding="utf-8",
    )
    src = wiki_root / "wiki/concepts/source.md"
    src.write_text(
        "---\nid: K-source\ntype: concept\ntitle: Source\n"
        "created: 2026-05-09T00:00:00+00:00\n"
        "updated: 2026-05-09T00:00:00+00:00\n---\n\n"
        "# Source\n\nSee [[fooo bar]] for context.\n",
        encoding="utf-8",
    )
    _cfg, _root, storage = await api_module._with_storage(wiki_root)
    try:
        for path, title in [
            ("wiki/concepts/foo-bar.md", "Foo Bar"),
            ("wiki/concepts/source.md", "Source"),
        ]:
            await storage.upsert_document(
                DocumentRecord(
                    doc_id=_wiki_doc_id(path), path=path, title=title,
                    hash=f"hash-{path}", mtime=0.0,
                    layer=Layer.WIKI, active=True,
                )
            )
    finally:
        await storage.close()


def test_lint_propose_apply_cli_full_loop(
    asgi_client: tuple[Any, ServerRuntime],
    patch_transport_factory: Callable[[], None],
    seeded_wiki: Path,
) -> None:
    patch_transport_factory()

    # 1. propose --rule broken_wikilink
    r1 = _run(["lint", "propose", "--rule", "broken_wikilink", "--plain"])
    assert r1.exit_code == 0, r1.stdout
    assert "succeeded" in r1.stdout.lower()
    assert "apply with:" in r1.stdout.lower()

    # Extract task_id from `dikw client lint apply <id>` hint line.
    task_id: str | None = None
    for line in r1.stdout.splitlines():
        if "lint apply" in line.lower():
            task_id = line.strip().split()[-1]
            break
    assert task_id is not None, f"could not extract task_id from: {r1.stdout!r}"

    # 2. proposals listing should include the new propose task as not-applied.
    # Use JSON format so the assertion sees the full task_id (the rich
    # Table renderer truncates long UUIDs with ellipses).
    r2 = _run(["lint", "proposals", "--format", "json"])
    assert r2.exit_code == 0, r2.stdout
    assert task_id in r2.stdout

    # 3. apply <task_id>
    r3 = _run(["lint", "apply", task_id, "--plain"])
    assert r3.exit_code == 0, r3.stdout
    assert "applied" in r3.stdout.lower()

    # 4. on-disk file now references the resolved target.
    rewritten = (seeded_wiki / "wiki/concepts/source.md").read_text(
        encoding="utf-8"
    )
    assert "[[Foo Bar]]" in rewritten


def test_lint_propose_invalid_rule_rejected_by_server(
    asgi_client: tuple[Any, ServerRuntime],
    patch_transport_factory: Callable[[], None],
) -> None:
    """An unknown --rule value is rejected at the pydantic-body validator
    on the server. The CLI surfaces that as a non-zero exit."""
    patch_transport_factory()
    r = _run(["lint", "propose", "--rule", "no_such_rule"])
    assert r.exit_code != 0


def test_lint_apply_unknown_proposal_id_fails(
    asgi_client: tuple[Any, ServerRuntime],
    patch_transport_factory: Callable[[], None],
) -> None:
    patch_transport_factory()
    r = _run(["lint", "apply", "no-such-id", "--plain"])
    assert r.exit_code != 0
    assert "failed" in r.stdout.lower() or "not found" in r.stdout.lower()
