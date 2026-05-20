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

    # 1. propose --rule broken_wikilink — ``--wait`` so the test sees
    # the succeeded summary + ``apply with:`` hint line instead of the
    # async-default task-handle JSON.
    r1 = _run(["client", "lint", "propose", "--rule", "broken_wikilink", "--plain", "--wait"])
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
    r2 = _run(["client", "lint", "proposals", "--format", "json"])
    assert r2.exit_code == 0, r2.stdout
    assert task_id in r2.stdout

    # 2.5. Each propose row in the JSON payload MUST carry its
    # ``result.proposals`` array. The 0.2.0 ``GET /v1/tasks`` is summary-
    # only (no ``result``), so ``lint proposals`` must fan out to
    # ``GET /v1/tasks/{id}/result`` to hydrate the propose payload
    # before serializing — otherwise the rendered table shows
    # ``proposals=0`` for every row even though there are real fixes
    # queued, and downstream agents see an empty ``proposals`` array
    # in JSON mode. Guard with a strict shape assertion.
    import json as _json2
    body2 = _json2.loads(r2.stdout)
    row = next(
        (r for r in body2["proposals"] if r.get("task_id") == task_id),
        None,
    )
    assert row is not None, body2
    assert isinstance(row.get("result"), dict), row
    assert row["result"].get("proposals"), row

    # 3. apply <task_id> — ``--wait`` so the test sees the apply report.
    r3 = _run(["client", "lint", "apply", task_id, "--plain", "--wait"])
    assert r3.exit_code == 0, r3.stdout
    assert "applied" in r3.stdout.lower()

    # 4. on-disk file now references the resolved target.
    rewritten = (seeded_wiki / "wiki/concepts/source.md").read_text(
        encoding="utf-8"
    )
    assert "[[Foo Bar]]" in rewritten

    # 5. After apply, ``proposals --format json`` must show the propose
    # task_id under ``applied_ids``. This guards the 0.2.0 regression
    # path: ``GET /v1/tasks`` no longer carries ``result``, so the
    # cross-reference now fans out to ``GET /v1/tasks/{id}/result`` for
    # every succeeded ``lint.apply`` row. If that fan-out is broken
    # (forgot to migrate / wrong key), ``applied_ids`` silently empties
    # while ``proposals`` still lists the propose row — and lint UX
    # gets stuck in "always pending" mode.
    import json as _json
    r4 = _run(["client", "lint", "proposals", "--format", "json"])
    assert r4.exit_code == 0, r4.stdout
    body = _json.loads(r4.stdout)
    assert task_id in body["applied_ids"], body

    # 6. table mode (opt-in via ``--format table``) must render the listing
    # without crashing — exercises the non-JSON ``render_lint_proposals_listing``
    # branch over the same hydrated rows.
    r5 = _run(["client", "lint", "proposals", "--format", "table"])
    assert r5.exit_code == 0, r5.stdout
    assert "lint proposals" in r5.stdout.lower()


@pytest.fixture()
async def orphan_seeded_wiki(client_wiki: Path) -> Path:
    """A 3-page wiki where:

    * ``parent.md`` and ``orphan.md`` share a ``sources`` entry — the
      orphan's parent-candidate score clears LINK_THRESHOLD, so the
      fixer must emit a ``link_from_existing`` proposal.
    * neither page has any inbound wikilinks, so both register as
      orphans; the fixer should propose a fix for the orphan only
      under ``--rule orphan_page`` filtering.
    """
    await _seed_orphan_pair(client_wiki)
    return client_wiki


async def _seed_orphan_pair(wiki_root: Path) -> None:
    parent = wiki_root / "wiki/concepts/parent-topic.md"
    parent.parent.mkdir(parents=True, exist_ok=True)
    parent.write_text(
        "---\nid: K-parent\ntype: concept\ntitle: Parent Topic\n"
        "sources: [sources/shared.md]\n"
        "created: 2026-05-09T00:00:00+00:00\n"
        "updated: 2026-05-09T00:00:00+00:00\n---\n\n"
        "# Parent Topic\n\nBroader prose about the topic to keep this "
        "comfortably past the stub byte threshold.\n",
        encoding="utf-8",
    )
    orphan = wiki_root / "wiki/concepts/orphan-detail.md"
    orphan.write_text(
        "---\nid: K-orphan\ntype: concept\ntitle: Orphan Detail\n"
        "sources: [sources/shared.md]\n"
        "created: 2026-05-09T00:00:00+00:00\n"
        "updated: 2026-05-09T00:00:00+00:00\n---\n\n"
        "# Orphan Detail\n\nNarrow observation that warrants its own "
        "page but no other page links to it, well past the stub bar.\n",
        encoding="utf-8",
    )
    _cfg, _root, storage = await api_module._with_storage(wiki_root)
    try:
        for path, title in [
            ("wiki/concepts/parent-topic.md", "Parent Topic"),
            ("wiki/concepts/orphan-detail.md", "Orphan Detail"),
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


def test_lint_orphan_page_propose_apply_roundtrip(
    asgi_client: tuple[Any, ServerRuntime],
    patch_transport_factory: Callable[[], None],
    orphan_seeded_wiki: Path,
) -> None:
    """End-to-end ``lint propose --rule orphan_page`` then ``lint apply``
    against the in-memory ASGI server: propose emits a link_from_existing
    proposal for at least one orphan, apply rewrites the parent body
    to include the orphan's backlink."""
    patch_transport_factory()

    r1 = _run(["client", "lint", "propose", "--rule", "orphan_page", "--plain", "--wait"])
    assert r1.exit_code == 0, r1.stdout
    assert "succeeded" in r1.stdout.lower()

    task_id: str | None = None
    for line in r1.stdout.splitlines():
        if "lint apply" in line.lower():
            task_id = line.strip().split()[-1]
            break
    assert task_id is not None, r1.stdout

    r2 = _run(["client", "lint", "apply", task_id, "--plain", "--wait"])
    assert r2.exit_code == 0, r2.stdout
    assert "applied" in r2.stdout.lower()

    # The orphan + parent share one ``sources`` entry → score 3.0,
    # exactly ``LINK_THRESHOLD``. Routing must pick link_from_existing
    # and the parent body must now reference the orphan via a backlink
    # under the stable ``## 相关`` heading. A leaf fallback at this score
    # would be a regression — assert the specific branch.
    parent_body = (orphan_seeded_wiki / "wiki/concepts/parent-topic.md").read_text(
        encoding="utf-8"
    )
    assert "[[Orphan Detail]]" in parent_body, (
        f"expected parent backlink at LINK_THRESHOLD score; got: {parent_body!r}"
    )


def test_lint_propose_invalid_rule_rejected_by_server(
    asgi_client: tuple[Any, ServerRuntime],
    patch_transport_factory: Callable[[], None],
) -> None:
    """An unknown --rule value is rejected at the pydantic-body validator
    on the server. The CLI surfaces that as a non-zero exit."""
    patch_transport_factory()
    r = _run(["client", "lint", "propose", "--rule", "no_such_rule"])
    assert r.exit_code != 0


def test_lint_apply_unknown_proposal_id_fails(
    asgi_client: tuple[Any, ServerRuntime],
    patch_transport_factory: Callable[[], None],
) -> None:
    patch_transport_factory()
    # ``--wait`` so the bad-id rejection surfaces as a failed task
    # final + non-zero exit instead of the async-default exit-0 handle.
    r = _run(["client", "lint", "apply", "no-such-id", "--plain", "--wait"])
    assert r.exit_code != 0
    assert "failed" in r.stdout.lower() or "not found" in r.stdout.lower()
