"""Engine-side tests for per-file ingest errors (PR 4).

The contract:
- Per-file failures land on ``IngestReport.errors`` instead of crashing
  the run, so a single bad markdown file in a 1000-file directory
  doesn't blow away the whole pass.
- Each failure also fires a ``partial("file_error", …)`` event so a
  streaming subscriber (CLI progress widget, NDJSON task stream) can
  render the failure live instead of waiting for the final report.
- ``IngestError.kind`` is one of ``unsupported_format`` / ``parse_error``
  / ``read_error`` / ``storage_error`` so callers can branch without
  pattern-matching ``message``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dikw_core import api
from dikw_core.config import dump_config_yaml, load_config

from .fakes import FakeEmbeddings, init_test_wiki
from .test_progress_reporter import ListReporter


def _seed_wiki(tmp_path: Path) -> Path:
    init_test_wiki(tmp_path)
    src_dir = tmp_path / "sources" / "demo"
    src_dir.mkdir(parents=True, exist_ok=True)
    return src_dir


def _widen_pattern_to_all(tmp_path: Path) -> None:
    """Patch ``dikw.yml`` so non-``.md`` files reach ``parse_any`` —
    the default ``**/*.md`` filters them out before our error path
    even runs, so triggering ``unsupported_format`` requires this."""
    cfg_path = tmp_path / "dikw.yml"
    cfg = load_config(cfg_path)
    cfg.sources[0].pattern = "**/*"
    cfg_path.write_text(dump_config_yaml(cfg), encoding="utf-8")


@pytest.mark.asyncio
async def test_ingest_records_parse_error_and_continues(tmp_path: Path) -> None:
    """A markdown file with broken YAML front-matter must produce one
    ``parse_error`` row in ``report.errors`` while the sibling good
    file still ingests successfully."""
    src_dir = _seed_wiki(tmp_path)
    (src_dir / "good.md").write_text(
        "# Good\n\nValid markdown content.\n", encoding="utf-8"
    )
    # ``: :`` after a key is a YAML scanner error — frontmatter.loads
    # surfaces it as yaml.scanner.ScannerError.
    (src_dir / "broken.md").write_text(
        "---\nbroken: : :\n---\n# Title\n", encoding="utf-8"
    )

    report = await api.ingest(tmp_path, embedder=FakeEmbeddings())

    assert report.scanned == 2
    assert report.added == 1, "good.md should still ingest cleanly"
    assert len(report.errors) == 1
    err = report.errors[0]
    assert err.path.endswith("broken.md")
    assert err.kind == "parse_error"
    assert err.message  # non-empty


@pytest.mark.asyncio
async def test_ingest_silently_skips_unsupported_format(tmp_path: Path) -> None:
    """A wide glob (``**/*``) sweeping non-markdown files into the run
    must NOT inflate ``report.errors`` — the prior behaviour was to
    skip silently, and a vault with thousands of asset files would
    otherwise drown the error channel. ``parse_error`` / ``read_error``
    / ``storage_error`` are the user-actionable surfaces this PR
    opens up."""
    src_dir = _seed_wiki(tmp_path)
    _widen_pattern_to_all(tmp_path)
    (src_dir / "good.md").write_text("# OK\n", encoding="utf-8")
    (src_dir / "data.txt").write_text("plain text\n", encoding="utf-8")
    (src_dir / "image.png").write_bytes(b"\x89PNG\r\n\x1a\n")

    report = await api.ingest(tmp_path, embedder=FakeEmbeddings())

    assert report.errors == ()
    # Only the .md file landed in scanned — preserves the prior wire
    # contract where ``scanned`` matches "supported files we tried".
    assert report.scanned == 1
    assert report.added == 1


@pytest.mark.asyncio
async def test_ingest_emits_file_error_partial_event(tmp_path: Path) -> None:
    """Per-file failures must fire on the reporter's ``partial`` channel
    with kind=``file_error`` so a streaming subscriber sees them live —
    the report-only path forces consumers to buffer the whole run before
    learning anything went wrong."""
    src_dir = _seed_wiki(tmp_path)
    (src_dir / "broken.md").write_text(
        "---\n: : :\n---\n# Title\n", encoding="utf-8"
    )

    reporter = ListReporter()
    await api.ingest(tmp_path, embedder=FakeEmbeddings(), reporter=reporter)

    file_errors = [
        ev for ev in reporter.events
        if ev.kind == "partial" and ev.payload.get("kind") == "file_error"
    ]
    assert len(file_errors) == 1
    payload = file_errors[0].payload["payload"]
    assert payload["kind"] == "parse_error"
    assert payload["path"].endswith("broken.md")
    assert payload["message"]


@pytest.mark.asyncio
async def test_ingest_records_storage_error_via_monkeypatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Force a post-parse failure (storage.replace_chunks raises) so
    the ``storage_error`` branch surfaces. We monkeypatch the storage
    method rather than rely on a real-world break because the only
    organic triggers are infra-level (DB down, disk full)."""
    src_dir = _seed_wiki(tmp_path)
    (src_dir / "boom.md").write_text("# Doom\n\nBody.\n", encoding="utf-8")

    original = api._with_storage

    async def patched(path: object) -> object:
        cfg, root, storage = await original(path)  # type: ignore[arg-type]

        async def boom(doc_id: object, chunks: object) -> object:
            del doc_id, chunks
            raise RuntimeError("simulated storage outage")

        storage.replace_chunks = boom  # type: ignore[method-assign]
        return cfg, root, storage

    monkeypatch.setattr(api, "_with_storage", patched)

    report = await api.ingest(tmp_path, embedder=FakeEmbeddings())
    assert len(report.errors) == 1
    err = report.errors[0]
    assert err.kind == "storage_error"
    assert err.path.endswith("boom.md")
    assert "simulated storage outage" in err.message


@pytest.mark.asyncio
async def test_ingest_records_read_error_for_invalid_utf8(
    tmp_path: Path,
) -> None:
    """A ``.md`` file containing non-UTF-8 bytes raises
    ``UnicodeDecodeError`` (a ``ValueError`` subclass, not ``OSError``).
    Without explicit handling it lands in the ``parse_error`` catch-all,
    which misleads callers branching on ``kind`` — UTF-8 decode failure
    is a read-side problem (the file as bytes can't be turned into text),
    not a parser-side syntax problem."""
    src_dir = _seed_wiki(tmp_path)
    # Latin-1 byte 0xA1 (¡) is invalid UTF-8 lead.
    (src_dir / "binary.md").write_bytes(
        b"# Title\n\nbody with bad byte: \xa1 here\n"
    )
    (src_dir / "good.md").write_text("# OK\n", encoding="utf-8")

    report = await api.ingest(tmp_path, embedder=FakeEmbeddings())

    assert report.added == 1
    assert len(report.errors) == 1
    err = report.errors[0]
    assert err.kind == "read_error", (
        f"UnicodeDecodeError must surface as read_error, got {err.kind}"
    )
    assert err.path.endswith("binary.md")


@pytest.mark.asyncio
async def test_storage_error_deactivates_doc_so_retry_reprocesses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A ``storage_error`` mid-file leaves the doc row written but its
    chunks not. Without explicit deactivation, the next ingest under
    an unchanged content hash hits the early-skip arm and the doc
    stays half-indexed forever. The fix: deactivate on storage_error
    so the skip arm falls through and re-runs the pipeline.
    """
    src_dir = _seed_wiki(tmp_path)
    (src_dir / "boom.md").write_text("# Doom\n\nbody.\n", encoding="utf-8")

    original = api._with_storage
    fail_chunks = True

    async def patched(path: object) -> object:
        cfg, root, storage = await original(path)  # type: ignore[arg-type]
        original_replace_chunks = storage.replace_chunks

        async def maybe_boom(doc_id: object, chunks: object) -> object:
            if fail_chunks:
                raise RuntimeError("simulated storage outage")
            return await original_replace_chunks(doc_id, chunks)

        storage.replace_chunks = maybe_boom  # type: ignore[method-assign]
        return cfg, root, storage

    monkeypatch.setattr(api, "_with_storage", patched)

    first = await api.ingest(tmp_path, embedder=FakeEmbeddings())
    assert len(first.errors) == 1
    assert first.errors[0].kind == "storage_error"

    # Confirm the doc is parked as inactive.
    cfg, _root, storage = await original(tmp_path)
    del cfg
    try:
        doc_id = api._doc_id_for(api.Layer.SOURCE, "sources/demo/boom.md")
        existing = await storage.get_document(doc_id)
        assert existing is not None and existing.active is False
    finally:
        await storage.close()

    # Lift the failure and retry — the early-skip must NOT silently
    # treat the inactive doc as "unchanged"; the pipeline must re-run.
    fail_chunks = False
    second = await api.ingest(tmp_path, embedder=FakeEmbeddings())
    assert second.errors == ()
    assert second.added + second.updated == 1
    assert second.chunks >= 1


@pytest.mark.asyncio
async def test_ingest_idempotent_run_clears_errors(tmp_path: Path) -> None:
    """Errors are per-run, not persistent. A re-ingest after fixing the
    bad file should report zero errors — proves errors aren't leaking
    across calls via storage state."""
    src_dir = _seed_wiki(tmp_path)
    bad = src_dir / "broken.md"
    bad.write_text("---\nbroken: : :\n---\n# T\n", encoding="utf-8")

    first = await api.ingest(tmp_path, embedder=FakeEmbeddings())
    assert len(first.errors) == 1

    bad.write_text("# Fixed\n\nNow valid.\n", encoding="utf-8")
    second = await api.ingest(tmp_path, embedder=FakeEmbeddings())
    assert second.errors == ()
    assert second.added == 1
