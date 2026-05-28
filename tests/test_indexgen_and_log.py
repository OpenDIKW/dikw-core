from __future__ import annotations

from pathlib import Path

from dikw_core.domains.knowledge.indexgen import INDEX_PATH, regenerate_index
from dikw_core.domains.knowledge.log import LOG_PATH, render_log
from dikw_core.domains.knowledge.page import build_page, write_page
from dikw_core.schemas import KnowledgeLogEntry


def test_regenerate_index_groups_by_folder(tmp_path: Path) -> None:
    (tmp_path / "knowledge").mkdir()
    write_page(
        tmp_path,
        build_page(
            title="DIKW pyramid",
            body="# DIKW pyramid\n\nA layered model for knowledge.",
            type_="concept",
        ),
    )
    write_page(
        tmp_path,
        build_page(
            title="Andrej Karpathy",
            body="# Andrej Karpathy\n\nComputer scientist.",
            type_="entity",
        ),
    )
    out = regenerate_index(tmp_path, updated="2026-04-21T12:00:00+00:00")
    assert out == tmp_path / INDEX_PATH
    text = out.read_text(encoding="utf-8")
    assert "## concepts" in text
    assert "## entities" in text
    assert "DIKW pyramid" in text
    assert "Andrej Karpathy" in text
    # summary line should include the first paragraph
    assert "A layered model" in text


def test_regenerate_index_when_empty(tmp_path: Path) -> None:
    (tmp_path / "knowledge").mkdir()
    out = regenerate_index(tmp_path, updated="now")
    text = out.read_text(encoding="utf-8")
    assert "No knowledge pages yet" in text


def test_render_log_is_newest_first(tmp_path: Path) -> None:
    entries = [
        KnowledgeLogEntry(ts=1_000.0, action="ingest", src="sources/a.md"),
        KnowledgeLogEntry(ts=2_000.0, action="synth", src="sources/a.md", dst="knowledge/notes/a.md"),
    ]
    out = render_log(tmp_path, entries, updated="now")
    text = out.read_text(encoding="utf-8")
    # newer entry (synth, ts=2000) should come before the older ingest entry
    assert text.index("synth") < text.index("ingest")
    assert "knowledge/notes/a.md" in text
    assert out == tmp_path / LOG_PATH


def test_render_log_empty(tmp_path: Path) -> None:
    out = render_log(tmp_path, [], updated="now")
    assert "No activity recorded yet" in out.read_text(encoding="utf-8")
