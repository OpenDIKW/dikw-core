"""Tests for ``build_import`` dispatching non-md inputs through a
converter plugin.

The dispatch path is single-file-only in v1: a non-md single file is
converted to md+assets in a temp staging directory, then the standard
import flow packages the staging directory like any other md tree.
Directory imports keep the strict md-only behaviour — convert files
individually first if you have mixed inputs.

Tests use a tiny ``_FakeConverter`` instead of a real PDF/EPUB plugin
so dikw-core's CI never grows ML-heavy dependencies; the contract
between importer and converter is what we want to lock in here.
"""

from __future__ import annotations

import gzip
import tarfile
import tempfile
from collections.abc import Callable
from io import BytesIO
from pathlib import Path
from typing import IO

import pytest

from dikw_core.client.converters import Converter, ConverterError
from dikw_core.client.importer import SourceImportError, build_import


class _FakeConverter:
    """Mirrors the user-facing convention: writes
    ``output_dir/<stem>.md`` and ``output_dir/assets/<stem>.<ext>`` so
    the original input is preserved as an asset alongside the converted
    markdown."""

    name = "fake"
    extensions = (".fake",)

    def __init__(self, body: str = "# fake doc\n", fail: bool = False) -> None:
        self._body = body
        self._fail = fail

    def convert(self, input_path: Path, output_dir: Path) -> None:
        if self._fail:
            raise RuntimeError("converter exploded")
        output_dir.mkdir(parents=True, exist_ok=True)
        # md_inspect only follows image-style refs (``![alt](path)`` or
        # ``![[file]]``). Plugin authors must image-ref every asset they
        # emit — including the original-as-provenance — so the importer
        # picks them up and the orphan check passes.
        body = (
            f"{self._body}\n\n![original](assets/{input_path.name})\n"
        )
        (output_dir / f"{input_path.stem}.md").write_text(
            body, encoding="utf-8"
        )
        assets_dir = output_dir / "assets"
        assets_dir.mkdir(exist_ok=True)
        assets_dir.joinpath(input_path.name).write_bytes(input_path.read_bytes())


def _resolver(converter: Converter) -> Callable[[str], Converter]:
    return lambda ext: converter


def _tar_names(payload: IO[bytes]) -> list[str]:
    payload.seek(0)
    buf = BytesIO(payload.read())
    with (
        gzip.GzipFile(fileobj=buf, mode="rb") as gz,
        tarfile.TarFile(fileobj=gz, mode="r") as tf,
    ):
        return sorted(m.name for m in tf.getmembers())


def test_non_md_single_file_no_converter_fails(tmp_path: Path) -> None:
    fake = tmp_path / "paper.fake"
    fake.write_bytes(b"fake bytes")
    with pytest.raises(SourceImportError) as exc:
        build_import(fake)
    assert ".fake" in str(exc.value)
    assert "dikw-plugins" in str(exc.value)


def test_non_md_single_file_dispatches_to_converter(tmp_path: Path) -> None:
    fake = tmp_path / "paper.fake"
    fake.write_bytes(b"original-fake-bytes")

    converter = _FakeConverter(body="# Paper\nconverted body\n")

    bundle = build_import(fake, converter_for=_resolver(converter))
    try:
        names = _tar_names(bundle.payload)
    finally:
        bundle.close()

    assert "sources/paper/paper.md" in names
    assert "sources/paper/assets/paper.fake" in names


def test_md_file_passes_through_without_converter(tmp_path: Path) -> None:
    md = tmp_path / "note.md"
    md.write_text("# Note\nbody\n", encoding="utf-8")

    bundle = build_import(md)
    try:
        names = _tar_names(bundle.payload)
    finally:
        bundle.close()

    assert names == ["sources/note.md"]


def test_converter_error_propagates_as_source_import_error(tmp_path: Path) -> None:
    fake = tmp_path / "paper.fake"
    fake.write_bytes(b"fake")

    def bad_resolver(ext: str) -> Converter:
        raise ConverterError("conflict between marker and mineru for '.fake'")

    with pytest.raises(SourceImportError) as exc:
        build_import(fake, converter_for=bad_resolver)
    msg = str(exc.value)
    assert "marker" in msg
    assert "mineru" in msg


def test_converter_runtime_failure_wrapped(tmp_path: Path) -> None:
    fake = tmp_path / "paper.fake"
    fake.write_bytes(b"fake")

    converter = _FakeConverter(fail=True)

    with pytest.raises(SourceImportError) as exc:
        build_import(fake, converter_for=_resolver(converter))
    msg = str(exc.value)
    assert "fake" in msg
    assert "converter exploded" in msg


def test_converter_producing_nothing_is_an_error(tmp_path: Path) -> None:
    """A buggy plugin that returns without writing anything must surface
    as a SourceImportError, not as a confusing 'no markdown files found'
    later in _resolve_input."""

    class _EmptyConverter:
        name = "empty"
        extensions = (".fake",)

        def convert(self, input_path: Path, output_dir: Path) -> None:
            output_dir.mkdir(parents=True, exist_ok=True)
            # writes nothing

    fake = tmp_path / "paper.fake"
    fake.write_bytes(b"fake")

    with pytest.raises(SourceImportError) as exc:
        build_import(fake, converter_for=_resolver(_EmptyConverter()))
    assert "empty" in str(exc.value) or "no markdown" in str(exc.value).lower()


def test_staging_cleaned_up_after_build(tmp_path: Path) -> None:
    """Every successful import should clean up its staging temp dir —
    no leftover ``dikw-import-*`` directories in tempfile.gettempdir()."""
    fake = tmp_path / "paper.fake"
    fake.write_bytes(b"fake")
    converter = _FakeConverter()

    tmp_root = Path(tempfile.gettempdir())
    before = {
        p
        for p in tmp_root.iterdir()
        if p.name.startswith("dikw-import-")
    }
    bundle = build_import(fake, converter_for=_resolver(converter))
    try:
        _tar_names(bundle.payload)
    finally:
        bundle.close()
    after = {
        p
        for p in tmp_root.iterdir()
        if p.name.startswith("dikw-import-")
    }
    assert after.issubset(before)


def test_staging_cleaned_up_on_failure(tmp_path: Path) -> None:
    """A converter that raises shouldn't leak its temp dir either."""
    fake = tmp_path / "paper.fake"
    fake.write_bytes(b"fake")
    converter = _FakeConverter(fail=True)

    tmp_root = Path(tempfile.gettempdir())
    before = {
        p
        for p in tmp_root.iterdir()
        if p.name.startswith("dikw-import-")
    }
    with pytest.raises(SourceImportError):
        build_import(fake, converter_for=_resolver(converter))
    after = {
        p
        for p in tmp_root.iterdir()
        if p.name.startswith("dikw-import-")
    }
    assert after.issubset(before)
