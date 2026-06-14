"""Top-level CLI tests for the local-only commands.

The only top-level commands that run in-process are ``version``,
``init``, ``serve`` and the ``auth`` subgroup. Every HTTP-bound command
lives under ``dikw client *`` — there are no top-level aliases.

The remote command surface (``dikw client status``, ``dikw client
ingest`` …) is exercised end-to-end against an in-memory ASGI server in
``tests/client/test_cli_e2e.py``. This file's job is to keep the
local-only commands honest and to guard against splice regressions.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from dikw_core.cli import app

from .conftest import removed_top_level_short_names

runner = CliRunner()


def test_version_prints_non_empty() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert result.stdout.strip()


def test_init_scaffolds_expected_tree(tmp_path: Path) -> None:
    wiki = tmp_path / "my-wiki"
    result = runner.invoke(
        app, ["init", str(wiki), "--description", "phase-5 cli test"]
    )
    assert result.exit_code == 0, result.stdout

    assert (wiki / "dikw.yml").is_file()
    assert (wiki / "sources").is_dir()
    # The default taxonomy scaffolds one folder per declared category, SINGULAR
    # (entity / concept / note) — no more plural type folders and no generated
    # knowledge/index.md or knowledge/log.md (the category tree is the catalogue;
    # the knowledge_log table is the history). See ADR-0004.
    assert (wiki / "knowledge" / "entity" / ".gitkeep").is_file()
    assert (wiki / "knowledge" / "concept" / ".gitkeep").is_file()
    assert (wiki / "knowledge" / "note" / ".gitkeep").is_file()
    assert not (wiki / "knowledge" / "index.md").exists()
    assert not (wiki / "knowledge" / "log.md").exists()
    assert (wiki / "prompts" / ".gitkeep").is_file()
    assert (wiki / "wisdom" / ".gitkeep").is_file()
    assert (wiki / ".dikw").is_dir()
    assert (wiki / ".gitignore").read_text().strip() == ".dikw/"


def test_init_refuses_to_overwrite_existing_wiki(tmp_path: Path) -> None:
    wiki = tmp_path / "my-wiki"
    first = runner.invoke(app, ["init", str(wiki)])
    assert first.exit_code == 0

    second = runner.invoke(app, ["init", str(wiki)])
    assert second.exit_code == 1


def test_serve_help_lists_options(monkeypatch: pytest.MonkeyPatch) -> None:
    """``dikw serve --help`` should at least mention the bind-host
    and token flags so the operator can configure auth posture
    without reading source code.

    Forces a wide terminal so rich/typer doesn't wrap option names
    across visual lines (CI's narrow default broke ``--host`` apart).
    """
    monkeypatch.setenv("COLUMNS", "200")
    monkeypatch.setenv("TERM", "dumb")  # disable colour escapes
    result = runner.invoke(app, ["serve", "--help"])
    assert result.exit_code == 0
    out = result.stdout
    assert "--host" in out
    assert "--token" in out
    assert "--port" in out


def test_root_bootstraps_client_telemetry_only_for_client_subgroup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The ``_root`` callback must invoke the env-only client OTel bootstrap
    ONLY for the ``client`` subgroup. Local commands (version/init/auth) make no
    httpx calls, and ``serve`` wires its OWN telemetry in the server lifespan —
    bootstrapping a client provider for it would lose OTel's process-once
    set_tracer_provider race and silently disable server telemetry.
    """
    import dikw_core.cli as cli_mod

    calls: list[str] = []

    def _spy(*, version: str) -> bool:
        calls.append(version)
        return False  # report inactive so no atexit/provider side effects fire

    monkeypatch.setattr(cli_mod, "configure_client_telemetry_from_env", _spy)

    # non-client subcommands never bootstrap client telemetry
    for args in (["version"], ["init", "--help"], ["serve", "--help"], ["auth", "--help"]):
        result = runner.invoke(app, args)
        assert result.exit_code == 0, (args, result.stdout)
    assert calls == [], f"client telemetry bootstrapped for a non-client command: {calls}"

    # the client subgroup bootstraps exactly once (--help still runs _root first)
    result = runner.invoke(app, ["client", "--help"])
    assert result.exit_code == 0, result.stdout
    assert len(calls) == 1


def test_root_registers_atexit_flush_when_client_telemetry_activates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """On a successful client-telemetry activation, ``_root`` must register an
    ``atexit`` flush — the short-lived CLI would otherwise exit before the
    BatchSpanProcessor drains its queue, dropping the client span (and with it
    the client→server stitch)."""
    import dikw_core.cli as cli_mod

    monkeypatch.setattr(
        cli_mod, "configure_client_telemetry_from_env", lambda *, version: True
    )
    registered: list[object] = []
    monkeypatch.setattr(cli_mod.atexit, "register", lambda fn: registered.append(fn))

    result = runner.invoke(app, ["client", "--help"])
    assert result.exit_code == 0, result.stdout
    assert cli_mod.shutdown_telemetry in registered


def test_top_level_app_registers_only_local_commands() -> None:
    """The top-level Typer app must register exactly the four local-only
    commands + the ``client`` subgroup — never any HTTP-bound verb.

    Inspects the live registry instead of parsing ``--help`` text;
    Rich's box-drawing glyphs make text scraping fragile, and the
    registry is the actual source of truth Typer dispatches against.
    """
    top_level = {c.name for c in app.registered_commands if c.name} | {
        g.name for g in app.registered_groups if g.name
    }
    assert top_level == {"version", "init", "serve", "auth", "client"}, (
        f"unexpected top-level surface: {sorted(top_level)}"
    )
    for forbidden in removed_top_level_short_names():
        assert forbidden not in top_level, (
            f"HTTP-bound short name {forbidden!r} leaked into top-level app"
        )
