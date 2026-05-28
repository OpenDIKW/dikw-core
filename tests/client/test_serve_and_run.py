"""Tests for ``dikw client serve-and-run``.

Two layers of coverage:

* Pure-function unit tests for the helpers (``find_free_port``,
  ``build_*``, ``terminate``, ``wait_until_ready``). These run in
  milliseconds and don't spawn any real subprocess.
* One end-to-end integration test that actually launches ``dikw serve``
  in a subprocess, runs ``dikw client info`` against it, and verifies
  the server is gone afterwards. This is the only test in the suite
  that exercises the full subprocess-management path; it's marked
  ``slow`` so CI can split it from the fast suite if it turns flaky on
  a constrained runner.

We deliberately don't mock ``subprocess.Popen`` for the integration
test — the bug surface that ``serve-and-run`` exists to manage
(orphaned processes, race between bind() and the first request,
shutdown signals on Windows vs POSIX) is *exactly* what mocks would
paper over.
"""

from __future__ import annotations

import dataclasses
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest

from dikw_core.client import serve_and_run as sar
from tests.fakes import init_test_base

# ---- pure-function helpers ---------------------------------------------


def test_find_free_port_returns_a_usable_port() -> None:
    port = sar.find_free_port()
    assert 1 <= port <= 65535
    # We can immediately rebind to it — the helper closed its socket.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("127.0.0.1", port))


def test_build_server_command_includes_token_only_when_set() -> None:
    base = sar.ServeAndRunOptions(
        base=Path("/tmp/w"),
        host="127.0.0.1",
        port=8765,
        token=None,
        ready_timeout=30.0,
        keep_alive=False,
        log_level="warning",
        inner_cmd=["status"],
    )
    cmd = sar.build_server_command(base)
    assert cmd[:3] == [sys.executable, "-m", "dikw_core.cli"]
    assert "serve" in cmd
    assert "--token" not in cmd

    with_token = sar.build_server_command(
        dataclasses.replace(base, token="s3cret")
    )
    assert "--token" in with_token
    assert with_token[with_token.index("--token") + 1] == "s3cret"


def test_effective_token_prefers_explicit_then_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``--token`` wins over ``DIKW_SERVER_TOKEN`` env, but env still
    activates when --token is unset — otherwise an authenticated
    server inherited via env never gets the right Authorization
    header on the readiness probe."""
    monkeypatch.delenv("DIKW_SERVER_TOKEN", raising=False)
    assert sar._effective_token(None) is None
    assert sar._effective_token("explicit") == "explicit"

    monkeypatch.setenv("DIKW_SERVER_TOKEN", "env-tok")
    assert sar._effective_token(None) == "env-tok"
    assert sar._effective_token("explicit") == "explicit"

    # Empty env value → treated as unset (avoids "" being interpreted
    # as "auth required with empty bearer").
    monkeypatch.setenv("DIKW_SERVER_TOKEN", "")
    assert sar._effective_token(None) is None


def test_build_inner_env_sets_url_and_optional_token() -> None:
    env = sar.build_inner_env("127.0.0.1", 9123, token=None)
    assert env["DIKW_SERVER_URL"] == "http://127.0.0.1:9123"
    assert "DIKW_SERVER_TOKEN" not in env


def test_client_host_brackets_ipv6_literals() -> None:
    """IPv6 hosts must be bracketed in the URL — ``http://::1:PORT``
    is an invalid URI (the colons collide with the port separator),
    so the server is unreachable from a client that took ``--host ::1``
    at face value."""
    assert sar._client_host("::1") == "[::1]"
    assert sar._client_host("2001:db8::1") == "[2001:db8::1]"
    # Already bracketed → leave alone.
    assert sar._client_host("[::1]") == "[::1]"
    # IPv6 wildcard maps to bracketed loopback.
    assert sar._client_host("::") == "[::1]"
    # IPv4 / hostnames untouched.
    assert sar._client_host("127.0.0.1") == "127.0.0.1"
    assert sar._client_host("localhost") == "localhost"
    # IPv4 wildcards still rewrite to loopback.
    assert sar._client_host("0.0.0.0") == "127.0.0.1"

    # Wildcard server bind must NOT leak into the client URL — the inner
    # CLI would try to dial 0.0.0.0 (not routable) and fail. Loopback is
    # the only sensible client-side rewrite when the server bound to
    # all-interfaces.
    env_with = sar.build_inner_env("0.0.0.0", 9123, token="abc")
    assert env_with["DIKW_SERVER_URL"] == "http://127.0.0.1:9123"
    assert env_with["DIKW_SERVER_TOKEN"] == "abc"

    env_v6 = sar.build_inner_env("::", 9123, token=None)
    assert env_v6["DIKW_SERVER_URL"] == "http://[::1]:9123"


def test_wait_until_ready_times_out_when_nothing_listens() -> None:
    """A port that nobody is bound to must surface as ``TimeoutError``
    quickly — the helper's job is to give up gracefully, not to hang
    the CLI on a wedged server."""
    port = sar.find_free_port()
    start = time.monotonic()
    with pytest.raises(TimeoutError):
        sar.wait_until_ready(f"http://127.0.0.1:{port}", timeout=0.5)
    elapsed = time.monotonic() - start
    # Bounded: the timeout is 0.5s, plus a few ms slack for the last
    # poll to complete. If we ever exceed 5s here, the deadline math
    # is broken.
    assert elapsed < 5.0


def test_wait_until_ready_sends_bearer_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the spawned server requires a token, the readiness probe
    must include ``Authorization: Bearer <token>``. Without it the
    server returns 401 and the caller waits out the entire timeout
    even though everything is healthy."""
    seen_headers: list[dict[str, str]] = []

    class _Resp:
        status = 200

        def __enter__(self) -> _Resp:
            return self

        def __exit__(self, *_: object) -> None:
            return None

    def fake_urlopen(req: Any, timeout: float) -> _Resp:  # type: ignore[no-untyped-def]
        seen_headers.append(dict(req.headers))
        return _Resp()

    monkeypatch.setattr(sar.urllib.request, "urlopen", fake_urlopen)
    sar.wait_until_ready(
        "http://127.0.0.1:9999", timeout=1.0, token="s3cret"
    )
    assert seen_headers, "urlopen was never called"
    # urllib title-cases header names ("Authorization").
    assert seen_headers[0].get("Authorization") == "Bearer s3cret"


def test_terminate_is_a_noop_for_already_exited_process() -> None:
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    # Should not raise even though the process has been gone for ms.
    sar.terminate(proc)


def test_build_inner_command_prefixes_bare_verb_with_client() -> None:
    """Inner argv against the spawned server is always a ``dikw client``
    invocation by definition, so a bare verb must be auto-prefixed.
    Otherwise the README quickstart ``dikw client serve-and-run --
    ingest --no-embed`` would spawn ``python -m dikw_core.cli ingest``
    and crash with ``No such command`` (top-level short names are gone
    in 0.1.0)."""
    argv = sar.build_inner_command(["ingest", "--no-embed"])
    # Tail must be the rewritten inner argv, head is the python-m shim.
    assert argv[-3:] == ["client", "ingest", "--no-embed"]
    assert argv[:3] == [sys.executable, "-m", "dikw_core.cli"]


def test_build_inner_command_preserves_explicit_client_prefix() -> None:
    """If the caller already spelled out ``client``, don't double-prefix.
    Power users (and our slow integration tests) pass the explicit form."""
    argv = sar.build_inner_command(["client", "info"])
    assert argv[-2:] == ["client", "info"]
    # Sanity: no double prefix, exactly one ``client`` token in the inner argv.
    assert argv.count("client") == 1


def test_run_refuses_empty_inner_cmd(tmp_path: Path) -> None:
    """No inner command is a usage error, not an implicit no-op."""
    init_test_base(tmp_path / "knowledge", description="empty inner cmd")
    rc = sar.run(
        sar.ServeAndRunOptions(
            base=tmp_path / "knowledge",
            host="127.0.0.1",
            port=0,
            token=None,
            ready_timeout=5.0,
            keep_alive=False,
            log_level="warning",
            inner_cmd=[],
        )
    )
    assert rc == 2


# ---- end-to-end integration --------------------------------------------


@pytest.fixture()
def wiki_for_serve(tmp_path: Path) -> Iterator[Path]:
    wiki = tmp_path / "knowledge"
    init_test_base(wiki, description="serve-and-run integration")
    yield wiki


@pytest.mark.slow
def test_serve_and_run_round_trips_info_command(
    wiki_for_serve: Path,
) -> None:
    """End-to-end: spawn the real server, run ``client info``, observe
    a successful exit and verify the server bound port is no longer
    accepting connections after we return.

    Marked ``slow`` because spinning up uvicorn + storage migration
    takes ~1-2s in cold cache; the unit tests above cover the
    fast-path logic.
    """
    port = sar.find_free_port()
    rc = sar.run(
        sar.ServeAndRunOptions(
            base=wiki_for_serve,
            host="127.0.0.1",
            port=port,
            token=None,
            ready_timeout=30.0,
            keep_alive=False,
            log_level="warning",
            inner_cmd=["client", "info"],
        )
    )
    assert rc == 0

    # Verify the server is gone — any failure mode is acceptable
    # (connect refused, hung socket on Windows TerminateProcess,
    # short read mid-shutdown). The contract we care about is that
    # this CLI invocation returned and the process is no longer
    # serving requests.
    try:
        with httpx.Client(timeout=0.5) as c:
            resp = c.get(f"http://127.0.0.1:{port}/v1/healthz")
            assert resp.status_code >= 400 or resp.status_code == 0
    except (
        httpx.ConnectError,
        httpx.RemoteProtocolError,
        httpx.ReadError,
        httpx.ReadTimeout,
        httpx.ConnectTimeout,
    ):
        pass  # expected — server is gone


@pytest.mark.slow
def test_serve_and_run_propagates_inner_failure(
    wiki_for_serve: Path,
) -> None:
    """If the inner command exits non-zero, ``serve-and-run`` returns
    that exit code so shell scripts can branch on it. Use a known-bad
    invocation (``--retrieval bogus`` is rejected by the server) so we
    don't depend on real provider keys to produce the failure."""
    port = sar.find_free_port()
    rc = sar.run(
        sar.ServeAndRunOptions(
            base=wiki_for_serve,
            host="127.0.0.1",
            port=port,
            token=None,
            ready_timeout=30.0,
            keep_alive=False,
            log_level="warning",
            inner_cmd=[
                "client",
                "eval",
                "--dataset",
                "does-not-exist",
                "--plain",
                # ``--wait`` so the test sees the dataset-not-found
                # failure propagate as a non-zero inner exit code.
                # Without it the inner CLI just prints the task handle
                # and exits 0 (task fails later, server-side).
                "--wait",
            ],
        )
    )
    assert rc != 0
