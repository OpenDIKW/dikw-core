"""``dikw client serve-and-run`` — one-shot server + command lifecycle.

Useful for "I just want to query my base without running ``dikw serve``
in another terminal".

Why two subprocesses instead of re-entering Typer in-process:
re-running click's main() from inside an already-active click command
breaks signal handling and exit-code propagation, and stacking two
long-running asyncio runs in one process is a flake magnet. The fork
cost is negligible next to the server's lifespan + storage migration.
"""

from __future__ import annotations

import dataclasses
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from .config import ENV_SERVER_TOKEN, ENV_SERVER_URL


def _effective_token(opts_token: str | None) -> str | None:
    """The token the spawned server will require — explicit ``--token``
    wins, otherwise the inherited ``DIKW_SERVER_TOKEN`` env var. The
    readiness probe must use this same value or it sees 401 and the
    caller waits out the entire ready timeout."""
    if opts_token is not None:
        return opts_token
    env = os.environ.get(ENV_SERVER_TOKEN)
    return env or None

# How long to wait between healthz polls. Short enough that a fast cold
# start (<1s) is barely-noticeable, long enough that we don't spam a
# struggling server during its lifespan startup.
_READY_POLL_INTERVAL = 0.1
# Per-poll HTTP timeout. Small because a healthy localhost endpoint
# replies in microseconds; a slower reply indicates the server is still
# booting and we should re-poll rather than wait it out on one request.
_READY_HTTP_TIMEOUT = 1.0
# How long to give the server after SIGTERM before we resort to SIGKILL.
# Uvicorn's graceful shutdown is sub-second on a base with no in-flight
# requests; 5s leaves headroom for slow lifespan teardown (storage
# adapter close, etc.) without wedging the CLI.
_TERMINATE_GRACE_SECONDS = 5.0
# Use ``python -m`` rather than the ``dikw`` entry-point script so we
# work inside CI matrices and process-wrappers that strip ``$PATH``.
_PYTHON_M_DIKW: list[str] = [sys.executable, "-m", "dikw_core.cli"]


@dataclass(frozen=True)
class ServeAndRunOptions:
    base: Path
    host: str
    port: int  # 0 means "pick free"
    token: str | None
    ready_timeout: float
    keep_alive: bool
    log_level: str
    inner_cmd: list[str]


def find_free_port() -> int:
    """Bind to port 0, read back the kernel's assignment, close.

    Tiny TOCTOU window between this call and the server actually
    binding; in practice the kernel doesn't hand out the same port
    twice in microseconds, and the alternative (bind + hand the FD to
    uvicorn) doesn't compose well across platforms.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class ServerExitedEarly(RuntimeError):
    """The spawned server died before /v1/healthz returned 200.

    Surfaces a clearer failure than the readiness timeout for the
    common misconfigurations — bad ``--host``/``--token`` combo,
    base without ``dikw.yml``, port conflict, or any other startup
    error that exits the subprocess before the bind completes.
    """


def wait_until_ready(
    url: str,
    *,
    timeout: float,
    token: str | None = None,
    proc: subprocess.Popen[bytes] | None = None,
) -> None:
    """Poll ``GET <url>/v1/healthz`` until 200 or ``timeout`` lapses.

    HTTP probe rather than stderr-banner parsing because Windows
    stderr buffering can stash the banner long after the bind. When
    ``proc`` is given, an early subprocess exit short-circuits the
    poll with ``ServerExitedEarly`` — saves a 30s timeout when the
    user passed e.g. ``--host 0.0.0.0`` without a token.

    ``token`` must be passed whenever the spawned server requires auth
    (any non-loopback ``--host``, or loopback with ``--token``); otherwise
    the probe sees 401 and the caller waits out the entire timeout.
    """
    deadline = time.monotonic() + timeout
    last_err: BaseException | None = None
    health_url = url.rstrip("/") + "/v1/healthz"
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    while time.monotonic() < deadline:
        if proc is not None and proc.poll() is not None:
            raise ServerExitedEarly(
                f"server subprocess exited with code {proc.returncode} "
                "before /v1/healthz responded"
            )
        try:
            req = urllib.request.Request(health_url, headers=headers)
            with urllib.request.urlopen(req, timeout=_READY_HTTP_TIMEOUT) as resp:
                if 200 <= resp.status < 300:
                    return
        except (urllib.error.URLError, ConnectionError, OSError) as e:
            last_err = e
        time.sleep(_READY_POLL_INTERVAL)
    raise TimeoutError(
        f"server at {health_url} did not become ready in {timeout}s"
        + (f" (last error: {last_err})" if last_err else "")
    )


def terminate(proc: subprocess.Popen[bytes]) -> None:
    """Stop a running server subprocess, escalating SIGTERM → SIGKILL.

    The grace window matters most on POSIX where uvicorn's signal
    handler closes the listening socket and drains in-flight
    requests; on Windows ``Popen.terminate`` is a hard kill via
    ``TerminateProcess`` regardless.
    """
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=_TERMINATE_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def build_server_command(opts: ServeAndRunOptions) -> list[str]:
    args = [
        *_PYTHON_M_DIKW,
        "serve",
        "--base",
        str(opts.base),
        "--host",
        opts.host,
        "--port",
        str(opts.port),
        "--log-level",
        opts.log_level,
    ]
    if opts.token is not None:
        args.extend(["--token", opts.token])
    return args


def build_inner_command(inner_cmd: list[str]) -> list[str]:
    # Inner argv targets the spawned server, so it's always a ``dikw
    # client …`` invocation — auto-prefix bare verbs, pass through if
    # already explicit.
    if inner_cmd and inner_cmd[0] != "client":
        inner_cmd = ["client", *inner_cmd]
    return [*_PYTHON_M_DIKW, *inner_cmd]


# Wildcard bind addresses (server listens on all interfaces) are NOT
# routable from a client. The server still binds to ``opts.host``; we
# remap only the URL the client uses to reach it back through loopback.
_WILDCARD_HOSTS = {"0.0.0.0", "::", "*", ""}


def _client_host(server_host: str) -> str:
    """Loopback address the in-process client should use to reach the
    server we just spawned, formatted for use inside an HTTP URL.

    ``::`` (IPv6 wildcard) routes to ``[::1]``; ``0.0.0.0`` / ``*``
    route to ``127.0.0.1``. IPv6 literals get bracketed so the colons
    don't collide with the URL's port separator (``http://[::1]:8765``,
    not ``http://::1:8765`` which is an invalid URI).
    """
    if server_host == "::":
        return "[::1]"
    if server_host in _WILDCARD_HOSTS:
        return "127.0.0.1"
    # Bracket bare IPv6 literals (``::1``, ``2001:db8::1``); leave
    # already-bracketed forms and IPv4 / hostnames untouched.
    if ":" in server_host and not server_host.startswith("["):
        return f"[{server_host}]"
    return server_host


def build_inner_env(
    host: str, port: int, token: str | None, *, keep_alive: bool = False
) -> dict[str, str]:
    env = dict(os.environ)
    env[ENV_SERVER_URL] = f"http://{_client_host(host)}:{port}"
    if token is not None:
        env[ENV_SERVER_TOKEN] = token
    if not keep_alive:
        # Signal to async-default op commands ("dikw client ingest" etc.) that
        # the temporary server dies the moment they exit, so they must
        # auto-wait instead of fire-and-forget. ``--keep-alive`` mode
        # leaves the server up and the agent contract holds — the user
        # can follow the task themselves.
        env[ENV_SERVE_AND_RUN_AUTO_WAIT] = "1"
    return env


# When ``dikw client serve-and-run`` spawns an inner CLI without ``--keep-alive``,
# this env var is set in the inner process so op commands know to
# auto-wait — otherwise async-default would print a handle, exit 0, and
# the temporary server would be torn down with the task still running.
ENV_SERVE_AND_RUN_AUTO_WAIT = "DIKW_SERVE_AND_RUN_AUTO_WAIT"


def run(opts: ServeAndRunOptions) -> int:
    """Run one server + inner-command lifecycle.

    Returns the inner command's exit code, or non-zero if the server
    never became ready / was misconfigured / the user passed no inner
    command.
    """
    if not opts.inner_cmd:
        sys.stderr.write(
            "serve-and-run: missing inner command "
            "(use `-- <cmd> [args...]`)\n"
        )
        return 2

    if opts.port == 0:
        opts = dataclasses.replace(opts, port=find_free_port())
    server_argv = build_server_command(opts)

    server = subprocess.Popen(server_argv)
    inner_rc = 1
    try:
        try:
            wait_until_ready(
                f"http://{_client_host(opts.host)}:{opts.port}",
                timeout=opts.ready_timeout,
                token=_effective_token(opts.token),
                proc=server,
            )
        except (TimeoutError, ServerExitedEarly) as e:
            sys.stderr.write(f"serve-and-run: {e}\n")
            return 1

        env = build_inner_env(
            opts.host, opts.port, opts.token, keep_alive=opts.keep_alive
        )
        inner_argv = build_inner_command(opts.inner_cmd)
        inner_rc = subprocess.run(inner_argv, env=env, check=False).returncode
    finally:
        if opts.keep_alive:
            sys.stderr.write(
                f"serve-and-run: server still running at "
                f"http://{opts.host}:{opts.port} (PID={server.pid})\n"
            )
        else:
            terminate(server)
    return inner_rc


# Public surface used by ``cli_app``; helpers are exercised via the
# module path in tests but aren't part of the contract.
__all__ = ["ServeAndRunOptions", "ServerExitedEarly", "run"]
