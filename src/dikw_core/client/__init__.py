"""Remote CLI for dikw-core.

Thin Typer + httpx + rich client that talks to a ``dikw serve`` instance
over HTTP/NDJSON. The package is intentionally engine-agnostic: nothing
in ``client/*`` imports from ``dikw_core.{api,storage,providers,server}``
— only ``rich``, ``typer``, ``httpx``, and stdlib. Adding a new
operation means adding one Typer command and matching the server's wire
contract; no engine changes required.

Submodules:
  * :mod:`config` — resolve server URL + token (CLI flags > env > toml > default)
  * :mod:`transport` — httpx wrapper, JSON + NDJSON helpers, error mapping
  * :mod:`importer` — pack a local sources/+assets/ tree as tar.gz + manifest
  * :mod:`progress` — NDJSON event → rich Progress / final-result renderers
  * :mod:`cli_app` — Typer app exposed as ``dikw client *``
"""

from .cli_app import app

__all__ = ["app"]
