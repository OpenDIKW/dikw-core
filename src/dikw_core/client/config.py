"""Client config resolution.

The client picks the server URL + bearer token from four sources, in
descending precedence:

1. Explicit args (``--server`` / ``--token``) — highest, defeats anything
   else for a single command.
2. Environment variables (``DIKW_SERVER_URL``, ``DIKW_SERVER_TOKEN``) —
   for shells/CI; the env always wins over file-based config.
3. ``~/.config/dikw/client.toml`` — per-user persistent default.
4. Built-in defaults (``http://127.0.0.1:8765``, no token).

Anything missing falls through to the next layer rather than erroring,
so the typical "I started ``dikw serve`` on this machine and want to
talk to it" flow needs no configuration at all.

Per-extension converter defaults (``client.converters``) follow the
same layering — ``DIKW_CLIENT_CONVERTER_<EXT>`` env wins over the toml's
``[default.converters]`` entry for that one extension, others stay on
toml. Converter selection on the CLI (``--converter=<name>``) is a
one-shot override applied per call and is not stored here.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_SERVER_URL = "http://127.0.0.1:8765"
ENV_SERVER_URL = "DIKW_SERVER_URL"
ENV_SERVER_TOKEN = "DIKW_SERVER_TOKEN"
ENV_CONVERTER_PREFIX = "DIKW_CLIENT_CONVERTER_"


@dataclass(frozen=True)
class ClientConfig:
    """Resolved client-side runtime settings the CLI passes to transport
    and importer code.

    ``converters`` maps lowercase file extension (``.pdf``) to the
    engine name (``marker``) the converter plugin should advertise.
    Empty when no plugin defaults have been configured — the importer
    then either picks the sole installed plugin or refuses with a
    conflict error.
    """

    server_url: str
    token: str | None
    converters: dict[str, str] = field(default_factory=dict)


def default_config_path() -> Path:
    """``~/.config/dikw/client.toml`` on Linux/macOS;
    ``%APPDATA%\\dikw\\client.toml`` on Windows.

    Falls back to the XDG default on platforms without ``%APPDATA%``."""
    if os.name == "nt":
        appdata = os.environ.get("APPDATA")
        if appdata:
            return Path(appdata) / "dikw" / "client.toml"
    xdg = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg) if xdg else Path.home() / ".config"
    return base / "dikw" / "client.toml"


def _load_file(path: Path) -> dict[str, object]:
    """Parse ``client.toml``; missing file → empty dict (not an error).

    The file shape is intentionally minimal — a flat ``[default]`` table
    with ``server_url`` / ``token`` / ``[default.converters]`` keys. We
    don't grow named profiles here until there's a real need; Phase-5
    plan keeps it single-tenant.
    """
    if not path.is_file():
        return {}
    with path.open("rb") as f:
        data = tomllib.load(f)
    default = data.get("default")
    if not isinstance(default, dict):
        return {}
    return {str(k): v for k, v in default.items()}


def _converters_from_file(file_cfg: dict[str, object]) -> dict[str, str]:
    """Extract the ``[default.converters]`` sub-table, normalising keys
    to lowercase and dropping non-string / empty entries silently — the
    same forgiving stance ``_strip_or_none`` takes elsewhere."""
    raw = file_cfg.get("converters")
    if not isinstance(raw, dict):
        return {}
    out: dict[str, str] = {}
    for k, v in raw.items():
        if not isinstance(k, str) or not isinstance(v, str):
            continue
        v = v.strip()
        if not v:
            continue
        out[k.lower()] = v
    return out


def _converters_from_env(env: dict[str, str] | os._Environ[str]) -> dict[str, str]:
    """Read ``DIKW_CLIENT_CONVERTER_<EXT>=<engine>`` entries into a
    ``{".pdf": "marker"}`` mapping. Empty values fall through (treated
    as "unset")."""
    out: dict[str, str] = {}
    for key, value in env.items():
        if not key.startswith(ENV_CONVERTER_PREFIX):
            continue
        if not isinstance(value, str):
            continue
        v = value.strip()
        if not v:
            continue
        ext_name = key[len(ENV_CONVERTER_PREFIX) :]
        if not ext_name:
            continue
        out["." + ext_name.lower()] = v
    return out


def resolve(
    *,
    server_url: str | None = None,
    token: str | None = None,
    config_path: Path | None = None,
    env: dict[str, str] | None = None,
) -> ClientConfig:
    """Compute the effective ``ClientConfig`` from CLI flags + env + file.

    Tests pass ``env`` and ``config_path`` to keep the lookup hermetic;
    the production CLI calls this with no kwargs, accepting the defaults.
    """
    src_env = os.environ if env is None else env
    file_cfg = _load_file(config_path or default_config_path())

    resolved_url = (
        server_url
        or src_env.get(ENV_SERVER_URL)
        or _strip_or_none(file_cfg.get("server_url"))
        or DEFAULT_SERVER_URL
    )
    resolved_token = (
        token
        or src_env.get(ENV_SERVER_TOKEN)
        or _strip_or_none(file_cfg.get("token"))
    )

    # Converters merge per-extension: toml supplies the baseline, env
    # entries override individual extensions. Same layering shape as the
    # server_url / token chain, but per-key rather than a single value.
    resolved_converters = _converters_from_file(file_cfg)
    resolved_converters.update(_converters_from_env(src_env))

    return ClientConfig(
        server_url=resolved_url.rstrip("/"),
        token=resolved_token or None,
        converters=resolved_converters,
    )


def _strip_or_none(value: object) -> str | None:
    """Coerce a TOML value to ``str | None``; empty strings become None.

    Empty-string overrides on disk are treated as "unset" rather than
    "explicitly empty" — otherwise a user who comments-out ``token = ""``
    can't fall through to env on the next run.
    """
    if not isinstance(value, str):
        return None
    s = value.strip()
    return s or None


__all__ = [
    "DEFAULT_SERVER_URL",
    "ENV_CONVERTER_PREFIX",
    "ENV_SERVER_TOKEN",
    "ENV_SERVER_URL",
    "ClientConfig",
    "default_config_path",
    "resolve",
]
