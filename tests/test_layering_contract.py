"""Layering-invariant contract — enforce the import-direction rules CLAUDE.md
documents under "Layering invariants" / "Named seams".

ruff/mypy/pytest are all blind to these: a forbidden import still type-checks
and runs. This walks every module's AST and asserts the direction rules so a
stray coupling fails in the fast suite instead of silently breaking the goal of
packaging ``client/`` as a standalone wheel (or dragging FastAPI into the engine).

Rules:
  A. ``client/*`` must not import ``dikw_core.{api, api_*, storage, providers,
     server, eval}`` — the client depends only on ``schemas`` (+ the
     dependency-light ``md_inspect``) within the package, so it stays
     wheel-packagable. (``eval`` matters now that the client restates the
     metric-direction convention rather than importing the engine's copy.)
  B. only ``server/*`` and the ``cli.py`` launcher may import fastapi / uvicorn /
     starlette — engine code must not depend on the web framework.
  C. only ``server/*`` and ``cli.py`` may import ``dikw_core.server`` — engine /
     client code must not depend on server plumbing.
"""

from __future__ import annotations

import ast
from pathlib import Path

import dikw_core

_SRC = Path(dikw_core.__file__).resolve().parent  # .../src/dikw_core

_WEB_FRAMEWORK = {"fastapi", "uvicorn", "starlette"}
# Engine subpackages the standalone client must never reach into (first path
# segment under ``dikw_core``). ``api`` / ``api_*`` are handled separately.
_CLIENT_FORBIDDEN_ROOTS = {"storage", "providers", "server", "eval"}


# ---- import extraction --------------------------------------------------


def _module_parts(path: Path) -> list[str]:
    rel = path.relative_to(_SRC.parent)  # dikw_core/<...>.py
    parts = list(rel.with_suffix("").parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return parts


def _package_parts(path: Path) -> list[str]:
    # A package (``__init__.py``) is its own package for relative resolution;
    # a regular module's package is its parent.
    parts = _module_parts(path)
    return parts if path.name == "__init__.py" else parts[:-1]


def _resolve_relative(path: Path, node: ast.ImportFrom) -> str | None:
    if node.level == 0:
        return node.module
    pkg = _package_parts(path)
    prefix = pkg[: len(pkg) - (node.level - 1)]
    if node.module:
        prefix = prefix + node.module.split(".")
    return ".".join(prefix) if prefix else None


def _imported_modules(path: Path) -> list[str]:
    """Absolute dotted module names ``path`` imports (relative imports resolved)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    mods: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            resolved = _resolve_relative(path, node)
            if resolved:
                mods.append(resolved)
    return mods


def _rel(path: Path) -> str:
    return path.relative_to(_SRC).as_posix()


# ---- rule predicates (unit-tested below on synthetic input) -------------


def _dikw_subroot(mod: str) -> str | None:
    """First path segment under ``dikw_core``, or None if not a dikw_core import."""
    parts = mod.split(".")
    if parts[0] != "dikw_core" or len(parts) < 2:
        return None
    return parts[1]


def _is_client_forbidden(mod: str) -> bool:
    root = _dikw_subroot(mod)
    if root is None:
        return False
    return root in _CLIENT_FORBIDDEN_ROOTS or root == "api" or root.startswith("api_")


def _is_web_framework(mod: str) -> bool:
    return mod.split(".")[0] in _WEB_FRAMEWORK


def _is_server_import(mod: str) -> bool:
    return _dikw_subroot(mod) == "server"


# ---- contract tests over the real tree ----------------------------------


def _modules() -> list[Path]:
    return sorted(_SRC.rglob("*.py"))


def test_client_does_not_import_engine() -> None:
    violations = [
        f"{_rel(p)} -> {mod}"
        for p in _modules()
        if _rel(p).startswith("client/")
        for mod in _imported_modules(p)
        if _is_client_forbidden(mod)
    ]
    assert not violations, (
        "client/* must not import "
        "dikw_core.{api,api_*,storage,providers,server,eval} "
        "(standalone-wheel invariant):\n" + "\n".join(violations)
    )


def test_engine_does_not_import_web_framework() -> None:
    violations = [
        f"{_rel(p)} -> {mod}"
        for p in _modules()
        if not _rel(p).startswith("server/") and _rel(p) != "cli.py"
        for mod in _imported_modules(p)
        if _is_web_framework(mod)
    ]
    assert not violations, (
        "only server/* and cli.py may import fastapi/uvicorn/starlette "
        "(engine must not depend on the web framework):\n" + "\n".join(violations)
    )


def test_only_server_and_cli_import_server_package() -> None:
    violations = [
        f"{_rel(p)} -> {mod}"
        for p in _modules()
        if not _rel(p).startswith("server/") and _rel(p) != "cli.py"
        for mod in _imported_modules(p)
        if _is_server_import(mod)
    ]
    assert not violations, (
        "only server/* and cli.py may import dikw_core.server "
        "(engine/client must not depend on server plumbing):\n" + "\n".join(violations)
    )


# ---- test the test: the predicates flag known-bad imports ---------------


def test_rule_predicates_flag_known_violations() -> None:
    # client forbidden set
    assert _is_client_forbidden("dikw_core.storage.base")
    assert _is_client_forbidden("dikw_core.api")
    assert _is_client_forbidden("dikw_core.api_core")
    assert _is_client_forbidden("dikw_core.providers")
    assert _is_client_forbidden("dikw_core.server.app")
    assert _is_client_forbidden("dikw_core.eval.runner")
    # …but the modules the client legitimately shares are allowed
    assert not _is_client_forbidden("dikw_core.schemas")
    assert not _is_client_forbidden("dikw_core.md_inspect")
    assert not _is_client_forbidden("httpx")
    # web framework + server-package predicates
    assert _is_web_framework("fastapi")
    assert _is_web_framework("uvicorn.config")
    assert not _is_web_framework("dikw_core.api")
    assert _is_server_import("dikw_core.server.app")
    assert not _is_server_import("dikw_core.schemas")
