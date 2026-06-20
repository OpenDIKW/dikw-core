#!/usr/bin/env python3
"""Real-environment end-to-end verification harness for dikw-core.

Two modes, one shared verb sequence:

* ``--mode local``  — scaffold a throwaway base in a temp dir, spawn a
  long-lived ``dikw serve`` (SQLite), run every ``dikw client`` verb
  against it, then destroy the base.
* ``--mode docker`` — build the dikw-core image **from the local working
  tree** (``uv build`` → install the wheel), bring up server + pgvector
  Postgres via a generated compose project on a free host port, run the
  SAME sequence, then ``docker compose down -v`` everything.

Karpathy's rule applied to verification: routing/scoping is deterministic
(which verbs exist, which run in which mode) and lives here as plain
dispatch; the probabilistic legs (synth quality) are judged elsewhere.

Provider posture is tiered + skip-loud: structural legs run with no keys;
real legs (``check``/embed/``synth``/vector-``retrieve``/``eval``) run when the
key env vars named by the active profile's ``provider.{llm,embedding}_api_key_env``
are present (from a gitignored ``.env``) — for the default template that's
``MINIMAX_API_KEY`` + ``GITEE_API_KEY`` — else they are SKIPPED loudly (a skip
is never a pass).

Default provider profile is MiniMax (M3) via ``anthropic_compat`` + Qwen3-
Embedding-0.6B via Gitee (the committed ``tests/fixtures/live-minimax-gitee
.dikw.yml`` template), overridable per-vendor via ``DIKW_E2E_*`` env vars.

Run:  ``uv run python tools/e2e_verify.py --mode local --corpus all``
"""

from __future__ import annotations

import argparse
import json
import os
import re
import secrets
import shutil
import signal
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
# ``python -m`` rather than the ``dikw`` entry-point so we survive PATH-
# stripping wrappers (same rationale as serve_and_run._PYTHON_M_DIKW).
_PYTHON_M_DIKW = [sys.executable, "-m", "dikw_core.cli"]

_TEMPLATE = _REPO_ROOT / "tests" / "fixtures" / "live-minimax-gitee.dikw.yml"
# Corpus name -> source dir copied verbatim into <base>/sources/<name>/.
# "assets" is the only one carrying image files (wiki-mini-mm/corpus/images).
_CORPORA: dict[str, Path] = {
    "notes": _REPO_ROOT / "tests" / "fixtures" / "notes",
    "mvp": _REPO_ROOT / "evals" / "datasets" / "mvp" / "corpus",
    "assets": _REPO_ROOT / "evals" / "datasets" / "wiki-mini-mm" / "corpus",
}

# The env vars that gate the tier-2 (real-provider) legs are NOT hardcoded:
# they are the vendor-canonical names declared by the active provider profile
# (``provider.llm_api_key_env`` + ``provider.embedding_api_key_env``), resolved
# at runtime via ``_required_key_envs``. The default template wires MiniMax
# (``MINIMAX_API_KEY``) + Gitee (``GITEE_API_KEY``).

_OBS_COMPOSE = _REPO_ROOT / "docs" / "observability" / "docker-compose.yml"
# Custom label stamped on every harness container/volume so ``--prune`` can
# sweep crashed-run leftovers regardless of project name; also the project-name
# prefix (``dikw-e2e-<port>``) by which build images are swept.
_LABEL = "dikw-e2e"


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
class Status(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    SKIP = "SKIPPED"


@dataclass
class LegResult:
    name: str
    status: Status
    detail: str = ""
    # The canonical ``dikw client`` leaf verb this leg exercised (e.g.
    # "pages list"), used by the coverage assertion. None for non-verb legs
    # (setup, cli-coverage itself).
    verb: str | None = None
    tier: int | None = None


def render_table(legs: list[LegResult]) -> str:
    name_w = max((len(leg.name) for leg in legs), default=4)
    name_w = max(name_w, len("leg"))
    lines = [f"{'leg':<{name_w}}  {'status':<9}  {'tier':<4}  detail",
             f"{'-' * name_w}  {'-' * 9}  {'-' * 4}  {'-' * 40}"]
    for leg in legs:
        tier = f"t{leg.tier}" if leg.tier else ""
        lines.append(f"{leg.name:<{name_w}}  {leg.status.value:<9}  {tier:<4}  {leg.detail}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Secrets & env
# --------------------------------------------------------------------------- #
def load_dotenv(path: Path, env: dict[str, str]) -> None:
    """Parse ``KEY=VALUE`` lines into ``env``; existing keys win (env > file).

    Deliberately tiny — no third-party dep — because the orchestrator runs as
    a plain ``uv run python tools/e2e_verify.py``, outside pytest-dotenv. Lines
    that are blank, comments, or lack ``=`` are ignored. Surrounding quotes on
    the value are stripped.
    """
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        if not key or key in env:  # env wins over the file
            continue
        val = val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in ("'", '"'):
            val = val[1:-1]
        env[key] = val


@dataclass
class KeyPosture:
    has_keys: bool
    missing: list[str] = field(default_factory=list)


def _required_key_envs(profile: Path) -> list[str]:
    """The env var names the real-provider legs need, read from the profile.

    Derived from ``provider.{llm,embedding}_api_key_env`` so a profile that
    targets, say, DeepSeek + Gitee gates on ``DEEPSEEK_API_KEY`` +
    ``GITEE_API_KEY`` without the harness hardcoding any vendor name. The
    ``openai_codex`` LLM uses OAuth (no env key), so its llm key is skipped.
    """
    from dikw_core.config import load_config

    p = load_config(profile).provider
    keys: list[str] = []
    if p.llm != "openai_codex":
        keys.append(p.llm_api_key_env)
    keys.append(p.embedding_api_key_env)
    seen: set[str] = set()
    return [k for k in keys if not (k in seen or seen.add(k))]


def resolve_posture(env: dict[str, str], required: list[str]) -> KeyPosture:
    missing = [k for k in required if not env.get(k)]
    return KeyPosture(has_keys=not missing, missing=missing)


def make_redactor(values: list[str | None]) -> Redactor:
    return Redactor([v for v in values if v])


class Redactor:
    """Masks known secret substrings (API keys, server token) in any text
    surfaced to the user — log dumps, command output, leg details."""

    def __init__(self, secrets_: list[str]) -> None:
        # Longest first so a key that contains a shorter one still fully masks.
        self._secrets = sorted(set(secrets_), key=len, reverse=True)

    def __call__(self, text: str) -> str:
        for s in self._secrets:
            if s:
                text = text.replace(s, "***")
        return text


# --------------------------------------------------------------------------- #
# CLI coverage manifest (deterministic scoping)
# --------------------------------------------------------------------------- #
# Verbs intentionally not driven by the sequence, each with a justification.
EXPLICIT_SKIP: dict[str, str] = {
    "serve-and-run": (
        "lifecycle primitive; this harness is its long-lived superset. "
        "Covered by tests/client/test_serve_and_run.py (-m slow)."
    ),
}


@dataclass
class CoverageManifest:
    live_leaves: set[str]
    skip: dict[str, str]


def client_leaf_verbs() -> set[str]:
    """Every executable ``dikw client`` leaf verb, derived from the LIVE Typer
    tree (never hardcoded) — same introspection tests/test_cli_surface.py uses.

    A group that is itself invokable (``invoke_without_command=True`` with a
    callback, e.g. ``lint``) is recorded as a leaf in addition to its
    subcommands, because the bare ``lint`` scan is a distinct behavior.
    """
    import click
    from typer.main import get_command

    from dikw_core.cli import app

    root = get_command(app)
    assert isinstance(root, click.Group)
    client = root.commands["client"]
    assert isinstance(client, click.Group)

    leaves: set[str] = set()

    def walk(grp: click.Group, prefix: str) -> None:
        for name in grp.commands:
            sub = grp.commands[name]
            path = f"{prefix} {name}".strip()
            if isinstance(sub, click.Group):
                if sub.callback is not None and getattr(sub, "invoke_without_command", False):
                    leaves.add(path)
                walk(sub, path)
            else:
                leaves.add(path)

    walk(client, "")
    return leaves


def build_coverage_manifest() -> CoverageManifest:
    return CoverageManifest(live_leaves=client_leaf_verbs(), skip=dict(EXPLICIT_SKIP))


def assert_full_cli_coverage(manifest: CoverageManifest, legs: list[LegResult]) -> LegResult:
    """Structural ⊇ check: every required verb must have been *attempted*.

    Coverage is the structural axis (was the verb exercised, PASS or loud
    SKIP); pass/fail is the behavioral axis (reported per-leg). A new verb
    added without a sequence step lands in ``missing`` → FAIL. A skip-list
    entry that no longer exists in the tree lands in ``stale_skips`` → FAIL.
    """
    # "covered" = attempted (any terminal status). Coverage is the structural
    # axis — *was the verb exercised* — kept orthogonal to the behavioral
    # pass/fail reported per-leg, so a verb's own failure doesn't also
    # masquerade here as a coverage gap.
    executed = {leg.verb for leg in legs if leg.verb}
    required = manifest.live_leaves - set(manifest.skip)
    missing = required - executed
    stale_skips = set(manifest.skip) - manifest.live_leaves
    ok = not missing and not stale_skips
    detail = f"required={len(required)} covered={len(required & executed)}"
    if missing:
        detail += f" MISSING={sorted(missing)}"
    if stale_skips:
        detail += f" STALE_SKIPS={sorted(stale_skips)}"
    return LegResult("cli-coverage", Status.PASS if ok else Status.FAIL, detail)


# --------------------------------------------------------------------------- #
# Provider dikw.yml builder
# --------------------------------------------------------------------------- #
def build_provider_yaml(*, mode: str, observe: bool, env: dict[str, str],
                        template: Path = _TEMPLATE) -> str:
    """Render the base's ``dikw.yml`` from the committed live template, applying
    per-vendor ``DIKW_E2E_*`` overrides and patching storage/telemetry per mode.

    Secrets are NEVER written here — they reach the providers via the env vars
    named by ``provider.{llm,embedding}_api_key_env`` (default template:
    MINIMAX_API_KEY / GITEE_API_KEY), exactly as the committed template documents.
    """
    from dikw_core.config import (
        PostgresStorageConfig,
        SQLiteStorageConfig,
        TelemetryConfig,
        dump_config_yaml,
        load_config,
    )

    cfg = load_config(template)
    p = cfg.provider
    p.llm = env.get("DIKW_E2E_LLM", p.llm)  # type: ignore[assignment]
    p.llm_model = env.get("DIKW_E2E_LLM_MODEL", p.llm_model)
    p.llm_base_url = env.get("DIKW_E2E_LLM_BASE_URL", p.llm_base_url)
    p.embedding = env.get("DIKW_E2E_EMBEDDING", p.embedding)  # type: ignore[assignment]
    p.embedding_model = env.get("DIKW_E2E_EMBEDDING_MODEL", p.embedding_model)
    p.embedding_base_url = env.get("DIKW_E2E_EMBEDDING_BASE_URL", p.embedding_base_url)
    if "DIKW_E2E_EMBEDDING_DIM" in env:
        p.embedding_dim = int(env["DIKW_E2E_EMBEDDING_DIM"])
    if "DIKW_E2E_EMBEDDING_BATCH" in env:
        p.embedding_batch_size = int(env["DIKW_E2E_EMBEDDING_BATCH"])

    if mode == "docker":
        # Wiki index on Postgres (the task store is wired separately via
        # DIKW_SERVER_TASKS_DSN in the compose env). ``postgres`` is the
        # compose service hostname on the internal network.
        cfg.storage = PostgresStorageConfig(dsn="postgresql://dikw:dikw@postgres:5432/dikw")
    else:
        cfg.storage = SQLiteStorageConfig()

    if observe:
        endpoint = "http://otel-collector:4318" if mode == "docker" else "http://localhost:4318"
        cfg.telemetry = TelemetryConfig(
            enabled=True, endpoint=endpoint, service_name="dikw-core-e2e", sample_ratio=1.0,
        )

    return dump_config_yaml(cfg)


def seed_corpus(sources_dir: Path, which: str) -> list[str]:
    """Copy the requested corpus/corpora into ``<base>/sources/<name>/``."""
    names = list(_CORPORA) if which == "all" else [which]
    seeded: list[str] = []
    for name in names:
        src = _CORPORA[name]
        if not src.is_dir():
            continue
        shutil.copytree(src, sources_dir / name, dirs_exist_ok=True)
        seeded.append(name)
    return seeded


# --------------------------------------------------------------------------- #
# Subprocess runner
# --------------------------------------------------------------------------- #
@dataclass
class CmdResult:
    returncode: int
    stdout: str
    stderr: str

    def json(self) -> Any:
        return json.loads(self.stdout)


def run_client(verb_args: list[str], env: dict[str, str], *, timeout: float = 300.0) -> CmdResult:
    argv = [*_PYTHON_M_DIKW, "client", *verb_args]
    proc = subprocess.run(
        argv, env=env, capture_output=True, text=True, timeout=timeout, check=False,
    )
    return CmdResult(proc.returncode, proc.stdout, proc.stderr)


# --------------------------------------------------------------------------- #
# The shared verb sequence (mode-agnostic)
# --------------------------------------------------------------------------- #
@dataclass
class SeqContext:
    env: dict[str, str]
    posture: KeyPosture
    redact: Redactor
    on_failure: Any  # callable(op_hint: str) -> str | None ; returns trace hint
    legs: list[LegResult] = field(default_factory=list)
    state: dict[str, Any] = field(default_factory=dict)

    def record(self, name: str, status: Status, detail: str = "", *,
               verb: str | None = None, tier: int | None = None) -> None:
        self.legs.append(LegResult(name, status, self.redact(detail), verb=verb, tier=tier))


def _all_page_paths(res: CmdResult | None, prefix: str) -> list[str]:
    if res is None:
        return []
    try:
        data = res.json()
    except json.JSONDecodeError:
        return []
    pages = data.get("pages", data) if isinstance(data, dict) else data
    if not isinstance(pages, list):
        return []
    return [p["path"] for p in pages
            if isinstance(p, dict) and isinstance(p.get("path"), str)
            and p["path"].startswith(prefix)]


def _discover_asset_id(env: dict[str, str], source_paths: list[str]) -> str | None:
    """The real asset id (sha256 of the bytes) is exposed in a page record's
    ``assets[].asset_id`` — read source pages until one references an asset."""
    for path in source_paths:
        r = run_client(["pages", "get", path], env, timeout=60.0)
        if r.returncode != 0:
            continue
        try:
            data = r.json()
        except json.JSONDecodeError:
            continue
        for a in data.get("assets") or []:
            aid = a.get("asset_id") if isinstance(a, dict) else None
            if isinstance(aid, str) and len(aid) == 64:
                return aid
    return None


def _step(ctx: SeqContext, name: str, verb: str, tier: int,
          args: list[str], *, expect: tuple[int, ...] = (0,),
          timeout: float = 300.0) -> CmdResult | None:
    """Run one ``dikw client`` invocation as a leg. Tier-2 steps SKIP loudly
    (still counted as covered) when keys are absent. The leg fails only on an
    exit code outside the documented contract; on failure the observability
    hook is asked for a trace hint."""
    if tier == 2 and not ctx.posture.has_keys:
        ctx.record(name, Status.SKIP, f"tier-2: keys absent ({'+'.join(ctx.posture.missing)})",
                   verb=verb, tier=tier)
        return None
    try:
        res = run_client(verb_args=verb.split() + args, env=ctx.env, timeout=timeout)
    except subprocess.TimeoutExpired:
        ctx.record(name, Status.FAIL, f"timed out after {timeout}s", verb=verb, tier=tier)
        return None
    if res.returncode in expect:
        first = (res.stdout.strip().splitlines() or [""])[0][:80]
        ctx.record(name, Status.PASS, f"exit={res.returncode} {first}", verb=verb, tier=tier)
        return res
    hint = ctx.on_failure(name) or ""
    tail = self_tail(res)
    ctx.record(name, Status.FAIL,
               f"exit={res.returncode} (want {expect}) {tail} {hint}".strip(),
               verb=verb, tier=tier)
    return res


def self_tail(res: CmdResult) -> str:
    err = (res.stderr or res.stdout).strip().splitlines()
    return (err[-1][:160] if err else "")


def _submit_async(ctx: SeqContext, verb: str, args: list[str]) -> str | None:
    """Fire an async-default op WITHOUT --wait and return its task_id, or None.
    The harness never sets DIKW_SERVE_AND_RUN_AUTO_WAIT, so async-default verbs
    genuinely return a handle and exit 0 — exactly what the cancel/timeout
    paths need."""
    res = run_client(verb.split() + args, ctx.env, timeout=60.0)
    if res.returncode != 0:
        return None
    try:
        tid = res.json().get("task_id")
    except (json.JSONDecodeError, AttributeError):
        # Fallback for a dirty/non-JSON stdout. Task ids are full UUIDs
        # (str(uuid.uuid4()) — hyphenated), so the char class must include `-`.
        m = re.search(r'"task_id"\s*:\s*"([0-9a-fA-F-]+)"', res.stdout)
        tid = m.group(1) if m else None
    # Guard the missing/null case: str(None) would be the truthy literal "None".
    return str(tid) if tid else None


def run_sequence(ctx: SeqContext) -> None:
    """Empty base -> ingest -> synth -> query -> lint -> delete, exercising
    every ``dikw client`` verb. Each step appends a LegResult tagged with the
    verb it covers and its tier."""
    env = ctx.env

    # --- 1-3 read-only identity / counts (tier-1) ---
    _step(ctx, "info", "info", 1, [])
    _step(ctx, "status", "status", 1, ["--format", "table"])
    _step(ctx, "health", "health", 1, ["--format", "json"])

    # --- 4 provider probe (tier-2) ---
    _step(ctx, "check (providers)", "check", 2, [], expect=(0,))

    # --- 5 import a standalone file (sync verb, tier-1) ---
    # Corpora are filesystem-seeded into sources/ at setup; ``import`` is
    # exercised on a fresh standalone md so it can't collide with the seed.
    have_assets = "assets" in (ctx.state.get("seeded") or [])
    import_md = ctx.state["tmp"] / "imported.md"
    import_md.write_text(
        "# E2E Imported Note\n\nA standalone markdown file committed via "
        "`dikw client import`.\n", encoding="utf-8")
    _step(ctx, "import", "import", 1, [str(import_md)], expect=(0,))

    # --- 7a hermetic ingest (tier-1): chunk + FTS + materialize assets ---
    _step(ctx, "ingest (--no-embed)", "ingest", 1, ["--no-embed", "--wait"], timeout=600.0)
    # --- 7b real ingest (tier-2): embeddings. No --strict: an image asset with
    # no multimodal embedder configured is a supported config (text embeds,
    # asset-vectors skipped), and --strict would wrongly abort the whole run. ---
    _step(ctx, "ingest (embed)", "ingest", 2, ["--wait"], timeout=900.0)
    # --- 8 idempotent re-ingest (tier-1) ---
    _step(ctx, "ingest (idempotent)", "ingest", 1, ["--no-embed", "--wait"], timeout=600.0)

    # --- 9-12 structural reads (tier-1) ---
    res = _step(ctx, "pages list", "pages list", 1, ["--layer", "source", "--format", "json"])
    source_paths = _all_page_paths(res, "sources/")
    knowledge_path = None
    if source_paths:
        _step(ctx, "pages get", "pages get", 1, [source_paths[0]])
        _step(ctx, "pages provenance (src)", "pages provenance", 1,
              [source_paths[0], "--direction", "in"])
    else:
        ctx.record("pages get", Status.SKIP, "no source page discovered", verb="pages get", tier=1)
        ctx.record("pages provenance (src)", Status.SKIP, "no source page discovered",
                   verb="pages provenance", tier=1)
    _step(ctx, "graph get", "graph get", 1, [])

    # --- assets round-trip (tier-1: assets materialize during ingest) ---
    asset_id = _discover_asset_id(env, source_paths) if have_assets else None
    if asset_id:
        out = ctx.state["tmp"] / "asset.bin"
        r = _step(ctx, "assets get", "assets get", 1,
                  [asset_id, "--output", str(out)], expect=(0,))
        if r is not None and r.returncode == 0 and not out.is_file():
            ctx.legs[-1] = LegResult("assets get", Status.FAIL,
                                     "exit=0 but no output file", verb="assets get", tier=1)
    elif have_assets:
        ctx.record("assets get", Status.SKIP, "no asset referenced by any source page",
                   verb="assets get", tier=1)
    else:
        ctx.record("assets get", Status.SKIP,
                   "no asset corpus seeded (need --corpus assets/all)",
                   verb="assets get", tier=1)

    # --- 13 synth K-layer + verify + judge in one task (tier-2). --all forces a
    # full re-synth so the task always has work. expect (0,1): with --verify the
    # CLI exits 1 when the post-synth self-check finds content issues (ungrounded
    # claims / unresolved wikilinks) — a content-dependent report outcome across
    # LLM runs, NOT a synth failure (cf. lint's 0/1, cli_app.py synth_cmd). A
    # genuine synth task failure is caught below: keys present + zero knowledge
    # pages makes the K-dependent legs FAIL. ---
    _step(ctx, "synth (--all --verify --judge)", "synth", 2,
          ["--all", "--verify", "--judge", "--wait"], expect=(0, 1), timeout=1800.0)

    # --- 14-16 K-layer-dependent reads (tier-2) ---
    if ctx.posture.has_keys:
        kl = run_client(["pages", "list", "--layer", "knowledge", "--format", "json"], env,
                        timeout=60.0)
        knowledge_path = _first_page_path(kl, "knowledge/") if kl.returncode == 0 else None
    if knowledge_path:
        _step(ctx, "pages provenance (k)", "pages provenance", 2,
              [knowledge_path, "--direction", "out"])
        _step(ctx, "pages links", "pages links", 2, [knowledge_path, "--direction", "both"])
    else:
        # No knowledge page found. With keys, synth ran and should have produced
        # pages — their absence is a real failure (the safety net for synth's
        # tolerated 0/1 exit). Without keys, synth was skipped, so these skip too.
        st = Status.FAIL if ctx.posture.has_keys else Status.SKIP
        why = ("synth produced no knowledge pages" if ctx.posture.has_keys
               else "depends on synth (tier-2)")
        for nm, vb in (("pages provenance (k)", "pages provenance"), ("pages links", "pages links")):
            ctx.record(nm, st, why, verb=vb, tier=2)
    _step(ctx, "retrieve", "retrieve", 2,
          ["Karpathy software 2.0", "--limit", "5", "--format", "table"])

    # --- 17 wisdom write (tier-1 via --no-embed) ---
    _step(ctx, "wisdom write", "wisdom write", 1,
          ["--slug", "e2e-note", "--title", "E2E Harness Note",
           "--author", "e2e", "--body", "A hand-written W-layer page for the e2e harness.",
           "--no-embed"], expect=(0,))

    # --- 18-21 lint cycle (tier-1 structural) ---
    _step(ctx, "lint", "lint", 1, ["--format", "table"], expect=(0, 1))
    pid = _submit_async(ctx, "lint propose", [])
    if pid:
        run_client(["tasks", "wait", pid, "--plain"], env, timeout=300.0)
        ctx.record("lint propose", Status.PASS, f"task={pid}", verb="lint propose", tier=1)
    else:
        ctx.record("lint propose", Status.FAIL, "no task handle returned",
                   verb="lint propose", tier=1)
    _step(ctx, "lint proposals", "lint proposals", 1, ["--format", "table"], expect=(0,))
    if pid:
        _step(ctx, "lint apply", "lint apply", 1, [pid, "--wait"], expect=(0, 1))
    else:
        ctx.record("lint apply", Status.SKIP, "no proposal task id", verb="lint apply", tier=1)

    # --- 22 eval packaged dataset (tier-2). --cache off: the harness server runs
    # in a throwaway /base with no persistent snapshot cache root to resolve. ---
    _step(ctx, "eval", "eval", 2,
          ["--dataset", "mvp", "--retrieval", "all", "--cache", "off", "--wait"],
          expect=(0, 2), timeout=1800.0)

    # --- 23-25 tasks reads (tier-1) ---
    res = _step(ctx, "tasks list", "tasks list", 1, ["--format", "json"])
    a_task = None
    if res is not None:
        a_task = _first_task_id(res)
    if a_task:
        _step(ctx, "tasks status", "tasks status", 1, [a_task])
        _step(ctx, "tasks events", "tasks events", 1, [a_task, "--wait", "0"])
        # cancel an already-terminal task: idempotent, covers the verb (tier-1)
        _step(ctx, "tasks cancel", "tasks cancel", 1, [a_task], expect=(0,))
        # wait on a terminal task: returns its status (tier-1)
        _step(ctx, "tasks wait (terminal)", "tasks wait", 1, [a_task], expect=(0, 1))
    else:
        for nm, vb in (("tasks status", "tasks status"), ("tasks events", "tasks events"),
                       ("tasks cancel", "tasks cancel"), ("tasks wait (terminal)", "tasks wait")):
            ctx.record(nm, Status.SKIP, "no task id available", verb=vb, tier=1)

    # --- 26 advanced cancel(130)/timeout(124) behavior (tier-2: needs a slow op) ---
    _cancel_and_timeout_paths(ctx)

    # --- 27 delete (tier-1) ---
    _step(ctx, "delete", "delete", 1, ["wisdom/e2e/e2e-note.md", "--reason", "e2e cleanup"],
          expect=(0,))
    _step(ctx, "graph get (post-delete)", "graph get", 1, [])


def _cancel_and_timeout_paths(ctx: SeqContext) -> None:
    if not ctx.posture.has_keys:
        ctx.record("tasks wait (--timeout 124)", Status.SKIP, "needs a slow op (tier-2)",
                   verb="tasks wait", tier=2)
        return
    tid = _submit_async(ctx, "synth", ["--all"])
    if not tid:
        ctx.record("tasks wait (--timeout 124)", Status.SKIP, "could not submit slow synth",
                   verb="tasks wait", tier=2)
        return
    # client-side timeout budget -> exit 124, task NOT auto-cancelled. Tolerate a
    # raced terminal (0/1) the same way the 130 leg below does: a slow op that
    # finishes (or fails fast) within the 0.5s budget is a race, not a failure.
    r = run_client(["tasks", "wait", tid, "--timeout", "0.5", "--plain"], ctx.env, timeout=60.0)
    if r.returncode == 124:
        ctx.record("tasks wait (--timeout 124)", Status.PASS, "exit=124",
                   verb="tasks wait", tier=2)
    elif r.returncode in (0, 1):
        ctx.record("tasks wait (--timeout 124)", Status.PASS,
                   f"task reached terminal within budget (exit={r.returncode}) — race, not a failure",
                   verb="tasks wait", tier=2)
    else:
        ctx.record("tasks wait (--timeout 124)", Status.FAIL,
                   f"exit={r.returncode} (want 124)", verb="tasks wait", tier=2)
    # cancel the still-running task, then wait -> exit 130 (cancelled)
    run_client(["tasks", "cancel", tid], ctx.env, timeout=60.0)
    r2 = run_client(["tasks", "wait", tid, "--plain"], ctx.env, timeout=120.0)
    if r2.returncode == 130:
        ctx.record("tasks cancel->wait (130)", Status.PASS, "cancelled task -> exit 130",
                   verb="tasks cancel", tier=2)
    elif r2.returncode in (0, 1):
        ctx.record("tasks cancel->wait (130)", Status.PASS,
                   f"task already terminal (exit={r2.returncode}) — cancel raced completion",
                   verb="tasks cancel", tier=2)
    else:
        ctx.record("tasks cancel->wait (130)", Status.FAIL,
                   f"exit={r2.returncode} (want 130/0/1)", verb="tasks cancel", tier=2)


def _first_page_path(res: CmdResult, prefix: str) -> str | None:
    try:
        data = res.json()
    except json.JSONDecodeError:
        return None
    pages = data.get("pages", data) if isinstance(data, dict) else data
    if not isinstance(pages, list):
        return None
    for item in pages:
        path = item.get("path") if isinstance(item, dict) else None
        if isinstance(path, str) and path.startswith(prefix):
            return path
    return None


def _first_task_id(res: CmdResult) -> str | None:
    try:
        data = res.json()
    except json.JSONDecodeError:
        return None
    tasks = data.get("tasks", data) if isinstance(data, dict) else data
    if isinstance(tasks, list):
        for t in tasks:
            tid = t.get("task_id") or t.get("id") if isinstance(t, dict) else None
            if isinstance(tid, str):
                return tid
    return None


# --------------------------------------------------------------------------- #
# Lifecycles
# --------------------------------------------------------------------------- #
class LocalLifecycle:
    """Throwaway SQLite base + one long-lived ``dikw serve`` subprocess."""

    def __init__(self, args: argparse.Namespace, env: dict[str, str]) -> None:
        self.args = args
        self.env = env
        self.base: Path | None = None
        self.server: subprocess.Popen[bytes] | None = None
        self.token = secrets.token_urlsafe(24)
        self.port = 0
        self.log_path: Path | None = None
        self.tmp: Path | None = None

    def setup(self) -> dict[str, str]:
        from dikw_core.client.serve_and_run import find_free_port, wait_until_ready

        self.base = Path(tempfile.mkdtemp(prefix="dikw-e2e-"))
        self.tmp = Path(tempfile.mkdtemp(prefix="dikw-e2e-tmp-"))
        _run_or_raise([*_PYTHON_M_DIKW, "init", str(self.base)], self.env)
        (self.base / "dikw.yml").write_text(
            build_provider_yaml(mode="local", observe=self.args.observe, env=self.env,
                                 template=Path(self.args.provider_profile)),
            encoding="utf-8")
        self.seeded = seed_corpus(self.base / "sources", self.args.corpus)

        self.port = find_free_port()
        self.log_path = self.base / ".dikw" / "server.log"
        server_env = _server_env(self.env)
        argv = [*_PYTHON_M_DIKW, "serve", "--base", str(self.base), "--host", "127.0.0.1",
                "--port", str(self.port), "--token", self.token, "--log-level", "warning"]
        with self.log_path.open("wb") as logf:
            self.server = subprocess.Popen(argv, env=server_env, stdout=logf,
                                           stderr=subprocess.STDOUT)
        wait_until_ready(f"http://127.0.0.1:{self.port}", timeout=60.0, token=self.token,
                         proc=self.server)
        return _client_env(self.env, f"http://127.0.0.1:{self.port}", self.token,
                           observe=self.args.observe)

    def on_failure(self, op_hint: str) -> str | None:
        if self.log_path and self.log_path.is_file():
            return _tail_trace(self.log_path.read_text(encoding="utf-8", errors="replace"),
                               observe=self.args.observe)
        return None

    def teardown(self) -> None:
        from dikw_core.client.serve_and_run import terminate

        if self.server is not None:
            terminate(self.server)
        if not self.args.keep:
            for d in (self.base, self.tmp):
                if d is not None:
                    shutil.rmtree(d, ignore_errors=True)
        elif self.base is not None:
            sys.stderr.write(f"[--keep] base left at {self.base}\n")


class DockerLifecycle:
    """server + pgvector Postgres in a generated compose project, image built
    from the LOCAL working tree (not the released PyPI Dockerfile)."""

    def __init__(self, args: argparse.Namespace, env: dict[str, str]) -> None:
        self.args = args
        self.env = env
        self.token = secrets.token_urlsafe(24)
        self.ctx_dir: Path | None = None
        self.base: Path | None = None
        self.tmp: Path | None = None
        self.host_port = 0
        self.project = ""
        self.compose_file: Path | None = None

    def setup(self) -> dict[str, str]:
        from dikw_core.client.serve_and_run import find_free_port, wait_until_ready

        _require_docker()
        self.ctx_dir = Path(tempfile.mkdtemp(prefix="dikw-e2e-ctx-"))
        self.base = self.ctx_dir / "base"
        self.tmp = Path(tempfile.mkdtemp(prefix="dikw-e2e-tmp-"))
        wheel = _build_local_wheel(self.ctx_dir, self.env)
        shutil.copy2(wheel, self.ctx_dir / wheel.name)

        _run_or_raise([*_PYTHON_M_DIKW, "init", str(self.base)], self.env)
        (self.base / "dikw.yml").write_text(
            build_provider_yaml(mode="docker", observe=self.args.observe, env=self.env,
                                 template=Path(self.args.provider_profile)),
            encoding="utf-8")
        self.seeded = seed_corpus(self.base / "sources", self.args.corpus)

        self.host_port = find_free_port()
        self.project = f"dikw-e2e-{self.host_port}"
        extras = "postgres,otel" if self.args.observe else "postgres"
        (self.ctx_dir / "Dockerfile").write_text(_harness_dockerfile(wheel.name, extras),
                                                  encoding="utf-8")
        # Keep the build context lean: the seeded corpus + uv-build out dir are
        # mounted/copied separately, not baked into the image.
        (self.ctx_dir / ".dockerignore").write_text("base/\nwheel/\n", encoding="utf-8")
        self.compose_file = self.ctx_dir / "docker-compose.e2e.yml"
        self.compose_file.write_text(
            _harness_compose(
                self.host_port,
                self.args.observe,
                _required_key_envs(Path(self.args.provider_profile)),
            ),
            encoding="utf-8",
        )

        self._compose(["down", "-v", "--remove-orphans"], check=False)  # pre-clean a stuck prior run
        # Generous budget: a cold-cache runner pulls base images + pgvector and
        # runs `pip install` of the freshly-built wheel inside `--build`.
        self._compose(["up", "-d", "--build", "--wait"], check=True, timeout=1200.0)
        url = f"http://127.0.0.1:{self.host_port}"
        wait_until_ready(url, timeout=120.0, token=self.token)
        return _client_env(self.env, url, self.token, observe=self.args.observe)

    def _compose_env(self) -> dict[str, str]:
        # Every compose invocation re-interpolates the compose file, which uses
        # ``${DIKW_SERVER_TOKEN:?required}`` — so the token must be present for
        # up AND down AND logs, else teardown/logging fail to interpolate. The
        # UID/GID make the container write the bind mount as the host user
        # (getattr fallback for Windows, where the image's 1000 is fine).
        getuid = getattr(os, "getuid", lambda: 1000)
        getgid = getattr(os, "getgid", lambda: 1000)
        return {**self.env, "DIKW_SERVER_TOKEN": self.token,
                "DIKW_E2E_UID": str(getuid()), "DIKW_E2E_GID": str(getgid())}

    def _compose(self, args: list[str], *, check: bool, timeout: float = 300.0) -> None:
        assert self.compose_file is not None
        cmd = ["docker", "compose", "-f", str(self.compose_file), "-p", self.project, *args]
        subprocess.run(cmd, env=self._compose_env(), check=check, timeout=timeout)

    def on_failure(self, op_hint: str) -> str | None:
        if not self.compose_file:
            return None
        try:
            r = subprocess.run(
                ["docker", "compose", "-f", str(self.compose_file), "-p", self.project,
                 "logs", "--tail", "50", "dikw-core-local"],
                env=self._compose_env(), capture_output=True, text=True, check=False, timeout=30.0)
            return _tail_trace(r.stdout, observe=self.args.observe)
        except (subprocess.SubprocessError, OSError):
            return None

    def teardown(self) -> None:
        if self.args.keep:
            sys.stderr.write(
                f"[--keep] containers left in project {self.project}; "
                f"tear down with: docker compose -p {self.project} down -v --remove-orphans\n")
            return
        if self.compose_file is not None:
            # --rmi local also removes the auto-named ``<project>-dikw-core-local``
            # build image (``down`` alone leaves it behind); the pulled pgvector
            # base carries a custom ``image:`` tag so ``local`` spares it.
            self._compose(["down", "-v", "--rmi", "local", "--remove-orphans"], check=False)
        for d in (self.ctx_dir, self.tmp):
            if d is not None:
                shutil.rmtree(d, ignore_errors=True)


def _build_local_wheel(out_root: Path, env: dict[str, str]) -> Path:
    out = out_root / "wheel"
    out.mkdir(parents=True, exist_ok=True)
    subprocess.run(["uv", "build", "--wheel", "--out-dir", str(out)],
                   cwd=_REPO_ROOT, env=env, check=True, timeout=600.0)
    wheels = sorted(out.glob("dikw_core-*.whl"))
    if len(wheels) != 1:
        raise RuntimeError(f"expected exactly one wheel in {out}, found {[w.name for w in wheels]}")
    return wheels[0]


def _harness_dockerfile(wheel_name: str, extras: str) -> str:
    return (
        "FROM python:3.12-slim\n"
        "RUN useradd --create-home --uid 1000 dikw\n"
        f"COPY {wheel_name} /tmp/{wheel_name}\n"
        f'RUN pip install --no-cache-dir "/tmp/{wheel_name}[{extras}]"\n'
        "USER dikw\n"
        "WORKDIR /base\n"
        "EXPOSE 8765\n"
        'ENTRYPOINT ["dikw"]\n'
        'CMD ["serve", "--base", "/base", "--host", "0.0.0.0", "--port", "8765"]\n'
    )


def _harness_compose(host_port: int, observe: bool, provider_key_envs: list[str]) -> str:
    # Pass the profile's vendor-canonical key vars through to the container
    # (empty-default so a structural-only run without keys still composes).
    provider_keys = "".join(f"      {k}: ${{{k}:-}}\n" for k in provider_key_envs)
    healthcheck = (
        "import urllib.request as r,os; "
        "r.urlopen(r.Request('http://localhost:8765/v1/healthz', "
        "headers={'Authorization': 'Bearer '+os.environ['DIKW_SERVER_TOKEN']})).read()"
    )
    obs_endpoint = "      OTEL_EXPORTER_OTLP_ENDPOINT: http://otel-collector:4318\n" if observe else ""
    # --observe: join the separately-launched observability stack's network
    # (project ``dikw-e2e-obs`` → default network ``dikw-e2e-obs_default``,
    # brought up before this compose in main()) so the server resolves
    # ``otel-collector`` and exports spans to it. Listing networks opts the
    # service out of the implicit default, so ``default`` must be named too
    # (postgres stays reachable there).
    svc_networks = "    networks: [default, obs]\n" if observe else ""
    obs_network = (
        "\nnetworks:\n"
        "  obs:\n"
        "    external: true\n"
        "    name: dikw-e2e-obs_default\n"
    ) if observe else ""
    return f"""services:
  postgres:
    image: pgvector/pgvector:0.8.2-pg18
    labels: {{{_LABEL}: "1"}}
    environment:
      POSTGRES_USER: dikw
      POSTGRES_PASSWORD: dikw
      POSTGRES_DB: dikw
    volumes:
      - pgdata:/var/lib/postgresql
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U dikw -d dikw"]
      interval: 5s
      timeout: 5s
      retries: 10

  dikw-core-local:
    build:
      context: .
      dockerfile: Dockerfile
    labels: {{{_LABEL}: "1"}}
    # Run as the invoking host user so the ``./base`` bind mount (created on the
    # host before ``up``) is writable: on native Linux a fixed image UID 1000 ≠
    # the host UID can't write to it (macOS Docker Desktop maps ownership, so it
    # masks this). HOME=/tmp keeps a passwd-less UID from defaulting to ``/``.
    user: "${{DIKW_E2E_UID:-1000}}:${{DIKW_E2E_GID:-1000}}"
    depends_on:
      postgres:
        condition: service_healthy
    environment:
      DIKW_SERVER_TOKEN: ${{DIKW_SERVER_TOKEN:?required}}
      DIKW_SERVER_TASKS_DSN: "postgresql://dikw:dikw@postgres:5432/dikw"
      DIKW_TASK_REAP_ON_START: "1"
{provider_keys}      DIKW_LOG_FORMAT: json
      HOME: /tmp
{obs_endpoint}    ports:
      - "{host_port}:8765"
{svc_networks}    volumes:
      - ./base:/base
    healthcheck:
      test: ["CMD", "python", "-c", "{healthcheck}"]
      interval: 10s
      timeout: 5s
      retries: 5
      start_period: 30s

volumes:
  pgdata:
    labels: {{{_LABEL}: "1"}}
{obs_network}"""


# --------------------------------------------------------------------------- #
# Env / observability helpers
# --------------------------------------------------------------------------- #
def _client_env(base: dict[str, str], url: str, token: str, *, observe: bool) -> dict[str, str]:
    from dikw_core.client.config import ENV_SERVER_TOKEN, ENV_SERVER_URL
    from dikw_core.client.serve_and_run import ENV_SERVE_AND_RUN_AUTO_WAIT

    env = dict(base)
    env[ENV_SERVER_URL] = url
    env[ENV_SERVER_TOKEN] = token
    env["DIKW_LOG_FORMAT"] = "json"
    # Wide console so rich's `console.print_json` (used by --format json verbs)
    # never soft-wraps a long path/title mid-string and breaks our json.loads.
    env["COLUMNS"] = "10000"
    # macOS system proxy intercepts loopback otherwise (see memory).
    env["no_proxy"] = env["NO_PROXY"] = "127.0.0.1,localhost"
    # NEVER auto-wait: it would force async-default verbs to block, breaking the
    # cancel/timeout paths that depend on an immediate task handle.
    env.pop(ENV_SERVE_AND_RUN_AUTO_WAIT, None)
    if observe:
        env["OTEL_EXPORTER_OTLP_ENDPOINT"] = "http://localhost:4318"
    return env


def _server_env(base: dict[str, str]) -> dict[str, str]:
    # JSON logs so on-failure trace extraction can parse trace_id; no_proxy so a
    # macOS system proxy doesn't intercept the loopback healthz/requests. Server
    # telemetry is driven by dikw.yml (build_provider_yaml), not env.
    env = dict(base)
    env["DIKW_LOG_FORMAT"] = "json"
    env["no_proxy"] = env["NO_PROXY"] = "127.0.0.1,localhost"
    return env


def _tail_trace(log_text: str, *, observe: bool) -> str:
    """Pull the most recent trace_id from json server logs; on --observe, emit a
    Jaeger deep link."""
    trace_id = None
    for line in reversed(log_text.strip().splitlines()[-200:]):
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        tid = rec.get("trace_id")
        if tid and tid != "0" * 32:
            trace_id = tid
            break
    if not trace_id:
        return ""
    if observe:
        return f"trace_id={trace_id} jaeger=http://localhost:16686/trace/{trace_id}"
    return f"trace_id={trace_id} (run with --observe for the Jaeger link)"


def _run_or_raise(argv: list[str], env: dict[str, str]) -> None:
    subprocess.run(argv, env=env, check=True, timeout=120.0)


def _require_docker() -> None:
    if shutil.which("docker") is None:
        raise RuntimeError("docker CLI not found")
    r = subprocess.run(["docker", "info"], capture_output=True, check=False, timeout=30.0)
    if r.returncode != 0:
        raise RuntimeError("docker daemon not reachable (`docker info` failed)")


# --------------------------------------------------------------------------- #
# Observability stack (--observe)
# --------------------------------------------------------------------------- #
def observability_up(env: dict[str, str]) -> None:
    if not _OBS_COMPOSE.is_file():
        return
    subprocess.run(
        ["docker", "compose", "-f", str(_OBS_COMPOSE), "-p", "dikw-e2e-obs", "up", "-d", "--wait"],
        env=env, check=False, timeout=300.0)


def observability_down(env: dict[str, str]) -> None:
    if not _OBS_COMPOSE.is_file():
        return
    subprocess.run(
        ["docker", "compose", "-f", str(_OBS_COMPOSE), "-p", "dikw-e2e-obs", "down", "-v"],
        env=env, check=False, timeout=120.0)


def prune() -> int:
    """Sweep any leftover harness containers/volumes/images/networks."""
    if shutil.which("docker") is None:
        sys.stderr.write("docker not found\n")
        return 1
    ids = subprocess.run(["docker", "ps", "-aq", "--filter", f"label={_LABEL}=1"],
                         capture_output=True, text=True, check=False).stdout.split()
    if ids:
        subprocess.run(["docker", "rm", "-f", *ids], check=False)
    vols = subprocess.run(["docker", "volume", "ls", "-q", "--filter", f"label={_LABEL}=1"],
                          capture_output=True, text=True, check=False).stdout.split()
    if vols:
        subprocess.run(["docker", "volume", "rm", *vols], check=False)
    # Build images are auto-named ``<project>-…`` and the per-project default
    # network ``<project>_default`` carries no harness label (compose ``labels:``
    # stamp containers/volumes, not images/networks), so sweep both by the
    # ``dikw-e2e-`` project-name prefix instead. Networks must follow the
    # containers (an in-use network refuses removal) — harmless if already gone.
    repos = subprocess.run(["docker", "images", "--format", "{{.Repository}}:{{.Tag}}"],
                           capture_output=True, text=True, check=False).stdout.split()
    imgs = [r for r in repos if r.startswith(f"{_LABEL}-")]
    if imgs:
        subprocess.run(["docker", "rmi", "-f", *imgs], check=False)
    all_nets = subprocess.run(["docker", "network", "ls", "--format", "{{.Name}}"],
                              capture_output=True, text=True, check=False).stdout.split()
    nets = [n for n in all_nets if n.startswith(f"{_LABEL}-")]
    if nets:
        subprocess.run(["docker", "network", "rm", *nets], check=False)
    sys.stderr.write(
        f"pruned {len(ids)} containers, {len(vols)} volumes, "
        f"{len(imgs)} images, {len(nets)} networks\n")
    return 0


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="dikw-core end-to-end verification harness")
    ap.add_argument("--mode", choices=["local", "docker"], help="verification environment")
    ap.add_argument("--corpus", choices=["notes", "mvp", "assets", "all"], default="all")
    ap.add_argument("--observe", action="store_true",
                    help="bring up the otel/jaeger stack + wire telemetry")
    ap.add_argument("--keep", action="store_true", help="skip teardown (debug)")
    ap.add_argument("--prune", action="store_true",
                    help="sweep leftover harness containers/volumes by label and exit")
    ap.add_argument("--provider-profile", default=str(_TEMPLATE),
                    help="dikw.yml-shaped provider profile (default: the committed "
                         "MiniMax + Qwen template); point at your own to swap vendor/model")
    ap.add_argument("--env-file", default=str(_REPO_ROOT / ".env"))
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.prune:
        return prune()
    if not args.mode:
        sys.stderr.write("error: --mode {local,docker} is required (or --prune)\n")
        return 2

    env = dict(os.environ)
    load_dotenv(Path(args.env_file), env)
    required_keys = _required_key_envs(Path(args.provider_profile))
    posture = resolve_posture(env, required_keys)

    manifest = build_coverage_manifest()
    lifecycle = LocalLifecycle(args, env) if args.mode == "local" else DockerLifecycle(args, env)
    redact = make_redactor(
        [env.get(k) for k in required_keys] + [getattr(lifecycle, "token", None)]
    )

    def _sigterm(_signo: int, _frame: Any) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _sigterm)

    legs: list[LegResult] = []
    if args.observe:
        observability_up(env)
    try:
        try:
            client_env = lifecycle.setup()
        except Exception as exc:
            legs.append(LegResult("setup", Status.FAIL, redact(f"{type(exc).__name__}: {exc}")))
            raise
        legs.append(LegResult("setup", Status.PASS,
                              f"mode={args.mode} keys={'present' if posture.has_keys else 'absent'}"))
        ctx = SeqContext(env=client_env, posture=posture, redact=redact,
                         on_failure=lifecycle.on_failure, legs=legs)
        ctx.state["seeded"] = getattr(lifecycle, "seeded", [])
        ctx.state["tmp"] = lifecycle.tmp
        try:
            run_sequence(ctx)
        except Exception as exc:
            ctx.record("sequence", Status.FAIL, f"{type(exc).__name__}: {exc}")
        # …and coverage still runs, so the harness's core assertion never silently vanishes.
        legs.append(assert_full_cli_coverage(manifest, legs))
    except Exception:
        # The only thing that reaches here is the re-raised setup failure, which
        # was already recorded as a FAIL leg above; swallow it so teardown and
        # the summary table still run (run_sequence/coverage record their own).
        pass
    finally:
        lifecycle.teardown()
        if args.observe and not args.keep:
            observability_down(env)

    print()
    print(f"E2E VERIFY — mode={args.mode} corpus={args.corpus} "
          f"keys={'present' if posture.has_keys else 'absent'}")
    print(render_table(legs))
    n_fail = sum(1 for leg in legs if leg.status is Status.FAIL)
    n_skip = sum(1 for leg in legs if leg.status is Status.SKIP)
    verdict = "FAIL" if n_fail else ("PASS (with loud skips — NOT a full verification)"
                                     if n_skip else "PASS")
    print(f"RESULT: {verdict} ({len(legs)} legs, {n_fail} failed, {n_skip} skipped)")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
