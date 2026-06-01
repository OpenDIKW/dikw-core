"""Shared engine-facade core: storage open/migrate, base scaffolding, and
the embed-version helpers reused across ingest / retrieve / synth / lint /
wisdom.

Rank-1 module: depends only on ``api_types`` + config / providers / schemas
/ storage, never on the ``api`` facade or the cluster modules — so the
clusters can import these primitives without an import cycle. ``api``
re-exports the public names (``init_base`` / ``load_base`` / ``status`` /
``resolve_base_root``) plus the underscore-prefixed helpers the cluster
modules and tests reach for.
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from urllib.parse import urlparse

import yaml

from .api_types import BaseUpgradeRequired
from .config import (
    CONFIG_FILENAME,
    DikwConfig,
    ProviderConfig,
    default_config,
    dump_config_yaml,
    find_config,
    load_config,
)
from .providers import EmbeddingProvider, TransientProviderError
from .schemas import EmbeddingVersion, StorageCounts
from .storage import Storage, build_storage
from .storage.base import NotSupported

# Base-relative files written by ``dikw init`` regardless of taxonomy. The
# per-category ``knowledge/<category>/.gitkeep`` folders are created
# separately in ``init_base`` from the config's declared categories (the
# default tree is entity/concept/note). dikw-core no longer scaffolds a
# generated ``knowledge/index.md`` / ``knowledge/log.md`` — the category
# folder tree is the catalogue and the ``knowledge_log`` table is the history
# (see ADR-0004). ``prompts/`` is the user-owned tree for optional per-base
# prompt overrides (``synth.prompt_path`` / ``lint.fixer_prompts``).
KNOWLEDGE_INIT_FILES: dict[str, str] = {
    "sources/.gitkeep": "",
    "prompts/.gitkeep": "",
    "wisdom/.gitkeep": "",
    ".dikw/.gitkeep": "",
    ".gitignore": ".dikw/\n",
}


# ---- embed-version helpers ----------------------------------------------


def _qualified_provider(protocol: str, base_url: str) -> str:
    """Return ``"<protocol>@<host>"`` for embed_versions.provider.

    The ``embedding`` config field only names the wire protocol
    (``"openai_compat"``); the actual backend is whatever ``base_url``
    points at. We fold the host into the version-identity provider so
    e.g. OpenAI text-embedding-3-small and Gitee's namesake serve under
    distinct version_ids — their vectors live in different spaces and
    must not share a vec table.

    Empty ``base_url`` falls through as bare protocol — the multimodal
    config makes ``base_url`` optional ("use provider default") so we
    avoid synthesising a bogus host placeholder there.
    """
    if not base_url:
        return protocol
    host = urlparse(base_url).hostname or base_url
    return f"{protocol}@{host}"


async def _register_text_version(
    storage: Storage, cfg_provider: ProviderConfig
) -> int:
    """Register-and-activate the text ``embed_versions`` row from cfg.

    Activates the configured identity. Only safe to call from ``ingest``,
    which re-embeds the full corpus and so can flip the active version
    safely. Lint apply / wisdom write must use
    :func:`_resolve_active_text_version_for_inline_embed` instead — they
    only re-embed the pages they touch, so flipping active would strand
    every other vector in the now-inactive table and gut dense retrieval
    until the next full ingest (codex review finding, 0.4.0).
    """
    return await storage.upsert_embed_version(
        EmbeddingVersion(
            # Encode the endpoint host into ``provider`` so two
            # OpenAI-compatible vendors serving the same model name
            # don't collide on a single version_id (their vectors
            # live in different spaces and must not share a table).
            provider=_qualified_provider(
                cfg_provider.embedding, cfg_provider.embedding_base_url
            ),
            model=cfg_provider.embedding_model,
            revision=cfg_provider.embedding_revision,
            dim=cfg_provider.embedding_dim,
            normalize=cfg_provider.embedding_normalize,
            distance=cfg_provider.embedding_distance,
            modality="text",
        )
    )


async def _resolve_active_text_version_for_inline_embed(
    storage: Storage, cfg_provider: ProviderConfig | None = None
) -> tuple[int, str] | None:
    """Return ``(version_id, model)`` of the active text embed version, or None.

    Callers that re-embed only the pages they touch (lint apply, wisdom
    write) must NOT register-and-activate a new version on their own —
    flipping active here would strand every other vector in the
    now-inactive table and gut dense retrieval until ``dikw client
    ingest`` re-embeds the full corpus. Mirrors the pattern used by
    ``synthesize`` (api.py:2484-2497).

    Returns ``None`` when:

    1. No active text version exists yet (fresh base, no ingest run), or
    2. ``cfg_provider`` is supplied and its embedding identity has
       drifted from the active version's identity. The caller built
       its embedder from cfg, so its vectors would land under the
       wrong version table — defer to the next ingest's resume scan
       (which goes through the full register-and-activate path)
       instead of silently mixing vector spaces.

    Drift detection compares the **full** version identity:
    ``(provider, model, revision, dim, normalize, distance)``.
    ``(provider, model)`` defines which vec table; ``revision`` /
    ``dim`` / ``normalize`` / ``distance`` define the vector space.
    A dim mismatch would raise StorageError at upsert_embeddings time
    AFTER files were already mutated (lint apply Phase 0); a
    revision bump or normalize/distance flip would silently mix
    semantically-different vectors. Codex review finding, 0.4.0.
    """
    try:
        active_text = await storage.get_active_embed_version(modality="text")
    except NotSupported:
        return None
    if active_text is None or active_text.version_id is None:
        return None
    if cfg_provider is not None:
        cfg_provider_key = _qualified_provider(
            cfg_provider.embedding, cfg_provider.embedding_base_url
        )
        if (
            active_text.provider != cfg_provider_key
            or active_text.model != cfg_provider.embedding_model
            or active_text.revision != cfg_provider.embedding_revision
            or active_text.dim != cfg_provider.embedding_dim
            or active_text.normalize != cfg_provider.embedding_normalize
            or active_text.distance != cfg_provider.embedding_distance
        ):
            return None
    return active_text.version_id, active_text.model


async def _preflight_embedder(
    embedder: EmbeddingProvider, model: str
) -> None:
    """One-token embed call to surface permanent provider failures upfront.

    Lint apply / wisdom write mutate the filesystem before
    ``persist_knowledge`` / ``persist_wisdom`` reach the embed call.
    A permanent ``ProviderError`` (bad API key, 401, invalid model id)
    raised mid-persist aborts with partial state: documents row
    upserted, chunks replaced, but links / provenance not reconciled
    and no ApplyReport returned to the caller.

    Preflight makes the embed call BEFORE any filesystem mutation so
    misconfig surfaces while state is still clean. Permanent errors
    propagate as-is to the caller; transient errors are tolerated
    (the actual embed has its own retry-skip budget — they may still
    succeed). One round-trip cost per apply / wisdom write.

    Codex review finding, 0.4.0.
    """
    try:
        await embedder.embed(["preflight"], model=model)
    except TransientProviderError:
        # The real embed call will retry; preflight only blocks on
        # permanent failures.
        return


# ---- base scaffolding (Phase 0) -----------------------------------------


def _assert_base_upgraded(root: Path) -> None:
    """Refuse to operate against a pre-upgrade base layout.

    Two legacy shapes are flagged (both alpha rebuild-over-migrate, see
    ADR-0004 / ADR-0003):

    1. A ``<root>/wiki/`` tree carrying markdown (dikw-core ≤0.3.6, before
       the ``wiki/`` → ``knowledge/`` rename).
    2. A ``dikw.yml`` carrying the pre-0.5.0 ``schema.page_types`` key
       (before the configurable-``category``-taxonomy change). The on-disk
       knowledge layout and frontmatter (``type:`` → ``category:``) changed,
       so the base must be rebuilt under the new taxonomy.
    """
    legacy = root / "wiki"
    if legacy.is_dir() and any(legacy.rglob("*.md")):
        # A bare empty ``wiki/`` left behind by an earlier rename attempt is
        # harmless; only flag if it still carries markdown the user would lose.
        raise BaseUpgradeRequired(
            f"base at {root} carries a legacy `wiki/` directory from "
            "dikw-core ≤0.3.6; the K layer directory must be renamed to "
            "`knowledge/` and the database rebuilt. Run:\n"
            f"    cd {root} && mv wiki knowledge && rm -rf .dikw\n"
            "then start the server (`dikw serve --base .`) and run "
            "`dikw client ingest` to reindex. (If `knowledge/` already "
            "exists and you intended to merge content, do the merge by "
            "hand before retrying — the engine refuses to silently abandon "
            "wiki/*.md files.)"
        )

    cfg_path = root / CONFIG_FILENAME
    if cfg_path.is_file():
        try:
            raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError:
            return  # a malformed dikw.yml surfaces as a clearer error in load_config
        schema = raw.get("schema") if isinstance(raw, dict) else None
        if isinstance(schema, dict) and "page_types" in schema:
            raise BaseUpgradeRequired(
                f"base at {root} uses the pre-0.5.0 `schema.page_types` key; the K "
                "layer now classifies pages with a configurable `schema.categories` "
                "taxonomy and stores it in `category:` frontmatter (not `type:`). "
                "Declare your categories in dikw.yml, then rebuild:\n"
                f"    cd {root} && rm -rf knowledge .dikw\n"
                "(or move `knowledge/` aside to keep the old pages), then start the "
                "server (`dikw serve --base .`), run `dikw client ingest`, and "
                "`dikw client synth` to regenerate the knowledge tree. See "
                "docs/adr/0003-configurable-knowledge-taxonomy.md."
            )


def init_base(root: str | Path, *, description: str | None = None) -> Path:
    base_root = Path(root).resolve()
    base_root.mkdir(parents=True, exist_ok=True)

    existing = base_root / CONFIG_FILENAME
    if existing.exists():
        raise FileExistsError(f"{existing} already exists; refusing to overwrite")

    cfg = default_config(description=description or f"dikw base at {base_root.name}")
    existing.write_text(dump_config_yaml(cfg), encoding="utf-8")

    for rel_path, body in KNOWLEDGE_INIT_FILES.items():
        target = base_root / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.write_text(body, encoding="utf-8")

    # Scaffold one folder per declared category so a fresh Obsidian vault shows
    # the taxonomy immediately (default tree: entity / concept / note).
    for category in cfg.schema_.categories:
        keep = base_root / "knowledge" / category.path / ".gitkeep"
        keep.parent.mkdir(parents=True, exist_ok=True)
        if not keep.exists():
            keep.write_text("", encoding="utf-8")

    return base_root


def resolve_base_root(path: str | Path | None) -> Path:
    start = Path(path) if path is not None else Path.cwd()
    found = find_config(start)
    if found is None:
        raise FileNotFoundError(
            f"no {CONFIG_FILENAME} found at or above {start.resolve()}"
        )
    return found.parent


def load_base(path: str | Path | None = None) -> tuple[DikwConfig, Path]:
    root = resolve_base_root(path)
    _assert_base_upgraded(root)
    return load_config(root / CONFIG_FILENAME), root


async def _with_storage(path: str | Path | None) -> tuple[DikwConfig, Path, Storage]:
    cfg, root = load_base(path)
    storage = build_storage(
        cfg.storage, root=root, cjk_tokenizer=cfg.retrieval.cjk_tokenizer
    )
    # If connect or migrate raises, the partially opened pool / fd leaks
    # unless we close it on the failure path. Agents probe ``/v1/health``
    # frequently — a repeated migrate failure would otherwise accumulate
    # SQLite fds or Postgres pool slots.
    try:
        await storage.connect()
        await storage.migrate()
    except BaseException:
        with contextlib.suppress(Exception):
            await storage.close()
        raise
    return cfg, root, storage


async def status(path: str | Path | None = None) -> StorageCounts:
    _cfg, _root, storage = await _with_storage(path)
    try:
        return await storage.counts()
    finally:
        await storage.close()
