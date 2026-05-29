"""High-level engine facade — server routes (``dikw_core.server``) and
the eval runner depend on this module; CLI access is via ``dikw client``
which talks HTTP to a running server instead of importing the engine.

This module is a thin **re-export facade**: every verb lives in a focused
``api_*`` cluster module and is surfaced here so the public ``api.X``
surface (and ``__all__``) stays byte-stable. Nothing is *defined* here.

Cluster map (each imports ``api_core`` / ``api_types`` and the domains it
needs, never this facade — so the import graph is acyclic):

  * ``api_core``     — base scaffold, storage open/migrate, embed-version
                       helpers (``init_base`` / ``load_base`` / ``status``).
  * ``api_types``    — cross-cutting DTOs + exceptions.
  * ``api_health``   — ``health`` / ``check_providers`` + provider probes.
  * ``api_ingest``   — ``ingest`` (D-layer write entry).
  * ``api_pages``    — ``list_pages`` / ``read_page`` / ``read_asset``.
  * ``api_graph``    — ``list_links`` / ``read_provenance`` / ``list_graph``.
  * ``api_retrieve`` — ``retrieve`` (RRF-fused hybrid search; no LLM).
  * ``api_synth``    — ``synthesize`` (the K-layer authoring leg; the only
                       place an LLM enters the engine).
  * ``api_lint``     — ``lint`` / ``lint_propose`` / ``lint_apply``.
  * ``api_wisdom``   — ``write_wisdom_page`` (W-layer write entry).
"""

from __future__ import annotations

from .api_core import (
    _assert_base_upgraded as _assert_base_upgraded,
)
from .api_core import (
    _register_text_version as _register_text_version,
)
from .api_core import (
    _with_storage as _with_storage,
)
from .api_core import (
    init_base,
    load_base,
    status,
)
from .api_graph import (
    list_graph,
    list_links,
    read_provenance,
)
from .api_health import (
    _PROBE_PNG_1X1 as _PROBE_PNG_1X1,
)
from .api_health import (
    _sanitize_base_url as _sanitize_base_url,
)
from .api_health import (
    check_providers,
    health,
)
from .api_ingest import ingest
from .api_lint import lint
from .api_lint import (
    lint_apply as lint_apply,
)
from .api_lint import (
    lint_propose as lint_propose,
)
from .api_pages import (
    list_pages,
    read_asset,
    read_page,
)
from .api_retrieve import retrieve
from .api_synth import (
    _LEGACY_BACKFILL_SENTINEL as _LEGACY_BACKFILL_SENTINEL,
)
from .api_synth import (
    _persist_knowledge_page as _persist_knowledge_page,
)
from .api_synth import (
    _sr_replace as _sr_replace,
)
from .api_synth import (
    _synth_pages_from_source as _synth_pages_from_source,
)
from .api_synth import synthesize
from .api_types import (
    AssetNotFound,
    CheckReport,
    EmbeddingInfo,
    HealthReport,
    IngestError,
    IngestErrorKind,
    IngestReport,
    LayerCounts,
    LlmInfo,
    MultimodalInfo,
    PageNotFound,
    ProbeResult,
    ProvidersInfo,
    SynthReport,
)
from .api_types import (
    BaseUpgradeRequired as BaseUpgradeRequired,
)
from .api_wisdom import write_wisdom_page
from .config import find_config

# ``doc_id_for`` is re-exported under its historical private spelling
# ``_doc_id_for``: tests and ``tests/fakes`` reach for it through the
# facade to build deterministic doc ids. The suppression marks the
# rename-alias re-export (a redundant ``x as x`` alias can't express a
# rename), so ruff keeps it rather than pruning it as unused-here.
from .domains.data.path_norm import doc_id_for as _doc_id_for  # noqa: F401
from .schemas import (
    DerivedPage,
    IncomingLink,
    OutgoingLink,
    PageAnchor,
    PageLinksResult,
    PageProvenanceResult,
    PageReadResult,
    PageRef,
    ProvenanceSource,
    RetrieveResult,
    WisdomWriteReport,
)
from .schemas import (
    Layer as Layer,
)

__all__ = [
    "AssetNotFound",
    "CheckReport",
    "DerivedPage",
    "EmbeddingInfo",
    "HealthReport",
    "IncomingLink",
    "IngestError",
    "IngestErrorKind",
    "IngestReport",
    "LayerCounts",
    "LlmInfo",
    "MultimodalInfo",
    "OutgoingLink",
    "PageAnchor",
    "PageLinksResult",
    "PageNotFound",
    "PageProvenanceResult",
    "PageReadResult",
    "PageRef",
    "ProbeResult",
    "ProvenanceSource",
    "ProvidersInfo",
    "RetrieveResult",
    "SynthReport",
    "WisdomWriteReport",
    "check_providers",
    "find_config",
    "health",
    "ingest",
    "init_base",
    "lint",
    "list_graph",
    "list_links",
    "list_pages",
    "load_base",
    "read_asset",
    "read_page",
    "read_provenance",
    "retrieve",
    "status",
    "synthesize",
    "write_wisdom_page",
]
