---
paths:
  - "src/dikw_core/*.py"
  - "src/dikw_core/client/**"
  - "docs/architecture.md"
---

# Module map

Full module map. `docs/architecture.md` has the long form.

```
src/dikw_core/
├── api.py                 thin re-export facade — surfaces every verb (ingest, retrieve,
│                          synthesize, lint (+ propose/apply), list_pages, read_page,
│                          list_links, read_provenance, list_graph, read_asset, status,
│                          health, check_providers, write_wisdom_page, delete_page) so the
│                          public `api.X` surface + `__all__` stay byte-stable; defines nothing
├── api_*.py               verb clusters the facade re-exports: api_core (scaffold +
│                          `_with_storage` + embed-version helpers), api_types (DTOs +
│                          exceptions), api_health, api_ingest, api_pages, api_graph,
│                          api_retrieve, api_synth, api_lint, api_wisdom, api_delete, api_path_safety.
│                          Each imports api_core/api_types + its domains, never the facade
│                          (acyclic). Move a verb's body here, not into api.py
├── cli.py                 top-level Typer app: version, init, serve, auth subgroup, client subgroup
│                          (HTTP-bound commands live exclusively under `dikw client <verb>` —
│                          there are no top-level short aliases)
├── auth_cli.py            `dikw auth {login,import,status,list,logout}` — local OAuth token store at <base>/.dikw/auth.json
├── logging.py             init_logging() — DIKW_LOG_LEVEL clamp + DIKW_LOG_FORMAT (text/json, trace-id correlation); clamps httpx/httpcore/urllib3 to WARNING
├── telemetry.py           OTel seam (optional [otel] extra) — get_tracer/get_meter accessors + dikw.*/gen_ai.* attribute keys + span helpers (gen_ai_span, op_span, traced_op, task_span) + record_* metric helpers + entry-only SDK bootstrap (configure_telemetry / configure_client_telemetry_from_env / shutdown_telemetry); imports only opentelemetry+stdlib, never server. Full cookbook: docs/observability.md
├── md_inspect.py          standalone markdown preflight — frontmatter + image-ref extraction (no engine deps)
├── progress.py            ProgressReporter Protocol + CancelToken (engine-side progress contract)
├── config.py              pydantic config + YAML loader (dikw.yml)
├── schemas.py             cross-layer DTOs (cross the Storage Protocol boundary — no SQL types)
├── domains/               DIKW domain model — the four layers grouped together
│   ├── data/              D layer — sources + assets + SourceBackend registry (markdown only)
│   ├── info/              I layer — chunk, tokenize, embed, render, RRF-fused hybrid search (+ optional post-fusion cross-encoder rerank)
│   ├── knowledge/         K layer — knowledge pages filed under a configurable `category` tree
│   │                                  (`knowledge/<category>/<slug>.md`), [[wikilinks]], frontmatter `sources:` ↔
│   │                                  provenance edge, lint (incl. missing_provenance, uncategorized, title_slug_quality, missing_file, stale_index, untracked_file, dangling_provenance), lint_fix + lint_fixers/
│   └── wisdom/            W layer — hand-written documents; `page.py::author_from_path`
│                                    (`wisdom/<author>/<slug>.md` → author); indexed exclusively by
│                                    `api.write_wisdom_page` (`dikw client wisdom write` / `POST /v1/base/wisdom`).
│                                    `dikw client ingest` does NOT scan `<base>/wisdom/`.
├── providers/             LLMProvider + EmbeddingProvider + MultimodalEmbeddingProvider + RerankProvider Protocols
│                          (anthropic_compat, openai_compat, openai_codex, gitee_multimodal, rerank [openai_compat_rerank])
├── storage/               Storage Protocol + adapters (sqlite, postgres) + migrations/{sqlite,postgres}
├── eval/                  retrieval + synth-quality eval — metrics, judge, dataset loader, runner, fake embedder
├── prompts/               versioned LLM prompts (importlib.resources); `resolve()` + `_contract.py` validate
│                          per-base overrides (`synth.prompt_path` / `lint.fixer_prompts`)
├── server/                FastAPI app, auth, sync + task + import + retrieve + pages + assets + graph routes,
│                          NDJSON streaming, task subsystem
└── client/                Remote Typer CLI + httpx transport + NDJSON progress + sources importer + converter dispatch
                           + baseline.py (eval --against/--write-baseline regression gate — pure, no engine import)
```
