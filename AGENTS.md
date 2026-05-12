# AGENTS.md

> Guide for AI agents that **use** dikw-core as a knowledge backend. If you're
> here to contribute code to the engine itself, see [CLAUDE.md](./CLAUDE.md)
> instead.

## What dikw-core is, from your point of view

`dikw-core` is a Python service that turns a directory of markdown notes
(plus assets) into a queryable knowledge engine spanning the **D**ata →
**I**nformation → **K**nowledge → **W**isdom layers. The directory it points
at is called a **dikw base** — `dikw.yml` lives at its root, `sources/`
holds the raw notes, and the engine writes K-layer wiki pages and W-layer
distilled wisdom back into the same tree as plain markdown.

You talk to it over HTTP+NDJSON via a long-running `dikw serve` process.
The Typer CLI (`dikw client …`) is one client; you can write your own
agent loop with any HTTP library.

The full server spec — auth posture, NDJSON wire format, every route — is
in [`docs/server.md`](./docs/server.md).

## Bootstrap

You probably want [`INSTALL_FOR_AGENTS.md`](./INSTALL_FOR_AGENTS.md). It
walks through install → init a base → set keys → start the server →
first retrieve call in concrete commands.

## Endpoints you will actually use

The agent surface is intentionally small. The server is **manually
started by the human operator** — if you can't reach `GET /v1/health`,
ask the user to run `dikw serve` (don't try to start it yourself).

**dikw-core does NOT do LLM answer synthesis.** It hands you ranked
chunks + applicable wisdom + the parsed wiki tree. Composing those into
an answer is your job — you run your own LLM with your own prompt,
applying query rewrite / expansion / conversation context as you see
fit. dikw-core is stateless; agents have the context dikw-core doesn't.

| route | purpose | when to call |
| --- | --- | --- |
| `GET /v1/health` | server self-description (`base_root`, `version`, `storage_engine`, `layer_counts`, `providers`) | first call after attach — confirms the server is up and which base it's bound to |
| `POST /v1/retrieve` | retrieval-only NDJSON (chunks + page refs, no LLM call) | knowledge access — feed the chunks into your own LLM prompt |
| `GET /v1/base/pages` | list pages registered in the base, optional `?layer=` filter | discovering page paths to read |
| `GET /v1/base/pages/{path}` | full page body + chunk anchors aligned to the parsed coordinate space | reading a specific page after a retrieval hit lands you on it |
| `POST /v1/ingest` | ingest whatever is on disk under `<base>/sources/` (loaded there by `POST /v1/import` or by the user dropping files in) | when the user adds/edits markdown and wants the index refreshed |
| `GET /v1/status`, `POST /v1/lint`, `POST /v1/check` | counts, lint issues, provider connectivity | sanity checks the user may ask for |

CLI equivalents — all `dikw client` commands default to JSON output
suitable for piping into `jq` or an agent loop. Human-readable rendering
requires opting in via `--format table`:

```
dikw client health                           # JSON by default
dikw client retrieve "your question" --plain # pipe-safe final JSON (chunks + page_refs)
dikw client pages list                       # JSON by default
dikw client pages get sources/notes/alpha.md # JSON
dikw client import ./local-sources           # pre-flights + imports md packages
dikw client ingest                           # rendered progress; NOT pipeable
```

`retrieve` consumes the NDJSON event stream server-side and emits the
final JSON payload (chunks + page_refs) to stdout. Pass `--plain`
whenever you pipe, otherwise the "retrieving…" rich banner lands on
stdout and breaks `jq`. The `--format json` and `--plain` toggles are
orthogonal: `--format` picks the *final* shape, `--plain` suppresses
the *intermediate* status. If you want the raw NDJSON event stream,
talk to `POST /v1/retrieve` over HTTP directly.

## A typical retrieval-augmented loop

1. `GET /v1/health` — confirm the server is up and grab `base_root` so you
   know which base the user pointed it at.
2. `POST /v1/retrieve` with the question + a `limit`. Each chunk hit
   carries a `path`, `layer`, `anchor`, `start_off`/`end_off`, plus
   full chunk `text` (on both the intermediate `retrieval_done` partial
   *and* `final.result.chunks`), plus `page_refs` listing the parent
   pages. A streaming agent can prompt off the partial without waiting
   for `final`.
3. If you want full pages instead of just chunks, follow the page refs
   with `GET /v1/base/pages/{path}` — that returns the parsed body plus
   anchors so you can re-locate every chunk hit inside the page body.
4. Feed the chunks into your own LLM prompt and produce the final
   answer client-side. dikw-core does not own the synthesis step.

## Things that will trip you up

- **Server lifecycle.** `dikw serve` is started manually by the user, not
  by the agent. If you can't connect, surface that to the user — do not
  spawn it yourself.
- **Auth.** Loopback (default) is open. Non-loopback hosts require a
  `DIKW_SERVER_TOKEN` bearer. The server-bound config is reflected in
  `GET /v1/info`'s `auth_required` field; `/v1/health` is the rich
  bootstrap probe and intentionally omits auth state.
- **The "base" terminology.** When the docs or CLI say "base", they mean
  the whole bound directory (which contains `sources/`, `wiki/`,
  `wisdom/`, `.dikw/`, `dikw.yml`). The K-layer subdirectory is still
  called `wiki/` on disk — that's intentional, since the user opens it
  in Obsidian. Don't confuse "the dikw base" with "the wiki/ folder
  inside it".
- **`/v1/base/pages/{path}` is index-driven.** Only paths registered as
  `DocumentRecord` rows resolve. If a markdown file exists on disk but
  hasn't been ingested, the route returns 404. Use `GET /v1/base/pages`
  to enumerate what's actually queryable.
- **NDJSON, not SSE — but only on streaming routes.** `POST /v1/retrieve`
  streams NDJSON directly on its response body. The async-task ops
  (`POST /v1/ingest`, `POST /v1/synth`, `POST /v1/distill`,
  `POST /v1/eval`) instead return a JSON `TaskHandle`
  (`{"task_id": "..."}`); follow the task by **opening
  `GET /v1/tasks/{task_id}/events`** as the NDJSON stream. Either way
  the final event has `type=final` and earlier events are `progress` /
  `partial` / `task_started`. There is no `data:` SSE prefix.
- **Per-file ingest errors are non-fatal.** A bad markdown file produces
  one `partial` event with `kind=file_error` and lands on
  `IngestReport.errors`, but the run continues. CLI users can pass
  `--strict` to flip this to a hard fail.

## Pointers

- [`docs/architecture.md`](./docs/architecture.md) — module map, layer
  contracts, named seams. Worth reading if you're going to do anything
  more than basic retrieval.
- [`docs/design.md`](./docs/design.md) — approved design doc. Source of
  truth for *intent*; supersedes anything inferred from code shape.
- [`docs/getting-started.md`](./docs/getting-started.md) — end-user
  walkthrough. Useful as a script when you're guiding a user through
  setup.
- [`docs/providers.md`](./docs/providers.md) — per-vendor config cookbook
  (MiniMax + Gitee, OpenAI, DeepSeek, Ollama, GLM, Gemini-compat, …) and
  production gotchas around batch size, dim locking, retry, prompt
  caching.
- [`docs/server.md`](./docs/server.md) — full HTTP wire spec, security
  posture, deployment notes.
