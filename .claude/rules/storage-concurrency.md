---
paths:
  - "src/dikw_core/storage/**"
  - "src/dikw_core/server/runtime.py"
  - "src/dikw_core/server/tasks/**"
  - "src/dikw_core/server/*_op.py"
  - "src/dikw_core/server/routes_{tasks,import}.py"
  - "tests/server/test_route_write_lock_wiring.py"
  - "tests/test_storage_contract.py"
  - "tests/server/test_task_store_contract.py"
---

# Storage adapters and write serialization

Read this rule before you add a storage method, a server task, or anything that mutates the base concurrently.

- **Write serialization is two-layered — don't re-introduce the races.** Concurrency is single-uvicorn-worker asyncio cooperative interleaving PLUS `asyncio.to_thread` preemptive OS-thread parallelism. (1) **Per-adapter:** a `SQLiteStorage` instance shares ONE `sqlite3.Connection` across all the `to_thread` workers a verb fans out (retrieval runs fts/vec/asset via `asyncio.create_task`); that connection's Python-level state is NOT thread-safe, so every method body runs through `self._locked(_run)` which acquires a per-instance `threading.RLock` **inside** the worker thread (must be `threading`, not `asyncio` — `_run` executes off the event loop; the task store's history records an `asyncio.Lock` here deadlocking a long-poll `Condition`). The three only-`to_thread`-not-`_locked` calls are lifecycle (`_open`/`initialize_jieba`/`conn.close`). Any new adapter method MUST go through `_locked`; a bare `asyncio.gather` of read methods on one instance otherwise trips `sqlite3.InterfaceError` (worked around three times historically: `storage/base.py` `get_assets`, `eval/runner.py`, `server/tasks/store_sqlite.py`) — and `connect` sets `busy_timeout=30000` + `connect(timeout=30)` so a cross-connection WAL writer blocks-then-succeeds, not immediate `database is locked`. (2) **Base-level:** every base-mutating task acquires the server's one `ServerRuntime.ingest_lock` (`asyncio.Lock`) — ingest, import, wisdom write, delete, **synth, and lint apply** (synth/lint take it via the `lock=` param on `make_synth_runner`/`make_lint_apply_runner`); they write the same deterministic `doc_id` rows + on-disk page with no enclosing transaction, so omitting the lock lets two writers silently lose anchors / drift the embed version / deactivate a healthy page. Read paths don't take it. `ingest_lock` is **per-process**, so multi-replica Postgres still needs cross-replica doc-level locking (not yet implemented).
