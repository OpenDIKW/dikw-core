# Orphan page governance

`dikw client lint` reports an `orphan_page` when a K-layer knowledge page has
zero inbound `[[wikilinks]]` from other pages. A knowledge base accumulates
orphans naturally as synth fans out: an ingested source mentions a
person, place, or sub-concept once, synth emits a page for it, but no
other page links back. Left alone, orphans pollute the link graph
(graph-leg retrieval misses them) and rot into stale notes nobody
reads.

`dikw client lint propose --rule orphan_page` runs the
`OrphanPageFixer` over every orphan and produces an actionable
proposal for each — never a silent skip. The fixer picks ONE of four
strategies per orphan, deterministic-signals first.

## The four strategies

### 1. `delete_page` — soft-delete a stub

Triggered when the orphan's body (after frontmatter strip) is **under
40 bytes AND has no outbound `[[wikilinks]]` AND** one of:

- the body is empty,
- the body starts with a `TODO` / `FIXME` / `WIP` / `stub` /
  `placeholder` / `draft` marker, or
- the orphan has no `sources` and no `tags` in its frontmatter.

Apply moves the file to `<base>/trash/knowledge/<original-rel-path>` and
purges its storage rows. **Recovery**: drag the file back into `knowledge/`
and rerun `dikw client ingest`. A `trashed: { at, reason, proposal_id }`
block is injected into the frontmatter on the way to trash for audit;
strip it by hand on the way back.

Example proposal:

```jsonc
{
  "issue_kind": "orphan_page",
  "issue_path": "knowledge/notes/half-written.md",
  "operations": [
    { "kind": "delete_page", "path": "knowledge/notes/half-written.md" }
  ],
  "rationale": "delete stub — body is 14 bytes (TODO/FIXME/WIP marker); soft-delete to trash",
  "source": "heuristic"
}
```

### 2. `merge_into_existing_page` — LLM-only

Triggered when the orphan scores **above `MERGE_THRESHOLD = 6.0`**
against an existing K-page (e.g. two shared `sources` entries, or one
shared source plus strong embedding similarity) AND `--enable-llm` is
set on the propose call. The fixer reads both bodies, asks the LLM to
rewrite the parent absorbing the orphan's content, and emits a 2-op
proposal:

```jsonc
{
  "operations": [
    { "kind": "update_page", "path": "knowledge/concepts/parent.md",
      "new_body": "<merged body>", "new_frontmatter": "<unioned sources+tags>" },
    { "kind": "delete_page", "path": "knowledge/concepts/orphan.md" }
  ],
  "source": "llm"
}
```

Frontmatter `sources` and `tags` are unioned deterministically by the
fixer (parent-order preserving). The LLM is told not to touch
frontmatter. Apply soft-deletes the orphan exactly like strategy 1.

Skipped silently if `--enable-llm` is off — propose falls through to
strategy 3 (link) instead.

### 3. `link_from_existing_page` — backlink injection

Triggered when a candidate scores **above `LINK_THRESHOLD = 3.0`** but
below `MERGE_THRESHOLD` (or when merge ran but the LLM returned no
usable response). The fixer appends `[[<orphan-title>]]` under a
stable `## 相关` heading at the bottom of the parent page:

```markdown
## 相关

- [[Detail of Phenomenon]]
```

A single `update_page` op on the parent — the orphan itself is
untouched. Successive auto-fixes accumulate links in the same `##
相关` section rather than each adding its own heading.

The fixer **refuses** this strategy when:

- the orphan has no title (the backlink wouldn't resolve), or
- the orphan's title is shared by ≥ 2 K-pages (the backlink would
  resolve to a different page — fix the `duplicate_title` issue
  first), or
- the parent body already contains the backlink (race / stale
  `links` table; skip silently).

### 4. `mark_as_leaf` — acknowledged terminal note

Always-on tail strategy. When nothing else fits, the fixer writes:

```yaml
lint:
  skip: [orphan_page]
  reason: no high-confidence parent or merge candidate; acknowledged as leaf
```

into the orphan's frontmatter. The next `dikw client lint` run
suppresses the issue and surfaces the page under
`LintReport.acknowledged_leaves` (visible with `--format json`). Users
who want to silence a specific orphan ahead of time can hand-add the
same block.

The list is **additive** — pre-existing entries in `lint.skip` are
preserved, and a pre-existing `lint.reason` written by the user is
NOT overwritten.

## Scoring weights and thresholds

Calibrated against real synth output. All weights live in
`src/dikw_core/domains/knowledge/lint_fixers/orphan_page.py`:

| Signal                        | Weight | Note                                  |
|-------------------------------|--------|---------------------------------------|
| shared `sources` entry        | +3.0   | strongest signal — same D-page lineage |
| shared full tag               | +1.0   | e.g. both have `topic/engine`         |
| shared tag domain (namespace) | +0.5   | both have `topic/...`, different leaf |
| title-token Jaccard           | ×2.0   | normalized via `links.normalize_base` |
| embedding cosine similarity   | ×3.0   | max over orphan-chunk × knowledge-chunk    |

`LINK_THRESHOLD = 3.0`, `MERGE_THRESHOLD = 6.0`. Tighten by editing
the module-level constants; baseline numbers in `evals/BASELINES.md`.

## CLI flow

```powershell
# Inspect orphan count first
uv run dikw client lint | python -c "import sys,json; d=json.load(sys.stdin); print(d['by_kind'].get('orphan_page', 0))"

# Heuristic-only propose (no LLM, no API cost). Generates link /
# delete / mark_as_leaf proposals.
uv run dikw client lint propose --rule orphan_page --limit 50

# Re-run with LLM to unlock merge strategy.
uv run dikw client lint propose --rule orphan_page --limit 50 --enable-llm

# Review the proposal task — note the task_id printed at the end.
uv run dikw client lint proposals

# Apply one proposal task. Proposals are aborted on first-op failure
# (later ops are skipped with a "earlier op failed" reason) but earlier
# successful writes in the same proposal stay on disk — there is no
# rollback. For the 2-op merge case specifically, the parent rewrite is
# queued first and the orphan soft-delete second; a failure between
# them leaves the merged parent + the orphan duplicate on disk, which
# the user can recover from by re-running propose (the next pass sees
# the duplicate and routes it to delete_stub or merge again). Order
# was chosen so that the worst case is duplication, not silent data
# loss.
uv run dikw client lint apply <task_id>

# Re-run lint — orphan count should drop, acknowledged_leaves should
# show whatever ended up on the mark_as_leaf branch.
uv run dikw client lint --format json
```

## Pre-committing to leaf status

If you want a page to stay an intentional leaf forever, add the
frontmatter directly — no propose needed:

```yaml
---
title: My One-Off Note
lint:
  skip: [orphan_page]
  reason: standalone reference card, not a graph node
---
```

`run_lint` will skip the page on every pass and count it under
`acknowledged_leaves`.
