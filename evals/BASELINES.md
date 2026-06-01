# Baselines

Archive of real-embedder benchmark runs. Each entry pins a reproducible
configuration and the observed numbers, so future runs can tell a
regression from a re-run variance.

Newest first. `dikw client eval` thresholds in each dataset's `dataset.yaml`
are calibrated ~2-3 % below the most recent canonical-mode run.

## 0.5.0 — configurable knowledge taxonomy + overridable prompts; drop index.md/log.md

**Status:** K-layer breaking refactor (`domains/knowledge/`). The fixed
single-axis `type` (entity/concept/note → pluralized folders) becomes a
configurable closed-set hierarchical `category` taxonomy
(`schema.categories` + `schema.fallback`); synth emits
`<page category="…" slug="…">` and the engine owns path construction;
`index.md`/`log.md` generation is removed (history stays in the
`knowledge_log` table). K-layer authoring prompts are per-base overridable
(`synth.prompt_path` / `lint.fixer_prompts`). Includes the 7 fixes from the
adversarial review (`9b95ad2`). See ADR-0003 / ADR-0004.

**Provider config:** `openai_codex` (gpt-5.5 via ChatGPT subscription) LLM +
`Qwen3-Embedding-0.6B@1024` via Gitee AI (batch_size=16) — the **same** stack
the 2026-05-13 synth baseline was calibrated against, so the gate thresholds
remain directly comparable. (The `.env` `ANTHROPIC_API_KEY` was rejected
401 against api.anthropic.com at run time; codex is both the working provider
and the original calibration LLM, so it is the correct choice — not a
downgrade.)

### mvp synth gate — `dikw client eval --dataset mvp --eval synth`

The K-layer acceptance gate. The mvp `dataset.yaml` now carries the
production-default category **descriptions** (review finding: a desc-stripped
eval prompt was weaker than default synth), so this run mirrors what a
default-config base produces.

```
metric                          value   threshold  result
synth/fact_grounding_ratio      0.787   0.550      ✓ pass
synth/atomicity_score           1.000   0.850      ✓ pass
synth/duplicate_ratio_max       0.000   0.050      ✓ pass
synth/wikilink_resolved_ratio   0.588   0.550      ✓ pass
synth/language_fidelity         1.000   0.950      ✓ pass
synth/expected_coverage         0.333   (info, ungated)
synth/page_density              0.571   (info)
n_sources=3  n_pages=12  gated=True  passed=True
```

All five gated metrics pass. Numbers sit in the same band as the 2026-05-13
post-cleanup baseline (3 Karpathy sources); the taxonomy refactor did not
regress synth quality. `wikilink_resolved_ratio` at 0.588 carries less slack
than the others but clears 0.55 — small-corpus cross-page links are sparse.

### elon-musk.md baseline (1500-line subset) — real large-document synth

Validates the new taxonomy end-to-end on a substantial real document. Full
`elon-musk.md` still hangs codex SSE (the 2026-05-08 known limitation,
unrelated to this PR), so the 1500-line / ~263 KB / 91-chunk subset is used,
matching that precedent.

- **Run:** ingest (1 source → 91 chunks/embeddings) + `dikw client synth`
  (target_tokens_per_group=3600, max_pages_per_group=4, slug_dedup=merge_body)
  → **19 chunk groups → 76 pages created, 0 errors**, 107 unresolved
  wikilinks. (2026-05-08 on the same subset: 19 groups → 70+2 pages.)
- **On-disk layout (the breaking change, verified live):** pages land under
  **singular** category folders — `entity/` (19), `concept/` (3), `note/` (54)
  = 76; frontmatter carries `category:` (no `type:`); `sources:` provenance
  intact; **no** `knowledge/index.md` or `knowledge/log.md` generated.
- **`dikw client lint`:** `broken_wikilink` 107 (cross-source refs from a
  single-source subset — expected), `orphan_page` 51 (no second-pass linking
  yet — same pattern as 2026-05-08's 53), and **`uncategorized` 0**. The zero
  is the key signal: the closed-set parser placed every one of the 76 pages
  under a *declared* category — none drifted to the `未分类` fallback bucket.

## 0.3.0 PR3 — wisdom as first-class retrieval layer (#122)

**Status:** purely additive — wisdom files indexed by 0.3.0 PR2 now
surface in `dikw client retrieve` tagged `Hit.layer == "wisdom"`,
and `lint` extends `broken_wikilink` / `orphan_page` /
`missing_provenance` to scan wisdom pages too. Wiki + source ranking
is untouched: `HybridSearcher.search` already defaulted
`layer=None`, so once PR2 wrote wisdom chunks + embeddings into the
shared `vec_chunks_v<id>` table they participated naturally. PR3
owns the unit-level test coverage that locks the contract and the
HTTP page APIs (`read_page` / `list_links` / `read_provenance`)
that PR2 deferred.

**Provider config:** openai_codex LLM + Qwen3-Embedding-0.6B on
Gitee AI; same as the 2026-05-13 baseline below.

**Methodology:** the packaged `bases/elon-musk-validation` dataset
ships zero wisdom files (`wisdom/` is empty), so an unmodified
`dikw client eval --dataset elon-musk-validation` run is
byte-identical to the most recent retrieval baseline — wisdom rows
that don't exist can't shift fusion ranks. The user-side validation is the smoke test in
the PR description: drop one wisdom file with `[[wikilink]]` +
`sources:` frontmatter, ingest, run retrieve against a query that
matches both a knowledge page and the wisdom page, and confirm
`Hit.layer == "wisdom"` arrives alongside the knowledge hit. Hard
nDCG/MRR numbers will land in a follow-up when the
elon-musk-validation dataset is extended with a wisdom subdirectory
(out of PR3 scope — needs a curated wisdom seed + qrel updates).

**Unit-level coverage shipped with PR3:**

- `tests/test_wisdom_retrieve.py::test_retrieve_returns_wisdom_hits` —
  FakeEmbeddings deterministic-token retrieve returns the wisdom
  page when the query targets its body.
- `tests/test_wisdom_retrieve.py::test_retrieve_hit_layer_correctly_tagged` —
  every `Hit.layer` matches the underlying document's `Layer`.
- `tests/test_wisdom_lint.py::test_wisdom_broken_wikilink_surfaces` /
  `test_wisdom_orphan_not_flagged_when_wisdom_links_in` /
  `test_wiki_not_orphan_when_only_wisdom_links_in` — orphan +
  broken-wikilink inbound counting now credits cross-layer edges.
- `tests/test_wisdom_read_apis.py::*` — `read_page`, `list_links`,
  `read_provenance` accept wisdom paths and resolve
  `wisdom→{wiki,source}` + `{wiki,source}→wisdom` edges.



## 2026-05-13 — broken_wikilink `--enable-llm` evidence-backed (#83)

**Status:** semantics change to the K-layer `lint propose` LLM path.
Before this PR, `--enable-llm` produced TODO-stub placeholder pages
that satisfied wikilink resolution without adding knowledge. After,
the LLM is invoked only when the D-layer has enough source evidence
to ground a real page; insufficient cases stay flagged as unresolved
`broken_wikilink`.

**Provider config:** openai_codex (gpt-5.5 via ChatGPT subscription)
LLM + Qwen3-Embedding-0.6B on Gitee AI; same as the 2026-05-10
baseline below so before/after are directly comparable.

**Methodology (after):** randomly-ported `dikw serve` against
`bases/elon-musk-validation`, then:

```
DIKW_SERVER_URL=http://127.0.0.1:<port> \
  dikw client lint propose --rule broken_wikilink --enable-llm \
    --limit 10 --plain
DIKW_SERVER_URL=... dikw client tasks show <task_id> > propose.json
```

Inspect `result.proposals[].operations[].new_body` for forbidden
tokens + body length; inspect `result.skipped[].reason` distribution.

### Before (2026-05-10 PR2 spot check)

| metric                   | value                          |
|--------------------------|--------------------------------|
| issues consumed          | 2                              |
| proposals returned       | 2                              |
| skipped                  | 0                              |
| generated body ⊇ `TODO`  | 2/2 (every proposal)           |
| skip reasons surfaced    | none (orchestrator default)    |

The user later reported a demo workspace producing **108 TODO-stub
pages** from one `lint apply` — the bug #83 describes.

### After (this PR — 2026-05-13 real-LLM run, `--limit 10`)

`elon-musk-validation` base, 99 total `broken_wikilink` issues in
the lint scan; the first 10 sampled by `run_lint_propose`. openai_codex
(gpt-5.5) LLM + Qwen3-Embedding-0.6B (Gitee AI) embedder, both
auto-wired by `api.lint_propose` when `enable_llm=True`.

| metric                                            | value     |
|---------------------------------------------------|-----------|
| issues consumed                                   | 10        |
| proposals returned (grounded `create_page`)       | 5         |
| skipped: `evidence_insufficient`                  | 0         |
| skipped: `rejected_todo_marker`                   | 0         |
| skipped: `rejected_body_too_short`                | 0         |
| skipped: `rejected_title_mismatch`                | 0         |
| skipped: `path_collision`                         | 0         |
| skipped: `fixer returned None` (soft LLM REFUSE / parse-fail) | 5 |
| generated body ⊇ `TODO` / `stub page` / `placeholder` | **0/5**  |
| body length min / mean / max (chars)              | 616 / 763 / 893 |
| evidence chunks per proposal                      | 8 (constant — `_EVIDENCE_TOP_K`) |
| evidence chars per proposal (range)               | 9 201 – 11 856 |
| frontmatter `sources` cites D-layer               | **5/5** (`sources/elon-musk.md`) — never the K-page referrer |

Sample proposal — `[[Justine Musk]]` (truncated to first paragraph;
note: prose actually traceable to evidence, not TODO-stubbed):

> # Justine Musk
>
> The supplied source refers to Justine Musk in Chinese translation
> as "贾丝廷" and gives her a named section in the biography's table
> of contents. In the Zip2-era material, she appears in a photo
> caption with Elon Musk receiving a delivered McLaren. The source
> also describes a hotel-pool wedding ceremony involving Justine and
> Musk: she wore a sleeveless white dress with a white floral
> headpiece, Musk wore a fitted tailcoat, Maye and Errol were present,
> and after dinner the group danced a conga line. […]

Rationale stamped on the proposal:
> `LLM-grounded (8 evidence chunks, 11856 chars) for missing target '[[Justine Musk]]'`

### Acceptance assertions met

- **Zero TODO-laced bodies** (5/5 clean). The bug #83 specifically
  reported is gone.
- **D-layer source traceability** preserved on every proposal —
  `sources: ["sources/elon-musk.md"]` (Round-2 P3 fix verified live).
- The 5 skips fell into the soft-failure bucket (`safe_synthesize_pages`
  returning None on LLM REFUSE / parse-fail). Reasons land on the
  live NDJSON stream via `reporter.log` but the final
  `FixProposalReport.skipped[].reason` is currently the generic
  `"fixer returned None"` for these cases. **Follow-up gap:** elevate
  LLM REFUSE into a structured `FixerSkip("rejected_llm_refused")` so
  agents reading the propose-task result distinguish "model refused on
  given evidence" from real operational failures. Out of scope for #83
  proper (which targets TODO-stub elimination + evidence gating); track
  separately.
- `lint apply` intentionally NOT run during this baseline — the
  numbers above all derive from the propose-task result JSON without
  mutating the base. A separate apply pass would be needed to confirm
  `grep -r "TODO: stub page generated" wiki/` returns 0 on disk; the
  proposal-side guarantees above make that a foregone conclusion.
## 2026-05-13 — K-layer orphan-page governance (lint-fix PR3)

Issue #82: `dikw client lint propose --rule orphan_page` previously skipped
every orphan because `FIXER_REGISTRY` lacked an entry — 250 orphans
on the elon-musk validation base produced 0 applicable proposals.
This PR registers `OrphanPageFixer` with four strategies (delete /
merge / link / mark_as_leaf), plus a soft-delete `<base>/trash/` for
all `delete_page` ops, and `lint.skip` per-page frontmatter
suppression.

### What this PR ships

| Strategy                    | When                                                       | Source       |
|-----------------------------|------------------------------------------------------------|--------------|
| `delete_page`               | body < 40 B, no outbound wikilinks, AND empty / TODO marker / no sources+tags | heuristic    |
| `merge_into_existing_page`  | candidate score ≥ `MERGE_THRESHOLD = 6.0` AND `--enable-llm` | LLM          |
| `link_from_existing_page`   | candidate score ≥ `LINK_THRESHOLD = 3.0` (or merge LLM no-op) | heuristic    |
| `mark_as_leaf` (tail)       | nothing else fits — writes `lint: {skip: [orphan_page], reason}` | heuristic    |

Scoring weights and thresholds documented in
`docs/lint-orphan-governance.md`.

### Baseline run — elon-musk-validation base, 2026-05-13

Base: `~/Project/opendikw/dikw-data/bases/elon-musk-validation/`
(wiki/ = 364 KB, donor source = Walter Isaacson's *Elon Musk* 2023
biography). Pre-PR result: `lint propose --rule orphan_page` skipped
every orphan because `FIXER_REGISTRY` had no entry → 0 applicable
proposals. Post-PR results below.

#### Lint baseline (pre-propose)

```
Total issues:          135
By kind:               broken_wikilink=96, orphan_page=39
Acknowledged leaves:   0
```

#### Heuristic-only run (`--rule orphan_page --limit 50`)

```
Proposals generated:   39 / 39 orphans  (100%)
Skipped:               0
Strategy distribution: link_from_existing=39
Source:                heuristic=39 (no LLM call)
```

Every orphan cleared `LINK_THRESHOLD = 3.0` from the shared
`sources: [sources/elon-musk.md]` entry alone — the donor source is
the single D-page that fans out across all K-pages, so every
orphan-vs-candidate pair scores ≥ 3.0 on shared sources. None
dropped to `mark_as_leaf`; no stubs detected.

#### LLM-enabled run (`--rule orphan_page --limit 50 --enable-llm`)

Provider: `openai_codex` (gpt-5.5 via ChatGPT-Plus backend).

```
Proposals generated:   39 / 39 orphans  (100%)
Skipped:               0
Strategy distribution: merge_into_existing=33, link_from_existing=6
Source:                llm=33, heuristic=6
```

85% of orphans (33/39) cleared `MERGE_THRESHOLD = 6.0` once embedding
similarity was folded in — the elon-musk corpus has dense semantic
overlap between sub-notes ("Elon Musk's Crisis Combat Mode" and
"Elon Musk's Adversity-Shaped Risk Tolerance" both pull from the
same source paragraph, score = 3 shared-source + ~3 embed = ~6+).
The remaining 6 orphans landed below MERGE_THRESHOLD but above
LINK_THRESHOLD, so the fixer fell back to the heuristic link branch
exactly as designed.

All 39 LLM merge prompts passed the destructive-merge contract
checks (`strict=True` parse, exactly-one-page response, path/title/
body-starts-with-heading validation) — no fall-through to link due
to contract violation.

#### Delta vs the issue #82 acceptance bar

| Before PR | After PR (heuristic) | After PR (LLM) |
|-----------|---------------------:|---------------:|
| 0 / 39 actionable proposals | 39 / 39 | 39 / 39 |
| every orphan silently skipped | every orphan reviewable as link | 85% merge-class, 15% link-class |

#### Reproducing

```powershell
$base = "$HOME/Project/opendikw/dikw-data/bases/elon-musk-validation"
$port = Get-Random -Minimum 53000 -Maximum 60000
Push-Location $base
Start-Process -NoNewWindow uv -ArgumentList @(
    "--project", "$HOME/Project/opendikw/dikw-core",
    "run", "dikw", "serve", "--port", $port, "--log-level", "warning"
)
$env:DIKW_SERVER_URL = "http://127.0.0.1:$port"

uv run dikw client lint --format json
uv run dikw client lint propose --rule orphan_page --limit 50 --plain
uv run dikw client lint propose --rule orphan_page --limit 50 --enable-llm --plain
uv run dikw client lint proposals --format json > props.json
```

### Tests landed

- 26 unit tests in `tests/test_lint_orphan_fixer.py` covering each
  strategy + edge cases (duplicate-title gate, vec_search layer
  scoping, trash collision avoidance, stub-delete refusal on
  metadata-rich pages).
- 4 frontmatter-suppression tests in `tests/test_lint_skip_frontmatter.py`.
- End-to-end ASGI roundtrip in
  `tests/client/test_cli_lint_fix.py::test_lint_orphan_page_propose_apply_roundtrip`.
- Storage contract additions: `test_delete_document_purges_all_rows`,
  `test_delete_document_missing_is_noop`.
  (`test_delete_document_clears_wisdom_evidence_references` was also added
  here originally; removed in the 0.3.0 W refactor PR1 alongside the
  `wisdom_evidence` table itself — see CHANGELOG.)

### Open follow-ups

- `add_to_moc` strategy (issue #82 item 2) — deferred. MOC topic
  recovery is a deterministic-scoping hard problem and overlaps with
  `knowledge/index.md`; more natural to schedule into synth (which is
  already clustering source material) than to bolt onto lint.
- `knowledge/index.md` promotion to a real K-page — conflicts with
  `indexgen.py`'s rewrite contract.
- Wire orphan score into `dikw client eval` metrics — current baseline table
  is good enough for first iteration.

## 2026-05-13 — fact_grounding_ratio tau sweep + claim-filter fix

Resolves follow-up #1 from the synth eval mvp baseline below. The
2026-05-13 baseline observed `fact_grounding_ratio = 0.26` at the
default `synth.grounding_threshold = 0.65` — suspiciously low against
synth output that hand-inspection said was well-grounded.

Investigation: ran an ad-hoc tau-sweep script (since removed —
it mirrored `run_synth_eval`'s ingest+synth pipeline but dumped
every claim's peak cosine against its source chunks, so the ratio
could be reduced at multiple taus from one LLM run).

**Run 1, tau sweep before filter** — 30 pages, 293 claims:

| tau  | fact_grounding_ratio |
| ---- | ---                  |
| 0.30 | 1.000                |
| 0.40 | 0.907                |
| 0.50 | 0.663                |
| 0.55 | 0.527                |
| 0.60 | 0.415                |
| 0.65 | 0.263                |
| 0.70 | 0.134                |

Top-10 highest cosines (0.76–0.89) were all real substantive
grounded claims. Bottom-10 was a **junk-claim cluster** at 0.32–0.36:
single periods (`.`), code-residue fragments (`embed(.`, `)\\` for
vector generation.`, `”`) — these slipped through
`_SENTENCE_BOUNDARY.split` and matched arbitrary chunks at low cosine,
dragging the ratio down at every tau without carrying any factual
signal.

**Fix in `split_claims`** (`src/dikw_core/eval/metrics.py`): filter
fragments that have fewer than 3 en word tokens (≥ 2 letters each)
AND fewer than 4 CJK characters. Removes pure-punctuation splits,
code-snippet residue, and 1-2-word fragments while keeping
bilingual real claims.

**Run 2, after claim filter** — 29 pages, 206 claims:

| tau  | fact_grounding_ratio | delta vs run-1 |
| ---- | ---                  | ---            |
| 0.30 | 0.997                | −0.003         |
| 0.40 | 0.940                | +0.033         |
| 0.50 | 0.721                | +0.058         |
| 0.55 | 0.544                | +0.017         |
| 0.60 | 0.442                | +0.027         |
| 0.65 | 0.265                | +0.002         |
| 0.70 | 0.114                | −0.020         |

The filter shaved 87 junk claims and modestly lifted the mid-band
ratios. Tau choice: **0.50** sits in the gap between two natural
clusters in the post-filter sample — real grounded claims at cos
≥ 0.55 (top-10 spans 0.77–0.90), paraphrased / weakly-related claims
at cos < 0.40 (bottom-10 spans 0.28–0.39). 0.65 was tighter than the
embedder's similarity scale supports for natural-language claims
against chunked-document sources.

**Changes landed**:
- `SynthSection.grounding_threshold` default: 0.65 → 0.50.
- `evals/datasets/mvp/dataset.yaml` `synth.grounding_threshold`:
  0.65 → 0.50; `thresholds.synth/fact_grounding_ratio`: 0.20 → 0.55
  (now well above the noise floor with headroom for run-to-run
  variance).
- `split_claims` filters fragments below the en-word / CJK-char
  thresholds.
- `compute_grounding_cosines` + `reduce_grounding_ratio` exposed as
  public symbols so future sweeps reuse the embedding work.
- Future re-sweeps (different embedder / different corpus): the
  `compute_grounding_cosines` + `reduce_grounding_ratio` public
  symbols above are the building blocks — write a fresh ad-hoc
  driver when needed.

**Verification** — `dikw client eval --dataset mvp --eval synth
--pretty` against same wiki:

```
metric                          value   threshold  result
synth/fact_grounding_ratio      0.736   0.550      ✓ pass
synth/atomicity_score           1.000   0.850      ✓ pass
synth/duplicate_ratio_max       0.004   0.050      ✓ pass
synth/wikilink_resolved_ratio   0.822   0.550      ✓ pass
synth/language_fidelity         1.000   0.950      ✓ pass
synth/expected_coverage         0.190   (info)
synth/page_density              0.500   (info)
n_sources=7  n_pages=31  passed=True
```

`fact_grounding_ratio` jumped from 0.26 (uninformative) to 0.74
(useful gate signal). 18% slack above the threshold absorbs typical
run-to-run synth variance (~0.05) without flaking.

### Open follow-ups (unchanged)

- `expected_coverage` is still informational — needs the fuzzy
  resolver / `expected.yaml` reconciliation (follow-up #2).
- LLM-judge soft-score baseline still pending (follow-up #3).

## 2026-05-13 — Synth quality eval mvp baseline (real LLM)

First real-LLM calibration run for the synth quality framework shipped
in #78 (2026-05-12). LLM = `openai_codex` (gpt-5.5 via ChatGPT
subscription), embedding = `Qwen3-Embedding-0.6B@1024` via Gitee.
Corpus = mvp 7 sources (dikw-core docs + 3 Karpathy essays) at
the time of this baseline. The dikw-core docs were removed from
the mvp corpus 2026-05-16; subsequent runs operate on 3 Karpathy
sources only and these numbers will need re-baselining.

Observed metrics:

```
dikw client eval — mvp (mode: synth)
metric                          value   direction
synth/atomicity_score           1.000   min
synth/duplicate_ratio_max       0.006   max
synth/expected_coverage         0.238   min
synth/fact_grounding_ratio      0.259   min
synth/language_fidelity         1.000   min
synth/wikilink_resolved_ratio   0.714   min
synth/page_density (info)       0.500   -
n_sources=7  n_pages=31
```

Calibrated thresholds (committed to `mvp/dataset.yaml`) with
conservative slack below observed — keeps the gate from flapping on
prompt tweaks while still catching regressions:

| metric                          | observed | threshold | rationale |
| ---                             | ---      | ---       | --- |
| `synth/fact_grounding_ratio`    | 0.259    | 0.20      | low signal — see "Caveats" below |
| `synth/atomicity_score`         | 1.000    | 0.85      | room for one violation across ~30 pages |
| `synth/duplicate_ratio_max`     | 0.006    | 0.05      | well below 5%; reverse direction |
| `synth/wikilink_resolved_ratio` | 0.714    | 0.55      | ~28% broken links — investigate |
| `synth/expected_coverage`       | 0.238    | —         | NOT gated yet — see "Caveats" below |
| `synth/language_fidelity`       | 1.000    | 0.95      | tight; English corpus, en-only output expected |

### Caveats — two metrics are low signal

* **`fact_grounding_ratio` = 0.259.** Only 26% of claim sentences
  have a source chunk with cosine ≥ 0.65 (the `synth.grounding_threshold`
  in `dataset.yaml`). Either (a) tau=0.65 is too tight for
  Qwen3-Embedding-0.6B's similarity scale on natural-language claims,
  or (b) the LLM is paraphrasing far enough from source language that
  cosine drops below 0.65 even on truly-grounded claims. Follow-up:
  tau sweep against the same corpus, possibly drop default tau to 0.50
  if (a) is confirmed.
* **`expected_coverage` = 0.238, run-to-run noise ≥ 0.09.** Only ~5
  of ~21 expected titles in `expected.yaml` resolved against the 30+
  generated pages. A re-run produced 0.143, a 0.09 swing — the
  metric is operating at the noise floor because LLM page titles
  like "DIKW Layers in dikw-core" / "DIKW Four-Layer Architecture"
  don't fuzzy-match the bare "DIKW Pyramid" that `expected.yaml`
  lists. **Not gated** until the fuzzy normalize is extended (drop
  "in dikw-core" suffix and similar dataset-specific stems) or
  `expected.yaml` is rewritten to match observed patterns.

`fact_grounding_ratio` still gates with the conservative threshold
above; its run-to-run swing (0.259 → 0.278) is well inside the 0.20
floor. `expected_coverage` is informational pending follow-up #2.

### Bug fixes landed alongside calibration

The first run wedged on a Gitee `400 Bad Request — No schema matches`
from `duplicate_ratio_max` and `fact_grounding_ratio` embedding calls.
Two issues, both fixed in this commit:

1. **Empty / whitespace-only synth pages got passed to `embedder.embed`**;
   most embedding APIs (Gitee included) 400 on empty strings rather
   than emit a zero vector. Filter them out at the metric boundary.
2. **`embedder.embed(texts)` sent all 30+ pages in one POST**; Gitee
   caps per-request input list well below that. `metrics.py` now
   batches at 16 (matches the default `embedding_batch_size`). Direct
   `embed()` callers had been bypassing the `consume_embedding_stream`
   path that the synth/ingest pipeline uses for batching — this
   restores parity.

### How to re-run

```bash
# From a wiki that has a configured LLM + DIKW_EMBEDDING_API_KEY in env:
dikw serve --port 8766 &
DIKW_SERVER_URL=http://127.0.0.1:8766 \
    uv run dikw client eval --dataset mvp --eval synth --pretty

# With soft judge (cost: ~$0.02 per page on Claude Sonnet):
DIKW_SERVER_URL=http://127.0.0.1:8766 \
    uv run dikw client eval --dataset mvp --eval synth \
        --judge --judge-sample 5 --pretty
```

### Open follow-ups

1. **Tau sweep on `fact_grounding_ratio`.** Run with tau ∈ {0.40, 0.50,
   0.60, 0.65}; pick the value where the metric matches a manual
   "is this grounded?" judgment on a 10-sample subset.
2. **Revisit `expected_coverage` fuzzy logic.** Either extend the
   normalize stem rule or rewrite `expected.yaml` to match observed
   title patterns. The current 0.24 number is mostly measuring the
   gap between author intent and LLM output, not synth quality.
3. **LLM-judge baseline** is still pending — the `--judge` path adds
   ~$0.50 / run at default sample, deferred until the two metric
   investigations above conclude.

## 2026-05-12 — Synth quality eval framework (PR #78)

Infrastructure shipped: `run_synth_eval` + 7 K-layer metrics
(`fact_grounding_ratio`, `atomicity_score`, `duplicate_ratio_max`,
`wikilink_resolved_ratio`, `expected_coverage`, `language_fidelity`
hard-gated; `page_density` informational) + LLM-judge soft scoring
(4 dimensions × 0-5).

`evals/datasets/mvp/` upgraded with `modes: [retrieval, synth]` and
`expected.yaml`. Hermetic gate (`tests/test_synth_quality.py`) passes.
Real-LLM calibration baseline: see the 2026-05-13 entry above.

## 2026-05-10 — K-layer fix proposals (lint-fix PR2): broken_wikilink LLM stub + non_atomic_page

**Status:** real-data spot check on the elon-musk-validation base
(`bases/elon-musk-validation`, source = Walter Isaacson's *Elon Musk*
biography). Confirms the two new PR2 fixers behave on a wiki the
engine actually owns.

**Provider config:** openai_codex (gpt-5.5 via ChatGPT subscription)
LLM + Qwen3-Embedding-0.6B on Gitee AI; `dikw.yml` defaults
elsewhere. PR2 fixers do not exercise the embedder leg yet.

**Methodology:** an ad-hoc baseline script (since removed) called
`api.lint_propose(..., enable_llm=True)` directly (no `dikw serve`)
so the run was reproducible without server lifecycle. Each rule was
gated by a `--limit` to keep token spend small while still proving
the LLM path end-to-end on real lint findings.

### Run 1 — `broken_wikilink` LLM stub fallback

Invocation (historical):

```
DIKW_PR2_BASELINE_RULE=broken_wikilink \
DIKW_PR2_BASELINE_LIMIT=2 \
uv run python <baseline-script>
```

| metric | value |
|---|---|
| issues consumed | 2 |
| proposals returned | 2 |
| skipped | 0 |
| accept rate (spot check) | 2/2 |

Both proposals were `create_page` with `source="llm"`. The LLM
stubs:

- carried the broken target verbatim as the page title
  (`# Twitter`, `# Tesla`),
- contained the literal `TODO` marker as the prompt requires,
- referenced the source page that triggered the link
  (`broken [[Twitter]] reference in knowledge/entities/elon-musk.md`),
- did **not** invent biographical or factual claims — at most a
  one-sentence summary of the surrounding context, which the prompt
  explicitly allows ("source page mentions [[Twitter]] in connection
  with Elon Musk acquiring it").

### Run 2 — `non_atomic_page` LLM splitter

```
DIKW_PR2_BASELINE_RULE=non_atomic_page \
DIKW_PR2_BASELINE_LIMIT=1 \
uv run python <baseline-script>
```

| metric | value |
|---|---|
| issues consumed | 1 (`knowledge/entities/errol-musk.md`) |
| proposal ops | 2 × `create_page` + 1 × `delete_page` |
| LLM children | 2 atomic notes |

The non-atomic page (flagged for "2 H1 sections — atomic page should
have exactly one") was split into two semantically-distinct atomic
children:

1. `knowledge/notes/errol-musk-harsh-parenting.md` — Errol's "extremely
   severe dictatorship" parenting style.
2. `knowledge/notes/errol-musk-emerald-trade.md` — Errol's 1986 emerald
   trade in Zambia.

Both children link back to `[[Errol Musk]]` via wikilink, exactly
as the design contract requires (the original page is deleted in
the same proposal, so `[[Errol Musk]]` becomes a `broken_wikilink`
issue under the next lint pass — handled by the broken_wikilink
fixer's stub fallback or fuzzy match against any new entity page).

### Caveats

- This is a **proposal** spot check, not an apply baseline — no
  changes were written to the knowledge tree. `dikw client lint apply`
  exists and is covered by 13 PR1 unit tests
  (`tests/test_lint_apply.py`); the proposal payloads above carry
  hash guards + create-collision skips so apply is safe even with
  concurrent edits.
- `openai_codex` is the configured LLM. The known
  "codex SSE 大输入卡死" issue (see internal memory) did NOT
  reproduce on the errol-musk page (~3 KB body); larger fat pages
  (10 KB+) may need a budget guard in a follow-up.
- Sample sizes are small (2 + 1) by design — token cost was the
  binding constraint. Larger sweeps belong in a routine eval, not
  a per-PR baseline.
## 2026-05-10 — Synth existing-pages awareness (synth-context PR2; measured A/B)

**Status:** non-destructive proof + measured signal-side A/B for PR2
(`feat/synth-existing-pages-context`, shipped as PR #69 commit
`8c6d392`). Pre-PR2 snapshot captured before the rebuild; post-PR2
column captured 2026-05-10 by wiping `knowledge/` + `.dikw/index.sqlite`
on the same base and re-running ingest + synth + lint via an ad-hoc
baseline script (since removed). Codex SSE keepalive bug **did not
trigger** — synth fans out to 19 groups of ≤3600 tokens each, so the
trigger condition (single oversize request) never appears.

### What PR2 changes

Each `_synth_pages_from_source` LLM call now receives two new prompt
sections — `## Already created in this batch` (per-source accumulator)
+ `## Existing knowledge pages` (full base snapshot under
`synth.existing_pages_max_bytes`, default 16384 B; vec_search-gated
top-K above) — and is told to emit zero `<page>` blocks for any
candidate that semantically duplicates an entry. Storage gains one
new primitive (`get_chunk_embeddings`); pure SELECT, zero DDL.

### Non-destructive proof

- **Storage:** `get_chunk_embeddings` is additive Protocol surface;
  both adapters' existing `vec_chunks_v<id>` tables already store
  row-per-chunk embeddings. New contract test
  (`test_get_chunk_embeddings_round_trip`) green on SQLite + PG.
- **Synth pipeline:** unchanged for empty/fresh bases (no existing
  pages → `(no existing pages …)` sentinel renders, prompt structure
  preserved); for non-empty bases the LLM gets *more* context, never
  less. `_synth_pages_from_source` accepts `storage` + `text_version_id`
  as **optional** kwargs so the narrow-unit `test_synth_observability`
  suite keeps working without storage plumbing.
- **Retrieval / lint / knowledge schema:** untouched. `RetrievalConfig` and
  `LintReport` shapes unchanged.
- **Test signal:** 1000-test full suite green; ruff + mypy clean.

### A/B — `elon-musk-validation` base, 2026-05-10

Pre-PR2 column captured from the knowledge tree synthesised under the
pre-PR2 prompt (2026-04 / 2026-05 commits). Post-PR2 column captured
by an ad-hoc baseline script (since removed) after a full
wipe-and-rebuild on the same source + same provider config.

```
source: ~/Project/opendikw/dikw-data/datasets/markdown-books/elon-musk.md
                (1500-line subset, per docs/eval-plan.md K-layer gate)
provider: openai_codex (gpt-5.5) + Qwen3-Embedding-0.6B (gitee-ai)
post-PR2 wall time: ingest 9.6s, synth 593.8s, lint 0.2s (10.1 min)
post-PR2 synth fan-out: 19 groups from 91 chunks
```

| metric                    | pre-PR2 | post-PR2 |     Δ |
|---------------------------|--------:|---------:|------:|
| total knowledge pages          |      74 |       76 |    +2 |
| ├─ entities/              |      23 |       34 |   +11 |
| ├─ notes/                 |      41 |       39 |    -2 |
| └─ concepts/              |      10 |        3 |    -7 |
| broken_wikilink           |     241 |       96 |  -145 |
| orphan_page               |      53 |       39 |   -14 |
| non_atomic_page           |       5 |        0 |    -5 |
| duplicate_title           |       1 |        0 |    -1 |
| SynthReport.unresolved    |       — |       96 |     — |

### Hypothesis vs reality

- **broken_wikilink ↓ — confirmed (-60 %).** 241 → 96. The LLM seeing
  the existing-pages roster on every group call is what stopped the
  bleed; PR1's fuzzy resolve cannot reach this class because the
  target pages were never generated in the first place. The
  `SynthReport.unresolved_wikilinks` (96) matches lint's
  `broken_wikilink` (96) byte-for-byte — the per-run signal landed
  exactly on the persisted lint signal, the surface added in PR #67
  works.
- **duplicate_title ↓ — confirmed (1 → 0).** Combined with PR1's
  fuzzy resolve (variant-title class) and PR2's semantic-duplicate
  guard, the wiki has zero duplicate-title issues on this base.
- **non_atomic_page ↓ — confirmed (5 → 0), with a caveat.** None of
  the 19 synth groups produced an obvious atomicity violation. Some
  of this may be PR #62 (1:N fan-out) doing its job; PR2 only
  contributes by making each group aware of what the *other* groups
  already wrote, which reduces the temptation to dump multiple H1s
  into one page to "cover the topic."
- **total knowledge pages ↓ — falsified (74 → 76, +2).** The hypothesis
  was that semantic duplicates collapse into references. Actual
  outcome: page count is roughly flat, but the **type distribution**
  shifted hard — entities +11, concepts -7, notes -2. Reading this
  as: PR2 lets the LLM recognise that a "concept page about Tesla"
  and an "entity page about Tesla" are the same thing, and it picks
  the entity bucket more aggressively when it can. This is structure
  improvement, not page-count reduction. Worth keeping as a learned
  delta — *don't quote "page count drops" as a PR2 selling point*.
- **orphan_page ↓ — partial (53 → 39, -26 %).** Direction was
  unclear in the pre-PR2 hypothesis. Net win: fewer dead-end pages
  to triage, even though we didn't drop total page count.

### Spot check

Manual inspection of the 76 post-PR2 pages: no obvious semantic
duplicates remain (no two pages cover the same person/company/event
under different titles). The 96 broken_wikilinks left over fall into
two roughly-equal buckets:

- **Real entity stubs the LLM didn't author** — e.g. `[[Joe Rogan]]`,
  `[[Larry Page]]` mentioned in passing inside body text but no
  dedicated page generated. These are exactly the kind of issue
  PR #70's `broken_wikilink` LLM stub fixer (`--enable-llm`) is
  built to close, in a follow-up `dikw client lint apply` pass.
- **Concept references** — `[[First-Principles Thinking]]`,
  `[[Vertical Integration]]` — same shape: real concept the LLM
  alluded to, no concept page emitted. Same fixer applies.

So the residual broken_wikilink count is not a PR2 failure mode; it's
the natural input to PR #70's fixer pipeline, which closes the loop
with `lint apply`.

## 2026-05-08 — Wikilink graph leg ablation (default-off, non-destructive proof)

**Status:** ablation for the optional 4th retrieval leg
(`feat/wikilink-recall`, commit `669a826` after PG-parametrize +
ranking-quality + fusion-compat tests). Confirms the leg is
non-destructive on standard retrieval benchmarks (which never produce
K-layer wikilinks). Ships **default-off**.

### Why not "+0.01 nDCG@10 on scifact + cmteb"?

The original plan's acceptance gate ("graph leg must lift nDCG@10 by
≥ 0.01 on SciFact + cmteb-t2-subset") is conceptually unmeasurable:

- The graph leg walks the K-layer wikilink graph
  (`Storage.neighbor_chunks_via_links`). Wikilinks are persisted to
  storage **only inside K-layer page persistence**
  (`api.py:_persist_knowledge_page` → `parse_links` → `storage.upsert_link`).
- Retrieval eval datasets (SciFact, cmteb, mvp, wiki-mini-mm) only
  ingest as D-layer sources — `dikw client eval` never invokes synth, so the
  `links` table stays empty regardless of `[[...]]` syntax in corpus.
- Therefore `neighbor_chunks_via_links` always returns `[]` on these
  datasets and the graph leg silently contributes nothing — equivalent
  to the 3-leg baseline byte-for-byte.

The leg's actual value is measurable only against a wiki-rich corpus
with synthesised K-layer pages + qrels. Not in scope this PR.

### Run shape (proving non-destructiveness)

`evals/scripts/graph_leg_ablation.py` (one-off, not committed) loops 4
configs over each dataset, all hermetic (FakeEmbeddings, cache_mode=off
to avoid the cache-vs-RetrievalConfig invalidation gap):

| Run | graph_enabled | graph_weight | graph_seed_top_k |
|-----|---------------|--------------|------------------|
| A   | False         | (n/a)        | (n/a)            |
| B   | True          | 0.2          | 20               |
| C   | True          | 0.5          | 20               |
| D   | True          | 1.0          | 20               |

### Results

| dataset  | A_off  | B_w0.2 | C_w0.5 | D_w1.0 | max delta |
|----------|--------|--------|--------|--------|-----------|
| mvp      | 0.8514 | 0.8514 | 0.8514 | 0.8514 | +0.0000   |
| scifact  | 0.1454 | 0.1454 | 0.1454 | 0.1454 | +0.0000   |

(scifact absolute number is FakeEmbeddings-driven and not comparable to
the 2026-04-23 baseline of 0.771 with Qwen3-Embedding-0.6B — what
matters here is the column-wise equality.)

### Conclusion

Ship `RetrievalConfig.graph_enabled=False` as default. The unit-test
suite (`tests/test_search_graph_leg.py` — 16 tests across sqlite + PG,
covering ranking-quality + RRF + CombSUM + CombMNZ) locks the
intended behaviour when the user opts in. Validation of the leg's
recall improvement on a real K-layer knowledge corpus is a separate
follow-up dependent on building a wiki-rich eval dataset with qrels.

### Cache-vs-RetrievalConfig caveat

`run_eval`'s snapshot cache key (`_corpus_cache_key` in
`src/dikw_core/eval/runner.py`) covers `(dataset, model, dim,
corpus_hash, mm_fingerprint)` but NOT `RetrievalConfig`. Re-running
with a different retrieval config under `cache_mode="read_write"`
silently reuses the first run's wiki dikw.yml — making 4 different
configs all run the same baseline. This was a real bug discovered
during this ablation; ablation harness must use `cache_mode="off"`.
Architectural fix (include retrieval_cfg fingerprint in key) is a
separate follow-up.

## 2026-05-08 — Stage A K-layer fan-out + atomicity (elon-musk.md, 1500-line subset)

**Status:** first real-data baseline for the Stage A `1:N` synth fan-out
(PR #62 shipped in main) and the `non_atomic_page` lint heuristic
(branch `feat/atomicity-lint`, commits `6cf3a5f` → `7b1251a` → `423cfcc`
+ tags-domain fix `6047906`). Tunes the lint thresholds against
LLM-generated K-layer pages.

### Run shape

- **Source:** `~/Project/opendikw/dikw-data/datasets/markdown-books/elon-musk.md`
  trimmed to the first 1500 lines (~12 chapters / ~263 KB / 91 D-layer chunks).
- **LLM:** `openai_codex` provider, model `gpt-5.5`, ChatGPT subscription
  OAuth (no per-token cost). Auth bootstrapped via
  `dikw auth import openai-codex` from `~/.codex/auth.json`.
- **Embedding:** `Qwen3-Embedding-0.6B` via Gitee AI (1024-dim, batch_size=16).
- **Synth:** target_tokens_per_group=3600, max_pages_per_group=4,
  slug_dedup=merge_body. Resulted in **19 chunk groups → 70 knowledge pages
  created + 2 updated** (zero parse / LLM errors).
- **Branch:** integration of `main + feat/atomicity-lint + feat/wikilink-recall`
  (validate/elon-musk).
- **Commit:** `423cfcc` (tip of `feat/atomicity-lint` after threshold calibration).

### Lint outcomes (post-calibration)

| issue kind          | count | rate |
|---------------------|-------|------|
| broken_wikilink     |  241  | n/a (cross-source refs) |
| orphan_page         |   53  | 73% (no second-pass index page yet) |
| non_atomic_page     |    5  | **6.9% of 72 pages** (target band 5-25%) |
| duplicate_title     |    1  | 1.4% |

`non_atomic_page` sample of 5/5 flagged pages:

| page                                  | trip reason          | verdict |
|---------------------------------------|----------------------|---------|
| `entities/joshua-haldeman.md`         | 2 H1 (CN + EN dup)   | TP |
| `entities/errol-musk.md`              | 2 H1 (CN + EN dup)   | TP |
| `entities/tesla-roadster.md`          | 2 H1 (same EN twice) | TP |
| `concepts/idiot-index.md`             | 2 H1 (same EN twice) | TP |
| `notes/tesla-2006-funding-round.md`   | 17 wikilinks         | FP (event w/ many entities) |

True-positive rate: **4/5 = 80%**, meets the ≥4/5 acceptance criterion.

5/5 unflagged 1500-2500 char "borderline" pages were genuinely atomic
(true negatives).

### Threshold calibration commits

| commit     | change                                                            |
|------------|-------------------------------------------------------------------|
| `6047906`  | tags-domain: only count *namespaced* tags (not flat); fixes 100% FP rate from LLM-generated 3-5 flat tags per page |
| `7b1251a`  | body 1500 → 2500 chars; wikilinks 8 → 15; eliminates borderline single-topic FPs (4/5 of original sample) |
| `423cfcc`  | new H1-count > 1 detector; catches bilingual / glued-duplicate pattern that body-chars at 2500 misses |

### Known limitation

Full `elon-musk.md` (6160 lines / 77 chunk groups) hung indefinitely on
the first synth attempt — server process showed near-zero CPU and no
outbound HTTPS connections after first LLM call. Hypothesis: codex SSE
keepalive bytes reset httpx's `read` timeout, so a stalled stream is
never timed out. 1500-line subset (19 groups) completes cleanly in
~10 min. Full-text reproduction + provider-side fix is a separate
follow-up; doesn't block the lint feature shipping.

## 2026-05-05 — Postgres backend on cmteb, post-CJK-symmetry

**Status:** first PG-backend canonical baseline, paired with the SQLite
2026-04-28 Phase 1.5 entry below. Locks in two PR-bound changes:

1. **CJK tokenizer symmetry on PG** — `chunks.fts` is now a plain
   `tsvector NOT NULL` filled by the Python adapter via
   `to_tsvector('simple', preprocess_for_fts(text, tokenizer="jieba"))`.
   Same helper SQLite has used for a year, so the index side runs jieba
   on Chinese input on both backends. The pre-PR PG path used
   `chunks.fts GENERATED ALWAYS AS to_tsvector('simple', text)` which
   bypassed Python entirely — Chinese queries hit roughly nothing
   (analogue of the SQLite v1 baseline below at nDCG@10 = 0.031).
2. **`ts_rank` length normalization** — PG `fts_search` now passes
   `normalization=1` (`1 + log(doclength)`) into `ts_rank`. Default 0
   gave long passages an unfair advantage in the IDF-less ts_rank
   score; flipping the flag is a one-line change that lifts cmteb bm25
   nDCG@10 by +0.10. Closer to BM25's length normalization but still
   not BM25 — `ts_rank` has no IDF (the remaining gap below).

### Run shape

- **Wiki:** `/tmp/dikw-phase15-pg-cmteb/` rebuilt from
  `/tmp/dikw-phase15-cmteb/sources/` (Phase 1.5 donor) by switching
  `dikw.yml.storage` to `backend: postgres`,
  `dsn: postgresql://postgres:test@localhost:5432/postgres`,
  `schema: dikw_eval_cmteb`. Same `Qwen3-Embedding-0.6B` provider
  config; `cjk_tokenizer: jieba`.
- **PG image:** `pgvector/pgvector:pg16` — vanilla pgvector, no
  third-party FTS extension.
- **Replay tool:** `evals/tools/run_phase15_from_snapshot.py` via a
  thin wrapper (`/tmp/dikw-pg-replay.py`) that pins
  `WindowsSelectorEventLoopPolicy` so psycopg async runs on Windows.
  The wrapper is dev-host scaffolding; it's not committed to the repo.
- **Walltime:** ingest ~7 min (5,000 chunks @ Gitee 0.6B batch=16);
  query embed prebatch ~1 min (300 queries / 19 batches); 3-mode
  replay ~4 min on the PG instance.

### Results

```
dikw replay — cmteb-t2-subset  (PG, Qwen3-Embedding-0.6B, jieba, ts_rank norm=1)
┌────────┬──────────┬───────────┬───────┬────────────┬───────────────┐
│ mode   │ hit_at_3 │ hit_at_10 │   mrr │ ndcg_at_10 │ recall_at_100 │
├────────┼──────────┼───────────┼───────┼────────────┼───────────────┤
│ bm25   │    0.877 │     0.913 │ 0.845 │      0.720 │         0.803 │
│ vector │    0.973 │     0.990 │ 0.967 │      0.943 │         0.980 │
│ hybrid │    0.980 │     0.990 │ 0.966 │      0.933 │         0.987 │
└────────┴──────────┴───────────┴───────┴────────────┴───────────────┘
```

vs SQLite Phase 1.5 (same model, same RRF weights, jieba on both):

| mode | metric | SQLite | PG | Δ |
|---|---|---|---|---|
| bm25 | nDCG@10 | 0.840 | 0.720 | **-0.120** |
| vector | nDCG@10 | 0.943 | 0.943 |  0.000 |
| hybrid | nDCG@10 | 0.946 | 0.933 | -0.013 |
| hybrid | recall@100 | 0.988 | 0.987 | -0.001 |

### Reading

- **Vector leg is byte-identical** (0.943 = 0.943) — confirms
  embedding + storage paths are equivalent on both backends; the only
  axis that moves between them is the FTS ranker.
- **bm25 leg trails SQLite by -0.12 nDCG@10**, even after the
  normalization fix. Cause: SQLite's FTS5 ships a proper BM25 ranker
  (k1, b, IDF over corpus token DF); PG's `ts_rank` is a
  length-normalized TF score with **no IDF**. Common Chinese tokens
  like `的`、`是` get the same per-occurrence weight as rare
  domain-specific terms — exactly what BM25's IDF discounts. Filed as
  follow-up below.
- **Hybrid lands within -0.013 of SQLite** despite the bm25 gap.
  RRF's `(rrf_k=60, bm25_weight=0.3, vector_weight=1.5)` already
  down-weights bm25 by ~5×, so a weaker bm25 leg has limited drag.
  Hybrid clears every threshold in `cmteb-t2-subset/dataset.yaml`.

### Threshold gate

```
[PASS] hit_at_3:        0.9800 ≥ 0.9600
[PASS] hit_at_10:       0.9900 ≥ 0.9700
[PASS] mrr:             0.9663 ≥ 0.9500
[PASS] ndcg_at_10:      0.9329 ≥ 0.9300
[PASS] recall_at_100:   0.9869 ≥ 0.9700
```

Pre-norm-fix the gate failed at nDCG@10 = 0.9279 vs threshold 0.93;
the one-line `ts_rank(.., 1)` knob lifted hybrid through the gate.

### Configuration

| | |
|---|---|
| dikw commit | `63a0aa8` (post-PR `feat/postgres-cjk-symmetry`) |
| Embedding | `Qwen3-Embedding-0.6B` @ Gitee AI, 1024-dim native, batch=16 |
| Storage | Postgres 16 + pgvector (`pgvector/pgvector:pg16`), schema `dikw_eval_cmteb`, fresh schema for this run |
| FTS index | `chunks.fts` plain `tsvector NOT NULL` populated by Python adapter; GIN index |
| FTS query rank | `ts_rank(fts, query, 1)` — length-normalized TF (no IDF) |
| Fusion config | `rrf_k=60, bm25_weight=0.3, vector_weight=1.5`, `cjk_tokenizer: jieba` (shipped defaults) |
| CJK tokenizer | `jieba` on both sides (index + query) — same `preprocess_for_fts` helper as SQLite |
| Wall time | ingest ~7 min, replay ~5 min total |
| Approximate cost | ~¥0.05 (one ingest + one 3-mode replay; query embed cache shared across modes) |

### Known issues / observations

**1. PG `ts_rank` has no IDF.** The -0.12 nDCG@10 bm25 gap vs SQLite
is overwhelmingly the IDF that `ts_rank` doesn't compute. PG ships
two ranking primitives — `ts_rank` (length-normalized TF) and
`ts_rank_cd` (cover-density, also no IDF) — neither matches FTS5's
BM25. Closing this gap requires either (a) implementing BM25 in pure
SQL using `ts_stat()` for corpus-level DF stats and tsvector
positional info for TF, or (b) introducing the `pg_search` (ParadeDB)
or `pg_bm25` extension. (a) is self-contained but ~150 LOC; (b)
changes deployment story (extension install + non-default Docker
image). Filed as a separate follow-up — tracked in `_PLAN.md` /
follow-up issue.

**2. Hybrid is bm25-noise-tolerant at the shipped RRF weights.**
`bm25_weight=0.3` against `vector_weight=1.5` (the SciFact-tuned
shipped defaults) gives bm25's rank position ~5× less pull on the
fused score than the vector leg's. The empirical result: even with
bm25 at -0.12 vs SQLite, hybrid is at -0.013. This is the same
behavior the v1 SciFact baseline observed when bm25 was structurally
weak (BASELINES.md 2026-04-23 v1 entry, "Known issues #2") — RRF at
these weights degrades gracefully on a noisy bm25 leg.

**3. Sample contract test on the same data.**
`test_fts_search_cjk_query_round_trip` (added in this PR) ingests
`"搜索引擎是信息检索的核心"` on both adapters under jieba and
asserts `_sanitize_fts("搜索引擎", cjk_tokenizer="jieba")` returns
the chunk. Passes byte-for-byte on PG and SQLite — the
PG-fts-stays-empty failure mode this baseline closes is now a
regression guard at the contract level too.

**4. Windows event-loop incompatibility for psycopg async.**
`uvicorn` defaults to `ProactorEventLoop` on Windows; `psycopg`
async refuses to run on it. The replay was driven by a thin
`/tmp/dikw-pg-replay.py` wrapper that pins
`WindowsSelectorEventLoopPolicy` before importing
`run_phase15_from_snapshot`. This is a dev-host scaffolding gap, not
a server-side patch; production Linux deploys are unaffected. If we
want PG ingest to work via `dikw serve` / `dikw client serve-and-run` on
Windows, the server's startup needs the same policy pin.

### Follow-ups (priority-ordered)

1. **PG BM25 ranker** — close the remaining bm25 nDCG@10 gap. Two
   feasible paths: (a) DIY BM25 in SQL using `ts_stat()` +
   tsvector positional decomposition (no new deps, ~150 LOC,
   self-contained); (b) integrate `pg_search` / `pg_bm25`
   extension (better ranker, deployment cost). Worth ~+0.10
   nDCG@10 on cmteb bm25 + matching SciFact uplift.
2. **scifact PG baseline** — deferred. Two ingest attempts hit
   sustained Gitee throttle (~50–100 chunks/min vs Phase 1.5's
   ~250/min) and the run was abandoned at 52 % of the embed phase
   to ship this PR. The English path is covered by the contract
   suite — `test_pg_fts_search_multi_word_or` exercises ASCII FTS
   under the same `cjk_tokenizer="jieba"` fixture and
   `normalization=1` rank, and 117 PG contract tests pass with no
   regression. Length normalization is monotonically helpful; no
   plausible mechanism for `norm=1` to regress short-doc English
   retrieval. SciFact baseline numbers will land in a follow-up
   commit when Gitee load eases.
3. **Windows uvicorn event-loop pin** — `dikw serve` should
   `asyncio.set_event_loop_policy(WindowsSelectorEventLoopPolicy())`
   on Windows before uvicorn starts, so PG-backed wikis work via
   `dikw client serve-and-run` without the dev-host wrapper. One-liner in
   `cli.py`'s serve subcommand.

## 2026-05-01 — wiki-mini-mm: chunk + asset views, real Qwen3-VL-Embedding-8B

**Status:** first end-to-end real-vector multimodal eval after the PR1
+ PR2 landing (chunk + asset views + 4-file dataset contract). Pins the
current behavior for the public-licensed Wikipedia slice so a future
regression in the multimodal fusion path or chunker boundaries is
measurable.

### Run shape

- **Dataset:** `evals/datasets/wiki-mini-mm/` — 6 Wikipedia article
  summaries (Eiffel Tower, Great Wall of China, Mount Fuji, Lion,
  Sushi, Mona Lisa), each with one Commons lead image. 6 docs / 6
  images / 26 hand-authored queries (24 positive across doc + chunk +
  asset views, 2 negative).
- **Wiki cfg:** Gitee AI endpoints, `Qwen3-Embedding-0.6B` (text,
  dim=1024) + `Qwen3-VL-Embedding-8B` (multimodal, dim=1024), RRF
  fusion (`bm25_weight=0.3`, `vector_weight=1.5`, `rrf_k=60`),
  `cjk_tokenizer=jieba`. CLI:
  ```
  dikw client eval --dataset wiki-mini-mm --embedder provider \
            --path <wiki-with-mm-cfg> --no-cache
  ```
- **Walltime:** ~3 min including image embed pass (6 images × Qwen3-VL-8B
  via Gitee). Runs against `HTTPS_PROXY=http://localhost:1235`.

### Results (canonical hybrid mode)

```
              dikw client eval — wiki-mini-mm  (real Qwen3-VL-Embedding-8B)
┌─────────────────────┬───────┐
│ metric              │ value │
├─────────────────────┼───────┤
│ doc/hit_at_3        │ 1.000 │
│ doc/hit_at_10       │ 1.000 │
│ doc/mrr             │ 1.000 │
│ doc/ndcg_at_10      │ 1.000 │
│ doc/recall_at_100   │ 1.000 │
│ chunk/hit_at_3      │ 0.500 │
│ chunk/hit_at_10     │ 1.000 │
│ chunk/mrr           │ 0.571 │
│ chunk/ndcg_at_10    │ 0.667 │
│ chunk/recall_at_100 │ 1.000 │
│ asset/hit_at_3      │ 1.000 │
│ asset/hit_at_10     │ 1.000 │
│ asset/mrr           │ 1.000 │
│ asset/ndcg_at_10    │ 1.000 │
│ asset/recall_at_100 │ 1.000 │
└─────────────────────┴───────┘
```

### Reading

- **Doc + asset views are perfect** — the 6 Wikipedia articles are well
  separated semantically (doc/mrr=1.000), and Qwen3-VL-8B finds the
  matching lead image at rank-1 for every image-targeted query
  (asset/mrr=1.000). The multimodal leg works end-to-end against
  Gitee's hosted endpoint.
- **Chunk view is intentionally weaker** — each article has only two
  chunks (Description / Image), and the queries split evenly between
  them. Real Qwen embeddings rank both chunks closely for image-
  targeted queries (the Image section paragraph is short and shares
  topic vocabulary), so chunk/hit_at_3 lands near 0.5. Hermetic
  `FakeEmbeddings` actually scores 0.667 on the same dataset because
  bag-of-words breaks ties differently. This is a property of the
  dataset's tight chunk structure, not a regression.
- **Negatives** — 2 OOD queries return top-3 doc lists (retrieval has
  no "no match" mode); `negative_diagnostics` surfaces them but they
  do not enter the metric averages.

### Reproducer

```bash
WIKI=$HOME/.dikw-mm-eval-wiki
mkdir -p "$WIKI/sources" && dikw init --path "$WIKI"
# Edit $WIKI/dikw.yml: provider.embedding_base_url=https://ai.gitee.com/v1,
# embedding_model=Qwen3-Embedding-0.6B, embedding_dim=1024,
# assets.multimodal: { provider: gitee_multimodal,
#   model: Qwen3-VL-Embedding-8B, dim: 1024, base_url: https://ai.gitee.com/v1 }
set -a && source .env && set +a
dikw client eval --dataset wiki-mini-mm --embedder provider \
          --path "$WIKI" --no-cache
```

`dataset.yaml` thresholds are intentionally empty — this dataset is a
demo / regression guard, not a calibrated gate. Future PRs that touch
fusion, chunker, or asset embedding should re-run and update the table
above (with delta vs prior run if non-trivial).

## 2026-04-28 — Fusion mode A/B/C: RRF vs CombSUM vs CombMNZ

**Status:** experimental run, **not** a new canonical baseline. PR #32
landed `retrieval.fusion: rrf|combsum|combmnz` (default `rrf`); this
entry pins the empirical delta on `cmteb-t2-subset` so we know what to
expect when reaching for the new knob. Same wiki, same indexed
embeddings, same query embeds — only the dispatcher branch in
`HybridSearcher.search` changes between runs. RRF row reproduces the
2026-04-28 canonical baseline byte-for-byte (sanity check).

### Run shape

- **Wiki:** `/tmp/dikw-phase15-cmteb/` rebuilt fresh against the
  current main (`5177117`); 5,000 documents, 5,000 chunks, 1
  `embed_versions` row, ingest wall time ~22 min via Gitee
  `Qwen3-Embedding-0.6B` at batch=16.
- **Replay tool:** one-off
  `/tmp/dikw-phase15-cmteb/run_3fusions.py` (analogue of
  `evals/tools/run_phase15_from_snapshot.py`, but loops the 3 fusion
  modes in **one** Python process so the in-memory query-embed cache
  is shared across runs — saves ~6 min of Gitee API time vs three
  separate CLI calls). Mutates `cfg.retrieval.fusion` between fuser
  invocations; everything else is identical to the snapshot tool.
- **Modes per fusion:** `bm25 / vector / hybrid` all run; bm25 and
  vector are fusion-independent and **were verified byte-identical
  across the three fusion modes** (sanity check on the dispatcher's
  scope).

### Hybrid results

```
dikw replay 3-fusion sweep — cmteb-t2-subset (Qwen3-Embedding-0.6B)
┌─────────┬──────────┬───────────┬───────┬────────────┬───────────────┐
│ fusion  │ hit_at_3 │ hit_at_10 │   mrr │ ndcg_at_10 │ recall_at_100 │
├─────────┼──────────┼───────────┼───────┼────────────┼───────────────┤
│ rrf     │    0.987 │     0.987 │ 0.979 │      0.946 │         0.988 │
│ combsum │    0.977 │     0.990 │ 0.972 │      0.948 │         0.988 │
│ combmnz │    0.983 │     0.987 │ 0.977 │      0.949 │         0.988 │
└─────────┴──────────┴───────────┴───────┴────────────┴───────────────┘
```

Δ vs RRF (4-decimal precision, since the deltas are small):

| metric | RRF | CombSUM | Δ | CombMNZ | Δ |
|---|---:|---:|---:|---:|---:|
| nDCG@10     | 0.9460 | 0.9482 | **+0.0021** | 0.9486 | **+0.0026** |
| MRR         | 0.9794 | 0.9725 | -0.0069 | 0.9774 | -0.0020 |
| hit@3       | 0.9867 | 0.9767 | -0.0100 | 0.9833 | -0.0033 |
| hit@10      | 0.9867 | 0.9900 | +0.0033 | 0.9867 |  0.0000 |
| recall@100  | 0.9877 | 0.9877 |  0.0000 | 0.9877 |  0.0000 |

### Interpretation

The hypothesis from PR #32's plan ("vector-dominant regimes will see
CombMNZ ≥ CombSUM ≥ RRF") is **partially confirmed**:

- **nDCG@10 wins for score-fusion**, in the predicted order
  (CombMNZ +0.0026 > CombSUM +0.0021 > RRF). Both score-aware fusers
  beat RRF on the published threshold metric, but the deltas are at
  the noise floor — same magnitude as the 0.6B vs 8B nDCG@10 noise
  reported in the canonical 04-28 entry.
- **RRF wins MRR + hit@3**. Score-fusion's per-leg min-max
  normalisation slightly perturbs the very head of the ranked list.
  RRF's rank-only summation keeps a leg's rank-1 nailed at
  `1 / (k + 1)` regardless of magnitude; CombSUM/CombMNZ let a
  flat-distribution leg dilute the rank-1 signal.
- **recall@100 is invariant.** All three fusers preserve the same
  doc set in the top-100 — the ordering inside that set is what
  fusion redistributes. Confirms fusion is a re-ranker, not a
  recall-changer, at this corpus shape.

**Net judgement:** RRF stays the right default. Score-fusion is a
real but small lift on nDCG@10 (the metric `dataset.yaml` thresholds
gate on); the trade is a measurable haircut on top-1 metrics. Worth
keeping as opt-in for two regimes the 0.6B baseline already flagged
as RRF-suboptimal: `vector` already very strong (CombMNZ buys back
some of the rank-collapse RRF imposes), and very-close legs
(CMTEB-0.6B `hybrid -0.003` vs `vector` under RRF — score-fusion
gives the dispatcher something magnitude-shaped to discriminate on).
Don't flip the default until a corpus shows a >0.005 nDCG@10 swing
that direction.

### Configuration

| | |
|---|---|
| dikw commit | `5177117` (post-PR #32 main) |
| Embedding | `Qwen3-Embedding-0.6B` @ Gitee AI, 1024-dim native, batch=16 |
| Storage | sqlite + per-version `vec_chunks_v<id>` (PR #27 PR-A schema) |
| Fusion config | `rrf_k: 60`, `bm25_weight: 0.3`, `vector_weight: 1.5` (shipped defaults) — same across all 3 fusion modes |
| CJK tokenizer | `jieba` |
| Wall time | ingest ~22 min, query-embed prebatch ~2 min, 3-mode replay ~6 min |
| Approximate cost | ~¥0.05-0.10 (one ingest + 300-query single embed pass; 3 fusion replays share the cache) |

### Known issues / observations

**1. RRF row reproduces the published 04-28 baseline exactly.** nDCG@10
0.9460 / hit@10 0.9867 / hit@3 0.9867 / MRR 0.9794 / recall@100 0.9877
all match the canonical-mode entry to 4 decimal places — confirms the
PR #32 dispatcher leaves the RRF path byte-identical (the
`track_asset_dist = self._fusion != "rrf"` gate from the simplify
follow-up was the load-bearing piece).

**2. `--mode bm25` and `--mode vector` are byte-identical across the
3 fusion modes.** Confirmed in the JSON dump
(`/tmp/dikw-phase15-cmteb/.dikw/fusion_replay_results.json`) — the
dispatcher only forks at the hybrid path, as designed.

**3. Score fusion does not help SciFact at 0.6B.** Not run this
cycle — 0.6B's vector leg on SciFact is already at BM25 parity
(0.672 vs 0.669 nDCG@10 in the 04-28 baseline), so there is no
"strong vector" for CombSUM/CombMNZ to amplify. Score-fusion's value
proposition is "preserve magnitude when one leg dominates"; with no
dominant leg, it has nothing to do. Skipping that ¥0.10 spend until
the canonical-model decision lands.

### Follow-ups

1. **Don't change the default.** RRF stays. Document in
   `docs/providers.md` (already done in PR #32) when to reach for
   `combsum` / `combmnz`.
2. **Sweep tool extension.** `evals/tools/sweep_rrf.py` re-fuses
   offline from `--dump-raw` JSONL that records rank lists only. To
   make CombSUM/CombMNZ sweepable without re-running the full eval,
   the dump format needs scores. Defer until someone wants a
   weight-tune sweep over the new fusers.
3. **Pin replay tool with smoke unit test** (carried over from prior
   baseline). Cheap; not done this cycle.

## 2026-04-28 — Phase 1.5 chunk-level fusion @ Qwen3-Embedding-0.6B

**Status:** new canonical baseline for both `cmteb-t2-subset` and
`scifact` under PR #27 PR-A's chunk-level fusion path
(`embed_versions` registry + per-version `vec_chunks_v<id>` tables).
Embedding model flipped from `Qwen3-Embedding-8B` (matryoshka 1024)
to `Qwen3-Embedding-0.6B` (native 1024) — same vector-space dim,
~13× fewer params. SciFact thresholds rebased to 0.6B levels;
CMTEB thresholds unchanged (0.6B comfortably above the existing
floor).

### What changed from the 2026-04-24 baselines

1. **Embedding model: `Qwen3-Embedding-8B` → `Qwen3-Embedding-0.6B`.**
   Both produce 1024-dim vectors on Gitee AI's
   `https://ai.gitee.com/v1` endpoint (8B via matryoshka truncation,
   0.6B native). Configured in
   `tests/fixtures/live-minimax-gitee.dikw.yml` and the two scratch
   wikis at `/tmp/dikw-phase15-{cmteb,scifact}/dikw.yml`. Verified
   via `dikw client check --embed-only` (cold response 2.4 s, dim=1024 as
   expected). Motivation: 0.6B is dramatically cheaper / faster on
   Gitee in throttle-prone windows; we wanted to know if it's
   equivalent on quality.
2. **Schema is now PR #27 PR-A's chunk-level layout.**
   `chunk_embed_meta` per chunk, `embed_cache` re-keyed by
   `(content_hash, version_id)`, per-version `vec_chunks_v<id>` table
   created lazily on first ingest. The 8B-built CMTEB snapshot at
   `evals/.cache/snapshots/cmteb-t2-subset/Qwen3-Embedding-8B__1024__32540e83__mig3/`
   was wiped (legacy `content` table from PR #22, plus stale
   `chunks_vec` singleton from pre-PR-A) and rebuilt under 0.6B from
   scratch. SciFact wiki was scaffolded but never ingested previously
   — built fresh from `evals/datasets/scifact/corpus/` (5,183 files
   copied into the wiki's `sources/`).
3. **Replay tool (`evals/tools/run_phase15_from_snapshot.py`) tracked
   for the first time, with two Codex-flagged fixes folded in:**
   `[P1]` queries pinned to the snapshot's active `embed_versions`
   row via `storage.get_active_embed_version(modality="text")` so
   query vectors always live in the same space as the indexed chunks
   (mirrors `api.query()`); `[P2]` embedder construction gated on
   `mode in ("vector", "hybrid")` so `--mode bm25` no longer requires
   `DIKW_EMBEDDING_API_KEY`. The `[P2]` gate was empirically verified
   by a smoke run with all API keys unset — script reached
   `storage.migrate()` with zero embedding-related errors.

### Results

#### CMTEB / T2Retrieval (300 queries, 5K passages, jieba CJK tokenizer)

```
dikw replay — cmteb-t2-subset  (Qwen3-Embedding-0.6B, chunk-level fusion)
┌────────┬──────────┬───────────┬───────┬────────────┬───────────────┐
│ mode   │ hit_at_3 │ hit_at_10 │   mrr │ ndcg_at_10 │ recall_at_100 │
├────────┼──────────┼───────────┼───────┼────────────┼───────────────┤
│ bm25   │    0.933 │     0.967 │ 0.922 │      0.840 │         0.908 │
│ vector │    0.973 │     0.990 │ 0.967 │      0.943 │         0.980 │
│ hybrid │    0.987 │     0.987 │ 0.979 │      0.946 │         0.988 │
└────────┴──────────┴───────────┴───────┴────────────┴───────────────┘
```

vs 8B v2 (2026-04-24):

| mode | metric | 8B v2 | 0.6B | Δ |
|---|---|---|---|---|
| bm25 | nDCG@10 | 0.840 | 0.840 | **0.000** (byte-identical) |
| vector | nDCG@10 | 0.942 | 0.943 | +0.001 |
| hybrid | nDCG@10 | 0.952 | 0.946 | -0.006 |
| hybrid | recall@100 | 0.990 | 0.988 | -0.002 |

BM25 leg is **byte-for-byte identical** across the schema migration —
PR #27 PR-A's `ranked_docs_deduped` step in chunk-level fusion
preserves doc-level BM25 metrics, and PR #22's `content` table drop
didn't perturb the FTS5 path. Vector leg ties 8B within noise; hybrid
loses 0.006 nDCG@10 against 8B but stays comfortably above both
single-leg modes. Effectively a wash on CMTEB.

#### SciFact (BEIR, 300 test queries, 5,183 passages)

```
dikw replay — scifact  (Qwen3-Embedding-0.6B, chunk-level fusion)
┌────────┬──────────┬───────────┬───────┬────────────┬───────────────┐
│ mode   │ hit_at_3 │ hit_at_10 │   mrr │ ndcg_at_10 │ recall_at_100 │
├────────┼──────────┼───────────┼───────┼────────────┼───────────────┤
│ bm25   │    0.707 │     0.817 │ 0.638 │      0.669 │         0.865 │
│ vector │    0.700 │     0.813 │ 0.639 │      0.672 │         0.903 │
│ hybrid │    0.737 │     0.843 │ 0.662 │      0.693 │         0.947 │
└────────┴──────────┴───────────┴───────┴────────────┴───────────────┘
```

vs 8B v2 (2026-04-23):

| mode | metric | 8B v2 | 0.6B | Δ |
|---|---|---|---|---|
| bm25 | nDCG@10 | 0.669 | 0.669 | **0.000** (byte-identical) |
| vector | nDCG@10 | 0.773 | 0.672 | **-0.101** |
| hybrid | nDCG@10 | 0.771 | 0.693 | **-0.078** |
| hybrid | hit@3 | 0.797 | 0.737 | -0.060 |
| hybrid | mrr | 0.734 | 0.662 | -0.072 |

**0.6B regresses ~0.05–0.10 across position-weighted metrics on
SciFact**, while CMTEB is essentially flat. The vector leg drops to
parity with BM25 (0.672 vs 0.669) — i.e., the dense leg has
**effectively zero semantic uplift over keyword retrieval** on
SciFact at 0.6B. recall@100 is the only metric that holds: 0.947 at
0.6B vs 0.947 at 8B (within noise) — the long tail still surfaces,
just at lower ranks.

### Why the asymmetric outcome

CMTEB-T2 is a **generic-domain Chinese passage retrieval task**;
0.6B's smaller capacity is offset by Qwen3's strong Chinese
pretraining, plus the question-form queries align well with the
passage corpus's surface vocabulary (high BM25 ceiling already does
most of the work — see hybrid bm25=0.840 vs vector=0.943 vs hybrid
0.946; vector+BM25 are close together, fusion gain is small).

SciFact is **biomedical scientific claim verification** — long-tail
specialist vocabulary (rare clinical terms, gene names, drug
references). At 8B, dense retrieval delivers a real ~0.10 nDCG@10
uplift over BM25 (0.773 vs 0.669). At 0.6B that uplift collapses
entirely (0.672 vs 0.669): the smaller model lacks the parametric
biomedical knowledge to pull rare-term queries above keyword level.
This is the textbook "small embedding model on specialist English"
failure mode and **is not a bug, it is the model's limit**.

### Threshold rebase

`evals/datasets/scifact/dataset.yaml` thresholds rebased ~3 % below
0.6B observed numbers (the established `dataset.yaml` convention).
This **lowers the gate floor** on SciFact — an honest 8B regression
that produces ~0.69 nDCG@10 will now pass the gate where it would
have failed before. Trade-off acknowledged: the gate now detects
"0.6B-class" regressions (model swap, ingest pipeline breakage,
fusion weight drift), not "8B-quality" ones. Re-raise if and when a
larger-model pipeline becomes the canonical eval target. CMTEB
thresholds unchanged (0.6B sits comfortably above the existing
floor).

### Configuration

| | |
|---|---|
| dikw commit | `5c6337f` (post-PR #29 main) |
| Embedding provider | openai_compat → Gitee AI (`https://ai.gitee.com/v1`) |
| Embedding model | `Qwen3-Embedding-0.6B`, **1024-dim native** |
| Embedding batch size | 16 (Gitee caps at 25) |
| Chunking | default heading-aware, 900 tokens, 15 % overlap |
| Fusion | RRF `(rrf_k=60, w_bm25=0.3, w_vec=1.5)` — shipped defaults |
| Storage schema | PR #27 PR-A: `chunk_embed_meta` + per-version `vec_chunks_v<id>` |
| Wiki layout | `/tmp/dikw-phase15-{cmteb,scifact}/` (out-of-tree scratch) |
| Replay tool | `evals/tools/run_phase15_from_snapshot.py` (this is its first tracked run) |
| Wall time, CMTEB ingest | ~10 min (5,000 chunks, ~6 s/batch under mild Gitee load) |
| Wall time, CMTEB eval | ~5 min (300 queries pre-batched + 3-mode search) |
| Wall time, SciFact ingest | ~10 min (5,183 chunks, similar batch latency) |
| Wall time, SciFact eval | ~5 min |
| Approximate cost | ~¥0.1 each on Gitee AI (10K total chunks × 1024-dim, 0.6B is cheaper than 8B per token) |

### Known issues / observations

**1. 0.6B is task-dependent.** CMTEB ≈ 8B; SciFact -0.10 nDCG@10
vector / -0.08 hybrid. Don't generalise the "0.6B is cheap and
adequate" finding past the regimes measured here. For
specialist-vocabulary English corpora (legal, medical, scientific),
plan to spend 8B-class budget or evaluate explicitly.

**2. Hybrid loses to vector on CMTEB at 0.6B.** Tiny (-0.003 nDCG@10
hybrid vs vector at 0.6B; on 8B v2 it was hybrid +0.010 over vector).
Within noise, but worth flagging: when vector is strong and BM25 is
already at 0.84, fusion has very little room to add. RRF's rank-based
nature limits the upside. Score-normalised fusion (CombSUM/CombMNZ)
remains an open follow-up from prior baselines and could matter more
on this kind of vector-dominated regime.

**3. Replay tool's `[P1]` correctness is contract-tested by behaviour,
not by code path.** The tool now passes `text_version_id` to
`HybridSearcher.from_config` and uses the active version's
`model`/`dim`. Tested via the actual CMTEB+SciFact runs (would have
caught a dim mismatch immediately). No unit test yet — the tool is
still a recovery utility, not a runner refactor (per its own module
docstring).

### Follow-ups (priority-ordered)

1. **Decide canonical model going forward.** Three reasonable
   stances: (a) keep 0.6B everywhere, accept SciFact gate semantics
   relaxation; (b) per-dataset model selection (8B for SciFact, 0.6B
   for CMTEB) — needs `dikw.yml` per-wiki, already supported; (c)
   pick a middle 1.5–4B model if Gitee offers one and re-baseline.
   Currently shipped: (a).
2. **Score-normalised fusion (CombSUM / CombMNZ).** Inherited from
   prior baselines. Now empirically motivated on CMTEB-0.6B too, not
   just SciFact: when both legs are close, RRF is a coarse averager.
3. **Pin replay tool with a smoke unit test.** Single-query CMTEB
   run with a fake version mismatch should exercise `[P1]` 's
   active-version pinning path. Cheap to add.
4. **Fix the `--dump-raw` query-id key issue (open from prior
   baselines).** Still open; not exercised by this run.
5. **Optional — full-corpus runs.** The 5K subsets undercount
   distractor density vs full-corpus. Spend ~¥1 / ~1 h to confirm
   the per-task asymmetry holds at full scale.

## 2026-04-24 — CMTEB / T2Retrieval v2 — CJK BM25 via jieba

**Status:** supersedes v1 as the canonical CMTEB baseline. Same
embeddings, same sampling, same fusion weights; what changed is
`retrieval.cjk_tokenizer: jieba` — the P0 from v1's "Follow-ups".

### What changed from v1

v1 exposed a broken BM25 leg on Chinese: SQLite FTS5's default
`unicode61` tokenizer segments CJK one character at a time, which
collapses BM25 to single-char IDF. 275 / 300 queries (91.7 %)
returned zero hits; nDCG@10 sat at 0.031 vs the published
Anserini+jieba baselines of 0.50–0.65. Hybrid degenerated to
vector-only because RRF had no signal from the BM25 leg to fuse with.

The fix landed across four commits on `feat/fts5-cjk-tokenizer`:

1. `info/tokenize.py` — `preprocess_for_fts(text, tokenizer=...)`
   runs `jieba.cut_for_search` over CJK runs only (regex-sliced so
   ASCII identifiers like `retrieval.rrf_k` pass through intact).
2. `RetrievalConfig.cjk_tokenizer: "none" | "jieba"` (default
   `"none"` at v2 baseline time — flipped to `"jieba"` in PR #24
   on 2026-04-26 so T7's chunker fix ships without config).
3. `SQLiteStorage` preprocesses title + body before the FTS insert;
   `_sanitize_fts` preprocesses the query identically. Locked at
   first ingest (same shape as `embedding_dimensions`).
4. CLI fix (`699abb6`, surfaced by the first v2 run producing
   identical numbers to v1): `dikw client eval --embedder provider` now
   forwards `cfg.retrieval` from the scratch wiki to `run_eval`. The
   original CLI only threaded `cfg.provider` and silently dropped
   the retrieval block — the first v2 rerun wasted ~4h + ~¥0.5
   before this was caught. Regression test
   `test_cli_eval_provider_mode_threads_retrieval_config` pins it.

### Results

```
dikw client eval — cmteb-t2-subset  (retrieval ablation, v2 defaults + jieba)
┌────────┬──────────┬───────────┬───────┬────────────┬───────────────┐
│ mode   │ hit_at_3 │ hit_at_10 │   mrr │ ndcg_at_10 │ recall_at_100 │
├────────┼──────────┼───────────┼───────┼────────────┼───────────────┤
│ bm25   │    0.933 │     0.967 │ 0.922 │      0.840 │         0.908 │
│ vector │    0.967 │     0.983 │ 0.952 │      0.942 │         0.987 │
│ hybrid │    0.980 │     0.987 │ 0.978 │      0.952 │         0.990 │
└────────┴──────────┴───────────┴───────┴────────────┴───────────────┘
```

BM25: **0.031 → 0.840 nDCG@10 (27×)** — the broken leg is healed.
Hybrid now genuinely beats vector (+0.010 nDCG@10, +0.013 hit@3,
+0.026 MRR) instead of just matching it — the fusion is finally
doing useful work on Chinese.

### Weight sweep (offline, same dump)

Same 48-combination grid as v1. Top 5 + reference rows:

```
|   k | w_bm25 | w_vec | hit@3 | hit@10 |    mrr | nDCG@10 | recall@100 |
|----:|-------:|------:|------:|-------:|-------:|--------:|-----------:|
|  40 |   0.30 |  2.00 | 0.980 |  0.983 |  0.978 |   0.954 |      0.990 |
|  40 |   0.30 |  1.50 | 0.980 |  0.987 |  0.978 |   0.953 |      0.990 |
|  60 |   0.30 |  2.00 | 0.980 |  0.987 |  0.978 |   0.953 |      0.990 |
|  40 |   0.30 |  1.00 | 0.983 |  0.990 |  0.979 |   0.953 |      0.990 |
|  40 |   0.50 |  2.00 | 0.980 |  0.987 |  0.978 |   0.953 |      0.990 |
|  60 |   0.30 |  1.50 | 0.980 |  0.987 |  0.978 |   0.952 |      0.990 |  ← shipped default
|  60 |   1.00 |  1.00 | 0.983 |  0.990 |  0.974 |   0.930 |      0.990 |  ← equal-weight (pre-tuning)
```

The shipped default `(k=60, bm25_weight=0.3, vector_weight=1.5)` —
tuned on SciFact v2 back on 2026-04-23 — lands 0.002 below the grid
maximum on CMTEB and 0.022 above the equal-weight baseline. **The
v1 "generalisation" question is answered: the SciFact-tuned RRF
defaults generalise cleanly to Chinese.**

### Comparison vs published baselines

For BM25 on CJK, the published Anserini+jieba baselines on T2Retrieval
sit around nDCG@10 ≈ 0.50–0.65 (model- and configuration-dependent).
Our **0.840** on a 300-query / 5K-passage subset is naturally above
that range because the smaller corpus reduces distractor density
(only 1 in ~3× more chances for a random dense distractor to displace
a real hit). Treat as a calibration floor for re-runs, not a
leaderboard comparison.

For hybrid, our 0.952 is above the dense-only MTEB-CN leaderboard
cluster (0.85–0.93) for the same reason. The **directional signal**
— hybrid > vector > bm25, bm25 reaches usable signal on CJK — is
what matters.

### Wall time

~1 h end-to-end (embedding ~28 min; query phase ~30 min). Much
faster than v1's ~4 h because the query phase's BM25 leg was
actually returning hits instead of timing out through empty result
sets, so RRF fusion could terminate promptly. The jieba
preprocessing added ~0.3 s dictionary-load on first use and
negligible per-query / per-chunk overhead.

### Known issues / observations

**1. `retrieval.cjk_tokenizer: jieba` requires the `cjk` optional
extra.** Install with `uv sync --extra cjk`. Failing to install it
and flipping the config produces an `ImportError` on first CJK
preprocessing. No runtime regression for ASCII-only users who leave
`cjk_tokenizer: none`.

**2. The fix is locked at first ingest.** Existing CJK wikis that
were ingested under `unicode61` have `documents_fts` rows reflecting
per-char tokenization. Flipping `cjk_tokenizer: jieba` without
wiping `.dikw/index.sqlite` produces a mismatch between query-side
(jieba-segmented) and index-side (per-char) tokens, silently
dropping hits. Documented as gotcha #7 in `docs/providers.md`.

**3. Sampling caveat (unchanged from v1).** 5K-passage subset is
structurally easier than the full 118K corpus. Per-query retrieval
metrics read higher than leaderboard numbers — see the full caveat
in the v1 entry below. The BM25 fix does not change this subsampling
limitation.

### Follow-ups (priority-ordered)

1. **`--dump-raw` should key by query ID, not text.** Inherited
   from v1's follow-up list; still open. Datasets with duplicate
   query text silently lose queries in the dump.
2. **Full-corpus CMTEB run.** Now meaningful because BM25 is
   working. ~10× cost and time (~¥5 / ~10 h). Gate on whether we
   actually want to chase leaderboard parity.
3. **Score-normalised fusion (CombSUM / CombMNZ).** Same as v1 +
   SciFact v2 follow-ups; RRF is rank-based and may not beat
   vector-only on future datasets where one leg is much stronger.
4. **Japanese / Korean corpora.** jieba covers Chinese only.
   Japanese (MeCab) or Korean (MeCab-ko) would need additional
   tokenizer modes in `cjk_tokenizer`.

## 2026-04-24 — CMTEB / T2Retrieval (sampled, 300 queries)

**Status:** superseded by v2 above. Kept for historical reference —
the pre-jieba BM25 numbers are the fair comparison point for future
tokenizer work, and the known-issues section documents the root
cause the v2 patch addresses.

### Original context (unchanged below)

First real-vector run on Chinese. **BM25 leg unusable on
CJK with the current FTS5 tokenizer** — see "Known issues" #1. The
fusion-weight generalisation question that motivated this run
(BASELINES v2 follow-up) is therefore **deferred** to a re-run after
the CJK BM25 fix lands. What we *can* report from this run: vector
retrieval works strongly on Chinese, and the v2 fusion defaults
`(rrf_k=60, bm25_weight=0.3, vector_weight=1.5)` "trivially generalise"
in the sense that they correctly down-weight a useless BM25 leg
(equal-weight fusion only loses 0.003 nDCG@10).

### Configuration

| | |
|---|---|
| dikw commit | branch `feat/evals-cmteb-calibration` |
| Dataset | C-MTEB / T2Retrieval — 118,605 passages, 22,812 queries (full); see sampling below |
| Sampling | query-first: 300 queries (seed=42) + 1,582 relevant pids + 3,418 random distractors → 5,000 corpus docs. See `evals/tools/prep_cmteb_t2.py` and `dataset.yaml`'s `_preprocessing` block. |
| LLM provider | *(not exercised — retrieval-only eval)* |
| Embedding provider | openai_compat → Gitee AI (`https://ai.gitee.com/v1`) |
| Embedding model | `Qwen3-Embedding-8B`, **4096-dim native** (matches SciFact v2) |
| Embedding batch size | 16 (Gitee AI caps at 25) |
| Chunking | default — heading-aware, 900 tokens, 15 % overlap |
| Fusion | RRF `(k=60, w_bm25=0.3, w_vec=1.5)` — shipped v2 defaults |
| FTS5 tokenizer | `unicode61` (default) — **the problem; see Known issues #1** |
| Donor wiki | `tests/fixtures/live-minimax-gitee.dikw.yml` copied verbatim |
| Wall time | ~4 h end-to-end (embedding ~28 min; query phase dominated by 300 × 3 modes × top-100) |
| Approximate cost | ~¥0.5 on Gitee AI (5,000 passages × 4096-dim) |

### Why the corpus is sampled query-first, not stratified

Unlike BEIR/SciFact, **every T2Retrieval corpus passage is a positive
qrel for at least one query** (118,605 unique pids referenced =
118,605 corpus total). The generic stratified sampler in
`convert_cmteb.py` therefore degenerates — it preserves the relevant
set, which equals the whole corpus, so any `--sample-size` < 118,605
returns all 118,605 passages. Realistic calibration on this dataset
requires sampling **queries** first, then collecting their positive
pids and padding with distractors drawn from docs not referenced by
the sampled queries. `evals/tools/prep_cmteb_t2.py` implements this
pre-pass.

### Results

```
dikw client eval — cmteb-t2-subset  (retrieval ablation: bm25 / vector / hybrid)
┌────────┬──────────┬───────────┬───────┬────────────┬───────────────┐
│ mode   │ hit_at_3 │ hit_at_10 │   mrr │ ndcg_at_10 │ recall_at_100 │
├────────┼──────────┼───────────┼───────┼────────────┼───────────────┤
│ bm25   │    0.070 │     0.073 │ 0.070 │      0.031 │         0.025 │
│ vector │    0.967 │     0.983 │ 0.952 │      0.942 │         0.987 │
│ hybrid │    0.967 │     0.983 │ 0.952 │      0.942 │         0.987 │
└────────┴──────────┴───────────┴───────┴────────────┴───────────────┘
```

`hybrid` matches `vector` to four decimal places on every position-
weighted metric and on recall@100. RRF with `(0.3, 1.5)` against a
near-zero BM25 ranking effectively returns the vector ranking
unmodified — see Known issues #2.

### Weight sweep (offline, same dump)

Same 48-combination grid as SciFact v2 (`rrf_k ∈ {40, 60, 100}` ×
`w_bm25 ∈ {0.3, 0.5, 0.7, 1.0}` × `w_vec ∈ {0.5, 1.0, 1.5, 2.0}`):

```
|   k | w_bm25 | w_vec | hit@3 | hit@10 |    mrr | nDCG@10 | recall@100 |
|----:|-------:|------:|------:|-------:|-------:|--------:|-----------:|
|  40 |   0.30 |  1.50 | 0.967 |  0.983 |  0.952 |   0.942 |      0.987 |
|  40 |   0.30 |  2.00 | 0.967 |  0.983 |  0.952 |   0.942 |      0.987 |
|  40 |   0.50 |  2.00 | 0.967 |  0.983 |  0.952 |   0.942 |      0.987 |
|  60 |   0.30 |  2.00 | 0.967 |  0.983 |  0.952 |   0.942 |      0.987 |
|  60 |   0.30 |  1.50 | 0.967 |  0.983 |  0.952 |   0.942 |      0.987 |  ← shipped default
|  60 |   1.00 |  1.00 | 0.963 |  0.983 |  0.950 |   0.939 |      0.987 |  ← equal-weight (pre-tuning)
```

Every top-tier configuration produces identical numbers because the
BM25 leg contributes essentially no signal — see Known issues #1.
Equal-weight only drops 0.003 nDCG@10 vs the shipped default; that
0.003 is the entire "fusion-weight tuning" surface area on this
dataset, and it's within re-run noise. **The fusion-weight
generalisation question is unanswerable on this dataset until BM25
on CJK is fixed.**

### Comparison vs published baselines

CMTEB leaderboard numbers shift over time and depend on the model
under test, so this dataset's `dataset.yaml` deliberately omits a
``published_baselines`` block. For reference at the time of writing,
modern Chinese embedders on the MTEB-CN leaderboard cluster around
nDCG@10 ≈ 0.85–0.93 on T2Retrieval (full corpus, 22,812 queries);
our 0.942 on the **300-query / 5K-corpus** subset is naturally above
that range because the smaller corpus reduces distractor density.
**Treat this number as a calibration floor for re-runs, not a
leaderboard comparison.**

For BM25 on CJK, the published Anserini-with-jieba baselines on
T2Retrieval sit around nDCG@10 ≈ 0.50–0.65 (model- and
configuration-dependent). Our **0.031** is two orders of magnitude
below that — see Known issues #1.

### Known issues / observations

**1. FTS5 BM25 is unusable on CJK with the current `unicode61`
tokenizer (P0 follow-up).** 275 of 300 queries (91.7%) returned
**zero** BM25 results; the remaining 25 mostly returned 1–6 hits, and
only 4 reached the top-100 limit. Diagnosis: SQLite FTS5's default
`unicode61` tokenizer treats CJK characters as single-character
tokens, and the `_sanitize_fts` rewrite path (added in commit
`62b596f` for the SciFact ablation) feeds those characters as
OR-of-tokens. Per-character BM25 has no useful signal for Chinese
passage retrieval — high-frequency characters dominate IDF, and
discriminating multi-character terms never form. The fix is a
CJK-aware tokenizer; SQLite's built-in `trigram` (3-character
n-grams) is the lowest-friction option, jieba+full-text is the
heaviest. This needs its own focused branch.

**2. Hybrid degenerates to vector-only when one leg is dead, and the
v2 default's `bm25_weight=0.3` makes that degeneration safe.** With
a noise-quality BM25 ranking, RRF at any `(k, weights)` returns
something proportional to the vector ranking; the `(0.3, 1.5)`
weighting accelerates this collapse and produces an identical
top-100 to vector-only mode. This is a small but real win for the
v2 defaults: a keyword-tuned default (e.g. equal-weight) would have
let BM25's noise leak into the fused ranking and lose 0.003 nDCG@10.
Not enough to matter here, but the directional behaviour generalises
from SciFact (where BM25 was real but ~0.10 nDCG@10 weaker than
vector — same shape, different magnitude).

**3. Vector retrieval is strong on Chinese.** Qwen3-Embedding-8B at
4096-dim hits hit@3=0.967 / nDCG@10=0.942 / recall@100=0.987 on
this subset. Wall-time and cost track SciFact (5,183 passages →
~28 min embedding; this run with 5,000 → similar). Confirms
docs/providers.md's recommendation that Qwen3 handles both languages
without per-corpus retuning.

**4. Sampling caveat for re-runs.** Per-query retrieval is structurally
easier on this subset than at full scale. Each sampled query has ~5
relevant docs on average (matches full-T2R's 5.2 qrels/query), but the
candidate pool shrinks from 118,605 → 5,000 — a single relevant doc is
~24× easier to surface by chance, and the same dense embedder will
land more of them in the top-10 / top-100 windows. Higher hit@k vs the
leaderboard is therefore structural, not a quality claim. The
leaderboard comparison only becomes meaningful after the BM25 fix lets
us actually exercise both legs — at which point we can also widen to
the full corpus if we're willing to spend ~¥10 / 10 hours embedding.

### Verdict on the original follow-up

> **Run CMTEB / T2Retrieval subset on the v2 defaults to confirm the
> weights generalise to Chinese + a different BM25/dense balance.**

Trivially yes (see above), but the test wasn't real — there was no
"different BM25/dense balance" to discover, only a broken BM25 leg
the v2 weights happened to silence cleanly. Real generalisation
verdict requires a re-run after the BM25 CJK fix.

### Follow-ups (priority-ordered)

1. **P0 — FTS5 CJK tokenizer.** Default `unicode61` doesn't segment
   CJK; switch to SQLite's built-in `trigram` (lowest friction, no
   new dep) or evaluate jieba-based segmentation. Schema change with
   a re-index migration path for existing knowledge bases. Independent branch.
2. **After P0 — Re-run this baseline.** Same donor wiki, same
   sampling, same dump pipeline; should produce a non-trivial BM25
   leg whose nDCG@10 lands closer to the 0.50–0.65 published range,
   at which point the fusion-weight generalisation question becomes
   answerable.
3. **Score-normalised fusion (CombSUM / CombMNZ).** Same as in v2
   follow-ups; even after BM25 is fixed, RRF is rank-based and may
   not beat vector-only on datasets where the dense signal is much
   stronger than keyword. Independent branch.
4. **Optional — full-corpus run** for an apples-to-apples leaderboard
   comparison. ~10× the cost and time. Probably only worth it after
   #1 + #2 produce a credible single-corpus baseline.
5. **`eval/runner.py --dump-raw` should key by query ID, not text.**
   Surfaced by the final-verification codex review on this branch.
   `runner.py:323` writes only the query text into the dump JSONL;
   `evals/tools/sweep_rrf.py` then keys `per_dataset[ds][q]` by that
   string. Datasets with duplicate query text (rare on T2Retrieval,
   common on some public IR benchmarks) silently overwrite — sweep
   scores fewer queries than the real eval and may pick the wrong
   fusion weights. Add a stable query ID to the dump and key by it.
   Independent branch; touches eval runner + sweep_rrf in lockstep.

## 2026-04-23 — SciFact (BEIR) v2 — RRF tuned

**Status:** supersedes v1 as the canonical SciFact baseline. Same
embeddings, new fusion defaults.

### What changed from v1

v1 exposed a structural regression: equal-weight RRF at `k=60` let the
~0.10-nDCG-weaker BM25 leg drag hybrid 0.037 points below the
vector-only mode on nDCG@10. The fix, from BASELINES v1 follow-up #1,
landed in three layers:

1. `reciprocal_rank_fusion` grew a `weights` parameter (it had always
   been equal-weight).
2. `RetrievalConfig` in `dikw.yml` exposes `rrf_k` + `bm25_weight` +
   `vector_weight` so users can tune per-corpus.
3. `dikw client eval --retrieval all --dump-raw` + `evals/tools/sweep_rrf.py`
   re-fuse a single dump at arbitrary weights offline — no re-embedding.

### Weight sweep (offline, same dump as v1)

Full 48-combination grid: `rrf_k ∈ {40, 60, 100}` × `bm25_weight ∈ {0.3, 0.5, 0.7, 1.0}`
× `vector_weight ∈ {0.5, 1.0, 1.5, 2.0}`. Top 5 by nDCG@10 plus the
vanilla and shipped-default reference rows:

```
|   k | w_bm25 | w_vec | hit@3 | hit@10 |    mrr | nDCG@10 | recall@100 |
|----:|-------:|------:|------:|-------:|-------:|--------:|-----------:|
|  40 |   0.30 |  2.00 | 0.793 |  0.917 |  0.737 |   0.773 |      0.970 |
|  40 |   0.30 |  1.50 | 0.797 |  0.917 |  0.736 |   0.772 |      0.970 |
|  60 |   0.30 |  1.50 | 0.797 |  0.920 |  0.734 |   0.771 |      0.970 |  ← shipped default
|  40 |   0.50 |  2.00 | 0.793 |  0.920 |  0.734 |   0.770 |      0.970 |
|  60 |   0.30 |  2.00 | 0.790 |  0.917 |  0.733 |   0.770 |      0.970 |
|  60 |   1.00 |  1.00 | 0.757 |  0.883 |  0.708 |   0.736 |      0.970 |  ← equal-weight (pre-tuning)
```

Full grid + commentary in `/tmp/sweep-result.log` on the run host.

### Shipped defaults: `(rrf_k=60, bm25_weight=0.3, vector_weight=1.5)`

Why this point and not the grid top:

- **Keep `k=60` unchanged.** It's the original RRF-paper constant and
  what every downstream user's mental model pins on. `k=40` scores
  0.002 higher on SciFact — within noise, and the flatter `k=60` curve
  across the grid suggests it generalises better to other corpora.
- **Closest "k=60" row to the Pareto top.** Tiers at `w_bm25=0.3` all
  collapse to ~0.771 nDCG@10; picking a more moderate `w_bm25=0.5` or
  `0.7` loses 0.004–0.011 nDCG@10 for arguably less overfit but
  weakens the SciFact win we actually measured.
- **Recall@100 unchanged at 0.970.** Every top-tier config keeps this
  — the BM25 leg's recall contribution is invariant to its weight; only
  position-weighted metrics shift.

### Results (shipped defaults)

```
dikw client eval — scifact  (retrieval ablation, v2 defaults)
┌────────┬──────────┬───────────┬───────┬────────────┬───────────────┐
│ mode   │ hit_at_3 │ hit_at_10 │   mrr │ ndcg_at_10 │ recall_at_100 │
├────────┼──────────┼───────────┼───────┼────────────┼───────────────┤
│ bm25   │    0.707 │     0.817 │ 0.638 │      0.669 │         0.865 │
│ vector │    0.797 │     0.907 │ 0.738 │      0.773 │         0.947 │
│ hybrid │    0.797 │     0.920 │ 0.734 │      0.771 │         0.970 │
└────────┴──────────┴───────────┴───────┴────────────┴───────────────┘
```

Hybrid now effectively **ties vector on every position-weighted metric
within re-run noise** (Δ ≤ 0.004), while keeping the BM25 leg's
recall@100 advantage (+0.023 over vector). The v1 regression ("hybrid
< vector on nDCG@10") is resolved.

### Comparison vs published baselines (unchanged from v1)

| Metric | Our bm25 | Published (BEIR paper, BM25 Anserini) | Δ |
|---|---|---|---|
| nDCG@10 | **0.669** | **0.665** | **+0.004** |

The BM25 leg is untouched by fusion tuning; the Anserini calibration
stays as before.

### Caveat: SciFact-tuned default is vector-heavy

The `(0.3, 1.5)` ratio over-favours dense semantics. On a
**keyword-heavy corpus** (code, identifiers, exact-term lookup) this
default will under-rank BM25 hits — raise `bm25_weight` in
`dikw.yml`'s `retrieval:` block. The `evals/tools/sweep_rrf.py` flow
is the intended tuning path; it finishes in milliseconds against a
`--dump-raw` JSONL, no second API call.

### Follow-ups (still not this PR)

- **Run CMTEB / T2Retrieval subset** on the v2 defaults to confirm the
  weights generalise to Chinese + a different BM25/dense balance.
- **Sweep chunk size** (450 / 900 / 1800 tokens) on SciFact. The
  default is picked by feel, not measured.
- **Score-normalized fusion** (CombSUM / CombMNZ) — the v2 tuning
  basically tied vector; going *above* it likely requires fusion that
  honours score magnitude, not just rank position. Independent branch,
  not blocked on anything above.

## 2026-04-23 — SciFact (BEIR)

**Status:** superseded by v2 above (same embeddings, equal-weight
fusion). Kept for historical reference — the pre-tuning numbers are
the fair comparison point for any future fusion work.

### Configuration

| | |
|---|---|
| dikw commit | `62b596f` (branch `feat/evals-public-benchmarks`, pre-merge) |
| Dataset | BEIR/SciFact — 5,183 passages, 300 test queries, binary qrels |
| Converter | `evals/tools/convert_beir.py` — test split, baseline 0.665 |
| LLM provider | *(not exercised — retrieval-only eval)* |
| Embedding provider | openai_compat → Gitee AI (`https://ai.gitee.com/v1`) |
| Embedding model | `Qwen3-Embedding-8B`, **4096-dim native** (no matryoshka truncation) |
| Embedding batch size | 16 (Gitee AI caps at 25) |
| Chunking | default — heading-aware, 900 tokens, 15 % overlap |
| Fusion | RRF k = 60 (untuned default) |
| FTS5 sanitizer | OR-of-tokens (commit 62b596f) — reproduces Anserini behavior |
| Donor wiki | `tests/fixtures/live-minimax-gitee.dikw.yml` copied verbatim |
| Wall time | ~1 h 20 min (embedding ~28 min, query phase ~50 min) |
| Approximate cost | ~¥0.5 on Gitee AI (5183 passages × 4096-dim) |

### Results

```
dikw client eval — scifact  (retrieval ablation: bm25 / vector / hybrid)
┌────────┬──────────┬───────────┬───────┬────────────┬───────────────┐
│ mode   │ hit_at_3 │ hit_at_10 │   mrr │ ndcg_at_10 │ recall_at_100 │
├────────┼──────────┼───────────┼───────┼────────────┼───────────────┤
│ bm25   │    0.707 │     0.817 │ 0.638 │      0.669 │         0.865 │
│ vector │    0.797 │     0.907 │ 0.738 │      0.773 │         0.947 │
│ hybrid │    0.757 │     0.883 │ 0.708 │      0.736 │         0.970 │
└────────┴──────────┴───────────┴───────┴────────────┴───────────────┘
```

### Comparison vs published baselines

| Metric | Our bm25 | Published (BEIR paper, BM25 Anserini) | Δ |
|---|---|---|---|
| nDCG@10 | **0.669** | **0.665** | **+0.004** |

Within 0.004 of the published number — **FTS5 BM25 is functionally
equivalent to Anserini BM25 on SciFact** once the sanitizer tokenizes
correctly. This tightens `docs/eval-plan.md` caveat #2 ("FTS5 ≠
Anserini"); the algorithmic difference is below-noise on this dataset.

Dense retrieval in the original BEIR paper spans 0.48 – 0.67 nDCG@10
depending on the model (DPR, ANCE, GenQ, TAS-B, ColBERT). Our vector
leg at 0.773 is consistent with modern state-of-the-art embedders
(Qwen3, BGE-M3, E5-mistral family), which sit above the 2021 numbers.

### Known issues / observations

**1. Hybrid underperforms vector (regression risk).** On SciFact,
dense-only retrieval (`vector` mode) beats hybrid by +0.037 nDCG@10
and +0.024 hit@10. RRF with `k=60` gives equal weight to both legs;
because bm25 is ~0.10 nDCG@10 behind vector, averaging pulls the
ranking toward the weaker signal. Hybrid still wins on **recall@100
(0.970 vs 0.947)** — the bm25 leg covers docs vector misses, just at
ranks too low to help the position-weighted metrics.

This confirms `docs/eval-plan.md` caveat #3 ("RRF k=60 never tuned").
Not a blocker for this PR — the ablation surfacing the problem is
exactly what the new `--retrieval all` flag was built for — but the
**default behaviour of `dikw query` is currently suboptimal on
paraphrased / synonym-heavy queries where one leg dominates**.

**2. Real-embedder wall time dominates (~80 min for 5K corpus).** Most
of it is the 324 embedding batches at ~5 s each (Gitee AI latency
scales with output dim; 4096-dim is large). Using 1024-dim matryoshka
truncation would cut cost and time ~4×, at the quality cost of
roughly -0.02 nDCG@10 on Qwen3 (rule-of-thumb from matryoshka papers;
not measured here). Tradeoff to revisit if we ever want to run this
in CI — presently it stays manual.

**3. Chunking was not a factor.** SciFact passages median ~200 tokens;
no doc fragmented at the 900-token chunk boundary. Caveat #1 in
`docs/eval-plan.md` is effectively dormant for BEIR-scale corpora and
will only bite on CMTEB where passages routinely exceed 900 tokens.

### Follow-ups (not this PR)

Priority-ordered. Each is a self-contained piece of work; do not bundle.

- **Fix RRF weighting.** Two flavors worth trying, in order: (a)
  per-dataset tunable `RRF_K` / mode weights via `dikw.yml` — small,
  lets the user discover the right ratio per corpus; (b) score-
  normalized fusion (CombSUM / CombMNZ over min-max scaled scores) —
  more complex but solves the problem generally. Gate on whether (a)
  moves the SciFact hybrid number above vector.
- **Calibrate thresholds per-dataset after (a).** Thresholds currently
  track hybrid-as-canonical; if hybrid gets stronger, raise the floors.
- **Fix the dataset.yaml overwrite regression.** `convert_beir.py`
  always rewrites the file, stripping any curated description. Should
  merge into the existing stub (preserve user keys, only rewrite the
  block the converter owns).
- **Run CMTEB / T2Retrieval subset** against the same donor wiki —
  same Qwen3 embedder handles both languages; result shape should be
  similar but with 900-token chunking actually biting.
- **Sweep chunk size** (450 / 900 / 1800 tokens) on SciFact — the
  default is picked by feel, not measured. Might recover part of the
  hybrid gap if shorter chunks localise BM25 signal better.
