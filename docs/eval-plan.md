# Eval plan

Scope: how `dikw-core` measures whether answers are *good*. Captures the
tradeoff between the hand-rolled retrieval gate we just shipped and the
LLM-as-judge frameworks (RAGAS, TruLens, ARES) we might adopt next, so
the next iteration doesn't re-litigate the question.

Status: **current approach is Phase-A retrieval metrics only**. Decision
revisited when the triggers at the bottom fire.

## What we measure today

Eval is a first-class CLI subcommand — `dikw client eval` — driven by the
runner at `src/dikw_core/eval/runner.py`. It ingests a dataset's corpus
into a temp base with deterministic `FakeEmbeddings`, runs the queries
through `HybridSearcher`, and compares aggregate `hit@3`, `hit@10`, and
`MRR` against the dataset's own thresholds.

The MVP dogfood dataset (project docs + Karpathy essays + 10 queries)
lives at `evals/datasets/mvp/` with its own `dataset.yaml` specifying
the thresholds. The full three-file contract ("how to add a dataset")
is in [`evals/README.md`](../evals/README.md). The pytest gate
`tests/test_retrieval_quality.py` is now a ~10-line wrapper over the
same runner, so the CLI and the gate can never drift.

- **What's covered.** Retrieval (I layer): chunking + RRF fusion + storage
  lookup. Catches: wrong chunk boundaries, broken vec/FTS wiring, RRF bugs,
  storage-adapter regressions.
- **What's not.** Generation is not measured. The remaining engine-internal
  LLM leg (K-layer synth — W layer is hand-written, no LLM authoring
  path) gets partial coverage via
  `dikw client eval --eval synth` (added 2026-05-12; seven quantified metrics —
  `fact_grounding_ratio`, `atomicity_score`, `duplicate_ratio_max`,
  `wikilink_resolved_ratio`, `expected_coverage`, `language_fidelity`,
  `page_density`). Agent-side answer synthesis (which lives outside
  dikw-core entirely now that `query` is removed) is the caller's
  responsibility to evaluate.

## Options for generation-side eval

### Homegrown golden answers

Author ~20 Q/A pairs with **reference answers**, run `api.retrieve` +
an LLM call wired into the eval harness, compare output to reference
via string match or embedding cosine. The harness owns the LLM call;
the engine side stays retrieval-only.

- **Pros:** deterministic scoring, no extra deps, cheap to run.
- **Cons:** string/cosine match is brittle — paraphrased correct answers
  fail. Reference answers drift as the corpus evolves. Brittleness
  tempts you to either relax thresholds (useless) or constantly re-author
  references (expensive).

### RAGAS (LLM-as-judge)

RAGAS runs `faithfulness` (is the answer grounded in retrieved context?)
and `answer_relevancy` (does the answer address the question?) via a
separate LLM call per metric per query.

- **Pros:** robust to paraphrase; covers both retrieval and generation
  signals from one harness; decouples metric code from reference
  answers.
- **Cons:** ~2-4 extra LLM calls per query during eval — not free at 20+
  queries with MiniMax costs. LLM judges drift when the judge model is
  upgraded. Flaky at low sample counts.

### TruLens / ARES

Similar shape to RAGAS with different tradeoffs (TruLens is
heavier-weight infra; ARES uses synthetic-data training for the judge).
Neither clearly dominates RAGAS for our stage.

## Recommendation

Stay on **Phase-A retrieval metrics only** until one of the triggers
below fires. Rationale:

1. **Retrieval dominates answer quality at alpha.** If the right
   chunks aren't in context, the judge can't rescue the answer. Fixing
   retrieval first is strictly higher ROI.
2. **Deterministic > noisy.** Phase A is hermetic; RAGAS is LLM-dependent
   and adds flakiness + spend. At 10 queries, LLM-judge variance would
   drown the signal.
3. **W-layer is the real differentiator.** Wisdom is now a first-class
   retrieval layer (0.3.0): hits arrive tagged `Hit.layer == "wisdom"`
   so an agent can group / weight / cite them separately. The interesting
   generation-side metric is "does the agent's answer cite the wisdom
   pages alongside the knowledge pages for the same question?" — a bespoke
   check catches that more cleanly than generic faithfulness scores.

## Acceptance gates for K-layer and Retrieval changes

Independent of "do we have evals?" — once a change touches K-layer
synth/lint/knowledge schema or retrieval config, **the PR description must
cite an `evals/BASELINES.md` entry** that demonstrates either signal
or non-destructiveness. Two gates:

1. **K-layer changes** (`src/dikw_core/domains/knowledge/`,
   `src/dikw_core/api_synth.py`, the LLM authoring prompts under
   `src/dikw_core/prompts/` — `synthesize.md`, `lint_fix_orphan_merge.md`,
   `lint_fix_broken_wikilink_grounded.md` — knowledge page schema in
   `schemas.py`): run a real-data baseline against
   `~/Project/opendikw/dikw-data/datasets/markdown-books/elon-musk.md`
   (1500-line subset is the working size — the full text exposes a
   codex SSE keepalive timeout bug, see 2026-05-08 entry). Use the
   `openai_codex` provider so the LLM cost is zero. Add a BASELINES.md
   section with: source size, group/page/chunk counts, lint outcomes
   broken down by issue kind, and (for atomicity tweaks) a 5+5 sample
   judgement of TP/FP rates.

   **Also run** `dikw client eval --dataset mvp --eval synth --pretty` (and
   `--judge --judge-sample 5` once an LLM budget allows the soft layer)
   to capture the seven quantified K-layer metrics — `fact_grounding_ratio`,
   `atomicity_score`, `duplicate_ratio_max`, `wikilink_resolved_ratio`,
   `expected_coverage`, `language_fidelity`, and the informational
   `page_density`. This replaces the manual 5+5 TP/FP sampling for
   ongoing changes once the mvp thresholds are calibrated; the first
   real-LLM run that lands the calibrated numbers is itself the
   baseline entry.

   Alongside the gated five, the runner emits **informational diagnostics**
   the five gated metrics are blind to: `source_chunk_coverage`
   (under-generation — source chunks that no page claim lands on),
   `fallback_ratio_max` (taxonomy miscalibration — share of pages filed under
   the fallback category), and `slug_merge_ratio_max` (over-generation — the
   fraction of fan-out pages collapsed by slug dedup). The `_max` suffix marks
   the two lower-is-better ones for the direction convention. The LLM judge now
   reports a deterministic bootstrap 95% CI per dimension so a noisy small-sample
   mean isn't mistaken for a real move.

   **`fact_entailment_ratio` — the LLM grounding leg the cosine is blind to.**
   `fact_grounding_ratio` reduces to a cosine, so a fabricated specific ("GPT-4
   is 4x faster") and a supported gist ("GPT-4 is faster") land in the same band.
   The entailment judge pairs each claim with its nearest source chunk (reusing
   the grounding argmax — no re-embedding) and asks an LLM whether the evidence
   *entails* it (`yes`/`partial`/`no` → `1.0`/`0.5`/`0.0`), catching invented
   numbers/dates/ratios, superlatives, causal overreach, and contradictions. It
   carries its own bootstrap 95% CI on `entailment_summary`, mirrors the ratio
   into `informational` for display / the A/B harness, and is **opt-in**: it
   runs only when `--judge` is set **and** the dataset declares
   `judge.entailment_grounding_enabled: true`, so it costs nothing by default.
   It is also the first judge dimension promoted to a **conditional gate**: a
   dataset may declare a `synth/fact_entailment_ratio` floor (mvp gates `0.55`,
   calibrated 2026-06-05 to observed `0.775`, CI `[0.65, 0.90]`, n=20), and the
   runner enforces it **only when the judge actually ran** — a non-judge run
   (hermetic CI, plain `--eval synth`) drops the threshold rather than recording
   a spurious miss. So the entailment gate bites on real-LLM acceptance runs
   only; the sample size needed to trust the number is settled by the power
   analysis below.

   **`category_correctness_ratio` — is each page filed under the right
   category?** `category_distribution` / `fallback_ratio_max` see *where* pages
   landed but not whether that filing is correct. The taxonomy judge re-picks
   the best category independently from the page body and the closed declared
   set (fallback included), then scores synth's actual choice: exact `1.0`, a
   judge-acknowledged co-equal `0.5`, wrong `0.0`. The closed set is enforced at
   parse time — a verdict naming an undeclared category is rejected, never
   silently re-filed, the same Karpathy discipline synth itself follows.
   Informational (never gated), bootstrap 95% CI on `category_summary`, and
   **opt-in** under `judge.category_correctness_enabled: true` (+ `--judge`), so
   `$0` by default.

   **`wikilink_correctness_ratio` — does each resolved link point at the right
   page?** `wikilink_resolved_ratio` counts resolution, and the fuzzy resolver
   (NFKC + casefold + punctuation strip + plural stem) deliberately absorbs
   surface variation — so a wrong-referent link (`[[Mercury]]` in a planetary
   context resolving to the chemical-element page) makes the resolved ratio
   look *better* while silently corrupting the graph that feeds graph-leg
   retrieval. The judge reads each resolved page→page link in its body context
   (the `[[wikilink]]` as written stays visible) next to the target page the
   engine resolved it to — the `links` table is the deterministic truth, fuzzy
   results included — and answers `yes` (right referent; resolver-absorbed
   surface variants still count) / `partial` (related but imprecise — a
   broader/narrower/sibling page) / `no` (a homonym or different entity) →
   `1.0`/`0.5`/`0.0`. Informational (never gated), bootstrap 95% CI on
   `wikilink_summary`, **opt-in** under `judge.wikilink_correctness_enabled:
   true` (+ `--judge`), so `$0` by default.

   **`semantic_atomicity_ratio` — does each page develop exactly one
   concept?** `atomicity_score` is a *form* heuristic — body chars, H1/H2
   counts, distinct wikilink targets, tag domains — blind in both directions:
   a short single paragraph stuffed with three unrelated concepts passes
   every count, while a thorough single-concept page can trip the length
   counters. The judge reads each page's title + body alone (atomicity is
   intrinsic to the page; no source text needed) and answers `yes` (one
   concept; passing mentions and `[[wikilink]]` references don't count
   against it) / `partial` (one dominant concept plus a substantively
   developed tangent that deserves its own page) / `no` (multiple distinct
   concepts bolted together) → `1.0`/`0.5`/`0.0`. The prompt's tie-breakers:
   form never decides, and depth on one subject is not a violation.
   Informational (never gated), bootstrap 95% CI on
   `semantic_atomicity_summary`, **opt-in** under
   `judge.semantic_atomicity_enabled: true` (+ `--judge`), so `$0` by
   default.

   **Sizing the judge sample (`--judge-sample auto`).** A judge ratio is only as
   trustworthy as its CI is tight. The real calibrations all cleared the
   ±0.2 half-width target, but category only barely (entailment n=20 → ±0.13,
   category n=8 → ±0.19; wikilink n=16 cleared it trivially — a zero-variance
   1.0 run whose degenerate CI says nothing about discriminative power, see
   the 2026-06-10 BASELINES entry) — riding low score-variance, not a
   sufficient sample. At
   n=8 the worst-case (50/50) half-width is ±0.35, so a higher-variance metric
   would have failed; we want a size that *guarantees* the target regardless of
   variance. A [0,1] ratio's bootstrap 95% CI half-width is at most
   `1.96 * 0.5 / sqrt(n)` (worst case at a 50/50 split), so `n ≥ 25` clears ±0.2
   for *any* score distribution — a dataset-independent bound, which is why a
   multi-corpus empirical sweep can't push it higher. `recommended_judge_sample()`
   returns that `n` clamped to `[5, 50]`, exposed as `dikw client eval
   --judge-sample auto`; smaller datasets are judged in full.

   **Proving an optimization actually helped.** The LLM makes synth
   non-deterministic, so a single before/after eval can't separate a real gain
   from ±0.05 run-to-run noise. `evals/tools/ab_experiment.py` runs the same
   synth eval N times per arm and compares the two arms with a Welch t-test +
   direction-aware ship gate (`p < p_max` **and** `improvement > effect_min`).
   A tuning PR cites its `result.json` shipped/regressed verdict, not a single
   run. The harness is developer tooling (not on any CI gate); the human reads
   the table.

2. **Retrieval config changes** (any field on `RetrievalConfig`,
   anything under `src/dikw_core/domains/info/search.py`): run an
   ablation on at least one packaged dataset (mvp / scifact / cmteb)
   showing nDCG@10 doesn't regress. Note that **graph-leg-style K-layer
   features can't be measured on standard retrieval benchmarks** —
   those datasets ingest as D-layer only and never produce wikilinks
   in the storage `links` table. For those, the gate is "non-destructive
   when off; non-destructive when on with empty links".

The Stage A K-layer fan-out + atomicity-lint baseline (2026-05-08) and
the wikilink graph leg ablation (2026-05-08) are the worked examples.

## Triggers for revisiting

Adopt an LLM-as-judge framework (default: RAGAS) when **any** of:

- Retrieval metrics saturate (hit@10 ≥ 0.95 on a 30+-query set) and
  user-perceived quality still disappoints. The issue is in generation,
  not retrieval.
- The corpus grows past ~50 docs or the query set past ~30 pairs, at
  which point authoring golden answers becomes a bottleneck but judging
  N questions still costs O(N) LLM calls, not O(N²) author-hours.
- We grow `evals/datasets/elon-musk-validation` (or any packaged
  dataset) with a wisdom subdirectory + qrels and want to measure
  retrieval lift from cross-layer wisdom hits — wisdom MRR / hit@k
  alongside the existing knowledge base / source metrics.

Until then: grow the Q/A set, keep Phase A green, don't spin up a
judge harness.

## 公开 benchmark 校准

Phase A also covers comparing dikw's retriever against published BEIR
/ CMTEB baselines via [`evals/tools/convert_{beir,cmteb}.py`](../evals/README.md#public-benchmarks)
+ `dikw client eval --retrieval {bm25,vector,hybrid,all}`. The framing is
**calibration, not reproduction** — five things make exact-number
parity impossible (and not actually useful):

1. **Chunking at 900 tokens.** dikw runs every ingested doc through
   `info/chunk.py` before embedding / indexing. Most BEIR passages
   are 100–500 tokens (no fragmentation), but longer CMTEB passages
   split into multiple chunks; the doc-level hit@k still works
   correctly (chunks of the same doc share a stem) but the underlying
   index shape diverges from "passage retrieval as published".
2. **FTS5 ≠ Anserini BM25.** Our `bm25` mode goes through SQLite's
   FTS5 `bm25()` — same family of formulas, different IDF / length-norm
   constants and tokenizer. Treat ±0.10 nDCG@10 vs the published
   number as in-band; larger gaps suggest a real bug, not algorithm
   choice. **CJK corpora also need `retrieval.cjk_tokenizer: jieba`**
   in `dikw.yml` — the default `unicode61` tokenizer splits Chinese
   per-character, collapsing BM25 to single-char IDF (see CMTEB v1
   baseline). Without it the `bm25` row lands near 0.03 nDCG@10 on
   Chinese regardless of fusion choice.
3. **RRF weighted on the SciFact 2026-04-23 sweep.** `k=60` from the
   original RRF paper, per-leg weights `(bm25=0.3, vector=1.5)` picked
   because equal-weight left hybrid 0.037 nDCG@10 behind vector-only
   on BEIR/SciFact — dragged down by a ~0.10-nDCG-weaker BM25 leg. The
   sweep + tuning path: edit the `retrieval:` weights in `dikw.yml`
   and re-run `dikw client eval --retrieval all` to compare
   (`evals/tools/sweep_rrf.py` re-fuses cached rankings offline, but the
   raw dump it needs is no longer emitted by the client CLI). Keyword-heavy
   corpora (code, rare identifiers) likely want `bm25_weight ≥ 1.0` —
   override via `retrieval:` block in `dikw.yml`. Full sweep table:
   [`evals/BASELINES.md`](../evals/BASELINES.md).
4. **Embedding dim choice.** The benchmark stubs default to
   Qwen3-Embedding-0.6B at 1024-dim (native) for cost. Premium runs
   on Qwen3-Embedding-8B (1024 matryoshka or 4096 native) shift dense
   + hybrid numbers; pin the model + dim in `dataset.yaml`'s comments
   so re-runs reproduce.
5. **CMTEB sample sizing.** The Chinese benchmarks ship at 1M+
   passages; we sample down to ~5K. Fewer distractors → higher
   absolute metrics than the published full-corpus numbers. The
   sampling preserves all relevant docs (recall is honest), but
   precision-style metrics like hit@k will read higher than they
   would at full scale.

The useful signal across all five caveats is the **trend**:

* Does `bm25` land near published BM25 (within ±0.10)?
* Does `hybrid` beat both `bm25` and `vector` on the same chunking,
  on the same dataset?
* Does the same embedder do equally well on English (BEIR) and
  Chinese (CMTEB)?

If the answer to any of those is "no", the gap is informative — go
look at the FTS leg, the RRF weighting, or the embedder. If the
absolute number doesn't match the BEIR paper, that is *expected* and
not by itself a bug.
