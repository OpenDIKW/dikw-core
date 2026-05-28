# evals

Retrieval-quality evaluation for `dikw-core`. Datasets live here as a
filesystem convention; `dikw client eval` runs any of them.

## The three-file contract

Any subdirectory of `evals/datasets/` with these three things is a valid
dataset:

```
evals/datasets/<name>/
├── dataset.yaml       # name, description, thresholds
├── corpus/            # *.md — the docs to ingest
└── queries.yaml       # {q, expect_any: [doc_stem, …]} pairs
```

Missing any piece → `dikw client eval` prints a specific error and exits 2.

### `dataset.yaml`

```yaml
name: mvp
description: short human-readable summary
thresholds:
  hit_at_3: 0.80        # optional; runner only asserts on metrics you list
  hit_at_10: 0.80
  mrr: 0.60
```

Supported metric keys: `hit_at_3`, `hit_at_10`, `mrr`. Unknown keys reject
the dataset. Omitting `thresholds:` entirely is fine — the runner still
computes metrics but `report.passed` is always True (useful for
exploratory datasets you haven't calibrated yet).

### `corpus/`

Any `*.md` under this directory is ingested; subdirectories are fine. File stem (filename without extension) becomes the doc's
identity in `expect_any`.

### `queries.yaml`

```yaml
queries:
  - q: "What does Karpathy mean by deterministic scoping?"
    expect_any: [karpathy-gist, design]   # hit if any stem is in top-k
  - q: "How do I init a wiki?"
    expect_any: [getting-started, README]
```

Semantics: a query is a "hit at k" if *any* stem in `expect_any` appears
in the top-k retrieval result. Paraphrases often live in multiple docs,
and requiring *all* stems would be artificially punitive.

## Usage

```bash
uv run dikw client eval                         # run every packaged dataset
uv run dikw client eval --dataset mvp           # run by name (under evals/datasets/)
uv run dikw client eval --dataset ./my-corpus   # run an arbitrary directory
```

Exit codes: `0` all passed, `1` any threshold failed, `2` bad spec /
dataset not found / no datasets to run.

### Real-vector mode

Default is hermetic — `FakeEmbeddings` (deterministic bag-of-words, 64
dim, <1s) with no network or API keys. For real-vector evaluation
against a configured provider, point at a wiki:

```bash
uv run --env-file .env dikw client eval --dataset mvp --embedder provider --path ./my-wiki
```

The runner reads `provider.embedding_*` from the wiki's `dikw.yml` and
uses `build_embedder(cfg.provider)`. Gotchas around batch size,
dimensions, and dim-locking apply — see [`../docs/providers.md`](../docs/providers.md).

## Adding a dataset

Three steps:

1. `mkdir -p evals/datasets/<name>/{corpus,}`
2. Drop markdown files into `corpus/`; write `queries.yaml` with 5-20
   Q/A pairs (4 exact-fact + 3 paraphrase + 3 synthesis is a reasonable
   mix for retrieval).
3. First run: `uv run dikw client eval --dataset <name>` — skip thresholds in
   `dataset.yaml` or leave them loose. Observe the numbers, then edit
   `dataset.yaml` to leave ~1-2 queries of slack below the observed
   values so corpus tweaks don't flake the gate while a real regression
   still fails it.

## Public benchmarks

Two converters under [`tools/`](./tools/) materialise public bundles
into the three-file shape above. Once the directory exists,
`dikw client eval` picks it up automatically — there is no runtime format
plugin.

### BEIR (English) — e.g., SciFact

Bundle source: <https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/>

```bash
curl -L https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip \
    -o /tmp/scifact.zip
unzip /tmp/scifact.zip -d /tmp/scifact-src/
uv run python evals/tools/convert_beir.py \
    --source /tmp/scifact-src/scifact/ \
    --out evals/datasets/scifact/ \
    --baseline-bm25-ndcg10 0.665   # BEIR paper baseline; printed in dataset.yaml
```

The committed `evals/datasets/scifact/dataset.yaml` carries the
`published_baselines` block and the regen command; `corpus/` and
`queries.yaml` are gitignored — re-run the converter locally.

### CMTEB (Chinese) — e.g., T2Retrieval subset

CMTEB corpora are huge (T2Retrieval ~2.3M passages); the converter
adds **stratified sampling** that always preserves every passage
referenced by a positive qrel:

```bash
huggingface-cli download C-MTEB/T2Retrieval --repo-type dataset \
    --local-dir /tmp/t2-src/
# Convert any parquet shards to BEIR-shape JSONL first if needed
# (one-liner with pandas.to_json).
uv run python evals/tools/convert_cmteb.py \
    --source /tmp/t2-src/ \
    --out evals/datasets/cmteb-t2-subset/ \
    --sample-size 5000 \
    --random-seed 42
```

The seed and source totals land in the generated `dataset.yaml`'s
`_sampling` block so re-runs against the same source bundle reproduce
the same sample.

### Running with real embeddings + ablation

After conversion, run all three retrieval modes for an apples-to-apples
comparison against published BM25 baselines:

```bash
uv run --env-file scratch-bench-base/.env \
    dikw client eval --dataset scifact --embedder provider \
    --path ./scratch-bench-base --retrieval all
```

See [`docs/providers.md`](../docs/providers.md#public-benchmark-calibration-with-gitee-ai)
for the donor-wiki setup against Gitee AI (the cheapest currently-tested
embedder for benchmark-scale work).

### Tuning RRF for your corpus

`dikw client eval --retrieval all --dump-raw path.jsonl` + `evals/tools/sweep_rrf.py`
re-fuses the same ranked lists at arbitrary `(rrf_k, bm25_weight,
vector_weight)` combinations offline — no re-embedding, milliseconds
per sweep. Pin the winning row into your wiki's `dikw.yml`:

```yaml
retrieval:
  rrf_k: 60
  bm25_weight: 0.5          # raise for keyword-heavy corpora
  vector_weight: 1.0
  cjk_tokenizer: jieba      # required for CJK — see providers.md gotcha #7
```

See [`docs/providers.md`](../docs/providers.md#tuning-rrf-weights-for-your-corpus)
for the step-by-step and [`BASELINES.md`](./BASELINES.md) for the
SciFact sweep that picked the shipped defaults. The CMTEB v1 baseline
lists the CJK tokenizer gap; re-runs under `cjk_tokenizer: jieba` are
expected to lift BM25 from 0.03 toward the published 0.5 range.

### Comparability caveats

These numbers are **calibration**, not exact reproduction. See
[`docs/eval-plan.md`](../docs/eval-plan.md#公开-benchmark-校准) for
the full list (chunking at 900 tokens, FTS5 vs Anserini BM25, RRF
weights tuned per-corpus, embedding dimension choice). The useful
signal is the *trend* — does dikw's bm25 land near published BM25,
does hybrid actually win — not the third decimal place.

## See also

- [`docs/eval-plan.md`](../docs/eval-plan.md) — why retrieval-only Phase
  A, when to revisit LLM-as-judge.
- [`docs/providers.md`](../docs/providers.md) — per-vendor config for
  `--embedder provider` mode.
- `src/dikw_core/eval/` — the runner, metrics, and dataset loader.
