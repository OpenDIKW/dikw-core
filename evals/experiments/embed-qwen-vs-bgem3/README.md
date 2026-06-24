# Embedding comparison — Qwen3-Embedding-0.6B vs bge-m3

Horizontal **retrieval** comparison (deterministic for a fixed embedder +
corpus, so **1 run/arm**). Both arms embed via Gitee AI at 1024-dim, cosine,
normalized; the LLM is unused in retrieval mode (`llm_api_key_env` is a required
field, so `spec.yaml` points it at a present var).

## Arms

| arm | model | endpoint |
| --- | --- | --- |
| `qwen3-0.6b` | `Qwen3-Embedding-0.6B` | `https://ai.gitee.com/v1` |
| `bge-m3` | `bge-m3` | `https://ai.gitee.com/v1` |

Dataset: `cmteb-t2-subset` (packaged). Metrics are direction-aware (`^` higher
is better). `*` marks the best arm per row.

## Result

| metric | qwen3-0.6b | bge-m3 |
| --- | --- | --- |
| hit@3 | **0.987** | 0.987 |
| hit@10 | **0.987** | 0.987 |
| mrr | **0.979** | 0.975 |
| ndcg@10 | **0.946** | 0.941 |
| recall@100 | 0.988 | **0.990** |

Raw numbers in `comparison.json` + per-arm `qwen3-0.6b.json` / `bge-m3.json`.

## Read

Effectively a **tie**. Qwen3-Embedding-0.6B edges the ranking-quality metrics
(mrr, ndcg@10) and hit-rate ties at @3/@10; bge-m3 takes recall@100 by a hair
(0.990 vs 0.988). Given Qwen3-0.6B is the smaller model, it is the marginally
better cost/quality pick on this dataset — but the gaps are within noise for a
single deterministic pass, so treat this as calibration, not a verdict.

## Reproduce

```bash
uv run --env-file .env python evals/tools/compare_models.py compare \
    --spec evals/experiments/embed-qwen-vs-bgem3/spec.yaml \
    --exp evals/experiments/embed-qwen-vs-bgem3
```

Needs `GITEE_API_KEY` in `.env`.
