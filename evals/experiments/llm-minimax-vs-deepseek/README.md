# LLM comparison — MiniMax-M3 vs DeepSeek-V4-Pro

Horizontal **synth** comparison: same dataset, same embedder, two synth LLMs.
Synth output is non-deterministic, so **n=3 runs/arm** + a Welch t-test of each
metric against the baseline (first arm = `minimax-m3`). The embedder is held
fixed at `Qwen3-Embedding-0.6B`@1024 (Gitee) so only the LLM varies.

## Arms

| arm | model (protocol) | endpoint |
| --- | --- | --- |
| `minimax-m3` (baseline) | `MiniMax-M3` (`anthropic_compat`) | `https://api.minimaxi.com/anthropic` |
| `deepseek-v4-pro` | `deepseek-v4-pro` (`anthropic_compat`) | `https://api.deepseek.com/anthropic` |

Dataset: `mvp` (packaged). Each arm resolves its own key (`MINIMAX_API_KEY` /
`DEEPSEEK_API_KEY`) via its `provider.llm_api_key_env`.

### Neutral judge

All judge legs (grounding / entailment / category / wikilink / atomicity) run on
a single **third-party** model — `openai_codex` (`gpt-5.5`) — instead of each arm
grading its own output, removing the self-evaluation bias that otherwise makes
the judge dims non-comparable across arms. The judge reads its OAuth token from
`<repo>/.dikw/auth.json` (`dikw auth import openai-codex`); it needs no env key.
Judging is capped at `--judge-sample 25` items per leg (baseline-comparable, and
it bounds the slow reasoning judge's wall-clock).

## Result

Direction-aware (`^` higher better, `v` lower better); `*` = best per row;
`p` = Welch vs `minimax-m3`. mean±std over n=3.

| metric | minimax-m3 | deepseek-v4-pro |
| --- | --- | --- |
| judge/grounding | **4.812**±0.037 | 4.778±0.127 (p=0.69) |
| judge/clarity | **4.534**±0.204 | 4.528±0.127 (p=0.97) |
| judge/completeness | **4.214**±0.185 | 4.111±0.173 (p=0.52) |
| judge/atomicity | 5.000 | 5.000 (p=1.00) |
| synth/fact_grounding_ratio | **0.678**±0.036 | 0.651±0.041 (p=0.43) |
| synth/fact_entailment_ratio | 0.724±0.046 | **0.787**±0.050 (p=0.18) |
| synth/expected_coverage | 0.185±0.064 | **0.296**±0.064 (p=0.10) |
| synth/source_chunk_coverage | 0.698±0.027 | **0.746**±0.027 (p=0.10) |
| synth/semantic_atomicity_ratio | 0.932±0.026 | **0.958**±0.042 (p=0.41) |
| synth/wikilink_resolved_ratio | 0.347±0.284 | **0.760**±0.212 (p=0.12) |
| synth/page_density | **0.587**±0.027 | 0.571±0.000 (p=0.42) |
| synth/atomicity_score · language_fidelity · wikilink_correctness_ratio | 1.000 | 1.000 (p=1.00) |
| synth/duplicate_ratio_max · fallback_ratio_max · slug_merge_ratio_max | 0.000 | 0.000 (p=1.00) |

Full matrix in `comparison.json`; per-arm raw runs in `minimax-m3.json` /
`deepseek-v4-pro.json`.

## Read

**At n=3 no metric reaches p<0.05 — the two models are statistically
indistinguishable on every dimension.** Both are flawless on the deterministic
structural metrics (atomicity, language fidelity, wikilink correctness; zero
duplicate / fallback / slug-merge).

Directionally there is a consistent split worth a larger-n follow-up:

- **MiniMax-M3** edges the judge-graded *quality* dims (grounding, completeness)
  and `fact_grounding_ratio` / `page_density`.
- **DeepSeek-V4-Pro** edges the *coverage / recall* dims — `expected_coverage`
  (0.296 vs 0.185), `source_chunk_coverage`, `fact_entailment_ratio`, and
  `wikilink_resolved_ratio` (0.760 vs 0.347, though with large variance).

The two near-significant gaps (`expected_coverage`, `source_chunk_coverage`,
both p≈0.10) favour DeepSeek. Confirming any of these directions needs n≥8.

## Reproduce

```bash
uv run --env-file .env python evals/tools/compare_models.py compare-synth \
    --spec evals/experiments/llm-minimax-vs-deepseek/spec.yaml \
    --exp evals/experiments/llm-minimax-vs-deepseek \
    --judge-sample 25
```

Needs `MINIMAX_API_KEY` + `DEEPSEEK_API_KEY` + `GITEE_API_KEY` in `.env`, and a
codex token at `<repo>/.dikw/auth.json` for the neutral judge. `judge: true` and
`runs: 3` come from `spec.yaml`; `--judge-sample 25` is **required** — without it
the codex judge runs over every page/claim and a single judged run blows past
the harness per-run ceiling. Synth is non-deterministic, so exact numbers shift
run-to-run; the committed JSON is one n=3 sample.
