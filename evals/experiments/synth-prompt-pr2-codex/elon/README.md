# elon-musk A/B raw runs (synth prompt PR2)

`run_elon.py` is the **byte-exact driver** that produced `baseline-*.json` /
`intervention-*.json` / `diag-1.json` — content-identical to the PR1 copy
(working-tree line endings may differ), committed again so this experiment
stays self-contained (its module docstring still shows the author's original
scratch-path invocation; run it from this committed location instead).

How each run was produced:

```bash
# arm A from a main-worktree checkout, arm B from the PR branch (@7df3de5):
uv run --env-file .env python evals/experiments/synth-prompt-pr2-codex/elon/run_elon.py \
    --base <scratch-base> --out <arm>-<n>.json
```

- `<scratch-base>`: a dikw base whose `dikw.yml` uses `openai_codex`
  (gpt-5.5), Qwen3-Embedding-0.6B@1024 (Gitee), `llm_max_tokens_synth: 8192`,
  default `entity/concept/note` taxonomy, **default storage path** (the
  driver's `wipe()` only clears `.dikw/index.sqlite*`), seeded with the
  1500-line `elon-musk.md` subset under `sources/`.
- Each run: wipe K-state → `api.ingest` → `api.synthesize(verify=True,
  judge=True)` (grounding n=25) → whole-vault `api.lint` → JSON dump.
- `wikilink_resolved_ratio` here is **driver-computed**: `(on-disk link total
  − SynthReport.unresolved_wikilinks) / on-disk total`. It is not the eval
  package's metric of the same name — compare within this directory only.

The diagnostic leg (the resolved-ratio drop investigation):

- `post_elon_diag.py` ran immediately after `diag-1.json`'s run, before any
  base reset: dumps every `broken_wikilink` lint issue to
  `diag-1-broken.json` and snapshots the vault (snapshot not committed; the
  dump is the audit artifact).
- `classify_broken.py` classifies each dumped target against the snapshot
  using the engine's own normalize rules
  (`domains/knowledge/links._normalize_for_match`): collision-refusal vs
  near-miss title vs rule-3(c) deliberate forward link. PR2 verdict:
  **22/22 forward links, 0 collisions, 0 near-misses.**

Interpretation lives in `evals/BASELINES.md` (2026-06-12 PR2 entry).
