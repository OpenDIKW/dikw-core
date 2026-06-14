# elon-musk A/B raw runs (synth SP restructure)

`run_elon.py` is the **byte-exact driver** that produced `baseline-*.json` /
`intervention-*.json` — committed here so this experiment stays self-contained
(its module docstring still shows the author's original scratch-path
invocation; run it from this committed location instead).

How each run was produced:

```bash
# arm A from a main-worktree checkout, arm B from the PR branch (@f684af5):
uv run --env-file .env python evals/experiments/synth-sp-restructure-codex/elon/run_elon.py \
    --base <scratch-base> --out <arm>-<n>.json
```

- `<scratch-base>`: a dikw base whose `dikw.yml` uses `openai_codex`
  (gpt-5.5), Qwen3-Embedding-0.6B@1024 (Gitee), `llm_max_tokens_synth: 8192`,
  default `entity/concept/note` taxonomy, **default storage path** (the
  driver's `wipe()` only clears `knowledge/`, `trash/`, and
  `.dikw/index.sqlite*`), seeded with the 1500-line `elon-musk.md` subset under
  `sources/`.
- Each run: wipe K-state → `api.ingest` → `api.synthesize(verify=True,
  judge=True)` (grounding n=25) → whole-vault `api.lint` → JSON dump.
- `wikilink_resolved_ratio` here is **driver-computed**: `(on-disk link total
  − SynthReport.unresolved_wikilinks) / on-disk total`. It is not the eval
  package's metric of the same name — compare within this directory only.

No diagnostic broken-link leg this round: the resolved ratio **improved** on
both arms (baseline 0.895 → intervention 0.962), so there was no
resolved-ratio drop to audit (contrast the PR2 entry, where a single-source
forward-link drop needed `classify_broken.py`). The one metric that read
negative — `grounding_entailment_ratio` (n=25 judge) — is judge noise; the
ship verdict rests on the stable hermetic `mvp` grounding measures, not the
elon judge. Interpretation lives in `evals/BASELINES.md` (2026-06-14 entry).
