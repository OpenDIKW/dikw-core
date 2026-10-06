# Parser format-slip replay (#294)

`replay_parser.py` reads existing knowledge pages whose provenance includes
`sources/elon-musk.md`, rewraps their bodies and tags as synth responses, and
injects the issue's missing-close, bare-opener, and quoted-instruction shapes.
It compares the parser at `388b828` with the working tree and asserts identical
page paths, titles, bodies, tags, and provenance across the recovered variants.
The input base is read-only; no providers or LLM calls are involved.

```powershell
uv run python evals/experiments/synth-format-slips-294/replay_parser.py `
  --knowledge-dir <base>/knowledge `
  --out evals/experiments/synth-format-slips-294/replay.json
```

`replay.json` records the 2026-10-06 run over 506 exported real pages and the
input-content hash. This measures parser recovery and non-destructiveness on
existing generated content, not fresh LLM output quality or prompt efficacy.
True `max_tokens` and `length` controls still request retry.
