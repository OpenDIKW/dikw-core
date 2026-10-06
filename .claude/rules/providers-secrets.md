---
paths:
  - "src/dikw_core/providers/**"
  - "src/dikw_core/config.py"
  - "src/dikw_core/auth_cli.py"
  - ".env.example"
  - "docs/providers.md"
  - "tests/test_{provider,codex_auth,auth_cli,config}*.py"
---

# Providers and secrets

Read this rule before you change a provider, `ProviderConfig`, key handling, or `.env.example`.

- Secrets: the LLM and embedding key env-var **names** are config-driven — `ProviderConfig` carries two **required** fields, `llm_api_key_env` and `embedding_api_key_env`, naming the var each leg reads from env. There is **no** hardcoded key-var name and **no** fallback: `anthropic_compat`/`openai_compat` read exactly the named var (`build_llm`/`build_embedder`/`build_multimodal_embedder` thread it from cfg), and a named-but-unset var fails loud at call time with the var's name. Names are vendor-canonical by convention (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `DEEPSEEK_API_KEY`, `MINIMAX_API_KEY`, `GITEE_API_KEY`, …) so `.env` holds every vendor's key and each base picks its own; there is **no** dikw-invented `DIKW_EMBEDDING_API_KEY` name. LLM/embedding key separation is achieved by *naming distinct vars* (point both fields at one var to share a key). **`.env` is for secrets only**; non-secret config (URLs, models, dims, batch, display labels) lives in `dikw.yml`. Never hardcode or commit; `.env`/`.env.*` are gitignored (except `.env.example`). The `openai_codex` LLM is the exception: it doesn't read an env API key — it manages ChatGPT OAuth tokens in dikw's own per-base store at `<base>/.dikw/auth.json` (separate from codex CLI's `~/.codex/auth.json`, to avoid refresh_token rotation conflicts). Bootstrap with `dikw auth login openai-codex` (device-code flow) or `dikw auth import openai-codex` (one-shot copy from `~/.codex/auth.json`); dikw refreshes the access_token automatically before each call.
