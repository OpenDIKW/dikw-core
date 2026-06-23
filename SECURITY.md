# Security Policy

## Supported versions

`dikw-core` is in **alpha**. Security fixes land on `main` and ship in the next
release; only the **latest published release** on
[PyPI](https://pypi.org/project/dikw-core/) is supported. Please reproduce any
report against the latest release (or `main`) before filing.

| Version            | Supported |
| ------------------ | --------- |
| Latest release     | ✅        |
| Older releases     | ❌        |

## Reporting a vulnerability

**Please do not open a public issue for security problems.**

Report privately via GitHub's **[Security Advisories](https://github.com/OpenDIKW/dikw-core/security/advisories/new)**
("Report a vulnerability") on this repository. That keeps the report confidential
until a fix is available and lets us coordinate a disclosure with you.

When reporting, please include:

- a description of the issue and its impact;
- the affected version (`dikw version`) and deployment shape (SQLite vs Postgres,
  loopback vs networked `dikw serve`, container vs source);
- steps to reproduce, ideally a minimal proof of concept;
- any suggested remediation, if you have one.

We will acknowledge your report, keep you updated on remediation, and credit you
in the release notes unless you prefer to remain anonymous.

## Scope notes

A few things are by design rather than vulnerabilities:

- `dikw serve` on loopback (`127.0.0.1`) runs **without auth** by intent; binding
  to a non-loopback interface is rejected unless `DIKW_SERVER_TOKEN` is set. Treat
  a networked deployment as requiring the token (see [`docs/server.md`](./docs/server.md)).
- Provider API keys are read from the env vars named in `dikw.yml`
  (`llm_api_key_env` / `embedding_api_key_env`); never commit secrets — `.env`
  files are gitignored.

If you are unsure whether something is in scope, report it privately and we will
triage it.
