"""Local CI mirror — the deterministic floor that CI runs, in one command.

This is the single in-loop gate referenced by step 3 of the
``dikw-core-delivery-workflow`` skill. Run it before every commit so a red
ruff / mypy / pytest never bounces off CI:

    uv run python tools/check.py

Mirrors the ``lint-type-test`` job in ``.github/workflows/reusable-ci.yml``,
minus ``--cov`` (coverage makes ASGI / CliRunner tests flaky in the local
inner loop on Windows — trust CI coverage instead). Stops at the first
failing stage and exits with its return code.
"""

import subprocess
import sys

# (label, command) in the same order CI's lint-type-test job runs them.
STAGES: list[tuple[str, list[str]]] = [
    ("ruff", ["uv", "run", "ruff", "check", "."]),
    ("mypy", ["uv", "run", "mypy", "src"]),
    ("pytest (fast)", ["uv", "run", "pytest", "-m", "not slow and not perf"]),
]


def main() -> int:
    for label, cmd in STAGES:
        print(f"\n=== {label}: {' '.join(cmd)} ===", flush=True)
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"\n[FAIL] {label} failed (exit {result.returncode}). Fix it before committing.")
            return result.returncode
    print("\n[OK] All checks passed (ruff + mypy + fast pytest).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
