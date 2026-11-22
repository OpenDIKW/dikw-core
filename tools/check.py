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

import json
import subprocess
import sys
from pathlib import Path

# (label, command) in the same order CI's lint-type-test job runs them.
STAGES: list[tuple[str, list[str]]] = [
    (
        "harness-doctor",
        ["uv", "run", "--offline", "--no-project", "python", "scripts/harness-doctor", "check"],
    ),
    ("ruff", ["uv", "run", "ruff", "check", "."]),
    ("mypy", ["uv", "run", "mypy", "src"]),
    ("pytest (fast)", ["uv", "run", "pytest", "-m", "not slow and not perf"]),
]


def _capture_fingerprint(repo_root: Path) -> tuple[str, str] | None:
    """Best-effort pre-run tree fingerprint via ``tools/verify_hook.py``."""
    hook_script = Path(__file__).resolve().parent / "verify_hook.py"
    if not hook_script.is_file():
        return None
    proc = subprocess.run(
        [sys.executable, str(hook_script), "--repo-root", str(repo_root), "fingerprint"],
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return None
    try:
        data = json.loads(proc.stdout)
        return str(data["head_sha"]), str(data["tree_fingerprint"])
    except (json.JSONDecodeError, KeyError, TypeError):
        return None


def _record_verify_stamp(repo_root: Path, before: tuple[str, str] | None) -> bool:
    """Stamp only when pre-run fingerprint was captured and still matches."""
    hook_script = Path(__file__).resolve().parent / "verify_hook.py"
    if not hook_script.is_file():
        return True
    if before is None:
        print(
            "\n[WARN] verify-hook: could not capture pre-check fingerprint; stamp NOT recorded.",
            file=sys.stderr,
        )
        return False
    cmd = [
        sys.executable,
        str(hook_script),
        "--repo-root",
        str(repo_root),
        "stamp",
        "--expect-head",
        before[0],
        "--expect-fingerprint",
        before[1],
    ]
    for label, _ in STAGES:
        cmd.extend(["--stage", label])
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        detail = proc.stderr.strip() or f"verify-hook stamp exited with code {proc.returncode}"
        print(f"\n[WARN] {detail}", file=sys.stderr)
        return False
    return True


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    before = _capture_fingerprint(repo_root)
    for label, cmd in STAGES:
        print(f"\n=== {label}: {' '.join(cmd)} ===", flush=True)
        result = subprocess.run(cmd, cwd=repo_root)
        if result.returncode != 0:
            print(f"\n[FAIL] {label} failed (exit {result.returncode}). Fix it before committing.")
            return result.returncode
    stamped = _record_verify_stamp(repo_root, before)
    if stamped:
        print("\n[OK] All checks passed (harness-doctor + ruff + mypy + fast pytest).")
    else:
        print(
            "\n[OK] All checks passed, but stamp was not recorded — "
            "re-run `tools/check.py` before committing to refresh the stamp."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
