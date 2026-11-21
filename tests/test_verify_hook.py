"""Tests for ``tools/verify_hook.py`` — commit verification gate & stamp logic.

Covers:
  * command detection (``git commit`` vs non-commit commands, chained commands,
    multiline newline-separated commands, quoted strings)
  * ``[skip-verify: <reason>]`` explicit bypass parsing
  * stamp write / fingerprint invariance across ``git add`` (including renames),
    rejection of mid-run mutations, partial-staging mismatch detection, and
    invalidation on post-check edits or new commits
  * worktree resolution for ``git -C <other-repo> commit``
  * read-only ``check`` CLI subcommand (never mutates escape-hatch state)
  * gate modes: ``off``, ``observe`` (logs without blocking), ``enforce``
    (blocks unverified commits with exit 2, allows after 2 prior blocks in the
    same session so an agent never deadlocks)
  * Claude Code PreToolUse JSON stdin handler (graceful on non-Bash / malformed
    payloads)
"""

from __future__ import annotations

import importlib.util
import io
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def _scrub_git_repo_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent inherited Git hook env vars from redirecting temp-repo calls."""
    for var in (
        "GIT_DIR",
        "GIT_WORK_TREE",
        "GIT_INDEX_FILE",
        "GIT_OBJECT_DIRECTORY",
        "GIT_ALTERNATE_OBJECT_DIRECTORIES",
        "GIT_COMMON_DIR",
        "GIT_PREFIX",
    ):
        monkeypatch.delenv(var, raising=False)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "verify_hook", REPO_ROOT / "tools" / "verify_hook.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["verify_hook"] = mod
    spec.loader.exec_module(mod)
    return mod


def _init_git_repo(tmp_path: Path, name: str = "repo") -> Path:
    repo = tmp_path / name
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo, check=True)
    subprocess.run(["git", "config", "commit.gpgSign", "false"], cwd=repo, check=True)
    subprocess.run(["git", "config", "tag.gpgSign", "false"], cwd=repo, check=True)
    subprocess.run(["git", "config", "core.hooksPath", os.devnull], cwd=repo, check=True)
    (repo / "hello.py").write_text("x = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "hello.py"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "initial"], cwd=repo, check=True)
    return repo


# ---- command detection --------------------------------------------------


@pytest.mark.parametrize(
    "cmd",
    [
        "git commit -m 'feat: add thing'",
        "git add . && git commit -m 'feat: add thing'",
        "git add .\ngit commit -m 'multiline command'",
        "git add .\n\ngit commit -m 'double newline'",
        "git status # inspect changes\ngit commit -m 'after comment'",
        "git -C /some/path commit --amend --no-edit",
        "FOO=1 git commit -m 'msg'",
        "echo ok; git commit -m 'msg'",
    ],
)
def test_is_git_commit_command_matches_real_commits(cmd: str) -> None:
    vh = _load_module()
    assert vh.is_git_commit_command(cmd) is True


@pytest.mark.parametrize(
    "cmd",
    [
        "git status",
        "git log -n 5",
        "git diff --stat",
        "uv run python tools/check.py",
        "echo 'git commit -m fake'",
        "echo 'line1\ngit commit -m quoted'",
        "pytest tests/test_verify_hook.py",
        "git show HEAD",
    ],
)
def test_is_git_commit_command_ignores_non_commits(cmd: str) -> None:
    vh = _load_module()
    assert vh.is_git_commit_command(cmd) is False


def test_extract_skip_reason() -> None:
    vh = _load_module()
    assert (
        vh.extract_skip_reason("git commit -m 'docs: fix typo [skip-verify: comment only]'")
        == "comment only"
    )
    assert vh.extract_skip_reason("git commit -m 'docs: fix typo [skip-verify: ]'") is None
    assert vh.extract_skip_reason("git commit -m 'feat: normal commit'") is None


# ---- fingerprint & stamp lifecycle --------------------------------------


def test_stamp_survives_git_add_and_invalidates_on_edit(tmp_path: Path) -> None:
    vh = _load_module()
    repo = _init_git_repo(tmp_path)

    # Edit tracked file + create untracked file before staging.
    (repo / "hello.py").write_text("x = 2\n", encoding="utf-8")
    (repo / "new_file.py").write_text("y = 3\n", encoding="utf-8")

    stamp = vh.write_stamp(repo, stages=["ruff", "mypy", "pytest"])
    assert stamp is not None
    assert vh.verify_stamp(repo)[0] is True
    assert stamp["stages"] == ["ruff", "mypy", "pytest"]

    # Staging the exact same files must NOT invalidate the stamp.
    subprocess.run(["git", "add", "hello.py", "new_file.py"], cwd=repo, check=True)
    ok, reason = vh.verify_stamp(repo)
    assert ok is True, reason

    # Any post-check edit MUST invalidate the stamp.
    (repo / "hello.py").write_text("x = 999\n", encoding="utf-8")
    ok, reason = vh.verify_stamp(repo)
    assert ok is False
    assert "stale" in reason.lower()


def test_stamp_survives_staged_rename(tmp_path: Path) -> None:
    vh = _load_module()
    repo = _init_git_repo(tmp_path)

    # Rename a tracked file on disk before staging, stamp it, then stage with git add -A.
    (repo / "hello.py").rename(repo / "renamed.py")
    assert vh.write_stamp(repo, stages=["ruff"]) is not None

    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    ok, reason = vh.verify_stamp(repo)
    assert ok is True, reason


def test_stamp_rejects_mid_run_tree_change(tmp_path: Path) -> None:
    vh = _load_module()
    repo = _init_git_repo(tmp_path)
    (repo / "hello.py").write_text("x = 2\n", encoding="utf-8")
    before_head, before_fp = vh.compute_tree_fingerprint(repo)

    # Simulate an edit arriving while checks were in flight.
    (repo / "hello.py").write_text("x = 3\n", encoding="utf-8")
    stamped = vh.write_stamp(
        repo,
        stages=["ruff"],
        expected_head_sha=before_head,
        expected_fingerprint=before_fp,
    )
    assert stamped is None
    assert vh.verify_stamp(repo)[0] is False


def test_partially_staged_file_rejected_on_plain_commit(tmp_path: Path) -> None:
    vh = _load_module()
    repo = _init_git_repo(tmp_path)

    # Stage version 2, then edit working tree to version 3 and run checks on version 3.
    (repo / "hello.py").write_text("x = 2\n", encoding="utf-8")
    subprocess.run(["git", "add", "hello.py"], cwd=repo, check=True)
    (repo / "hello.py").write_text("x = 3\n", encoding="utf-8")
    vh.write_stamp(repo, stages=["ruff", "mypy", "pytest"])

    # Plain `git commit` (including attached `-m"add tests"` with 'a' in the message)
    # would commit the untested staged version 2 -> must fail.
    for cmd in (
        "git commit -m 'partial'",
        'git commit -m"add tests"',
        "git add README.md && git commit -m 'selective add'",
    ):
        d_plain = vh.evaluate_commit_gate(
            repo, command=cmd, mode="enforce", session_id=f"s-partial-{cmd}"
        )
        assert d_plain.allowed is False, cmd
        assert "staged index differs" in d_plain.message

    # `git commit -am ...`, `git commit --gpg-sign -a ...`, and `git add -A && git commit`
    # commit the tested working-tree version 3 -> allowed.
    for cmd in (
        "git commit -am 'all'",
        "git commit --gpg-sign -a -m 'signed all'",
        "git add -A && git commit -m 'full stage then commit'",
    ):
        d_all = vh.evaluate_commit_gate(
            repo, command=cmd, mode="enforce", session_id=f"s-partial-ok-{cmd}"
        )
        assert d_all.allowed is True, cmd


def test_compound_command_with_mutator_or_redirect_before_commit_is_blocked(tmp_path: Path) -> None:
    vh = _load_module()
    repo = _init_git_repo(tmp_path)
    (repo / "hello.py").write_text("x = 2\n", encoding="utf-8")
    vh.write_stamp(repo, stages=["ruff", "mypy", "pytest"])

    for cmd in (
        "ruff format . && git add . && git commit -m 'formatted'",
        "printf 'x = 999\\n' > hello.py && git add . && git commit -m 'redirected'",
    ):
        decision = vh.evaluate_commit_gate(
            repo,
            command=cmd,
            mode="enforce",
            session_id=f"s-mutator-{cmd}",
        )
        assert decision.allowed is False, cmd
        assert "separate calls" in decision.message


def test_stamp_invalidates_on_executable_bit_change(tmp_path: Path) -> None:
    vh = _load_module()
    repo = _init_git_repo(tmp_path)
    script = repo / "hello.py"
    script.write_text("x = 2\n", encoding="utf-8")
    script.chmod(0o644)
    vh.write_stamp(repo, stages=["ruff"])
    assert vh.verify_stamp(repo)[0] is True

    script.chmod(0o755)
    ok, reason = vh.verify_stamp(repo)
    assert ok is False
    assert "stale" in reason.lower()


def test_enforce_block_survives_unwritable_log_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    vh = _load_module()
    repo = _init_git_repo(tmp_path)
    (repo / "hello.py").write_text("x = 2\n", encoding="utf-8")

    # Point observe log path at a directory so opening it for append raises OSError.
    log_dir = repo / ".claude" / "verify-gate-observe.jsonl"
    log_dir.mkdir(parents=True)
    decision = vh.evaluate_commit_gate(
        repo, command="git commit -m 'unverified'", mode="enforce", session_id="s-log-err"
    )
    assert decision.allowed is False
    assert decision.exit_code == 2


def test_claude_pretool_uses_payload_cwd_for_sibling_worktree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    vh = _load_module()
    repo_a = _init_git_repo(tmp_path, "repo_a")
    repo_b = _init_git_repo(tmp_path, "repo_b")
    monkeypatch.setenv("DIKW_VERIFY_GATE_MODE", "enforce")

    # repo_a has a valid stamp; repo_b does NOT.
    (repo_a / "hello.py").write_text("x = 2\n", encoding="utf-8")
    vh.write_stamp(repo_a, stages=["ruff"])
    (repo_b / "hello.py").write_text("x = 9\n", encoding="utf-8")

    payload = json.dumps(
        {
            "session_id": "s-cwd",
            "cwd": str(repo_b),
            "tool_name": "Bash",
            "tool_input": {"command": "git commit -m 'in b'"},
        }
    )
    monkeypatch.setattr(sys, "stdin", io.StringIO(payload))
    assert vh.main(["--repo-root", str(repo_a), "claude-pretool"]) == 2

    # Once repo_b is stamped, committing from cwd=repo_b passes.
    vh.write_stamp(repo_b, stages=["ruff"])
    monkeypatch.setattr(sys, "stdin", io.StringIO(payload))
    assert vh.main(["--repo-root", str(repo_a), "claude-pretool"]) == 0


def test_stamp_invalidates_after_new_commit(tmp_path: Path) -> None:
    vh = _load_module()
    repo = _init_git_repo(tmp_path)
    (repo / "hello.py").write_text("x = 2\n", encoding="utf-8")
    vh.write_stamp(repo, stages=["ruff"])
    subprocess.run(["git", "add", "hello.py"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "second"], cwd=repo, check=True)

    ok, reason = vh.verify_stamp(repo)
    assert ok is False
    assert "head" in reason.lower() or "stale" in reason.lower()


def test_git_dash_c_checks_target_worktree_stamp(tmp_path: Path) -> None:
    vh = _load_module()
    repo_a = _init_git_repo(tmp_path, "repo_a")
    repo_b = _init_git_repo(tmp_path, "repo_b")

    # Repo A is verified, Repo B is NOT verified.
    (repo_a / "hello.py").write_text("x = 2\n", encoding="utf-8")
    vh.write_stamp(repo_a, stages=["ruff"])
    (repo_b / "hello.py").write_text("x = 9\n", encoding="utf-8")

    # Committing in repo_b via `git -C repo_b commit` from repo_a must NOT use repo_a's stamp.
    d_unverified = vh.evaluate_commit_gate(
        repo_a,
        command=f"git -C {shlex.quote(str(repo_b))} commit -m 'in b'",
        mode="enforce",
        session_id="s-wt",
    )
    assert d_unverified.allowed is False

    # Stamping repo_b authorizes committing in repo_b.
    vh.write_stamp(repo_b, stages=["ruff"])
    d_verified = vh.evaluate_commit_gate(
        repo_a,
        command=f"git -C {shlex.quote(str(repo_b))} commit -m 'in b'",
        mode="enforce",
        session_id="s-wt",
    )
    assert d_verified.allowed is True


# ---- gate modes: off / observe / enforce + escape hatch -----------------


def test_gate_off_allows_without_stamp(tmp_path: Path) -> None:
    vh = _load_module()
    repo = _init_git_repo(tmp_path)
    decision = vh.evaluate_commit_gate(
        repo,
        command="git commit -m 'unverified'",
        mode="off",
        session_id="sess-off",
    )
    assert decision.allowed is True
    assert decision.exit_code == 0


def test_gate_observe_logs_without_blocking(tmp_path: Path) -> None:
    vh = _load_module()
    repo = _init_git_repo(tmp_path)
    (repo / "hello.py").write_text("x = 2\n", encoding="utf-8")

    decision = vh.evaluate_commit_gate(
        repo,
        command="git commit -m 'unverified'",
        mode="observe",
        session_id="sess-obs",
    )
    assert decision.allowed is True
    assert decision.exit_code == 0
    assert "observe" in decision.message.lower()

    log_path = repo / ".claude" / "verify-gate-observe.jsonl"
    assert log_path.is_file()
    entries = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert len(entries) == 1
    assert entries[0]["mode"] == "observe"
    assert entries[0]["session_id"] == "sess-obs"
    assert entries[0]["would_block"] is True


def test_gate_enforce_blocks_twice_then_escape_hatch_allows_third(tmp_path: Path) -> None:
    vh = _load_module()
    repo = _init_git_repo(tmp_path)
    (repo / "hello.py").write_text("x = 2\n", encoding="utf-8")

    d1 = vh.evaluate_commit_gate(
        repo, command="git commit -m 'try 1'", mode="enforce", session_id="sess-enf"
    )
    assert d1.allowed is False
    assert d1.exit_code == 2

    d2 = vh.evaluate_commit_gate(
        repo, command="git commit -m 'try 2'", mode="enforce", session_id="sess-enf"
    )
    assert d2.allowed is False
    assert d2.exit_code == 2

    # 3rd attempt in the same session: escape hatch lets Claude through so it cannot deadlock.
    d3 = vh.evaluate_commit_gate(
        repo, command="git commit -m 'try 3'", mode="enforce", session_id="sess-enf"
    )
    assert d3.allowed is True
    assert d3.exit_code == 0
    assert "escape" in d3.message.lower() or "blocked twice" in d3.message.lower()


def test_gate_enforce_allows_with_valid_stamp_and_resets_counter(tmp_path: Path) -> None:
    vh = _load_module()
    repo = _init_git_repo(tmp_path)
    (repo / "hello.py").write_text("x = 2\n", encoding="utf-8")

    # Block once first.
    d1 = vh.evaluate_commit_gate(
        repo, command="git commit -m 'try 1'", mode="enforce", session_id="sess-reset"
    )
    assert d1.allowed is False

    # Now write a valid stamp and commit — allowed and counter resets.
    vh.write_stamp(repo, stages=["ruff", "mypy", "pytest"])
    d2 = vh.evaluate_commit_gate(
        repo, command="git commit -m 'verified'", mode="enforce", session_id="sess-reset"
    )
    assert d2.allowed is True
    assert d2.exit_code == 0

    # After another edit without re-checking, counter started fresh (blocks again twice).
    (repo / "hello.py").write_text("x = 3\n", encoding="utf-8")
    d3 = vh.evaluate_commit_gate(
        repo, command="git commit -m 'unverified again'", mode="enforce", session_id="sess-reset"
    )
    d4 = vh.evaluate_commit_gate(
        repo, command="git commit -m 'unverified again 2'", mode="enforce", session_id="sess-reset"
    )
    assert d3.allowed is False
    assert d4.allowed is False


def test_gate_enforce_allows_explicit_skip_reason(tmp_path: Path) -> None:
    vh = _load_module()
    repo = _init_git_repo(tmp_path)
    (repo / "hello.py").write_text("x = 2\n", encoding="utf-8")

    decision = vh.evaluate_commit_gate(
        repo,
        command="git commit -m 'docs: typo [skip-verify: doc typo only]'",
        mode="enforce",
        session_id="sess-skip",
    )
    assert decision.allowed is True
    assert decision.exit_code == 0


# ---- CLI subcommands (read-only check & Claude PreToolUse) --------------


def test_check_subcommand_is_read_only_and_nonzero_when_missing(tmp_path: Path) -> None:
    vh = _load_module()
    repo = _init_git_repo(tmp_path)

    # Repeated check calls must keep returning 1 and never create state files.
    for _ in range(4):
        assert vh.main(["--repo-root", str(repo), "check"]) == 1
    assert not (repo / ".claude" / ".verify-gate-state.json").exists()
    assert not (repo / ".claude" / "verify-gate-observe.jsonl").exists()

    # Once stamped, check returns 0.
    assert vh.main(["--repo-root", str(repo), "stamp", "--stage", "ruff"]) == 0
    assert vh.main(["--repo-root", str(repo), "check"]) == 0


def test_claude_pretool_reads_env_mode_and_handles_payloads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    vh = _load_module()
    repo = _init_git_repo(tmp_path)
    (repo / "hello.py").write_text("x = 2\n", encoding="utf-8")
    monkeypatch.setenv("DIKW_VERIFY_GATE_MODE", "enforce")

    # Non-commit command exits 0 immediately.
    payload_non_commit = json.dumps(
        {"session_id": "s1", "tool_name": "Bash", "tool_input": {"command": "git status"}}
    )
    monkeypatch.setattr(sys, "stdin", io.StringIO(payload_non_commit))
    assert vh.main(["--repo-root", str(repo), "claude-pretool"]) == 0

    # Unverified git commit in enforce mode exits 2 with guidance on stderr.
    payload_commit = json.dumps(
        {
            "session_id": "s1",
            "tool_name": "Bash",
            "tool_input": {"command": "git commit -m 'wip'"},
        }
    )
    monkeypatch.setattr(sys, "stdin", io.StringIO(payload_commit))
    assert vh.main(["--repo-root", str(repo), "claude-pretool"]) == 2
    assert "tools/check.py" in capsys.readouterr().err

    # Malformed stdin never crashes the hook (fails open with exit 0).
    monkeypatch.setattr(sys, "stdin", io.StringIO("not-json"))
    assert vh.main(["--repo-root", str(repo), "claude-pretool"]) == 0
