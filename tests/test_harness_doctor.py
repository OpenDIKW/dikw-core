"""Black-box acceptance tests for the published harness-doctor command."""

import json
import os
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def run(root, *args, cwd=None, stdin=None):
    return subprocess.run(
        [sys.executable, str(root / "scripts/harness-doctor"), *args],
        cwd=cwd or root,
        input=stdin,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )


@pytest.fixture
def repo(tmp_path, monkeypatch):
    # A Git hook in the caller must not redirect any of this fixture's child processes.
    for name in (
        "GIT_DIR",
        "GIT_WORK_TREE",
        "GIT_INDEX_FILE",
        "GIT_OBJECT_DIRECTORY",
        "GIT_ALTERNATE_OBJECT_DIRECTORIES",
        "GIT_COMMON_DIR",
        "GIT_PREFIX",
    ):
        monkeypatch.delenv(name, raising=False)
    root = tmp_path / "work tree"
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    (root / "scripts").mkdir()
    shutil.copy(ROOT / "scripts/harness-doctor", root / "scripts")
    shutil.copytree(ROOT / "harness", root / "harness")
    shutil.copy(ROOT / "AGENTS.md", root)
    assert run(root, "sync").returncode == 0
    return root


def test_drift_repair_and_session_receipt(repo):
    (repo.parent / "CLAUDE.md").write_text("Parent instructions", encoding="utf-8")
    assert run(repo).returncode == 0
    for client in ("claude", "codex", "antigravity"):
        result = run(repo, "session-start", client, cwd=repo / "scripts", stdin="{}")
        assert result.returncode == 0
        body = json.loads(result.stdout)
        receipt = (
            body["injectSteps"][0]["systemMessage"]["systemMessage"]
            if client == "antigravity"
            else body["hookSpecificOutput"]["additionalContext"]
        )
        assert "harness-doctor: OK" in receipt
        if client == "claude":
            assert "AGENTS.md" in receipt and "read it" in receipt

    skill = next((repo / ".agents/skills").glob("*/SKILL.md"))
    resources = next((repo / ".agents/skills").rglob("ADR-FORMAT.md"))
    role = repo / ".codex/agents/reviewer.toml"
    assert tomllib.loads(role.read_text("utf-8"))["name"] == "reviewer"
    originals = {}
    for path in (skill, resources, role, repo / ".agents/hooks.json"):
        originals[path] = path.read_bytes()
        path.unlink()
    role.write_text("drift\n", encoding="utf-8")
    result = run(repo, "check")
    assert result.returncode == 1
    assert "line 1, byte 0: expected" in result.stdout
    for path in originals:
        assert path.relative_to(repo).as_posix() in result.stdout
    assert "harness-doctor: FAIL" in run(repo, "session-start", "codex").stdout
    assert run(repo, "sync").returncode == 0
    assert run(repo, "sync").returncode == 0
    assert all(p.read_bytes() == content for p, content in originals.items())
    assert run(repo).returncode == 0


def test_unknown_files_and_instruction_budget(repo):
    extra = repo / ".claude/rules/incubating.md"
    extra.write_text("keep my work\n", encoding="utf-8")
    assert "unexpected" in run(repo).stdout
    run(repo, "sync")
    assert extra.read_text("utf-8") == "keep my work\n"
    extra.unlink()
    local = repo / "CLAUDE.local.md"
    local.write_text("Personal instructions\n", encoding="utf-8")
    assert run(repo).returncode == 0
    subprocess.run(["git", "add", "-f", "--", "CLAUDE.local.md"], cwd=repo, check=True)
    assert "competing project instructions: CLAUDE.local.md" in run(repo).stdout
    local.unlink()
    for old in ("GEMINI.md", ".agents/AGENTS.md", "_agent/GEMINI.md"):
        conflicting = repo / old
        conflicting.parent.mkdir(exist_ok=True)
        conflicting.write_text("other root", encoding="utf-8")
        assert old in run(repo).stdout
        conflicting.unlink()
    guide = repo / "AGENTS.md"
    original = guide.read_text("utf-8").splitlines()
    guide.write_text("\n".join(original + ["line"] * (200 - len(original))) + "\n", "utf-8")
    assert run(repo).returncode == 0
    with guide.open("a", encoding="utf-8") as stream:
        stream.write("line 201\n")
    assert run(repo).returncode == 1


def test_preserves_claude_settings_and_unregistered_hooks(repo):
    path = repo / ".claude/settings.json"
    config = json.loads(path.read_bytes())
    config["permissions"] = {"deny": ["Read(private.txt)"]}
    del config["hooks"]["SessionStart"]
    path.write_text(json.dumps(config), encoding="utf-8-sig")
    assert run(repo, "sync").returncode == 0
    repaired = json.loads(path.read_bytes())
    assert repaired["permissions"] == config["permissions"]
    assert repaired["hooks"]["SessionStart"]
    repaired["hooks"]["Stop"] = [{"hooks": [{"type": "command", "command": "echo local"}]}]
    path.write_text(json.dumps(repaired), encoding="utf-8")
    assert run(repo, "sync").returncode == 1
    assert json.loads(path.read_bytes()) == repaired
    for malformed in (b'{"permissions":', b"[]"):
        path.write_bytes(malformed)
        result = run(repo, "sync")
        assert result.returncode == 1 and ".claude/settings.json" in result.stdout
        assert path.read_bytes() == malformed


@pytest.mark.parametrize(
    "name", [".claude/settings.json", ".codex/hooks.json", ".agents/hooks.json"]
)
def test_sync_protects_unregistered_and_malformed_hooks(repo, name):
    path = repo / name
    registered = json.loads(path.read_bytes())
    owner = registered["harness-doctor"] if name.startswith(".agents/") else registered["hooks"]
    original = owner["SessionStart"]
    replacement = json.loads(json.dumps(original))
    handler = replacement[0].get("hooks", replacement)[0]
    handler["command"] = "node scripts/session-bootstrap.mjs"
    handler.pop("statusMessage", None)
    for changed in (
        [*original, {"type": "command", "command": "echo local"}],
        {"custom": True},
        replacement,
    ):
        owner["SessionStart"] = changed
        content = json.dumps(registered).encode()
        path.write_bytes(content)
        result = run(repo, "sync")
        assert result.returncode == 1 and name in result.stdout
        assert path.read_bytes() == content


def test_skill_executable_mode(repo):
    suffix = "skills/grill-with-docs/check.sh"
    source = repo / "harness" / suffix
    copies = [repo / adapter / suffix for adapter in (".claude", ".agents")]
    source.write_bytes(b"#!/bin/sh\nexit 0\n")
    if os.name == "nt":
        subprocess.run(["git", "add", "--chmod=+x", "--", source], cwd=repo, check=True)
        assert "git add --chmod=+x" in run(repo, "sync").stdout
        subprocess.run(["git", "add", "--chmod=+x", "--", *copies], cwd=repo, check=True)
    else:
        source.chmod(0o755)
        assert run(repo, "sync").returncode == 0
        for target in copies:
            subprocess.run([target], cwd=repo, check=True)
    assert run(repo, "check").returncode == 0
    if os.name == "nt":
        subprocess.run(["git", "update-index", "--chmod=-x", "--", copies[0]], cwd=repo, check=True)
    else:
        copies[0].chmod(0o644)
    assert "executable mode .claude/" in run(repo).stdout
    if os.name != "nt":
        assert run(repo, "sync").returncode == 0
        subprocess.run([copies[0]], cwd=repo, check=True)
        assert run(repo).returncode == 0


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission and umask behavior")
def test_generated_files_preserve_mode_and_respect_umask(repo):
    role = repo / ".claude/agents/reviewer.md"
    role.chmod(0o640)
    role.write_text("change\n", encoding="utf8")
    assert run(repo, "sync").returncode == 0
    assert role.stat().st_mode & 0o777 == 0o640
    role.unlink()
    previous = os.umask(0o027)
    try:
        assert run(repo, "sync").returncode == 0
    finally:
        os.umask(previous)
    assert role.stat().st_mode & 0o777 == 0o640


def test_shared_routing_cases(repo):
    fixture = json.loads((repo / "harness/tests/routing.json").read_text("utf-8"))
    path = repo / "harness/manifest.json"
    manifest = json.loads(path.read_text("utf-8"))
    for name, globs in fixture["patterns"].items():
        manifest["rules"][name] = ["fixture/" + glob for glob in globs]
        (repo / f"harness/rules/{name}.md").write_text(name + "\n", "utf-8")
    path.write_text(json.dumps(manifest), "utf-8")
    assert run(repo, "sync").returncode == 0
    selector = (repo / ".agents/rules/config.md").read_text("utf-8").splitlines()[2]
    assert (
        json.loads(selector.removeprefix("globs: "))
        == "**/fixture/src/**/config.py,**/fixture/src/**/config.ts"
    )
    deeper = repo / "nested"
    deeper.mkdir()
    for case in fixture["cases"]:
        paths = ["../fixture/" + p for p in case["paths"]]
        result = run(repo, "rules", *paths, cwd=deeper)
        assert result.returncode == 0
        assert result.stdout.splitlines() == [f".codex/rules/areas/{s}.md" for s in case["want"]]
