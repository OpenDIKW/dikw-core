"""Tests for ``run_lint_propose`` orchestrator (single task + serial loop)."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from dikw_core.domains.knowledge.lint import LintIssue, LintReport
from dikw_core.domains.knowledge.lint_fix import (
    Fixer,
    FixerContext,
    FixOperation,
    FixProposal,
    FixProposalReport,
    run_lint_propose,
)
from dikw_core.domains.knowledge.page import build_page
from dikw_core.progress import CancelToken


@dataclass
class _ListReporter:
    """Captures every event for assertions."""

    events: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    token: CancelToken = field(default_factory=CancelToken)

    async def progress(
        self, *, phase: str, current: int = 0, total: int = 0,
        detail: dict[str, Any] | None = None,
    ) -> None:
        self.events.append(
            ("progress", {"phase": phase, "current": current,
                          "total": total, "detail": detail or {}})
        )

    async def log(self, level: str, message: str) -> None:
        self.events.append(("log", {"level": level, "message": message}))

    async def partial(self, kind: str, payload: dict[str, Any]) -> None:
        self.events.append(("partial", {"kind": kind, "payload": payload}))

    def cancel_token(self) -> CancelToken:
        return self.token


def _issue(kind: str, path: str = "knowledge/concepts/x.md", line: int = 1) -> LintIssue:
    return LintIssue(kind=kind, path=path, detail=f"[[stub]] for {path}", line=line)  # type: ignore[arg-type]


def _proposal_for(issue: LintIssue) -> FixProposal:
    return FixProposal(
        proposal_id=f"p-{issue.path}",
        issue_kind=issue.kind,
        issue_path=issue.path,
        issue_detail=issue.detail,
        issue_line=issue.line,
        operations=[
            FixOperation(
                kind="update_page",
                path=issue.path,
                new_frontmatter={},
                new_body="changed",
                expected_hash="deadbeef",
            )
        ],
        rationale="stub",
        source="heuristic",
    )


@dataclass
class _ScriptedFixer:
    """Returns scripted proposals (or raises) per issue path."""

    by_path: dict[str, FixProposal | None | Exception] = field(default_factory=dict)
    kind: str = "broken_wikilink"
    seen: list[str] = field(default_factory=list)

    async def propose(
        self, issue: LintIssue, ctx: FixerContext, reporter: Any
    ) -> FixProposal | None:
        self.seen.append(issue.path)
        scripted = self.by_path.get(issue.path, None)
        if isinstance(scripted, Exception):
            raise scripted
        return scripted


def _ctx(tmp_path: Path) -> FixerContext:
    return FixerContext(
        storage=None, llm=None, embedding=None,
        base_root=tmp_path, all_pages=[],
    )


@pytest.mark.asyncio
async def test_orchestrator_filters_by_rule_and_applies_limit(
    tmp_path: Path,
) -> None:
    issues = [
        _issue("broken_wikilink", "knowledge/a.md"),
        _issue("orphan_page", "knowledge/b.md"),
        _issue("broken_wikilink", "knowledge/c.md"),
        _issue("broken_wikilink", "knowledge/d.md"),
    ]
    fixer = _ScriptedFixer(
        by_path={
            "knowledge/a.md": _proposal_for(issues[0]),
            "knowledge/c.md": _proposal_for(issues[2]),
            "knowledge/d.md": _proposal_for(issues[3]),
        }
    )
    registry: dict[str, Fixer] = {"broken_wikilink": fixer}  # type: ignore[dict-item]
    reporter = _ListReporter()

    report = await run_lint_propose(
        report=LintReport(issues=issues),
        rule="broken_wikilink",
        limit=2,
        ctx=_ctx(tmp_path),
        reporter=reporter,
        registry=registry,
    )

    assert isinstance(report, FixProposalReport)
    assert len(report.proposals) == 2
    # Only first two broken_wikilink issues seen — orphan filtered out, last skipped by limit
    assert fixer.seen == ["knowledge/a.md", "knowledge/c.md"]
    progress_events = [e for e in reporter.events if e[0] == "progress"]
    assert progress_events
    assert progress_events[0][1]["phase"] == "lint_propose"
    # last progress reports total = 2 (post-filter / post-limit)
    assert progress_events[-1][1]["total"] == 2


@pytest.mark.asyncio
async def test_orchestrator_records_fixer_returning_none_as_skipped(
    tmp_path: Path,
) -> None:
    issues = [_issue("broken_wikilink", "knowledge/a.md")]
    fixer = _ScriptedFixer(by_path={"knowledge/a.md": None})
    registry: dict[str, Fixer] = {"broken_wikilink": fixer}  # type: ignore[dict-item]

    report = await run_lint_propose(
        report=LintReport(issues=issues),
        rule=None, limit=10,
        ctx=_ctx(tmp_path),
        reporter=_ListReporter(),
        registry=registry,
    )
    assert report.proposals == []
    assert len(report.skipped) == 1
    assert report.skipped[0]["issue_path"] == "knowledge/a.md"
    assert "fixer returned None" in report.skipped[0]["reason"]


@pytest.mark.asyncio
async def test_orchestrator_records_fixer_exception_as_skipped(
    tmp_path: Path,
) -> None:
    issues = [
        _issue("broken_wikilink", "knowledge/a.md"),
        _issue("broken_wikilink", "knowledge/b.md"),
    ]
    fixer = _ScriptedFixer(
        by_path={
            "knowledge/a.md": RuntimeError("boom"),
            "knowledge/b.md": _proposal_for(issues[1]),
        }
    )
    registry: dict[str, Fixer] = {"broken_wikilink": fixer}  # type: ignore[dict-item]
    reporter = _ListReporter()

    report = await run_lint_propose(
        report=LintReport(issues=issues),
        rule=None, limit=10,
        ctx=_ctx(tmp_path),
        reporter=reporter,
        registry=registry,
    )
    # Exception in one issue must not fail the whole task.
    assert len(report.proposals) == 1
    assert report.proposals[0].issue_path == "knowledge/b.md"
    assert any(
        s["issue_path"] == "knowledge/a.md" and "boom" in s["reason"]
        for s in report.skipped
    )
    log_events = [e for e in reporter.events if e[0] == "log"]
    assert log_events  # warning logged for the failing fixer


@pytest.mark.asyncio
async def test_orchestrator_skips_issue_when_no_fixer_registered(
    tmp_path: Path,
) -> None:
    issues = [_issue("orphan_page", "knowledge/a.md")]
    # Empty registry — orphan_page has no fixer in PR1.
    report = await run_lint_propose(
        report=LintReport(issues=issues),
        rule=None, limit=10,
        ctx=_ctx(tmp_path),
        reporter=_ListReporter(),
        registry={},
    )
    assert report.proposals == []
    assert len(report.skipped) == 1
    assert "no fixer" in report.skipped[0]["reason"].lower()


@pytest.mark.asyncio
async def test_orchestrator_honours_cancel_token(tmp_path: Path) -> None:
    issues = [_issue("broken_wikilink", f"knowledge/p{i}.md") for i in range(5)]

    @dataclass
    class _CancellingFixer:
        token: CancelToken
        kind: str = "broken_wikilink"
        seen: list[str] = field(default_factory=list)

        async def propose(
            self, issue: LintIssue, ctx: FixerContext, reporter: Any
        ) -> FixProposal | None:
            self.seen.append(issue.path)
            if len(self.seen) == 2:
                self.token.cancel()
            return _proposal_for(issue)

    reporter = _ListReporter()
    fixer = _CancellingFixer(token=reporter.token)
    registry: dict[str, Fixer] = {"broken_wikilink": fixer}  # type: ignore[dict-item]

    with pytest.raises(asyncio.CancelledError):
        await run_lint_propose(
            report=LintReport(issues=issues),
            rule=None, limit=10,
            ctx=_ctx(tmp_path),
            reporter=reporter,
            registry=registry,
        )
    # Cancellation is checked at the top of each loop iteration; the
    # cancelling fixer ran twice (set the token on iter 2), and the
    # third iteration's pre-check raised before invoking the fixer.
    assert fixer.seen == ["knowledge/p0.md", "knowledge/p1.md"]


@pytest.mark.asyncio
async def test_orchestrator_uses_default_registry_when_none_given(
    tmp_path: Path,
) -> None:
    """Production callers omit ``registry`` and get the package default
    (the ``FIXER_REGISTRY`` from ``lint_fixers``)."""
    # No issues → no fixer dispatch, but the call must not raise on the
    # registry default — which would only manifest at non-empty issues.
    report = await run_lint_propose(
        report=LintReport(issues=[]),
        rule=None, limit=10,
        ctx=_ctx(tmp_path),
        reporter=_ListReporter(),
    )
    assert report.proposals == []
    assert report.skipped == []


@pytest.mark.asyncio
async def test_propose_kind_is_kept_in_sync(tmp_path: Path) -> None:
    """Built-in registry's broken_wikilink fixer must declare ``kind`` so
    the orchestrator can match issue.kind against it."""
    from dikw_core.domains.knowledge.lint_fixers import FIXER_REGISTRY

    fixer = FIXER_REGISTRY["broken_wikilink"]
    assert fixer.kind == "broken_wikilink"

    # Construct a minimal real fix scenario so the registry's actual
    # fixer is exercised end-to-end inside the orchestrator (not just
    # the registry lookup).
    target_page = build_page(title="Existing Page", body="# Existing Page\n", type_="concept")
    src_page = build_page(
        title="Source",
        body="# Source\n\nLink: [[existing page]]\n",
        type_="concept",
    )
    src_abs = tmp_path / src_page.path
    src_abs.parent.mkdir(parents=True, exist_ok=True)
    src_abs.write_text(
        "---\ntitle: Source\n---\n\n"
        "# Source\n\nLink: [[existing page]]\n",
        encoding="utf-8",
    )

    ctx = FixerContext(
        storage=None, llm=None, embedding=None,
        base_root=tmp_path, all_pages=[target_page, src_page],
    )
    issue = LintIssue(
        kind="broken_wikilink",
        path=src_page.path,
        detail="[[existing page]] has no matching knowledge page",
        line=3,
    )
    report = await run_lint_propose(
        report=LintReport(issues=[issue]),
        rule="broken_wikilink", limit=10,
        ctx=ctx, reporter=_ListReporter(),
    )
    assert len(report.proposals) == 1
    assert "[[Existing Page]]" in report.proposals[0].operations[0].new_body  # type: ignore[arg-type]
