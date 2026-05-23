"""Tests for the per-rule lint fixers.

PR1 shipped only ``broken_wikilink`` with a heuristic-only path: fuzzy
match the broken target against existing K-layer page titles and propose
a ``[[link]]`` rewrite when the match is confident enough; otherwise the
fixer returns ``None`` so the orchestrator skips the issue (the
evidence-backed LLM repair path shipped later in #83).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from dikw_core.domains.knowledge.lint import LintIssue, LintReport
from dikw_core.domains.knowledge.lint_fix import (
    FixerContext,
    FixOperation,
    FixProposal,
    WikiPageMeta,
    run_lint_propose,
)
from dikw_core.domains.knowledge.lint_fixers import (
    broken_wikilink as bwl_mod,
)
from dikw_core.domains.knowledge.lint_fixers.broken_wikilink import (
    BrokenWikilinkFixer,
)
from dikw_core.domains.knowledge.wiki import build_page
from dikw_core.schemas import Hit, Layer

from .fakes import FakeLLM


def _make_page(title: str, body: str) -> Any:
    """Build a real ``WikiPage`` so tests can write the same path layout
    on disk that production synth would produce."""
    return build_page(title=title, body=body, type_="concept")


def _meta_from(page: Any) -> WikiPageMeta:
    return WikiPageMeta(path=page.path, title=page.title)


def _ctx(
    *,
    pages: list[Any],
    wiki_root: Path,
    llm: Any | None = None,
    enable_llm: bool = False,
    cfg: Any = None,
) -> FixerContext:
    """Build a FixerContext for unit tests.

    LLM-enabled paths require a non-None ``cfg`` in production (see
    fixer guards); default to ``_default_cfg()`` so tests don't have
    to thread it through every call. Heuristic-only tests keep
    ``cfg=None`` and ``enable_llm=False`` — that branch never reads cfg.
    """
    if enable_llm and cfg is None:
        cfg = _default_cfg()
    return FixerContext(
        storage=None,
        llm=llm if llm is not None else FakeLLM(),  # type: ignore[arg-type]
        embedding=None,
        wiki_root=wiki_root,
        all_pages=[_meta_from(p) for p in pages],
        enable_llm=enable_llm,
        cfg=cfg,
    )


def _default_cfg() -> Any:
    """Build a minimal ``DikwConfig`` for fixer tests.

    Used by both ``broken_wikilink`` LLM-stub tests and the
    ``non_atomic_page`` tests so the fixer's ``ctx.cfg.provider.*`` /
    ``ctx.cfg.schema_.*`` / ``ctx.cfg.synth.*`` reads land on real
    defaults rather than ``None``.
    """
    from dikw_core.config import DikwConfig

    return DikwConfig()


@pytest.mark.asyncio
async def test_fuzzy_match_above_threshold_proposes_update_page(
    tmp_path: Path,
) -> None:
    """A broken ``[[foo bar]]`` should rewrite to the existing ``Foo Bar``
    page when normalized titles match within the 0.85 ratio band."""
    wiki_root = tmp_path
    target_page = _make_page("Foo Bar", "# Foo Bar\nbody\n")
    src_page = _make_page(
        "Source Page",
        "# Source Page\n\nSee [[foo  bar]] for context.\n",
    )
    src_abs = wiki_root / src_page.path
    src_abs.parent.mkdir(parents=True, exist_ok=True)
    src_abs.write_text(
        "---\ntitle: Source Page\n---\n\n"
        "# Source Page\n\nSee [[foo  bar]] for context.\n",
        encoding="utf-8",
    )

    issue = LintIssue(
        kind="broken_wikilink",
        path=src_page.path,
        detail="[[foo  bar]] has no matching wiki page",
        line=3,
    )
    fixer = BrokenWikilinkFixer()
    proposal = await fixer.propose(
        issue,
        _ctx(pages=[target_page, src_page], wiki_root=wiki_root),
        reporter=_NullReporter(),
    )

    assert proposal is not None
    assert isinstance(proposal, FixProposal)
    assert proposal.source == "heuristic"
    assert proposal.issue_path == src_page.path
    assert len(proposal.operations) == 1
    op = proposal.operations[0]
    assert isinstance(op, FixOperation)
    assert op.kind == "update_page"
    assert op.path == src_page.path
    assert op.expected_hash is not None  # we will hash the on-disk file
    assert op.new_body is not None
    # Rewrite should land an exact-title link.
    assert "[[Foo Bar]]" in op.new_body
    assert "[[foo  bar]]" not in op.new_body


@pytest.mark.asyncio
async def test_fuzzy_match_miss_returns_none_in_pr1(tmp_path: Path) -> None:
    """When no existing title is close enough, PR1's heuristic-only
    path returns ``None`` (the evidence-backed LLM repair path shipped
    later in #83)."""
    wiki_root = tmp_path
    other = _make_page("Completely Different", "# Completely Different\n")
    src_page = _make_page(
        "Source Page",
        "# Source Page\n\nSee [[xyz123abc]] here.\n",
    )
    src_abs = wiki_root / src_page.path
    src_abs.parent.mkdir(parents=True, exist_ok=True)
    src_abs.write_text(
        "---\ntitle: Source Page\n---\n\n"
        "# Source Page\n\nSee [[xyz123abc]] here.\n",
        encoding="utf-8",
    )

    issue = LintIssue(
        kind="broken_wikilink",
        path=src_page.path,
        detail="[[xyz123abc]] has no matching wiki page",
        line=3,
    )
    fixer = BrokenWikilinkFixer()
    proposal = await fixer.propose(
        issue,
        _ctx(pages=[other, src_page], wiki_root=wiki_root),
        reporter=_NullReporter(),
    )
    assert proposal is None


@pytest.mark.asyncio
async def test_match_excludes_self_page(tmp_path: Path) -> None:
    """A page must not propose a link to itself even if the broken
    target normalizes to the page's own title."""
    wiki_root = tmp_path
    src_page = _make_page(
        "Source Page",
        "# Source Page\n\nSee [[source page]].\n",
    )
    src_abs = wiki_root / src_page.path
    src_abs.parent.mkdir(parents=True, exist_ok=True)
    src_abs.write_text(
        "---\ntitle: Source Page\n---\n\n"
        "# Source Page\n\nSee [[source page]].\n",
        encoding="utf-8",
    )

    issue = LintIssue(
        kind="broken_wikilink",
        path=src_page.path,
        detail="[[source page]] has no matching wiki page",
        line=3,
    )
    fixer = BrokenWikilinkFixer()
    proposal = await fixer.propose(
        issue,
        _ctx(pages=[src_page], wiki_root=wiki_root),
        reporter=_NullReporter(),
    )
    assert proposal is None


@pytest.mark.asyncio
async def test_fuzzy_match_works_on_non_ascii_titles(tmp_path: Path) -> None:
    """The earlier ASCII-only ``[a-z0-9]`` normalize regex stripped
    every CJK glyph and made the fixer a no-op for multilingual bases.
    Verify a mixed-case CJK target now resolves."""
    wiki_root = tmp_path
    target_page = _make_page("卡尔曼滤波", "# 卡尔曼滤波\nbody\n")
    src_page = _make_page(
        "Source CJK",
        "# Source CJK\n\nReference [[卡尔曼 滤波]] here.\n",
    )
    src_abs = wiki_root / src_page.path
    src_abs.parent.mkdir(parents=True, exist_ok=True)
    src_abs.write_text(
        "---\ntitle: Source CJK\n---\n\n"
        "# Source CJK\n\nReference [[卡尔曼 滤波]] here.\n",
        encoding="utf-8",
    )

    issue = LintIssue(
        kind="broken_wikilink",
        path=src_page.path,
        detail="[[卡尔曼 滤波]] has no matching wiki page",
        line=3,
    )
    fixer = BrokenWikilinkFixer()
    proposal = await fixer.propose(
        issue,
        _ctx(pages=[target_page, src_page], wiki_root=wiki_root),
        reporter=_NullReporter(),
    )
    assert proposal is not None, "non-ASCII titles must not be silently skipped"
    assert "[[卡尔曼滤波]]" in proposal.operations[0].new_body  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_extracts_target_from_detail_when_issue_lacks_explicit_target(
    tmp_path: Path,
) -> None:
    """The lint scanner formats detail as ``[[<target>]] has no matching ...``;
    the fixer must parse that to recover the broken target text."""
    wiki_root = tmp_path
    target_page = _make_page("Karpathy Rule", "# Karpathy Rule\nbody\n")
    src_page = _make_page(
        "Source",
        "# Source\n\nApply [[karpathy rule]] here.\n",
    )
    src_abs = wiki_root / src_page.path
    src_abs.parent.mkdir(parents=True, exist_ok=True)
    src_abs.write_text(
        "---\ntitle: Source\n---\n\n"
        "# Source\n\nApply [[karpathy rule]] here.\n",
        encoding="utf-8",
    )

    issue = LintIssue(
        kind="broken_wikilink",
        path=src_page.path,
        detail="[[karpathy rule]] has no matching wiki page",
        line=3,
    )
    fixer = BrokenWikilinkFixer()
    proposal = await fixer.propose(
        issue,
        _ctx(pages=[target_page, src_page], wiki_root=wiki_root),
        reporter=_NullReporter(),
    )
    assert proposal is not None
    assert "[[Karpathy Rule]]" in proposal.operations[0].new_body  # type: ignore[arg-type]


class _NullReporter:
    """Minimal ``ProgressReporter`` for fixer-unit tests."""

    def __init__(self) -> None:
        from dikw_core.progress import CancelToken

        self._token = CancelToken()

    async def progress(
        self, *, phase: str, current: int = 0, total: int = 0,
        detail: dict[str, Any] | None = None,
    ) -> None:
        return None

    async def log(self, level: str, message: str) -> None:
        return None

    async def partial(self, kind: str, payload: dict[str, Any]) -> None:
        return None

    def cancel_token(self) -> Any:
        return self._token


class _ListReporter:
    """Captures progress/log/partial events so orchestrator-level tests
    can assert on the live event stream as well as on the final report."""

    def __init__(self) -> None:
        from dikw_core.progress import CancelToken

        self._token = CancelToken()
        self.events: list[tuple[str, dict[str, Any]]] = []

    async def progress(
        self, *, phase: str, current: int = 0, total: int = 0,
        detail: dict[str, Any] | None = None,
    ) -> None:
        self.events.append((
            "progress",
            {"phase": phase, "current": current, "total": total,
             "detail": detail or {}},
        ))

    async def log(self, level: str, message: str) -> None:
        self.events.append(("log", {"level": level, "message": message}))

    async def partial(self, kind: str, payload: dict[str, Any]) -> None:
        self.events.append(("partial", {"kind": kind, "payload": payload}))

    def cancel_token(self) -> Any:
        return self._token


# --- PR3 (#83): broken_wikilink evidence-backed LLM repair -------------------
#
# Semantics change: ``--enable-llm`` no longer means "allow TODO stubs"; it
# means "allow the LLM to write a real grounded page IFF the D/I layer has
# enough source evidence". The fixer pulls evidence chunks via
# ``_collect_evidence`` (real impl uses HybridSearcher); tests
# monkeypatch that function to script evidence presence/absence and verify
# both the gating (no evidence → skip + structured reason, agent-visible)
# and the post-generation body checks (reject TODO markers, reject too-short
# bodies).


def _make_broken_link_setup(
    tmp_path: Path, broken_target: str = "Whole New Topic"
) -> tuple[Path, Any, Any]:
    """Build a wiki-root + a source page that references a missing target.

    No existing K-page resembles ``broken_target``, so the heuristic
    must miss; LLM-grounded tests then assert that the fallback fires
    when evidence is present and is silent when not.
    """
    wiki_root = tmp_path
    src_page = _make_page(
        "Source",
        f"# Source\n\nSee [[{broken_target}]] for context.\n",
    )
    src_abs = wiki_root / src_page.path
    src_abs.parent.mkdir(parents=True, exist_ok=True)
    src_abs.write_text(
        f"---\ntitle: Source\n---\n\n# Source\n\nSee [[{broken_target}]] for context.\n",
        encoding="utf-8",
    )
    issue = LintIssue(
        kind="broken_wikilink",
        path=src_page.path,
        detail=f"[[{broken_target}]] has no matching wiki page",
        line=3,
    )
    return wiki_root, src_page, issue


# A 400+ char real-content body: passes the _MIN_BODY_CHARS=200 floor and
# contains no _FORBIDDEN_BODY_TOKENS. Used by happy-path + canonicalization
# tests where we want the grounded path to succeed end-to-end.
_GROUNDED_LLM_RESPONSE = (
    "<page path=\"wiki/concepts/whole-new-topic.md\" type=\"concept\">\n"
    "---\n"
    "tags: [grounded]\n"
    "sources: [\"sources/foo.md\"]\n"
    "---\n"
    "\n"
    "# Whole New Topic\n"
    "\n"
    "Whole New Topic is the concept the source material introduces. "
    "The evidence describes specific characteristics: practitioners use "
    "it to coordinate across teams, the approach measurably reduces "
    "review time, and adoption has spread across multiple production "
    "projects. Concrete examples in the source illustrate the workflow "
    "from initial setup through ongoing maintenance.\n"
    "</page>\n"
)

# Same shell, but the body still carries a TODO marker — the post-generation
# guard must reject this even when evidence was sufficient.
_TODO_LLM_RESPONSE = (
    "<page path=\"wiki/concepts/whole-new-topic.md\" type=\"concept\">\n"
    "---\n"
    "tags: [stub]\n"
    "---\n"
    "\n"
    "# Whole New Topic\n"
    "\n"
    "TODO: stub page for the broken `[[Whole New Topic]]` reference. "
    "Even with enough evidence we must refuse pages whose body still "
    "contains the literal TODO marker, otherwise broken_wikilink: 0 "
    "would once again hide unrepaired knowledge gaps.\n"
    "</page>\n"
)

# Body too short — passes the TODO-token check but fails the body length
# floor. Guards against the LLM producing "Topic A is a topic." filler.
_SHORT_LLM_RESPONSE = (
    "<page path=\"wiki/concepts/whole-new-topic.md\" type=\"concept\">\n"
    "---\n"
    "tags: [grounded]\n"
    "---\n"
    "\n"
    "# Whole New Topic\n"
    "\n"
    "Whole New Topic is a topic.\n"
    "</page>\n"
)


def _make_hit(*, chunk_id: int, text: str, source_path: str = "sources/foo.md") -> Hit:
    """Build a D-layer ``Hit`` good enough to feed the evidence check.

    Only the fields the fixer reads matter: ``text`` (counted toward the
    char threshold + rendered into the prompt) and ``path`` / ``title``
    (citation in the prompt). The rest of the schema's required fields
    get plausible placeholders.
    """
    return Hit(
        doc_id=f"D-{chunk_id}",
        chunk_id=chunk_id,
        seq=chunk_id,
        score=1.0 - chunk_id * 0.01,
        snippet=text[:80],
        path=source_path,
        title="Source File",
        layer=Layer.SOURCE,
        start=0,
        end=len(text),
        text=text,
    )


def _patch_evidence(monkeypatch: pytest.MonkeyPatch, hits: list[Hit]) -> dict[str, int]:
    """Swap ``_collect_evidence`` with a script that returns ``hits``.

    Returns a counter dict so callers can assert call count — e.g., the
    ``enable_llm=False`` test verifies the evidence pipeline is not even
    invoked when the user hasn't opted in.
    """
    counter = {"calls": 0}

    async def fake(
        ctx: Any, target: str, excerpt: str, issue_path: str
    ) -> list[Hit]:
        counter["calls"] += 1
        return list(hits)

    monkeypatch.setattr(bwl_mod, "_collect_evidence", fake)
    return counter


def _grounded_evidence_hits() -> list[Hit]:
    """Two chunks totaling ~500 chars — well above the _MIN_EVIDENCE_CHARS=200
    floor, comfortably above _MIN_EVIDENCE_CHUNKS=1."""
    return [
        _make_hit(
            chunk_id=1,
            text=(
                "Whole New Topic was introduced last quarter to coordinate "
                "engineering teams across distributed projects. Practitioners "
                "report measurable reductions in review time after adopting "
                "the workflow described here."
            ),
        ),
        _make_hit(
            chunk_id=2,
            text=(
                "The Whole New Topic playbook covers initial setup, the "
                "weekly cadence, and the rollback procedure. Three production "
                "projects have adopted it so far, each documenting their "
                "experience in the engineering log."
            ),
            source_path="sources/bar.md",
        ),
    ]


def _orchestrator_skipped_reasons(
    report: Any,
) -> list[str]:
    """Pull the ``reason`` field off every skip record for assertions."""
    return [str(s.get("reason", "")) for s in report.skipped]


@pytest.mark.asyncio
async def test_broken_wikilink_llm_grounded_when_enough_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With ``enable_llm=True`` and sufficient D/I evidence the fixer must
    produce a ``create_page`` proposal whose body is the grounded LLM
    response — no TODO marker, no stub language."""
    wiki_root, src_page, issue = _make_broken_link_setup(tmp_path)
    _patch_evidence(monkeypatch, _grounded_evidence_hits())
    fake = FakeLLM(response_text=_GROUNDED_LLM_RESPONSE)
    fixer = BrokenWikilinkFixer()
    proposal = await fixer.propose(
        issue,
        _ctx(pages=[src_page], wiki_root=wiki_root, llm=fake, enable_llm=True),
        reporter=_NullReporter(),
    )

    assert proposal is not None, "grounded path must fire when evidence is enough"
    assert isinstance(proposal, FixProposal)
    assert proposal.source == "llm"
    assert proposal.issue_path == src_page.path
    assert len(proposal.operations) == 1
    op = proposal.operations[0]
    assert isinstance(op, FixOperation)
    assert op.kind == "create_page"
    assert op.expected_hash is None
    assert op.path.startswith("wiki/")
    assert op.path.endswith(".md")
    assert op.new_body is not None
    assert "Whole New Topic" in op.new_body
    # The whole point of #83 — body must not regress to TODO stubs.
    assert "TODO" not in op.new_body
    assert "stub page" not in op.new_body.lower()
    # Rationale must surface the evidence count so reviewers see the
    # repair was grounded, not hallucinated.
    assert "evidence" in proposal.rationale.lower()
    # The prompt must inject the evidence chunks so the LLM can ground
    # its prose against real source text.
    assert fake.last_user is not None
    assert "Whole New Topic" in fake.last_user
    assert "review time" in fake.last_user  # excerpt from chunk #1


@pytest.mark.asyncio
async def test_broken_wikilink_llm_skipped_when_evidence_insufficient(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No evidence in D/I → the fixer must NOT call the LLM and the
    orchestrator must record a structured ``evidence_insufficient`` reason
    so agents reading ``FixProposalReport.skipped`` see why the wikilink
    stays unrepaired (not the generic 'fixer returned None')."""
    wiki_root, src_page, issue = _make_broken_link_setup(tmp_path)
    counter = _patch_evidence(monkeypatch, [])  # zero chunks
    fake = FakeLLM(response_text=_GROUNDED_LLM_RESPONSE)

    ctx = _ctx(pages=[src_page], wiki_root=wiki_root, llm=fake, enable_llm=True)
    reporter = _ListReporter()
    report = await run_lint_propose(
        report=LintReport(issues=[issue]),
        rule="broken_wikilink",
        limit=10,
        ctx=ctx,
        reporter=reporter,
        registry={"broken_wikilink": BrokenWikilinkFixer()},  # type: ignore[dict-item]
    )

    assert report.proposals == []
    reasons = _orchestrator_skipped_reasons(report)
    assert len(reasons) == 1
    assert reasons[0].startswith("evidence_insufficient"), reasons
    # Evidence pipeline was invoked, LLM was not.
    assert counter["calls"] == 1
    assert fake.last_user is None, "LLM must not be called without evidence"
    # Reporter also saw the reason on the live stream so --plain users
    # see it without waiting for the final report.
    log_messages = [
        e[1]["message"] for e in reporter.events if e[0] == "log"
    ]
    assert any(
        "evidence_insufficient" in m for m in log_messages
    ), log_messages


@pytest.mark.asyncio
async def test_broken_wikilink_llm_rejects_todo_body(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Evidence is sufficient but the LLM still emits a TODO-laced body
    (prompt regression, model misbehavior). The fixer must reject the
    proposal and surface ``rejected_todo_marker`` as the skip reason —
    a defence-in-depth so #83 cannot resurface through a prompt drift."""
    wiki_root, src_page, issue = _make_broken_link_setup(tmp_path)
    _patch_evidence(monkeypatch, _grounded_evidence_hits())
    fake = FakeLLM(response_text=_TODO_LLM_RESPONSE)

    ctx = _ctx(pages=[src_page], wiki_root=wiki_root, llm=fake, enable_llm=True)
    report = await run_lint_propose(
        report=LintReport(issues=[issue]),
        rule="broken_wikilink",
        limit=10,
        ctx=ctx,
        reporter=_ListReporter(),
        registry={"broken_wikilink": BrokenWikilinkFixer()},  # type: ignore[dict-item]
    )

    assert report.proposals == []
    reasons = _orchestrator_skipped_reasons(report)
    assert len(reasons) == 1
    assert reasons[0].startswith("rejected_todo_marker"), reasons


@pytest.mark.asyncio
async def test_broken_wikilink_llm_rejects_title_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Evidence is sufficient and the body is real prose, but the LLM
    titled the page after a *related* concept (not the broken target).
    Applying that would add an unrelated K-page while leaving the
    original ``[[Whole New Topic]]`` reference still broken — exactly
    the silent failure mode #83 is fighting. The fixer must reject the
    proposal with ``rejected_title_mismatch`` so agents see why."""
    wiki_root, src_page, issue = _make_broken_link_setup(tmp_path)
    _patch_evidence(monkeypatch, _grounded_evidence_hits())
    mismatched_response = (
        "<page path=\"wiki/concepts/related-topic.md\" type=\"concept\">\n"
        "---\n"
        "tags: [grounded]\n"
        "---\n"
        "\n"
        "# Related Topic\n"
        "\n"
        "Related Topic is a different concept that the evidence happens "
        "to mention. It is not the canonical target of the broken link, "
        "so creating this page would not resolve the original wikilink "
        "and would only pollute the K-layer with an unrelated entry. "
        "The fixer must catch this before proposing a create_page op.\n"
        "</page>\n"
    )
    fake = FakeLLM(response_text=mismatched_response)

    ctx = _ctx(pages=[src_page], wiki_root=wiki_root, llm=fake, enable_llm=True)
    report = await run_lint_propose(
        report=LintReport(issues=[issue]),
        rule="broken_wikilink",
        limit=10,
        ctx=ctx,
        reporter=_ListReporter(),
        registry={"broken_wikilink": BrokenWikilinkFixer()},  # type: ignore[dict-item]
    )

    assert report.proposals == []
    reasons = _orchestrator_skipped_reasons(report)
    assert len(reasons) == 1
    assert reasons[0].startswith("rejected_title_mismatch"), reasons


@pytest.mark.asyncio
async def test_broken_wikilink_collect_evidence_falls_back_to_bm25_on_hybrid_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``mode='hybrid'`` embeds the query first; if the embedder raises
    (no creds, transient outage, dim mismatch) the whole search blows
    up. Without a fallback the fixer is recorded as ``fixer raised`` —
    losing every BM25-recoverable repair on a base whose embedding leg
    is temporarily degraded. We must retry with ``mode='bm25'`` so
    lexical evidence still drives grounded proposals."""

    class _FailingHybridSearcher:
        instance: _FailingHybridSearcher | None = None
        hybrid_called: int = 0
        bm25_called: int = 0

        @classmethod
        def from_config(
            cls, storage: Any, embedder: Any, retrieval_cfg: Any, **_kw: Any
        ) -> _FailingHybridSearcher:
            inst = cls()
            cls.instance = inst
            return inst

        async def search(
            self, q: str, *, limit: int, layer: Any, mode: str
        ) -> list[Hit]:
            _ = (q, limit, layer)
            if mode == "hybrid":
                type(self).hybrid_called += 1
                raise RuntimeError("embedding provider unreachable")
            type(self).bm25_called += 1
            return _grounded_evidence_hits()

    class _StubStorage:
        async def get_active_embed_version(self, *, modality: str) -> Any:
            _ = modality
            return None

    monkeypatch.setattr(bwl_mod, "HybridSearcher", _FailingHybridSearcher)
    wiki_root, src_page, issue = _make_broken_link_setup(tmp_path)
    fake = FakeLLM(response_text=_GROUNDED_LLM_RESPONSE)
    ctx = FixerContext(
        storage=_StubStorage(),  # type: ignore[arg-type]
        llm=fake,
        embedding=None,
        wiki_root=wiki_root,
        all_pages=[_meta_from(src_page)],
        enable_llm=True,
        cfg=_default_cfg(),
    )
    fixer = BrokenWikilinkFixer()
    proposal = await fixer.propose(issue, ctx, reporter=_NullReporter())

    assert _FailingHybridSearcher.hybrid_called == 1
    assert _FailingHybridSearcher.bm25_called == 1, (
        "fixer must retry with mode='bm25' after hybrid raises"
    )
    assert proposal is not None, "BM25 fallback must produce a grounded proposal"
    assert proposal.source == "llm"


@pytest.mark.asyncio
async def test_broken_wikilink_llm_rejects_singular_target_plural_title(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``[[Network]]`` (singular target) + LLM page titled ``# Networks``
    (regular plural). ``_normalize_for_match`` stems both sides to
    ``network`` — a symmetric compare would accept this — but the real
    resolver indexes stored titles via ``_normalize_base`` (no stem) so
    a page indexed as ``networks`` would NOT resolve a stemmed lookup of
    ``network``. The fixer must mirror the resolver's asymmetric
    semantics or this proposal silently leaves ``broken_wikilink`` open."""
    wiki_root = tmp_path
    src_page = _make_page(
        "Source",
        "# Source\n\nSee [[Network]] for context.\n",
    )
    src_abs = wiki_root / src_page.path
    src_abs.parent.mkdir(parents=True, exist_ok=True)
    src_abs.write_text(
        "---\ntitle: Source\n---\n\n# Source\n\nSee [[Network]] for context.\n",
        encoding="utf-8",
    )
    issue = LintIssue(
        kind="broken_wikilink",
        path=src_page.path,
        detail="[[Network]] has no matching wiki page",
        line=3,
    )
    _patch_evidence(monkeypatch, _grounded_evidence_hits())
    plural_response = (
        "<page path=\"wiki/concepts/networks.md\" type=\"concept\">\n"
        "---\n"
        "tags: [grounded]\n"
        "---\n"
        "\n"
        "# Networks\n"
        "\n"
        "Networks (plural) is a category encompassing many specific "
        "network kinds. The evidence describes several examples and "
        "their use cases. While related, this page title does not "
        "match the singular link target and would leave the link "
        "broken under the resolver's asymmetric normalize rules.\n"
        "</page>\n"
    )
    fake = FakeLLM(response_text=plural_response)
    ctx = _ctx(pages=[src_page], wiki_root=wiki_root, llm=fake, enable_llm=True)
    report = await run_lint_propose(
        report=LintReport(issues=[issue]),
        rule="broken_wikilink",
        limit=10,
        ctx=ctx,
        reporter=_ListReporter(),
        registry={"broken_wikilink": BrokenWikilinkFixer()},  # type: ignore[dict-item]
    )

    assert report.proposals == []
    reasons = _orchestrator_skipped_reasons(report)
    assert len(reasons) == 1
    assert reasons[0].startswith("rejected_title_mismatch"), reasons


@pytest.mark.asyncio
async def test_broken_wikilink_llm_grounded_preserves_evidence_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The grounded create_page op must cite the D-layer evidence
    chunks' source paths in its ``sources:`` frontmatter, not the
    K-layer page that contained the broken wikilink. Without this
    override, ``parse_synthesis_response`` would stamp the referrer
    path on every generated stub, breaking source traceability for
    pages that were specifically built from D-layer evidence."""
    wiki_root, src_page, issue = _make_broken_link_setup(tmp_path)
    hits = _grounded_evidence_hits()  # two hits with paths sources/foo.md and sources/bar.md
    _patch_evidence(monkeypatch, hits)
    fake = FakeLLM(response_text=_GROUNDED_LLM_RESPONSE)
    fixer = BrokenWikilinkFixer()
    proposal = await fixer.propose(
        issue,
        _ctx(pages=[src_page], wiki_root=wiki_root, llm=fake, enable_llm=True),
        reporter=_NullReporter(),
    )

    assert proposal is not None
    op = proposal.operations[0]
    assert op.new_frontmatter is not None
    op_sources = op.new_frontmatter.get("sources", [])
    assert isinstance(op_sources, list)
    # Must cite the D-layer evidence paths, NOT the K-page referrer.
    evidence_paths = {h.path for h in hits if h.path}
    assert set(op_sources) == evidence_paths, (
        f"expected sources = {evidence_paths!r}, got {op_sources!r}"
    )
    assert issue.path not in op_sources


@pytest.mark.asyncio
async def test_broken_wikilink_llm_rejects_too_short_body(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One-sentence filler ("Topic A is a topic.") would technically
    resolve the wikilink but adds no knowledge. Body-length floor
    rejects it with ``rejected_body_too_short`` so the link stays
    flagged as broken."""
    wiki_root, src_page, issue = _make_broken_link_setup(tmp_path)
    _patch_evidence(monkeypatch, _grounded_evidence_hits())
    fake = FakeLLM(response_text=_SHORT_LLM_RESPONSE)

    ctx = _ctx(pages=[src_page], wiki_root=wiki_root, llm=fake, enable_llm=True)
    report = await run_lint_propose(
        report=LintReport(issues=[issue]),
        rule="broken_wikilink",
        limit=10,
        ctx=ctx,
        reporter=_ListReporter(),
        registry={"broken_wikilink": BrokenWikilinkFixer()},  # type: ignore[dict-item]
    )

    assert report.proposals == []
    reasons = _orchestrator_skipped_reasons(report)
    assert len(reasons) == 1
    assert reasons[0].startswith("rejected_body_too_short"), reasons


@pytest.mark.asyncio
async def test_broken_wikilink_llm_disabled_returns_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When ``enable_llm`` is False, a heuristic miss must still return
    None — the LLM path is opt-in, and the evidence pipeline must NOT
    even run (no storage hits, no embedder spend)."""
    wiki_root, src_page, issue = _make_broken_link_setup(tmp_path)
    counter = _patch_evidence(monkeypatch, _grounded_evidence_hits())
    fake = FakeLLM(response_text=_GROUNDED_LLM_RESPONSE)
    fixer = BrokenWikilinkFixer()
    proposal = await fixer.propose(
        issue,
        _ctx(pages=[src_page], wiki_root=wiki_root, llm=fake, enable_llm=False),
        reporter=_NullReporter(),
    )
    assert proposal is None
    # Neither the evidence pipeline nor the LLM should fire when the
    # user hasn't opted in.
    assert counter["calls"] == 0
    assert fake.last_user is None


@pytest.mark.asyncio
async def test_broken_wikilink_llm_failure_returns_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An LLM exception in the grounded path must not fail the whole
    propose task — return None so the orchestrator records a soft skip.

    We deliberately do NOT route this through FixerSkip: provider outages
    are operational noise, not product semantics. The existing
    ``safe_synthesize_pages`` soft-failure contract keeps logging them
    via reporter; the proposal report's ``skipped`` field stays clean
    of "LLM call failed" entries that would distract from the
    evidence-vs-quality decisions agents care about."""
    wiki_root, src_page, issue = _make_broken_link_setup(tmp_path)
    _patch_evidence(monkeypatch, _grounded_evidence_hits())

    class _RaisingLLM:
        async def complete(self, **kwargs: Any) -> Any:
            raise RuntimeError("simulated LLM outage")

    fixer = BrokenWikilinkFixer()
    proposal = await fixer.propose(
        issue,
        _ctx(
            pages=[src_page],
            wiki_root=wiki_root,
            llm=_RaisingLLM(),
            enable_llm=True,
        ),
        reporter=_NullReporter(),
    )
    assert proposal is None


@pytest.mark.asyncio
async def test_broken_wikilink_llm_strips_alias_and_anchor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``[[Target|label]]`` / ``[[Target#section]]`` resolve against a
    page titled ``Target``. The grounded page MUST be built around the
    bare canonical name; otherwise the LLM titles the page with the
    suffix (``Target|label``) and the next lint pass keeps reporting
    the wikilink as broken."""
    wiki_root, src_page, _ = _make_broken_link_setup(
        tmp_path, broken_target="Whole New Topic|Custom Label"
    )
    _patch_evidence(monkeypatch, _grounded_evidence_hits())
    fake = FakeLLM(response_text=_GROUNDED_LLM_RESPONSE)
    issue = LintIssue(
        kind="broken_wikilink",
        path=src_page.path,
        detail="[[Whole New Topic|Custom Label]] has no matching wiki page",
        line=3,
    )
    fixer = BrokenWikilinkFixer()
    proposal = await fixer.propose(
        issue,
        _ctx(pages=[src_page], wiki_root=wiki_root, llm=fake, enable_llm=True),
        reporter=_NullReporter(),
    )
    assert proposal is not None
    # Prompt must inject the bare canonical target — `[[Whole New Topic]]`
    # — even though the source body the prompt also embeds still
    # contains the raw `[[Whole New Topic|Custom Label]]` reference.
    assert fake.last_user is not None
    assert "[[Whole New Topic]]" in fake.last_user
    assert "[[Whole New Topic|Custom Label]]" in fake.last_user  # in source_context
    assert proposal.rationale.endswith("'[[Whole New Topic]]'")

    # Same check for `#anchor` syntax.
    fake_anchor = FakeLLM(response_text=_GROUNDED_LLM_RESPONSE)
    issue_anchor = LintIssue(
        kind="broken_wikilink",
        path=src_page.path,
        detail="[[Whole New Topic#background]] has no matching wiki page",
        line=3,
    )
    proposal_anchor = await fixer.propose(
        issue_anchor,
        _ctx(
            pages=[src_page], wiki_root=wiki_root,
            llm=fake_anchor, enable_llm=True,
        ),
        reporter=_NullReporter(),
    )
    assert proposal_anchor is not None
    assert fake_anchor.last_user is not None
    assert "[[Whole New Topic]]" in fake_anchor.last_user
    assert "[[Whole New Topic#background]]" not in fake_anchor.last_user.split(
        "Broken wikilink target:", 1
    )[1].split("\n", 1)[0]


@pytest.mark.asyncio
async def test_broken_wikilink_llm_unparseable_returns_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """LLM emits no usable ``<page>`` block (e.g. apologies / refusal):
    fixer must skip rather than synthesise an empty page. Treated as a
    soft failure (same channel as provider outages), not a FixerSkip —
    the grounded prompt explicitly offers a ``REFUSE: ...`` exit when
    evidence is insufficient, which lands here."""
    wiki_root, src_page, issue = _make_broken_link_setup(tmp_path)
    _patch_evidence(monkeypatch, _grounded_evidence_hits())
    fake = FakeLLM(response_text="REFUSE: insufficient evidence")
    fixer = BrokenWikilinkFixer()
    proposal = await fixer.propose(
        issue,
        _ctx(pages=[src_page], wiki_root=wiki_root, llm=fake, enable_llm=True),
        reporter=_NullReporter(),
    )
    assert proposal is None


# --- PR2: non_atomic_page fixer ---------------------------------------------


_NON_ATOMIC_LLM_RESPONSE = (
    "<page path=\"wiki/concepts/topic-a.md\" type=\"concept\">\n"
    "---\n"
    "tags: [child]\n"
    "---\n"
    "\n"
    "# Topic A\n"
    "\n"
    "First atomic child page.\n"
    "</page>\n"
    "\n"
    "<page path=\"wiki/concepts/topic-b.md\" type=\"concept\">\n"
    "---\n"
    "tags: [child]\n"
    "---\n"
    "\n"
    "# Topic B\n"
    "\n"
    "Second atomic child page.\n"
    "</page>\n"
)


def _make_fat_page_on_disk(tmp_path: Path) -> tuple[Path, Any, Any]:
    """Write a non-atomic K-page to disk + the matching ``LintIssue``."""
    wiki_root = tmp_path
    page_path = "wiki/concepts/grab-bag.md"
    abs_path = wiki_root / page_path
    abs_path.parent.mkdir(parents=True, exist_ok=True)
    body = (
        "---\n"
        "id: K-grab\n"
        "type: concept\n"
        "title: Grab Bag\n"
        "---\n"
        "\n"
        "# Grab Bag\n"
        "\n"
        "## Topic A\n\nFirst topic discussion.\n\n"
        "## Topic B\n\nSecond topic discussion.\n"
    )
    abs_path.write_text(body, encoding="utf-8")
    page = _make_page("Grab Bag", body)  # only used for ctx.all_pages
    issue = LintIssue(
        kind="non_atomic_page",
        path=page_path,
        detail="page looks like multiple atomic notes glued together: 2 H2 sections",
        line=None,
    )
    return wiki_root, page, issue


@pytest.mark.asyncio
async def test_non_atomic_page_splits_into_n_create_plus_one_delete(
    tmp_path: Path,
) -> None:
    """Happy path: LLM emits 2 child pages → fixer proposes 2 create_page
    + 1 delete_page (original)."""
    from dikw_core.domains.knowledge.lint_fixers.non_atomic_page import (
        NonAtomicPageFixer,
    )

    wiki_root, page, issue = _make_fat_page_on_disk(tmp_path)
    fake = FakeLLM(response_text=_NON_ATOMIC_LLM_RESPONSE)
    fixer = NonAtomicPageFixer()
    proposal = await fixer.propose(
        issue,
        _ctx(
            pages=[page],
            wiki_root=wiki_root,
            llm=fake,
            enable_llm=True,
            cfg=_default_cfg(),
        ),
        reporter=_NullReporter(),
    )

    assert proposal is not None
    assert proposal.source == "llm"
    assert proposal.issue_kind == "non_atomic_page"
    creates = [op for op in proposal.operations if op.kind == "create_page"]
    deletes = [op for op in proposal.operations if op.kind == "delete_page"]
    assert len(creates) == 2, "LLM emitted 2 child pages"
    assert len(deletes) == 1, "original page must be deleted"
    assert deletes[0].path == issue.path
    assert deletes[0].expected_hash is not None  # concurrent-edit guard
    # External wikilinks are intentionally NOT rewritten — the design
    # leaves [[Grab Bag]] in other pages to be fixed by a follow-up
    # broken_wikilink propose run (decision A in the plan).
    paths = {op.path for op in creates}
    assert "wiki/concepts/grab-bag.md" not in paths
    assert all(p.startswith("wiki/") and p.endswith(".md") for p in paths)


@pytest.mark.asyncio
async def test_non_atomic_page_skips_when_synth_returns_one(tmp_path: Path) -> None:
    """If the LLM only finds one atomic page in the body, splitting is
    pointless — fixer returns None."""
    from dikw_core.domains.knowledge.lint_fixers.non_atomic_page import (
        NonAtomicPageFixer,
    )

    wiki_root, page, issue = _make_fat_page_on_disk(tmp_path)
    one_page_response = (
        "<page path=\"wiki/concepts/single-atomic.md\" type=\"concept\">\n"
        "---\ntags: [single]\n---\n\n# Single Atomic\n\nOnly one.\n"
        "</page>\n"
    )
    fake = FakeLLM(response_text=one_page_response)
    fixer = NonAtomicPageFixer()
    proposal = await fixer.propose(
        issue,
        _ctx(
            pages=[page],
            wiki_root=wiki_root,
            llm=fake,
            enable_llm=True,
            cfg=_default_cfg(),
        ),
        reporter=_NullReporter(),
    )
    assert proposal is None


@pytest.mark.asyncio
async def test_non_atomic_page_skips_on_synth_error(tmp_path: Path) -> None:
    """Hard parse failure (no <page> blocks) → fixer returns None
    rather than failing the whole propose task."""
    from dikw_core.domains.knowledge.lint_fixers.non_atomic_page import (
        NonAtomicPageFixer,
    )

    wiki_root, page, issue = _make_fat_page_on_disk(tmp_path)
    fake = FakeLLM(response_text="LLM refused to split.")
    fixer = NonAtomicPageFixer()
    proposal = await fixer.propose(
        issue,
        _ctx(
            pages=[page],
            wiki_root=wiki_root,
            llm=fake,
            enable_llm=True,
            cfg=_default_cfg(),
        ),
        reporter=_NullReporter(),
    )
    assert proposal is None


_TRUNCATED_NON_ATOMIC_RESPONSE = (
    "<page path=\"wiki/concepts/topic-a.md\" type=\"concept\">\n"
    "---\ntags: [child]\n---\n\n# Topic A\n\nFirst child.\n"
    "</page>\n\n"
    "<page path=\"wiki/concepts/topic-b.md\" type=\"concept\">\n"
    "---\ntags: [child]\n---\n\n# Topic B\n\nSecond child.\n"
    "</page>\n\n"
    # Third <page> opener with no </page> — synth marks this retry=True.
    "<page path=\"wiki/concepts/topic-c.md\" type=\"concept\">\n"
    "---\ntags: [child]\n---\n\n# Topic C\n\nThird child body keeps "
    "going but max_tokens cut it off here."
)


@pytest.mark.asyncio
async def test_non_atomic_page_refuses_truncated_split(tmp_path: Path) -> None:
    """A response with two complete <page> blocks plus an unclosed third
    is a SynthesisPartialError(retry=True). The destructive non_atomic_page
    fixer must NOT accept it — applying would delete the original page
    and silently drop Topic C's content with it."""
    from dikw_core.domains.knowledge.lint_fixers.non_atomic_page import (
        NonAtomicPageFixer,
    )

    wiki_root, page, issue = _make_fat_page_on_disk(tmp_path)
    fake = FakeLLM(response_text=_TRUNCATED_NON_ATOMIC_RESPONSE)
    fixer = NonAtomicPageFixer()
    proposal = await fixer.propose(
        issue,
        _ctx(
            pages=[page],
            wiki_root=wiki_root,
            llm=fake,
            enable_llm=True,
            cfg=_default_cfg(),
        ),
        reporter=_NullReporter(),
    )
    assert proposal is None, (
        "truncated split must be refused; otherwise apply deletes the "
        "original after creating only the complete subset"
    )


_DETERMINISTIC_PARTIAL_RESPONSE = (
    "<page path=\"wiki/concepts/topic-a.md\" type=\"concept\">\n"
    "---\ntags: [child]\n---\n\n# Topic A\n\nFirst child.\n"
    "</page>\n\n"
    "<page path=\"wiki/concepts/topic-b.md\" type=\"concept\">\n"
    "---\ntags: [child]\n---\n\n# Topic B\n\nSecond child.\n"
    "</page>\n\n"
    # A third complete <page> block whose body is missing the required
    # ATX `# Title` line — parse_synthesis_response treats it as a
    # deterministic partial (retry=False).
    "<page path=\"wiki/concepts/topic-c.md\" type=\"concept\">\n"
    "---\ntags: [child]\n---\n\nNo title heading here, just prose.\n"
    "</page>\n"
)


@pytest.mark.asyncio
async def test_non_atomic_page_refuses_deterministic_partial(
    tmp_path: Path,
) -> None:
    """Even when the partial parse is deterministic (one malformed block
    among valid ones, retry=False), the destructive non_atomic_page
    fixer must refuse — accepting "2 valid + 1 dropped" would delete
    the original and silently lose Topic C's intended content."""
    from dikw_core.domains.knowledge.lint_fixers.non_atomic_page import (
        NonAtomicPageFixer,
    )

    wiki_root, page, issue = _make_fat_page_on_disk(tmp_path)
    fake = FakeLLM(response_text=_DETERMINISTIC_PARTIAL_RESPONSE)
    fixer = NonAtomicPageFixer()
    proposal = await fixer.propose(
        issue,
        _ctx(
            pages=[page],
            wiki_root=wiki_root,
            llm=fake,
            enable_llm=True,
            cfg=_default_cfg(),
        ),
        reporter=_NullReporter(),
    )
    assert proposal is None, (
        "deterministic partial split must be refused in strict mode"
    )


@pytest.mark.asyncio
async def test_non_atomic_page_aborts_on_child_collision(
    tmp_path: Path,
) -> None:
    """If ANY child path collides with an existing K-page, the fixer
    must abort the whole split — silently filtering the colliding
    child would still emit delete_page for the original, dropping the
    colliding child's content with it. User resolves by hand."""
    from dikw_core.domains.knowledge.lint_fixers.non_atomic_page import (
        NonAtomicPageFixer,
    )

    wiki_root, page, issue = _make_fat_page_on_disk(tmp_path)
    # An existing K-page at the path the LLM's first child would claim.
    colliding_existing = _make_page("Topic A", "# Topic A\nexisting body\n")
    fake = FakeLLM(response_text=_NON_ATOMIC_LLM_RESPONSE)
    fixer = NonAtomicPageFixer()
    proposal = await fixer.propose(
        issue,
        _ctx(
            # Include the colliding page in ctx.all_pages so the fixer
            # sees it via the ``existing_paths`` set.
            pages=[page, colliding_existing],
            wiki_root=wiki_root,
            llm=fake,
            enable_llm=True,
            cfg=_default_cfg(),
        ),
        reporter=_NullReporter(),
    )
    assert proposal is None, (
        "any child-path collision must abort the whole non_atomic_page split"
    )


@pytest.mark.asyncio
async def test_non_atomic_page_skips_when_llm_disabled(tmp_path: Path) -> None:
    """Without ``enable_llm``, the fixer cannot run (no heuristic-only
    path makes sense for this rule) — must skip silently."""
    from dikw_core.domains.knowledge.lint_fixers.non_atomic_page import (
        NonAtomicPageFixer,
    )

    wiki_root, page, issue = _make_fat_page_on_disk(tmp_path)
    fake = FakeLLM(response_text=_NON_ATOMIC_LLM_RESPONSE)
    fixer = NonAtomicPageFixer()
    proposal = await fixer.propose(
        issue,
        _ctx(
            pages=[page],
            wiki_root=wiki_root,
            llm=fake,
            enable_llm=False,
            cfg=_default_cfg(),
        ),
        reporter=_NullReporter(),
    )
    assert proposal is None
    assert fake.last_user is None  # no LLM call


# --- CJK short-target regression --------------------------------------------
#
# ``_MIN_TARGET_LEN = 4`` is a fuzzy-match guardrail (3-char ASCII
# substrings hit 0.85 against too many titles by chance). The early
# version of the fixer enforced it at the top of ``propose()``, before
# the LLM-stub branch — which silently dropped 2-3 char CJK targets
# like ``[[秦朝]]`` even when ``--enable-llm`` was set. The fix moves
# the gate to the heuristic branch only; the LLM path stays gated by
# ``enable_llm`` alone.

_CJK_GROUNDED_LLM_RESPONSE = (
    "<page path=\"wiki/concepts/qin-dynasty.md\" type=\"concept\">\n"
    "---\n"
    "tags: [history]\n"
    "sources: [\"sources/foo.md\"]\n"
    "---\n"
    "\n"
    "# 秦朝\n"
    "\n"
    "秦朝是中国历史上第一个统一的中央集权王朝,根据证据所述,公元前 221 "
    "年由秦始皇建立。证据描述了郡县制取代分封制、统一度量衡与文字、修筑长城 "
    "等关键举措。这些制度奠定了此后两千余年中国官僚体系的基础,虽然王朝本身 "
    "国祚短暂,仅历两世便告倾覆。秦朝还推行了书同文、车同轨,统一货币与度量衡, "
    "并大规模修建驰道与水利工程,为后世留下了深远影响。考古发现表明,秦代陵墓 "
    "形制与兵马俑数量也反映出当时中央集权对人力与资源的高度调动。\n"
    "</page>\n"
)


@pytest.mark.asyncio
async def test_broken_wikilink_short_cjk_target_enters_llm_when_enabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``[[秦朝]]`` is 2 CJK chars — below the 4-char heuristic gate.
    With ``enable_llm=True`` AND sufficient D/I evidence the grounded
    LLM path MUST still fire; the heuristic length gate guards fuzzy
    match, not the LLM path."""
    wiki_root, src_page, issue = _make_broken_link_setup(
        tmp_path, broken_target="秦朝"
    )
    _patch_evidence(monkeypatch, _grounded_evidence_hits())
    fake = FakeLLM(response_text=_CJK_GROUNDED_LLM_RESPONSE)
    fixer = BrokenWikilinkFixer()
    proposal = await fixer.propose(
        issue,
        _ctx(pages=[src_page], wiki_root=wiki_root, llm=fake, enable_llm=True),
        reporter=_NullReporter(),
    )

    assert proposal is not None, (
        "grounded LLM path must fire for short CJK targets when enabled"
    )
    assert proposal.source == "llm"
    assert proposal.operations[0].kind == "create_page"
    # FakeLLM captured the prompt — the broken target was injected.
    assert fake.last_user is not None
    assert "秦朝" in fake.last_user


@pytest.mark.asyncio
async def test_broken_wikilink_short_cjk_target_skipped_when_llm_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same short CJK target with ``enable_llm=False``: propose returns
    None and the LLM is NEVER called — confirms the relaxed gate didn't
    accidentally enable the LLM path unconditionally."""
    wiki_root, src_page, issue = _make_broken_link_setup(
        tmp_path, broken_target="秦朝"
    )
    counter = _patch_evidence(monkeypatch, _grounded_evidence_hits())
    fake = FakeLLM(response_text=_CJK_GROUNDED_LLM_RESPONSE)
    fixer = BrokenWikilinkFixer()
    proposal = await fixer.propose(
        issue,
        _ctx(pages=[src_page], wiki_root=wiki_root, llm=fake, enable_llm=False),
        reporter=_NullReporter(),
    )

    assert proposal is None
    assert counter["calls"] == 0
    assert fake.last_user is None
