"""HTTP-level tests for ``POST /v1/synth``.

Goes through the ``TaskManager`` plumbing exercised in
``test_ingest_task.py``; this file focuses on the synth-specific event
vocabulary + final shape rather than re-testing event tape replay or
cancellation (already covered for ingest).
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import httpx
import pytest

from dikw_core import api as api_module
from dikw_core.providers import LLMResponse
from dikw_core.server import synth_op as synth_op_module

from ..fakes import FakeEmbeddings, FakeLLM
from .conftest import wait_task_terminal as _wait_terminal

FIXTURES = Path(__file__).parent.parent / "fixtures" / "notes"


def _patch_synth_factories(
    monkeypatch: pytest.MonkeyPatch, *, llm: FakeLLM
) -> None:
    monkeypatch.setattr(synth_op_module, "build_llm", lambda _cfg, **_kw: llm)
    monkeypatch.setattr(
        synth_op_module, "build_embedder", lambda _cfg: FakeEmbeddings()
    )


class _ScriptedSynthLLM:
    """Returns one canned ``<page>`` block per source path, matched by
    substring against the user prompt body."""

    def __init__(self, by_source: dict[str, str]) -> None:
        self._by_source = by_source

    async def complete(
        self, *, system: str, user: str, model: str, **_: Any
    ) -> LLMResponse:
        for src_path, body in self._by_source.items():
            if src_path in user:
                return LLMResponse(text=body, finish_reason="end_turn")
        raise AssertionError(f"no scripted page for prompt: {user[:200]}")

    def complete_stream(self, **_: Any) -> Any:
        raise NotImplementedError


class _ScriptedSynthAndJudgeLLM(_ScriptedSynthLLM):
    """A synth scripted LLM that also answers the entailment judge — synth
    prompts return a canned page, entailment prompts a canned verdict."""

    def __init__(self, by_source: dict[str, str], *, verdict: str = "yes") -> None:
        super().__init__(by_source)
        self._verdict = verdict

    async def complete(
        self, *, system: str, user: str, model: str, **_: Any
    ) -> LLMResponse:
        if "entailment judge" in system.lower():
            return LLMResponse(
                text=f'{{"verdict": "{self._verdict}"}}', finish_reason="end_turn"
            )
        return await super().complete(system=system, user=user, model=model)


# ---- synth -------------------------------------------------------------


@pytest.mark.asyncio
async def test_synth_task_emits_per_source_progress_and_final_report(
    server_client: httpx.AsyncClient,
    base_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Seed three source markdown files + ingest them so synth has
    # documents to process.
    dest = base_root / "sources" / "notes"
    dest.mkdir(parents=True, exist_ok=True)
    for src in FIXTURES.glob("*.md"):
        shutil.copy2(src, dest / src.name)
    await api_module.ingest(base_root, embedder=FakeEmbeddings())

    script = {
        "sources/notes/karpathy-wiki.md": (
            '<page path="knowledge/concepts/karpathy.md" type="concept">\n'
            "---\ntags: [karpathy]\n---\n\n"
            "# Karpathy\n\nDeterministic scoping matters.\n"
            "</page>"
        ),
        "sources/notes/dikw.md": (
            '<page path="knowledge/concepts/dikw.md" type="concept">\n'
            "---\ntags: [dikw]\n---\n\n"
            "# DIKW\n\nFour layers stacked.\n"
            "</page>"
        ),
        "sources/notes/retrieval.md": (
            '<page path="knowledge/concepts/retrieval.md" type="concept">\n'
            "---\ntags: [retrieval]\n---\n\n"
            "# Retrieval\n\nRRF fuses BM25 with dense.\n"
            "</page>"
        ),
    }
    _patch_synth_factories(
        monkeypatch, llm=FakeLLM()  # placeholder, overridden below
    )
    # Override build_llm to return the scripted stub for synth.
    monkeypatch.setattr(
        synth_op_module, "build_llm", lambda _cfg, **_kw: _ScriptedSynthLLM(script)
    )

    submit = await server_client.post(
        "/v1/synth", json={"force_all": True, "no_embed": False}
    )
    assert submit.status_code == 200, submit.text
    handle = submit.json()
    assert handle["op"] == "synth"
    task_id = handle["task_id"]

    row = await _wait_terminal(server_client, task_id, timeout=15.0)
    assert row["status"] == "succeeded", row

    result = (await server_client.get(f"/v1/tasks/{task_id}/result")).json()[
        "result"
    ]
    # SynthReport fields land verbatim in the final result.
    assert result["candidates"] == 3
    assert result["created"] == 3
    assert result["errors"] == 0

    # Event tape carries one progress event per source, all phase=synth.
    resp = await server_client.get(
        f"/v1/tasks/{task_id}/events",
        params={"from_seq": 0, "limit": 1000, "wait": 0},
    )
    events = resp.json()["events"]
    synth_progress = [
        e for e in events if e["type"] == "progress" and e["phase"] == "synth"
    ]
    assert len(synth_progress) == 3
    assert {e["detail"]["outcome"] for e in synth_progress} == {"created"}
    assert events[-1]["type"] == "final" and events[-1]["status"] == "succeeded"


@pytest.mark.asyncio
async def test_synth_verify_returns_verify_section(
    server_client: httpx.AsyncClient,
    base_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``POST /v1/synth {verify: true}`` folds a ``SynthVerifyReport`` into the
    task result under ``verify`` — the gated booleans (``passed`` / leg-ok) and
    the duplicate leg (run because the server builds a real embedder)."""
    dest = base_root / "sources" / "notes"
    dest.mkdir(parents=True, exist_ok=True)
    for src in FIXTURES.glob("*.md"):
        shutil.copy2(src, dest / src.name)
    await api_module.ingest(base_root, embedder=FakeEmbeddings())

    script = {
        "sources/notes/karpathy-wiki.md": (
            '<page category="concept" slug="karpathy">\n'
            "---\ntags: [karpathy]\n---\n\n"
            "# Karpathy\n\nDeterministic scoping beats probabilistic guessing.\n"
            "</page>"
        ),
        "sources/notes/dikw.md": (
            '<page category="concept" slug="dikw">\n'
            "---\ntags: [dikw]\n---\n\n"
            "# DIKW\n\nData becomes information becomes knowledge becomes wisdom.\n"
            "</page>"
        ),
        "sources/notes/retrieval.md": (
            '<page category="concept" slug="retrieval">\n'
            "---\ntags: [retrieval]\n---\n\n"
            "# Retrieval\n\nReciprocal rank fusion blends sparse and dense hits.\n"
            "</page>"
        ),
    }
    _patch_synth_factories(monkeypatch, llm=FakeLLM())
    monkeypatch.setattr(
        synth_op_module, "build_llm", lambda _cfg, **_kw: _ScriptedSynthLLM(script)
    )

    submit = await server_client.post(
        "/v1/synth", json={"force_all": True, "no_embed": False, "verify": True}
    )
    assert submit.status_code == 200, submit.text
    task_id = submit.json()["task_id"]

    row = await _wait_terminal(server_client, task_id, timeout=15.0)
    assert row["status"] == "succeeded", row

    result = (await server_client.get(f"/v1/tasks/{task_id}/result")).json()[
        "result"
    ]
    verify = result["verify"]
    assert verify is not None
    assert verify["pages_checked"] == 3
    # Distinct bodies, no broken links → clean gated legs; the duplicate leg
    # ran (the server builds a real embedder).
    assert verify["duplicate_checked"] is True
    assert verify["lint_ok"] is True
    assert verify["persist_ok"] is True
    assert verify["passed"] is True


@pytest.mark.asyncio
async def test_synth_verify_judge_returns_grounding_section(
    server_client: httpx.AsyncClient,
    base_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``POST /v1/synth {verify, judge}`` threads ``judge`` through
    SynthSubmit → make_synth_runner → api.synthesize and returns the report-only
    grounding leg in the serialized result. This is the only path the CLI
    exercises, so it pins the server plumbing the engine tests can't."""
    dest = base_root / "sources" / "notes"
    dest.mkdir(parents=True, exist_ok=True)
    for src in FIXTURES.glob("*.md"):
        shutil.copy2(src, dest / src.name)
    await api_module.ingest(base_root, embedder=FakeEmbeddings())

    # Pages carry ``sources:`` so the grounding leg has provenance to follow.
    script = {
        "sources/notes/karpathy-wiki.md": (
            '<page category="concept" slug="karpathy">\n'
            "---\ntags: [karpathy]\nsources: [sources/notes/karpathy-wiki.md]\n---\n\n"
            "# Karpathy\n\nDeterministic scoping beats probabilistic guessing.\n"
            "</page>"
        ),
        "sources/notes/dikw.md": (
            '<page category="concept" slug="dikw">\n'
            "---\ntags: [dikw]\nsources: [sources/notes/dikw.md]\n---\n\n"
            "# DIKW\n\nData becomes information becomes knowledge becomes wisdom.\n"
            "</page>"
        ),
        "sources/notes/retrieval.md": (
            '<page category="concept" slug="retrieval">\n'
            "---\ntags: [retrieval]\nsources: [sources/notes/retrieval.md]\n---\n\n"
            "# Retrieval\n\nReciprocal rank fusion blends sparse and dense hits.\n"
            "</page>"
        ),
    }
    _patch_synth_factories(monkeypatch, llm=FakeLLM())
    monkeypatch.setattr(
        synth_op_module,
        "build_llm",
        lambda _cfg, **_kw: _ScriptedSynthAndJudgeLLM(script, verdict="yes"),
    )

    submit = await server_client.post(
        "/v1/synth",
        json={"force_all": True, "no_embed": False, "verify": True, "judge": True},
    )
    assert submit.status_code == 200, submit.text
    task_id = submit.json()["task_id"]

    row = await _wait_terminal(server_client, task_id, timeout=15.0)
    assert row["status"] == "succeeded", row

    result = (await server_client.get(f"/v1/tasks/{task_id}/result")).json()[
        "result"
    ]
    verify = result["verify"]
    assert verify is not None
    assert verify["grounding_requested"] is True
    assert verify["grounding_checked"] is True
    assert verify["grounding_n_judged"] > 0
    # Canned "yes" judge → ratio 1.0, and it is report-only: passed unaffected.
    assert verify["grounding_entailment_ratio"] == pytest.approx(1.0)
    assert verify["passed"] is True


@pytest.mark.asyncio
async def test_synth_without_verify_has_null_verify_section(
    server_client: httpx.AsyncClient,
    base_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A default synth (no ``verify``) leaves ``result.verify`` null — the
    post-pass is strictly opt-in."""
    dest = base_root / "sources" / "notes"
    dest.mkdir(parents=True, exist_ok=True)
    for src in FIXTURES.glob("*.md"):
        shutil.copy2(src, dest / src.name)
    await api_module.ingest(base_root, embedder=FakeEmbeddings())

    script = {
        "sources/notes/karpathy-wiki.md": (
            '<page category="concept" slug="karpathy">\n---\n---\n\n'
            "# Karpathy\n\nScoping is deterministic.\n</page>"
        ),
        "sources/notes/dikw.md": (
            '<page category="concept" slug="dikw">\n---\n---\n\n'
            "# DIKW\n\nFour layers stacked.\n</page>"
        ),
        "sources/notes/retrieval.md": (
            '<page category="concept" slug="retrieval">\n---\n---\n\n'
            "# Retrieval\n\nRRF fuses BM25 with dense.\n</page>"
        ),
    }
    _patch_synth_factories(monkeypatch, llm=FakeLLM())
    monkeypatch.setattr(
        synth_op_module, "build_llm", lambda _cfg, **_kw: _ScriptedSynthLLM(script)
    )

    submit = await server_client.post(
        "/v1/synth", json={"force_all": True, "no_embed": False}
    )
    assert submit.status_code == 200, submit.text
    task_id = submit.json()["task_id"]
    row = await _wait_terminal(server_client, task_id, timeout=15.0)
    assert row["status"] == "succeeded", row

    result = (await server_client.get(f"/v1/tasks/{task_id}/result")).json()[
        "result"
    ]
    assert result["verify"] is None


@pytest.mark.asyncio
async def test_synth_verify_populated_report_survives_asdict(
    server_client: httpx.AsyncClient,
    base_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A FAILing verify with a non-empty `lint_findings` (tuple of nested frozen
    `SynthVerifyLintFinding`) must survive `dataclasses.asdict` on the server
    task path and reach the engine-free client as a list of plain dicts. Pins
    the nested-tuple serialization contract a future @property regression would
    break."""
    dest = base_root / "sources" / "notes"
    dest.mkdir(parents=True, exist_ok=True)
    for src in FIXTURES.glob("*.md"):
        shutil.copy2(src, dest / src.name)
    await api_module.ingest(base_root, embedder=FakeEmbeddings())

    # One page carries a dangling [[Ghost Reference]] nobody authors → the
    # broken_wikilink lint leg fails verify.
    script = {
        "sources/notes/dikw.md": (
            '<page category="concept" slug="dikw">\n---\n---\n\n'
            "# DIKW\n\nFour layers. See [[Ghost Reference]] which is dangling.\n"
            "</page>"
        ),
        "sources/notes/karpathy-wiki.md": (
            '<page category="concept" slug="karpathy">\n---\n---\n\n'
            "# Karpathy\n\nDeterministic scoping.\n</page>"
        ),
        "sources/notes/retrieval.md": (
            '<page category="concept" slug="retrieval">\n---\n---\n\n'
            "# Retrieval\n\nRRF fuses BM25 with dense.\n</page>"
        ),
    }
    _patch_synth_factories(monkeypatch, llm=FakeLLM())
    monkeypatch.setattr(
        synth_op_module, "build_llm", lambda _cfg, **_kw: _ScriptedSynthLLM(script)
    )

    submit = await server_client.post(
        "/v1/synth", json={"force_all": True, "no_embed": False, "verify": True}
    )
    task_id = submit.json()["task_id"]
    row = await _wait_terminal(server_client, task_id, timeout=15.0)
    assert row["status"] == "succeeded", row

    verify = (await server_client.get(f"/v1/tasks/{task_id}/result")).json()[
        "result"
    ]["verify"]
    assert verify["passed"] is False
    assert verify["lint_ok"] is False
    findings = verify["lint_findings"]
    assert isinstance(findings, list) and findings
    assert any(f["kind"] == "broken_wikilink" for f in findings)
    # Each finding is a plain dict (the nested frozen dataclass flattened).
    assert all({"kind", "path", "detail"} <= set(f) for f in findings)


@pytest.mark.asyncio
async def test_synth_verify_no_embed_loud_skips_duplicate_leg(
    server_client: httpx.AsyncClient,
    base_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`no_embed=True` leaves the runner without an embedder, so the duplicate
    leg is skipped: `duplicate_checked` is False and `duplicate_ratio` survives
    asdict as JSON null — never silently passed off as measured."""
    dest = base_root / "sources" / "notes"
    dest.mkdir(parents=True, exist_ok=True)
    for src in FIXTURES.glob("*.md"):
        shutil.copy2(src, dest / src.name)
    await api_module.ingest(base_root, embedder=FakeEmbeddings())

    script = {
        "sources/notes/dikw.md": (
            '<page category="concept" slug="dikw">\n---\n---\n\n'
            "# DIKW\n\nFour layers stacked.\n</page>"
        ),
        "sources/notes/karpathy-wiki.md": (
            '<page category="concept" slug="karpathy">\n---\n---\n\n'
            "# Karpathy\n\nScoping is deterministic.\n</page>"
        ),
        "sources/notes/retrieval.md": (
            '<page category="concept" slug="retrieval">\n---\n---\n\n'
            "# Retrieval\n\nRRF fuses BM25 with dense.\n</page>"
        ),
    }
    _patch_synth_factories(monkeypatch, llm=FakeLLM())
    monkeypatch.setattr(
        synth_op_module, "build_llm", lambda _cfg, **_kw: _ScriptedSynthLLM(script)
    )

    submit = await server_client.post(
        "/v1/synth", json={"force_all": True, "no_embed": True, "verify": True}
    )
    task_id = submit.json()["task_id"]
    row = await _wait_terminal(server_client, task_id, timeout=15.0)
    assert row["status"] == "succeeded", row

    verify = (await server_client.get(f"/v1/tasks/{task_id}/result")).json()[
        "result"
    ]["verify"]
    assert verify["duplicate_checked"] is False
    assert verify["duplicate_ratio"] is None
