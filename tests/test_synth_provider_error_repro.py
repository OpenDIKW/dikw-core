"""Pins issue #134 — single ProviderError must not abort the whole task.

The 0.3.6 ``_synth_pages_from_source`` lets a ``ProviderError`` from
``llm.complete`` (codex empty-response, auth/quota flap, anything raised
by the provider) escape the per-group loop — group N+1 onwards never
runs and the caller (``synthesize``) cannot persist prior groups' work.

The 0.4.0 fix wraps ``llm.complete`` in a bounded per-group retry loop:
on ``ProviderError`` we retry up to ``cfg.synth.provider_error_retries``
times with linear backoff, then skip the group (record as a parse-style
error so ``synth_source_done`` isn't written) and continue with the
next group. This file pins both the happy path (retry-then-succeed)
and the exhausted path (skip-then-continue) so a future refactor
cannot quietly re-introduce the issue-#134 regression.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from dikw_core import api
from dikw_core.config import DikwConfig
from dikw_core.progress import CancelToken
from dikw_core.providers.base import LLMResponse, ProviderError
from dikw_core.schemas import ChunkRecord

from .fakes import make_provider_cfg
from .test_progress_reporter import ListReporter


@dataclass
class FlakyLLM:
    """LLM that raises ``ProviderError`` on the configured call indices.

    Mirrors the codex empty-response failure mode from issue #134:
    ``response.output=None`` + zero text deltas → ``ProviderError``.
    Call counter is global (across all groups), so a test can script
    "raise on call 0 then succeed", "raise twice then succeed",
    "raise forever", etc.
    """

    response_text: str
    raise_on_calls: set[int]
    call_count: int = field(default=0, init=False)

    async def complete(
        self,
        *,
        system: str,
        user: str,
        model: str,
        max_tokens: int = 4096,
        temperature: float = 0.2,
        tools: Any = None,
    ) -> LLMResponse:
        _ = (system, user, model, max_tokens, temperature, tools)
        idx = self.call_count
        self.call_count += 1
        if idx in self.raise_on_calls:
            raise ProviderError(
                "OpenAI codex backend returned response.output=None and "
                "shipped zero text deltas; the SDK reducer fallback has "
                "no partial response to surface."
            )
        return LLMResponse(text=self.response_text, finish_reason="end_turn")


def _three_groups() -> tuple[str, list[ChunkRecord]]:
    sections = [
        "# Section one\n\nAlpha alpha alpha alpha alpha.\n",
        "# Section two\n\nBravo bravo bravo bravo bravo.\n",
        "# Section three\n\nCharlie charlie charlie charlie charlie.\n",
    ]
    body = "".join(sections)
    chunks: list[ChunkRecord] = []
    cursor = 0
    for seq, text in enumerate(sections):
        end = cursor + len(text)
        chunks.append(
            ChunkRecord(
                doc_id="D::sources/multi.md",
                seq=seq,
                start=cursor,
                end=end,
                text=text,
            )
        )
        cursor = end
    return body, chunks


_VALID_PAGE = '<page path="knowledge/x.md" type="concept">\n# X\n\nbody\n</page>'
_TEMPLATE = (
    "src={source_path} body={source_body} idx={group_index}/"
    "{group_total} headings={group_outline} max={max_pages} "
    "types={allowed_types}"
)


def _cfg(*, retries: int = 2, backoff: float = 0.0) -> DikwConfig:
    """Build a config with tiny groups + fast retries for unit tests.

    ``backoff=0.0`` keeps the retry loop instant; the retry-event
    contract test pins the configured value separately.
    """
    cfg = DikwConfig(provider=make_provider_cfg())
    cfg.synth.target_tokens_per_group = 40
    cfg.synth.provider_error_retries = retries
    cfg.synth.provider_error_retry_backoff_seconds = backoff
    return cfg


@pytest.mark.asyncio
async def test_retry_then_succeed_on_provider_error() -> None:
    """Group 2's first attempt raises, retry succeeds → all 3 groups
    process, no errors counted, 4 total LLM calls (1 + 1-retry + 1 + 1).
    """
    body, chunks = _three_groups()
    # Calls: g1 attempt-0 (idx=0, succeed), g2 attempt-0 (idx=1, raise),
    # g2 attempt-1 (idx=2, succeed), g3 attempt-0 (idx=3, succeed).
    llm = FlakyLLM(response_text=_VALID_PAGE, raise_on_calls={1})

    outcome = await api._synth_pages_from_source(
        llm=llm,
        template=_TEMPLATE,
        cfg=_cfg(retries=2),
        source_path="sources/multi.md",
        source_body=body,
        chunks=chunks,
        cancel=CancelToken(),
        reporter=ListReporter(),
    )

    assert llm.call_count == 4, "expected 1 retry on g2"
    assert outcome.groups_processed == 3
    assert outcome.parse_errors == 0
    assert len(outcome.pages) == 3


@pytest.mark.asyncio
async def test_skip_group_after_exhausted_retries_and_continue() -> None:
    """Group 2 raises on EVERY attempt → skip group 2, record error,
    continue with group 3.
    """
    body, chunks = _three_groups()
    # g1 attempt-0 succeeds (idx=0). g2 attempts at idx 1, 2, 3 all
    # raise (retries=2 → 3 attempts total). g3 attempt-0 succeeds at
    # idx=4.
    llm = FlakyLLM(response_text=_VALID_PAGE, raise_on_calls={1, 2, 3})

    outcome = await api._synth_pages_from_source(
        llm=llm,
        template=_TEMPLATE,
        cfg=_cfg(retries=2),
        source_path="sources/multi.md",
        source_body=body,
        chunks=chunks,
        cancel=CancelToken(),
        reporter=ListReporter(),
    )

    assert llm.call_count == 5, (
        "g1 once, g2 three attempts (all raise), g3 once"
    )
    assert outcome.groups_processed == 3
    assert outcome.parse_errors == 1, "the skipped g2 must count as 1 error"
    assert len(outcome.pages) == 2, "only g1 and g3 produced pages"
    # Diagnostic note must include enough to find the source/group.
    joined_notes = "\n".join(outcome.log_notes)
    assert "group 2/3" in joined_notes
    assert "provider" in joined_notes.lower()


@pytest.mark.asyncio
async def test_progress_emits_retrying_and_skipped_events() -> None:
    """Per-group event tape must surface ``retrying`` (one per failed
    attempt that still has retries left) and ``skipped`` (terminal,
    when retries are exhausted) so NDJSON consumers can render the
    state distinct from a clean ``error``.
    """
    body, chunks = _three_groups()
    # g1 succeeds, g2 always raises (3 attempts), g3 succeeds.
    llm = FlakyLLM(response_text=_VALID_PAGE, raise_on_calls={1, 2, 3})
    reporter = ListReporter()

    await api._synth_pages_from_source(
        llm=llm,
        template=_TEMPLATE,
        cfg=_cfg(retries=2),
        source_path="sources/multi.md",
        source_body=body,
        chunks=chunks,
        cancel=CancelToken(),
        reporter=reporter,
    )

    llm_events = [
        e for e in reporter.events
        if e.kind == "progress" and e.payload["phase"] == "synth_llm"
    ]
    statuses = [e.payload["detail"].get("status") for e in llm_events]
    # First two attempts emit "retrying"; third (terminal) emits "skipped".
    assert statuses.count("retrying") == 2
    assert statuses.count("skipped") == 1

    retrying = next(
        e for e in llm_events
        if e.payload["detail"].get("status") == "retrying"
    )
    rd = retrying.payload["detail"]
    assert rd["source_path"] == "sources/multi.md"
    assert rd["group_pos"] == 2
    assert rd["attempt"] == 1
    assert rd["max_attempts"] == 3
    assert rd["error_kind"] == "ProviderError"
    assert rd["error_msg"]

    skipped = next(
        e for e in llm_events
        if e.payload["detail"].get("status") == "skipped"
    )
    sd = skipped.payload["detail"]
    assert sd["source_path"] == "sources/multi.md"
    assert sd["group_pos"] == 2
    assert sd["attempts"] == 3
    assert sd["reason"] == "provider_error"
    assert sd["error_kind"] == "ProviderError"


@pytest.mark.asyncio
async def test_retry_loop_honors_cancel_between_attempts() -> None:
    """Between attempts the loop must check ``CancelToken`` so a
    user-issued cancel during a flapping run terminates promptly
    instead of consuming the full backoff budget.
    """
    body, chunks = _three_groups()
    llm = FlakyLLM(response_text=_VALID_PAGE, raise_on_calls={0, 1, 2, 3, 4})
    cancel = CancelToken()

    # Pre-cancel before the call; the first ProviderError should
    # see ``raise_if_cancelled()`` fire before sleeping for retry.
    # NOTE: ``CancelledError`` is a ``BaseException`` (not ``Exception``)
    # in Python 3.8+, so ``pytest.raises(Exception)`` would NOT catch
    # it. Use ``BaseException`` to pin both: cancellation must surface
    # and must not get swallowed by the retry loop's ProviderError
    # catch.
    cancel.cancel()

    with pytest.raises(BaseException) as exc_info:
        await api._synth_pages_from_source(
            llm=llm,
            template=_TEMPLATE,
            cfg=_cfg(retries=5),
            source_path="sources/multi.md",
            source_body=body,
            chunks=chunks,
            cancel=cancel,
            reporter=ListReporter(),
        )
    assert "cancel" in str(exc_info.value).lower() or (
        type(exc_info.value).__name__ in {"CancelledError", "TaskCancelled"}
    )


@pytest.mark.asyncio
async def test_retries_zero_means_no_retry() -> None:
    """``provider_error_retries=0`` → first failure terminates the
    group immediately (one attempt, no retry, ``skipped`` event).
    Validates the off-switch for users who want the legacy fail-fast.
    """
    body, chunks = _three_groups()
    llm = FlakyLLM(response_text=_VALID_PAGE, raise_on_calls={1})
    reporter = ListReporter()

    outcome = await api._synth_pages_from_source(
        llm=llm,
        template=_TEMPLATE,
        cfg=_cfg(retries=0),
        source_path="sources/multi.md",
        source_body=body,
        chunks=chunks,
        cancel=CancelToken(),
        reporter=reporter,
    )

    assert llm.call_count == 3, "g1 + g2 (no retry) + g3"
    assert outcome.parse_errors == 1

    statuses = [
        e.payload["detail"].get("status")
        for e in reporter.events
        if e.kind == "progress" and e.payload["phase"] == "synth_llm"
    ]
    # With retries=0, "retrying" never fires.
    assert "retrying" not in statuses
    assert statuses.count("skipped") == 1
