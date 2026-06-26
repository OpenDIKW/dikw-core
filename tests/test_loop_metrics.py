"""Test-the-tool for the delivery-loop effectiveness metrics
(``tools/loop_metrics.py``, Tier 3 of the delivery-loop roadmap).

The script reports "cost per accepted change" signals over merged PRs. Two data
tiers, and these cases pin each pure function that turns raw ``gh`` data into a
metric:

  * ``classify_first_pass_green`` — a PR is first-pass-green iff no *required*
    check ever concluded a failure across its commits (absence of a check is not
    a failure, so it degrades gracefully across history where the required set
    changed); required is matched by case-insensitive substring fragment.
  * ``count_escapes`` — non-author inline review comments (the findings that
    leaked past in-loop review). Deterministic count, no LLM actionability judge.
  * ``parse_receipt`` — best-effort extraction of ``codex_rounds`` and the
    ``fresh_review`` verdict from a rendered ``## Delivery receipt`` PR body
    (n/a when the section is absent — most historical PRs).
  * ``summarize`` — folds per-PR records into the aggregate rates, counting
    first-pass-green only over PRs we actually have CI data for.

The receipt fixture mirrors the real shape rendered into PR #242's body
(``codex (1 round)`` in a table cell, ``fresh-review **pass**`` in a heading).
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.loop_metrics import (  # noqa: E402
    DEFAULT_REQUIRED_FRAGMENTS,
    CheckResult,
    PRMetrics,
    Review,
    ReviewComment,
    _parse_gh_objects,
    classify_first_pass_green,
    count_escapes,
    has_changes_requested,
    parse_receipt,
    summarize,
)

# A receipt section shaped like the one rendered into PR #242's body.
_RECEIPT_BODY = """## What

Adds a per-branch delivery artifact.

## Delivery receipt (dogfooded — this PR ran through the loop it edits)

| # | step | status | evidence |
|---|------|--------|----------|
| 4 | codex (1 round) | done | 2 WARN fixed, 2 NOTE resolved |

### step 5 — fresh-review **pass** (clean subagent, all findings notes)
| # | finding | verdict | action |
|---|---|---|---|
| 1 | something | CONFIRMED | FIXED |
"""

_NO_RECEIPT_BODY = """## What

A normal PR with no delivery receipt at all.

## Summary by CodeRabbit
* docs
"""


# --- classify_first_pass_green -------------------------------------------------


def _cr(name: str, conclusion: str) -> CheckResult:
    return CheckResult(name=name, conclusion=conclusion)


def test_first_pass_green_all_required_success() -> None:
    results = [
        _cr("lint-type-test (3.12)", "success"),
        _cr("lint-type-test (3.13)", "success"),
        _cr("Postgres contract tests", "success"),
        _cr("Server e2e (serve-and-run)", "success"),
        _cr("codecov/patch", "success"),
        _cr("CodeQL", "failure"),  # non-required → ignored
    ]
    assert classify_first_pass_green(results, DEFAULT_REQUIRED_FRAGMENTS) is True


def test_first_pass_green_required_failure_anywhere_is_red() -> None:
    # Same context appears success on a later commit but failed earlier → not green.
    results = [
        _cr("lint-type-test (3.12)", "failure"),
        _cr("lint-type-test (3.12)", "success"),
        _cr("codecov/patch", "success"),
    ]
    assert classify_first_pass_green(results, DEFAULT_REQUIRED_FRAGMENTS) is False


def test_first_pass_green_ignores_nonrequired_failures() -> None:
    results = [
        _cr("Analyze Python", "failure"),
        _cr("Trivy", "failure"),
        _cr("lint-type-test (3.12)", "success"),
    ]
    assert classify_first_pass_green(results, DEFAULT_REQUIRED_FRAGMENTS) is True


def test_first_pass_green_substring_and_case_insensitive() -> None:
    # Real check-run names carry the "ci /" workflow prefix; fragment matches by substring.
    results = [_cr("ci / Lint-Type-Test (3.13)", "TIMED_OUT")]
    assert classify_first_pass_green(results, DEFAULT_REQUIRED_FRAGMENTS) is False


def test_first_pass_green_absent_check_is_not_failure() -> None:
    # Old PR predating Server e2e: it simply never ran → not counted as a failure.
    results = [_cr("lint-type-test (3.12)", "success")]
    assert classify_first_pass_green(results, DEFAULT_REQUIRED_FRAGMENTS) is True


def test_first_pass_green_error_state_counts_as_failure() -> None:
    # Commit statuses (codecov) use "error"/"failure"/"success", not check-run conclusions.
    results = [_cr("codecov/patch", "error")]
    assert classify_first_pass_green(results, DEFAULT_REQUIRED_FRAGMENTS) is False


# --- _parse_gh_objects (the array-vs-NDJSON parser) ----------------------------


def test_parse_gh_objects_json_array() -> None:
    # Shape from `gh ... --json f1,f2` (no --jq): a single JSON array.
    out = '[{"number": 1}, {"number": 2}]'
    assert _parse_gh_objects(out) == [{"number": 1}, {"number": 2}]


def test_parse_gh_objects_ndjson_stream() -> None:
    # Shape from `gh api --paginate --jq '.[] | {x}'`: one compact object per line.
    out = '{"name": "a"}\n{"name": "b"}\n{"name": "c"}\n'
    assert _parse_gh_objects(out) == [{"name": "a"}, {"name": "b"}, {"name": "c"}]


def test_parse_gh_objects_empty_and_blank() -> None:
    assert _parse_gh_objects("") == []
    assert _parse_gh_objects("   \n  \n") == []


def test_parse_gh_objects_single_object() -> None:
    assert _parse_gh_objects('{"a": 1}') == [{"a": 1}]


def test_parse_gh_objects_skips_unparseable_and_non_dict_lines() -> None:
    # A malformed line and a bare-array line are dropped, not fatal.
    out = '{"ok": 1}\nnot-json\n[1, 2]\n{"ok": 2}'
    assert _parse_gh_objects(out) == [{"ok": 1}, {"ok": 2}]


# --- count_escapes -------------------------------------------------------------


def test_count_escapes_only_non_author_inline_comments() -> None:
    comments = [
        ReviewComment(author="coderabbitai"),
        ReviewComment(author="coderabbitai"),
        ReviewComment(author="holo"),  # the PR author's own reply → not an escape
    ]
    assert count_escapes(comments, author="holo") == 2


def test_count_escapes_empty() -> None:
    assert count_escapes([], author="holo") == 0


def test_has_changes_requested_non_author_review() -> None:
    reviews = [
        Review(author="someone", state="CHANGES_REQUESTED"),
        Review(author="holo", state="APPROVED"),
    ]
    assert has_changes_requested(reviews, author="holo") is True


def test_has_changes_requested_author_self_review_ignored() -> None:
    # A PR author cannot "request changes" on their own PR, but guard anyway.
    reviews = [Review(author="holo", state="CHANGES_REQUESTED")]
    assert has_changes_requested(reviews, author="holo") is False


def test_has_changes_requested_none() -> None:
    reviews = [Review(author="someone", state="APPROVED")]
    assert has_changes_requested(reviews, author="holo") is False


# --- parse_receipt -------------------------------------------------------------


def test_parse_receipt_present() -> None:
    facts = parse_receipt(_RECEIPT_BODY)
    assert facts.has_receipt is True
    assert facts.codex_rounds == 1
    assert facts.fresh_review == "pass"


def test_parse_receipt_absent() -> None:
    facts = parse_receipt(_NO_RECEIPT_BODY)
    assert facts.has_receipt is False
    assert facts.codex_rounds is None
    assert facts.fresh_review is None


def test_parse_receipt_empty_body() -> None:
    facts = parse_receipt("")
    assert facts.has_receipt is False


def test_parse_receipt_scoped_to_section_not_whole_body() -> None:
    # Narrative BEFORE the receipt mentions misleading values; the real receipt
    # section (and a CodeRabbit block AFTER it) must not bleed into the scrape.
    body = """## Why

An earlier abandoned attempt took codex 9 rounds and the fresh-review came back
blocking before we reworked it.

## Delivery receipt

| 4 | codex (1 round) | done | ok |

### step 5 — fresh-review **pass**

## Summary by CodeRabbit
mentions codex 7 rounds in release notes
"""
    facts = parse_receipt(body)
    assert facts.has_receipt is True
    assert facts.codex_rounds == 1
    assert facts.fresh_review == "pass"


# --- summarize -----------------------------------------------------------------


def _m(
    number: int,
    *,
    green: bool,
    escapes: int,
    seen: int,
    codex: int | None = None,
    receipt: bool = False,
) -> PRMetrics:
    return PRMetrics(
        number=number,
        title=f"PR {number}",
        author="holo",
        first_pass_green=green,
        escapes=escapes,
        required_checks_seen=seen,
        codex_rounds=codex,
        fresh_review=None,
        has_receipt=receipt,
        changes_requested=False,
    )


def test_summarize_rates() -> None:
    records = [
        _m(1, green=True, escapes=0, seen=5, codex=1, receipt=True),
        _m(2, green=False, escapes=3, seen=5),
        _m(3, green=True, escapes=1, seen=0),  # no CI data → excluded from green rate
    ]
    agg = summarize(records)
    assert agg.total == 3
    assert agg.with_ci == 2
    assert agg.first_pass_green == 1
    assert agg.first_pass_green_rate == 0.5
    assert agg.total_escapes == 4
    assert agg.mean_escapes == 4 / 3
    assert agg.with_receipt == 1
    assert agg.mean_codex_rounds == 1.0


def test_summarize_empty() -> None:
    agg = summarize([])
    assert agg.total == 0
    assert agg.first_pass_green_rate is None
    assert agg.mean_escapes is None
    assert agg.mean_codex_rounds is None
