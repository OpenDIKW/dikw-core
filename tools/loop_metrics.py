"""Delivery-loop effectiveness metrics — Tier 3 of the delivery-loop roadmap.

0xCodez's north-star for a maturing agent loop is **cost per accepted change**:
if you cannot measure the loop, you cannot tell whether it is getting better. For
a solo maintainer the tractable proxies are *first-pass-green rate* (did the
in-loop self-verify stop CI from ever going red?) and *escape rate* (how many
findings leaked past in-loop review to a post-PR reviewer?). This walks the
merged PRs in a rolling window and reports them.

Two data tiers, by robustness:

  * **GitHub API (robust, every historical PR).** ``first_pass_green`` — no
    *required* check ever concluded a failure across the PR's commits; and
    ``escapes`` — **inline** review comments authored by someone other than the
    PR author (plus a ``changes_requested`` flag for a non-author blocking
    review). Inline-only is deliberate: PR-level issue comments are dominated by
    bot summaries / "LGTM" / the receipt itself, and deciding which of those is a
    *finding* would need an LLM — disallowed on this read path — so the count is
    a deterministic, honest under-count of inline findings rather than a noisy
    over-count of all comments.
  * **PR-body delivery receipt (best-effort, only PRs that carry one).**
    ``codex_rounds`` and the ``fresh_review`` verdict, regex-scraped from the
    rendered ``## Delivery receipt`` section (see the ``dikw-core-delivery-
    workflow`` skill). ``n/a`` when absent — which is most history today, so the
    API tier carries the script until receipts accumulate (the receipt is the
    instrument, this script is the readout).

Karpathy's rule holds on the read path: every metric is a deterministic count or
parse — no LLM judges actionability, so ``escapes`` is an honest upper bound that
includes nitpicks rather than a model's guess at which comments "mattered".

The pure functions (:func:`classify_first_pass_green`, :func:`required_seen`,
:func:`count_escapes`, :func:`has_changes_requested`, :func:`parse_receipt`,
:func:`receipt_section`, :func:`_parse_gh_objects`, :func:`summarize`) are
unit-tested in ``tests/test_loop_metrics.py``; :func:`main` adds the ``gh``
plumbing — mirroring the split in ``tools/check_baselines.py``.
"""

from __future__ import annotations

import argparse
import datetime
import json
import re
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass

# Required-check name *fragments* (lowercased, substring-matched). Real check-run
# names carry the "ci /" workflow prefix and the job suffix, so we match loosely.
# Mirrors memory ``codeql-ci-required-checks`` — the five contexts branch
# protection actually blocks merge on (CodeQL / Trivy / Analyze Python are
# non-required and deliberately excluded). ``--required`` overrides at the CLI.
DEFAULT_REQUIRED_FRAGMENTS: tuple[str, ...] = (
    "lint-type-test (3.12)",
    "lint-type-test (3.13)",
    "postgres contract",
    "server e2e",
    "codecov/patch",
)

# Conclusions (check-runs) and states (commit statuses) that mean the check was
# genuinely red. check-runs use ``failure``/``timed_out``/…; commit statuses
# (e.g. codecov/patch) use ``error``/``failure``. NB ``cancelled`` and ``stale``
# are deliberately NOT here: GitHub Actions concurrency auto-cancels a superseded
# run on re-push, which is not a CI failure — counting it would deflate the rate.
_FAILING: frozenset[str] = frozenset(
    {
        "failure",
        "timed_out",
        "startup_failure",
        "action_required",
        "error",
    }
)

# Match the receipt section heading rendered into a PR body — "## Delivery
# receipt" optionally followed by a parenthetical ("(dogfooded — …)").
_RECEIPT_HEADING_RE = re.compile(r"^##\s+Delivery receipt\b", re.IGNORECASE | re.MULTILINE)
_H2_LINE_RE = re.compile(r"^##\s+")
_FENCE_RE = re.compile(r"^\s*```")
# Codex round count, however the receipt spells it: "codex (1 round)",
# "codex (≤3) | done | 2 rounds". Confined to one line on purpose — the canonical
# receipt renders the count in a single table cell, and spanning lines would risk
# grabbing an unrelated number elsewhere in the section. Fails closed to None.
_CODEX_ROUNDS_RE = re.compile(r"codex[^\n]*?(\d+)\s*rounds?\b", re.IGNORECASE)
# Fresh-review verdict: "fresh-review **pass**" / "fresh-review: blocking". Word
# boundaries on the alternation so it does not match "pass" inside "bypass" or
# "blocking" inside "unblocking".
_FRESH_REVIEW_RE = re.compile(
    r"fresh-review\b[^\n]*?\*{0,2}\b(pass|blocking|changes[ _]requested)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class CheckResult:
    """One check-run conclusion or commit-status state for a PR commit."""

    name: str
    conclusion: str


@dataclass(frozen=True)
class ReviewComment:
    """An inline review comment (``/pulls/{n}/comments``), reduced to its author."""

    author: str


@dataclass(frozen=True)
class Review:
    """A review submission (``/pulls/{n}/reviews``) — author + state."""

    author: str
    state: str


@dataclass(frozen=True)
class ReceiptFacts:
    """What the best-effort receipt scrape recovered from a PR body."""

    has_receipt: bool
    codex_rounds: int | None
    fresh_review: str | None


@dataclass(frozen=True)
class PRMetrics:
    """Per-PR row folded into the aggregate."""

    number: int
    title: str
    author: str
    first_pass_green: bool
    escapes: int
    required_checks_seen: int  # distinct required fragments seen, not a raw count
    full_required: bool  # saw the *entire* required set → comparable in the rate
    codex_rounds: int | None
    fresh_review: str | None
    has_receipt: bool
    changes_requested: bool


@dataclass(frozen=True)
class Aggregate:
    """Window-level rollup."""

    total: int
    with_ci: int
    first_pass_green: int
    first_pass_green_rate: float | None
    total_escapes: int
    mean_escapes: float | None
    with_receipt: int
    receipt_coverage: float | None
    mean_codex_rounds: float | None
    changes_requested: int


# --- pure functions ------------------------------------------------------------


def _is_required(name: str, required_fragments: Sequence[str]) -> bool:
    low = name.lower()
    return any(frag in low for frag in required_fragments)


def required_seen(
    results: Sequence[CheckResult], required_fragments: Sequence[str]
) -> set[str]:
    """The set of required fragments that matched at least one check result.

    Distinct fragments, not a raw count — a check that ran on three commits (and
    in both the check-runs and statuses feeds) counts its fragment once.
    """
    names = [r.name.lower() for r in results]
    return {frag for frag in required_fragments if any(frag in n for n in names)}


def classify_first_pass_green(
    results: Sequence[CheckResult], required_fragments: Sequence[str]
) -> bool:
    """True iff no *required* check ever concluded a failure.

    ``results`` is the union of check-runs (queried with ``filter=all`` so a
    same-commit re-run cannot hide an earlier failure) + commit statuses across
    **all** of the PR's commits, so a required check that failed on any commit —
    or any attempt — makes the PR not-first-pass-green. A required check that
    never ran (absent) is not a failure; this degrades gracefully across history
    where the required set changed.
    """
    for r in results:
        if _is_required(r.name, required_fragments) and r.conclusion.lower() in _FAILING:
            return False
    return True


def count_escapes(comments: Sequence[ReviewComment], author: str) -> int:
    """Inline review comments authored by someone other than the PR author.

    An honest upper bound on findings that leaked past in-loop review — it counts
    nitpicks too, by design (no LLM decides which comments "mattered"). Empty /
    null authors are dropped (a deleted account yields no usable login).
    """
    return sum(1 for c in comments if c.author and c.author != author)


def has_changes_requested(reviews: Sequence[Review], author: str) -> bool:
    """True iff a non-author submitted a CHANGES_REQUESTED review."""
    return any(r.author != author and r.state.upper() == "CHANGES_REQUESTED" for r in reviews)


def receipt_section(body: str) -> str | None:
    """The text of the ``## Delivery receipt`` section, or None if absent.

    Bounded heading → next top-level (H2) heading / EOF so a codex / fresh-review
    mention in the surrounding narrative (a "## Why" above, a "## Summary by
    CodeRabbit" below) can't bleed into the scrape. Fenced code blocks are
    skipped when looking for the bound, because the skill mandates pasting real
    machine output into Evidence — a ``## `` line inside a pasted ``` fence is
    content, not a real section break.
    """
    heading = _RECEIPT_HEADING_RE.search(body)
    if not heading:
        return None
    start = heading.end()
    offset = start
    in_fence = False
    for line in body[start:].splitlines(keepends=True):
        if _FENCE_RE.match(line):
            in_fence = not in_fence
        elif not in_fence and _H2_LINE_RE.match(line):
            return body[start:offset]
        offset += len(line)
    return body[start:]


def parse_receipt(body: str) -> ReceiptFacts:
    """Best-effort scrape of the rendered ``## Delivery receipt`` PR-body section."""
    section = receipt_section(body)
    if section is None:
        return ReceiptFacts(has_receipt=False, codex_rounds=None, fresh_review=None)
    rounds_match = _CODEX_ROUNDS_RE.search(section)
    fresh_match = _FRESH_REVIEW_RE.search(section)
    fresh = fresh_match.group(1).lower().replace("_", " ") if fresh_match else None
    return ReceiptFacts(
        has_receipt=True,
        codex_rounds=int(rounds_match.group(1)) if rounds_match else None,
        fresh_review=fresh,
    )


def summarize(records: Sequence[PRMetrics]) -> Aggregate:
    """Fold per-PR rows into window rates.

    ``first_pass_green_rate`` is computed only over PRs that saw the **full**
    required set (``full_required``), so old PRs that predate some required
    checks — or any partial set — don't skew an apples-to-oranges rate.
    ``mean_codex_rounds`` averages only PRs whose receipt reported a count.
    """
    total = len(records)
    with_ci = sum(1 for r in records if r.full_required)
    green = sum(1 for r in records if r.full_required and r.first_pass_green)
    total_escapes = sum(r.escapes for r in records)
    with_receipt = sum(1 for r in records if r.has_receipt)
    changes_requested = sum(1 for r in records if r.changes_requested)
    codex_values = [r.codex_rounds for r in records if r.codex_rounds is not None]
    return Aggregate(
        total=total,
        with_ci=with_ci,
        first_pass_green=green,
        first_pass_green_rate=(green / with_ci) if with_ci else None,
        total_escapes=total_escapes,
        mean_escapes=(total_escapes / total) if total else None,
        with_receipt=with_receipt,
        receipt_coverage=(with_receipt / total) if total else None,
        mean_codex_rounds=(sum(codex_values) / len(codex_values)) if codex_values else None,
        changes_requested=changes_requested,
    )


def _parse_gh_objects(text: str) -> list[dict[str, object]]:
    """Parse ``gh`` stdout into a list of objects, handling both shapes it emits.

    A single JSON array (``--json`` without ``--jq``) parses whole; an NDJSON
    stream (``--paginate`` with a per-object ``--jq`` — one compact object per
    line, which ``json.loads`` cannot parse whole) falls back to line-by-line.
    Non-dict and unparseable lines are dropped. Pure (no I/O) so it is
    unit-tested directly.
    """
    out = text.strip()
    if not out:
        return []
    try:
        parsed = json.loads(out)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, list):
        return [o for o in parsed if isinstance(o, dict)]
    if isinstance(parsed, dict):
        return [parsed]
    objs: list[dict[str, object]] = []
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            objs.append(obj)
    return objs


# --- gh I/O (thin; not unit-tested) --------------------------------------------


class _GhError(RuntimeError):
    """A ``gh`` invocation failed (non-zero exit) — recoverable per-PR."""


def _run_gh(args: list[str]) -> str:
    """Run ``gh`` and return stdout.

    A missing ``gh`` binary is fatal (every call would fail) → ``sys.exit``. A
    non-zero exit raises :class:`_GhError` so the caller can isolate a transient
    per-PR failure instead of aborting a whole multi-PR window.
    """
    try:
        return subprocess.run(
            ["gh", *args], capture_output=True, text=True, check=True
        ).stdout
    except FileNotFoundError:
        sys.exit("error: `gh` CLI not found on PATH — install/authenticate it first.")
    except subprocess.CalledProcessError as exc:
        raise _GhError(f"`gh {' '.join(args)}` failed: {exc.stderr.strip()}") from exc


def _gh_obj(args: list[str]) -> dict[str, object]:
    """Parse ``gh`` stdout as a single JSON object (e.g. ``pr view --json``)."""
    parsed = json.loads(_run_gh(args) or "{}")
    return parsed if isinstance(parsed, dict) else {}


def _gh_list(args: list[str]) -> list[dict[str, object]]:
    """Run ``gh`` and parse its stdout as a list of objects (see _parse_gh_objects)."""
    return _parse_gh_objects(_run_gh(args))


def _as_dict(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}


def _as_list(value: object) -> list[object]:
    return value if isinstance(value, list) else []


def _check_results_for_commit(repo: str, sha: str) -> list[CheckResult]:
    """Union of check-runs + commit statuses for one commit SHA.

    check-runs uses ``filter=all`` (every attempt, so a same-commit re-run can't
    hide an earlier failure); statuses uses the **plural** ``/statuses`` endpoint
    (full history) rather than the combined ``/status`` (latest-per-context only).
    Both are paginated.
    """
    results: list[CheckResult] = []
    runs = _gh_list(
        [
            "api", f"repos/{repo}/commits/{sha}/check-runs?filter=all", "--paginate",
            "--jq", ".check_runs[] | {name, conclusion}",
        ]
    )
    for obj in runs:
        results.append(
            CheckResult(name=str(obj.get("name", "")), conclusion=str(obj.get("conclusion") or ""))
        )
    statuses = _gh_list(
        [
            "api", f"repos/{repo}/commits/{sha}/statuses", "--paginate",
            "--jq", ".[] | {context, state}",
        ]
    )
    for obj in statuses:
        results.append(
            CheckResult(name=str(obj.get("context", "")), conclusion=str(obj.get("state") or ""))
        )
    return results


def _collect_pr(
    repo: str, number: int, required_fragments: Sequence[str], include_bots: bool
) -> PRMetrics | None:
    """Gather one PR's metrics, or None if it is bot-authored and not included."""
    pr = _gh_obj(
        [
            "pr", "view", str(number), "--repo", repo,
            "--json", "number,title,author,body,commits,reviews",
        ]
    )
    author_obj = _as_dict(pr.get("author"))
    if not include_bots and author_obj.get("is_bot"):
        return None
    author = str(author_obj.get("login") or "")
    shas = [
        str(c.get("oid", ""))
        for c in _as_list(pr.get("commits"))
        if isinstance(c, dict) and c.get("oid")
    ]
    all_checks: list[CheckResult] = []
    for sha in shas:
        all_checks.extend(_check_results_for_commit(repo, sha))
    seen = required_seen(all_checks, required_fragments)

    comments = _gh_list(
        [
            "api", f"repos/{repo}/pulls/{number}/comments", "--paginate",
            "--jq", ".[] | {login: .user.login}",
        ]
    )
    review_comments = [ReviewComment(author=str(o.get("login") or "")) for o in comments]
    reviews = [
        Review(
            author=str(_as_dict(r.get("author")).get("login") or ""),
            state=str(r.get("state", "")),
        )
        for r in _as_list(pr.get("reviews"))
        if isinstance(r, dict)
    ]
    facts = parse_receipt(str(pr.get("body") or ""))
    number_val = pr.get("number")
    return PRMetrics(
        number=number_val if isinstance(number_val, int) else number,
        title=str(pr.get("title", "")),
        author=author,
        first_pass_green=classify_first_pass_green(all_checks, required_fragments),
        escapes=count_escapes(review_comments, author),
        required_checks_seen=len(seen),
        full_required=len(seen) == len(required_fragments),
        codex_rounds=facts.codex_rounds,
        fresh_review=facts.fresh_review,
        has_receipt=facts.has_receipt,
        changes_requested=has_changes_requested(reviews, author),
    )


def _merged_pr_numbers(repo: str, limit: int, since: str | None) -> list[int]:
    """Merged PR numbers in the window.

    When ``since`` is given it is filtered **server-side** via the ``merged:>=``
    search qualifier (so ``--limit`` caps a date-correct window, not a
    naive-string slice of the newest ``limit`` PRs), and a truncation warning
    fires if the cap is hit.
    """
    args = ["pr", "list", "--repo", repo, "--limit", str(limit), "--json", "number,mergedAt"]
    if since:
        args += ["--search", f"merged:>={since} sort:created-desc"]
    else:
        args += ["--state", "merged"]
    rows = _gh_list(args)
    numbers = [n for r in rows if isinstance((n := r.get("number")), int)]
    if since and len(numbers) >= limit:
        print(
            f"warning: --limit {limit} reached with --since {since}; the window may be "
            f"truncated — raise --limit to be sure it covers the full date range.",
            file=sys.stderr,
        )
    return numbers


# --- rendering -----------------------------------------------------------------


def _fmt_rate(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.0f}%"


def _fmt_num(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def _render_text(records: Sequence[PRMetrics], agg: Aggregate) -> str:
    lines = [
        "# Delivery-loop metrics",
        "",
        f"PRs in window: {agg.total}  (measured on the full required set: {agg.with_ci})",
        (
            f"first-pass-green: {agg.first_pass_green}/{agg.with_ci} "
            f"= {_fmt_rate(agg.first_pass_green_rate)}"
        ),
        (
            f"escapes (non-author inline comments): total {agg.total_escapes}, "
            f"mean {_fmt_num(agg.mean_escapes)}/PR"
        ),
        f"changes-requested PRs: {agg.changes_requested}",
        (
            f"receipt coverage: {agg.with_receipt}/{agg.total} "
            f"= {_fmt_rate(agg.receipt_coverage)}"
        ),
        f"mean codex rounds (where recorded): {_fmt_num(agg.mean_codex_rounds)}",
        "",
        "| PR | green | escapes | CR | codex | fresh-review | title |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in records:
        green = "?" if not r.full_required else ("✓" if r.first_pass_green else "✗")
        cr = "✗" if r.changes_requested else ""
        codex = str(r.codex_rounds) if r.codex_rounds is not None else "—"
        fresh = r.fresh_review or "—"
        title = r.title if len(r.title) <= 50 else r.title[:47] + "..."
        lines.append(
            f"| #{r.number} | {green} | {r.escapes} | {cr} | {codex} | {fresh} | {title} |"
        )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--repo", default="OpenDIKW/dikw-core", help="owner/name (default: %(default)s)"
    )
    parser.add_argument(
        "--limit", type=int, default=20, help="max merged PRs to scan (default: %(default)s)"
    )
    parser.add_argument(
        "--since", default=None,
        help="only PRs merged on/after this ISO date YYYY-MM-DD (server-side filter)",
    )
    parser.add_argument(
        "--required", action="append", default=None,
        help="required-check name fragment (repeatable; overrides the default set)",
    )
    parser.add_argument(
        "--include-bots", action="store_true",
        help="include bot-authored PRs (default: skip them — they bypass the human loop)",
    )
    parser.add_argument(
        "--json", action="store_true", help="emit machine-readable JSON instead of a table"
    )
    args = parser.parse_args(argv)

    if args.since:
        try:
            datetime.date.fromisoformat(args.since)
        except ValueError:
            return _fail(f"--since must be an ISO date (YYYY-MM-DD), got {args.since!r}")

    required = (
        tuple(f.lower() for f in args.required) if args.required else DEFAULT_REQUIRED_FRAGMENTS
    )
    try:
        numbers = _merged_pr_numbers(args.repo, args.limit, args.since)
    except _GhError as exc:
        return _fail(str(exc))
    if not numbers:
        print("no merged PRs in window", file=sys.stderr)
        return 0

    records: list[PRMetrics] = []
    skipped_bots = 0
    for n in numbers:
        try:
            rec = _collect_pr(args.repo, n, required, args.include_bots)
        except _GhError as exc:
            print(f"warning: skipping PR #{n}: {exc}", file=sys.stderr)
            continue
        if rec is None:
            skipped_bots += 1
            continue
        records.append(rec)
    if skipped_bots:
        print(f"note: skipped {skipped_bots} bot-authored PR(s) (use --include-bots)", file=sys.stderr)
    if not records:
        print("no human-authored PRs in window", file=sys.stderr)
        return 0
    agg = summarize(records)

    if args.json:
        payload = {"aggregate": asdict(agg), "prs": [asdict(r) for r in records]}
        print(json.dumps(payload, indent=2))
    else:
        print(_render_text(records, agg))
    return 0


def _fail(message: str) -> int:
    print(f"error: {message}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
