"""Content check for ``evals/BASELINES.md`` additions — the gate behind
``.github/workflows/eval-gate.yml``.

The original gate only asked *was BASELINES.md diffed at all?* — a false-green:
a single blank-line edit satisfied it, and a reviewer (usually the author) had
to eyeball whether the entry was real. This parses the **added** lines of a PR's
BASELINES.md diff and asserts they form a *substantive, change-type-appropriate*
baseline entry:

  * always — a genuinely NEW dated/versioned entry header (`## <YYYY-MM-DD|X.Y.Z> — …`)
    whose text is not already in the base-revision file, so neither a one-char
    edit to an old entry nor a copy-pasted/stale header passes;
  * K-layer/synth diff (``domains/knowledge/**``, ``api_synth.py``, or the K-layer
    authoring prompts) — the entry demonstrates EITHER signal OR non-destructiveness
    (docs/eval-plan.md's exact framing) AND references the mandated ``elon-musk``
    baseline corpus (CLAUDE.md "Things not to do"; ``feedback_real_data_validation``).
    *Signal* = ≥ N of the seven ``synth`` metrics OR a lint-outcome count (issue-kind
    / TP-FP, the shape lint-fix PRs baseline with), on numeric lines. *Non-
    destructiveness* = an explicit marker (byte-identical / non-destructive / purely
    additive / no-regression) for an additive change that legitimately produces no
    numbers (e.g. #122: a packaged dataset ships zero wisdom files, so the retrieval
    eval is byte-identical) — accepted instead of fabricated numbers;
  * Retrieval diff (``domains/info/**``) — the entry records an ablation
    (nDCG / hit@k / MRR / recall);
  * storage-only diff — the new-entry header is sufficient (storage behavior is
    gated by the real-Postgres contract suite, not by eval numbers).

This is the *content* check (the entry exists, is new, has the right shape and
corpus). Actually re-running the eval to verify the numbers is a separate,
heavier gate (roadmap Phase 2). Token checks are substring/numeric-line based,
not semantic — e.g. the corpus leg cannot tell an honest ``elon-musk`` mention
from a contrived one; that residual softness is a conscious Phase-0 limitation.
The pure :func:`check_baseline_addition` is unit-tested in
``tests/test_check_baselines.py``; :func:`main` adds the git plumbing the
workflow calls.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys

BASELINES_PATH = "evals/BASELINES.md"

# Path surfaces the eval-gate triggers on (mirror eval-gate.yml `on.paths` — the
# two must be edited together). K-layer authoring (synth + the lint-fix LLM
# authoring prompts) lives in domains/knowledge/** AND api_synth.py AND three
# prompts; a synth-quality PR commonly touches only the latter (e.g. #172:
# api_synth.py + prompts/synthesize.md, no domains/knowledge change), so the
# baseline gate must watch them or it never fires on the dominant synth-PR shape.
# RetrievalConfig lives in the broad config.py and is left out deliberately
# (watching config.py wholesale would gate every config edit).
_KNOWLEDGE_PREFIXES = ("src/dikw_core/domains/knowledge/",)
_KNOWLEDGE_FILES = (
    "src/dikw_core/api_synth.py",
    "src/dikw_core/prompts/synthesize.md",
    "src/dikw_core/prompts/lint_fix_orphan_merge.md",
    "src/dikw_core/prompts/lint_fix_broken_wikilink_grounded.md",
)
_INFO_PREFIXES = ("src/dikw_core/domains/info/",)
_STORAGE_PREFIXES = ("src/dikw_core/storage/",)

# The seven synth-quality metrics from `dikw client eval --dataset mvp --eval
# synth` (PR #78). The canonical mvp gate entry reports all seven; the most
# metric-sparse already-merged K-layer A/B entry (2026-06-07) spells out five.
_SYNTH_METRICS: tuple[str, ...] = (
    "fact_grounding_ratio",
    "atomicity_score",
    "duplicate_ratio_max",
    "wikilink_resolved_ratio",
    "expected_coverage",
    "language_fidelity",
    "page_density",
)
# Floor, not exact-7: requiring all seven would spuriously block the author's own
# current A/B-table practice (which names five), while three cleanly separates a
# real synth baseline from a prose-only / empty edit.
_MIN_SYNTH_METRICS = 3

# Lint-fix PRs (the merged #68/#70/#82/#83 arc) baseline against elon-musk with
# lint-outcome counts, not the synth metrics (docs/eval-plan.md lists this as the
# primary K-layer baseline content). Accept either shape for the quantitative leg.
_LINT_OUTCOME_RE = re.compile(
    r"broken_wikilink|orphan_page|non_atomic_page|missing_provenance"
    r"|uncategorized|duplicate_title|\bTP\b|\bFP\b"
)

# eval-plan.md: a K-layer baseline demonstrates "either signal or non-
# destructiveness". An additive / coverage change can legitimately produce no
# numbers (e.g. #122: the packaged dataset ships zero wisdom files, so the
# retrieval eval is byte-identical) — accept an explicit non-destructiveness
# assertion instead of forcing fabricated numbers or the no-baseline-needed label
# (whose "non-functional refactor" wording misdescribes an additive change).
_NON_DESTRUCTIVE_RE = re.compile(
    r"byte-identical|non-destructive|behaviou?r-neutral|purely additive|no[ -]regression",
    re.IGNORECASE,
)

# Retrieval-ablation vocabulary used in BASELINES.md retrieval entries: nDCG@10 /
# ndcg_at_10, hit_at_3 / hit@k, mrr / MRR, recall_at_100 / recall@100. Every
# alternative requires a value-bearing suffix (`_`/`@`) so prose ("Recall that…",
# "nDCG is great") does not satisfy the leg.
_RETRIEVAL_METRIC_RE = re.compile(r"ndcg[_@]|hit[_@]|recall[_@]|\bmrr\b", re.IGNORECASE)

# A new top-level entry header must LEAD with a date (`2026-06-07`) or semver
# (`0.5.0`), matching the real convention `## <date|semver> — title`. Anchoring
# to the start rejects `### ` sub-sections, table rows, and `## ` headings that
# merely mention a date/version elsewhere (`## TODO before 2026-12-31`).
_ENTRY_HEADER_RE = re.compile(r"^##\s+(?:\d{4}-\d{2}-\d{2}|\d+\.\d+\.\d+)\b")

_DIGIT_RE = re.compile(r"\d")

_LABEL_HINT = (
    "If this is a non-functional refactor (rename / comment / typing only), "
    "label the PR 'no-baseline-needed'."
)


def _entry_headers(lines: list[str]) -> list[str]:
    """The dated/versioned entry-header lines among ``lines`` (stripped)."""
    return [line.strip() for line in lines if _ENTRY_HEADER_RE.match(line)]


def check_baseline_addition(
    added_lines: list[str],
    *,
    existing_headers: set[str],
    touches_knowledge: bool,
    touches_info: bool,
    touches_storage: bool,
) -> list[str]:
    """Return a list of violation messages for a BASELINES.md addition.

    ``added_lines`` are the post-``+`` text lines added to BASELINES.md in the
    PR diff (the leading ``+`` already stripped). ``existing_headers`` is the set
    of entry-header lines (stripped) already present in the base-revision file,
    used to reject a reused/stale header. An empty return list == pass.
    """
    violations: list[str] = []
    full_text = "\n".join(added_lines)
    # Numeric (value-bearing) lines — a reported outcome co-occurs with a number,
    # which distinguishes a real metric/count row from a passing prose mention.
    value_text = "\n".join(line for line in added_lines if _DIGIT_RE.search(line))

    # 1. A genuinely new entry — not a one-char edit to an existing one, and not
    #    a reused/stale/copy-pasted header. Subsumes the old presence check.
    new_headers = [h for h in _entry_headers(added_lines) if h not in existing_headers]
    if not new_headers:
        violations.append(
            "evals/BASELINES.md has no NEW dated/versioned entry header "
            "(`## <YYYY-MM-DD|X.Y.Z> — …` not already in the file) among the added "
            "lines. A K-layer / Retrieval / storage change needs a new baseline entry "
            f"showing real-data outcome — don't reuse or edit an old header. {_LABEL_HINT}"
        )

    # 2. K-layer/synth: signal OR non-destructiveness (docs/eval-plan.md) + the
    #    mandated elon-musk corpus. Signal = synth metrics or lint counts on numeric
    #    lines; non-destructiveness = an explicit marker for an additive change.
    if touches_knowledge:
        found = [m for m in _SYNTH_METRICS if m in value_text]
        has_signal = len(found) >= _MIN_SYNTH_METRICS or bool(
            _LINT_OUTCOME_RE.search(value_text)
        )
        if not has_signal and not _NON_DESTRUCTIVE_RE.search(full_text):
            violations.append(
                "K-layer/synth change (domains/knowledge/** | api_synth.py | "
                "prompts/{synthesize,lint_fix_*}.md): the new BASELINES entry shows "
                f"neither signal nor a non-destructiveness claim. Provide ≥ "
                f"{_MIN_SYNTH_METRICS} of {list(_SYNTH_METRICS)} from `dikw client eval "
                "--dataset mvp --eval synth`, OR lint-outcome counts (issue-kind / TP-FP) "
                "on numeric lines, OR an explicit non-destructiveness marker "
                "(byte-identical / non-destructive / purely additive / no-regression) "
                "for an additive change (docs/eval-plan.md: 'either signal or "
                "non-destructiveness')."
            )
        if "elon-musk" not in full_text:
            violations.append(
                "K-layer/synth change: the new BASELINES entry must reference the "
                "mandated `elon-musk` baseline corpus "
                "(CLAUDE.md 'Things not to do' / feedback_real_data_validation)."
            )

    # 3. Retrieval: an ablation with retrieval metrics.
    if touches_info and not _RETRIEVAL_METRIC_RE.search(full_text):
        violations.append(
            "Retrieval change (domains/info/**): the new BASELINES entry names no "
            "retrieval metric (nDCG / hit@k / MRR / recall). Record an ablation "
            "across ≥1 packaged dataset."
        )

    # storage-only: the new-entry header check (1) is the whole requirement —
    # storage correctness is gated by the real-Postgres contract suite.
    _ = touches_storage
    return violations


# ---- git plumbing (not unit-tested; exercised by the workflow) -----------


def _git(args: list[str]) -> str:
    return subprocess.run(
        ["git", *args], check=True, capture_output=True, text=True
    ).stdout


def _added_lines(base_sha: str, head_sha: str, path: str) -> list[str]:
    # --unified=0: no context lines, so every `+` line is a real addition.
    # Three-dot (merge-base): isolate the PR's own changes and match GitHub's
    # `on.paths` semantics, so a base-branch advance can't leak into the diff.
    out = _git(["diff", "--unified=0", f"{base_sha}...{head_sha}", "--", path])
    return [
        line[1:]
        for line in out.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    ]


def _changed_files(base_sha: str, head_sha: str) -> list[str]:
    return _git(["diff", "--name-only", f"{base_sha}...{head_sha}"]).splitlines()


def _base_headers(base_sha: str, path: str) -> set[str]:
    """Entry headers already in the base-revision file (empty if it's new here)."""
    try:
        out = _git(["show", f"{base_sha}:{path}"])
    except subprocess.CalledProcessError:
        return set()  # path did not exist at base — a brand-new BASELINES.md
    return set(_entry_headers(out.splitlines()))


def _touches(changed: list[str], prefixes: tuple[str, ...], files: tuple[str, ...]) -> bool:
    return any(f in files or any(f.startswith(p) for p in prefixes) for f in changed)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-sha", required=True)
    parser.add_argument("--head-sha", required=True)
    parser.add_argument("--baselines-path", default=BASELINES_PATH)
    args = parser.parse_args(argv)

    try:
        added = _added_lines(args.base_sha, args.head_sha, args.baselines_path)
        changed = _changed_files(args.base_sha, args.head_sha)
        existing = _base_headers(args.base_sha, args.baselines_path)
    except subprocess.CalledProcessError as e:
        print(
            "::error::git diff failed (base/head SHA unreachable? check "
            f"actions/checkout fetch-depth: 0): {e}"
        )
        return 1

    violations = check_baseline_addition(
        added,
        existing_headers=existing,
        touches_knowledge=_touches(changed, _KNOWLEDGE_PREFIXES, _KNOWLEDGE_FILES),
        touches_info=_touches(changed, _INFO_PREFIXES, ()),
        touches_storage=_touches(changed, _STORAGE_PREFIXES, ()),
    )
    if violations:
        for v in violations:
            print(f"::error::{v}")
        return 1
    print("::notice::evals/BASELINES.md content check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
