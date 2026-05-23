"""Fixer for ``missing_provenance`` issues.

Pure deterministic — no LLM call, no fuzzy match. Reads the live
frontmatter ``sources:`` list and emits a single
``reconcile_provenance`` op that calls
``storage.replace_provenance_from`` with that snapshot. The fixer never
modifies the wiki file: frontmatter is the source of truth (the wiki
tree is a user-editable Obsidian vault) and this op only syncs storage
to it. See ``docs/adr/0001-provenance-as-separate-edge.md``.

Typical lifecycle: on bases that existed before the provenance feature
shipped, the first ``dikw client lint`` run emits one
``missing_provenance`` issue per K-page with a ``sources:`` frontmatter
block; ``dikw client lint fix apply --kind missing_provenance``
backfills them in one pass. The same issue also fires when a user
hand-edits ``sources:`` outside of synth / lint-apply — covered by the
same fix path.
"""

from __future__ import annotations

import uuid
from typing import Any

import frontmatter

from ..lint import LintKind
from ..lint_fix import (
    FixerContext,
    FixOperation,
    FixProposal,
    bytes_sha256,
)
from ..wiki import frontmatter_str_list


class MissingProvenanceFixer:
    """Reconcile the ``provenance`` storage table from a K-page's
    frontmatter ``sources:`` list.

    Always ``source="heuristic"`` (the spelling for deterministic
    fixers in :class:`FixProposal`) — no LLM call, no provider
    dependencies.
    """

    kind: LintKind = "missing_provenance"

    async def propose(
        self,
        issue: Any,
        ctx: FixerContext,
        reporter: Any,
    ) -> FixProposal | None:
        abs_path = (ctx.wiki_root / issue.path).resolve()
        if not abs_path.is_file():
            # File vanished between scan and propose — nothing to
            # reconcile from. The next lint pass will reflect storage
            # truth (zero sources expected, so no issue surfaces).
            return None

        # Read once: hash + frontmatter from the same bytes so the
        # ``expected_hash`` stamp matches what the apply pass will
        # verify against, with no second disk read in between.
        file_bytes = abs_path.read_bytes()
        post = frontmatter.loads(file_bytes.decode("utf-8"))
        # ``frontmatter_str_list`` is the shared malformed-shape guard
        # — symmetric with ``persist_wiki_page`` and ``run_lint``. A
        # YAML scalar / dict / null collapses to ``[]`` rather than
        # being iterated character-by-character into apply.
        source_paths = frontmatter_str_list(post.metadata, "sources")

        op = FixOperation(
            kind="reconcile_provenance",
            path=issue.path,
            source_paths=source_paths,
            expected_hash=bytes_sha256(file_bytes),
        )
        return FixProposal(
            proposal_id=str(uuid.uuid4()),
            issue_kind=issue.kind,
            issue_path=issue.path,
            issue_detail=issue.detail,
            issue_line=None,
            operations=[op],
            rationale=(
                f"reconcile {len(source_paths)} provenance edge(s) "
                "from frontmatter sources:"
            ),
            source="heuristic",
        )
