"""Replay exported elon-musk knowledge pages with issue #294's format slips.

No LLM calls or writes to the input base. This measures parser recovery and
content preservation, not the quality of newly generated pages.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import types
from pathlib import Path

import frontmatter
import yaml

from dikw_core.domains.knowledge.synthesize import SynthesisError, parse_synthesis_response


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--knowledge-dir", type=Path, required=True)
    parser.add_argument("--before-ref", default="388b828")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    before = types.ModuleType("dikw_core.domains.knowledge._before_294")
    before.__package__ = "dikw_core.domains.knowledge"
    sys.modules[before.__name__] = before
    source = subprocess.check_output(
        ["git", "show", f"{args.before_ref}:src/dikw_core/domains/knowledge/synthesize.py"],
        text=True,
        encoding="utf-8",
    )
    exec(compile(source, "before_294.py", "exec"), before.__dict__)

    blocks = []
    categories = set()
    corpus_hash = hashlib.sha256()
    for path in sorted(args.knowledge_dir.rglob("*.md")):
        post = frontmatter.load(path)
        if "sources/elon-musk.md" not in post.get("sources", []):
            continue
        category = path.parent.relative_to(args.knowledge_dir).as_posix()
        categories.add(category)
        corpus_hash.update(path.read_bytes())
        tags = yaml.safe_dump({"tags": post.get("tags", [])}, allow_unicode=True).strip()
        blocks.append(
            f'<page category="{category}" slug="{path.stem}">\n'
            f"---\n{tags}\n---\n\n{post.content}\n</page>"
        )
    assert blocks, "no elon-musk pages in input directory"
    raw = "\n\n".join(blocks)
    kwargs = {
        "source_path": "sources/elon-musk.md",
        "allowed_categories": tuple(sorted(categories)),
        "finish_reason": "end_turn",
    }
    expected = parse_synthesis_response(raw, **kwargs)

    def signature(pages):
        return [(p.path, p.title, p.body, p.tags, p.sources) for p in pages]

    rows = {}
    for shape, text in {
        "valid": raw,
        "missing_closes": raw.replace("</page>", ""),
        "bare_openers": raw.replace("<page category=", "<page>\n<page category="),
        "quoted_instruction": "Emit ZERO `<page>` blocks if covered.\n" + raw,
    }.items():
        try:
            old = before.parse_synthesis_response(text, **kwargs)
            old_result = {"accepted": True, "pages": len(old)}
        except before.SynthesisError as exc:
            old_result = {"accepted": False, "surviving_pages": len(getattr(exc, "pages", []))}
        new = parse_synthesis_response(text, **kwargs)
        assert signature(new) == signature(expected), shape
        rows[shape] = {"before": old_result, "after_pages": len(new), "content_identical": True}

    for reason in ("max_tokens", "length"):
        try:
            parse_synthesis_response(raw, **(kwargs | {"finish_reason": reason}))
        except SynthesisError as exc:
            assert exc.retry
        else:
            raise AssertionError(f"{reason} truncation was accepted")
    result = {
        "input_pages": len(blocks),
        "input_sha256": corpus_hash.hexdigest(),
        "rows": rows,
        "truncation_controls": "passed",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
