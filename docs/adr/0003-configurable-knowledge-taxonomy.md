# Configurable hierarchical knowledge taxonomy (category) replaces fixed page-type folders

**Status**: Accepted, 0.5.0.

## Context

Through 0.4.x the K layer classified every page on a single hard-wired axis,
the page `type` (`entity` / `concept` / `note`), and filed it two levels deep
at `knowledge/<type-英文复数>/<slug>.md` (`entities/` `concepts/` `notes/`,
via an English-pluralisation map `type_to_folder`). `type` was already
configurable as a flat string list (`schema.page_types`), but the folder
naming, the two-level shape, the reverse-parse (`type_from_path` assumed
exactly three path segments), and the synth prompt all baked in the
type-folder convention.

Enterprises classify knowledge by their own taxonomy — product lines,
technical domains, processes, customers — which is (a) hierarchical, not flat,
and (b) often non-English, where `type + "s"` pluralisation is meaningless
(`产品s`). The fixed single-axis `type` could not express this.

The key reframing (from the user): **`type` is not a separate concept from
`category` — it is one degenerate instance of it.** `entity`/`concept`/`note`
is simply the default taxonomy (depth 1, three nodes). So the change is not
"add a category concept alongside type" but "generalise the one hard-wired
classification axis into a configurable hierarchical tree."

## Decision

One classification axis, `category`, declared as a hierarchical tree in config:

- `schema.categories` (replaces `schema.page_types`) is a list of nodes, each a
  `/`-separated path of arbitrary depth (`产品/移动端`, `技术/架构`) with an
  optional `desc`. `schema.fallback` (default `未分类`) is the bucket for pages
  synth cannot confidently classify. The default `categories` is
  `entity`/`concept`/`note` (carrying the historic prompt descriptions), so a
  fresh `dikw init` behaves as before.
- **Closed set.** The taxonomy is authoritative: synth's LLM may only file a
  page under a declared path (chosen by emitting `category="<path>"`); an
  unrecognised value falls to `fallback`. This is Karpathy's rule — deterministic
  scoping (config defines the valid categories) feeds probabilistic reasoning
  (the LLM picks which one). A wrong category is a fixable re-file; an invented
  folder would be irreversible taxonomy drift.
- **On disk:** `knowledge/<category-path>/<slug>.md`. The category path is used
  verbatim as the folder (Unicode allowed — the closed set means the folder name
  is exactly the operator-authored config string, never unsanitised LLM output);
  config-load validates each segment (NFC, reject `..` / absolute / backslash /
  filesystem-reserved chars) and the existing base-containment guard covers the
  rest. `type_to_folder` / `_TYPE_FOLDERS` pluralisation is deleted.
- **Frontmatter:** the page-classification key is `category:` (replaces `type:`).
  Clean break — no `type` / `page_types` alias. `documents` has no `type` column,
  so this is purely an on-disk + config change with no DB schema migration.
- **Prompt overrides** ride alongside (see the companion change): `synth.prompt_path`
  and `lint.fixer_prompts` let a base supply its own authoring prompts, validated
  against a placeholder/output contract at load.

Migration: no dedicated tool. The upgrade guard detects a pre-taxonomy base and
raises `BaseUpgradeRequired`, directing the user to declare `categories`, clear
`knowledge/` + `.dikw/`, and re-run `dikw client synth` to rebuild K under the
new taxonomy (alpha policy — rebuild over carry-forward).

## Consequences

- **+** Enterprises express their own classification; the engine ships a sensible
  default and stays out of the way.
- **+** Read paths are untouched — `list_pages` / `read_page` / graph / wikilink
  resolve / lint are storage- and title-driven and directory-agnostic, so arbitrary
  nesting has zero retrieval impact. Only the write path, the reverse-parse
  (`type_from_path` → `category_from_path`), the scaffold, and the synth prompt change.
- **+** No DB migration (`type` was never a column).
- **−** Breaking on-disk + config change. Existing bases must re-synth (see migration).
- **−** Single axis: a base that adopts a business taxonomy loses the
  entity/concept/note distinction unless it encodes it into the tree
  (`技术/概念`, …). Accepted deliberately over an orthogonal two-axis folder shape,
  which doubled prompt/index/lint complexity for a distinction most enterprise
  taxonomies don't want.

## Alternatives considered

1. **Keep `type` flat-configurable, only fix pluralisation.** Doesn't deliver
   hierarchy, which is the actual enterprise need. Rejected.
2. **Orthogonal two-axis folders (`<business>/<type>/`).** Keeps entity/concept/note
   as a second dimension. Rejected — the user explicitly does not want both axes,
   and it forces every prompt/index/lint consumer to reason about two dimensions.
3. **Open taxonomy (LLM invents category paths from a description).** Maximally
   flexible but violates "scoping is deterministic" — every page could spawn a new
   folder, with no operator control and unbounded drift. Rejected.
4. **slug + label split (ASCII folder, Chinese display label).** More cross-platform-robust
   but less WYSIWYG and more verbose config. Rejected — the closed set makes raw
   Unicode folders safe, and the user wanted the on-disk folders to *be* the Chinese names.

## References

- `docs/design.md` "On-Disk Base Layout" + "The Four Layers" — current contract.
- ADR-0004 (drop generated index.md/log.md) — companion K-layout change shipped together.
- CHANGELOG `[0.5.0]` — the release entry + ⚠️ Breaking notes.
