Your page-authoring policy is in the system instructions. This message carries the operational detail those rules defer to (page length, linking density, tags, output language), the exact **Output format**, and finally the per-call inputs — the category list (fixed for this knowledge base), this call's section numbers, the knowledge-base context, and the source text.

## Page length

The atomicity rule is standing policy; these are the length norms it defers to.

- A body under ~200 characters is **rarely** worth its own page — a bare stub or TODO is better folded into a related page and referenced with a `[[wikilink]]` from there.
- Typical atomic pages run **300–1500 characters**: long enough to stand alone, short enough to stay single-subject. Do not pad to reach a length; if a subject genuinely warrants only a sentence, fold it into a neighbouring page rather than emit a stub.
- Atomic does **not** mean thin. Before closing a page, make sure it captures every substantive fact this section offers about its subject — concise means no padding, never dropped facts.

## Fan-out

This call sees only **one section** of the source document — the section numbers and the page cap appear under **Task** near the end of this prompt, and the section text follows as the **SOURCE DOCUMENT** block at the very end. Identify the distinct concepts, entities, and notes in this section that deserve their own page in the knowledge base. Output one `<page>` block per item.

- Emit **zero** blocks if this section contains nothing worth a knowledge page (boilerplate, navigation, copyright notices, table of contents).
- Emit at most the stated **page cap**. If the section has fewer distinct topics, emit fewer.
- Emit pages in **descending order of importance**, and never open a `<page>` block you cannot finish — if your output budget cuts the response short, the least important page should be the one lost.
- Reuse the section's heading structure as a hint for natural page boundaries, but do not feel bound by it — merge two H2 sections into one page when they cover the same atomic subject, or split one H2 into multiple pages when it conflates topics.

## Linking

The honest-linking and faithfulness rules are standing policy; these are the mechanics they defer to.

1. Link **inline**, where the reference occurs in the prose — never as a trailing "see also" list. Every `[[wikilink]]` target must be one of: **(a)** a page listed in the knowledge-base context near the end of this prompt (write its title **verbatim**), **(b)** the title of another `<page>` you emit in this response, or **(c)** a concept or entity clearly substantial enough to deserve its own page later — a deliberate forward link that `dikw client lint` tracks until the page exists. Do **not** wikilink names, places, or terms that merely appear in passing: a link must point at something a reader would genuinely open.
2. **Link density**: a well-linked page naturally lands around **2–4 wikilinks per 500 characters** once every load-bearing reference is linked — substantially more than that usually signals manufactured links, which dilute the graph and lower grounding. When in doubt, leave plain text.

## Tags

Pick **2–5 short tags** per page. Prefer a small, reusable vocabulary over bespoke phrases — for example `entity`, `concept`, `process`, `historical`, `technical`, `definition`. Tags may be namespaced (`area/topic`, e.g. `ml/architecture`), but a single page should stay within **one** namespace domain — mixing `ml/...` and `biology/...` on one page is a signal it is really two pages.

## Output language

The source-language rule is standing policy: emit page titles, body text, tags, and new wikilink titles in the **dominant language** of the source section — never translate a concept into another language. These are the mechanics it defers to.

- For mixed-language sources, follow the language of the chunk you are summarising; a single page should not switch languages mid-paragraph.
- The `slug` must be lowercase ASCII kebab-case regardless of title language. For a non-ASCII title, use a short pinyin or English-equivalent slug (e.g. title `神经网络` → slug `neural-network` or `shen-jing-wang-luo`); the page title itself stays in the source language. The `category` path may be non-ASCII — copy it verbatim from the Category list.

## Example

Two worked examples. Note the atomic single-subject body, the inline `[[wikilinks]]` placed exactly where the prose leans on them, and how each page stays in its source language. (The `category` values below are illustrative — in real output, copy the best-fitting path **verbatim** from the Category list near the end of this prompt.)

<page category="concept" slug="transformer-architecture">
---
tags: [concept, deep-learning]
---

# Transformer architecture

The transformer is a neural-network architecture that replaces recurrence with [[self-attention]], letting it process every token of a sequence in parallel rather than strictly left-to-right. First introduced for machine translation, it now underpins the modern [[large language model]].

Its core block pairs multi-head [[self-attention]] with a position-wise feed-forward network, wrapped in residual connections and layer normalisation — a unit that stacks cleanly to great depth.
</page>

<page category="entity" slug="qin-shi-huang">
---
tags: [entity, historical]
---

# 秦始皇

秦始皇（前259–前210年）是[[秦朝]]的开国皇帝，于公元前221年完成对六国的统一，结束了[[战国时期]]长达数百年的割据混战，建立起中国历史上第一个中央集权的统一帝国。

他废除分封、推行郡县制，统一文字、度量衡与货币，为后世两千年的政治制度奠定基础。对外，他连接并扩建北方既有城墙，即[[万里长城]]的前身。其严刑峻法在身后引发动荡，而他奠定的统一格局为[[汉朝]]所继承。
</page>

## Output format

For each page, emit exactly one `<page>` block, wrapped verbatim. Do **not** emit prose outside the blocks.

```
<page category="<category-path>" slug="<slug>">
---
tags: [tag1, tag2]
---

# Page Title

Body paragraphs here. Use [[Wikilinks]] for references.
</page>
```

- `category` is one path copied **verbatim** from the Category list below; omitting the attribute is a **last resort**, per the Closed-taxonomy invariant (the engine then files the page under its fallback bucket for a human to reclassify).
- `slug` is lowercase, kebab-case, ASCII-only. The engine files the page at `knowledge/<category>/<slug>.md`.
- The first line of the body must be an ATX `# Page Title` matching the page title you choose.
- In the front-matter, emit **only** `tags`. Do **not** add `title`, `id`, `category`, `sources`, `created`, `updated`, or `lint` — the engine manages those and silently ignores them if you include them (`title` comes from the body `# Page Title`; `category` and `slug` from the `<page>` attributes).

## Category list

{categories}

## Task

This call covers **section {group_index} of {group_total}** of the source document — emit at most **{max_pages}** `<page>` blocks for this section.

## Knowledge-base context

{existing_pages_section}

**Reusing an existing page is always better than regenerating similar content.** When the section above lists pages — under `Existing knowledge pages` or `Already created in this batch` — scan them before emitting any page and decide:

- **Semantic duplicate** — the candidate states the same fact at the same granularity as a listed page. Emit **ZERO** `<page>` blocks for it; instead reference the existing page via `[[Title]]` in your other pages' bodies. Do not regenerate it.
- **Different facet** — the candidate is a genuinely new angle, sub-topic, or finer slice. Emit a new page and link it to the related existing one. Example: if `[[Elon Musk]]` already exists, a page on `[[SpaceX reusable-rocket program]]` is a new facet, not a duplicate.

This applies to BOTH:
- pages already in the knowledge base (titles may not match exactly — use judgement; prefer reference over regeneration on ambiguity)
- pages just created earlier in this same batch (MUST reference, not regenerate)

(The `Priority targets` list, when present, is the opposite case: those pages do **not** exist yet — creating one when this section covers it is encouraged, never a duplicate.)

Each existing-page bullet is formatted `- Title [slug] (category)`. The `[slug]` is that page's file identifier — use it only to tell apart two pages that share a title; always write the **title** (not the slug) inside `[[ ]]`.

SOURCE DOCUMENT — path: {source_path}

Section headings (in order): {group_outline}

```
{source_body}
```
