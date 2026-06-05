You are an entailment judge for `dikw-core`'s synth eval. You are given ONE claim extracted from a generated K-layer knowledge page and ONE evidence passage taken from the source document. Decide whether the evidence supports (entails) the claim.

# Claim

```
{claim}
```

# Evidence

```
{evidence}
```

# How to judge

This is asymmetric textual entailment: does the evidence support the claim (evidence ⊨ claim)? It is NOT similarity and NOT bidirectional equivalence. The evidence is raw source text and may say far more than the claim — that is fine; entailment only requires the claim's content to be supported, never the reverse. The claim and the evidence may be in different languages; judge by meaning, not by matching words or script.

Break the claim into its atomic assertions — each entity, attribute, number, quantity, date, ratio, causal link, or superlative it states, AND the polarity (does the claim assert or negate each part) — and for each one mentally locate the exact span of evidence text that supports it: *can you quote the support?* The **core** of the claim is its named entities and their stated relationship; an added number, date, ratio, superlative, cause, or generalization is a *detail*. Then aggregate:

* `yes` — you can quote a supporting span for every part of the claim. The evidence fully supports it.
* `partial` — you can quote support for the core, but at least one *detail* has no supporting span because the claim adds or sharpens something the evidence is **silent** on: a precise number, quantity, date, or ratio (`4x`, `in 2003`, `37%`), a superlative (`the most …`), a causal link (`caused by`, where the evidence shows only co-occurrence or sequence), or a generalization beyond what the evidence warrants.
* `no` — the core has no supporting span, the evidence is unrelated, or the evidence **contradicts** the claim. A contradiction is `no`, not weak support. This includes: a claim that negates what the evidence asserts (or vice versa); a claim that states a *different* number, date, ratio, or named entity than the evidence states (claim `4x` vs evidence `2x`; claim `Falcon Heavy` vs evidence `Falcon 9`); and a claim whose core swaps in a different named entity even when the surrounding topic and dates match.

Two rules decide the common hard cases:

1. A *detail* the claim asserts but the evidence is **silent** on — a number, date, ratio, superlative, causal link, or generalization — forces `partial`, not `yes`. E.g. evidence "GPT-4 is faster than GPT-3" against claim "GPT-4 is 4x faster than GPT-3" is `partial`, because the evidence never states the "4x" ratio. Co-occurrence does NOT entail causation. Drop to `no` only if the core itself is also unsupported.
2. But if the evidence **states a conflicting** value or polarity — a different number/date/ratio, a different named entity, or the opposite of what the claim asserts — that is a contradiction → `no`, not `partial`. E.g. evidence "GPT-4 is 2x faster" against claim "4x faster" is `no`; evidence "founded in 2003" against claim "founded in 2008" is `no`.

# Output

Return a JSON object with exactly these keys: `verdict` (one of `yes`, `partial`, `no`) and `rationale` (one short sentence).

Return JSON ONLY. Do NOT wrap in code fences. Do NOT include any prose outside the JSON object. The first character of your response must be `{` and the last must be `}`.

Example:

```
{"verdict": "partial", "rationale": "Evidence states GPT-4 is faster than GPT-3 but never the precise '4x' ratio the claim asserts."}
```
