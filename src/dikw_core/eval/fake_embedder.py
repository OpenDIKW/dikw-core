"""Deterministic bag-of-words embedder for hermetic eval.

Ships inside the package (unlike the test-only ``tests/fakes.py``) so
``dikw client eval`` can run without API keys or network access. The algorithm is
trivial on purpose: tokenise, hash each word to a fixed-size bucket, L2
normalise the count vector. Semantically similar inputs (overlapping
bag-of-words) cluster together, which is enough to exercise the retrieval
pipeline deterministically at <1ms per doc.

**Not** a replacement for a real embedder — quality is far below any
production model. Use ``--embedder provider`` on ``dikw client eval`` to switch
to the knowledge base's configured provider for real-vector evaluation.
"""

from __future__ import annotations

import hashlib
import math
import re

EMBED_DIM = 64
_WORD = re.compile(r"[A-Za-z]+")


def _tokens(text: str) -> list[str]:
    return [w.lower() for w in _WORD.findall(text)]


def _bucket(tok: str) -> int:
    h = hashlib.sha1(tok.encode("utf-8")).digest()
    return int.from_bytes(h[:4], "big") % EMBED_DIM


class FakeEmbeddings:
    """Deterministic bag-of-words embeddings over a fixed ``EMBED_DIM`` space."""

    async def embed(self, texts: list[str], *, model: str) -> list[list[float]]:
        _ = model
        out: list[list[float]] = []
        for text in texts:
            vec = [0.0] * EMBED_DIM
            for tok in _tokens(text):
                vec[_bucket(tok)] += 1.0
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            out.append([v / norm for v in vec])
        return out


__all__ = ["EMBED_DIM", "FakeEmbeddings"]
