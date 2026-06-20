"""Live probe for the multimodal embedding leg.

Hits the configured ``assets.multimodal`` endpoint with several real input
shapes and reports observed batching / dim / cross-modal cosine behavior in
a ``dikw check``-style table. Used to validate provider quirks (Gitee
``Qwen3-VL-Embedding-2B`` vs ``8B``, batch caps, ``dimensions`` parameter
handling) before we rely on those quirks in production code.

Why this lives here and not in ``tests/``:

* Real API spend per run; not hermetic.
* Output is for human eyes — a table + verdict, not pass/fail asserts.
* Settles questions ``test_multimodal_provider.py`` deliberately can't:
  every test in that file uses ``FakeMultimodalEmbedding``.

Why a separate script and not a ``dikw`` subcommand: this is one-shot
diagnostic, not a feature. Same pedigree as ``run_phase15_from_snapshot.py``.

Usage
-----

::

    set -a && source .env && set +a
    uv run python -m evals.tools.probe_multimodal \\
        --config "C:\\Users\\HE LE\\scratch-dikw-thinking-models\\dikw.yml" \\
        --image  "C:\\Users\\HE LE\\Project\\second-brain\\.tmp\\思维模型\\images\\images\\00050.jpeg" \\
        --image  "C:\\Users\\HE LE\\Project\\second-brain\\.tmp\\思维模型\\images\\images\\00051.jpeg" \\
        --image  "C:\\Users\\HE LE\\Project\\second-brain\\.tmp\\思维模型\\images\\images\\00052.jpeg" \\
        --image  "C:\\Users\\HE LE\\Project\\second-brain\\.tmp\\思维模型\\images\\images\\00053.jpeg"

Bypasses ``GiteeMultimodalEmbedding`` and constructs requests via ``httpx``
so the wire payload (especially the ``dimensions`` field, which the production
provider does not yet send) can be controlled per-probe.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import math
import sys
import time
from pathlib import Path
from typing import Any

import httpx
from rich.console import Console
from rich.table import Table

from dikw_core.config import load_config
from dikw_core.providers.base import _resolve_key

_DEFAULT_BASE_URL = "https://ai.gitee.com/v1"
_HTTP_TIMEOUT = 60.0
_PROBE_TEXTS = [
    "一只蓝色的猫",
    "深度学习模型",
    "黑天鹅事件",
    "供需曲线图",
]
_DEFAULT_QUERY = "一张猫的照片"


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0.0 or nb == 0.0:
        return float("nan")
    return dot / (na * nb)


def _encode_image(path: Path) -> str:
    """Read an image file and return a ``data:`` URI for the wire payload."""
    raw = path.read_bytes()
    suffix = path.suffix.lower()
    mime = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }.get(suffix, "application/octet-stream")
    b64 = base64.b64encode(raw).decode("ascii")
    return f"data:{mime};base64,{b64}"


async def _embed_call(
    client: httpx.AsyncClient,
    *,
    base_url: str,
    api_key: str,
    model: str,
    inputs: list[dict[str, str]],
    dimensions: int | None,
) -> tuple[list[list[float]], int]:
    """One round-trip; returns ``(vectors, elapsed_ms)``."""
    payload: dict[str, Any] = {"model": model, "input": inputs}
    if dimensions is not None:
        payload["dimensions"] = dimensions
    started = time.perf_counter()
    resp = await client.post(
        f"{base_url}/embeddings",
        json=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    resp.raise_for_status()
    data = resp.json()
    rows = sorted(data.get("data", []), key=lambda r: r.get("index", 0))
    return [list(r["embedding"]) for r in rows], elapsed_ms


def _summarize(
    name: str,
    n_in: int,
    vectors: list[list[float]] | None,
    elapsed_ms: int,
    error: str | None,
    expected_dim: int | None,
) -> dict[str, Any]:
    """Shape one probe's outcome into a row dict for the rich table."""
    if error is not None:
        return {
            "name": name,
            "n_in": n_in,
            "n_out": 0,
            "dim": 0,
            "ms": elapsed_ms,
            "ok": False,
            "detail": error,
        }
    assert vectors is not None
    n_out = len(vectors)
    dim = len(vectors[0]) if vectors else 0
    ok_count = n_out == n_in
    ok_dim = expected_dim is None or dim == expected_dim
    detail_parts: list[str] = []
    if not ok_count:
        detail_parts.append(f"expected {n_in} vectors, got {n_out}")
    if not ok_dim:
        detail_parts.append(f"expected dim={expected_dim}, got {dim}")
    if ok_count and ok_dim:
        detail_parts.append("OK")
    return {
        "name": name,
        "n_in": n_in,
        "n_out": n_out,
        "dim": dim,
        "ms": elapsed_ms,
        "ok": ok_count and ok_dim,
        "detail": "; ".join(detail_parts),
    }


async def _run_probes(
    base_url: str,
    api_key: str,
    model: str,
    images: list[Path],
    query: str,
    batch_size: int,
    dim: int | None,
) -> tuple[list[dict[str, Any]], list[tuple[str, float]], list[tuple[str, str, float]]]:
    """Execute the four probes in sequence; return rows + cosine top-list +
    pairwise image-image cosines (for diagnosing degenerate embeddings)."""
    rows: list[dict[str, Any]] = []
    cosines: list[tuple[str, float]] = []
    pairwise: list[tuple[str, str, float]] = []
    image_vectors: dict[str, list[float]] = {}
    # Force a fresh connection per request — Gitee's keepalive drops sockets
    # mid-batch (per gitee_multimodal.py:138-143 comment).
    limits = httpx.Limits(max_keepalive_connections=0)
    async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT, limits=limits) as client:
        text_inputs = [{"text": t} for t in _PROBE_TEXTS[:batch_size]]
        try:
            vectors, ms = await _embed_call(
                client,
                base_url=base_url,
                api_key=api_key,
                model=model,
                inputs=text_inputs,
                dimensions=dim,
            )
            rows.append(
                _summarize("text-only batch", len(text_inputs), vectors, ms, None, dim)
            )
        except Exception as e:
            rows.append(
                _summarize(
                    "text-only batch",
                    len(text_inputs),
                    None,
                    0,
                    f"{type(e).__name__}: {e}",
                    dim,
                )
            )

        image_batch = [_encode_image(p) for p in images[:batch_size]]
        image_inputs = [{"image": img} for img in image_batch]
        try:
            vectors, ms = await _embed_call(
                client,
                base_url=base_url,
                api_key=api_key,
                model=model,
                inputs=image_inputs,
                dimensions=dim,
            )
            rows.append(
                _summarize("image-only batch", len(image_inputs), vectors, ms, None, dim)
            )
        except Exception as e:
            rows.append(
                _summarize(
                    "image-only batch",
                    len(image_inputs),
                    None,
                    0,
                    f"{type(e).__name__}: {e}",
                    dim,
                )
            )

        # 2 text + remaining slots filled with images, capped at batch_size.
        mix_n_text = min(2, batch_size)
        mix_n_image = max(0, batch_size - mix_n_text)
        mixed_inputs: list[dict[str, str]] = [
            {"text": t} for t in _PROBE_TEXTS[:mix_n_text]
        ]
        for i in range(mix_n_image):
            if i < len(image_batch):
                mixed_inputs.append({"image": image_batch[i]})
        try:
            vectors, ms = await _embed_call(
                client,
                base_url=base_url,
                api_key=api_key,
                model=model,
                inputs=mixed_inputs,
                dimensions=dim,
            )
            rows.append(
                _summarize("mixed batch", len(mixed_inputs), vectors, ms, None, dim)
            )
        except Exception as e:
            rows.append(
                _summarize(
                    "mixed batch",
                    len(mixed_inputs),
                    None,
                    0,
                    f"{type(e).__name__}: {e}",
                    dim,
                )
            )

        # Cross-modal cosine sanity: one query text + each image individually
        # so a per-input shape bug doesn't poison the comparison. Single
        # inputs always work even on broken-batching providers.
        try:
            q_vec_rows, _ = await _embed_call(
                client,
                base_url=base_url,
                api_key=api_key,
                model=model,
                inputs=[{"text": query}],
                dimensions=dim,
            )
            q_vec = q_vec_rows[0]
            ms_total = 0
            for path in images:
                v_rows, ms = await _embed_call(
                    client,
                    base_url=base_url,
                    api_key=api_key,
                    model=model,
                    inputs=[{"image": _encode_image(path)}],
                    dimensions=dim,
                )
                ms_total += ms
                image_vectors[path.name] = v_rows[0]
                cosines.append((path.name, _cosine(q_vec, v_rows[0])))
            cosines.sort(key=lambda x: -x[1])
            names = list(image_vectors.keys())
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    pairwise.append(
                        (
                            names[i],
                            names[j],
                            _cosine(image_vectors[names[i]], image_vectors[names[j]]),
                        )
                    )
            rows.append(
                {
                    "name": "cross-modal cosine",
                    "n_in": 1 + len(images),
                    "n_out": 1 + len(images),
                    "dim": len(q_vec),
                    "ms": ms_total,
                    "ok": True,
                    "detail": (
                        f"top: {cosines[0][0]}={cosines[0][1]:.3f}"
                        if cosines
                        else "no images probed"
                    ),
                }
            )
        except Exception as e:
            rows.append(
                {
                    "name": "cross-modal cosine",
                    "n_in": 1 + len(images),
                    "n_out": 0,
                    "dim": 0,
                    "ms": 0,
                    "ok": False,
                    "detail": f"{type(e).__name__}: {e}",
                }
            )

    return rows, cosines, pairwise


def _render(
    rows: list[dict[str, Any]],
    cosines: list[tuple[str, float]],
    pairwise: list[tuple[str, str, float]],
    *,
    base_url: str,
    model: str,
    dim_sent: int | None,
) -> None:
    console = Console()
    console.print()
    console.print(f"[bold]endpoint:[/bold] {base_url}/embeddings")
    console.print(f"[bold]model:   [/bold] {model}")
    console.print(
        f"[bold]dim sent:[/bold] {dim_sent if dim_sent is not None else '(omitted)'}"
    )
    console.print()

    table = Table(title="probe_multimodal", show_lines=False)
    table.add_column("probe", no_wrap=True)
    table.add_column("in", justify="right")
    table.add_column("out", justify="right")
    table.add_column("dim", justify="right")
    table.add_column("ms", justify="right")
    table.add_column("status", no_wrap=True)
    table.add_column("detail")
    for row in rows:
        status = "[green]✓ OK[/green]" if row["ok"] else "[red]✗ FAIL[/red]"
        table.add_row(
            row["name"],
            str(row["n_in"]),
            str(row["n_out"]),
            str(row["dim"]),
            str(row["ms"]),
            status,
            row["detail"],
        )
    console.print(table)

    if cosines:
        console.print()
        console.print(
            "[bold]cross-modal cosine[/bold] (query text vs each image, 8-decimal):"
        )
        for name, sim in cosines:
            console.print(f"  {sim:+.8f}  {name}")
        console.print()
    if pairwise:
        console.print(
            "[bold]pairwise image-image cosine[/bold] "
            "(degenerate-embedding sniff test; ~1.0 = collapse):"
        )
        for a, b, sim in pairwise:
            console.print(f"  {sim:+.8f}  {a} ↔ {b}")
        console.print()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Live probe for the multimodal embedding leg. Hits the configured "
            "endpoint with text-only / image-only / mixed batches plus a "
            "cross-modal cosine sanity check, then prints a table."
        ),
    )
    p.add_argument(
        "--config",
        type=Path,
        required=True,
        help="path to dikw.yml (assets.multimodal section is read)",
    )
    p.add_argument(
        "--image",
        type=Path,
        action="append",
        required=True,
        help="path to a real image (repeat for batch tests; need ≥ batch_size)",
    )
    p.add_argument(
        "--query",
        type=str,
        default=_DEFAULT_QUERY,
        help=f"query text for the cross-modal cosine probe (default: {_DEFAULT_QUERY!r})",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="batch size for the batch probes (default: 4)",
    )
    p.add_argument(
        "--dim",
        type=int,
        default=None,
        help=(
            "send this value as the 'dimensions' wire field. "
            "Defaults to assets.multimodal.dim from dikw.yml. "
            "Pass 0 to omit the field entirely (test the model's native default)."
        ),
    )
    p.add_argument(
        "--model",
        type=str,
        default=None,
        help=(
            "override the model name from dikw.yml (e.g. probe a different "
            "multimodal model on the same endpoint without editing the wiki config)."
        ),
    )
    return p.parse_args()


async def _main() -> int:
    args = _parse_args()

    cfg = load_config(args.config)
    mm = cfg.assets.multimodal
    if mm is None:
        print(
            "ERROR: dikw.yml at "
            f"{args.config} has no assets.multimodal section",
            file=sys.stderr,
        )
        return 2

    base_url = (mm.base_url or _DEFAULT_BASE_URL).rstrip("/")
    model = args.model if args.model else mm.model

    if args.dim is None:
        dim_to_send: int | None = mm.dim
    elif args.dim == 0:
        dim_to_send = None
    else:
        dim_to_send = args.dim

    for path in args.image:
        if not path.exists():
            print(f"ERROR: image not found: {path}", file=sys.stderr)
            return 2

    api_key = _resolve_key(None, cfg.provider.embedding_api_key_env)

    rows, cosines, pairwise = await _run_probes(
        base_url=base_url,
        api_key=api_key,
        model=model,
        images=list(args.image),
        query=args.query,
        batch_size=args.batch_size,
        dim=dim_to_send,
    )

    _render(
        rows,
        cosines,
        pairwise,
        base_url=base_url,
        model=model,
        dim_sent=dim_to_send,
    )

    failed = [r["name"] for r in rows if not r["ok"]]
    if failed:
        print(f"FAIL: {', '.join(failed)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(_main()))
