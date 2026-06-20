"""Tests for the MultimodalEmbeddingProvider Protocol contract.

Pins the Protocol's invariants — input/output ordering, batch handling,
text-only and image-only and mixed inputs — using a deterministic
``FakeMultimodalEmbedding`` so the test doesn't depend on any real
provider's HTTP surface.
"""

from __future__ import annotations

from dikw_core.providers.base import MultimodalEmbeddingProvider
from dikw_core.schemas import ImageContent, MultimodalInput
from tests.fakes import FakeMultimodalEmbedding


def test_fake_satisfies_protocol() -> None:
    """Static-duck-typing sanity: the fake implements the Protocol."""
    fake: MultimodalEmbeddingProvider = FakeMultimodalEmbedding(dim=4)
    assert hasattr(fake, "embed")


async def test_text_only_input_produces_vector() -> None:
    fake = FakeMultimodalEmbedding(dim=4)
    vectors = await fake.embed(
        [MultimodalInput(text="hello world")], model="fake-mm-v1"
    )
    assert len(vectors) == 1
    assert len(vectors[0]) == 4


async def test_image_only_input_produces_vector() -> None:
    fake = FakeMultimodalEmbedding(dim=8)
    vectors = await fake.embed(
        [
            MultimodalInput(
                images=[ImageContent(bytes=b"\x89PNG\r\n", mime="image/png")]
            )
        ],
        model="fake-mm-v1",
    )
    assert len(vectors) == 1
    assert len(vectors[0]) == 8


async def test_output_order_matches_input_order() -> None:
    """Deterministic provider: same input twice → same vector. Inputs in a
    batch must come back in input order so callers can pair with their
    source rows."""
    fake = FakeMultimodalEmbedding(dim=4)
    inputs = [
        MultimodalInput(text="alpha"),
        MultimodalInput(text="beta"),
        MultimodalInput(text="gamma"),
    ]
    a = await fake.embed(inputs, model="fake-mm-v1")
    b = await fake.embed(inputs, model="fake-mm-v1")
    assert a == b
    # Order: the second input's vector must differ from the first's.
    assert a[0] != a[1]


async def test_text_and_image_distinguished() -> None:
    """A text input and an image input — even with conceptually similar
    payload — produce distinct vectors so the embedding actually carries
    modality signal."""
    fake = FakeMultimodalEmbedding(dim=4)
    text_vec = (
        await fake.embed([MultimodalInput(text="png")], model="x")
    )[0]
    img_vec = (
        await fake.embed(
            [MultimodalInput(images=[ImageContent(bytes=b"png", mime="image/png")])],
            model="x",
        )
    )[0]
    assert text_vec != img_vec


async def test_empty_batch_returns_empty() -> None:
    fake = FakeMultimodalEmbedding(dim=4)
    assert await fake.embed([], model="any") == []


# ---- Gitee multimodal request shape tests (no network) ---------------------
#
# Gitee's multimodal embeddings endpoint speaks one wire format across all
# multimodal models it serves (Qwen3-VL-Embedding-8B, jina-clip-v2, …):
# per-modality input dicts, ``[{"text": "..."}, {"image": "data:..."}]``.
# The serializer is therefore model-agnostic; these tests pin its shape
# without mentioning any specific model name.


def test_serialize_text_only() -> None:
    """Text-only input → ``{"text": "..."}``."""
    from dikw_core.providers.gitee_multimodal import _serialize_input

    out = _serialize_input(MultimodalInput(text="hello"))
    assert out == {"text": "hello"}


def test_serialize_image_only() -> None:
    """Image-only input → ``{"image": "data:<mime>;base64,..."}``."""
    from dikw_core.providers.gitee_multimodal import _serialize_input

    out = _serialize_input(
        MultimodalInput(
            images=[ImageContent(bytes=b"\x89PNG\r\n", mime="image/png")]
        )
    )
    assert isinstance(out, dict)
    assert set(out.keys()) == {"image"}
    assert out["image"].startswith("data:image/png;base64,")


def test_serialize_rejects_combined_text_and_image() -> None:
    """Gitee's multimodal endpoint embeds per-modality; combined inputs must
    raise loudly rather than silently drop a side."""
    import pytest

    from dikw_core.providers.base import ProviderError
    from dikw_core.providers.gitee_multimodal import _serialize_input

    with pytest.raises(ProviderError, match="per-modality"):
        _serialize_input(
            MultimodalInput(
                text="caption",
                images=[ImageContent(bytes=b"\x89PNG", mime="image/png")],
            )
        )


def test_serialize_rejects_multiple_images_per_input() -> None:
    """Each MultimodalInput maps to one wire input; >1 image would lose
    pairing between source and embedding."""
    import pytest

    from dikw_core.providers.base import ProviderError
    from dikw_core.providers.gitee_multimodal import _serialize_input

    with pytest.raises(ProviderError, match="one image"):
        _serialize_input(
            MultimodalInput(
                images=[
                    ImageContent(bytes=b"\x89PNG", mime="image/png"),
                    ImageContent(bytes=b"\xff\xd8", mime="image/jpeg"),
                ]
            )
        )


def test_factory_resolves_gitee_multimodal() -> None:
    """``build_multimodal_embedder('gitee_multimodal')`` resolves to the
    unified Gitee provider — model-agnostic, the model name in
    ``assets.multimodal.model`` discriminates which Gitee model runs."""
    from dikw_core.providers import build_multimodal_embedder
    from dikw_core.providers.gitee_multimodal import GiteeMultimodalEmbedding

    inst = build_multimodal_embedder("gitee_multimodal", api_key_env="GITEE_API_KEY")
    assert isinstance(inst, GiteeMultimodalEmbedding)


def test_factory_rejects_legacy_gitee_qwen_vl_literal() -> None:
    """The transient ``gitee_qwen_vl`` literal split one Gitee endpoint into
    two provider strings even though the wire shape is identical. After the
    unify pass it must be rejected loudly so stale ``dikw.yml`` configs
    surface immediately instead of silently falling back."""
    import pytest

    from dikw_core.providers import build_multimodal_embedder
    from dikw_core.providers.base import ProviderError

    with pytest.raises(ProviderError, match="unknown multimodal embedding provider"):
        build_multimodal_embedder("gitee_qwen_vl", api_key_env="GITEE_API_KEY")
