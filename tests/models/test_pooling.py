"""`pool_token_dim`: the memory-lean pooling of the token dimension.

`_pool_logits` used to open with ``logits.float()`` — a float32 copy of the
entity logits, the largest tensor in a training step and twice the size of the
bfloat16 original, which autograd then held until backward had run. These tests
pin the two things that replacement has to get right: it must not save a
float32 copy, and it must pool to the same values as the code it replaced.
"""

import math

import pytest
import torch

from d3text.models.models import pool_token_dim

POOLINGS = ("logsumexp", "logmeanexp", "max", "mean")


def reference(logits, pooling, dim=1):
    """The implementation `pool_token_dim` replaced, verbatim."""
    x = logits.float()
    if pooling == "logsumexp":
        pooled = torch.logsumexp(x, dim=dim)
    elif pooling == "logmeanexp":
        pooled = torch.logsumexp(x, dim=dim) - math.log(x.shape[dim])
    elif pooling == "max":
        pooled = torch.amax(x, dim=dim)
    elif pooling == "mean":
        pooled = torch.mean(x, dim=dim)
    else:
        raise ValueError(pooling)
    return pooled.to(logits.dtype)


def _saved(fn, logits, pooling):
    """(result, shapes-and-dtypes autograd packed into the graph)."""
    saved: list[tuple[tuple[int, ...], torch.dtype]] = []

    def pack(tensor):
        saved.append((tuple(tensor.shape), tensor.dtype))
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        out = fn(logits, pooling)
    return out, saved


@pytest.mark.parametrize("pooling", POOLINGS)
def test_pooling_saves_no_float32_copy_of_the_logits(pooling):
    """The behavioural red: one full-size float32 save before, none after."""
    logits = torch.randn(2, 3000, 37, dtype=torch.bfloat16, requires_grad=True)

    _, new = _saved(pool_token_dim, logits, pooling)
    _, old = _saved(reference, logits, pooling)

    full_float32 = (tuple(logits.shape), torch.float32)
    assert new.count(full_float32) == 0
    # the guard that the red is real: the old path did save exactly one
    assert old.count(full_float32) == (1 if pooling != "mean" else 0)


@pytest.mark.parametrize("pooling", POOLINGS)
def test_pooling_is_bitwise_identical_in_bfloat16(pooling):
    """bfloat16 is the autocast path every GPU run takes.

    The float32 arithmetic is reordered — summed slice by slice rather than in
    one reduction — but the difference is far below a bfloat16 ulp, so it does
    not survive the cast back and the pooled logits are unchanged.
    """
    torch.manual_seed(0)
    logits = torch.randn(3, 2600, 41, dtype=torch.bfloat16) * 4
    assert torch.equal(
        pool_token_dim(logits, pooling), reference(logits, pooling)
    )


@pytest.mark.parametrize("pooling", POOLINGS)
def test_pooling_gradients_match(pooling):
    torch.manual_seed(0)
    base = torch.randn(2, 2100, 23, dtype=torch.bfloat16) * 4
    grads = []
    for fn in (pool_token_dim, reference):
        logits = base.clone().requires_grad_(True)
        out = fn(logits, pooling)
        torch.manual_seed(1)
        out.backward(torch.randn_like(out.float()).to(out.dtype))
        grads.append(logits.grad)
    # bfloat16 has an 8-bit significand; the reordering lands well inside it
    assert torch.allclose(grads[0].float(), grads[1].float(), atol=1e-4)


@pytest.mark.parametrize("pooling", POOLINGS)
@pytest.mark.parametrize("tokens", [1, 7, 2048, 2049, 4096, 5000])
def test_pooling_matches_across_the_chunk_boundary(pooling, tokens):
    """Slices are 2048 tokens wide, so the partial trailing slice, the exact
    multiple, and the shorter-than-one-slice case are all distinct paths."""
    torch.manual_seed(0)
    logits = torch.randn(2, tokens, 13, dtype=torch.bfloat16) * 4
    assert torch.equal(
        pool_token_dim(logits, pooling), reference(logits, pooling)
    )


@pytest.mark.parametrize("pooling", POOLINGS)
def test_pooling_handles_fully_masked_documents(pooling):
    """`forward` masks padding to `_neg_inf` (-1e9) before pooling, and a
    document that is all padding must not become a NaN."""
    logits = torch.full((2, 64, 3), -1e9, dtype=torch.bfloat16)
    logits[0, :5, 1] = 3.0
    pooled = pool_token_dim(logits, pooling)
    assert torch.isfinite(pooled).all()
    assert torch.equal(pooled, reference(logits, pooling))


def test_logsumexp_survives_an_all_negative_infinity_column():
    """True -inf is not reachable from `_neg_inf`, but the shift-by-the-max
    trick makes `x - max` a NaN there, so the guard is worth pinning."""
    logits = torch.full((1, 8, 2), -float("inf"))
    pooled = pool_token_dim(logits, "logsumexp")
    assert torch.equal(pooled, reference(logits, "logsumexp"))
    assert bool((pooled == -float("inf")).all())


# --------------------------------------------------------------------------- #
# the reduction is still per document                                          #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("pooling", POOLINGS)
def test_pooling_is_per_document(pooling):
    """Slicing runs along the *token* axis only.

    The document axis and the logits axis survive every step — the running max
    and the running sum are both `[document, logits]` — so a document's pooled
    vector is a function of its own tokens and nothing else. Pooling a batch
    must therefore give, row by row, exactly what pooling each document alone
    gives.
    """
    torch.manual_seed(0)
    # deliberately different scales per document: any cross-document leak in
    # the running max or the running sum would show up immediately
    batch = torch.stack(
        [
            torch.randn(3000, 17, dtype=torch.bfloat16) * scale + offset
            for scale, offset in ((1.0, 0.0), (8.0, -20.0), (0.1, 50.0))
        ]
    )

    together = pool_token_dim(batch, pooling)
    for document in range(batch.shape[0]):
        alone = pool_token_dim(batch[document : document + 1], pooling)
        assert torch.equal(together[document : document + 1], alone)


@pytest.mark.parametrize("pooling", POOLINGS)
def test_pooling_ignores_the_other_documents_entirely(pooling):
    """The complementary direction: changing one document must not move any
    other document's pooled logits."""
    torch.manual_seed(0)
    batch = torch.randn(3, 2500, 11, dtype=torch.bfloat16) * 3
    before = pool_token_dim(batch, pooling)

    batch[1] = batch[1] * 100 + 40  # only document 1 changes
    after = pool_token_dim(batch, pooling)

    assert torch.equal(before[0], after[0])
    assert torch.equal(before[2], after[2])
    assert not torch.equal(before[1], after[1])
