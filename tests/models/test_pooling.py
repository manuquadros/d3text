"""`pool_token_dim`: the memory-lean pooling of the token dimension.

`_pool_logits` used to open with `logits.float()` — a float32 copy of the
largest tensor in a training step, held by autograd until backward had run.
These pin the two things the replacement must get right: it must not save a
float32 copy, and it must pool to the same values.
"""

import math

import pytest
import torch

from d3text.models.base import pool_chunk_tokens, pool_token_dim

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

    The float32 arithmetic is reordered — summed slice by slice — but the
    difference is far below a bfloat16 ulp and does not survive the cast back.
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

    The running max and the running sum are both `[document, logits]`, so
    pooling a batch must give, row by row, exactly what pooling each document
    alone gives.
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


# --------------------------------------------------------------------------- #
# padding must be invisible: the masked normalisers                            #
# --------------------------------------------------------------------------- #
def masked_reference(logits, mask, pooling):
    """Pool each document's real tokens alone — the oracle a mask must match."""
    rows = []
    for document in range(logits.shape[0]):
        real = logits[document][mask[document]].unsqueeze(0)
        rows.append(reference(real, pooling)[0])
    return torch.stack(rows)


def _padded_batch():
    """Two documents of very different lengths, filled the way `forward`
    fills: real logits where the mask is set, -1e9 elsewhere."""
    torch.manual_seed(0)
    logits = torch.full((2, 1200, 7), -1e9)
    mask = torch.zeros(2, 1200, dtype=torch.bool)
    mask[0, :150] = True
    mask[1, :1200] = True
    logits[mask] = torch.randn(150 + 1200, 7) * 4
    return logits, mask


@pytest.mark.parametrize("pooling", POOLINGS)
def test_masked_pooling_matches_pooling_each_document_alone(pooling):
    """A document's pooled logits are a function of the document, not of how
    long its batch companions were. Without the mask, `logmeanexp` normalises
    by the padded length — shifting the short document by -log(1200/150) on
    every column — and `mean` sums the -1e9 fills into its numerator."""
    logits, mask = _padded_batch()

    pooled = pool_token_dim(logits, pooling, mask=mask)

    assert torch.allclose(
        pooled, masked_reference(logits, mask, pooling), atol=1e-5
    )


@pytest.mark.parametrize("pooling", ("logmeanexp", "mean"))
def test_the_padded_normaliser_was_the_bug(pooling):
    """The guard that the oracle above can go red: without the mask, the
    padded document's pooled logits do NOT match pooling it alone."""
    logits, mask = _padded_batch()

    unmasked = pool_token_dim(logits, pooling)

    assert not torch.allclose(
        unmasked[0], masked_reference(logits, mask, pooling)[0], atol=1e-2
    )


def test_masked_mean_sends_no_gradient_to_padding():
    logits, mask = _padded_batch()
    logits.requires_grad_(True)

    pool_token_dim(logits, "mean", mask=mask).sum().backward()

    assert logits.grad is not None
    assert torch.all(logits.grad[~mask] == 0)
    assert torch.all(logits.grad[mask] != 0)


def _ragged_batch():
    """One padded batch whose documents have three *different* real lengths.

    The lengths have to differ: dividing by the padded length and dividing by
    the real count are the same arithmetic on a row that fills the batch.
    """
    torch.manual_seed(0)
    logits = torch.full((3, 900, 5), -1e9)
    mask = torch.zeros(3, 900, dtype=torch.bool)
    for document, tokens in enumerate((37, 400, 900)):
        mask[document, :tokens] = True
    logits[mask] = torch.randn(int(mask.sum()), 5) * 4
    return logits, mask


def differentiable_masked_mean(logits, mask):
    """The masked mean written so autograd differentiates it: the fills
    zeroed out of the numerator, each row over its own real token count."""
    weights = mask.unsqueeze(-1).to(logits.dtype)
    return (logits * weights).sum(dim=1) / mask.sum(dim=1, keepdim=True)


def test_masked_mean_scales_its_gradient_by_the_real_token_count():
    """`_ChunkedMean.backward` reads none of its input.

    So the divisor is written out a second time there with nothing forcing the
    two copies to agree — and the padded length leaves the pooled values right
    while every document's gradient is too small by `real / padded`.
    """
    logits, mask = _ragged_batch()
    torch.manual_seed(1)
    upstream = torch.randn(3, 5)

    grads = []
    for pool in (
        lambda x: pool_token_dim(x, "mean", mask=mask),
        lambda x: differentiable_masked_mean(x, mask),
    ):
        candidate = logits.clone().requires_grad_(True)
        pool(candidate).backward(upstream)
        grads.append(candidate.grad)

    assert torch.allclose(grads[0], grads[1])


def test_a_fully_masked_document_pools_to_zero_under_masked_mean():
    """The masked mean alone does not return the fill for an all-padding row.

    Its numerator is the empty sum over the floored count of one, so the row
    pools to 0.0 and sigmoids to 0.5. No document reaching the pooling has zero
    real tokens, so this pins the arithmetic rather than a decision.
    """
    logits = torch.full((2, 64, 3), -1e9)
    mask = torch.zeros(2, 64, dtype=torch.bool)
    mask[0, :5] = True
    logits[0, :5] = 3.0

    assert torch.equal(
        pool_token_dim(logits, "mean", mask=mask)[1], torch.zeros(3)
    )
    for pooling in ("logsumexp", "logmeanexp", "max"):
        assert torch.equal(
            pool_token_dim(logits, pooling, mask=mask)[1],
            torch.full((3,), -1e9),
        )


def test_masked_mean_refuses_a_mask_edited_between_forward_and_backward():
    """The mask is saved as an input, so its version counter is checked.
    Editing it in place while the graph is alive would otherwise send the
    gradient to a different set of tokens than the ones that were summed."""
    logits, mask = _ragged_batch()
    logits.requires_grad_(True)
    pooled = pool_token_dim(logits, "mean", mask=mask)

    mask[0, 500] = True

    with pytest.raises(RuntimeError, match="inplace operation"):
        pooled.sum().backward()


@pytest.mark.parametrize("pooling", POOLINGS)
def test_a_fully_masked_document_stays_finite_under_a_mask(pooling):
    """`token_counts` floors at one so an all-padding row divides by 1 and
    takes log(1), never NaN."""
    logits = torch.full((2, 64, 3), -1e9)
    mask = torch.zeros(2, 64, dtype=torch.bool)
    mask[0, :5] = True
    logits[0, :5] = 3.0

    pooled = pool_token_dim(logits, pooling, mask=mask)

    assert torch.isfinite(pooled).all()


@pytest.mark.parametrize("pooling", POOLINGS)
def test_a_mask_with_no_padding_changes_nothing(pooling):
    """With every token real, the masked and unmasked paths must agree
    exactly — the mask only ever corrects for padding."""
    torch.manual_seed(0)
    logits = torch.randn(3, 500, 11, dtype=torch.bfloat16) * 4
    mask = torch.ones(3, 500, dtype=torch.bool)

    assert torch.equal(
        pool_token_dim(logits, pooling, mask=mask),
        pool_token_dim(logits, pooling),
    )


# --------------------------------------------------------------------------- #
# the slice width adapts to the batch                                          #
# --------------------------------------------------------------------------- #
def test_slice_width_holds_the_element_count_flat():
    """A fixed token width made the slice `[documents, 2048, entities]`, which
    grows with the batch. Budgeting elements keeps every slice the same size."""
    width = 6862
    sizes = {
        documents * pool_chunk_tokens(documents, width) * width
        for documents in (1, 2, 4, 8, 16, 32)
    }
    # integer division leaves a little slack; nothing near the 8x of a fixed
    # token width
    assert max(sizes) / min(sizes) < 1.05


def test_slice_width_narrows_as_the_batch_widens():
    """Non-increasing in the batch, and never over budget. Stated as the
    invariant rather than as a ratio: the width is an integer division, so it
    truncates, and a tolerance tight enough to be meaningful would red on the
    truncation rather than on a real regression."""
    width = 6862
    widths = [
        pool_chunk_tokens(documents, width)
        for documents in (1, 2, 4, 8, 16, 32, 128)
    ]
    assert widths == sorted(widths, reverse=True)
    for documents, chunk in zip((1, 2, 4, 8, 16, 32, 128), widths):
        assert documents * chunk * width <= 14_000_000


def test_slice_width_never_reaches_zero():
    """A batch wide enough to exceed the budget on a single token must still
    advance, or the pooling loop would not terminate."""
    assert pool_chunk_tokens(10**6, 10**6) == 1
    assert pool_chunk_tokens(0, 0) >= 1


@pytest.mark.parametrize("pooling", POOLINGS)
@pytest.mark.parametrize("documents", [1, 3, 8])
def test_pooling_is_unchanged_by_the_adaptive_width(pooling, documents):
    """The width is a memory knob only: whatever it resolves to, the pooled
    logits must equal the whole-tensor float32 reduction."""
    torch.manual_seed(0)
    logits = torch.randn(documents, 3000, 29, dtype=torch.bfloat16) * 4
    assert torch.equal(
        pool_token_dim(logits, pooling), reference(logits, pooling)
    )


# --------------------------------------------------------------------------- #
# a document with no tokens                                                    #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("pooling", POOLINGS)
def test_pooling_refuses_an_empty_token_dimension(pooling):
    """A document trimmed to no tokens must be refused by all four poolings.

    `logsumexp` returned `-inf`, scoring a content-free document as a correct
    negative; `mean` returned `NaN` and poisoned the epoch's loss; `logmeanexp`
    died inside `math.log`; only `max` named the dimension.
    """
    logits = torch.zeros(1, 0, 5, dtype=torch.bfloat16)

    with pytest.raises(ValueError, match="holds no tokens"):
        pool_token_dim(logits, pooling)


@pytest.mark.parametrize("pooling", POOLINGS)
def test_a_batch_with_one_empty_document_is_refused(pooling):
    """The dimension is empty for the whole batch or for none of it — padding
    is what hides the content-free document among real ones — so the guard is
    stated on the tensor rather than per row."""
    logits = torch.randn(4, 0, 3)

    with pytest.raises(ValueError, match="holds no tokens"):
        pool_token_dim(logits, pooling)


def test_the_refusal_names_the_shape():
    """A `NaN` in the loss is only actionable if the error says what was
    empty."""
    with pytest.raises(ValueError, match=r"\(1, 0, 5\)"):
        pool_token_dim(torch.zeros(1, 0, 5), "logsumexp")
