"""`TokenBudgetBatchSampler`: batches bounded by padded chunk count.

Peak VRAM is linear in a batch's *padded* token count, and a batch pads to its
longest document. Batching by a fixed document count therefore leaves peak
memory a lottery over which documents the sampler drew — the reason a run
trains for a while and then dies. These tests pin the bound, and pin that
bounding it costs no data.
"""

import pytest
from torch.utils.data import SequentialSampler

from d3text.data.data import TokenBudgetBatchSampler, get_batch_loader

# index -> chunk count; deliberately spans a 30x range, as the corpus does
LENGTHS = {0: 2, 1: 30, 2: 1, 3: 3, 4: 1, 5: 12, 6: 1, 7: 1, 8: 4, 9: 2}


def padded_cost(batch):
    """What the batch will actually allocate: every document padded to the
    longest one in it."""
    return len(batch) * max(LENGTHS[i] for i in batch)


def sampler(budget, order=None):
    return TokenBudgetBatchSampler(
        sampler=iter(order if order is not None else sorted(LENGTHS)),
        lengths=LENGTHS,
        budget=budget,
    )


@pytest.mark.parametrize("budget", [1, 2, 4, 8, 16, 32, 64, 1000])
def test_no_batch_exceeds_the_budget(budget):
    """The whole point. The single exception is a document longer than the
    budget on its own, which no batching can bound."""
    for batch in sampler(budget):
        assert padded_cost(batch) <= budget or len(batch) == 1


@pytest.mark.parametrize("budget", [1, 4, 16, 1000])
def test_every_document_is_drawn_exactly_once(budget):
    """Bounding memory must not silently drop or duplicate training data."""
    drawn = [index for batch in sampler(budget) for index in batch]
    assert sorted(drawn) == sorted(LENGTHS)


def test_an_over_budget_document_is_yielded_alone():
    """Not dropped and not truncated: index 1 is 30 chunks against a budget of
    8, and it still has to train."""
    batches = list(sampler(8))
    assert [1] in batches


def test_batch_size_varies_with_document_length():
    """The behaviour a fixed document count cannot have: short documents ride
    together, a long one travels nearly alone."""
    sizes = [len(batch) for batch in sampler(16)]
    assert min(sizes) == 1
    assert max(sizes) > 1


def test_a_long_document_does_not_drag_short_ones_into_its_padding():
    """Index 1 (30 chunks) must not be batched with the 1-chunk documents: at
    a budget of 32 that pair would allocate 2 * 30 = 60."""
    for batch in sampler(32):
        if 1 in batch:
            assert batch == [1]


def test_budget_must_be_positive():
    with pytest.raises(ValueError):
        TokenBudgetBatchSampler(sampler=iter([]), lengths=LENGTHS, budget=0)


def test_no_length_is_reported():
    """Documented absence: the batch count depends on the draw order, so the
    sampler declines to guess rather than report a wrong one."""
    with pytest.raises(TypeError):
        len(sampler(16))


# --------------------------------------------------------------------------- #
# through the real loader                                                      #
# --------------------------------------------------------------------------- #
def test_loader_without_max_chunks_keeps_the_fixed_document_count(tiny_brenda):
    """Both-ways guard: the default path is unchanged."""
    loader = get_batch_loader(dataset=tiny_brenda.present, batch_size=2)
    sizes = [len(batch) for batch in loader]
    assert sorted(sizes) == [1, 2]  # three documents, two per batch


def test_loader_with_max_chunks_bounds_the_padded_batch(tiny_brenda):
    """Chunk counts are [2, 5, 1]; a budget of 4 admits {0, 2} (2*2=4) but
    never lets the 5-chunk document share a batch."""
    lengths = tiny_brenda.present.sequence_lengths
    loader = get_batch_loader(
        dataset=tiny_brenda.present,
        batch_size=99,
        sampler=SequentialSampler(tiny_brenda.present),
        max_chunks=4,
    )
    seen = []
    for batch in loader:
        indices = [lengths_index(lengths, doc) for doc in batch]
        seen.extend(indices)
        assert (
            len(batch) * max(lengths[i] for i in indices) <= 4
            or len(batch) == 1
        )
    assert sorted(seen) == [0, 1, 2]


def lengths_index(lengths, doc):
    """Recover a document's row index from the chunk count the loader returned
    (the fixture's three documents have distinct chunk counts)."""
    n_chunks = doc["doc_id"].shape[-1]
    matches = [i for i, n in lengths.items() if n == n_chunks]
    assert len(matches) == 1
    return matches[0]


def test_the_zero_sentinel_keeps_the_fixed_document_count(tiny_brenda):
    """`ModelConfig.batch_max_chunks` carries "off" as 0, because TOML has no
    null, so the loader must read 0 the same way it reads None."""
    loader = get_batch_loader(
        dataset=tiny_brenda.present, batch_size=2, max_chunks=0
    )
    assert sorted(len(batch) for batch in loader) == [1, 2]


# --------------------------------------------------------------------------- #
# a split frame the encodings file does not cover                              #
# --------------------------------------------------------------------------- #
def test_an_index_without_a_length_is_skipped():
    """A pmid in the split frame but absent from the encodings HDF5 has no
    entry in `sequence_lengths`. Looking it up used to raise `KeyError` before
    the first batch was yielded, killing the run on a stale artifact."""
    batches = list(
        TokenBudgetBatchSampler(
            sampler=iter([0, 99, 2]), lengths=LENGTHS, budget=16
        )
    )
    assert [index for batch in batches for index in batch] == [0, 2]


def test_skipping_costs_no_other_document_its_place():
    """The skip must not close the batch it lands in, nor spend budget on a
    document that is never yielded."""
    with_gap = list(
        TokenBudgetBatchSampler(
            sampler=iter([2, 99, 4]), lengths=LENGTHS, budget=16
        )
    )
    assert with_gap == [[2, 4]]


def test_loader_with_max_chunks_survives_a_pmid_missing_from_the_hdf5(
    tiny_brenda,
):
    """The full path: `evaluate` passes `batch_max_chunks`, so the best known
    config batches this way, and one uncovered pmid used to end the run."""
    loader = get_batch_loader(
        dataset=tiny_brenda.full,
        batch_size=99,
        sampler=SequentialSampler(tiny_brenda.full),
        max_chunks=4,
    )
    batches = list(loader)
    assert sum(len(batch) for batch in batches) == 3
    assert all(batch for batch in batches)
