"""Grounding a tagged span in the store's mentions: the overlap join, the type
filter, and a narrowing that never picks.

The type filter is the invariant worth pinning twice over: the candidates of a
stored mention are of any type, so a span that took an overlapping mention's
IDs unfiltered would ground a bacterium in an enzyme. Narrowing is pinned in
both directions because the empty-intersection case is the one that must leave
the set alone rather than empty it.
"""

import h5py
import numpy
import pytest
import torch
from d3text import token_labels
from d3text.mention_metrics import PredictedMention
from d3text.models.token_supervision import (
    StoredMention,
    TokenLabelReader,
    resolve_mentions,
)
from d3text.token_labels import BRENDA_LABELS, DocumentLabels

BACTERIA = BRENDA_LABELS.by_prefix["bac"]
ENZYMES = BRENDA_LABELS.by_prefix["enz"]
STRAINS = BRENDA_LABELS.by_prefix["str"]


def mention(positions: list[int], *entity_ids: str) -> StoredMention:
    """One stored mention covering `positions`, its candidates `entity_ids`."""
    return StoredMention(
        entity_ids=frozenset(entity_ids),
        positions=torch.tensor(positions, dtype=torch.int64),
    )


def test_a_span_takes_every_overlapping_mentions_candidates() -> None:
    """Overlap, not containment: a mention reaching into the span from either
    side counts, one clear of it does not, and the span's own coordinates come
    back untouched."""
    stored = (
        mention([4, 5], "enz1"),
        mention([6, 7], "enz2"),
        mention([9], "enz3"),
    )

    (resolved,) = resolve_mentions(
        [PredictedMention(start=5, end=7, type_code=ENZYMES)], stored
    )

    assert (resolved.start, resolved.end, resolved.type_code) == (
        5,
        7,
        ENZYMES,
    )
    assert resolved.entity_ids == frozenset({"enz1", "enz2"})


def test_a_span_takes_no_candidate_of_another_type() -> None:
    """A shared token grounds each span in its own tagged type, whether the
    other type's ID arrives on a mention of its own or on the same
    cross-type candidate set."""
    stored = (mention([3], "bac7", "enz9"), mention([3], "enz8"))

    (resolved,) = resolve_mentions(
        [PredictedMention(start=3, end=4, type_code=BACTERIA)], stored
    )

    assert resolved.entity_ids == frozenset({"bac7"})


def test_a_span_the_store_grounds_in_nothing_is_nil() -> None:
    """NIL is an answer: a typed span over no mention of its type keeps the
    empty set and stays in the output."""
    stored = (mention([7], "enz1"), mention([0, 1], "bac2"))

    resolved = resolve_mentions(
        [
            PredictedMention(start=0, end=2, type_code=ENZYMES),
            PredictedMention(start=4, end=5, type_code=ENZYMES),
        ],
        stored,
    )

    assert [span.entity_ids for span in resolved] == [
        frozenset(),
        frozenset(),
    ]


def test_an_ambiguous_span_narrows_to_what_the_document_names_alone() -> None:
    """The document names one of the two candidates through a mention of one
    candidate only, so the span keeps that one."""
    stored = (mention([2, 3], "str5", "str6"), mention([20], "str6"))

    (resolved,) = resolve_mentions(
        [PredictedMention(start=2, end=4, type_code=STRAINS)], stored
    )

    assert resolved.entity_ids == frozenset({"str6"})


def test_an_ambiguous_span_stays_whole_when_nothing_narrows_it() -> None:
    """Narrowing never picks. The document's unambiguous strain is neither
    candidate, so an empty intersection leaves the set whole rather than
    emptying it or choosing within it."""
    stored = (mention([2, 3], "str5", "str6"), mention([20], "str8"))

    (resolved,) = resolve_mentions(
        [PredictedMention(start=2, end=4, type_code=STRAINS)], stored
    )

    assert resolved.entity_ids == frozenset({"str5", "str6"})


def test_two_nested_spans_of_different_types_both_resolve() -> None:
    """A species inside a strain designation is a different entity that can
    hold a different relation, so the nested spans emit both rather than the
    longer one winning the shared tokens."""
    stored = (mention([4, 5, 6], "str11"), mention([4, 5], "bac12"))

    resolved = resolve_mentions(
        [
            PredictedMention(start=4, end=7, type_code=STRAINS),
            PredictedMention(start=4, end=6, type_code=BACTERIA),
        ],
        stored,
    )

    assert [span.entity_ids for span in resolved] == [
        frozenset({"str11"}),
        frozenset({"bac12"}),
    ]


def test_a_type_code_the_space_does_not_declare_is_refused() -> None:
    """A code outside the space would silently ground every span of that type
    in nothing, which reads as a tagger that found no entity rather than as a
    head sized to another schema."""
    with pytest.raises(KeyError, match="is not an entity-type code"):
        resolve_mentions(
            [
                PredictedMention(
                    start=0, end=1, type_code=len(BRENDA_LABELS.types) + 1
                )
            ],
            (),
        )


def test_the_stores_own_mentions_resolve_a_tagged_span(tmp_path) -> None:
    """End to end on the axis both sides share: `exact_mentions`' tuple feeds
    straight in, so the store's token positions and the tagger's span
    coordinates need no translation between them."""
    candidate_ids = (
        frozenset({"str11", "str12"}),
        frozenset({"bac12"}),
        frozenset({"str12"}),
    )
    path = tmp_path / "labels.hdf5"
    with h5py.File(path, "w") as store:
        token_labels.write_label_space(
            store, BRENDA_LABELS, stamp=token_labels.IndexStamp(digest="test")
        )
        token_labels.store_token_labels(
            store,
            "77",
            DocumentLabels(
                codes=numpy.zeros((1, 8), dtype=numpy.int8),
                spans=numpy.zeros(
                    (len(candidate_ids), token_labels.SPAN_COLUMNS),
                    dtype=numpy.int32,
                ),
                text_length=0,
                candidate_ids=candidate_ids,
                anchors=numpy.array(
                    [[0, 0, 2, 5], [1, 0, 2, 4], [2, 0, 7, 8]],
                    dtype=numpy.int32,
                ),
            ),
        )
    stored = TokenLabelReader(path).exact_mentions("77", numpy.ones((1, 8)))

    assert stored is not None
    resolved = resolve_mentions(
        [
            PredictedMention(start=2, end=5, type_code=STRAINS),
            PredictedMention(start=2, end=4, type_code=BACTERIA),
        ],
        stored,
    )

    assert [span.entity_ids for span in resolved] == [
        frozenset({"str12"}),
        frozenset({"bac12"}),
    ]
