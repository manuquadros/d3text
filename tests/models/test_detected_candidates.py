"""Where a relation candidate comes from: the tagger's spans, the store's links.

`forward` runs the span tagger over the hidden states it already has, cuts its
argmax into typed spans, and grounds each span in the exact mentions the label
store holds for that document. What comes out is a candidate *set* per argument,
which is what these pin: which sets a document proposes, which pairs of them
reach the relation classifier, and that a gold pair only gets a row of its own
where no detected pair already covers it.

The tagger is replaced by a function returning fixed logits, so a test states
the spans it means rather than coaxing a randomly initialised head into
producing them. Everything is CPU and synthetic — no BRENDA data.
"""

import h5py
import numpy
import pytest
import torch
from d3text import token_labels
from d3text.models.config import ModelConfig
from d3text.models.ete import ETEBrendaModel
from d3text.models.model_types import IndexedRelation
from d3text.models.token_supervision import StoredMention, TokenLabelReader
from d3text.schema import BRENDA_SCHEMA
from d3text.token_labels import BRENDA_LABELS, OUTSIDE, LabelSpace

pytestmark = pytest.mark.slow

BACTERIA = BRENDA_LABELS.by_prefix["bac"]
ENZYMES = BRENDA_LABELS.by_prefix["enz"]

TOKENS = 8
HAS_ENZYME = BRENDA_SCHEMA.relation_names.index("HasEnzyme")
NONE = BRENDA_SCHEMA.none_relation_index

# The permuted space of `test_the_grounding_reads_the_readers_own_label_space`:
# the same four types, so a tagger head sized to one fits the other, with code 1
# meaning bacteria instead of strains.
PERMUTED_LABELS = LabelSpace(
    types=("bacteria", "strains", "other_organisms", "enzymes"),
    prefixes=("bac", "str", "oth", "enz"),
)


@pytest.fixture
def ete(patch_base_model, empty_token_label_store):
    """A real ETE model over the BRENDA schema, on CPU.

    The schema has to be BRENDA's: the candidates a span is grounded in are
    filtered by the label space's ID prefixes, and the schema is what has to be
    able to type them back.
    """
    model = ETEBrendaModel(
        schema=BRENDA_SCHEMA,
        config=ModelConfig(
            base_model="prajjwal1/bert-mini",
            hidden_layers=[8],
            ramp_epochs=0,
            token_labels_store=str(empty_token_label_store),
        ),
        device="cpu",
    )
    model.eval()
    return model


def tags(model, *spans):
    """Make the tagger predict `spans`, each `(start, end, code)`."""
    width = 1 + len(BRENDA_LABELS.types)
    logits = torch.full((1, TOKENS, width), -1.0)
    logits[..., OUTSIDE] = 0.0
    for start, end, code in spans:
        logits[0, start:end, code] = 1.0
    object.__setattr__(model, "token_tagger", lambda hidden: logits)


def mention(positions, *entity_ids):
    """One stored exact mention covering `positions`."""
    return StoredMention(
        entity_ids=frozenset(entity_ids),
        positions=torch.tensor(positions, dtype=torch.int64),
    )


def run(model, stored, **kwargs):
    """One forward pass over a single synthetic document."""
    torch.manual_seed(0)
    with torch.no_grad():
        return model(
            torch.randn(1, TOKENS, 256),
            torch.ones(1, TOKENS, dtype=torch.bool),
            stored_mentions={0: stored} if stored else None,
            **kwargs,
        )


def rows_of(meta):
    return list(
        zip(
            meta["sequence"].tolist(),
            meta["arg_pred_i"].tolist(),
            meta["arg_pred_j"].tolist(),
        )
    )


def test_two_grounded_spans_of_admitted_types_become_one_pair(ete):
    """The base case: each span takes its own mention's candidates, and the two
    arguments are paired once, keyed by the ids those sets were interned to."""
    tags(ete, (0, 2, BACTERIA), (4, 6, ENZYMES))

    *_, relations = run(ete, (mention([0, 1], "bac1"), mention([4, 5], "enz1")))

    assert relations is not None
    meta, logits = relations
    assert logits.shape == (1, len(BRENDA_SCHEMA.relation_names))
    assert ete._argument_sets == (frozenset({"bac1"}), frozenset({"enz1"}))
    assert rows_of(meta) == [(0, 0, 1)]


def test_spans_sharing_a_candidate_set_are_one_argument(ete):
    """Two mentions of the same entity are one argument, not two: a pair per
    mention would score the same relation twice and split the tokens of its
    representation between them.

    The two bacteria spans are kept apart by an untagged token, as in
    `test_a_nil_span_proposes_no_argument`: `spans_from_codes` returns maximal
    runs of one code, so written adjacent they would be the one span and no two
    spans would share a set at all.
    """
    tags(ete, (0, 2, BACTERIA), (3, 5, BACTERIA), (6, 8, ENZYMES))
    stored = (
        mention([0, 1], "bac1"),
        mention([3, 4], "bac1"),
        mention([6, 7], "enz1"),
    )

    *_, relations = run(ete, stored)

    assert relations is not None
    meta, _ = relations
    assert ete._argument_sets == (frozenset({"bac1"}), frozenset({"enz1"}))
    assert rows_of(meta) == [(0, 0, 1)]

    # "Not split between them" is a claim about positions, which the sets and
    # the row count cannot see: both spans' tokens have to pool into the one
    # argument's representation.
    arguments = ete._tagged_arguments(
        torch.zeros(1, TOKENS, 8),
        torch.ones(1, TOKENS, dtype=torch.bool),
        {0: stored},
    )
    assert {
        candidates: positions.tolist()
        for candidates, positions in arguments[0].items()
    } == {
        frozenset({"bac1"}): [0, 1, 3, 4],
        frozenset({"enz1"}): [6, 7],
    }


def test_a_nil_span_proposes_no_argument(ete):
    """A typed span the store grounds in nothing is NIL, and relations train on
    grounded arguments only — so it contributes no argument at all rather than
    an empty one every gold pair would then appear to be covered by.

    The three spans are kept apart by an untagged token each: `spans_from_codes`
    merges adjacent runs of one code, so a NIL span written against a grounded
    one of the same type would be the same span.
    """
    tags(ete, (0, 2, BACTERIA), (3, 4, ENZYMES), (5, 7, ENZYMES))

    *_, relations = run(ete, (mention([0, 1], "bac1"), mention([5, 6], "enz1")))

    assert relations is not None
    meta, _ = relations
    assert ete._argument_sets == (frozenset({"bac1"}), frozenset({"enz1"}))
    assert rows_of(meta) == [(0, 0, 1)]


def test_a_type_inadmissible_pair_is_not_proposed(ete):
    """No relation type pairs two enzymes, so the schema alone fixes the label
    and the pair must not reach the relation classifier."""
    tags(ete, (0, 2, ENZYMES), (4, 6, ENZYMES))

    *_, relations = run(ete, (mention([0, 1], "enz1"), mention([4, 5], "enz2")))

    assert relations is None


def test_one_argument_alone_proposes_nothing(ete):
    tags(ete, (0, 2, BACTERIA))

    *_, relations = run(ete, (mention([0, 1], "bac1"),))

    assert relations is None


def test_a_detected_argument_holds_only_ids_the_store_grounds_it_in(ete):
    """The gold-leak guard.

    Gold positions are available on this pass and name entities the store
    mentions nowhere. A detected argument reading them — the gold-only masks
    `entity_positions` is built from, say — would propose gold entities and
    nothing else, which is the capability this proposer exists to replace.
    """
    tags(ete, (0, 2, BACTERIA), (4, 6, ENZYMES))

    *_, relations = run(
        ete,
        (mention([0, 1], "bac1"), mention([4, 5], "enz1")),
        gold_entity_positions={
            0: {"bac9": torch.tensor([0, 1]), "enz9": torch.tensor([4, 5])}
        },
    )

    assert relations is not None
    assert ete._argument_sets == (frozenset({"bac1"}), frozenset({"enz1"}))


def test_a_detected_pair_covering_gold_takes_its_label_alone(ete):
    """Detected first, and only once: the detected row is the representation the
    evaluation scores, so it is the one that has to carry the gold label. A gold
    row beside it would leave that representation trained on `none`.
    """
    tags(ete, (0, 2, BACTERIA), (4, 6, ENZYMES))
    gold = [
        IndexedRelation(
            docix=0,
            subject="bac2",
            object="enz1",
            label=torch.tensor(HAS_ENZYME),
        )
    ]

    *_, relations = run(
        ete,
        (mention([0, 1], "bac1", "bac2"), mention([4, 5], "enz1")),
        gold_relations=gold,
        gold_entity_positions={
            0: {"bac2": torch.tensor([0, 1]), "enz1": torch.tensor([4, 5])}
        },
    )

    assert relations is not None
    meta, logits = relations
    # The ambiguous detected set, not the gold singleton: one row, and its
    # subject is still every entity that mention could name.
    assert ete._argument_sets == (
        frozenset({"bac1", "bac2"}),
        frozenset({"enz1"}),
    )
    assert rows_of(meta) == [(0, 0, 1)]

    _, _, targets = ete.align_relation_predictions(gold, meta, logits)
    assert targets.tolist() == [HAS_ENZYME]


def test_a_gold_pair_no_detection_covers_still_gets_a_row(ete):
    """The fallback: the store places both arguments, but the tagger found only
    one of them, so the pair would otherwise never be trained at all."""
    tags(ete, (4, 6, ENZYMES))
    gold = [
        IndexedRelation(
            docix=0,
            subject="bac2",
            object="enz1",
            label=torch.tensor(HAS_ENZYME),
        )
    ]

    *_, relations = run(
        ete,
        (mention([4, 5], "enz1"),),
        gold_relations=gold,
        gold_entity_positions={
            0: {"bac2": torch.tensor([0, 1]), "enz1": torch.tensor([4, 5])}
        },
    )

    assert relations is not None
    meta, logits = relations
    assert ete._argument_sets == (frozenset({"bac2"}), frozenset({"enz1"}))
    assert rows_of(meta) == [(0, 0, 1)]

    _, _, targets = ete.align_relation_predictions(gold, meta, logits)
    assert targets.tolist() == [HAS_ENZYME]


def test_a_gold_pair_the_store_places_nowhere_gets_no_row(ete):
    """No anchor, no representation: an argument the store mentions nowhere has
    no positions to pool it from, and the bookkeeping counts it as such rather
    than charging it to detection."""
    tags(ete, (4, 6, ENZYMES))
    gold = [
        IndexedRelation(
            docix=0,
            subject="bac2",
            object="enz1",
            label=torch.tensor(HAS_ENZYME),
        )
    ]
    anchored = {0: {"enz1": torch.tensor([4, 5])}}

    *_, relations = run(
        ete,
        (mention([4, 5], "enz1"),),
        gold_relations=gold,
        gold_entity_positions=anchored,
    )

    assert relations is None
    assert ete.unscored_gold_relations(gold, None, anchored) == (
        [],
        [HAS_ENZYME],
    )


def test_the_grounding_reads_the_readers_own_label_space(ete, tmp_path):
    """The codes mean what the *store* says they mean.

    Under BRENDA's own space code 1 is a strain; under this store's it is a
    bacterium. Grounding through the default space rather than the reader's
    would keep the candidates of the wrong type's prefix, so the span would
    ground in another entity entirely and still look resolved.
    """
    path = tmp_path / "permuted.hdf5"
    with h5py.File(path, "w") as store:
        token_labels.write_label_space(
            store,
            PERMUTED_LABELS,
            stamp=token_labels.IndexStamp(digest="permuted"),
        )
    ete._token_labels = TokenLabelReader(path, space=PERMUTED_LABELS)

    tags(ete, (0, 2, 1), (4, 6, ENZYMES))

    *_, relations = run(
        ete, (mention([0, 1], "bac1", "str1"), mention([4, 5], "enz1"))
    )

    assert relations is not None
    assert ete._argument_sets == (frozenset({"bac1"}), frozenset({"enz1"}))


def test_a_document_the_store_lacks_contributes_no_mentions(
    patch_base_model, tmp_path
):
    """A store covering fewer documents than the split is a data gap, and the
    gap has to stay visible: such a document grounds no span, so it proposes no
    candidate, and its gold relations have no anchor either."""
    path = tmp_path / "labels.hdf5"
    with h5py.File(path, "w") as store:
        token_labels.write_label_space(
            store,
            BRENDA_LABELS,
            stamp=token_labels.IndexStamp(digest="one-document"),
        )
        token_labels.store_token_labels(
            store,
            "11",
            token_labels.DocumentLabels(
                codes=numpy.zeros((1, TOKENS + 2), dtype=numpy.int8),
                spans=numpy.zeros((1, token_labels.SPAN_COLUMNS), numpy.int32),
                text_length=0,
                candidate_ids=(frozenset({"bac1"}),),
                anchors=numpy.array([[0, 0, 1, 3]], dtype=numpy.int32),
            ),
        )
    model = ETEBrendaModel(
        schema=BRENDA_SCHEMA,
        config=ModelConfig(
            base_model="prajjwal1/bert-mini",
            hidden_layers=[8],
            ramp_epochs=0,
            token_labels_store=str(path),
        ),
        device="cpu",
    )
    batch = [
        {
            "id": torch.tensor(pmid),
            "sequence": {
                "attention_mask": torch.ones(1, TOKENS + 2, dtype=torch.int64)
            },
        }
        for pmid in (11, 12)
    ]

    mentions = model._stored_mentions(batch)

    assert set(mentions) == {0}
    assert mentions[0][0].entity_ids == frozenset({"bac1"})


def test_strict_targets_ignore_a_row_whose_argument_is_ambiguous(stub):
    """The strict score's rule, and what its gap to the intersection score is
    made of: the row covers the gold pair, so it keeps the typed target it is
    trained on, while strict reads it as `none` and counts the gold unscored."""
    model = stub(
        ETEBrendaModel,
        entity_logits_pooling="logsumexp",
        relations_none_index=NONE,
        _argument_sets=(frozenset({"bac1", "bac2"}), frozenset({"enz1"})),
        _argument_groups={
            "bac1": frozenset({0}),
            "bac2": frozenset({0}),
            "enz1": frozenset({1}),
        },
    )
    gold = [
        IndexedRelation(
            docix=0,
            subject="bac2",
            object="enz1",
            label=torch.tensor(HAS_ENZYME),
        )
    ]
    scored = {
        "sequence": torch.tensor([0]),
        "arg_pred_i": torch.tensor([0]),
        "arg_pred_j": torch.tensor([1]),
    }

    _, _, targets = model.align_relation_predictions(
        gold, scored, torch.randn(1, len(BRENDA_SCHEMA.relation_names))
    )
    # `_strict_relation_targets` takes the pooled meta as host-side
    # (sequence, arg_pred_i, arg_pred_j) triples, not the device tensors.
    strict, missed = model._strict_relation_targets(gold, [(0, 0, 1)])

    assert targets.tolist() == [HAS_ENZYME]
    assert strict.tolist() == [NONE]
    assert missed == [HAS_ENZYME]


def test_gold_covers_every_row_its_ambiguous_argument_reaches(stub):
    """A gold pair keys every row that could be it, not one of them.

    `bac2` is a candidate of two arguments — narrowing is document-global, so
    two ambiguous mentions sharing an entity both stay whole — and either
    argument paired with `enz1` could be the gold pair. A covering row the
    expansion failed to name would be trained toward `none` while the typed
    prediction on it still counted as a false positive.
    """
    model = stub(
        ETEBrendaModel,
        entity_logits_pooling="logsumexp",
        relations_none_index=NONE,
        _argument_sets=(
            frozenset({"bac1", "bac2"}),
            frozenset({"bac2", "bac3"}),
            frozenset({"enz1", "enz2"}),
        ),
        _argument_groups={
            "bac1": frozenset({0}),
            "bac2": frozenset({0, 1}),
            "bac3": frozenset({1}),
            "enz1": frozenset({2}),
            "enz2": frozenset({2}),
        },
    )
    gold = [
        IndexedRelation(
            docix=0,
            subject="bac2",
            object="enz1",
            label=torch.tensor(HAS_ENZYME),
        )
    ]
    scored = {
        "sequence": torch.tensor([0, 0]),
        "arg_pred_i": torch.tensor([0, 1]),
        "arg_pred_j": torch.tensor([2, 2]),
    }

    meta, _, targets = model.align_relation_predictions(
        gold, scored, torch.randn(2, len(BRENDA_SCHEMA.relation_names))
    )

    assert rows_of(meta) == [(0, 0, 2), (0, 1, 2)]
    assert targets.tolist() == [HAS_ENZYME, HAS_ENZYME]
