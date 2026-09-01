"""Abstaining a class head's document-level negative where the text
mentions the type anyway, without BRENDA linking it (DEC-04).

A real ``BrendaClassificationModel`` over the tiny injected BERT
(``patch_base_model``) and a hand-written token-label store — the model's
four classes are built in `BRENDA_SCHEMA`'s own declaration order
(strains, bacteria, other_organisms, enzymes) because that is the order
`token_labels.LabelSpace` assigns its codes 1..4 from, and the mask relies
on the two agreeing.
"""

import h5py
import numpy
import pytest
import torch
from d3text import token_labels
from d3text.models.config import ModelConfig
from d3text.models.entity_linking import BrendaClassificationModel
from d3text.schema import BRENDA_SCHEMA
from d3text.token_labels import BRENDA_LABELS, DocumentLabels

CLASS_NAMES = list(
    BRENDA_LABELS.types
)  # strains, bacteria, other_organisms, enzymes
STRAINS, BACTERIA, OTHER_ORGANISMS, ENZYMES = range(len(CLASS_NAMES))


def write_store(path, spans_by_document):
    with h5py.File(path, "w") as store:
        token_labels.write_label_space(
            store,
            BRENDA_LABELS,
            stamp=token_labels.IndexStamp(digest="test-index"),
        )
        for pubmed_id, spans in spans_by_document.items():
            token_labels.store_token_labels(
                store,
                pubmed_id,
                DocumentLabels(
                    codes=numpy.zeros((0,), dtype=numpy.int8),
                    spans=numpy.asarray(spans, dtype=numpy.int32).reshape(
                        -1, token_labels.SPAN_COLUMNS
                    ),
                    text_length=0,
                ),
            )
    return path


def build_model(
    patch_base_model, store, abstain=True, min_chars=0, min_chars_by_class=None
):
    return BrendaClassificationModel(
        schema=BRENDA_SCHEMA,
        class_matrix=torch.zeros(1, len(CLASS_NAMES)),
        entity_index={"str1": 0},
        config=ModelConfig(
            base_model="prajjwal1/bert-mini",
            hidden_layers=[8],
            token_labels_store=str(store),
            class_negative_abstention=abstain,
            class_negative_abstention_min_chars=min_chars,
            class_negative_abstention_min_chars_by_class=min_chars_by_class
            or {},
            # Isolated from the consistency term, which reads the class
            # logits too and would confound the "abstained logit does not
            # move the loss" assertion below.
            consistency_weight=0.0,
        ),
        device="cpu",
    )


def batch_of(pubmed_ids):
    return [{"id": torch.tensor(pmid)} for pmid in pubmed_ids]


# --------------------------------------------------------------------------- #
# ModelConfig                                                                  #
# --------------------------------------------------------------------------- #
def test_abstention_requires_a_label_store() -> None:
    with pytest.raises(ValueError, match="token_labels_store"):
        ModelConfig(class_negative_abstention=True)


def test_abstention_with_a_store_is_accepted() -> None:
    ModelConfig(
        token_labels_store="/some/store.hdf5", class_negative_abstention=True
    )


# --------------------------------------------------------------------------- #
# The mask                                                                     #
# --------------------------------------------------------------------------- #
def test_a_document_negative_mentioning_the_type_is_abstained(
    patch_base_model, tmp_path
) -> None:
    """The false negative DEC-04 measures: no gold bacterium, but the text
    matched a bacterium's surface form (gold-linked or not)."""
    store = write_store(
        tmp_path / "labels.hdf5",
        {"11": [(0, 20, BACTERIA + 1, 0)]},
    )
    model = build_model(patch_base_model, store)
    class_true = torch.zeros(1, len(CLASS_NAMES))  # every class negative

    mask = model.class_negative_abstain_mask(batch_of([11]), class_true)

    assert mask is not None
    expected = torch.zeros_like(mask)
    expected[0, BACTERIA] = True
    assert torch.equal(mask, expected)


def test_a_document_negative_with_no_mention_is_not_abstained(
    patch_base_model, tmp_path
) -> None:
    store = write_store(tmp_path / "labels.hdf5", {"11": []})
    model = build_model(patch_base_model, store)
    class_true = torch.zeros(1, len(CLASS_NAMES))

    mask = model.class_negative_abstain_mask(batch_of([11]), class_true)

    assert mask is not None
    assert not bool(mask.any())


def test_a_gold_positive_is_never_abstained(patch_base_model, tmp_path) -> None:
    """Abstention only ever removes a negative assertion; a real positive
    target must still be trained on even where the dictionary also matched
    it — abstaining that would throw away the one signal DEC-04 is not
    disputing."""
    store = write_store(
        tmp_path / "labels.hdf5",
        {"11": [(0, 20, BACTERIA + 1, 1)]},
    )
    model = build_model(patch_base_model, store)
    class_true = torch.zeros(1, len(CLASS_NAMES))
    class_true[0, BACTERIA] = 1

    mask = model.class_negative_abstain_mask(batch_of([11]), class_true)

    assert mask is not None
    assert not bool(mask.any())


def test_a_document_the_store_lacks_is_not_abstained(
    patch_base_model, tmp_path
) -> None:
    store = write_store(tmp_path / "labels.hdf5", {})
    model = build_model(patch_base_model, store)
    class_true = torch.zeros(1, len(CLASS_NAMES))

    mask = model.class_negative_abstain_mask(batch_of([404]), class_true)

    assert mask is not None
    assert not bool(mask.any())


def test_the_mask_is_none_without_the_config_flag(
    patch_base_model, tmp_path
) -> None:
    store = write_store(
        tmp_path / "labels.hdf5", {"11": [(0, 20, BACTERIA + 1, 0)]}
    )
    model = build_model(patch_base_model, store, abstain=False)
    class_true = torch.zeros(1, len(CLASS_NAMES))

    assert model.class_negative_abstain_mask(batch_of([11]), class_true) is None


def test_a_short_mention_does_not_abstain_the_negative(
    patch_base_model, tmp_path
) -> None:
    """A dictionary match shorter than `class_negative_abstention_min_chars`
    must not, on its own, remove the negative supervision for that class —
    otherwise every incidental one- or two-character match abstains the
    negative it should still assert, which is what collapsed the class head
    toward predicting positive on nearly every document."""
    store = write_store(
        tmp_path / "labels.hdf5",
        {"11": [(0, 3, BACTERIA + 1, 0)]},  # a 3-character match
    )
    model = build_model(patch_base_model, store, min_chars=8)
    class_true = torch.zeros(1, len(CLASS_NAMES))

    mask = model.class_negative_abstain_mask(batch_of([11]), class_true)

    assert mask is not None
    assert not bool(mask.any())


def test_a_long_enough_mention_still_abstains_the_negative(
    patch_base_model, tmp_path
) -> None:
    """The gate excludes short matches, not every match: one at or above the
    configured length still abstains, as an ungated match always did."""
    store = write_store(
        tmp_path / "labels.hdf5",
        {"11": [(0, 8, BACTERIA + 1, 0)]},  # exactly the cutoff
    )
    model = build_model(patch_base_model, store, min_chars=8)
    class_true = torch.zeros(1, len(CLASS_NAMES))

    mask = model.class_negative_abstain_mask(batch_of([11]), class_true)

    assert mask is not None
    expected = torch.zeros_like(mask)
    expected[0, BACTERIA] = True
    assert torch.equal(mask, expected)


def test_the_cutoff_is_overridable_per_class(
    patch_base_model, tmp_path
) -> None:
    """BUG-92: a uniform cutoff cannot serve `bacteria` and `strains` at
    once — `class_negative_abstention_min_chars_by_class` raises one class's
    cutoff without moving the class-wide default the other still uses."""
    store = write_store(
        tmp_path / "labels.hdf5",
        {
            "11": [
                (0, 10, BACTERIA + 1, 0),  # 10 chars
                (20, 30, STRAINS + 1, 0),  # 10 chars
            ]
        },
    )
    model = build_model(
        patch_base_model,
        store,
        min_chars=8,  # the class-wide default: both spans would pass it
        min_chars_by_class={"bacteria": 20},  # bacteria alone needs more
    )
    class_true = torch.zeros(1, len(CLASS_NAMES))

    mask = model.class_negative_abstain_mask(batch_of([11]), class_true)

    assert mask is not None
    expected = torch.zeros_like(mask)
    expected[0, STRAINS] = True  # 10 >= the unmodified 8-char default
    assert torch.equal(mask, expected)  # bacteria: 10 < its own 20-char cutoff


def test_a_class_not_overridden_keeps_the_default_cutoff(
    patch_base_model, tmp_path
) -> None:
    store = write_store(
        tmp_path / "labels.hdf5", {"11": [(0, 10, STRAINS + 1, 0)]}
    )
    model = build_model(
        patch_base_model,
        store,
        min_chars=8,
        min_chars_by_class={"bacteria": 20},  # does not name strains
    )
    class_true = torch.zeros(1, len(CLASS_NAMES))

    mask = model.class_negative_abstain_mask(batch_of([11]), class_true)

    assert mask is not None
    expected = torch.zeros_like(mask)
    expected[0, STRAINS] = True
    assert torch.equal(mask, expected)


# --------------------------------------------------------------------------- #
# The loss reads the mask                                                     #
# --------------------------------------------------------------------------- #
def test_abstained_class_loss_does_not_move_with_the_abstained_logit(
    patch_base_model, tmp_path
) -> None:
    """The property the whole mechanism is for: once a (document, class) pair
    is abstained, that class's prediction stops affecting the class loss."""
    store = write_store(
        tmp_path / "labels.hdf5", {"11": [(0, 20, BACTERIA + 1, 0)]}
    )
    model = build_model(patch_base_model, store)
    class_true = torch.zeros(1, len(CLASS_NAMES))
    abstain = model.class_negative_abstain_mask(batch_of([11]), class_true)
    assert abstain is not None and bool(abstain[0, BACTERIA])

    entity_true = torch.zeros(1, 1)  # width num_of_entities - 1 (no UNK)
    entity_logits = torch.zeros(1, model.num_of_entities)

    def class_loss_at(bacteria_logit: float) -> float:
        # Width num_of_classes (OOS included); compute_entity_loss drops it.
        class_logits = torch.zeros(1, model.num_of_classes)
        class_logits[0, model.class_columns[BACTERIA]] = bacteria_logit
        _, loss = model.compute_entity_loss(
            predictions=(entity_logits, class_logits),
            targets=(entity_true, class_true),
            class_abstain=abstain,
        )
        return loss.item()

    assert class_loss_at(0.0) == pytest.approx(class_loss_at(50.0))
