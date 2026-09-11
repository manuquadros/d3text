"""The store, and the meaning it has to carry with it."""

import ast
import collections
import dataclasses
import importlib.util
import inspect
import itertools
import re
import subprocess
import sys

import h5py
import numpy
import pytest
import torch
from beartype.roar import BeartypeCallHintParamViolation
from conftest import _ENZYME, _FORMS, _STAMP, _empty_labels, _encode
from d3text import surface_forms, token_labels
from d3text.constraints import NonNegative
from d3text.models.token_supervision import TokenLabelReader
from d3text.schema import BRENDA_SCHEMA
from d3text.utils import aggregate_embeddings


def test_the_label_store_round_trips(tmp_path, index) -> None:
    """The targets live beside the encodings, keyed by pubmed id."""
    text = "catalase and cholesterol oxidase"
    encoding = _encode(text)
    labels = token_labels.document_token_labels(
        text, index, {"enz2"}, encoding["offset_mapping"]
    )
    path = tmp_path / "labels.hdf5"

    with h5py.File(path, "w-", libver="latest") as store:
        token_labels.write_label_space(store, stamp=_STAMP)
        token_labels.store_token_labels(store, "10822008", labels)

    with h5py.File(path, "r") as store:
        stored = token_labels.load_token_labels(store, "10822008")
        assert numpy.array_equal(stored.codes, labels.codes)
        assert numpy.array_equal(stored.spans, labels.spans)
        assert stored.text_length == len(text)
        with pytest.raises(KeyError):
            token_labels.load_token_labels(store, "99999999")


_CANDIDATE_FORMS = {
    "enz1": ["AS-A", "cholesterol oxidase"],
    "enz5": ["AS-A"],
    "enz2": ["catalase"],
    "bac3": ["Streptomyces griseocarneus"],
}

_CANDIDATE_TEXT = (
    "AS-A and catalase and catalases from Streptomyces griseocarneus, "
    "then cholesterol oxidase and AS-A again with catalase"
)


def test_every_exact_mention_is_stored_with_its_whole_candidate_set(
    tmp_path,
) -> None:
    """Gold or not, ambiguous or not: a detected span is to be linked through
    what the store holds, so storing only gold mentions' IDs would link
    nothing but gold. A fuzzy mention names no entity to link to and stores
    none, and the gold masks stay gold-only."""
    index = surface_forms.build_index(_CANDIDATE_FORMS)
    text = _CANDIDATE_TEXT[: _CANDIDATE_TEXT.index(",")]
    labels = token_labels.document_token_labels(
        text, index, {"bac3"}, _encode(text)["offset_mapping"]
    )
    path = tmp_path / "labels.hdf5"

    with h5py.File(path, "w-", libver="latest") as store:
        token_labels.write_label_space(store, stamp=_STAMP)
        token_labels.store_token_labels(store, "10822008", labels)

    with h5py.File(path, "r") as store:
        stored = token_labels.load_token_labels(store, "10822008")

    assert [
        mention.fuzzy for mention in token_labels.find_mentions(text, index)
    ] == [False, False, True, False]
    assert stored.candidate_ids == (
        frozenset({"enz1", "enz5"}),
        frozenset({"enz2"}),
        frozenset(),
        frozenset({"bac3"}),
    )
    assert {row for row, *_ in stored.anchors.tolist()} == {0, 1, 3}
    assert set(stored.entity_token_masks) == {"bac3"}


def test_the_anchors_place_each_exact_mention_on_the_aggregated_axis(
    tmp_path,
) -> None:
    """Checked against the offset mapping merged the way the embeddings are,
    over windows narrow enough that mentions straddle overlaps: an anchor that
    followed its own window rather than the merge would place an overlapping
    mention's tokens twice, or not at all."""
    index = surface_forms.build_index(_CANDIDATE_FORMS)
    encoding = _encode(_CANDIDATE_TEXT, max_length=32)
    labels = token_labels.document_token_labels(
        _CANDIDATE_TEXT, index, {"bac3"}, encoding["offset_mapping"]
    )
    path = tmp_path / "labels.hdf5"

    with h5py.File(path, "w-", libver="latest") as store:
        token_labels.write_label_space(store, stamp=_STAMP)
        token_labels.store_token_labels(store, "10822008", labels)

    mentions = TokenLabelReader(path).exact_mentions(
        "10822008", encoding["attention_mask"]
    )
    offsets = aggregate_embeddings(
        torch.as_tensor(encoding["offset_mapping"]),
        torch.as_tensor(encoding["attention_mask"]),
    )
    starts, ends = offsets[:, 0], offsets[:, 1]
    expected = [
        (
            mention.entity_ids,
            torch.nonzero(
                (ends > starts)
                & (starts < mention.end)
                & (ends > mention.start)
            )
            .squeeze(-1)
            .tolist(),
        )
        for mention in token_labels.find_mentions(_CANDIDATE_TEXT, index)
        if not mention.fuzzy
    ]

    windows_per_row = collections.Counter(
        row for row, *_ in labels.anchors.tolist()
    )
    assert max(windows_per_row.values()) > 1, "no mention straddles windows"
    assert mentions is not None
    assert [
        (mention.entity_ids, mention.positions.tolist()) for mention in mentions
    ] == expected


def test_writing_a_document_again_replaces_its_targets(tmp_path) -> None:
    """The second write is the one read back, whole.

    A re-run of the precompute command rewrites documents it already holds,
    so a store that kept the first write, or held a group's old spans beside
    new codes, would train on the run before the fix.
    """
    first = _empty_labels()
    second = token_labels.DocumentLabels(
        codes=numpy.array([0, _ENZYME, _ENZYME, 0, 0], dtype=numpy.int8),
        spans=numpy.array([[2, 10, _ENZYME, 1]], dtype=numpy.int32),
        text_length=12,
        candidate_ids=(frozenset({"enz1"}),),
    )
    path = tmp_path / "labels.hdf5"

    with h5py.File(path, "w-", libver="latest") as store:
        token_labels.write_label_space(store, stamp=_STAMP)
        token_labels.store_token_labels(store, "10822008", first)
    with h5py.File(path, "r+") as store:
        token_labels.store_token_labels(store, "10822008", second)

    with h5py.File(path, "r") as store:
        stored = token_labels.load_token_labels(store, "10822008")

    assert numpy.array_equal(stored.codes, second.codes)
    assert numpy.array_equal(stored.spans, second.spans)
    assert stored.text_length == second.text_length
    assert stored.candidate_ids == second.candidate_ids


def test_the_store_records_what_its_codes_mean(tmp_path) -> None:
    """The artifact has to say which column is which type.

    A store written under one declaration order and read under another scores
    every type against another type's target, silently, since the shapes agree.
    """
    path = tmp_path / "labels.hdf5"

    with h5py.File(path, "w-", libver="latest") as store:
        token_labels.write_label_space(store, stamp=_STAMP)

    with h5py.File(path, "r") as store:
        recorded = token_labels.read_label_space(store)

    assert recorded == token_labels.BRENDA_LABELS
    assert recorded.types == BRENDA_SCHEMA.class_names


def test_a_store_written_under_another_order_reads_back_as_that_order(
    tmp_path,
) -> None:
    """The failure the recording exists to catch, made visible.

    Reversing the declaration keeps every width identical, so the attribute is
    the only thing separating the two stores.
    """
    reversed_space = token_labels.LabelSpace(
        types=token_labels.BRENDA_LABELS.types[::-1],
        prefixes=token_labels.BRENDA_LABELS.prefixes[::-1],
    )
    path = tmp_path / "labels.hdf5"

    with h5py.File(path, "w-", libver="latest") as store:
        token_labels.write_label_space(store, reversed_space, stamp=_STAMP)

    with h5py.File(path, "r") as store:
        recorded = token_labels.read_label_space(store)

    assert recorded == reversed_space
    assert recorded != token_labels.BRENDA_LABELS


def test_reading_a_store_under_another_label_space_is_refused(
    tmp_path,
) -> None:
    """The read side of the recording, which is the side that gets it wrong.

    Recording the order on the way in only helps if the way out compares it,
    and a reader that must remember to call `read_label_space` first is one
    that will one day not.
    """
    permuted = token_labels.LabelSpace(
        types=token_labels.BRENDA_LABELS.types[::-1],
        prefixes=token_labels.BRENDA_LABELS.prefixes[::-1],
    )
    path = tmp_path / "labels.hdf5"

    with h5py.File(path, "w-", libver="latest") as store:
        token_labels.write_label_space(store, permuted, stamp=_STAMP)
        token_labels.store_token_labels(store, "10822008", _empty_labels())

    with h5py.File(path, "r") as store:
        with pytest.raises(ValueError, match="records the label space"):
            token_labels.load_token_labels(store, "10822008")

        under_its_own_space = token_labels.load_token_labels(
            store, "10822008", permuted
        )

    assert numpy.array_equal(
        under_its_own_space.codes, _empty_labels().codes
    ), "a store read under the space it records still reads"


def test_a_store_written_under_another_pairing_rule_is_refused(
    tmp_path, monkeypatch
) -> None:
    """The types, prefixes and codes are the pairing's inputs, and re-pairing
    them in `by_prefix` leaves all three alike, so a read comparing only those
    trains every type against another type's target. Refused both ways: an
    older build can read a newer store as easily as the reverse."""
    space = token_labels.BRENDA_LABELS
    paired = space.by_prefix
    before = tmp_path / "before.hdf5"
    after = tmp_path / "after.hdf5"

    with h5py.File(before, "w-", libver="latest") as store:
        token_labels.write_label_space(store, stamp=_STAMP)

    with monkeypatch.context() as patched:
        patched.setattr(
            token_labels.LabelSpace,
            "by_prefix",
            property(lambda s: dict(zip(s.prefixes, reversed(s.codes)))),
        )
        assert space.by_prefix != paired, "the rule has to really re-pair"
        with h5py.File(after, "w-", libver="latest") as store:
            token_labels.write_label_space(store, stamp=_STAMP)

        with h5py.File(before, "r") as store:
            with pytest.raises(ValueError, match="this build does not use"):
                token_labels.read_label_space(store)

    with h5py.File(after, "r") as store:
        with pytest.raises(ValueError, match="this build does not use"):
            token_labels.read_label_space(store)


def test_targets_cannot_be_written_without_their_label_space(
    tmp_path, index
) -> None:
    """A store of unattributed codes cannot be repaired, only regenerated, so
    it must not be possible to start one."""
    path = tmp_path / "labels.hdf5"

    with h5py.File(path, "w-", libver="latest") as store:
        with pytest.raises(KeyError, match="records no label space"):
            token_labels.store_token_labels(store, "10822008", _empty_labels())


def test_a_store_that_records_no_label_space_is_refused(tmp_path) -> None:
    path = tmp_path / "labels.hdf5"

    with h5py.File(path, "w-", libver="latest") as store:
        pass

    with h5py.File(path, "r") as store:
        with pytest.raises(KeyError, match="records no label space"):
            token_labels.read_label_space(store)


def test_a_store_written_under_another_ignore_index_is_refused(
    tmp_path,
) -> None:
    """`IGNORE_INDEX` is torch's `ignore_index` and the targets are handed to
    the loss unchanged, so a store that spelled it differently would train on
    the tokens this scheme abstains from."""
    path = tmp_path / "labels.hdf5"

    with h5py.File(path, "w-", libver="latest") as store:
        token_labels.write_label_space(store, stamp=_STAMP)
        store.attrs["ignore_index"] = -1

    with h5py.File(path, "r") as store:
        with pytest.raises(ValueError, match="this build does not use"):
            token_labels.read_label_space(store)


def test_a_store_written_before_the_mention_spans_is_refused(
    tmp_path,
) -> None:
    """A format-1 store keys each document to a bare code array.

    It can neither be read as format-2 nor completed without re-running the
    matcher, so it is refused rather than defaulted into.
    """
    path = tmp_path / "labels.hdf5"

    with h5py.File(path, "w-", libver="latest") as store:
        token_labels.write_label_space(store, stamp=_STAMP)
        store.attrs["d3text_token_labels_format"] = 1

    with h5py.File(path, "r") as store:
        with pytest.raises(ValueError, match="format-1 label store"):
            token_labels.read_label_space(store)
        with pytest.raises(ValueError, match="regenerate it"):
            token_labels.load_token_labels(store, "10822008")


def test_a_store_written_before_every_mention_carried_its_ids_is_refused(
    tmp_path,
) -> None:
    """A format-5 store traces gold entities' mentions and no others, so a
    linker reading it would propose gold alone; and nothing in it can recover
    the other mentions' IDs short of re-running the matcher."""
    path = tmp_path / "labels.hdf5"

    with h5py.File(path, "w-", libver="latest") as store:
        token_labels.write_label_space(store, stamp=_STAMP)
        token_labels.store_token_labels(store, "10822008", _empty_labels())
        store.attrs["d3text_token_labels_format"] = 5

    with h5py.File(path, "r") as store:
        with pytest.raises(ValueError, match="format-5 label store"):
            token_labels.read_label_space(store)
        with pytest.raises(ValueError, match="regenerate it"):
            token_labels.load_token_labels(store, "10822008")


def test_an_offset_mapping_of_the_wrong_shape_is_rejected() -> None:
    with pytest.raises(ValueError, match="size-2 axis"):
        token_labels.project_onto_tokens(
            numpy.zeros(4, dtype=numpy.int8), numpy.zeros((2, 3))
        )


def test_a_whole_encoding_is_refused_in_place_of_its_offset_mapping(
    index,
) -> None:
    """Passing `encoding` for `encoding["offset_mapping"]` fails on the type.

    Typed `Any`, the parameter let a `BatchEncoding` through to `asarray`,
    which read it as an array of its key names and raised about shapes.
    """
    text = "catalase"
    encoding = _encode(text)

    with pytest.raises(BeartypeCallHintParamViolation, match="offset_mapping"):
        token_labels.document_token_labels(text, index, {"enz2"}, encoding)
    with pytest.raises(BeartypeCallHintParamViolation, match="offset_mapping"):
        token_labels.project_onto_tokens(
            numpy.zeros(len(text), dtype=numpy.int8), encoding
        )


def test_a_token_entirely_past_the_labelled_text_is_rejected() -> None:
    """A bound outrunning `labels` must not read as a real negative.

    Clipped to the array, the third token's `[12, 15]` would read an empty
    interval and come out `OUTSIDE` — a silent negative rather than a
    rejected offset mapping.
    """
    labels = numpy.zeros(10, dtype=numpy.int8)
    labels[0:3] = 1
    offsets = [[0, 0], [0, 3], [12, 15], [0, 0]]

    with pytest.raises(ValueError, match="10"):
        token_labels.project_onto_tokens(labels, offsets)


def test_an_offset_mapping_from_a_longer_string_is_rejected(index) -> None:
    """`document_token_labels` inherits the check through the projection."""
    text = "catalase"
    longer_encoding = _encode("catalase and cholesterol oxidase")

    with pytest.raises(ValueError, match="past the"):
        token_labels.document_token_labels(
            text, index, {"enz2"}, longer_encoding["offset_mapping"]
        )


def test_a_store_records_the_index_its_targets_were_matched_against(
    tmp_path,
) -> None:
    """The label space says what the codes mean; it does not say which strings
    earned one, and that is a separate thing the artifact has to carry."""
    path = tmp_path / "labels.hdf5"

    with h5py.File(path, "w-", libver="latest") as store:
        token_labels.write_label_space(store, stamp=_STAMP)

    with h5py.File(path, "r") as store:
        recorded = token_labels.read_index_stamp(store)

    assert recorded == _STAMP


def test_the_recorded_digest_is_readable_from_the_path_alone(tmp_path) -> None:
    """`train` and `evaluate` record and compare the store's index without
    ever holding a surface-form index of their own, so the digest has to be
    reachable from the configured path and nothing else."""
    path = tmp_path / "labels.hdf5"

    with h5py.File(path, "w-", libver="latest") as store:
        token_labels.write_label_space(store, stamp=_STAMP)

    assert token_labels.store_index_digest(path) == _STAMP.digest
    # TOML's spelling of "no store", and the default: a model configured
    # without one has no provenance to record rather than a missing file.
    assert token_labels.store_index_digest("") is None


def test_the_recorded_rules_digest_is_readable_from_the_path_alone(
    tmp_path,
) -> None:
    """The checkpoint records this digest beside the index one, and both have
    to be reachable the same way: from the configured path, holding no store
    open."""
    path = tmp_path / "labels.hdf5"

    with h5py.File(path, "w-", libver="latest") as store:
        token_labels.write_label_space(store, stamp=_STAMP)

    expected = token_labels._rules_digest(token_labels.labelling_rules())
    assert token_labels.store_labelling_rules_digest(path) == expected
    assert token_labels.store_labelling_rules_digest("") is None


def test_a_rules_only_change_leaves_the_index_digest_untouched(
    tmp_path, monkeypatch
) -> None:
    """The gap this whole ticket is about: a store rebuilt after a labelling
    rule moves is byte-identical in its index digest to one built before,
    because the rule never touches the index. Only the rules digest catches
    it, which is why `evaluate` has to compare both."""
    before = tmp_path / "before.hdf5"
    with h5py.File(before, "w-", libver="latest") as store:
        token_labels.write_label_space(store, stamp=_STAMP)
    before_index = token_labels.store_index_digest(before)
    before_rules = token_labels.store_labelling_rules_digest(before)

    monkeypatch.setattr(surface_forms, "FUZZY_MIN_LENGTH", 20)
    after = tmp_path / "after.hdf5"
    with h5py.File(after, "w-", libver="latest") as store:
        token_labels.write_label_space(store, stamp=_STAMP)
    after_index = token_labels.store_index_digest(after)
    after_rules = token_labels.store_labelling_rules_digest(after)

    assert before_index == after_index == _STAMP.digest
    assert before_rules != after_rules


def test_a_store_matched_against_another_index_is_refused(tmp_path) -> None:
    """The whole point: which strings name entities is a property of the
    index, and an index is a function of the datasets pooled and of the
    extractors that pooled them. Neither shows up in the types, the prefixes
    or the codes, so appending under a second index leaves one file whose two
    halves label the same string differently.
    """
    elsewhere = token_labels.IndexStamp.from_index(
        surface_forms.build_index({**_FORMS, "oth7": ["Jaculus orientalis"]}),
        sources=("another-split.csv",),
    )
    path = tmp_path / "labels.hdf5"

    with h5py.File(path, "w-", libver="latest") as store:
        token_labels.write_label_space(store, stamp=_STAMP)
        token_labels.store_token_labels(store, "10822008", _empty_labels())

    with h5py.File(path, "r") as store:
        with pytest.raises(ValueError, match="disagree about which strings"):
            token_labels.check_index(store, elsewhere)

        assert token_labels.check_index(store, _STAMP) == _STAMP


def test_the_refusal_names_the_inputs_and_the_command_that_rebuilds_it(
    tmp_path,
) -> None:
    """A refusal that only says two hashes differ leaves the operator nowhere:
    the recorded inputs are what identifies the artifact in hand, and the
    command is what replaces it."""
    elsewhere = token_labels.IndexStamp(
        digest="0" * 64, sources=("another-split.csv",)
    )
    path = tmp_path / "labels.hdf5"

    with h5py.File(path, "w-", libver="latest") as store:
        token_labels.write_label_space(store, stamp=_STAMP)

    with h5py.File(path, "r") as store:
        with pytest.raises(ValueError) as refusal:
            token_labels.check_index(store, elsewhere)

    message = str(refusal.value)
    assert "split.csv" in message
    assert "another-split.csv" in message
    assert "precompute-token-labels" in message


def test_a_store_from_before_the_index_was_recorded_is_refused(
    tmp_path,
) -> None:
    """It loads clean and cannot say what it was matched against, which is the
    defect: a tagger would train on whatever the store happens to hold. The
    refusal has to carry the command that replaces it, since there is nothing
    to migrate.
    """
    path = tmp_path / "labels.hdf5"

    with h5py.File(path, "w-", libver="latest") as store:
        token_labels.write_label_space(store, stamp=_STAMP)
        token_labels.store_token_labels(store, "10822008", _empty_labels())
        del store.attrs["surface_form_index_digest"]
        del store.attrs["surface_form_index_sources"]
        store.attrs["d3text_token_labels_format"] = 2

    with h5py.File(path, "r") as store:
        for refuse in (
            lambda: token_labels.read_label_space(store),
            lambda: token_labels.read_index_stamp(store),
            lambda: token_labels.load_token_labels(store, "10822008"),
        ):
            with pytest.raises(ValueError) as refusal:
                refuse()
            assert "precompute-token-labels" in str(refusal.value)


def test_targets_cannot_be_written_without_the_index_that_placed_them(
    tmp_path,
) -> None:
    """A store already full of targets nothing can attribute is unrepairable,
    so the write path refuses one the way it refuses a missing label space."""
    path = tmp_path / "labels.hdf5"

    with h5py.File(path, "w-", libver="latest") as store:
        token_labels.write_label_space(store, stamp=_STAMP)
        del store.attrs["surface_form_index_digest"]

        with pytest.raises(KeyError, match="records no surface-form index"):
            token_labels.store_token_labels(store, "10822008", _empty_labels())


def test_a_store_records_the_rules_that_placed_its_targets(tmp_path) -> None:
    """The index says which strings name entities; it does not say what the
    sweep did with that answer, and the targets are a function of both."""
    path = tmp_path / "labels.hdf5"

    with h5py.File(path, "w-", libver="latest") as store:
        token_labels.write_label_space(store, stamp=_STAMP)

    with h5py.File(path, "r") as store:
        assert (
            token_labels.read_labelling_rules(store)
            == token_labels.labelling_rules()
        )
        assert (
            token_labels.check_labelling_rules(store)
            == token_labels.labelling_rules()
        )


def test_a_store_placed_by_other_labelling_rules_is_refused(
    tmp_path, monkeypatch
) -> None:
    """The gap this exists for: a rule change the index digest cannot see.

    `MAX_MENTION_GAP`, the longest-match window and the fuzzy fallback each
    decide which spans a byte-identical index yields, so a stamp over the
    index alone accepts a store labelled by code this build no longer runs and
    appends today's spans beside yesterday's.
    """
    path = tmp_path / "labels.hdf5"

    with h5py.File(path, "w-", libver="latest") as store:
        token_labels.write_label_space(store, stamp=_STAMP)
        token_labels.store_token_labels(store, "10822008", _empty_labels())

    monkeypatch.setattr(surface_forms, "FUZZY_MIN_LENGTH", 20)
    rebuilt = surface_forms.build_index(_FORMS)

    assert (
        surface_forms.index_digest(rebuilt) == _STAMP.digest
    ), "the index has to be the half that did not move"
    assert (
        token_labels.find_mentions("catalases are active", rebuilt) == []
    ), "the rule has to be one that really relabels the corpus"

    with h5py.File(path, "r+", libver="latest") as store:
        with pytest.raises(ValueError, match="FUZZY_MIN_LENGTH"):
            token_labels.check_index(store, _STAMP)
        with pytest.raises(ValueError, match="placed by labelling rules"):
            token_labels.store_token_labels(store, "10822009", _empty_labels())


def test_the_fingerprint_covers_the_whole_matching_path() -> None:
    """What the fingerprint reaches, spelled out so that widening it is a
    reviewed change rather than a silent one. A helper added to the sweep and
    not listed here is one the guard cannot see.

    `LabelSpace` and `SurfaceFormIndex` are absent on purpose: the sweep is
    handed them rather than constructing them, and each is covered already.
    """
    assert set(token_labels.labelling_rules()) == {
        "surface_forms.ACCESSION",
        "surface_forms.COMMON_WORD_ZIPF",
        "surface_forms.FUZZY_CANDIDATE_MAX_TERMS",
        "surface_forms.FUZZY_CUTOFF",
        "surface_forms.FUZZY_MIN_LENGTH",
        "surface_forms.PLACEHOLDER_FORMS",
        "surface_forms.SurfaceFormIndex.fuzzy_ids",
        "surface_forms.SurfaceFormIndex.lookup",
        "surface_forms.SurfaceFormIndex.may_start",
        "surface_forms.THOUSANDS",
        "surface_forms.UNIT_SYMBOLS",
        "surface_forms._QUANTITY",
        "surface_forms._WORD",
        "surface_forms._is_placeholder",
        "surface_forms.form_key",
        "surface_forms.has_letter",
        "surface_forms.is_common_word",
        "surface_forms.is_quantity",
        "surface_forms.word_spans",
        "token_labels.ANCHOR_COLUMNS",
        "token_labels.DocumentLabels",
        "token_labels.IGNORE_INDEX",
        "token_labels.MAX_MENTION_GAP",
        "token_labels.Mention",
        "token_labels.OUTSIDE",
        "token_labels.SPAN_COLUMNS",
        "token_labels._ABBREVIATION_DOT",
        "token_labels._EPITHET",
        "token_labels._code_of",
        "token_labels._contiguous_run",
        "token_labels._entity_token_presence",
        "token_labels._is_genus_initial",
        "token_labels._mention_anchors",
        "token_labels._mention_type",
        "token_labels._overlapping_tokens",
        "token_labels.character_labels_from_spans",
        "token_labels.document_token_labels",
        "token_labels.find_mentions",
        "token_labels.gold_entity_mention_spans",
        "token_labels.mention_spans",
        "token_labels.project_onto_tokens",
    }


def test_a_rules_fingerprint_reads_the_code_and_not_the_prose() -> None:
    """A guard that fired on a reflowed docstring would be switched off.

    Every refusal costs a corpus relabel, so the fingerprint has to move on
    what changes a span and stay put on what does not.
    """

    def documented():
        def rule(value: int) -> int:
            """One thing."""
            return value + 1

        return rule

    def rewritten():
        def rule(value: int) -> int:
            """Something else entirely, and at more length.

            With a second paragraph nobody has to reread.
            """
            # And a comment.
            return value + 1

        return rule

    def altered():
        def rule(value: int) -> int:
            """One thing."""
            return value + 2

        return rule

    fingerprint = token_labels._source_fingerprint

    assert fingerprint(documented()) == fingerprint(rewritten())
    assert fingerprint(documented()) != fingerprint(altered())


def test_a_rules_fingerprint_reads_the_code_and_not_the_types() -> None:
    """beartype enforces an annotation, so a narrowed one can refuse a call it
    used to accept, loudly; it cannot relabel one it accepts. Hashing it made
    `max_gap: int` -> `max_gap: NonNegative` cost a corpus relabel."""

    def typed():
        def rule(value: int) -> int:
            return value + 1

        return rule

    def narrowed():
        def rule(value: NonNegative) -> int:
            return value + 1

        return rule

    def untyped():
        def rule(value):
            return value + 1

        return rule

    def altered():
        def rule(value: int) -> int:
            return value + 2

        return rule

    fingerprint = token_labels._source_fingerprint

    assert fingerprint(typed()) == fingerprint(narrowed())
    assert fingerprint(typed()) == fingerprint(untyped())
    assert fingerprint(typed()) != fingerprint(altered())


def test_retyping_a_local_keeps_its_rules_fingerprint() -> None:
    """A function body's annotation is never evaluated, so `x: T = v` binds
    just as `x = v` does, and a bare `x: T` declares `x` whatever `T` is."""

    def typed():
        def rule(value):
            found: list[int] = [value]
            seen: int
            seen = found[0]
            return seen + 1

        return rule

    def retyped():
        def rule(value):
            found: list[NonNegative] = [value]
            seen: "NonNegative | None"
            seen = found[0]
            return seen + 1

        return rule

    def unannotated():
        def rule(value):
            found = [value]
            seen: int
            seen = found[0]
            return seen + 1

        return rule

    fingerprint = token_labels._source_fingerprint

    assert fingerprint(typed()) == fingerprint(retyped())
    assert fingerprint(typed()) == fingerprint(unannotated())


def test_a_bare_local_annotation_still_makes_its_name_local() -> None:
    """A function body's `found: T` with no value binds nothing but makes
    `found` local, so a read of the global becomes an `UnboundLocalError`:
    erasing the annotation must not erase the statement."""

    def declared():
        def rule(value):
            found: list[int]
            return found[0] + value  # noqa: F821

        return rule

    def undeclared():
        def rule(value):
            return found[0] + value  # noqa: F821

        return rule

    fingerprint = token_labels._source_fingerprint

    assert fingerprint(declared()) != fingerprint(undeclared())


def test_retyping_a_methods_parameter_keeps_its_fingerprint() -> None:
    """A method is a function wherever its class stands, so its annotations
    go like any other function's, whether the method, its class or a rule
    that defines the class is what gets fingerprinted."""

    def typed():
        class Span:
            def shifted(self, by: int) -> int:
                moved: int = by + 1
                return moved

        return Span

    def narrowed():
        class Span:
            def shifted(self, by: NonNegative) -> "NonNegative":
                moved: NonNegative = by + 1
                return moved

        return Span

    def altered():
        class Span:
            def shifted(self, by: int) -> int:
                moved: int = by + 2
                return moved

        return Span

    def nesting():
        def rule(by: int) -> int:
            class Span:
                def shifted(self, by: int) -> int:
                    return by + 1

            return Span().shifted(by)

        return rule

    def nesting_narrowed():
        def rule(by: int) -> int:
            class Span:
                def shifted(self, by: NonNegative) -> "NonNegative":
                    return by + 1

            return Span().shifted(by)

        return rule

    fingerprint = token_labels._source_fingerprint
    span, narrowed_span, altered_span = typed(), narrowed(), altered()

    assert fingerprint(span) == fingerprint(narrowed_span)
    assert fingerprint(span.shifted) == fingerprint(narrowed_span.shifted)
    assert fingerprint(nesting()) == fingerprint(nesting_narrowed())
    assert fingerprint(span) != fingerprint(altered_span)
    assert fingerprint(span.shifted) != fingerprint(altered_span.shifted)


_SPAN_MODULE = """\
import dataclasses
from typing import ClassVar


@dataclasses.dataclass
class Span:
    start: int
    fuzzy: {annotation} = False


def rule(start):
    @dataclasses.dataclass
    class Span:
        start: int
        fuzzy: {annotation} = False

    return Span(start)
"""

_PROBES = itertools.count()


def _probe_module(tmp_path, monkeypatch, source, namespace=None):
    """A module run from a file, where `inspect.getsource` can read it back.

    Registered in `sys.modules`, since `inspect` finds a class's file there;
    `namespace` seeds its globals before `source` runs."""
    name = f"_fingerprint_probe_{next(_PROBES)}"
    path = tmp_path / f"{name}.py"
    path.write_text(source)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    vars(module).update(
        (key, value)
        for key, value in (namespace or {}).items()
        if not key.startswith("__")
    )
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("annotation", ["ClassVar[bool]", "int"])
@pytest.mark.parametrize("placement", ["Span", "rule"])
def test_retyping_a_dataclass_field_still_moves_the_fingerprint(
    tmp_path, monkeypatch, annotation, placement
) -> None:
    """In a class body the annotation is what makes a name a field, and
    `fuzzy: ClassVar[bool] = False` is none: `__init__` stops taking it. So
    class annotations are hashed as written, at module level and in a class a
    rule defines, and a type-only retype costing a relabel is the price."""
    field = _probe_module(
        tmp_path, monkeypatch, _SPAN_MODULE.format(annotation="bool")
    )
    changed = _probe_module(
        tmp_path, monkeypatch, _SPAN_MODULE.format(annotation=annotation)
    )
    names = {each.name for each in dataclasses.fields(changed.Span)}

    assert ("fuzzy" in names) == (annotation == "int"), "`ClassVar` unmakes it"
    assert token_labels._source_fingerprint(
        getattr(field, placement)
    ) != token_labels._source_fingerprint(getattr(changed, placement))


def test_the_max_gap_retype_keeps_find_mentions_fingerprint(
    tmp_path, monkeypatch
) -> None:
    """`5d4e3b9` narrowed `find_mentions`' `max_gap` from `int` to
    `NonNegative` and changed nothing else in it, yet every store labelled
    before it was refused as placed by other rules. Undoing that one edit on
    the live rule must leave its fingerprint where it is."""
    narrowed = inspect.getsource(inspect.unwrap(token_labels.find_mentions))
    assert narrowed.count("max_gap: NonNegative") == 1, "the retype has moved"
    widened = narrowed.replace("max_gap: NonNegative", "max_gap: int")
    probes = [
        _probe_module(
            tmp_path, monkeypatch, source, vars(token_labels)
        ).find_mentions
        for source in (narrowed, widened)
    ]
    fingerprint = token_labels._source_fingerprint

    assert (
        fingerprint(probes[0])
        == token_labels.labelling_rules()["token_labels.find_mentions"]
    ), "the probe has to read as the live rule does"
    assert fingerprint(probes[0]) == fingerprint(probes[1])


def test_the_fingerprint_covers_the_classes_the_sweep_constructs() -> None:
    """`find_mentions` never passes `fuzzy=` on its exact branch, so
    `Mention.fuzzy`'s default decides what every exact mention asserts while
    appearing in no function body: flipping it would relabel the corpus with
    every function on the walk byte-identical."""
    rules = token_labels.labelling_rules()
    hashed = ast.unparse(token_labels._rule_tree(token_labels.Mention))

    assert "fuzzy: bool = False" in hashed
    assert rules["token_labels.Mention"] == token_labels._source_fingerprint(
        token_labels.Mention
    )


def test_a_classes_fingerprint_reads_its_fields_and_not_their_notes() -> None:
    """A class is hashed whole, and a field's note is prose the way a
    docstring is: rewriting one must not cost a relabel, a changed default
    must. Only the first string of a body is a docstring to `ast`."""

    def noted():
        @dataclasses.dataclass
        class Span:
            """One thing."""

            fuzzy: bool = False
            """Whether the span is a near-miss."""

        return Span

    def bare():
        @dataclasses.dataclass
        class Span:
            """One thing."""

            fuzzy: bool = False

        return Span

    def flipped():
        @dataclasses.dataclass
        class Span:
            """One thing."""

            fuzzy: bool = True

        return Span

    fingerprint = token_labels._source_fingerprint

    assert fingerprint(noted()) == fingerprint(bare())
    assert fingerprint(bare()) != fingerprint(flipped())


def test_a_rules_decorators_are_part_of_its_fingerprint() -> None:
    """beartype's import hook recompiles this package so that a decorated
    function's code starts at its `def`, where `inspect.getsource` then
    begins, so every rule on the walk lost its decorators from the hash.

    `is_common_word` is the walk's one decorated rule, and `lru_cache` cannot
    change an answer, which is why nothing noticed.
    """
    rule = surface_forms.is_common_word
    hashed = token_labels._rule_tree(rule).body[0]

    assert not inspect.getsource(inspect.unwrap(rule)).startswith(
        "@"
    ), "the rule has to be one beartype recompiled"
    assert isinstance(hashed, ast.FunctionDef)
    assert [ast.unparse(decorator) for decorator in hashed.decorator_list] == [
        "lru_cache(maxsize=None)"
    ]
    assert token_labels.labelling_rules()[
        "surface_forms.is_common_word"
    ] == token_labels._source_fingerprint(rule)


@pytest.mark.parametrize(
    "head", [r"\d" * 67, "[a-z]" * 40], ids=["escaped", "plain"]
)
def test_a_pattern_constant_fingerprints_its_whole_text_and_flags(
    head: str, monkeypatch
) -> None:
    """A pattern's repr truncates its string's repr to 200 characters, quote
    and doubled backslashes included, so the escaped pair collides at 135
    characters of text. Hashing that repr let an edit to a long regex's tail
    relabel the corpus unrefused.
    """

    def fingerprint(value: object) -> str:
        monkeypatch.setattr(token_labels, "_EPITHET", value)
        return token_labels.labelling_rules()["token_labels._EPITHET"]

    first, second = re.compile(head + "x"), re.compile(head + "y")

    assert repr(first) == repr(second), "the edit has to be past the repr's cut"
    assert fingerprint(first) != fingerprint(second)
    assert fingerprint(frozenset({first})) != fingerprint(frozenset({second}))
    assert fingerprint(re.compile(head)) != fingerprint(
        re.compile(head, re.IGNORECASE)
    )


_FROZENSET_CONSTANT = """
from d3text import token_labels

value = frozenset(("alpha", "beta", "gamma", "delta", "epsilon", "zeta"))
token_labels.MAX_MENTION_GAP = value
print(repr(value))
print(token_labels.labelling_rules()["token_labels.MAX_MENTION_GAP"])
"""


def test_a_frozenset_constant_fingerprints_alike_under_every_hash_seed(
    tmp_path, monkeypatch
) -> None:
    """A `frozenset` iterates in hash order, which `PYTHONHASHSEED` randomises
    per process: hashed as written, one would refuse every store at random
    and name a constant nobody touched.

    In subprocesses because one interpreter agrees with itself however the
    repr is built; the reprs are checked to differ, so the seeds reorder it.
    """
    written, fingerprints = set(), set()
    for seed in ("0", "1", "2"):
        monkeypatch.setenv("PYTHONHASHSEED", seed)
        result = subprocess.run(
            [sys.executable, "-c", _FROZENSET_CONSTANT],
            capture_output=True,
            text=True,
            cwd=tmp_path,
            check=True,
        )
        value, fingerprint = result.stdout.splitlines()[-2:]
        written.add(value)
        fingerprints.add(fingerprint)

    assert len(written) > 1, "the seeds have to reorder the set"
    assert len(fingerprints) == 1


def test_a_store_from_before_the_rules_were_recorded_is_refused(
    tmp_path,
) -> None:
    """It loads clean and cannot say what placed its targets, which is the
    defect: the spans would be extended by whatever the sweep does today."""
    path = tmp_path / "labels.hdf5"

    with h5py.File(path, "w-", libver="latest") as store:
        token_labels.write_label_space(store, stamp=_STAMP)
        del store.attrs["labelling_rules"]

        with pytest.raises(KeyError, match="records no labelling rules"):
            token_labels.store_token_labels(store, "10822008", _empty_labels())
