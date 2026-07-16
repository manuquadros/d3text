"""Ported from the deleted src/tests/test_dict_tagger.py (entities layer).

Re-targeted at the live d3text.models.dict_tagger. Tagging itself needs no
model and no network; the `from_schema` tests read term lists off disk, from
`tmp_path` or from the vocabulary files committed under `DATA_DIR`.
"""

from pathlib import Path

from d3text.data.data import DATA_DIR
from d3text.datasets.brenda import BRENDA_SCHEMA
from d3text.models.dict_tagger import DictTagger
from d3text.schema import EntityType, RelationType, Schema
from d3text.utils import Token, repr_sequence

sample = [
    Token(string="on", offset=(3475, 3477), prediction="O", gold_label=None),
    Token(string="the", offset=(3478, 3481), prediction="O", gold_label=None),
    Token(
        string="production",
        offset=(3482, 3492),
        prediction="O",
        gold_label=None,
    ),
    Token(string="of", offset=(3493, 3495), prediction="O", gold_label=None),
    Token(string="COX", offset=(3496, 3499), prediction="O", gold_label=None),
    Token(string=".", offset=(3499, 3500), prediction="O", gold_label=None),
]


def test_dict_tagger_merges_matching_span() -> None:
    expected = [
        Token(
            string="on", offset=(3475, 3477), prediction="O", gold_label=None
        ),
        Token(
            string="the", offset=(3478, 3481), prediction="O", gold_label=None
        ),
        Token(
            string="production of COX",
            offset=(3482, 3499),
            prediction="process",
            gold_label=None,
        ),
        Token(string=".", offset=(3499, 3500), prediction="O", gold_label=None),
    ]

    dtagger = DictTagger(
        vocabs={"process": ["production of COX", "enzyme activity alteration"]}
    )
    assert list(dtagger.tag(sample)) == expected


def test_dict_tagger_leaves_non_o_tokens_untouched() -> None:
    tokens = [
        Token(
            string="production",
            offset=(3482, 3492),
            prediction="enzyme",
            gold_label=None,
        ),
        Token(
            string="of", offset=(3493, 3495), prediction="O", gold_label=None
        ),
    ]
    # The first token already carries a non-"O" prediction, so it must be
    # yielded unchanged rather than pulled into a dictionary match.
    dtagger = DictTagger(vocabs={"process": ["production of"]})
    assert list(dtagger.tag(tokens)) == tokens


def test_dict_tagger_below_cutoff_no_match() -> None:
    # A vocabulary term that does not resemble the input at all should not
    # trigger a match; every token is passed through unchanged.
    dtagger = DictTagger(
        vocabs={"process": ["completely unrelated phrase"]}, cutoff=93.0
    )
    assert list(dtagger.tag(sample)) == sample


def test_dict_tagger_cutoff_gates_imperfect_matches() -> None:
    # Scores 82.35 against the term, so this pins only the cutoff-100
    # rejection: the case difference is already below the default cutoff.
    near_miss = [
        Token(
            string="production of cox",
            offset=(0, 17),
            prediction="O",
            gold_label=None,
        )
    ]
    strict = DictTagger(vocabs={"process": ["production of COX"]}, cutoff=100.0)
    assert list(strict.tag(near_miss)) == near_miss


def test_dict_tagger_window_cap_limits_span() -> None:
    # _find_best_match only considers windows up to min(len, 10) tokens, so a
    # 12-token phrase can never be matched as a single span.
    tokens = [Token(f"w{i}", (i * 3, i * 3 + 2), "O", None) for i in range(12)]
    full_phrase = repr_sequence(tokens)
    tagged = list(
        DictTagger(vocabs={"p": [full_phrase]}, cutoff=93.0).tag(tokens)
    )
    assert len(tagged) == 12  # nothing merged
    assert all(tok.prediction == "O" for tok in tagged)


def _schema(*entity_types: EntityType) -> Schema:
    """A schema carrying only what `Schema` demands — the tagger reads the
    entity types and never the relations."""
    return Schema(
        entity_types=entity_types,
        relation_types=(RelationType(name="none", is_none=True),),
    )


def _write_vocab(directory: Path, name: str, *terms: str) -> None:
    (directory / name).write_text("\n".join(terms) + "\n")


def test_from_schema_labels_matches_with_the_entity_type_name(
    tmp_path: Path,
) -> None:
    _write_vocab(tmp_path, "enzymes.txt", "production of COX")
    tagger = DictTagger.from_schema(
        _schema(
            EntityType(name="enzymes", prefix="enz", vocab_path="enzymes.txt")
        ),
        tmp_path,
    )

    tagged = list(tagger.tag(sample))

    assert [tok.string for tok in tagged] == [
        "on",
        "the",
        "production of COX",
        ".",
    ]
    assert [tok.prediction for tok in tagged] == ["O", "O", "enzymes", "O"]


def test_from_schema_skips_entity_types_without_a_vocab_path(
    tmp_path: Path,
) -> None:
    _write_vocab(tmp_path, "enzymes.txt", "production of COX")
    tagger = DictTagger.from_schema(
        _schema(
            EntityType(name="bacteria", prefix="bac"),
            EntityType(name="enzymes", prefix="enz", vocab_path="enzymes.txt"),
        ),
        tmp_path,
    )

    assert [vocab.label for vocab in tagger._vocabs] == ["enzymes"]


def test_from_schema_resolves_vocab_path_relative_to_data_dir(
    tmp_path: Path,
) -> None:
    entity_type = EntityType(
        name="enzymes", prefix="enz", vocab_path="enzymes.txt"
    )
    here, there = tmp_path / "here", tmp_path / "there"
    here.mkdir()
    there.mkdir()
    _write_vocab(here, "enzymes.txt", "production of COX")
    _write_vocab(there, "enzymes.txt", "completely unrelated phrase")

    assert [
        tok.prediction
        for tok in DictTagger.from_schema(_schema(entity_type), here).tag(
            sample
        )
    ] == ["O", "O", "enzymes", "O"]
    assert (
        list(DictTagger.from_schema(_schema(entity_type), there).tag(sample))
        == sample
    )


def test_from_schema_forwards_the_cutoff(tmp_path: Path) -> None:
    _write_vocab(tmp_path, "enzymes.txt", "production of COX")
    schema = _schema(
        EntityType(name="enzymes", prefix="enz", vocab_path="enzymes.txt")
    )
    # Scores 94.12 against the term: matched at the default cutoff, rejected
    # at 100, so a cutoff the classmethod dropped would show up here.
    near_miss = [Token("production of COx", (0, 17), "O", None)]

    assert [
        tok.prediction
        for tok in DictTagger.from_schema(schema, tmp_path).tag(near_miss)
    ] == ["enzymes"]
    assert (
        list(
            DictTagger.from_schema(schema, tmp_path, cutoff=100.0).tag(
                near_miss
            )
        )
        == near_miss
    )


def test_from_schema_builds_a_vocab_per_named_brenda_term_list() -> None:
    tagger = DictTagger.from_schema(BRENDA_SCHEMA, DATA_DIR)

    named = [et.name for et in BRENDA_SCHEMA.entity_types if et.vocab_path]
    assert [vocab.label for vocab in tagger._vocabs] == named
    assert "other_organisms" not in named
