"""The synonym -> entity-ID index, and the two forms that must not be in it.

Everything here runs off ``brenda_references/tests/test_files/testdb.json``,
which is tracked, 40 kB, and carries the same four-table shape as the 1.1 GB
``documents.json`` the pipeline actually reads — including, usefully, `More`
as a synonym of the enzyme `Aliphatic nitrilase`.
"""

import json
import pathlib
import unittest.mock

import pytest
from d3text import surface_forms
from d3text.datasets.brenda import BRENDA_SCHEMA

_TESTDB = (
    pathlib.Path(__file__).resolve().parent.parent
    / "brenda_references"
    / "tests"
    / "test_files"
    / "testdb.json"
)

# `enzymes["3008"]` is `Aliphatic nitrilase`, and `More` is one of its
# synonyms; it is the entity the placeholder deletion has to leave reachable.
_NITRILASE = "enz3008"


@pytest.fixture(scope="module")
def tables() -> dict[str, dict[str, object]]:
    return surface_forms.load_entity_tables(_TESTDB)


@pytest.fixture(scope="module")
def forms(tables) -> dict[str, list[str]]:
    return surface_forms.brenda_surface_forms(
        tables,
        (
            document.get("other_organisms") or {}
            for document in tables["documents"].values()
        ),
    )


@pytest.fixture(scope="module")
def index(forms) -> surface_forms.SurfaceFormIndex:
    return surface_forms.build_index(forms)


def test_load_entity_tables_reads_a_small_dump_whole(tables) -> None:
    """The tail-seek is for the 1.1 GB dump; a fixture is just JSON.

    The `documents` table is the difference that matters: the tail route cannot
    reach it, and it is the only place other-organism names exist.
    """
    assert set(tables) == {"documents", "enzymes", "bacteria", "strains"}
    assert tables == json.loads(_TESTDB.read_text(encoding="utf8"))


def test_more_placeholder_is_absent_from_the_index(index, tables) -> None:
    """`More` is a curation marker, not a name, and it is in the fixture."""
    assert "More" in tables["enzymes"]["3008"]["synonyms"]

    assert index.lookup(["More"]) == frozenset()
    assert index.lookup(["more"]) == frozenset()


def test_category_nouns_are_absent_from_the_index(index) -> None:
    """A mention of "plants" links to no particular organism."""
    for noun in surface_forms.PLACEHOLDER_FORMS:
        assert index.lookup([noun]) == frozenset(), noun
        assert index.lookup([noun.capitalize()]) == frozenset(), noun


def test_the_enzyme_more_stood_in_for_stays_reachable(index) -> None:
    """The deletion must cost a form, not an entity."""
    assert _NITRILASE in index.lookup(["Aliphatic", "nitrilase"])
    assert _NITRILASE in index.entity_ids


def test_dropping_the_placeholders_removes_no_entity(forms, index) -> None:
    """Every entity reachable without the deletions is reachable with them.

    The sharp version of the previous test: it is not enough that one enzyme
    survives, no entity may lose its last handle. Compared against an index
    built with the deletion disabled rather than against a hardcoded list, so
    the assertion keeps meaning when the fixture grows.
    """
    with unittest.mock.patch.object(
        surface_forms, "PLACEHOLDER_FORMS", frozenset()
    ):
        unfiltered = surface_forms.build_index(forms)

    assert unfiltered.entity_ids == index.entity_ids
    assert len(unfiltered) > len(index)


def test_a_category_noun_keeps_its_id_behind_a_modifier() -> None:
    """Drop them or require a modifier — this is the modifier reading.

    Only the bare form is dropped, so a form that merely contains a category
    noun is untouched.
    """
    index = surface_forms.build_index(
        {"enz1": ["protease"], "enz2": ["alkaline protease"]}
    )

    assert index.lookup(["protease"]) == frozenset()
    assert index.lookup(["alkaline", "protease"]) == {"enz2"}


def test_symbol_forms_are_matched_case_sensitively() -> None:
    """`FOR` is an enzyme, `for` is a preposition, and case is all there is."""
    index = surface_forms.build_index({"enz1": ["FOR"], "enz2": ["catalase"]})

    assert index.lookup(["FOR"]) == {"enz1"}
    assert index.lookup(["for"]) == frozenset()


def test_descriptive_forms_fold_case() -> None:
    """A long lowercase name collides with no English word, so it can fold."""
    index = surface_forms.build_index({"enz2": ["cholesterol oxidase"]})

    assert index.lookup(["Cholesterol", "Oxidase"]) == {"enz2"}


def test_punctuation_inside_a_form_is_not_compared() -> None:
    """`MMP-3` and `MMP 3` are the same enzyme written two ways."""
    index = surface_forms.build_index({"enz1": ["MMP-3"]})

    assert index.lookup(["MMP", "3"]) == {"enz1"}


def test_forms_shorter_than_the_minimum_carry_no_id() -> None:
    """`CO` is cholesterol oxidase in BRENDA and carbon monoxide elsewhere."""
    index = surface_forms.build_index({"enz1": ["CO", "COD"]})

    assert index.lookup(["CO"]) == frozenset()
    assert index.lookup(["COD"]) == {"enz1"}


def test_one_form_can_name_several_entities() -> None:
    """A surface form is not owned by one entity; the index is a multimap."""
    index = surface_forms.build_index(
        {"enz1": ["nitrilase"], "enz2": ["nitrilase"]}
    )

    assert index.lookup(["nitrilase"]) == {"enz1", "enz2"}


def test_other_organism_names_come_from_the_documents(index, forms) -> None:
    """`oth` IDs have no table anywhere; their names are inline on documents.

    `documents.json` carries four tables and none of them is
    `other_organisms`, so an index scoped to the three entity tables would
    label every other-organism mention negative — the exact assertion the
    third target exists to avoid.
    """
    assert index.lookup(["Brevibacterium", "sterolicum"]) == {"oth978"}
    assert any(entity.startswith("oth") for entity in forms)


def test_other_organism_names_are_pooled_across_documents() -> None:
    """One document's naming has to be usable while labelling another.

    An unannotated mention is by definition not in its own document's column,
    so the only source of the name is some other document.
    """
    pooled = surface_forms.other_organism_forms(
        [{"1": "Nocardia erythropolis"}, {"1": "Nocardia rhodochrous"}]
    )

    assert set(pooled["1"]) == {
        "Nocardia erythropolis",
        "Nocardia rhodochrous",
    }


def test_strain_forms_leave_out_the_taxon_name(tables) -> None:
    """A strain's `taxon` names the species, not the strain.

    Indexing it would attach strain IDs to bacterium mentions.
    """
    extracted = surface_forms.strain_forms(tables["strains"])

    assert "ATCC 201872" in extracted["289"]
    assert "Schizosaccharomyces pombe" not in extracted["289"]


def test_prefixes_agree_with_the_corpus_schema() -> None:
    """The index keys entities the way a split frame's gold set spells them.

    `BRENDA_SCHEMA` is where the corpus side declares this; the index cannot
    import it without dragging the BRENDA data layer into a leaf module, so
    the two are pinned together here instead.
    """
    declared = {
        entity_type.name: entity_type.prefix
        for entity_type in BRENDA_SCHEMA.entity_types
    }

    assert dict(surface_forms.BRENDA_PREFIXES) == declared
