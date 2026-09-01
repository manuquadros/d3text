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

# Spelled out rather than read off `surface_forms.PLACEHOLDER_FORMS`: a test
# that iterates the constant asserts only that whatever is in it is dropped, so
# deleting a noun deletes it from the expectations too, silently.
_CATEGORY_NOUNS = (
    "plant",
    "plants",
    "mutant",
    "strain",
    "bacteria",
    "bacterium",
    "yeast",
    "protease",
)


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


@pytest.mark.parametrize("noun", _CATEGORY_NOUNS)
def test_a_category_noun_carries_no_id_in_any_casing(noun) -> None:
    """A mention of "plants" links to no particular organism.

    The uppercase spelling is the one that rests on the deletion alone: an
    all-caps form is symbol-like, so `index_key` never asks the frequency
    guard about it, while five of the eight nouns are common enough English
    that the guard hides whether they are still in the set. Built here rather
    than read off `index`, since the tracked fixture registers no bare
    category noun at all and so cannot tell a dropped one from an absent one.
    """
    spellings = (noun, noun.capitalize(), noun.upper())
    index = surface_forms.build_index(
        {f"enz{n}": [spelling] for n, spelling in enumerate(spellings)}
    )

    assert index.entity_ids == frozenset()


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


def test_ordinary_english_designations_carry_no_id() -> None:
    """BRENDA registers `sensitive` as a strain, and the literature uses it."""
    index = surface_forms.build_index(
        {
            "str1": ["sensitive"],
            "str2": ["original"],
            "str3": ["yielding"],
            "enz1": ["nitrilase"],
        }
    )

    assert index.lookup(["sensitive"]) == frozenset()
    assert index.lookup(["original"]) == frozenset()
    assert index.lookup(["yielding"]) == frozenset()
    assert index.lookup(["nitrilase"]) == {"enz1"}


def test_the_frequency_guard_spares_symbol_forms() -> None:
    """Case already separates `FOR` from `for`, so frequency must not judge it.

    The guard reads general English, where `for` is about as common as a word
    gets. Asking it about a form whose case is load-bearing would delete the
    enzyme on the strength of the preposition's frequency.
    """
    index = surface_forms.build_index({"enz1": ["FOR"], "enz2": ["HAS"]})

    assert index.lookup(["FOR"]) == {"enz1"}
    assert index.lookup(["HAS"]) == {"enz2"}
    assert index.lookup(["for"]) == frozenset()


def test_a_common_word_keeps_its_id_behind_a_modifier() -> None:
    """Same modifier reading as the category nouns: only the bare form goes."""
    index = surface_forms.build_index(
        {"str1": ["original"], "str2": ["original Kluyver isolate"]}
    )

    assert index.lookup(["original"]) == frozenset()
    assert index.lookup(["original", "Kluyver", "isolate"]) == {"str2"}


def test_bacterial_genera_survive_the_frequency_guard() -> None:
    """The cutoff is calibrated to sit above the genera, and that is fragile.

    `escherichia` (2.63), `pseudomonas` (2.59) and `bacillus` (2.70) are the
    closest legitimate names to `COMMON_WORD_ZIPF`, so they are what a raised
    threshold or a re-estimated frequency table would take first — silently,
    and at the cost of most of the bacterial channel.
    """
    genera = ["escherichia", "pseudomonas", "bacillus", "streptomyces"]
    index = surface_forms.build_index(
        {f"bac{n}": [genus] for n, genus in enumerate(genera)}
    )

    for n, genus in enumerate(genera):
        assert index.lookup([genus]) == {f"bac{n}"}, genus


def test_an_entity_named_only_by_an_english_word_becomes_unreachable() -> None:
    """Deliberate, and the one place a hygiene rule may cost an entity.

    Keeping the key to preserve reachability would not make the strain
    findable — every occurrence of `sensitive` in the literature would answer
    to it — while the mentions it invents land across the whole corpus. The
    entity is the cheaper loss, so this must not be "fixed" back.
    """
    index = surface_forms.build_index({"str1": ["sensitive"]})

    assert index.entity_ids == frozenset()


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
        "N. erythropolis",
        "Nocardia rhodochrous",
        "N. rhodochrous",
    }


def test_strain_forms_leave_out_the_taxon_name(tables) -> None:
    """A strain's `taxon` names the species, not the strain.

    Indexing it would attach strain IDs to bacterium mentions.
    """
    extracted = surface_forms.strain_forms(tables["strains"])

    assert "ATCC 201872" in extracted["289"]
    assert "Schizosaccharomyces pombe" not in extracted["289"]


def test_every_indexed_id_wears_a_prefix_the_corpus_schema_declares(
    forms,
) -> None:
    """The index keys entities the way a split frame's gold set spells them.

    Asserted over the IDs the index actually emits rather than over
    `BRENDA_PREFIXES`, which is now derived from `BRENDA_SCHEMA` and would
    therefore agree with it by construction. What is still worth pinning is
    that the derivation reaches every namespace: a prefix that disagreed with
    the corpus would not fail, it would build an index no gold set can match.
    """
    declared = {
        entity_type.prefix for entity_type in BRENDA_SCHEMA.entity_types
    }

    used = {entity_id[:3] for entity_id in forms}

    assert used == declared


def test_abbreviated_genus_shortens_a_binomial() -> None:
    """`Escherichia coli` is written `E. coli` in running text."""
    assert surface_forms.abbreviated_genus("Escherichia coli") == "E. coli"


def test_abbreviated_genus_keeps_the_strain_qualifier() -> None:
    """The strain-qualified form abbreviates the same way the bare one does."""
    assert (
        surface_forms.abbreviated_genus("Escherichia coli K-12")
        == "E. coli K-12"
    )


@pytest.mark.parametrize(
    "form",
    [
        "DSM 20745",  # culture-collection number: no epithet follows
        "ATCC 25922",
        "E. coli",  # already abbreviated: no lowercase run in the genus
        "Candidatus Liberibacter",  # capitalized second word is no epithet
        "nitrilase",  # single word
        "",
    ],
)
def test_abbreviated_genus_declines_non_binomials(form) -> None:
    """A form that does not open with a binomial gets no abbreviation.

    `DSM 20745` mangled to `D. 20745` would be a phantom surface form
    attached to a real strain ID — a silent mislabel, not a recall gain.
    """
    assert surface_forms.abbreviated_genus(form) is None


def test_bacteria_forms_carry_the_abbreviated_variant() -> None:
    """37% synonym coverage, median 0: the abbreviation must be generated."""
    extracted = surface_forms.bacteria_forms(
        {"42": {"organism": "Bacillus subtilis", "synonyms": []}}
    )

    assert "B. subtilis" in extracted["42"]
    assert "Bacillus subtilis" in extracted["42"]


def test_strain_designations_carry_the_abbreviated_variant() -> None:
    """A designation opening with the binomial abbreviates; numbers do not."""
    extracted = surface_forms.strain_forms(
        {
            "7": {
                "designations": ["Escherichia coli K-12", "DSM 20745"],
                "cultures": [],
            }
        }
    )

    assert "E. coli K-12" in extracted["7"]
    assert "DSM 20745" in extracted["7"]
    assert "D. 20745" not in extracted["7"]


def _bacterium_names(names: list[str]) -> list[str]:
    return surface_forms.bacteria_forms(
        {"1": {"organism": names[0], "synonyms": names[1:]}}
    )["1"]


def _strain_names(names: list[str]) -> list[str]:
    return surface_forms.strain_forms(
        {"1": {"designations": names, "cultures": []}}
    )["1"]


def _other_organism_names(names: list[str]) -> list[str]:
    return surface_forms.other_organism_forms([{"1": name} for name in names])[
        "1"
    ]


_NAME_BEARING = pytest.mark.parametrize(
    "extract",
    [_bacterium_names, _strain_names, _other_organism_names],
    ids=["bacteria", "strains", "other_organisms"],
)


@_NAME_BEARING
def test_every_name_bearing_extractor_abbreviates_the_genus(
    extract,
) -> None:
    """Genus abbreviation is a property of the index, not of one extractor.

    Running text writes a species `C. albicans` after naming it once in full,
    so an extractor that indexes only the full binomial makes that mention
    unreachable — and an entity type whose names are harvested from running
    text in the first place is the last one that can afford to skip it.
    Asserted over all three so no single extractor can drift out of step.
    """
    forms = extract(["Candida albicans"])

    assert "Candida albicans" in forms
    assert "C. albicans" in forms


@_NAME_BEARING
@pytest.mark.parametrize("name", ["rice", "HIV-1", "DSM 20745"])
def test_no_name_bearing_extractor_abbreviates_a_non_binomial(
    extract, name
) -> None:
    """The binomial guard has to hold wherever the expansion is applied.

    `D. 20745` would be a phantom form attached to a real ID, so widening the
    expansion to a third extractor must not widen what it mangles.
    """
    assert extract([name]) == [name]


def test_fuzzy_ids_finds_an_inflectional_variant() -> None:
    """`oxidases` is one edit from the registered `oxidase`."""
    index = surface_forms.build_index({"enz1": ["oxidase"]})

    assert index.fuzzy_ids("oxidases") == {"enz1"}


def test_fuzzy_ids_is_empty_for_an_unrelated_word() -> None:
    index = surface_forms.build_index({"enz1": ["oxidase"]})

    assert index.fuzzy_ids("temperature") == frozenset()


def test_fuzzy_ids_declines_words_below_the_length_floor() -> None:
    """A short word is closer to everything, which is what the floor avoids."""
    index = surface_forms.build_index({"enz1": ["oda"]})

    assert index.fuzzy_ids("odd") == frozenset()


def test_fuzzy_ids_declines_a_common_english_word() -> None:
    """`protein` scores 80 against the unrelated enzyme `prorenin`.

    Filtering the query, not just the candidates, is what keeps a loose
    cutoff from spending real negative signal on ordinary vocabulary that
    happens to sit near a technical name.
    """
    index = surface_forms.build_index({"enz1": ["prorenin"]})

    assert index.fuzzy_ids("protein") == frozenset()


def test_fuzzy_ids_respects_the_symbol_case_policy() -> None:
    """A short symbol keeps its case; folding it would collide with English."""
    index = surface_forms.build_index({"enz1": ["MMP3"]})

    assert index.fuzzy_ids("MMP3X") == {"enz1"}
    assert index.fuzzy_ids("mmp3x") == frozenset()


def test_fuzzy_ids_ignores_multiword_forms() -> None:
    """Multi-word forms are out of scope; `lookup`'s own tolerance covers them.

    `streptomyce` is missing the final `s` of `Streptomyces`, but the only
    registered form is the two-word binomial, and fuzzy matching is only ever
    asked of a single word.
    """
    index = surface_forms.build_index({"bac1": ["Streptomyces griseocarneus"]})

    assert index.fuzzy_ids("streptomyce") == frozenset()


def test_fuzzy_ids_memoizes_repeated_words() -> None:
    """A second call for the same word must not re-score it.

    Word occurrence in running text is Zipfian, so `fuzzy_ids` is asked of
    the same word thousands of times across a corpus; the result is a pure
    function of `(word, index, cutoff)`, so the second call has to be a
    cache hit rather than a second `process.extractOne` scan.
    """
    index = surface_forms.build_index({"enz1": ["oxidase"]})

    with unittest.mock.patch.object(
        surface_forms.process,
        "extractOne",
        wraps=surface_forms.process.extractOne,
    ) as extract_one:
        first = index.fuzzy_ids("oxidases")
        assert extract_one.call_count > 0

        calls_after_first = extract_one.call_count
        second = index.fuzzy_ids("oxidases")

    assert extract_one.call_count == calls_after_first
    assert second == first


def test_abbreviated_variants_are_reachable_through_the_index() -> None:
    """The gap this closes: text says `E. coli`, the table says the binomial."""
    index = surface_forms.build_index(
        surface_forms.brenda_surface_forms(
            {
                "bacteria": {
                    "9": {"organism": "Escherichia coli", "synonyms": []}
                }
            }
        )
    )

    assert index.lookup(["E", "coli"]) == {"bac9"}


def test_the_index_digest_is_the_same_for_two_builds_of_one_index() -> None:
    """A fingerprint that moved between processes could refuse nothing.

    Iteration order over the tables and over an entity's forms is not part of
    what an index means, so neither may reach the digest.
    """
    forwards = surface_forms.build_index(
        {
            "enz1": ["cholesterol oxidase", "COD"],
            "bac3": ["Streptomyces griseocarneus"],
        }
    )
    backwards = surface_forms.build_index(
        {
            "bac3": ["Streptomyces griseocarneus"],
            "enz1": ["COD", "cholesterol oxidase"],
        }
    )

    assert surface_forms.index_digest(forwards) == surface_forms.index_digest(
        forwards
    )
    assert surface_forms.index_digest(forwards) == surface_forms.index_digest(
        backwards
    )


def test_the_index_digest_moves_when_an_extractor_indexes_more() -> None:
    """The axis a dataset list alone would miss.

    Giving an extractor the abbreviated-genus expansion changes which tokens
    the store labels while leaving its types, prefixes and codes identical, so
    only a fingerprint of the index itself can tell the two artifacts apart.
    """
    verbatim = surface_forms.build_index({"oth7": ["Candida albicans"]})
    expanded = surface_forms.build_index(
        surface_forms.brenda_surface_forms({}, [{"7": "Candida albicans"}])
    )

    assert expanded.lookup(["C", "albicans"]) == {"oth7"}
    assert not verbatim.lookup(["C", "albicans"])
    assert surface_forms.index_digest(verbatim) != surface_forms.index_digest(
        expanded
    )


def test_the_index_digest_moves_when_another_entity_owns_a_form() -> None:
    """The same keys under different IDs label the same token as another type,
    so the digest has to read the entity IDs and not only the keys."""
    one = surface_forms.build_index({"oth7": ["Jaculus orientalis"]})
    other = surface_forms.build_index({"oth8": ["Jaculus orientalis"]})

    assert surface_forms.index_digest(one) != surface_forms.index_digest(other)
