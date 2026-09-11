"""The synonym -> entity-ID index, and the two forms that must not be in it.

Everything here runs off ``brenda_references/tests/test_files/testdb.json``,
which is tracked, 40 kB, and carries the same four-table shape as the 1.1 GB
``documents.json`` the pipeline actually reads — including, usefully, `More`
as a synonym of the enzyme `Aliphatic nitrilase`.
"""

import json
import pathlib
import subprocess
import sys
import tempfile
import unittest.mock
from collections.abc import Callable
from typing import Any

import pytest
from d3text import surface_forms, token_labels
from d3text.schema import BRENDA_SCHEMA

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

_PRINT_HEAVY_MODULES = (
    "import sys; print(sorted(m for m in sys.modules if m.startswith(("
    "'torch', 'd3text.data', 'd3text.datasets', 'brenda_references', "
    "'lpsn_interface'))))"
)

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
    "archaeon",
    "plasmid",
    "yeast",
    "protease",
)


@pytest.fixture(scope="module")
def tables() -> dict[str, dict[str, Any]]:
    return surface_forms.load_entity_tables(_TESTDB)


@pytest.fixture(scope="module")
def forms(tables: dict[str, dict[str, Any]]) -> dict[str, list[str]]:
    return surface_forms.brenda_surface_forms(
        tables,
        (
            document.get("other_organisms") or {}
            for document in tables["documents"].values()
        ),
    )


@pytest.fixture(scope="module")
def index(forms: dict[str, list[str]]) -> surface_forms.SurfaceFormIndex:
    return surface_forms.build_index(forms)


def test_load_entity_tables_reads_a_small_dump_whole(
    tables: dict[str, dict[str, Any]],
) -> None:
    """The tail-seek is for the 1.1 GB dump; a fixture is just JSON.

    The `documents` table is the difference that matters: the tail route cannot
    reach it, and it is the only place other-organism names exist.
    """
    assert set(tables) == {"documents", "enzymes", "bacteria", "strains"}
    assert tables == json.loads(_TESTDB.read_text(encoding="utf8"))


def test_load_entity_tables_seeks_the_tail_past_a_shrunk_window(
    tables: dict[str, dict[str, Any]],
) -> None:
    """The 1.1 GB dump's only route, pinned against the whole-file one.

    `_TAIL_SEARCH_BYTES` is lowered below the fixture's size to force the
    seek branch, while staying wide enough for the shrunk tail to still carry
    `enzymes`; `documents` is left out of the comparison for the reason
    `test_load_entity_tables_reads_a_small_dump_whole` gives.
    """
    with unittest.mock.patch.object(
        surface_forms, "_TAIL_SEARCH_BYTES", 30_000
    ):
        seeked = surface_forms.load_entity_tables(_TESTDB)

    assert set(seeked) == {"enzymes", "bacteria", "strains"}
    for name in ("enzymes", "bacteria", "strains"):
        assert seeked[name] == tables[name]


def test_load_entity_tables_raises_when_the_tail_carries_no_key() -> None:
    """A window too narrow to reach the entity-table key fails loudly.

    Silently returning nothing would misreport a misconfigured window as an
    empty dump.
    """
    with unittest.mock.patch.object(
        surface_forms, "_TAIL_SEARCH_BYTES", 20_000
    ):
        with pytest.raises(ValueError, match="carries no"):
            surface_forms.load_entity_tables(_TESTDB)


def test_more_placeholder_is_absent_from_the_index(
    index: surface_forms.SurfaceFormIndex, tables: dict[str, dict[str, Any]]
) -> None:
    """`More` is a curation marker, not a name, and it is in the fixture."""
    assert "More" in tables["enzymes"]["3008"]["synonyms"]

    assert index.lookup(["More"]) == frozenset()
    assert index.lookup(["more"]) == frozenset()


@pytest.mark.parametrize("noun", _CATEGORY_NOUNS)
def test_a_category_noun_carries_no_id_in_any_casing(noun: str) -> None:
    """A mention of "plants" links to no particular organism.

    The uppercase spelling is the one that rests on the deletion alone: an
    all-caps form is symbol-like, so `_index_key` never asks the frequency
    guard about it, while seven of the ten nouns are common enough English
    that the guard hides whether they are still in the set. Built here rather
    than read off `index`, since the tracked fixture registers no bare
    category noun at all and so cannot tell a dropped one from an absent one.
    """
    spellings = (noun, noun.capitalize(), noun.upper())
    index = surface_forms.build_index(
        {f"enz{n}": [spelling] for n, spelling in enumerate(spellings)}
    )

    assert index.entity_ids == frozenset()


def test_the_enzyme_more_stood_in_for_stays_reachable(
    index: surface_forms.SurfaceFormIndex,
) -> None:
    """The deletion must cost a form, not an entity."""
    assert _NITRILASE in index.lookup(["Aliphatic", "nitrilase"])
    assert _NITRILASE in index.entity_ids


def test_dropping_the_placeholders_loses_only_placeholder_named_entities(
    forms: dict[str, list[str]],
) -> None:
    """Only an entity named by nothing but a placeholder is lost.

    The sharp version of the previous test: it is not enough that one enzyme
    survives, no entity with a real name may lose its last handle. Compared
    against an index built with the deletion disabled rather than against a
    hardcoded list, so the assertion keeps meaning when the fixture grows. The
    probes are what keep that comparison from being vacuous: every placeholder
    the tracked fixture registers is also ordinary English, so the frequency
    guard deletes it either way. `PROTEASE` is all-caps, a spelling the guard
    is never asked about, and `alkaline protease` keeps its entity reachable
    once it goes. The shipped dump files `plasmid` and `archaeon` each as a
    bacterium with no other name, and such a record is the one loss allowed.
    """
    probe = dict(forms) | {
        "enz999999": ["PROTEASE", "alkaline protease"],
        "bac999999": ["plasmid"],
    }
    index = surface_forms.build_index(probe)
    with unittest.mock.patch.object(
        surface_forms, "PLACEHOLDER_FORMS", frozenset()
    ):
        unfiltered = surface_forms.build_index(probe)

    assert unfiltered.entity_ids - index.entity_ids == {"bac999999"}
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


def _plasmid_index() -> surface_forms.SurfaceFormIndex:
    """The bacteria the shipped dump names `plasmid`, `archaeon` and
    `plasmid R100`, beside the enzyme `plasmin`."""
    return surface_forms.build_index(
        surface_forms.brenda_surface_forms(
            {
                "bacteria": {
                    "19375": {"organism": "plasmid", "synonyms": []},
                    "20072": {"organism": "archaeon", "synonyms": []},
                    "19397": {"organism": "plasmid R100", "synonyms": []},
                },
                "enzymes": {
                    "15373": {"recommended_name": "plasmin", "synonyms": []}
                },
            }
        )
    )


def test_a_bare_placeholder_is_a_trained_negative() -> None:
    """Keyed, `plasmid` made every plasmid in the literature a mention of one
    bacterium; unkeyed, it fell through to the fuzzy layer and scored 85.7
    against `plasmin` (`plasmids` 80.0), so it stayed abstained on as an
    enzyme near-miss. Neither may happen, and the designation keeps its ID."""
    text = (
        "Plasmids and the plasmid were cured, as was the archaeon; "
        "plasmid R100 was not."
    )
    gold = {"bac19375", "bac20072"}

    mentions = token_labels.find_mentions(text, _plasmid_index())
    rows = token_labels.mention_spans(mentions, gold)
    labels = token_labels.character_labels(len(text), mentions, gold)

    assert [
        (text[start:end], code, is_gold)
        for start, end, code, is_gold in rows.tolist()
    ] == [("plasmid R100", token_labels.BRENDA_LABELS.code_of("bac19397"), 0)]
    for word in ("Plasmids", "plasmid", "archaeon"):
        start = text.index(word)
        assert set(labels[start : start + len(word)]) == {token_labels.OUTSIDE}


def test_a_near_miss_beside_a_placeholder_keeps_its_abstention() -> None:
    """Only a placeholder and its plural are refused, not every word near one.

    `plasmins` scores 80.0 against `plasmid`, and `Bacteroidia`, a class of
    bacteria, 84.2 against `bacteria` but 81.8 against `Bacteroides`; refusing
    either would turn an abstention on a name into a trained negative on it.
    """
    index = surface_forms.build_index(
        {"enz15373": ["plasmin"], "bac769": ["Bacteroides"]}
    )

    assert index.fuzzy_ids("plasmins") == {"enz15373"}
    assert index.fuzzy_ids("Bacteroidia") == {"bac769"}


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
    """Case separates `CAMP` from `camp`, so frequency must not judge it.

    The guard reads general English, which folds case and so answers for the
    campsite. Asking it about a spelling English never writes would delete the
    enzyme on the strength of the ordinary word's frequency.
    """
    index = surface_forms.build_index({"enz1": ["CAMP"], "enz2": ["ChAT"]})

    assert index.lookup(["CAMP"]) == {"enz1"}
    assert index.lookup(["ChAT"]) == {"enz2"}
    assert index.lookup(["camp"]) == frozenset()


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
    """`CAMP` is an enzyme, `camp` is a field, and case is all there is."""
    index = surface_forms.build_index({"enz1": ["CAMP"], "enz2": ["catalase"]})

    assert index.lookup(["CAMP"]) == {"enz1"}
    assert index.lookup(["camp"]) == frozenset()


def test_descriptive_forms_fold_case() -> None:
    """A long lowercase name collides with no English word, so it can fold."""
    index = surface_forms.build_index({"enz2": ["cholesterol oxidase"]})

    assert index.lookup(["Cholesterol", "Oxidase"]) == {"enz2"}


def test_punctuation_inside_a_form_is_not_compared() -> None:
    """`MMP-3` and `MMP 3` are the same enzyme written two ways."""
    index = surface_forms.build_index({"enz1": ["MMP-3"]})

    assert index.lookup(["MMP", "3"]) == {"enz1"}


def test_a_thousands_separator_is_not_a_word_boundary() -> None:
    """`DSM 22,228` is one deposit number, and `DSM 22` is another strain's.

    Splitting it offers the shorter window to the index, which answers
    confidently with the wrong strain rather than failing.
    """
    assert surface_forms.form_words("Orbus hercynius DSM 22,228") == [
        "Orbus",
        "hercynius",
        "DSM",
        "22228",
    ]


@pytest.mark.parametrize(
    ("text", "words"),
    [
        ("ATCC 35984, 35983", ["ATCC", "35984", "35983"]),
        ("ATCC 35984,35983", ["ATCC", "35984", "35983"]),
        ("NBRC 15308, 100", ["NBRC", "15308", "100"]),
    ],
)
def test_a_comma_between_two_deposit_numbers_still_splits(
    text: str, words: list[str]
) -> None:
    """Gluing a list of deposits invents an accession no collection issued,
    so both halves of the rule are load-bearing: the third case's list item is
    itself three digits, and only the space separates it from a separator."""
    assert surface_forms.form_words(text) == words


def test_word_spans_stay_anchored_to_the_text_as_written() -> None:
    """The join happens in the word, never in the text.

    These offsets are what a caller paints labels with, so dropping the comma
    from the string instead would shift every span after it by one.
    """
    text = "DSM 22,228 and ATCC 6538"

    assert surface_forms.word_spans(text) == [
        ("DSM", 0, 3),
        ("22228", 4, 10),
        ("and", 11, 14),
        ("ATCC", 15, 19),
        ("6538", 20, 24),
    ]
    assert text[4:10] == "22,228"


def test_the_index_reads_a_separator_the_way_the_text_does() -> None:
    """Both sides key through `form_words`, so the two cannot disagree.

    No BRENDA culture number is spelled with a separator today, which is why
    the defect was only ever in the text; one that appeared tomorrow would
    still be reachable.
    """
    index = surface_forms.build_index({"str1": ["DSM 22,228"]})

    assert index.lookup(surface_forms.form_words("DSM 22,228")) == {"str1"}


def test_a_deposit_number_written_without_its_space_still_resolves() -> None:
    """BRENDA records `ATCC 14990` and the literature writes `ATCC14990`.

    The index is keyed by a form's words, so the two spellings are two keys
    and a mention writing the second one reached nothing at all.
    """
    index = surface_forms.build_index({"str1": ["ATCC 14990"]})

    assert index.lookup(surface_forms.form_words("ATCC14990")) == {"str1"}
    assert index.lookup(surface_forms.form_words("ATCC 14990")) == {"str1"}


def test_a_deposit_number_recorded_without_a_space_still_resolves() -> None:
    """The other direction, which is not hypothetical: a thirtieth of the
    accessions in BRENDA's own `cultures` table carry no separator, and the
    text that names those writes the space."""
    index = surface_forms.build_index({"str1": ["DSM642"]})

    assert index.lookup(surface_forms.form_words("DSM 642")) == {"str1"}
    assert index.lookup(surface_forms.form_words("DSM642")) == {"str1"}


def test_a_deposit_inside_a_designation_respells_with_it() -> None:
    """A strain's forms are mostly full designations, not bare accessions."""
    index = surface_forms.build_index(
        {"str1": ["Staphylococcus aureus ATCC 6538"]}
    )
    respelled = surface_forms.form_words("Staphylococcus aureus ATCC6538")

    assert index.lookup(respelled) == {"str1"}


@pytest.mark.parametrize(
    ("designation", "respelled"),
    [("IP 32953", "IP32953"), ("ST 131", "ST131"), ("PAO1", "PAO 1")],
)
def test_a_designation_shaped_like_a_deposit_gains_no_spelling(
    designation: str, respelled: str
) -> None:
    """The acronym is the only thing separating these three from accessions.

    A rule reading any capitals-then-digits as a deposit would hand the sweep
    keys no collection ever issued, on strings that are ordinary designations.
    """
    index = surface_forms.build_index({"str1": [designation]})

    assert index.lookup(surface_forms.form_words(designation)) == {"str1"}
    assert index.lookup(surface_forms.form_words(respelled)) == frozenset()


def test_a_deposit_number_respells_only_in_the_acronym_case() -> None:
    """The acronyms are matched case-sensitively, `AS` being a collection and
    also two ordinary letters, and the respelling keeps that policy rather
    than inventing a looser one: a lowercased form folds as it always did and
    gains nothing."""
    index = surface_forms.build_index({"str1": ["atcc 14990"]})

    assert index.lookup(["ATCC", "14990"]) == {"str1"}
    assert index.lookup(["atcc14990"]) == frozenset()


def test_a_deposit_number_hyphenated_reaches_both_spellings() -> None:
    """`ATCC-14990` already keys as the spaced form, the hyphen being a word
    boundary; it is the joined spelling that has to be added."""
    index = surface_forms.build_index({"str1": ["ATCC-14990"]})

    assert index.lookup(["ATCC", "14990"]) == {"str1"}
    assert index.lookup(["ATCC14990"]) == {"str1"}


def test_the_module_does_not_import_the_brenda_data_layer() -> None:
    """Building an index must cost neither the data layer nor torch.

    The accession grammar this module keys deposits by is shared with
    `d3text.datasets.culture_numbers`, and every module of that package runs
    its `__init__`, which reaches BRENDA and drops an `lpsn.log` into the
    working directory — so the grammar lives here and the dependency runs the
    other way. Checked in a subprocess, so the verdict is about the module's
    own imports and not about what an earlier test file left in `sys.modules`.
    """
    probe = f"import d3text.surface_forms; {_PRINT_HEAVY_MODULES}"
    with tempfile.TemporaryDirectory() as directory:
        result = subprocess.run(
            [sys.executable, "-c", probe],
            capture_output=True,
            text=True,
            cwd=directory,
            check=True,
        )
        littered = sorted(
            path.name for path in pathlib.Path(directory).iterdir()
        )

    assert result.stdout.strip().endswith("[]"), result.stdout
    assert littered == [], f"importing the module littered its cwd: {littered}"


def test_this_suite_imports_no_more_of_the_tree_than_the_module() -> None:
    """Collecting this file must cost neither the data layer nor torch.

    Reaching the schema through `d3text.datasets` once cost this leaf's suite
    four seconds of torch and BRENDA per run. Loaded in a subprocess, since
    under pytest an earlier file may already have paid that.
    """
    probe = (
        "import importlib.util; "
        "spec = importlib.util.spec_from_file_location('suite', "
        f"{str(pathlib.Path(__file__).resolve())!r}); "
        "module = importlib.util.module_from_spec(spec); "
        f"spec.loader.exec_module(module); {_PRINT_HEAVY_MODULES}"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        check=True,
    )

    assert result.stdout.strip().endswith("[]"), result.stdout


def test_forms_shorter_than_the_minimum_carry_no_id() -> None:
    """`CO` is carbon monoxide elsewhere, `COD` chemical oxygen demand.

    Both name an enzyme in BRENDA, and case separates neither of them from the
    sense the rest of the literature gives it.
    """
    index = surface_forms.build_index({"enz1": ["CO", "COD", "CODH"]})

    assert index.lookup(["CO"]) == frozenset()
    assert index.lookup(["COD"]) == frozenset()
    assert index.lookup(["CODH"]) == {"enz1"}


def test_a_three_character_acronym_carries_no_id() -> None:
    """`PCR`, `PBS` and `LPS` are all registered enzyme symbols.

    None of them names an enzyme in running text, and case cannot tell them
    from the method, the buffer and the polysaccharide they usually are,
    because the competing sense is an acronym too. Forms this short were the
    commonest enzyme "mentions" in every corpus measured, under a different
    cast each time, which is why the bar is on length and not on a list.
    """
    index = surface_forms.build_index(
        {"enz1": ["PCR"], "enz2": ["PBS"], "enz3": ["LPS"], "enz4": ["CODH"]}
    )

    assert index.lookup(["PCR"]) == frozenset()
    assert index.lookup(["PBS"]) == frozenset()
    assert index.lookup(["LPS"]) == frozenset()
    assert index.lookup(["CODH"]) == {"enz4"}


def test_a_form_spelled_the_way_english_spells_it_carries_no_id() -> None:
    """The frequency guard follows the spelling, not the lookup table.

    `Name`, `alpha` and `2019` are registered designations short enough to
    read as symbols, so they went into the case-sensitive table and the guard
    never reached them — but running text writes them in exactly that casing,
    which is the premise that table rests on.
    """
    index = surface_forms.build_index(
        {
            "str1": ["Name"],
            "str2": ["alpha"],
            "str3": ["2019"],
            "enz1": ["CAMP"],
        }
    )

    assert index.lookup(["Name"]) == frozenset()
    assert index.lookup(["alpha"]) == frozenset()
    assert index.lookup(["2019"]) == frozenset()
    assert index.lookup(["CAMP"]) == {"enz1"}


def test_one_form_can_name_several_entities() -> None:
    """A surface form is not owned by one entity; the index is a multimap."""
    index = surface_forms.build_index(
        {"enz1": ["nitrilase"], "enz2": ["nitrilase"]}
    )

    assert index.lookup(["nitrilase"]) == {"enz1", "enz2"}


def test_other_organism_names_come_from_the_documents(
    index: surface_forms.SurfaceFormIndex, forms: dict[str, list[str]]
) -> None:
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


def test_pooling_the_other_organism_names_leaves_them_as_written() -> None:
    """The abbreviation is added by a second call, and not every caller wants
    it: one resolving these names against an outside nomenclature needs the
    corpus's own spellings, since an abbreviation that nomenclature lists under
    another taxon can only cost the entity its identifier."""
    pooled = surface_forms.pooled_other_organism_names(
        [{"1": "Nocardia erythropolis"}, {"1": "Nocardia rhodochrous"}]
    )

    assert pooled == {"1": ["Nocardia erythropolis", "Nocardia rhodochrous"]}


_ISOMERASE = {
    "enzymes": {
        "1": {"recommended_name": "glucose isomerase", "ec_class": "5.3.2.1"}
    }
}
"""One enzyme whose EC number is also a plausible section number."""


def _isomerase_index() -> surface_forms.SurfaceFormIndex:
    return surface_forms.build_index(
        surface_forms.brenda_surface_forms(_ISOMERASE)
    )


def test_an_ec_number_carries_its_id_only_where_it_is_written_as_one() -> None:
    """The index is keyed by a form's words, so a bare `5.3.2.1` would enter
    it as `5 3 2 1` and every section number, corpus size and confidence
    interval of that shape would name an enzyme. Registering the number as the
    literature writes it is what separates the two, and it must not cost the
    written-out name."""
    index = _isomerase_index()

    assert index.lookup(["5", "3", "2", "1"]) == frozenset()
    assert index.lookup(["EC", "5", "3", "2", "1"]) == {"enz1"}
    assert index.lookup(["glucose", "isomerase"]) == {"enz1"}


def test_a_section_number_stays_a_trained_negative() -> None:
    """The document-level consequence, and the one that reaches the loss.

    A multi-word key survives any "ignore symbol-like matches" filter
    downstream, so a section number keyed as an EC number is indistinguishable
    from a match on a written-out name. The qualified number in the same
    sentence is a real mention and has to stay one, since an unmatched span is
    painted `OUTSIDE` rather than withheld.
    """
    index = _isomerase_index()
    text = "See section 5.3.2.1; the enzyme (EC 5.3.2.1) was assayed."
    section = text.index("5.3.2.1")
    written = text.index("EC 5.3.2.1")

    labels = token_labels.character_labels(
        len(text),
        token_labels.find_mentions(text, index),
        gold_entity_ids={"enz1"},
    )

    assert set(labels[section : section + len("5.3.2.1")]) == {
        token_labels.OUTSIDE
    }
    assert set(labels[written : written + len("EC 5.3.2.1")]) == {
        token_labels.BRENDA_LABELS.code_of("enz1")
    }


def test_the_older_dotted_ec_spelling_reaches_its_enzyme() -> None:
    """`E.C. 5.3.2.1` keys as `E C 5 3 2 1`, which ordinary text produces no
    more than it does `EC 5 3 2 1`, so the older style can carry the ID while
    the bare number beside it still names nothing."""
    index = _isomerase_index()
    text = "See section 5.3.2.1; the enzyme (E.C. 5.3.2.1) was assayed."

    mentions = token_labels.find_mentions(text, index)

    assert [
        (text[mention.start : mention.end], mention.entity_ids)
        for mention in mentions
    ] == [("E.C. 5.3.2.1", {"enz1"})]


@pytest.mark.parametrize(
    ("text", "entity_id", "other_form"),
    [
        (
            "pp. 3577-3580 were consulted",
            "str15133",
            ["NRRL", "B", "3577"],
        ),
        (
            "lot 9005-74 was discarded",
            "str15138",
            ["CDC", "9005", "74"],
        ),
    ],
)
def test_a_bare_strain_designation_is_not_read_off_running_text(
    tables: dict[str, dict[str, Any]],
    text: str,
    entity_id: str,
    other_form: list[str],
) -> None:
    """A page-range fragment and a lot number must not train the entity head.

    `str15133` is designated the bare `3577` (also a page-range fragment) and
    `str15138` the bare `9005-74` (also a lot number); neither carries an
    `EC`-style qualifier to separate the real designation from the digits a
    document also happens to spell that way. Both strains keep other,
    letter-bearing designations reachable — `NRRL B-3577` and `CDC 9005-74`
    among them — so the guard costs a form, not the entity.
    """
    index = surface_forms.build_index(
        surface_forms.brenda_surface_forms({"strains": tables["strains"]})
    )

    mentions = token_labels.find_mentions(text, index)

    assert not any(entity_id in mention.entity_ids for mention in mentions)
    assert entity_id in index.lookup(other_form)


def test_strain_forms_leave_out_the_taxon_name(
    tables: dict[str, dict[str, Any]],
) -> None:
    """A strain's `taxon` names the species, not the strain.

    Indexing it would attach strain IDs to bacterium mentions.
    """
    extracted = surface_forms.strain_forms(
        tables["strains"], tables["bacteria"]
    )

    assert "ATCC 201872" in extracted["289"]
    assert "Schizosaccharomyces pombe" not in extracted["289"]


def test_strain_forms_requires_the_bacteria_table() -> None:
    """`typhimurium` is an epithet only a bacterium's synonym names, so a call
    leaving the table out would keep it a strain key, silently."""
    with pytest.raises(TypeError):
        surface_forms.strain_forms(_anonymous_strains("typhimurium"))


# How many taxonless, depositless records the shipped dump files each
# descriptor under. Spelled out rather than derived from
# `surface_forms.DESCRIPTOR_MIN_RECORDS`, for the reason `_CATEGORY_NOUNS` is.
_DESCRIPTORS = {
    "CuZn-SOD": 21,
    "Mn-SOD": 19,
    "Fe-SOD": 13,
    "DsbA homologous": 13,
    "type S": 13,
}


def _anonymous_strains(*designations: str) -> dict[str, dict[str, Any]]:
    """One strain record per designation, with neither taxon nor deposit."""
    return {
        str(n): {"taxon": None, "cultures": [], "designations": [designation]}
        for n, designation in enumerate(designations)
    }


def _strain_index(
    table: dict[str, dict[str, Any]],
    bacteria: dict[str, dict[str, Any]] | None = None,
) -> surface_forms.SurfaceFormIndex:
    return surface_forms.build_index(
        surface_forms.brenda_surface_forms(
            {"strains": table, "bacteria": bacteria or {}}
        )
    )


@pytest.mark.parametrize("designation", list(_DESCRIPTORS))
def test_a_descriptor_many_anonymous_strains_share_carries_no_id(
    designation: str,
) -> None:
    """A protein name or phenotype in BRENDA's strain field is filed as one
    anonymous record per organism, so its key reaches a score of strains, none
    of which a text could mean by it."""
    index = _strain_index(
        _anonymous_strains(*[designation] * _DESCRIPTORS[designation])
    )

    assert index.lookup(surface_forms.form_words(designation)) == frozenset()
    assert index.entity_ids == frozenset()


def test_a_descriptor_does_not_swallow_the_species_after_it() -> None:
    """The sweep splits `wild-type` at the hyphen and takes the longest match
    first, so a `type S` key consumed the `S.` and left `pyogenes` a trained
    negative on a bacterium name."""
    index = _strain_index(
        _anonymous_strains(*["type S"] * _DESCRIPTORS["type S"]),
        {"1": {"organism": "Streptococcus pyogenes", "synonyms": []}},
    )
    text = "wild-type S. pyogenes"
    species = text.index("S. pyogenes")

    mentions = token_labels.find_mentions(text, index)

    assert mentions == [
        token_labels.Mention(
            start=species,
            end=species + len("S. pyogenes"),
            entity_ids=frozenset({"bac1"}),
        )
    ]


@pytest.mark.parametrize(
    "designations",
    [
        ["IL1403"] * 4,
        ["BL21-(DE3)", "BL21(DE3)", "BL21 (DE3)", "BL21-DE3"],
    ],
    ids=["IL1403", "BL21(DE3)"],
)
def test_a_real_strain_four_anonymous_records_share_keeps_its_id(
    designations: list[str],
) -> None:
    """The dump's largest anonymous groups of a real strain, both of them
    gold-linked in the splits: the bar has to sit above them. `BL21(DE3)` is
    four spellings of one key, which is what the index counts."""
    index = _strain_index(_anonymous_strains(*designations))

    assert index.lookup(surface_forms.form_words(designations[0])) == {
        f"str{n}" for n in range(4)
    }


def test_a_shared_designation_with_a_deposit_keeps_its_id() -> None:
    """`Marburg` is seventeen records across three species, each with a
    culture number, so it names real strains however many share it."""
    table = {
        str(n): {
            "taxon": "Methanothermobacter marburgensis",
            "cultures": [{"strain_number": f"DSM {2133 + n}"}],
            "designations": ["Marburg"],
        }
        for n in range(17)
    }

    index = _strain_index(table)

    assert index.lookup(["Marburg"]) == {f"str{n}" for n in range(17)}


def _label_rows(
    text: str, index: surface_forms.SurfaceFormIndex, gold: frozenset[str]
) -> list[tuple[str, int, int]]:
    """Each `mention_spans` row as the text it covers, its code and flag."""
    spans = token_labels.mention_spans(
        token_labels.find_mentions(text, index), gold
    )
    return [
        (text[start:end], code, is_gold)
        for start, end, code, is_gold in spans.tolist()
    ]


def test_a_bare_species_epithet_is_a_trained_negative() -> None:
    """str16702's one designation is `typhimurium`, the epithet of the synonym
    `Salmonella typhimurium`, so with no taxon or deposit a bare "Typhimurium"
    was a strain mention. The longer forms ending on the word must keep it."""
    index = _strain_index(
        _anonymous_strains("typhimurium", "serovar Typhimurium"),
        {
            "1": {
                "organism": "Salmonella enterica",
                "synonyms": ["Salmonella typhimurium"],
            }
        },
    )
    text = "S. Typhimurium, serovar Typhimurium: Typhimurium grew."
    gold = frozenset({"bac1", "str1"})
    bare = text.rindex("Typhimurium")
    space = token_labels.BRENDA_LABELS

    labels = token_labels.character_labels(
        len(text), token_labels.find_mentions(text, index), gold
    )

    assert _label_rows(text, index, gold) == [
        ("S. Typhimurium", space.code_of("bac1"), 1),
        ("serovar Typhimurium", space.code_of("str1"), 1),
    ]
    assert set(labels[bare : bare + len("Typhimurium")].tolist()) == {
        token_labels.OUTSIDE
    }


@pytest.mark.parametrize(
    ("designation", "bacteria", "named"),
    [
        (
            "indica",
            {"1": {"organism": "Pseudomonas indica", "synonyms": []}},
            {},
        ),
        (
            "Japonica",
            {"1": {"organism": "Shewanella japonica", "synonyms": []}},
            {},
        ),
        (
            "mrakii",
            {},
            {
                "9": {
                    "taxon": {"name": "Cyberlindnera mrakii"},
                    "cultures": [],
                    "designations": ["Kodama 169"],
                }
            },
        ),
    ],
    ids=["bacterium", "capitalized", "strain-taxon"],
)
def test_an_epithet_designation_is_painted_outside(
    designation: str,
    bacteria: dict[str, dict[str, Any]],
    named: dict[str, dict[str, Any]],
) -> None:
    """The epithet may come from a bacterium's name or another strain's taxon,
    and matches case-folded: BRENDA capitalizes `Japonica` as it would a
    cultivar group."""
    index = _strain_index(
        {**_anonymous_strains(designation), **named}, bacteria
    )
    text = f"Seeds of the {designation} group were sown."
    gold = frozenset({"str0"})

    labels = token_labels.character_labels(
        len(text), token_labels.find_mentions(text, index), gold
    )

    assert _label_rows(text, index, gold) == []
    assert set(labels.tolist()) == {token_labels.OUTSIDE}


@pytest.mark.parametrize("designation", ["gantai", "azul"])
def test_a_lowercase_designation_no_binomial_names_keeps_its_id(
    designation: str,
) -> None:
    """BRENDA writes cultivar names lowercase, as it does epithets, and
    `gantai` and `azul` are gold-linked in training: the epithet has to come
    from a binomial, not from the word's shape."""
    index = _strain_index(
        _anonymous_strains(designation),
        {"1": {"organism": "Pseudomonas indica", "synonyms": []}},
    )
    text = f"seeds of cv. {designation} were sown"
    gold = frozenset({"str0"})

    assert _label_rows(text, index, gold) == [
        (designation, token_labels.BRENDA_LABELS.code_of("str0"), 1)
    ]


def test_an_epithet_leaves_a_record_with_a_taxon_as_well() -> None:
    """str11536, formerly *Synechocystis aquatilis*, keeps `aquatilis` beside
    the bare-genus taxon `Cyanobacterium` and a deposit, and every bare hit of
    it in the splits is another species' epithet, as in this training
    sentence: the taxon and the deposit must not spare the word."""
    index = _strain_index(
        {
            "11536": {
                "taxon": {"name": "Cyanobacterium"},
                "cultures": [{"strain_number": "NBRC 102756"}],
                "designations": ["MBIC10216", "aquatilis"],
            }
        },
        {"1": {"organism": "Rahnella aquatilis", "synonyms": []}},
    )
    text = (
        "To determine whether ORF427 was translated, AroAR. aquatilis was "
        "expressed and purified."
    )

    assert _label_rows(text, index, frozenset({"bac1"})) == []
    assert index.lookup(["MBIC10216"]) == {"str11536"}
    assert index.lookup(["NBRC", "102756"]) == {"str11536"}


def test_an_epithet_leaves_a_strain_truly_named_like_one() -> None:
    """The cost of reading the word rather than the record: `Album` names
    str14092, a *Pseudarthrobacter oxydans* strain, and equals the epithet of
    `Methylomicrobium album`. The splits never write it bare for the strain,
    and its longer designation and deposits keep it reachable."""
    extracted = surface_forms.strain_forms(
        {
            "14092": {
                "taxon": {"name": "Pseudarthrobacter oxydans"},
                "cultures": [{"strain_number": "DSM 20120"}],
                "designations": ["Album", "Album ATCC14359"],
            }
        },
        {"1": {"organism": "Methylomicrobium album", "synonyms": []}},
    )

    assert extracted["14092"] == ["Album ATCC14359", "DSM 20120"]


def test_every_indexed_id_wears_a_prefix_the_corpus_schema_declares(
    forms: dict[str, list[str]],
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
def test_abbreviated_genus_declines_non_binomials(form: str) -> None:
    """A form that does not open with a binomial gets no abbreviation.

    `DSM 20745` mangled to `D. 20745` would be a phantom surface form
    attached to a real strain ID — a silent mislabel, not a recall gain.
    """
    assert surface_forms.abbreviated_genus(form) is None


@pytest.mark.parametrize(
    "form",
    [
        "Agaricus sp.",
        "Agaricus sp",
        "Bacillus spp.",
        "Bacillus spp",
        "Firmicutes bacterium",
    ],
)
def test_abbreviated_genus_declines_a_bare_placeholder(form: str) -> None:
    """A bare placeholder is named by its genus alone, so it keeps the genus.

    `A. sp.` would be one key for every unnamed species of an `A` genus.
    """
    assert surface_forms.abbreviated_genus(form) is None


@pytest.mark.parametrize(
    ("form", "abbreviated"),
    [
        ("Pseudomonas sp. P51", "P. sp. P51"),
        ("Paracoccus sp. N81106", "P. sp. N81106"),
        ("Acinetobacter sp. NIPH 1859", "A. sp. NIPH 1859"),
        ("Firmicutes bacterium TAB5", "F. bacterium TAB5"),
    ],
)
def test_abbreviated_genus_keeps_a_placeholder_s_designation(
    form: str, abbreviated: str
) -> None:
    """The designation identifies the organism, and text abbreviates it.

    `Paracoccus sp. PC1, P. sp. N81106` is how a list of strains reads.
    """
    assert surface_forms.abbreviated_genus(form) == abbreviated


@pytest.mark.parametrize(
    ("form", "abbreviated"),
    [
        ("Bacillus sphaericus", "B. sphaericus"),
        ("Clostridium sporogenes", "C. sporogenes"),
        ("Trichinella spiralis", "T. spiralis"),
    ],
)
def test_abbreviated_genus_keeps_an_epithet_opening_like_a_placeholder(
    form: str, abbreviated: str
) -> None:
    """The placeholder guard matches whole words, not the prefix `sp`."""
    assert surface_forms.abbreviated_genus(form) == abbreviated


def test_bacteria_forms_carry_the_abbreviated_variant() -> None:
    """37% synonym coverage, median 0: the abbreviation must be generated."""
    extracted = surface_forms.bacteria_forms(
        {"42": {"organism": "Bacillus subtilis", "synonyms": []}}
    )

    assert "B. subtilis" in extracted["42"]
    assert "Bacillus subtilis" in extracted["42"]


def test_a_genus_synonym_names_the_genus_and_none_of_its_species(
    index: surface_forms.SurfaceFormIndex,
    tables: dict[str, dict[str, Any]],
) -> None:
    """The dump hands `Pseudomonas sp. GM41` the genus's synonyms verbatim.

    Indexed as they stand, `Zestomonas` names every record under the genus, so
    a mention of the genus anchors whichever of them a document holds as gold.
    """
    assert "Zestomonas" in tables["bacteria"]["20021"]["synonyms"]

    assert index.lookup(["Zestomonas"]) == {"bac5085"}


def test_a_species_record_keeps_its_multiword_synonyms() -> None:
    """Only the one-word synonym goes; a binomial still names the species."""
    extracted = surface_forms.bacteria_forms(
        {
            "1": {
                "organism": "Pseudomonas sp. P51",
                "synonyms": ["Zestomonas", "Pseudomonas putida"],
            }
        }
    )

    assert "Zestomonas" not in extracted["1"]
    assert "Pseudomonas putida" in extracted["1"]
    assert "P. putida" in extracted["1"]


@pytest.mark.parametrize("organism", ["Pseudomonas", ""])
def test_a_record_not_named_below_the_genus_keeps_a_one_word_synonym(
    organism: str,
) -> None:
    """A genus's synonyms are its names. A record with no name of its own has
    nothing to judge a synonym by, and dropping it there would cost the entity
    its last form rather than one form."""
    extracted = surface_forms.bacteria_forms(
        {"1": {"organism": organism, "synonyms": ["Zestomonas"]}}
    )

    assert "Zestomonas" in extracted["1"]


def test_strain_designations_carry_the_abbreviated_variant() -> None:
    """A designation opening with the binomial abbreviates; numbers do not."""
    extracted = surface_forms.strain_forms(
        {
            "7": {
                "designations": ["Escherichia coli K-12", "DSM 20745"],
                "cultures": [],
            }
        },
        {},
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
        {"1": {"designations": names, "cultures": []}}, {}
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
    extract: Callable[[list[str]], list[str]],
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
    extract: Callable[[list[str]], list[str]], name: str
) -> None:
    """The binomial guard has to hold wherever the expansion is applied.

    `D. 20745` would be a phantom form attached to a real ID, so widening the
    expansion to a third extractor must not widen what it mangles.
    """
    assert extract([name]) == [name]


@_NAME_BEARING
@pytest.mark.parametrize(
    "name", ["Agaricus sp.", "Bacillus spp.", "Firmicutes bacterium"]
)
def test_no_name_bearing_extractor_abbreviates_a_bare_placeholder(
    extract: Callable[[list[str]], list[str]], name: str
) -> None:
    """All three populations hold unnamed-species placeholders."""
    assert extract([name]) == [name]


def test_fuzzy_ids_finds_an_inflectional_variant() -> None:
    """`oxidases` is one edit from the registered `oxidase`."""
    index = surface_forms.build_index({"enz1": ["oxidase"]})

    assert index.fuzzy_ids("oxidases") == {"enz1"}


def test_fuzzy_ids_is_empty_for_an_unrelated_word() -> None:
    """A word nothing registered resembles gets no ID, not the nearest one."""
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


def test_fuzzy_ids_declines_a_word_carrying_no_letter() -> None:
    """`fuzz.ratio` cannot tell two numbers apart the way it tells two words.

    Digits are interchangeable under character overlap, so `10000` reaches the
    cutoff of the registered `10008` at exactly 80.0. Reading a thousands
    separator into the number is what carries such a word over the length
    floor — `10,000` used to split into `10` and `000` — and a hit there
    spends a centrifugation speed's negative signal on an abstention.
    """
    index = surface_forms.build_index({"str1": ["10008"]})

    assert index.fuzzy_ids("10000") == frozenset()


def test_fuzzy_ids_still_reads_an_alphanumeric_accession() -> None:
    """The guard is letterlessness, not the presence of digits.

    An unspaced deposit number is exactly what the fuzzy layer is asked of
    once the exact index misses it, so a designation carrying letters must
    keep reaching one.
    """
    index = surface_forms.build_index({"str2": ["NCIMB 8826"]})

    assert index.fuzzy_ids("NCIMB8827") == {"str2"}


def test_a_centrifugation_speed_stays_a_trained_negative() -> None:
    """The document-level consequence: `10,000` is labelled, not abstained on.

    A fuzzy mention is forced to `IGNORE_INDEX`, so a near-hit on the number
    withdraws those characters from the loss entirely rather than mislabelling
    them.
    """
    index = surface_forms.build_index({"str1": ["10008"]})
    text = "The lysate was centrifuged at 10,000 x g for 10 min."
    speed = text.index("10,000")

    labels = token_labels.character_labels(
        len(text),
        token_labels.find_mentions(text, index),
        gold_entity_ids=frozenset(),
    )

    assert set(labels[speed : speed + len("10,000")]) == {token_labels.OUTSIDE}


@pytest.mark.parametrize(
    ("text", "quantity", "key"),
    [
        ("Cells were pelleted at 3,000g for 10 min.", "3,000g", "3000"),
        ("spun at maximum speed (21,100x g) at 4 C", "21,100x", "210x"),
        ("a 128bp fragment of the promoter", "128bp", "1278b"),
        ("eluted at 22min in mobile phase B", "22min", "22Lin"),
    ],
)
def test_a_quantity_stays_a_trained_negative(
    text: str, quantity: str, key: str
) -> None:
    """A unit on a number does not make it a variant of a designation.

    Each carries a letter past the guard refusing a bare number, and scores at
    least 80 against `key` on its digits alone. Read off the label rows, since
    it is the sweep and not `fuzzy_ids` that sees the word's context.
    """
    index = surface_forms.build_index({"str1": [key]})
    at = text.index(quantity)

    spans = token_labels.mention_spans(
        token_labels.find_mentions(text, index), frozenset()
    )
    labels = token_labels.character_labels_from_spans(len(text), spans)

    assert spans.tolist() == []
    assert set(labels[at : at + len(quantity)]) == {token_labels.OUTSIDE}


@pytest.mark.parametrize(
    ("text", "word", "key"),
    [
        ("the type strain DSM 20074T was used", "20074T", "20074"),
        ("the type strain DSM 22,228T was used", "22,228T", "22228"),
        ("the type strain NRRL B-14,911T was used", "14,911T", "14911"),
        ("integrated into the 10403s chromosome", "10403s", "10403S"),
    ],
)
def test_a_designation_shaped_like_a_quantity_keeps_its_abstention(
    text: str, word: str, key: str
) -> None:
    """`20074T` has the shape of `3000g`; `10403s` is a strain in lowercase.

    Neither is a quantity: a deposit number is known by its acronym, separator
    or not, and one letter after a number suffixes a designation rather than
    naming a unit. Dropping either abstention trains a strain as a negative.
    """
    index = surface_forms.build_index({"str1": [key]})
    at = text.index(word)

    spans = token_labels.mention_spans(
        token_labels.find_mentions(text, index), frozenset()
    )

    assert spans.tolist() == [
        [at, at + len(word), token_labels.BRENDA_LABELS.code_of("str1"), 0]
    ]


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
    """Text says `E. coli` where the table says the binomial."""
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


def test_bare_placeholders_sharing_an_initial_share_no_key() -> None:
    """Unnamed species of two unrelated genera must not meet in the index.

    Abbreviated, both would reach `A sp`, the only key they had in common.
    """
    index = surface_forms.build_index(
        surface_forms.brenda_surface_forms(
            {
                "bacteria": {
                    "1": {"organism": "Aneurinibacillus sp.", "synonyms": []}
                }
            },
            [{"2": "Agaricus sp."}],
        )
    )

    assert not index.lookup(["A", "sp"])
    assert not [
        key for key in (*index.exact, *index.folded) if len(key.split()[0]) == 1
    ]
    assert index.lookup(["Aneurinibacillus", "sp"]) == {"bac1"}
    assert index.lookup(["Agaricus", "sp"]) == {"oth2"}


def test_a_placeholder_s_designation_is_reachable_abbreviated() -> None:
    """Text lists `Paracoccus sp. PC1, P. sp. N81106`; the second must match."""
    index = surface_forms.build_index(
        surface_forms.brenda_surface_forms(
            {
                "bacteria": {
                    "3": {"organism": "Paracoccus sp. N81106", "synonyms": []}
                }
            }
        )
    )

    assert index.lookup(["P", "sp", "N81106"]) == {"bac3"}


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


def test_fuzzy_buckets_are_independent_of_entity_insertion_order() -> None:
    """A tied fuzzy match must not flip with the order entities were pooled.

    Set equality of the buckets would pass either way; `process.extractOne`
    breaks a tied score by position, so only comparing the bucket tuples (and
    the tie-break they produce) can catch bucket order tracking insertion
    order instead of the tables.
    """
    forwards = surface_forms.build_index(
        {"enz1": ["zqxvbn"], "enz2": ["zqxvbm"]}
    )
    backwards = surface_forms.build_index(
        {"enz2": ["zqxvbm"], "enz1": ["zqxvbn"]}
    )

    assert (
        forwards.folded_singles_by_first_letter
        == backwards.folded_singles_by_first_letter
    )
    assert forwards.fuzzy_ids("zqxvbz") == backwards.fuzzy_ids("zqxvbz")


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
