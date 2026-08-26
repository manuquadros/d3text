"""The linker contract: longest match, set-valued answers, NIL, nesting.

Everything runs off small in-memory indexes; the real `SurfaceFormIndex`
implementation is used throughout, so these pin the linker against the same
lookup machinery the labelling pipeline matches with.
"""

import pytest
from d3text import surface_forms
from d3text.linking import DictionaryLinker, Linker


def _linker(forms_by_entity: dict[str, list[str]]) -> DictionaryLinker:
    return DictionaryLinker(surface_forms.build_index(forms_by_entity))


def test_dictionary_linker_satisfies_the_protocol() -> None:
    assert isinstance(_linker({"enz1": ["catalase"]}), Linker)


def test_longest_contiguous_match_wins() -> None:
    """Over `Streptomyces griseocarneus` the species wins; the genus that a
    shorter window also matches is not emitted beside it."""
    linker = _linker(
        {
            "bac1": ["Streptomyces"],
            "bac2": ["Streptomyces griseocarneus"],
        }
    )

    assert linker.link("Streptomyces griseocarneus", "bacteria") == {"bac2"}


def test_shorter_match_survives_when_no_longer_one_exists() -> None:
    linker = _linker({"bac1": ["Streptomyces"]})

    assert linker.link("Streptomyces griseocarneus", "bacteria") == {"bac1"}


def test_ambiguous_form_emits_every_entity() -> None:
    """`AS-A` names four separate enzymes; the answer is the set, because
    only a consumer with context can narrow it."""
    linker = _linker(
        {
            "enz1": ["AS-A"],
            "enz2": ["AS-A"],
            "enz3": ["AS-A"],
            "enz4": ["AS-A"],
        }
    )

    assert linker.link("AS-A", "enzymes") == {"enz1", "enz2", "enz3", "enz4"}


def test_the_type_filters_which_wordlist_answers() -> None:
    """One span, two types, two answers — the nested-emission contract.

    A strain designation embedding the species binomial resolves to the
    strain when linked as a strain and to the nested species when linked as
    a bacterium, so both entities are reachable from the single tagged span.
    """
    linker = _linker(
        {
            "str7": ["Escherichia coli K-12"],
            "bac9": ["Escherichia coli"],
        }
    )

    assert linker.link("Escherichia coli K-12", "strains") == {"str7"}
    assert linker.link("Escherichia coli K-12", "bacteria") == {"bac9"}


def test_no_match_is_nil_not_an_error() -> None:
    linker = _linker({"enz1": ["catalase"]})

    assert linker.link("unheard-of protein", "enzymes") == frozenset()


def test_a_match_of_the_wrong_type_is_nil() -> None:
    """The type conditions the answer: an enzyme name linked as a bacterium
    resolves to nothing rather than to the enzyme."""
    linker = _linker({"enz1": ["catalase"]})

    assert linker.link("catalase", "bacteria") == frozenset()


def test_an_unknown_entity_type_raises() -> None:
    linker = _linker({"enz1": ["catalase"]})

    with pytest.raises(KeyError, match="viruses"):
        linker.link("catalase", "viruses")


def test_abbreviated_variants_link_back_to_the_binomial_entity() -> None:
    """The dictionary-gap closure, end to end: the table holds only the
    binomial, the text holds `E. coli`, and the link still resolves."""
    linker = DictionaryLinker(
        surface_forms.build_index(
            surface_forms.brenda_surface_forms(
                {
                    "bacteria": {
                        "9": {"organism": "Escherichia coli", "synonyms": []}
                    }
                }
            )
        )
    )

    assert linker.link("E. coli", "bacteria") == {"bac9"}
