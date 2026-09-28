"""The nomenclature that supplies enzymeNER's gold, and what it refuses to.

A span is gold when Expasy gives its surface form exactly one EC number.
Nothing here consults BRENDA's surface forms: choosing the subset from the
linker's own index would make its answer the gold.
"""

import pathlib

import pytest
from d3text.datasets import expasy
from d3text.identifier_bridge import (
    EC_NUMBER,
    BridgeRow,
    ExternalMention,
    IdentifierBridge,
)
from d3text.linking import DictionaryLinker
from d3text.linking_eval import score_linking
from d3text.surface_forms import build_index

# Records as Expasy writes them: a wrapped name continues on the next line of
# the same tag and ends at the full stop, and a hyphen at the wrap point is
# not a word break.
NOMENCLATURE = "\n".join(
    (
        "CC   Release of 10-Jun-2026",
        "//",
        "ID   1.1.1.1",
        "DE   alcohol dehydrogenase.",
        "AN   aldehyde reductase.",
        "//",
        "ID   1.13.11.25",
        "DE   3,4-dihydroxy-9,10-secoandrosta-1,3,5(10)-triene-9,17-dione 4,5-",
        "DE   dioxygenase.",
        "AN   steroid 4,5-dioxygenase.",
        "//",
        "ID   1.1.1.170",
        "DE   3beta-hydroxysteroid-4alpha-carboxylate 3-dehydrogenase",
        "DE   (decarboxylating).",
        "//",
        "ID   1.1.1.5",
        "DE   Transferred entry: 1.1.1.303 and 1.1.1.304.",
        "//",
        "ID   1.1.1.74",
        "DE   Deleted entry.",
        "//",
        "ID   1.13.12.5",
        "DE   Renilla-luciferin 2-monooxygenase.",
        "AN   luciferase.",
        "//",
        "ID   1.13.12.7",
        "DE   Photinus-luciferin 4-monooxygenase (ATP-hydrolysing).",
        "AN   luciferase.",
        "//",
        "ID   3.2.1.23",
        "DE   beta-galactosidase.",
        "//",
        "ID   1.1.1.100",
        "DE   3-oxoacyl-[acyl-carrier-protein] reductase.",
        "//",
        "ID   1.1.1.212",
        "DE   3 oxoacyl [acyl carrier protein] reductase.",
        "//",
    )
)

ADH = "alcohol dehydrogenase"


@pytest.fixture
def nomenclature(tmp_path: pathlib.Path) -> expasy.EnzymeNomenclature:
    path = tmp_path / expasy.NOMENCLATURE
    path.write_text(NOMENCLATURE + "\n", encoding=expasy.ENCODING)
    return expasy.load_nomenclature(path)


def _mention(surface: str, start: int = 0) -> ExternalMention:
    return ExternalMention(
        document="PMC1:S1",
        start=start,
        end=start + len(surface),
        surface=surface,
        external_id=None,
    )


# --------------------------------------------------------------------------- #
# The subset is the nomenclature's, and it is not BRENDA's                     #
# --------------------------------------------------------------------------- #
def test_only_a_name_with_one_ec_number_becomes_gold(
    nomenclature: expasy.EnzymeNomenclature,
) -> None:
    """The licence for scoring exact-match top-1. `luciferase` is four
    reactions sharing a word, so no answer to it can be graded."""
    assert nomenclature.sole_ec(ADH) == "1.1.1.1"
    assert nomenclature.ec_numbers("luciferase") == {"1.13.12.5", "1.13.12.7"}
    assert nomenclature.sole_ec("luciferase") is None
    assert nomenclature.sole_ec("Taq polymerase") is None


def test_the_judged_subset_moves_with_the_nomenclature_alone(
    nomenclature: expasy.EnzymeNomenclature,
) -> None:
    """Every EC number names exactly one BRENDA enzyme, so the bridge filters
    nothing and this is the *only* filter there is. It reads no BRENDA
    surface form, which is what keeps the spans it keeps from being the spans
    the linker happens to answer with one candidate."""
    spans = [_mention(ADH), _mention("luciferase", 40), _mention("kinase", 60)]

    stamped = nomenclature.assign(spans)

    assert [(mention.surface, mention.external_id) for mention in stamped] == [
        (ADH, "1.1.1.1"),
        ("luciferase", "1.13.12.5"),
        ("luciferase", "1.13.12.7"),
        ("kinase", None),
    ]


def test_an_unheld_name_keeps_the_span_in_the_denominator(
    nomenclature: expasy.EnzymeNomenclature,
) -> None:
    """A span the nomenclature cannot name is not a span that did not happen:
    dropping it would report the coverage as a share of the names Expasy
    knows, which is a denominator nobody asked about."""
    stamped = nomenclature.assign([_mention("Taq polymerase")])

    assert len(stamped) == 1
    assert stamped[0].external_id is None


# --------------------------------------------------------------------------- #
# Normalization, and the ambiguity it must not hide                            #
# --------------------------------------------------------------------------- #
def test_the_corpus_spelling_reaches_the_nomenclature_spelling(
    nomenclature: expasy.EnzymeNomenclature,
) -> None:
    """Expasy spells Greek letters out in Latin and the corpus writes them as
    letters, so without the folding these two are different enzymes."""
    assert (
        nomenclature.sole_ec("\N{GREEK SMALL LETTER BETA}-Galactosidase")
        == "3.2.1.23"
    )
    assert expasy.normalize("HMG\N{HYPHEN}CoA") == "hmg coa"


_GREEK_SPELLING = {
    "\N{GREEK SMALL LETTER ALPHA}": "alpha",
    "\N{GREEK CAPITAL LETTER ALPHA}": "alpha",
    "\N{GREEK SMALL LETTER BETA}": "beta",
    "\N{GREEK CAPITAL LETTER BETA}": "beta",
    "\N{GREEK SMALL LETTER GAMMA}": "gamma",
    "\N{GREEK CAPITAL LETTER GAMMA}": "gamma",
    "\N{GREEK SMALL LETTER DELTA}": "delta",
    "\N{GREEK CAPITAL LETTER DELTA}": "delta",
    "\N{GREEK SMALL LETTER EPSILON}": "epsilon",
    "\N{GREEK CAPITAL LETTER EPSILON}": "epsilon",
    "\N{GREEK SMALL LETTER ZETA}": "zeta",
    "\N{GREEK CAPITAL LETTER ZETA}": "zeta",
    "\N{GREEK SMALL LETTER ETA}": "eta",
    "\N{GREEK CAPITAL LETTER ETA}": "eta",
    "\N{GREEK SMALL LETTER THETA}": "theta",
    "\N{GREEK CAPITAL LETTER THETA}": "theta",
    "\N{GREEK SMALL LETTER IOTA}": "iota",
    "\N{GREEK CAPITAL LETTER IOTA}": "iota",
    "\N{GREEK SMALL LETTER KAPPA}": "kappa",
    "\N{GREEK CAPITAL LETTER KAPPA}": "kappa",
    "\N{GREEK SMALL LETTER LAMDA}": "lambda",
    "\N{GREEK CAPITAL LETTER LAMDA}": "lambda",
    "\N{GREEK SMALL LETTER MU}": "mu",
    "\N{GREEK CAPITAL LETTER MU}": "mu",
    "\N{GREEK SMALL LETTER NU}": "nu",
    "\N{GREEK CAPITAL LETTER NU}": "nu",
    "\N{GREEK SMALL LETTER XI}": "xi",
    "\N{GREEK CAPITAL LETTER XI}": "xi",
    "\N{GREEK SMALL LETTER OMICRON}": "omicron",
    "\N{GREEK CAPITAL LETTER OMICRON}": "omicron",
    "\N{GREEK SMALL LETTER PI}": "pi",
    "\N{GREEK CAPITAL LETTER PI}": "pi",
    "\N{GREEK SMALL LETTER RHO}": "rho",
    "\N{GREEK CAPITAL LETTER RHO}": "rho",
    "\N{GREEK SMALL LETTER SIGMA}": "sigma",
    "\N{GREEK CAPITAL LETTER SIGMA}": "sigma",
    "\N{GREEK SMALL LETTER FINAL SIGMA}": "sigma",
    "\N{GREEK SMALL LETTER TAU}": "tau",
    "\N{GREEK CAPITAL LETTER TAU}": "tau",
    "\N{GREEK SMALL LETTER UPSILON}": "upsilon",
    "\N{GREEK CAPITAL LETTER UPSILON}": "upsilon",
    "\N{GREEK SMALL LETTER PHI}": "phi",
    "\N{GREEK CAPITAL LETTER PHI}": "phi",
    "\N{GREEK SMALL LETTER CHI}": "chi",
    "\N{GREEK CAPITAL LETTER CHI}": "chi",
    "\N{GREEK SMALL LETTER PSI}": "psi",
    "\N{GREEK CAPITAL LETTER PSI}": "psi",
    "\N{GREEK SMALL LETTER OMEGA}": "omega",
    "\N{GREEK CAPITAL LETTER OMEGA}": "omega",
}


def test_every_greek_letter_folds_to_its_latin_spelling() -> None:
    """Each of the 24 letters, in both cases, and final sigma folds to its
    Latin spelling."""
    for letter, spelling in _GREEK_SPELLING.items():
        assert expasy.normalize(f"{letter}-test") == f"{spelling} test"


def test_an_accented_greek_letter_folds_to_the_letter_not_the_accent() -> None:
    """`unicodedata.name` of an accented letter ends in the diacritic's
    name, not the letter's, so naming the character directly collides
    distinct letters onto their diacritic ('tonos', 'psili', ...); the
    letter must be recovered from the accented character's base form."""
    assert (
        expasy.normalize("\N{GREEK SMALL LETTER ALPHA WITH TONOS}-test")
        == "alpha test"
    )
    assert (
        expasy.normalize("\N{GREEK CAPITAL LETTER ALPHA WITH PSILI}-test")
        == "alpha test"
    )


def test_sharp_s_still_folds_to_beta() -> None:
    """ß is Latin, not Greek, so it falls outside any Unicode-name-based
    Greek folding; the deliberate override must survive independently
    of how the Greek letters themselves are folded."""
    assert (
        expasy.normalize("\N{LATIN SMALL LETTER SHARP S}-test") == "beta test"
    )


def test_a_name_normalization_makes_ambiguous_stays_ambiguous(
    nomenclature: expasy.EnzymeNomenclature,
) -> None:
    """Folding hyphens to spaces is what buys the coverage, and it can make
    two names collide that the file spells apart. The collision has to
    surface as ambiguity — resolving it to whichever record was read last
    would be a gold chosen by file order."""
    key = expasy.normalize("3-oxoacyl-[acyl-carrier-protein] reductase")

    assert key == expasy.normalize("3 oxoacyl [acyl carrier protein] reductase")
    assert nomenclature.by_name[key] == {"1.1.1.100", "1.1.1.212"}
    assert (
        nomenclature.sole_ec("3-oxoacyl-[acyl-carrier-protein] reductase")
        is None
    )


# --------------------------------------------------------------------------- #
# The flat file's own shape                                                    #
# --------------------------------------------------------------------------- #
def test_a_wrapped_name_is_one_name_not_two(
    nomenclature: expasy.EnzymeNomenclature,
) -> None:
    """Read line by line, a wrapped name is two names the file never holds and
    the enzyme's real name is missing — a silent hole in the gold."""
    assert (
        nomenclature.sole_ec(
            "3,4-dihydroxy-9,10-secoandrosta-1,3,5(10)-triene-9,17-dione "
            "4,5-dioxygenase"
        )
        == "1.13.11.25"
    )
    assert (
        nomenclature.sole_ec(
            "3beta-hydroxysteroid-4alpha-carboxylate 3-dehydrogenase "
            "(decarboxylating)"
        )
        == "1.1.1.170"
    )
    assert nomenclature.sole_ec("dioxygenase") is None


def test_a_records_status_is_not_one_of_its_names(
    nomenclature: expasy.EnzymeNomenclature,
) -> None:
    """`Deleted entry` and `Transferred entry` sit in the field names live in,
    so a parser that trusts the field indexes two enzymes nobody named."""
    assert nomenclature.ec_numbers("Deleted entry") == frozenset()
    assert (
        nomenclature.ec_numbers("Transferred entry: 1.1.1.303") == frozenset()
    )
    assert "1.1.1.5" not in {
        ec for found in nomenclature.by_name.values() for ec in found
    }


def test_alternate_names_are_indexed_beside_the_official_one(
    nomenclature: expasy.EnzymeNomenclature,
) -> None:
    assert nomenclature.sole_ec("aldehyde reductase") == "1.1.1.1"
    assert nomenclature.unambiguous == len(nomenclature) - 2


# --------------------------------------------------------------------------- #
# The whole chain, on fixtures                                                 #
# --------------------------------------------------------------------------- #
def test_gold_from_the_nomenclature_can_contradict_the_dictionary(
    nomenclature: expasy.EnzymeNomenclature,
) -> None:
    """The end-to-end anti-circularity property for enzymes. BRENDA's index
    says this name is `enz1`; Expasy's EC number belongs to `enz2`. A gold
    re-derived from the index would return the linker's own answer and score
    1.000, and the four spans would not partition the way they do."""
    bridge = IdentifierBridge.from_rows(
        EC_NUMBER,
        [
            BridgeRow("enz2", "1.1.1.1", "ec_class"),
            BridgeRow("enz3", "3.2.1.23", "ec_class"),
        ],
    )
    linker = DictionaryLinker(
        build_index({"enz1": [ADH], "enz3": ["beta-galactosidase"]})
    )
    spans = [
        _mention(ADH),
        _mention("beta-galactosidase", 40),
        _mention("luciferase", 80),
        _mention("Taq polymerase", 100),
    ]

    report = score_linking(
        mentions=nomenclature.assign(spans),
        bridge=bridge,
        linker=linker,
        entity_types=["enzymes"],
        namespace=EC_NUMBER,
    )

    assert (report.annotated, report.judged) == (4, 2)
    assert (report.outside_bridge, report.ambiguous_gold) == (1, 1)
    assert (report.strict.correct, report.strict.wrong) == (1, 1)


@pytest.mark.integration
def test_the_published_nomenclature_parses_into_names_not_statuses() -> None:
    """The same shape over the real 8,456 records, where the wrapped names and
    the status records are the two things a naive parser silently gets wrong."""
    path = (
        pathlib.Path.home()
        / "Downloads"
        / "expasy-enzyme"
        / expasy.NOMENCLATURE
    )
    if not path.exists():
        pytest.skip(f"no ENZYME nomenclature at {path}")

    loaded = expasy.load_nomenclature(path)

    assert len(loaded) == 16746
    assert loaded.unambiguous == 16305
    assert loaded.sole_ec("alcohol dehydrogenase") == "1.1.1.1"
    assert not [
        key for key in loaded.by_name if key.startswith("deleted entry")
    ]
