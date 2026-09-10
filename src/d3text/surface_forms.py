"""BRENDA's surface forms, indexed by the entity IDs they name.

Not "what is this entity called" but "which entities could this string be",
which is what distant supervision needs. The index is keyed by the *words* of a
form rather than by the form itself, so no hyphenation convention has to be
modelled. Deliberately a leaf: building an index costs neither the BRENDA data
layer nor torch. See the surface-forms page of the documentation.
"""

import collections
import hashlib
import json
import os
import pathlib
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from rapidfuzz import fuzz, process
from wordfreq import zipf_frequency

from d3text.constraints import EntityId, FuzzyScore
from d3text.schema import BRENDA_SCHEMA

MIN_FORM_LENGTH = 4
"""Shortest form that may carry an ID.

Three characters and under is a namespace every field writes in at once, and
case cannot separate its senses because the competing one is an acronym too:
`PCR`, `PBS`, `LPS` and `CAP` are registered enzyme symbols that name a method,
a buffer, a polysaccharide and a phenotype in running text. No enzyme and no
bacterium loses its last form to this bar.
"""

MAX_FORM_WORDS = 8
"""Longest form, in words. Also the widest window the sweep has to try."""

SYMBOL_MAX_LENGTH = 5
"""Length at or below which a surface form is read as a symbol, not a name."""

COMMON_WORD_ZIPF = 3.0
"""Zipf frequency above which a one-word form names nothing.

BRENDA registers ordinary English as strain designations and as place and
surnames, and `Yes`, `alpha` and `2019` are as much of it as `sensitive` is.
Asked of any form `is_english_spelling` admits, folding or not. Does not
replace `PLACEHOLDER_FORMS`, which covers words common only in this
literature.
"""

FUZZY_MIN_LENGTH = 4
"""Shortest word a near-hit is asked of.

Below this `fuzz.ratio`'s own length-normalization already refuses almost
everything a loose cutoff would admit, so the floor avoids wasted lookups
rather than changing the outcome.
"""

FUZZY_CUTOFF: FuzzyScore = 80.0
"""How close a word must score to a known single-word form to abstain on it.

Loose by design and not calibrated: a fuzzy hit can only ever turn a token into
`IGNORE_INDEX`, so a wrong hit costs one token of negative signal rather than a
mislabelled positive.
"""

FUZZY_CANDIDATE_MAX_TERMS = 20_000
"""Ceiling on a first-letter bucket's size before a fuzzy lookup skips it.

`process.extractOne` is linear in the candidate count, so an unbounded bucket
turns one common initial letter into the `O(terms)` cost this module avoids.
"""

PLACEHOLDER_FORMS = frozenset(
    {
        "more",
        "plant",
        "plants",
        "mutant",
        "strain",
        "bacteria",
        "bacterium",
        "yeast",
        "protease",
    }
)
"""Single-word forms that name no particular entity, and are dropped.

Only the *bare* form goes, so `alkaline protease` and `Bacillus strain 168`
keep their IDs.
"""

DESCRIPTOR_MIN_RECORDS = 5
"""Anonymous strain records a designation must be shared by to be dropped.

An anonymous record has neither a taxon nor a culture number, so its
designation is all that identifies it. No real strain in the dump is filed that
way under more than four; `CuZn-SOD`, `type S` and the other protein names and
descriptors are filed under five to twenty-one.
"""

BRENDA_PREFIXES: Mapping[str, str] = {
    entity_type.name: entity_type.prefix
    for entity_type in BRENDA_SCHEMA.entity_types
}
"""Entity-table name -> the prefix its numeric IDs wear in a corpus row.

Read off the schema rather than restated: a prefix disagreeing with the
corpus's spelling does not fail, it produces an index whose keys no gold set
can ever match.
"""

_ENTITY_TABLE_KEY = b'"enzymes": {"'
_TAIL_SEARCH_BYTES = 256 * 1024 * 1024

_WORD = re.compile(r"[^\W_]+")

THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")
"""A comma inside one number, as against one between two of them.

The literature writes both — `DSM 22,228` is a single deposit number and
`ATCC 35984, 35983` is two — so a comma joins the digits around it only where
exactly three digits follow it and nothing else does. `NBRC 15308, 100` is why
the absent space is load-bearing: its second item is itself three digits, and
gluing it fabricates an accession no collection ever issued.
"""

COLLECTIONS = frozenset(
    {
        "ACM",
        "AS",
        "ATCC",
        "BCC",
        "BCRC",
        "CBMAI",
        "CBS",
        "CCAC",
        "CCAP",
        "CCM",
        "CCMM",
        "CCMP",
        "CCRC",
        "CCT",
        "CCUG",
        "CDBB",
        "CECT",
        "CFBP",
        "CGMCC",
        "CIP",
        "CLIB",
        "CNCTC",
        "CRBIP",
        "DBVPG",
        "DSM",
        "FGSC",
        "FRR",
        "HAMBI",
        "HUT",
        "IAM",
        "ICMP",
        "IFO",
        "IHEM",
        "IMET",
        "IMI",
        "JCM",
        "KACC",
        "KCTC",
        "LMD",
        "LMG",
        "MUCL",
        "MUM",
        "NBIMCC",
        "NBRC",
        "NCAIM",
        "NCCB",
        "NCDO",
        "NCFB",
        "NCIB",
        "NCIM",
        "NCIMB",
        "NCMB",
        "NCPF",
        "NCPPB",
        "NCTC",
        "NCYC",
        "NIES",
        "NRRL",
        "PCC",
        "PDDCC",
        "RCC",
        "SAG",
        "TBRC",
        "TISTR",
        "UAMH",
        "UTEX",
        "VKM",
        "VTT",
    }
)
"""Acronyms of the culture collections BRENDA's deposits are held in.

Closed, and matched case-sensitively: `AS` is a collection and also two
ordinary letters, and the difference an accession has from a strain designation
is the acronym, not the shape.
"""

_ACCESSION_BODY = r"(?:[A-Za-z]{1,3}[-.])?\d+(?:[./-]\d+)*[A-Za-z]?"

ACCESSION = re.compile(
    r"(?<![A-Za-z0-9])"
    r"("
    + "|".join(sorted(COLLECTIONS, key=lambda name: (-len(name), name)))
    + r")(?![A-Za-z])[ -]{0,2}"
    r"(" + _ACCESSION_BODY + r")"
    r"(?![A-Za-z0-9])"
)
"""One culture-collection accession: acronym, optional separator, number.

The acronym decides, not the shape: `PAO1`, `IP 32953` and `ST 131` are strain
designations written exactly like deposits, so a pattern taking any
capitals-then-digits reads three of them as accessions. The separator is
optional because the literature drops it — `ATCC14990` for BRENDA's
`ATCC 14990` — which is what `accession_spellings` exists to reconcile.
"""

_BINOMIAL_GENUS = re.compile(r"^[A-Z][a-z]+(?= [a-z]{2})")
"""A genus opening a binomial: capitalized word, then a lowercase epithet.

The lookahead is the guard, so a culture-collection number never comes back
mangled.
"""

EC_PREFIX = "EC "
"""What an EC number has to be written with to be read as one.

The index is keyed by a form's words, so a bare `5.3.2.1` registers as the key
`5 3 2 1`, and every section number, corpus size and confidence interval of
that shape then names an enzyme. The literature writes `EC 5.3.2.1`, and the
qualifier is the only thing separating the number from the digits.
"""


def word_spans(text: str) -> list[tuple[str, int, int]]:
    """The alphanumeric runs of `text`, each with where it sits in `text`.

    Underscore is excluded deliberately: `\\w` admits it, and a gene name
    written `pyr_C` should tokenize the way `pyr-C` does. A `THOUSANDS` comma
    is not a boundary, so `DSM 22,228` yields the word `22228` — but the
    offsets stay anchored to the text as written, comma included, because a
    caller painting labels over the document has nothing else to index by.

    :param text: the string to split.
    :return: its alphanumeric runs, each as `(word, start, end)`.
    """
    separators = {match.start() for match in THOUSANDS.finditer(text)}
    spans: list[tuple[str, int, int]] = []
    for match in _WORD.finditer(text):
        if spans and match.start() - 1 in separators:
            word, start, _ = spans[-1]
            spans[-1] = (word + match.group(), start, match.end())
        else:
            spans.append((match.group(), match.start(), match.end()))
    return spans


def form_words(text: str) -> list[str]:
    """The alphanumeric runs of `text`, in order.

    :param text: the string to split.
    :return: its alphanumeric runs, as `word_spans` reads them.
    """
    return [word for word, _, _ in word_spans(text)]


def form_key(words: Sequence[str]) -> str:
    """The lookup key for a word sequence.

    :param words: the words of a form, as `form_words` returns them.
    :return: the key those words index under.
    """
    return " ".join(words)


def is_symbol_like(term: str) -> bool:
    """Whether case is load-bearing for `term`.

    Case is the only feature separating the enzyme symbol `FOR` from the
    English word `for`; descriptive names collide with no English word and so
    can afford to fold.

    :param term: a surface form.
    :return: whether it must keep its case.
    """

    return len(term) <= SYMBOL_MAX_LENGTH or any(
        character.isupper() for character in term[1:]
    )


def is_english_spelling(term: str) -> bool:
    """Whether English writes `term` in the casing it is written in here.

    `wordfreq` folds case, so its answer describes the word rather than this
    spelling of it. That makes the frequency guard meaningful for a form
    running text also produces — `Yes` opening a sentence, `alpha`, `2019` —
    and meaningless for one it never produces, where `FOR` would be deleted on
    the strength of the preposition.

    :param term: a single-word surface form.
    :return: whether it is spelled the way English spells that word.
    """
    return term == term.lower() or term == term.capitalize()


@lru_cache(maxsize=None)
def is_common_word(word: str) -> bool:
    """Whether general English uses `word` too often for it to name anything.

    Memoized, since both callers ask it of the same running-prose words over
    and over across a corpus.

    :param word: a single word.
    :return: whether its Zipf frequency reaches `COMMON_WORD_ZIPF`.
    """
    return zipf_frequency(word.lower(), "en") >= COMMON_WORD_ZIPF


@dataclass(frozen=True, slots=True)
class SurfaceFormIndex:
    """Surface form -> the entity IDs that form could name.

    Two tables rather than one because the case policy is per form, not per
    index: `exact` is keyed by the form's words as written, `folded` by the
    same words lowercased.
    """

    exact: Mapping[str, frozenset[str]]
    folded: Mapping[str, frozenset[str]]
    max_words: int
    exact_first_words: frozenset[str]
    folded_first_words: frozenset[str]
    exact_singles_by_first_letter: Mapping[str, tuple[str, ...]]
    folded_singles_by_first_letter: Mapping[str, tuple[str, ...]]
    _fuzzy_cache: dict[tuple[str, float], frozenset[str]] = field(
        default_factory=dict, compare=False, repr=False
    )
    """Memo of `fuzzy_ids` keyed by `(word, cutoff)`.

    Sound because the tables it scores against are fixed once the frozen index
    is built. Mutating this dict's *contents* needs no `object.__setattr__`;
    only reassigning the attribute would.
    """

    def lookup(self, words: Sequence[str]) -> frozenset[str]:
        """Every entity ID some form of which is exactly `words`.

        Both tables are read and their answers unioned, since a window can be a
        symbol of one entity and a descriptive name of another.

        :param words: the words of the candidate span.
        :return: the entity IDs any form of which is exactly those words.
        """
        key = form_key(words)
        return self.exact.get(key, frozenset()) | self.folded.get(
            key.lower(), frozenset()
        )

    def may_start(self, word: str) -> bool:
        """Whether any form begins with `word`.

        The sweep asks this once per position so that ordinary prose costs two
        set lookups rather than `MAX_FORM_WORDS` window joins.

        :param word: the word at the sweep's current position.
        :return: whether any form starts with it.
        """
        return (
            word in self.exact_first_words
            or word.lower() in self.folded_first_words
        )

    def fuzzy_ids(
        self, word: str, cutoff: FuzzyScore = FUZZY_CUTOFF
    ) -> frozenset[EntityId]:
        """Entity IDs of single-word forms `word` is a close variant of.

        Asked only of a word `lookup` already found nothing for, and gated by
        `is_common_word` on the *query* as well as the candidates: at this
        cutoff an ordinary English word can score within it of an unrelated
        technical one. A word carrying no letter is refused outright, since
        `fuzz.ratio` reads digits as interchangeable and a number one digit
        from a deposit number is a different deposit rather than a variant of
        one. Memoized on the index.

        :param word: a word no exact form matched.
        :param cutoff: the `fuzz.ratio` score a candidate must reach.
        :return: the entity IDs of the near-hits, empty if there are none.
        """
        cache_key = (word, cutoff)
        cached = self._fuzzy_cache.get(cache_key)
        if cached is not None:
            return cached

        if (
            len(word) < FUZZY_MIN_LENGTH
            or not any(character.isalpha() for character in word)
            or is_common_word(word)
        ):
            self._fuzzy_cache[cache_key] = frozenset()
            return frozenset()

        ids: set[str] = set()

        exact_candidates = self.exact_singles_by_first_letter.get(word[:1], ())
        if 0 < len(exact_candidates) <= FUZZY_CANDIDATE_MAX_TERMS:
            found = process.extractOne(
                word, exact_candidates, scorer=fuzz.ratio, score_cutoff=cutoff
            )
            if found is not None:
                ids |= self.exact[found[0]]

        folded_word = word.lower()
        folded_candidates = self.folded_singles_by_first_letter.get(
            folded_word[:1], ()
        )
        if 0 < len(folded_candidates) <= FUZZY_CANDIDATE_MAX_TERMS:
            found = process.extractOne(
                folded_word,
                folded_candidates,
                scorer=fuzz.ratio,
                score_cutoff=cutoff,
            )
            if found is not None:
                ids |= self.folded[found[0]]

        result = frozenset(ids)
        self._fuzzy_cache[cache_key] = result
        return result

    @property
    def entity_ids(self) -> frozenset[str]:
        """Every entity the index can still reach.

        `PLACEHOLDER_FORMS` is judged against this; `COMMON_WORD_ZIPF`
        deliberately is not, since a key that names everything makes its entity
        no more findable.
        """
        reachable: set[str] = set()
        for table in (self.exact, self.folded):
            for ids in table.values():
                reachable |= ids
        return frozenset(reachable)

    def __len__(self) -> int:
        return len(self.exact) + len(self.folded)


def accession_spellings(form: str) -> list[str]:
    """`form`, plus the ways running text respells the deposits it carries.

    BRENDA records a deposit number as `ATCC 14990` and the literature writes
    `ATCC14990` in about a tenth of its mentions; since the index is keyed by
    a form's words, the two are different keys and only one of them is held.
    Both spellings are produced so that either recovers the strain. A form
    carrying no accession — `PAO1`, `IP 32953`, `ST 131` — comes back alone,
    which is what the closed acronym list in `ACCESSION` is for.

    :param form: a surface form as BRENDA spells it.
    :return: `form` first, then its respellings, without duplicates.
    """
    spellings = [form]
    for separator in ("", " "):
        respelled = ACCESSION.sub(rf"\g<1>{separator}\g<2>", form)
        if respelled not in spellings:
            spellings.append(respelled)
    return spellings


def index_keys(form: str) -> list[tuple[str, bool]]:
    """Every key `form` is reachable under, each with whether it is folded.

    One key usually, two where `accession_spellings` finds a deposit number
    the corpus also writes the other way round.

    :param form: a surface form as BRENDA spells it.
    :return: the keys and their folding, empty if the form carries no ID.
    """
    keys: list[tuple[str, bool]] = []
    for spelling in accession_spellings(form):
        keyed = _index_key(spelling)
        if keyed is not None and keyed not in keys:
            keys.append(keyed)
    return keys


def _index_key(form: str) -> tuple[str, bool] | None:
    """`form`'s lookup key and whether it is case-folded, or None if dropped.

    The frequency guard is asked of every single-word form whatever branch it
    routes to, since it is the form's *spelling* and not its table that
    decides whether the question is meaningful.
    """
    stripped = form.strip()
    if len(stripped) < MIN_FORM_LENGTH:
        return None

    words = form_words(stripped)
    if not words or len(words) > MAX_FORM_WORDS:
        return None

    key = form_key(words)
    if len(words) == 1:
        if key.lower() in PLACEHOLDER_FORMS:
            return None
        if is_english_spelling(key) and is_common_word(key):
            return None

    if is_symbol_like(stripped):
        return key, False
    return key.lower(), True


def build_index(
    forms_by_entity: Mapping[str, Iterable[str]],
) -> SurfaceFormIndex:
    """Invert `forms_by_entity`, which maps a *prefixed* ID to its forms.

    Prefixed because that is the spelling the corpus uses, and a label compared
    against a document's gold set is only useful in that spelling.

    :param forms_by_entity: prefixed entity ID -> its surface forms.
    :return: the index those forms define.
    """
    exact: collections.defaultdict[str, set[str]] = collections.defaultdict(set)
    folded: collections.defaultdict[str, set[str]] = collections.defaultdict(
        set
    )
    max_words = 0

    for entity_id, forms in forms_by_entity.items():
        for form in forms:
            for key, fold in index_keys(form):
                (folded if fold else exact)[key].add(entity_id)
                max_words = max(max_words, key.count(" ") + 1)

    return SurfaceFormIndex(
        exact={key: frozenset(ids) for key, ids in exact.items()},
        folded={key: frozenset(ids) for key, ids in folded.items()},
        max_words=max_words,
        exact_first_words=frozenset(key.split(" ", 1)[0] for key in exact),
        folded_first_words=frozenset(key.split(" ", 1)[0] for key in folded),
        exact_singles_by_first_letter=_singles_by_first_letter(exact),
        folded_singles_by_first_letter=_singles_by_first_letter(folded),
    )


def index_digest(index: SurfaceFormIndex) -> str:
    """A fingerprint of every form `index` can match, and of what it names.

    Sorted and explicitly encoded, so the same index digests the same in any
    process on any machine. It is what lets an artifact labelled from an index
    refuse a later run whose index differs — by its inputs, by the extractors
    that pooled them, or by the filters `index_keys` applies.

    :param index: the index to fingerprint.
    :return: the hex SHA-256 of its two lookup tables.
    """
    digest = hashlib.sha256()
    tables = (("exact", index.exact), ("folded", index.folded))
    for table_name, table in tables:
        for key in sorted(table):
            entities = " ".join(sorted(table[key]))
            line = f"{table_name}\t{key}\t{entities}\n"
            digest.update(line.encode("utf8"))
    return digest.hexdigest()


def _singles_by_first_letter(
    table: Mapping[str, set[str]],
) -> dict[str, tuple[str, ...]]:
    """Single-word keys of `table`, bucketed by their first character.

    This is what keeps `SurfaceFormIndex.fuzzy_ids` from scoring a word against
    the whole population. Sorted so a bucket is a pure function of `table`:
    `process.extractOne` breaks a tied score by position, and an unsorted
    bucket would carry `table`'s insertion order instead.
    """
    buckets: collections.defaultdict[str, list[str]] = collections.defaultdict(
        list
    )
    for key in table:
        if " " not in key and key:
            buckets[key[0]].append(key)
    return {letter: tuple(sorted(keys)) for letter, keys in buckets.items()}


def enzyme_forms(table: Mapping[str, Any]) -> dict[str, list[str]]:
    """Enzyme ID -> recommended name, qualified EC number and synonyms.

    The number is indexed only in its `EC_PREFIX` spelling, and kept rather
    than dropped because an unmatched span is painted `OUTSIDE`: an EC number
    the index does not hold trains the one spelling that names an enzyme
    unambiguously as a negative.

    :param table: the dump's `enzymes` table.
    :return: each enzyme's surface forms.
    """
    return {
        entity_id: [
            record.get("recommended_name") or "",
            _ec_number_form(record.get("ec_class") or ""),
            *(record.get("synonyms") or []),
        ]
        for entity_id, record in table.items()
    }


def _ec_number_form(ec_class: str) -> str:
    """`5.3.2.1` -> `EC 5.3.2.1`, and an absent number -> no form at all."""
    number = ec_class.strip()
    return f"{EC_PREFIX}{number}" if number else ""


def abbreviated_genus(form: str) -> str | None:
    """`Escherichia coli K-12` -> `E. coli K-12`, or None off a binomial.

    Restates `brenda_references.utils.abbreviate_bacteria`'s convention rather
    than importing it, because this module is a leaf and that one is not.

    :param form: a candidate surface form.
    :return: the genus-abbreviated form, or None if it opens with no binomial.
    """
    stripped = form.strip()
    genus = _BINOMIAL_GENUS.match(stripped)
    if genus is None:
        return None
    return f"{stripped[0]}.{stripped[genus.end() :]}"


def with_abbreviated_genus(forms: Iterable[str]) -> list[str]:
    """`forms`, each binomial-opening one followed by its abbreviation.

    Only 37% of BRENDA's bacteria carry any synonym, so the form running text
    uses is usually absent while the full binomial is present.

    :param forms: surface forms of one entity.
    :return: those forms plus the abbreviations they imply.
    """
    expanded: list[str] = []
    for form in forms:
        expanded.append(form)
        abbreviated = abbreviated_genus(form)
        if abbreviated is not None:
            expanded.append(abbreviated)
    return expanded


def bacteria_forms(table: Mapping[str, Any]) -> dict[str, list[str]]:
    """Bacterium ID -> organism name, LPSN synonyms, and their abbreviations.

    :param table: the dump's `bacteria` table.
    :return: each bacterium's surface forms.
    """
    return {
        entity_id: with_abbreviated_genus(
            [
                record.get("organism") or "",
                *(record.get("synonyms") or []),
            ]
        )
        for entity_id, record in table.items()
    }


def strain_forms(table: Mapping[str, Any]) -> dict[str, list[str]]:
    """Strain ID -> designations and culture-collection numbers.

    Left out: the `taxon`, which names the species; a letterless form, which
    running text spells as page ranges and lot numbers; and a designation
    `DESCRIPTOR_MIN_RECORDS` anonymous records share, which describes a protein
    or a phenotype rather than naming a strain.

    :param table: the dump's `strains` table.
    :return: each strain's surface forms.
    """
    descriptors = _descriptor_keys(table)
    return {
        entity_id: with_abbreviated_genus(
            [
                form
                for form in (
                    *_named_designations(record, descriptors),
                    *(
                        culture.get("strain_number") or ""
                        for culture in (record.get("cultures") or [])
                    ),
                )
                if any(character.isalpha() for character in form)
            ]
        )
        for entity_id, record in table.items()
    }


def _is_anonymous(record: Mapping[str, Any]) -> bool:
    """Whether a strain record has neither a taxon nor a culture number."""
    return not record.get("taxon") and not any(
        culture.get("strain_number") for culture in record.get("cultures") or []
    )


def _descriptor_keys(
    table: Mapping[str, Any],
) -> frozenset[tuple[str, bool]]:
    """Index keys `DESCRIPTOR_MIN_RECORDS` or more anonymous records share.

    Counted by key rather than by string, so `CuZn-SOD` and `CuZn SOD` are one
    designation, as they are to the index.
    """
    holders: collections.defaultdict[tuple[str, bool], set[str]] = (
        collections.defaultdict(set)
    )
    for entity_id, record in table.items():
        if not _is_anonymous(record):
            continue
        for designation in record.get("designations") or []:
            for key in index_keys(designation):
                holders[key].add(entity_id)
    return frozenset(
        key
        for key, holding in holders.items()
        if len(holding) >= DESCRIPTOR_MIN_RECORDS
    )


def _named_designations(
    record: Mapping[str, Any], descriptors: frozenset[tuple[str, bool]]
) -> list[str]:
    """`record`'s designations, less the descriptors if it is anonymous.

    A record with a taxon or a culture number keeps every designation: it is
    identified by something besides the string.
    """
    designations: list[str] = list(record.get("designations") or [])
    if not _is_anonymous(record):
        return designations
    return [
        designation
        for designation in designations
        if descriptors.isdisjoint(index_keys(designation))
    ]


def pooled_other_organism_names(
    columns: Iterable[Mapping[str, str]],
) -> dict[str, list[str]]:
    """Other-organism ID -> the names the corpus calls it, as written.

    Pooled across every document on purpose: a document mentioning an organism
    it was not annotated with is exactly the case the abstain target exists
    for, and that mention is only recognizable from another document's naming
    of it.

    :param columns: the per-document id -> name mappings, which is the shape
        both the TinyDB `documents` table and the split CSVs' column hold.
    :return: each other-organism's pooled names, verbatim.
    """
    names: collections.defaultdict[str, list[str]] = collections.defaultdict(
        list
    )
    for column in columns:
        for entity_id, name in column.items():
            if isinstance(name, str) and name:
                names[str(entity_id)].append(name)
    return dict(names)


def other_organism_forms(
    columns: Iterable[Mapping[str, str]],
) -> dict[str, list[str]]:
    """Other-organism ID -> pooled document names and their abbreviations.

    These names are the only ones harvested from running text, which is where
    an abbreviated genus is likeliest to be what the text actually says. A
    caller resolving the names against an outside nomenclature wants
    `pooled_other_organism_names` instead: the abbreviation is a spelling
    running text uses, not a name a reference database answers for.

    :param columns: the per-document id -> name mappings, which is the shape
        both the TinyDB `documents` table and the split CSVs' column hold.
    :return: each other-organism's pooled names, with the abbreviations they
        imply.
    """
    return {
        entity_id: with_abbreviated_genus(forms)
        for entity_id, forms in pooled_other_organism_names(columns).items()
    }


def brenda_surface_forms(
    tables: Mapping[str, Mapping[str, Any]],
    other_organisms: Iterable[Mapping[str, str]] = (),
    prefixes: Mapping[str, str] = BRENDA_PREFIXES,
) -> dict[str, list[str]]:
    """Prefixed entity ID -> surface forms, over all four ID namespaces.

    A table absent from `tables` contributes nothing rather than raising, since
    `load_entity_tables`'s tail-parse route cannot reach `documents`.

    :param tables: the dump's entity tables, by table name.
    :param other_organisms: the per-document id -> name mappings.
    :param prefixes: table name -> the ID prefix the corpus spells it with.
    :return: every entity's surface forms, under its prefixed ID.
    """
    extracted = {
        "enzymes": enzyme_forms(tables.get("enzymes", {})),
        "bacteria": bacteria_forms(tables.get("bacteria", {})),
        "strains": strain_forms(tables.get("strains", {})),
        "other_organisms": other_organism_forms(other_organisms),
    }

    return {
        prefixes[name] + str(entity_id): forms
        for name, by_entity in extracted.items()
        for entity_id, forms in by_entity.items()
    }


def load_entity_tables(
    path: str | os.PathLike[str],
) -> dict[str, dict[str, Any]]:
    """The tables of a TinyDB dump, without loading a 1.1 GB file.

    A dump larger than the tail-search window is parsed off its tail, which
    yields `enzymes`, `bacteria` and `strains` but **not** `documents`;
    anything smaller is read whole.

    :param path: the dump to read.
    :return: its entity tables, by table name.
    :raises ValueError: if a large dump carries no entity table in its tail.
    """
    dump = pathlib.Path(path)
    if dump.stat().st_size <= _TAIL_SEARCH_BYTES:
        with dump.open("r", encoding="utf8") as handle:
            loaded: dict[str, dict[str, Any]] = json.load(handle)
        return loaded

    with dump.open("rb") as handle:
        handle.seek(dump.stat().st_size - _TAIL_SEARCH_BYTES)
        tail = handle.read()

    offset = tail.find(_ENTITY_TABLE_KEY)
    if offset < 0:
        msg = (
            f"{dump} carries no {_ENTITY_TABLE_KEY.decode()!r} in its last "
            f"{_TAIL_SEARCH_BYTES} bytes; its entity tables are elsewhere"
        )
        raise ValueError(msg)

    tables: dict[str, dict[str, Any]] = json.loads(
        "{" + tail[offset:].decode("utf8")
    )
    return tables
