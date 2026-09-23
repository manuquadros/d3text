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
import math
import os
import pathlib
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache, partial
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
        "archaeon",
        "plasmid",
        "yeast",
        "protease",
        "constitutive",
        "archaea",
        "protozoa",
    }
)
"""Single-word forms that name no particular entity, and are dropped.

Only the *bare* form goes, so `alkaline protease` and `Bacillus strain 168`
keep their IDs. `SurfaceFormIndex.fuzzy_ids` reads the set too: a dropped
form must not come back as a near-miss of whatever key sits closest to it.
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

UNIT_SYMBOLS = frozenset(
    {
        "aa",
        "bp",
        "kb",
        "Kb",
        "kbp",
        "Mb",
        "nt",
        "Da",
        "kDa",
        "KDa",
        "kD",
        "MDa",
        "fg",
        "pg",
        "ng",
        "mg",
        "kg",
        "fM",
        "pM",
        "nM",
        "mM",
        "mol",
        "fmol",
        "pmol",
        "nmol",
        "mmol",
        "nl",
        "nL",
        "ml",
        "mL",
        "dl",
        "dL",
        "nm",
        "mm",
        "cm",
        "km",
        "ps",
        "ns",
        "ms",
        "sec",
        "min",
        "mins",
        "hr",
        "hrs",
        "ºC",
        "oC",
        "rpm",
        "xg",
        "mV",
        "kV",
        "Hz",
        "kHz",
        "MHz",
        "kJ",
        "kcal",
        "kPa",
        "MPa",
        "psi",
        "pt",
        *(
            micro + unit
            for micro in ("µ", "μ", "u")
            for unit in ("g", "l", "L", "m", "M", "mol", "s")
        ),
    }
)
"""Unit symbols running text glues to a number: `128bp`, `110aa`, `22min`.

Multi-letter only. One letter after a number is how a strain designation is
suffixed — `168T`, `10403S`, `14028s` and `210x` are all keys — so `g`, `s` or
`x` cannot tell a quantity from a name; no key ends in any of these. The micro
prefix comes in both micro signs and as ASCII `u`, as the literature writes it.
"""

_QUANTITY = re.compile(r"\d+([^\W\d_]+)")
"""A number and the letters glued to it, the shape `is_quantity` reads."""

_BINOMIAL_GENUS = re.compile(r"^[A-Z][a-z]+(?= [a-z]{2})")
"""A genus opening a binomial: capitalized word, then a lowercase epithet.

The lookahead is the guard, so a culture-collection number never comes back
mangled.
"""

_BARE_PLACEHOLDERS = frozenset({"sp", "spp", "bacterium"})
"""Placeholder words that identify nothing when they end the form.

`Firmicutes bacterium` abbreviated stays `Firmicutes bacterium`: dropping the
abbreviation is the only guard `bacterium` gets. `Paracoccus sp. N81106` keeps
its abbreviation regardless, since the designation still identifies it.
"""

_CASE_SENSITIVE_PLACEHOLDERS = frozenset({"sp", "spp"})
"""`_BARE_PLACEHOLDERS` whose abbreviation is generated, case kept intact.

`Agaricus sp.` abbreviated is `A. sp.`, a key every unnamed species of an `A`
genus shares — real running text does abbreviate a species left unnamed this
way. Folding it to `a sp` would also catch the prose `a sp.`/`a bacterium`
writes lowercase, so `_index_key` keeps this shape's case instead of letting
`is_symbol_like` fold it; `_ABBREVIATED_PLACEHOLDER_KEY` is what recognises
it there. `bacterium` is left out: no abbreviated-`bacterium` collision or
lost gold span was ever observed to fix.
"""

_ABBREVIATED_PLACEHOLDER_KEY = re.compile(r"^[A-Z]\.\s*spp?\.?$")
"""A genus-initial abbreviation of a bare `sp.`/`spp.` placeholder, exactly.

`N. sp`, `B. sp`, `T. sp` are how running text writes an unnamed species —
real organism mentions that must reach the exact table, not the folded one
the lowercase prose `a sp`/`a bacterium` also reads into, and the only
feature separating the two is case.
"""

EC_PREFIXES: tuple[str, ...] = ("EC ", "E.C. ")
"""The spellings an EC number has to be written in to be read as one.

The index is keyed by a form's words, so a bare `5.3.2.1` registers as the key
`5 3 2 1`, and every section number, corpus size and confidence interval of
that shape then names an enzyme. The literature writes `EC 5.3.2.1`, or in the
older style `E.C. 5.3.2.1`, and the qualifier is the only thing separating the
number from the digits.
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


def has_letter(form: str) -> bool:
    """Whether `form` carries at least one letter.

    Answers the same of a form as written and of its words, since every
    letter is a word character and so no split of the form can drop one.

    :param form: a surface form, a matched span, or one of their words.
    :return: whether any character of it is alphabetic.
    """
    return any(character.isalpha() for character in form)


def _is_unit_symbol(letters: str) -> bool:
    """Whether `letters` is a `UNIT_SYMBOLS` member, gravity's `x` included.

    Centrifugation drops the `x` in running text -- `1,000g` for the same
    `1,000xg` -- so the bare letter counts too when prefixing it with `x`
    would land in `UNIT_SYMBOLS`.
    """
    return letters in UNIT_SYMBOLS or f"x{letters}" in UNIT_SYMBOLS


def is_quantity(text: str, start: int, end: int) -> bool:
    """Whether the word at `text[start:end]` measures something, not names it.

    A number glued to a `UNIT_SYMBOLS` symbol is a quantity, and so is one
    written with a `THOUSANDS` separator whatever follows it, as `3,000g` is:
    only a deposit number groups its digits and still names something, which
    is why a word an `ACCESSION` reads into its number never counts -- unless
    the accession match stops short of the word, at the collection's own
    separator, and what is left over reads as a unit rather than a deposit
    suffix: `AS 1,000g` is a quantity sitting behind a collection acronym,
    `DSM 22,228T` a deposit still wearing its type-strain letter.

    :param text: the text the word was read from.
    :param start: where the word starts, as `word_spans` reports it.
    :param end: where the word ends.
    :return: whether the word is a number carrying a unit rather than a name.
    """
    written = text[start:end]
    number = _QUANTITY.fullmatch(written.replace(",", ""))
    if number is None:
        return False
    if "," not in written and number.group(1) not in UNIT_SYMBOLS:
        return False
    # ponytail: a comma-grouped deposit ending in a real unit letter (no
    # observed key does) would misread as a quantity here; widen past
    # `_is_unit_symbol` if one turns up.
    return not any(
        deposit.start(2) <= start < deposit.end(2)
        and (deposit.end(2) == end or not _is_unit_symbol(number.group(1)))
        for deposit in ACCESSION.finditer(text, 0, end)
    )


def _is_placeholder(word: str) -> bool:
    """Whether `word` is a `PLACEHOLDER_FORMS` entry, or one with an `s`.

    In any casing, as `_index_key` drops the entries: `plasmids` names no
    entity any more than `plasmid` does.
    """
    folded = word.lower()
    return (
        folded in PLACEHOLDER_FORMS
        or folded.removesuffix("s") in PLACEHOLDER_FORMS
    )


@dataclass(frozen=True, slots=True, eq=False)
class SurfaceFormIndex:
    """Surface form -> the entity IDs that form could name.

    Two tables rather than one because the case policy is per form, not per
    index: `exact` is keyed by the form's words as written, `folded` by the
    same words lowercased.

    `eq=False` leaves hash/equality at `object`'s identity-based default,
    rather than the dataclass-generated pair `frozen=True` would otherwise
    add: `exact` and `folded` are built as plain `dict`s (`build_index`), so
    a compared-field hash would raise `TypeError` on every instance, a
    promise `Mapping[str, frozenset[str]]` cannot keep without also making
    the two `*_singles_by_first_letter` tables genuinely immutable.
    """

    exact: Mapping[str, frozenset[str]]
    folded: Mapping[str, frozenset[str]]
    max_words: int
    exact_first_words: frozenset[str]
    folded_first_words: frozenset[str]
    exact_singles_by_first_letter: Mapping[str, Mapping[int, tuple[str, ...]]]
    folded_singles_by_first_letter: Mapping[str, Mapping[int, tuple[str, ...]]]
    _fuzzy_cache: dict[tuple[str, float], frozenset[str]] = field(
        default_factory=dict, compare=False, repr=False
    )
    """Memo of `fuzzy_ids` keyed by `(word, cutoff)`.

    Sound because the tables it scores against are fixed once the frozen index
    is built. Mutating this dict's *contents* needs no `object.__setattr__`;
    only reassigning the attribute would.
    """

    excluded_words: frozenset[str] = frozenset()
    """Case-folded single words `fuzzy_ids` must refuse beside a placeholder.

    Populated by `build_index`'s caller from `excluded_single_words`: an
    epithet or a descriptor is dropped from `strain_forms` for naming a
    species or an anonymous group rather than a particular record, the same
    reason a `PLACEHOLDER_FORMS` entry is dropped, so a near-miss on the same
    word must be refused the same way. Defaults to empty for an index built
    from forms alone, with no table to read the exclusion from.
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
        one. So is a `PLACEHOLDER_FORMS` entry or its plural, or a word in
        `excluded_words`: each was dropped for naming no particular entity —
        a placeholder, an epithet, a descriptor — and a near-hit would only
        hand the word to whichever key sits nearest it instead. Memoized on
        the index.

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
            or not has_letter(word)
            or is_common_word(word)
            or _is_placeholder(word)
            or word.lower() in self.excluded_words
        ):
            self._fuzzy_cache[cache_key] = frozenset()
            return frozenset()

        ids: set[str] = set()

        exact_key = _nearest_key(
            word, self.exact_singles_by_first_letter.get(word[:1], {}), cutoff
        )
        if exact_key is not None:
            ids |= self.exact[exact_key]

        folded_word = word.lower()
        folded_key = _nearest_key(
            folded_word,
            self.folded_singles_by_first_letter.get(folded_word[:1], {}),
            cutoff,
        )
        if folded_key is not None:
            ids |= self.folded[folded_key]

        result = frozenset(ids)
        self._fuzzy_cache[cache_key] = result
        return result

    @property
    def entity_ids(self) -> frozenset[str]:
        """Every entity the index can still reach.

        `PLACEHOLDER_FORMS` is judged against this, and may cost it only a
        record named by a placeholder alone, such as the bacteria BRENDA calls
        `plasmid` and `archaeon`. `COMMON_WORD_ZIPF` deliberately is not, since
        a key that names everything makes its entity no more findable.
        """
        reachable: set[str] = set()
        for table in (self.exact, self.folded):
            for ids in table.values():
                reachable |= ids
        return frozenset(reachable)

    def __len__(self) -> int:
        return len(self.exact) + len(self.folded)


def _respell(separator: str, suffix: str, match: re.Match[str]) -> str:
    # `_ACCESSION_BODY`'s trailing `[A-Za-z]?` may already be captured here;
    # appending `suffix` on top would double it.
    #
    # Module-level, not a closure inside `accession_spellings`: the package
    # is beartyped at import by `beartype_this_package`, which decorates a
    # nested function every time its `def` runs and memoises the result by
    # function object, so the four closures built per call — one per
    # separator/suffix pair — were held for the life of the process. Bound
    # to its pair with `functools.partial`, which the claw hook never sees.
    body = match.group(2)
    added = "" if body[-1:].isalpha() else suffix
    return f"{match.group(1)}{separator}{body}{added}"


def accession_spellings(form: str) -> list[str]:
    """`form`, plus the ways running text respells the deposits it carries.

    BRENDA records a deposit number as `ATCC 14990` and the literature writes
    `ATCC14990` in about a tenth of its mentions, and `DSM 20074T` where the
    trailing `T` marks it as the species' type strain, glued to the digits
    with no space; since the index is keyed by a form's words, each is a
    different key and only one of them is held. Every spelling is produced so
    that any of them recovers the strain. A form carrying no accession —
    `PAO1`, `IP 32953`, `ST 131` — comes back alone, which is what the closed
    acronym list in `ACCESSION` is for.

    :param form: a surface form as BRENDA spells it.
    :return: `form` first, then its respellings, without duplicates.
    """
    spellings = [form]
    for separator in ("", " "):
        for suffix in ("", "T"):
            respelled = ACCESSION.sub(
                partial(_respell, separator, suffix), form
            )
            if respelled not in spellings:
                spellings.append(respelled)
    return spellings


def index_keys(form: str) -> list[tuple[str, bool]]:
    """Every key `form` is reachable under, each with whether it is folded.

    One key usually, more where `accession_spellings` finds a deposit number
    the corpus also writes another way round or marks as a type strain.

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

    if is_symbol_like(stripped) or _ABBREVIATED_PLACEHOLDER_KEY.match(stripped):
        return key, False
    return key.lower(), True


def build_index(
    forms_by_entity: Mapping[str, Iterable[str]],
    excluded_words: frozenset[str] = frozenset(),
) -> SurfaceFormIndex:
    """Invert `forms_by_entity`, which maps a *prefixed* ID to its forms.

    Prefixed because that is the spelling the corpus uses, and a label compared
    against a document's gold set is only useful in that spelling.

    :param forms_by_entity: prefixed entity ID -> its surface forms.
    :param excluded_words: case-folded single words `fuzzy_ids` must refuse a
        near-miss on, as `excluded_single_words` reads them off the same
        tables `forms_by_entity` was built from. Defaults to none.
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
        excluded_words=excluded_words,
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
) -> dict[str, dict[int, tuple[str, ...]]]:
    """Single-word keys of `table`, by their first character, then length.

    This is what keeps `SurfaceFormIndex.fuzzy_ids` from scoring a word against
    the whole population. Sorted so a bucket is a pure function of `table`:
    `process.extractOne` breaks a tied score by position, and an unsorted
    bucket would carry `table`'s insertion order instead.
    """
    buckets: dict[str, dict[int, list[str]]] = {}
    for key in sorted(table):
        if " " not in key and key:
            buckets.setdefault(key[0], {}).setdefault(len(key), []).append(key)
    return {
        letter: {length: tuple(keys) for length, keys in by_length.items()}
        for letter, by_length in buckets.items()
    }


def length_band_ratios(cutoff: FuzzyScore) -> tuple[float, float] | None:
    """Bounds on `len(term) / len(query)` for a term that can reach `cutoff`.

    `fuzz.ratio`, and `fuzz.QRatio` on the strings it processes, score `200 *
    M / (len(a) + len(b))` with `M` at most the shorter length, which gives the
    inclusive band `q * cutoff / (200 - cutoff) <= t <= q * (200 - cutoff) /
    cutoff`.

    :param cutoff: the score a term has to be able to reach.
    :return: the band, or None for a degenerate cutoff — scoring a term that
        cannot win only costs time, so declining to prune is the safe answer.
    """
    if not 0.0 < cutoff < 200.0:
        return None

    return cutoff / (200.0 - cutoff), (200.0 - cutoff) / cutoff


def _nearest_key(
    query: str, by_length: Mapping[int, tuple[str, ...]], cutoff: FuzzyScore
) -> str | None:
    """The key a scan of the whole first-letter bucket would pick, or None.

    Scores only the lengths `length_band_ratios` admits, rounded outwards. The
    cap is measured on the whole bucket and a tie across lengths goes to the
    smaller key, the one first in sorted order, so no answer moves.
    """
    if not 0 < sum(map(len, by_length.values())) <= FUZZY_CANDIDATE_MAX_TERMS:
        return None

    # Declared bare: beartype's claw checks an annotated assignment on every
    # call, which here is once per bucket per fuzzy lookup.
    lengths: Iterable[int]
    best: tuple[float, str] | None
    ratios = length_band_ratios(cutoff)
    if ratios is None:
        lengths = by_length
    else:
        shortest, longest = ratios
        lengths = range(
            math.floor(len(query) * shortest),
            math.ceil(len(query) * longest) + 1,
        )

    best = None
    for length in lengths:
        keys = by_length.get(length)
        if keys is None:
            continue
        found = process.extractOne(
            query, keys, scorer=fuzz.ratio, score_cutoff=cutoff
        )
        if found is None:
            continue
        key, score, _ = found
        ranked = (-score, key)
        if best is None or ranked < best:
            best = ranked
    return None if best is None else best[1]


def enzyme_forms(table: Mapping[str, Any]) -> dict[str, list[str]]:
    """Enzyme ID -> recommended name, qualified EC number and synonyms.

    The number is indexed only in its `EC_PREFIXES` spellings, and kept rather
    than dropped because an unmatched span is painted `OUTSIDE`: an EC number
    the index does not hold trains a spelling that names an enzyme
    unambiguously as a negative.

    :param table: the dump's `enzymes` table.
    :return: each enzyme's surface forms.
    """
    return {
        entity_id: [
            record.get("recommended_name") or "",
            *_ec_number_forms(record.get("ec_class") or ""),
            *(record.get("synonyms") or []),
        ]
        for entity_id, record in table.items()
    }


def _ec_number_forms(ec_class: str) -> list[str]:
    """`5.3.2.1` -> its `EC_PREFIXES` spellings; an absent number -> none."""
    number = ec_class.strip()
    return [f"{prefix}{number}" for prefix in EC_PREFIXES] if number else []


def abbreviated_genus(form: str) -> str | None:
    """`Escherichia coli K-12` -> `E. coli K-12`, or None off a binomial.

    Restates `brenda_references.utils.abbreviate_bacteria`'s convention rather
    than importing it, because this module is a leaf and that one is not.

    :param form: a candidate surface form.
    :return: the genus-abbreviated form, or None if it opens with no binomial
        or is a bare `Genus bacterium` placeholder such as `Firmicutes
        bacterium`. A bare `Genus sp.`/`Genus spp.` placeholder still
        abbreviates -- `_index_key` is what keeps its case from folding.
    """
    stripped = form.strip()
    genus = _BINOMIAL_GENUS.match(stripped)
    if genus is None:
        return None
    remainder = stripped[genus.end() :]
    placeholder = remainder.strip().removesuffix(".")
    if placeholder in _BARE_PLACEHOLDERS - _CASE_SENSITIVE_PLACEHOLDERS:
        return None
    return f"{stripped[0]}.{remainder}"


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


_GENUS_PSEUDO_ID = "genus"
"""Stem of a bare-genus pseudo-entity ID, numbered per `bacteria_forms` call.

Letters only, so the prefixed ID (`"bac" + "genus0"`) still fullmatches
`EntityId`'s `[a-z]+[0-9]+`. Never a real BRENDA ID -- no bacterium record is
keyed by anything but a digit string -- so it can never coincide with a
document's actual gold entity.
"""


def bacteria_forms(table: Mapping[str, Any]) -> dict[str, list[str]]:
    """Bacterium ID -> organism name, LPSN synonyms, and their abbreviations.

    A one-word synonym is dropped from a record whose own name is longer: the
    dump hands every record under a genus that genus's synonyms, and a bare
    genus name names none of them. A genus that owns no genus-level record of
    its own -- so a bare mention of it would otherwise match no key at all --
    gets a pseudo-entity keyed to its bare name instead. That ID is never a
    document's gold entity, so `character_labels_from_spans` always writes
    `IGNORE_INDEX` where the genus is mentioned rather than `OUTSIDE`: the
    same abstention an unmatched EC number is denied and a fuzzy near-miss
    already gets, for a genus the table only ever places a species under.
    The same one-word/binomial-first-word split applies to every synonym too,
    not only a record's own `organism`: a reclassified genus that is never
    itself an `organism` value can still surface as the first word of a
    synonym binomial elsewhere in the dump.

    :param table: the dump's `bacteria` table.
    :return: each bacterium's surface forms, plus one abstain-only
        pseudo-entity per bare genus the table names no genus-level record
        for.
    """
    forms: dict[str, list[str]] = {}
    genus_records: set[str] = set()
    bare_genera: set[str] = set()

    def bucket_genus(name: str) -> None:
        words = form_words(name)
        if len(words) == 1:
            genus_records.add(words[0])
        elif len(words) > 1:
            genus_match = _BINOMIAL_GENUS.match(name.strip())
            if genus_match is not None:
                bare_genera.add(genus_match.group())

    for entity_id, record in table.items():
        organism = record.get("organism") or ""
        synonyms = record.get("synonyms") or []
        bucket_genus(organism)
        for synonym in synonyms:
            bucket_genus(synonym)
        if len(form_words(organism)) > 1:
            synonyms = [
                synonym for synonym in synonyms if len(form_words(synonym)) != 1
            ]
        forms[entity_id] = with_abbreviated_genus([organism, *synonyms])

    for position, genus in enumerate(sorted(bare_genera - genus_records)):
        forms[f"{_GENUS_PSEUDO_ID}{position}"] = [genus]

    return forms


def strain_forms(
    table: Mapping[str, Any], bacteria: Mapping[str, Any]
) -> dict[str, list[str]]:
    """Strain ID -> designations and culture-collection numbers.

    Left out: the `taxon`, which names the species; a letterless form, which
    running text spells as page ranges and lot numbers; a bare
    culture-collection acronym, which is a deposit number with its number
    missing; a designation `DESCRIPTOR_MIN_RECORDS` anonymous records share,
    which describes a protein or a phenotype rather than naming a strain; and
    a one-word designation equal to a species epithet, which running text
    writes as the epithet.

    :param table: the dump's `strains` table.
    :param bacteria: the dump's `bacteria` table, whose names are read with the
        strains' taxa for the epithets; some, `typhimurium` among them, only a
        bacterium names.
    :return: each strain's surface forms.
    """
    descriptors = _descriptor_keys(table)
    epithets = _species_epithets(table, bacteria)
    return {
        entity_id: with_abbreviated_genus(
            [
                form
                for form in (
                    *_named_designations(record, descriptors, epithets),
                    *(
                        culture.get("strain_number") or ""
                        for culture in (record.get("cultures") or [])
                    ),
                )
                if has_letter(form) and not _is_collection_acronym(form)
            ]
        )
        for entity_id, record in table.items()
    }


def excluded_single_words(
    tables: Mapping[str, Mapping[str, Any]],
) -> frozenset[str]:
    """Single words `strain_forms` drops off `tables` without dropping the ID.

    An epithet and a single-word descriptor are folded away because the word
    names a species or an anonymous group, never a particular strain — the
    same reason a `PLACEHOLDER_FORMS` entry is dropped — so `fuzzy_ids` must
    refuse a near-miss on either the way it already refuses one on a
    placeholder. A multi-word descriptor (`type S`, `CuZn-SOD`) needs no
    entry: `fuzzy_ids` is only ever asked of one word at a time, so a key
    that never was one cannot be near-missed as one.

    :param tables: the dump's entity tables, the same mapping
        `brenda_surface_forms` reads.
    :return: the words, case-folded, `SurfaceFormIndex.fuzzy_ids` must refuse.
    """
    strains = tables.get("strains", {})
    bacteria = tables.get("bacteria", {})
    descriptors = _descriptor_keys(strains)
    return _species_epithets(strains, bacteria) | {
        key.lower() for key, _ in descriptors if " " not in key
    }


def _is_collection_acronym(form: str) -> bool:
    """Whether `form` is one word and that word a `COLLECTIONS` acronym.

    Case-folded, unlike the `ACCESSION` match, which needs the case to tell an
    acronym from the ordinary letters running text spells it with: here the
    field holds the acronym however BRENDA entered it. Asked of the culture
    numbers too, since a deposit truncated to its acronym names no more than a
    designation written that way does.
    """
    words = form_words(form)
    return len(words) == 1 and words[0].upper() in COLLECTIONS


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
    record: Mapping[str, Any],
    descriptors: frozenset[tuple[str, bool]],
    epithets: frozenset[str],
) -> list[str]:
    """`record`'s designations, less epithets, and descriptors if anonymous.

    A record with a taxon or a culture number keeps every descriptor: it is
    identified by something besides the string. It loses an epithet all the
    same, since text writes the word as the epithet however the record is
    filed; that costs a strain truly named like one, such as `Album`.
    """
    designations: list[str] = [
        designation
        for designation in record.get("designations") or []
        if not _is_species_epithet(designation, epithets)
    ]
    if not _is_anonymous(record):
        return designations
    return [
        designation
        for designation in designations
        if descriptors.isdisjoint(index_keys(designation))
    ]


def _species_epithet(name: str) -> str | None:
    """The epithet of the binomial `name` opens with, or None.

    A binomial as `abbreviated_genus` reads one, whose second word must also be
    wholly lowercase letters and no placeholder: `sp.` in `Pseudomonas sp. P51`
    and `bacterium` in `Coryneform bacterium` are no epithets.
    """
    stripped = name.strip()
    genus = _BINOMIAL_GENUS.match(stripped)
    if genus is None:
        return None
    word = stripped[genus.end() :].split(maxsplit=1)[0]
    if re.fullmatch(r"[a-z]+", word) is None or word in _BARE_PLACEHOLDERS:
        return None
    return word


def _species_epithets(
    strains: Mapping[str, Any], bacteria: Mapping[str, Any]
) -> frozenset[str]:
    """Every species epithet of a bacterium's names or a strain's taxon.

    Read off the dump rather than off a word's shape: BRENDA writes cultivar
    names lowercase too, and `gantai` and `azul` name real strains.
    """
    names: list[str] = [
        name
        for record in bacteria.values()
        for name in (
            record.get("organism") or "",
            *(record.get("synonyms") or []),
        )
    ]
    for record in strains.values():
        taxon = record.get("taxon")
        if isinstance(taxon, Mapping):
            names.append(taxon.get("name") or "")
    return frozenset(
        epithet
        for name in names
        if (epithet := _species_epithet(name)) is not None
    )


def _is_species_epithet(designation: str, epithets: frozenset[str]) -> bool:
    """Whether `designation` is one word, and that word one of `epithets`.

    Case-folded, since BRENDA capitalizes some as it would a cultivar group:
    `Japonica`.
    """
    words = form_words(designation)
    return len(words) == 1 and words[0].lower() in epithets


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
        "strains": strain_forms(
            tables.get("strains", {}), tables.get("bacteria", {})
        ),
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
