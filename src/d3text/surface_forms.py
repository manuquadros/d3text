"""BRENDA's surface forms, indexed by the entity IDs they name.

Not "what is this entity called" but "which entities could this string be",
which is what distant supervision needs. The index is keyed by the *words* of a
form rather than by the form itself, so no hyphenation convention has to be
modelled. Deliberately a leaf: building an index costs neither the BRENDA data
layer nor torch. See the surface-forms page of the documentation.
"""

import collections
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

from d3text.schema import BRENDA_SCHEMA

MIN_FORM_LENGTH = 3
"""Shortest form that may carry an ID.

One- and two-character forms are almost all element symbols, figure labels and
units, which no amount of case sensitivity separates from real names.
"""

MAX_FORM_WORDS = 8
"""Longest form, in words. Also the widest window the sweep has to try."""

SYMBOL_MAX_LENGTH = 5
"""Length at or below which a surface form is read as a symbol, not a name."""

COMMON_WORD_ZIPF = 3.0
"""Zipf frequency above which a one-word case-folded form names nothing.

BRENDA registers ordinary English as strain designations and as place and
surnames, each long enough to clear `MIN_FORM_LENGTH` and lowercase enough to
fold. Does not replace `PLACEHOLDER_FORMS`, which covers words common only in
this literature.
"""

FUZZY_MIN_LENGTH = 4
"""Shortest word a near-hit is asked of.

Below this `fuzz.ratio`'s own length-normalization already refuses almost
everything a loose cutoff would admit, so the floor avoids wasted lookups
rather than changing the outcome.
"""

FUZZY_CUTOFF = 80.0
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

_BINOMIAL_GENUS = re.compile(r"^[A-Z][a-z]+(?= [a-z]{2})")
"""A genus opening a binomial: capitalized word, then a lowercase epithet.

The lookahead is the guard, so a culture-collection number never comes back
mangled.
"""


def form_words(text: str) -> list[str]:
    """The alphanumeric runs of `text`, in order.

    Underscore is excluded deliberately: `\\w` admits it, and a gene name
    written `pyr_C` should tokenize the way `pyr-C` does.

    :param text: the string to split.
    :return: its alphanumeric runs.
    """
    return [match.group() for match in _WORD.finditer(text)]


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
        self, word: str, cutoff: float = FUZZY_CUTOFF
    ) -> frozenset[str]:
        """Entity IDs of single-word forms `word` is a close variant of.

        Asked only of a word `lookup` already found nothing for, and gated by
        `is_common_word` on the *query* as well as the candidates: at this
        cutoff an ordinary English word can score within it of an unrelated
        technical one. Memoized on the index.

        :param word: a word no exact form matched.
        :param cutoff: the `fuzz.ratio` score a candidate must reach.
        :return: the entity IDs of the near-hits, empty if there are none.
        """
        cache_key = (word, cutoff)
        cached = self._fuzzy_cache.get(cache_key)
        if cached is not None:
            return cached

        if len(word) < FUZZY_MIN_LENGTH or is_common_word(word):
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


def index_key(form: str) -> tuple[str, bool] | None:
    """`form`'s lookup key and whether it is case-folded, or None if dropped.

    The frequency guard is asked last, and only of the folding branch, because
    it is that branch's own premise that decides whether the question is
    meaningful.

    :param form: a surface form as BRENDA spells it.
    :return: its key and whether that key is folded, or None if it carries no
        ID.
    """
    stripped = form.strip()
    if len(stripped) < MIN_FORM_LENGTH:
        return None

    words = form_words(stripped)
    if not words or len(words) > MAX_FORM_WORDS:
        return None

    if len(words) == 1 and words[0].lower() in PLACEHOLDER_FORMS:
        return None

    key = form_key(words)
    if is_symbol_like(stripped):
        return key, False

    if len(words) == 1 and is_common_word(key):
        return None
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
            keyed = index_key(form)
            if keyed is None:
                continue
            key, fold = keyed
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


def _singles_by_first_letter(
    table: Mapping[str, set[str]],
) -> dict[str, tuple[str, ...]]:
    """Single-word keys of `table`, bucketed by their first character.

    This is what keeps `SurfaceFormIndex.fuzzy_ids` from scoring a word against
    the whole population.
    """
    buckets: collections.defaultdict[str, list[str]] = collections.defaultdict(
        list
    )
    for key in table:
        if " " not in key and key:
            buckets[key[0]].append(key)
    return {letter: tuple(keys) for letter, keys in buckets.items()}


def enzyme_forms(table: Mapping[str, Any]) -> dict[str, list[str]]:
    """Enzyme ID -> recommended name, EC number and synonyms.

    :param table: the dump's `enzymes` table.
    :return: each enzyme's surface forms.
    """
    return {
        entity_id: [
            record.get("recommended_name") or "",
            record.get("ec_class") or "",
            *(record.get("synonyms") or []),
        ]
        for entity_id, record in table.items()
    }


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

    The strain's `taxon` name is left out: it names the *species*, so counting
    it would label bacterium mentions as strain evidence.

    :param table: the dump's `strains` table.
    :return: each strain's surface forms.
    """
    return {
        entity_id: with_abbreviated_genus(
            [
                *(record.get("designations") or []),
                *(
                    culture.get("strain_number") or ""
                    for culture in (record.get("cultures") or [])
                ),
            ]
        )
        for entity_id, record in table.items()
    }


def other_organism_forms(
    columns: Iterable[Mapping[str, str]],
) -> dict[str, list[str]]:
    """Other-organism ID -> pooled document names and their abbreviations.

    Pooled across every document on purpose: a document mentioning an organism
    it was not annotated with is exactly the case the abstain target exists
    for, and that mention is only recognizable from another document's naming
    of it. These names are the only ones harvested from running text, which is
    where an abbreviated genus is likeliest to be what the text actually says.

    :param columns: the per-document id -> name mappings, which is the shape
        both the TinyDB `documents` table and the split CSVs' column hold.
    :return: each other-organism's pooled names, with the abbreviations they
        imply.
    """
    names: collections.defaultdict[str, list[str]] = collections.defaultdict(
        list
    )
    for column in columns:
        for entity_id, name in column.items():
            if isinstance(name, str) and name:
                names[str(entity_id)].append(name)
    return {
        entity_id: with_abbreviated_genus(forms)
        for entity_id, forms in names.items()
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
