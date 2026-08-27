"""BRENDA's surface forms, indexed by the entity IDs they name.

Distant supervision needs the inverse of the entity tables: not "what is this
entity called" but "which entities could this string be". `build_index` is that
inverse, and `d3text.token_labels` is its only intended reader.

**Exact lookup, not fuzzy scoring.** `models.dict_tagger.Vocab` already matches
surface forms, and it is the wrong tool at this scale: it scores a query against
every term in a length band, which is ~50 s per fulltext over the ~160k forms
BRENDA carries, and its cutoff was calibrated against a scorer that no longer
exists. A false hit here is not a wrong prediction but a *silently mislabelled
training token*, so the trade this module wants is the opposite one — cheap and
literal. What it keeps from `dict_tagger` is the part that is a decision rather
than an algorithm: `is_symbol_like`, which now lives here because the case
policy is a property of the dictionary and both readers must not drift apart on
it.

The index is keyed by the *words* of a form rather than by the form itself, so
`D-3-hydroxybutyrate dehydrogenase` and `D 3 hydroxybutyrate dehydrogenase`
reach the same entry and no hyphenation convention has to be modelled.

Deliberately a leaf: the only `d3text` module it imports is `d3text.schema`,
which is itself a leaf, and it imports nothing from `brenda_references`, so
building an index costs neither the BRENDA data layer nor torch. The entity
tables arrive as plain mappings, which is what the TinyDB dump already is on
disk.
"""

import collections
import json
import os
import pathlib
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from wordfreq import zipf_frequency

from d3text.schema import BRENDA_SCHEMA

MIN_FORM_LENGTH = 3
"""Shortest form that may carry an ID.

One- and two-character forms are almost all element symbols, figure labels and
units; `CO` names cholesterol oxidase in BRENDA and carbon monoxide everywhere
else, and no amount of case sensitivity separates those.
"""

MAX_FORM_WORDS = 8
"""Longest form, in words. Also the widest window the sweep has to try."""

SYMBOL_MAX_LENGTH = 5
"""Length at or below which a surface form is read as a symbol, not a name."""

COMMON_WORD_ZIPF = 3.0
"""Zipf frequency above which a one-word case-folded form names nothing.

BRENDA registers ordinary English as strain designations — `sensitive`,
`original`, `yielding`, `hybrid`, `aerobic` — and as place and surnames:
`california`, `shanghai`, `berlin`, `johnson`. Each is long enough to clear
`MIN_FORM_LENGTH` and lowercase enough to fold, so neither the length bar nor
the case policy sees them, and `sensitive` alone then claims a strain mention
in a quarter of the corpus.

Frequency is the discriminating feature because the two populations barely
overlap: of 4,190 one-word folded keys in the full index only 431 register in
general English at all, the other 90% being technical names general text has
no use for. 3.0 is where the two bands meet — the bacterial genera sit just
under it (`escherichia` 2.63, `pseudomonas` 2.59, `bacillus` 2.70) and the
ordinary words just over (`aerobic` 3.19, `yielding` 3.40, `hybrid` 4.11).
Measured over the whole dictionary this drops 90 keys of 160,109 and removes
1.8 spurious document-firings per document.

The one taxonomic casualty is `salmonella` (3.09), and it is a cheap one: the
bare genus fires on the same documents its binomials do, so the entity is
still found by `Salmonella enterica` and the genus-alone key was double
counting a single mention.

**Not a replacement for `PLACEHOLDER_FORMS`.** General frequency cannot see a
noun that is common only in this literature: `plasmid` (2.68), `protease`
(2.78) and `constitutive` (2.66) all pass this guard and name no particular
entity. The two rules cover different populations and both are needed.
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

`More` is BRENDA's curation marker for "this enzyme has further entries
elsewhere". It is registered as a synonym of 1,123 separate enzymes and it is
an ordinary English word, so every occurrence of it in running text would
resolve to a thousand entities at once. The rest are category nouns: a mention
of "plants" links to no organism, and `protease` is the one that survives the
symbol/descriptive split, since it is long and lowercase and so folds
legitimately.

Only the *bare* form goes. A form is dropped when it is one word and that word
is in this set, so `alkaline protease` and `Bacillus strain 168` keep their
IDs — which is the "require a modifier" reading of the same rule.
"""

BRENDA_PREFIXES: Mapping[str, str] = {
    entity_type.name: entity_type.prefix
    for entity_type in BRENDA_SCHEMA.entity_types
}
"""Entity-table name -> the prefix its numeric IDs wear in a corpus row.

Read off the schema rather than restated, because a prefix that disagrees with
the corpus's spelling does not fail — it produces an index whose keys no gold
set can ever match, and every mention it finds is then labelled as belonging to
no annotated entity.
"""

_ENTITY_TABLE_KEY = b'"enzymes": {"'
_TAIL_SEARCH_BYTES = 256 * 1024 * 1024

_WORD = re.compile(r"[^\W_]+")

_BINOMIAL_GENUS = re.compile(r"^[A-Z][a-z]+(?= [a-z]{2})")
"""A genus opening a binomial: capitalized word, then a lowercase epithet.

The lookahead is the guard: ``DSM 20745`` and ``ATCC 25922`` open with no
lowercase epithet, ``Candidatus Foo`` capitalizes its second word, and an
already-abbreviated ``E. coli`` has no lowercase run after its initial — none
of them match, so none gets a nonsense abbreviation.
"""


def form_words(text: str) -> list[str]:
    """The alphanumeric runs of `text`, in order.

    Underscore is excluded from the class deliberately: `\\w` admits it, and a
    gene name written `pyr_C` should tokenize the way `pyr-C` does.
    """
    return [match.group() for match in _WORD.finditer(text)]


def form_key(words: Sequence[str]) -> str:
    """The lookup key for a word sequence."""
    return " ".join(words)


def is_symbol_like(term: str) -> bool:
    """Whether case is load-bearing for `term`.

    Case is the only feature separating the enzyme symbol `FOR` from the
    English word `for`, `ARE` from `are`, `HAS` from `has`; all three are real
    BRENDA entities, so folding case away over the whole vocabulary trades a
    handful of recovered variants for a match in nearly every sentence. Two
    shapes carry that risk: a short form, and one with a capital past its
    first character (`MMP-3`, `HerE`, `CelL`) — the initial capital alone is
    just a sentence or a genus and says nothing.

    Descriptive names (`catalase`, `cytochrome c oxidase`) collide with no
    English word, so they are the population that can afford to fold.
    """

    return len(term) <= SYMBOL_MAX_LENGTH or any(
        character.isupper() for character in term[1:]
    )


def is_common_word(word: str) -> bool:
    """Whether general English uses `word` too often for it to name anything.

    Asked only of forms that are a single word *and* have already been judged
    descriptive enough to fold case, which is what keeps it safe. A symbol
    keeps its case and is therefore never compared against the English word
    it shares letters with — `FOR` the enzyme survives this while `for` was
    never a key to begin with — and a multi-word form is exempt because the
    modifier is what makes it specific, exactly as `PLACEHOLDER_FORMS` reads
    `alkaline protease`.
    """
    return zipf_frequency(word.lower(), "en") >= COMMON_WORD_ZIPF


@dataclass(frozen=True, slots=True)
class SurfaceFormIndex:
    """Surface form -> the entity IDs that form could name.

    Two tables rather than one because the case policy is per form, not per
    index: `exact` is keyed by the form's words as written, `folded` by the
    same words lowercased. `lookup` reads both and unions the answers, since a
    window can legitimately be a symbol of one entity and a descriptive name of
    another, and choosing between them at match time would be a guess.
    """

    exact: Mapping[str, frozenset[str]]
    folded: Mapping[str, frozenset[str]]
    max_words: int
    exact_first_words: frozenset[str]
    folded_first_words: frozenset[str]

    def lookup(self, words: Sequence[str]) -> frozenset[str]:
        """Every entity ID some form of which is exactly `words`."""
        key = form_key(words)
        return self.exact.get(key, frozenset()) | self.folded.get(
            key.lower(), frozenset()
        )

    def may_start(self, word: str) -> bool:
        """Whether any form begins with `word`.

        The sweep asks this once per position so that the overwhelming
        majority of tokens — ordinary prose — cost two set lookups rather than
        `MAX_FORM_WORDS` window joins.
        """
        return (
            word in self.exact_first_words
            or word.lower() in self.folded_first_words
        )

    @property
    def entity_ids(self) -> frozenset[str]:
        """Every entity the index can still reach.

        `PLACEHOLDER_FORMS` is judged against this: dropping `More` is only
        safe because each of the 1,123 enzymes it stood in for keeps a real
        name.

        `COMMON_WORD_ZIPF` is deliberately **not**, and the difference is the
        point. It costs 56 entities their last key, 52 of them strains
        registered under nothing but an ordinary English word. Keeping such a
        key to preserve reachability is the trade run backwards: the entity is
        not thereby findable, since every occurrence of `sensitive` in the
        literature would answer to it, and the mentions it manufactures are
        spread across the whole corpus rather than confined to the one entity
        lost. A name that names everything names nothing.
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

    None covers the five reasons a form carries no ID: it is too short, it
    tokenizes to nothing, it is longer than the sweep's widest window, it is
    a bare `PLACEHOLDER_FORMS` entry, or it is one ordinary English word.

    The frequency guard is asked last, and only of the folding branch, because
    it is the branch's own premise that decides whether the question is even
    meaningful: a form that keeps its case is separated from its English
    homograph by the case, so `FOR` must survive a test `for` would fail.
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

    Prefixed because that is the spelling the corpus uses: an entity is
    `enz3494` in a split frame's `entities` column and `"3494"` in
    `documents.json`, and a label that has to be compared against a document's
    gold set is only useful in the former.
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
    )


def enzyme_forms(table: Mapping[str, Any]) -> dict[str, list[str]]:
    """Enzyme ID -> recommended name, EC number and synonyms."""
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

    The same genus -> initial-plus-dot convention as `abbreviate_bacteria` in
    `brenda_references.utils`, restated here rather than imported because this
    module is a leaf and that one is not — and guarded, where that one is not,
    to forms actually opening with a binomial, so a culture-collection number
    never comes back mangled.
    """
    stripped = form.strip()
    genus = _BINOMIAL_GENUS.match(stripped)
    if genus is None:
        return None
    return f"{stripped[0]}.{stripped[genus.end() :]}"


def with_abbreviated_genus(forms: Iterable[str]) -> list[str]:
    """`forms`, each binomial-opening one followed by its abbreviation.

    The dictionary's bacterial gap in one number: only 37% of BRENDA's
    bacteria carry any synonym at all (median 0), so the form running text
    actually uses — `E. coli`, `B. subtilis` — is usually absent while the
    full binomial is present. Generating the abbreviation from the binomial
    closes that gap without waiting on LPSN; without it, a measurement of the
    linker measures the dictionary instead. Genus initials collide across
    genera, which the index absorbs the way it absorbs every shared form: the
    key reaches both entity sets.
    """
    expanded: list[str] = []
    for form in forms:
        expanded.append(form)
        abbreviated = abbreviated_genus(form)
        if abbreviated is not None:
            expanded.append(abbreviated)
    return expanded


def bacteria_forms(table: Mapping[str, Any]) -> dict[str, list[str]]:
    """Bacterium ID -> organism name, LPSN synonyms, and their abbreviations."""
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

    The strain's `taxon` name is deliberately left out: it names the
    *species*, so counting it as a strain mention would label bacterium
    mentions as strain evidence. A designation that itself opens with the
    binomial (`Escherichia coli K-12`) also contributes its genus-abbreviated
    variant, which is the strain-qualified form running text uses.
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
    """Other-organism ID -> every name any document gave it.

    `documents.json` has no `other_organisms` table — the four it carries are
    `documents`, `enzymes`, `bacteria` and `strains` — so the only place these
    names exist is inline, one document at a time, in the id -> name mapping
    each document carries. `columns` is an iterable of exactly those mappings,
    which is the shape both the TinyDB `documents` table and the split CSVs'
    `other_organisms` column hold.

    Pooled across every document on purpose: a document that mentions an
    organism it was not annotated with is the case the `ignore` target exists
    for, and that mention can only be recognized from some *other* document's
    naming of it.
    """
    names: collections.defaultdict[str, list[str]] = collections.defaultdict(
        list
    )
    for column in columns:
        for entity_id, name in column.items():
            if isinstance(name, str) and name:
                names[str(entity_id)].append(name)
    return dict(names)


def brenda_surface_forms(
    tables: Mapping[str, Mapping[str, Any]],
    other_organisms: Iterable[Mapping[str, str]] = (),
    prefixes: Mapping[str, str] = BRENDA_PREFIXES,
) -> dict[str, list[str]]:
    """Prefixed entity ID -> surface forms, over all four ID namespaces.

    A table absent from `tables` contributes nothing rather than raising: the
    tail-parse route in `load_entity_tables` cannot reach `documents`, and a
    caller that only wants enzymes should not have to fabricate the rest.
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

    The shipped `documents.json` is 1.1 GB of document records followed by the
    three entity tables, which are its last ~8 MB. A dump larger than
    `_TAIL_SEARCH_BYTES` is therefore parsed off its tail — which yields
    `enzymes`, `bacteria` and `strains` but **not** `documents`, so an `oth`
    namespace built from that route has to get its names from the split CSVs
    instead. Anything smaller is read whole, which is what the tracked test
    fixture and any hand-built dump take.

    :raises ValueError: if a large dump carries no entity table in its tail,
        which is what a differently ordered dump looks like from here.
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
