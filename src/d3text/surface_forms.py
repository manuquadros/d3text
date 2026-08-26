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

Deliberately a leaf: it imports nothing from `d3text` and nothing from
`brenda_references`, so building an index costs neither the BRENDA data layer
nor torch. The entity tables arrive as plain mappings, which is what the TinyDB
dump already is on disk.
"""

import collections
import json
import os
import pathlib
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

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
    "enzymes": "enz",
    "bacteria": "bac",
    "strains": "str",
    "other_organisms": "oth",
}
"""Entity-table name -> the prefix its numeric IDs wear in a corpus row.

Must agree with `d3text.datasets.brenda.BRENDA_SCHEMA`, which is where the
corpus side declares the same thing; `tests/test_surface_forms.py` pins the two
together rather than importing the schema here, since reaching it would drag
the whole BRENDA data layer into a leaf.
"""

_ENTITY_TABLE_KEY = b'"enzymes": {"'
_TAIL_SEARCH_BYTES = 256 * 1024 * 1024

_WORD = re.compile(r"[^\W_]+")


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

        The hygiene deletions are judged against this: dropping a form is only
        safe if the entities it named are reachable by some other form.
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

    None covers the four reasons a form carries no ID: it is too short, it
    tokenizes to nothing, it is longer than the sweep's widest window, or it is
    a bare `PLACEHOLDER_FORMS` entry.
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


def bacteria_forms(table: Mapping[str, Any]) -> dict[str, list[str]]:
    """Bacterium ID -> organism name and its LPSN synonyms."""
    return {
        entity_id: [
            record.get("organism") or "",
            *(record.get("synonyms") or []),
        ]
        for entity_id, record in table.items()
    }


def strain_forms(table: Mapping[str, Any]) -> dict[str, list[str]]:
    """Strain ID -> designations and culture-collection numbers.

    The strain's `taxon` name is deliberately left out: it names the
    *species*, so counting it as a strain mention would label bacterium
    mentions as strain evidence.
    """
    return {
        entity_id: [
            *(record.get("designations") or []),
            *(
                culture.get("strain_number") or ""
                for culture in (record.get("cultures") or [])
            ),
        ]
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
