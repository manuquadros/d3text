"""Expasy ENZYME: EC numbers for enzyme names, assigned outside BRENDA.

enzymeNER marks spans and no identifiers, so this nomenclature is what turns a
span into gold. It is a dictionary, which makes the enzyme evaluation weaker
than a corpus of hand-assigned identifiers — its own resolution errors are
indistinguishable from the linker's. See the evaluation page of the
documentation.
"""

import collections
import os
import pathlib
import re
import unicodedata
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace

from d3text.identifier_bridge import ExternalMention

NOMENCLATURE = "enzyme.dat"
"""The flat file, as Expasy publishes it."""

ENCODING = "latin-1"
"""The file's encoding; it is not valid UTF-8."""

NOT_A_NAME = ("Deleted entry", "Transferred entry")
"""`DE` texts that report a record's status instead of naming an enzyme."""

_TAG = slice(0, 2)
_CONTENT = slice(5, None)
_RECORD_END = "//"
_IDENTIFIER = "ID"
_NAME_TAGS = ("DE", "AN")

_GREEK = {
    "\N{GREEK SMALL LETTER ALPHA}": "alpha",
    "\N{GREEK CAPITAL LETTER ALPHA}": "alpha",
    "\N{GREEK SMALL LETTER BETA}": "beta",
    "\N{GREEK CAPITAL LETTER BETA}": "beta",
    "\N{LATIN SMALL LETTER SHARP S}": "beta",
    "\N{GREEK SMALL LETTER GAMMA}": "gamma",
    "\N{GREEK CAPITAL LETTER GAMMA}": "gamma",
    "\N{GREEK SMALL LETTER DELTA}": "delta",
    "\N{GREEK CAPITAL LETTER DELTA}": "delta",
    "\N{GREEK SMALL LETTER EPSILON}": "epsilon",
    "\N{GREEK SMALL LETTER ZETA}": "zeta",
    "\N{GREEK SMALL LETTER ETA}": "eta",
    "\N{GREEK SMALL LETTER THETA}": "theta",
    "\N{GREEK SMALL LETTER KAPPA}": "kappa",
    "\N{GREEK SMALL LETTER LAMDA}": "lambda",
    "\N{GREEK SMALL LETTER MU}": "mu",
    "\N{GREEK SMALL LETTER PI}": "pi",
    "\N{GREEK SMALL LETTER RHO}": "rho",
    "\N{GREEK SMALL LETTER SIGMA}": "sigma",
    "\N{GREEK SMALL LETTER TAU}": "tau",
    "\N{GREEK SMALL LETTER PHI}": "phi",
    "\N{GREEK SMALL LETTER CHI}": "chi",
    "\N{GREEK SMALL LETTER PSI}": "psi",
    "\N{GREEK SMALL LETTER OMEGA}": "omega",
    "\N{GREEK CAPITAL LETTER OMEGA}": "omega",
}
"""Greek letters Expasy spells out in Latin and the corpus writes as letters."""

_PUNCTUATION = str.maketrans(
    {
        **dict.fromkeys("‐‑‒–—―−", "-"),
        **dict.fromkeys("‘’′", "'"),
        **dict.fromkeys("“”", '"'),
    }
)

_WHITESPACE = re.compile(r"\s+")


def normalize(name: str) -> str:
    """An enzyme name reduced to the key both resources are looked up under.

    Hyphens become spaces, which is the whole of what this buys beyond case
    and Unicode folding; a key two EC numbers then share stays ambiguous
    rather than resolving to one of them.

    :param name: a name as either resource spells it.
    :return: the lookup key, empty if the name reduces to nothing.
    """
    folded = unicodedata.normalize("NFKC", name)
    folded = "".join(_GREEK.get(character, character) for character in folded)
    folded = folded.translate(_PUNCTUATION).lower().replace("-", " ")
    return _WHITESPACE.sub(" ", folded).strip(" .")


def _joined(parts: Sequence[str]) -> str:
    """Continuation lines rejoined the way the flat file wrapped them."""
    text = ""
    for part in parts:
        if not text:
            text = part
        elif text.endswith("-"):
            text += part
        else:
            text = f"{text} {part}"
    return text


def parse_records(lines: Iterable[str]) -> Iterator[tuple[str, list[str]]]:
    """The file's records as EC number and the names it lists for it.

    A `DE`/`AN` text runs until the line that ends it with a full stop, so a
    wrapped name read line by line is a name neither resource holds.

    :param lines: the flat file's lines.
    :return: one `(ec_number, names)` pair per record carrying a name, with
        `NOT_A_NAME` statuses dropped.
    """
    ec_number = ""
    names: list[str] = []
    pending: list[str] = []
    pending_tag = ""

    def flush() -> None:
        if pending:
            names.append(_joined(pending).rstrip("."))
            pending.clear()

    for line in lines:
        tag, content = line[_TAG], line[_CONTENT].strip()
        if tag != pending_tag:
            flush()
            pending_tag = tag
        if tag == _IDENTIFIER:
            ec_number, names = content, []
        elif tag in _NAME_TAGS:
            pending.append(content)
            text = _joined(pending)
            if text.endswith("."):
                names.append(text[:-1])
                pending.clear()
        elif tag == _RECORD_END and ec_number:
            named = [name for name in names if not name.startswith(NOT_A_NAME)]
            if named:
                yield ec_number, named
            ec_number, names = "", []


@dataclass(frozen=True, slots=True)
class EnzymeNomenclature:
    """Normalized enzyme name -> the EC numbers Expasy gives it.

    A set per name rather than one EC because `luciferase` names four
    reactions; `sole_ec` is where that matters, and it is what selects the
    judged subset.
    """

    by_name: Mapping[str, frozenset[str]]

    def ec_numbers(self, name: str) -> frozenset[str]:
        """Every EC number `name` could denote.

        :param name: the name as written, normalized on the way in.
        :return: the EC numbers, empty if the nomenclature does not hold it.
        """
        return self.by_name.get(normalize(name), frozenset())

    def sole_ec(self, name: str) -> str | None:
        """The one EC number `name` denotes, or None if not exactly one.

        :param name: the name as written.
        :return: the EC number, or None if the nomenclature holds none or
            several for it.
        """
        found = self.ec_numbers(name)
        if len(found) != 1:
            return None
        return next(iter(found))

    def assign(
        self, mentions: Iterable[ExternalMention]
    ) -> list[ExternalMention]:
        """Stamp each span with the EC numbers its surface form denotes.

        A span denoting several is emitted once per EC, so the scorer counts
        it as gold-side ambiguity rather than the caller silently picking one;
        a span the nomenclature does not hold keeps a `None` identifier and
        stays in the coverage denominator.

        :param mentions: the corpus's spans, which carry no identifier.
        :return: the spans with their EC numbers, longer than `mentions`
            wherever a surface form denotes more than one.
        """
        stamped: list[ExternalMention] = []
        for mention in mentions:
            found = self.ec_numbers(mention.surface)
            if not found:
                stamped.append(replace(mention, external_id=None))
                continue
            stamped.extend(
                replace(mention, external_id=ec_number)
                for ec_number in sorted(found)
            )
        return stamped

    @property
    def unambiguous(self) -> int:
        """How many names denote exactly one EC number.

        :return: the count, which is the population `sole_ec` can answer for.
        """
        return sum(1 for found in self.by_name.values() if len(found) == 1)

    def __len__(self) -> int:
        return len(self.by_name)


def load_nomenclature(
    path: str | os.PathLike[str],
) -> EnzymeNomenclature:
    """Read `enzyme.dat` and index its names.

    :param path: the flat file to read.
    :return: the nomenclature, indexed by normalized name.
    :raises FileNotFoundError: if the file is missing.
    """
    lines = pathlib.Path(path).read_text(encoding=ENCODING).splitlines()
    index: collections.defaultdict[str, set[str]] = collections.defaultdict(set)
    for ec_number, names in parse_records(lines):
        for name in names:
            key = normalize(name)
            if key:
                index[key].add(ec_number)

    return EnzymeNomenclature(
        by_name={key: frozenset(found) for key, found in index.items()}
    )


__all__ = [
    "ENCODING",
    "NOMENCLATURE",
    "NOT_A_NAME",
    "EnzymeNomenclature",
    "load_nomenclature",
    "normalize",
    "parse_records",
]
