"""BRENDA entity IDs paired with identifiers no BRENDA dictionary produced.

Nothing here resolves anything: a script builds the table where the outside
resource lives, and the evaluation path and CI read only what it emitted. The
namespace is recorded inside the file, because a taxid table and an EC table
are both `entity_id -> string`, so loading one where the other was meant
raises nothing at all. See the evaluation page of the documentation.
"""

import os
import pathlib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

NCBI_TAXID = "ncbi_taxid"
"""Namespace of NCBI taxonomy identifiers, as S800 annotators assigned them."""

EC_NUMBER = "ec_number"
"""Namespace of Enzyme Commission numbers, as the ENZYME nomenclature lists
them."""

_NAMESPACE_KEY = "namespace"
_COLUMNS = ("entity_id", "external_id", "source")
_SEPARATOR = "\t"


@dataclass(frozen=True, slots=True)
class ExternalMention:
    """One span an annotator outside this project marked, and its identifier.

    The offsets are half-open, like every other span in this package, whatever
    convention the corpus file on disk uses. `external_id` is None where the
    outside authority named none — a corpus that marks spans without naming
    them, or a name its nomenclature does not hold — which keeps the span in
    the coverage denominator instead of dropping it out of the arithmetic.
    """

    document: str
    start: int
    end: int
    surface: str
    external_id: str | None


@dataclass(frozen=True, slots=True)
class BridgeRow:
    """One entity's outside identifier, and how the pairing was made.

    `source` is free-form rather than an enumeration, so that adding a bridge
    does not force a format change on the tables already written.
    """

    entity_id: str
    external_id: str
    source: str


@dataclass(frozen=True, slots=True)
class IdentifierBridge:
    """An entity table read in both directions, and its namespace.

    `by_external` is a set per identifier rather than a single entity because
    BRENDA curates the same taxon more than once — two rows for one species,
    each with its own ID — and collapsing them would silently pick one.
    `sole_entity` is where that matters.
    """

    namespace: str
    by_entity: Mapping[str, str]
    by_external: Mapping[str, frozenset[str]]
    sources: Mapping[str, str]

    @classmethod
    def from_rows(
        cls, namespace: str, rows: Iterable[BridgeRow]
    ) -> "IdentifierBridge":
        """Index `rows` both ways.

        :param namespace: the authority the identifiers belong to.
        :param rows: the pairings to index.
        :return: the bridge, indexed in both directions.
        :raises ValueError: if an entity appears twice, which would leave the
            forward direction depending on row order.
        """
        by_entity: dict[str, str] = {}
        sources: dict[str, str] = {}
        by_external: dict[str, set[str]] = {}
        for row in rows:
            if row.entity_id in by_entity:
                raise ValueError(
                    f"{row.entity_id!r} carries two {namespace} identifiers: "
                    f"{by_entity[row.entity_id]!r} and {row.external_id!r}"
                )
            by_entity[row.entity_id] = row.external_id
            sources[row.entity_id] = row.source
            by_external.setdefault(row.external_id, set()).add(row.entity_id)

        return cls(
            namespace=namespace,
            by_entity=by_entity,
            by_external={
                external: frozenset(entities)
                for external, entities in by_external.items()
            },
            sources=sources,
        )

    def external_id(self, entity_id: str) -> str | None:
        """`entity_id`'s outside identifier, or None if the table has none.

        :param entity_id: the BRENDA entity to look up.
        :return: the identifier, or None if the table pairs it with nothing.
        """
        return self.by_entity.get(entity_id)

    def entity_ids(self, external_id: str) -> frozenset[str]:
        """Every entity the table pairs with `external_id`.

        :param external_id: the outside authority's identifier.
        :return: the entities carrying it, empty if none do.
        """
        return self.by_external.get(external_id, frozenset())

    def sole_entity(self, external_id: str) -> str | None:
        """The one entity carrying `external_id`, or None if not exactly one.

        The selection is made entirely on the gold side; keeping the spans the
        linker resolves uniquely would make its own answer the gold.

        :param external_id: the outside authority's identifier.
        :return: the sole entity, or None if none or several carry it.
        """
        entities = self.entity_ids(external_id)
        if len(entities) != 1:
            return None
        return next(iter(entities))

    def __len__(self) -> int:
        return len(self.by_entity)


def write_bridge(
    path: str | os.PathLike[str], namespace: str, rows: Iterable[BridgeRow]
) -> int:
    """Write `rows` as a bridge table and return how many landed.

    Sorted, so rebuilding the table against a refreshed resource diffs what
    actually changed rather than the dictionary's iteration order.

    :param path: where to write the table.
    :param namespace: the authority the identifiers belong to.
    :param rows: the pairings to write.
    :return: how many rows landed.
    :raises ValueError: if a field carries the separator or a newline, which
        would produce a file that reads back as different rows than it holds.
    """
    listed = sorted(
        rows, key=lambda row: (row.entity_id, row.external_id, row.source)
    )
    for row in listed:
        for field in (row.entity_id, row.external_id, row.source):
            if _SEPARATOR in field or "\n" in field:
                raise ValueError(
                    f"{field!r} carries a separator or newline, so the row "
                    f"for {row.entity_id!r} cannot be written as a table"
                )

    lines = [
        _SEPARATOR.join((_NAMESPACE_KEY, namespace)),
        _SEPARATOR.join(_COLUMNS),
        *(
            _SEPARATOR.join((row.entity_id, row.external_id, row.source))
            for row in listed
        ),
    ]
    pathlib.Path(path).write_text("\n".join(lines) + "\n", encoding="utf8")
    return len(listed)


def load_bridge(
    path: str | os.PathLike[str], expect: str | None = None
) -> IdentifierBridge:
    """Read a bridge table written by `write_bridge`.

    :param path: the table to read.
    :param expect: The namespace the caller is scoring against. Given, a table
        recording a different one is refused rather than silently scored — the
        failure it prevents is arithmetic on identifiers from two authorities,
        which produces a number rather than an error.
    :return: the bridge, indexed in both directions.
    :raises ValueError: on a malformed header, a malformed row, or a namespace
        other than `expect`.
    """
    lines = pathlib.Path(path).read_text(encoding="utf8").splitlines()
    if len(lines) < 2:
        raise ValueError(f"{path} carries no bridge header")

    header = lines[0].split(_SEPARATOR)
    if len(header) != 2 or header[0] != _NAMESPACE_KEY:
        raise ValueError(
            f"{path} opens with {lines[0]!r}, not a "
            f"{_NAMESPACE_KEY!r} declaration"
        )
    namespace = header[1]
    if expect is not None and namespace != expect:
        raise ValueError(
            f"{path} records {namespace!r} identifiers, but {expect!r} were "
            f"asked for: the two name different things"
        )

    if tuple(lines[1].split(_SEPARATOR)) != _COLUMNS:
        raise ValueError(
            f"{path} declares columns {lines[1]!r}, expected "
            f"{_SEPARATOR.join(_COLUMNS)!r}"
        )

    rows = []
    for number, line in enumerate(lines[2:], start=3):
        if not line:
            continue
        fields = line.split(_SEPARATOR)
        if len(fields) != len(_COLUMNS):
            raise ValueError(
                f"{path}:{number} has {len(fields)} fields, expected "
                f"{len(_COLUMNS)}: {line!r}"
            )
        rows.append(BridgeRow(*fields))

    return IdentifierBridge.from_rows(namespace, rows)


__all__ = [
    "EC_NUMBER",
    "NCBI_TAXID",
    "BridgeRow",
    "ExternalMention",
    "IdentifierBridge",
    "load_bridge",
    "write_bridge",
]
