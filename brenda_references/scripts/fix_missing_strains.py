import asyncio
import itertools
import logging
import math
from collections.abc import Callable, Mapping

from aiotinydb import AIOTinyDB
from aiotinydb.storage import AIOJSONStorage
from tinydb import where
from tqdm import tqdm

from apiadapters.straininfo import (
    AsyncStrainInfoAdapter,
    normalize_strain_names,
)
from apiadapters.straininfo import Strain as StrainInfoStrain
from brenda_references.collection_numbers import is_collection_number
from brenda_references.config import config
from brenda_references.utils import CachingMiddleware
from d3types import Strain

logger = logging.getLogger(__name__)


def known_designations(strains: Mapping[int, Strain]) -> dict[str, int]:
    """Every normalized designation in `strains`, mapped to its BRENDA id.

    Mirrors `StrainInfoAdapterBase.retrieve_strain_models`'s own
    `known_names` build, so a name reaches the same id whichever loop
    consumes it.

    :param strains: the batch being resolved, keyed by BRENDA strain id.
    :return: normalized designation -> BRENDA strain id.
    """
    return {
        name: ix
        for ix, model in strains.items()
        for name in normalize_strain_names(model.designations)
    }


def matching_designations(
    entry: StrainInfoStrain, known_names: Mapping[str, int]
) -> frozenset[str]:
    """Names on `entry` that also name a BRENDA strain in `known_names`.

    :param entry: one record StrainInfo returned for the batch.
    :param known_names: designation -> BRENDA strain id, from
        `known_designations`.
    :return: the overlap, empty if StrainInfo returned `entry` for a name
        none of the batch's designations produced.
    """
    names = entry.designations | frozenset(
        culture.strain_number for culture in entry.cultures
    )
    return frozenset(name for name in names if name in known_names)


def gated_pairing(
    entry: StrainInfoStrain,
    known_names: Mapping[str, int],
    is_collection_number: Callable[[str], bool] = is_collection_number,
) -> tuple[int, str] | None:
    """The BRENDA strain `entry` may join to, admitting a registry id only.

    `retrieve_strain_models` (in the sibling `apiadapters` package, not
    editable here) pairs a returned record with whichever BRENDA strain
    shares its *first* matching designation or culture number, with no
    organism constraint — a short or generic designation reaches whichever
    record StrainInfo happened to return first, which is very often a
    different species. Restricting admission to a designation shaped like a
    real culture-collection accession keeps only the pairings that are
    unambiguous by construction.

    :param entry: one record StrainInfo returned for the batch.
    :param known_names: designation -> BRENDA strain id, from
        `known_designations`.
    :param is_collection_number: the shape predicate; overridable for tests.
    :return: `(BRENDA strain id, the designation that matched)`, or None if
        no admissible designation names both a BRENDA strain and `entry`.
    """
    for name in sorted(matching_designations(entry, known_names)):
        if is_collection_number(name):
            return known_names[name], name
    return None


async def run() -> None:  # noqa: D103
    async with (
        AIOTinyDB(
            config["documents"],
            storage=CachingMiddleware(AIOJSONStorage),
        ) as docdb,
        AsyncStrainInfoAdapter() as straininfo,
    ):
        straininfo.storage = docdb

        batch_size = 100
        total = math.ceil(
            docdb.table("strains").count(where("id") == None) / batch_size,
        )
        joined = unjoinable = 0
        for batch in tqdm(
            itertools.batched(
                docdb.table("strains").search(where("id") == None),
                batch_size,
            ),
            total=total,
        ):
            strains = {doc.doc_id: Strain.model_validate(doc) for doc in batch}
            known_names = known_designations(strains)

            ids = await straininfo.get_strain_ids(list(known_names.keys()))
            records = await straininfo.get_strain_data(ids)

            accepted: dict[int, StrainInfoStrain] = {}
            for entry in records:
                pairing = gated_pairing(entry, known_names)
                if pairing is not None:
                    accepted[pairing[0]] = entry.model_copy()
                    joined += 1
                elif matching_designations(entry, known_names):
                    unjoinable += 1
                    logger.info(
                        "StrainInfo id %s retrieved but its matching "
                        "designation has no collection-number shape; "
                        "leaving %s unjoined.",
                        entry.id,
                        sorted(matching_designations(entry, known_names)),
                    )

            await asyncio.gather(
                *(
                    docdb.table("strains").update(
                        strain.model_dump(), doc_ids=[key]
                    )
                    for key, strain in accepted.items()
                ),
            )

        logger.info(
            "Joined %d strains on a collection-number-shaped designation; "
            "left %d retrieved records unjoined.",
            joined,
            unjoinable,
        )


def main() -> None:  # noqa: D103
    asyncio.run(run())
