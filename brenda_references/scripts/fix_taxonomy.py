"""Reclassify organisms in the `other_organisms` fields of the doc database.

When one of those organisms is a bacteria in the NCBI taxonomy, move it to the
bacteria field of the document and add an entry for it in the `bacteria` table
of the database.

Similarly, if one of those organisms is a bacterial strain, move it to the
strain field of the document and add an entry for it in the `strains` table.
Furthermore, extract the species part of the strain name, if there is one, and
make sure it is reflected both in the bacteria field of the document and on the
bacteria table of the database.
"""

import logging
from collections.abc import Mapping

from apiadapters.straininfo import StrainInfoAdapter
from brenda_references.docdb import BrendaDocDB
from d3types import Strain
from taxonomy import ncbitax
from tinydb.table import Document as TinyDBDoc
from tqdm import tqdm

logger = logging.getLogger(__name__)


def update_doc_bacteria(
    docdb: BrendaDocDB, doc: TinyDBDoc, bacname: str
) -> int:
    """Add `bacname` to `docdb` and to `doc`, under its existing or new id.

    `bacname` is added to `doc["bacteria"]` even when it is already one of
    the designations of an existing `docdb.bacteria` record — only the
    record itself is skipped in that case — since `doc["bacteria"]` is
    otherwise left with no reference to it at all.

    :return: the id `bacname` is stored under in `docdb.bacteria`.
    """
    bacid = docdb.insert_bacteria_record(bacname)

    if str(bacid) not in {str(key) for key in doc["bacteria"]}:
        doc["bacteria"][bacid] = bacname
        docdb.update_record(table="documents", fields=doc, doc_id=doc.doc_id)

    return bacid


def update_doc_strain(
    docdb: BrendaDocDB, doc: TinyDBDoc, strainname: str
) -> int:
    """Update `doc` in `docdb` with `strainname`, under its existing or new id.

    `strainname` is added as a new record if it is not one of the
    designations of an existing record in docdb.strains; either way, its id
    is appended to `doc["strains"]` if not already there.

    :return: the id `strainname` is stored under in `docdb.strains`.
    """
    match = docdb.strain_by_designation(strainname)

    if match is not None:
        docdb.add_strain_synonyms(doc_id=match.doc_id, synonyms={strainname})
        strainid = match.doc_id
    else:
        with StrainInfoAdapter() as si:
            model = si.retrieve_strain_models(
                {0: Strain(designations=frozenset({strainname}))}  # type: ignore[call-arg]
            )

        strainid = docdb.insert(table="strains", record=model[0].model_dump())

    if strainid not in doc["strains"]:
        doc["strains"].append(strainid)
        docdb.update_record(table="documents", fields=doc, doc_id=doc.doc_id)

    return strainid


def _remap_organism_relations(
    doc: TinyDBDoc,
    old_ids: set[int],
    new_bacteria_id: Mapping[int, int],
    new_strain_id: Mapping[int, int],
    prior_strains: set[int],
) -> None:
    """Rewrite relation arguments naming an id `old_ids` removes.

    Only organism-holding slots are touched: a HasEnzyme subject and a
    HasSpecies object. A HasEnzyme object is an `ec_class_id` and a
    HasSpecies subject is a strain id (`db.BRENDA.enzyme_relations`); either
    can equal a value in `old_ids` by coincidence rather than because it
    names the organism being reclassified, so neither is ever inspected
    here. `prior_strains` guards the same coincidence for a HasEnzyme
    subject: strain ids come from the `strains` table's own counter, so one
    can equal an `old_ids` value while already correctly naming a different,
    pre-existing strain — that subject is left alone rather than
    overwritten.

    :param doc: the document being fixed; `doc["relations"]` is mutated in
        place.
    :param old_ids: the `other_organisms` ids being removed this pass.
    :param new_bacteria_id: `old_ids` value -> the bacteria id it now lives
        under, for ids a bacterium was split off for.
    :param new_strain_id: `old_ids` value -> the strain id it now lives
        under, for ids a strain was split off for.
    :param prior_strains: `doc["strains"]` as it stood before this pass'
        reclassification loop ran.
    """
    for pair in doc["relations"].get("HasEnzyme", []):
        subject = pair["subject"]
        if subject in old_ids and subject not in prior_strains:
            if subject in new_strain_id:
                pair["subject"] = new_strain_id[subject]
            elif subject in new_bacteria_id:
                pair["subject"] = new_bacteria_id[subject]

    for pair in doc["relations"].get("HasSpecies", []):
        obj = pair["object"]
        if obj in old_ids and obj in new_bacteria_id:
            pair["object"] = new_bacteria_id[obj]


def fix_taxonomy(docdb: BrendaDocDB) -> None:
    """Make sure there are no bacteria in the other_organisms field."""
    docs = tuple(
        doc for doc in docdb.references if doc.get("other_organisms", {})
    )

    for doc in tqdm(docs):
        doc_id = doc.doc_id
        # Snapshotted before the reclassification loop below appends to
        # doc["strains"]: a subject already among these is a genuine
        # pre-existing strain, not this pass' own output, and must not be
        # treated as a numeric collision with an old other_organisms id.
        prior_strains: set[int] = set(doc["strains"])

        delete_from_other: set[str] = set()
        bacteria_names: dict[str, str] = {}
        strain_names: dict[str, str] = {}

        for _id, orgname in doc["other_organisms"].items():
            decomposed = ncbitax.decompose_name(orgname)

            if decomposed is None:
                logger.warning(
                    "decompose_name could not place %r (doc %s); "
                    "left in other_organisms",
                    orgname,
                    doc_id,
                )
                continue

            species, strain = decomposed.species, decomposed.strain

            if not species and not strain:
                logger.warning(
                    "decompose_name placed %r at the taxonomy root only "
                    "(doc %s); left in other_organisms",
                    orgname,
                    doc_id,
                )
                continue

            delete_from_other.add(_id)

            if species:
                if not strain and species not in orgname:
                    bacteria_names[_id] = orgname
                else:
                    bacteria_names[_id] = species

            if strain:
                strain_names[_id] = strain
            else:
                suffix = orgname.removeprefix(species or "").strip()
                if suffix:
                    strain_names[_id] = suffix

        # Cached by name, not just by id: two other_organisms ids in one doc
        # can decompose to the same bacterium or strain, and the second
        # lookup should reuse the first's id instead of re-querying LPSN or
        # StrainInfo for a name already resolved this pass.
        bacteria_id_cache: dict[str, int] = {}
        new_bacteria_id: dict[str, int] = {}
        for _id, name in bacteria_names.items():
            if name not in bacteria_id_cache:
                bacteria_id_cache[name] = update_doc_bacteria(docdb, doc, name)
            new_bacteria_id[_id] = bacteria_id_cache[name]

        strain_id_cache: dict[str, int] = {}
        new_strain_id: dict[str, int] = {}
        for _id, name in strain_names.items():
            if name not in strain_id_cache:
                strain_id_cache[name] = update_doc_strain(docdb, doc, name)
            new_strain_id[_id] = strain_id_cache[name]

        if delete_from_other:
            _remap_organism_relations(
                doc,
                old_ids={int(_id) for _id in delete_from_other},
                new_bacteria_id={
                    int(key): value for key, value in new_bacteria_id.items()
                },
                new_strain_id={
                    int(key): value for key, value in new_strain_id.items()
                },
                prior_strains=prior_strains,
            )
            docdb.update_record(
                table="documents",
                fields={
                    "relations": doc["relations"],
                    "other_organisms": {
                        k: v
                        for k, v in doc["other_organisms"].items()
                        if k not in delete_from_other
                    },
                },
                doc_id=doc_id,
            )


if __name__ == "__main__":
    with BrendaDocDB() as docdb:
        fix_taxonomy(docdb)
