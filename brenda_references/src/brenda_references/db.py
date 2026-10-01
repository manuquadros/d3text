"""The interface to the BRENDA database."""

import os
import re
from collections.abc import Iterable
from functools import lru_cache
from types import TracebackType
from typing import Any, NamedTuple, Self

from d3types import (
    EC,
    Bacteria,
    BaseEC,
    BaseOrganism,
    BaseReference,
    HasEnzyme,
    HasSpecies,
    Organism,
    StrainRef,
)
from rapidfuzz import fuzz, process
from sqlalchemy.engine import URL, Engine
from sqlalchemy.orm import declarative_base
from sqlalchemy.schema import Column
from sqlalchemy.sql.expression import Select
from sqlalchemy.types import Integer, String
from sqlmodel import Field, Session, SQLModel, create_engine, select
from taxonomy import ncbitax

from .config import config

Base = declarative_base()

BACTERIA_TAX_ID = 2


class Protein_Connect(SQLModel, table=True):  # type: ignore
    """Model mapping to the protein_connect table of brenda_conn"""

    __table_args__ = {"keep_existing": True}
    __tablename__ = "protein_connect"
    protein_connect_id: int = Field(
        primary_key=True,
        description="ID of the protein-organism connection in BRENDA.",
    )
    organism_id: int = Field(
        nullable=False,
        description="Reference to the organism taking part in the relation.",
    )
    ec_class_id: int = Field(
        nullable=False,
        description="Reference to the EC Class of the protein.",
    )
    protein_organism_strain_id: int | None = Field(
        description="Reference to a specific strain related to the protein, "
        "if available."
    )
    reference_id: int = Field(
        nullable=False,
        description="Reference to an article in which the connection"
        " is attested.",
    )


class _Reference(SQLModel, BaseReference, table=True):  # type: ignore
    """Model mapping to the `reference` table of brenda_conn"""

    __table_args__ = {"keep_existing": True}
    __tablename__ = "reference"
    reference_id: int = Field(primary_key=True)


class _Organism(SQLModel, BaseOrganism, table=True):  # type: ignore
    """Model mapping to the `organism` table of brenda_conn"""

    __table_args__ = {"keep_existing": True}
    __tablename__ = "organism"
    organism_id: int = Field(primary_key=True)


class _EC(SQLModel, BaseEC, table=True):  # type: ignore
    """Model mapping to the `ec_class` table of brenda_conn"""

    __table_args__ = {"keep_existing": True}
    __tablename__ = "ec_class"
    ec_class_id: int = Field(primary_key=True)


class _Protein(SQLModel, table=True):  # type: ignore
    """Model mapping to the `protein` table of brenda_conn"""

    __table_args__ = {"keep_existing": True}
    __tablename__ = "protein"
    protein_id: int = Field(primary_key=True)


class _Strain(Base):  # type: ignore
    """Model mapping to the `strain` table of brenda_conn"""

    __table_args__ = {"keep_existing": True}
    __tablename__ = "protein_organism_strain"

    id: int = Column("protein_organism_strain_id", Integer, primary_key=True)
    name: str = Column("organism_strain", String)


class EC_Synonyms_Connect(SQLModel, table=True):  # type: ignore
    """Model mapping to the `synonyms_connect` table of brenda_conn"""

    __table_args__ = {"keep_existing": True}
    __tablename__ = "synonyms_connect"
    synonyms_connect_id: int = Field(primary_key=True)
    ec_class_id: int
    synonyms_id: int
    reference_id: int


class EC_Synonyms(SQLModel, table=True):  # type: ignore
    """Model mapping to the `ec_synonyms` table of brenda_conn"""

    __table_args__ = {"keep_existing": True}
    __tablename__ = "synonyms"
    synonyms_id: int = Field(primary_key=True)
    synonyms: str


with open(config["sources"]["bacteria"], encoding="utf-8") as sl:
    bacteria = set(s.strip() for s in sl.readlines())


class BRENDA:
    def __init__(self):
        self.engine = get_engine()
        SQLModel.metadata = Base.metadata
        SQLModel.metadata.create_all(self.engine)
        self.session = Session(self.engine)

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.session.close()

    def references(self) -> Iterable[_Reference]:
        """Retrieve list of literature references in BRENDA."""
        query: Select = select(_Reference).execution_options(yield_per=64)
        return self.session.scalars(query)

    def count_references(self) -> int:
        return self.session.query(_Reference.reference_id).count()

    def enzyme_relations(self, reference_id: int) -> dict[str, Any]:
        """Return entities and relations attested in `reference_id`."""
        query = (
            select(Protein_Connect, _Organism, _EC, _Strain)
            .join(
                _Organism, Protein_Connect.organism_id == _Organism.organism_id
            )
            .join(_EC, Protein_Connect.ec_class_id == _EC.ec_class_id)
            .outerjoin(
                _Strain,
                Protein_Connect.protein_organism_strain_id == _Strain.id,
            )
            .where(Protein_Connect.reference_id == reference_id)
        )
        records = self.session.exec(query).fetchall()

        output: dict[str, Any] = {
            key: set()
            for key in ("enzymes", "bacteria", "strains", "other_organisms")
        }
        output["triples"] = {}

        for record in records:
            organism, no_activity_organism = clean_name(
                record._Organism, "organism"
            )

            if record._Strain:
                strain, no_activity_strain = clean_name(record._Strain, "name")

                if not no_activity_strain:
                    output["triples"].setdefault("HasEnzyme", set()).add(
                        HasEnzyme(
                            subject=strain.id, object=record._EC.ec_class_id
                        ),
                    )

                output["triples"].setdefault("HasSpecies", set()).add(
                    HasSpecies(subject=strain.id, object=organism.organism_id),
                )
                output["strains"].add(strain)
            else:
                if not no_activity_organism:
                    output["triples"].setdefault("HasEnzyme", set()).add(
                        HasEnzyme(
                            subject=organism.organism_id,
                            object=record._EC.ec_class_id,
                        ),
                    )

            if is_bacteria(organism.organism):
                output["bacteria"].add(
                    Bacteria.model_validate(organism, from_attributes=True),
                )
            else:
                output["other_organisms"].add(
                    Organism.model_validate(organism, from_attributes=True),
                )

            output["enzymes"].add(
                EC.model_validate(record._EC, from_attributes=True)
            )

        return output

    @lru_cache(maxsize=512)
    def ec_synonyms(self, ec_class_id: int) -> list[str]:
        """The synonyms BRENDA records for one EC class.

        :param ec_class_id: the EC class to look up.
        :return: its synonyms.
        """
        query = (
            select(EC_Synonyms.synonyms)
            .join_from(
                EC_Synonyms,
                EC_Synonyms_Connect,
                EC_Synonyms_Connect.synonyms_id == EC_Synonyms.synonyms_id,
            )
            .where(EC_Synonyms_Connect.ec_class_id == ec_class_id)
        )

        synonyms = self.session.exec(query).all()

        return synonyms


def get_engine() -> Engine:
    """Establish a connection to the BRENDA database.

    The server and the login are read from `BRENDA_HOST`, `BRENDA_USER` and
    `BRENDA_PASSWORD`. The host lives there rather than in `config.toml`
    because it names a private server and the config is shipped with the
    package.

    :return: the engine.
    :raises KeyError: if any of the three is unset.
    """
    try:
        host, user, password = (
            os.environ["BRENDA_HOST"],
            os.environ["BRENDA_USER"],
            os.environ["BRENDA_PASSWORD"],
        )
    except KeyError as err:
        err.add_note(
            "Please set the BRENDA_HOST, BRENDA_USER and BRENDA_PASSWORD"
            " environment variables"
        )
        raise

    db_conn_info = config["database"]
    url_object = URL.create(
        drivername=db_conn_info["backend"],
        host=host,
        database=db_conn_info["database"],
        username=user,
        password=password,
    )

    return create_engine(url_object)


class Classification(NamedTuple):
    """Whether a name is a bacterium, and whether NCBI lineage decided it.

    `by_lineage` is false when the name resolved to no taxid, so the bacteria
    name list decided.
    """

    bacterium: bool
    by_lineage: bool


def classify_organism(organism: str) -> Classification:
    """Decide whether `organism` is under NCBI Bacteria, and how.

    The name is resolved to a taxid directly, then through the species of its
    decomposed name. A name that resolves neither way is matched against the
    bacteria name list.

    :param organism: the organism name as BRENDA gives it.
    :return: the decision, with `by_lineage` false for a name-list match.
    """
    tax_id = ncbitax.resolve_any_tax_id(organism)

    if tax_id is None:
        decomposed = ncbitax.decompose_name(organism)
        if decomposed is not None and decomposed.species:
            tax_id = ncbitax.resolve_any_tax_id(decomposed.species)

    if tax_id is not None:
        return Classification(
            ncbitax.is_descendant(tax_id, BACTERIA_TAX_ID), by_lineage=True
        )

    _, ratio, _ = process.extract(
        organism, bacteria, scorer=fuzz.QRatio, limit=1
    )[0]

    return Classification(ratio > 90, by_lineage=False)


def is_bacteria(organism: str) -> bool:
    """Check whether `organism` is a taxon under NCBI Bacteria (taxid 2).

    Resolution order: `classify_organism`.

    :param organism: the organism name as BRENDA gives it.
    :return: whether it descends from Bacteria; archaea do not.
    """
    return classify_organism(organism).bacterium


def clean_name(
    model: SQLModel | _Strain,
    fieldname: str,
    pattern: str = "no activity (in|by) ",
) -> tuple[SQLModel, bool] | StrainRef:
    """Strip a pattern out of `fieldname` on an SQLModel.

    :param model: the model to update.
    :param fieldname: the field to clean up.
    :param pattern: regular expression matching the offending strings.
    :return: the updated model, and whether the pattern was found.
    """
    name, count = re.subn(rf"{pattern}", "", getattr(model, fieldname))

    if isinstance(model, _Strain):
        data = model.__dict__
        data[fieldname] = name
        return StrainRef(id=data["id"], name=data["name"]), bool(count)

    return model.copy(update={fieldname: name}), bool(count)
