"""
Provides the interface to the BRENDA database.

`brenda_references`
    Provides a list of article references from BRENDA.

`brenda_enzyme_relations`
    Retrieves all relation triples and their participating entities for a given
    reference.

`ec_synonyms`
    Retrieves all synonyms of a given EC Class, linked to the references where
    they are attested in the database.
"""

import os
import re
from collections.abc import Iterable
from functools import lru_cache
from types import TracebackType
from typing import Any, Self

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

from .config import config

Base = declarative_base()


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
        """For a given EC class, fetch a list of synonym, reference_id pairs."""
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

    Login information stored in the BRENDA_USER and BRENDA_PASSWORD environment
    variables.
    """
    try:
        user, password = (
            os.environ["BRENDA_USER"],
            os.environ["BRENDA_PASSWORD"],
        )
    except KeyError as err:
        err.add_note(
            "Please set the BRENDA_USER and BRENDA_PASSWORD",
            " environment variables",
        )
        raise

    db_conn_info = config["database"]
    url_object = URL.create(
        drivername=db_conn_info["backend"],
        host=db_conn_info["host"],
        database=db_conn_info["database"],
        username=user,
        password=password,
    )

    return create_engine(url_object)


def is_bacteria(organism: str) -> bool:
    """Check whether `organism` is the name of a bacteria."""
    _, ratio, _ = process.extract(
        organism, bacteria, scorer=fuzz.QRatio, limit=1
    )[0]

    return ratio > 90


def clean_name(
    model: SQLModel | _Strain,
    fieldname: str,
    pattern: str = "no activity (in|by) ",
) -> tuple[SQLModel, bool] | StrainRef:
    """Utility function to remove a string from `fieldname` in an SQLModel.

    :param model: The SQLModel to be updated.
    :param fieldname: The field of `model` where the offending string is to
        found and cleaned up.
    :param pattern: Regular expression characterizing the set of offending
        strings.

    :return: Tuple containing the updated model and a boolean value indicating
        whether the pattern was found in the models `fieldname`.

    Example:
    --------
    There are 1161 rows in the `organism` table of brenda_conn where the
    `organism` field contains a string of the form "no activity in Eptesicus
    fuscus" or "no activity by Mycobacterium smegmatis MSMEI_6484".::

        clean_name(Organism, "organism", "no activity (in|by) ")

    would lead to those fields being stripped of the extraneous string and to
    a return value of `True`, to be handled by the caller.
    """
    name, count = re.subn(rf"{pattern}", "", getattr(model, fieldname))

    if isinstance(model, _Strain):
        data = model.__dict__
        data[fieldname] = name
        return StrainRef(id=data["id"], name=data["name"]), bool(count)

    return model.copy(update={fieldname: name}), bool(count)
