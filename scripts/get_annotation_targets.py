#!/usr/bin/env python

import getpass
import json
import tomllib
from argparse import ArgumentParser

from config import species_list
from rapidfuzz import fuzz, process
from sqlalchemy import URL
from sqlmodel import Field, Session, SQLModel, create_engine, select
from tqdm import tqdm


class Protein_Connect(SQLModel, table=True):  # type: ignore
    protein_connect_id: int = Field(primary_key=True)
    organism_id: int = Field(nullable=False)
    ec_class_id: int = Field(nullable=False)
    protein_organism_strain_id: int | None = Field()
    reference_id: int = Field(nullable=False)
    protein_id: int = Field(nullable=False)


class Reference(SQLModel, table=True):  # type: ignore
    reference_id: int = Field(primary_key=True)
    authors: str = Field(nullable=False)
    title: str = Field(nullable=False)
    journal: str = Field()
    volume: str = Field()
    pages: str = Field()
    year: int = Field()
    pubmed_id: str = Field()
    path: str = Field()


class Organism(SQLModel, table=True):  # type: ignore
    organism_id: int = Field(primary_key=True)
    organism: str = Field(nullable=False)


class EC_Class(SQLModel, table=True):  # type: ignore
    ec_class_id: int = Field(primary_key=True)
    ec_class: str = Field(nullable=False)
    recommended_name: str = Field(nullable=False)


class Protein(SQLModel, table=True):  # type: ignore
    protein_id: int = Field(primary_key=True)
    protein: str | None = Field()


with open(species_list, "r") as sl:
    bacteria = set(s.strip() for s in sl.readlines())


def is_bacteria(organism: str) -> bool:
    _, ratio, _ = process.extract(
        organism, bacteria, scorer=fuzz.QRatio, limit=1
    )[0]

    return ratio > 90


def main():
    # Get the credentials for the database connection
    argparser = ArgumentParser()
    argparser.add_argument(
        "config", help="File containing database connection information."
    )
    argparser.add_argument(
        "output",
        help="Output file to hold enzyme-strain relations to be resolved.",
    )
    args = argparser.parse_args()
    config_file = args.config
    output_file = args.output

    with open(config_file, mode="rb") as cf:
        db_conn_info = tomllib.load(cf)

    # Initialize the DB engine
    user = input("User: ")
    password = getpass.getpass(prompt="Password: ")
    url_object = URL.create(
        drivername=db_conn_info["backend"],
        host=db_conn_info["host"],
        database=db_conn_info["database"],
        username=user,
        password=password,
    )
    engine = create_engine(url_object, echo=True)

    # Query enzyme-organism connections that are not associated with a strain
    # and return those that
    with Session(engine) as session:
        query = (
            select(Protein_Connect, Organism, EC_Class, Protein, Reference)
            .join(Organism, Protein_Connect.organism_id == Organism.organism_id)
            .join(EC_Class, Protein_Connect.ec_class_id == EC_Class.ec_class_id)
            .join(Protein, Protein_Connect.protein_id == Protein.protein_id)
            .join(
                Reference,
                Protein_Connect.reference_id == Reference.reference_id,
            )
            .where(Protein_Connect.protein_organism_strain_id is None)
        )

        with open(output_file, "a") as out:
            for record in tqdm(session.exec(query)):
                if is_bacteria(record.Organism.organism):
                    d = {}
                    for field in record:
                        d |= field
                    json.dump(d, out)
                    out.write("\n")
