#!/usr/bin/env python

from argparse import ArgumentParser

from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError
from sqlmodel import Session, create_engine
from tqdm import tqdm

from backend.db import SQLModel
from xmlparser import get_chunks, get_text, parse_file


def store(file: str, engine: Engine) -> None:
    tree = parse_file(file)
    chunks = get_chunks(tree)

    try:
        text = get_text(tree)
    except ValueError as error:
        print(error)
        print(f"File: {file}")

    with Session(engine) as session:
        session.add(text)
        session.commit()
        session.refresh(text)
        for chunk in chunks:
            chunk.source = text.id
            session.add(chunk)
        session.commit()


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("filenames", nargs="+")
    args = parser.parse_args()
    for file in args.filenames:
        p = store(file)
