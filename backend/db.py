import os
from typing import Optional

from datamodel import (Annotation, Annotator, Response, SQLModel, Text,
                       TextChunk)
from log import logger
from multimethod import multimethod
from pydantic import EmailStr
from sqlalchemy.sql.functions import random
from sqlmodel import Session, create_engine, select

from xmlparser import get_chunk

db_path = os.path.join(os.path.dirname(__file__), "database.db")

engine = create_engine(
    f"sqlite:///{db_path}",
    echo=True,
    connect_args={"check_same_thread": False},
)


def db_init() -> None:
    SQLModel.metadata.create_all(engine)


def get_annotator(email: EmailStr, name: Optional[str] = None) -> Annotator:
    with Session(engine) as session:
        annotator = session.exec(
            select(Annotator).where(Annotator.email == email)
        ).first()

    if annotator is None:
        annotator = Annotator(email=email, name=name)
        create_annotator(**annotator.model_dump())

    return annotator


def create_annotator(email: EmailStr, name: str) -> None:
    with Session(engine) as session:
        session.add(Annotator(email=email, name=name))
        session.commit()


def add_annotation(annotator: EmailStr, chunk_id: int, annotation: str) -> None:
    with Session(engine) as session:
        session.add(
            Annotation(
                annotator=annotator, chunk=chunk_id, annotation=annotation
            )
        )
        session.commit()


@multimethod
def query(pmid: int, start: int) -> Response:
    with Session(engine) as session:
        chunk, article = next(
            session.exec(
                select(TextChunk, Text)
                .join(Text)
                .where(Text.pmid == pmid)
                .where(TextChunk.start == start)
            )
        )

    return Response(
        article=article,
        chunk=chunk,
        content=get_chunk(article.content, chunk.start, chunk.stop),
    )


@query.register
def _(pmid: int) -> Text:
    with Session(engine) as session:
        article = next(session.exec(select(Text).where(Text.pmid == pmid)))

    return article


@query.register
def _() -> Response:
    with Session(engine) as session:
        annotation = next(
            session.exec(select(Annotation).order_by(random()).limit(1))
        )
        chunk = next(
            session.exec(
                select(TextChunk)
                .where(TextChunk.id == annotation.chunk)
                .limit(1)
            )
        )
        article = next(
            session.exec(select(Text).where(Text.id == chunk.source).limit(1))
        )

    return Response(article=article, chunk=chunk, content=annotation.annotation)
