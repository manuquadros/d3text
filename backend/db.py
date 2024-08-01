import os

from datamodel import (Annotation, Annotator, Response, SQLModel, Text,
                       TextChunk)
from log import logger
from sqlalchemy.sql.functions import random
from sqlmodel import Session, create_engine, select

from xmlparser import get_chunk, transform_article

db_path = os.path.join(os.path.dirname(__file__), "database.db")

engine = create_engine(
    f"sqlite:///{db_path}",
    echo=True,
    connect_args={"check_same_thread": False},
)


def db_init() -> None:
    SQLModel.metadata.create_all(engine)


def random_chunk() -> Response:
    with Session(engine) as session:
        chunk = next(
            session.exec(select(TextChunk).order_by(random()).limit(1))
        )
        article = next(
            session.exec(select(Text).where(Text.id == chunk.source).limit(1))
        )

    return Response(
        article=article,
        chunk=chunk,
        content=get_chunk(article.content, chunk.start, chunk.stop),
    )


def query_chunk(pmid: int, start: int) -> Response:
    with Session(engine) as session:
        chunk, article = next(
            session.exec(
                select(TextChunk, Text)
                .join(Text)
                .where(Text.pmid == pmid)
                .where(TextChunk.start == start)
            )
        )

    content = get_chunk(article.content, chunk.start, chunk.stop)
    logger.debug(content)
    content = transform_article(content)
    logger.debug(content)

    return Response(
        article=article,
        chunk=chunk,
        content=content,
    )


def query_article(pmid: int) -> Text:
    with Session(engine) as session:
        article = next(session.exec(select(Text).where(Text.pmid == pmid)))

    article.content = transform_article(article.content)
    return article
