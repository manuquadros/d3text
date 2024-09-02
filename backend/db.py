import os
from collections.abc import Iterable, Iterator
from typing import Optional

from datamodel import (Annotation, Annotator, HtmlChunk, Response, SQLModel,
                       Text, TextChunk)
from multimethod import multimethod
from pydantic import EmailStr
from sqlalchemy.exc import IntegrityError
from sqlalchemy.sql.functions import random
from sqlmodel import Session, col, create_engine, select
from tokenizers.normalizers import BertNormalizer

from xmlparser import transform_article

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


def add_annotation(
    ann: Annotation,
    force: Optional[bool] = False,
) -> None:
    with Session(engine) as session:
        try:
            session.add(
                Annotation(
                    annotator=ann.annotator,
                    chunk=ann.chunk,
                    annotation=ann.annotation,
                )
            )
            session.commit()
        except IntegrityError:
            session.rollback()
            if force:
                record = session.exec(
                    select(Annotation).where(
                        Annotation.annotator == ann.annotator
                        and Annotation.chunk == ann.chunk_id
                    )
                ).one()
                record.annotation = ann.annotation
                session.commit()
            else:
                raise


def save_annotations(annotations: Iterable[Annotation]) -> None:
    how_many: int = 0
    for ann in annotations:
        try:
            add_annotation(ann)
            how_many += 1
        except IntegrityError:
            pass

    print(f"Successfuly added {how_many} new annotations.")


def get_unannotated(
    annotator: Optional[EmailStr] = None, batch_size: Optional[int] = None
) -> Iterator[Response]:
    if annotator is not None:
        annotated = select(Annotation.chunk).where(
            Annotation.annotator == annotator
        )
    else:
        annotated = select(Annotation.chunk)

    query = (
        select(TextChunk, Text)
        .join(Text)
        .where(col(TextChunk.id).not_in(annotated))
    )

    if batch_size is not None:
        query = query.limit(batch_size)

    with Session(engine) as session:
        results = session.exec(query).all()

    for result in results:
        article = result[1]
        chunk = result[0]
        yield Response(
            article=article,
            chunk=chunk,
            content=chunk.content,
        )


@multimethod
def query(pmid: int, pos: int) -> Response:
    with Session(engine) as session:
        chunk, article = next(
            session.exec(
                select(TextChunk, Text)
                .join(Text)
                .where(Text.pmid == pmid)
                .where(TextChunk.pos == pos)
            )
        )

    return Response(
        article=article,
        chunk=chunk,
        content=chunk.content,
    )


@query.register
def _(pmid: int) -> Response:
    with Session(engine) as session:
        article = next(session.exec(select(Text).where(Text.pmid == pmid)))

    return Response(article=article, chunk=None, content=compile_text(article))


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


def compile_text(text: Text) -> str:
    with Session(engine) as session:
        chunks = session.exec(
            select(TextChunk)
            .where(TextChunk.source == text.id)
            .order_by(TextChunk.pos.asc())
        ).all()

    content = "\n".join(ck.content for ck in chunks[1:])
    print(content)

    return transform_article(f"<article>\n{text.meta}\n{content}</article>")


def get_batch(annotator_email: EmailStr, batch_size: int) -> list[HtmlChunk]:
    chunks = []
    for item in get_unannotated(annotator_email, batch_size):
        chunks.append(response_to_article(item))

    return chunks


def response_to_article(item: Response) -> HtmlChunk:
    content = BertNormalizer(lowercase=False).normalize_str(item.content)

    return HtmlChunk(
        article_id=item.article.id,
        chunk_id=item.chunk.id if item.chunk else None,
        metadata=item.article.meta,
        body=content,
    )
