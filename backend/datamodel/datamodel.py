from pydantic import BaseModel, EmailStr, NonNegativeInt, PositiveInt
from sqlmodel import Field, SQLModel, UniqueConstraint


class Annotator(SQLModel, table=True):
    email: EmailStr = Field(primary_key=True)
    name: str = Field(nullable=False)


class Text(SQLModel, table=True):
    id: int = Field(primary_key=True)
    pmid: PositiveInt = Field(nullable=False, unique=True)
    doi: str = Field(nullable=False)
    meta: str = Field(nullable=False)

class TextChunk(SQLModel, table=True):
    """
    start: the document position of the <p> tag that starts the chunk.
    stop: the document position immediately after the last <p> tag of the chunk.
    """

    id: int = Field(primary_key=True)
    source: int | None = Field(
        default=None, nullable=False, foreign_key="text.id"
    )
    content: str = Field(nullable=False)
    pos: NonNegativeInt = Field(nullable=False)


class Annotation(SQLModel, table=True):
    __table_args__ = (
        UniqueConstraint(
            "annotator", "chunk", name="unique_annotator_plus_chunk"
        ),
    )
    id: int = Field(primary_key=True)
    annotator: EmailStr = Field(nullable=False, foreign_key="annotator.email")
    chunk: int = Field(nullable=False, foreign_key="textchunk.id")
    annotation: str = Field(nullable=False)


class HtmlChunk(BaseModel):
    article_id: int
    chunk_id: int | None
    metadata: str
    body: str


class Response(BaseModel):
    article: Text
    chunk: TextChunk | None
    content: str
