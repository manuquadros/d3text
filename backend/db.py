from pydantic import EmailStr, PositiveInt
from sqlmodel import Field, SQLModel


class Annotator(SQLModel, table=True):
    email: EmailStr = Field(primary_key=True)
    name: str = Field(nullable=False)


class Text(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    pmid: PositiveInt = Field(nullable=False, unique=True)
    doi: str = Field(nullable=False)
    content: str = Field(nullable=False)


class TextChunk(SQLModel, table=True):
    """
    start: the document position of the <p> tag that starts the chunk.
    stop: the document position immediately after the last <p> tag of the chunk.
    """

    id: int | None = Field(default=None, primary_key=True)
    source: int | None = Field(
        default=None, nullable=False, foreign_key="text.id"
    )
    start: PositiveInt = Field(nullable=False)
    stop: PositiveInt = Field(nullable=False)


class Annotation(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    annotator: EmailStr = Field(nullable=False, foreign_key="annotator.email")
    chunk: int = Field(nullable=False, foreign_key="textchunk.id")
    annotation: str = Field(nullable=False)
