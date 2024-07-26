from pydantic import EmailStr, PositiveInt
from sqlmodel import Field, SQLModel, create_engine


class Annotator(SQLModel, table=True):
    email: EmailStr = Field(primary_key=True)
    name: str = Field(nullable=False)


class Text(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    pmid: int = Field(nullable=False)
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
    start: int = Field(nullable=False)
    stop: int = Field(nullable=False)


class Annotation(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    annotator: EmailStr = Field(nullable=False, foreign_key="annotator.email")
    chunk: int = Field(nullable=False, foreign_key="textchunks.id")
    annotation: str = Field(nullable=False)


engine = create_engine(
    "sqlite:///annotations.db",
    echo=True,
    connect_args={"check_same_thread": False},
)
