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


class Annotation(SQLModel, table=True):
    id: int = Field(primary_key=True)
    annotator: EmailStr = Field(nullable=False, foreign_key="annotator.email")
    text: int = Field(nullable=False, foreign_key="text.pmid")
    offset_start: PositiveInt = Field(nullable=False)
    offset_end: PositiveInt = Field(nullable=False)
    annotation: str = Field(nullable=False)


engine = create_engine(
    "sqlite:///annotations.db",
    echo=True,
    connect_args={"check_same_thread": False},
)
