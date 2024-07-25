from db import Annotation, Annotator, SQLModel, Text, engine
from fastapi import FastAPI
from sqlmodel import Session, select

app = FastAPI()


@app.on_event("startup")
def on_startup():
    SQLModel.metadata.create_all(engine)


@app.get("/annotators/")
def read_annotators():
    with Session(engine) as session:
        annotators = session.exec(select(Annotator)).all()
        return annotators
