from db import Annotation, Annotator, SQLModel, Text, engine
from fastapi import FastAPI
from sqlmodel import Session, select
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = ["http://localhost:5173"]

@app.get("/annotators/")
def read_annotators():
    with Session(engine) as session:
        annotators = session.exec(select(Annotator)).all()
        return annotators
app.add_middleware(CORSMiddleware, allow_origins=origins)
