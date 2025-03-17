from typing import Optional, Annotated

from db import db_init, query, save_annotation
from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import EmailStr

from xmlparser import transform_article

app = FastAPI()

origins = ["http://localhost:5173"]

app.add_middleware(CORSMiddleware, allow_origins=origins)


@app.on_event("startup")
def on_startup():
    db_init()


@app.get("/segment/")
def show_segment(pmid: Optional[int] = None, start: Optional[int] = None) -> str:
    args = [arg for arg in (pmid, start) if arg is not None]

    return get_response_json(*args)


@app.get("/annotation/")
def show_annotation(annotator: EmailStr, id: int) -> str:
    return get_response_json(annotator, id)


@app.put("/annotation/")
def store_annotation(
    annotator: Annotated[EmailStr, Form()],
    id: Annotated[int, Form()],
    annotation: Annotated[str, Form()],
) -> None:
    save_annotation(annotator, id, annotation)


def get_response_json(*args) -> str:
    response = query(*args)
    response.content = transform_article(response.content)
    return response.model_dump_json()
