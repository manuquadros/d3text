from typing import Optional

from db import db_init, query
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import EmailStr

from xmlparser import fromstring, tostring, transform_article

app = FastAPI()

origins = ["http://localhost:5173"]

app.add_middleware(CORSMiddleware, allow_origins=origins)


@app.on_event("startup")
def on_startup():
    db_init()


@app.get("/segment/")
def show_segment(
    pmid: Optional[int] = None, start: Optional[int] = None
) -> str:
    args = [arg for arg in (pmid, start) if arg is not None]

    return get_response_json(*args)


@app.get("/annotation/")
def show_annotation(annotator: EmailStr, id: int) -> str:
    return get_response_json(annotator, id)


def get_response_json(*args) -> str:
    response = query(*args)
    response.content = transform_article(response.content)
    return response.model_dump_json()
