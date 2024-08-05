from typing import Optional

from datamodel import Response
from db import db_init, query
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from xmlparser import transform_article

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
    print(args)

    return get_response_json(*args)


def get_response_json(*args) -> str:
    response = query(*args)
    print(response)
    response.content = transform_article(response.content)
    print(response)
    return response.model_dump_json()
