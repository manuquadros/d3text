from typing import Optional

from datamodel import Response
from db import db_init, query_article, query_chunk, random_chunk
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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
    if pmid is not None and start is not None:
        print(f"Getting chunk {start} from PMID {pmid}")
        segment = query_chunk(pmid, start)
    elif pmid is not None:
        print(f"Retrieving article {pmid}")
        segment = query_article(pmid)
    else:
        print("Getting random chunk")
        segment = random_chunk()

    return segment.model_dump_json()
