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
def show_segment(pmid: int = 0, start: int = 0) -> str:
    if pmid and start:
        print(f"Getting chunk {start} from PMID {pmid}")
        segment = query_chunk(pmid, start)
    elif pmid:
        print(f"Retrieving article {pmid}")
        segment = query_article(pmid)
    else:
        print("Getting random chunk")
        segment = random_chunk()

    return segment.model_dump_json()
