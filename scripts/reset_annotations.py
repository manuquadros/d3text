#!/usr/bin/env python

from db import Annotation, Session, engine
from sqlmodel import delete


def reset_annotations() -> None:
    with Session(engine) as session:
        session.exec(delete(Annotation))
        session.commit()


def main() -> None:
    reset_annotations()
