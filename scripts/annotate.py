#!/usr/bin/env python

import os
from itertools import starmap

import torch
import torch._dynamo
from datamodel import Annotation
from db import (get_annotator, get_batch, query, response_to_article,
                save_annotations)

from config import load_model_config
from entities.models import NERCTagger
from entities.serialize import serialize_triples
from xmlparser import reinsert_tags, remove_tags

os.environ["TRANSFORMERS_OFFLINE"] = "1"


def main() -> None:
    # chunk = db.query(17077506, 0)

    config = load_model_config("entities/models/current_model_config.toml")

    model = NERCTagger(config=config)
    model.load_state_dict(torch.load("entities/models/current_model.pt"))
    model.to(model.device)
    model.eval()

    print("Model loaded")

    annotator = get_annotator("strain_annotator@dsmz.de", "Strain Annotator")
    batch = tuple(get_batch(annotator.email, config.batch_size))
    # batch = [response_to_article(query(11914155))]
    counter = 0

    while batch:
        batch_bodies = [sample.body for sample in batch]
        stripped = [remove_tags(body) for body in batch_bodies]
        predictions = model.predict(stripped)
        serialized = map(serialize_triples, predictions)

        retagged = starmap(reinsert_tags, zip(serialized, batch_bodies))

        annotations = (
            Annotation(
                annotator=annotator.email,
                chunk=article.chunk_id,
                annotation="<annotation>"
                + article.metadata
                + annotated
                + "</annotation>",
            )
            for article, annotated in zip(batch, retagged)
        )

        counter += save_annotations(annotations, force=True)

        batch = tuple(get_batch(annotator.email, config.batch_size))

    if counter:
        print(f"Successfuly added {counter} new annotations.")
    else:
        print("No new annotations to add.")
