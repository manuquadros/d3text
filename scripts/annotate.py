#!/usr/bin/env python

import os
from itertools import starmap

import db
import torch
import torch._dynamo
from datamodel import Annotation

from config import load_model_config
from entities.models import NERCTagger
from entities.serialize import serialize_triples
from utils import get_batch
from xmlparser import reinsert_tags, remove_tags, split_metadata_body

os.environ["TRANSFORMERS_OFFLINE"] = "1"


def main() -> None:
    # chunk = db.query(17077506, 0)

    config = load_model_config("entities/models/current_model_config.toml")
    model = NERCTagger(config=config)
    model.load_state_dict(torch.load("entities/models/current_model.pt"))
    model.to(model.device)
    model.eval()

    print("Model loaded")

    annotator = db.get_annotator("strain_annotator@dsmz.de", "Strain Annotator")
    batch = get_batch(annotator.email, config.batch_size)

    while batch:
        batch_bodies = [sample.body for sample in batch]
        stripped = [remove_tags(body) for body in batch_bodies]
        print("\n---\n".join(stripped))
        predictions = model.predict(stripped)
        serialized = starmap(serialize_triples, zip(predictions, batch_bodies))
        retagged = starmap(reinsert_tags, zip(serialized, batch_bodies))

        print(list(retagged))

        annotations = (
            Annotation(
                annotator=annotator.email,
                chunk=article.chunk_id,
                annotation="<chunk>"
                + article.metadata
                + annotated
                + "</chunk>",
            )
            for article, annotated in zip(batch, retagged)
        )

        db.save_annotations(annotations)

        break

        batch = get_batch(annotator.email, config.batch_size)
