#!/usr/bin/env python
"""Running a checkpoint over articles and keeping what it predicted.

`evaluate` forwards the same documents through the same heads, scores them
and writes numbers: every span, entity id and relation it built is consumed
by a metric and dropped. This command keeps them instead and scores nothing,
writing one JSON object per document. The command reference documents that
record, including which fields are written `null` rather than empty and why
the two are different answers.
"""

import argparse
import json
import logging
import pathlib
from collections.abc import Callable, Sequence
from typing import cast

import h5py
from tqdm import tqdm

from d3text import (
    checkpoint,
    corpus,
    encodings_store,
    factory,
    linking_corpora,
    runtime,
)
from d3text.cli import args as cli_args
from d3text.data.data import encodings_path
from d3text.linking import DictionaryLinker, Linker
from d3text.linking_eval import TaggedSpan
from d3text.models import token_supervision
from d3text.models.config import encodings, load_model_config
from d3text.models.model_types import BatchItem
from d3text.models.ete import PredictedRelation
from d3text.schema import BRENDA_SCHEMA

logger = logging.getLogger(__name__)


def command_line_args() -> argparse.Namespace:
    """Parse the command line.

    :return: the parsed arguments; `datasets` defaults to the configured
        corpus files.
    """
    parser = argparse.ArgumentParser(
        prog="infer",
        description=(
            "Run a checkpoint over articles and write what it predicted."
        ),
    )
    parser.add_argument(
        "config",
        help="The training configuration the checkpoint was made with",
    )
    parser.add_argument("checkpoint", help="Checkpoint written by `train`")
    parser.add_argument("output", help="JSON Lines file to write")
    parser.add_argument(
        "datasets",
        nargs="*",
        type=cli_args.readable_path,
        help=(
            "corpus files to predict over; defaults to the configured "
            "corpus. Every document must already be in the encodings store "
            "the config's base model names"
        ),
    )

    args = parser.parse_args()
    args.datasets = cli_args.resolve_datasets(parser, args.datasets)
    return args


def document_record(
    document: str,
    spans: Sequence[TaggedSpan],
    linker: Linker | None,
    relations: Sequence[PredictedRelation] | None,
) -> dict[str, object]:
    """One document's predictions, as the output file carries them.

    Offsets index the `corpus.document_text` output for this document, which
    is what the tagger read; the surface is written beside them because a
    consumer that assembles the text differently needs it to find the span
    again.

    :param document: the article's pubmed id.
    :param spans: every mention the tagger proposed, in its own order.
    :param linker: the linker to resolve each span's surface through, or
        None where the machine could build none, which is written through
        as a `null` `entity_ids` rather than as an empty set.
    :param relations: the pairs the relation head labelled, or None where it
        was asked nothing — no relation head in the checkpoint, or no pair
        proposed to put to it — which is written through rather than
        flattened onto the empty list.
    :return: the record, ready for `json.dumps`.
    """
    return {
        "document": document,
        "spans": [
            {
                "start": span.start,
                "end": span.end,
                "surface": span.surface,
                "entity_type": span.entity_type,
                "entity_ids": (
                    None
                    if linker is None
                    else sorted(linker.link(span.surface, span.entity_type))
                ),
            }
            for span in spans
        ],
        "relations": (
            None
            if relations is None
            else [
                {
                    "predicate": relation.predicate,
                    "arguments": [
                        sorted(relation.arguments[0]),
                        sorted(relation.arguments[1]),
                    ],
                }
                for relation in relations
            ]
        ),
    }


def build_linker() -> Linker | None:
    """The linker every span's surface is resolved through, if there is one.

    :return: a linker over the BRENDA surface-form index, or None on a
        machine that holds none of the files it is built from —
        `linking_corpora.brenda_index` says which is missing.
    """
    index = linking_corpora.brenda_index()
    if index is None:
        logger.warning(
            "no surface-form index could be built, so no span is linked and "
            "every `entity_ids` is written as null rather than as an empty "
            "set the linker chose"
        )
        return None
    return DictionaryLinker(index)


def main() -> None:
    """Predict over the named or configured corpus files; write the records.

    :raises SystemExit: if the checkpoint carries no span tagger, so there
        is nothing for this command to predict, or if no document of the
        named corpus files could be run at all.
    """
    args = command_line_args()
    config = load_model_config(args.config)
    # See `train.main`: after the config so the seed comes from it, before
    # any CUDA work.
    runtime.configure(seed=config.seed)

    logger.info("Loading checkpoint...")
    saved = checkpoint.load(args.checkpoint)
    store_path = encodings_path(encodings[config.base_model])
    # The offsets these records carry index the text this store was built
    # from, so a store rebuilt since the checkpoint trained moves every one
    # of them.
    encodings_store.encodings_provenance(
        saved.encodings_digest,
        encodings_store.store_content_digest(store_path),
    )

    logger.info("Initializing model...")
    model = factory.build_model(config, BRENDA_SCHEMA)
    model.register_load_state_dict_pre_hook(factory.fix_keys_hook)
    model.load_state_dict(saved.state_dict)
    model.to(model.device)
    model.eval()

    token_tagger = token_supervision.resolve_token_tagger(model)
    if token_tagger is None:
        raise SystemExit(
            f"COULD NOT RUN: {config.model_class} carries no span tagger, so "
            f"it proposes no mention and no relation argument; there is "
            f"nothing for this command to write. Predict with a checkpoint "
            f"trained against a token-label store."
        )

    # Only `ETEBrendaModel` declares this, and a checkpoint without it
    # predicts spans alone; the records then say so rather than carrying an
    # empty relation list.
    raw_relations = getattr(model, "predicted_relations", None)
    if raw_relations is None:
        logger.warning(
            "%s carries no relation head, so every record's `relations` is "
            "written as null rather than as an empty list",
            config.model_class,
        )
    predicted_relations = cast(
        Callable[[Sequence[BatchItem]], list[PredictedRelation] | None] | None,
        raw_relations,
    )

    linker = build_linker()

    written = skipped = 0
    with (
        h5py.File(store_path, "r") as store,
        pathlib.Path(args.output).open("w", encoding="utf8") as output,
    ):
        for dataset in args.datasets:
            total, rows = corpus.stream_rows(dataset, corpus.STREAM_BATCH)
            for pubmed_id, text in tqdm(rows, total=total, desc=dataset.name):
                document = str(pubmed_id)
                group = store.get(document)
                if not text or not encodings_store.is_finished_group(group):
                    skipped += 1
                    continue

                spans = token_supervision.predicted_spans_from_store(
                    store,
                    None,
                    {document: text},
                    model.get_token_embeddings,
                    model.hidden,
                    token_tagger,
                    model.autocast_context,
                )
                # A second forward over the same document, whose
                # trunk pass the embeddings cache serves where the machine
                # has one on. Folding it into the pass above means
                # restating the span grounding here, over `forward`'s own
                # hidden state and token logits, the way `evaluate_model`
                # does; worth it only if this ever measures as slow.
                relations = (
                    None
                    if predicted_relations is None
                    else predicted_relations(
                        [
                            token_supervision.store_batch_item(
                                group, int(pubmed_id)
                            )
                        ]
                    )
                )
                record = document_record(document, spans, linker, relations)
                output.write(json.dumps(record) + "\n")
                written += 1

    if not written:
        raise SystemExit(
            f"COULD NOT RUN: none of the {skipped} documents of "
            f"{', '.join(str(path) for path in args.datasets)} has text and "
            f"a finished group in {store_path}, so nothing was predicted. "
            f"Build the store with `precompute-encodings` first."
        )
    logger.info(
        "wrote %d document%s to %s; %d had no text or no finished group in "
        "%s and were not run",
        written,
        "" if written == 1 else "s",
        args.output,
        skipped,
        store_path,
    )


if __name__ == "__main__":
    main()
