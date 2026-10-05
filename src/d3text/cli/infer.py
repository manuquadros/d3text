#!/usr/bin/env python
"""Running a checkpoint over articles and keeping what it predicted.

Where `evaluate` scores the spans, entity ids and relations it builds and
drops them, this command keeps them, scores nothing, and writes one JSON
object per document; the command reference documents that record.
"""

import argparse
import json
import logging
import pathlib
from collections.abc import Callable, Sequence
from typing import cast

from tqdm import tqdm

from d3text import (
    checkpoint,
    corpus,
    encodings_store,
    factory,
    runtime,
    surface_forms,
    token_labels,
    utils,
)
from d3text.cli import args as cli_args
from d3text.linking import DictionaryLinker, Linker
from d3text.linking_eval import TaggedSpan
from d3text.models import token_supervision
from d3text.models.config import load_model_config
from d3text.models.model_types import BatchItem
from d3text.models.ete import PredictedRelation
from d3text.models.token_supervision import StoredMention
from d3text.schema import BRENDA_SCHEMA

logger = logging.getLogger(__name__)


def command_line_args() -> argparse.Namespace:
    """Parse the command line.

    :return: the parsed arguments.
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
        nargs="+",
        metavar="DATASET",
        type=cli_args.readable_path,
        help="corpus files to predict over",
    )

    return parser.parse_args()


def document_record(
    document: str,
    spans: Sequence[TaggedSpan],
    linker: Linker | None,
    relations: Sequence[PredictedRelation] | None,
) -> dict[str, object]:
    """One document's predictions, as the output file carries them.

    Offsets index the `corpus.document_text` output for this document, which
    is the text this command tokenized; the surface is written beside them
    because a consumer that assembles the text differently needs it to find
    the span again.

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


def build_linker(saved: checkpoint.Checkpoint) -> Linker | None:
    """The linker every span's surface is resolved through, if there is one.

    Reads `saved.surface_form_index` rather than rebuilding it from BRENDA's
    data: that index is what `train` built alongside the weights, so linking
    here has no dependency on `brenda_references` or the BRENDA data.

    :param saved: the checkpoint this run loaded.
    :return: a linker over the checkpoint's own surface-form index, or None
        for a checkpoint written before this was recorded, or for a training
        run that could not build one (`linking_corpora.brenda_index`'s
        warning names why) — retrain where that index can be built.
    """
    if saved.surface_form_index is None:
        logger.warning(
            "this checkpoint carries no surface-form index, so no span is "
            "linked and every `entity_ids` is written as null rather than as "
            "an empty set the linker chose; retrain where that index can "
            "be built"
        )
        return None
    return DictionaryLinker(saved.surface_form_index)


def grounding_index(
    saved: checkpoint.Checkpoint,
) -> surface_forms.SurfaceFormIndex | None:
    """The index relation arguments are matched against in the given text.

    Warns, rather than refusing, unless the checkpoint's digests show this
    index and this build's labelling rules placed its training targets:
    `train` builds its index apart from the label store's, so only the
    digests can say the two agree.

    :param saved: the checkpoint this run loaded.
    :return: `saved.surface_form_index`, or None when it carries none, so
        that no relation argument can be grounded.
    """
    index = saved.surface_form_index
    if index is None:
        logger.warning(
            "this checkpoint carries no surface-form index to ground "
            "relation arguments against, so every `relations` is written "
            "as null"
        )
        return None

    recorded_index = saved.token_labels_digest
    recorded_rules = saved.labelling_rules_digest
    drift: str | None
    if recorded_index is None:
        drift = "the checkpoint records no token-label provenance"
    elif recorded_index != (current_index := surface_forms.index_digest(index)):
        drift = (
            f"training matched index {recorded_index[:12]}, this one is "
            f"{current_index[:12]}"
        )
    elif recorded_rules is None:
        drift = (
            "the index matches, but the checkpoint records no "
            "labelling-rules digest to compare this build's rules with"
        )
    else:
        try:
            current_rules = token_labels.labelling_rules_digest()
        except OSError as error:
            drift = f"this build's labelling rules are unknown: {error}"
        else:
            drift = (
                None
                if current_rules == recorded_rules
                else f"training labelled by rules {recorded_rules[:12]}, "
                f"this build by {current_rules[:12]}"
            )
    if drift is None:
        return index
    logger.warning(
        "relations are grounded on the mentions the checkpoint's "
        "surface-form index matches under this build's labelling rules, "
        "which need not be the ones the relation head trained on: %s",
        drift,
    )
    return index


def main() -> None:
    """Tokenize the named corpus files, predict over them, write the records.

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
    # The loader `precompute-encodings` tokenizes the training corpus with.
    tokenizer = utils.load_fast_tokenizer(config.base_model)

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
        Callable[
            [Sequence[BatchItem], dict[int, tuple[StoredMention, ...]]],
            list[PredictedRelation] | None,
        ]
        | None,
        raw_relations,
    )
    index = None if predicted_relations is None else grounding_index(saved)

    linker = build_linker(saved)

    written = skipped = 0
    with pathlib.Path(args.output).open("w", encoding="utf8") as output:
        for dataset in args.datasets:
            total, rows = corpus.stream_rows(dataset, corpus.STREAM_BATCH)
            for pubmed_id, text in tqdm(rows, total=total, desc=dataset.name):
                if not text:
                    skipped += 1
                    continue

                document = str(pubmed_id)
                # The windowing `precompute-encodings` and
                # `precompute-token-labels` tokenize with.
                (encoding,) = encodings_store.document_encodings(
                    utils.split_and_tokenize(tokenizer, [text]), 1
                )
                # The caches and the token-label store key a document by its
                # id alone, and this text need not be what any of them was
                # filled from.
                with model.uncached_embeddings():
                    spans = token_supervision.predicted_spans_from_encoding(
                        encoding,
                        int(pubmed_id),
                        text,
                        document,
                        model.get_token_embeddings,
                        model.hidden,
                        token_tagger,
                        model.autocast_context,
                    )
                    # A second trunk forward; folding it into the pass above
                    # would restate the span grounding `evaluate_model` does.
                    # Only if it is slow.
                    relations = None
                    if predicted_relations is not None and index is not None:
                        mentions = token_supervision.live_mentions(
                            text, encoding, index
                        )
                        relations = predicted_relations(
                            [
                                token_supervision.store_batch_item(
                                    encoding, int(pubmed_id)
                                )
                            ],
                            {0: mentions} if mentions else {},
                        )
                record = document_record(document, spans, linker, relations)
                output.write(json.dumps(record) + "\n")
                written += 1

    if not written:
        raise SystemExit(
            f"COULD NOT RUN: none of the {skipped} documents of "
            f"{', '.join(str(path) for path in args.datasets)} has text, so "
            f"nothing was predicted."
        )
    logger.info(
        "wrote %d document%s to %s; %d had no text and were not run",
        written,
        "" if written == 1 else "s",
        args.output,
        skipped,
    )


if __name__ == "__main__":
    main()
