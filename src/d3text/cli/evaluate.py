#!/usr/bin/env python

import argparse
import logging
import os
import pathlib
import warnings
from collections.abc import Mapping
from typing import cast

import h5py

from d3text import (
    checkpoint,
    data,
    encodings_store,
    factory,
    linking_corpora,
    runtime,
    token_labels,
    tracking,
)
from d3text.datasets import enzymener, s800
from d3text.datasets.brenda import (
    BRENDA_SCHEMA,
    brenda_dataset,
    encodings_path,
)
from d3text.linking_eval import TaggedSpan
from d3text.models import token_supervision
from d3text.models.config import encodings, load_model_config, machine_config
from d3text.vocabulary import Vocabulary

logger = logging.getLogger(__name__)


def command_line_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="evaluate",
        description=("Evaluate a model with the provided configuration."),
    )
    parser.add_argument(
        "config", help="Configuration file for the model to be evaluated."
    )
    parser.add_argument("model_state_dict", help="Model state dict")

    return parser.parse_args()


def load_evaluation_dataset(
    config_base_model: str,
    vocabulary: Vocabulary,
) -> data.EntityRelationDataset:
    """The dataset to score a checkpoint on, indexed the way it was trained.

    The recorded vocabulary is authoritative and the training split is not
    loaded at all: every checkpoint this code reads carries one, so the corpus
    never gets to decide which class owns which column.

    :param config_base_model: the model the encodings must have been built
        with.
    :param vocabulary: the checkpoint's recorded column order.
    :return: the indexed splits.
    """
    return brenda_dataset(
        schema=BRENDA_SCHEMA,
        encodings=encodings[config_base_model],
        vocabulary=vocabulary,
        split_names=("test",),
        base_model=config_base_model,
    )


def token_labels_provenance(
    recorded: str | None,
    current: str | None,
    recorded_rules: str | None = None,
    current_rules: str | None = None,
) -> str:
    """Say whether this run's label store is the one the checkpoint trained on.

    Warns where `token_labels.check_index` refuses, because the two guard
    different things: that one is about to *extend* a store whose halves would
    then label the same string differently, while a mismatch here only makes
    the detection scores incomparable with the training run's. The scores are
    still the scores; silence about which dictionary set their denominator is
    what has to go.

    The index digest and the rules digest answer different questions — which
    strings name entities, and what the sweep did with that answer — and
    either can move while the other does not, so a store rebuilt under
    unchanged rules and a store re-labelled by changed rules against a
    byte-identical index both have to be caught. The rules half is compared
    only when both sides carry one: a checkpoint written before it was
    recorded has `recorded_rules=None` even though `recorded` is a real
    digest, and that absence must not read as a mismatch.

    :param recorded: the index digest the checkpoint carries, if any.
    :param current: the index digest of the store this run reads, if any.
    :param recorded_rules: the labelling-rules digest the checkpoint carries,
        if any.
    :param current_rules: the labelling-rules digest of the store this run
        reads, if any.
    :return: the value for the run's `checkpoint_token_labels` tag.
    """
    if recorded is None:
        if current is None:
            return "unused"
        warnings.warn(
            "this checkpoint records no token-label provenance, so nothing "
            f"says whether its targets came from index {current[:12]}, the "
            "one this run reads. The detection metrics count gold spans this "
            "store's dictionary named, which is the training store's only if "
            "it has not been rebuilt since.",
            RuntimeWarning,
            stacklevel=2,
        )
        return "unrecorded"

    if recorded != current:
        reads = f"index {current[:12]}" if current else "no label store"
        warnings.warn(
            "this checkpoint was trained on targets from surface-form index "
            f"{recorded[:12]}, but this run reads {reads}; the two "
            "dictionaries name different strings, so the detection metrics "
            "count a different set of gold spans than the model was trained "
            "against and are not comparable with that run's.",
            RuntimeWarning,
            stacklevel=2,
        )
        return "mismatched"

    if (
        recorded_rules is not None
        and current_rules is not None
        and recorded_rules != current_rules
    ):
        warnings.warn(
            "this checkpoint was trained on targets placed by labelling "
            f"rules {recorded_rules[:12]}, but this run's store was labelled "
            f"by rules {current_rules[:12]}; the same surface-form index "
            f"({current[:12]}) named the strings, but the rules that turned "
            "them into spans have moved, so the detection metrics count a "
            "different set of gold spans than the model was trained against "
            "and are not comparable with that run's.",
            RuntimeWarning,
            stacklevel=2,
        )
        return "mismatched"

    return "matched"


def encodings_provenance(recorded: str | None, current: str | None) -> str:
    """Say whether this run's token ids are the ones the checkpoint trained on.

    Warns for `token_labels_provenance`'s reason: a corpus re-tokenized under
    a newer tokenizer revision or a corrected `document_text` gives the heads
    different inputs for the same document, which makes the scores
    incomparable with the training run's without making them wrong.

    :param recorded: the digest the checkpoint carries, if any.
    :param current: the digest of the store this run reads, if any.
    :return: the value for the run's `checkpoint_encodings` tag.
    """
    if recorded is not None and recorded == current:
        return "matched"

    if recorded is None:
        warnings.warn(
            "this checkpoint records no encodings digest, so nothing says "
            "which tokenization produced the inputs it was trained on; these "
            "test scores are that run's only if the store has not been "
            "rebuilt since.",
            RuntimeWarning,
            stacklevel=2,
        )
        return "unrecorded"

    if current is None:
        warnings.warn(
            f"this checkpoint was trained on encodings {recorded[:12]} and "
            "the store this run reads carries no digest of its own, so "
            "whether it holds the same token ids cannot be established; "
            "rebuild it with `precompute-encodings` to stamp it.",
            RuntimeWarning,
            stacklevel=2,
        )
        return "unstamped"

    warnings.warn(
        f"this checkpoint was trained on encodings {recorded[:12]} but this "
        f"run reads {current[:12]}; the two files hold different token ids "
        "for the same documents, so these scores are not comparable with "
        "that run's.",
        RuntimeWarning,
        stacklevel=2,
    )
    return "mismatched"


def report_linking(root: str | None) -> dict[str, float]:
    """Score the dictionary linker on the external corpora under `root`.

    Logged from here rather than from `evaluate_model` because it reads no
    checkpoint: the linker holds no learned parameters, so the block is a
    property of the surface-form index and is the same for every model.

    :param root: the directory holding the corpora, or None where the machine
        has none.
    :return: the metrics logged, empty where the block was skipped.
    """
    block = linking_corpora.linking_block(root)
    if not block.reports:
        return {}

    summary = block.summary()
    logger.info("\n=== Linking metrics (dictionary linker, outside gold) ===")
    logger.info(summary)
    tracking.log_text(summary, "test/linking_report.txt")
    metrics = block.metrics()
    tracking.log_metrics(metrics)
    return metrics


def _readable_texts(
    store: h5py.File,
    corpus: str,
    texts: Mapping[str, str],
    encodings_file: str | os.PathLike[str],
) -> Mapping[str, str]:
    """`texts`, or `{}` where scoring them would misreport a missing store.

    `predicted_spans_from_store` already skips a document whose group is
    absent or unfinished with no signal of its own, so a store built without
    `precompute-encodings --{corpus}` would otherwise hand
    `report_predicted_linking` an all-empty (or partial) span list that scores
    identically to a tagger that ran and found nothing — every gold mention
    of the unread documents charged as a missed detection instead of the
    precompute gap it actually is. Refused rather than scored against the
    documents that are readable: a report whose population silently shrinks
    from one run to the next would look like a change in the model.

    :param store: an open encodings store.
    :param corpus: which corpus's groups to check (`"s800"` or `"enzymener"`).
    :param texts: the corpus's document id to full text mapping.
    :param encodings_file: `store`'s path, named only for the log line.
    :return: `texts` unchanged if every document's group is present and
        finished; `{}` otherwise.
    """
    readable = token_supervision.readable_documents(store, corpus, texts)
    if len(readable) < len(texts):
        logger.warning(
            "%s holds a finished %s group for %d of %d document(s), so the "
            "predicted-linking report for %s is skipped; build it with "
            "`precompute-encodings --%s` first",
            encodings_file,
            corpus,
            len(readable),
            len(texts),
            corpus,
            corpus,
        )
        return {}
    logger.info(
        "%s: scoring predicted linking over %d document(s)",
        corpus,
        len(readable),
    )
    return texts


def report_predicted_linking(
    root: str | None,
    encodings_file: str | os.PathLike[str],
    model: object,
) -> dict[str, float]:
    """Score the dictionary linker through the checkpoint's own spans.

    Skipped wherever `report_linking` skips, and also where `model` detects
    no span at all — `NERClassificationModel` has no `token_tagger` — where
    the encodings store naming `encodings_file` is not on disk, or (per
    corpus, via `_readable_texts`) where that store holds no finished group,
    or only some, for a corpus with gold on disk — a `precompute-encodings
    --s800`/`--enzymener` gap, never scored as a tagger that ran and
    detected nothing.

    :param root: the directory holding the corpora, or None where the machine
        has none.
    :param encodings_file: the encodings store the checkpoint's base model was
        trained on, read for the S800/enzymeNER groups
        `precompute-encodings --s800`/`--enzymener` wrote into it.
    :param model: the loaded checkpoint, forwarded through
        `token_supervision.predicted_spans_from_store`. Typed loosely
        (`object`, not `factory.ConfigurableModel`) because `main`'s own
        tests drive this through stub models beartype would otherwise
        refuse at the call boundary.
    :return: the metrics logged, empty where the block was skipped.
    """
    token_tagger = token_supervision.resolve_token_tagger(model)
    if token_tagger is None:
        return {}
    # A real `token_tagger` only ever comes from a real checkpoint: every
    # stub the tests reach this line with declares none, and returns above.
    typed_model = cast(factory.ConfigurableModel, model)
    if root is None or not os.path.exists(encodings_file):
        return {}
    directory = pathlib.Path(root).expanduser()

    predicted: dict[str, list[TaggedSpan]] = {}
    with h5py.File(encodings_file, "r") as store:
        try:
            organism_texts = s800.load_s800(
                directory / linking_corpora.S800
            ).texts
        except (ValueError, FileNotFoundError):
            organism_texts = {}
        if organism_texts:
            organism_texts = _readable_texts(
                store, "s800", organism_texts, encodings_file
            )
        if organism_texts:
            predicted["s800"] = token_supervision.predicted_spans_from_store(
                store,
                "s800",
                organism_texts,
                typed_model.get_token_embeddings,
                typed_model.hidden,
                token_tagger,
                typed_model.autocast_context,
            )

        try:
            enzyme_texts = enzymener.load_enzymener(
                directory / linking_corpora.ENZYMENER
            ).texts
        except (ValueError, FileNotFoundError):
            enzyme_texts = {}
        if enzyme_texts:
            enzyme_texts = _readable_texts(
                store, "enzymener", enzyme_texts, encodings_file
            )
        if enzyme_texts:
            predicted["enzymener"] = (
                token_supervision.predicted_spans_from_store(
                    store,
                    "enzymener",
                    enzyme_texts,
                    typed_model.get_token_embeddings,
                    typed_model.hidden,
                    token_tagger,
                    typed_model.autocast_context,
                )
            )

    if not predicted:
        return {}
    block = linking_corpora.predicted_linking_block(root, predicted)
    if not block.reports:
        return {}

    summary = block.summary()
    logger.info("\n=== Linking metrics (dictionary linker, own spans) ===")
    logger.info(summary)
    tracking.log_text(summary, "test/predicted_linking_report.txt")
    metrics = block.metrics()
    tracking.log_metrics(metrics)
    return metrics


def main() -> None:
    args = command_line_args()
    config = load_model_config(args.config)
    # See `train.main`: after the config so the seed comes from it, before any
    # CUDA work.
    runtime.configure(seed=config.seed)

    # Read before the corpus: the vocabulary it carries decides how the corpus
    # is indexed, and a missing or unreadable checkpoint should not cost the
    # ~300 MB load first.
    logger.info("Loading checkpoint...")
    saved = checkpoint.load(args.model_state_dict)
    # Before the corpus for the same reason: an operator who is about to score
    # against the wrong dictionary should hear it now, not after the load.
    labels_provenance = token_labels_provenance(
        saved.token_labels_digest,
        token_labels.store_index_digest(config.token_labels_store),
        saved.labelling_rules_digest,
        token_labels.store_labelling_rules_digest(config.token_labels_store),
    )
    stale_rules = token_labels.stale_labelling_rules(config.token_labels_store)
    if stale_rules is not None:
        logger.warning("%s", stale_rules)
    inputs_provenance = encodings_provenance(
        saved.encodings_digest,
        encodings_store.store_content_digest(
            encodings_path(encodings[config.base_model])
        ),
    )

    logger.info("Loading evaluation dataset...")
    dataset = load_evaluation_dataset(
        config_base_model=config.base_model,
        vocabulary=saved.vocabulary,
    )
    eval_data = data.get_batch_loader(
        dataset=dataset.data["test"],
        batch_size=1,
        max_chunks=config.batch_max_chunks,
    )

    logger.info("Initializing model...")
    model = factory.build_model(config, BRENDA_SCHEMA)
    model.register_load_state_dict_pre_hook(factory.fix_keys_hook)
    model.load_state_dict(saved.state_dict)

    # BrendaClassificationModel (and ETE, through it) declares this whether or
    # not it has a span tagger; NERClassification never does, and has nothing
    # to split by novelty.
    if hasattr(model, "training_entity_ids"):
        model.training_entity_ids = saved.vocabulary.entity_ids

    model.to(model.device)

    # A run of its own rather than the training run that produced the
    # checkpoint: attaching to that one needs its id recorded inside the
    # checkpoint, which no existing checkpoint carries. The `checkpoint` tag is
    # what links the two, and `stage = "eval"` keeps test-set numbers out of a
    # run list being scanned for training curves.
    with tracking.run(
        name=tracking.stamped(pathlib.Path(args.model_state_dict).stem),
        params=config.model_dump(),
        tags={
            "stage": "eval",
            "checkpoint": args.model_state_dict,
            "checkpoint_token_labels": labels_provenance,
            "checkpoint_encodings": inputs_provenance,
            **tracking.provenance_tags(config.model_class, config.base_model),
            **tracking.environment_tags(config.base_model),
        },
    ):
        tracking.log_metrics(
            {
                **factory.dataset_metrics(dataset, BRENDA_SCHEMA),
                **factory.model_metrics(model),
            }
        )
        tracking.log_artifact(args.config)
        model.evaluate_model(eval_data)
        report_linking(machine_config().linking_corpora)
        report_predicted_linking(
            machine_config().linking_corpora,
            encodings_path(encodings[config.base_model]),
            model,
        )


if __name__ == "__main__":
    main()
