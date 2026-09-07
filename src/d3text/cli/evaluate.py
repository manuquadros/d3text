#!/usr/bin/env python

import argparse
import logging
import pathlib
import warnings

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
from d3text.datasets.brenda import (
    BRENDA_SCHEMA,
    brenda_dataset,
    encodings_path,
)
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
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Truncate the training split, and with it the entity vocabulary "
            "derived from it. Only consulted for a checkpoint written before "
            "the vocabulary was recorded, where it must reproduce the value "
            "the training run was given; a recorded vocabulary makes it "
            "unnecessary and it is ignored."
        ),
    )

    return parser.parse_args()


def load_evaluation_dataset(
    config_base_model: str,
    vocabulary: Vocabulary | None,
    limit: int | None,
) -> data.EntityRelationDataset:
    """The dataset to score a checkpoint on, indexed the way it was trained.

    A recorded vocabulary is authoritative and the training split is not loaded
    at all, which is also what makes `--limit` irrelevant: the flag resized the
    entity head by resizing the split it was derived from. Without one the
    order is rebuilt from the training split behind a warning, valid only if
    `--limit`, `noise=` and the corpus all match the training run.

    :param config_base_model: the model the encodings must have been built
        with.
    :param vocabulary: the checkpoint's recorded column order, if it has one.
    :param limit: the training-split truncation to reproduce, read only when
        rebuilding.
    :return: the indexed splits.
    """
    encodings_file = encodings[config_base_model]

    if vocabulary is not None:
        if limit is not None:
            warnings.warn(
                "--limit is ignored: the checkpoint records its own entity "
                f"vocabulary ({len(vocabulary)} entities), so the evaluation "
                "does not derive one from the training split",
                RuntimeWarning,
                stacklevel=2,
            )
        return brenda_dataset(
            schema=BRENDA_SCHEMA,
            encodings=encodings_file,
            vocabulary=vocabulary,
            split_names=("test",),
            base_model=config_base_model,
        )

    warnings.warn(
        "this checkpoint records no entity vocabulary, so the entity and "
        "class columns are being rebuilt from the training split. They match "
        "the ones it was trained on only if --limit, the noise counts and the "
        "corpus are all as they were then; a mismatch in width fails on load, "
        "and one in order does not fail at all.",
        RuntimeWarning,
        stacklevel=2,
    )
    return brenda_dataset(
        schema=BRENDA_SCHEMA,
        encodings=encodings_file,
        limit=limit,
        base_model=config_base_model,
    )


def token_labels_provenance(recorded: str | None, current: str | None) -> str:
    """Say whether this run's label store is the one the checkpoint trained on.

    Warns where `token_labels.check_index` refuses, because the two guard
    different things: that one is about to *extend* a store whose halves would
    then label the same string differently, while a mismatch here only makes
    the detection scores incomparable with the training run's. The scores are
    still the scores; silence about which dictionary set their denominator is
    what has to go.

    :param recorded: the digest the checkpoint carries, if any.
    :param current: the digest of the store this run reads, if any.
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

    if recorded == current:
        return "matched"

    reads = f"index {current[:12]}" if current else "no label store"
    warnings.warn(
        "this checkpoint was trained on targets from surface-form index "
        f"{recorded[:12]}, but this run reads {reads}; the two dictionaries "
        "name different strings, so the detection metrics count a different "
        "set of gold spans than the model was trained against and are not "
        "comparable with that run's.",
        RuntimeWarning,
        stacklevel=2,
    )
    return "mismatched"


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


def main() -> None:
    runtime.configure()
    args = command_line_args()
    config = load_model_config(args.config)

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
    )
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
        limit=args.limit,
    )
    eval_data = data.get_batch_loader(
        dataset=dataset.data["test"],
        batch_size=1,
        max_chunks=config.batch_max_chunks,
    )

    logger.info("Initializing model...")
    model = factory.build_model(config, dataset, BRENDA_SCHEMA)
    model.register_load_state_dict_pre_hook(factory.fix_keys_hook)
    model.load_state_dict(saved.state_dict)

    model.to(model.device)

    # A run of its own rather than the training run that produced the
    # checkpoint: attaching to that one needs its id recorded inside the
    # checkpoint, which no existing checkpoint carries. The `checkpoint` tag is
    # what links the two, and `stage = "eval"` keeps test-set numbers out of a
    # run list being scanned for training curves.
    with tracking.run(
        name=tracking.stamped(pathlib.Path(args.model_state_dict).stem),
        params={**config.model_dump(), "limit": args.limit},
        tags={
            "stage": "eval",
            "checkpoint": args.model_state_dict,
            "checkpoint_vocabulary": (
                "recorded" if saved.vocabulary is not None else "rebuilt"
            ),
            "checkpoint_token_labels": labels_provenance,
            "checkpoint_encodings": inputs_provenance,
            **tracking.provenance_tags(config.model_class, config.base_model),
            **tracking.environment_tags(),
        },
    ):
        tracking.log_metrics(
            {
                **factory.dataset_metrics(dataset),
                **factory.model_metrics(model),
            }
        )
        tracking.log_artifact(args.config)
        model.evaluate_model(eval_data)
        report_linking(machine_config().linking_corpora)


if __name__ == "__main__":
    main()
