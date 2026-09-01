#!/usr/bin/env python

import argparse
import logging
import pathlib
import warnings

from d3text import checkpoint, data, factory, runtime, tracking
from d3text.datasets.brenda import BRENDA_SCHEMA, brenda_dataset
from d3text.models.config import encodings, load_model_config
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


def main() -> None:
    runtime.configure()
    args = command_line_args()
    config = load_model_config(args.config)

    # Read before the corpus: the vocabulary it carries decides how the corpus
    # is indexed, and a missing or unreadable checkpoint should not cost the
    # ~300 MB load first.
    logger.info("Loading checkpoint...")
    saved = checkpoint.load(args.model_state_dict)

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


if __name__ == "__main__":
    main()
