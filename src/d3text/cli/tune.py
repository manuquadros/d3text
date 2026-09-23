#!/usr/bin/env python

import argparse
import ast
import csv
import gc
import logging
import pathlib
from functools import lru_cache
from pprint import pformat

import torch
import torch._dynamo

from d3text import data, factory, runtime, tracking, utils
from d3text.cli.args import non_negative_limit
from d3text.datasets.brenda import BRENDA_SCHEMA, brenda_dataset
from d3text.models.config import ModelConfig, encodings, load_tuning_config
from d3text.training.trainer import Trainer

logger = logging.getLogger(__name__)


def _logged_configs(path: str) -> list[ModelConfig]:
    """Read configurations already attempted in a tuning results file."""
    output = pathlib.Path(path)
    if not output.exists() or output.stat().st_size == 0:
        return []

    configs = []
    with output.open(newline="") as stream:
        for row in csv.DictReader(stream):
            values = {}
            for field in ModelConfig.model_fields:
                if field not in row:
                    continue
                raw = row[field]
                try:
                    values[field] = ast.literal_eval(raw)
                except (ValueError, SyntaxError):
                    values[field] = raw
            configs.append(ModelConfig(**values))
    return configs


@lru_cache(maxsize=1)
def _dataset_for(base_model: str, limit: int | None):
    """Build the dataset and training-split class frequencies for one
    `(base_model, limit)` pair, keeping only the most recent pair resident.

    A sweep's dataset depends on nothing else it varies (lr, dropout,
    pooling, ...), so caching on this key alone lets trials that only change
    those skip the ~500 MB split-CSV parse that `brenda_dataset` pays.

    :param base_model: the trial's base transformer, keys `encodings`.
    :param limit: the `--limit` flag's value, or None.
    :return: the dataset and its training split's class frequencies.

    Left return-unannotated on purpose: `brenda_dataset` and
    `compute_frequencies` already carry the real contract, and beartype
    would otherwise re-check this thin wrapper's return against the full
    `EntityRelationDataset`/`BrendaDataset` shape, which only a live corpus
    satisfies — exactly what a unit test stubs away.
    """
    dataset = brenda_dataset(
        schema=BRENDA_SCHEMA,
        encodings=encodings[base_model],
        limit=limit,
        base_model=base_model,
        split_names=("train", "val"),
    )
    class_freqs = data.compute_frequencies(
        dataset.data["train"], column="classes"
    )
    return dataset, class_freqs


def command_line_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="tuning",
        description=(
            "Tune a model with the provided configuration and saves the results"
            "output file."
        ),
    )
    parser.add_argument("config", help="Tuning config file.")
    parser.add_argument("output", help="Location to save the results.")
    parser.add_argument("--limit", type=non_negative_limit, default=None)

    return parser.parse_args()


def main() -> None:
    runtime.configure()
    args = command_line_args()
    logger.info("Loading hyperparameter configurations...")
    configs = load_tuning_config(
        args.config, excluded=_logged_configs(args.output)
    )

    failed = 0
    attempted = 0
    for trial, config in enumerate(configs):
        attempted += 1
        # Reseeded per trial rather than once for the sweep: otherwise each
        # trial starts from the RNG state the trial before it left, and a
        # configuration's score depends on where in the sweep it was drawn.
        runtime.set_seed(config.seed)
        logger.info("%s", pformat(config.model_dump(), sort_dicts=False))

        trainer = model = train_data_loader = val_data_loader = None

        # The run opens before any of setup runs, not just around `fit`:
        # every param and tag it opens with comes from `config`/`args`
        # alone, so a config a model constructor rejects, or one whose
        # parameters do not fit the device at `model.to`, gets the same
        # FAILED run and NaN row as a trial that dies mid-epoch, instead of
        # taking the whole sweep down with it.
        try:
            with tracking.run(
                name=tracking.stamped(f"{config.model_class}-{trial:03d}"),
                params={**config.model_dump(), "limit": args.limit},
                tags={
                    "stage": "tuning",
                    "sweep": args.config,
                    "trial": str(trial),
                    **tracking.provenance_tags(
                        config.model_class, config.base_model
                    ),
                    **tracking.environment_tags(config.base_model),
                },
            ):
                try:
                    logger.info("Loading dataset...")
                    dataset, class_freqs = _dataset_for(
                        config.base_model, args.limit
                    )
                    train_data = dataset.data["train"]
                    train_data_loader = data.get_batch_loader(
                        dataset=train_data,
                        batch_size=config.batch_size,
                        max_chunks=config.batch_max_chunks,
                    )
                    val_data_loader = data.get_batch_loader(
                        dataset=dataset.data["val"],
                        batch_size=config.batch_size,
                        max_chunks=config.batch_max_chunks,
                    )

                    logger.info("Loading model...")
                    model = factory.build_model(
                        config,
                        BRENDA_SCHEMA,
                        class_freqs=class_freqs,
                    )
                    model.to(model.device)

                    # Only a prediction until the first batch actually
                    # drives the backend; the `finally` below retags with
                    # what happened.
                    compiled = model.compile_trunk()
                    tracking.set_tags({"compiled": str(compiled).lower()})
                    trainer = Trainer(model)

                    tracking.log_metrics(
                        {
                            **factory.dataset_metrics(dataset),
                            **factory.model_metrics(model),
                        }
                    )
                    logger.info("Running config...")
                    trainer.fit(
                        train_data=train_data_loader,
                        val_data=val_data_loader,
                        save_checkpoint=False,
                    )
                finally:
                    # The backend does not run until the first batch, so the
                    # tag set after `compile_trunk` records only what was
                    # installed; this is the first point that can say what
                    # the epochs actually ran. It sits in a `finally`
                    # because a trial that died mid-epoch is the one someone
                    # later filters for when asking whether the compiler was
                    # implicated. Guarded because a trial that died before
                    # `build_model` never had one to ask.
                    if model is not None:
                        tracking.set_tags(
                            {"compiled": str(model.trunk_is_compiled()).lower()}
                        )
                utils.log_config(
                    args.output, config, val_loss=trainer.best_val_loss
                )
        except Exception:
            failed += 1
            logger.exception("Trial %d failed", trial)
            utils.log_config(args.output, config, val_loss=float("nan"))
        finally:
            # The next trial's model is built before the loop rebinds these,
            # so without this two are resident at once; a trial that died
            # during setup never bound some of them, hence the `None`
            # prebinding above. `gc.collect()` because the eager fallback
            # leaves a cycle on the model. On unified memory the overshoot
            # arrives as the kernel OOM killer, not a CUDA error.
            del trainer, model, train_data_loader, val_data_loader
            torch._dynamo.reset()
            gc.collect()
            torch.cuda.empty_cache()

    if failed and failed == attempted:
        raise SystemExit(f"tuning: all {failed} trials failed")


if __name__ == "__main__":
    main()
