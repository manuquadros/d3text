#!/usr/bin/env python

import argparse
import logging
import typing
from pprint import pformat

import torch
import torch._dynamo
from d3text import data, factory, runtime, tracking, utils
from d3text.factory import ConfigurableModel
from d3text.models.config import encodings, load_tuning_config

logger = logging.getLogger(__name__)


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
    parser.add_argument("--limit", type=int, default=None)

    return parser.parse_args()


def main() -> None:
    runtime.configure()
    args = command_line_args()
    logger.info("Loading hyperparameter configurations...")
    configs = load_tuning_config(args.config)

    for trial, config in enumerate(configs):
        encodings_file = encodings[config.base_model]

        logger.info("%s", pformat(config.model_dump()))
        logger.info("Loading dataset...")
        dataset = data.brenda_dataset(
            encodings=encodings_file, limit=args.limit
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
            dataset,
            entity_freqs=data.compute_frequencies(
                train_data, column="entities"
            ),
            class_freqs=data.compute_frequencies(train_data, column="classes"),
        )

        model.to(model.device)
        if config.base_layers_to_unfreeze:
            model.unfreeze_encoder_layers(n=config.base_layers_to_unfreeze)

        compiled = False
        if runtime.is_triton_compatible():
            try:
                # Typed as a bare callable, but it returns a wrapper that
                # forwards attribute access to the module it wrapped.
                model = typing.cast(
                    ConfigurableModel, torch.compile(model, dynamic=True)
                )
                compiled = True
            except Exception as e:
                logger.warning("Failed to compile with Triton: %s", e)
                logger.warning(
                    "Skipping torch.compile(): GPU too old for Triton"
                )

        with tracking.run(
            name=tracking.stamped(f"{config.model_class}-{trial:03d}"),
            params={**config.model_dump(), "limit": args.limit},
            tags={
                "stage": "tuning",
                "sweep": args.config,
                "trial": str(trial),
                "compiled": str(compiled).lower(),
                **tracking.provenance_tags(
                    config.model_class, config.base_model
                ),
                **tracking.environment_tags(),
            },
        ):
            tracking.log_metrics(
                {
                    **factory.dataset_metrics(dataset),
                    **factory.model_metrics(model),
                }
            )
            try:
                logger.info("Running config...")
                model.train_model(
                    train_data=train_data_loader,
                    val_data=val_data_loader,
                    save_checkpoint=False,
                )
            except Exception:
                logger.exception("Trial %d failed", trial)
                raise
            else:
                utils.log_config(
                    args.output, config, val_loss=model.best_val_loss
                )


if __name__ == "__main__":
    main()
