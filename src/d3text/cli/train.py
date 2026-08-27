#!/usr/bin/env python

import argparse
import logging
import pathlib

import torch
import torch._dynamo
from d3text import checkpoint, data, factory, runtime, tracking
from d3text.models.config import encodings, load_model_config
from d3text.training.trainer import Trainer
from d3text.vocabulary import Vocabulary
from torch.profiler import ProfilerActivity, profile
from torch.utils.data import SequentialSampler

logger = logging.getLogger(__name__)


def command_line_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="train",
        description=(
            "Train a model with the provided configuration and saves the resulting"
            "parameters in the file provided with the -f flag."
        ),
    )
    parser.add_argument(
        "config", help="Configuration file for the model to be trained."
    )
    parser.add_argument("output", help="Location to save the trained model.")
    parser.add_argument("-prof", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--log-checkpoint",
        action="store_true",
        help=(
            "Upload the saved checkpoint to the MLflow run. Off by default: "
            "the state dict carries the frozen base model, so it is hundreds "
            "of MB per run."
        ),
    )

    return parser.parse_args()


def main() -> None:
    runtime.configure()
    args = command_line_args()
    config = load_model_config(args.config)
    batch_size = config.batch_size
    encodings_file = encodings[config.base_model]

    logger.info("Loading dataset...")
    dataset = data.brenda_dataset(
        encodings=encodings_file,
        limit=args.limit,
        base_model=config.base_model,
    )

    train_data = dataset.data["train"]
    logger.info("Initializing model...")
    model = factory.build_model(
        config,
        dataset,
        entity_freqs=data.compute_frequencies(train_data, column="entities"),
        class_freqs=data.compute_frequencies(train_data, column="classes"),
    )

    model.to(model.device)

    logger.info("model size: %.3fMB", factory.model_size_mb(model))

    if args.prof:
        torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH)
        train_data_loader = data.get_batch_loader(
            dataset=train_data,
            batch_size=batch_size,
            sampler=SequentialSampler(data_source=train_data),
        )
        logger.info("Profiling:")
        batch = next(iter(train_data_loader))
        logger.info("Profiled batch: %s", batch[0]["id"].item())
        with torch.no_grad():
            _ = model.compute_batch_losses(batch)
        # inputs = model.get_token_embeddings(batch)
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            with_stack=True,
            profile_memory=True,
        ) as prof:
            for _ in range(25):
                model.compute_batch_losses(batch)
        logger.info(
            "%s",
            prof.key_averages(group_by_stack_n=20).table(
                sort_by="self_cpu_time_total", row_limit=20
            ),
        )
    else:
        train_data_loader = data.get_batch_loader(
            dataset=train_data,
            batch_size=batch_size,
            max_chunks=config.batch_max_chunks,
        )
        val_data_loader = data.get_batch_loader(
            dataset=dataset.data["val"],
            batch_size=batch_size,
            max_chunks=config.batch_max_chunks,
        )
        compiled = runtime.compile_model(model)
        logger.info("Training:")
        with tracking.run(
            name=tracking.stamped(pathlib.Path(args.output).stem),
            params={**config.model_dump(), "limit": args.limit},
            tags={
                "stage": "train",
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
            best_state = Trainer(model).fit(
                train_data=train_data_loader,
                val_data=val_data_loader,
                save_checkpoint=True,
            )
            if best_state is None:
                # With validation data and `save_checkpoint=True` the trainer
                # snapshots every epoch that improves on the one before, so it
                # comes back empty only when none ever did — a run whose
                # validation loss was NaN throughout. Those parameters still
                # cost what they cost; the warning is what says they are not a
                # chosen best epoch.
                logger.warning(
                    "Training kept no best-epoch snapshot; saving the "
                    "parameters the last epoch left in place."
                )
                best_state = model.state_dict()

            # The vocabulary travels with the weights: the entity head's
            # columns are positional and this training split is the only thing
            # that says which entity owns which. `evaluate` reads it back
            # rather than re-deriving it from a corpus that has since moved.
            checkpoint.save(
                args.output,
                best_state,
                Vocabulary.from_index(dataset.entity_index, dataset.class_map),
            )
            tracking.log_artifact(args.config)
            if args.log_checkpoint:
                tracking.log_artifact(args.output)

        logger.info("Model saved to %s.", args.output)


if __name__ == "__main__":
    main()
