#!/usr/bin/env python

import argparse
import itertools
import logging
import pathlib

from d3text import (
    checkpoint,
    data,
    encodings_store,
    factory,
    runtime,
    token_labels,
    tracking,
)
from d3text.cli.args import non_negative_limit
from d3text.datasets.brenda import BRENDA_SCHEMA, brenda_dataset
from d3text.models.base import Model
from d3text.models.config import (
    encodings_path,
    load_model_config,
    token_labels_path,
)
from d3text.progress import batch_progress
from d3text.training.trainer import Trainer
from d3text.vocabulary import Vocabulary
from torch.profiler import ProfilerActivity, profile, schedule
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

_PROFILE_WARMUP_STEPS = 10
_PROFILE_ACTIVE_STEPS = 10


def command_line_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="train",
        description=(
            "Train a model with the provided configuration and save the "
            "resulting checkpoint to the given output file."
        ),
    )
    parser.add_argument(
        "config", help="Configuration file for the model to be trained."
    )
    parser.add_argument("output", help="Location to save the trained model.")
    parser.add_argument("-prof", action="store_true")
    parser.add_argument("--limit", type=non_negative_limit, default=None)
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


def profile_training(model: Model, loader: DataLoader) -> None:
    """Profile real training steps and log the costliest operators.

    Each step is what `Model.run_epoch` runs — forward, backward, clip and
    optimizer step, with the trunk compiled and batches from the training
    loader — so the table measures training rather than one repeated forward.
    The warmup steps keep compilation and allocator growth out of the table.

    :param model: the model to profile; its weights are updated, so it is not
        worth saving afterwards.
    :param loader: the training loader, drawn from as training would.
    """
    update = Trainer(model).update
    model.compile_trunk()
    model.train()
    steps = _PROFILE_WARMUP_STEPS + _PROFILE_ACTIVE_STEPS
    on_cuda = model.device.startswith("cuda")
    activities = [ProfilerActivity.CPU]
    if on_cuda:
        activities.append(ProfilerActivity.CUDA)
    logger.info(
        "Profiling %d training steps after %d warmup steps:",
        _PROFILE_ACTIVE_STEPS,
        _PROFILE_WARMUP_STEPS,
    )
    with profile(
        activities=activities,
        schedule=schedule(
            wait=0,
            warmup=_PROFILE_WARMUP_STEPS,
            active=_PROFILE_ACTIVE_STEPS,
            repeat=1,
        ),
        with_stack=True,
        profile_memory=True,
        acc_events=True,
    ) as prof:
        taken = 0
        for batch in itertools.islice(
            model.prefetch_layer_boundary_reads(batch_progress(loader)), steps
        ):
            update.zero_grad()
            update(*model.compute_losses(batch, 0).values())
            prof.step()
            taken += 1
    if taken < steps:
        logger.warning(
            "The training split ran out after %d of %d steps; the profile "
            "covers %d of %d active steps.",
            taken,
            steps,
            max(0, taken - _PROFILE_WARMUP_STEPS),
            _PROFILE_ACTIVE_STEPS,
        )
    logger.info(
        "%s",
        prof.key_averages(group_by_stack_n=20).table(
            sort_by="self_device_time_total"
            if on_cuda
            else "self_cpu_time_total",
            row_limit=20,
        ),
    )


def main() -> None:
    args = command_line_args()
    config = load_model_config(args.config)
    # After the config is read so the seed comes from it, and still before any
    # CUDA work: parsing arguments and reading a TOML file touch no device, and
    # the caching allocator reads its environment variable when it first
    # initialises.
    runtime.configure(seed=config.seed)
    batch_size = config.batch_size
    encodings_file = encodings_path(config.base_model)
    labels_path = token_labels_path(config)
    labels_digest = token_labels.store_index_digest(labels_path)
    rules_digest = token_labels.store_labelling_rules_digest(labels_path)
    stale_rules = token_labels.stale_labelling_rules(labels_path)
    if stale_rules is not None:
        logger.warning("%s", stale_rules)
    encodings_digest = encodings_store.store_content_digest(encodings_file)

    logger.info("Loading dataset...")
    dataset = brenda_dataset(
        schema=BRENDA_SCHEMA,
        encodings=encodings_file,
        limit=args.limit,
        base_model=config.base_model,
        split_names=("train", "val"),
    )

    train_data = dataset.data["train"]
    logger.info("Initializing model...")
    model = factory.build_model(
        config,
        BRENDA_SCHEMA,
        class_freqs=data.compute_frequencies(train_data, column="classes"),
    )

    model.to(model.device)

    logger.info("model size: %.3fMB", factory.model_size_mb(model))

    if args.prof:
        profile_training(
            model,
            data.get_batch_loader(
                dataset=train_data,
                batch_size=batch_size,
                max_chunks=config.batch_max_chunks,
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
        compiled = model.compile_trunk()
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
                **tracking.environment_tags(config.base_model),
            },
        ):
            tracking.log_metrics(
                {
                    **factory.dataset_metrics(dataset, BRENDA_SCHEMA),
                    **factory.model_metrics(model),
                }
            )
            try:
                best_state = Trainer(model).fit(
                    train_data=train_data_loader,
                    val_data=val_data_loader,
                    save_checkpoint=True,
                )
            finally:
                # The backend does not run until the first batch, so the tag
                # set when the run opened records what was installed; this is
                # the first point it can say what the epochs actually
                # executed. It sits in a `finally` because a run that died
                # mid-epoch is the one someone later filters for when asking
                # whether the compiler was implicated.
                tracking.set_tags(
                    {"compiled": str(model.trunk_is_compiled()).lower()}
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

            # The vocabulary travels with the weights: the class head's
            # columns are positional and this training split is the only thing
            # that says which class owns which, and which entities it named.
            # `evaluate` reads it back rather than re-deriving it from a corpus
            # that has since moved. The three store digests travel for the same
            # reason: which strings the label dictionary named, and what the
            # sweep did with that answer, is what set the span targets, and
            # which ids the encodings hold is what the heads ever saw.
            checkpoint.save(
                args.output,
                best_state,
                Vocabulary.from_class_map(dataset.class_map),
                token_labels_digest=labels_digest,
                labelling_rules_digest=rules_digest,
                encodings_digest=encodings_digest,
            )
            tracking.log_artifact(args.config)
            if args.log_checkpoint:
                tracking.log_artifact(args.output)

        logger.info("Model saved to %s.", args.output)


if __name__ == "__main__":
    main()
