#!/usr/bin/env python

import argparse
import functools
import itertools
import logging
import pathlib
import tempfile

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
from d3text.cli.args import non_negative_limit
from d3text.datasets.brenda import BRENDA_SCHEMA, brenda_dataset
from d3text.models.base import Model, Step
from d3text.models.config import (
    encodings_path,
    load_model_config,
    token_labels_path,
)
from d3text.progress import batch_progress
from d3text.training.trainer import ResumeFile, Trainer
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
    parser.add_argument(
        "output",
        help=(
            "Location to save the trained model, or, under -prof, the "
            "chrome trace to write instead."
        ),
    )
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
    parser.add_argument(
        "--register-model",
        metavar="NAME",
        default=None,
        help=(
            "Log the trained model to the MLflow run as a PyTorch model and "
            "register it as a new version of NAME, tagged with the git "
            "release it was trained from."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Continue the run whose resume file, OUTPUT with the suffix "
            ".resume.pt, the last finished epoch left behind."
        ),
    )

    args = parser.parse_args()
    if args.prof and args.resume:
        parser.error("-prof trains no run to resume")
    if args.register_model is not None:
        # Refused here rather than at the end: a multi-hour run should not
        # finish before saying it had nowhere to register.
        if args.prof:
            parser.error("-prof trains no model to register")
        if not tracking.enabled():
            parser.error(
                f"--register-model needs a tracking server; set "
                f"{tracking.TRACKING_URI_VAR}"
            )
    return args


def profile_training(
    model: Model, loader: DataLoader, trace_path: str | pathlib.Path
) -> None:
    """Profile real training steps, log the costliest operators and export a
    chrome trace.

    Each step is a full `Model.run_epoch` step on real batches, after
    warmup steps that keep compilation and allocator growth out of the
    table. Stacks stay on so the trace separates a step's Python-level
    phases, which the table cannot.

    :param model: the model to profile; its weights are updated, so it is not
        worth saving afterwards.
    :param loader: the training loader, drawn from as training would.
    :param trace_path: where to write the chrome trace (JSON, loadable in
        chrome://tracing or https://ui.perfetto.dev). No checkpoint is
        written under `-prof`, so `train`'s `OUTPUT` argument names this
        file instead.
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
    model.log_pass_stats(Step.TRAINING)
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
    prof.export_chrome_trace(str(trace_path))
    logger.info("Wrote chrome trace to %s", trace_path)


def main() -> None:
    args = command_line_args()
    config = load_model_config(args.config)
    # After the config (for the seed), before any CUDA work: the caching
    # allocator reads its environment variable when it first initialises.
    runtime.configure(seed=config.seed)
    # Before training, so a malformed manifest (`ValueError`, uncaught)
    # fails fast rather than after `Trainer.fit`; `-prof` saves nothing.
    surface_form_index = None
    if not args.prof:
        logger.info("Building surface-form index...")
        surface_form_index = linking_corpora.brenda_index()
        if surface_form_index is None:
            logger.warning(
                "no surface-form index could be built, so this "
                "checkpoint carries none and `infer` will link no span "
                "against it; linking_corpora.brenda_index's own warning "
                "names why"
            )
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
    model = factory.build_model_for_dataset(
        config,
        BRENDA_SCHEMA,
        dataset,
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
            args.output,
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
        recompile_hits_before = runtime.recompile_limit_hits()
        resume_file = ResumeFile(
            pathlib.Path(args.output).with_suffix(".resume.pt"),
            inputs={
                "config": config.model_dump(mode="json"),
                "limit": args.limit,
                "token_labels_digest": labels_digest,
                "labelling_rules_digest": rules_digest,
                "encodings_digest": encodings_digest,
            },
        )
        resume_from = resume_file.read() if args.resume else None
        logger.info("Training:")
        with tracking.run(
            name=tracking.stamped(pathlib.Path(args.output).stem),
            run_id=None if resume_from is None else resume_from["run_id"],
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
                    resume_file=resume_file,
                    resume_from=resume_from,
                )
            finally:
                # Only now known what the epochs ran under; `finally` so a
                # run that died mid-epoch still says whether it was compiled.
                hit = runtime.recompile_limit_hits() > recompile_hits_before
                tracking.set_tags(
                    {
                        "compiled": str(model.trunk_is_compiled()).lower(),
                        "recompile_limit_hit": str(hit).lower(),
                    }
                )
            if best_state is None:
                # Empty only if no epoch ever improved (validation loss NaN
                # throughout); the warning says these are not a best epoch.
                logger.warning(
                    "Training kept no best-epoch snapshot; saving the "
                    "parameters the last epoch left in place."
                )
                best_state = model.state_dict()

            # Vocabulary and store digests pin what the positional heads and
            # span targets meant at training time; the surface-form index
            # lets `infer` link without the BRENDA data.
            save = functools.partial(
                checkpoint.save,
                vocabulary=Vocabulary.from_class_map(dataset.class_map),
                token_labels_digest=labels_digest,
                labelling_rules_digest=rules_digest,
                encodings_digest=encodings_digest,
                surface_form_index=surface_form_index,
            )
            save(args.output, best_state)
            resume_file.path.unlink()
            tracking.log_artifact(args.config)
            if args.log_checkpoint:
                tracking.log_artifact(args.output)
            if args.register_model is not None:
                # The pickled module carries the weights, so the sidecar
                # carries everything but, rather than a second copy.
                model.load_state_dict(best_state, strict=True)
                with tempfile.TemporaryDirectory() as scratch:
                    sidecar = pathlib.Path(scratch, "checkpoint_metadata.pt")
                    save(sidecar, {})
                    tracking.register_model(
                        model,
                        sidecar,
                        args.register_model,
                        tracking.provenance_tags(
                            config.model_class, config.base_model
                        ),
                    )

        logger.info("Model saved to %s.", args.output)


if __name__ == "__main__":
    main()
