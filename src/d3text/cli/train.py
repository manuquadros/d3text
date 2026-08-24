#!/usr/bin/env python

import argparse
import pathlib
import typing

import torch
import torch._dynamo
from d3text import data, factory, runtime, tracking
from d3text.factory import ConfigurableModel
from d3text.models.config import encodings, load_model_config
from torch.profiler import ProfilerActivity, profile
from torch.utils.data import SequentialSampler


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

    print("Loading dataset...")
    if args.limit is not None:
        dataset = data.brenda_dataset(
            encodings=encodings_file, limit=args.limit
        )
    else:
        dataset = data.brenda_dataset(encodings=encodings_file)

    train_data = dataset.data["train"]
    print("Initializing model...")
    model = factory.build_model(
        config,
        dataset,
        entity_freqs=data.compute_frequencies(train_data, column="entities"),
        class_freqs=data.compute_frequencies(train_data, column="classes"),
    )

    model.to(model.device)
    if config.base_layers_to_unfreeze:
        model.unfreeze_encoder_layers(n=config.base_layers_to_unfreeze)

    print(f"model size: {factory.model_size_mb(model):.3f}MB")

    if args.prof:
        torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH)
        train_data_loader = data.get_batch_loader(
            dataset=train_data,
            batch_size=batch_size,
            sampler=SequentialSampler(data_source=train_data),
        )
        print("Profiling:")
        batch = next(iter(train_data_loader))
        print(batch[0]["id"].item())
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
        print(
            prof.key_averages(group_by_stack_n=20).table(
                sort_by="self_cpu_time_total", row_limit=20
            )
        )
    else:
        # Use memory efficient attention if available
        if hasattr(model.base_model, "config"):
            model.base_model.config.use_memory_efficient_attention = True
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
        compiled = False
        if runtime.is_triton_compatible():
            try:
                # `torch.compile` is typed as returning a bare callable, but it
                # hands back a wrapper that forwards attribute access to the
                # module it wrapped — which is also why the checkpoint saved
                # below carries the `_orig_mod.` prefix `evaluate` strips.
                model = typing.cast(
                    ConfigurableModel, torch.compile(model, dynamic=True)
                )
                compiled = True
            except Exception as e:
                print(f"Failed to compile with Triton: {e}")
                print("Skipping torch.compile(): GPU too old for Triton")
        print("Training:")
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
            model.train_model(
                train_data=train_data_loader,
                val_data=val_data_loader,
                save_checkpoint=True,
            )

            torch.save(model.state_dict(), args.output)
            tracking.log_artifact(args.config)
            if args.log_checkpoint:
                tracking.log_artifact(args.output)

        print(f"Model saved to {args.output}.")


if __name__ == "__main__":
    main()
