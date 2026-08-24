#!/usr/bin/env python

import argparse
import pathlib

import torch
from d3text import data, factory, runtime, tracking
from d3text.models.config import encodings, load_model_config


def command_line_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="evaluate",
        description=("Evaluate a model with the provided configuration."),
    )
    parser.add_argument(
        "config", help="Configuration file for the model to be evaluated."
    )
    parser.add_argument("model_state_dict", help="Model state dict")
    parser.add_argument("--limit", type=int, default=None)

    return parser.parse_args()


def main() -> None:
    runtime.configure()
    args = command_line_args()
    config = load_model_config(args.config)

    print("Loading evaluation dataset...")
    if args.limit is not None:
        dataset = data.brenda_dataset(
            encodings=encodings[config.base_model], limit=args.limit
        )
    else:
        dataset = data.brenda_dataset(encodings=encodings[config.base_model])
    eval_data = data.get_batch_loader(
        dataset=dataset.data["test"],
        batch_size=1,
        max_chunks=config.batch_max_chunks,
    )

    print("Initializing model...")
    model = factory.build_model(config, dataset)
    model.register_load_state_dict_pre_hook(factory.fix_keys_hook)
    state_dict = torch.load(args.model_state_dict)
    model.load_state_dict(state_dict)

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
