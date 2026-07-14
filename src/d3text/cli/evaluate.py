#!/usr/bin/env python

import argparse

import torch
from d3text import data, factory, runtime
from d3text.datasets.brenda import BRENDA_SCHEMA, brenda_dataset
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
        dataset = brenda_dataset(
            BRENDA_SCHEMA,
            encodings=encodings[config.base_model],
            limit=args.limit,
        )
    else:
        dataset = brenda_dataset(
            BRENDA_SCHEMA, encodings=encodings[config.base_model]
        )
    eval_data = data.get_batch_loader(
        dataset=dataset.data["test"], batch_size=1
    )

    print("Initializing model...")
    model = factory.build_model(config, dataset)
    model.register_load_state_dict_pre_hook(factory.fix_keys_hook)
    state_dict = torch.load(args.model_state_dict)
    model.load_state_dict(state_dict)

    model.to(model.device)
    model.evaluate_model(eval_data)


if __name__ == "__main__":
    main()
