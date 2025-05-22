#!/usr/bin/env python

import argparse

import torch
from d3text import data, models
from d3text.models.config import load_model_config


def command_line_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="evaluate.py",
        description=("Evaluate a model with the provided configuration."),
    )
    parser.add_argument(
        "config", help="Configuration file for the model to be evaluated."
    )
    parser.add_argument("model_state_dict", help="Model state dict")

    return parser.parse_args()


if __name__ == "__main__":
    args = command_line_args()
    config = load_model_config(args.config)

    print("Loading evaluation dataset...")
    dataset = data.brenda_dataset()
    eval_data = data.get_batch_loader(
        dataset=dataset.data["test"], batch_size=1
    )

    print("Initializing model...")
    mclass = getattr(models, config.model_class)
    model = mclass(classes=dataset.class_map, config=config)
    state_dict = torch.load(args.model_state_dict)
    model.load_state_dict(state_dict)

    model.to(model.device)
    model.evaluate_model(eval_data)
