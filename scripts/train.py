#!/usr/bin/env python

import argparse

import torch
import torch._dynamo

from d3text import data, models
from d3text.models.config import load_model_config  # , save_model_config

# import os


torch.set_float32_matmul_precision("high")


def command_line_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="train.py",
        description=(
            "Train a model with the provided configuration and saves the resulting"
            "parameters in the file provided with the -f flag."
        ),
    )
    parser.add_argument(
        "config", help="Configuration file for the model to be trained."
    )
    parser.add_argument("output", help="Location to save the trained model.")

    return parser.parse_args()


if __name__ == "__main__":
    args = command_line_args()
    config = load_model_config(args.config)
    batch_size = config.batch_size

    print("Loading dataset...")
    dataset = data.brenda_dataset()
    train_data = dataset.data["train"]
    train_data_loader = data.get_batch_loader(
        dataset=train_data, batch_size=batch_size
    )
    val_data_loader = data.get_batch_loader(
        dataset=dataset.data["val"], batch_size=batch_size
    )

    print("Initializing model...")
    mclass = getattr(models, config.model_class)
    model = mclass(classes=dataset.class_map, config=config)

    model.to(model.device)

    model.compile(dynamic=True)

    print("Training:")
    model.train_model(train_data=train_data_loader)

    # print(model.evaluate_model(val_data))

    # torch.save(model.state_dict(), args.output)
    # model.save_config(os.path.splitext(args.output)[0] + "_config.toml")

    # print(f"Model saved to {args.output}.")
