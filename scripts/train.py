#!/usr/bin/env python

import argparse

import torch
import torch._dynamo

from config import load_model_config
from entities import data, models

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


def main():
    args = command_line_args()
    config = load_model_config(args.config)

    ds = data.preprocess_dataset(
        data.only_species_and_strains800(upsample=False),
        validation_split=True,
        test_split=False,
    )
    train_data = data.get_loader(
        dataset_config=ds,
        split="train",
        batch_size=config.batch_size,
    )
    val_data = data.get_loader(
        dataset_config=ds,
        split="validation",
        batch_size=config.batch_size,
    )

    model = models.NERCTagger(
        num_labels=len(train_data.classes),
        config=config,
    )

    model.to(model.device)

    model.compile(mode="reduce-overhead")
    model.train_model(
        train_data=train_data, val_data=val_data, output_loss=False
    )

    print(model.evaluate_model(val_data))

    torch.save(model.state_dict(), args.output)

    print(f"Model saved to {args.output}.")
