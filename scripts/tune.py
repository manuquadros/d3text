#!/usr/bin/env python

import argparse
import os
from pprint import pp

import torch
import torch._dynamo
from d3text import data, models, utils
from d3text.models.config import encodings, load_tuning_config

torch.set_float32_matmul_precision("high")
os.environ["PYTORCH_HIP_ALLOC_CONF"] = "expandable_segments:True"


def command_line_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="tune.py",
        description=(
            "Tune a model with the provided configuration and saves the results"
            "output file."
        ),
    )
    parser.add_argument("config", help="Tuning config file.")
    parser.add_argument("output", help="Location to save the results.")
    parser.add_argument("--limit", type=int, default=None)

    return parser.parse_args()


def is_triton_compatible() -> bool:
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability()
    return (major, minor) >= (7, 0)


if __name__ == "__main__":
    args = command_line_args()
    print("Loading hyperparameter configurations...")
    configs = load_tuning_config(args.config)

    for config in configs:
        encodings_file = encodings[config.base_model]

        pp(config.model_dump())
        print("Loading dataset...")
        if args.limit is not None:
            dataset = data.brenda_dataset(
                encodings=encodings_file, limit=args.limit
            )
        else:
            dataset = data.brenda_dataset(encodings=encodings_file)
        train_data = dataset.data["train"]
        train_data_loader = data.get_batch_loader(
            dataset=train_data, batch_size=config.batch_size
        )
        val_data_loader = data.get_batch_loader(
            dataset=dataset.data["val"], batch_size=config.batch_size
        )

        print("Loading model...")
        mclass = getattr(models, config.model_class)
        model = mclass(classes=dataset.class_map, config=config)

        model.to(model.device)
        if config.base_layers_to_unfreeze:
            model.unfreeze_encoder_layers(n=config.base_layers_to_unfreeze)

        # Use memory efficient attention if available
        if hasattr(model.base_model, "config"):
            model.base_model.config.use_memory_efficient_attention = True

        if is_triton_compatible():
            try:
                model = torch.compile(model, dynamic=True)
            except Exception as e:
                print(f"Failed to compile with Triton: {e}")
                print("Skipping torch.compile(): GPU too old for Triton")

        try:
            print("Running config...")
            model.train_model(
                train_data=train_data_loader,
                val_data=val_data_loader,
                save_checkpoint=False,
            )
        except Exception as e:
            print(f"{e}")
            raise
        else:
            utils.log_config(args.output, config, val_loss=model.best_val_loss)
