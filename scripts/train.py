#!/usr/bin/env python

import argparse
import os

import torch
import torch._dynamo
from d3text import data, models
from d3text.models.config import encodings, load_model_config

os.environ["PYTORCH_HIP_ALLOC_CONF"] = "expandable_segments:True"


def print_model_size(model: torch.nn.Module) -> None:
    """Compute and print model size.
    Piotr Bialecki @ https://discuss.pytorch.org/t/finding-model-size/130275/2
    """
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    print("model size: {:.3f}MB".format(size_all_mb))


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


def is_triton_compatible() -> bool:
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability()
    return (major, minor) >= (7, 0)


if __name__ == "__main__":
    args = command_line_args()
    config = load_model_config(args.config)
    batch_size = config.batch_size
    encodings_file = encodings[config.base_model]

    print("Loading dataset...")
    dataset = data.brenda_dataset(limit=100, encodings=encodings_file)
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
    if config.base_layers_to_unfreeze:
        model.unfreeze_encoder_layers(n=config.base_layers_to_unfreeze)

    # Use memory efficient attention if available
    if hasattr(model.base_model, "config"):
        model.base_model.config.use_memory_efficient_attention = True

    print_model_size(model)

    if is_triton_compatible():
        try:
            model = torch.compile(model, dynamic=True)
        except Exception as e:
            print(f"Failed to compile with Triton: {e}")
            print("Skipping torch.compile(): GPU too old for Triton")

    print("Training:")
    model.train_model(
        train_data=train_data_loader,
        val_data=val_data_loader,
        save_checkpoint=True,
    )

    torch.save(model.state_dict(), args.output)
    model.save_config(os.path.splitext(args.output)[0] + "_config.toml")

    print(f"Model saved to {args.output}.")
