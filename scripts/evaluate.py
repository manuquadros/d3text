#!/usr/bin/env python

import argparse

import torch
from d3text import data, models
from d3text.models.config import encodings, load_model_config


def command_line_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="evaluate.py",
        description=("Evaluate a model with the provided configuration."),
    )
    parser.add_argument(
        "config", help="Configuration file for the model to be evaluated."
    )
    parser.add_argument("model_state_dict", help="Model state dict")
    parser.add_argument("--limit", type=int, default=None)

    return parser.parse_args()


def fix_keys_hook(
    module: torch.nn.Module,
    state_dict: dict,
    prefix: str,
    local_metadata: dict,
    strict: bool,
    missing_keys: list,
    unexpected_keys: list,
    error_msgs: list,
) -> None:
    new_dict = {
        key.replace("_orig_mod.", ""): state_dict[key] for key in state_dict
    }
    state_dict.clear()
    state_dict.update(new_dict)


if __name__ == "__main__":
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
        dataset=dataset.data["test"], batch_size=1
    )

    print("Initializing model...")
    mclass = getattr(models, config.model_class)
    model = mclass(
        classes=dataset.class_map,
        config=config,
        class_matrix=dataset.class_matrix,
        entity_index=dataset.entity_index,
    )
    model.register_load_state_dict_pre_hook(fix_keys_hook)
    state_dict = torch.load(args.model_state_dict)
    model.load_state_dict(state_dict)

    model.to(model.device)
    model.evaluate_model(eval_data)
