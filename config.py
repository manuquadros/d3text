import configparser
import os
import tomllib

import tomli_w

from utils import ModelConfig


def load_model_config(path: str) -> ModelConfig:
    with open(path, "rb") as config_file:
        model_config = ModelConfig(**tomllib.load(config_file))

    return model_config


def save_model_config(config: dict, path: str) -> None:
    with open(path, "wb") as config_file:
        tomli_w.dump(config, config_file)


config = configparser.ConfigParser()
config.read(os.path.join(os.path.dirname(__file__), "config.ini"))

brenda_json = config["data"]["brenda_json"]
references_file = config["data"]["references_file"]
species_list = config["data"]["species_list"]
enzymes_list = config["data"]["enzymes_list"]
strains_list = config["data"]["strains_list"]
literature_folder = config["data"]["literature_folder"]
esummaries = config["data"]["esummaries"]

entrez_email = os.environ.get("ENTREZ_EMAIL")
