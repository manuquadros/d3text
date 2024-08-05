import configparser
import os
import tomllib

from entities.utils import ModelConfig


def load_model_config(path: str) -> ModelConfig:
    with open(path, "rb") as config_file:
        config = ModelConfig(**tomllib.load(config_file))

    return config


config = configparser.ConfigParser()
config.read("config.ini")

brenda_json = config["data"]["brenda_json"]
references_file = config["data"]["references_file"]
species_list = config["data"]["species_list"]
literature_folder = config["data"]["literature_folder"]
esummaries = config["data"]["esummaries"]

entrez_email = os.environ.get("ENTREZ_EMAIL")
