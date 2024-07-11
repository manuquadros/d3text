import re
from dataclasses import dataclass

import datasets
import s800classed

from entities import utils


def species800(upsample: bool = True) -> datasets.Dataset:
    dataset = s800classed.load()
    if upsample:
        dataset["train"] = utils.upsample(dataset["train"], 'Strain')
        
    return dataset


def only_species_and_strains800(upsample: bool = True) -> datasets.Dataset:
    dataset = species800(upsample=upsample)
    dataset = dataset.map(
        lambda sample: keep_only(["Species", "Strain"], sample, oos=True)
    )

    return dataset


def keep_only(keep: list[str], sample: dict, oos: bool) -> dict:
    """
    Return a new `sample` with only the labels specified in `keep`.
    """
    keep_regex = re.compile(rf"[BI]-({'|'.join(keep)})" r"|(?<![^\/])O+(?![^\/])")
    # keep_regex will match any label contained in `keep` plus any sequence
    # of the letter O, as long as it is not part of another label.

    sample["nerc_tags"] = [
        "/".join(
            map(
                lambda label: label if keep_regex.match(label) else replace(label, oos),
                tag.split("/"),
            )
        )
        for tag in sample["nerc_tags"]
    ]

    return sample


def replace(label: str, oos: bool) -> str:
    if oos:
        return label[:2] + "OOS"
    else:
        return "O"
