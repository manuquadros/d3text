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


