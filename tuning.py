import random

import torch._dynamo

from entities import data, models

torch.set_float32_matmul_precision("high")

import os

os.environ["TRANSFORMERS_OFFLINE"] = "1"

ds = data.load_dataset(data.only_species_and_strains800())

for _ in range(3):
    configs = list(models.model_configs())
    random.shuffle(configs)

    for config in configs:
        nt = models.NERCTagger(ds, config=config)
        nt.cuda()

        torch._dynamo.reset()
        model = torch.compile(nt, mode="max-autotune", fullgraph=True)
        model.train_model()

        _, report = model.evaluate_model()
        print(report)
