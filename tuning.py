import os
import random

import torch._dynamo

from entities import data, models

torch.set_float32_matmul_precision("high")

os.environ["TRANSFORMERS_OFFLINE"] = "1"

ds = data.load_dataset(data.only_species_and_strains800(upsample=False))

ds = data.preprocess_dataset(
    data.only_species_and_strains800(upsample=False), validation_split=False
)

configs = list(models.model_configs())
random.shuffle(configs)

for config in configs:
    for n in range(4):
        nt = models.NERCTagger(ds, config=config)
        nt.cuda()

        torch._dynamo.reset()
        model = torch.compile(nt, mode="max-autotune", fullgraph=True)
        model.train_model()

        _, report = model.evaluate_model()
        print(report)
