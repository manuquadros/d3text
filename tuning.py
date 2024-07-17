import dataclasses
import os
import random

import torch
import torch._dynamo
from sklearn.model_selection import KFold

from entities import data, models, utils

torch.set_float32_matmul_precision("high")

os.environ["TRANSFORMERS_OFFLINE"] = "1"

ds = data.preprocess_dataset(
    data.only_species_and_strains800(upsample=False), validation_split=False
)
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True)

configs = list(models.model_configs())
random.shuffle(configs)

for config in configs:
    fold_val_losses: list[float] = []
    strain_f1: list[float] = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(ds.data["train"])):
        train_loader: torch.utils.data.DataLoader = torch.utils.data.DataLoader(
            dataset=ds.data["train"],
            batch_size=config.batch_size,
            sampler=torch.utils.data.SubsetRandomSampler(train_idx),
        )
        val_loader: torch.utils.data.DataLoader = torch.utils.data.DataLoader(
            dataset=ds.data["train"],
            batch_size=config.batch_size,
            sampler=torch.utils.data.SubsetRandomSampler(val_idx),
        )
        train_data = dataclasses.replace(ds, data=train_loader)
        val_data = dataclasses.replace(ds, data=val_loader)

        nt = models.NERCTagger(
            num_labels=len(train_data.classes), config=config
        )
        nt.cuda()

        torch._dynamo.reset()
        model = torch.compile(nt, mode="max-autotune", fullgraph=True)
        _, val_loss = model.train_model(
            train_data=train_data, val_data=val_data
        )

        report = model.evaluate_model(output_dict=True)
        strain_f1.append(report["Strain"]["f1-score"])
        fold_val_losses.append(val_loss)

    utils.log_config(
        "models.csv", config, val_losses=fold_val_losses, strain_f1=strain_f1
    )
