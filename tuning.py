import dataclasses
import os
import random

import numpy as np
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
    print(config)
    fold_val_losses: list[float] = []
    strain_f1_values: list[float] = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(ds.data["train"])):
        torch._dynamo.reset()
        print("-" * 7)
        print(f"Fold {fold + 1}:")
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
        test_loader: torch.utils.data.DataLoader = torch.utils.data.DataLoader(
            dataset=ds.data["test"],
            batch_size=config.batch_size,
        )
        train_data = dataclasses.replace(ds, data=train_loader)
        val_data = dataclasses.replace(ds, data=val_loader)
        test_data = dataclasses.replace(ds, data=test_loader)

        nt = models.NERCTagger(
            num_labels=len(train_data.classes), config=config
        )
        nt.to(nt.device)

        # mode="reduce-overhead" gives the best results on my hardware (20 SMs).
        # With more than 80 streaming processors, one could try "max-autotune"
        nt.compile(mode="reduce-overhead")
        val_loss = nt.train_model(train_data=train_data, val_data=val_data)

        report = nt.evaluate_model(test_data, output_dict=True)
        f1 = report["Strain"]["f1-score"]
        strain_f1_values.append(f1)
        fold_val_losses.append(val_loss)
        print(f"Validation loss on this fold: {val_loss:.5f}")
        print(f"F1 for Strain on this fold: {f1:.2f}")

    val_loss = np.mean(fold_val_losses)
    val_loss_std = np.std(fold_val_losses)
    strain_f1 = np.mean(strain_f1_values)
    strain_f1_std = np.std(strain_f1_values)

    utils.log_config(
        "models.csv",
        config,
        val_loss=val_loss,
        val_loss_std=val_loss_std,
        strain_f1=strain_f1,
        strain_f1_std=strain_f1_std,
    )
