import math

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf


def plot_hyperparameter(df: pd.DataFrame, file_name: str, metric: str) -> None:
    fig, ax = plt.subplots(2, 4, sharey="row", figsize=(20, 10))

    for coln, column in enumerate(
        df.columns[
            ~df.columns.isin(
                [
                    "val_loss",
                    "num_epochs",
                    "patience",
                    "val_loss_std",
                    "strain_f1",
                    "strain_f1_std",
                ]
            )
        ]
    ):
        sns.boxplot(
            x=column,
            y=metric,
            data=df,
            ax=ax[math.floor(coln / 4), coln % 4],
        )

    fig.savefig(file_name)


models = pd.read_csv("models.csv")
models = models[models["val_loss"] <= 0.15]

### Regression

results = smf.ols(
    "val_loss ~ optimizer * lr * lr_scheduler * batch_size + dropout * hidden_layers * hidden_size + normalization",
    data=models,
).fit()

print(results.summary())

results = smf.ols(
    "strain_f1 ~ optimizer * lr * lr_scheduler * batch_size + dropout * hidden_layers * hidden_size + normalization",
    data=models,
).fit()

print(results.summary())

plot_hyperparameter(models, "models_loss.png", "val_loss")
plot_hyperparameter(models, "models_F1.png", "strain_f1")
