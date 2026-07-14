"""`Trainer`: one training run over one model.

The split with `Model` is by lifetime. A model is a set of heads that can price
a batch and walk an epoch; it outlives any particular run and is the thing a
checkpoint restores. An optimizer, a scheduler, a gradient scaler, an
early-stopping counter and a best-so-far snapshot are *the run's* — they mean
nothing to a model loaded for evaluation, and holding them on the model meant
`Model.state_dict()` and the epoch loop had to keep stepping around each other.

The `Trainer` is itself the `Optimization` it hands to `Model.run_epoch`: the
loop asks only for `zero_grad` and `update`, and how the gradients are scaled,
clipped and applied stays here.

## Checkpoint format

`save` writes a bare `state_dict` — the *best* epoch's, not the last one's.
Nothing else is serialized: no optimizer state (a run is not resumable), no
config (`evaluate` is given the same `config.toml` the run was), and no entity
index (it is a pure function of the corpus and the schema, and is rebuilt).

`train` compiles the model before fitting it, so the keys it writes carry
`torch.compile`'s `_orig_mod.` prefix; `factory.fix_keys_hook` strips it on the
way back in.

Which weights those are depends on what the run was given:

- validation data and `save_checkpoint` (the default): the epoch with the
  lowest validation loss, restored into the model at the end of `fit` so the
  live model and the file agree;
- no validation data, or `save_checkpoint=False` (tuning, which throws the
  weights away and keeps the loss): the weights as the last epoch left them.
"""

from copy import deepcopy

import torch
from jaxtyping import Float
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from d3text.models.base import Model, Step
from d3text.models.config import optimizers, schedulers

__all__ = ["Trainer", "print_epoch_stats"]


def print_epoch_stats(
    losses: dict[str, float], denominator: int, step: Step
) -> None:
    for obj, loss in losses.items():
        tqdm.write(f"Average ({obj}) {step} loss: {loss / denominator:.4f}")

    total_loss = sum(losses.values())
    tqdm.write(f"Average {step} loss: {total_loss / denominator:.4f}")


class Trainer:
    """Fit `model`, keeping the best epoch's weights.

    :param model: the model to train. Its `ModelConfig` says how: optimizer,
        learning rate, scheduler, epochs, patience.
    :param save_checkpoint: keep a copy of the best epoch's weights. Off for
        hyperparameter search, which wants the validation loss and nothing
        else, and would otherwise deepcopy a state dict per improving epoch.
    """

    def __init__(self, model: Model, save_checkpoint: bool = True) -> None:
        self.model = model
        self.config = model.config
        self.save_checkpoint = save_checkpoint

        self.optimizer, self.scheduler = self._setup()
        self.scaler = torch.amp.GradScaler(model.device)

        self.stop_counter = 0
        self.best_val_loss = float("inf")
        self.best_epoch = -1
        self.best_model_state: dict[str, Tensor] | None = None

    def _setup(
        self,
    ) -> tuple[
        torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler | None
    ]:
        """The optimizer and learning-rate scheduler the config names."""
        optimizer = optimizers[self.config.optimizer](
            self.model.parameters(), lr=self.config.lr
        )

        scheduler = None
        match self.config.lr_scheduler:
            case "exponential":
                scheduler = schedulers["exponential"](optimizer, gamma=0.95)
            case "reduce_on_plateau":
                scheduler = schedulers["reduce_on_plateau"](
                    optimizer, min_lr=0.0001, patience=2, factor=0.5
                )

        return optimizer, scheduler

    def zero_grad(self) -> None:
        self.optimizer.zero_grad(set_to_none=True)

    def update(self, *losses: Float[Tensor, ""]) -> None:
        """Step on the sum of the batch's losses.

        The models weight their own objectives — `compute_losses` returns what
        the optimizer is meant to step on — so summing is all that is left to
        do here.
        """
        loss: Float[Tensor, ""] = torch.stack(losses).sum()

        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()

    def fit(
        self,
        train_data: DataLoader,
        val_data: DataLoader | None = None,
    ) -> float | None:
        """Train for `config.num_epochs`, or until validation loss stops
        improving.

        :returns: the best validation loss, or `None` if there was no
            validation data to compute one over.
        """
        for epoch in trange(
            self.config.num_epochs,
            dynamic_ncols=True,
            position=0,
            desc="Epochs",
            leave=True,
        ):
            self.model.train()
            losses, denominator = self.model.run_epoch(
                data=train_data,
                step=Step.TRAINING,
                epoch=epoch,
                optimization=self,
            )

            print_epoch_stats(
                losses=losses, denominator=denominator, step=Step.TRAINING
            )

            if val_data is not None:
                val_loss = self._validate(val_data=val_data, epoch=epoch)
                tqdm.write(f"Average validation loss: {val_loss:.5f}")
                self._step_scheduler(val_loss)

                # The ramp epochs are a warm-up: a model still holding one of
                # its objectives back at a fraction of its weight is not the
                # model early stopping is there to judge.
                if epoch <= self.model.ramp_epochs:
                    self.stop_counter = 0

                if self._early_stop(val_loss, epoch):
                    tqdm.write(
                        f"Model converged; best epoch was {self.best_epoch}."
                    )
                    break

            tqdm.write("-" * 50)

        self._restore_best()

        return None if val_data is None else self.best_val_loss

    def _validate(self, val_data: DataLoader, epoch: int) -> float:
        self.model.eval()
        losses, denominator = self.model.run_epoch(
            data=val_data, step=Step.VALIDATION, epoch=epoch
        )

        print_epoch_stats(
            losses=losses, denominator=denominator, step=Step.VALIDATION
        )

        return sum(losses.values()) / denominator

    def _step_scheduler(self, val_loss: float) -> None:
        if self.scheduler is None:
            return

        if isinstance(
            self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
        ):
            # Its `step` takes the monitored metric, not an epoch — and it is
            # not an `LRScheduler` subclass, so only an isinstance check tells
            # the two signatures apart.
            self.scheduler.step(val_loss)
        else:
            self.scheduler.step()

    def _early_stop(self, val_loss: float, epoch: int) -> bool:
        """Whether `config.patience` epochs have passed without an improvement
        to the validation loss.

        An improving epoch is the one whose weights `save` will write, so this
        is where the snapshot is taken.
        """
        if val_loss <= self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_epoch = epoch
            self.stop_counter = 0
            if self.save_checkpoint:
                self.best_model_state = deepcopy(self.model.state_dict())
        else:
            self.stop_counter += 1

        return self.stop_counter > self.config.patience

    def _restore_best(self) -> None:
        """Leave the model holding the weights `save` would write, whether the
        run early-stopped or ran every epoch out. Without this, a run that never
        converged would be evaluated on its last epoch and checkpointed on its
        best — two different models."""
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state, strict=True)

    def save(self, path: str) -> None:
        """Write the best epoch's `state_dict` to `path`.

        With no checkpoint kept — no validation data, or `save_checkpoint`
        off — there is no "best" epoch to write, so this writes the weights the
        run ended on.
        """
        if self.best_model_state is not None:
            torch.save(self.best_model_state, path)
        else:
            tqdm.write(
                "No best epoch was kept (no validation data, or checkpointing "
                "off): saving the weights the run ended on."
            )
            torch.save(self.model.state_dict(), path)
