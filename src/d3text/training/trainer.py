"""The epoch schedule: optimizer, LR scheduler, early stopping, telemetry.

`Model` computes losses; `Trainer` decides what is done with them. The split
is what lets a model be constructed, loaded and evaluated without carrying an
optimizer, a best-epoch snapshot and a stop counter around with it.
"""

import logging
import time
from copy import deepcopy
from typing import Any, cast

import torch
from torch import Tensor
from torch._dynamo.eval_frame import OptimizedModule
from torch.utils.data import DataLoader
from tqdm import trange

from d3text import tracking
from d3text.models.base import (
    Model,
    Step,
    epoch_rate_metrics,
    print_epoch_stats,
)
from d3text.models.config import optimizers, schedulers
from d3text.training.update import BatchUpdate

logger = logging.getLogger(__name__)


class Trainer:
    """Trains `model` for `model.config.num_epochs`, or until it converges.

    Single-use: the optimizer, scheduler and gradient scaler are built once
    in `__init__` and never rebuilt, so a second `fit()` call would resume
    their state — including the LR schedule — rather than start a fresh run.
    Construct a new `Trainer` per training run.
    """

    model: Model
    best_model_state: dict[str, Any] | None

    def __init__(self, model: Model | OptimizedModule) -> None:
        # `torch.compile` hands back an `OptimizedModule` that forwards every
        # attribute to the model it wrapped, so the trainer drives it exactly
        # as it drives an uncompiled one — but it is not a `Model`, and
        # beartype checks this annotation at call time. Naming both is what
        # keeps `train`'s compile branch runnable while still rejecting a
        # module that carries no config.
        self.model = cast(Model, model)
        self.config = self.model.config
        self.optimizer, self.scheduler = self._setup()
        self.update = BatchUpdate(self.model, self.optimizer, self.model.device)

        self.stop_counter = 0
        self.best_model_state = None
        self.best_val_loss = float("inf")
        self.best_epoch = -1

    def _setup(
        self,
    ) -> tuple[
        torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler | None
    ]:
        """Setup optimizer and learning rate scheduler.

        Returns:
            Tuple of (optimizer, scheduler)
        """
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

    def fit(
        self,
        train_data: DataLoader,
        val_data: DataLoader | None = None,
        save_checkpoint: bool = True,
    ) -> dict[str, Any] | None:
        """Generic training loop for all models.

        :returns: the parameters a checkpoint should be written from — the
            best epoch's, copied while that epoch was current — or ``None``
            when the run kept no snapshot to hand back: ``save_checkpoint``
            off, or no validation data to choose a best epoch by. Handing them
            over is what frees the caller from knowing that `fit` also loads
            the snapshot into the model on its way out; a caller that saved
            the model instead was relying on that mutation without naming it,
            and nothing at the call site would have noticed it stop happening.
            The best validation loss is on `best_val_loss`, where `tune` reads
            it.
        """
        self.stop_counter = 0
        self.best_model_state = None
        self.best_val_loss = float("inf")
        self.best_epoch = -1
        epochs_run = 0
        stopped_early = False

        for epoch in trange(
            self.config.num_epochs,
            dynamic_ncols=True,
            position=0,
            desc="Epochs",
            leave=True,
        ):
            self.model.train()
            self.update.reset_grad_norms()
            tracking.log_metrics(
                {
                    "learning_rate": self.optimizer.param_groups[0]["lr"],
                    **{
                        f"loss_weight/{objective}": weight
                        for objective, weight in self.model.epoch_loss_weights(
                            epoch
                        ).items()
                    },
                },
                step=epoch,
            )
            started = time.perf_counter()
            losses, denominator = self.model.run_epoch(
                data=train_data,
                step=Step.TRAINING,
                epoch=epoch,
                update=self.update,
            )
            train_seconds = time.perf_counter() - started
            epochs_run = epoch + 1

            tracking.log_metrics(
                {
                    **print_epoch_stats(
                        losses=losses,
                        denominator=denominator,
                        step=Step.TRAINING,
                    ),
                    **self.update.grad_norm_metrics(),
                    **epoch_rate_metrics(
                        batches=denominator,
                        seconds=train_seconds,
                        step=Step.TRAINING,
                    ),
                },
                step=epoch,
            )

            if val_data is not None:
                val_loss = self._validate(val_data=val_data, epoch=epoch)

                if self.scheduler is not None:
                    if self.config.lr_scheduler == "reduce_on_plateau":
                        # ReduceLROnPlateau.step takes the monitored metric, not
                        # an epoch; it is not an LRScheduler subclass.
                        cast(
                            torch.optim.lr_scheduler.ReduceLROnPlateau,
                            self.scheduler,
                        ).step(val_loss)
                    else:
                        self.scheduler.step()

                logger.info("Average validation loss: %.5f", val_loss)

                early_stop = self._early_stop(
                    val_loss, epoch=epoch, save_checkpoint=save_checkpoint
                )
                tracking.log_metrics(
                    {
                        "early_stopping/epochs_without_improvement": float(
                            self.stop_counter
                        )
                    },
                    step=epoch,
                )
                if early_stop:
                    stopped_early = True
                    break

            logger.info("-" * 50)

        if val_data is not None:
            # Both exits from the loop leave the model holding the last epoch
            # trained, which is the best one only when the run ended on it.
            if (
                save_checkpoint
                and self.best_model_state is not None
                and self.best_epoch != epochs_run - 1
            ):
                logger.info(
                    "%s Loading the best epoch's parameters.",
                    "Model converged."
                    if stopped_early
                    else "Ran out of epochs.",
                )
                self.model.load_state_dict(self.best_model_state, strict=True)

            # The summary the run list is scanned by. `epochs_after_best`
            # answers what `best_val_loss` alone cannot: a run that stopped
            # with several epochs since its best had converged, while one that
            # ended at its best was still improving when `num_epochs` ran out.
            tracking.log_metrics(
                {
                    "best_val_loss": self.best_val_loss,
                    "best_epoch": float(self.best_epoch),
                    "epochs_run": float(epochs_run),
                    "epochs_after_best": float(
                        epochs_run - 1 - self.best_epoch
                    ),
                    "stopped_early": float(stopped_early),
                }
            )

        return self.best_model_state

    def _early_stop(
        self, val_loss: float, epoch: int, save_checkpoint: bool
    ) -> bool:
        """Stop training after `self.config.patience` epochs have passed
        without improvement to `metric` according to the `goal`. Most likely
        we will want to minimize validation loss.

        If `save_checkpoint` is True, store the best model state in
        `self.best_model_state`.

        `epoch` is carried here rather than tracked in `fit` so that the epoch
        and the loss it belongs to are written by the same comparison; two
        comparisons in two places is how `best_epoch` came to disagree with
        `best_val_loss` in the first place.
        """
        if val_loss <= self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_epoch = epoch
            self.stop_counter = 0
            if save_checkpoint:
                self.best_model_state = self._cpu_state_dict()
        else:
            self.stop_counter += 1

        if self.stop_counter > self.config.patience:
            return True
        else:
            return False

    def _cpu_state_dict(self) -> dict[str, Any]:
        """A detached CPU copy of the model's current parameters.

        `deepcopy(state_dict())` preserved each tensor's device, so on CUDA the
        best-epoch snapshot was a second resident copy of the whole model — the
        frozen base model included, 0.4 GiB of it — pinned for the rest of the
        run and briefly doubled at every improving epoch, since the new copy is
        built before the old one is dropped. Nothing ever reads it on-device:
        it is `torch.save`d, or loaded back once at convergence, and
        `load_state_dict` copies each tensor to its parameter's own device
        either way.

        `copy=True` is load-bearing, and only on CPU runs: `.to("cpu")` on a
        tensor already there returns *self*, which would leave the snapshot
        aliasing the live parameters and tracking them as training continued.
        """
        return {
            key: (
                value.detach().to("cpu", copy=True)
                if isinstance(value, Tensor)
                else deepcopy(value)
            )
            for key, value in self.model.state_dict().items()
        }

    def _validate(
        self,
        val_data: DataLoader,
        epoch: int,
    ) -> float:
        self.model.eval()
        started = time.perf_counter()
        losses, denominator = self.model.run_epoch(
            data=val_data,
            step=Step.VALIDATION,
            epoch=epoch,
            update=self.update,
        )
        seconds = time.perf_counter() - started

        tracking.log_metrics(
            {
                **print_epoch_stats(
                    losses=losses,
                    denominator=denominator,
                    step=Step.VALIDATION,
                ),
                **epoch_rate_metrics(
                    batches=denominator,
                    seconds=seconds,
                    step=Step.VALIDATION,
                ),
            },
            step=epoch,
        )

        return sum(losses.values()) / denominator
