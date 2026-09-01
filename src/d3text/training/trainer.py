"""The epoch schedule: optimizer, LR scheduler, early stopping, telemetry.

`Model` computes losses; `Trainer` decides what is done with them. The split is
what lets a model be constructed, loaded and evaluated without carrying an
optimizer, a best-epoch snapshot and a stop counter around with it.
"""

import logging
import time
from copy import deepcopy
from typing import Any, assert_never, cast

import torch
from torch import Tensor
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

    Single-use: the optimizer, scheduler and gradient scaler are built once and
    never rebuilt, so a second `fit()` would resume their state — the LR
    schedule included — rather than start a fresh run.
    """

    best_model_state: dict[str, Any] | None

    def __init__(self, model: Model) -> None:
        self.model = model
        self.config = self.model.config
        self.optimizer, self.scheduler = self._setup()
        self.update = BatchUpdate(
            self.model,
            self.optimizer,
            self.model.device,
            amp_dtype=self.model.amp_dtype,
        )

        self.stop_counter = 0
        self.best_model_state = None
        self.best_val_loss = float("inf")
        self.best_epoch = -1

    def _setup(
        self,
    ) -> tuple[
        torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler | None
    ]:
        """Build the optimizer and the learning-rate scheduler.

        :return: the optimizer, and the scheduler if the config asks for one.
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
            case "":
                pass
            case unreachable:
                assert_never(unreachable)

        return optimizer, scheduler

    def fit(
        self,
        train_data: DataLoader,
        val_data: DataLoader | None = None,
        save_checkpoint: bool = True,
    ) -> dict[str, Any] | None:
        """Train `model`, stopping early if validation stops improving.

        :param train_data: the split to train on.
        :param val_data: the split to score each epoch, if any.
        :param save_checkpoint: whether to keep the best epoch's parameters.
        :return: the parameters a checkpoint should be written from — the best
            epoch's, copied while that epoch was current — or None when the run
            kept no snapshot. Handing them back frees the caller from knowing
            that `fit` also loads the snapshot into the model on its way out.
            The best validation loss is on `best_val_loss`.
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

            # `epochs_after_best` answers what `best_val_loss` alone cannot: a
            # run that stopped with several epochs since its best had
            # converged, while one that ended at its best was still improving
            # when `num_epochs` ran out. Both are undefined without a
            # validation split, so they stay gated on one existing.
            tracking.log_metrics(
                {
                    "best_val_loss": self.best_val_loss,
                    "best_epoch": float(self.best_epoch),
                    "epochs_after_best": float(
                        epochs_run - 1 - self.best_epoch
                    ),
                }
            )

        # `epochs_run` and `stopped_early` need no validation split to mean
        # something — `stopped_early` is always `False` without one, the same
        # value it would have if validation existed but the run finished its
        # full schedule without early-stopping — so both are logged
        # unconditionally. This is the summary a run list is scanned by; a
        # run with no validation split must still report how many epochs it
        # ran.
        tracking.log_metrics(
            {
                "epochs_run": float(epochs_run),
                "stopped_early": float(stopped_early),
            }
        )

        return self.best_model_state

    def _early_stop(
        self, val_loss: float, epoch: int, save_checkpoint: bool
    ) -> bool:
        """Whether `patience` epochs have passed without improvement.

        `epoch` is carried here rather than tracked in `fit` so the epoch and
        the loss it belongs to are written by the same comparison; two
        comparisons in two places is how `best_epoch` came to disagree with
        `best_val_loss`.

        :param val_loss: this epoch's validation loss.
        :param epoch: the epoch it belongs to.
        :param save_checkpoint: whether to snapshot an improving epoch.
        :return: whether to stop.
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
        snapshot was a second resident copy of the whole model, frozen base
        included, pinned for the rest of the run. `copy=True` is load-bearing
        on CPU runs: `.to("cpu")` on a tensor already there returns *self*,
        which would leave the snapshot aliasing the live parameters.

        :return: the snapshot.
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
