"""The epoch schedule: optimizer, LR scheduler, early stopping, telemetry.

`Model` computes losses; `Trainer` decides what is done with them. The split is
what lets a model be constructed, loaded and evaluated without carrying an
optimizer, a best-epoch snapshot and a stop counter around with it.
"""

import logging
import math
import time
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, assert_never, cast

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import trange

from d3text import tracking
from d3text.constraints import NonNegative
from d3text.models.base import (
    Model,
    Step,
    epoch_rate_metrics,
    print_epoch_stats,
)
from d3text.models.config import PLATEAU_PATIENCE, optimizers, schedulers
from d3text.training.update import BatchUpdate

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Selection:
    """One epoch's validation standing, as `Trainer` reports and compares it.

    `score` is the geometric mean of the selection metrics: 0 when any one is
    0, so a collapsed task vetoes the epoch. `rank` is what epochs are
    compared by: first the number of non-zero metrics, then the geometric
    mean of those, so epochs the veto flattens to 0 are still ordered.
    """

    score: float
    rank: float

    @classmethod
    def from_values(cls, values: Sequence[float]) -> "Selection":
        """The standing of an epoch whose selection metrics are `values`.

        :param values: the selection metrics, each finite and non-negative.
        :return: the epoch's score and rank.
        """
        nonzero = [value for value in values if value > 0]
        partial = math.prod(nonzero) ** (1 / len(nonzero)) if nonzero else 0.0
        # One float, because `ReduceLROnPlateau.step` takes one. The partial
        # mean is squashed into [0, 1) so the non-zero count dominates it
        # whatever scale the metrics are on.
        return cls(
            score=partial if len(nonzero) == len(values) else 0.0,
            rank=len(nonzero) + partial / (1 + partial),
        )


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
        self.best_selection_score = float("-inf")
        self.best_rank = float("-inf")
        self.best_epoch = -1

    def _setup(
        self,
    ) -> tuple[
        torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler | None
    ]:
        """Build the optimizer and the learning-rate scheduler.

        Trainable base-model parameters (`config.unfrozen_top_layers`) get
        their own param group at `config.base_model_lr`, and the class head's
        parameters get their own at `config.class_head_lr`, each falling back
        to `lr` when unset — everything else trains at `lr`, as before.

        :return: the optimizer, and the scheduler if the config asks for one.
        """
        # `getattr`: a `Model` built to drive `Trainer` alone (as in its
        # tests) owns neither submodule, and every parameter is then "other".
        base_model = getattr(self.model, "base_model", None)
        base_model_param_ids = (
            {id(p) for p in base_model.parameters()}
            if base_model is not None
            else set()
        )
        classifier = getattr(self.model, "classifier", None)
        classifier_param_ids = (
            {id(p) for p in classifier.parameters()}
            if classifier is not None
            else set()
        )
        base_model_params: list[torch.nn.Parameter] = []
        classifier_params: list[torch.nn.Parameter] = []
        other_params: list[torch.nn.Parameter] = []
        for param in self.model.parameters():
            if not param.requires_grad:
                continue
            if id(param) in base_model_param_ids:
                base_model_params.append(param)
            elif id(param) in classifier_param_ids:
                classifier_params.append(param)
            else:
                other_params.append(param)

        param_groups = [{"params": other_params, "lr": self.config.lr}]
        if base_model_params:
            param_groups.append(
                {
                    "params": base_model_params,
                    "lr": self.config.base_model_lr or self.config.lr,
                }
            )
        if classifier_params:
            param_groups.append(
                {
                    "params": classifier_params,
                    "lr": self.config.class_head_lr or self.config.lr,
                }
            )

        optimizer = optimizers[self.config.optimizer](
            param_groups, lr=self.config.lr
        )

        scheduler = None
        match self.config.lr_scheduler:
            case "exponential":
                scheduler = schedulers["exponential"](optimizer, gamma=0.95)
            case "reduce_on_plateau":
                # mode="max": stepped with the selection score (higher is
                # better), the same quantity `_early_stop` compares — never
                # validation loss.
                scheduler = schedulers["reduce_on_plateau"](
                    optimizer,
                    mode="max",
                    min_lr=0.0001,
                    patience=PLATEAU_PATIENCE,
                    factor=0.5,
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
            The best selection score is on `best_selection_score`.
        """
        self.stop_counter = 0
        self.best_model_state = None
        self.best_selection_score = float("-inf")
        self.best_rank = float("-inf")
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
                epoch=epoch,
                update=self.update,
            )
            train_seconds = time.perf_counter() - started
            logger.info(
                "Epoch %d training time: %.2f s", epoch + 1, train_seconds
            )
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
                selection = self._validate(val_data=val_data, epoch=epoch)

                if self.scheduler is not None:
                    if self.config.lr_scheduler == "reduce_on_plateau":
                        # Takes the metric, not an epoch. The rank, not the
                        # score: one collapsed metric would pin the score at
                        # 0 and cut the rate while the others still improve.
                        cast(
                            torch.optim.lr_scheduler.ReduceLROnPlateau,
                            self.scheduler,
                        ).step(selection.rank)
                    else:
                        self.scheduler.step()

                early_stop = self._early_stop(
                    selection, epoch=epoch, save_checkpoint=save_checkpoint
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

            # Undefined without a validation split, so gated on one existing.
            tracking.log_metrics(
                {
                    "best_selection_score": self.best_selection_score,
                    "best_epoch": float(self.best_epoch),
                    "epochs_after_best": float(
                        epochs_run - 1 - self.best_epoch
                    ),
                }
            )

        # Unconditional: both mean something without a validation split
        # (`stopped_early` is then `False`), and a run list is scanned by them.
        tracking.log_metrics(
            {
                "epochs_run": float(epochs_run),
                "stopped_early": float(stopped_early),
            }
        )

        return self.best_model_state

    def _selection_score(
        self, val_data: DataLoader, epoch: NonNegative
    ) -> Selection:
        """This epoch's standing over the validation metrics.

        Scored through the model's own `evaluate_model`, under
        `prefix="validation"`, rather than a second computation of the same
        numbers — so what `_early_stop` compares is exactly what the
        `validation/*` metrics on the tracking run already say.

        :param val_data: the split to score.
        :param epoch: the epoch it belongs to; `evaluate_model`'s tracking
            step.
        :return: the standing over `config.selection_metrics`, or the model
            class's `default_selection_metrics` when that is empty.
        :raises ValueError: no metric is configured and the model class
            names no default, a configured name is absent from what
            `evaluate_model` reports, or a reported value is non-finite or
            negative.
        """
        names = (
            self.config.selection_metrics
            or self.model.default_selection_metrics
        )
        if not names:
            raise ValueError(
                f"{type(self.model).__name__} names no "
                "default_selection_metrics and config.selection_metrics is "
                "empty; name one explicitly rather than falling back to "
                "validation loss"
            )

        scored = self.model.evaluate_model(
            val_data, prefix="validation", log_reports=False, step=epoch
        )
        missing = [name for name in names if f"validation/{name}" not in scored]
        if missing:
            available = sorted(
                key.removeprefix("validation/") for key in scored
            )
            raise ValueError(
                f"selection metric(s) {missing} not reported by "
                f"{type(self.model).__name__}.evaluate_model; it reports "
                f"{available}"
            )

        values = [scored[f"validation/{name}"] for name in names]
        non_finite = {
            name: value
            for name, value in zip(names, values, strict=True)
            if not math.isfinite(value)
        }
        if non_finite:
            raise ValueError(
                f"selection metric(s) {non_finite} are non-finite; NaN "
                "would never win the best-epoch comparison and +inf "
                "would always win it silently"
            )
        negative = {
            name: value
            for name, value in zip(names, values, strict=True)
            if value < 0
        }
        if negative:
            raise ValueError(
                f"selection metric(s) {negative} are negative; a geometric "
                "mean over them is undefined"
            )
        selection = Selection.from_values(values)
        logger.info(
            "Epoch %d selection score: %.4f (geometric mean of %s)",
            epoch + 1,
            selection.score,
            ", ".join(
                f"{name}={value:.4f}"
                for name, value in zip(names, values, strict=True)
            ),
        )
        return selection

    def _early_stop(
        self, selection: Selection, epoch: NonNegative, save_checkpoint: bool
    ) -> bool:
        """Whether `patience` epochs have passed without improvement.

        `epoch` is carried here rather than tracked in `fit` so the epoch and
        the score it belongs to are written by the same comparison; two
        comparisons in two places is how `best_epoch` came to disagree with
        `best_selection_score`.

        :param selection: this epoch's standing; only a strictly higher rank
            than the best so far is an improvement.
        :param epoch: the epoch it belongs to.
        :param save_checkpoint: whether to snapshot an improving epoch.
        :return: whether to stop.
        """
        if selection.rank > self.best_rank:
            self.best_rank = selection.rank
            self.best_selection_score = selection.score
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

    def _validate(self, val_data: DataLoader, epoch: NonNegative) -> Selection:
        """Score the validation split once, timed.

        No validation loss is computed: nothing reads it, and a loss pass
        would run every head over the split a second time.

        :param val_data: the split to score.
        :param epoch: the epoch it belongs to.
        :return: the epoch's standing.
        """
        started = time.perf_counter()
        selection = self._selection_score(val_data=val_data, epoch=epoch)
        seconds = time.perf_counter() - started
        logger.info("Epoch %d validation time: %.2f s", epoch + 1, seconds)
        self.model.log_pass_stats(Step.VALIDATION)

        # No `batches_per_second`: `TokenBudgetBatchSampler` has no length,
        # and only a loss pass counted the batches.
        tracking.log_metrics(
            {
                f"{Step.VALIDATION}/epoch_seconds": seconds,
                f"{Step.VALIDATION}/selection_score": selection.score,
            },
            step=epoch,
        )
        return selection
