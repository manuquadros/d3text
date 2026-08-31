"""The per-step weight update: backward, unscale, clip, step.

Its own module rather than part of `trainer` because `Model.run_epoch` — which
stays on the model — is what calls it, while `trainer` imports the model
classes; putting the two together would make that a cycle.
"""

import torch
from jaxtyping import Float
from torch import Tensor

GRAD_CLIP_NORM = 1.0


class BatchUpdate:
    """The optimizer half of a training batch, and its gradient telemetry.

    `run_epoch` computes the losses; this applies them. It owns the optimizer,
    the gradient scaler and the epoch's gradient-norm accumulators, none of
    which are properties of the model being trained.
    """

    # Plain attributes rather than buffers: they are per-epoch telemetry, and a
    # buffer would follow the parameters into every checkpoint.
    _grad_norm_sum: Tensor | None
    _grad_norm_clipped: Tensor | None

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str,
        amp_dtype: torch.dtype = torch.float16,
    ) -> None:
        """`amp_dtype` is the dtype the forward autocasts to.

        Loss scaling only exists to keep fp16 gradients out of the subnormal
        range, so it is enabled for float16 alone: bfloat16 has float32's
        exponent range and nothing to rescue, while the scaler still costs a
        scale multiply, an `unscale_` division over every gradient, and a
        `.item()` inside `step` that synchronises the host against the device
        on every optimizer step. A disabled scaler passes `scale`, `unscale_`,
        `step` and `update` straight through, so `__call__` is the same code
        either way. The default is the conservative one: a caller that does
        not say which dtype it autocasts to gets the scaling.
        """
        self.model = model
        self.optimizer = optimizer
        self.scaler = torch.amp.GradScaler(
            device, enabled=amp_dtype == torch.float16
        )
        self.reset_grad_norms()

    def zero_grad(self) -> None:
        self.optimizer.zero_grad(set_to_none=True)

    def __call__(self, *losses: Float[Tensor, ""]) -> None:
        loss: Float[Tensor, ""] = torch.stack(losses).sum()

        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        self._record_grad_norm(
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), GRAD_CLIP_NORM
            )
        )
        self.scaler.step(self.optimizer)
        self.scaler.update()

    def reset_grad_norms(self) -> None:
        """Drop the previous epoch's gradient-norm accumulators."""
        self._grad_norm_sum = None
        self._grad_norm_clipped = None
        self._grad_norm_steps = 0

    def _record_grad_norm(self, grad_norm: Tensor) -> None:
        """Accumulate one step's pre-clip gradient norm, without a sync.

        `clip_grad_norm_` returns the norm it measured *before* clipping, which
        is the only informative one — after clipping it is `GRAD_CLIP_NORM` by
        construction on every step that clipped at all. The sum is kept on the
        accelerator and read once per epoch: an `.item()` per optimizer step
        would serialise the training loop against the device.
        """
        norm = grad_norm.detach()
        clipped = (norm > GRAD_CLIP_NORM).to(norm.dtype)
        if self._grad_norm_sum is None or self._grad_norm_clipped is None:
            self._grad_norm_sum = norm.clone()
            self._grad_norm_clipped = clipped
        else:
            self._grad_norm_sum = self._grad_norm_sum + norm
            self._grad_norm_clipped = self._grad_norm_clipped + clipped
        self._grad_norm_steps += 1

    def grad_norm_metrics(self) -> dict[str, float]:
        """The epoch's mean pre-clip gradient norm and its clipping rate.

        Empty when no optimizer step ran — a validation-only pass, or a model
        whose `run_epoch` never applies the update — so that nothing logs a
        gradient statistic for an epoch that computed no gradients. A clipping
        rate pinned at 1.0 is the signal that `GRAD_CLIP_NORM` is doing the
        optimising rather than the learning rate.
        """
        if not self._grad_norm_steps or self._grad_norm_sum is None:
            return {}

        steps = float(self._grad_norm_steps)
        metrics = {"training/grad_norm": self._grad_norm_sum.item() / steps}
        if self._grad_norm_clipped is not None:
            metrics["training/grad_clip_rate"] = (
                self._grad_norm_clipped.item() / steps
            )

        return metrics
