"""Classifier and relation heads used by the models in this package."""

import math
from typing import cast

import torch
import torch.nn as nn
from jaxtyping import Bool, Float
from torch import Tensor
from d3text.constraints import (
    FREQUENCY_CLAMP_EPS,
    Positive,
    PositiveReal,
    UnitInterval,
)


class ClassificationHead(nn.Module):
    """The entity-class head of the end-to-end models."""

    def __init__(
        self,
        input_size: Positive,
        n_classes: Positive,
        class_freqs: Float[Tensor, " classes"] | None = None,
        oos_index: int = -1,
    ) -> None:
        """Build the class output layer.

        :param input_size: number of input features.
        :param n_classes: number of output entity classes.
        :param class_freqs: class label frequencies, to seed the bias.
        :param oos_index: column of the unsupervised OOS class, which carries
            no frequency and so is seeded from a prior instead.
        """
        super().__init__()
        self.class_classifier = nn.Linear(input_size, n_classes)
        if class_freqs is not None:
            initialize_classifier_bias(
                linear=cast(nn.Linear, self.class_classifier),
                freqs=class_freqs,
                sentinel_index=oos_index,
                sentinel_prior=0.9,
            )

    def forward(self, input: Tensor) -> Tensor:
        return self.class_classifier(input)


class BiaffineRelationClassifier(nn.Module):
    """The relation head of the end-to-end model, scoring argument pairs."""

    def __init__(
        self,
        hidden_size: Positive,
        num_relations: Positive,
        separate_predicate_layer: bool,
        biaff_hidden_size: Positive,
    ):
        """Build the biaffine relation scorer.

        :param hidden_size: number of features in each argument's input
            representation.
        :param num_relations: number of output relation labels.
        :param separate_predicate_layer: project the object argument through
            its own hidden layer instead of sharing the subject's.
        :param biaff_hidden_size: width of the projected representations the
            bilinear and linear terms score.
        """
        super().__init__()
        self.separate_predicate_layer = separate_predicate_layer
        self.hidden_linear = nn.Sequential(
            nn.Linear(
                in_features=hidden_size,
                out_features=biaff_hidden_size,
                bias=True,
            ),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        if separate_predicate_layer:
            self.hidden_linear_y = nn.Sequential(
                nn.Linear(
                    in_features=hidden_size,
                    out_features=biaff_hidden_size,
                    bias=True,
                ),
                nn.GELU(),
                nn.Dropout(0.1),
            )
        else:
            self.hidden_linear_y = self.hidden_linear

        self.bilinear = nn.Parameter(
            torch.randn(num_relations, biaff_hidden_size, biaff_hidden_size)
        )
        nn.init.xavier_uniform_(self.bilinear)
        self.linear = nn.Linear(biaff_hidden_size * 2, num_relations)
        self.bias = nn.Parameter(torch.zeros(num_relations))

    def forward(
        self,
        x: Float[Tensor, "pairs features"],
        y: Float[Tensor, "pairs features"],
    ) -> Float[Tensor, "pairs logits"]:
        x = self.hidden_linear(x)
        y = self.hidden_linear_y(y)
        bilinear_term = torch.einsum("bi,rid,bj->br", x, self.bilinear, y)
        linear_term = self.linear(torch.cat([x, y], dim=-1))
        return bilinear_term + linear_term + self.bias


def initialize_classifier_bias(
    linear: torch.nn.Linear,
    freqs: torch.Tensor,
    eps: PositiveReal = FREQUENCY_CLAMP_EPS,
    sentinel_index: int | None = -1,
    sentinel_prior: UnitInterval = 0.1,
) -> None:
    """Initialize classifier bias using log odds from label frequencies.

    :param linear: the layer whose bias to seed.
    :param freqs: the supervised labels' frequencies, in column order.
    :param eps: floor keeping the log odds finite.
    :param sentinel_index: the head's one unsupervised column — OOS on the
        class head — which has no frequency; defaults to the last column,
        where the models put it. Pass `None` for a head with no sentinel
        column.
    :param sentinel_prior: the probability to seed that column from.
    """
    device = linear.weight.device
    dtype = linear.weight.dtype

    p = freqs.clamp(eps, 1 - eps).to(device=device, dtype=dtype)
    log_odds = torch.log(p) - torch.log1p(-p)  # logit(p)

    with torch.no_grad():
        if sentinel_index is None:
            if log_odds.numel() != linear.out_features:
                raise ValueError(
                    f"freqs len {log_odds.numel()} != out_features {linear.out_features}"
                )
            linear.bias.copy_(log_odds)
            return

        expected = linear.out_features - 1
        if log_odds.numel() != expected:
            raise ValueError(
                f"freqs len {log_odds.numel()} != expected {expected} "
                f"(out_features-1) for layer with a sentinel column"
            )

        sentinel = sentinel_index % linear.out_features
        kept = torch.tensor(
            [
                column
                for column in range(linear.out_features)
                if column != sentinel
            ],
            device=device,
        )
        bias = torch.empty(linear.out_features, device=device, dtype=dtype)
        bias[kept] = log_odds
        prior = max(min(sentinel_prior, 1 - eps), eps)
        bias[sentinel] = math.log(prior) - math.log1p(-prior)
        linear.bias.copy_(bias)


class PermutationBatchNorm1d(nn.BatchNorm1d):
    """`nn.BatchNorm1d` over a padded `(document, token, features)` block,
    with statistics (and the running-stat update) restricted to the
    positions a mask marks real.

    Plain `nn.BatchNorm1d` run over every position, padding included, lets a
    padded position — a constant `GELU(bias)` activation, since padding is
    zero before the first `Linear` — drag the batch mean and shrink the
    variance in proportion to how much padding the batch carries; a real
    token's normalized value would then depend on how long the other
    documents in its batch are. Selecting the real positions before
    delegating to `nn.BatchNorm1d.forward` keeps every other option
    (`momentum`, `affine`, `track_running_stats`) working exactly as it
    does upstream.
    """

    def forward(  # type: ignore[override]
        # Deliberately not Liskov-substitutable: every caller of this class
        # is `base._run_hidden_layer`, which special-cases it by `isinstance`
        # specifically to pass the extra `mask` argument the mask-aware
        # statistics need.
        self,
        input: Float[Tensor, "document token features"],
        mask: Bool[Tensor, "document token"],
    ) -> Float[Tensor, "document token features"]:
        """Normalize `input` over the positions `mask` marks real.

        Padding positions are left at zero so the output keeps `input`'s
        shape; the token loss (`IGNORE_INDEX`) and `_mask_padding` exclude
        them downstream.

        :param input: the padded token-feature block.
        :param mask: which positions carry a real token.
        :return: the normalized block, the same shape as `input`.
        :raises ValueError: if `mask` marks no position of `input` real.
        """
        real = input[mask]
        if real.shape[0] == 0:
            raise ValueError(
                "PermutationBatchNorm1d got a batch with no real positions"
            )
        output = torch.zeros_like(input)
        output[mask] = super().forward(real)
        return output
