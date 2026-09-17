"""Classifier and relation heads used by the models in this package."""

import math
from typing import cast

import torch
import torch.nn as nn
from jaxtyping import Float
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
    def __init__(
        self,
        hidden_size: Positive,
        num_relations: Positive,
        separate_predicate_layer: bool = False,
        biaff_hidden_size: Positive = 32,
    ):
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

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        # x, y: [B, D]
        x = self.hidden_linear(x)
        y = self.hidden_linear_y(y)
        bilinear_term = torch.einsum(
            "bi,rid,bj->br", x, self.bilinear, y
        )  # [B, R]
        linear_term = self.linear(torch.cat([x, y], dim=-1))  # [B, R]
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
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input = torch.permute(input, (0, 2, 1))
        out = torch.permute(super().forward(input), (0, 2, 1))
        return out
