"""Classifier and relation heads used by the models in this package.

Also holds `PermutationBatchNorm1d`, the hidden-block normalisation layer
`base.py` builds into its hidden layers and runs through
`base._run_hidden_layer` — not a head, but placed here rather than moved
next to its only caller.
"""

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
        """Score `input` into per-class logits.

        :param input: features in the last dimension; other dimensions
            are kept as-is.
        :return: `input` with its last dimension replaced by one logit
            per class; which class a column is is the caller's own
            convention, not something this head tracks.
        """
        return self.class_classifier(input)


class BiaffineRelationClassifier(nn.Module):
    """The relation head of the end-to-end model, scoring argument pairs."""

    def __init__(
        self,
        hidden_size: Positive,
        num_relations: Positive,
        separate_predicate_layer: bool,
        biaff_hidden_size: Positive,
        dropout: UnitInterval,
    ) -> None:
        """Build the biaffine relation scorer.

        :param hidden_size: number of features in each argument's input
            representation.
        :param num_relations: number of output relation labels.
        :param separate_predicate_layer: project the pair's second
            argument (`y`) through its own hidden layer instead of
            sharing the first argument's (`x`'s).
        :param biaff_hidden_size: width of the projected representations the
            bilinear and linear terms score.
        :param dropout: dropout probability for both hidden projections.
        """
        super().__init__()
        self.separate_predicate_layer = separate_predicate_layer
        self.hidden_linear = nn.Sequential(
            nn.Linear(hidden_size, biaff_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        if separate_predicate_layer:
            self.hidden_linear_y = nn.Sequential(
                nn.Linear(hidden_size, biaff_hidden_size),
                nn.GELU(),
                nn.Dropout(dropout),
            )
        else:
            self.hidden_linear_y = self.hidden_linear

        self.bilinear = nn.Parameter(
            torch.randn(num_relations, biaff_hidden_size, biaff_hidden_size)
        )
        nn.init.xavier_uniform_(self.bilinear)
        self.linear = nn.Linear(biaff_hidden_size * 2, num_relations)

    def forward(
        self,
        x: Float[Tensor, "pairs features"],
        y: Float[Tensor, "pairs features"],
    ) -> Float[Tensor, "pairs logits"]:
        """Score each argument pair over every relation label.

        `x` and `y` are the pair's first and second argument, in
        ascending `ArgumentGroups` id order; they carry no subject or
        object role, since a gold pair's arguments are sorted before
        scoring and it is the relation label itself that says which
        argument type fills which role.

        :param x: the first argument's representation, one row per pair.
        :param y: the second argument's representation, one row per
            pair.
        :return: one row of relation-label logits per pair.
        """
        x = self.hidden_linear(x)
        y = self.hidden_linear_y(y)
        bilinear_term = torch.einsum("bi,rij,bj->br", x, self.bilinear, y)
        linear_term = self.linear(torch.cat([x, y], dim=-1))
        return bilinear_term + linear_term


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
    :raises ValueError: if `freqs` has the wrong number of elements for
        `linear`'s output width, given whether `sentinel_index` is set, or if
        `sentinel_index` is not a column of `linear`.
    """
    device = linear.weight.device
    dtype = linear.weight.dtype

    log_odds = torch.logit(freqs.to(device=device, dtype=dtype), eps=eps)

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

        if not -linear.out_features <= sentinel_index < linear.out_features:
            raise ValueError(
                f"sentinel_index {sentinel_index} outside "
                f"[{-linear.out_features}, {linear.out_features})"
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
    """`nn.BatchNorm1d` over a padded `(document, token, features)` block.

    Statistics and the running-stat update cover only the positions a mask
    marks real, so a token's normalized value cannot depend on how long the
    other documents in its batch are.
    """

    def forward(  # type: ignore[override]
        # Deliberately not Liskov-substitutable: its one caller,
        # `base._run_hidden_layer`, special-cases it to pass `mask`.
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
