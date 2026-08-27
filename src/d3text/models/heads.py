"""Classifier and relation heads used by the models in this package.

Split out of what used to be models.py so each head has one home,
independent of the model classes that use it.
"""

import math
from typing import cast

import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor


class ClassificationHead(nn.Module):
    """Define a classification head for end-to-end models."""

    def __init__(
        self,
        input_size: int,
        n_entities: int,
        n_classes: int,
        entity_freqs: Float[Tensor, " entities"] | None = None,
        class_freqs: Float[Tensor, " classes"] | None = None,
        unk_index: int = -1,
        oos_index: int = -1,
    ) -> None:
        """Initialize the classification head.

        :param input_size: number of input features
        :param n_entities: number of output entities
        :param n_classes: number of output entity classes
        :param unk_index: column of the unsupervised UNK entity, which carries
            no frequency and so is seeded from a prior instead
        :param oos_index: idem for the OOS class
        """
        super().__init__()
        self.entity_classifier = nn.Sequential(
            nn.Linear(
                in_features=input_size,
                out_features=input_size,
                bias=True,
            ),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(input_size, n_entities),
        )
        # self.entity_classifier = nn.Linear(input_size, n_entities)
        self.class_classifier = nn.Linear(input_size, n_classes)
        if entity_freqs is not None:
            initialize_classifier_bias(
                linear=cast(nn.Linear, self.entity_classifier[-1]),
                freqs=entity_freqs,
                sentinel_index=unk_index,
            )
        if class_freqs is not None:
            initialize_classifier_bias(
                linear=cast(nn.Linear, self.class_classifier),
                freqs=class_freqs,
                sentinel_index=oos_index,
                sentinel_prior=0.9,
            )

    def forward(self, input: Tensor) -> tuple[Tensor, Tensor]:
        entity_logits = self.entity_classifier(input)
        class_logits = self.class_classifier(input)

        return entity_logits, class_logits


class BiaffineRelationClassifier(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_relations: int,
        separate_predicate_layer: bool = False,
        biaff_hidden_size: int = 32,
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
    eps: float = 1e-5,
    sentinel_index: int | None = -1,
    sentinel_prior: float = 0.1,
) -> None:
    """Initialize classifier bias using log odds from label frequencies.

    `freqs` covers the supervised labels only, in column order. `sentinel_index`
    names the head's one unsupervised column — UNK for an entity head, OOS for a
    class head — which has no frequency and is seeded from `sentinel_prior`
    instead. It defaults to the last column, where both models put it; pass
    `None` for a head with no sentinel column.
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
