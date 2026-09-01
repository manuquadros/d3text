"""The divisor of the masked token loss.

Nothing consumes `masked_token_cross_entropy` yet — the tagger head that will
read the distant-supervision targets is a later piece of work — so these are
the only thing standing between the divisor and the trap it exists to avoid.
"""

import pytest
import torch
from d3text.models.base import (
    masked_bce_with_logits,
    masked_token_cross_entropy,
)
from d3text.models.config import ModelConfig
from d3text.token_labels import IGNORE_INDEX
from pydantic import ValidationError


def _batch() -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(0)
    preds = torch.randn(8, 3, generator=generator)
    targets = torch.tensor([0, 1, IGNORE_INDEX, 2, IGNORE_INDEX, 1, 0, 2])
    return preds, targets


def test_the_divisor_is_the_unmasked_count() -> None:
    """Not the token count — the trap `focal_cross_entropy` documents.

    Dividing by the sequence length scales every real token's loss by the share
    of the document that happened to be masked.
    """
    preds, targets = _batch()
    kept = targets != IGNORE_INDEX
    total = torch.nn.functional.cross_entropy(
        preds[kept], targets[kept], reduction="none"
    ).sum()

    loss = masked_token_cross_entropy(preds, targets)

    torch.testing.assert_close(loss, total / int(kept.sum()))
    assert not torch.isclose(loss, total / targets.numel())


def test_it_matches_the_ignore_index_spelling() -> None:
    """`cross_entropy(..., ignore_index=...)` divides the same way.

    Which is the point: the two spellings must not be able to disagree, and
    the explicit one is here so the divisor is visible at the call site.
    """
    preds, targets = _batch()

    torch.testing.assert_close(
        masked_token_cross_entropy(preds, targets),
        torch.nn.functional.cross_entropy(
            preds, targets, ignore_index=IGNORE_INDEX
        ),
    )


def test_masked_tokens_do_not_dilute_the_kept_ones() -> None:
    """Adding ignored tokens must not move the loss at all.

    A document that matches more uncurated entities gets more ignored targets,
    and that must cost the curated ones nothing.
    """
    preds, targets = _batch()
    padding = torch.zeros(5, 3)

    before = masked_token_cross_entropy(preds, targets)
    after = masked_token_cross_entropy(
        torch.cat([preds, padding]),
        torch.cat([targets, torch.full((5,), IGNORE_INDEX)]),
    )

    torch.testing.assert_close(before, after)


def test_an_all_masked_batch_is_a_differentiable_zero() -> None:
    """Reachable from a short document whose every match is uncurated."""
    preds = torch.randn(4, 3, requires_grad=True)
    targets = torch.full((4,), IGNORE_INDEX)

    loss = masked_token_cross_entropy(preds, targets)
    loss.backward()

    assert loss.item() == 0.0
    assert preds.grad is not None
    assert torch.equal(preds.grad, torch.zeros_like(preds))


def test_a_batch_with_nothing_masked_is_plain_cross_entropy() -> None:
    preds, targets = _batch()
    targets = targets.clamp(min=0)

    torch.testing.assert_close(
        masked_token_cross_entropy(preds, targets),
        torch.nn.functional.cross_entropy(preds, targets),
    )


def test_weighting_defaults_to_unweighted_and_changes_nothing() -> None:
    """The new keyword must be a strict opt-in.

    A model with no `token_labels_store` never sets `token_loss_weighting`,
    so the default has to reproduce the previous call exactly.
    """
    preds, targets = _batch()
    torch.testing.assert_close(
        masked_token_cross_entropy(preds, targets),
        masked_token_cross_entropy(preds, targets, weighting="unweighted"),
    )


def _one_gradient_step(
    weighting: str, focal_gamma: float = 2.0
) -> torch.Tensor:
    """Probability the tagger assigns each true class after one SGD step.

    18 majority-class tokens and one each of two minority classes, from a head
    confidently predicting the majority everywhere. Returning the post-step
    softmax makes `weighting`'s effect on those two tokens comparable across
    calls without depending on where training converges.
    """
    logits = torch.zeros(20, 3)
    logits[:, 0] = 5.0
    logits.requires_grad_(True)
    targets = torch.tensor([0] * 18 + [1, 2])

    loss = masked_token_cross_entropy(
        logits, targets, weighting=weighting, focal_gamma=focal_gamma
    )
    (grad,) = torch.autograd.grad(loss, logits)
    with torch.no_grad():
        updated = logits - grad
        probs = updated.softmax(dim=-1)
        return torch.stack([probs[18, 1], probs[19, 2]])


@pytest.mark.parametrize("weighting", ("balanced", "focal"))
def test_weighting_shifts_predictions_toward_the_minority_classes(
    weighting: str,
) -> None:
    """The mechanism `token_loss_weighting` exists for: not just that it
    runs, but that it measurably moves the minority-class tokens the plain
    average leaves behind.
    """
    unweighted = _one_gradient_step("unweighted")
    weighted = _one_gradient_step(weighting)

    assert bool((weighted > unweighted).all())


def test_token_loss_weighting_defaults_to_unweighted() -> None:
    assert ModelConfig().token_loss_weighting == "unweighted"


def test_token_loss_weighting_rejects_an_unknown_scheme() -> None:
    with pytest.raises(ValidationError):
        ModelConfig(token_loss_weighting="bogus")


def _class_batch() -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(0)
    logits = torch.randn(4, 3, generator=generator)
    targets = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [1.0, 1.0, 0.0]]
    )
    return logits, targets


def test_no_abstain_mask_is_plain_bce_with_logits() -> None:
    logits, targets = _class_batch()

    torch.testing.assert_close(
        masked_bce_with_logits(logits, targets),
        torch.nn.functional.binary_cross_entropy_with_logits(logits, targets),
    )


def test_the_divisor_is_the_kept_pair_count() -> None:
    """Not the whole matrix — the same trap the token loss avoids.

    Dividing by every `(document, class)` pair would scale a document's real
    targets down by however many pairs it happened to abstain.
    """
    logits, targets = _class_batch()
    abstain = torch.zeros_like(targets, dtype=torch.bool)
    abstain[2, 0] = True
    abstain[3, 2] = True
    kept = ~abstain

    loss = masked_bce_with_logits(logits, targets, abstain=abstain)

    elementwise = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, targets, reduction="none"
    )
    torch.testing.assert_close(loss, elementwise[kept].sum() / kept.sum())
    assert not torch.isclose(loss, elementwise.sum() / elementwise.numel())


def test_abstained_pairs_do_not_move_the_kept_ones() -> None:
    """Abstaining more (document, class) pairs must cost the rest nothing."""
    logits, targets = _class_batch()

    none = masked_bce_with_logits(
        logits, targets, abstain=torch.zeros_like(targets, dtype=torch.bool)
    )
    one = masked_bce_with_logits(
        logits[1:2], targets[1:2], abstain=torch.zeros(1, 3, dtype=torch.bool)
    )
    padded_targets = torch.cat([targets[1:2], torch.zeros(3, 3)])
    padded_logits = torch.cat([logits[1:2], torch.zeros(3, 3)])
    padded_abstain = torch.cat(
        [
            torch.zeros(1, 3, dtype=torch.bool),
            torch.ones(3, 3, dtype=torch.bool),
        ]
    )

    padded = masked_bce_with_logits(
        padded_logits, padded_targets, abstain=padded_abstain
    )

    assert none.shape == ()  # sanity: still a scalar with a real mask
    torch.testing.assert_close(one, padded)


def test_an_all_abstained_batch_is_a_differentiable_zero() -> None:
    logits = torch.randn(4, 3, requires_grad=True)
    targets = torch.zeros(4, 3)
    abstain = torch.ones(4, 3, dtype=torch.bool)

    loss = masked_bce_with_logits(logits, targets, abstain=abstain)
    loss.backward()

    assert loss.item() == 0.0
    assert logits.grad is not None
    assert torch.equal(logits.grad, torch.zeros_like(logits))


def test_downweight_defaults_to_a_hard_abstain() -> None:
    """DEC-04 option 1's already-run configs never set `downweight`, so the
    default has to reproduce their exact hard-abstain numbers."""
    logits, targets = _class_batch()
    abstain = torch.zeros_like(targets, dtype=torch.bool)
    abstain[2, 0] = True
    abstain[3, 2] = True

    torch.testing.assert_close(
        masked_bce_with_logits(logits, targets, abstain=abstain),
        masked_bce_with_logits(
            logits, targets, abstain=abstain, downweight=0.0
        ),
    )


def test_downweight_keeps_a_fraction_of_the_abstained_pairs() -> None:
    """DEC-04 option 2: an abstained pair contributes `downweight` times its
    own term to both the numerator and the divisor, rather than nothing."""
    logits, targets = _class_batch()
    abstain = torch.zeros_like(targets, dtype=torch.bool)
    abstain[2, 0] = True
    abstain[3, 2] = True
    weight = torch.where(abstain, 0.4, 1.0)

    loss = masked_bce_with_logits(
        logits, targets, abstain=abstain, downweight=0.4
    )

    elementwise = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, targets, reduction="none"
    )
    torch.testing.assert_close(
        loss, (elementwise * weight).sum() / weight.sum()
    )


def test_downweight_of_one_cancels_the_abstention() -> None:
    """The other endpoint: full weight on every pair is plain BCE, matching
    what `class_negative_abstention = False` already gives."""
    logits, targets = _class_batch()
    abstain = torch.zeros_like(targets, dtype=torch.bool)
    abstain[2, 0] = True
    abstain[3, 2] = True

    torch.testing.assert_close(
        masked_bce_with_logits(
            logits, targets, abstain=abstain, downweight=1.0
        ),
        torch.nn.functional.binary_cross_entropy_with_logits(logits, targets),
    )


def test_class_negative_downweight_defaults_to_zero() -> None:
    assert ModelConfig().class_negative_downweight == 0.0


def test_class_negative_downweight_rejects_out_of_range_values() -> None:
    with pytest.raises(ValidationError):
        ModelConfig(class_negative_downweight=1.5)
    with pytest.raises(ValidationError):
        ModelConfig(class_negative_downweight=-0.1)
