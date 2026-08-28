"""The divisor of the masked token loss.

Nothing consumes `masked_token_cross_entropy` yet — the tagger head that will
read the distant-supervision targets is a later piece of work — so these are
the only thing standing between the divisor and the trap it exists to avoid.
"""

import torch
from d3text.models.base import (
    masked_bce_with_logits,
    masked_token_cross_entropy,
)
from d3text.token_labels import IGNORE_INDEX


def _batch() -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(0)
    preds = torch.randn(8, 3, generator=generator)
    targets = torch.tensor([0, 1, IGNORE_INDEX, 2, IGNORE_INDEX, 1, 0, 2])
    return preds, targets


def test_the_divisor_is_the_unmasked_count() -> None:
    """Not the token count — the trap `focal_cross_entropy` documents.

    2.8% of the tokens carry no answer. Summing the kept terms and dividing by
    the sequence length scales every real token's loss by the share of the
    document that happened to be masked, so a document with more uncurated
    entities in it teaches less about the ones it does have.
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

    The property the divisor buys, stated directly: a document that matches
    more uncurated entities gets more `ignore` targets, and that must cost the
    curated ones nothing.
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
    targets down by however many pairs it happened to abstain, teaching less
    from documents with more abstentions rather than the same amount from
    fewer terms.
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
