"""Regression tests for the vectorised `align_relation_predictions` (PERF-01).

The alignment used to convert `rel_meta` to Python lists and walk the candidate
rows one at a time. These tests pin the two properties the vectorised version
has to keep: it must produce exactly what that loop produced, and it must reach
its answer without reading the candidate tensors back to the host.

Everything runs on CPU with synthetic tensors — no data, network, or GPU.
"""

from collections import defaultdict

import pytest
import torch
from d3text.models.model_types import IndexedRelation
from d3text.models.ete import ETEBrendaModel

POOLINGS = ("logsumexp", "logmeanexp", "max", "mean")

_ENTITY_TO_INDEX = {"A": 0, "B": 1, "C": 4, "D": 9}
_NONE_INDEX = 2


def _model(stub, pooling="logsumexp"):
    return stub(
        ETEBrendaModel,
        entity_logits_pooling=pooling,
        entity_to_index=_ENTITY_TO_INDEX,
        relations_none_index=_NONE_INDEX,
    )


def _reference(model, true_relations, rel_meta, rel_logits):
    """The pre-PERF-01 implementation, transcribed.

    Kept verbatim rather than simplified: it is the oracle the vectorised path
    is checked against, so any tidying here would weaken the check.
    """
    if rel_logits is None or rel_logits.numel() == 0:
        return None

    def _as_list(x):
        return x.detach().cpu().tolist()

    seq_list = _as_list(rel_meta["sequence"])
    subj_list = _as_list(rel_meta["arg_pred_i"])
    obj_list = _as_list(rel_meta["arg_pred_j"])

    device = rel_logits.device
    groups = defaultdict(list)
    for row_idx, (d, i, j) in enumerate(zip(seq_list, subj_list, obj_list)):
        groups[(int(d), int(i), int(j))].append(row_idx)

    if not groups:
        return None

    gold_by_key = defaultdict(list)
    for tr in true_relations:
        try:
            subj_ix = int(model.entity_to_index[tr.subject])
            obj_ix = int(model.entity_to_index[tr.object])
        except KeyError:
            continue
        gold_by_key[(int(tr.docix), subj_ix, obj_ix)].append(int(tr.label))

    pooled_logits, pooled_targets = [], []
    pooled_seq, pooled_subj, pooled_obj = [], [], []
    none_idx = model.relations_none_index

    for (d, i, j), row_idxs in groups.items():
        pooled_logits.append(model._pool_logits(rel_logits[row_idxs], dim=0))
        labels = gold_by_key.get((d, i, j))
        if labels:
            if any(lbl != none_idx for lbl in labels):
                target = next(lbl for lbl in labels if lbl != none_idx)
            else:
                target = labels[0]
        else:
            target = int(none_idx)
        pooled_targets.append(target)
        pooled_seq.append(d)
        pooled_subj.append(i)
        pooled_obj.append(j)

    meta = {
        "sequence": torch.tensor(pooled_seq, dtype=torch.long, device=device),
        "arg_pred_i": torch.tensor(
            pooled_subj, dtype=torch.long, device=device
        ),
        "arg_pred_j": torch.tensor(pooled_obj, dtype=torch.long, device=device),
    }
    return (
        meta,
        torch.stack(pooled_logits, dim=0).to(device),
        torch.tensor(pooled_targets, dtype=torch.long, device=device),
    )


def _duplicated_batch():
    """Candidate rows with duplicate triples, out of sorted order.

    Two documents; doc 1 is proposed before doc 0 and triple (1, 4, 9) appears
    three times, so grouping, first-appearance ordering and multi-row pooling
    are all exercised at once.
    """
    triples = [
        (1, 4, 9),
        (0, 0, 1),
        (1, 4, 9),
        (0, 1, 4),
        (1, 4, 9),
        (0, 0, 1),
    ]
    meta = {
        "sequence": torch.tensor([t[0] for t in triples]),
        "arg_pred_i": torch.tensor([t[1] for t in triples]),
        "arg_pred_j": torch.tensor([t[2] for t in triples]),
    }
    return meta, torch.randn(len(triples), 3)


def _gold():
    return [
        # matches the (0, 0, 1) group
        IndexedRelation(
            docix=0, subject="A", object="B", label=torch.tensor(0)
        ),
        # matches the (1, 4, 9) group
        IndexedRelation(
            docix=1, subject="C", object="D", label=torch.tensor(1)
        ),
        # no candidate pair proposed (0, 4, 9): must not disturb any group.
        # Its label differs from every matched group's, so a join that lets it
        # land on a neighbouring key changes that key's target.
        IndexedRelation(
            docix=0, subject="C", object="D", label=torch.tensor(0)
        ),
        # subject absent from entity_to_index: dropped
        IndexedRelation(
            docix=0, subject="Z", object="B", label=torch.tensor(1)
        ),
    ]


@pytest.mark.parametrize("pooling", POOLINGS)
def test_align_matches_the_python_loop_reference(stub, pooling):
    model = _model(stub, pooling)
    meta_in, rel_logits = _duplicated_batch()
    gold = _gold()

    got = model.align_relation_predictions(gold, meta_in, rel_logits)
    want = _reference(model, gold, meta_in, rel_logits)

    got_meta, got_logits, got_targets = got
    want_meta, want_logits, want_targets = want

    for key in ("sequence", "arg_pred_i", "arg_pred_j"):
        assert got_meta[key].tolist() == want_meta[key].tolist(), key
    assert got_targets.tolist() == want_targets.tolist()
    assert torch.allclose(got_logits, want_logits, atol=1e-6)


def test_align_pools_every_row_of_a_repeated_triple(stub):
    """Guards the oracle itself: a group of three must not collapse to one row's
    value, which is what a grouping bug would silently produce."""
    model = _model(stub, "logsumexp")
    meta_in, rel_logits = _duplicated_batch()
    _, pooled, _ = model.align_relation_predictions([], meta_in, rel_logits)
    repeated = rel_logits[[0, 2, 4]]  # the three (1, 4, 9) rows
    assert torch.allclose(
        pooled[0], torch.logsumexp(repeated, dim=0), atol=1e-6
    )
    assert not torch.allclose(pooled[0], repeated[0])


class _HostReadIsAFailure(torch.Tensor):
    """A tensor that treats a host round-trip as a test failure.

    Stands in for a CUDA tensor: `.cpu()`, `.tolist()`, `.item()` and `.numpy()`
    are precisely the calls that block the launch queue on a real device, and
    are invisible on CPU. Subclass instances survive `detach`/`to`/indexing, so
    the guard follows the tensors through the call.
    """

    def _fail(self, name):
        raise AssertionError(
            f"align_relation_predictions read a candidate tensor with .{name}()"
        )

    def cpu(self, *args, **kwargs):
        self._fail("cpu")

    def tolist(self):
        self._fail("tolist")

    def item(self):
        self._fail("item")

    def numpy(self, *args, **kwargs):
        self._fail("numpy")


@pytest.mark.parametrize("pooling", POOLINGS)
def test_align_never_reads_the_candidate_tensors_back_to_the_host(
    stub, pooling
):
    model = _model(stub, pooling)
    meta_in, rel_logits = _duplicated_batch()
    guarded_meta = {
        key: value.as_subclass(_HostReadIsAFailure)
        for key, value in meta_in.items()
    }
    guarded_logits = rel_logits.as_subclass(_HostReadIsAFailure)

    meta, pooled, targets = model.align_relation_predictions(
        _gold(), guarded_meta, guarded_logits
    )

    # The result must still be usable, not merely sync-free.
    assert pooled.shape == (3, 3)
    assert targets.shape == (3,)
    assert set(meta) == {"sequence", "arg_pred_i", "arg_pred_j"}


def test_align_keeps_the_gradient_flowing_to_every_pooled_row(stub):
    model = _model(stub, "logsumexp")
    meta_in, rel_logits = _duplicated_batch()
    rel_logits.requires_grad_(True)
    _, pooled, _ = model.align_relation_predictions([], meta_in, rel_logits)
    pooled.sum().backward()
    assert rel_logits.grad is not None
    assert (rel_logits.grad != 0).all()


def test_align_rejects_an_unknown_pooling(stub):
    model = _model(stub, "bogus")
    with pytest.raises(ValueError, match="Unknown pooling"):
        model.align_relation_predictions([], *_duplicated_batch())
