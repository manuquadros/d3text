"""What every tracked metric's y-axis actually is.

MLflow charts a metric under its key and offers no field for a unit, a
direction, or the denominator an average was taken over, so the keys are
written to say what they are and this module renders the glossary
`tracking.run` posts as the run's description. A leaf: no imports from
`d3text`, none from mlflow.
"""

from __future__ import annotations

import re
from typing import Final, NamedTuple


class Entry(NamedTuple):
    """One metric family: how it is keyed, and what its y-axis measures."""

    pattern: str
    """Regex matching the metric keys this entry documents."""

    display: str
    """How the family is written in the glossary table."""

    axis: str
    """What one point on the chart is."""

    unit: str
    """The y-axis unit, or `—` where the quantity is dimensionless."""


_PER_EPOCH: Final = (
    Entry(
        r"(training|validation)/loss_total",
        "`{training,validation}/loss_total`",
        "Sum of that pass's per-objective means below. The training one is "
        "the quantity back-propagated; the validation one is what early "
        "stopping and `best_val_loss` compare",
        "loss per batch",
    ),
    Entry(
        r"(training|validation)/loss_(entity|class|relation|token)",
        "`{training,validation}/loss_<objective>`",
        "One objective's loss, summed over the pass's batches and divided by "
        "the batch count. `relation` is scaled by `loss_weight/relation` "
        "before it gets here, so its curve moves when the ramp moves and not "
        "only when the model does",
        "loss per batch",
    ),
    Entry(
        r"loss_weight/\w+",
        "`loss_weight/<objective>`",
        "The multiplier that objective was trained under this epoch. Only "
        "`relation` ever moves: it ramps 0.1 → 1.0 over `ramp_epochs`",
        "—",
    ),
    Entry(
        r"learning_rate",
        "`learning_rate`",
        "The optimizer's learning rate as the epoch began",
        "—",
    ),
    Entry(
        r"training/grad_norm",
        "`training/grad_norm`",
        "Global L2 norm of the gradient *before* clipping, averaged over the "
        "epoch's optimizer steps",
        "—",
    ),
    Entry(
        r"training/grad_clip_rate",
        "`training/grad_clip_rate`",
        "Share of the epoch's optimizer steps whose pre-clip norm exceeded "
        "`GRAD_CLIP_NORM`. Pinned at 1.0 means the clip, not the learning "
        "rate, is setting the step size",
        "fraction, 0–1",
    ),
    Entry(
        r"(training|validation)/epoch_seconds",
        "`{training,validation}/epoch_seconds`",
        "Wall-clock time that pass took, this epoch",
        "seconds",
    ),
    Entry(
        r"(training|validation)/batches_per_second",
        "`{training,validation}/batches_per_second`",
        "That pass's batches divided by its `epoch_seconds`. Batches, not "
        "documents: `TokenBudgetBatchSampler` makes the documents per batch a "
        "function of document length, so this is throughput of work, not of "
        "corpus",
        "batches per second",
    ),
    Entry(
        r"early_stopping/epochs_without_improvement",
        "`early_stopping/epochs_without_improvement`",
        "Consecutive epochs since the best validation loss. The run stops "
        "once it passes `patience`",
        "epochs",
    ),
)

_SUMMARY: Final = (
    Entry(
        r"best_val_loss",
        "`best_val_loss`",
        "Lowest `validation/loss_total` any epoch reached (logged once, at "
        "the end)",
        "loss per batch",
    ),
    Entry(
        r"best_epoch",
        "`best_epoch`",
        "Which epoch that was, zero-based",
        "epoch index",
    ),
    Entry(
        r"epochs_after_best",
        "`epochs_after_best`",
        "Epochs trained after the best one. Zero means the run was still "
        "improving when `num_epochs` ran out",
        "epochs",
    ),
    Entry(
        r"epochs_run",
        "`epochs_run`",
        "Epochs completed, early stop included",
        "epochs",
    ),
    Entry(
        r"stopped_early",
        "`stopped_early`",
        "1 if patience ended the run, 0 if it ran its full schedule",
        "boolean",
    ),
)

_CONTEXT: Final = (
    Entry(
        r"dataset/(train|val|test)_documents",
        "`dataset/<split>_documents`",
        "Documents the split was built to hold, before anything was read",
        "documents",
    ),
    Entry(
        r"dataset/test_documents_(scored|missing)",
        "`dataset/test_documents_{scored,missing}`",
        "Documents the evaluation pass actually scored, and those the "
        "encodings file did not back",
        "documents",
    ),
    Entry(
        r"dataset/entities",
        "`dataset/entities`",
        "Entity-head columns, `UNK` included — the training split's "
        "vocabulary, so `--limit` moves it",
        "columns",
    ),
    Entry(
        r"dataset/classes",
        "`dataset/classes`",
        "Class-head columns, `OOS` included",
        "columns",
    ),
    Entry(
        r"model/size_mb",
        "`model/size_mb`",
        "Resident size of the parameters and buffers, frozen base model "
        "included",
        "MiB",
    ),
    Entry(
        r"model/(parameters|trainable_parameters)",
        "`model/{parameters,trainable_parameters}`",
        "Parameter count, and the part of it that receives gradients",
        "parameters",
    ),
    Entry(
        r"model/trainable_fraction",
        "`model/trainable_fraction`",
        "Trainable share of the parameters. Near 1.0 means the encoder was "
        "not frozen",
        "fraction, 0–1",
    ),
)

_TEST: Final = (
    Entry(
        r"test/(entity|class)_micro_f1",
        "`test/{entity,class}_micro_f1`",
        "Micro-averaged F1 over the head's columns at its decision "
        "threshold, `UNK`/`OOS` excluded",
        "F1, 0–1",
    ),
    Entry(
        r"test/(entity|class)_micro_ap",
        "`test/{entity,class}_micro_ap`",
        "Micro-averaged average precision — threshold-free, so it separates "
        "a badly calibrated head from an uninformative one",
        "AP, 0–1",
    ),
    Entry(
        r"test/entity_lrap",
        "`test/entity_lrap`",
        "Label ranking average precision: how high the true entities rank "
        "among all columns, per document",
        "LRAP, 0–1",
    ),
    Entry(
        r"test/relation_(macro|micro)_f1_typed",
        "`test/relation_{macro,micro}_f1_typed`",
        "Relation F1 over `HasEnzyme` and `HasSpecies` only. `none` is "
        "excluded: it is the majority class and the one nobody asked about",
        "F1, 0–1",
    ),
    Entry(
        r"test/relation_accuracy",
        "`test/relation_accuracy`",
        "Share of candidate pairs labelled correctly, `none` included — so "
        "it is high for a head that predicts `none` throughout",
        "fraction, 0–1",
    ),
    Entry(
        r"test/relation_candidate_pairs",
        "`test/relation_candidate_pairs`",
        "Pairs the entity head's hard mask proposed. The relation scores are "
        "over these, not over the corpus's pairs",
        "pairs",
    ),
    Entry(
        r"test/relation_none_share",
        "`test/relation_none_share`",
        "Share of those pairs whose gold label is `none`. It is a property "
        "of the current entity head, so it changes between checkpoints",
        "fraction, 0–1",
    ),
    Entry(
        r"test/\w+_(gold|predicted)_positives",
        "`test/<task>_{gold,predicted}_positives`",
        "Positive labels in the gold data and in the predictions. These are "
        "what tell a head predicting *nothing* from one predicting the "
        "*wrong* thing — both score micro-F1 0",
        "labels",
    ),
    Entry(
        r"test/\w+_labels_predicted",
        "`test/<task>_labels_predicted`",
        "Distinct columns the head ever fired on. A collapse onto one "
        "frequent label shows up here and nowhere else",
        "columns",
    ),
    Entry(
        r"test/detection_(precision|recall|f1)",
        "`test/detection_{precision,recall,f1}`",
        "Span-tagger detection scores over all entity types pooled",
        "score, 0–1",
    ),
    Entry(
        r"test/detection_\w+_(precision|recall)",
        "`test/detection_<type>_{precision,recall}`",
        "The same, per entity type",
        "score, 0–1",
    ),
    Entry(
        r"test/detection_(true_positives|false_positives|false_negatives)",
        "`test/detection_{true,false}_{positives,negatives}`",
        "The counts those scores are computed from",
        "spans",
    ),
    Entry(
        r"test/detection_documents(_missing_labels)?",
        "`test/detection_documents{,_missing_labels}`",
        "Documents the detection pass covered, and those carrying no token "
        "labels to score against",
        "documents",
    ),
    Entry(
        r"test/detection_ignore(d_predictions|_regions|_firing_rate)",
        "`test/detection_ignore*`",
        "Predictions dropped by an ignore region, the regions themselves, "
        "and the share of predictions they removed",
        "counts / fraction",
    ),
)

ENTRIES: Final = _PER_EPOCH + _SUMMARY + _CONTEXT + _TEST

STAGES: Final[dict[str, tuple[Entry, ...]]] = {
    "train": _PER_EPOCH + _SUMMARY + _CONTEXT,
    "tuning": _PER_EPOCH + _SUMMARY + _CONTEXT,
    "eval": _CONTEXT + _TEST,
}

_HEADER: Final = {
    "train": (
        "Per-epoch metrics are **averages over the pass's batches** — the "
        "step axis is the epoch number. The summary metrics are logged once, "
        "at the end, with no step."
    ),
    "tuning": (
        "Per-epoch metrics are **averages over the pass's batches** — the "
        "step axis is the epoch number. The summary metrics are logged once, "
        "at the end, with no step."
    ),
    "eval": (
        "One evaluation pass over the test split; every metric is logged "
        "once, with no step axis."
    ),
}


def describe(metric: str) -> Entry | None:
    """The glossary entry documenting `metric`, or `None` if none does.

    :param metric: the metric key.
    :return: its entry, or None — which is the thing worth catching in a test,
        since it means a metric reaches the server with no record of what its
        y-axis measures.
    """
    for entry in ENTRIES:
        if re.fullmatch(entry.pattern, metric):
            return entry

    return None


def glossary(stage: str) -> str:
    """The metric glossary for a run of `stage`, as a Markdown table.

    :param stage: the run's stage tag, which selects the table.
    :return: the Markdown to post as the run's description.
    """
    entries = STAGES.get(stage)
    if entries is None:
        return ""

    rows = "\n".join(
        f"| {entry.display} | {entry.axis} | {entry.unit} |"
        for entry in entries
    )

    return (
        f"### Metrics\n\n{_HEADER[stage]}\n\n"
        "| Metric | What one point is | Unit |\n|---|---|---|\n"
        f"{rows}\n"
    )
