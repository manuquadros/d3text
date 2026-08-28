#!/usr/bin/env python
"""Measure whether the document-level class objective localizes.

The models are trained on one 0/1 vector per document over the schema's class
names: `_pool_logits` collapses `[document, token, class]` before the loss sees
it, and multiple-instance learning is supposed to do the rest. Whether the
per-token class logits that survive that collapse actually land *on* the entity
mentions has never been measured, and the answer decides how much of a span
tagger has to be built.

This probe answers it without annotation. It re-runs the frozen base model over
a sample of validation documents, pushes the token embeddings through the
trained hidden block and class head, and compares the per-token class
probabilities against gold mentions located by string matching — each document's
*own* linked entities only, so a match is a mention the document is actually
annotated for.

The matcher here is deliberately standalone rather than `DictTagger`'s: `Vocab`
silently drops all but a fraction of a wordlist, and a probe cannot be built on
a matcher with a known hole in it.

Surface forms come from three places, none of which needs the BRENDA database:

- ``bacteria`` and ``other_organisms`` are ``{id: name}`` dicts in the split CSV;
- ``enzymes`` get their recommended name and synonyms from the ``enzymes`` table
  at the tail of ``documents.json``;
- ``strains`` get their designations and culture-collection numbers from that
  file's ``strains`` table.

Run it from a writable directory: it reaches `brenda_references`, which pulls in
`lpsn_interface` and its import-time `lpsn.log`.

Usage::

    pdm run python scripts/localization_probe.py <config.toml> <model.pt> \
        --documents 200 --out probe.json
"""

import argparse
import ast
import json
import logging
import pathlib
import re
import sys
from collections.abc import Iterable, Iterator, Mapping, Sequence
from typing import Any, TypedDict

import pandas as pd
import torch
from torch import Tensor

from d3text import checkpoint, corpus, logs
from d3text.datasets.brenda import BRENDA_SCHEMA
from d3text.factory import MODEL_CLASSES, ConfigurableModel, fix_keys_hook
from d3text.models.config import ModelConfig, load_model_config
from d3text.utils import (
    aggregate_embeddings,
    load_fast_tokenizer,
    split_and_tokenize,
)
from d3text.vocabulary import Vocabulary

# Under the `d3text` hierarchy on purpose: `logs.configure()` puts its handler
# on that logger and sets `propagate = False`, so a `__main__` logger reaches
# no handler at all and the probe runs to completion in silence.
logger = logging.getLogger("d3text.localization_probe")

# The tail of `documents.json` carries three tables the split CSVs do not:
# enzyme synonyms, bacterial synonyms and strain designations. Every *document*
# record spells `"enzymes"` with a list, so this key is unique to the table and
# is what anchors the tail parse.
ENZYME_TABLE_KEY = b'"enzymes": {"'

# How far back from EOF to look for it. The three tables together are ~8 MB;
# this is slack, not a measurement.
TAIL_SEARCH_BYTES = 256 * 1024 * 1024

# A surface form shorter than this is not evidence: two characters match
# somewhere in every document of a 4,000-token biochemistry paper.
MIN_SURFACE_FORM = 3

# Forms at or below this length, or carrying no lowercase letter, are matched
# case-sensitively. `fuzz.QRatio` case-folds, which is how `for` comes to match
# the enzyme abbreviation `FOR`; the same trap is open to a plain search.
ACRONYM_LENGTH = 5

# The probability above which a token counts as "fired". Reported alongside the
# threshold-free ranking measures precisely because a head trained under
# `pos_weight` is not calibrated.
FIRE_THRESHOLD = 0.5

# Windows pushed through the frozen base model at once.
WINDOW_BATCH = 8


class ClassStats(TypedDict):
    """One class's counters, pooled over the documents positive for it."""

    documents_positive: int
    documents_with_located_mentions: int
    entities_gold: int
    entities_located: int
    tokens: int
    tokens_gold: int
    tokens_fired: int
    tokens_fired_and_gold: int
    prob_sum_gold: float
    prob_sum_background: float
    auc_sum: float
    auc_documents: int
    top1_hits: int
    topk_hits: int
    topk_total: int
    negative_documents: int
    negative_tokens: int
    negative_tokens_fired: int
    noise_documents: int
    noise_tokens: int
    noise_tokens_fired: int
    # The pooled objective the head was actually trained on, kept beside the
    # per-token one so a silent channel can be told from an untrained one.
    pooled_positive_documents: int
    pooled_negative_documents: int
    pooled_fired_on_positive: int
    pooled_fired_on_negative: int
    pooled_prob_sum_positive: float
    pooled_prob_sum_negative: float


def new_stats() -> ClassStats:
    return ClassStats(
        documents_positive=0,
        documents_with_located_mentions=0,
        entities_gold=0,
        entities_located=0,
        tokens=0,
        tokens_gold=0,
        tokens_fired=0,
        tokens_fired_and_gold=0,
        prob_sum_gold=0.0,
        prob_sum_background=0.0,
        auc_sum=0.0,
        auc_documents=0,
        top1_hits=0,
        topk_hits=0,
        topk_total=0,
        negative_documents=0,
        negative_tokens=0,
        negative_tokens_fired=0,
        noise_documents=0,
        noise_tokens=0,
        noise_tokens_fired=0,
        pooled_positive_documents=0,
        pooled_negative_documents=0,
        pooled_fired_on_positive=0,
        pooled_fired_on_negative=0,
        pooled_prob_sum_positive=0.0,
        pooled_prob_sum_negative=0.0,
    )


def read_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="localization-probe",
        description=(
            "Compare per-token class probabilities against dictionary-located "
            "gold mentions on a sample of validation documents."
        ),
    )
    parser.add_argument(
        "config", help="Config the checkpoint was trained under."
    )
    parser.add_argument("model_state_dict", help="Checkpoint to probe.")
    parser.add_argument(
        "--documents",
        type=int,
        default=200,
        help="Curated validation documents to probe (default 200).",
    )
    parser.add_argument(
        "--noise-documents",
        type=int,
        default=50,
        help=(
            "Out-of-domain noise articles to probe alongside them. They carry "
            "no entities, so they measure the other half of the objective: "
            "what the head does on text no mention could be in."
        ),
    )
    parser.add_argument(
        "--encodings",
        default=None,
        help=(
            "Precomputed encodings HDF5. When given, each document's "
            "tokenization is checked against the one the model trained on and "
            "a mismatch is counted rather than silently probed."
        ),
    )
    parser.add_argument(
        "--out", default=None, help="Write the metrics here as JSON."
    )
    return parser.parse_args()


def find_entity_tables(path: pathlib.Path) -> dict[str, dict[str, Any]]:
    """The ``enzymes``/``bacteria``/``strains`` tables of the TinyDB dump.

    Parsed off the tail rather than loaded whole: the file is 1.1 GB of
    document records and the three tables that carry surface forms are the last
    8 MB of it.

    :raises ValueError: if the enzyme table's key is not in the tail, which is
        what a dump written in another key order would look like.
    """
    size = path.stat().st_size
    start = max(0, size - TAIL_SEARCH_BYTES)
    with path.open("rb") as dump:
        dump.seek(start)
        tail = dump.read()

    offset = tail.find(ENZYME_TABLE_KEY)
    if offset < 0:
        msg = (
            f"{path} carries no {ENZYME_TABLE_KEY.decode()!r} in its last "
            f"{TAIL_SEARCH_BYTES} bytes; the entity tables are elsewhere in it"
        )
        raise ValueError(msg)

    tables: dict[str, dict[str, Any]] = json.loads(
        "{" + tail[offset:].decode("utf8")
    )
    return tables


def surface_forms(
    entities: Mapping[str, Iterable[str]],
) -> dict[str, list[str]]:
    """Drop forms too short or too generic to be evidence, and dedupe."""
    return {
        entity_id: sorted(
            {
                form.strip()
                for form in forms
                if form and len(form.strip()) >= MIN_SURFACE_FORM
            }
        )
        for entity_id, forms in entities.items()
    }


def enzyme_forms(table: Mapping[str, Any]) -> dict[str, list[str]]:
    """Enzyme ID -> recommended name, synonyms and EC number."""
    return surface_forms(
        {
            entity_id: [
                record.get("recommended_name") or "",
                record.get("ec_class") or "",
                *(record.get("synonyms") or []),
            ]
            for entity_id, record in table.items()
        }
    )


def bacteria_forms(table: Mapping[str, Any]) -> dict[str, list[str]]:
    """Bacterium ID -> organism name and its LPSN synonyms."""
    return surface_forms(
        {
            entity_id: [
                record.get("organism") or "",
                *(record.get("synonyms") or []),
            ]
            for entity_id, record in table.items()
        }
    )


def strain_forms(table: Mapping[str, Any]) -> dict[str, list[str]]:
    """Strain ID -> designations and culture-collection numbers.

    The strain's ``taxon`` name is deliberately left out: it names the
    *species*, so counting it as a strain mention would score the strain
    channel on bacterium mentions.
    """
    return surface_forms(
        {
            entity_id: [
                *(record.get("designations") or []),
                *(
                    culture.get("strain_number") or ""
                    for culture in (record.get("cultures") or [])
                ),
            ]
            for entity_id, record in table.items()
        }
    )


def compile_forms(forms: Sequence[str]) -> re.Pattern[str] | None:
    """One regex matching any of `forms` on a token boundary.

    Split into a case-folded and a case-sensitive alternation: a long
    descriptive name is safe to fold, an acronym is not — folded, the
    determiner ``for`` matches the enzyme abbreviation ``FOR``.
    """
    foldable = [
        form
        for form in forms
        if len(form) > ACRONYM_LENGTH and any(c.islower() for c in form)
    ]
    exact = [form for form in forms if form not in set(foldable)]

    branches = []
    if foldable:
        branches.append("(?i:" + "|".join(re.escape(f) for f in foldable) + ")")
    if exact:
        branches.append("|".join(re.escape(f) for f in exact))
    if not branches:
        return None

    return re.compile(
        r"(?<![A-Za-z0-9])(?:" + "|".join(branches) + r")(?![A-Za-z0-9])"
    )


class GoldMentions(TypedDict):
    spans: list[tuple[int, int]]
    entities_gold: int
    entities_located: int


def locate_mentions(
    text: str, forms_by_entity: Mapping[str, Sequence[str]]
) -> GoldMentions:
    """Character spans in `text` of any surface form of any given entity."""
    spans: list[tuple[int, int]] = []
    located = 0
    for forms in forms_by_entity.values():
        pattern = compile_forms(forms)
        if pattern is None:
            continue
        hits = [
            (match.start(), match.end()) for match in pattern.finditer(text)
        ]
        if hits:
            located += 1
            spans.extend(hits)

    return GoldMentions(
        spans=spans,
        entities_gold=len(forms_by_entity),
        entities_located=located,
    )


def token_mask_from_spans(
    offsets: Tensor, spans: Sequence[tuple[int, int]]
) -> Tensor:
    """Which tokens overlap any of `spans`, given their character offsets."""
    mask = torch.zeros(offsets.shape[0], dtype=torch.bool)
    if not spans:
        return mask

    starts = offsets[:, 0]
    ends = offsets[:, 1]
    for span_start, span_end in spans:
        mask |= (starts < span_end) & (ends > span_start)
    return mask


def document_token_logits(
    model: ConfigurableModel,
    tokenizer: Any,
    text: str,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Per-token class probabilities, offsets, input ids, and the pooled vector.

    Runs the same two steps the training forward does — the frozen base model,
    then the hidden block and the class head — and then pools the result twice
    over: once not at all, which is what this probe is about, and once through
    the model's own `_pool_logits`, which is what the loss actually saw.

    The pooled vector is what makes the unpooled one legible. A channel that is
    silent on every token is only evidence of a *degenerate* solution if the
    document objective it was trained on is meanwhile being met; a channel
    silent at both levels is evidence of nothing but an undertrained head.
    """
    encoding = split_and_tokenize(tokenizer=tokenizer, inputs=text)
    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]
    offsets = encoding["offset_mapping"]

    windows = []
    with torch.no_grad():
        for start in range(0, input_ids.shape[0], WINDOW_BATCH):
            stop = start + WINDOW_BATCH
            with model.autocast_context():
                hidden = model.base_model(
                    input_ids=input_ids[start:stop].to(
                        model.device, dtype=torch.int
                    ),
                    attention_mask=attention_mask[start:stop].to(model.device),
                ).last_hidden_state
            windows.append(hidden.detach().cpu())

    embeddings = aggregate_embeddings(
        torch.cat(windows).to(dtype=model.amp_dtype), attention_mask
    )
    aggregated_offsets = aggregate_embeddings(offsets, attention_mask)
    aggregated_ids = aggregate_embeddings(
        input_ids.unsqueeze(-1), attention_mask
    )

    width = int(model.class_columns.numel())
    if embeddings.shape[0] == 0:
        # A document whose text tokenizes to nothing but [CLS]/[SEP]. The noise
        # pool holds one (pmid 21434216: 167 characters of whitespace once the
        # JATS tags come off, which `document_text`'s emptiness check passes
        # because whitespace is truthy). It is not a measurement either way, so
        # the caller drops it -- but note the two poolings disagree about it
        # rather than both refusing it: `logsumexp` over an empty token dim
        # returns -inf, which sigmoids to a confident 0 and reads as a document
        # correctly predicted negative, while `amax` raises outright.
        return (
            torch.empty(0, width),
            aggregated_offsets,
            aggregated_ids.squeeze(-1),
            torch.full((width,), float("nan")),
        )

    with torch.no_grad(), model.autocast_context():
        _, class_logits = model.classifier(
            model.hidden(embeddings.to(model.device))
        )
        probabilities = torch.sigmoid(model.drop_oos(class_logits).float())
        # `[document, token, class]` with `dim=1`, so this takes the same
        # `pool_token_dim` path the training forward takes rather than the
        # general fallback.
        pooled = model._pool_logits(class_logits.unsqueeze(0), dim=1)
        document = torch.sigmoid(model.drop_oos(pooled).float()).squeeze(0)

    return (
        probabilities.cpu(),
        aggregated_offsets,
        aggregated_ids.squeeze(-1),
        document.cpu(),
    )


def token_auc(probabilities: Tensor, gold: Tensor) -> float | None:
    """P(a random gold token outranks a random background token).

    Rank-based rather than thresholded: the class head is trained under a
    `pos_weight` and its sigmoid is not calibrated, so where the probabilities
    sit says less than how they are ordered.
    """
    positives = int(gold.sum())
    negatives = int(gold.numel() - positives)
    if positives == 0 or negatives == 0:
        return None

    order = torch.argsort(probabilities)
    ranks = torch.empty_like(order, dtype=torch.float64)
    ranks[order] = torch.arange(
        1, probabilities.numel() + 1, dtype=torch.float64
    )
    # Ties get their mean rank, so a head that assigns one probability to every
    # token scores 0.5 rather than whatever the sort order happened to be.
    _, inverse, counts = torch.unique(
        probabilities, return_inverse=True, return_counts=True
    )
    tie_sums = torch.zeros(counts.numel(), dtype=torch.float64).scatter_add_(
        0, inverse, ranks
    )
    ranks = (tie_sums / counts)[inverse]

    rank_sum = float(ranks[gold].sum())
    return (rank_sum - positives * (positives + 1) / 2) / (
        positives * negatives
    )


def build_model(
    config: ModelConfig, vocabulary: Vocabulary, state_dict: dict
) -> ConfigurableModel:
    """The checkpoint's model, built from its own recorded vocabulary.

    Deliberately not `factory.build_model`: that takes an
    `EntityRelationDataset`, and a probe that only needs the heads should not
    pay for the 560 MB training split to get them. The vocabulary carries
    everything the constructor asks for.
    """
    model_class = MODEL_CLASSES[config.model_class]
    model = model_class(
        schema=BRENDA_SCHEMA,
        class_matrix=vocabulary.class_matrix(),
        config=config,
        entity_index=vocabulary.entity_index,
    )
    model.register_load_state_dict_pre_hook(fix_keys_hook)
    model.load_state_dict(state_dict)
    model.to(model.device)
    model.eval()
    return model


def validation_documents(limit: int) -> pd.DataFrame:
    """The curated validation split, with the ``{id: name}`` dicts intact.

    `brenda_references.validation_data` is not used here even though this is
    its split: `preprocess_labels` reduces the bacteria and other-organism
    dicts to their keys, and the values are exactly the surface forms this
    probe needs. The row filter it applies is reproduced below.
    """
    from brenda_references.brenda_references import DATA_DIR

    frame = pd.read_csv(
        pathlib.Path(str(DATA_DIR)) / "validation_data.csv", index_col=0
    )
    frame = frame.dropna(subset=["abstract", "fulltext"])
    for column in ("enzymes", "bacteria", "strains", "other_organisms"):
        frame[column] = frame[column].apply(ast.literal_eval)

    # `validation_data`'s own filter: a document with bacteria but no strain is
    # dropped from the split, so probing one would probe a document the model
    # never validated on.
    keep = ~(
        frame["bacteria"].astype("bool") & ~frame["strains"].astype("bool")
    )
    return frame[keep].head(limit)


def noise_articles(limit: int) -> pd.DataFrame:
    """The validation split's own block of the out-of-domain noise pool."""
    from brenda_references.brenda_references import noise_documents

    if limit <= 0:
        return pd.DataFrame()
    return noise_documents("validation", limit)


def document_entities(
    row: pd.Series,
    tables: Mapping[str, dict[str, list[str]]],
) -> dict[str, dict[str, list[str]]]:
    """This document's linked entities, by class, with their surface forms.

    Names for the two organism columns come off the row itself, which is the
    only place they exist; enzymes and strains are looked up by ID.
    """
    return {
        "enzymes": {
            str(entity_id): tables["enzymes"].get(str(entity_id), [])
            for entity_id in row["enzymes"]
        },
        "strains": {
            str(entity_id): tables["strains"].get(str(entity_id), [])
            for entity_id in row["strains"]
        },
        "bacteria": {
            str(entity_id): surface_forms(
                {
                    str(entity_id): [
                        name,
                        *tables["bacteria"].get(str(entity_id), []),
                    ]
                }
            )[str(entity_id)]
            for entity_id, name in row["bacteria"].items()
        },
        "other_organisms": {
            str(entity_id): surface_forms({str(entity_id): [name]})[
                str(entity_id)
            ]
            for entity_id, name in row["other_organisms"].items()
        },
    }


def iterate_rows(
    frame: pd.DataFrame, noise: bool
) -> Iterator[tuple[str, str, pd.Series, bool]]:
    for _, row in frame.iterrows():
        text = corpus.document_text(row["abstract"], row["fulltext"])
        if not text:
            continue
        yield str(row["pubmed_id"]), text, row, noise


def encoded_ids(store: Any, pubmed_id: str) -> Tensor | None:
    if store is None or pubmed_id not in store:
        return None
    group = store[pubmed_id]
    ids = torch.as_tensor(group["input_ids"][()], dtype=torch.long)
    mask = torch.as_tensor(group["attention_mask"][()], dtype=torch.long)
    return aggregate_embeddings(ids.unsqueeze(-1), mask).squeeze(-1)


def summarize(stats: Mapping[str, ClassStats]) -> dict[str, dict[str, float]]:
    """Turn the counters into the rates the ticket asks to read."""
    summary: dict[str, dict[str, float]] = {}
    for name, counts in stats.items():
        tokens = counts["tokens"]
        gold = counts["tokens_gold"]
        fired = counts["tokens_fired"]
        background = tokens - gold
        base_rate = gold / tokens if tokens else 0.0
        precision = counts["tokens_fired_and_gold"] / fired if fired else 0.0
        summary[name] = {
            "documents_positive": float(counts["documents_positive"]),
            "documents_scored": float(
                counts["documents_with_located_mentions"]
            ),
            "entity_location_rate": (
                counts["entities_located"] / counts["entities_gold"]
                if counts["entities_gold"]
                else 0.0
            ),
            "gold_token_share": base_rate,
            "fire_rate": fired / tokens if tokens else 0.0,
            "precision": precision,
            "recall": counts["tokens_fired_and_gold"] / gold if gold else 0.0,
            "lift": precision / base_rate if base_rate else float("nan"),
            "mean_prob_gold": counts["prob_sum_gold"] / gold if gold else 0.0,
            "mean_prob_background": (
                counts["prob_sum_background"] / background
                if background
                else 0.0
            ),
            "token_auc": (
                counts["auc_sum"] / counts["auc_documents"]
                if counts["auc_documents"]
                else float("nan")
            ),
            "top1_hit_rate": (
                counts["top1_hits"] / counts["auc_documents"]
                if counts["auc_documents"]
                else float("nan")
            ),
            "topk_precision": (
                counts["topk_hits"] / counts["topk_total"]
                if counts["topk_total"]
                else float("nan")
            ),
            "negative_document_fire_rate": (
                counts["negative_tokens_fired"] / counts["negative_tokens"]
                if counts["negative_tokens"]
                else float("nan")
            ),
            "document_recall": (
                counts["pooled_fired_on_positive"]
                / counts["pooled_positive_documents"]
                if counts["pooled_positive_documents"]
                else float("nan")
            ),
            "document_false_positive_rate": (
                counts["pooled_fired_on_negative"]
                / counts["pooled_negative_documents"]
                if counts["pooled_negative_documents"]
                else float("nan")
            ),
            "document_mean_prob_positive": (
                counts["pooled_prob_sum_positive"]
                / counts["pooled_positive_documents"]
                if counts["pooled_positive_documents"]
                else float("nan")
            ),
            "document_mean_prob_negative": (
                counts["pooled_prob_sum_negative"]
                / counts["pooled_negative_documents"]
                if counts["pooled_negative_documents"]
                else float("nan")
            ),
            "noise_document_fire_rate": (
                counts["noise_tokens_fired"] / counts["noise_tokens"]
                if counts["noise_tokens"]
                else float("nan")
            ),
        }
    return summary


def report(summary: Mapping[str, Mapping[str, float]]) -> None:
    columns = (
        ("docs+", "documents_positive", "%7.0f"),
        ("scored", "documents_scored", "%7.0f"),
        ("ent.loc", "entity_location_rate", "%8.3f"),
        ("gold%", "gold_token_share", "%8.4f"),
        ("fire%", "fire_rate", "%8.4f"),
        ("prec", "precision", "%8.4f"),
        ("rec", "recall", "%8.4f"),
        ("lift", "lift", "%8.2f"),
        ("AUC", "token_auc", "%8.3f"),
        ("top1", "top1_hit_rate", "%8.3f"),
        ("top-k", "topk_precision", "%8.3f"),
        ("neg fire%", "negative_document_fire_rate", "%10.4f"),
        ("noise fire%", "noise_document_fire_rate", "%12.4f"),
        ("doc rec", "document_recall", "%9.3f"),
        ("doc FPR", "document_false_positive_rate", "%9.3f"),
    )
    header = "%-16s" % "class" + "".join(
        ("%" + fmt.split("%")[1].split(".")[0].lstrip("-") + "s") % label
        for label, _, fmt in columns
    )
    logger.info("%s", header)
    for name, values in summary.items():
        logger.info(
            "%s",
            "%-16s" % name
            + "".join(fmt % values[key] for _, key, fmt in columns),
        )


def main() -> None:
    logs.configure()
    args = read_args()

    config = load_model_config(args.config)
    saved = checkpoint.load(args.model_state_dict)
    if saved.vocabulary is None:
        logger.error(
            "%s records no vocabulary; the probe cannot know which class owns "
            "which column.",
            args.model_state_dict,
        )
        sys.exit(1)

    logger.info("Building the model from the checkpoint's vocabulary...")
    model = build_model(config, saved.vocabulary, saved.state_dict)
    tokenizer = load_fast_tokenizer(config.base_model)
    class_names = model.known_classes

    logger.info("Reading the entity surface forms...")
    from brenda_references.brenda_references import DATA_DIR

    tables = find_entity_tables(pathlib.Path(str(DATA_DIR)) / "documents.json")
    forms = {
        "enzymes": enzyme_forms(tables["enzymes"]),
        "bacteria": bacteria_forms(tables["bacteria"]),
        "strains": strain_forms(tables["strains"]),
    }

    store = None
    if args.encodings:
        import h5py
        import hdf5plugin  # noqa: F401  (registers the Zstd filter)

        store = h5py.File(args.encodings, "r")

    stats = {name: new_stats() for name in class_names}
    documents = 0
    tokenization_mismatches = 0
    empty_documents = 0

    rows = list(iterate_rows(validation_documents(args.documents), noise=False))
    rows += list(iterate_rows(noise_articles(args.noise_documents), noise=True))

    for pubmed_id, text, row, is_noise in rows:
        probabilities, offsets, ids, pooled = document_token_logits(
            model, tokenizer, text
        )
        trained_ids = encoded_ids(store, pubmed_id)
        if trained_ids is not None and (
            trained_ids.shape != ids.shape or not torch.equal(trained_ids, ids)
        ):
            tokenization_mismatches += 1
            continue

        if probabilities.shape[0] == 0:
            empty_documents += 1
            logger.warning(
                "%s tokenizes to no tokens at all; skipped.", pubmed_id
            )
            continue

        documents += 1
        gold = (
            {name: {} for name in class_names}
            if is_noise
            else document_entities(row, forms)
        )

        for column, name in enumerate(class_names):
            counts = stats[name]
            probability = probabilities[:, column]
            fired = probability >= FIRE_THRESHOLD

            # A noise article is a negative document for every class, which is
            # exactly the role the corpus appends it in.
            positive_document = not is_noise and bool(gold[name])
            document_probability = float(pooled[column])
            if positive_document:
                counts["pooled_positive_documents"] += 1
                counts["pooled_prob_sum_positive"] += document_probability
                counts["pooled_fired_on_positive"] += int(
                    document_probability >= FIRE_THRESHOLD
                )
            else:
                counts["pooled_negative_documents"] += 1
                counts["pooled_prob_sum_negative"] += document_probability
                counts["pooled_fired_on_negative"] += int(
                    document_probability >= FIRE_THRESHOLD
                )

            if is_noise:
                counts["noise_documents"] += 1
                counts["noise_tokens"] += probability.numel()
                counts["noise_tokens_fired"] += int(fired.sum())
                continue

            entities = gold[name]
            if not entities:
                counts["negative_documents"] += 1
                counts["negative_tokens"] += probability.numel()
                counts["negative_tokens_fired"] += int(fired.sum())
                continue

            counts["documents_positive"] += 1
            mentions = locate_mentions(text, entities)
            counts["entities_gold"] += mentions["entities_gold"]
            counts["entities_located"] += mentions["entities_located"]
            if not mentions["spans"]:
                continue

            counts["documents_with_located_mentions"] += 1
            mask = token_mask_from_spans(offsets, mentions["spans"])
            counts["tokens"] += probability.numel()
            counts["tokens_gold"] += int(mask.sum())
            counts["tokens_fired"] += int(fired.sum())
            counts["tokens_fired_and_gold"] += int((fired & mask).sum())
            counts["prob_sum_gold"] += float(probability[mask].sum())
            counts["prob_sum_background"] += float(probability[~mask].sum())

            auc = token_auc(probability, mask)
            if auc is not None:
                counts["auc_sum"] += auc
                counts["auc_documents"] += 1
                counts["top1_hits"] += int(mask[int(probability.argmax())])
                k = int(mask.sum())
                top_k = torch.topk(probability, k).indices
                counts["topk_hits"] += int(mask[top_k].sum())
                counts["topk_total"] += k

        if documents % 25 == 0:
            logger.info("Probed %d documents.", documents)

    if store is not None:
        store.close()

    logger.info(
        "Probed %d documents (%d skipped on a tokenization mismatch, %d as "
        "empty).",
        documents,
        tokenization_mismatches,
        empty_documents,
    )
    summary = summarize(stats)
    report(summary)

    if args.out:
        payload = {
            "checkpoint": args.model_state_dict,
            "config": args.config,
            "pooling": config.entity_logits_pooling,
            "documents": documents,
            "tokenization_mismatches": tokenization_mismatches,
            "empty_documents": empty_documents,
            "fire_threshold": FIRE_THRESHOLD,
            "summary": summary,
            "counts": stats,
        }
        pathlib.Path(args.out).write_text(json.dumps(payload, indent=2))
        logger.info("Wrote %s.", args.out)


if __name__ == "__main__":
    main()
