#!/usr/bin/env python

"""Produce the per-token distant-supervision targets, offline.

One HDF5 store of `d3text.token_labels` targets, keyed by pubmed id and shaped
like the encodings the tagger reads. It needs no encodings file: re-tokenizing
`corpus.document_text` reproduces the stored `input_ids` element for element. A
leaf, like the other two precompute commands. Two passes over each corpus file,
since the other-organism names exist only inline in the documents.
"""

import argparse
import dataclasses
import logging
import multiprocessing
import os
import pathlib
from collections.abc import Iterable, Iterator

import h5py
import numpy
import transformers
from d3text import corpus, logs, surface_forms, token_labels, utils
from d3text.cli import args as cli_args
from numpy.typing import NDArray
from tqdm import tqdm

logger = logging.getLogger(__name__)

_Task = tuple[str, str, frozenset[str]]
"""A document still to label: its key, its text and its gold entity IDs."""

_Result = tuple[str, token_labels.DocumentLabels, NDArray[numpy.bool_], str]
"""A labelled document: its key, its targets, which of its token positions
are real content -- `False` at a padding, `[CLS]` or `[SEP]` position, the
ones `project_onto_tokens` forces to `IGNORE_INDEX` regardless of any
surface-form match -- and its `token_labels.document_fingerprint`."""

_pool_index: surface_forms.SurfaceFormIndex | None = None
_pool_tokenizer: transformers.PreTrainedTokenizerFast | None = None
"""Set by `_label_pooled` in the parent, before the pool forks.

A forked worker inherits both through copy-on-write rather than either being
pickled or rebuilt per task — the same reasoning that keeps the store
single-writer applies here: cheap to inherit once, wasteful to repeat per
document.
"""


def build_index(
    entity_tables: pathlib.Path, datasets: list[pathlib.Path]
) -> surface_forms.SurfaceFormIndex:
    """The surface-form index, over all four ID namespaces.

    :param entity_tables: the TinyDB dump holding the three entity tables.
    :param datasets: the corpus files to pool other-organism names from.
    :return: the index to match against.
    """
    tables = surface_forms.load_entity_tables(entity_tables)
    return surface_forms.build_index(
        surface_forms.brenda_surface_forms(
            tables,
            (
                names
                for dataset in datasets
                for names in corpus.other_organism_names(
                    dataset, corpus.STREAM_BATCH
                )
            ),
        ),
        excluded_words=surface_forms.excluded_single_words(tables),
    )


def label_document(
    text: str,
    gold_entity_ids: frozenset[str],
    index: surface_forms.SurfaceFormIndex,
    tokenizer: transformers.PreTrainedTokenizerFast,
) -> tuple[token_labels.DocumentLabels, NDArray[numpy.bool_]]:
    """One document's targets, in the geometry its encodings have.

    The mention spans, and every exact mention's candidate IDs and token
    anchors, come back with them and are stored with them, so a run cannot
    leave a document described by its codes alone.

    :param text: the document text the encodings were built from.
    :param gold_entity_ids: the entities this document is linked to.
    :param index: the surface forms to match.
    :param tokenizer: the tokenizer the encodings were built with.
    :return: the codes, the spans they were projected from, and the
        mentions' candidate IDs and anchors; and a content mask, `True`
        where the encoding's offset spans a real character range -- `False`
        at padding and at `[CLS]`/`[SEP]`, which `project_onto_tokens` always
        maps to `IGNORE_INDEX` regardless of any surface-form match, so
        counting them as abstention would conflate a window-geometry
        property with the matching rules' own abstention rate.
    """
    encoding = utils.split_and_tokenize(tokenizer=tokenizer, inputs=text)
    labels = token_labels.document_token_labels(
        text, index, gold_entity_ids, encoding["offset_mapping"]
    )
    offsets = numpy.asarray(encoding["offset_mapping"])
    content_mask = offsets[..., 1] > offsets[..., 0]
    return labels, content_mask


def _label_task(task: _Task) -> _Result:
    """One pooled worker's unit of work.

    Runs in a forked worker and returns the labels rather than writing them —
    the parent is the only process allowed to touch the HDF5 store.

    :param task: the document's key, text and gold entity IDs.
    :return: the key, paired with its labels, their content mask, and this
        document's fingerprint.
    :raises RuntimeError: if a worker's pool globals were never set, which
        means it ran before `_label_pooled` assigned them.
    """
    if _pool_index is None or _pool_tokenizer is None:
        msg = "worker pool globals were not set before the pool started"
        raise RuntimeError(msg)
    key, text, gold_entity_ids = task
    labels, content_mask = label_document(
        text, gold_entity_ids, _pool_index, _pool_tokenizer
    )
    fingerprint = token_labels.document_fingerprint(text, gold_entity_ids)
    return key, labels, content_mask, fingerprint


def _merge_duplicate_pubmed_ids(
    documents: Iterable[corpus.CorpusDocument],
) -> Iterator[corpus.CorpusDocument]:
    """Union rows that share a `pubmed_id` into one document.

    BRENDA curates one row per enzyme a paper documents, so a paper naming
    several enzymes reaches this stream as several rows sharing one
    `pubmed_id` and text, each carrying only part of the gold set. The store
    below is keyed by `pubmed_id` and would otherwise keep whichever row it
    labels last, silently dropping every other row's gold entities -- the
    corpus-reader twin of what `brenda_references.merge_duplicate_documents`
    fixes for training's own read of the same CSVs -- and applies the same
    tiebreaker, so a group whose rows disagree on text is not read one way
    for training and another for the label store: the group's `text` and
    `path` come from its first row with a non-null `path`, or its first row
    if none has one.

    :param documents: the corpus stream to merge, in read order.
    :return: one document per `pubmed_id`, with `entity_ids` and
        `other_organisms` unioned across its group.
    """
    merged: dict[corpus.PubmedId, corpus.CorpusDocument] = {}
    for document in documents:
        existing = merged.get(document.pubmed_id)
        if existing is None:
            merged[document.pubmed_id] = document
            continue
        primary = (
            document
            if existing.path is None and document.path is not None
            else existing
        )
        merged[document.pubmed_id] = dataclasses.replace(
            primary,
            entity_ids=existing.entity_ids | document.entity_ids,
            other_organisms={
                **existing.other_organisms,
                **document.other_organisms,
            },
        )
    yield from merged.values()


def _pending_documents(
    store: h5py.File,
    total: int,
    documents: Iterable[corpus.CorpusDocument],
    force_regenerate: bool,
) -> Iterator[_Task]:
    """The documents of `documents` that still need a fresh label.

    Applies the resume-skip and "no text" checks here, in the caller's
    process, which is also the only process allowed to write `store` — a
    document skipped or emptied here is never handed to a worker.

    A stored group is skipped only where it is complete (`holds_token_labels`)
    and its `token_labels.document_fingerprint` still matches the text and
    gold set the corpus gives that document now. Completeness alone is not
    enough: it still passes for a group whose document changed underneath it
    (a BRENDA refresh, a `corpus.document_text` change). Nor are
    `open_store`'s store-level stamps (the surface-form index digest, the
    tokenizer), already checked before this function runs — those cover the
    whole store and never move for one document's text or gold set. A group
    predating `token_labels.document_fingerprint` carries none, which this
    reads the same as a mismatch — always relabelled once, after which it
    carries one.

    :param store: the open label store; mutated for a document the corpus now
        gives no text, whose stale group (if any) is deleted.
    :param total: `documents`' row count, for the progress bar.
    :param documents: the corpus documents to consider, in stream order.
    :param force_regenerate: whether to re-label a document already stored.
    :return: `(key, text, gold_entity_ids)` for every document to label.

    `documents` is wrapped in `tqdm` here, not by the caller: passing an
    already-`tqdm`-wrapped iterator across this beartype-checked module's
    subscripted `Iterable[T]` parameter silently drops its first element (a
    beartype bug, not a `tqdm` one); wrapping after the checked boundary
    avoids it.
    """
    for document in tqdm(documents, position=1, desc="Rows", total=total):
        key = str(document.pubmed_id)
        if (
            token_labels.holds_token_labels(store, key)
            and not force_regenerate
            and token_labels.stored_document_fingerprint(store, key)
            == token_labels.document_fingerprint(
                document.text, document.entity_ids
            )
        ):
            continue

        if not document.text:
            logger.warning(
                "%s has neither an abstract nor a fulltext; "
                "storing no targets for it.",
                key,
            )
            # Reached with -f, for a group an interrupted run left
            # unfinished, or for a plain rerun whose fingerprint check found
            # a complete group stale because the document now has no text.
            # Either way the corpus now says this document has no text, so
            # whatever is stored for it goes.
            if key in store:
                del store[key]
            continue

        yield key, document.text, document.entity_ids


def _label_pooled(
    store: h5py.File,
    pending: Iterator[_Task],
    index: surface_forms.SurfaceFormIndex,
    tokenizer: transformers.PreTrainedTokenizerFast,
    workers: int,
) -> tuple[int, int]:
    """Label and store `pending`'s documents across a forked worker pool.

    Results are consumed as they complete (`imap_unordered`) rather than in
    submission order — safe because the store is keyed by pubmed id — but
    each is still stored by this, the parent process, the moment it arrives.

    :param store: the open label store to write into.
    :param pending: the documents to label.
    :param index: the surface forms to match, inherited by every worker.
    :param tokenizer: the tokenizer the encodings were built with, likewise
        inherited.
    :param workers: worker processes to run; must be greater than 1.
    :return: the abstained (`IGNORE_INDEX`) and total content-token counts
        labelled this call, padding and `[CLS]`/`[SEP]` excluded from both.
    """
    global _pool_index, _pool_tokenizer
    _pool_index = index
    _pool_tokenizer = tokenizer
    # Each worker holds its own copy of the tokenizer; its Rust-side thread
    # pool would otherwise oversubscribe the machine on top of this one.
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    ignored = total = 0
    # `fork` is what lets a worker inherit `index` and `tokenizer` by
    # copy-on-write instead of paying to pickle or rebuild either per task;
    # the platform default is fork only on Linux, so it is requested by name.
    with multiprocessing.get_context("fork").Pool(workers) as pool:
        # `imap_unordered` drives `pending` from a thread internal to `Pool`,
        # so that thread's reads and deletes of `store` (inside
        # `_pending_documents`) interleave with this loop's writes across two
        # OS threads of this same process — never a worker process. h5py
        # serializes every HDF5 call through its own global lock, so this is
        # still the one process, the parent, doing all the writing.
        for key, labels, content_mask, fingerprint in pool.imap_unordered(
            _label_task, pending
        ):
            content_codes = labels.codes[content_mask]
            ignored += int((content_codes == token_labels.IGNORE_INDEX).sum())
            total += content_codes.size
            token_labels.store_token_labels(
                store, key, labels, document_fingerprint=fingerprint
            )
    return ignored, total


def read_args() -> argparse.Namespace:
    """Parse and validate the command line.

    Every path is checked before the entity tables and the tokenizer are read:
    the tables are 1.1 GB and the index build scans every corpus file.

    :return: the parsed arguments.
    """
    parser = argparse.ArgumentParser(
        prog="precompute-token-labels",
        description=(
            "Place per-token distant-supervision targets for the documents of "
            "the provided data frames, by matching BRENDA's surface forms."
        ),
    )
    parser.add_argument(
        "base_model",
        help="the model whose tokenizer the encodings were built with",
    )
    parser.add_argument(
        "-e",
        "--entity-tables",
        type=cli_args.readable_path,
        default=None,
        help=(
            "BRENDA's TinyDB dump, holding the entity tables; defaults to "
            "the documents.json brenda_references is configured with"
        ),
    )
    parser.add_argument("output_path", help="HDF5 store to write")
    parser.add_argument(
        "datasets",
        nargs="*",
        type=cli_args.readable_path,
        help=(
            "corpus files to label; defaults to the splits and noise pools "
            "`brenda_references` is configured with, which is the set a "
            "training run reads"
        ),
    )
    parser.add_argument(
        "-f",
        "--force-regenerate",
        action="store_true",
        help=(
            "re-label the documents the store already holds from the passed "
            "datasets, even one whose stored group still matches its text "
            "and gold set; a plain rerun already relabels a group whose "
            "fingerprint is missing or stale"
        ),
    )
    parser.add_argument(
        "-j",
        "--workers",
        type=int,
        default=os.cpu_count() or 1,
        help=(
            "worker processes to label documents in parallel, over a "
            "forked pool that inherits the surface-form index and the "
            "tokenizer (default: every logical CPU); 1 or 0 labels "
            "serially in this process, with no pool at all"
        ),
    )

    args = parser.parse_args()
    args.datasets = cli_args.resolve_datasets(parser, args.datasets)

    if args.entity_tables is None:
        # Imported here, not at module scope, for the same reason
        # `resolve_datasets` imports `brenda_references` locally: a run given
        # its own -e pays nothing for the import.
        import brenda_references

        configured = brenda_references.documents_path()
        if not configured.is_file():
            parser.error(
                f"the configured entity tables dump is not present: "
                f"{configured} — fetch the data, or name the file with "
                "-e/--entity-tables"
            )
        args.entity_tables = configured

    output = pathlib.Path(args.output_path)
    if not output.parent.is_dir():
        parser.error(f"{output.parent} is not a directory")
    args.output_path = output

    if args.workers < 0:
        parser.error("--workers must not be negative")

    return args


def open_store(
    path: pathlib.Path,
    stamp: token_labels.IndexStamp,
    tokenizer: token_labels.TokenizerStamp,
) -> h5py.File:
    """The label store, with what produced its targets recorded or checked.

    A resumed store is checked rather than re-stamped: continuing under a
    different label space, against a surface-form index this invocation would
    build differently, or under another tokenizer or window geometry, leaves
    a file whose halves mean different things. The same argument refuses a
    store of an older layout, and the answer to any of them is a
    regeneration.

    :param path: the store to open or create.
    :param stamp: the surface-form index this invocation will label against.
    :param tokenizer: the tokenizer and window geometry this invocation will
        project its targets through.
    :return: the open store.
    :raises KeyError: if an existing store records no label space, index or
        tokenizer.
    :raises ValueError: if it records another label space, layout version,
        surface-form index, tokenizer or window geometry.
    """
    if not path.exists():
        store = h5py.File(path, "w-", libver="latest")
        token_labels.write_label_space(
            store, token_labels.BRENDA_LABELS, stamp=stamp, tokenizer=tokenizer
        )
        return store

    store = h5py.File(path, "r+", libver="latest")
    try:
        recorded = token_labels.read_label_space(store)
        if recorded != token_labels.BRENDA_LABELS:
            msg = (
                f"{path} holds targets over {recorded.types}, but this build "
                f"labels over {token_labels.BRENDA_LABELS.types}; "
                "regenerate it"
            )
            raise ValueError(msg)
        token_labels.check_index(store, stamp)
        token_labels.check_tokenizer(store, tokenizer)
    except (KeyError, ValueError):
        store.close()
        raise
    return store


def main() -> None:
    logs.configure()
    args = read_args()

    index = build_index(args.entity_tables, args.datasets)
    stamp = token_labels.IndexStamp.from_index(
        index,
        sources=[str(args.entity_tables), *(str(d) for d in args.datasets)],
    )
    logger.info(
        "Indexed %d surface forms over %d entities, as index %s.",
        len(index),
        len(index.entity_ids),
        stamp.digest[:12],
    )
    tokenizer = utils.load_fast_tokenizer(args.base_model)
    tokenizer_stamp = token_labels.TokenizerStamp.from_tokenizer(
        tokenizer,
        args.base_model,
        window_length=utils.WINDOW_LENGTH,
        window_stride=utils.WINDOW_STRIDE,
    )

    ignored_tokens = labelled_tokens = 0
    with open_store(args.output_path, stamp, tokenizer_stamp) as store:
        for dataset in tqdm(args.datasets, position=0, desc="Datasets"):
            total, documents = corpus.stream_documents(
                dataset, corpus.STREAM_BATCH
            )
            pending = _pending_documents(
                store,
                total,
                _merge_duplicate_pubmed_ids(documents),
                args.force_regenerate,
            )

            if args.workers > 1:
                ignored, labelled = _label_pooled(
                    store, pending, index, tokenizer, args.workers
                )
            else:
                ignored = labelled = 0
                for key, text, gold_entity_ids in pending:
                    labels, content_mask = label_document(
                        text, gold_entity_ids, index, tokenizer
                    )
                    content_codes = labels.codes[content_mask]
                    ignored += int(
                        (content_codes == token_labels.IGNORE_INDEX).sum()
                    )
                    labelled += content_codes.size
                    token_labels.store_token_labels(
                        store,
                        key,
                        labels,
                        document_fingerprint=token_labels.document_fingerprint(
                            text, gold_entity_ids
                        ),
                    )
            ignored_tokens += ignored
            labelled_tokens += labelled

    # Abstention (IGNORE_INDEX) is designed, not a defect (see
    # docs/distant-supervision.md), but its rate is a property of the surface-
    # form index and the matching rules, both of which move quietly across
    # commits -- logging it here is what makes a rules or index change that
    # shifts the rate visible at build time rather than found later by
    # diffing two stores. Counted over content tokens only (padding and
    # [CLS]/[SEP] excluded from both halves) so the rate reflects the
    # matching rules, not each document's share of window padding.
    if labelled_tokens:
        logger.info(
            "Abstained (IGNORE_INDEX) on %d of %d content tokens labelled "
            "this run (%.1f%%).",
            ignored_tokens,
            labelled_tokens,
            100 * ignored_tokens / labelled_tokens,
        )


if __name__ == "__main__":
    main()
