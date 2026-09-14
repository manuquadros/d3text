#!/usr/bin/env python

"""Produce the per-token distant-supervision targets, offline.

One HDF5 store of `d3text.token_labels` targets, keyed by pubmed id and shaped
like the encodings the tagger reads. It needs no encodings file: re-tokenizing
`corpus.document_text` reproduces the stored `input_ids` element for element. A
leaf, like the other two precompute commands. Two passes over each corpus file,
since the other-organism names exist only inline in the documents.
"""

import argparse
import logging
import multiprocessing
import os
import pathlib
from collections.abc import Iterable, Iterator

import h5py
import transformers
from d3text import corpus, logs, surface_forms, token_labels, utils
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Rows pulled into memory at a time. Not a flag, for the reason
# `precompute-encodings` gives: it trades nothing a caller cares about, and the
# corpus is streamed precisely so it need not be tuned.
STREAM_BATCH = 1000

_Task = tuple[str, str, frozenset[str]]
"""A document still to label: its key, its text and its gold entity IDs."""

_Result = tuple[str, token_labels.DocumentLabels]
"""A labelled document: its key, paired with its targets."""

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
                for names in corpus.other_organism_names(dataset, STREAM_BATCH)
            ),
        )
    )


def label_document(
    text: str,
    gold_entity_ids: frozenset[str],
    index: surface_forms.SurfaceFormIndex,
    tokenizer: transformers.PreTrainedTokenizerFast,
) -> token_labels.DocumentLabels:
    """One document's targets, in the geometry its encodings have.

    The mention spans, and every exact mention's candidate IDs and token
    anchors, come back with them and are stored with them, so a run cannot
    leave a document described by its codes alone.

    :param text: the document text the encodings were built from.
    :param gold_entity_ids: the entities this document is linked to.
    :param index: the surface forms to match.
    :param tokenizer: the tokenizer the encodings were built with.
    :return: the codes, the spans they were projected from, and the
        mentions' candidate IDs and anchors.
    """
    encoding = utils.split_and_tokenize(tokenizer=tokenizer, inputs=text)
    return token_labels.document_token_labels(
        text, index, gold_entity_ids, encoding["offset_mapping"]
    )


def _label_task(task: _Task) -> _Result:
    """One pooled worker's unit of work.

    Runs in a forked worker and returns the labels rather than writing them —
    the parent is the only process allowed to touch the HDF5 store.

    :param task: the document's key, text and gold entity IDs.
    :return: the key, paired with its labels.
    :raises RuntimeError: if a worker's pool globals were never set, which
        means it ran before `_label_pooled` assigned them.
    """
    if _pool_index is None or _pool_tokenizer is None:
        msg = "worker pool globals were not set before the pool started"
        raise RuntimeError(msg)
    key, text, gold_entity_ids = task
    return key, label_document(
        text, gold_entity_ids, _pool_index, _pool_tokenizer
    )


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
        if token_labels.holds_token_labels(store, key) and not force_regenerate:
            continue

        if not document.text:
            logger.warning(
                "%s has neither an abstract nor a fulltext; "
                "storing no targets for it.",
                key,
            )
            # Reached with -f, or for a group an interrupted run left
            # unfinished. The corpus now says this document has no text, so
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
) -> None:
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
    """
    global _pool_index, _pool_tokenizer
    _pool_index = index
    _pool_tokenizer = tokenizer
    # Each worker holds its own copy of the tokenizer; its Rust-side thread
    # pool would otherwise oversubscribe the machine on top of this one.
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

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
        for key, labels in pool.imap_unordered(_label_task, pending):
            token_labels.store_token_labels(store, key, labels)


def _readable(path: str) -> pathlib.Path:
    resolved = pathlib.Path(path)
    if not resolved.is_file():
        raise argparse.ArgumentTypeError(f"{path} is not a readable file")
    return resolved


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
        "entity_tables",
        type=_readable,
        help="BRENDA's TinyDB dump, holding the entity tables",
    )
    parser.add_argument("output_path", help="HDF5 store to write")
    parser.add_argument("datasets", nargs="+", type=_readable)
    parser.add_argument(
        "-f",
        "--force-regenerate",
        action="store_true",
        help="re-label documents the store already holds",
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

    output = pathlib.Path(args.output_path)
    if not output.parent.is_dir():
        parser.error(f"{output.parent} is not a directory")
    args.output_path = output

    if args.workers < 0:
        parser.error("--workers must not be negative")

    return args


def open_store(path: pathlib.Path, stamp: token_labels.IndexStamp) -> h5py.File:
    """The label store, with what produced its targets recorded or checked.

    A resumed store is checked rather than re-stamped: continuing under a
    different label space, or against a surface-form index this invocation
    would build differently, leaves a file whose halves mean different things.
    The same argument refuses a store of an older layout, and the answer to
    any of them is a regeneration.

    :param path: the store to open or create.
    :param stamp: the surface-form index this invocation will label against.
    :return: the open store.
    :raises KeyError: if an existing store records no label space or index.
    :raises ValueError: if it records another label space, layout version or
        surface-form index.
    """
    if not path.exists():
        store = h5py.File(path, "w-", libver="latest")
        token_labels.write_label_space(
            store, token_labels.BRENDA_LABELS, stamp=stamp
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

    with open_store(args.output_path, stamp) as store:
        for dataset in tqdm(args.datasets, position=0, desc="Datasets"):
            total, documents = corpus.stream_documents(dataset, STREAM_BATCH)
            pending = _pending_documents(
                store, total, documents, args.force_regenerate
            )

            if args.workers > 1:
                _label_pooled(store, pending, index, tokenizer, args.workers)
            else:
                for key, text, gold_entity_ids in pending:
                    token_labels.store_token_labels(
                        store,
                        key,
                        label_document(text, gold_entity_ids, index, tokenizer),
                    )


if __name__ == "__main__":
    main()
