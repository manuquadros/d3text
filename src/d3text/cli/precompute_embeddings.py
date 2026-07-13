#!/usr/bin/env python
import argparse
import os
import pathlib
import queue
import threading
import typing
from concurrent.futures import (
    FIRST_COMPLETED,
    Future,
    ThreadPoolExecutor,
    as_completed,
    wait,
)

import blosc2
import lmdb
import polars as pl
import torch
import tqdm
import transformers
from d3text import utils

CPU_COUNT = os.cpu_count() or 1
COMP_THREADS = max(1, CPU_COUNT // 2)
MAX_BACKLOG = max(8, COMP_THREADS * 2)


def read_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("base_model")
    p.add_argument("output_path")
    p.add_argument("datasets", nargs="+")
    p.add_argument(
        "-f",
        "--force-regenerate",
        action="store_true",
        help="re-embed documents already stored in the output LMDB",
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=50,
        help="token windows per forward pass; tune for your VRAM",
    )
    p.add_argument(
        "--max_length",
        type=int,
        default=None,
        help="tokens per window (default: the base model's context window)",
    )
    p.add_argument("--commit_every", type=int, default=100)
    p.add_argument(
        "--stream_batch", type=int, default=1000
    )  # rows per Polars slice
    return p.parse_args()


def window_size(
    max_length: int | None, model_config: transformers.PretrainedConfig
) -> int:
    """Resolve the number of tokens per window `embed_document` splits into.

    The tokenizer cannot be asked for this. `model_max_length` is a ~1e30
    sentinel whenever the tokenizer config declares no limit — which is the
    case for the default base model — and `split_and_tokenize` pads *to*
    `max_length`, so that sentinel asks for an impossible tensor. The position
    embeddings are the real cap: a longer window indexes past the table.
    """
    limit: int = model_config.max_position_embeddings
    if max_length is None:
        return limit
    if not 1 <= max_length <= limit:
        msg = (
            f"--max_length must be between 1 and {limit}, the context window "
            f"of {model_config.name_or_path}; got {max_length}."
        )
        raise ValueError(msg)
    return max_length


def stored_keys(env: lmdb.Environment) -> set[bytes]:
    """The pubmed ids already embedded in `env`.

    Keys only: the values are the compressed embeddings, and pulling those in
    just to test for presence would defeat the point of skipping them.
    """
    with env.begin() as txn:
        return set(txn.cursor().iternext(keys=True, values=False))


def tensor_to_bytes(t: torch.Tensor) -> bytes:
    a = t.detach().to(torch.float16).contiguous().cpu().numpy()
    return typing.cast(
        bytes,
        blosc2.pack_array(
            a,
            codec=blosc2.Codec.ZSTD,
            clevel=9,
            filter=blosc2.Filter.BITSHUFFLE,
        ),
    )


def writer_thread(
    env: lmdb.Environment,
    in_q: queue.Queue[tuple[bytes, bytes]],
    stop_evt: threading.Event,
    commit_every: int,
    pbar_written: tqdm.tqdm,
) -> None:
    tdb = env.begin(write=True)
    n_since = 0
    try:
        while True:
            if stop_evt.is_set() and in_q.empty():
                break
            try:
                k, v = in_q.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                tdb.put(k, v)
            except lmdb.MapFullError:
                tdb.commit()
                env.sync()
                stop_evt.set()
                continue

            n_since += 1
            pbar_written.update(1)
            if n_since >= commit_every:
                tdb.commit()
                tdb = env.begin(write=True)
                n_since = 0

        try:
            tdb.commit()
        except lmdb.Error:
            pass
        env.sync()
    except Exception:
        try:
            tdb.abort()
        except Exception:
            pass
        raise


def stream_rows(path: pathlib.Path, batch_size: int):
    """Yield (pmid, text) in small batches to keep RAM flat."""
    if path.suffix == ".csv":
        lazy = pl.scan_csv(path).drop("")
    elif path.suffix == ".json":
        lazy = pl.scan_ndjson(path).rename({"body": "fulltext"})
    else:
        raise ValueError(f"{path} has an unrecognized file format.")

    lazy = lazy.select(
        pl.col("pubmed_id"),
        pl.concat_str(
            [
                pl.col("abstract").fill_null(""),
                pl.col("fulltext").fill_null(""),
            ],
            separator="\n",
        ).alias("text"),
    )

    # total rows for tqdm
    total = lazy.select(pl.len()).collect().item()

    def _iter():
        for start in range(0, total, batch_size):
            df = lazy.slice(start, batch_size).collect()
            for pmid, text in df.iter_rows():
                yield pmid, text
            del df

    return total, _iter()


def main() -> None:
    # help CUDA memory fragmentation a bit
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    args = read_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = utils.load_fast_tokenizer(args.base_model)
    model = (
        transformers.AutoModel.from_pretrained(args.base_model)
        .to(device)
        .eval()
    )
    max_len = window_size(args.max_length, model.config)

    # LMDB env
    env = lmdb.open(args.output_path, map_size=100 * 1024**3)

    # Snapshot taken before any writing, so a document is judged against what
    # a *previous* run stored, not against this run's own output.
    already_embedded = set() if args.force_regenerate else stored_keys(env)

    for dataset in args.datasets:
        path = pathlib.Path(dataset)
        print(f"\nProcessing {path}")

        total_rows, row_iter = stream_rows(path, args.stream_batch)
        skipped = 0

        # In-flight compression jobs -> the pmid key each will be stored under.
        # Local to the dataset: a shared dict would let one dataset's undrained
        # leftovers be written while the next is processed.
        futures: dict[Future[bytes], bytes] = {}

        # queues + bars
        out_q: queue.Queue[tuple[bytes, bytes]] = queue.Queue(maxsize=124)
        stop_evt = threading.Event()

        pbar_emb = tqdm.tqdm(
            total=total_rows,
            desc="Embedded",
            position=0,
            leave=False,
            dynamic_ncols=True,
        )
        pbar_written = tqdm.tqdm(
            total=total_rows,
            desc="Written ",
            position=1,
            leave=False,
            dynamic_ncols=True,
        )

        # start writer
        wt = threading.Thread(
            target=writer_thread,
            args=(env, out_q, stop_evt, args.commit_every, pbar_written),
            daemon=True,
        )
        wt.start()

        # compression pool
        with (
            ThreadPoolExecutor(max_workers=COMP_THREADS) as pool,
            torch.inference_mode(),
        ):
            for pmid, text in row_iter:
                if stop_evt.is_set():
                    break

                key = str(pmid).encode()
                if key in already_embedded:
                    skipped += 1
                    pbar_emb.update(1)
                    pbar_written.total = total_rows - skipped
                    continue

                emb = utils.embed_document(
                    text,
                    tokenizer=tokenizer,
                    model=model,
                    stride=20,
                    batch_size=args.batch_size,
                    max_len=max_len,
                )
                pbar_emb.update(1)

                # submit for compression
                f = pool.submit(tensor_to_bytes, emb)
                futures[f] = key

                if len(futures) >= MAX_BACKLOG:
                    done, _ = wait(
                        list(futures.keys()), return_when=FIRST_COMPLETED
                    )
                    for d in done:
                        out_q.put((futures.pop(d), d.result()))

            # Drain unconditionally: the in-loop flush above is what keeps the
            # backlog *below* MAX_BACKLOG, so repeating that guard here would
            # be false exactly when there is still work in flight. A dataset
            # shorter than MAX_BACKLOG would then write nothing at all.
            for done_future in as_completed(list(futures)):
                if stop_evt.is_set():
                    # The writer stopped consuming (map full), so further puts
                    # would block forever once out_q fills.
                    break
                out_q.put((futures.pop(done_future), done_future.result()))

        # signal writer to finish; join
        stop_evt.set()
        wt.join()

        # close bars
        pbar_emb.close()
        pbar_written.close()

        if skipped:
            print(
                f"Skipped {skipped} documents already embedded in "
                f"{args.output_path}; pass -f to re-embed them."
            )

    print("Done.")


if __name__ == "__main__":
    main()
