"""Measurement: should the embeddings store be LMDB or HDF5?

`hdf5plugin` ships the same Blosc2 codec `embeddings_store` drives directly, so
the comparison is made at identical codec settings on real activations, or it
measures the codec rather than the container. Three numbers decide it and they
do not agree: size is a near-tie, whole-document reads favour LMDB, and
row-range reads favour HDF5, which decompresses only the chunks it needs.
`--slice` sets the fraction a row-range read asks for and `--chunk-rows` sweeps
HDF5 chunk shapes. Both stores are written through the same bf16
int16-reinterpretation, since neither HDF5 nor numpy has a bfloat16 type.
"""

import argparse
import os
import shutil
import statistics
import time

import h5py
import hdf5plugin
import lmdb
import numpy as np
import torch
import transformers
from d3text.embeddings_store import bytes_to_tensor, tensor_to_bytes
from d3text.utils.utils import aggregate_embeddings

p = argparse.ArgumentParser()
p.add_argument("--docs", type=int, default=12)
p.add_argument("--base-model", default="michiyasunaga/BioLinkBERT-base")
p.add_argument(
    "--encodings", default="data/biolinkbert-base-zstd-22-encodings.hdf5"
)
p.add_argument("--batch-size", type=int, default=8)
p.add_argument(
    "--out",
    default=os.environ.get("TMPDIR", "/tmp"),
    help="where the two throwaway stores are built; needs room for both.",
)
p.add_argument(
    "--slice",
    type=float,
    default=0.2,
    help=(
        "fraction of a document's rows the range read asks for, taken from "
        "the middle. The pipeline issues no such read today — this measures "
        "an advantage HDF5 would only realise if one were added."
    ),
)
p.add_argument(
    "--chunk-rows",
    type=int,
    nargs="+",
    default=[64, 256, 1024, 4096, 16384],
    help=(
        "HDF5 chunk shapes to sweep, in rows; a value at or above the "
        "document's token count stores it as a single chunk."
    ),
)
a = p.parse_args()

# The codec `embeddings_store._CPARAMS` drives blosc2 with, spelled as the HDF5
# filter. Identical settings are the whole point: any difference in the numbers
# below is then the container's, not the compressor's.
CODEC = dict(
    hdf5plugin.Blosc2(cname="zstd", clevel=5, filters=hdf5plugin.Blosc2.SHUFFLE)
)

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = transformers.AutoModel.from_pretrained(a.base_model).to(dev).eval()

f = h5py.File(a.encodings, "r")
keys = list(f.keys())
step = max(1, len(keys) // a.docs)
sample = keys[::step][: a.docs]

embs: dict[str, torch.Tensor] = {}
fwd: list[float] = []
for key in sample:
    ids = torch.from_numpy(f[key]["input_ids"][:].astype(np.int64))
    mask = torch.from_numpy(f[key]["attention_mask"][:].astype(np.int64))
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    outs = []
    with torch.inference_mode():
        for s in range(0, ids.size(0), a.batch_size):
            with torch.amp.autocast(device_type=dev.type, dtype=torch.bfloat16):
                outs.append(
                    model(
                        ids[s : s + a.batch_size].to(dev),
                        mask[s : s + a.batch_size].to(dev),
                    )
                    .last_hidden_state.detach()
                    .cpu()
                )
    emb = aggregate_embeddings(torch.cat(outs, dim=0), mask)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    fwd.append(time.perf_counter() - t0)
    embs[key] = emb
f.close()

raw = sum(e.shape[0] * e.shape[1] * 2 for e in embs.values())
n = len(embs)
print(
    f"{n} documents on {dev.type}, "
    f"{statistics.mean(e.shape[0] for e in embs.values()):.0f} tokens/doc mean, "
    f"{raw / 2**20:.2f} MiB raw bf16"
)
print(f"base-model forward: {statistics.mean(fwd) * 1000:.1f} ms/doc\n")


def as_int16(tensor: torch.Tensor) -> np.ndarray:
    """The bf16 bit pattern, which is what either container actually stores."""
    return tensor.to(torch.bfloat16).contiguous().view(torch.int16).numpy()


def warm(path: str) -> None:
    """Page-cache the store, so the comparison is decompression, not IO."""
    for root, _, files in os.walk(path) if os.path.isdir(path) else ():
        for name in files:
            os.system(f"cat {os.path.join(root, name)} > /dev/null")
    if os.path.isfile(path):
        os.system(f"cat {path} > /dev/null")


def du(path: str) -> int:
    if os.path.isfile(path):
        return os.path.getsize(path)
    return sum(os.path.getsize(os.path.join(path, x)) for x in os.listdir(path))


lmdb_path = os.path.join(a.out, "bench-emb.lmdb")
shutil.rmtree(lmdb_path, ignore_errors=True)
env = lmdb.open(lmdb_path, map_size=max(8 * 1024**3, raw * 4))
with env.begin(write=True) as txn:
    for key, emb in embs.items():
        txn.put(key.encode(), tensor_to_bytes(emb))
env.close()
warm(lmdb_path)

env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False)
t0 = time.perf_counter()
with env.begin(buffers=True) as txn:
    lmdb_full = {k: bytes_to_tensor(bytes(txn.get(k.encode()))) for k in sample}
lmdb_full_s = time.perf_counter() - t0

t0 = time.perf_counter()
with env.begin(buffers=True) as txn:
    for key in sample:
        # No partial read exists: the blob is one frame, so a row range costs
        # a full inflate and then a slice of the result.
        emb = bytes_to_tensor(bytes(txn.get(key.encode())))
        rows = emb.shape[0]
        lo = int(rows * (0.5 - a.slice / 2))
        emb[lo : lo + int(rows * a.slice)]
lmdb_slice_s = time.perf_counter() - t0

print(
    f"{'store':30s} {'MiB':>8s} {'ratio':>7s} {'full':>10s} "
    f"{int(a.slice * 100)}% slice"
)
print(
    f"{'LMDB + blosc2 (today)':30s} {du(lmdb_path) / 2**20:8.2f} "
    f"{raw / du(lmdb_path):6.2f}x {lmdb_full_s * 1000 / n:8.2f}ms "
    f"{lmdb_slice_s * 1000 / n:8.2f}ms"
)

for chunk_rows in a.chunk_rows:
    path = os.path.join(a.out, "bench-emb.hdf5")
    if os.path.exists(path):
        os.remove(path)
    with h5py.File(path, "w", libver="latest") as g:
        for key, emb in embs.items():
            arr = as_int16(emb)
            g.create_dataset(
                key,
                data=arr,
                dtype="int16",
                chunks=(min(arr.shape[0], chunk_rows), arr.shape[1]),
                **CODEC,
            )
    warm(path)

    g = h5py.File(path, "r")
    t0 = time.perf_counter()
    h5_full = {
        k: torch.from_numpy(g[k][()]).view(torch.bfloat16) for k in sample
    }
    h5_full_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    for key in sample:
        rows = g[key].shape[0]
        lo = int(rows * (0.5 - a.slice / 2))
        g[key][lo : lo + int(rows * a.slice)]
    h5_slice_s = time.perf_counter() - t0
    g.close()

    # The containers must agree bit for bit, or the timings compare two
    # different stores rather than two ways of holding one.
    for key in sample:
        assert torch.equal(lmdb_full[key], h5_full[key]), key

    label = f"HDF5 + same blosc2, {chunk_rows} rows"
    print(
        f"{label:30s} {du(path) / 2**20:8.2f} {raw / du(path):6.2f}x "
        f"{h5_full_s * 1000 / n:8.2f}ms {h5_slice_s * 1000 / n:8.2f}ms"
    )
    os.remove(path)

print(
    f"\nforward is {statistics.mean(fwd) * 1000:.1f} ms/doc; a read must beat "
    f"that to be worth building at all."
)
