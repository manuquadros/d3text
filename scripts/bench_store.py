"""Measurement: is an LMDB get + blosc2 decompress cheaper than the forward?

Sources token ids from the precomputed encodings HDF5 rather than the corpus
text, so the input is byte-identical to what the training path feeds the base
model. Also sizes the store: the ticket's 112 GiB train figure is
*uncompressed*, and the disk budget is set by what BITSHUFFLE+ZSTD9 actually
achieves on these activations, which has not been measured.
"""

import argparse
import statistics
import time

import h5py
import hdf5plugin  # noqa: F401  registers the Zstd filter h5py needs
import numpy as np
import torch
import transformers
from d3text.embeddings_store import bytes_to_tensor, tensor_to_bytes
from d3text.utils.utils import aggregate_embeddings

p = argparse.ArgumentParser()
p.add_argument("--docs", type=int, default=25)
p.add_argument("--base-model", default="michiyasunaga/BioLinkBERT-base")
p.add_argument(
    "--encodings", default="data/biolinkbert-base-zstd-22-encodings.hdf5"
)
p.add_argument("--batch-size", type=int, default=8)
a = p.parse_args()

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = transformers.AutoModel.from_pretrained(a.base_model).to(dev).eval()
amp = (
    torch.bfloat16
    if (dev.type == "cuda" and torch.cuda.is_bf16_supported())
    else torch.float16
)

f = h5py.File(a.encodings, "r")
# Evenly spaced through the file rather than the first N: chunk counts vary by
# an order of magnitude and the head of the file is not a sample of the tail.
keys = list(f.keys())
step = max(1, len(keys) // a.docs)
sample = keys[::step][: a.docs]

fwd, comp, decomp, raw, packed, toks, chunks = [], [], [], [], [], [], []

for i, key in enumerate(sample, 1):
    ids = torch.from_numpy(f[key]["input_ids"][:].astype(np.int64))
    mask = torch.from_numpy(f[key]["attention_mask"][:].astype(np.int64))

    if dev.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    outs = []
    with torch.inference_mode():
        for s in range(0, ids.size(0), a.batch_size):
            bi = ids[s : s + a.batch_size].to(dev)
            bm = mask[s : s + a.batch_size].to(dev)
            with torch.amp.autocast(device_type=dev.type, dtype=amp):
                outs.append(model(bi, bm).last_hidden_state.detach().cpu())
    emb = aggregate_embeddings(torch.cat(outs, dim=0), mask)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    blob = tensor_to_bytes(emb)
    t2 = time.perf_counter()
    back = bytes_to_tensor(blob)
    t3 = time.perf_counter()
    assert back.shape == emb.shape, (back.shape, emb.shape)

    fwd.append(t1 - t0)
    comp.append(t2 - t1)
    decomp.append(t3 - t2)
    raw.append(emb.numel() * 2)
    packed.append(len(blob))
    toks.append(emb.shape[0])
    chunks.append(ids.size(0))
    print(
        f"{i:3d} pmid={key} chunks={ids.size(0):3d} tokens={emb.shape[0]:6d} "
        f"fwd={t1 - t0:7.3f}s pack={t2 - t1:6.3f}s unpack={t3 - t2:6.3f}s "
        f"raw={raw[-1] / 2**20:7.2f}MiB packed={packed[-1] / 2**20:7.2f}MiB "
        f"ratio={raw[-1] / packed[-1]:5.2f}x",
        flush=True,
    )


def s(name, xs, unit=""):
    print(
        f"{name:24s} mean={statistics.mean(xs):9.4f}{unit} "
        f"median={statistics.median(xs):9.4f}{unit} "
        f"min={min(xs):9.4f}{unit} max={max(xs):9.4f}{unit}"
    )


name = torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu"
print(f"\n=== {len(sample)} documents on {dev} ({name}), amp={amp} ===")
s("forward (s)", fwd)
s("pack fp16+zstd9 (s)", comp)
s("unpack (s)", decomp)
s("chunks", [float(c) for c in chunks])
s("tokens", [float(n) for n in toks])
s("raw MiB", [r / 2**20 for r in raw])
s("packed MiB", [q / 2**20 for q in packed])

ratio = sum(raw) / sum(packed)
print(f"\naggregate compression ratio : {ratio:.3f}x")
print(
    f"forward / unpack            : {statistics.mean(fwd) / statistics.mean(decomp):.1f}x faster to read"
)
print(
    f"forward / (unpack+pack)     : {statistics.mean(fwd) / (statistics.mean(decomp) + statistics.mean(comp)):.1f}x"
)
print(
    "\n-- store size projection (ticket's uncompressed GiB / measured ratio) --"
)
tot = 0.0
for split, gib in (("train", 112.1), ("validation", 15.7), ("test", 15.2)):
    print(f"  {split:11s} {gib:6.1f} GiB raw -> {gib / ratio:6.1f} GiB on disk")
    tot += gib
print(f"  {'ALL':11s} {tot:6.1f} GiB raw -> {tot / ratio:6.1f} GiB on disk")
