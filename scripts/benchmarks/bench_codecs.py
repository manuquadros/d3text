"""Which compression actually suits the precomputed-embeddings store.

`tensor_to_bytes` uses fp16 + BITSHUFFLE + ZSTD clevel 9 and achieves 1.145x,
which is close to not compressing at all for a second of CPU per document. This
sweeps the alternatives on real base-model activations and reports the only
three numbers that decide it: bytes per document, round-trip time, and how far
the reconstruction moves the vectors the heads consume.

Error is measured against the fp32 view of the live forward, not against the
fp16 store, so the existing cast is charged to the schemes that make it.
"""

import argparse
import statistics
import time

import blosc2
import h5py
import hdf5plugin  # noqa: F401  registers the Zstd filter h5py needs
import numpy as np
import torch
import transformers
from d3text.utils.utils import aggregate_embeddings

p = argparse.ArgumentParser()
p.add_argument("--docs", type=int, default=6)
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

ZSTD, LZ4, BLOSCLZ, ZLIB = (
    blosc2.Codec.ZSTD,
    blosc2.Codec.LZ4,
    blosc2.Codec.BLOSCLZ,
    blosc2.Codec.ZLIB,
)
NOF, SHUF, BITSHUF, TRUNC = (
    blosc2.Filter.NOFILTER,
    blosc2.Filter.SHUFFLE,
    blosc2.Filter.BITSHUFFLE,
    blosc2.Filter.TRUNC_PREC,
)


def _blosc(arr, codec, clevel, filters, metas):
    return blosc2.compress2(
        arr, codec=codec, clevel=clevel, filters=filters, filters_meta=metas
    )


def _unblosc(blob, dtype, shape):
    return np.frombuffer(blosc2.decompress2(blob), dtype=dtype).reshape(shape)


def lossless(codec, clevel, filt):
    """fp16, then a blosc2 pipeline — the family the store already uses."""

    def enc(x32):
        arr = x32.astype(np.float16)
        return _blosc(arr, codec, clevel, [filt], [0]), arr.shape

    def dec(blob, shape):
        return _unblosc(blob, np.float16, shape).astype(np.float32)

    return enc, dec


def truncated(bits, clevel=5):
    """fp16 with `bits` mantissa bits zeroed before the entropy coder.

    fp16 carries 10 mantissa bits; these activations do not need all of them,
    and the low ones are the noise that defeats zstd.
    """

    def enc(x32):
        arr = x32.astype(np.float16)
        return _blosc(arr, ZSTD, clevel, [TRUNC, BITSHUF], [bits, 0]), arr.shape

    def dec(blob, shape):
        return _unblosc(blob, np.float16, shape).astype(np.float32)

    return enc, dec


def int8_affine(clevel=5):
    """Per-row min/max quantisation to uint8, scales stored alongside."""

    def enc(x32):
        lo = x32.min(axis=1, keepdims=True)
        hi = x32.max(axis=1, keepdims=True)
        scale = np.maximum((hi - lo) / 255.0, 1e-12)
        q = np.rint((x32 - lo) / scale).astype(np.uint8)
        body = _blosc(q, ZSTD, clevel, [SHUF], [0])
        side = _blosc(
            np.concatenate([lo, scale], axis=1).astype(np.float32),
            ZSTD,
            clevel,
            [SHUF],
            [0],
        )
        return (body, side), x32.shape

    def dec(blob, shape):
        body, side = blob
        q = _unblosc(body, np.uint8, shape).astype(np.float32)
        s = _unblosc(side, np.float32, (shape[0], 2))
        return q * s[:, 1:2] + s[:, 0:1]

    return enc, dec


def fp8_e4m3(clevel=5):
    """torch's fp8, byte-viewed so blosc2 sees plain uint8."""

    def enc(x32):
        t = torch.from_numpy(x32).to(torch.float8_e4m3fn)
        arr = t.view(torch.uint8).numpy()
        return _blosc(arr, ZSTD, clevel, [SHUF], [0]), arr.shape

    def dec(blob, shape):
        raw = _unblosc(blob, np.uint8, shape).copy()
        t = torch.from_numpy(raw).view(torch.float8_e4m3fn)
        return t.to(torch.float32).numpy()

    return enc, dec


def raw_fp16():
    def enc(x32):
        arr = x32.astype(np.float16)
        return arr.tobytes(), arr.shape

    def dec(blob, shape):
        return (
            np.frombuffer(blob, dtype=np.float16)
            .reshape(shape)
            .astype(np.float32)
        )

    return enc, dec


SCHEMES: list[tuple[str, tuple]] = [
    ("CURRENT fp16+bitshuf+zstd9", lossless(ZSTD, 9, BITSHUF)),
    ("fp16+bitshuf+zstd5", lossless(ZSTD, 5, BITSHUF)),
    ("fp16+bitshuf+zstd1", lossless(ZSTD, 1, BITSHUF)),
    ("fp16+shuffle+zstd5", lossless(ZSTD, 5, SHUF)),
    ("fp16+nofilter+zstd5", lossless(ZSTD, 5, NOF)),
    ("fp16+bitshuf+lz4", lossless(LZ4, 5, BITSHUF)),
    ("fp16+shuffle+blosclz", lossless(BLOSCLZ, 5, SHUF)),
    ("fp16+bitshuf+zlib5", lossless(ZLIB, 5, BITSHUF)),
    ("fp16 raw (no codec)", raw_fp16()),
    ("trunc 2 mantissa bits", truncated(2)),
    ("trunc 3 mantissa bits", truncated(3)),
    ("trunc 4 mantissa bits", truncated(4)),
    ("trunc 5 mantissa bits", truncated(5)),
    ("trunc 6 mantissa bits", truncated(6)),
    ("int8 per-row affine", int8_affine()),
    ("fp8 e4m3", fp8_e4m3()),
]

f = h5py.File(a.encodings, "r")
keys = list(f.keys())
step = max(1, len(keys) // a.docs)
sample = keys[::step][: a.docs]

docs: list[np.ndarray] = []
for key in sample:
    ids = torch.from_numpy(f[key]["input_ids"][:].astype(np.int64))
    mask = torch.from_numpy(f[key]["attention_mask"][:].astype(np.int64))
    outs = []
    with torch.inference_mode():
        for s in range(0, ids.size(0), a.batch_size):
            bi = ids[s : s + a.batch_size].to(dev)
            bm = mask[s : s + a.batch_size].to(dev)
            with torch.amp.autocast(device_type=dev.type, dtype=amp):
                outs.append(model(bi, bm).last_hidden_state.detach().cpu())
    emb = aggregate_embeddings(torch.cat(outs, dim=0), mask)
    docs.append(emb.to(torch.float32).numpy())
    print(f"loaded {key}: {emb.shape}", flush=True)

raw_total = sum(d.size * 2 for d in docs)  # fp16 bytes, the store's unit
print(f"\n{len(docs)} documents, {raw_total / 2**20:.1f} MiB as raw fp16\n")

hdr = (
    f"{'scheme':28s} {'MiB/doc':>8s} {'vs fp16':>8s} {'pack s':>8s} "
    f"{'unpack s':>9s} {'cos err':>10s} {'max abs':>9s} {'store GiB':>10s}"
)
print(hdr)
print("-" * len(hdr))

rows = []
for name, (enc, dec) in SCHEMES:
    sizes, packs, unpacks, coss, maxes = [], [], [], [], []
    for d in docs:
        t0 = time.perf_counter()
        blob, shape = enc(d)
        t1 = time.perf_counter()
        back = dec(blob, shape)
        t2 = time.perf_counter()

        n = sum(len(b) for b in blob) if isinstance(blob, tuple) else len(blob)
        sizes.append(n)
        packs.append(t1 - t0)
        unpacks.append(t2 - t1)

        num = (back * d).sum(axis=1)
        den = np.linalg.norm(back, axis=1) * np.linalg.norm(d, axis=1) + 1e-12
        coss.append(float(np.mean(1.0 - num / den)))
        maxes.append(float(np.abs(back - d).max()))

    mib = statistics.mean(sizes) / 2**20
    ratio = raw_total / sum(sizes) * len(docs) / len(docs)
    ratio = (sum(d.size * 2 for d in docs) / len(docs)) / statistics.mean(sizes)
    store = 143.0 / ratio
    rows.append(
        (
            name,
            mib,
            ratio,
            statistics.mean(packs),
            statistics.mean(unpacks),
            statistics.mean(coss),
            statistics.mean(maxes),
            store,
        )
    )
    print(
        f"{name:28s} {mib:8.2f} {ratio:7.2f}x {statistics.mean(packs):8.3f} "
        f"{statistics.mean(unpacks):9.4f} {statistics.mean(coss):10.2e} "
        f"{statistics.mean(maxes):9.4f} {store:9.1f}"
    )

print(
    "\ncos err = mean per-token (1 - cosine similarity) against the fp32 view "
    "of the live forward.\nstore GiB projects the ticket's 143.0 GiB "
    "uncompressed all-splits total through each scheme's ratio."
)
