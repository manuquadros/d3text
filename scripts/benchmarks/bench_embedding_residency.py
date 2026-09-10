"""Measurement: does keeping token embeddings on the GPU cost peak VRAM?

Runs both placements over the same pre-drawn batches in one process,
alternating across rounds, and reports peak allocated bytes, the seconds inside
`get_token_embeddings`, and whether the two agree bit for bit. One process is
what makes the timings trustworthy — two benchmark processes sharing a card
invent OOMs. `--order` swaps which placement goes first, because on a thermally
throttled card the first-measured one is favoured by enough to flip the sign of
a small difference. Equivalence is checked under `eval()`, since dropout would
make two passes differ for unrelated reasons.

Both arms are written out, and neither is the live method: borrowing
`type(model).get_token_embeddings` for the "cpu" arm would measure the
on-device placement twice, and the zero difference it would report is exactly
what a successful confirmation looks like.

`--source` picks which regime is measured. A machine-local `config.toml` can
set `cpu_embeddings_cache_mb` or `embeddings_store`, and both arms consult the
same process-wide cache and the same store, so a machine configured for
training turns the comparison into a read of one source timed against a read
of the same source. `off` forces both sources off so every batch pays the
base-model forward — the regime the on-device change targets. `configured`
leaves `config.toml` as-is and measures the *hit* path's residency instead, a
real regime of its own but not a substitute for `off`: there the on-device arm
holds every document's tensor beside the padded buffer, where the round-trip
arm moves only the finished buffer to the card. `select_source_regime` records
the cache and the store each arm actually got, so the JSON says which regime a
number belongs to.
"""

import argparse
import itertools
import json
import statistics as st
import time
from collections.abc import Sequence
from types import ModuleType
from typing import cast

import torch
from torch.nn.utils.rnn import pad_sequence

from d3text import data, factory, runtime
from d3text.datasets.brenda import BRENDA_SCHEMA, brenda_dataset
from d3text.models import base as M
from d3text.models.config import encodings, load_model_config
from d3text.models.model_types import BatchItem
from d3text.utils.utils import aggregate_embeddings

SOURCE_OFF = "off"
SOURCE_CONFIGURED = "configured"
SOURCE_REGIMES = (SOURCE_OFF, SOURCE_CONFIGURED)


def select_source_regime(
    regime: str, base_model: str, module: ModuleType = M
) -> dict[str, bool | str]:
    """Force the cache and the store off, or leave them as `config.toml` set.

    The record reads the live cache and the same cached `embeddings_store`
    call the arms make, not the config, so a configured store that fails to
    open or was written for another base model is recorded as absent.

    :param regime: `"off"` to force every batch through the base-model
        forward; `"configured"` to leave the machine config as-is.
    :param base_model: the base model the arms will open the store for.
    :param module: the `d3text.models.base` module, injectable for testing.
    :return: the regime and each arm's actual source availability, meant to
        be recorded verbatim in the run's JSON.
    """
    if regime == SOURCE_OFF:
        module.cpu_embeddings_cache = None
        module.mconfig.embeddings_store = None

    cache_on = module.cpu_embeddings_cache is not None
    store_on = module.embeddings_store(base_model) is not None

    return {
        "regime": regime,
        "cpu_arm_cache": cache_on,
        "cpu_arm_store": store_on,
        "gpu_arm_cache": cache_on,
        "gpu_arm_store": store_on,
    }


def _token_embeddings(
    self: M.Model, batch: Sequence[BatchItem], *, on_device: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    """`Model.get_token_embeddings`, with where it aggregates as a flag.

    One body for both arms makes "these differ only in residency" a property
    of the code rather than a promise, so every release the shipped method
    performs has to be performed here too.
    """
    device = self.device if on_device else "cpu"
    cache = M.cpu_embeddings_cache
    store = M.embeddings_store(self.config.base_model)
    inputs: list[None | torch.Tensor] = [None] * len(batch)
    missing = []

    for ix, item in enumerate(batch):
        doc_id = int(item["id"].item())
        if cache is not None:
            hit = cache.get(M.cpu_cache_key(self.config.base_model, doc_id))
            if hit is not None:
                inputs[ix] = hit
                continue
        if store is not None:
            stored = store.get(
                doc_id, expected_tokens=M.document_token_count(item)
            )
            if stored is not None:
                inputs[ix] = stored.to(dtype=self.amp_dtype)
                continue
        missing.append((ix, item))

    if missing:
        with torch.no_grad():
            bi = self.batch_input_tensors([i for _, i in missing])
            attn = bi["attention_mask"].to(self.device, non_blocking=True)
            with self.autocast_context():
                output = self.base_model(
                    input_ids=bi["input_ids"].to(
                        self.device, dtype=torch.int, non_blocking=True
                    ),
                    attention_mask=attn,
                ).last_hidden_state.detach()
        if not on_device:
            # Rebinding the one name is what ends the card residency here, as
            # the pre-change method's inline `.cpu()` did (as in `204e2af`); a
            # second name would hold both copies and charge this arm card
            # memory that method never used.
            output = output.cpu()
        # The host-side mask is `bi`'s own, never a copy back down: charging
        # the round-trip arm a transfer the pre-change method never made
        # would bias the very number this script exists to report.
        chunk_masks = attn if on_device else bi["attention_mask"]
        out_iter, mask_iter = iter(output), iter(chunk_masks)
        for ix, item in missing:
            n = item["doc_id"].shape[-1]
            outs = torch.stack(tuple(itertools.islice(out_iter, n))).to(
                dtype=self.amp_dtype
            )
            masks = torch.stack(tuple(itertools.islice(mask_iter, n)))
            emb = aggregate_embeddings(outs, masks)
            inputs[ix] = emb
            if cache is not None and not cache.full():
                cache.set(
                    M.cpu_cache_key(
                        self.config.base_model, int(item["id"].item())
                    ),
                    emb.cpu(),
                )

        # The shipped method drops the hidden states before it pads, so both
        # arms do: left bound, a whole `[chunks, WINDOW_LENGTH, embedding]`
        # tensor would sit beside the padded buffer on the on-device arm
        # alone. On the round-trip arm they are host memory by now, which
        # `max_memory_allocated` cannot see, so this neither costs nor
        # credits that arm anything.
        del output, out_iter

    # Hits wait on the host until the hidden states are gone, as in the
    # shipped method; moved earlier, they would inflate the on-device arm's
    # forward alone. On the round-trip arm `device` is the host, where every
    # tensor already is, so this moves nothing there.
    embeddings = [
        e.to(device, non_blocking=True)
        for e in cast(list[torch.Tensor], inputs)
    ]
    max_len = max(e.shape[0] for e in embeddings)
    padded = pad_sequence(embeddings, batch_first=True, padding_value=0.0)
    masks = torch.zeros(
        (len(embeddings), max_len), dtype=torch.bool, device=device
    )
    for i, e in enumerate(embeddings):
        masks[i, : e.shape[0]] = True
    return padded.to(self.device, non_blocking=True), masks.to(
        self.device, non_blocking=True
    )


def cpu_impl(
    self: M.Model, batch: Sequence[BatchItem]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Aggregate and pad on the host, then move the batch to the card."""
    return _token_embeddings(self, batch, on_device=False)


def gpu_impl(
    self: M.Model, batch: Sequence[BatchItem]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Aggregate and pad on the card; no round-trip."""
    return _token_embeddings(self, batch, on_device=True)


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Compare aggregating token embeddings on the CPU against doing it "
            "on the GPU: peak VRAM, time, and whether they agree bit for bit."
        )
    )
    p.add_argument("config", help="model config TOML, e.g. a tuned config")
    p.add_argument(
        "--budget",
        type=int,
        default=80,
        help="chunk budget per batch (TokenBudgetBatchSampler); peak VRAM "
        "scales with this, so it is the knob that sets the regime",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="training documents to load; also sizes the entity head, so it "
        "changes peak VRAM and is part of a measurement's identity",
    )
    p.add_argument(
        "--batches", type=int, default=8, help="batches measured per round"
    )
    p.add_argument(
        "--warmup",
        type=int,
        default=2,
        help="batches run before measurement starts, per variant",
    )
    p.add_argument(
        "--rounds",
        type=int,
        default=3,
        help="alternating passes over the measured batches; the median is "
        "reported, since a single round is thermal noise",
    )
    p.add_argument(
        "--order",
        default="cpu,gpu",
        help="which placement is measured first within each round; run both "
        "orders to tell an ordering artifact from a real difference",
    )
    p.add_argument(
        "--source",
        choices=SOURCE_REGIMES,
        default=SOURCE_OFF,
        help="'off' forces the CPU cache and embeddings store off so every "
        "batch pays the base-model forward, the regime the on-device "
        "change targets; 'configured' leaves config.toml as-is and "
        "measures the hit path instead. The VM confirmation needs both.",
    )
    a = p.parse_args()

    runtime.configure()
    cfg = load_model_config(a.config)
    # Before the dataset and the model: `embeddings_store` caches what it
    # opens, so `off` has to reach the config before anything calls it.
    source_info = select_source_regime(a.source, cfg.base_model)
    ds = brenda_dataset(
        schema=BRENDA_SCHEMA,
        encodings=encodings[cfg.base_model],
        limit=a.limit,
    )
    train = ds.data["train"]
    model = factory.build_model(
        cfg,
        ds,
        BRENDA_SCHEMA,
        entity_freqs=data.compute_frequencies(train, column="entities"),
        class_freqs=data.compute_frequencies(train, column="classes"),
    )
    model.to(model.device)

    impls = {"cpu": cpu_impl, "gpu": gpu_impl}

    loader = data.get_batch_loader(
        dataset=train, batch_size=cfg.batch_size, max_chunks=a.budget
    )
    it = iter(loader)
    batches = []
    for _ in range(a.warmup + a.batches):
        try:
            batches.append(next(it))
        except StopIteration:
            break
    warm, measured = batches[: a.warmup], batches[a.warmup :]
    if not measured:
        print(json.dumps({"budget": a.budget, "error": "no batches"}))
        return

    # Equivalence under eval(): dropout off, so any difference is real.
    model.eval()
    equiv = {"max_abs_delta": 0.0, "masks_equal": True, "shapes_equal": True}
    with torch.no_grad():
        for b in measured[:2]:
            ec, mc = cpu_impl(model, b)
            eg, mg = gpu_impl(model, b)
            equiv["shapes_equal"] &= (
                ec.shape == eg.shape and mc.shape == mg.shape
            )
            if ec.shape == eg.shape:
                # Compare on-device: both are already resident, and a
                # float32 CPU copy of a large batch thrashes the host.
                d = (ec.float() - eg.float()).abs().max().item()
                equiv["max_abs_delta"] = max(equiv["max_abs_delta"], d)
            equiv["masks_equal"] &= bool(torch.equal(mc, mg))
    del ec, mc, eg, mg
    torch.cuda.empty_cache()

    model.train()

    def step(b):
        losses = model.compute_batch_losses(b)
        loss = sum(losses) if isinstance(losses, tuple) else losses
        loss.backward()
        model.zero_grad(set_to_none=True)

    def run(variant, bs, measure):
        type(model).get_token_embeddings = impls[variant]
        emb_s = 0.0
        if measure:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        orig = impls[variant]

        def timed(self, batch):
            nonlocal emb_s
            torch.cuda.synchronize()
            t = time.perf_counter()
            r = orig(self, batch)
            torch.cuda.synchronize()
            emb_s += time.perf_counter() - t
            return r

        type(model).get_token_embeddings = timed
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for b in bs:
            step(b)
        torch.cuda.synchronize()
        total = time.perf_counter() - t0
        type(model).get_token_embeddings = impls[variant]
        return {
            "total_s": total,
            "emb_s": emb_s,
            "peak_MiB": torch.cuda.max_memory_allocated() / 2**20,
        }

    order = tuple(a.order.split(","))
    for v in order:
        run(v, warm, measure=False)

    res = {"cpu": [], "gpu": []}
    err = None
    try:
        for _ in range(a.rounds):
            for v in order:
                res[v].append(run(v, measured, measure=True))
    except torch.cuda.OutOfMemoryError as e:
        err = f"OOM: {str(e)[:120]}"

    out = {
        "budget": a.budget,
        "limit": a.limit,
        "model_class": cfg.model_class,
        "entity_columns": len(ds.entity_index),
        "batches": len(measured),
        "docs_per_batch": [len(b) for b in measured],
        "chunks_per_batch": [
            sum(int(i["doc_id"].shape[-1]) for i in b) for b in measured
        ],
        "equivalence": equiv,
        "error": err,
        "order": a.order,
        "source_regime": source_info,
    }
    for v in ("cpu", "gpu"):
        if res[v]:
            out[f"{v}_peak_MiB"] = max(r["peak_MiB"] for r in res[v])
            out[f"{v}_emb_s"] = st.median(r["emb_s"] for r in res[v])
            out[f"{v}_total_s"] = st.median(r["total_s"] for r in res[v])
            out[f"{v}_emb_all"] = [round(r["emb_s"], 4) for r in res[v]]
    print("BENCH " + json.dumps(out))


if __name__ == "__main__":
    main()
