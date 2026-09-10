"""Measurement: does keeping token embeddings on the GPU cost peak VRAM?

Runs both placements over the same pre-drawn batches in one process,
alternating across rounds, and reports peak allocated bytes, the seconds inside
`get_token_embeddings`, and whether the two agree bit for bit. One process is
what makes the timings trustworthy — two benchmark processes sharing a card
invent OOMs. `--order` swaps which placement goes first, because on a thermally
throttled card the first-measured one is favoured by enough to flip the sign of
a small difference. Equivalence is checked under `eval()`, since dropout would
make two passes differ for unrelated reasons. Results and the write-up are in
`design/oom-06/`.

`--source` picks which regime is measured. A machine-local `config.toml` can
set `cpu_embeddings_cache_mb` or `embeddings_store`, and both `cpu_impl` (the
production `get_token_embeddings`) and `gpu_impl` below consult the same
process-wide cache, so a machine configured for training turns the comparison
into a cache (or, for the store, a store) read timed against another read of
the same source, with the equivalence check trivially true either way. `off`
forces both sources off so every batch pays the base-model forward — the
regime the on-device change targets. `configured` leaves `config.toml` as-is
and instead measures the *hit* path's residency, which is a real regime on
its own (a stub, all-store-hit corpus peaks higher on the on-device arm) but
not a substitute for `off`; only `cpu_impl` ever queries the store, so under
`configured` the two arms are not reading from the same source set, and
`select_source_regime` records exactly what each arm can read from so that
asymmetry shows up in the JSON instead of behind a clean-looking number.
"""

import argparse
import itertools
import json
import statistics as st
import time
from types import ModuleType
from typing import cast

import torch
from torch.nn.utils.rnn import pad_sequence

from d3text import data, factory, runtime
from d3text.datasets.brenda import BRENDA_SCHEMA, brenda_dataset
from d3text.models import base as M
from d3text.models.config import encodings, load_model_config
from d3text.utils.utils import aggregate_embeddings

SOURCE_OFF = "off"
SOURCE_CONFIGURED = "configured"
SOURCE_REGIMES = (SOURCE_OFF, SOURCE_CONFIGURED)


def select_source_regime(
    regime: str, module: ModuleType = M
) -> dict[str, bool | str]:
    """Force the CPU cache and embeddings store off, or leave them exactly
    as `config.toml` set them, and report what each arm can read from.

    `cpu_impl` and `gpu_impl` both check `module.cpu_embeddings_cache`, but
    only `cpu_impl` queries `module.embeddings_store` (gated on
    `module.mconfig.embeddings_store`), so the two arms are symmetric only
    when the store is off. Recording that per arm is what stops a
    `configured` run's asymmetry from reading as a clean comparison.

    :param regime: `"off"` to force every batch through the base-model
        forward; `"configured"` to leave the machine config as-is.
    :param module: the `d3text.models.base` module, injectable for testing.
    :return: the regime and each arm's actual source availability, meant to
        be recorded verbatim in the run's JSON.
    """
    if regime == SOURCE_OFF:
        module.cpu_embeddings_cache = None
        module.mconfig.embeddings_store = None

    cache_on = module.cpu_embeddings_cache is not None
    store_on = bool(module.mconfig.embeddings_store)

    return {
        "regime": regime,
        "cpu_arm_cache": cache_on,
        "cpu_arm_store": store_on,
        "gpu_arm_cache": cache_on,
        "gpu_arm_store": False,
    }


def gpu_impl(self, batch):
    """PERF-02: aggregate and pad on the GPU; no CPU round-trip."""
    cache = M.cpu_embeddings_cache
    inputs: list[None | torch.Tensor] = [None] * len(batch)
    missing = []
    for ix, item in enumerate(batch):
        key = M.cpu_cache_key(self.config.base_model, int(item["id"].item()))
        hit = cache.get(key) if cache is not None else None
        if hit is not None:
            inputs[ix] = hit.to(self.device, non_blocking=True)
        else:
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
        out_iter, mask_iter = iter(output), iter(attn)
        for ix, item in missing:
            n = item["doc_id"].shape[-1]
            outs = torch.stack(tuple(itertools.islice(out_iter, n))).to(
                dtype=self.amp_dtype
            )
            masks = torch.stack(tuple(itertools.islice(mask_iter, n)))
            emb = aggregate_embeddings(outs, masks)
            inputs[ix] = emb
            if cache is not None and self.training and not cache.full():
                cache.set(
                    M.cpu_cache_key(
                        self.config.base_model, int(item["id"].item())
                    ),
                    emb.cpu(),
                )

    embeddings = cast(list[torch.Tensor], inputs)
    max_len = max(e.shape[0] for e in embeddings)
    padded = pad_sequence(embeddings, batch_first=True, padding_value=0.0)
    masks = torch.zeros(
        (len(embeddings), max_len), dtype=torch.bool, device=self.device
    )
    for i, e in enumerate(embeddings):
        masks[i, : e.shape[0]] = True
    return padded, masks


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
    source_info = select_source_regime(a.source)
    cfg = load_model_config(a.config)
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

    cpu_impl = type(model).get_token_embeddings
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
