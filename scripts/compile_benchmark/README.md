# Does `torch.compile` pay for this model?

`runtime.compile_model` is called on every training run, and the `compiled`
tag it produces says a graph was installed — not that the run was faster.
Nothing established that compiling is worth doing. The workload argues both
ways: the base transformer is frozen, so the compiled region is only the heads
and the pooling, and the batches are ragged, which is why `dynamic=True` is
passed at all. A dynamic graph that still retraces per shape pays tracing cost
on every new one and can lose to eager outright.

This directory measures it. Two arms train the same model on the same data
from one generated config, and differ in exactly one thing: whether
`D3TEXT_DISABLE_COMPILE` is set.

## What it needs

**A Triton-capable GPU: compute capability 7.0 or newer.** `run_arms.py`
refuses to start on anything else and that refusal is the deliverable's most
important line. Below 7.0, `runtime.is_triton_compatible()` is False, so
`compile_model` returns False *whatever the switch says*, both arms run eager,
and the benchmark would report a confident speedup of 1.00 drawn from pure
timing noise. **The P100 VM is compute capability 6.0 and cannot host this
benchmark**; run it on a card that can.

It also needs what any training run needs: the BRENDA splits, the encodings
store named by the config's `base_model`, and a writable working directory.

## Running it

```bash
bash scripts/compile_benchmark/run.sh
```

Knobs are environment variables —
`COMPILE_BENCH_{CONFIG,EPOCHS,LIMIT,REPEATS,OUT,PDM}` — and the header of
`run.sh` lists them. The pieces are usable on their own:

```bash
pdm run python scripts/compile_benchmark/run_arms.py cfg_base.toml \
    --epochs 3 --limit 500 --repeats 3 --out out/
pdm run python scripts/compile_benchmark/compare_arms.py out/run.json
```

A handful of epochs on a few hundred documents, not convergence: this is a
timing comparison and nothing about it improves with a trained model.

## Why the arms are interleaved

A card throttles under a sustained load, so two arms run back to back are two
arms run at two different clock speeds, and the second one is not slower
because of the switch. The arms are therefore interleaved, each repeat
reverses their order, and the report takes the **median** over repeats rather
than trusting one pair of numbers. Wall-clock timing in which arm order is
confounded with thermal state is not a measurement.

The first epoch is reported apart from the ones after it, because it is the
epoch that pays for tracing. Whether that cost is worth paying depends on how
many epochs a real run has to amortize it over, which is a question the reader
has and the tool does not.

## A crashed arm is a result

`nn.Module.compile` is **lazy**: it installs `_compiled_call_impl` and
returns without invoking inductor, so `compile_model`'s `try` cannot see a
backend failure. Inductor first runs inside the first forward, well outside
that call. This is not hypothetical — on an RTX 1000 Ada the compiled arm dies
about nine seconds into epoch 0 with

```
BackendCompilerFailed: backend='inductor' raised:
AssertionError: Node convert_element_type_16 was invalid, but is output
```

while the eager arm trains normally. So "the compiled arm does not survive
epoch 0 on this card" is an answer the tool has to be able to express, and it
does: the wrapper writes its metrics file in a `finally` with `completed:
false` and the exception's first line, the runner records the arm and carries
on, and `compare_arms.py` names the arm that failed before printing any
timings. A one-armed table is always labelled as one.

## What is not a result

Exactly one condition invalidates the comparison rather than answering it: an
arm that compiled when it should not have, or the reverse. `compare_arms.py`
reports that as `THE ARMS ARE NOT COMPARABLE` and exits non-zero; every other
outcome, a dead arm included, exits zero because it is a finding.

## Output

Everything lands in `out/` (self-ignoring). `run.json` holds every run's
per-epoch metrics, exit status and error; `train_<arm>_<repeat>.log` the full
console output; `report.md` the tables; `arms.json` the headline medians.
Nothing is resumable on purpose — half a benchmark resumed hours later on a
cold card is not half a measurement.
