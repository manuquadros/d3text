# Runtime, logging and experiment tracking

## Process-wide runtime configuration

TF32, the float32 matmul precision, the CUDA/HIP caching allocator, tokenizer
parallelism, the RNG seed and where the library's log records go are all
*process*-global and sticky: the first writer wins, and nothing undoes it.
Setting them while a module is being imported makes a run's numerics depend on
import order — which is how `tune` came to train at a different matmul precision
from `train`, its own setting landing after the one `d3text.models` applied on
the way in.

So they belong to whoever owns the process, not to whichever module happens to
be imported first. `runtime.configure()` is called from a script's `main()`;
tests, notebooks and the precompute scripts inherit torch's own defaults unless
they ask for these. Call it **before any CUDA work**: the caching allocator
reads its environment variable when it first initialises and ignores it
thereafter.

### GPU checks

`unsupported_gpu_architecture` says so if the installed torch ships no kernels
for the present GPU. ROCm has no equivalent of PTX: a wheel carries object code
for the architectures it was built for and nothing else, so a card outside that
list fails at the *first* device allocation with `HIP error: invalid device
function` — arbitrarily deep into whatever ran first, and with
`torch.cuda.is_available()` having answered True all along. HIP builds only: a
CUDA wheel embeds PTX and JITs forward-compatibly, and `gcnArchName` is a ROCm
property in the first place. Anything unexpected reads as nothing to report — a
startup check that ends a run is worse than the crash it was meant to explain.

`is_triton_compatible` asks up front whether `torch.compile`'s Triton backend
can target the GPU (compute capability 7.0, Volta, or newer). Asking up front
matters because `torch.compile` is lazy: on an older card it returns a wrapper
quite happily and only fails at the first forward pass, long past the
`try/except` the call site wraps it in.

### torch.compile and the runtime type checker

`beartype_this_package()` wraps every annotated function in this package in a
checker that runs `isinstance(x, Float[Tensor, ...])`, and dynamo cannot
evaluate that call. Tracing into jaxtyping's `__instancecheck__` builds a guard
on a bound method's object id that fails on the very frame that created it, and
torch aborts with `AssertionError: Guard failed on the same frame it was
created`. Where it does not trace in, it constant-folds the check through
`issubclass` to `False` instead, and beartype rejects a tensor that is perfectly
valid. Either way the run dies before its first batch.

`exclude_type_checkers_from_dynamo` skips those frames, leaving the checks
themselves running eagerly and unchanged; only the model's own frames are
compiled. All three entries are needed — skipping the two packages still leaves
the generated wrapper traced, and skipping the wrapper alone lets dynamo pick
`__instancecheck__` up as a top-level frame of its own. It is idempotent,
because `SKIP_DIRS` is a process-global list backing a compiled regex.

`compile_model` uses `nn.Module.compile` rather than `torch.compile`. The latter
hands back an `OptimizedModule` wrapper, and every attribute it forwards comes
back bound to the module it wrapped — so a method called on the wrapper runs on
the *uncompiled* model, and the `self(...)` inside it never reaches the compiled
graph. That is the whole call pattern here: the trainer drives
`model.run_epoch(...)`, which is three frames above the only forward call.
Compiling in place installs the graph on the model's own `__call__`, which every
one of those frames goes through. The return value is read off the model rather
than off the call succeeding, so the `compiled` tag on a run says the graph is
installed and not merely that nothing raised.

## Console logging

The library logs through `logging.getLogger(__name__)` and installs nothing on
the way in: importing `d3text` must not decide where anyone else's records go —
the same first-writer-wins hazard `runtime.configure` exists for. `configure` is
called from an entry point (`runtime.configure` does it for `train`, `tune` and
`evaluate`; the precompute commands call it themselves) and puts one handler on
the `d3text` logger with `propagate = False`, so the root logger and any
configuration the importing application already has are left alone. Calling it
twice replaces the handler rather than doubling every line.

`d3text/__init__.py`'s two missing-dependency notices stay bare `print`s on
purpose: they fire while the package is being imported, before any entry point
could have configured a handler, so a logger would drop them.

**The handler writes through `tqdm.write`.** A plain stream write lands in
whatever terminal line a live progress bar occupies and smears it, which is why
the training loop wrote its epoch numbers with `tqdm.write` in the first place;
routing them through `logging` had to keep that property, not trade it for a
verbosity knob. `TqdmLoggingHandler` resolves its stream at emit time rather
than storing it, so a handler installed before a stream is swapped — pytest's
capture, a redirect — still writes where stdout currently points.

`WritableStream` is narrower than `typing.TextIO`, which is a protocol wide
enough that `io.StringIO` does not satisfy it — and a stream a test can read
back is the only way to pin what the handler wrote.

`LevelPrefixFormatter` names the level of anything more urgent than INFO and
nothing else. INFO is the narration these commands printed verbatim before it
moved behind `logging`, so it has to keep printing verbatim; a warning that
looks exactly like narration is a warning nobody reads.

`D3TEXT_LOG_LEVEL` selects the verbosity, and an unparseable value falls back to
INFO rather than raising: losing a multi-hour run to a typo in a verbosity knob
would be a poor trade.

## Uncaught exceptions

`BaseException.add_note` is the obvious way to attach context to an exception
raised by someone else's code — pydantic's `ValidationError`, say — but
stackprinter renders only the traceback and the exception's own message. A note
attached anywhere in this package therefore reaches pytest and a plain `python
-c`, which use the stdlib hook, and is dropped from every console script, which
is the one path it was written for. `excepthook.with_notes` wraps the hook so
the notes follow the traceback.

## Progress bars

`TokenBudgetBatchSampler` deliberately has no `__len__` — how many batches an
epoch takes depends on the order the inner sampler draws — so `len(loader)`
raises, tqdm gets no total, and the bar degrades to a bare counter. The
*document* count of a split is fixed whatever the batching, so `batch_progress`
counts documents and carries the batch count as a postfix. `split_documents`
asks the dataset rather than the loader, and is defined once so the bar's
shortfall warning and the logged coverage metrics cannot disagree.

The bar can stop short of its total: a document whose pmid is missing from the
HDF5 file is dropped by `BrendaDataset._getitems` and never reaches a batch.
When *every* document a batch was drawn for is missing, the batch collates to
`[]`; that batch is dropped rather than yielded, because each of the six epoch
and evaluation loops would otherwise hand it to `ground_truth`, whose
`torch.concat(())` raises. `evaluate` loads with `batch_size=1`, so there one
missing pmid is one empty batch.

Dropping it is a skip, not a raise: a stale encodings file is exactly the
condition that produces this, and it must not cost a multi-hour run its
remaining hours. It is also not silent — the shortfall is logged once when the
pass ends, instead of once per batch or not at all. The shortfall and the
dropped batches are counted independently and reported as two messages:
`_getitems` drops a missing row on its own, so the usual shape of a stale
encodings file is a split that loses documents without any batch losing all of
them, and reporting the count only alongside a dropped batch would leave that
case silent.

## Experiment tracking

Every entry point in `d3text.tracking` is a **no-op unless
`MLFLOW_TRACKING_URI` is set**, so importing the module — or calling it from the
training loop — changes nothing for tests, notebooks, or a run on a machine with
no tracking server:

```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000   # must be http(s)
```

The variable, rather than a config key, is what selects tracking because the
tracking server is a property of the *machine* the run happens on, exactly like
the torch flavour — the same `config.toml` has to work on the VM that has a
server and on the laptop that does not. It has to name an `http(s)://` server:
the dependency is `mlflow-skinny`, which ships no local store backend.

The module is a **leaf** but for `d3text.metric_docs`, which is itself one;
`mlflow` is imported only on first use, and `torch` only inside
`environment_tags`. That is what lets the models log without dragging a tracking
client into every import of the package.

**Tracking never propagates a failure into the run.** A server that is down, an
expired token, or a client too old for the API disables tracking for the rest of
the process with a single warning; a multi-hour training run must not die
because a metric could not be posted. A run is closed as `FAILED` when its block
raises, so a crashed training run is distinguishable in the UI from one that
merely stopped early — and the exception is re-raised untouched either way.

### Provenance

`git_commit` returns the short hash, `-dirty` if the tree was edited, and `None`
when the answer would be a guess: no git, no repository (a non-editable install
into site-packages), or a detached/empty HEAD. The dirty check is `git diff
--quiet HEAD`, which compares **tracked** files only. `git status --porcelain`
would be wrong: `CLAUDE.md`, `design/` and `ncbitax/` live in the tree untracked
and un-ignored on purpose, so it would report every run as dirty and the flag
would stop meaning anything.

The commit goes into the run *name* as well as the tags, because the name is the
only column always visible in a run list — scanning a sweep for "which of these
ran before the pooling change" should not need a click per run. It also suffixes
the default experiment name, so runs from different code auto-namespace into
different experiments rather than piling into one; setting
`MLFLOW_EXPERIMENT_NAME` overrides that outright, for a sweep that wants every
trial in one place regardless of commit.

`provenance_tags` records the model and base model as *tags* even though both
are already in the params via `ModelConfig.model_dump()`: a param is one click
deep, and these are the questions asked while *scanning* a run list, so they go
where they can be shown as columns and filtered on (`tags.model =
"ETEBrendaModel"`).

`environment_tags` records the machine and torch build. A sweep is normally
spread over the machines that were free — a P100 VM, an RTX Ada box, a laptop on
CPU — and the accelerator is what explains a run that is three times slower, or
that differs numerically, from the run beside it. `torch.__version__` carries
the flavour suffix (`+cu128`, `+rocm…`, bare for CPU), which is the same thing
`TORCH_FLAVOUR` selected at lock time.

### The metric glossary

MLflow charts a metric under its key and nothing else: there is no place in the
API to record a unit, a direction, or the denominator an average was taken over.
A key like `training/class` therefore leaves the reader to guess whether the
axis is a loss, an F1, or a count, and `batches_per_second` leaves open which
pass it timed.

Two things close that gap: the keys are written to say what they are
(`loss_`-prefixed, `epoch_seconds`), and `d3text.metric_docs` renders the
glossary that `tracking.run` posts as the run's description, where the UI shows
it above the charts — the only free-text field the UI shows on the run page
itself, written as the `mlflow.note.content` tag.

`describe` returning `None` is the thing worth catching in a test: it means a
metric reaches the tracking server with no record anywhere of what its y-axis
measures, which is the state the module exists to end. The module is a leaf — no
imports from `d3text`, none from mlflow — so the glossary can be rendered,
tested or printed without a tracking server or a model in the process.

A per-class table is not a metric: it has one row per label and is read whole,
once, when a micro-average turns out to hide something. `log_text` writes it
beside the metrics so the run stays self-contained, rather than in a terminal
scrollback that outlives nothing.

::: d3text.runtime

::: d3text.logs

::: d3text.excepthook

::: d3text.progress

::: d3text.tracking

::: d3text.metric_docs
