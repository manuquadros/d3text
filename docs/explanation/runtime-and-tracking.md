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

The seed is part of a run's identity, not an implementation detail, so it is
a `ModelConfig` field rather than only `configure`'s default (42): the number
a run used is then in the config it was launched from and in the params
MLflow records, and a sweep can vary it, since the spread over seeds is what
says whether two configurations differ at all.

That is why `train` and `evaluate` call `configure` *after* reading their
config rather than as the first statement of `main()`: parsing arguments and
reading a TOML file touch no device, so the allocator rule above is still
satisfied. `tune` configures once for the sweep and then calls
`runtime.set_seed` per trial — otherwise each trial starts from whatever RNG
state the trial before it left, and a configuration's score depends on where
in the sweep it happened to be drawn.

### Two traps in the config keys

`torch.backends.cuda.matmul.allow_tf32` aliases the matmul precision itself
(`True` ↔ `"high"`, `False` ↔ `"highest"`) rather than naming an independent
knob, so writing both `float32_matmul_precision` and this flag leaves
whichever write ran last in effect. Only cuDNN's TF32 flag
(`cudnn_allow_tf32`) is genuinely separate.

Each backend also reads only its own allocator variable: a CUDA build
ignores `PYTORCH_HIP_ALLOC_CONF` entirely, and `configure()` picks the
variable matching the installed torch build and calls `setdefault` on it, so
a value already set in the environment wins over `config.toml`.

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

Compiling is opt-in, through `D3TEXT_COMPILE`, because it has not paid for
this model. Measured with the top of the trunk training, the compiled run's
first epoch took about six times as long as eager, and later epochs were under
one percent faster — hundreds of epochs to recover the warmup, against runs
that stop well short of that. Much of the warmup is dynamo recompiling code
whose shapes change every batch, and the steady-state gain is small either way.

`train` and `tune` compile the trunk's trainable top encoder layers, not the
whole model: `Model.compile_trunk` is what `D3TEXT_COMPILE` gates, not a
`model.compile()` on the model `train`/`tune` build. Most of a training step
is orchestration — pooling, chunked reductions, the relation heads — that
dynamo would otherwise trace and guard on Python values (a batch's document
count, a list of gold relations) that change every step, hitting the
eight-recompile budget within a couple of dozen steps and running eager for
the rest of the process. The trunk is the one part with regular shapes and
heavy GPU work, and it is reached by two different routes — the full
forward in `Model._embed_missing`, a `BertEncoder` call passing kwargs, and
the cached-prefix replay in `Model._replay_top_layers`, a bare `for layer in
top_layers` loop passing positionals. Compiling each `encoder.layer` module
in place would leave both routes sharing `BertLayer.forward`'s one code
object, so dynamo would guard on every way the two routes and the
frozen/trainable layers differ — a parameter's `requires_grad`, the
attention-mask convention, kwargs vs positional, `hidden_states` dtype — and
exceed the eight-recompile budget within two epochs. Routing both paths
through one wrapper, `Model._trunk_top`, instead gives `compile_trunk` a
single function to guard. `unfrozen_top_layers=0` (the default) builds no
wrapper, so `compile_trunk` is a no-op then: that trunk runs under
`no_grad`, or is answered from the CPU cache or a precomputed store, none of
which has a recompile to save.

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

`runtime.compile_model` uses `nn.Module.compile` rather than `torch.compile`.
The latter hands back an `OptimizedModule` wrapper, and every attribute it
forwards comes back bound to the module it wrapped — so a method called on
the wrapper runs on the *uncompiled* module, and a `self(...)` inside it
never reaches the compiled graph. Compiling in place installs the graph on
the module's own `__call__` instead, which is why `Model.compile_trunk`
calls `runtime.compile_model` on `_trunk_top`, a real (if parameter-free)
`nn.Module`, rather than compiling a bare function: both of `_trunk_top`'s
callers — `_embed_missing` and `_replay_top_layers` — reach it through a
plain Python call, `self._trunk_top(...)`, which is `__call__` underneath.

Installing the graph is all that call does. The backend is not asked for a
kernel until the first forward, and under `dynamic=True` it is asked again at
every recompile — inside the training loop, past the `try` the compile is
wrapped in, so an inductor failure there killed the run at epoch 0 and left a
Triton-capable machine *less* able to train than one that had to stay eager.
`_install_eager_fallback` wraps the installed call so a dynamo exception drops
the wrapper back to eager and re-runs the call there; only dynamo's own
exceptions are caught, because those mean the compile failed rather than the
model, and anything the model itself raises has to keep propagating.

That wrapper can only guard what passes through `__call__`, and half the graph
does not: AOTAutograd compiles the backward separately, and lowers it at the
first `loss.backward()` — a call into the autograd engine, not into the model.
Torch does attempt that lowering while the forward is compiling, but it
suppresses a failure there and retries it lazily, so a backend error on the
backward graph surfaced from `backward()` as a bare `RuntimeError` and killed
the run exactly as the forward case used to.
`_compile_the_backward_with_the_forward` makes the first attempt the only one,
leaving a single guarded point at which either half can fail — and it fails
before there is a loss, so nothing has to unwind a half-taken optimizer step.

That fallback clears the graph, which is what keeps the `compiled` tag
truthful. `compile_trunk`'s return value is read off `_trunk_top` rather than
off the call succeeding, but it can still only report what was *installed*,
so `train` and `tune` read `Model.trunk_is_compiled` again once training is
over and set the tag from that — the tag then says what the epochs executed.
Both do it from a `finally`, so the retag happens however the epochs ended: a
run that died is exactly the one someone later filters for when asking
whether the compiler was implicated.

## Console logging

The library logs through `logging.getLogger(__name__)` and installs nothing on
the way in: importing `d3text` must not decide where anyone else's records go —
the same first-writer-wins hazard `runtime.configure` exists for. `configure` is
called from an entry point (`runtime.configure` does it for `train`, `tune` and
`evaluate`; the precompute commands call it themselves) and puts one handler on
the `d3text` logger, and on the `brenda_references` logger, each with
`propagate = False`, so the root logger and any configuration the importing
application already has are left alone. `brenda_references` is routed
alongside `d3text` rather than left to fend for itself, because it is a
production dependency on the `train`/`evaluate`/`tune` import path and its
modules log under their own `__name__` rather than naming `d3text` — nothing
in the dependency itself decides where its records go. Calling `configure`
twice replaces both handlers rather than doubling every line.

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

**Yield from a progress bar; never return one.** beartype deep-checks a return
value annotated `Iterable[T]` by taking one item off it to inspect. That is
free for a list, and a bare iterator is skipped outright, but a `tqdm` is
`Sized` and its `__iter__` consumes whatever it wraps, so the item beartype
sampled never reaches the caller: one element vanishes per call, always the
first, with nothing raised and no count to notice it by. Measured on beartype
0.22.9. `negative_screen._limited` yields from its bar for that reason, and is
annotated `Iterator[T]` — not `Sized`, so it buys no deep check, and a `tqdm`
is not an instance of one, so a rewrite that hands the bar back is rejected
loudly instead of shipping one document fewer.

## Experiment tracking

Every entry point in `d3text.tracking` is a **no-op unless
`MLFLOW_TRACKING_URI` is set**, so importing the module — or calling it from the
training loop — changes nothing for tests, notebooks, or a run on a machine with
no tracking server.

The variable, rather than a config key, is what selects tracking because the
tracking server is a property of the *machine* the run happens on, exactly like
the torch flavour — the same `config.toml` has to work on the VM that has a
server and on the laptop that does not. It has to name an `http(s)://` server:
the dependency is `mlflow-skinny`, which ships no local store backend.

### Why the keys say what they are

The keys a run logs are listed in [the metric
reference](../reference/metrics.md). Two of them exist only to make a curve
interpretable: `loss_weight/*` is [the ramp weight](models.md#the-relation-loss-ramp)
each objective trained under that epoch, and `training/grad_clip_rate` is
[the clip rate](cli-and-training.md#the-weight-update) — without them, a loss
curve that bends because the schedule moved cannot be told from one that bends
because the model changed. `epochs_after_best` separates a converged run from
one still improving when `num_epochs` ran out: zero means the last epoch was
the best.

Adding a metric means adding its glossary entry:
`tests/training/test_trainer.py` drives a real `fit` and fails on any logged
key `metric_docs.describe` can't resolve, so an undocumented one can't land
quietly. **Renamed keys don't back-fill** — a run logged before a rename
keeps the old name, so a chart spanning both eras needs both.

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
into site-packages), or an empty HEAD. A detached HEAD still identifies the
code exactly, so it is stamped like any other. The dirty check is `git diff
--quiet HEAD`, which compares **tracked** files only. `git status --porcelain`
would be wrong: a checkout routinely holds untracked, un-ignored files — a
local `config.toml`, downloaded data, editor state — so it would report every
run as dirty and the flag would stop meaning anything.

`git_describe` answers the other half. `git_commit` says which code exactly;
`git_describe` says which release that code descends from — `v0.2.0` on a
tagged commit, `v0.2.0-12-gabc1234` twelve commits later — which is the form
a paper can cite and a reader can type. It matches against `v[0-9]*`, so a
tag that is not a release cannot become the anchor, and it is `None` when
there is no release tag to describe against.

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
the flavour suffix (`+cu130`, `+cu128`, bare for CPU), which is the same thing
`TORCH_FLAVOUR` selected at lock time.

It also records the `config.toml` settings the run was launched under.
Those are per-machine and deliberately uncommitted, which is exactly why the
run has to carry them: nothing else ever writes down that this one ran with
`float32_matmul_precision = "medium"`, and that key alone is the difference
between fp32 and bf16 arithmetic — enough for two runs at one commit, on one
config, to disagree. `embeddings_store` and `linking_corpora` go in as
whether they were set rather than as their paths: what reproduces a run is
that embeddings came from a store at all, and a path is this machine's
directory layout, not provenance.

A store being *configured* is not a store having *served* the run: one that
cannot be opened, or that another model wrote, disables itself and the run
recomputes. So `run` stamps `embeddings_store_lookups` and
`embeddings_store_coverage` as it closes — how many documents were looked up
in a store, and what share of them came back from one. A run off the store
and a run that recomputed are
[not numerically comparable](data.md#a-stored-embedding-and-a-live-one-are-not-the-same-number),
so that share is what says whether two runs' numbers may be put beside each
other at all; zero lookups is a run that computed every embedding itself.

The counters belong to the reader, which is cached for the life of the
process and never reset, so each run carries the *difference* across its own
scope: `tuning` opens a run per trial in one process, and raw totals would
make every trial's number include the trials before it. They are written
before the run is closed because the reader's own summary is logged at
process exit, by which time the last run is long gone.

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
