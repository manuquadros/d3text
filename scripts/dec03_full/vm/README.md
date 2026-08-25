# The pooling experiment on the VM — one command

```bash
cd /vol/storage/dev/d3text   # the checkout on the VM
git pull                     # must include the embeddings-store reader
tmux new -s dec03 'bash scripts/dec03_full/vm/run.sh 2>&1 | tee vm-run.log'
```

That is the whole thing. It takes roughly **five hours** and leaves
`/vol/storage/dec03-vm-<date>.tar.gz` — that tarball is what to send back.

`Ctrl-B` then `d` detaches and leaves it running; `tmux attach -t dec03` picks
it up again, from a second ssh session if the first one dropped. **tmux rather
than `nohup … &`**: five hours over ssh outlives at least one connection, and a
tmux window can be watched live and interrupted, where a backgrounded job can
only be tailed. The session closes when the run ends, which is what the `tee`
is for — `vm-run.log` holds the same output whether anyone was attached or not.
On a machine with no tmux, `nohup bash scripts/dec03_full/vm/run.sh >
vm-run.log 2>&1 &` is the same run without the reattaching.

**Everything large goes on `/vol/storage`, not in `$HOME`:** the ~101 GiB
store, and the tarball. `$HOME` on that machine is not where the checkout
lives and is not sized for this. The script says all four paths in its first
four lines, so `head -4 vm-run.log` is enough to check before walking away.
Anywhere without a `/vol/storage`, they fall back to `$HOME` and the run works
unchanged — this is a default, not a requirement.

## What it does, in order

| Stage | What | Roughly |
|---|---|---|
| `preflight` | GPU, RAM, free disk, corpus files, encodings, and that this checkout *has* the store reader | seconds |
| `profile_card` | `bench_card.py` — base-model and entity-head throughput per dtype and per batch size on this GPU | ~3 min |
| `profile_build` | `profile_build.py` — where the store build's wall clock goes per document, plus what the volume's disk can do | ~4 min |
| `bench` | `bench_store.py` — is reading the store still cheaper than the forward on **this** card? It was 27.8× on the laptop and the margin narrows on a faster GPU. **Stops the run below `DEC03_BENCH_MIN`** | ~10 min |
| `build_store` | `precompute-embeddings` over all three splits and the noise pool | ~2 h, **~101 GiB** |
| `coverage` | how many of each split's documents the store actually holds. **Stops the run below `DEC03_MIN_COVERAGE`** | ~1 min |
| `configure` | writes `config.toml` to point at the store (the old one is kept in `out/config.toml.before`) | instant |
| `smoke` | 2 epochs over 20 documents, purely to prove the store is being *read* and that the scorer and the step profiler run | ~5 min |
| `profile_step` | `profile_step.py` — peak VRAM and a per-phase time breakdown of a training step at several chunk budgets, with the store serving | ~10 min |
| `train_*` / `score_*` | the two arms, full training split, 6 epochs each, then per-class document scores on val and test | ~2 h |
| `bundle` | tars up every log, json, and timing | seconds |

## The three profile stages

**They change nothing about the arms.** No config value moves, nothing in
`src/` is edited, and the instrumentation is monkeypatched inside the profiler
rather than added to the library — so this run answers the pooling question
exactly as it would have, and costs about twenty minutes more. What they add is
the evidence for how the *next* run should be configured, because four of the
current settings are guesses that the logs cannot check:

| What it measures | What it decides |
|---|---|
| fp32 vs fp16 vs bf16, for the base model and for a `[T,128]@[128,6862]` head GEMM | whether `Model.amp_dtype`'s bf16 is costing the arms. `torch.cuda.is_bf16_supported()` answers for *emulation* as well as hardware, so on a pre-Ampere card it says yes to a format the card has no units for — while `precompute-embeddings` hardcodes fp16 |
| tokens/s and peak VRAM against windows per forward, 8 → 256 | whether a batcher that crosses documents is worth writing. The shipped one batches *within* a document and nothing here has more than 29 windows, so `DEC03_EMB_BATCH` never fires |
| the build's per-document phase breakdown, and the volume's O_DIRECT throughput | whether the build is GPU-bound as `perf_baseline.md` claims, or stalled on the unpinned D2H, the Python-loop aggregation and the synchronous corpus read — and whether the arms' store reads are minutes or hours. The throughput half writes a 1 GiB file to `$DEC03_VOL` and deletes it; the result is `out/volume_io.txt` |
| peak VRAM and a step's phases at `batch_max_chunks` ∈ 64…512, at the full 6862-column vocabulary | `batch_max_chunks`, and whether `num_workers` is now the binding constraint. The 512 ceiling in `perf_baseline.md` was measured at `--limit 200` and *before* the store removed the base-model forward that set the high-water mark; both corrections move it, in opposite directions |

**They are the only stages that do not stop the run.** A failure is logged,
stamped `FAILED`, and skipped past. The stages that do stop — `bench`,
`coverage`, `smoke` — protect the result, and past them lies five hours of
measuring the wrong thing. A profile protects nothing, so a typo in one must
not cost the run its remaining hours. `DEC03_SKIP_PROFILE=1` skips all three.

**It resumes.** Each finished stage drops a file in `out/stamps/`; rerunning
skips those. Kill it, reboot, rerun the same command — `precompute-embeddings`
itself also skips documents already stored, so even a half-built store picks up
where it left off. `DEC03_FORCE=1` reruns everything; your original
`config.toml` backup survives that, since `configure` only ever backs up a file
it did not write itself.

**It stops at the first failure** rather than continuing into stages that would
be measuring the wrong thing. Two of those stops are new and deliberate: a card
where the store no longer pays for itself, and a store that does not hold the
corpus. Both would otherwise be discovered five hours later, from a log.

**`out/` ignores itself.** It holds two checkpoints of a few hundred MB and
sits inside a tracked directory, so `scripts/dec03_full/.gitignore` covers it.
The script drops a `.gitignore` in it as well, because `DEC03_OUT` can point
anywhere and a redirected run lands outside the rule the repository carries.

## Knobs

| Variable | Default | Why you would change it |
|---|---|---|
| `DEC03_VOL` | `/vol/storage`, or `$HOME` if there is none | The volume the store and the tarball go on. |
| `DEC03_REPO` | the checkout this script is in | Normally leave it: the path is derived from the script's own location, so a checkout that moves takes the run with it. |
| `DEC03_STORE` | `$DEC03_VOL/d3text-embeddings` | Put the ~101 GiB somewhere with room. Deliberately outside the repo: `data/` is neither tracked nor ignored. |
| `DEC03_OUT` | `scripts/dec03_full/vm/out` | Where logs and results collect. |
| `DEC03_BUNDLE` | `$DEC03_VOL/dec03-vm-<date>.tar.gz` | Where the tarball lands. |
| `DEC03_EMB_BATCH` | 50 | Token windows per forward while building the store. Lower it if that stage OOMs. |
| `DEC03_UNTIL` | unset | Run up to and including this stage, then hold. `DEC03_UNTIL=coverage` builds and checks the store — two hours that depend on nothing about the model — while a decision about the arms is still open. Rerunning without it resumes at the first stage that never ran. |
| `DEC03_BENCH_MIN` | 3.0 | How much cheaper reading must be than recomputing before the run commits to two hours and 101 GiB. |
| `DEC03_MIN_COVERAGE` | 0.99 | How much of each split the store must hold. Lower it only after reading `out/store_coverage.json` and deciding the gap is real. |
| `DEC03_PROFILE_DOCS` | 150 | Documents the build profile times. Lower it to shorten that stage; below ~50 the windows-per-document spread stops being a sample. |
| `DEC03_PROFILE_BUDGETS` | `64,128,256,512` | `batch_max_chunks` values the step profile sweeps. 64 is what the arms run; the rest are there to find the card's real ceiling, and a budget that OOMs is recorded rather than raised. |
| `DEC03_SKIP_PROFILE` | unset | Skips all three profile stages. |
| `DEC03_PDM` | `~/.local/bin/pdm` | |

There is **no base-model knob.** It is read out of `cfg_logsumexp.toml`, and
the run refuses to start if the two arms disagree about it. A variable here
could only ever build a store keyed on one model's windows for a run that
trains another, and every document would then quietly miss.

`batch_max_chunks` is **not** a knob here either. It stays at the 64 the configs
carry, which is what the laptop's arm A ran at, so the two machines' runs stay
comparable; the P100 baseline found 512 to be its ceiling at 99.2% of the card,
and an OOM five hours in costs more than the throughput is worth.

## What the two arms are

`cfg_logsumexp.toml` and `cfg_logmeanexp.toml` differ in exactly one line —
`entity_logits_pooling` — and both run through `seeded_train.py`, so
initialization and batch order are shared. Six epochs, full training split, no
`--limit`: that last is the whole point, since every earlier measurement of this
question was made at `--limit 500`, where `noise=450` puts the split at 47%
noise instead of the corpus's own 4.8%.

Scoring is per-class **document** precision/recall/F1 at p ≥ 0.5 on validation
and test — not the pooled validation loss, which is what hid the collapse in
the first place: a channel that predicts nothing scores near-zero loss on the
75–83% of documents where its class is absent.

## If a stage fails

Everything is in `$OUT`. The likely ones:

- **`preflight`** names what is missing. Free disk and a checkout without the
  store reader are the two that stop it outright. It also prints `note:` lines
  for what this card silently ignores — a pre-Volta card gets no
  `torch.compile`, a pre-Ampere one gets nothing from
  `float32_matmul_precision` or `cudnn_allow_tf32` and reports bf16 support it
  does not have in hardware. None of that fails, and none of it appeared in a
  log before.
- **`profile_card` / `profile_build` / `profile_step`** do not stop the run;
  see above. Their logs are `out/profile_*.log` and their numbers
  `out/profile_*.json`.
- **`bench`** means reading the store is no longer enough cheaper than the
  forward to be worth building. The measurement is in `out/bench_store.log`;
  `DEC03_BENCH_MIN` is how to overrule it.
- **`coverage`** means the store does not hold the documents the run will ask
  for — the wrong corpus, the wrong key, or a build that stopped early and got
  stamped as finished. `out/store_coverage.json` names how many are missing per
  split and gives ten examples.
- **`smoke`** fails four ways, and says which: the training run itself crashed;
  the store was never opened (check `embeddings_store` in `config.toml`); it was
  opened but **served no document**; or it disagrees with the encodings about a
  document's token count. The last means the store was built with a different
  window than `precompute-encodings` used — rebuild it without passing
  `--max_length`.

A store that is not read is not a wrong number, it is a slow run: every
document it fails to answer for silently falls back to the base model, which is
exactly the cost the store exists to remove. That is why `smoke` reads the log
rather than trusting the exit code, and why the reader now logs a line the
first time it serves a document and a hit rate when the process ends — grep
`served .* of .* documents` in any run's log.
