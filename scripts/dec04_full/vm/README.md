# DEC-04's falsification test on the VM — one command

```bash
cd /vol/storage/dev/d3text   # the checkout on the VM
git pull                     # must include the designation guard
~/.local/bin/pdm install -L locks/cu118.lock --frozen-lockfile
tmux new -s dec04 'bash scripts/dec04_full/vm/run.sh 2>&1 | tee vm-run.log'
```

**The `pdm install` is not optional this time.** The designation guard needs a
general-English frequency table (`wordfreq`), which is a new dependency, and a
checkout that has the code but not the package builds the *unguarded* index —
reproducing exactly the mislabelling this run exists to measure away. Preflight
refuses to start without it, so a forgotten install costs seconds rather than
hours. All four lockfiles were regenerated for it.

Roughly **three to four hours** if the DEC-03 embeddings store is still on the
volume, and considerably longer if it is not. It leaves
`/vol/storage/dec04-vm-<date>.tar.gz` — that tarball is what to send back.

`Ctrl-B` then `d` detaches; `tmux attach -t dec04` picks it up again. On a
machine with no tmux, `nohup bash scripts/dec04_full/vm/run.sh > vm-run.log
2>&1 &` is the same run without the reattaching.

## What question this answers

DEC-04 measures that BRENDA's document-level class label is wrong on half the
rows of two classes: 51.7% of `bacteria`-negative documents name a bacterium
anyway, and about a third of the `other_organisms` negatives name an organism.
It holds three options, and this run tests the interim one.

**Option 3** — leave the document-level loss alone and let the token-level
objective carry the localization — makes a falsifiable prediction. DEC-02
measured the `other_organisms` channel scoring gold mention tokens *below*
ordinary prose, which is what the label noise predicts: a positive document
pushes up one token, a false-negative document pushes down all of them. If
token supervision supplies the localization the pooled loss cannot, that
inversion must disappear.

`lift` is the statistic — mean probability on gold mention tokens over mean
probability on background — and 1.0 is the line. DEC-02's `logmeanexp` arm put
`other_organisms` at **0.822**. The run prints its verdict at the end and
writes it to `out/verdict.json`; the three outcomes are *option 3 survives*,
*option 3 falsified*, and *premise absent* (the baseline showed no inversion,
so there was nothing to undo and the run cannot decide).

**Two arms, not one, and the baseline is not redundant** with DEC-02's
published numbers. Those were taken at `--limit 500`, where `noise=450` puts
the training split at 47% noise against the corpus's own 4.8%, and under a
pooling DEC-03 has since replaced. Comparing a new tagger arm against them
would confound three changes on the one channel under test.

It also produces **FEAT-06's detection recall** — the number FEAT-01 has been
blocked on — from `evaluate` on the tagger arm.

## What it does, in order

| Stage | What | Roughly |
|---|---|---|
| `preflight` | DEC-03's checks (GPU, disk, corpus, and that the encodings still tokenize to what the corpus reader produces), plus that this checkout *has* the designation guard. **Runs every time, never stamped** | seconds |
| `token_labels` | `precompute-token-labels` over the three splits and the noise pool — 12,399 documents | ~40 min |
| `audit` | that the guard actually took, and the realised label distribution. **Stops the run** | ~2 min |
| `tagger_config` | writes `out/cfg_tagger.toml`, and checks the two arms differ in exactly one line | instant |
| `configure` | points `config.toml` at the DEC-03 embeddings store if it is still there | instant |
| `smoke` | 20 documents, purely to prove the labels are being *read* | ~5 min |
| `train_baseline` / `probe_baseline` | full training split, 6 epochs, no token supervision | ~1.5 h |
| `train_tagger` / `probe_tagger` | the same, with the tagger head | ~1.5 h |
| `compare` | the verdict | instant |
| `detection` | `evaluate` on the tagger arm — FEAT-06's recall | ~10 min |
| `bundle` | tars up every log, json and timing | seconds |

The `audit` stage is the one worth understanding, because it exists to catch a
failure that is otherwise invisible. **The label store records its label space
but not the dictionary that filled it** ([BUG-60](../../../design/tickets/BUG-60.md)),
so a store built before the guard and one built after are indistinguishable
from the inside. Training on the stale one puts `sensitive` down as a strain
mention in a quarter of the corpus and reports nothing unusual. The audit
rebuilds the index and asserts that ten ordinary words reach no entity and that
eight real names still do — the second half mattering as much as the first,
since `escherichia` (Zipf 2.63) sits just under the cutoff and losing it would
cost most of the bacterial channel.

**It resumes.** Each finished stage stamps `out/stamps/`; rerunning skips
those. `precompute-token-labels` also skips documents it already holds, so even
a half-built store picks up where it left off. `DEC04_FORCE=1` reruns
everything.

**It stops at the first failure**, except `detection` — that answers a
different ticket, so it is logged rather than fatal, and a failure there does
not cost the verdict.

## Knobs

| Variable | Default | Why you would change it |
|---|---|---|
| `DEC04_VOL` | `/vol/storage`, or `$HOME` if there is none | The volume the label store and the tarball go on. |
| `DEC04_LABELS` | `$DEC04_VOL/d3text-token-labels.hdf5` | The token-label store. A few hundred MB. Outside the repo on purpose: `data/` is neither tracked nor ignored. |
| `DEC04_STORE` | `$DEC04_VOL/d3text-embeddings` | The DEC-03 embeddings store. If it is not there the run still works, hours slower. |
| `DEC04_ENCODINGS` | `data/biolinkbert-base-zstd-22-encodings.hdf5` | Only the probe uses it, to cross-check tokenization. |
| `DEC04_OUT` | `scripts/dec04_full/vm/out` | Where logs and results collect. |
| `DEC04_BUNDLE` | `$DEC04_VOL/dec04-vm-<date>.tar.gz` | Where the tarball lands. |
| `DEC04_UNTIL` | unset | Run up to and including this stage, then hold. `DEC04_UNTIL=audit` builds and checks the labels — 40 minutes that depend on nothing about the model — while a decision about the arms is still open. |
| `DEC04_PROBE_DOCS` | 200 | Validation documents each probe scores. |
| `DEC04_AUDIT_DOCS` | 400 | Documents the audit samples for the label distribution. |
| `DEC04_FORCE` | unset | Rerun every stage. Your original `config.toml` backup survives it. |
| `DEC04_PDM` | `~/.local/bin/pdm` | |

There is **no base-model knob**, for the reason the DEC-03 runner gives: the
labels are placed by re-tokenizing with that model's tokenizer, so a store
built under one model addresses another's encodings nowhere at all — and
misses silently, one masked document at a time. It is read out of
`cfg_baseline.toml`.

`--limit` is not a knob either. Both arms run the full training split, which is
the whole point: every earlier measurement of this question was made at
`--limit 500`.

## If a stage fails

Everything is in `out/`. The likely ones:

- **`preflight`** names what is missing. The new one is the designation guard;
  the fix is `pdm install -L locks/<flavour>.lock --frozen-lockfile`.
- **`audit`** means the label store was built with an unguarded dictionary, or
  `COMMON_WORD_ZIPF` moved. `out/label_audit.log` names which words leaked and
  which were lost. The fix is to rebuild: `rm out/stamps/token_labels` and
  rerun with `DEC04_FORCE=1`.
- **`smoke`** fails two ways and says which: the training run crashed, or no
  document in the split had token labels at all — which means the store was
  built over other splits than the arms train on.
- **`smoke` crashing with `AssertionError: Guard failed on the same frame it
  was created`** is neither of those and is not this run's doing. It is
  `torch.compile` tracing `jaxtyping`'s `__instancecheck__`, and it reproduces
  on the baseline config with no token labels anywhere in it. **The P100 does
  not hit it**: compute capability 6.0 cannot host Triton, so
  `runtime.is_triton_compatible()` skips compilation and the DEC-03 arms
  trained through this same stack. A card that *can* compile does hit it, and
  the workaround until it has a ticket of its own is `TORCHDYNAMO_DISABLE=1
  bash scripts/dec04_full/vm/run.sh` — which changes what is measured only by
  removing a compilation step, not a computation.
- **`compare`** reporting *premise absent* is not a failure. It means the
  baseline arm showed no anti-localization for the supervision to undo, so this
  run cannot test option 3's prediction. That is itself worth knowing: it would
  say the effect DEC-02 measured was a property of the pooling DEC-03 removed
  rather than of the label noise, and DEC-04's diagnosis would need revisiting.

## What this run does *not* settle

It tests option 3 and nothing else. If the verdict is *falsified*, the choice
between **option 1** (abstain at the document level for a class the text
matches) and **option 2** (down-weight rather than abstain) is still open, and
neither is implemented. What has landed is the guard option 1 was waiting on —
without it the abstention mask would inherit the same ordinary-English
designations and excuse a quarter of the strain negatives from the loss.

The detection recall it produces is measured against distant labels, so it
scores agreement with the matcher rather than correctness, and it is blind to
entities BRENDA does not carry. FEAT-06 says to report it as what it is.
