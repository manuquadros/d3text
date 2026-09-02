# FEAT-10's re-measurement on the VM — one command

```bash
cd /vol/storage/dev/d3text   # the checkout on the VM
git pull                     # must include the abbreviated-genus other-organism forms
TMPDIR=~/.cache/pdm-tmp ~/.local/bin/pdm install -L locks/cu118.lock --frozen-lockfile
tmux new -s feat10 'bash scripts/feat10_recall/vm/run.sh 2>&1 | tee vm-run.log'
```

Roughly **six hours** if the earlier embeddings store is still on the volume,
and considerably longer if it is not. It leaves
`/vol/storage/feat10-vm-<date>.tar.gz` — that tarball is what to send back.

`Ctrl-B` then `d` detaches; `tmux attach -t feat10` picks it up again.

**The label store is rebuilt, not reused, and the rebuild is the long pole that
depends on nothing else.** `8cb932b` made the store record a digest of the
surface-form index it was labelled against and refuse a mismatch, so every
store on that volume is now refused on open; `eb3addc` then changed what that
index holds for `other_organisms`, which is the one class this run is about.
`FEAT10_UNTIL=token_labels` runs exactly that stage — start it before the arms
are settled, since nothing about the comparison changes it.

## What question this answers

Two changes have landed against stage 1 and neither has been measured on real
data.

`c9d4ba2` gave `masked_token_cross_entropy` the `balanced` and `focal`
weightings the relation head already had, which is the recall lever FEAT-10
found missing — the tagger decides a token's type by a plain argmax over a
loss whose majority class is 91.2% of kept tokens, so it defaults toward
`OUTSIDE`. It is fixture-tested on a synthetic imbalance and nothing has run it
over the corpus.

`eb3addc` wrapped `other_organism_forms` in `with_abbreviated_genus`, so
`C. albicans` now names what `E. coli` already did. That was measured against
S800's spans through the *linker*; its effect on the *tagger* — the 14.5%
recall that makes `other_organisms` the worst of the four types — is unmeasured
because it changes the training targets, not just the matcher.

One run covers both, and produces the per-type detection recall FEAT-01 is
blocked on at a commit that can be cited. The number it replaces, 42.7%, was
stamped `b99ade7-dirty`.

## What it does, in order

| Stage | What | Roughly |
|---|---|---|
| `preflight` | DEC-04's, called with this run's paths. **Runs every time, never stamped** | seconds |
| `token_labels` | `precompute-token-labels` over the three splits and the noise pool, under format 3 | ~40–60 min |
| `audit` | that the designation guard took, and the realised label distribution. **Stops the run** | ~2 min |
| `configs` | one config per arm, and a check that any two differ in exactly one line | instant |
| `configure` | points `config.toml` at the embeddings store if it is still there | instant |
| `smoke` | 20 documents, purely to prove the labels are being *read* | ~5 min |
| `train_<arm>` / `eval_<arm>` | full training split, 6 epochs, then `evaluate` — once per arm | ~1.6 h each |
| `compare` | the three arms' detection scores side by side | instant |
| `bundle` | tars up every log, json and timing | seconds |

**It resumes.** Each finished stage stamps `out/stamps/`; rerunning skips
those, and `precompute-token-labels` skips documents it already holds, so even
a half-built store picks up where it left off. `FEAT10_FORCE=1` reruns
everything.

`evaluate` runs through `evaluate_json.py`, which wraps nothing but the metric
logging: the per-type detection numbers reach MLflow and the returned dict, and
the console carries only the three overall ones, so a machine with no tracking
server would otherwise score the arms and keep none of the answer.

## The arms

Three, differing in one config line each:

| Arm | `token_loss_weighting` | |
|---|---|---|
| `unweighted` | `"unweighted"` | the current default, and the fresh baseline |
| `balanced` | `"balanced"` | per-batch inverse frequency |
| `focal` | `"focal"` | `(1 − p_t)**token_focal_gamma · CE`, γ at its 2.0 default |

The unweighted arm is **not** redundant with FEAT-06's published 42.7%: the
label store underneath it has just been replaced, so a comparison against that
number would confound the weighting with the targets. `FEAT10_ARMS="unweighted
balanced"` drops the third and about ninety minutes with it.

γ is left at its default rather than swept. A sweep needs the three-arm result
first: if `focal` at 2.0 does not move recall, the γ that would is not the next
question.

## Reading the result

`out/arms.log` holds the table; `out/arms.json` the merged metrics. It is a
table and not a verdict, because FEAT-10's question is a tradeoff: a recall
lever that buys 10 points of recall for 15 of precision has answered the
question and not settled it. Three things to read together —

- **Per type, not just overall.** The four types started 5× apart, and a
  weighting that lifts the mean by lifting `bacteria` further has not addressed
  the class the ticket names.
- **The document heads.** The tagger shares a trunk with the entity, class and
  relation heads, so the last table says what the lever cost the rest of the
  model.
- **What the scores are against.** Detection is scored against the distant
  labels the arms train on, so it measures agreement with the matcher, and is
  blind to entities BRENDA does not carry.

## Knobs

| Variable | Default | Why you would change it |
|---|---|---|
| `FEAT10_VOL` | `/vol/storage`, or `$HOME` | The volume the label store and the tarball go on. |
| `FEAT10_LABELS` | `$FEAT10_VOL/d3text-token-labels-fmt3.hdf5` | The token-label store. A new name, since the old file is refused rather than upgraded. |
| `FEAT10_STORE` | `$FEAT10_VOL/d3text-embeddings` | The precomputed embeddings. Absent, the run works and is hours slower. |
| `FEAT10_ENCODINGS` | `data/biolinkbert-base-zstd-22-encodings.hdf5` | Preflight cross-checks tokenization against it. |
| `FEAT10_ARMS` | `unweighted balanced focal` | Fewer arms, or one at a time. |
| `FEAT10_UNTIL` | unset | Run up to and including this stage, then hold. `FEAT10_UNTIL=token_labels` is the hour that depends on nothing. |
| `FEAT10_OUT` | `scripts/feat10_recall/vm/out` | Where logs and results collect. |
| `FEAT10_BUNDLE` | `$FEAT10_VOL/feat10-vm-<date>.tar.gz` | Where the tarball lands. |
| `FEAT10_AUDIT_DOCS` | 400 | Documents the audit samples. |
| `FEAT10_SMOKE_DOCS` | 20 | Documents the smoke run trains on. |
| `FEAT10_FORCE` | unset | Rerun every stage. Your original `config.toml` backup survives it. |
| `FEAT10_PDM` | `~/.local/bin/pdm` | |

There is no base-model knob and no `--limit`, for the reasons DEC-04's runner
gives: the labels are placed by re-tokenizing with the base model's tokenizer,
and `--limit` selects the entity vocabulary, so an arm truncated differently is
a different model rather than a smaller run.

## If a stage fails

Everything is in `out/`. The likely ones:

- **`token_labels` refusing to open the store** names the invocation that
  replaces it. That is format 3 working; point `FEAT10_LABELS` at a new file
  rather than deleting the old one, which is still the store the published
  numbers were measured against.
- **`audit`** means the index was built unguarded, or `COMMON_WORD_ZIPF` moved.
  `out/label_audit.log` names which words leaked and which were lost.
- **`smoke`** fails two ways and says which: the training run crashed, or no
  document in the split had token labels — which means the store was built over
  other splits than the arms train on.
- **`smoke` crashing with `AssertionError: Guard failed on the same frame it
  was created`** is neither, and is not this run's doing. It is `torch.compile`
  tracing `jaxtyping`'s `__instancecheck__`. The P100 does not hit it —
  compute capability 6.0 cannot host Triton, so compilation is skipped — and on
  a card that can, `D3TEXT_DISABLE_COMPILE=1` is the switch (`965e588`), which
  changes what is measured only by removing a compilation step.
