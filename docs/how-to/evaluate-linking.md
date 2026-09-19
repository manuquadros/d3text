# Score the linker against external corpora

Goal: the `test/linking_*` blocks in an evaluation — the dictionary linker's
accuracy against identifiers assigned by an outside authority, per entity
type.

This block measures the surface-form index, not the checkpoint: the linker
has no learned parameters, so the numbers move only when the index does.
The corpora are downloads, so the block is optional; an evaluation without
them skips it and finishes.

## 1. Download the corpora

Put each under one directory, in the layout `d3text.linking_corpora`
expects (the constants there are the source of these names):

| Corpus | Path under the root | Provides |
| --- | --- | --- |
| Species-800 | `Species-800/S800.tsv` and `Species-800/abstracts/` | Species spans with NCBI taxids, for bacteria and other organisms |
| enzymeNER | `enzymeNER/GoldSet.txt` and `enzymeNER/GoldSetAnnot.txt` | Enzyme spans, unnamed |
| ENZYME nomenclature | `expasy-enzyme/enzyme.dat` | EC numbers for enzymeNER's spans. Without it enzymeNER is skipped: the corpus names no identifiers |
| NLP4Pheno | `nlp4pheno/export.json` | Strain spans carrying culture-collection numbers |

The first three are found under their publisher's own filenames. NLP4Pheno
publishes several dated Label Studio exports that do not annotate the same
spans, so the block reads exactly `nlp4pheno/export.json` and nothing else:
copy or symlink the export you chose to that name, and record which one in
the run's notes. An `nlp4pheno/` directory without that file is warned
about.

## 2. Make sure the BRENDA inputs are present

The block builds the surface-form index from `documents.json` and all three
splits, and checks each against `SHA256SUMS` first. A missing or mismatched
file skips the block with a warning; run
[`pull_data.py --check`](fetch-the-data.md) if that happens.

The bridge tables joining BRENDA entities to outside identifiers —
`data/organism_taxids.tsv`, `data/enzyme_ec_numbers.tsv`,
`data/strain_numbers.tsv` — are tracked in git. Rebuilding one needs the
outside resource (an NCBI taxonomy dump, a registry API) and is done by the
matching `scripts/build_*_bridge.py`.

## 3. Name the root in `config.toml`

```toml
linking_corpora = "~/corpora"
```

## 4. Evaluate

```bash
pdm run evaluate <config.toml> <checkpoint.pt>
```

The linking blocks print last, after every other metric, and are logged
under `test/linking_<namespace>_*` — one namespace per authority
(`ncbi_taxid`, `ec_number`, `strain_number`). Each block names the index
digest it was measured under and states its coverage: the accuracy is over
the spans whose identifier resolves to exactly one BRENDA entity, and the
dropped spans are the hard ones.

## Reading the numbers

- Read every accuracy beside its `_coverage`.
- The `strain_number` block is largely circular — the culture number the
  gold is keyed on is itself a form in the dictionary — and prints its own
  caveat; the standalone `scripts/score_strain_linking.py` separates the
  circular share from the rest.
- The `ec_number` block's high lenient accuracy is mostly shared IUBMB
  nomenclature between BRENDA and Expasy, not disambiguation.

Why the subset is chosen on the gold side, and what each corpus can and
cannot show: [Scoring linking against outside
identifiers](../explanation/evaluation.md#scoring-linking-against-outside-identifiers).

## Scoring a tagger's own spans

The block above scores the linker on the annotators' spans, so a detection
miss never reaches it. `d3text.linking_eval.score_predicted_linking` scores
the same corpora through a tagger's predicted spans instead (the
`test/predicted_linking_<namespace>_*` keys), and `precompute-encodings`
can encode a corpus into the store for that purpose (`--s800`,
`--enzymener`). `evaluate` does not yet run that path; a script has to
drive it.
