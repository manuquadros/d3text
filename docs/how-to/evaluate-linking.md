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

## Rebuild a bridge table

The bridge tables are tracked in git; rebuild one after the BRENDA dump or
the outside resource it pairs against changes, and commit the result.
`$DATA` below is the directory `pull_data.py` printed.

```bash
pdm run python scripts/build_organism_taxid_bridge.py \
    "$DATA/documents.json" data/organism_taxids.tsv \
    "$DATA/training_data.csv" "$DATA/validation_data.csv" "$DATA/test_data.csv"
pdm run python scripts/build_enzyme_ec_bridge.py \
    "$DATA/documents.json" data/enzyme_ec_numbers.tsv
pdm run python scripts/build_strain_number_bridge.py \
    "$DATA/documents.json" data/strain_numbers.tsv
```

The organism bridge needs the NCBI taxonomy dump, which `ncbitax` downloads
on first lookup unless `NCBITAX_AUTO_DOWNLOAD=0` is set. It takes all three
splits because an other organism is named only in their inline column: one
missing from every split cannot be paired. The enzyme and strain bridges are
identifier joins over the dump alone. Each prints how much of its population
it paired.

## Score one corpus on its own

Each corpus has a standalone scorer. None needs a checkpoint or the BRENDA
SQL database:

```bash
pdm run python scripts/score_species_linking.py \
    "$DATA/documents.json" data/organism_taxids.tsv ~/corpora/Species-800 \
    "$DATA/training_data.csv" "$DATA/validation_data.csv" "$DATA/test_data.csv"
pdm run python scripts/score_enzyme_linking.py \
    "$DATA/documents.json" data/enzyme_ec_numbers.tsv \
    ~/corpora/enzymeNER ~/corpora/expasy-enzyme/enzyme.dat
pdm run python scripts/score_strain_linking.py \
    "$DATA/documents.json" data/strain_numbers.tsv \
    ~/corpora/nlp4pheno/export.json
```

The species scorer prints three reports — bacteria, other organisms, then
both together — and needs the splits for the organism bridge's reason: an
index built without them holds no other-organism form, and the linker would
answer NIL to every such span. The strain scorer also prints how many judged
spans spell the accession the way the index holds it, and the score over the
rest when any are left.

## Scoring a tagger's own spans

The block above scores the linker on the annotators' spans, so a detection
miss never reaches it. `d3text.linking_eval.score_predicted_linking` scores
the same corpora through a tagger's predicted spans instead (the
`test/predicted_linking_<namespace>_*` keys), and `precompute-encodings`
can encode a corpus into the store for that purpose (`--s800`,
`--enzymener`). `evaluate` runs that path after the gold-span block, for a
checkpoint with a span tagger whose encodings store holds those corpora.

A store missing the corpus entirely, or holding only some of its documents,
logs no `test/predicted_linking_*` metric for that corpus rather than
scoring it: a document the store never encoded is not a detection the
tagger missed, and would otherwise be counted as one. Encode the corpus
fully with `precompute-encodings --s800`/`--enzymener` first.
