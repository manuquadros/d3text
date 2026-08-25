# brenda_references

The BRENDA reference corpus: the training / validation / test DataFrames and
the relation preprocessing behind them. `d3text` consumes this as a library.

## Running the scripts

`scripts/` holds the ad-hoc data-collection drivers that build and repair the
document database — pulling references out of the BRENDA mirror, fetching
abstracts and full texts from NCBI, resolving strains, reporting corpus
statistics. They are not part of the distribution: this is a src-layout
package, so the wheel carries `src/brenda_references` and nothing else. Run
them from this directory, where their own package is importable:

```bash
python -m scripts.statistics
```

`-m` rather than a path so that `scripts/__init__.py` runs and installs
beartype's import hook, which is where these scripts get their runtime type
checking from.

They expect the BRENDA connection in the environment (`BRENDA_HOST`,
`BRENDA_USER`, `BRENDA_PASSWORD`) and the document database named by
`src/brenda_references/config.toml`. Several of them still import modules that
have since moved or been deleted, so treat them as a record of how the corpus
was assembled rather than as maintained tooling.
