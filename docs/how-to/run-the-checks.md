# Run the checks

Goal: the same verdict CI gives, before a change is pushed. All commands run
from the repository root; `pdm run` is required because the tools live in
the project's virtual environment.

## Tests

```bash
pdm run pytest tests/                              # everything but integration tests
pdm run pytest tests/test_utils.py::test_name      # one test
pdm run pytest -m integration                      # needs the full data files, network or a GPU
```

`-m 'not integration'` is the default (`pyproject.toml`). `gpu`-marked
tests run wherever a CUDA device is present and skip otherwise; to force the
skip path, `CUDA_VISIBLE_DEVICES="" pdm run pytest tests/`.

`tests/test_data.py` needs the full BRENDA data files and runs for over 30
seconds; it is integration-only.

The data layer's own tests live beside it:
`pdm run pytest brenda_references/tests/`.

## Lint, format and type-check

```bash
pdm run ruff check src/ scripts/ tests/
pdm run ruff format src/ scripts/ tests/
pdm run mypy src/
```

`mypy src/` reports `Success: no issues found`; any error is new. Both
commands are repository-root-relative and never see `brenda_references/`;
run them against that directory explicitly when it changes.

`ruff` is pinned exactly in `pyproject.toml` because its verdict moves with
its version; moving the pin is a change of its own.

## Documentation

```bash
pdm run mkdocs build --strict
```

The only check the docs site gets. Fails on a link that resolves to nothing,
a docstring field list griffe cannot parse, or a `:::` block naming a
module that does not import. Run it after touching a docstring or a page.

`pdm run mkdocs serve` previews the site at `http://127.0.0.1:8000`.

## Lockfiles

```bash
pdm run check-locks
```

Asserts all four lockfiles are current against `pyproject.toml` and
`brenda_references/pyproject.toml`. CI runs the CPU one; see
[Change a dependency](change-a-dependency.md).

## Runtime type checks

`beartype` (a development dependency) wraps every annotated function in the
package at import, so a call violating an annotation raises in tests and in
any environment that has it installed. Production installs without it run
unchecked.
