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

## Imports

```bash
pdm run check-imports
```

Imports the module behind each declared console script, one subprocess per
entry point, and names the ones that fail. None of the checks above performs
an import — ruff imports nothing, and mypy resolves circular imports
statically — so all three pass on a tree whose package raises `ImportError`
and whose every command is dead. The suite notices but does not report it:
the same broken import breaks collection of the test modules that would name
the failure, and pytest abandons the session on a collection error, leaving
dozens of identical tracebacks and no test result. CI's test job goes red for
the same reason, and just as illegibly.

One subprocess per entry point is the whole point — within a single process a
module already in `sys.modules` launders the import ordering that produces the
cycle. Takes about half a minute. The sources of the checkout it runs from
take precedence over anything installed, so it answers for the tree in front
of you rather than for the environment.

## Dead code

```bash
pdm run check-deadcode
```

Reports every function, class, method, attribute and variable in `src/` and
`scripts/` that nothing in `src/`, `scripts/` or `tests/` refers to, and
fails if there is one. Code that only a test calls counts as used. Settings
are in `[tool.deadcode]` in `pyproject.toml`.

Something only a framework calls — a pydantic validator, a
`logging.Handler.emit` override, an `ast.NodeTransformer.visit_*` method —
looks unused to it. Add its name to `ignore-names` there. The
`ignore-*-if-decorated-with` options would be the natural tool for the
first case, but the pinned deadcode version reads them and never applies
them, and it applies `ignore-definitions-if-inherits-from` only to class
attributes, not to methods.

The wrapper exists because the `deadcode` command exits 0 whatever it
finds, and because it reports nothing — rather than failing — when `only`
matches no file. `deadcode` is pinned exactly for the same reason as `ruff`.

## Documentation

```bash
pdm run mkdocs build --strict
```

The only check the docs site gets; CI runs it on every push. Fails on a
link that resolves to nothing,
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
