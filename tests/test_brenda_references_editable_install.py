"""`brenda_references` must run from the checkout, not a copied wheel.

pdm only installs a path dependency editable when it is also declared in a
dev dependency group; the production dependency alone installs a snapshot
copy into `site-packages`, silently freezing whatever `brenda_references/src/`
held at the last install. A snapshot inside pdm's default in-project
`.venv` still resolves under the repository root, so the check below has to
be narrower than "under the repo" -- it must be under the checkout's own
`brenda_references/src/`.
"""

import pathlib

import brenda_references

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_EDITABLE_SRC = _REPO_ROOT / "brenda_references" / "src"


def test_brenda_references_imports_from_the_editable_checkout() -> None:
    installed = pathlib.Path(brenda_references.__file__).resolve()

    assert installed.is_relative_to(_EDITABLE_SRC), (
        f"brenda_references imported from {installed}, not from "
        f"{_EDITABLE_SRC}: it is installed as a copy, not editable. Run "
        "`TMPDIR=~/.cache/pdm-tmp pdm install -L locks/<flavour>.lock "
        "--frozen-lockfile`."
    )
