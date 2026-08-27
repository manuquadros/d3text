"""`brenda_references/tests/` used to run only when someone `cd`ed into the
subpackage and ran pytest by hand: the root `pyproject.toml`'s `testpaths`
points only at the top-level `tests/`, and `brenda_references` had no
`.github/` of its own, so nothing in CI ever collected its suite.

It is not folded into the root job here. `brenda_references` is its own pdm
project with its own git-URL dependencies (`apiadapters`, `d3types`, `gme`,
`lpsn-interface`, `ncbitax`); resolving it standalone needs the same
`use_uv = "false"` guard the root `pyproject.toml` already carries, plus its
own copy of the `lpsn-interface`/`ncbitax` `[tool.pdm.resolution.overrides]`
the root project documents, so a dedicated workflow that installs and tests
it from its own directory is the smaller, self-contained change.
"""

import pathlib
import tomllib

REPO_ROOT = pathlib.Path(__file__).parents[1]
WORKFLOW = REPO_ROOT / ".github/workflows/brenda-references-tests.yml"
PACKAGE_ROOT = REPO_ROOT / "brenda_references"


def test_a_workflow_installs_and_runs_the_subpackage_suite():
    text = WORKFLOW.read_text()

    assert "working-directory: brenda_references" in text
    assert "pdm install -G dev" in text
    assert "pytest" in text


def test_the_subpackage_disables_pdms_uv_backend():
    """Without this, a machine whose global pdm config defaults to
    `use_uv = true` silently rewrites this package's git-URL dependencies
    into `[tool.uv.sources]` the moment `pdm install` or `pdm lock` runs
    here, dropping `[tool.pdm.resolution.overrides]` in the process (uv has
    no equivalent), which is what the next test depends on.
    """
    with (PACKAGE_ROOT / "pdm.toml").open("rb") as f:
        config = tomllib.load(f)

    assert config["use_uv"] == "false"


def test_the_subpackage_overrides_the_two_split_git_dependencies():
    """`d3types` pins `lpsn-interface` through a different URL spelling than
    the one `brenda_references` declares for itself; pdm (unlike uv) treats a
    direct-reference URL as part of a package's identity, so an unresolved
    standalone lock sees two different packages named `lpsn-interface`.
    `ncbitax` carries the same split.
    """
    with (PACKAGE_ROOT / "pyproject.toml").open("rb") as f:
        config = tomllib.load(f)

    overrides = config["tool"]["pdm"]["resolution"]["overrides"]
    assert overrides["lpsn-interface"].endswith("lpsn-interface")
    assert overrides["ncbitax"].endswith("ncbitax")


def test_the_subpackage_caps_python_below_gmes_upper_bound():
    """`gme` (a dependency of `brenda_references`) requires `python<3.13`;
    without the same upper bound here, a standalone lock rejects every
    candidate of it.
    """
    with (PACKAGE_ROOT / "pyproject.toml").open("rb") as f:
        config = tomllib.load(f)

    assert config["project"]["requires-python"] == ">=3.12,<3.13"


def test_the_subpackage_registers_the_integration_marker_and_excludes_it():
    with (PACKAGE_ROOT / "pyproject.toml").open("rb") as f:
        config = tomllib.load(f)

    ini = config["tool"]["pytest"]["ini_options"]
    assert any("integration" in marker for marker in ini["markers"])
    assert "not integration" in " ".join(ini["addopts"])


def test_the_taxdump_and_data_pull_dependent_tests_are_marked_integration():
    """These need resources a generic CI runner does not have: the NCBI
    taxdump archive `ncbitax` reads, and the BRENDA database export
    `scripts/pull_data.py` fetches with credentials. Marking them
    `integration` keeps them in the suite, documented, without demanding
    they pass in the default run -- the same treatment the root project
    already gives comparable tests.
    """
    targets = {
        "brenda_references/tests/test_apis.py": [
            "test_bacteria_post_init_lpsn_id",
            "test_strain_in_bacteria_name_is_detected",
        ],
        "brenda_references/tests/test_taxonomy.py": [
            "test_fix_bacteria",
            "test_fix_strains",
            "test_29345379",
        ],
        "brenda_references/tests/test_scripts.py": [
            "test_generate_dataset_data_dir_holds_the_splits",
        ],
    }

    for relpath, names in targets.items():
        lines = (REPO_ROOT / relpath).read_text().splitlines()
        for name in names:
            def_line = next(
                i for i, line in enumerate(lines) if f"def {name}(" in line
            )
            assert (
                "@pytest.mark.integration" in lines[def_line - 1]
            ), f"{relpath}::{name} is not marked integration"
