"""The tracked scripts that write `config.toml` must write one that loads.

Three experiment drivers generate the repo-root `config.toml` wholesale, so a
key renamed in Python alone leaves them writing a file every `train`,
`evaluate` and `tune` then dies on in `machine_config()` — and regenerating it
is how the breakage repairs itself after a hand edit.
`tests/test_scripts_importable.py` cannot see this: it resolves Python imports
and does not read shell.
"""

import pathlib
import subprocess
import tomllib

import pytest
from d3text.models import config as cfg

REPO_ROOT = pathlib.Path(__file__).parents[1]
CONFIG_ASSIGNMENT = 'config="$REPO/config.toml"'
GENERATORS = sorted(
    path
    for path in (REPO_ROOT / "scripts").rglob("run.sh")
    if CONFIG_ASSIGNMENT in path.read_text()
)


def configure_body(script: pathlib.Path) -> str:
    """The script's `configure` function, on its own.

    Sourcing the script whole would run the experiment it drives, so the
    function is cut out by its opening line and the unindented `}` that closes
    it — the inner brace group ends indented, on `} > "$config"`.
    """
    lines = script.read_text().splitlines()
    start = [i for i, line in enumerate(lines) if line == "configure () {"]
    assert start, f"{script} has no `configure () {{` to extract"
    end = [i for i, line in enumerate(lines[start[0] :]) if line == "}"]
    assert end, f"{script}'s configure function has no closing brace"

    return "\n".join(lines[start[0] : start[0] + end[0] + 1])


def test_the_generated_configs_are_covered():
    """A discovery that silently matched nothing would leave the check below
    passing on an empty parametrisation."""
    assert GENERATORS, (
        f"no tracked script assigns {CONFIG_ASSIGNMENT}; the discovery, not "
        "the tree, is what to fix if these scripts still generate a config"
    )


@pytest.mark.parametrize("script", GENERATORS, ids=lambda path: path.parts[-3])
def test_a_generated_config_names_only_machine_config_fields(script, tmp_path):
    """Run the generator's `configure` against a throwaway repo and load what
    it wrote. Validating is not enough on its own: a deprecated key that
    `MachineConfig` still migrates would validate, and the point is that no
    tracked file goes on writing one."""
    repo, out, store = (tmp_path / name for name in ("repo", "out", "store"))
    for directory in (repo, out, store):
        directory.mkdir()

    harness = "\n".join(
        (
            "set -euo pipefail",
            "log () { :; }",
            f'REPO="{repo}"',
            f'OUT="{out}"',
            f'STORE="{store}"',
            configure_body(script),
            "configure",
        )
    )
    subprocess.run(["bash", "-c", harness], check=True, capture_output=True)

    with (repo / "config.toml").open("rb") as generated:
        contents = tomllib.load(generated)

    assert contents, f"{script} wrote an empty config.toml"
    assert set(contents) <= set(cfg.MachineConfig.model_fields), (
        f"{script} writes keys MachineConfig does not have: "
        f"{sorted(set(contents) - set(cfg.MachineConfig.model_fields))}"
    )
    cfg.MachineConfig(**contents)


def test_no_tracked_file_still_assigns_the_document_budget():
    """The key the megabyte budget replaced must not be set anywhere.

    Naming it is fine — the migration in `config.py` has to — so the pattern
    matches an assignment, which is what a shell script echoing it into a
    config.toml writes and what a config.toml carries. The key comes from the
    module rather than a literal, so this file cannot match itself.
    """
    assignment = cfg.DOCUMENT_BUDGET_KEY + r"[ \t]*="

    found = subprocess.run(
        ["git", "grep", "--name-only", "-E", "-e", assignment],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    assert found.returncode in (0, 1), found.stderr
    assert found.returncode == 1, (
        "the budget in documents is still configured in: "
        f"{', '.join(found.stdout.split())}"
    )
