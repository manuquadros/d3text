"""Pure unit tests for models/config.py."""

import ast
import inspect
import json
import pathlib
import random
import subprocess

import pytest
import tomlkit
from pydantic import ValidationError

from d3text import factory
from d3text.models import config as cfg

REPO_ROOT = pathlib.Path(__file__).parents[2]


def write_tuning_grid(path: pathlib.Path, **grid: list) -> str:
    """Write a tuning grid as TOML. JSON renders each list compatibly."""
    path.write_text(
        "\n".join(f"{key} = {json.dumps(vs)}" for key, vs in grid.items())
    )
    return str(path)


def test_model_config_defaults():
    c = cfg.ModelConfig()
    assert c.model_class == "ETEBrendaModel"
    assert c.optimizer == "adam"
    assert c.batch_size == 32
    assert c.hidden_layers == [32]
    assert c.entity_entropy_threshold == 0.8
    assert c.biaffine_hidden_size == 32


def test_model_config_round_trip(tmp_path):
    original = cfg.ModelConfig(lr=0.01, batch_size=8, dropout=0.2)
    path = tmp_path / "model.toml"
    cfg.save_model_config(original.model_dump(), str(path))
    loaded = cfg.load_model_config(str(path))
    assert loaded == original


def test_negative_lr_rejected():
    with pytest.raises(ValidationError):
        cfg.ModelConfig(lr=-1.0)


def test_negative_entity_entropy_threshold_rejected():
    with pytest.raises(ValidationError):
        cfg.ModelConfig(entity_entropy_threshold=-0.1)


def test_non_positive_biaffine_hidden_size_rejected():
    with pytest.raises(ValidationError):
        cfg.ModelConfig(biaffine_hidden_size=0)


def test_unknown_field_rejected():
    """A field with no reader must fail loudly, not be silently dropped.

    entity_loss_scaling_factor was removed because nothing consumed it;
    configs that still carry it (or any other unrecognised key) must be
    rejected rather than accepted with the key quietly ignored.
    """
    with pytest.raises(ValidationError):
        cfg.ModelConfig(entity_loss_scaling_factor=1.0)


@pytest.mark.parametrize(
    "field, value",
    [
        ("normalization", "LayerNorm"),
        ("normalization", "Layer"),
        ("lr_scheduler", "exponentail"),
        ("lr_scheduler", "plateau"),
    ],
)
def test_misspelled_behaviour_selector_rejected(field, value):
    """A typo in either used to fall through a `match` whose unmatched arm is
    a no-op, training with no normalization / no scheduler and looking
    configured in every log."""
    with pytest.raises(ValidationError):
        cfg.ModelConfig(**{field: value})


@pytest.mark.parametrize(
    "field, value",
    [
        ("normalization", "layer"),
        ("normalization", "batch"),
        ("normalization", "none"),
        ("lr_scheduler", ""),
        ("lr_scheduler", "exponential"),
        ("lr_scheduler", "reduce_on_plateau"),
    ],
)
def test_behaviour_selector_accepts_every_spelling_in_use(field, value):
    assert getattr(cfg.ModelConfig(**{field: value}), field) == value


def test_machine_config_rejects_negative_cache():
    with pytest.raises(ValidationError):
        cfg.MachineConfig(cpu_embeddings_cache_size=-1)


def test_machine_config_runtime_defaults():
    """The runtime keys are optional: a config.toml predating them (or no file
    at all) still yields the settings the scripts have been running with."""
    mc = cfg.MachineConfig(cpu_embeddings_cache_size=0)
    assert mc.float32_matmul_precision == "medium"
    assert mc.cudnn_allow_tf32 is True
    assert mc.expandable_segments is True
    assert mc.tokenizers_parallelism is True


def test_machine_config_rejects_unknown_matmul_precision():
    with pytest.raises(ValidationError):
        cfg.MachineConfig(
            cpu_embeddings_cache_size=0, float32_matmul_precision="fastest"
        )


def test_machine_config_rejects_unknown_key():
    """A misspelled key must fail loudly rather than be dropped.

    Every field here is a performance or allocator knob, so a key that is
    ignored leaves the feature at its default and is indistinguishable from a
    slow machine.
    """
    with pytest.raises(ValidationError):
        cfg.MachineConfig(
            cpu_embeddings_cache_size=0, embeddings_stor="/nowhere"
        )


def test_example_config_still_loads():
    """The shipped example is what a machine copies to config.toml, so every
    key it names has to be one MachineConfig accepts."""
    example = tomlkit.loads((REPO_ROOT / "config.toml.example").read_text())
    assert cfg.MachineConfig(**example).float32_matmul_precision == "medium"


def test_machine_config_error_names_the_config_file(tmp_path, monkeypatch):
    """The rejection surfaces at import of d3text.models.base, far from the
    file that caused it, and pydantic names only the offending field."""
    original_open = pathlib.Path.open
    bad = tmp_path / "config.toml"
    bad.write_text("cpu_embeddings_cache_size = 0\nembeddings_stor = '/x'\n")

    def open_bad_config(self, *args, **kwargs):
        if self.name == "config.toml":
            return original_open(bad, *args, **kwargs)
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(pathlib.Path, "open", open_bad_config)
    with pytest.raises(ValidationError) as caught:
        cfg.machine_config()
    assert any("config.toml" in note for note in caught.value.__notes__)


def test_machine_config_falls_back_when_file_missing(monkeypatch):
    """machine_config() must not raise when config.toml is absent."""
    original_open = pathlib.Path.open

    def open_missing_config(self, *args, **kwargs):
        if self.name == "config.toml":
            raise FileNotFoundError(self)
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(pathlib.Path, "open", open_missing_config)
    mc = cfg.machine_config()
    assert mc.cpu_embeddings_cache_size == 0


def test_load_tuning_config_replays_a_sweep_from_an_injected_rng(tmp_path):
    """The same seed must redraw the same sweep, so a tuning run can be
    reproduced. Drawing from the unseeded global `random` cannot do this."""
    path = write_tuning_grid(
        tmp_path / "tuning.toml",
        optimizer=["adam", "adamw", "nadam"],
        lr=[0.1, 0.01, 0.001],
        hidden_layers=[32, 64],
    )

    first = cfg.load_tuning_config(path, rng=random.Random(0))
    again = cfg.load_tuning_config(path, rng=random.Random(0))
    other = cfg.load_tuning_config(path, rng=random.Random(1))

    assert len(first) == cfg.SWEEP_SIZE
    assert first == again
    assert first != other, "a different seed must draw a different sweep"


def test_load_tuning_config_does_not_draw_from_the_global_rng(tmp_path):
    """The sweep must come from its own generator, not the global `random`
    stream: drawn from the global one, a sweep is silently a function of
    whatever last seeded the process, and two runs under the same seed explore
    the identical 250 configurations instead of independent samples.

    (Asserting the global state is *untouched* would not work: `beartype`
    spot-checks a returned container by indexing it at random, so every
    beartyped function returning a list advances the global stream.)
    """
    path = write_tuning_grid(
        tmp_path / "tuning.toml",
        optimizer=["adam", "adamw", "nadam"],
        lr=[0.1, 0.01, 0.001],
        hidden_layers=[32, 64],
    )

    random.seed(7)
    first = cfg.load_tuning_config(path)
    random.seed(7)
    again = cfg.load_tuning_config(path)

    assert first != again


def test_load_tuning_config_accepts_a_grid_with_boolean_fields(tmp_path):
    """TOML booleans must survive into ModelConfig.

    tomlkit's Integer/Float/String/Array subclass their builtins, so pydantic
    accepts them as-is; `bool` cannot be subclassed, so a TOML bool inside an
    array arrives as `tomlkit.items.Bool` and fails validation. Every field in
    the repo's own tuning_config.toml is affected.
    """
    path = write_tuning_grid(
        tmp_path / "tuning.toml",
        optimizer=["adam"],
        hidden_layers=[32],
        common_hidden_block=[True, False],
        separate_predicate_layer=[True, False],
    )

    configs = cfg.load_tuning_config(path, rng=random.Random(0))

    assert configs
    assert {c.common_hidden_block for c in configs} == {True, False}
    assert all(isinstance(c.common_hidden_block, bool) for c in configs)


def test_load_tuning_config_takes_a_grid_smaller_than_the_sweep_whole(tmp_path):
    """A grid with fewer configurations than the sweep size is a legitimate
    config, not a `Sample larger than population` crash."""
    path = write_tuning_grid(
        tmp_path / "tuning.toml", optimizer=["adam"], hidden_layers=[32]
    )

    configs = cfg.load_tuning_config(path, rng=random.Random(0))

    assert 0 < len(configs) < cfg.SWEEP_SIZE
    assert all(c.optimizer == "adam" for c in configs)


def test_tuning_config_is_tracked_in_git():
    """`.gitignore`'s `*config.toml`, which exists to hide the machine-local
    `config.toml`, also swallowed `tuning_config.toml`. The test below then
    asserted on a file that only ever existed in working copies that happened
    to have a stray one, so presence on disk proves nothing here: only
    tracking does.
    """
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "tuning_config.toml"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    assert tracked.returncode == 0, (
        "tuning_config.toml is not tracked; a fresh clone would not have it: "
        f"{tracked.stderr.strip()}"
    )


TRACKED_MODEL_CONFIGS = [
    "scripts/dec03_full/cfg_logsumexp.toml",
    "scripts/dec03_full/cfg_logmeanexp.toml",
    "scripts/dec04_full/cfg_baseline.toml",
    "tests/best_config_so_far.toml",
]


@pytest.mark.parametrize("relative_path", TRACKED_MODEL_CONFIGS)
def test_tracked_experiment_config_validates(relative_path):
    """A config left behind by a `ModelConfig` field removal fails only when
    a run reaches `load_model_config`, hours into an experiment on a VM. This
    walks the fixed list of tracked, `ModelConfig`-shaped experiment configs
    (not a glob: `tuning_config.toml` is sweep-shaped, and
    `src/d3text/models/current_model_config.toml` /
    `tests/teste_output_config.toml` are dumps with a different shape) and
    fails locally instead.
    """
    path = REPO_ROOT / relative_path
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", relative_path],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert (
        tracked.returncode == 0
    ), f"{relative_path} is not tracked: {tracked.stderr.strip()}"

    cfg.load_model_config(str(path))


def test_committed_tuning_config_names_a_buildable_model_class():
    """The repo's own tuning grid must name a model the factory can build.

    Asserted against the registry the CLI actually resolves through, not
    against whatever `d3text.models` happens to export: a name can be an
    attribute of that package without naming a model at all.
    """
    with (REPO_ROOT / "tuning_config.toml").open() as f:
        grid = tomlkit.load(f).unwrap()

    for name in grid["model_class"]:
        assert (
            name in factory.MODEL_CLASSES
        ), f"tuning_config.toml names {name!r}"


def _names_mentioned(node: ast.AST) -> set[str]:
    """Every bare name appearing anywhere under `node`."""
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


def _module_reference_graph(tree: ast.Module) -> dict[str, set[str]]:
    """Map each module-level name to the names its own definition mentions."""
    graph: dict[str, set[str]] = {}
    for stmt in tree.body:
        if isinstance(stmt, (ast.FunctionDef, ast.ClassDef)):
            targets, mentions = [stmt.name], _names_mentioned(stmt)
        elif isinstance(stmt, ast.Assign):
            targets = [t.id for t in stmt.targets if isinstance(t, ast.Name)]
            mentions = _names_mentioned(stmt.value)
        elif isinstance(stmt, ast.AnnAssign) and isinstance(
            stmt.target, ast.Name
        ):
            targets = [stmt.target.id]
            mentions = (
                set() if stmt.value is None else _names_mentioned(stmt.value)
            )
        else:
            continue
        for target in targets:
            graph.setdefault(target, set()).update(mentions)
    return graph


def _reachable_from(graph: dict[str, set[str]], roots: list[str]) -> set[str]:
    """The transitive closure of `graph` over `roots`."""
    seen: set[str] = set()
    pending = list(roots)
    while pending:
        name = pending.pop()
        if name not in seen:
            seen.add(name)
            pending.extend(graph.get(name, ()))
    return seen


def test_every_declared_model_config_is_reachable_from_a_loader():
    """A `ModelConfig` class no loader can build configures nothing.

    Reachable means named in the reference graph of the module's own
    loaders — the module-level functions whose return annotation names
    `ModelConfig` — directly or through a module-level name they use, such
    as a dispatch table. `ModelConfig` itself is checked too, so a broken
    walk fails here rather than passing vacuously.
    """
    tree = ast.parse(inspect.getsource(cfg))
    loaders = [
        stmt.name
        for stmt in tree.body
        if isinstance(stmt, ast.FunctionDef)
        and stmt.returns is not None
        and "ModelConfig" in _names_mentioned(stmt.returns)
    ]
    assert loaders, "config.py declares no loader returning a ModelConfig"

    reachable = _reachable_from(_module_reference_graph(tree), loaders)
    declared = {
        name
        for name, value in vars(cfg).items()
        if isinstance(value, type) and issubclass(value, cfg.ModelConfig)
    }
    unreachable = sorted(declared - reachable)
    assert not unreachable, (
        f"{unreachable} is named by none of {loaders}, so no TOML can "
        "select it"
    )
