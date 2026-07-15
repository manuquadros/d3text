"""Pure unit tests for models/config.py."""

import json
import pathlib
import random

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


def test_ete_config_requires_layer_lists():
    with pytest.raises(ValidationError):
        cfg.ETEModelConfig()  # entity_layers / class_layers are required
    ete = cfg.ETEModelConfig(entity_layers=[8], class_layers=[4])
    assert ete.entity_layers == [8]
    assert ete.class_layers == [4]


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


def test_committed_tuning_config_names_a_buildable_model_class():
    """The repo's own tuning grid must name a model the factory can build.

    Asserted against the registry the CLI actually resolves through, not
    against whatever `d3text.models` happens to export: a name can be an
    attribute of that package without naming a model at all.
    """
    with (REPO_ROOT / "tuning_config.toml").open() as f:
        grid = tomlkit.load(f).unwrap()

    for name in grid["model_class"]:
        assert name in factory.MODEL_CLASSES, (
            f"tuning_config.toml names {name!r}"
        )
