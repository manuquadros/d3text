"""Pure unit tests for models/config.py."""

import pathlib

import pytest
from pydantic import ValidationError

from d3text.models import config as cfg


def test_model_config_defaults():
    c = cfg.ModelConfig()
    assert c.model_class == "ETEBrendaModel"
    assert c.optimizer == "adam"
    assert c.batch_size == 32
    assert c.hidden_layers == [32]


def test_model_config_round_trip(tmp_path):
    original = cfg.ModelConfig(lr=0.01, batch_size=8, dropout=0.2)
    path = tmp_path / "model.toml"
    cfg.save_model_config(original.model_dump(), str(path))
    loaded = cfg.load_model_config(str(path))
    assert loaded == original


def test_negative_lr_rejected():
    with pytest.raises(ValidationError):
        cfg.ModelConfig(lr=-1.0)


def test_machine_config_rejects_negative_cache():
    with pytest.raises(ValidationError):
        cfg.MachineConfig(cpu_embeddings_cache_size=-1)


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
