"""The per-trial config dump `tune` writes to the log.

`pprint.pp` (the old call) hardcodes `sort_dicts=False`; `pprint.pformat`
(what it was replaced with) defaults `sort_dicts` to `True`. Left at that
default the dump prints alphabetically instead of in `ModelConfig` field
order, which is what a reader expects when comparing it against the TOML.
"""

import argparse

import pytest

from d3text.cli import tune
from d3text.models.config import ModelConfig


class _StopAfterDump(Exception):
    """Raised from the first thing `main` does after the log line, so the
    test never has to drive a real dataset/model/trainer through it."""


@pytest.fixture
def stop_after_config_dump(monkeypatch):
    monkeypatch.setattr(
        tune,
        "command_line_args",
        lambda: argparse.Namespace(
            config="unused.toml", output="unused.csv", limit=None
        ),
    )
    monkeypatch.setattr(
        tune, "load_tuning_config", lambda path: [ModelConfig()]
    )
    monkeypatch.setitem(tune.encodings, ModelConfig().base_model, "unused.hdf5")

    def blow_up(**kwargs):
        raise _StopAfterDump

    monkeypatch.setattr(tune, "brenda_dataset", blow_up)


def test_dump_key_order_matches_model_dump_field_order(
    stop_after_config_dump, capsys
):
    with pytest.raises(_StopAfterDump):
        tune.main()

    dump = ModelConfig().model_dump()
    field_order = list(dump.keys())

    printed = capsys.readouterr().out
    dump_start = printed.index("{")
    dump_end = printed.index("}\n", dump_start) + 1
    dump_text = printed[dump_start:dump_end]

    printed_order = [
        line.split(":", 1)[0].strip().strip("'")
        for line in dump_text.strip("{}").splitlines()
        if line.strip()
    ]
    assert printed_order == field_order
