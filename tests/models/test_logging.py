"""The training loop's narration goes through `logging`, not to stdout.

`print` and `tqdm.write` write whatever the code decided to say, whenever it
decided to say it. Nothing downstream can quiet a sweep's per-epoch chatter, or
turn a warning into something a run's log can be filtered on, and a library
that writes to stdout has taken a decision that belongs to whoever owns the
process. These pin that the lines are now records on
`d3text.models.models`, with the level deciding whether they reach the console.
"""

import logging

import pytest
import torch
import torch.nn as nn
from d3text import logs
from d3text.models.models import Model, Step, print_epoch_stats


class FakeEncoder:
    """The two views `unfreeze_encoder_layers` reads off a base model."""

    def __init__(self, layers: int = 3) -> None:
        self.params = {
            f"encoder.layer.{index}.attention.weight": nn.Parameter(
                torch.zeros(1), requires_grad=False
            )
            for index in range(layers)
        }

    def state_dict(self) -> dict[str, nn.Parameter]:
        return self.params

    def named_parameters(self) -> list[tuple[str, nn.Parameter]]:
        return list(self.params.items())


@pytest.fixture
def console(
    restore_package_logger: logging.Logger,
    capsys: pytest.CaptureFixture[str],
) -> pytest.CaptureFixture[str]:
    """The package configured as an entry point configures it: INFO to stdout."""
    logs.configure(logging.INFO)
    return capsys


@pytest.fixture
def silenced(
    restore_package_logger: logging.Logger,
    capsys: pytest.CaptureFixture[str],
) -> pytest.CaptureFixture[str]:
    """The package at WARNING — what `D3TEXT_LOG_LEVEL=WARNING` buys."""
    logs.configure(logging.WARNING)
    return capsys


def test_print_epoch_stats_writes_what_it_returns(
    console: pytest.CaptureFixture[str],
) -> None:
    returned = print_epoch_stats(
        {"entity": 4.0, "class": 2.0}, denominator=4, step=Step.TRAINING
    )

    out = console.readouterr().out

    assert returned == {
        "training/entity": 1.0,
        "training/class": 0.5,
        "training/total": 1.5,
    }
    assert "Average (entity) training loss: 1.0000" in out
    assert "Average (class) training loss: 0.5000" in out
    assert "Average training loss: 1.5000" in out


def test_print_epoch_stats_is_silent_above_its_level(
    silenced: pytest.CaptureFixture[str],
) -> None:
    """The whole point of the change: an epoch's narration is suppressible
    without suppressing the numbers, which still go to the caller and to
    MLflow."""
    returned = print_epoch_stats(
        {"entity": 4.0}, denominator=4, step=Step.VALIDATION
    )

    captured = silenced.readouterr()

    assert returned == {
        "validation/entity": 1.0,
        "validation/total": 1.0,
    }
    assert captured.out == ""
    assert captured.err == ""


def test_unfreeze_encoder_layers_names_what_it_unfroze(
    console: pytest.CaptureFixture[str], stub
) -> None:
    model = stub(Model, base_model=FakeEncoder())

    model.unfreeze_encoder_layers(n=1)

    out = console.readouterr().out

    assert "Trainable: encoder.layer.2.attention.weight" in out
    assert "encoder.layer.0" not in out


def test_unfreeze_encoder_layers_is_silent_above_its_level(
    silenced: pytest.CaptureFixture[str], stub
) -> None:
    encoder = FakeEncoder()
    model = stub(Model, base_model=encoder)

    model.unfreeze_encoder_layers(n=1)

    captured = silenced.readouterr()

    assert captured.out == ""
    assert captured.err == ""
    # Quieting the narration must not quiet the work it narrates.
    assert (
        encoder.params["encoder.layer.2.attention.weight"].requires_grad is True
    )
