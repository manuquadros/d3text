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
from d3text import logs
from d3text.models.models import Step, print_epoch_stats


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
