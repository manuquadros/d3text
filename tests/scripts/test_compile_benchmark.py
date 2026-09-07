"""The compile A/B is only worth reading if its two arms really are two arms.

Wall-clock timing is the easiest measurement to get confidently wrong: nothing
crashes when a second variable drifts between the arms, when every epoch's
timing is flattened onto the last one, or when the arm that died is quietly
dropped and the survivor's column is printed as a comparison. These pin the
four places that could happen.
"""

import importlib.util
import pathlib
import tomllib

import pytest
from d3text import runtime, tracking

_SCRIPTS = pathlib.Path(__file__).resolve().parents[2] / "scripts"
_DIRECTORY = _SCRIPTS / "compile_benchmark"


def _load(name: str):
    """One benchmark script as a module, without putting `scripts/` on the
    path: every name under it is a top-level one, so an import would shadow
    installed packages for the rest of the session."""
    path = _DIRECTORY / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


run_arms = _load("run_arms")
train_json = _load("train_json")
compare_arms = _load("compare_arms")

SECONDS = compare_arms.SECONDS


def _plan(base_env: dict[str, str], repeats: int = 2):
    return run_arms.plan(
        pathlib.Path("/tmp/config.toml"),
        pathlib.Path("/tmp/out"),
        limit=500,
        repeats=repeats,
        base_env=base_env,
    )


def test_the_arms_differ_only_in_the_compile_switch() -> None:
    """The comparison's whole validity: anything else differing between the
    arms would be attributed to compiling. The base environment here already
    exports the switch, which is the case an implementation that only *sets* it
    for the eager arm gets wrong — both arms would then run eager and no number
    of repeats would show it."""
    runs = _plan({"PATH": "/usr/bin", runtime.COMPILE_DISABLE_VARIABLE: "1"})

    environments = {run.arm: run.env for run in runs}
    differing = {
        key
        for key in set(environments[run_arms.COMPILED])
        | set(environments[run_arms.EAGER])
        if environments[run_arms.COMPILED].get(key)
        != environments[run_arms.EAGER].get(key)
    }

    assert differing == {runtime.COMPILE_DISABLE_VARIABLE}
    assert (
        runtime.COMPILE_DISABLE_VARIABLE not in environments[run_arms.COMPILED]
    )


def test_every_run_trains_the_same_thing() -> None:
    """The command lines may differ only where a run's own metrics file is
    named; the config, the epoch budget and `--limit` are the comparison."""
    runs = _plan({"PATH": "/usr/bin"})

    shapes = {
        tuple(part for part in run.command if part != str(run.metrics))
        for run in runs
    }

    assert len(shapes) == 1


def test_the_arms_are_interleaved_and_their_order_alternates() -> None:
    """Both arms once a repeat, and neither always first: on a card that heats
    up under load, an arm that always goes second always runs warmer, and a
    fixed A,B order folds that drift into the answer."""
    order = run_arms.schedule(4)

    by_repeat: dict[int, list[str]] = {}
    for repeat, arm in order:
        by_repeat.setdefault(repeat, []).append(arm)

    assert len(by_repeat) == 4
    assert all(
        sorted(arms) == sorted(run_arms.ARMS) for arms in by_repeat.values()
    )
    first = [arms[0] for arms in by_repeat.values()]
    assert first.count(run_arms.COMPILED) == first.count(run_arms.EAGER)


def test_the_generated_config_pins_the_epoch_budget_for_both_arms() -> None:
    """`patience` follows `num_epochs` so early stopping cannot cut one arm
    short: an arm that ran fewer epochs would contribute fewer samples to the
    median than the arm it is compared against."""
    source = pathlib.Path(_DIRECTORY / "cfg_base.toml")

    written = tomllib.loads(run_arms.benchmark_config(source, 7))

    assert written["num_epochs"] == 7
    assert written["patience"] == 7


def test_per_epoch_metrics_are_kept_under_their_own_epoch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The capture must not flatten the epochs into one dict. The first epoch
    is the one that pays for tracing, so a capture keeping only the last value
    would hide precisely the cost this benchmark exists to price."""
    monkeypatch.setattr(tracking, "log_metrics", tracking.log_metrics)
    collected = train_json.capture_epoch_metrics()

    tracking.log_metrics({SECONDS: 12.0}, 0)
    tracking.log_metrics({SECONDS: 7.0}, 1)
    tracking.log_metrics({"dataset/entities": 3.0})

    assert collected["0"][SECONDS] == 12.0
    assert collected["1"][SECONDS] == 7.0
    assert collected[train_json.NO_STEP]["dataset/entities"] == 3.0


def test_a_crashed_arm_keeps_the_epochs_it_did_finish() -> None:
    """`nn.Module.compile` is lazy, so a graph that cannot be built raises
    inside the first epoch rather than at the call that installed it. That
    makes a dead arm a result, and a result has to survive to the report."""
    record = train_json.summarize(
        {"0": {SECONDS: 9.0}, train_json.NO_STEP: {"model/size_mb": 1.0}},
        {"compiled": True},
        "BackendCompilerFailed: backend='inductor' raised",
    )

    assert record["completed"] is False
    assert record["epochs"] == {"0": {SECONDS: 9.0}}
    assert "BackendCompilerFailed" in str(record["error"])


def test_an_exception_is_summarized_to_one_quotable_line() -> None:
    """Inductor's failures are paragraphs; a table has room for a line."""
    summary = train_json.error_summary(
        AssertionError("Node convert_element_type_16\n  was invalid")
    )

    assert summary == (
        "AssertionError: Node convert_element_type_16 was invalid"
    )


def test_an_arm_that_did_not_do_what_its_name_says_is_flagged() -> None:
    """A compiled arm that compiled and then crashed is a result; one that
    never compiled at all is a broken experiment, and the two must not be
    confused — the second invalidates every number beside it."""
    crashed = {"compiled": True, "completed": False, "error": "boom"}
    never_compiled = {"compiled": False, "completed": True}

    assert run_arms.switch_failure(run_arms.COMPILED, crashed) is None
    assert run_arms.switch_failure(run_arms.COMPILED, never_compiled)
    assert run_arms.switch_failure(run_arms.EAGER, {"compiled": True})
    assert run_arms.switch_failure(run_arms.EAGER, {"compiled": False}) is None


def _run(arm: str, repeat: int, **overrides):
    record = {
        "arm": arm,
        "repeat": repeat,
        "completed": True,
        "compiled": arm == run_arms.COMPILED,
        "epochs": {"0": {SECONDS: 10.0}, "1": {SECONDS: 8.0}},
        "error": None,
        "log": f"train_{arm}_{repeat}.log",
        "switch_failure": None,
    }
    record.update(overrides)
    return record


def test_a_dead_arm_is_named_rather_than_dropped() -> None:
    """The failure mode this guards is a table that looks complete: with one
    arm gone, the survivor's column is not a comparison and the report must say
    so. It is still exit 0 — a card on which compiling does not work is an
    answer, not a broken run."""
    runs = [
        _run(
            run_arms.COMPILED,
            0,
            completed=False,
            epochs={},
            error="BackendCompilerFailed: inductor raised",
        ),
        _run(run_arms.EAGER, 0),
    ]

    message, status = compare_arms.verdict(runs)

    assert status == 0
    assert "compiled: 0/1 runs completed" in message
    assert "BackendCompilerFailed" in message
    assert "not a comparison" in message


def test_a_miswired_switch_invalidates_the_whole_comparison() -> None:
    """The one outcome that is not a finding: if an arm compiled when it should
    not have, no timing beside it can be read as a compilation effect."""
    runs = [
        _run(run_arms.COMPILED, 0, switch_failure="it never compiled"),
        _run(run_arms.EAGER, 0),
    ]

    message, status = compare_arms.verdict(runs)

    assert status == 1
    assert "NOT COMPARABLE" in message


def test_the_first_epoch_is_not_pooled_with_the_ones_after_it() -> None:
    """The first epoch carries the tracing cost, so it is not a sample from the
    same population; averaging it in would let the epoch budget decide the
    verdict."""
    runs = [_run(run_arms.COMPILED, 0), _run(run_arms.COMPILED, 1)]

    summary = compare_arms.summarize(runs)

    assert summary["first_epoch_seconds"] == 10.0
    assert summary["later_epoch_seconds"] == 8.0
    assert summary["whole_run_seconds"] == 18.0
