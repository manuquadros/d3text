#!/usr/bin/env python
"""Time training with `torch.compile` on and off, interleaving the arms.

Both arms train the same model on the same data from one generated config and
differ in exactly one thing: whether `D3TEXT_DISABLE_COMPILE` is set. What they
cannot share is the machine's thermal state, and a card throttles under a
sustained load — so arms run back to back would confound the switch with the
order they ran in. They are therefore interleaved, each repeat reverses their
order, and the report medians over repeats instead of trusting one pair of
numbers.

An arm that dies is recorded rather than fatal. `nn.Module.compile` is lazy, so
a graph that cannot be built raises inside the first epoch and not at the call
that installed it; "the compiled arm does not survive epoch 0 on this card" is
a finding, and the run that produced it must reach the table.

    pdm run python scripts/compile_benchmark/run_arms.py cfg_base.toml \\
        --epochs 3 --limit 500 --repeats 3
"""

import argparse
import dataclasses
import datetime
import json
import os
import pathlib
import subprocess
import sys
from collections.abc import Mapping
from typing import Any

import tomlkit
import torch
from d3text import runtime

COMPILED = "compiled"
EAGER = "eager"
ARMS = (COMPILED, EAGER)

_HERE = pathlib.Path(__file__).resolve().parent
WRAPPER = _HERE / "train_json.py"
# `train` reads relative data paths and writes `lpsn.log` into the working
# directory, so the arms run from the checkout rather than from wherever the
# benchmark was launched.
REPO = _HERE.parent.parent


@dataclasses.dataclass(frozen=True)
class Run:
    """One arm's turn: what to run, in what environment, and where it lands."""

    arm: str
    repeat: int
    command: list[str]
    env: dict[str, str]
    metrics: pathlib.Path


def read_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run_arms",
        description="Time training with torch.compile on and off.",
    )
    parser.add_argument("config", help="the model config both arms train from")
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="epochs per arm; a timing A/B wants a handful, not convergence",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=500,
        help="training documents per arm; 0 for the whole split",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="how many times each arm runs; the report takes the median",
    )
    parser.add_argument("--out", default=str(_HERE / "out"))
    return parser.parse_args()


def unsupported_machine() -> str | None:
    """Say so if this machine cannot compile, which makes the arms identical.

    The failure this exists to prevent is not a crash: on a card below compute
    capability 7.0 `compile_model` returns False whatever the switch says, both
    arms run eager, and the benchmark reports a confident speedup of one drawn
    from pure timing noise.

    :return: the diagnostic, or None where the comparison can be made.
    """
    if not torch.cuda.is_available():
        return (
            "no CUDA device, so torch.compile's Triton backend has nothing "
            "to target and both arms would time the same eager code"
        )

    if not runtime.is_triton_compatible():
        major, minor = torch.cuda.get_device_capability(0)
        return (
            f"{torch.cuda.get_device_name(0)} is compute capability "
            f"{major}.{minor} and Triton needs 7.0 or newer, so "
            f"`compile_model` returns False on this card whatever "
            f"{runtime.COMPILE_DISABLE_VARIABLE} says and both arms would "
            f"run eager"
        )

    return None


def benchmark_config(source: pathlib.Path, epochs: int) -> str:
    """`source` with the epoch count both arms are to run.

    `patience` follows `num_epochs` so that neither arm can be cut short by
    early stopping: an arm that stopped an epoch early would contribute fewer
    epochs to the median than the arm it is being compared against.

    :param source: the model config to derive from.
    :param epochs: how many epochs each arm runs.
    :return: the TOML to write and hand to both arms.
    """
    document = tomlkit.parse(source.read_text())
    document["num_epochs"] = epochs
    document["patience"] = epochs
    return tomlkit.dumps(document)


def arm_environment(base: Mapping[str, str], arm: str) -> dict[str, str]:
    """`base` carrying the compile switch this arm needs, and nothing else.

    The compiled arm *removes* the variable rather than leaving it alone: a
    shell that had already exported it would otherwise run both arms eager,
    which no amount of repeating would show up as anything but a null result.

    :param base: the environment to derive from, ordinarily `os.environ`.
    :param arm: which arm the environment is for.
    :return: the environment to run that arm in.
    """
    env = dict(base)
    if arm == EAGER:
        env[runtime.COMPILE_DISABLE_VARIABLE] = "1"
    else:
        env.pop(runtime.COMPILE_DISABLE_VARIABLE, None)

    return env


def schedule(repeats: int) -> list[tuple[int, str]]:
    """The order the arms run in: each once a repeat, alternating who is first.

    Alternating rather than a fixed A,B: on a machine that heats up under load
    the arm that always goes second always runs on a warmer card, and a fixed
    order would fold that drift into the answer.

    :param repeats: how many times each arm runs.
    :return: `(repeat, arm)` pairs, in the order they are to run.
    """
    order = []
    for repeat in range(repeats):
        arms = ARMS if repeat % 2 == 0 else tuple(reversed(ARMS))
        order.extend((repeat, arm) for arm in arms)

    return order


def plan(
    config: pathlib.Path,
    out: pathlib.Path,
    *,
    limit: int | None,
    repeats: int,
    base_env: Mapping[str, str],
) -> list[Run]:
    """Every run this benchmark will make, in the order it will make them.

    Built up front rather than as each arm starts so that the whole plan can be
    asserted over: the arms are comparable only if nothing but the switch
    differs between them, and that is a property of the set, not of one run.

    :param config: the config both arms train from.
    :param out: the directory the metrics and the scratch checkpoint go in.
    :param limit: training documents per arm, or None for the whole split.
    :param repeats: how many times each arm runs.
    :param base_env: the environment to derive each arm's from.
    :return: the runs to make.
    """
    # One scratch path for every run: the checkpoint carries the frozen base
    # model, a timing run's weights are worth nothing, and a path that varied
    # by arm would be one more thing differing between them.
    checkpoint = out / "model.pt"

    runs = []
    for repeat, arm in schedule(repeats):
        metrics = out / f"metrics_{arm}_{repeat}.json"
        command = [
            sys.executable,
            str(WRAPPER),
            str(metrics),
            str(config),
            str(checkpoint),
        ]
        if limit is not None:
            command += ["--limit", str(limit)]

        runs.append(
            Run(
                arm=arm,
                repeat=repeat,
                command=command,
                env=arm_environment(base_env, arm),
                metrics=metrics,
            )
        )

    return runs


def switch_failure(arm: str, record: Mapping[str, Any]) -> str | None:
    """Say so if the compile switch did not do what the arm's name says.

    This is the one condition that invalidates the comparison rather than
    answering it: an arm compiled when it should not have been, or the reverse,
    means the two arms were never the two things being compared. A compiled arm
    that compiled and then *crashed* is not this — that is a result — but one
    that compiled and then fell back to eager is, since the epochs it
    contributed to the median are then a mixture of both arms.

    :param arm: which arm produced the record.
    :param record: what its wrapper wrote.
    :return: the diagnostic, or None when the arm is what it claims.
    """
    compiled = record.get("compiled")
    if arm == COMPILED and compiled is not True:
        if record.get("graph_installed"):
            return (
                f"the {arm} arm installed a graph and finished eager: the "
                f"compiler backend failed at some forward and the epochs it "
                f"timed are not all compiled ones"
            )
        return (
            f"the {arm} arm reports compiled={compiled!r}: torch.compile "
            f"installed no graph, so this repeat times eager against eager"
        )
    if arm == EAGER and compiled is not False:
        return (
            f"the {arm} arm reports compiled={compiled!r}: "
            f"{runtime.COMPILE_DISABLE_VARIABLE} did not reach it"
        )

    return None


def log(message: str) -> None:
    stamp = datetime.datetime.now().isoformat(timespec="seconds")
    print(f"[{stamp}] {message}", flush=True)


def execute(run: Run, out: pathlib.Path) -> dict[str, Any]:
    """Run one arm and read back what it recorded, crash included.

    :param run: the arm's turn.
    :param out: the directory its log goes in.
    :return: the wrapper's record, with the arm, repeat, exit status and log
        added; `completed` is False where the arm died.
    """
    logfile = out / f"train_{run.arm}_{run.repeat}.log"
    log(f"START {run.arm} repeat {run.repeat} -> {logfile.name}")
    with logfile.open("w") as handle:
        completed = subprocess.run(
            run.command,
            env=run.env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            cwd=REPO,
        )

    if run.metrics.exists():
        record: dict[str, Any] = json.loads(run.metrics.read_text())
    else:
        # The wrapper writes its file in a `finally`, so a missing one means
        # the process died before reaching it — an import error, or a kill.
        record = {
            "compiled": None,
            "graph_installed": None,
            "completed": False,
            "error": f"no metrics were written; see {logfile.name}",
            "epochs": {},
            "context": {},
        }

    record.update(
        arm=run.arm,
        repeat=run.repeat,
        returncode=completed.returncode,
        log=logfile.name,
        switch_failure=switch_failure(run.arm, record),
    )

    if record["completed"]:
        log(f"DONE  {run.arm} repeat {run.repeat}")
    else:
        log(f"FAIL  {run.arm} repeat {run.repeat}: {record['error']}")

    return record


def main() -> None:
    args = read_args()

    unsupported = unsupported_machine()
    if unsupported is not None:
        raise SystemExit(
            f"this machine cannot run the benchmark: {unsupported}"
        )

    out = pathlib.Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / ".gitignore").write_text("*\n")

    config = out / "config.toml"
    config.write_text(
        benchmark_config(pathlib.Path(args.config).resolve(), args.epochs)
    )
    limit = args.limit if args.limit else None

    runs = plan(
        config, out, limit=limit, repeats=args.repeats, base_env=os.environ
    )
    log(f"{len(runs)} runs, {args.epochs} epochs each, limit {limit}")

    results = []
    for run in runs:
        run.metrics.unlink(missing_ok=True)
        results.append(execute(run, out))

    destination = out / "run.json"
    destination.write_text(
        json.dumps(
            {
                "config": str(config),
                "epochs": args.epochs,
                "limit": limit,
                "repeats": args.repeats,
                "runs": results,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    (out / "model.pt").unlink(missing_ok=True)
    log(f"wrote {destination}")
    print(
        "\nnow: pdm run python scripts/compile_benchmark/compare_arms.py "
        f"{destination}"
    )


if __name__ == "__main__":
    main()
