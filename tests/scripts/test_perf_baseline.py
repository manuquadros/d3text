"""``perf_baseline.sh``: arm B's chrome-trace output path.

`export_chrome_trace` stages a `.gz` destination's whole JSON trace in a
temp file under `TMPDIR`, so arm B hands torch a plain `.json` path and the
script gzips it into the output directory. The closing `wrote` summary line
names the trace only when one exists.
"""

import gzip
import os
import pathlib
import re
import subprocess

_SCRIPT = (
    pathlib.Path(__file__).resolve().parents[2]
    / "scripts/benchmarks/perf_baseline.sh"
)


def _arm_b_output() -> str:
    """Return arm B's OUTPUT argument (the `-prof` `run train` call)."""
    text = _SCRIPT.read_text()
    match = re.search(
        r'run train "\$OUT/baseline\.toml" "(\$OUT/[^"]+)"[^\n]*\\\n'
        r'\s*--limit "\$LIMIT" -prof',
        text,
    )
    assert match, "could not find arm B's `run train ... -prof` call"
    return match.group(1)


def test_arm_b_output_named_in_summary_line() -> None:
    """The closing `wrote` line must name arm B's trace file."""
    output = _arm_b_output()
    filename = output.rsplit("/", 1)[-1]
    text = _SCRIPT.read_text()
    match = re.search(r'echo "wrote [^\n]*"', text)
    assert match, "could not find the closing `wrote` summary line"
    assert filename in match.group(0)


_STUB_PDM = """#!/usr/bin/env bash
# `run train CONFIG OUTPUT ... -prof`: log OUTPUT, write $STUB_TRACE there.
if [[ "$1 $2" == "run train" && " $* " == *" -prof "* ]]; then
  echo "$4" > "$STUB_LOG"
  if [[ -n "${STUB_TRACE:-}" ]]; then printf '%s' "$STUB_TRACE" > "$4"; fi
  exit "${STUB_EXIT:-0}"
fi
[[ "$1 $2" == "run train" ]] && exit 0
exit 1
"""


def _run_script(
    tmp_path: pathlib.Path,
    trace: str | None,
    failing_gzip: bool = False,
    exit_status: int = 0,
) -> tuple[subprocess.CompletedProcess[str], pathlib.Path, str]:
    """Run the real script against stub `pdm` and `nvidia-smi`.

    :return: the finished process, the output dir, and arm B's OUTPUT.
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    pdm = bin_dir / "pdm"
    pdm.write_text(_STUB_PDM)
    smi = bin_dir / "nvidia-smi"
    smi.write_text("#!/usr/bin/env bash\nexit 0\n")
    stubs = [pdm, smi]
    if failing_gzip:
        stubs.append(bin_dir / "gzip")
        stubs[-1].write_text("#!/usr/bin/env bash\nexit 1\n")
    for stub in stubs:
        stub.chmod(0o755)
    config = tmp_path / "config.toml"
    config.write_text("num_epochs = 9\npatience = 9\n")
    out = tmp_path / "out"
    out.mkdir()
    (out / "prof.trace.json.gz").write_bytes(gzip.compress(b"stale"))
    log = tmp_path / "arm_b_output"
    env = {
        **os.environ,
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "PDM": str(pdm),
        "CONFIG": str(config),
        "FORCE_DIRTY": "1",
        "STUB_LOG": str(log),
    }
    env.pop("STUB_TRACE", None)
    env.pop("STUB_EXIT", None)
    if trace is not None:
        env["STUB_TRACE"] = trace
    if exit_status != 0:
        env["STUB_EXIT"] = str(exit_status)
    proc = subprocess.run(
        ["bash", str(_SCRIPT), str(out)],
        cwd=_SCRIPT.parents[2],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    return proc, out, log.read_text().strip()


def test_arm_b_torch_output_is_uncompressed(tmp_path: pathlib.Path) -> None:
    """Arm B hands torch a non-`.gz` path.

    A `.gz` path makes torch stage the whole JSON trace under `TMPDIR`.
    """
    _, _, arm_b_output = _run_script(tmp_path, trace="{}")
    assert not arm_b_output.endswith(".gz")


def test_arm_b_trace_gzipped_in_place(tmp_path: pathlib.Path) -> None:
    """The script gzips arm B's trace into the output dir, keeping no `.json`."""
    payload = '{"traceEvents": []}'
    proc, out, arm_b_output = _run_script(tmp_path, trace=payload)
    gz = out / "prof.trace.json.gz"
    assert gzip.decompress(gz.read_bytes()).decode() == payload
    assert not pathlib.Path(arm_b_output).exists()
    assert "prof.trace.json.gz" in proc.stdout.splitlines()[-1]


def test_missing_trace_not_claimed(tmp_path: pathlib.Path) -> None:
    """With no trace written, the closing line omits it and warns.

    The output dir holds a trace from an earlier run, which must not count.
    """
    proc, _, _ = _run_script(tmp_path, trace=None)
    wrote = proc.stdout.splitlines()[-1]
    assert wrote.startswith("wrote ")
    assert "prof.trace" not in wrote
    assert "arm B wrote no trace" in proc.stderr


def test_failed_gzip_keeps_json(tmp_path: pathlib.Path) -> None:
    """A failed gzip leaves the `.json` trace, and only that is claimed."""
    proc, out, _ = _run_script(tmp_path, trace="{}", failing_gzip=True)
    assert (out / "prof.trace.json").read_text() == "{}"
    assert not (out / "prof.trace.json.gz").exists()
    wrote = proc.stdout.splitlines()[-1]
    assert "prof.trace.json," in wrote
    assert "prof.trace.json.gz" not in wrote


def test_train_exit_nonzero_moves_trace_aside(tmp_path: pathlib.Path) -> None:
    """When train exits non-zero, any partial trace is moved aside, not claimed.

    If train dies partway through writing the trace (e.g., ENOSPC), the
    `.json` file is left incomplete. Arm B must not gzip it or claim it in
    the closing `wrote` line. Instead, it is moved to `.partial.json` for
    debugging and a warning names the exit status.
    """
    payload = '{"traceEvents": []}'
    proc, out, _ = _run_script(tmp_path, trace=payload, exit_status=42)
    # Partial trace exists, complete trace files do not.
    partial = out / "prof.trace.partial.json"
    assert partial.read_text() == payload
    assert not (out / "prof.trace.json").exists()
    assert not (out / "prof.trace.json.gz").exists()
    # The closing `wrote` line does not mention the trace.
    wrote = proc.stdout.splitlines()[-1]
    assert wrote.startswith("wrote ")
    assert "prof.trace" not in wrote
    # stderr names the exit status and the partial file.
    assert "exit status 42" in proc.stderr
    assert "prof.trace.partial.json" in proc.stderr
