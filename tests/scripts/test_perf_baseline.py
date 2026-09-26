"""``perf_baseline.sh``: arm B's chrome-trace output path.

Parses the script text rather than running it. `export_chrome_trace` gzips
its output only when the destination path ends in `.gz`, so arm B's OUTPUT
must carry that suffix, and the closing `wrote` summary line must name it.
"""

import pathlib
import re

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


def test_arm_b_output_ends_in_gz() -> None:
    """`export_chrome_trace` gzips only when the path ends in `.gz`."""
    assert _arm_b_output().endswith(".gz")


def test_arm_b_output_named_in_summary_line() -> None:
    """The closing `wrote` line must name arm B's trace file."""
    output = _arm_b_output()
    filename = output.rsplit("/", 1)[-1]
    text = _SCRIPT.read_text()
    match = re.search(r'echo "wrote [^\n]*"', text)
    assert match, "could not find the closing `wrote` summary line"
    assert filename in match.group(0)
