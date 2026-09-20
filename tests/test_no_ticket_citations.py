"""No file in this repository cites an identifier from the backlog.

The backlog is a separate git repository, and a ticket's shard is deleted
when a fix closes it, so an identifier written into a comment, docstring or
page here becomes a reference a reader of this repository cannot follow.
The directory names past experiments left behind carry no hyphen before
their digits, so the pattern below reads those as the paths they are.
"""

import pathlib
import subprocess

_ROOT = pathlib.Path(__file__).resolve().parent.parent

# An area prefix, a hyphen and two digits, as the backlog allocates them.
_CITATION = r"\b(feat|bug|arch|doc|dec|perf|test|data)-[0-9]{2}\b"


def _git(*args: str) -> str:
    """Run git at the repository root, tolerating grep's "nothing matched".

    `git grep` exits 1 when nothing matches, which is the passing case here.
    Every other non-zero exit is a broken invocation, and must not reach the
    caller as an empty result -- that would be indistinguishable from a
    clean tree.
    """
    done = subprocess.run(
        ["git", *args], cwd=_ROOT, capture_output=True, text=True
    )
    assert done.returncode in (0, 1), done.stderr
    return done.stdout


def test_no_tracked_file_cites_a_ticket_id() -> None:
    """Pins the absence of backlog identifiers across the tracked tree.

    The first assertion is not redundant with the second: a search that
    found nothing and a search that looked at nothing produce the same empty
    output, so without a listing that has to be populated, a checkout git
    cannot read would report a clean tree.
    """
    assert len(_git("ls-files").split()) > 100, "the listing broke"
    assert _git("grep", "-nEi", "-e", _CITATION) == ""
