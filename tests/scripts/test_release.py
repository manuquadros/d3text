"""The release script refuses rather than half-releasing.

Every check here guards a step that cannot be undone once it is published: a
tag is immutable the moment anyone fetches it, and a changelog commit with no
tag beside it is a release that does not exist. The classification of a commit
subject is pinned too, because it decides whether a release happens at all.
"""

import importlib.util
import pathlib

import pytest

_SCRIPT = pathlib.Path(__file__).resolve().parents[2] / "scripts/release.py"


def _load_release():
    """The release script as a module, without putting `scripts/` on the path.

    Every name under `scripts/` is a top-level one, so importing by path keeps
    the whole directory from shadowing installed packages for the rest of the
    session.
    """
    spec = importlib.util.spec_from_file_location(_SCRIPT.stem, _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


release = _load_release()


@pytest.mark.parametrize(
    "subject",
    [
        "build: pin ruff",
        "chore: tidy",
        "ci: publish the docs site",
        "test: cover the loader",
        "docs: explain the seed",
        "build(deps): bump torch",
    ],
)
def test_these_subjects_release_nothing(subject):
    assert release.INTERNAL_RE.match(subject)


@pytest.mark.parametrize(
    "subject",
    [
        "feat: wire a tagger's spans into the report",
        "fix: seed the noise pool",
        "perf: batch by chunk budget",
        "refactor: split the model module",
        # A break is a release even under an otherwise internal prefix:
        # dropping a Python version is not invisible to anyone installing.
        "build!: require Python 3.12",
    ],
)
def test_these_subjects_are_worth_a_release(subject):
    assert not release.INTERNAL_RE.match(subject)


def test_a_release_is_cut_from_main_only(monkeypatch):
    """A tag made on a branch names commits that may never reach `main`, and
    it cannot be moved once it is pushed."""
    monkeypatch.setattr(release, "_git", lambda *a, **k: "a-side-branch")

    with pytest.raises(release.ReleaseError, match="HEAD is on"):
        release.release(dry_run=True, push=False, version="v9.9.9")


def test_a_version_git_cliff_could_not_compute_is_refused(monkeypatch):
    """`--bumped-version` printing a diagnostic rather than a version would
    otherwise become the tag name."""
    monkeypatch.setattr(release, "_git", lambda *a, **k: release.BRANCH)
    monkeypatch.setattr(release, "current_tag", lambda: "v0.1.0")

    with pytest.raises(release.ReleaseError, match="not a vX.Y.Z version"):
        release.release(dry_run=True, push=False, version="0.2")


def test_an_existing_tag_stops_the_release_before_anything_changes(
    monkeypatch,
):
    """Reaching `git tag` with a name already taken would fail *after* the
    changelog commit had landed, leaving a release commit with no release."""
    monkeypatch.setattr(release, "current_tag", lambda: "v0.1.0")
    monkeypatch.setattr(
        release,
        "_git",
        lambda *args, **kwargs: (
            release.BRANCH if args[0] == "symbolic-ref" else "v0.2.0"
        ),
    )

    with pytest.raises(release.ReleaseError, match="already exists"):
        release.release(dry_run=True, push=False, version="v0.2.0")


def test_nothing_to_release_is_not_an_error(monkeypatch, capsys):
    """Exit 1, not a raise: a scheduled or scripted caller asking for a
    release it turns out not to need has not failed."""
    monkeypatch.setattr(release, "current_tag", lambda: "v0.2.0")
    monkeypatch.setattr(release, "_git", lambda *a, **k: release.BRANCH)

    assert release.release(dry_run=True, push=False, version="v0.2.0") == 1
    assert "nothing to release" in capsys.readouterr().out
