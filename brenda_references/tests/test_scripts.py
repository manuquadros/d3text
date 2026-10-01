"""Guards on the ad-hoc scripts that read and write the data files."""

import hashlib
import pathlib

import pytest

from brenda_references import data_paths as package
from scripts import pull_data


def test_file_digest_matches_a_plain_sha256(tmp_path: pathlib.Path) -> None:
    """Pins the digest itself, not just that some string comes back."""
    target = tmp_path / "blob.bin"
    target.write_bytes(b"some bytes to hash")

    assert (
        pull_data.file_digest(target)
        == hashlib.sha256(b"some bytes to hash").hexdigest()
    )


def test_every_script_agrees_with_the_package_on_the_data_dir() -> None:
    """One home for the path, or a writer misses every reader.

    `pull_data.py` used to derive its own path from `__file__`, which is
    the checkout, while the loader derived it from the installed package,
    which under a non-editable install is `site-packages`. The two
    disagreed silently: the fetch reported success and every later read of
    a blob still raised `FileNotFoundError`. The same disagreement would
    send a regenerated split somewhere nothing loads.
    """
    assert pull_data.DATA_DIR == package.DATA_DIR


@pytest.mark.integration
def test_data_dir_holds_the_splits() -> None:
    assert (package.DATA_DIR / "training_data.csv").is_file()


def _run_main(monkeypatch: pytest.MonkeyPatch, *argv: str) -> dict[str, object]:
    """Run `pull_data.main` with the Hub stubbed; return the download kwargs."""
    import huggingface_hub

    seen: dict[str, object] = {}
    monkeypatch.setattr(
        huggingface_hub, "snapshot_download", lambda **kw: seen.update(kw)
    )
    monkeypatch.setattr(pull_data, "verify", lambda expected: [])
    monkeypatch.setattr("sys.argv", ["pull_data.py", *argv])
    assert pull_data.main() == 0
    return seen


def test_download_uses_the_pinned_hub_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A moving `main` would strand a checkout whose manifest names old blobs."""
    seen = _run_main(monkeypatch)

    assert seen["revision"] == pull_data.read_revision(package.HUB_REVISION)


def test_revision_flag_overrides_the_pin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    other = "0" * 40

    assert _run_main(monkeypatch, "--revision", other)["revision"] == other


def test_revision_flag_is_refused_with_check(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        "sys.argv", ["pull_data.py", "--check", "--revision", "0" * 40]
    )

    with pytest.raises(SystemExit):
        pull_data.main()

    assert "--revision has no effect with --check" in capsys.readouterr().err


@pytest.mark.parametrize("ref", ["main", "9acffbc", ""])
def test_read_revision_refuses_a_moving_ref(
    tmp_path: pathlib.Path, ref: str
) -> None:
    pin = tmp_path / "HUB_REVISION"
    pin.write_text(ref + "\n")

    with pytest.raises(SystemExit):
        pull_data.read_revision(pin)
