"""Guards on where the data blobs are looked for.

The blobs ship through the Hugging Face Hub, so the directory holding them
has to be the same one whether this package is installed editable or copied
into `site-packages`. A package-relative answer is not: it moves with the
install, the downloader keeps writing to the checkout, and every read of a
blob that is in no git raises `FileNotFoundError` on a fresh install.
"""

import pathlib

import pytest

from brenda_references import data_paths
from brenda_references.config import config


def test_env_override_wins(monkeypatch, tmp_path: pathlib.Path) -> None:
    monkeypatch.setenv("BRENDA_DATA_DIR", str(tmp_path))

    assert data_paths.resolve_data_dir() == tmp_path


def test_env_override_expands_user(monkeypatch) -> None:
    monkeypatch.setenv("BRENDA_DATA_DIR", "~/brenda-blobs")

    assert data_paths.resolve_data_dir() == pathlib.Path.home() / "brenda-blobs"


def test_unfilled_package_dir_is_not_the_answer(
    monkeypatch, tmp_path: pathlib.Path
) -> None:
    """A package directory carrying only the manifest must be passed over.

    This is the fresh-install regression: a non-editable install's `data/`
    holds the manifest and nothing else, and resolving to it sends the
    loader somewhere `pull_data.py` never writes.
    """
    package_data = tmp_path / "site-packages" / "brenda_references" / "data"
    package_data.mkdir(parents=True)
    (package_data / "SHA256SUMS").write_text("")
    monkeypatch.delenv("BRENDA_DATA_DIR", raising=False)
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "share"))
    monkeypatch.setattr(data_paths, "_LEGACY_DIR", package_data)

    assert data_paths.resolve_data_dir() == (
        tmp_path / "share" / "brenda-references"
    )


def test_filled_package_dir_is_kept(
    monkeypatch, tmp_path: pathlib.Path
) -> None:
    """An editable checkout that already holds the blobs keeps reading them.

    Pinning this is what makes the change a no-op for an existing setup
    rather than a 1.85 GB re-download.
    """
    checkout_data = tmp_path / "checkout" / "data"
    checkout_data.mkdir(parents=True)
    (checkout_data / "documents.json").write_text("")
    monkeypatch.delenv("BRENDA_DATA_DIR", raising=False)
    monkeypatch.setattr(data_paths, "_LEGACY_DIR", checkout_data)

    assert data_paths.resolve_data_dir() == checkout_data


@pytest.mark.parametrize("home", ["/xdg", None])
def test_default_dir_is_outside_the_package(monkeypatch, home) -> None:
    if home is None:
        monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    else:
        monkeypatch.setenv("XDG_DATA_HOME", home)

    assert data_paths._LEGACY_DIR not in data_paths._default_dir().parents


def test_corpus_files_covers_every_file_a_split_is_built_from() -> None:
    """The precompute default is the set `load_split` actually reads.

    The two sets drifting apart is not a cosmetic difference: a split is
    loaded with a block of each noise pool appended, so a pool the default
    omits is a document no store holds and every consumer silently drops.
    """
    assert set(data_paths.corpus_files()) == {
        data_paths.split_path("training"),
        data_paths.split_path("validation"),
        data_paths.split_path("test"),
        data_paths.noise_pool_path("psycholinguistics"),
        data_paths.noise_pool_path("enzyme_negative"),
    }


def test_the_file_names_come_from_the_configuration() -> None:
    """A name changed in `config.toml` moves the path the loaders read."""
    configured = config["datasets"]
    assert (
        data_paths.split_path("validation").name
        == configured["splits"]["validation"]
    )
    assert (
        data_paths.noise_pool_path("enzyme_negative").name
        == configured["noise_pools"]["enzyme_negative"]
    )


def test_an_unconfigured_split_is_named_rather_than_missing() -> None:
    """A typo raises here, not as a `FileNotFoundError` far downstream."""
    with pytest.raises(KeyError, match="configured: training"):
        data_paths.split_path("train")
