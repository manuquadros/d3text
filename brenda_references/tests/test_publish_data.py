"""`publish_data.py` re-pins from the commit its own upload made.

The Hub is faked: `huggingface_hub.HfApi` is replaced, so nothing here
touches the network.
"""

import hashlib
import pathlib
from typing import Any

import httpx
import huggingface_hub
import pytest

from scripts import publish_data

OLD_PIN = "a" * 40
NEW_OID = "b" * 40


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class FakeApi:
    """A Hub whose head is `head`; `create_commit` records its calls.

    `raises` makes the commit fail; `oid` overrides the id it returns, as
    huggingface_hub does with the head's id when every file is already there.
    """

    calls: list[dict[str, Any]] = []
    raises: Exception | None = None
    head = OLD_PIN
    oid = NEW_OID

    def repo_info(self, repo_id: str, **kwargs: Any) -> Any:
        return type("Info", (), {"sha": FakeApi.head})()

    def create_commit(self, repo_id: str, **kwargs: Any) -> Any:
        if FakeApi.raises is not None:
            raise FakeApi.raises
        FakeApi.calls.append({"repo_id": repo_id, **kwargs})
        return type("Info", (), {"oid": FakeApi.oid})()


@pytest.fixture
def hub(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> pathlib.Path:
    """A data dir of three blobs, its manifest and pin, and a fake Hub."""
    data = tmp_path / "data"
    data.mkdir()
    for name in ("a.json", "b.csv", "c.csv"):
        (data / name).write_bytes(name.encode())
    manifest = tmp_path / "SHA256SUMS"
    manifest.write_text(
        "".join(
            f"{_sha(n.encode())}  {n}\n" for n in ("a.json", "b.csv", "c.csv")
        )
    )
    pin = tmp_path / "HUB_REVISION"
    pin.write_text(OLD_PIN + "\n")
    monkeypatch.setattr(publish_data, "DATA_DIR", data)
    monkeypatch.setattr(publish_data, "MANIFEST", manifest)
    monkeypatch.setattr(publish_data, "HUB_REVISION", pin)
    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    FakeApi.calls = []
    FakeApi.raises = None
    FakeApi.head = OLD_PIN
    FakeApi.oid = NEW_OID
    return tmp_path


def _run(monkeypatch: pytest.MonkeyPatch, *argv: str) -> int:
    monkeypatch.setattr("sys.argv", ["publish_data.py", *argv])
    return publish_data.main()


def _state(root: pathlib.Path) -> tuple[bytes, bytes]:
    return (
        (root / "SHA256SUMS").read_bytes(),
        (root / "HUB_REVISION").read_bytes(),
    )


def test_one_changed_file_is_uploaded_and_repinned_to_the_commit(
    hub: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (hub / "data" / "b.csv").write_bytes(b"new rows")

    assert _run(monkeypatch, "-m", "redraw") == 0

    (call,) = FakeApi.calls
    assert [op.path_in_repo for op in call["operations"]] == ["b.csv"]
    assert call["parent_commit"] == OLD_PIN
    assert call["commit_message"] == "redraw"
    assert (hub / "SHA256SUMS").read_text() == (
        f"{_sha(b'a.json')}  a.json\n"
        f"{_sha(b'new rows')}  b.csv\n"
        f"{_sha(b'c.csv')}  c.csv\n"
    )
    assert (hub / "HUB_REVISION").read_text() == NEW_OID + "\n"


def test_dry_run_uploads_and_writes_nothing(
    hub: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (hub / "data" / "b.csv").write_bytes(b"new rows")
    before = _state(hub)

    assert _run(monkeypatch, "--dry-run") == 0

    assert FakeApi.calls == []
    assert _state(hub) == before


def test_nothing_changed_exits_1_without_uploading(
    hub: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    before = _state(hub)

    assert _run(monkeypatch, "-m", "noop") == 1

    assert FakeApi.calls == []
    assert _state(hub) == before


def test_a_missing_manifest_file_aborts_before_any_upload(
    hub: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (hub / "data" / "b.csv").unlink()
    (hub / "data" / "c.csv").write_bytes(b"changed")
    before = _state(hub)

    with pytest.raises(SystemExit) as exc:
        _run(monkeypatch, "-m", "x")

    assert exc.value.code not in (0, None)
    assert FakeApi.calls == []
    assert _state(hub) == before


def test_a_refused_commit_leaves_manifest_and_pin_untouched(
    hub: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (hub / "data" / "b.csv").write_bytes(b"new rows")
    before = _state(hub)
    response = httpx.Response(
        412, request=httpx.Request("POST", "https://hub.invalid/commit")
    )
    FakeApi.raises = huggingface_hub.errors.HfHubHTTPError(
        "stale parent", response=response
    )

    with pytest.raises(SystemExit) as exc:
        _run(monkeypatch, "-m", "x")

    assert exc.value.code not in (0, None)
    assert _state(hub) == before


def test_a_real_commit_message_is_required_to_upload(
    hub: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (hub / "data" / "b.csv").write_bytes(b"new rows")

    with pytest.raises(SystemExit):
        _run(monkeypatch)

    assert FakeApi.calls == []


def test_a_moved_head_is_refused_before_any_commit(
    hub: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The head moved past the pin: refuse rather than commit or re-pin.

    `parent_commit` alone cannot catch this: when every changed file is
    already identical on the Hub, huggingface_hub sends no commit and returns
    the head's id, which the script would then pin.
    """
    (hub / "data" / "b.csv").write_bytes(b"new rows")
    FakeApi.head = "c" * 40
    FakeApi.oid = FakeApi.head
    before = _state(hub)

    with pytest.raises(SystemExit) as exc:
        _run(monkeypatch, "-m", "x")

    assert exc.value.code not in (0, None)
    assert FakeApi.calls == []
    assert _state(hub) == before


def test_files_already_on_the_hub_repin_the_manifest_without_a_commit(
    hub: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The returned id is the pin itself, so no commit was made."""
    (hub / "data" / "b.csv").write_bytes(b"new rows")
    FakeApi.oid = OLD_PIN

    assert _run(monkeypatch, "-m", "x") == 0

    out = capsys.readouterr().out
    assert "Published" not in out
    assert OLD_PIN in out
    assert f"{_sha(b'new rows')}  b.csv" in (hub / "SHA256SUMS").read_text()
    assert (hub / "HUB_REVISION").read_text() == OLD_PIN + "\n"


def test_a_returned_id_that_is_not_a_full_sha_is_not_pinned(
    hub: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (hub / "data" / "b.csv").write_bytes(b"new rows")
    FakeApi.oid = "main"
    before = _state(hub)

    with pytest.raises(SystemExit) as exc:
        _run(monkeypatch, "-m", "x")

    assert exc.value.code not in (0, None)
    assert _state(hub) == before
