"""`main()` must drive `sync_doc_db` to completion.

`sync_doc_db` is a coroutine function: a console script calling it directly
builds an unawaited coroutine and exits having done nothing, so `main()`, the
`sync-doc-db` entry point, has to run it on an event loop.
"""

import asyncio

import brenda_references.brenda_references as bref


def test_main_runs_sync_doc_db_coroutine_to_completion(monkeypatch) -> None:
    calls: list[str] = []

    async def fake_sync_doc_db() -> None:
        # Proves the coroutine is actually awaited on an event loop, not
        # merely constructed and discarded: a bare call to an unawaited
        # coroutine function never reaches this line.
        await asyncio.sleep(0)
        calls.append("ran")

    monkeypatch.setattr(bref, "sync_doc_db", fake_sync_doc_db)

    bref.main(argv=[])

    assert calls == ["ran"]


def test_main_declared_as_the_sync_doc_db_console_script() -> None:
    import pathlib
    import tomllib

    pyproject = (
        pathlib.Path(__file__).parents[1]
        / "brenda_references"
        / "pyproject.toml"
    )
    with pyproject.open("rb") as handle:
        config = tomllib.load(handle)

    scripts = config["project"]["scripts"]
    assert scripts.get("sync-doc-db") == (
        "brenda_references.brenda_references:main"
    )
