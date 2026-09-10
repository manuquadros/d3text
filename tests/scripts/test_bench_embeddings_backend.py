"""The LMDB arms of the backend benchmark must read the way the production
reader does. `bytes_to_tensor` was widened (`3e791c7`) to accept a
`memoryview` so a reader under `buffers=True` need not copy the mapped page
in; a call site that still wraps the read in `bytes(...)` pays an ~8-11 MiB
memcpy per document that the reader it is supposed to be timing does not.

The script cannot be imported directly: it loads a real transformer model
and opens a real HDF5 corpus file at module scope. So this pins the exact
`bytes_to_tensor(...)` call expressions from the source and evaluates each
against a stub `txn` whose `get` returns a `memoryview`, checking what
actually reaches `bytes_to_tensor`.
"""

import ast
import pathlib

_SCRIPT = (
    pathlib.Path(__file__).resolve().parents[2]
    / "scripts"
    / "benchmarks"
    / "bench_embeddings_backend.py"
)


def _bytes_to_tensor_call_sites() -> list[str]:
    source = _SCRIPT.read_text()
    tree = ast.parse(source)
    sites = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "bytes_to_tensor"
        ):
            segment = ast.get_source_segment(source, node)
            assert segment is not None
            sites.append(segment)
    return sites


class _StubTxn:
    def get(self, _key: bytes) -> memoryview:
        return memoryview(b"\x00" * 32)


def test_lmdb_reads_forward_the_mapped_page_not_a_copy() -> None:
    sites = _bytes_to_tensor_call_sites()
    assert len(sites) == 2

    for site in sites:
        received: list[object] = []
        namespace = {
            "bytes_to_tensor": lambda packed: received.append(packed),
            "txn": _StubTxn(),
            "k": "doc",
            "key": "doc",
        }
        exec(site, namespace)

        assert len(received) == 1, site
        assert isinstance(received[0], memoryview), site
        assert not isinstance(received[0], bytes), site
