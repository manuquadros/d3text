"""The benchmark's LMDB arms hand `bytes_to_tensor` a `memoryview`, uncopied.

A `bytes(...)` wrap times a per-document copy the production reader never
makes. The script loads a model at import, so its call expressions are read
from source and evaluated against a stub `txn`.
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
