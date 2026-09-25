"""The residency benchmark's arms, equivalence check and source regime record.

Written out rather than borrowed from the live method, an arm can drift from
it, and a release the shipped method makes and an arm skips can be worth a
whole `[chunks, WINDOW_LENGTH, embedding]` block of the peak the script
reports. The regime record is what tells a forward-path run from a hit-path
one in the JSON, so it has to credit each arm with what it actually reads.
"""

import importlib.util
import math
import pathlib
import types
import weakref

import lmdb
import pytest
import torch
from d3text.embeddings_store import StoreProvenance, write_provenance
from d3text.models.base import (
    ByteBudgetCache,
    Model,
    cpu_cache_key,
    document_token_count,
)
from d3text.models.config import ModelConfig
from d3text.utils.utils import aggregate_embeddings

_SCRIPT = (
    pathlib.Path(__file__).resolve().parents[2]
    / "scripts"
    / "benchmarks"
    / "bench_embedding_residency.py"
)


def _load():
    """Load the benchmark by file path, keeping `scripts/` off `sys.path`.

    Every name under `scripts/` is top-level, so putting it on the path would
    shadow installed packages for the rest of the session.
    """
    spec = importlib.util.spec_from_file_location(_SCRIPT.stem, _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bench = _load()

_BASE_MODEL = ModelConfig(model_class="NERClassificationModel").base_model


def _stub_module(cache_on: bool, store: str | None) -> types.SimpleNamespace:
    """A `d3text.models.base` stand-in whose store opens whenever named."""
    module = types.SimpleNamespace(
        cpu_embeddings_cache=object() if cache_on else None,
        mconfig=types.SimpleNamespace(
            embeddings_store=({_BASE_MODEL: store} if store else {})
        ),
    )
    module.embeddings_store = lambda base_model: (
        object() if module.mconfig.embeddings_store.get(base_model) else None
    )
    return module


def _item(pmid, n_chunks, token):
    return {
        "id": torch.tensor(pmid),
        "doc_id": torch.zeros(n_chunks, dtype=torch.uint8),
        "sequence": {
            "input_ids": torch.zeros(n_chunks, token, dtype=torch.long),
            "attention_mask": torch.ones(n_chunks, token, dtype=torch.long),
        },
    }


def test_off_forces_the_cache_and_store_off_regardless_of_config() -> None:
    """A machine's `config.toml` can set either knob; `off` must win over
    both so the benchmark measures the base-model forward, not a cache or
    store read."""
    stub = _stub_module(cache_on=True, store="/mnt/embeddings")

    info = bench.select_source_regime(
        bench.SOURCE_OFF, _BASE_MODEL, module=stub
    )

    assert stub.cpu_embeddings_cache is None
    assert stub.mconfig.embeddings_store == {}
    assert info == {
        "regime": bench.SOURCE_OFF,
        "cpu_arm_cache": False,
        "cpu_arm_store": False,
        "gpu_arm_cache": False,
        "gpu_arm_store": False,
    }


def test_configured_leaves_the_machine_config_untouched() -> None:
    """`configured` measures the hit path on purpose, so it must not force
    either knob off."""
    stub = _stub_module(cache_on=True, store="/mnt/embeddings")

    info = bench.select_source_regime(
        bench.SOURCE_CONFIGURED, _BASE_MODEL, module=stub
    )

    assert stub.cpu_embeddings_cache is not None
    assert stub.mconfig.embeddings_store == {_BASE_MODEL: "/mnt/embeddings"}
    assert info["regime"] == bench.SOURCE_CONFIGURED


def test_source_help_does_not_overpromise_the_hit_path(monkeypatch, capsys):
    """`--source`'s help must say `configured` only measures the hit path
    where a source is live, not unconditionally -- a store that fails to
    open, was written for another base model, or a zero cache budget
    silently runs the same forward `off` does."""
    monkeypatch.setattr("sys.argv", ["bench_embedding_residency.py", "--help"])
    with pytest.raises(SystemExit):
        bench.main()

    help_text = " ".join(capsys.readouterr().out.split())
    assert "wherever a source is live" in help_text
    assert "measures the hit path instead" not in help_text


def test_a_configured_store_is_recorded_as_symmetric_between_the_arms(
    stub, monkeypatch
) -> None:
    """Both arms are credited with a store that opens, and both do read it.

    They share one body, so a record crediting only one arm would pass off a
    like-for-like comparison of hits as a store read timed against a forward.
    """
    hidden, token = 4, 64
    read_by: list[str] = []
    arm = ""

    class FakeStore:
        def get(self, pubmed_id, expected_tokens):
            read_by.append(arm)
            return torch.zeros(expected_tokens, hidden)

    monkeypatch.setattr("d3text.models.base.cpu_embeddings_cache", None)
    monkeypatch.setattr(
        "d3text.models.base.embeddings_store", lambda _base_model: FakeStore()
    )

    info = bench.select_source_regime(bench.SOURCE_CONFIGURED, _BASE_MODEL)

    assert info["cpu_arm_store"] is True
    assert info["gpu_arm_store"] is True

    model = stub(
        Model,
        device="cpu",
        amp_dtype=torch.bfloat16,
        config=ModelConfig(model_class="NERClassificationModel"),
    )

    for arm in ("cpu", "gpu"):
        getattr(bench, f"{arm}_impl")(model, [_item(100, 1, token)])

    assert read_by == ["cpu", "gpu"]


def test_no_store_configured_is_symmetric() -> None:
    """No store set: neither arm is credited with one; both share the cache."""
    stub = _stub_module(cache_on=True, store=None)

    info = bench.select_source_regime(
        bench.SOURCE_CONFIGURED, _BASE_MODEL, module=stub
    )

    assert info["cpu_arm_store"] is False
    assert info["gpu_arm_store"] is False
    assert info["cpu_arm_cache"] == info["gpu_arm_cache"] is True


@pytest.mark.parametrize(
    "written_by", [None, "another/base-model"], ids=["unopenable", "mismatched"]
)
def test_a_configured_store_that_does_not_open_is_recorded_as_absent(
    written_by, tmp_path, monkeypatch
) -> None:
    """A store the config names but `embeddings_store` refuses is no source.

    Both arms then run the forward, so crediting them with the configured
    store would label a forward-path number as a hit-path one.
    """
    path = tmp_path / "store"
    if written_by is None:
        # A missing path is built by the run rather than refused, so the one
        # that cannot open is a file where the directory should be.
        path.write_bytes(b"")
    else:
        with lmdb.open(str(path), map_size=2**20) as env:
            write_provenance(
                env,
                StoreProvenance(
                    base_model=written_by, max_length=512, stride=20
                ),
            )
    monkeypatch.setattr(
        bench.M.mconfig, "embeddings_store", {_BASE_MODEL: str(path)}
    )
    # `embeddings_store` is cached per base model, so a store another test
    # opened, or failed to, must not answer for this one.
    bench.M.embeddings_store.cache_clear()
    try:
        info = bench.select_source_regime(bench.SOURCE_CONFIGURED, _BASE_MODEL)
    finally:
        bench.M.embeddings_store.cache_clear()

    assert info["cpu_arm_store"] is False
    assert info["gpu_arm_store"] is False


@pytest.mark.parametrize("arm", ["cpu_impl", "gpu_impl"])
def test_neither_arm_pads_while_the_hidden_states_are_reachable(
    arm, stub, monkeypatch
):
    """Both arms release the forward's output where the shipped method does.

    Reachability is Python-level, so the on-device arm is checkable on a CPU.
    Only that arm would pay for a miss, the other's copy being host memory by
    then, but the two share one body and are pinned together.
    """
    hidden, token = 4, 64
    forward_output: list[weakref.ref] = []
    alive_when_padding: list[bool] = []

    def fake_base_model(input_ids, attention_mask):
        n_seq, seq_len = input_ids.shape
        hidden_states = torch.zeros(n_seq, seq_len, hidden)
        forward_output.append(weakref.ref(hidden_states))
        # The tensor to weakref is the one `.detach()` hands back, and it must
        # not be reachable from the object holding it, or the reference cycle
        # would leave the release to the collector.
        return types.SimpleNamespace(
            last_hidden_state=types.SimpleNamespace(
                detach=lambda: hidden_states
            )
        )

    real_pad_sequence = bench.pad_sequence

    def recording_pad_sequence(*args, **kwargs):
        alive_when_padding.append(forward_output[0]() is not None)
        return real_pad_sequence(*args, **kwargs)

    monkeypatch.setattr("d3text.models.base.cpu_embeddings_cache", None)
    monkeypatch.setattr(
        "d3text.models.base.embeddings_store", lambda _base_model: None
    )
    monkeypatch.setattr(bench, "pad_sequence", recording_pad_sequence)

    model = stub(
        Model,
        device="cpu",
        amp_dtype=torch.bfloat16,
        base_model=fake_base_model,
        config=ModelConfig(model_class="NERClassificationModel"),
    )

    embeddings, masks = getattr(bench, arm)(model, [_item(100, 2, token)])

    assert alive_when_padding == [False]
    assert embeddings.shape[0] == 1 and embeddings.shape[-1] == hidden
    assert masks.shape == embeddings.shape[:2]


@pytest.mark.parametrize("arm", ["cpu_impl", "gpu_impl"])
def test_neither_arm_moves_a_hit_before_the_hidden_states_are_released(
    arm, stub, monkeypatch, watch_device_moves
):
    """Both arms keep hits on the host through the forward, as the method does.

    An on-device arm that moved them earlier would charge the forward, the
    phase the peak comparison turns on, a residency the shipped method does
    not have. On the round-trip arm the move names the host and costs
    nothing, but the two share one body and are pinned together.
    """
    hidden, token = 4, 64
    cached_doc, stored_doc, fresh_doc = 100, 200, 300
    rows = document_token_count(_item(stored_doc, 1, token))
    forward_output: list[weakref.ref] = []
    released_at_move: list[bool] = []

    def watched(tensor):
        return watch_device_moves(
            tensor,
            lambda: released_at_move.append(
                bool(forward_output) and forward_output[0]() is None
            ),
        )

    cache = ByteBudgetCache(max_bytes=2**20)
    cache.set(
        cpu_cache_key(
            ModelConfig(model_class="NERClassificationModel").base_model,
            cached_doc,
        ),
        watched(torch.zeros(rows, hidden, dtype=torch.float16)),
    )

    class FakeStore:
        def get(self, pubmed_id, expected_tokens):
            if pubmed_id != stored_doc:
                return None
            return watched(
                torch.zeros(expected_tokens, hidden, dtype=torch.bfloat16)
            )

    def fake_base_model(input_ids, attention_mask):
        n_seq, seq_len = input_ids.shape
        hidden_states = torch.zeros(n_seq, seq_len, hidden)
        forward_output.append(weakref.ref(hidden_states))
        return types.SimpleNamespace(
            last_hidden_state=types.SimpleNamespace(
                detach=lambda: hidden_states
            )
        )

    monkeypatch.setattr("d3text.models.base.cpu_embeddings_cache", cache)
    monkeypatch.setattr(
        "d3text.models.base.embeddings_store", lambda _base_model: FakeStore()
    )

    model = stub(
        Model,
        device="cpu",
        amp_dtype=torch.float16,
        base_model=fake_base_model,
        config=ModelConfig(model_class="NERClassificationModel"),
    )

    getattr(bench, arm)(
        model,
        [_item(doc, 1, token) for doc in (cached_doc, stored_doc, fresh_doc)],
    )

    assert len(forward_output) == 1
    assert released_at_move == [True, True]


def test_a_declined_write_never_copies_to_the_host(stub, monkeypatch):
    """Mirrors the fix in `Model.get_token_embeddings`: a document too big
    for what is left must not pay the device-to-host copy before `set`
    declines it on its own accounting -- the drift this script exists to
    avoid.

    The budget leaves 4 bytes after the first document -- not zero, so
    `full()` alone would not short-circuit the second -- and 4 is still
    less than the second, same-sized document costs.
    """
    hidden = 4

    def fake_base_model(input_ids, attention_mask):
        n_seq, seq_len = input_ids.shape
        return types.SimpleNamespace(
            last_hidden_state=torch.zeros(n_seq, seq_len, hidden)
        )

    cache = ByteBudgetCache(max_bytes=12)
    monkeypatch.setattr("d3text.models.base.cpu_embeddings_cache", cache)
    monkeypatch.setattr(
        "d3text.models.base.embeddings_store", lambda _base_model: None
    )
    monkeypatch.setattr(
        bench, "aggregate_embeddings", lambda outs, masks: outs[:, 0, :]
    )

    original_cpu = torch.Tensor.cpu
    calls: list[int] = []

    def counting_cpu(self, *args, **kwargs):
        calls.append(1)
        return original_cpu(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "cpu", counting_cpu)

    model = stub(
        Model,
        device="cpu",
        amp_dtype=torch.bfloat16,
        base_model=fake_base_model,
        config=ModelConfig(model_class="NERClassificationModel"),
    )

    # 1 row x 4 columns x 2 bytes (bfloat16) = 8, under the 12-byte budget.
    bench.gpu_impl(model, [_item(700, 1, 6)])
    assert cache.get(cpu_cache_key(_BASE_MODEL, 700)) is not None
    assert len(calls) == 1

    # 4 bytes remain -- not full, but not enough for another 8-byte
    # document. Must be declined without paying for the copy that used to
    # run before `set`'s own check.
    bench.gpu_impl(model, [_item(701, 1, 6)])
    assert cache.get(cpu_cache_key(_BASE_MODEL, 701)) is None
    assert len(calls) == 1


@pytest.mark.gpu
def test_the_on_device_arm_aggregates_against_the_host_mask(stub, monkeypatch):
    """`gpu_impl` must hand `aggregate_embeddings` the same CPU mask the
    round-trip arm hands it, never the copy moved onto the card for the
    forward -- `aggregate_embeddings` requires a host mask and raises on a
    device one, so a device mask handed back here fails every CUDA run of
    this arm, not just this assertion.
    """
    hidden, token = 4, 64
    aggregated_on: list[str] = []
    real_aggregate = aggregate_embeddings

    def recording_aggregate(outs, masks):
        aggregated_on.append(masks.device.type)
        return real_aggregate(outs, masks)

    def fake_base_model(input_ids, attention_mask):
        n_seq, seq_len = input_ids.shape
        return types.SimpleNamespace(
            last_hidden_state=torch.zeros(n_seq, seq_len, hidden, device="cuda")
        )

    monkeypatch.setattr("d3text.models.base.cpu_embeddings_cache", None)
    monkeypatch.setattr(
        "d3text.models.base.embeddings_store", lambda _base_model: None
    )
    monkeypatch.setattr(bench, "aggregate_embeddings", recording_aggregate)

    model = stub(
        Model,
        device="cuda",
        amp_dtype=torch.float16,
        base_model=fake_base_model,
        config=ModelConfig(model_class="NERClassificationModel"),
    )

    bench.gpu_impl(model, [_item(100, 1, token)])

    assert aggregated_on == ["cpu"]


def _arm_output(values, masks=None, dtype=torch.bfloat16):
    embeddings = torch.tensor(values, dtype=dtype).reshape(1, -1, 1)
    if masks is None:
        masks = [True] * embeddings.shape[1]
    return embeddings, torch.tensor([masks], dtype=torch.bool)


def _equivalence(monkeypatch, pairs):
    """`equivalence` over arms that hand back batch `k`'s pair from `pairs`."""
    monkeypatch.setattr(bench, "cpu_impl", lambda _model, k: pairs[k][0])
    monkeypatch.setattr(bench, "gpu_impl", lambda _model, k: pairs[k][1])
    return bench.equivalence(object(), range(len(pairs)))


@pytest.mark.parametrize("nan_batch", [0, 1], ids=["first", "second"])
@pytest.mark.parametrize("nan_arm", ["cpu", "gpu"])
def test_a_nan_in_one_arm_is_recorded_as_disagreement(
    nan_arm, nan_batch, monkeypatch
) -> None:
    """A NaN must never read as agreement, whichever arm or batch it is in.

    `max(0.0, nan)` is `0.0`, so a running maximum loses a NaN that arrives
    as its second argument, and reports the very agreement the check exists
    to be able to refute.
    """
    pairs = []
    for batch in range(2):
        cpu, gpu = _arm_output([0.5, 1.0]), _arm_output([0.5, 1.0])
        if batch == nan_batch:
            (cpu if nan_arm == "cpu" else gpu)[0][0, 0, 0] = math.nan
        pairs.append((cpu, gpu))

    record = _equivalence(monkeypatch, pairs)

    assert math.isnan(record["max_abs_delta"])
    assert record["bit_identical"] is False


def test_signed_zeros_are_not_bit_identical(monkeypatch) -> None:
    """`-0.0 == 0.0`, so only a comparison of the raw bits can refute them."""
    record = _equivalence(
        monkeypatch, [(_arm_output([-0.0, 1.0]), _arm_output([0.0, 1.0]))]
    )

    assert record["max_abs_delta"] == 0.0
    assert record["bit_identical"] is False


def test_a_difference_in_the_masks_alone_is_not_bit_identical(
    monkeypatch,
) -> None:
    """Identical embeddings under different masks are not the same output."""
    record = _equivalence(
        monkeypatch,
        [
            (
                _arm_output([0.5, 0.0], masks=[True, False]),
                _arm_output([0.5, 0.0], masks=[True, True]),
            )
        ],
    )

    assert record["max_abs_delta"] == 0.0
    assert record["bit_identical"] is False


@pytest.mark.parametrize(
    "dtype",
    [torch.bfloat16, torch.float16, torch.float32],
    ids=["bf16", "fp16", "fp32"],
)
def test_identical_outputs_are_recorded_as_agreement(
    dtype, monkeypatch
) -> None:
    """The bit comparison must not refute outputs that really are identical."""
    values = torch.randn(2, 5, 3, generator=torch.Generator().manual_seed(0))
    masks = torch.ones(2, 5, dtype=torch.bool)
    pairs = [
        ((values.to(dtype), masks), (values.to(dtype).clone(), masks.clone()))
        for _ in range(2)
    ]

    assert _equivalence(monkeypatch, pairs) == {
        "bit_identical": True,
        "max_abs_delta": 0.0,
        "masks_equal": True,
        "shapes_equal": True,
    }


def test_an_oom_during_equivalence_is_recorded_not_raised(monkeypatch) -> None:
    """An OOM here must land in the JSON like a measured round's, not as an
    uncaught traceback with nothing written -- this phase runs before the
    measured rounds even start, so a budget sweep near the card's limit can
    hit it first."""

    def raises_oom(_model, _batches):
        raise torch.cuda.OutOfMemoryError("mock OOM")

    monkeypatch.setattr(bench, "equivalence", raises_oom)

    equiv, equiv_error = bench.run_equivalence(object(), range(2))

    assert equiv is None
    assert equiv_error == "equivalence OOM: mock OOM"


def test_equivalence_without_an_oom_is_passed_through(monkeypatch) -> None:
    """The success path must still return `equivalence`'s record, untouched,
    with no error recorded."""
    sentinel = {"bit_identical": True}
    monkeypatch.setattr(bench, "equivalence", lambda _model, _batches: sentinel)

    equiv, equiv_error = bench.run_equivalence(object(), range(2))

    assert equiv is sentinel
    assert equiv_error is None


def test_no_batch_s_outputs_outlive_it_into_the_next_batch_s_arms(
    monkeypatch,
) -> None:
    """Both arms of batch `k + 1` run with all four of batch `k`'s released.

    The phase runs outside the OOM guard, so an output held over adds a
    batch to what the card carries, and a budget near the limit would end the
    run with no JSON. Reachability is Python-level, so a CPU suffices.
    """
    outputs: list[tuple[int, weakref.ref]] = []
    held_over: list[int] = []

    def arm(_model, batch):
        held_over.append(
            sum(ref() is not None for k, ref in outputs if k < batch)
        )
        embeddings = torch.zeros(1, 2, 1)
        masks = torch.ones(1, 2, dtype=torch.bool)
        outputs.extend(
            (batch, weakref.ref(tensor)) for tensor in (embeddings, masks)
        )
        return embeddings, masks

    monkeypatch.setattr(bench, "cpu_impl", arm)
    monkeypatch.setattr(bench, "gpu_impl", arm)

    bench.equivalence(object(), range(3))

    assert len(outputs) == 12
    assert held_over == [0] * 6
