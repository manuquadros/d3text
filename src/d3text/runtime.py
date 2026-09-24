"""Process-wide runtime configuration.

TF32, the matmul precision, the caching allocator, tokenizer parallelism, the
RNG seed and the log handler are all process-global and sticky, so setting them
at import time makes a run's numerics depend on import order. `configure()` is
called from a script's `main()`; everything else inherits torch's own defaults.
"""

import logging
import os
from typing import Any

import torch

from . import logs
from .models.config import MachineConfig, machine_config

logger = logging.getLogger(__name__)

#: Opts a run into `compile_model`; unset, every run trains eager. Opt-in
#: because compiling has not paid for this model: with the trunk training,
#: warmup cost minutes against a steady-state gain under one percent per epoch.
#: An environment variable rather than a `config.toml` key or CLI flag, on the
#: `D3TEXT_LOG_LEVEL` precedent: whether compiling pays is a property of the
#: machine and the invocation, not of the model config, and a model config is
#: shared across the machines that run it. Any non-empty value enables.
COMPILE_VARIABLE = "D3TEXT_COMPILE"


def has_bf16_hardware() -> bool:
    """Whether this GPU runs bfloat16 in silicon rather than by emulation.

    `torch.cuda.is_bf16_supported()` answers a different question and returns
    True on cards with no bf16 units at all. Asked by compute capability, which
    is readable on every torch version.

    :return: whether bf16 arithmetic is native here.
    """
    if not torch.cuda.is_available():
        return False

    return torch.cuda.get_device_capability() >= (8, 0)


def select_amp_dtype(device: str) -> torch.dtype:
    """Pick bf16 wherever it is safe, fp16 only where the backend demands it.

    CPU takes no hardware check: bf16 is software-emulated on every build, and
    it is what PyTorch's own CPU autocast defaults to — fp16's much narrower
    exponent range genuinely overflows CPU-scale activations that bf16,
    sharing fp32's exponent range, does not. Each GPU backend is then asked
    independently: compute capability is meaningless under HIP, so the
    device-name allowlist is the sole authority for ROCm.

    :param device: the device this model runs its forward pass on.
    :return: the autocast dtype to use.
    """
    if not device.startswith("cuda"):
        return torch.bfloat16

    is_rocm = getattr(torch.version, "hip", None) is not None

    if is_rocm:
        device_name = (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""
        )
        bf16_ok = any(k in device_name for k in ("MI200", "MI250", "MI3"))
    else:
        bf16_ok = has_bf16_hardware()

    return torch.bfloat16 if bf16_ok else torch.float16


def set_seed(seed: int) -> None:
    """Seed the global RNG every sampler and initialiser draws from.

    Separate from `configure` because a tuning sweep reseeds between trials:
    two configurations are only comparable if each starts from the same RNG
    state rather than inheriting whatever the trial before it left behind.

    :param seed: the value to seed with.
    """
    # Seeds the global generator that `data.g` hands to the samplers.
    torch.manual_seed(seed)


def configure(
    config: MachineConfig | None = None, *, seed: int | None = 42
) -> None:
    """Apply this machine's runtime settings, defaulting to `config.toml`.

    Call once from a script entry point, before any CUDA work: the caching
    allocator reads its environment variable when it first initialises and
    ignores it thereafter. Also installs the package's console log handler.

    :param config: the machine settings to apply; read from `config.toml` if
        omitted.
    :param seed: the global RNG seed; `None` leaves it untouched.
    """
    settings = machine_config() if config is None else config

    # Here rather than in each `main()` so a command cannot be written that
    # forgets it: the library logs instead of printing, so an unconfigured
    # process would run to completion in total silence.
    logs.configure()

    os.environ["TOKENIZERS_PARALLELISM"] = (
        "true" if settings.tokenizers_parallelism else "false"
    )

    if settings.expandable_segments:
        # Each backend reads only its own variable, so setting the other one is
        # what made this a silent no-op on CUDA. `setdefault` keeps an
        # operator's own allocator settings from being overwritten.
        if torch.version.hip:
            os.environ.setdefault(
                "PYTORCH_HIP_ALLOC_CONF", "expandable_segments:True"
            )
        elif torch.version.cuda:
            os.environ.setdefault(
                "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"
            )

    # `torch.backends.cuda.matmul.allow_tf32` is an alias for this same setting
    # (True <-> "high", False <-> "highest"), so it needs no knob of its own —
    # two would silently overwrite each other. cuDNN's is the separate one.
    torch.set_float32_matmul_precision(settings.float32_matmul_precision)
    torch.backends.cudnn.allow_tf32 = settings.cudnn_allow_tf32

    if seed is not None:
        set_seed(seed)

    # Last, so the allocator variables above are already in place before
    # anything here touches the driver.
    unsupported = unsupported_gpu_architecture()
    if unsupported is not None:
        logger.warning("%s", unsupported)


#: Presents the card as a different architecture to the ROCm runtime. The
#: RDNA2 parts share an ISA, so pointing a gfx1032 at the gfx1030 kernels the
#: wheel does ship is what makes it run at all.
HSA_OVERRIDE_VARIABLE = "HSA_OVERRIDE_GFX_VERSION"


def _architecture(name: str) -> str:
    """The bare `gfxNNNN`, dropping the feature flags a device or a wheel

    may or may not spell out.
    """
    return name.split(":", 1)[0]


def unsupported_gpu_architecture() -> str | None:
    """Say so if the installed torch ships no kernels for the present GPU.

    A ROCm wheel carries object code for its build list and no PTX, so a card
    outside that list fails at the *first* device allocation with `HIP error:
    invalid device function`, with `torch.cuda.is_available()` having answered
    True all along.

    :return: the diagnostic, or None where there is nothing to report — HIP
        builds only, and anything unexpected reads as nothing, since a startup
        check that ends a run is worse than the crash it was meant to explain.
    """
    try:
        if not torch.version.hip or not torch.cuda.is_available():
            return None

        compiled = [_architecture(arch) for arch in torch.cuda.get_arch_list()]
        if not compiled:
            return None

        device = _architecture(torch.cuda.get_device_properties(0).gcnArchName)
        if device in compiled:
            return None

        return (
            f"This torch build ships no kernels for {device}: it was compiled "
            f"for {' '.join(compiled)}. GPU work will fail at the first "
            f"allocation with 'HIP error: invalid device function'. Setting "
            f"{HSA_OVERRIDE_VARIABLE} to a supported architecture of the same "
            f"family (10.3.0 for gfx1030) runs the card under those kernels."
        )
    except Exception:
        return None


def is_triton_compatible() -> bool:
    """Whether `torch.compile`'s Triton backend can target this machine's GPU.

    Asked up front because `torch.compile` is lazy: on an older card it returns
    a wrapper quite happily and only fails at the first forward pass.

    :return: whether the GPU is compute capability 7.0 or newer.
    """
    if not torch.cuda.is_available():
        return False

    return torch.cuda.get_device_capability() >= (7, 0)


_TYPE_CHECKER_PACKAGES = ("beartype", "jaxtyping")

# beartype rewrites each checked function into a wrapper whose code object
# reports this in place of a path, so there is no directory dynamo could match
# the wrapper against.
_BEARTYPE_WRAPPER_FILE = "<@beartype"

_type_checkers_excluded = False


def exclude_type_checkers_from_dynamo() -> None:
    """Keep `torch.compile` from tracing the runtime type checker.

    Dynamo cannot evaluate jaxtyping's `__instancecheck__`: tracing in builds a
    guard that fails on the frame that created it, and constant-folding it
    instead rejects a perfectly valid tensor. All three entries are needed, and
    the call is idempotent because `SKIP_DIRS` backs a compiled regex.
    """
    global _type_checkers_excluded
    if _type_checkers_excluded:
        return

    from torch._dynamo import trace_rules

    for package in _TYPE_CHECKER_PACKAGES:
        trace_rules.add(package)
    trace_rules.SKIP_DIRS.append(_BEARTYPE_WRAPPER_FILE)
    trace_rules._recompile_re()

    _type_checkers_excluded = True


def compile_model(model: torch.nn.Module) -> bool:
    """Compile `model`'s forward **in place**, reporting whether it took.

    `nn.Module.compile` rather than `torch.compile`, whose `OptimizedModule`
    wrapper forwards attributes bound to the module it wrapped — so a method
    called on the wrapper runs uncompiled, which is the whole call pattern
    here.

    :param model: the model to compile.
    :return: whether a graph is installed, read off the model rather than off
        the call not raising. A backend that fails later clears it again, so
        ask `is_compiled` for what the model is executing now.
    """
    if not os.environ.get(COMPILE_VARIABLE):
        logger.info(
            "Skipping torch.compile(): %s is not set",
            COMPILE_VARIABLE,
        )
        return False

    if not is_triton_compatible():
        logger.info("Skipping torch.compile(): no Triton-capable GPU")
        return False

    exclude_type_checkers_from_dynamo()

    try:
        _compile_the_backward_with_the_forward()
        _use_eager_dropout_masks()
        # `dynamic=True`: batches are ragged, so a static-shape graph would
        # recompile on nearly every one.
        model.compile(dynamic=True)
    except Exception as error:
        logger.warning("Failed to compile with Triton: %s", error)
        return False

    _install_eager_fallback(model)

    return is_compiled(model)


def _compile_the_backward_with_the_forward() -> None:
    """Make a backward-graph compile failure raise at the forward.

    AOTAutograd already lowers the backward while the forward is compiling,
    but it swallows a failure there and retries the lowering lazily inside
    `loss.backward()` — which is not a call into the model, so
    `_install_eager_fallback` never sees it and the run dies on the raw
    backend error. Making the first attempt the only one puts both halves of
    the graph behind the one guard. It is process-global rather than scoped
    around the `model.compile()` call, which installs a wrapper and nothing
    more: the backend does not run until the first forward, and runs again at
    every recompile.
    """
    import torch._functorch.config

    torch._functorch.config.force_non_lazy_backward_lowering = True


def _use_eager_dropout_masks() -> None:
    """Make a compiled dropout draw the same masks as eager under one seed.

    Inductor generates dropout's random mask with its own philox RNG rather
    than torch's, so under the same seed a compiled run drew different noise
    than an eager one and the two trained on different data, though neither
    computed anything wrong. Process-global rather than scoped around
    `model.compile()`, for the same reason
    `_compile_the_backward_with_the_forward` is: the backend runs lazily at
    the first forward and again at every recompile, past any `with` a call
    site could wrap it in.
    """
    import torch._inductor.config

    torch._inductor.config.fallback_random = True


def _install_eager_fallback(model: torch.nn.Module) -> None:
    """Make a backend failure at a forward drop `model` back to eager.

    `nn.Module.compile` only installs the graph: the backend runs at the first
    forward and, under `dynamic=True`, again at every recompile — inside the
    training loop, where the call that asked for the compile can no longer
    guard it, so an inductor failure killed the run outright. Only dynamo's own
    exceptions are caught, because those mean the compile failed rather than
    the model, which is what makes retrying the call eagerly safe; anything the
    model itself raises propagates untouched.

    :param model: a model `nn.Module.compile` has been called on.
    """
    from torch._dynamo.exc import TorchDynamoException

    compiled_call = model._compiled_call_impl
    if compiled_call is None:
        return

    def call_with_eager_fallback(*args: Any, **kwargs: Any) -> Any:
        try:
            return compiled_call(*args, **kwargs)
        except TorchDynamoException as error:
            logger.warning(
                "Falling back to eager execution: the compiler backend "
                "failed (%s)",
                error,
            )
            model._compiled_call_impl = None
            return model._call_impl(*args, **kwargs)

    model._compiled_call_impl = call_with_eager_fallback


def is_compiled(model: torch.nn.Module) -> bool:
    """Whether `model`'s own `__call__` dispatches to a compiled graph.

    Reports what the model executes *now*: a backend failure clears the graph,
    so this and not `compile_model`'s return value is what a finished run's
    `compiled` tag should be read from.

    :param model: the model to inspect.
    :return: whether a graph is installed.
    """
    return getattr(model, "_compiled_call_impl", None) is not None
