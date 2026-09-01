"""Process-wide runtime configuration.

TF32, the matmul precision, the caching allocator, tokenizer parallelism, the
RNG seed and the log handler are all process-global and sticky, so setting them
at import time makes a run's numerics depend on import order. `configure()` is
called from a script's `main()`; everything else inherits torch's own defaults.
"""

import logging
import os

import torch

from . import logs
from .models.config import MachineConfig, machine_config

logger = logging.getLogger(__name__)

#: Disables `compile_model` outright, regardless of Triton compatibility.
#: An environment variable rather than a `config.toml` key or CLI flag, on the
#: `D3TEXT_LOG_LEVEL` precedent: `runtime.configure()` runs before
#: `command_line_args()` in `train`/`tune`/`evaluate`, so a parsed flag could
#: never reach here, and whether compiling pays is a property of the machine
#: and the run, not of the model config. Any non-empty value disables.
COMPILE_DISABLE_VARIABLE = "D3TEXT_DISABLE_COMPILE"


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
        # Seeds the global generator that `data.g` hands to the samplers.
        torch.manual_seed(seed)

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
    :return: whether the graph is installed, read off the model rather than off
        the call not raising.
    """
    if os.environ.get(COMPILE_DISABLE_VARIABLE):
        logger.info(
            "Skipping torch.compile(): %s is set",
            COMPILE_DISABLE_VARIABLE,
        )
        return False

    if not is_triton_compatible():
        logger.info("Skipping torch.compile(): no Triton-capable GPU")
        return False

    exclude_type_checkers_from_dynamo()

    try:
        # `dynamic=True`: batches are ragged, so a static-shape graph would
        # recompile on nearly every one.
        model.compile(dynamic=True)
    except Exception as error:
        logger.warning("Failed to compile with Triton: %s", error)
        return False

    return is_compiled(model)


def is_compiled(model: torch.nn.Module) -> bool:
    """Whether `model`'s own `__call__` dispatches to a compiled graph.

    :param model: the model to inspect.
    :return: whether a graph is installed.
    """
    return getattr(model, "_compiled_call_impl", None) is not None
