"""Process-wide runtime configuration.

TF32, the float32 matmul precision, the CUDA/HIP caching allocator, tokenizer
parallelism, the RNG seed and where the library's log records go are all
*process*-global and sticky: the first writer wins, and nothing undoes it.
Setting them while a module is being imported makes a run's numerics depend on
import order — which is how ``scripts/tune.py`` came to train at a different
matmul precision from ``scripts/train.py``, its own setting landing after the
one ``d3text.models`` applied on the way in.

So they belong to whoever owns the process, not to whichever module happens to
be imported first. `configure()` is called from a script's ``main()``; tests,
notebooks, and the precompute scripts inherit torch's own defaults unless they
ask for these.
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
    """Apply this machine's runtime settings, defaulting to ``config.toml``.

    Call once from a script entry point, before any CUDA work: the caching
    allocator reads its environment variable when it first initialises and
    ignores it thereafter. ``seed=None`` leaves the global RNG untouched.

    Also installs the package's console log handler at the verbosity
    ``D3TEXT_LOG_LEVEL`` asks for; see `d3text.logs`.
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
    """The bare ``gfxNNNN``, dropping the ``:sramecc+:xnack-`` feature flags
    that a device or a wheel may or may not spell out."""
    return name.split(":", 1)[0]


def unsupported_gpu_architecture() -> str | None:
    """Say so if the installed torch ships no kernels for the present GPU.

    ROCm has no equivalent of PTX: a wheel carries object code for the
    architectures it was built for and nothing else, so a card outside that
    list fails at the *first* device allocation with ``HIP error: invalid
    device function`` — arbitrarily deep into whatever ran first, and with
    ``torch.cuda.is_available()`` having answered True all along.

    Returns the diagnostic, or `None` where there is nothing to report. HIP
    builds only: a CUDA wheel embeds PTX and JITs forward-compatibly, and
    ``gcnArchName`` is a ROCm property in the first place. Anything unexpected
    reads as nothing to report — a startup check that ends a run is worse than
    the crash it was meant to explain.
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

    Triton needs compute capability 7.0 (Volta) or newer. Asking up front
    matters because `torch.compile` is lazy: on an older card it returns a
    wrapper quite happily and only fails at the first forward pass, long past
    the ``try/except`` the call site wraps it in.
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

    ``beartype_this_package()`` (see ``d3text/__init__.py``) wraps every
    annotated function in this package in a checker that runs
    ``isinstance(x, Float[Tensor, ...])``, and dynamo cannot evaluate that
    call. Tracing into jaxtyping's ``__instancecheck__`` builds a guard on a
    bound method's object id that fails on the very frame that created it, and
    torch aborts with ``AssertionError: Guard failed on the same frame it was
    created``. Where it does not trace in, it constant-folds the check through
    ``issubclass`` to ``False`` instead, and beartype rejects a tensor that is
    perfectly valid. Either way the run dies before its first batch.

    Skipping these frames leaves the checks themselves running, eagerly and
    unchanged; only the model's own frames are compiled. All three entries are
    needed — skipping the two packages still leaves the generated wrapper
    traced, and skipping the wrapper alone lets dynamo pick
    ``__instancecheck__`` up as a top-level frame of its own.

    Idempotent, because ``SKIP_DIRS`` is a process-global list backing a
    compiled regex: appending on every call would grow both without end.
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

    `nn.Module.compile` rather than `torch.compile`: the latter hands back an
    `OptimizedModule` wrapper, and every attribute it forwards comes back bound
    to the module it wrapped — so a method called on the wrapper runs on the
    *uncompiled* model, and the ``self(...)`` inside it never reaches the
    compiled graph. That is the whole call pattern here: the trainer drives
    ``model.run_epoch(...)``, which is three frames above the only forward
    call. Compiling in place installs the graph on the model's own
    ``__call__``, which every one of those frames goes through.

    The return value is read off the model rather than off the call
    succeeding, so the ``compiled`` tag on a run says the graph is installed
    and not merely that nothing raised.
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
    """Whether `model`'s own ``__call__`` dispatches to a compiled graph."""
    return getattr(model, "_compiled_call_impl", None) is not None
