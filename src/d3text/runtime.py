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
    if not is_triton_compatible():
        logger.info("Skipping torch.compile(): no Triton-capable GPU")
        return False

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
