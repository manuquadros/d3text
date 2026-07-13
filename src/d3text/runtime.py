"""Process-wide runtime configuration.

TF32, the float32 matmul precision, the CUDA/HIP caching allocator, tokenizer
parallelism, and the RNG seed are all *process*-global and sticky: the first
writer wins, and nothing undoes it. Setting them while a module is being
imported makes a run's numerics depend on import order — which is how
``scripts/tune.py`` came to train at a different matmul precision from
``scripts/train.py``, its own setting landing after the one ``d3text.models``
applied on the way in.

So they belong to whoever owns the process, not to whichever module happens to
be imported first. `configure()` is called from a script's ``main()``; tests,
notebooks, and the precompute scripts inherit torch's own defaults unless they
ask for these.
"""

import os

import torch

from .models.config import MachineConfig, machine_config


def configure(
    config: MachineConfig | None = None, *, seed: int | None = 42
) -> None:
    """Apply this machine's runtime settings, defaulting to ``config.toml``.

    Call once from a script entry point, before any CUDA work: the caching
    allocator reads its environment variable when it first initialises and
    ignores it thereafter. ``seed=None`` leaves the global RNG untouched.
    """
    settings = machine_config() if config is None else config

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
