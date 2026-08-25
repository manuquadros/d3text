"""Run `train` with the global generator seeded, so two pooling settings
differ only in the pooling.

`train.main()` calls `runtime.configure()` itself, and that call's `seed`
defaults to 42 — so seeding here and *then* calling main would simply be
overwritten, and the file would look like it pinned a seed while pinning
nothing. Binding the seed to the function instead makes main's own call the one
that applies it.
"""

import functools

from d3text import runtime

SEED = 0

runtime.configure = functools.partial(  # type: ignore[assignment]
    runtime.configure, seed=SEED
)

from d3text.cli import train  # noqa: E402

train.main()
