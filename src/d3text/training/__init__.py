"""The training loop: the epoch schedule, and the per-step weight update.

Deliberately re-exports nothing. `d3text.models.base` imports
`d3text.training.update`, so a re-export of `.trainer` here — which imports
`d3text.models.base` in turn — would close that cycle at import time.
"""
