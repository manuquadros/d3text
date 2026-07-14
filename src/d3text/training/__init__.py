"""Driving a model through a training run.

A `Model` knows what one batch costs and how to walk a set of them; everything
that is true of a *run* rather than of a model — the optimizer, the scheduler,
the gradient scaler, early stopping, the checkpoint — lives here.
"""

from .trainer import Trainer as Trainer
from .trainer import print_epoch_stats as print_epoch_stats
