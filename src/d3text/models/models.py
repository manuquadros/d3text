"""Backwards-compatible re-exports of the model classes.

The models used to live here, in one 2.5k-line module; they now live in
`base`, `heads`, `ner`, `entity_linking` and `ete`. This module keeps the old
import path working. New code should import from the module that defines what
it wants.

Only the public names are re-exported. The module *globals* the old file
carried — `cpu_embeddings_cache`, `aggregate_embeddings` — are deliberately
absent: re-exporting a value rebinds it, so patching it here would leave the
definition in `base` untouched, and a test that thought it had switched the
embeddings cache off would silently still be using it. `monkeypatch.setattr`
raises on a missing attribute, so patching the old path now fails loudly
instead.
"""

from .base import Model as Model
from .base import Step as Step
from .base import get_pool_fn as get_pool_fn
from .base import label_columns as label_columns
from .base import load_base_model as load_base_model
from .base import print_epoch_stats as print_epoch_stats
from .entity_linking import (
    BrendaClassificationModel as BrendaClassificationModel,
)
from .entity_linking import ordered_entities as ordered_entities
from .ete import ETEBrendaModel as ETEBrendaModel
from .ete import balanced_class_weights as balanced_class_weights
from .ete import focal_cross_entropy as focal_cross_entropy
from .ete import get_batch_entities as get_batch_entities
from .heads import BiaffineRelationClassifier as BiaffineRelationClassifier
from .heads import ClassificationHead as ClassificationHead
from .heads import PermutationBatchNorm1d as PermutationBatchNorm1d
from .heads import initialize_classifier_bias as initialize_classifier_bias
from .ner import NERClassificationModel as NERClassificationModel
