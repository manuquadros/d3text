"""Backwards-compatible re-exports of the model classes.

The models used to live here, in one 2.5k-line module; they now live in
`base`, `heads`, `ner`, `brenda` and `relations`. This module keeps the old
import path working. New code should import from the module that defines what
it wants.

Three things are deliberately *not* on this path.

The training loop: it left the model classes for `d3text.training.Trainer`, so
`train_model` and `print_epoch_stats` are gone from here rather than re-exported
from it.

`BrendaClassificationModel` and `ETEBrendaModel`: there is now one model class,
`BrendaModel`, which holds a `RelationExtractor` or does not. The two old names
survive only as the strings a `config.model_class` may carry, and `d3text.factory`
is what turns them into a model. Aliasing them here would put back the confusion
the merge removed — both names would be the same class, so `isinstance` could no
longer tell an end-to-end model from a two-head one. The question that used to be
asked that way is now `model.relations is None`.

The module *globals* the old file carried — `cpu_embeddings_cache`,
`aggregate_embeddings`: re-exporting a value rebinds it, so patching it here
would leave the definition in `base` untouched, and a test that thought it had
switched the embeddings cache off would silently still be using it.
`monkeypatch.setattr` raises on a missing attribute, so patching the old path now
fails loudly instead.
"""

from .base import Model as Model
from .base import Step as Step
from .base import label_columns as label_columns
from .base import load_base_model as load_base_model
from .base import pool_logits as pool_logits
from .brenda import BrendaModel as BrendaModel
from .brenda import Logits as Logits
from .brenda import Targets as Targets
from .brenda import ordered_entities as ordered_entities
from .heads import BiaffineRelationClassifier as BiaffineRelationClassifier
from .heads import ClassificationHead as ClassificationHead
from .heads import PermutationBatchNorm1d as PermutationBatchNorm1d
from .heads import initialize_classifier_bias as initialize_classifier_bias
from .ner import NERClassificationModel as NERClassificationModel
from .relations import AlignedRelations as AlignedRelations
from .relations import RelationExtractor as RelationExtractor
from .relations import RelationPairs as RelationPairs
from .relations import balanced_class_weights as balanced_class_weights
from .relations import focal_cross_entropy as focal_cross_entropy
