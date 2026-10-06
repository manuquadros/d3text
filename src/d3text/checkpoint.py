"""Writing and reading a checkpoint that carries its own provenance.

A bare `state_dict` does not say which column is which class, nor which
labels, encodings or surface forms it was trained against; `save` writes those
beside the weights. Files older than `FORMAT` are refused, not partly read.
"""

import dataclasses
import os
from collections.abc import Mapping
from typing import Any

import torch
import torch.serialization

from d3text.surface_forms import (
    SurfaceFormIndex,
    index_from_payload,
    index_to_payload,
)
from d3text.vocabulary import Vocabulary

# A key no `state_dict` can carry: parameter names are dotted attribute paths,
# so this cannot collide with one, and its presence is what tells the two
# on-disk shapes apart.
FORMAT_KEY = "d3text_checkpoint_format"
FORMAT = 2

STATE_DICT_KEY = "state_dict"
VOCABULARY_KEY = "vocabulary"
# Optional within the format, not a bump: a reader that does not know a key
# reads the checkpoint it read before, and a bump would refuse every old file.
TOKEN_LABELS_DIGEST_KEY = "token_labels_digest"
LABELLING_RULES_DIGEST_KEY = "labelling_rules_digest"
ENCODINGS_DIGEST_KEY = "encodings_digest"
SURFACE_FORM_INDEX_KEY = "surface_form_index"
RELATION_THRESHOLDS_KEY = "relation_thresholds"


@dataclasses.dataclass(frozen=True)
class Checkpoint:
    """The contents of a `.pt` file: weights, and what they mean.

    :param state_dict: the parameters exactly as `torch.save` received them,
        including the `_orig_mod.` prefixes a checkpoint written under
        `torch.compile` carries.
    :param vocabulary: the class-head column order the weights were trained
        on, and the entity IDs the training split named.
    :param token_labels_digest: the digest of the surface-form index the
        token-level targets were matched against, or `None` for a run that
        read no label store and for a checkpoint written before it was
        recorded.
    :param labelling_rules_digest: the digest of the labelling rules that
        placed those targets, or `None` for a run that read no label store
        and for a checkpoint written before it was recorded — including one
        that does carry a `token_labels_digest`, from before this field
        existed.
    :param encodings_digest: the content digest of the encodings store the
        inputs were read from, or `None` for a checkpoint written before it
        was recorded and for a run whose store carried none.
    :param surface_form_index: the surface-form index `train` built from the
        BRENDA data on its machine, for `infer` to link spans against without
        needing that data itself, or `None` for a checkpoint written before
        this was recorded, or for a training run that could not build one
        (`linking_corpora.brenda_index`'s warning names why).
    :param relation_thresholds: the per-relation decision thresholds
        calibrated on the validation split, by relation name, or `None` for a
        model with no relation head and for a checkpoint never calibrated,
        whose relations are then decided by argmax.
    """

    state_dict: dict[str, Any]
    vocabulary: Vocabulary
    token_labels_digest: str | None = None
    labelling_rules_digest: str | None = None
    encodings_digest: str | None = None
    surface_form_index: SurfaceFormIndex | None = None
    relation_thresholds: dict[str, float] | None = None


def save(
    path: str | os.PathLike[str],
    state_dict: dict[str, Any],
    vocabulary: Vocabulary,
    token_labels_digest: str | None = None,
    labelling_rules_digest: str | None = None,
    encodings_digest: str | None = None,
    surface_form_index: SurfaceFormIndex | None = None,
    relation_thresholds: Mapping[str, float] | None = None,
) -> None:
    """Write `state_dict`, its vocabulary and where its data came from.

    The write replaces `path` atomically, so a failed one leaves any earlier
    file there intact.

    The vocabulary goes in as plain builtins rather than as a pickled
    `Vocabulary`, so the file stays loadable under `weights_only=True`; the
    surface-form index travels the same way, through
    `d3text.surface_forms.index_to_payload`.

    :param path: where to write.
    :param state_dict: the parameters to store.
    :param vocabulary: the class columns that interpret them.
    :param token_labels_digest: the surface-form index digest of the label
        store the run's token-level targets came from, if it read one.
    :param labelling_rules_digest: the labelling-rules digest of that same
        store, if it read one.
    :param encodings_digest: the content digest of the encodings store the
        run's inputs came from, if it records one.
    :param surface_form_index: the index `train` built from the BRENDA data,
        for `infer` to link against, if it built one.
    :param relation_thresholds: the calibrated decision threshold of each
        typed relation, by name, if the model was calibrated.
    """
    partial = os.path.join(
        os.path.dirname(path), f"{os.path.basename(path)}.partial"
    )
    torch.save(
        {
            FORMAT_KEY: FORMAT,
            STATE_DICT_KEY: state_dict,
            VOCABULARY_KEY: vocabulary.to_payload(),
            TOKEN_LABELS_DIGEST_KEY: token_labels_digest,
            LABELLING_RULES_DIGEST_KEY: labelling_rules_digest,
            ENCODINGS_DIGEST_KEY: encodings_digest,
            SURFACE_FORM_INDEX_KEY: (
                None
                if surface_form_index is None
                else index_to_payload(surface_form_index)
            ),
            RELATION_THRESHOLDS_KEY: (
                None
                if relation_thresholds is None
                else {name: float(t) for name, t in relation_thresholds.items()}
            ),
        },
        partial,
    )
    os.replace(partial, path)


def load(
    path: str | os.PathLike[str],
    map_location: torch.serialization.MAP_LOCATION = None,
) -> Checkpoint:
    """Read `path`, refusing any shape this code cannot interpret.

    :param path: the checkpoint to read.
    :param map_location: passed through to `torch.load`.
    :return: the weights, the vocabulary and the store digests.
    :raises ValueError: on a checkpoint whose recorded format this code does
        not know, an older one carrying the entity-linking head, a bare
        `state_dict` from before the format key existed, a `state_dict` that
        is not a dict with string keys, or a digest field that is not a string
        or None. Reading the weights and ignoring the rest is how a format
        change becomes a wrong-numbers bug instead of an error.
    """
    # Explicit, not relied on as a default: `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD`
    # only overrides a call site that left this unset, and this file is
    # written to stay loadable under `weights_only=True` (see `save`).
    contents = torch.load(path, map_location=map_location, weights_only=True)

    if not isinstance(contents, dict) or FORMAT_KEY not in contents:
        raise ValueError(
            f"{os.fspath(path)} is a bare state dict from before "
            f"{FORMAT_KEY!r} was written. It carries the entity-linking head, "
            "which this d3text no longer builds; retrain to get a format-"
            f"{FORMAT} checkpoint."
        )

    version = contents[FORMAT_KEY]
    if version != FORMAT:
        # `isinstance` because `version` came off disk: a string there would
        # make the comparison a TypeError rather than the message it is for.
        older = isinstance(version, int) and version < FORMAT
        detail = (
            " It carries the entity-linking head, which this d3text no "
            f"longer builds; retrain to get a format-{FORMAT} checkpoint."
            if older
            else ""
        )
        raise ValueError(
            f"{os.fspath(path)} is a format-{version} checkpoint; this "
            f"d3text reads format {FORMAT}.{detail}"
        )

    try:
        state_dict = contents[STATE_DICT_KEY]
        payload = contents[VOCABULARY_KEY]
    except KeyError as error:
        raise ValueError(
            f"{os.fspath(path)} declares format {version} but is missing "
            f"{error}"
        ) from None

    # Validate state_dict is a dict with string keys
    if not isinstance(state_dict, dict):
        raise ValueError(
            f"{os.fspath(path)} {STATE_DICT_KEY!r} must be a dict, "
            f"found {type(state_dict).__name__}"
        )

    if not all(isinstance(key, str) for key in state_dict.keys()):
        raise ValueError(
            f"{os.fspath(path)} {STATE_DICT_KEY!r} keys must all be strings, "
            f"found non-string key(s)"
        )

    # Validate digests are str or None
    for digest_key in (
        TOKEN_LABELS_DIGEST_KEY,
        LABELLING_RULES_DIGEST_KEY,
        ENCODINGS_DIGEST_KEY,
    ):
        digest_value = contents.get(digest_key)
        if digest_value is not None and not isinstance(digest_value, str):
            raise ValueError(
                f"{os.fspath(path)} {digest_key!r} must be a string or None, "
                f"found {type(digest_value).__name__}"
            )

    raw_index = contents.get(SURFACE_FORM_INDEX_KEY)
    return Checkpoint(
        state_dict=state_dict,
        vocabulary=Vocabulary.from_payload(payload),
        token_labels_digest=contents.get(TOKEN_LABELS_DIGEST_KEY),
        labelling_rules_digest=contents.get(LABELLING_RULES_DIGEST_KEY),
        encodings_digest=contents.get(ENCODINGS_DIGEST_KEY),
        surface_form_index=(
            None if raw_index is None else index_from_payload(raw_index)
        ),
        relation_thresholds=_relation_thresholds(
            path, contents.get(RELATION_THRESHOLDS_KEY)
        ),
    )


def _relation_thresholds(
    path: str | os.PathLike[str], raw: object
) -> dict[str, float] | None:
    """The stored thresholds, checked: a name -> probability mapping or None."""
    if raw is None:
        return None
    if not isinstance(raw, dict) or not all(
        isinstance(name, str)
        and isinstance(value, float)
        and 0.0 <= value <= 1.0
        for name, value in raw.items()
    ):
        raise ValueError(
            f"{os.fspath(path)} carries {RELATION_THRESHOLDS_KEY!r} as "
            f"{raw!r}, not a mapping of relation names to probabilities"
        )
    return dict(raw)
