"""Writing and reading a checkpoint that carries its own provenance.

A bare `state_dict` is not self-describing: its class head is a matrix of the
right *width* and nothing more, and nothing in it says which dictionary its
token-level targets were matched against, which tokenization produced its
inputs, or which entities a surface form of the tagger's own type can name.
`save` writes the `Vocabulary`, the label store's surface-form index digest,
its labelling-rules digest, the encodings store's content digest and the
linker's own surface-form index next to the weights and `load` hands them
back.

Format 2 dropped the entity-linking head, so a file written under format 1 —
or the bare `state_dict` that predates the format key — holds parameters this
code cannot build a model for. Both are refused outright rather than read for
the part that still fits.
"""

import dataclasses
import os
from typing import Any

import torch

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
# Optional within the format rather than a format of its own: a reader that
# does not know the key reads exactly the checkpoint it read before, and
# bumping would refuse every file already on disk to gain nothing. The
# surface-form index is optional the same way: it qualifies what `infer` can
# do with the weights, not how to interpret them.
TOKEN_LABELS_DIGEST_KEY = "token_labels_digest"
LABELLING_RULES_DIGEST_KEY = "labelling_rules_digest"
ENCODINGS_DIGEST_KEY = "encodings_digest"
SURFACE_FORM_INDEX_KEY = "surface_form_index"


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
    """

    state_dict: dict[str, Any]
    vocabulary: Vocabulary
    token_labels_digest: str | None = None
    labelling_rules_digest: str | None = None
    encodings_digest: str | None = None
    surface_form_index: SurfaceFormIndex | None = None


def save(
    path: str | os.PathLike[str],
    state_dict: dict[str, Any],
    vocabulary: Vocabulary,
    token_labels_digest: str | None = None,
    labelling_rules_digest: str | None = None,
    encodings_digest: str | None = None,
    surface_form_index: SurfaceFormIndex | None = None,
) -> None:
    """Write `state_dict`, its vocabulary and where its data came from.

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
    """
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
        },
        path,
    )


def load(
    path: str | os.PathLike[str],
    map_location: Any = None,
) -> Checkpoint:
    """Read `path`, refusing any shape this code cannot interpret.

    :param path: the checkpoint to read.
    :param map_location: passed through to `torch.load`.
    :return: the weights, the vocabulary and the store digests.
    :raises ValueError: on a checkpoint whose recorded format this code does
        not know, an older one carrying the entity-linking head, or a bare
        `state_dict` from before the format key existed. Reading the weights
        and ignoring the rest is how a format change becomes a wrong-numbers
        bug instead of an error.
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
    )
