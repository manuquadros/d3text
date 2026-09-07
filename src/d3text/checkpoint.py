"""Writing and reading a checkpoint that carries its own provenance.

A bare `state_dict` is not self-describing: its entity head is a matrix of the
right *width* and nothing more, and nothing in it says which dictionary its
token-level targets were matched against or which tokenization produced its
inputs. `save` writes the `Vocabulary`, the label store's surface-form index
digest and the encodings store's content digest next to the weights and `load`
hands them back. Checkpoints written before any of them existed still load,
reported as `None`.
"""

import dataclasses
import os
from typing import Any

import torch

from d3text.vocabulary import Vocabulary

# A key no `state_dict` can carry: parameter names are dotted attribute paths,
# so this cannot collide with one, and its presence is what tells the two
# on-disk shapes apart.
FORMAT_KEY = "d3text_checkpoint_format"
FORMAT = 1

STATE_DICT_KEY = "state_dict"
VOCABULARY_KEY = "vocabulary"
# Optional within format 1 rather than a format of its own: a reader that does
# not know the key reads exactly the checkpoint it read before, and bumping
# would refuse every file already on disk to gain nothing.
TOKEN_LABELS_DIGEST_KEY = "token_labels_digest"
ENCODINGS_DIGEST_KEY = "encodings_digest"


@dataclasses.dataclass(frozen=True)
class Checkpoint:
    """The contents of a `.pt` file: weights, and what they mean.

    :param state_dict: the parameters exactly as `torch.save` received them,
        including the `_orig_mod.` prefixes a checkpoint written under
        `torch.compile` carries.
    :param vocabulary: the column order the heads were trained on, or `None`
        for a checkpoint written before it was recorded.
    :param token_labels_digest: the digest of the surface-form index the
        token-level targets were matched against, or `None` for a run that
        read no label store and for a checkpoint written before it was
        recorded.
    :param encodings_digest: the content digest of the encodings store the
        inputs were read from, or `None` for a checkpoint written before it
        was recorded and for a run whose store carried none.
    """

    state_dict: dict[str, Any]
    vocabulary: Vocabulary | None
    token_labels_digest: str | None = None
    encodings_digest: str | None = None

    @property
    def is_legacy(self) -> bool:
        """Whether this checkpoint records no vocabulary of its own."""
        return self.vocabulary is None


def save(
    path: str | os.PathLike[str],
    state_dict: dict[str, Any],
    vocabulary: Vocabulary,
    token_labels_digest: str | None = None,
    encodings_digest: str | None = None,
) -> None:
    """Write `state_dict`, its vocabulary and where its data came from.

    The vocabulary goes in as plain builtins rather than as a pickled
    `Vocabulary`, so the file stays loadable under `weights_only=True`.

    :param path: where to write.
    :param state_dict: the parameters to store.
    :param vocabulary: the column order that interprets them.
    :param token_labels_digest: the surface-form index digest of the label
        store the run's token-level targets came from, if it read one.
    :param encodings_digest: the content digest of the encodings store the
        run's inputs came from, if it records one.
    """
    torch.save(
        {
            FORMAT_KEY: FORMAT,
            STATE_DICT_KEY: state_dict,
            VOCABULARY_KEY: vocabulary.to_payload(),
            TOKEN_LABELS_DIGEST_KEY: token_labels_digest,
            ENCODINGS_DIGEST_KEY: encodings_digest,
        },
        path,
    )


def load(
    path: str | os.PathLike[str],
    map_location: Any = None,
) -> Checkpoint:
    """Read `path`, whichever of the two on-disk shapes it holds.

    :param path: the checkpoint to read.
    :param map_location: passed through to `torch.load`.
    :return: the weights, the vocabulary and the two store digests, all but
        the weights `None` for a legacy file.
    :raises ValueError: on a checkpoint whose recorded format this code does
        not know. Silently reading its `state_dict` and ignoring the rest is
        how a format change becomes a wrong-numbers bug instead of an error.
    """
    contents = torch.load(path, map_location=map_location)

    if not isinstance(contents, dict) or FORMAT_KEY not in contents:
        return Checkpoint(state_dict=contents, vocabulary=None)

    version = contents[FORMAT_KEY]
    if version != FORMAT:
        raise ValueError(
            f"{os.fspath(path)} is a format-{version} checkpoint; this "
            f"d3text reads format {FORMAT}"
        )

    try:
        state_dict = contents[STATE_DICT_KEY]
        payload = contents[VOCABULARY_KEY]
    except KeyError as error:
        raise ValueError(
            f"{os.fspath(path)} declares format {version} but is missing "
            f"{error}"
        ) from None

    return Checkpoint(
        state_dict=state_dict,
        vocabulary=Vocabulary.from_payload(payload),
        token_labels_digest=contents.get(TOKEN_LABELS_DIGEST_KEY),
        encodings_digest=contents.get(ENCODINGS_DIGEST_KEY),
    )
