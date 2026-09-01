"""Writing and reading a checkpoint that carries its own vocabulary.

A bare `state_dict` is not self-describing: its entity head is a matrix of the
right *width* and nothing more. `save` writes the `Vocabulary` next to the
weights and `load` hands it back. Checkpoints written before this existed still
load, reported as `vocabulary=None`.
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


@dataclasses.dataclass(frozen=True)
class Checkpoint:
    """The contents of a `.pt` file: weights, and what they mean.

    :param state_dict: the parameters exactly as `torch.save` received them,
        including the `_orig_mod.` prefixes a checkpoint written under
        `torch.compile` carries.
    :param vocabulary: the column order the heads were trained on, or `None`
        for a checkpoint written before it was recorded.
    """

    state_dict: dict[str, Any]
    vocabulary: Vocabulary | None

    @property
    def is_legacy(self) -> bool:
        """Whether this checkpoint records no vocabulary of its own."""
        return self.vocabulary is None


def save(
    path: str | os.PathLike[str],
    state_dict: dict[str, Any],
    vocabulary: Vocabulary,
) -> None:
    """Write `state_dict` and the `vocabulary` that interprets it to `path`.

    The vocabulary goes in as plain builtins rather than as a pickled
    `Vocabulary`, so the file stays loadable under `weights_only=True`.

    :param path: where to write.
    :param state_dict: the parameters to store.
    :param vocabulary: the column order that interprets them.
    """
    torch.save(
        {
            FORMAT_KEY: FORMAT,
            STATE_DICT_KEY: state_dict,
            VOCABULARY_KEY: vocabulary.to_payload(),
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
    :return: the weights and the vocabulary, the latter `None` for a legacy
        file.
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
    )
