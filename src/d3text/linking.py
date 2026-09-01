"""Resolving a tagged mention to the BRENDA entities it could name.

Deliberately not part of the model: it holds no learned parameters, so it can
be swapped without touching a checkpoint. The answer is a *set*, since a
surface form is not owned by one entity, and the empty set is an answer — a NIL
mention — rather than a failure.
"""

from typing import Protocol, runtime_checkable

from d3text.surface_forms import SurfaceFormIndex, form_words
from d3text.token_labels import BRENDA_LABELS, LabelSpace


@runtime_checkable
class Linker(Protocol):
    """Span text + tagged type -> the entity IDs the span could name."""

    def link(self, mention: str, entity_type: str) -> frozenset[str]:
        """Every entity ID of `entity_type` that `mention` could name.

        :param mention: the span's text.
        :param entity_type: the type the tagger assigned it.
        :return: the candidate IDs; empty means NIL, which is an answer in its
            own right.
        """
        ...


class DictionaryLinker:
    """Longest contiguous match against the tagged type's slice of the index.

    Longest-first is the disambiguation rule, and between equally long matches
    the IDs are unioned. The type conditions the *filter*, not the sweep, so
    nested entities of another type stay reachable from the same span.
    """

    def __init__(
        self,
        index: SurfaceFormIndex,
        space: LabelSpace = BRENDA_LABELS,
    ) -> None:
        self._index = index
        self._space = space
        self._prefixes = dict(zip(space.types, space.prefixes))

    def link(self, mention: str, entity_type: str) -> frozenset[str]:
        try:
            prefix = self._prefixes[entity_type]
        except KeyError:
            msg = (
                f"{entity_type!r} is not an entity type of this linker's "
                f"label space; known: {list(self._prefixes)}"
            )
            raise KeyError(msg) from None

        words = form_words(mention)
        widest = min(len(words), self._index.max_words)
        for length in range(widest, 0, -1):
            found = frozenset(
                entity_id
                for start in range(len(words) - length + 1)
                for entity_id in self._index.lookup(
                    words[start : start + length]
                )
                if entity_id.startswith(prefix)
            )
            if found:
                return found
        return frozenset()


__all__ = ["DictionaryLinker", "Linker"]
