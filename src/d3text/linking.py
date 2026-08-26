"""Resolving a tagged mention to the BRENDA entities it could name.

The tagger proposes typed spans; something has to turn a span into entity IDs,
and that something is deliberately **not part of the model**: it holds no
learned parameters, so it can be swapped — a dictionary today, a bi-encoder
retriever later that catches the variation edit distance misses — without
touching a checkpoint. `Linker` is that seam.

Two facts of the contract are load-bearing:

- **The answer is a set, not an ID.** A surface form is not owned by one
  entity — `AS-A` names four separate enzymes — and a species nested inside a
  strain designation is meant to yield both entities rather than force a
  choice at link time. Whoever consumes the set (the relation head, an
  evaluation) is the one with the context to narrow it.
- **The empty set is an answer**, not a failure: a typed span the dictionary
  cannot resolve is a NIL mention, emitted with no ID and scored as *correct*
  exactly when the mention has no BRENDA entity.

`DictionaryLinker` matches only what the tagger proposed — a handful of
lookups per document, each against one type's slice of the index, instead of
one query per n-gram window over the whole vocabulary. That ordering is what
makes linking cheap; the index itself is `d3text.surface_forms`' exact,
case-aware one, whose trade-offs (and why not the fuzzy `dict_tagger.Vocab`)
are argued there.
"""

from typing import Protocol, runtime_checkable

from d3text.surface_forms import SurfaceFormIndex, form_words
from d3text.token_labels import BRENDA_LABELS, LabelSpace


@runtime_checkable
class Linker(Protocol):
    """Span text + tagged type -> the entity IDs the span could name."""

    def link(self, mention: str, entity_type: str) -> frozenset[str]:
        """Every entity ID of `entity_type` that `mention` could name.

        Empty means NIL: the mention resolves to no known entity of that
        type, which is an answer in its own right.
        """
        ...


class DictionaryLinker:
    """Longest contiguous match against the tagged type's slice of the index.

    Longest-first is the disambiguation rule: over ``Streptomyces
    griseocarneus`` the species wins and the bare genus is never emitted,
    because a window that long matched and every shorter window lies inside
    it. Between equally long matches nothing here can choose, so their IDs
    are unioned — the same arity `Linker` promises for ambiguous forms.

    The type conditions the *filter*, not the sweep, so nested entities of
    another type stay reachable from the same span: linking ``Escherichia
    coli K-12`` as a strain yields the designation's ID, and linking the
    same span as a bacterium yields the nested species, which is how one
    span emits both entities.
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
