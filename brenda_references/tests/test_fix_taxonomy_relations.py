"""Pin `fix_taxonomy` rewriting HasEnzyme/HasSpecies arguments it orphans.

`fix_taxonomy` used to delete a reclassified `other_organisms` id without
ever touching `doc["relations"]`, so a HasEnzyme subject or HasSpecies
object naming that id was silently dropped by `preprocess_relations`. These
tests exercise the remap against a real `BrendaDocDB(storage="memory")` —
no on-disk doc database — with `ncbitax.decompose_name` and
`StrainInfoAdapter` faked so nothing reaches the network;
`insert_bacteria_record` still runs for real, against the package's bundled
(offline) LPSN dump, and finds none of these fictional names there. The
strain-collision test also carries its result through `preprocess_labels`
to prove nothing is dropped.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import pandas as pd
import pytest
from brenda_references.docdb import BrendaDocDB
from d3types import Strain
from scripts import fix_taxonomy
from taxonomy.ncbitax import DecomposedName
from tinydb.table import Document as TinyDBDoc

from brenda_references.brenda_references import preprocess_labels


class FakeStrainInfoAdapter:
    """Stands in for `StrainInfoAdapter`: echoes the models it is given.

    The real adapter resolves designations against the StrainInfo network
    API; these tests only need `retrieve_strain_models` to be a no-op so
    `update_doc_strain` can insert the model it built locally.
    """

    def __enter__(self) -> "FakeStrainInfoAdapter":
        return self

    def __exit__(self, *exc_info: object) -> None:
        return None

    def retrieve_strain_models(
        self, strains: dict[int, Strain]
    ) -> dict[int, Strain]:
        return strains


@pytest.fixture(autouse=True)
def _fake_strain_network(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        fix_taxonomy, "StrainInfoAdapter", FakeStrainInfoAdapter
    )


def _decompose_from(
    mapping: dict[str, DecomposedName],
) -> Callable[[str], DecomposedName | None]:
    def fake(name: str) -> DecomposedName | None:
        return mapping.get(name)

    return fake


def test_ec_id_colliding_with_a_reclassified_organism_id_left_untouched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A HasEnzyme object equal to a removed other_organisms id is not
    translated, even though that id is also, correctly, remapped as a
    subject elsewhere in the same document's relations.

    Reproduces attempt 1's defect: remapping subject *and* object through
    one id->id map turned an unrelated EC id into the reclassified
    organism's new id, fabricating a label for an enzyme absent from the
    document.
    """
    monkeypatch.setattr(
        fix_taxonomy.ncbitax,
        "decompose_name",
        _decompose_from(
            {
                "Xenobacterium alpha S1": DecomposedName(
                    species="Xenobacterium alpha", strain="S1"
                ),
            }
        ),
    )

    with BrendaDocDB(storage="memory") as docdb:
        docdb.bacteria.insert(
            TinyDBDoc({"organism": "Some Known Bacterium", "synonyms": []}, 9)
        )
        docdb.documents.insert(
            TinyDBDoc(
                {
                    "other_organisms": {"501": "Xenobacterium alpha S1"},
                    "bacteria": {"9": "Some Known Bacterium"},
                    "strains": [],
                    "relations": {
                        "HasEnzyme": [
                            # Unrelated pair: object 501 is an EC id that
                            # happens to equal the other_organisms id about
                            # to be removed.
                            {"subject": 9, "object": 501},
                            # The reclassified organism's own pair.
                            {"subject": 501, "object": 42},
                        ],
                        "HasSpecies": [],
                    },
                },
                1,
            )
        )

        fix_taxonomy.fix_taxonomy(docdb)
        doc = docdb.documents.get(doc_id=1)

    has_enzyme = doc["relations"]["HasEnzyme"]
    assert has_enzyme[0] == {"subject": 9, "object": 501}

    new_strain_id = doc["strains"][0]
    assert has_enzyme[1] == {"subject": new_strain_id, "object": 42}
    assert "501" not in doc["other_organisms"]


def test_bacteria_and_strain_get_distinct_ids_and_subject_prefers_strain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An organism that splits into both a species and a strain is stored
    under two different ids, and a HasEnzyme subject naming it moves to the
    strain id, never the bacteria id.
    """
    monkeypatch.setattr(
        fix_taxonomy.ncbitax,
        "decompose_name",
        _decompose_from(
            {
                "Gamma species G7": DecomposedName(
                    species="Gamma species", strain="G7"
                ),
            }
        ),
    )

    with BrendaDocDB(storage="memory") as docdb:
        # Seeded so the bacteria table's next id is well past any strain
        # id this test produces; a fix that accidentally reused one id for
        # both would otherwise pass by coincidence.
        docdb.bacteria.insert(
            TinyDBDoc({"organism": "placeholder", "synonyms": []}, 100)
        )
        docdb.documents.insert(
            TinyDBDoc(
                {
                    "other_organisms": {"5": "Gamma species G7"},
                    "bacteria": {},
                    "strains": [],
                    "relations": {
                        "HasEnzyme": [{"subject": 5, "object": 3}],
                        "HasSpecies": [],
                    },
                },
                2,
            )
        )

        fix_taxonomy.fix_taxonomy(docdb)
        doc = docdb.documents.get(doc_id=2)

    assert list(doc["bacteria"].keys()) == [101]
    assert doc["bacteria"][101] == "Gamma species"
    assert doc["strains"] == [1]
    assert set(doc["bacteria"].keys()).isdisjoint(doc["strains"])
    assert doc["relations"]["HasEnzyme"][0] == {"subject": 1, "object": 3}


def test_strain_collision_credits_each_organism_its_own_strain(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Two other_organisms ids reclassified in one pass must not be
    credited to each other's strain, even when one of them numerically
    collides with a strain the other reuses.

    Reproduces attempt 2's defect: guarding a HasEnzyme subject against
    `doc["strains"]` read *after* the reclassification loop saw this pass'
    own appended ids and treated the second organism's id (which happened
    to equal the id the first organism's strain was reused under) as
    already a valid strain, leaving its subject unremapped and silently
    correct-looking instead of dropped.
    """
    monkeypatch.setattr(
        fix_taxonomy.ncbitax,
        "decompose_name",
        _decompose_from(
            {
                "Alpha one A1": DecomposedName(
                    species="Alpha one", strain="A1"
                ),
                "Beta two B2": DecomposedName(species="Beta two", strain="B2"),
            }
        ),
    )

    with BrendaDocDB(storage="memory") as docdb:
        # Strain 20 already designates "A1" — coincidentally the same
        # number as the other_organisms id "20" ("Beta two B2") being
        # reclassified.
        docdb.strains.insert(TinyDBDoc({"designations": ["A1"]}, 20))
        docdb.documents.insert(
            TinyDBDoc(
                {
                    "other_organisms": {
                        "10": "Alpha one A1",
                        "20": "Beta two B2",
                    },
                    "bacteria": {},
                    "strains": [],
                    "relations": {
                        "HasEnzyme": [
                            {"subject": 10, "object": 1},
                            {"subject": 20, "object": 2},
                        ],
                        "HasSpecies": [],
                    },
                },
                3,
            )
        )

        with caplog.at_level(logging.WARNING):
            fix_taxonomy.fix_taxonomy(docdb)

        doc = docdb.documents.get(doc_id=3)

    assert not [m for m in caplog.messages if "left in other_organisms" in m]
    assert doc["strains"] == [20, 21]

    has_enzyme = doc["relations"]["HasEnzyme"]
    assert has_enzyme[0] == {"subject": 20, "object": 1}
    assert has_enzyme[1] == {"subject": 21, "object": 2}

    frame = pd.DataFrame(
        [
            {
                "bacteria": repr(doc["bacteria"]),
                "other_organisms": repr(doc["other_organisms"]),
                "strains": repr(doc["strains"]),
                "enzymes": repr([1, 2]),
                "relations": repr(doc["relations"]),
            }
        ]
    )

    with caplog.at_level(logging.WARNING):
        processed = preprocess_labels(frame)

    assert not [m for m in caplog.messages if "dropped" in m]

    pairs = processed.iloc[0]["relations"][0]
    assert ("enz1", "str20") in pairs
    assert ("enz2", "str21") in pairs
    assert list(pairs[("enz1", "str20")]) == [1.0, 0.0, 0.0]
    assert list(pairs[("enz2", "str21")]) == [1.0, 0.0, 0.0]


def test_root_only_decomposition_stays_in_other_organisms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`decompose_name` can return a hit with neither species nor strain
    (only the taxonomy root reached). Reclassifying it used to crash with
    `TypeError` from `str.removeprefix(None)`; it must instead stay put.
    """
    monkeypatch.setattr(
        fix_taxonomy.ncbitax,
        "decompose_name",
        _decompose_from(
            {"Unplaceable organism": DecomposedName(species=None, strain=None)}
        ),
    )

    with BrendaDocDB(storage="memory") as docdb:
        docdb.documents.insert(
            TinyDBDoc(
                {
                    "other_organisms": {"1": "Unplaceable organism"},
                    "bacteria": {},
                    "strains": [],
                    "relations": {"HasEnzyme": [], "HasSpecies": []},
                },
                4,
            )
        )

        fix_taxonomy.fix_taxonomy(docdb)
        doc = docdb.documents.get(doc_id=4)

    assert doc["other_organisms"] == {"1": "Unplaceable organism"}
    assert doc["bacteria"] == {}
    assert doc["strains"] == []
