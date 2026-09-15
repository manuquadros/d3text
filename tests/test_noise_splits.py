"""The psycholinguistics noise pool: disjoint per split, stable per process.

These pin a `brenda_references` function from d3text's own gated suite on
purpose. The pool is drawn once per split inside `brenda_dataset`, so a
regression here does not raise — it silently changes which documents a run
trains and evaluates on, and the only suite anything runs before a commit is
this one.
"""

import subprocess
import sys

import pandas as pd
import pytest
from brenda_references.brenda_references import (
    DATA_DIR,
    NOISE_BLOCKS,
    noise_documents,
    psycholinguistics_data,
)


def _pool(size: int = 1000) -> pd.DataFrame:
    return pd.DataFrame({"pubmed_id": range(size), "abstract": [""] * size})


@pytest.fixture
def stub_pool(monkeypatch):
    """Stand in for the 76 MB pool, keeping these tests off the data files."""
    pool = _pool()
    monkeypatch.setattr(
        "brenda_references.brenda_references.psycholinguistics_data",
        lambda: pool,
    )
    return pool


def test_noise_blocks_do_not_overlap():
    """No fraction of the pool belongs to two splits.

    This is the property that keeps a noise article out of both training and
    test; every other guarantee below rests on it.
    """
    spans = sorted(NOISE_BLOCKS.values())
    for (_, first_end), (second_start, _) in zip(spans, spans[1:]):
        assert first_end <= second_start


def test_each_split_draws_from_its_own_block(stub_pool):
    draws = {
        split: set(noise_documents(split, 50)["pubmed_id"])
        for split in NOISE_BLOCKS
    }
    for split, drawn in draws.items():
        others = set().union(
            *(ids for name, ids in draws.items() if name != split)
        )
        assert not drawn & others


def test_repeated_draws_return_the_same_documents(stub_pool):
    """The exhausted-iterator regression.

    `psycholinguistics_data` used to hand back a `@cache`d *iterator*, so the
    second draw in a process got only the tail the first had left. A tuning
    sweep re-builds the dataset once per trial, which meant trial 2's
    validation and test splits ran with no noise at all and no trial was
    comparable to any other.
    """
    first = noise_documents("training", 450)
    second = noise_documents("training", 450)
    assert list(first["pubmed_id"]) == list(second["pubmed_id"])
    assert len(second) == 450


def test_a_short_block_raises_rather_than_returning_fewer(stub_pool):
    """Running short must fail loudly: silently handing back fewer noise
    documents than asked for is exactly how the old bug ran whole sweeps."""
    too_many = len(stub_pool) + 1
    with pytest.raises(ValueError, match="fewer than"):
        noise_documents("training", too_many)


def test_an_unknown_split_raises(stub_pool):
    with pytest.raises(ValueError, match="no noise block"):
        noise_documents("holdout", 10)


def test_no_noise_requested_draws_nothing(stub_pool):
    assert noise_documents("training", 0).empty


_PERMUTATION_PROBE = """
import json
from brenda_references.brenda_references import noise_documents
print(json.dumps({
    split: [int(x) for x in noise_documents(split, 20)["pubmed_id"]]
    for split in ("training", "validation", "test")
}))
"""


@pytest.mark.integration
def test_the_permutation_is_the_same_in_every_process():
    """`train` and `evaluate` are separate processes and must agree.

    The permutation used to be unseeded, so each process drew a different one:
    measured on the real pool, 25 of `evaluate`'s 50 test-noise articles had
    been in `train`'s training noise. Only a subprocess reproduces this — a
    second call in *this* process would be served by `@cache` and agree
    trivially.
    """
    runs = {
        subprocess.run(
            [sys.executable, "-c", _PERMUTATION_PROBE],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        for _ in range(2)
    }
    assert len(runs) == 1


# Neither the 1.1 GB BRENDA entity dump nor the 76 MB psycholinguistics pool
# ships in the repo, so this test can only run where both have been fetched
# locally / on a self-hosted runner. Guard on the files it reads, so a fresh
# checkout and hosted CI skip cleanly instead of erroring.
_DOCUMENTS_PATH = DATA_DIR / "documents.json"
_PSYLING_PATH = DATA_DIR / "pmc_linguistics_articles.json"


@pytest.mark.integration
@pytest.mark.skipif(
    not (_DOCUMENTS_PATH.exists() and _PSYLING_PATH.exists()),
    reason=(
        f"needs the BRENDA entity dump at {_DOCUMENTS_PATH} and the "
        f"psycholinguistics pool at {_PSYLING_PATH}; local/self-hosted only"
    ),
)
def test_psycholinguistics_data_names_no_enzyme():
    """The pool's own claim, checked rather than asserted.

    `psycholinguistics_data` hardcodes every row's `enzymes` column to `[]`;
    this rescreens the whole pool against the same surface-form index and
    descriptive-match reading the labelling pipeline matches against, and
    fails if any row still names an enzyme under it.
    """
    from d3text import negative_screen, surface_forms
    from d3text.corpus import document_text

    index = surface_forms.build_index(
        surface_forms.brenda_surface_forms(
            surface_forms.load_entity_tables(_DOCUMENTS_PATH)
        )
    )

    contaminated = [
        row["pmc_id"]
        for _, row in psycholinguistics_data().iterrows()
        if negative_screen.DESCRIPTIVE.rejects(
            negative_screen.matched_forms(
                document_text(row["abstract"], row["fulltext"]), index
            )
        )
    ]
    assert not contaminated
