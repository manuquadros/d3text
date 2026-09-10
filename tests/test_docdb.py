from brenda_references.docdb import BrendaDocDB

import pathlib

TESTDB_PATH = pathlib.Path(__file__).parent / "test_files/testdb.json"


def test_strain_search():
    with BrendaDocDB(path=TESTDB_PATH) as docdb:
        assert docdb.strain_by_designation("ATCC 51142") is None

        designations = ("Schizosaccharomyces pombe", "ATCC 201872", "GK1")
        for name in designations:
            assert docdb.strain_by_designation(name).doc_id == 289


def test_bacteria_search():
    with BrendaDocDB(path=TESTDB_PATH) as docdb:
        assert docdb.bacteria_by_name("Crocosphaera subtropica") is None

        for name in ("Streptomyces septatus", "Streptomyces griseocarneus"):
            assert docdb.bacteria_by_name(name).doc_id == 6027


def test_strain_search_skips_null_designations():
    """A `designations: null` record must not stop the search for another."""
    with BrendaDocDB(storage="memory") as docdb:
        docdb.strains.insert(
            {
                "taxon": None,
                "cultures": [],
                "designations": None,
            }
        )
        valid_id = docdb.strains.insert(
            {
                "taxon": None,
                "cultures": [],
                "designations": ["Valid Designation"],
            }
        )

        assert docdb.strain_by_designation("Valid Designation").doc_id == (
            valid_id
        )


def test_bacteria_search_skips_null_synonyms():
    """A `synonyms: null` record must not stop the search for another."""
    with BrendaDocDB(storage="memory") as docdb:
        docdb.bacteria.insert(
            {"organism": "Null Synonyms Organism", "synonyms": None}
        )
        valid_id = docdb.bacteria.insert(
            {"organism": "Valid Organism", "synonyms": ["Valid Synonym"]}
        )

        assert docdb.bacteria_by_name("Valid Synonym").doc_id == valid_id


def test_fulltext_articles_skips_unparseable_fulltext():
    """`null` and non-XML `fulltext` values are excluded, not raised on."""
    with BrendaDocDB(storage="memory") as docdb:
        docdb.documents.insert({"fulltext": None})
        docdb.documents.insert({"fulltext": "plain text, not XML"})
        valid_id = docdb.documents.insert(
            {"fulltext": "<article><body>text</body></article>"}
        )

        articles = docdb.fulltext_articles()

        assert len(articles) == 1
        assert articles[0].doc_id == valid_id
