import pandas as pd
from apiadapters.ncbi.parser import is_scanned
from brenda_references.brenda_references import preprocess_labels


def test_is_scanned():
    xml = """<jats:body xmlns:jats=\"https://jats.nlm.nih.gov/ns/archiving/1.3/\">\n    <jats:supplementary-material content-type=\"scanned-pages\" position=\"float\">\n      <jats:graphic xmlns:xlink=\"http://www.w3.org/1999/xlink\" position=\"float\" xlink:href=\"brjcancer00184-0039.tif\" xlink:role=\"969\" xlink:title=\"scanned-page\"></jats:graphic>\n      <jats:graphic xmlns:xlink=\"http://www.w3.org/1999/xlink\" position=\"float\" xlink:href=\"brjcancer00184-0040.tif\" xlink:role=\"970\" xlink:title=\"scanned-page\"></jats:graphic>\n      <jats:graphic xmlns:xlink=\"http://www.w3.org/1999/xlink\" position=\"float\" xlink:href=\"brjcancer00184-0041.tif\" xlink:role=\"971\" xlink:title=\"scanned-page\"></jats:graphic>\n      <jats:graphic xmlns:xlink=\"http://www.w3.org/1999/xlink\" position=\"float\" xlink:href=\"brjcancer00184-0042.tif\" xlink:role=\"972\" xlink:title=\"scanned-page\"></jats:graphic>\n      <jats:graphic xmlns:xlink=\"http://www.w3.org/1999/xlink\" position=\"float\" xlink:href=\"brjcancer00184-0043.tif\" xlink:role=\"973\" xlink:title=\"scanned-page\"></jats:graphic>\n      <jats:graphic xmlns:xlink=\"http://www.w3.org/1999/xlink\" position=\"float\" xlink:href=\"brjcancer00184-0044.tif\" xlink:role=\"974\" xlink:title=\"scanned-page\"></jats:graphic>\n      <jats:graphic xmlns:xlink=\"http://www.w3.org/1999/xlink\" position=\"float\" xlink:href=\"brjcancer00184-0045.tif\" xlink:role=\"975\" xlink:title=\"scanned-page\"></jats:graphic>\n      <jats:graphic xmlns:xlink=\"http://www.w3.org/1999/xlink\" position=\"float\" xlink:href=\"brjcancer00184-0046.tif\" xlink:role=\"976\" xlink:title=\"scanned-page\"></jats:graphic>\n      <jats:graphic xmlns:xlink=\"http://www.w3.org/1999/xlink\" position=\"float\" xlink:href=\"brjcancer00184-0047.tif\" xlink:role=\"977\" xlink:title=\"scanned-page\"></jats:graphic>\n    </jats:supplementary-material>\n  </jats:body>"""

    assert is_scanned(xml) is True


def test_none_fill_spells_pairs_the_way_the_typed_keys_are_spelled() -> None:
    """The `none` fill must key a pair the way a typed key of it would be.

    The fill walks the entity columns (bacteria, enzymes, strains,
    other_organisms) while the typed keys are sorted, and the two orders
    disagree for a (strain, other_organism) pair. A fill key spelled in
    column order would miss a typed key of the same pair, leaving the
    document holding that pair twice under two different labels.
    """
    frame = pd.DataFrame(
        {
            "bacteria": ["{}"],
            "enzymes": ["[5]"],
            "strains": ["[3]"],
            "other_organisms": ["{7: 'Vibrio sp.'}"],
            "relations": ["{'HasEnzyme': [{'subject': 3, 'object': 5}]}"],
        }
    )

    processed = preprocess_labels(frame)
    pairs = processed["relations"].iloc[0][0]

    assert processed["entities"].iloc[0] == ["enz5", "str3", "oth7"]
    assert set(pairs) == {
        ("enz5", "str3"),
        ("enz5", "oth7"),
        ("oth7", "str3"),
    }
    assert pairs[("enz5", "str3")].tolist() == [1.0, 0.0, 0.0]
