from entities.serialize import serialize_triples
from utils import Token


def test_merge_same_resource():
    source = (
        Token(
            string="ATCC",
            offset=(121, 125),
            prediction="B-Strain",
            gold_label=None,
        ),
        Token(
            string="255",
            offset=(126, 129),
            prediction="I-Strain",
            gold_label=None,
        ),
        Token(
            string="##44",
            offset=(129, 131),
            prediction="I-Strain",
            gold_label=None,
        ),
        Token(
            string="is",
            offset=(132, 134),
            prediction="O",
            gold_label=None,
        ),
        Token(
            string="ATCC",
            offset=(135, 138),
            prediction="B-Strain",
            gold_label=None,
        ),
        Token(
            string="255",
            offset=(139, 142),
            prediction="I-Strain",
            gold_label=None,
        ),
        Token(
            string="##44",
            offset=(142, 146),
            prediction="I-Strain",
            gold_label=None,
        ),
    )

    space = " " * 121

    target = (
        '<div prefix="d3o: https://purl.dsmz.de/schema/">'
        f"{space}"
        '<span class="entity" resource="#T1" typeof="d3o:Strain">ATCC 25544</span>'
        ' is <span class="entity" resource="#T1" typeof="d3o:Strain">ATCC 25544</span>'
        "</div>"
    )

    assert serialize_triples(source) == target
