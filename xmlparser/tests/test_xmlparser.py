from copy import deepcopy

from xmlparser import (Element, chars, clean_namespaces, copy_curies, curies,
                       fromstring, merge_children, promote_spans,
                       reinsert_tags, remove_tags, tostring, transform_article)

tryptophan = "<div>with the indole precursor <sc>l</sc>-tryptophan, we observed</div>"
italic = (
    "<div>with the <italic>indole precursor l-tryptophan</italic>, we observed</div>"
)
spaced_tag_string = (
    '<sec id="s4.12"><title>CE-ESI-TOF-MS target analysis.</title></sec>'
)
spanseq = (
    '<root><italic><span resource="#T3" typeof="d3o:Strain">P</span></italic>'
    '<span resource="#T3" typeof="d3o:Strain">2</span>'
    '<sub><span resource="#T3" typeof="d3o:Strain">1</span></sub></root>'
)
spanlifted = (
    '<root><span resource="#T3" typeof="d3o:Strain"><italic>P</italic></span>'
    '<span resource="#T3" typeof="d3o:Strain">2</span>'
    '<span resource="#T3" typeof="d3o:Strain"><sub>1</sub></span></root>'
)


def test_copy_curies():
    empty = Element("div")

    another_empty = deepcopy(empty)

    first = Element(
        "div",
        attrib={"prefix": "d3o: https://purl.dsmz.de/schema/"},
    )
    target1 = Element(
        "div",
        attrib={"prefix": "schema: http://schema.org/ dc: http://purl.org/dc/terms/"},
    )
    target2 = Element(
        "div",
        attrib={
            "prefix": (
                "schema: http://schema.org/ dc: http://purl.org/dc/terms/ "
                "d3o: https://purl.dsmz.de/schema/"
            )
        },
    )

    copy_curies(source=first, target=target1)
    assert curies(target1) == curies(target2)

    copy_curies(source=target1, target=first)
    assert curies(first) == curies(target2)

    assert tostring(empty, method="c14n2") == tostring(another_empty, method="c14n2")

    copy_curies(source=empty, target=first)
    assert tostring(empty, method="c14n2") == tostring(another_empty, method="c14n2")

    assert curies(first) == curies(target2)

    copy_curies(source=empty, target=another_empty)

    assert tostring(empty, method="c14n2") == tostring(another_empty, method="c14n2")


def test_extract_curies():
    div = Element(
        "div",
        attrib={
            "prefix": (
                "schema: http://schema.org/ dc: http://purl.org/dc/terms/ "
                "d3o: https://purl.dsmz.de/schema/"
            )
        },
    )

    assert curies(div) == {
        "schema: http://schema.org/",
        "dc: http://purl.org/dc/terms/",
        "d3o: https://purl.dsmz.de/schema/",
    }


def test_non_tag_chars_iterator_works() -> None:
    assert list(chars("precursor <sc>l</sc>-tryptophan")) == [
        "p",
        "r",
        "e",
        "c",
        "u",
        "r",
        "s",
        "o",
        "r",
        " ",
        "<sc>l</sc>",
        "-",
        "t",
        "r",
        "y",
        "p",
        "t",
        "o",
        "p",
        "h",
        "a",
        "n",
    ]


def test_remove_and_reinsert_tags_are_inverses() -> None:
    assert reinsert_tags(remove_tags(tryptophan), tryptophan) == tryptophan
    assert (
        reinsert_tags(remove_tags(spaced_tag_string), spaced_tag_string)
        == spaced_tag_string
    )


def test_remove_and_insert_with_annotation_is_valid_html() -> None:
    annotated_tryptophan = (
        "with the indole precursor "
        '<span typeof="entity">l</span>-tryptophan, we observed'
    )
    expected_tryptophan = (
        "<div>with the indole precursor "
        '<sc><span typeof="entity">l</span></sc>-tryptophan, we observed</div>'
    )
    assert reinsert_tags(annotated_tryptophan, tryptophan) == expected_tryptophan

    annotated_italic = (
        'with the indole precursor <span typeof="entity">'
        "l-tryptophan</span>, we observed"
    )
    expected_italic = (
        "<div>with the <italic>indole precursor "
        '<span typeof="entity">l-tryptophan</span></italic>, we observed</div>'
    )
    assert reinsert_tags(annotated_italic, italic) == expected_italic


def test_reinsert_tags_preserves_outer_div():
    original = [
        '<p>The bacteria were grown on the GYS medium in a 2-liter scale fermentor in batch mode operation under pH and temperature controlled conditions. Under this conditions the cell yield was doubled (9.5 mg/ml vs. 4.8 mg/ml dry cell weight) and the cultivation time was reduced to one third (60 vs. 180 hours) as compared with shaken flasks. These results are in good agreement with the literature [<xref ref-type="bibr" rid="B12">12</xref>]. We found that addition of 2 g/l cholesterol to the culture broth [<xref ref-type="bibr" rid="B12">12</xref>], prepared as an aqueous emulsion with the aid of Tween 80 at a weight ratio 2:1 results in a high yield of COX production [<xref ref-type="bibr" rid="B9">9</xref>], but the preparation procedure of that emulsion had a marked influence in the final enzyme yield, although not on the cell weight, as seen in Table <xref ref-type="table" rid="T1">1</xref>. The spray-dry method resulted advantageous because the cholesterol :Tween 80 emulsion formed readily and COX production increased in overall by three times with respect to the preparation of the cholesterol:Tween 80 mixture at the flame. Enzyme production improvement resulted larger as cell-linked (3.8-fold) than as extracellular (2.3-fold). This overall increase of COX production can be due to a better availability of cholesterol to the cell since particle size obtained by spray-dry is smaller.</p>',
        '<chunk-body><p xmlns="https://jats.nlm.nih.gov/ns/archiving/1.3/" xmlns:ali="http://www.niso.org/schemas/ali/1.0/" xmlns:xlink="http://www.w3.org/1999/xlink" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">Distribution of COX activity among detergent depleted and detergent rich phases after induction of phase separation of cell extracts done with the indicated concentration of Triton X-114. (a) Total activity; (b) Specific activity.</p><p xmlns="https://jats.nlm.nih.gov/ns/archiving/1.3/" xmlns:ali="http://www.niso.org/schemas/ali/1.0/" xmlns:xlink="http://www.w3.org/1999/xlink" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">As Triton X-114 concentration is increased, COX partitions towards the detergent rich phase, increasing its specific activity (Figures <xref ref-type="fig" rid="F2">2b</xref> and <xref ref-type="fig" rid="F3">3b</xref>) thus resulting in enzyme purification and also in enzyme concentration since the volume of the detergent-rich phase is much lower than the initial volume. The 1% concentration of detergent was an exception to this rule since COX partitioned toward the depleted phase under our working conditions. Partitioning of commercial COX in buffers containing 1% Triton X-114 occurred toward the rich phase and was very influenced by the buffer concentration [<xref ref-type="bibr" rid="B16">16</xref>]. Therefore, it seems that the composition of phase separation media is extremely important to the partitioning of particular proteins.</p><p xmlns="https://jats.nlm.nih.gov/ns/archiving/1.3/" xmlns:ali="http://www.niso.org/schemas/ali/1.0/" xmlns:xlink="http://www.w3.org/1999/xlink" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">The purification was made evident by running samples of COX from cells and culture broth in SDS-PAGE gels. Figure <xref ref-type="fig" rid="F4">4</xref> shows that in both cases the detergent-rich phase was enriched in some proteins, including COX, whereas the depleted phase showed other different protein bands.</p><fig xmlns="https://jats.nlm.nih.gov/ns/archiving/1.3/" xmlns:ali="http://www.niso.org/schemas/ali/1.0/" xmlns:xlink="http://www.w3.org/1999/xlink" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" position="float" id="F4">\n          <label>Figure 4</label>\n          <caption>\n            <p>SDS-PAGE of COX fractions using 3% Triton X-114 for extraction, purification and concentration, (a) Cell extracts: lane 1, Mw markers; lane 2, commercial COX; lane 3, total extracted proteins; lane 4, proteins in detergent depleted phase; lane 5, proteins in detergent rich phase, (b) Culture broth: lane 1, Mw markers; lane 2, commercial COX; lane 3, proteins in detergent rich phase; lane 4, total proteins in culture broth; lane 5, proteins in detergent depleted phase. Arrows indicated the COX band.</p>\n          </caption>\n          <graphic xlink:href="1472-6750-2-3-4"/>\n        </fig><p xmlns="https://jats.nlm.nih.gov/ns/archiving/1.3/" xmlns:ali="http://www.niso.org/schemas/ali/1.0/" xmlns:xlink="http://www.w3.org/1999/xlink" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">SDS-PAGE of COX fractions using 3% Triton X-114 for extraction, purification and concentration, (a) Cell extracts: lane 1, Mw markers; lane 2, commercial COX; lane 3, total extracted proteins; lane 4, proteins in detergent depleted phase; lane 5, proteins in detergent rich phase, (b) Culture broth: lane 1, Mw markers; lane 2, commercial COX; lane 3, proteins in detergent rich phase; lane 4, total proteins in culture broth; lane 5, proteins in detergent depleted phase. Arrows indicated the COX band.</p><p xmlns="https://jats.nlm.nih.gov/ns/archiving/1.3/" xmlns:ali="http://www.niso.org/schemas/ali/1.0/" xmlns:xlink="http://www.w3.org/1999/xlink" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">An exceptional result was obtained when performing COX purification from the culture broth supplemented with a 6% w/v Triton X-114. The total activity recovered after phase separation was ca. 3.5-fold that measured in the broth before phase separation. This result suggests that soluble COX produced by the culture is not fully active and that it can be activated by a treatment with 6% Triton X-114 but not with 4% or less. Further increase of Triton X-114 concentration results in no improvement with respect to 6% (results not shown). This phenomenon was not observed with COX extracted from cells, therefore the enzyme most likely exists in a fully active form in the cells.</p></chunk-body>',
    ]

    serialized = [
        '<div prefix="d3o: https://purl.dsmz.de/schema/">The <span resource="#T1" typeof="d3o:OOS">bacteria</span> were grown on the GYS medium in a 2-liter scale fermentor in batch mode operation under pH and temperature controlled conditions. Under this conditions the cell yield was doubled (9.5 mg/ml vs. 4.8 mg/ml dry cell weight) and the cultivation time was reduced to one third (60 vs. 180 hours) as compared with shaken flasks. These results are in good agreement with the literature [12]. We found that addition of 2 g/l cholesterol to the culture broth [12], prepared as an aqueous emulsion with the aid of Tween 80 at a weight ratio 2:1 results in a high yield of COX production [9], but the preparation procedure of that emulsion had a marked influence in the final enzyme yield, although not on the cell weight, as seen in Table 1. The spray-dry method resulted advantageous because the cholesterol :Tween <span resource="#T2" typeof="d3o:Strain">80</span> emulsion formed readily and COX production increased in overall by three times with respect to the preparation of the cholesterol:Tween <span resource="#T3" typeof="d3o:Strain">80</span> mixture at the flame. Enzyme production improvement resulted larger as cell-linked (3.8-fold) than as extracellular (2.3-fold). This overall increase of COX production can be due to a better availability of cholesterol to the cell since particle size obtained by spray-dry is smaller.</div>',
        '<div prefix="d3o: https://purl.dsmz.de/schema/">Distribution of COX activity among detergent depleted and detergent rich phases after induction of phase separation of cell extracts done with the indicated concentration of Triton X-114. (a) Total activity; (b) Specific activity.As Triton X-114 concentration is increased, COX partitions towards the detergent rich phase, increasing its specific activity (Figures 2b and 3b) thus resulting in enzyme purification and also in enzyme concentration since the volume of the detergent-rich phase is much lower than the initial volume. The 1% concentration of detergent was an exception to this rule since COX partitioned toward the depleted phase under our working conditions. Partitioning of commercial COX in buffers containing 1% Triton X-114 occurred toward the rich phase and was very influenced by the buffer concentration [16]. Therefore, it seems that the composition of phase separation media is extremely important to the partitioning of particular proteins.The purification was made evident by running samples of COX from cells and culture broth in SDS-PAGE gels. Figure 4 shows that in both cases the detergent-rich phase was enriched in some proteins, including COX, whereas the depleted phase showed other different protein bands.           Figure 4                        SDS-PAGE of COX fractions using 3% Triton X-114 for extraction, purification and concentration, (a) Cell extracts: lane 1, Mw markers; lane 2, commercial COX; lane 3, total extracted proteins; lane 4, proteins in detergent depleted phase; lane 5, proteins in detergent rich phase, (b) Culture broth: lane 1, Mw markers; lane 2, commercial COX; lane 3, proteins in detergent rich phase; lane 4, total proteins in culture broth; lane 5, proteins in detergent depleted phase. Arrows indicated the COX band.                               SDS-PAGE of COX fractions using 3% Triton X-114 for extraction, purification and concentration, (a) Cell extracts: lane 1, Mw markers; lane 2, commercial COX; lane 3, total extracted proteins; lane 4, proteins in detergent depleted phase; lane 5, proteins in detergent rich phase, (b) Culture broth: lane 1, Mw markers; lane 2, commercial COX; lane 3, proteins in detergent rich phase; lane 4, total proteins in culture broth; lane 5, proteins in detergent depleted phase. Arrows indicated the COX band.An exceptional result was obtained when performing COX purification from the culture broth supplemented with a 6% w/v Triton X-114. The total activity recovered after phase separationseparation was ca. 3.5-fold that measured in the broth before phase separation. This result suggests that soluble COX produced by the culture is not fully active and that it can be activated by a treatment with 6% Triton X-114 but not with 4% or less. Further increase of Triton X-114 concentration results in no improvement with respect to 6% (results not shown). This phenomenon was not observed with COX extracted from cells, therefore the enzyme most likely exists in a fully active form in the cells.</div>',
    ]

    goal = [
        '<p prefix="d3o: https://purl.dsmz.de/schema/">The <span resource="#T1" typeof="d3o:OOS">bacteria</span> were grown on the GYS medium in a 2-liter scale fermentor in batch mode operation under pH and temperature controlled conditions. Under this conditions the cell yield was doubled (9.5 mg/ml vs. 4.8 mg/ml dry cell weight) and the cultivation time was reduced to one third (60 vs. 180 hours) as compared with shaken flasks. These results are in good agreement with the literature [<xref ref-type="bibr" rid="B12">12</xref>]. We found that addition of 2 g/l cholesterol to the culture broth [<xref ref-type="bibr" rid="B12">12</xref>], prepared as an aqueous emulsion with the aid of Tween 80 at a weight ratio 2:1 results in a high yield of COX production [<xref ref-type="bibr" rid="B9">9</xref>], but the preparation procedure of that emulsion had a marked influence in the final enzyme yield, although not on the cell weight, as seen in Table <xref ref-type="table" rid="T1">1</xref>. The spray-dry method resulted advantageous because the cholesterol :Tween <span resource="#T2" typeof="d3o:Strain">80</span> emulsion formed readily and COX production increased in overall by three times with respect to the preparation of the cholesterol:Tween <span resource="#T3" typeof="d3o:Strain">80</span> mixture at the flame. Enzyme production improvement resulted larger as cell-linked (3.8-fold) than as extracellular (2.3-fold). This overall increase of COX production can be due to a better availability of cholesterol to the cell since particle size obtained by spray-dry is smaller.</p>',
        '<chunk-body prefix="d3o: https://purl.dsmz.de/schema/"><p>Distribution of COX activity among detergent depleted and detergent rich phases after induction of phase separation of cell extracts done with the indicated concentration of Triton X-114. (a) Total activity; (b) Specific activity.As Triton X-114 concentration is increased, COX partitions towards the detergent rich phase, increasing its specific activity (Figures <xref ref-type="fig" rid="F2">2b</xref> and <xref ref-type="fig" rid="F3">3b</xref>) thus resulting in enzyme purification and also in enzyme concentration since the volume of the detergent-rich phase is much lower than the initial volume. The 1% concentration of detergent was an exception to this rule since COX partitioned toward the depleted phase under our working conditions. Partitioning of commercial COX in buffers containing 1% Triton X-114 occurred toward the rich phase and was very influenced by the buffer concentration [<xref ref-type="bibr" rid="B16">16</xref>The purification was made evident by running samples of COX from cells and culture broth in SDS-PAGE gels. Figure <xref ref-type="fig" rid="F4">4</xref> shows that in both cases the detergent-rich phase was enriched in some proteins, including COX, whereas the depleted phase showed other different protein bands.</p><fig xmlns:xlink="http://www.w3.org/1999/xlink" position="float" id="F4">           <label>Figure 4</label>           <caption>             <p>SDS-PAGE of COX fractions using 3% Triton X-114 for extraction, purification and concentration, (a) Cell extracts: lane 1, Mw markers; lane 2, commercial COX; lane 3, total extracted proteins; lane 4, proteins in detergent depleted phase; lane 5, proteins in detergent rich phase, (b) Culture broth: lane 1, Mw markers; lane 2, commercial COX; lane 3, proteins in detergent rich phase; lane 4, total proteins in culture broth; lane 5, proteins in detergent depleted phase. Arrows indicated the COX band.</p>           </caption>           <graphic xlink:href="1472-6750-2-3-4"></graphic>         </fig><p>SDS-PAGE of COX fractions using 3% Triton X-114 for extraction, purification and concentration, (a) Cell extracts: lane 1, Mw markers; lane 2, commercial COX; lane 3, total extracted proteins; lane 4, proteins in detergent depleted phase; lane 5, proteins in detergent rich phase, (b) Culture broth: lane 1, Mw markers; lane 2, commercial COX; lane 3, proteins in detergent rich phase; lane 4, total proteins in culture broth; lane 5, proteins in detergent depleted phase. Arrows indicated the COX band.An exceptional result was obtained when performing COX purification from the culture broth supplemented with a 6% w/v Triton X-114. The total activity recovered after phase separationseparation was ca. 3.5-fold that measured in the broth before phase separation. This result suggests that soluble COX produced by the culture is not fully active and that it can be activated by a treatment with 6% Triton X-114 but not with 4% or less. Further increase of Triton X-114 concentration results in no improvement with respect to 6% (results not shown). This phenomenon was not observed with COX extracted from cells, therefore the enzyme most likely exists in a fully active form in </p></chunk-body>',
    ]

    for s, o, g in zip(serialized, original, goal):
        assert reinsert_tags(s, o) == g


def test_div_with_attribs():
    div = (
        '<div prefix="d3o: https://purl.dsmz.de/schema/"> '
        "Crystallization and preliminary X-ray diffraction analysis of two N-terminal "
        "fragments of the DNA-cleavage domain of topoisomerase IV from <span"
        ' resource="#T1" typeof="d3o:Bacteria">Staphylococcus aureus</span>'
    )
    assert next(chars(div)) == '<div prefix="d3o: https://purl.dsmz.de/schema/"> '


def test_spans_to_the_top():
    tree = fromstring(spanseq)
    assert tostring(promote_spans(tree), encoding="unicode") == spanlifted


def test_cousin_spans_should_be_merged_when_possible():
    tree = fromstring(spanlifted)
    assert (
        tostring(merge_children(tree), encoding="unicode")
        == '<root><span resource="#T3" typeof="d3o:Strain"><italic>P</italic>2<sub>1</sub></span></root>'
    )


def test_transform_article():
    chunk = """<chunk><journal-meta xmlns="https://dtd.nlm.nih.gov/ns/archiving/2.3/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">       <journal-id journal-id-type="nlm-ta">BMC Biotechnol</journal-id>       <journal-title>BMC Biotechnology</journal-title>       <issn pub-type="epub">1472-6750</issn>       <publisher>         <publisher-name>BioMed Central</publisher-name>         <publisher-loc>London</publisher-loc>       </publisher>     </journal-meta> <article-meta xmlns="https://dtd.nlm.nih.gov/ns/archiving/2.3/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">       <article-id pub-id-type="accession">PMC101390</article-id>       <article-id pub-id-type="pmcid">PMC101390</article-id>       <article-id pub-id-type="pmc-uid">101390</article-id>       <article-id pub-id-type="pmid">11914155</article-id>       <article-id pub-id-type="publisher-id">1472-6750-2-3</article-id>       <article-id pub-id-type="pmid">11914155</article-id>       <article-id pub-id-type="doi">10.1186/1472-6750-2-3</article-id>       <article-categories>         <subj-group subj-group-type="heading">           <subject>Research Article</subject>         </subj-group>       </article-categories>       <title-group>         <article-title><italic>Rhodococcus erythropolis</italic> ATCC 25544 as a suitable source of cholesterol oxidase: cell-linked and extracellular enzyme synthesis, purification and concentration</article-title>       </title-group>       <contrib-group>         <contrib contrib-type="author" id="A1">           <name>             <surname>Sojo</surname>             <given-names>Mar M</given-names>           </name>           <xref ref-type="aff" rid="I1">1</xref>           <email>msojo@um.es</email>         </contrib>         <contrib contrib-type="author" corresp="yes" id="A2">           <name>             <surname>Bru</surname>             <given-names>Roque R</given-names>           </name>           <xref ref-type="aff" rid="I2">2</xref>           <email>Roque.Bru@ua.es</email>         </contrib>         <contrib contrib-type="author" id="A3">           <name>             <surname>García-Carmona</surname>             <given-names>Francisco F</given-names>           </name>           <xref ref-type="aff" rid="I1">1</xref>           <email>gcarmona@um.es</email>         </contrib>       </contrib-group>       <aff id="I1"><label>1</label>Departamento de Bioquímica y Biología Molecular-A, Facultad de Biología, Universidad de Murcia, E-30100 Murcia, Spain</aff>       <aff id="I2"><label>2</label>Departamento de Agroquímica y Bioquímica, Facultad de Ciencias, Universidad de Alicante, E-0 3 080 Alicante, Spain</aff>       <pub-date pub-type="collection">         <year>2002</year>       </pub-date>       <pub-date pub-type="epub">         <day>26</day>         <month>3</month>         <year>2002</year>       </pub-date>       <volume>2</volume>       <fpage>3</fpage>       <lpage>3</lpage>       <ext-link xmlns:xlink="http://www.w3.org/1999/xlink" ext-link-type="uri" xlink:href="http://www.biomedcentral.com/1472-6750/2/3"/>       <history>         <date date-type="received">           <day>17</day>           <month>1</month>           <year>2002</year>         </date>         <date date-type="accepted">           <day>26</day>           <month>3</month>           <year>2002</year>         </date>       </history>       <permissions>         <copyright-statement>Copyright © 2002 Sojo et al; licensee BioMed Central Ltd. This is an Open Access article: verbatim copying and redistribution of this article are permitted in all media for any purpose, provided this notice is preserved along with the article's original URL.</copyright-statement>         <copyright-year>2002</copyright-year>         <copyright-holder>Sojo et al; licensee BioMed Central Ltd. This is an Open Access article: verbatim copying and redistribution of this article are permitted in all media for any purpose, provided this notice is preserved along with the article's original URL.</copyright-holder>       </permissions>       <abstract>         <sec>           <title>Background</title>           <p>The suitability of the strain <italic>Rhodococcus erythropolis</italic> ATCC 25544 grown in a two-liter fermentor as a source of cholesterol oxidase has been investigated. The strain produces both cell-linked and extracellular cholesterol oxidase in a high amount, that can be extracted, purified and concentrated by using the detergent Triton X-114.</p>         </sec>         <sec>           <title>Results</title>           <p>A spray-dry method of preparation of the enzyme inducer cholesterol in Tween 20 was found to be superior in both convenience and enzyme synthesis yield to one of heat-mixing. Both were similar as far as biomass yield is concerned. Cell-linked cholesterol oxidase was extracted with Triton X-114, and this detergent was also used for purification and concentration, following temperature-induced detergent phase separation. Triton X-114 was utilized to purify and to concentrate the cell-linked and the extracellular enzyme. Cholesterol oxidase was found mainly in the resulting detergent-rich phase. When Triton X-114 concentration was set to 6% w/v the extracellular, but not the cell-extracted enzyme, underwent a 3.4-fold activation after the phase separation process. This result is interpreted in the light of interconvertible forms of the enzyme that do not seem to be in equilibrium. Fermentation yielded 360 U/ml (672 U/ml after activation), 36% of which was extracellular (65% after activation). The Triton X-114 phase separation step yielded 11.6-fold purification and 20.3-fold concentration.</p>         </sec>         <sec>           <title>Conclusions</title>           <p>The results of this work may make attractive and cost-effective the implementation of this bacterial strain and this detergent in a purification-based industrial production scheme of commercial cholesterol oxidase.</p>         </sec>       </abstract>       <kwd-group>         <kwd>Cholesterol oxidase</kwd>         <kwd>           <italic>Rhodococcus erythropolis ATCC 25544</italic>         </kwd>         <kwd>enzyme purification</kwd>         <kwd>Triton X-114</kwd>         <kwd>phase separation</kwd>       </kwd-group>     </article-meta><chunk-body> <p xmlns="https://dtd.nlm.nih.gov/ns/archiving/2.3/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">In a previous work [<xref ref-type="bibr" rid="B9">9</xref>] we described the cell-bound and extracellular cholesterol oxidase activities from <italic><span resource="#T1" typeof="d3o:Bacteria">R. erythropolis</span></italic> <span resource="#T2" typeof="d3o:Strain">ATCC 25544</span>, achieving in optimal conditions 55% cell-bound and 45% extracellular activity. Their enzymatic properties strongly supported the idea that the particulate and the extracellular cholesterol oxidases are two different forms of the same enzyme with an estimated molecular mass of 55 kDa. In this work we optimize the culture conditions in a 2-liter fermentor of this extracellular cholesterol oxidase producer strain and carry out the extraction, partial purification and concentration of both types of cholesterol oxidase by using Triton X-114 phase separation. The results obtained are very promising for the use of this strain and this technique in the industrial processing of <span resource="#T3" typeof="d3o:OOS">bacteria</span> to obtain cholesterol oxidase.</p></chunk-body></chunk>"""

    goal = """<chunk><div class="metadata"><p>Excerpt from:<strong><italic xmlns="https://dtd.nlm.nih.gov/ns/archiving/2.3/">Rhodococcus erythropolis</italic> ATCC 25544 as a suitable source of cholesterol oxidase: cell-linked and extracellular enzyme synthesis, purification and concentration</strong></p><p>Authors:<name xmlns="https://dtd.nlm.nih.gov/ns/archiving/2.3/"><surname>Sojo,</surname>Mar M            -</name><name xmlns="https://dtd.nlm.nih.gov/ns/archiving/2.3/"><surname>Bru,</surname>Roque R            -</name><name xmlns="https://dtd.nlm.nih.gov/ns/archiving/2.3/"><surname>García-Carmona,</surname>Francisco F</name></p><p>DOI: 10.1186/1472-6750-2-3</p></div><div xmlns="https://jats.nlm.nih.gov/ns/archiving/1.3/" class="chunk-body"><p xmlns="https://dtd.nlm.nih.gov/ns/archiving/2.3/">In a previous work [<xref ref-type="bibr" rid="B9">9</xref>] we described the cell-bound and extracellular cholesterol oxidase activities from<italic><span resource="#T1" typeof="d3o:Bacteria">R. erythropolis</span></italic><span resource="#T2" typeof="d3o:Strain">ATCC 25544</span>, achieving in optimal conditions 55% cell-bound and 45% extracellular activity. Their enzymatic properties strongly supported the idea that the particulate and the extracellular cholesterol oxidases are two different forms of the same enzyme with an estimated molecular mass of 55 kDa. In this work we optimize the culture conditions in a 2-liter fermentor of this extracellular cholesterol oxidase producer strain and carry out the extraction, partial purification and concentration of both types of cholesterol oxidase by using Triton X-114 phase separation. The results obtained are very promising for the use of this strain and this technique in the industrial processing of<span resource="#T3" typeof="d3o:OOS">bacteria</span>to obtain cholesterol oxidase.</p></div></chunk>"""

    goal = tostring(
        clean_namespaces(fromstring(goal)),
        method="c14n2",
        strip_text=True,
        pretty_print=True,
    )
    transformed = tostring(
        fromstring(transform_article(chunk)),
        method="c14n2",
        strip_text=True,
        pretty_print=True,
    )

    assert transformed == goal


def test_transform_article_keeps_curies():
    chunk = '<chunk><chunk-body prefix="d3o: https://purl.dsmz.de/schema/"><p>Distribution of COX activity among detergent depleted and detergent rich phases after induction of phase separation of cell extracts done with the indicated concentration of Triton X-114. (a) Total activity; (b) Specific activity.As Triton X-114 concentration is increased, COX partitions towards the detergent rich phase, increasing its specific activity (Figures <xref ref-type="fig" rid="F2">2b</xref> and <xref ref-type="fig" rid="F3">3b</xref>) thus resulting in enzyme purification and also in enzyme concentration since the volume of the detergent-rich phase is much lower than the initial volume. The 1% concentration of detergent was an exception to this rule since COX partitioned toward the depleted phase under our working conditions. Partitioning of commercial COX in buffers containing 1% Triton X-114 occurred toward the rich phase and was very influenced by the buffer concentration [<xref ref-type="bibr" rid="B16">16</xref>The purification was made evident by running samples of COX from cells and culture broth in SDS-PAGE gels. Figure <xref ref-type="fig" rid="F4">4</xref> shows that in both cases the detergent-rich phase was enriched in some proteins, including COX, whereas the depleted phase showed other different protein bands.</p></chunk-body></chunk>'

    goal = """<chunk>
    <div class="metadata">
    <p>Excerpt from:
    <strong/></p>
    
    <p>Authors: </p>
    <p>DOI: </p>
    </div>
    <div class="chunk-body" prefix="d3o: https://purl.dsmz.de/schema/"><p>Distribution of COX activity among detergent depleted and detergent rich phases after induction of phase separation of cell extracts done with the indicated concentration of Triton X-114. (a) Total activity; (b) Specific activity.As Triton X-114 concentration is increased, COX partitions towards the detergent rich phase, increasing its specific activity (Figures <xref ref-type="fig" rid="F2">2b</xref> and <xref ref-type="fig" rid="F3">3b</xref>) thus resulting in enzyme purification and also in enzyme concentration since the volume of the detergent-rich phase is much lower than the initial volume. The 1% concentration of detergent was an exception to this rule since COX partitioned toward the depleted phase under our working conditions. Partitioning of commercial COX in buffers containing 1% Triton X-114 occurred toward the rich phase and was very influenced by the buffer concentration [<xref ref-type="bibr" rid="B16">16</xref>The purification was made evident by running samples of COX from cells and culture broth in SDS-PAGE gels. Figure <xref ref-type="fig" rid="F4">4</xref> shows that in both cases the detergent-rich phase was enriched in some proteins, including COX, whereas the depleted phase showed other different protein bands.</p></div></chunk>"""

    assert tostring(
        fromstring(transform_article(chunk)), method="c14n2", strip_text=True
    ) == tostring(clean_namespaces(fromstring(goal)), method="c14n2", strip_text=True)
