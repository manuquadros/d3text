from xmlparser import (chars, fromstring, merge_children, promote_spans,
                       reinsert_tags, remove_tags, tostring, transform_article)

tryptophan = (
    "<div>with the indole precursor <sc>l</sc>-tryptophan, we observed</div>"
)
italic = "<div>with the <italic>indole precursor l-tryptophan</italic>, we observed</div>"
spaced_tag_string = (
    '<sec id="s4.12"><title>CE-ESI-TOF-MS target analysis.</title></sec>'
)
spanseq = (
    '<root><italic><span resource="#T3" typeof="ncbitaxon:Strain">P</span></italic>'
    '<span resource="#T3" typeof="ncbitaxon:Strain">2</span>'
    '<sub><span resource="#T3" typeof="ncbitaxon:Strain">1</span></sub></root>'
)
spanlifted = (
    '<root><span resource="#T3" typeof="ncbitaxon:Strain"><italic>P</italic></span>'
    '<span resource="#T3" typeof="ncbitaxon:Strain">2</span>'
    '<span resource="#T3" typeof="ncbitaxon:Strain"><sub>1</sub></span></root>'
)


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
    assert (
        reinsert_tags(annotated_tryptophan, tryptophan) == expected_tryptophan
    )

    annotated_italic = (
        'with the indole precursor <span typeof="entity">'
        "l-tryptophan</span>, we observed"
    )
    expected_italic = (
        "<div>with the <italic>indole precursor "
        '<span typeof="entity">l-tryptophan</span></italic>, we observed</div>'
    )
    assert reinsert_tags(annotated_italic, italic) == expected_italic


def test_div_with_attribs():
    div = (
        '<div prefix="ncbitaxon: http://purl.obolibrary.org/obo/NCBITaxon_"> '
        "Crystallization and preliminary X-ray diffraction analysis of two N-terminal "
        "fragments of the DNA-cleavage domain of topoisomerase IV from <span"
        ' resource="#T1" typeof="ncbitaxon:Species">Staphylococcus aureus</span>'
    )
    assert (
        next(chars(div))
        == '<div prefix="ncbitaxon: http://purl.obolibrary.org/obo/NCBITaxon_"> '
    )


def test_spans_to_the_top():
    tree = fromstring(spanseq)
    assert tostring(promote_spans(tree), encoding="unicode") == spanlifted


def test_cousin_spans_should_be_merged_when_possible():
    tree = fromstring(spanlifted)
    assert (
        tostring(merge_children(tree), encoding="unicode")
        == '<root><span resource="#T3" typeof="ncbitaxon:Strain"><italic>P</italic>2<sub>1</sub></span></root>'
    )

def test_transform_article():
    chunk = """<chunk><journal-meta xmlns="https://dtd.nlm.nih.gov/ns/archiving/2.3/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">       <journal-id journal-id-type="nlm-ta">BMC Biotechnol</journal-id>       <journal-title>BMC Biotechnology</journal-title>       <issn pub-type="epub">1472-6750</issn>       <publisher>         <publisher-name>BioMed Central</publisher-name>         <publisher-loc>London</publisher-loc>       </publisher>     </journal-meta> <article-meta xmlns="https://dtd.nlm.nih.gov/ns/archiving/2.3/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">       <article-id pub-id-type="accession">PMC101390</article-id>       <article-id pub-id-type="pmcid">PMC101390</article-id>       <article-id pub-id-type="pmc-uid">101390</article-id>       <article-id pub-id-type="pmid">11914155</article-id>       <article-id pub-id-type="publisher-id">1472-6750-2-3</article-id>       <article-id pub-id-type="pmid">11914155</article-id>       <article-id pub-id-type="doi">10.1186/1472-6750-2-3</article-id>       <article-categories>         <subj-group subj-group-type="heading">           <subject>Research Article</subject>         </subj-group>       </article-categories>       <title-group>         <article-title><italic>Rhodococcus erythropolis</italic> ATCC 25544 as a suitable source of cholesterol oxidase: cell-linked and extracellular enzyme synthesis, purification and concentration</article-title>       </title-group>       <contrib-group>         <contrib contrib-type="author" id="A1">           <name>             <surname>Sojo</surname>             <given-names>Mar M</given-names>           </name>           <xref ref-type="aff" rid="I1">1</xref>           <email>msojo@um.es</email>         </contrib>         <contrib contrib-type="author" corresp="yes" id="A2">           <name>             <surname>Bru</surname>             <given-names>Roque R</given-names>           </name>           <xref ref-type="aff" rid="I2">2</xref>           <email>Roque.Bru@ua.es</email>         </contrib>         <contrib contrib-type="author" id="A3">           <name>             <surname>García-Carmona</surname>             <given-names>Francisco F</given-names>           </name>           <xref ref-type="aff" rid="I1">1</xref>           <email>gcarmona@um.es</email>         </contrib>       </contrib-group>       <aff id="I1"><label>1</label>Departamento de Bioquímica y Biología Molecular-A, Facultad de Biología, Universidad de Murcia, E-30100 Murcia, Spain</aff>       <aff id="I2"><label>2</label>Departamento de Agroquímica y Bioquímica, Facultad de Ciencias, Universidad de Alicante, E-0 3 080 Alicante, Spain</aff>       <pub-date pub-type="collection">         <year>2002</year>       </pub-date>       <pub-date pub-type="epub">         <day>26</day>         <month>3</month>         <year>2002</year>       </pub-date>       <volume>2</volume>       <fpage>3</fpage>       <lpage>3</lpage>       <ext-link xmlns:xlink="http://www.w3.org/1999/xlink" ext-link-type="uri" xlink:href="http://www.biomedcentral.com/1472-6750/2/3"/>       <history>         <date date-type="received">           <day>17</day>           <month>1</month>           <year>2002</year>         </date>         <date date-type="accepted">           <day>26</day>           <month>3</month>           <year>2002</year>         </date>       </history>       <permissions>         <copyright-statement>Copyright © 2002 Sojo et al; licensee BioMed Central Ltd. This is an Open Access article: verbatim copying and redistribution of this article are permitted in all media for any purpose, provided this notice is preserved along with the article's original URL.</copyright-statement>         <copyright-year>2002</copyright-year>         <copyright-holder>Sojo et al; licensee BioMed Central Ltd. This is an Open Access article: verbatim copying and redistribution of this article are permitted in all media for any purpose, provided this notice is preserved along with the article's original URL.</copyright-holder>       </permissions>       <abstract>         <sec>           <title>Background</title>           <p>The suitability of the strain <italic>Rhodococcus erythropolis</italic> ATCC 25544 grown in a two-liter fermentor as a source of cholesterol oxidase has been investigated. The strain produces both cell-linked and extracellular cholesterol oxidase in a high amount, that can be extracted, purified and concentrated by using the detergent Triton X-114.</p>         </sec>         <sec>           <title>Results</title>           <p>A spray-dry method of preparation of the enzyme inducer cholesterol in Tween 20 was found to be superior in both convenience and enzyme synthesis yield to one of heat-mixing. Both were similar as far as biomass yield is concerned. Cell-linked cholesterol oxidase was extracted with Triton X-114, and this detergent was also used for purification and concentration, following temperature-induced detergent phase separation. Triton X-114 was utilized to purify and to concentrate the cell-linked and the extracellular enzyme. Cholesterol oxidase was found mainly in the resulting detergent-rich phase. When Triton X-114 concentration was set to 6% w/v the extracellular, but not the cell-extracted enzyme, underwent a 3.4-fold activation after the phase separation process. This result is interpreted in the light of interconvertible forms of the enzyme that do not seem to be in equilibrium. Fermentation yielded 360 U/ml (672 U/ml after activation), 36% of which was extracellular (65% after activation). The Triton X-114 phase separation step yielded 11.6-fold purification and 20.3-fold concentration.</p>         </sec>         <sec>           <title>Conclusions</title>           <p>The results of this work may make attractive and cost-effective the implementation of this bacterial strain and this detergent in a purification-based industrial production scheme of commercial cholesterol oxidase.</p>         </sec>       </abstract>       <kwd-group>         <kwd>Cholesterol oxidase</kwd>         <kwd>           <italic>Rhodococcus erythropolis ATCC 25544</italic>         </kwd>         <kwd>enzyme purification</kwd>         <kwd>Triton X-114</kwd>         <kwd>phase separation</kwd>       </kwd-group>     </article-meta><chunk-body> <div><p xmlns="https://dtd.nlm.nih.gov/ns/archiving/2.3/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">In a previous work [<xref ref-type="bibr" rid="B9">9</xref>] we described the cell-bound and extracellular cholesterol oxidase activities from <italic><span resource="#T1" typeof="ncbitaxon:Species">R. erythropolis</span></italic> <span resource="#T2" typeof="ncbitaxon:Strain">ATCC 25544</span>, achieving in optimal conditions 55% cell-bound and 45% extracellular activity. Their enzymatic properties strongly supported the idea that the particulate and the extracellular cholesterol oxidases are two different forms of the same enzyme with an estimated molecular mass of 55 kDa. In this work we optimize the culture conditions in a 2-liter fermentor of this extracellular cholesterol oxidase producer strain and carry out the extraction, partial purification and concentration of both types of cholesterol oxidase by using Triton X-114 phase separation. The results obtained are very promising for the use of this strain and this technique in the industrial processing of <span resource="#T3" typeof="ncbitaxon:OOS">bacteria</span> to obtain cholesterol oxidase.</p></div></chunk-body></chunk>"""

    goal = """<chunk><div xmlns="https://jats.nlm.nih.gov/ns/archiving/1.3/" class="metadata"><p>Excerpt from:<strong>Rhodococcus erythropolis ATCC 25544 as a suitable source of cholesterol oxidase: cell-linked and extracellular enzyme synthesis, purification and concentration</strong></p><p>Authors:<name xmlns="https://dtd.nlm.nih.gov/ns/archiving/2.3/"><surname>Sojo,</surname>Mar M            -</name><name xmlns="https://dtd.nlm.nih.gov/ns/archiving/2.3/"><surname>Bru,</surname>Roque R            -</name><name xmlns="https://dtd.nlm.nih.gov/ns/archiving/2.3/"><surname>García-Carmona,</surname>Francisco F</name></p><p>DOI: 10.1186/1472-6750-2-3</p></div><div xmlns="https://jats.nlm.nih.gov/ns/archiving/1.3/" class="chunk-body"><p xmlns="https://dtd.nlm.nih.gov/ns/archiving/2.3/">In a previous work [<xref ref-type="bibr" rid="B9">9</xref>] we described the cell-bound and extracellular cholesterol oxidase activities from<italic><span resource="#T1" typeof="ncbitaxon:Species">R. erythropolis</span></italic><span resource="#T2" typeof="ncbitaxon:Strain">ATCC 25544</span>, achieving in optimal conditions 55% cell-bound and 45% extracellular activity. Their enzymatic properties strongly supported the idea that the particulate and the extracellular cholesterol oxidases are two different forms of the same enzyme with an estimated molecular mass of 55 kDa. In this work we optimize the culture conditions in a 2-liter fermentor of this extracellular cholesterol oxidase producer strain and carry out the extraction, partial purification and concentration of both types of cholesterol oxidase by using Triton X-114 phase separation. The results obtained are very promising for the use of this strain and this technique in the industrial processing of<span resource="#T3" typeof="ncbitaxon:OOS">bacteria</span>to obtain cholesterol oxidase.</p></div></chunk>"""

    goal = tostring(fromstring(goal), method="c14n2", strip_text=True, pretty_print=True)
    transformed = tostring(fromstring(transform_article(chunk)), method="c14n2", strip_text=True, pretty_print=True)
    
    assert transformed == goal
