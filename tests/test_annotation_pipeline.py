import pytest
import torch
from d3text.models import NERCTagger
from d3text.models.config import load_model_config
from d3text.models.serialize import serialize_triples
from xmlparser import reinsert_tags, remove_tags

config = load_model_config("entities/models/current_model_config.toml")
model = NERCTagger(config=config)
model.load_state_dict(torch.load("entities/models/current_model.pt"))
model.to(model.device)
model.eval()


# @pytest.mark.skip(reason="only for integration testing")
def test_annotation_returns_the_same_text():
    samples = [
        '<chunk-body> <title xmlns="https://dtd.nlm.nih.gov/ns/archiving/2.3/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">Background</title> <p xmlns="https://dtd.nlm.nih.gov/ns/archiving/2.3/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">Microbial cholesterol oxidases (EC 1.1.3.6) (COX) catalyze the oxidation and isomerization of cholesterol to 4-cholesten-3-one. Interest in these enzymes mostly relies in their utility in the determination of cholesterol in biological samples such as serum and foods [<xref ref-type="bibr" rid="B1">1</xref>], and also in the bioconversion of a number of 3β-hydroxysteroids in organic solvents [<xref ref-type="bibr" rid="B2">2</xref>] and in reverse micelles [<xref ref-type="bibr" rid="B3">3</xref>] (for a recent review see [<xref ref-type="bibr" rid="B5">5</xref>]). Since earliest reports on crude preparations from <italic>Mycobacterium sp.</italic>[<xref ref-type="bibr" rid="B4">4</xref>], cholesterol oxidases have been described in a number of bacteria and fungi [<xref ref-type="bibr" rid="B5">5</xref>]. Enzymatic properties of cholesterol oxidase from <italic>Rhodococcus</italic> strains (some of which named formerly as <italic>Nocardid</italic>) are particularly suitable for use in the analytical determination of cholesterol, in which the hydrogen peroxide formed is used in a chromogenic reaction catalyzed by horseradish peroxidase [<xref ref-type="bibr" rid="B6">6</xref>].</p> <p xmlns="https://dtd.nlm.nih.gov/ns/archiving/2.3/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">The <italic>Rhodococcus</italic> enzyme has been usually reported to be membrane bound, extractable from whole cells by treatment with detergents or trypsin, although no phospholipids are detected in the enzyme extracts [<xref ref-type="bibr" rid="B7">7</xref>]. More recent reports have demonstrated the production of both extracellular and cell-bound cholesterol oxidase by strains of this genus such as <italic>Rhodococcus</italic> sp. GK1 [<xref ref-type="bibr" rid="B8">8</xref>], <italic>R. erythropolis</italic> ATCC 25544 [<xref ref-type="bibr" rid="B9">9</xref>] and the pathogenic specie <italic>R. equi</italic>[<xref ref-type="bibr" rid="B10">10</xref>,<xref ref-type="bibr" rid="B11">11</xref>].</p> <p xmlns="https://dtd.nlm.nih.gov/ns/archiving/2.3/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">The kinetics of enzyme synthesis at both bench and large scale by <italic>Nocardia rhodochrous</italic> (renamed as <italic>Rhodococcus rhodochrous</italic>), a strain that produces only a cell-bound COX, has been studied and the growing conditions for bacterial enzyme synthesis in fermentor were defined [<xref ref-type="bibr" rid="B12">12</xref>].</p> </chunk-body>',
        '<p xmlns="https://dtd.nlm.nih.gov/ns/archiving/2.3/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">The bacteria were grown on the GYS medium in a 2-liter scale fermentor in batch mode operation under pH and temperature controlled conditions. Under this conditions the cell yield was doubled (9.5 mg/ml vs. 4.8 mg/ml dry cell weight) and the cultivation time was reduced to one third (60 vs. 180 hours) as compared with shaken flasks. These results are in good agreement with the literature [<xref ref-type="bibr" rid="B12">12</xref>]. We found that addition of 2 g/l cholesterol to the culture broth [<xref ref-type="bibr" rid="B12">12</xref>], prepared as an aqueous emulsion with the aid of Tween 80 at a weight ratio 2:1 results in a high yield of COX production [<xref ref-type="bibr" rid="B9">9</xref>], but the preparation procedure of that emulsion had a marked influence in the final enzyme yield, although not on the cell weight, as seen in Table <xref ref-type="table" rid="T1">1</xref>. The spray-dry method resulted advantageous because the cholesterol :Tween 80 emulsion formed readily and COX production increased in overall by three times with respect to the preparation of the cholesterol:Tween 80 mixture at the flame. Enzyme production improvement resulted larger as cell-linked (3.8-fold) than as extracellular (2.3-fold). This overall increase of COX production can be due to a better availability of cholesterol to the cell since particle size obtained by spray-dry is smaller.</p>',
    ]

    for sample in samples:
        stripped = remove_tags(sample)
        prediction = next(model.predict(stripped))
        serialized = serialize_triples(prediction)
        retagged = reinsert_tags(serialized, sample)

        assert remove_tags(retagged).strip() == stripped.strip()


def test_dict_tagger_detects_enzymes():
    sample = "In a previous work [9] we described the cell-bound and extracellular cholesterol oxidase activities from R. erythropolis ATCC 25544, achieving in optimal conditions 55% cell-bound and 45% extracellular activity. Their enzymatic properties strongly supported the idea that the particulate and the extracellular cholesterol oxidases are two different forms of the same enzyme with an estimated molecular mass of 55 kDa. In this work we optimize the culture conditions in a 2-liter fermentor of this extracellular cholesterol oxidase producer strain and carry out the extraction, partial purification and concentration of both types of cholesterol oxidase by using Triton X-114 phase separation. The results obtained are very promising for the use of this strain and this technique in the industrial processing of bacteria to obtain cholesterol oxidase."

    prediction = next(model.predict(sample))

    for token in prediction:
        if token.string in ("cholesterol", "cholesterol oxidase"):
            assert token.prediction == "Enzyme"
