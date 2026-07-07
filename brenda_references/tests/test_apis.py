import pytest
from apiadapters.ncbi import AsyncNCBIAdapter
from apiadapters.straininfo import StrainInfoAdapter
from brenda_references import expand_doc
from d3types import Bacteria, Document, Organism, Strain
from lpsn_interface import get_lpsn, lpsn_id, lpsn_synonyms, name_parts

get_lpsn()
straininfo = StrainInfoAdapter()
caldanaerobacter = Organism(id=1, organism="Caldanaerobacter subterraneus")
thermoanaerobacter = Organism(id=2, organism="Thermoanaerobacter subterraneus")


def test_bacteria_post_init_lpsn_id() -> None:
    bac = Bacteria.model_validate(caldanaerobacter, from_attributes=True)
    assert bac.lpsn_id == 774333


def test_strain_in_bacteria_name_is_detected() -> None:
    bac = Bacteria(id=234, organism="Pyrococcus horikoshii OT3")
    assert name_parts("Pyrococcus horikoshii OT3")["strain"] == "OT3"
    assert bac.organism == "Pyrococcus horikoshii"


def test_subterraneus_synonyms() -> None:
    assert thermoanaerobacter.organism in lpsn_synonyms(
        caldanaerobacter.organism
    )
    assert caldanaerobacter.organism in lpsn_synonyms(
        thermoanaerobacter.organism
    )


def test_lpsn_id_works() -> None:
    assert lpsn_id("Clostridium difficile") == 774867
    assert lpsn_id("Agrobacterium") == 515059


def test_strain_id_retrieval() -> None:
    assert {11469, 35283, 38539, 39812, 66369, 309797, 341518}.issubset(
        straininfo.get_strain_ids("K-12")
    )


@pytest.mark.skip(reason="adjust the format of the test data before testing")
def test_strain_data_retrieval() -> None:
    resp = straininfo.get_strain_data(11469)
    assert resp is not None

    strain = next(iter(resp))
    assert strain == Strain.model_validate(
        {
            "straininfo_id": 11469,
            "taxon": "Escherichia coli",
            "synonyms": {
                "LMG 18221t2",
                "K12 O Rough H48",
                "Lederberg strain K12",
                "Lederberg K12",
                "K12",
                "CIP 54.117, IFO 3301",
                "K-12",
                "CCTM La 2193",
                "LMG 18221t1",
                "J. Lederberg. K12 O Rough H48",
                "CCUG46621",
                "PCM 2560",
                "E. Wollman, Inst. Pasteur",
                "NCTC 10538 - CIP",
            },
            "cultures": {
                "LMG 18221",
                "IFO 3301",
                "NCFB 1984",
                "NCIMB 10083",
                "NCTC 10538",
                "DSM 11250",
                "NCDO 1984",
                "NCIB 10083",
                "NCDO1990",
                "CIP 54.117",
                "CECT 433",
                "CCUG 46621",
                "HUT 8106",
                "NBRC 3301",
                "BCRC 16081",
                "CCRC 16081",
                "CCUG 49263",
                "CGMCC 1.3344",
                "CFBP 5947",
                "VTT E-032275",
                "CNCTC 7388",
            },
        },
    )


def test_strain_info_api_url() -> None:
    assert (
        straininfo.strain_info_api_url(["K-12", "NE1"])
        == "https://api.straininfo.dsmz.de/v1/search/strain/str_des/K-12,NE1"
    )

    assert (
        straininfo.strain_info_api_url(["K-12"])
        == "https://api.straininfo.dsmz.de/v1/search/strain/str_des/K-12"
    )

    assert (
        straininfo.strain_info_api_url([39812, 66469])
        == "https://api.straininfo.dsmz.de/v1/data/strain/max/39812,66469"
    )


@pytest.mark.asyncio
async def test_expand_doc_gets_pmc_open() -> None:
    doc = Document(
        authors="",
        title="",
        journal="",
        volume="",
        pages="",
        year=1986,
        pubmed_id="15018644",
        path="",
    )

    async with AsyncNCBIAdapter() as ncbi:
        updated_doc = await expand_doc(ncbi, doc)
        assert updated_doc.pmc_open is True
