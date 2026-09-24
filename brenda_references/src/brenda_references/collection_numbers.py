"""Culture-collection accession shape, gating what a StrainInfo match may join.

`fix_missing_strains.py` is the only caller. `apiadapters.straininfo.
StrainInfoAdapterBase.retrieve_strain_models` joins a record StrainInfo
returns to whichever BRENDA strain shares its *first* matching designation,
with no organism constraint, so a short or generic designation (`A2`, `F1`,
`UC`) regularly names a strain of an unrelated species. A designation with
the shape of a real culture-collection accession — an acronym a real
collection issues, followed by its deposit number — is a registry identifier
and unambiguous by construction; nothing else is safe to join on.
"""

import re

COLLECTIONS = frozenset(
    {
        "ACM",
        "ACOI",
        "ATCC",
        "ATHUBA",
        "AS",
        "BACA",
        "BCC",
        "BCRC",
        "BEA",
        "BMCC",
        "BRFM",
        "CAIM",
        "CBMAI",
        "CBS",
        "CCAC",
        "CCAP",
        "CCM",
        "CCMM",
        "CCMP",
        "CCOS",
        "CCP",
        "CCRC",
        "CCT",
        "CCTCC",
        "CCUG",
        "CCY",
        "CDBB",
        "CECT",
        "CFBP",
        "CGMCC",
        "CGSC",
        "CIM",
        "CIP",
        "CIRMBP",
        "CLIB",
        "CNCTC",
        "CPCC",
        "CRBIP",
        "DBVPG",
        "DCG",
        "DMST",
        "DSM",
        "DSMZ",
        "ETH",
        "FBCC",
        "FGSC",
        "FRR",
        "HAMBI",
        "HKI",
        "HUT",
        "IAFB",
        "IAM",
        "ICCF",
        "ICMP",
        "IEGM",
        "IFO",
        "IHEM",
        "IMET",
        "IMI",
        "IMRU",
        "IPPAS",
        "ITEM",
        "ITM",
        "JCM",
        "JMRC",
        "KACC",
        "KCCM",
        "KCTC",
        "KMM",
        "KPD",
        "LEGE",
        "LEGECC",
        "LMD",
        "LMG",
        "MSCL",
        "MSCU",
        "MTCC",
        "MUCL",
        "MUM",
        "NBIMCC",
        "NBRC",
        "NCAIM",
        "NCCB",
        "NCDO",
        "NCFB",
        "NCIB",
        "NCIM",
        "NCIMB",
        "NCMA",
        "NCMB",
        "NCPF",
        "NCPPB",
        "NCPV",
        "NCTC",
        "NCYC",
        "NIES",
        "NIVA",
        "NORCCA",
        "NRRL",
        "OCM",
        "PCC",
        "PCM",
        "PDDCC",
        "PTCC",
        "PYCC",
        "RAH",
        "RAV",
        "RAX",
        "RCC",
        "SAG",
        "SCCAP",
        "STH",
        "STI",
        "TBRC",
        "TISTR",
        "TUCC",
        "UAMH",
        "UCCCB",
        "UHCC",
        "UIO",
        "ULC",
        "UMCC",
        "UTCC",
        "UTEX",
        "VKM",
        "VTT",
        "YIM",
        "ZIMET",
    }
)
"""Acronyms of the culture collections BRENDA's strain designations name.

Mirrors `d3text.surface_forms.COLLECTIONS` — vendored from DSMZ's cafi
(https://github.com/LeibnizDSMZ/cafi, commit
effeca350ac72faeb01d19c2c14830a905c5d116, `src/cafi/data/acr_db.json`),
data licensed CC-BY-4.0, attribution DSMZ / LeibnizDSMZ, less the acronyms
`d3text.surface_forms._ACCESSION_COLLISIONS` excludes for reading a
non-deposit as an accession (`ST` for MLST sequence types among them).
`brenda_references` cannot import the d3text module holding that
derivation: `d3text` depends on `brenda_references`, not the other way
round, and adding the reverse edge would be a real dependency cycle, not a
header worth adding for one frozenset — so the vetted list is copied rather
than shared, and `tests/test_surface_forms.py` pins the two copies equal.
Matched case-sensitively for the reason `d3text.surface_forms.COLLECTIONS`
gives: `AS` is a collection and also two ordinary letters.
"""

_BODY = r"\d+(?:[.\-/]\d+)*[A-Za-z]?"

_PATTERN = re.compile(
    r"^("
    + "|".join(
        sorted(COLLECTIONS, key=lambda acronym: (-len(acronym), acronym))
    )
    + r")[ .\-]{0,2}("
    + _BODY
    + r")$"
)
"""An acronym, an optional short separator, then a digit-led deposit number.

Anchored on the whole string, unlike `d3text.surface_forms.ACCESSION`, which
`find`s an accession inside running text: `normalize_strain_names` has
already isolated one designation per call here, so there is nothing to
search for the pattern *within*. The body allows a dot/dash/slash-joined
numeric tail (`CBS 111.30`) for the collections that number that way; a
designation the grammar reads only in part — `ATCC BAA-245`, `LMG
16656QC1/01` — is rejected rather than truncated, since the truncation would
assert an accession BRENDA never wrote.
"""


def is_collection_number(designation: str) -> bool:
    """Whether `designation` has culture-collection-accession shape.

    :param designation: one BRENDA strain designation.
    :return: True if it reads as a known collection's acronym followed by a
        deposit number, false otherwise.
    """
    return _PATTERN.match(designation.strip()) is not None


__all__ = ["COLLECTIONS", "is_collection_number"]
