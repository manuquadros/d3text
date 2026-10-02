"""Convert StrainInfo API records into the canonical `d3types.Strain`."""

from apiadapters.straininfo import StrainRecord
from d3types import Strain


def strain_record_to_d3types(record: StrainRecord) -> Strain:
    """Validate a StrainInfo strain record as a `d3types.Strain`.

    The record's top-level fields map onto the model's as they are; its
    `relation.culture` and `relation.designation` lists become `cultures`
    and `designations`, each empty when the record has none.

    :param record: one strain object returned by the StrainInfo v1 API.
    :return: the record as a `d3types.Strain`.
    :raises pydantic.ValidationError: if a field of the record does not fit
        the type of the `d3types.Strain` field it fills.
    """
    relation = record["relation"]
    return Strain.model_validate(
        {
            **record,
            "cultures": relation.get("culture", []),
            "designations": relation.get("designation", []),
        }
    )
