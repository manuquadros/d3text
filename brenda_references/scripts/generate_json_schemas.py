"""Generate JSON schemas from Pydantic models in the brenda_references package.

This module provides functionality to automatically generate JSON schemas for all
Pydantic BaseModel subclasses defined in the brenda_types module.

Example:
    $ python generate_json_schemas.py > brenda_types.schema.json

The script will print the JSON schema for each Pydantic model found in brenda_types.
"""

from pprint import pp

from pydantic import BaseModel

from brenda_references import brenda_types


def main() -> None:  # noqa: D103
    for name in dir(brenda_types):
        attr = getattr(brenda_types, name)

        if isinstance(attr, type) and issubclass(attr, BaseModel) and attr != BaseModel:
            pp(attr.model_json_schema())


if __name__ == "__main__":
    main()
