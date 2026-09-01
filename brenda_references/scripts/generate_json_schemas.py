"""Generate JSON schemas from Pydantic models in the d3types package.

This module provides functionality to automatically generate JSON schemas for all
Pydantic BaseModel subclasses defined in the d3types module.

Example:
    $ python generate_json_schemas.py > d3types.schema.json

The script will print the JSON schema for each Pydantic model found in d3types.
"""

from pprint import pp

from pydantic import BaseModel

import d3types


def main() -> None:  # noqa: D103
    for name in dir(d3types):
        attr = getattr(d3types, name)

        if (
            isinstance(attr, type)
            and issubclass(attr, BaseModel)
            and attr != BaseModel
        ):
            pp(attr.model_json_schema())


if __name__ == "__main__":
    main()
