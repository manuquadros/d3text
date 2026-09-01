try:
    from d3text.excepthook import install as _install_excepthook

    _install_excepthook(style="darkbg2")
except ModuleNotFoundError:
    print(
        "pip install stackprinter if you want stackprinter's exception messages."
    )

try:
    from beartype.claw import beartype_this_package

    beartype_this_package()
except ModuleNotFoundError:
    print("pip install beartype if you want runtime type-checking.")

# Imported after `beartype_this_package()` so the claw's import hook is already
# installed and `schema` gets instrumented like the rest of the package.
from d3text.schema import (  # noqa: E402
    EntityType as EntityType,
    RelationType as RelationType,
    Schema as Schema,
)
