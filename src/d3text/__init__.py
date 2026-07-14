try:
    import stackprinter

    stackprinter.set_excepthook(style="darkbg2")
except ModuleNotFoundError:
    print(
        "pip install stackprinter if you want stackprinter's exception messages."
    )

try:
    from beartype.claw import beartype_this_package

    beartype_this_package()
except ModuleNotFoundError:
    print("pip install beartype if you want runtime type-checking.")

# Imported below the claw so that beartype's import hook instruments it.
from d3text.schema import EntityType, RelationType, Schema  # noqa: E402

__all__ = ["EntityType", "RelationType", "Schema"]
