try:
    from beartype.claw import beartype_this_package
except ModuleNotFoundError:
    # `beartype` is declared by no runtime dependency of this package — it is
    # a dev-group extra of the root project, which is the only reason these
    # scripts ever imported. The claw is an enhancement, not a precondition.
    pass
else:
    beartype_this_package()
