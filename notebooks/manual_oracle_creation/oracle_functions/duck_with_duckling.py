from vif.falcon.oracle.guided_oracle.expressions import (
    added,
    removed,
    angle,
    color,
    placement,
    position,
    size,
    shape,
    within,
)


def test_valid_customization() -> bool:
    return (
        added("small duck")
        and placement("small duck", "big duck", "under")
        and placement("small duck", "big duck", "left")
    )
