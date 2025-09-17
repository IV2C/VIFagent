from vif.falcon.oracle.guided_oracle.expressions import (
    present,
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
    return position(
        "inner circle",
        "left part of the outer circle, between the two leftmost dots",
        0.5,
        "horizontal",
    )
