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
    return present("monkey's tongue") and placement(
        "monkey's tongue", "monkey's mouth", "right"
    )
