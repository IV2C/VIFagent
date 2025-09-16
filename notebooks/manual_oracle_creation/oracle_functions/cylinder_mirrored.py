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
    return present("cylinder on the left of the vertical axis") and not present(
        "T0 circle inside the cylinder on the left side of the vertical axis"
    )
