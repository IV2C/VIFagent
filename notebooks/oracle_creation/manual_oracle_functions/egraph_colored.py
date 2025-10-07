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
    return color("left node in the E_5 zone", "blue") and color(
        "right node in the E_5 zone", "blue"
    )
