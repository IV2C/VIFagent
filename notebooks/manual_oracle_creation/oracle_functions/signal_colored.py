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
        color("upper part of the rectangle on the left", "very ligth red")
        and color("middle part of the rectangle on the left", "light red")
        and color("lower part of the rectangle on the left", "red")
    )
