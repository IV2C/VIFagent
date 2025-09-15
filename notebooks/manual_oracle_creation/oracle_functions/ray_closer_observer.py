from vif.falcon.oracle.guided_oracle.expressions import (
    added,
    removed,
    angle,
    color,
    placement,
    position,
    size,
    shape,
    Axis,
    within,
)


def test_valid_customization() -> bool:
    return position(
        "bottom-right observer diopter", "barrier", 0.3, Axis.vertical
    ) and position("bottom-right observer diopter", "barrier", 0.4, Axis.horizontal)
