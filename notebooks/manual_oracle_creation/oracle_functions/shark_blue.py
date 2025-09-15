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
    return color("pectoral fins", "blue") and color("outer shark's skin", "blue")
