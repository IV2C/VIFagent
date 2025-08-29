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
        color("Topmost circular node of the hidden layer", "red")
        and color("Bottommost circular node of the hidden layer", "red")
        and color("Middle circular node of the hidden layer", "red")
    )
