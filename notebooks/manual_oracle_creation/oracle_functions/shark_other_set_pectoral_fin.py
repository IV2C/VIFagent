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
    Direction
)


def test_valid_customization() -> bool:
    return added("second set of pectoral fins lower on the shark's body") and placement(
        "second set of pectoral fins lower on the shark's body",
        "upper pecotral fins",
        Direction.right
    ) and placement(
        "second set of pectoral fins lower on the shark's body",
        "upper pecotral fins",
        Direction.under
    )