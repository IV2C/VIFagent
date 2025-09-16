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
    return (
        placement("Memory", "CPU", "over")
        and placement("Storage", "CPU", "over")
        and placement("input", "CPU", "left")
    )
