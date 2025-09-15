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
        added("Motor 1")
        and added("Motor 2")
        and within("Motor 1", "green container")
        and within("Motor 2", "green container")
    )
