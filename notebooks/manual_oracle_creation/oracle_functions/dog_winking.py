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
    return (size("left eye", (1, 0.7)) and size("right eye", (1, 1))) or (
        size("right eye", (1, 0.7)) and size("left eye", (1, 1))
    )
