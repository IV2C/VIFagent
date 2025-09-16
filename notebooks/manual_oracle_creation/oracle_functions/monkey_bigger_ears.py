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
    return size("monkey's right ear", (1.5, 1.5)) and size(
        "monkey's left ear", (1.5, 1.5)
    )
