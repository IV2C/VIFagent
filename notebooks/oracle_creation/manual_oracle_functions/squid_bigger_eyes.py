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
    return size("squid's left eye", (1.5, 1.5)) and size(
        "squid's right eye", (1.5, 1.5)
    )
