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
        size("squid's bottom left tentacle", (1.5, 1.0))
        and size("squid's bottom right tentacle", (1.5, 1.0))
        and size("squid's top left tentacle", (1.0, 1.3))
        and size("squid's top right tentacle", (1.0, 1.3))
    )
