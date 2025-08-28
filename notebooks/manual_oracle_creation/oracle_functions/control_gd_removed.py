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
        removed("GD box at the top of the diagram")
        and removed("white node linked to the GS box on its right")
        and added("Arrow from GS directly to the Y black node")
        and removed("Arrow annotated with Z at the top of the diagram")
    )
