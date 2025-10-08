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
        not present("GD box at the top of the diagram")
        and not present("white node linked to the GS box on its right")
        and present("Arrow from GS directly to the Y black node")
        and not present("Arrow annotated with Z at the top of the diagram")
    )
