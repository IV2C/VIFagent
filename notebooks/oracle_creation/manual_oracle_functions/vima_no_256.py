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
        not present("black vertical rectangle over VecSum")
        and not present("black vertical rectangle over Stencil")
        and not present("black vertical rectangle over MatMult")
    )
