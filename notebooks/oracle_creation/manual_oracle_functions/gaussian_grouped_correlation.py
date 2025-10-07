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
        present("dashed box")
        and present('annotation "correlation"')
        and position('annotation "correlation"', "dashed box", "under")
        and position('annotation "correlation"', "dashed box", "right")
    )
