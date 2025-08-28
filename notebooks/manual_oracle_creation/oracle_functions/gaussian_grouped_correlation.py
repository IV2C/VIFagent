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
        added("dashed box")
        and added('annotation "correlation"')
        and position('annotation "correlation"', "dashed box", "under")
        and position('annotation "correlation"', "dashed box", "right")
    )
