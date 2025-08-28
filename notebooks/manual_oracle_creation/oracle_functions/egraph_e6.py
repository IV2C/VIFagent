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
        added("E_6 zone")
        and within("leftmost node", "E_6 zone")
        and within("node at the end of the ab arrow", "E_6 zone")
    )
