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
        added("left eye's pupil")
        and added("right eye's pupil")
        and color("left eye sclera", "white")
        and color("right eye sclera", "white")
    )
