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
        present("left eye's pupil")
        and present("right eye's pupil")
        and color("left eye sclera", "white")
        and color("right eye sclera", "white")
    )
