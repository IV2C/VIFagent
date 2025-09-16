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
    leye = "dog's left eye"
    reye = "dog's right eye"
    return (
        size(leye, (1.3, 1.3))
        and size(reye, (1.3, 1.3))
        and color(f"{leye} sclera", "white")
        and color(f"{reye} sclera", "white")
        and color(f"{leye} pupil", "black")
        and color(f"{reye} pupil", "black")
    )
