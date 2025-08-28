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
    f3s = "F_3 force vector"
    force_cond = (
        added(f3s)
        and placement(f3s, "F_1 force vector", "right")
        and placement(f3s, "F_2 force vector", "left")
    )
    w1_cond = added("w31 interval") and placement("w31 interval","w11 interval","right") and placement("w31 interval","w21 interval","left")
    w2_cond = added("w32 interval") and placement("w32 interval","w12 interval","right") and placement("w32 interval","w22 interval","left")
    return force_cond and w1_cond and w2_cond
