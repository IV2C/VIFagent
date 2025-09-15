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
    anot = "text annotation k"
    position_over = (
        placement(anot + "1","top-left black filled rectangle","over")
        and placement(anot + "2","top-right black filled rectangle","over")
        and placement(anot + "3","bottom-left black filled rectangle","over")
        and placement(anot + "4","bottom-right black filled rectangle","over")
    )
    position_right = (
        placement(anot + "1","top-left black filled rectangle","right")
        and placement(anot + "2","top-right black filled rectangle","right")
        and placement(anot + "3","bottom-left black filled rectangle","right")
        and placement(anot + "4","bottom-right black filled rectangle","right")
    )
    return (
        added(anot + "1")
        and added(anot + "2")
        and added(anot + "3")
        and added(anot + "4")
    ) and position_over and position_right
