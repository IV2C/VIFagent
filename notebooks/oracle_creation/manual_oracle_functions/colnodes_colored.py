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
    red_dots_condition = (
        color("top most dot in the red zone", "red")
        and color("dot in middle of the red zone", "red")
        and color("bottom most dot in the red zone", "red")
    )
    green_dots_condition = color("left most dot in the green zone", "green") and color(
        "right most dot in the green zone", "green"
    )
    blue_dots_condition = color("right most dot in the blue zone", "blue") and color(
        "bottom most dot in the blue zone", "blue"
    )

    dot_in_both_zone_condition = color(
        "dot in the green and blue zone", "green and blue"
    )

    return (
        red_dots_condition
        and green_dots_condition
        and blue_dots_condition
        and dot_in_both_zone_condition
    )
