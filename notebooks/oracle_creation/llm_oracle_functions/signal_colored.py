from vif.falcon.oracle.guided_oracle.property_expression import visual_property
from vif.falcon.oracle.guided_oracle.expressions import (
    OracleExpression,
    aligned,
    present,
    angle,
    placement,
    position,
    color,
    shape,
    size,
    within,
    mirrored,
)
def test_valid_customization() -> bool:
    left_rectangle_color_A1 = color("left red rectangle at height corresponding to A1", "light pink")
    left_rectangle_color_A2 = color("left red rectangle at height corresponding to A2", "pink")
    left_rectangle_color_A3 = color("left red rectangle at height corresponding to A3", "red")
    return left_rectangle_color_A1 and left_rectangle_color_A2 and left_rectangle_color_A3
