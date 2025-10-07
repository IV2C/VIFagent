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
    cow_face_color_changed = color("cow face", "brown")
    white_dots_present = present("three white dots") and placement("three white dots", "left side of cow face", direction="under") and placement("three white dots", "cow's left eye", direction="under")
    return cow_face_color_changed and white_dots_present
