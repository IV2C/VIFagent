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
    mirrored_cylinder = mirrored("cylinder", axis="horizontal")
    axes_adapted = visual_property("the length of the coordinate axes are adapted to the new position of the cylinder")
    other_elements_not_mirrored = not mirrored("T", axis="horizontal") and not mirrored("T0", axis="horizontal") and not mirrored("axes", axis="horizontal")
    return mirrored_cylinder and axes_adapted and other_elements_not_mirrored
