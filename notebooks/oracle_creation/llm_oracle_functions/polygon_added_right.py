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
    return present("second polygon with 5 sides") and placement("second polygon with 5 sides", "first polygon", direction="right") and visual_property("The second polygon is numbered from 1 to 5")
