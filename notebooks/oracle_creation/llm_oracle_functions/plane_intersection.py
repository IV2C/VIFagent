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
    return present("Pi_2") and shape("Pi_2", "plane") and aligned("Pi_2", "x-y-plane", axis="horizontal") and present("intersection of Pi_2 and existing plane") and within("intersection of Pi_2 and existing plane", "z-axis")
