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
    return present("E_6") and within("bottom-left vertex on the left", "E_6") and within("bottom-left vertex on the right", "E_6") and angle("E_6", degree=-45)
