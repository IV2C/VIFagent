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
    return present("white circle behind left black eye") and present("white circle behind right black eye") and within("white circle behind left black eye", "left black eye") and within("white circle behind right black eye", "right black eye")
