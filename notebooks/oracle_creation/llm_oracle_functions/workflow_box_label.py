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
    return visual_property("The top-left black rectangle is labeled with k_1 on its above right") and \
           visual_property("The top-right black rectangle is labeled with k_2 on its above right") and \
           visual_property("The bottom-left black rectangle is labeled with k_3 on its above right") and \
           visual_property("The bottom-right black rectangle is labeled with k_4 on its above right")
