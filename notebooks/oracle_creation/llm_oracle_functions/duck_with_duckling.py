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
    return present("smaller duck") and placement("smaller duck", "duck", direction="left") and placement("smaller duck", "duck", direction="under") and size("smaller duck", (0.5, 0.5))
