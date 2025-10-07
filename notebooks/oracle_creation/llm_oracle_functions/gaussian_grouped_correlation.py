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
    return present("dashed rectangle") and within("uC1", "dashed rectangle") and within("uCC", "dashed rectangle") and within("tC1", "dashed rectangle") and within("tCC", "dashed rectangle") and within("C1", "dashed rectangle") and within("CC", "dashed rectangle") and present('"Correlation" label') and within('"Correlation" label', "dashed rectangle") and placement('"Correlation" label', "dashed rectangle", direction="bottom right")
