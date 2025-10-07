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
    return not present("Block 6") and not present("Block 7") and not present("Block 8") \
           and not present("arrow pointing to Block 8") \
           and size("'Execute' box", (1.0, 2/3)) \
           and position("Block 5", "Core 1", ratio=1.0, axis="vertical")
