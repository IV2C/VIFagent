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
    moved_gpu = placement("GPU with 4 Cores", "OpenCL Program", direction="right")
    moved_execute_section = placement("the section under GPU with 4 Cores", "OpenCL Program", direction="right")
    moved_execute_arrow = placement('"Execute" arrow', '"Compile" arrow', direction="right")
    return moved_gpu and moved_execute_section and moved_execute_arrow
