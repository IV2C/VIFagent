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
    vertices_within_E5 = ["vertex within E5 at the bottom left", "vertex within E5 at the center", "vertex within E5 at the bottom right"]
    return all(color(vertex, "blue") for vertex in vertices_within_E5)
