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
    left_aligned = aligned("left sphere 1", "left sphere 2", axis="vertical")
    right_aligned = aligned("right sphere 1", "right sphere 2", axis="vertical")
    arrow_adjusted = visual_property("The arrow is adjusted to point from the aligned left spheres to the aligned right spheres")
    return left_aligned and right_aligned and arrow_adjusted
