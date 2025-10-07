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
    node1 = color("node at the top-right of the green zone", "green") 
    node2 = color("node at the bottom-right of the red zone", "red") 
    node3 = color("node at the top-left of the blue zone", "blue") 
    node4 = color("central node", "green") and color("central node", "blue") and color("central node", "red") 
    node5 = color("node at the bottom-left of the red zone", "red") 
    node6 = color("node at the top-right of the blue zone", "blue") 
    node7 = color("node at the left of the green zone", "green") 
    node8 = color("node at the bottom of the blue zone", "blue") 
    return node1 and node2 and node3 and node4 and node5 and node6 and node7 and node8
