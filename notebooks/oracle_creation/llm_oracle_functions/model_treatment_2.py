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
    # Check if the new treatment is added
    new_treatment_added = present("second treatment") and present("B_t") and present("B_t+1")
    
    # Check if the existing treatment is renamed
    treatment_renamed = present("first treatment") and not present("treatment")
    
    # Check if the new treatment has the same color as the first treatment
    same_color = color("second treatment", color("first treatment"))
    
    # Check if the new treatment is between the first treatment and outcome
    placement_check = placement("outcome", "second treatment", direction="right") and placement("second treatment", "first treatment", direction="right")
    
    # Check if the nodes B_t and B_t+1 are linked the same way as A_t and A_t+1
    linkage_check = visual_property("B_t and B_t+1 are linked in the same way as A_t and A_t+1")
    
    return new_treatment_added and treatment_renamed and same_color and placement_check and linkage_check
