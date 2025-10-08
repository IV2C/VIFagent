from vif.falcon.oracle.guided_oracle.expressions import (present,removed,angle,color,placement,position,size,shape,within)
from vif.falcon.oracle.guided_oracle.property_expression import visual_property
def test_valid_customization() -> bool:
    second_treatment_added = present("box annotated with second treatment") and present("circle with B_t") and present("circle with B_t+1")
    treatment_rbox_replace = present("first treatment box") and not present("box annotated with treatment only")
    placement_box = placement("outcome", "second treatment", direction="right") and placement("second treatment", "first treatment", direction="right")
    
    return second_treatment_added and treatment_rbox_replace and placement_box