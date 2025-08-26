from vif.falcon.oracle.guided_oracle.expressions import (added, angle,color,placement,position,size,shape,within)
def test_valid_customization() -> bool:
    added_condition = added("right eye") and added("left eye") and added("right eye pupils") and added("left eye pupils")
    within_condition = within("right eye","bee's head") and within("left eye","bee's head")
    return added_condition and within_condition