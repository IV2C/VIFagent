from vif.falcon.oracle.guided_oracle.expressions import (present, angle,color,placement,position,size,shape,within)
def test_valid_customization() -> bool:
    added_condition = present("right eye") and present("left eye") and present("right eye pupils") and present("left eye pupils")
    within_condition = within("right eye","bee's head") and within("left eye","bee's head")
    return added_condition and within_condition