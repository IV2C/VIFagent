from vif.falcon.oracle.guided_oracle.expressions import (Axis, present,removed,angle,color,placement,position,size,shape,within)
def test_valid_customization() -> bool:
  return position("left eye","right eye",0.5,"horizontal")