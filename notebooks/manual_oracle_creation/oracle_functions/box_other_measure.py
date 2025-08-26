from vif.falcon.oracle.guided_oracle.expressions import (added,removed,angle,color,placement,position,size,shape,within)
def test_valid_customization() -> bool:
  return added("L2 interval") and placement("L2 interval","b1 square","right") and placement("L2 interval","b3 square","over")