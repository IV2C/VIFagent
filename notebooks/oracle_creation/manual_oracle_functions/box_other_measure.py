from vif.falcon.oracle.guided_oracle.expressions import (present,removed,angle,color,placement,position,size,shape,within)
def test_valid_customization() -> bool:
  return present("L2 interval") and placement("L2 interval","b1 square","right") and placement("L2 interval","b3 square","over")