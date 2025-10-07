from vif.falcon.oracle.guided_oracle.expressions import (aligned, present,removed,angle,color,placement,position,size,shape,within)
def test_valid_customization() -> bool:
  return aligned("right side of the copper box on top","right side of the green box","horizontal")