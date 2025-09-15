from vif.falcon.oracle.guided_oracle.expressions import (added,removed,angle,color,placement,position,size,shape,within)
def test_valid_customization() -> bool:
  return removed("gray box with two blue squares and two yellow triangle")