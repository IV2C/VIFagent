from vif.falcon.oracle.guided_oracle.expressions import (present,removed,angle,color,placement,position,size,shape,within)
def test_valid_customization() -> bool:
  return size("cat's abdomen",(1.3,1)) and size("cat's head",(1.3,1)) and size("cat's feet",(1.3,1))