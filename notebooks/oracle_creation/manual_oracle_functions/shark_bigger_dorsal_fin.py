from vif.falcon.oracle.guided_oracle.expressions import (present,removed,angle,color,placement,position,size,shape,within)
def test_valid_customization() -> bool:
  return size("dorsal fin",(1.5,1.0))