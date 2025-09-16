from vif.falcon.oracle.guided_oracle.expressions import (present,removed,angle,color,placement,position,size,shape,within)
def test_valid_customization() -> bool:
  return placement("AC/DC converter","DC/AC converter","left") and not placement("AC/DC converter","DC/AC converter","over")