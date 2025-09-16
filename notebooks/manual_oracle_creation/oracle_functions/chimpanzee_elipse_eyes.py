from vif.falcon.oracle.guided_oracle.expressions import (present,removed,angle,color,placement,position,size,shape,within)
def test_valid_customization() -> bool:
  return shape("chimpanzee's left eye","ellipse") and shape("chimpanzee's right eye","ellipse")