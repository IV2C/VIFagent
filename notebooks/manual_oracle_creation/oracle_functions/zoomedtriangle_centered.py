from vif.falcon.oracle.guided_oracle.expressions import (added, aligned,removed,angle,color,placement,position,size,shape,within)
def test_valid_customization() -> bool:
  return aligned("green dot in the red circle","red circle","vertical") and aligned("green dot in the red circle","red circle","horizontal")