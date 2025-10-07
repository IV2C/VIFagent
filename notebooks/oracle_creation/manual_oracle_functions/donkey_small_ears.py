from vif.falcon.oracle.guided_oracle.expressions import (present,removed,angle,color,placement,position,size,shape,within)
def test_valid_customization() -> bool:
    return size("donkey's left ear",(1,0.5)) and size("donkey's right ear",(1,0.5))
  