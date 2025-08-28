from vif.falcon.oracle.guided_oracle.expressions import (added,removed,angle,color,placement,position,size,shape,within)
def test_valid_customization() -> bool:
  return placement("4g box","Camera box","left") and placement("NFC box","Camera box","right")