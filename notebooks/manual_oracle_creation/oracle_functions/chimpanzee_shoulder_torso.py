from vif.falcon.oracle.guided_oracle.expressions import (present,removed,angle,color,placement,position,size,shape,within)
def test_valid_customization() -> bool:
  return present("chimpanzee's torso") and placement("chimpanzee's torso","chimpanzee's face","under")