from vif.falcon.oracle.guided_oracle.expressions import (added,removed,angle,color,placement,position,size,shape,within)
def test_valid_customization() -> bool:
  return placement("entire GPU box","OpenCL program box","right") and placement("execute arrow","entire GPU box","right") and not placement("entire GPU box","OpenCL program box","under")