from vif.falcon.oracle.guided_oracle.expressions import (added,removed,angle,color,placement,position,size,shape,within)
def test_valid_customization() -> bool:
  arrow_size_check = size("first blue arrow",(1,0.8)) and size("second blue arrow",(1,0.8)) and size("third blue arrow",(1,0.8)) and size("fourth blue arrow",(1,0.8))
  block_removal_check = removed("Block annotated with number 6") and removed("Block annotated with number 7") and removed("Block annotated with number 8")
  return block_removal_check and arrow_size_check