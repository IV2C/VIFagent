from vif.falcon.oracle.guided_oracle.expressions import (added,removed,angle,color,placement,position,size,shape,within)
def test_valid_customization() -> bool:
  toprow = "on the topmost row"
  return (
      removed(f"H box {toprow}") and
      removed(f"RZ(x1) {toprow}") and
      removed(f"RZ(θ10) {toprow}") #and RZ(θ2)?
      and removed(f"link from RZ(θ10) {toprow} to the bottom row")
  )