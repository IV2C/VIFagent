from vif.falcon.oracle.guided_oracle.expressions import (present,removed,angle,color,placement,position,size,shape,within)
def test_valid_customization() -> bool:
  toprow = "on the topmost row"
  return (
      not present(f"H box {toprow}") and
      not present(f"RZ(x1) {toprow}") and
      not present(f"RZ(θ10) {toprow}") #and RZ(θ2)?
      and not present(f"link from RZ(θ10) {toprow} to the bottom row")
  )