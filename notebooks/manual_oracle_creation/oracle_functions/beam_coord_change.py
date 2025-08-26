from vif.falcon.oracle.guided_oracle.expressions import (angle,color,placement,position,size,shape,within)
def test_valid_customization() -> bool:
    return placement("coordinate system","red F","over") and placement("coordinate system","top EA annotation","under")