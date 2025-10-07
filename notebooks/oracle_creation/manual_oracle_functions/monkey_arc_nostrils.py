from vif.falcon.oracle.guided_oracle.expressions import (present,removed,angle,color,placement,position,size,shape,within)
def test_valid_customization() -> bool:
    return shape("left nostril","arc") and shape("right nostril","arc")