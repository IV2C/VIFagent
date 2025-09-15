from vif.falcon.oracle.guided_oracle.expressions import (added,removed,angle,color,placement,position,size,shape,within)
def test_valid_customization() -> bool:
    return placement(
        "top sphere on the left", "bottom sphere on the left", "under"
    ) and placement("top sphere on the right", "bottom sphere on the right", "under")