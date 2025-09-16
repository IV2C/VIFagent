from vif.falcon.oracle.guided_oracle.expressions import (present,removed,angle,color,placement,position,size,shape,within)
def test_valid_customization() -> bool:
    bottom_left_node = "black node at the bottom left outside and on the left side of the red zone"
    bottom_node = "black node at the bottom on the left of the blue zone"
    link_one = f"link between the {bottom_left_node} and the node at the bottom in the left zone"
    link_two = f"link between the {bottom_node} and the node at the bottom in the blue zone"
    return removed(bottom_left_node) and removed(bottom_node) and removed(link_one) and removed(link_two)