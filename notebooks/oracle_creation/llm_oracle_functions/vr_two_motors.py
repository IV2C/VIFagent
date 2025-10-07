from vif.falcon.oracle.guided_oracle.property_expression import visual_property
from vif.falcon.oracle.guided_oracle.expressions import (
    OracleExpression,
    aligned,
    present,
    angle,
    placement,
    position,
    color,
    shape,
    size,
    within,
    mirrored,
)
def test_valid_customization() -> bool:
    # Check if "Motors" is replaced by "Motor 1" and "Motor 2"
    motor_split = not present("Motors") and present("Motor 1") and present("Motor 2")
    
    # Check if "Motor 1" and "Motor 2" are next to each other
    motors_placement = placement("Motor 1", "Motor 2", direction="left") or placement("Motor 2", "Motor 1", direction="left")
    
    # Check if both "Motor 1" and "Motor 2" are linked to the "Robot controller"
    motor_links = within("Motor 1", "green container") and within("Motor 2", "green container") and within("green container", "orange container")
    
    # Check if the containers are updated accordingly
    container_update = within("Robot controller", "orange container") and within("Deep Learning module", "orange container") and within("Data transfer node", "orange container") and within("Sensors", "green container")
    
    return motor_split and motors_placement and motor_links and container_update
