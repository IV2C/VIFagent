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
    left_eye_color_changed = color("left eye of the chimpanzee", "white")
    right_eye_color_changed = color("right eye of the chimpanzee", "white")
    left_pupil_added = present("black pupil in the left eye of the chimpanzee")
    right_pupil_added = present("black pupil in the right eye of the chimpanzee")
    left_pupil_position = placement("black pupil in the left eye of the chimpanzee", "left eye of the chimpanzee", direction="right")
    right_pupil_position = placement("black pupil in the right eye of the chimpanzee", "right eye of the chimpanzee", direction="left")
    return left_eye_color_changed and right_eye_color_changed and left_pupil_added and right_pupil_added and left_pupil_position and right_pupil_position
