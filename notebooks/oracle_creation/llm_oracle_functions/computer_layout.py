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
    memory_above_cpu = placement("MEMORY", "CPU", direction="over")
    storage_above_cpu = placement("STORAGE", "CPU", direction="over")
    input_left_of_cpu = placement("INPUT", "CPU", direction="left")
    correct_layout = memory_above_cpu and storage_above_cpu and input_left_of_cpu
    arrows_adapted = visual_property("The arrows are adapted so that they do not overlap with the elements")
    return correct_layout and arrows_adapted
