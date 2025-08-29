from vif.falcon.oracle.guided_oracle.expressions import (
    added,
    removed,
    angle,
    color,
    placement,
    position,
    size,
    shape,
    within,
)


def test_valid_customization() -> bool:
    st_box = 'Box labelled with "second treatment"'
    bt = "circular node labelled Bt"
    bt_1 = "circular node labelled Bt+1"
    added_condition = added(st_box) and added(bt) and added(bt_1)
    return added_condition and within(bt, st_box) and within(bt_1, st_box)
