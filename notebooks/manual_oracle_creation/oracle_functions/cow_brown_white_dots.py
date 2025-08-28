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
    added_dots = (
        added("bottom White dot under the right eye of the cow")
        and added("top-right White dot under the right eye of the cow")
        and added("top-left White dot under the right eye of the cow")
    )
    return added_dots and color("cow's fur","brown")
