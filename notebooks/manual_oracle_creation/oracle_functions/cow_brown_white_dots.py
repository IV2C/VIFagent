from vif.falcon.oracle.guided_oracle.expressions import (
    present,
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
        present("bottom White dot under the right eye of the cow")
        and present("top-right White dot under the right eye of the cow")
        and present("top-left White dot under the right eye of the cow")
    )
    return added_dots and color("cow's fur","brown")
