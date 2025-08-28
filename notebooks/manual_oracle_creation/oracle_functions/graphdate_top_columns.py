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
    months = ["Jul", "Oct", "Jan 2021", "Apr annotation with arrow"]
    return (
        placement("Jan 2020", "bottom F annotation", "under")
        and placement("Apr", "bottom F annotation", "under")
        and placement("Jul", "bottom F annotation", "under")
        and placement("oct", "bottom F annotation", "under")
        and placement("Jan 2021", "bottom F annotation", "under")
    )
