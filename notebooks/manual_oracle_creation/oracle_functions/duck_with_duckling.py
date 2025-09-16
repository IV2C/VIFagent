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
    return (
        present("small duck")
        and placement("small duck", "big duck", "under")
        and placement("small duck", "big duck", "left")
    )
