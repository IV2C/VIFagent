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
    n5 = 'node in the Input layer with "input 5" going into'
    hd = "node of the hidden layer"
    t = f"topmost {hd}"
    m = f"middle {hd}"
    b = f"bottommost {hd}"
    return (
        removed(n5)
        and removed('arrow labelled "input 5"')
        and removed(f"arrow going from the fifth node of the input layer to the {t}")
        and removed(f"arrow going from the fifth node of the input layer to the {m}")
        and removed(f"arrow going from the fifth node of the input layer to the {b}")
    )
