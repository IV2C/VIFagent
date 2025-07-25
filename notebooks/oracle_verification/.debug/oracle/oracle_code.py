def test_valid_customization() -> bool:
    """
    Asserts that the square has been rotated by 45 degrees clockwise.
    """
    # Clockwise rotation is represented by a negative angle.
    return angle("blue square", degree=-45)
