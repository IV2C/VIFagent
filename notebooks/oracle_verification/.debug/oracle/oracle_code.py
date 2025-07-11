def test_valid_customization() -> bool:
    """
    Asserts that the top light purple square has been rotated by 45 degrees clockwise.
    """
    # A clockwise rotation is represented by a negative angle.
    return angle("Top light purple square", degree=-45)
