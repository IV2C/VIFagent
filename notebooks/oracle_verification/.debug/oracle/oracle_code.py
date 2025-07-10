def test_valid_customization() -> bool:
    """
    Asserts that the top light purple square has been rotated by 45 degrees clockwise.
    """
    # The prompt specifies a clockwise rotation of 45 degrees.
    # The angle function takes a positive degree for clockwise rotation.
    return angle("Top light purple square", degree=45)
