DETECTION_PROMPT: str = """Detect all of the labels, if present, labels can appear multiple times. Output a json list where each entry contains the 2D bounding box in "box_2d" and this/these labels in "label":
{labels}.
"""

SEGMENTATION_PROMPT:str ="""
Give the segmentation masks for the {labels}. Output a JSON list of segmentation masks where each entry contains the 2D bounding box in the key "box_2d", the segmentation mask in key "mask", and the text label in the key "label".
Each label should be present once at most. Always generate the full mask, not just <start_of_mask>.
Important Note: The exact same labels as the ones asked should be used, do not modify them."""
