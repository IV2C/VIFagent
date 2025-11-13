DETECTION_PROMPT: str = """Detect the {label}, if present, labels can appear multiple times. Output a json list where each entry contains the 2D bounding box in "box_2d" and the label in "label"."""

SEGMENTATION_PROMPT:str ="""
Give the segmentation mask for the {label}. Output a JSON list of segmentation masks where each entry contains the 2D bounding box in the key "box_2d", the segmentation mask in key "mask", and the text label in the key "label".
Always generate the full mask, not just <start_of_mask>."""
