DETECTION_PROMPT: str = """Detect, with no more than 20 items. Output a json list where each entry contains the 2D bounding box in "box_2d" and each of these labels:
{labels} 
in a field "label"."""

PINPOINT_PROMPT = """Among the following features:
{features}

Identify the ones most likely to be affected by the instruction:
"{instruction}"

Begin with reasoning as needed.
Conclude with the keyword ANSWER: on a line by itself, followed by a parsable JSON array (no extra text), listing features sorted from most to least likely to be edited.

Format the last two lines exactly like this:
ANSWER:
["feature1", "feature2", ...]"""
