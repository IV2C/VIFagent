DETECTION_PROMPT: str = """Detect, with no more than 20 items. Output a json list where each entry contains the 2D bounding box in "box_2d" and each of these labels:
{labels} 
in a field "label"."""

PINPOINT_PROMPT = """Among the following features:
{features}

Give me the ones that are the most probable of being edited by the instruction:
"{instruction}"

Start with your usual reasoning or explanation as needed. Then, at the end of your response, output the keyword ANSWER: followed by your definitive answer and the next line, in the following parsable JSON array format (no extra text):
ANSWER:
["feature1", "feature2", ...]"""
