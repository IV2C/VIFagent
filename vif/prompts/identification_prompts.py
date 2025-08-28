DETECTION_PROMPT: str = """Detect, if present, with no more than 20 items. Output a json list where each entry contains the 2D bounding box in "box_2d" and these labels in "label":
{labels} .
"""

SEGMENTATION_PROMPT:str ="""
Give the segmentation masks for the {labels}. Output a JSON list of segmentation masks where each entry contains the 2D bounding box in the key "box_2d", the segmentation mask in key "mask", and the text label in the key "label".
Only identify if the exact descriptive label is present, if not, do not output a mask for the label. Each label should be present once at most. Always generate the full mask, not just <start_of_mask>."""

PINPOINT_PROMPT = """Given the following list of features:
`{features}`

And the instruction:
`"{instruction}"`

Your task:

1. Reason step by step to determine the effects of the instruction, explicitely follow each of these steps.
    - Describe the image.
    - Describe the instruction, with regard to the image.
    - Explicitly reason about the descriptions you have made and the image, then, from the given list only, identify which features will be modified or deleted. Finally identify which features will be added (i.e., not in the original list). Give a first answer.
    - Explicitly Double check your answer.
2. After reasoning, conclude with the keyword `ANSWER:` on its own line, followed by exactly three JSON arrays, one per line:
    1. Features from the original list that will be modified.
    2. Features from the original list that will be deleted.
    3. New features that will be added. Each feature must be between 1 and 4 words long.

Final format (strict):

ANSWER:  
["featureA", "featureB", ...]  
["featureC", ...]  
["new_feature", ...]  

"""


PINPOINT_PROMPT = """Given the following list of features:
`{features}`

And the instruction:
`"{instruction}"`

Your task:

1. Reason step by step to determine the effects of the instruction, explicitely follow each of these steps.
    - Describe the image.
    - Describe the instruction, with regard to the image.
    - Explicitly reason about the descriptions you have made and the image, then, from the given list only, identify which features will be modified or deleted. Finally identify which features will be added (i.e., not in the original list). Give a first answer.
    - Explicitly Double check your answer.
2. After reasoning, conclude with the keyword `ANSWER:` on its own line, followed by exactly three JSON arrays, one per line:
    1. Features from the original list that will be modified.
    2. Features from the original list that will be deleted.
    3. New features that will be added. Each feature must be between 1 and 4 words long.

Final format (strict):

ANSWER:  
["featureA", "featureB", ...]  
["featureC", ...]  
["new_feature", ...]  

"""