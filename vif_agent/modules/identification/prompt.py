DETECTION_PROMPT: str = """Detect, with no more than 20 items. Output a json list where each entry contains the 2D bounding box in "box_2d" and each of these labels:
{labels} 
in a field "label"."""

PINPOINT_PROMPT = """Given the following list of features:
`{features}`

And the instruction:
`"{instruction}"`

Your task:

1. Reason step by step to determine the effects of the instruction.
2. From the given list only, identify which features will be modified or deleted.
3. Identify any new features that will be added (i.e., not in the original list).

After reasoning, conclude with the keyword `ANSWER:` on its own line, followed by exactly three JSON arrays, one per line:

1. Features from the original list that will be modified.
2. Features from the original list that will be deleted.
3. New features that will be added.

Format (strict):

ANSWER:  
["featureA", "featureB", ...]  
["featureC", ...]  
["new_feature", ...]  

"""
