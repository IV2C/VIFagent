#labels must be a list
DETECTION_PROMPT: str = """Detect, with no more than 20 items. Output a json list where each entry contains the 2D bounding box in "box_2d" and each of these labels:
{labels} 
in a field "label"."""
FEATURE_IDENTIFIER_PROMPT:str = """Give me a JSON describing the image
The first field "description" contains a high-level description of the image.  
The second field "features" contains a list of all the specific features  in the image.
Notes: 
- *Very single feature of the image must listed, i.e. every single shape, as a feature, no matter how important the shape is.*
- Each feature MUST have a precise name, preferably with position and color attributes.  
- Features MUST be described in one to five words. 
- Only existing features in the image are to be listed.
- Features MUST describe as many instances of things as possible, with a name allowing to pinpoint which feature is where.
- The JSON MUST be between code blocks.
- You MUST follow the exact pattern below, with each feature in the "feature" array.

Output format:
```json
{
  "description": "high-level description",
  "features": [
    "feature1",
    "feature2",
    "feature3",
    ...
  ]
}
```
"""



SYSTEM_PROMPT_CLARIFY:str = """You are an interpretation agent.
You will receive an image and a customization instruction.
Your task is to rewrite the instruction as a clear, step-by-step guide focused only on the image.
Clarify any vague parts, and describe precisely what changes should be made to the image.
Do not include anything unrelated to the image. Respond only with the rewritten instruction in plain text.
"""


IT_PROMPT: str = """
{instruction}
```
{content}
```
"""