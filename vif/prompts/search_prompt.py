
FEATURE_SEARCH_PROMPT:str = """Give me a JSON describing the image
The first field "description" contains a high-level description of the image.  
The second field "features" contains a list of all the specific features  in the image.
Notes: 
- Every single feature of the image must listed, i.e. every single shape, as a feature, no matter how important the shape is.
- Each feature MUST have a precise name, preferably with position and color attributes.  
- Features MUST be described in one to five words. 
- Only existing features in the image are to be listed.
- Features MUST describe as many specific feature as possible, with a name allowing to pinpoint which thing is where.
- Each feature must be unique in the image, i.e. if there are two instances of the same feature, you must add two features and allow to distinct between both using position, color, or other attributes.
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