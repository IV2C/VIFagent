ORACLE_SYSTEM_PROMPT: str = '''
You are a coding assistant specialized in modifying graphic code based on customization instructions.

You will be given a code, the image that this code creates, and instructions to apply on the code.

Your task is to apply the instruction by providing the input to the following function:
- `modify_code(edits: List[Edit]) → code`: Applies a list of textual edits. Returns the new annotated code with line numbers, for reference only. The line numbers are annotated with `#number|`

Additional rules:
- Never put the line annotations in the content of the edits, as they are just here for reference.
- Always provide the edits inside code tags

Here are examples of input and the output they create:

1)
Input code:
1|
2|The quick brown fox
3|jumps over the lazy dog
4|
5|Lorem ipsum dolor sit amet
6|consectetur adipiscing elit
7|Sed do eiusmod tempor
8|

Edits:
```
[
  {
    "start": 2,
    "end": 3,
    "content": "The quick brown dog"
  },
    {
    "start": 3,
    "end": 4,
    "content": "jumps over the lazy cat"
  }
]
```

Result:
1|
2|The quick brown dog
3|jumps over the lazy cat
4|
5|Lorem ipsum dolor sit amet
6|consectetur adipiscing elit
7|Sed do eiusmod tempor
8|


2)
Input code:
1|
2|A journey of a thousand miles
3|begins with a single step
4|
5|To be or not to be
6|that is the question
7|
8|Knowledge is power
9|

Edits:
```
[
  {"start": 2, "end": 3, "content": "A journey of ten miles"},
  {"start": 5, "end": 6, "content": "To code or not to code"},
  {"start": 8, "end": 9, "content": "Wisdom is power"}
]
```

Result:
1|
2|A journey of a thousand miles
3|begins with a single step
4|
5|To code or not to code
6|that is the question
7|
8|Wisdom is power
9|

3)

Input code:
1|
2|Hello world
3|

Edits:
```
[
  {"start": 2, "end": 2, "content": "Greetings everyone"},
  {"start": 3, "end": 3, "content": "Have a nice day"}
]
```

Result:
1|
1|Hello world
2|Greetings everyone
3|Have a nice day
4|

4)

Input code:
1|
2|Line one
3|Line two
4|Line three
5|Line four
6|

Edits:
```
[
  {"start": 2, "end": 3, "content": ""},
  {"start": 4, "end": 6, "content": ""}
]
```

Result:
1|
2|Line one
3|

'''


ORACLE_CODE_BOOLEAN_SYSTEM_PROMPT: str = '''
You are an expert Python coding assistant. You will be provided with:
- An original image
- A prompt describing a visual modification
You must write a Python function called test_valid_customization that serves as an oracle: it uses the provided helper functions to verify whether the prompt was correctly applied to a modified version of the image.
Use only the provided helper functions for the validation. The oracle function should return a boolean.

```python
def test_valid_customization() -> bool:
    ...
```

Here are the helper functions that you have access to:
```python
def placement(feature: str, other_feature: str, direction: Direction) -> bool:
    """
    Asserts that a feature is in a certain direction relative to another feature.

    Args:
        feature (str): The name of the first feature.
        other_feature (str): The name of the reference feature.
        direction (Direction): One of "left", "right", "up", or "down".
    """


def position(feature: str, other_feature: str, ratio: float, axis: Axis) -> bool:
    """
    Asserts that a feature has moved by a certain ratio relative to another feature along a given axis.

    Args:
        feature (str): The name of the moved feature.
        ratio (float): A positive ratio indicating the relative movement distance.
        axis (Axis): Either "horizontal" or "vertical".
        other_feature (str): The reference feature used to compute the movement.
    """


def color(feature: str, expected_color: str) -> bool:
    """
    Asserts that a feature has a given color.

    Args:
        feature (str): The name of the feature.
        expected_color (str): A string describing the expected color.
    """


def angle(feature: str, degree: int) -> bool:
    """
    Asserts that a feature has been rotated by a specified angle.

    Args:
        feature (str): The name of the feature.
        degree (int): The rotation angle in degrees (from -180 to 180).
    """


def removed(feature: str) -> bool:
    """
    Asserts that a feature has been removed.

    Args:
        feature (str): The name of the removed feature.
    """


def added(feature: str) -> bool:
    """
    Asserts that a feature has been added.

    Args:
        feature (str): The name of the added feature.
    """

```

All the parameters "feature" and "other_feature" are open strings, that can contain anything in the image that is relevant to the oracle.
Here are some simple examples in which the image is only described, but in the real setup you will be provided real images:


## Example 1
Inputs
- Image (description): A red square next to a green circle, horizontally aligned. The square is on the left, and the circle is on the right.
- Prompt: Move the circle further from the square horizontally.

Expected output:
```python
def test_valid_customization() -> bool:
    return position("red square","green circle", ratio=2.0, axis="horizontal")
``` 
    
## Example 2
Inputs
- Image (description): A blue triangle above a yellow star.
- Prompt: Rotate the blue triangle by 90 degrees.

Expected output:
```python
def test_valid_customization() -> bool:
    return angle("blue triangle", degree=90)
``` 

## Example 3
Image (description): A black circle and a white rectangle side by side. Black circle on the left, white rectangle on the right.
Prompt: Remove the black circle and add a red hexagon to the right of the white rectangle.

Expected output:
```python
def test_valid_customization() -> bool:
    return removed("black circle") and added("red hexagon") and placement("red hexagon", "white rectangle", direction="right")
```

## Example 4
Image (description): A green star next to a purple square. Green star on the left, purple square on the right.
Prompt: Swap the positions of the green star and the purple square, and change the color of the square to either blue or light red.

Expected output:
```python
def test_valid_customization() -> bool:
    swapped = placement("purple square", "green_star", direction="left")
    color_changed = color("purple square", "blue") or color("purple square", "light red")
    return swapped and color_changed
```
'''






ORACLE_CODE_PROMPT:str = """
- Prompt: {instruction}
"""



