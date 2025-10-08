ORACLE_SYSTEM_PROMPT: str = """
You are a coding assistant specialized in modifying graphic code based on customization instructions.

You will be given a code, the image that this code creates, and instructions to apply on the code.

You will have to return
Your task is to apply the instruction by providing the input to the following function:
- `modify_code(edits: List[Edit]) → code`: Applies a list of textual edits. Returns the new annotated code with line numbers, for reference only. The line numbers are annotated with `#number|`

Each edit is in the form 
{
  "start": int,
  "end": int,
  "content": string
}

These edits will be applied to the code, each edit will replace the code at the lines [start:end[.


Additional rules:
- Never put the line annotations in the content of the edits, as they are just here for reference.
- Always provide the edits inside code tags

Here are examples of input code with edits and the output code the edits create:

1)
Input:
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
    "end": 4,
    "content": "The quick brown dog\\njumps over the lazy cat"
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
Input:
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

Input:
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

Input:
1|Line one
2|Line two
3|Line three
4|Line four
5|

Edits:
```
[
  {"start": 2, "end": 3, "content": ""},
  {"start": 3, "end": 5, "content": ""}
]
```

Result:
1|Line one
2|

"""


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
        direction (Direction): One of "left", "right", "over", or "under".(e.g. "feature is on the left/right of the other_feature" or "feature is over/under the other_feature")
    """


def position(feature: str, other_feature: str, ratio: float, axis: Axis) -> bool:
    """
    Asserts that a feature has moved by a certain ratio relative to another feature along a given axis.

    Args:
        feature (str): The name of the moved feature.
        other_feature (str): The reference feature used to compute the movement.
        ratio (float): A positive ratio indicating the relative movement distance, when compared to the original distance between the two features.
        axis (Axis): Either "horizontal" or "vertical".
    """


def color(feature: str, expected_color: str) -> bool:
    """
    Asserts that a feature has a given color. A precise arbitrary open string color string must be provided, i.e. the closest one possible to the expected color.

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

def size(feature: str, ratio: Tuple[float, float]) -> bool:
    """
    Asserts that a feature has been resized by given scaling factors along x and y.

    Args:
        feature (str): The name of the resized feature.
        ratio (Tuple[float, float]): Scaling factors applied to the feature’s width (x) and height (y), relative to its own original dimensions. 
    """
    
def within(feature: str, other_feature: str) -> bool:
    """
    Asserts that a feature is contained in another feature.

    Args:
        feature (str): The name of the feature contained in "other_feature".
        other_feature (str): The name of the other feature that contains "feature".
    """
    
def shape(feature: str, shape: str) -> bool:
    """
    Asserts that a feature looks like a certain shape.

    Args:
        feature (str): The name of feature.
        shape (str): An open string describing the shape of the feature (square, triangle, ellipse,etc.).
    """

def present(feature: str) -> bool:
    """
    Asserts that a specific feature is present in the image.

    Args:
        feature (str): The name of the feature present.
    """
    
def mirrored(feature: str, axis:Axis) -> bool:
    """
    Asserts that a feature is mirrored along an axis.

    Args:
        feature (str): The name of the feature.
        axis (Axis): Either "horizontal" (left/right) or "vertical"(up/down).

    """

def aligned(feature: str, other_feature: str, axis:Axis) -> bool:
    """
    Asserts that a feature is aligned with another another feature vertically or horizontally.

    Args:
        feature (str): The name of the first feature.
        other_feature (str): The name of the second feature.
        axis (Axis): Either "horizontal" (left/right) or "vertical"(up/down).
    """
```

All the parameters "feature" and "other_feature" are open strings, that can contain anything in the image that is relevant to the oracle, and must be unambiguous unique features, for example, do not use "circle" if there are two circles in the image.
You can give very detailed description of the feature you are searching for to make them unambiguous. When using these parameters, ensure the features are identifiable both in the initial and modified image.
For example, if the instruction describes a color change, do not use the color as an attribute of the feature because it will not be the same in the modified image.
You can use boolean operators such as "not", "and", and "or" for each type of condition, as well as intermediary variables. However you cannot use keywords like any() or all() 
The examples below show examples with overly simple features, in the real case you will have to provide highly detailed and higher level features. 
Here are some very simple examples in which the image is only described, but in the real setup you will be provided real images:


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
    return not present("black circle") and present("red hexagon") and placement("red hexagon", "white rectangle", direction="right")
```

## Example 4
Image (description): A green star next to a purple square. Green star on the left, purple square on the right.
Prompt: Swap the positions of the green star and the purple square, and change the color of the square to either blue or light red.

Expected output:
```python
def test_valid_customization() -> bool:
    swapped = placement("square", "green star", direction="left")
    color_changed = color("square", "blue") or color("square", "light red")
    return swapped and color_changed
```

## Example 5
Image (description): A blue rectangle of a certain width w and height h.
Prompt: Make the rectangle twice as tall.

Expected output:
```python
def test_valid_customization() -> bool:
   return size("blue rectangle", (1.0,2.0))
```

## Example 6
Image (description): A drawing of a face.
Prompt: Make the ears in the shape of a square.

Expected output:
```python
def test_valid_customization() -> bool:
   return shape("left ear","square") and shape("right ear","square") 
```

## Example 7
Image (description): A drawing of a face.
Prompt: Add a nose in the shape of a square.

Expected output:
```python
def test_valid_customization() -> bool:
   return present("nose") and within("nose","face") and shape("nose","square")
```
'''


ORACLE_PROPERTY_USAGE_PROMPT = '''
You also have access to a visual property checker helper function, to use when the other ones are not sufficient:
```
def visual_property(property:str) -> bool:
    """
    Asserts that a property is applied on the second image.

    Args:
        property:str
    """
```

This function can be used the same way as the other helper functions, for example:
## Example 8
Image (description): A drawing of a person.
Prompt: Make the person look scared.

Expected output:
```python
def test_valid_customization() -> bool:
   return visual_property("The person is now looking scared")
```
'''



ORACLE_CODE_PROMPT: str = """
- Prompt: {instruction}
"""
