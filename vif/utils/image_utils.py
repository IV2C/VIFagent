import base64
from concurrent.futures import Future, ThreadPoolExecutor
from io import BytesIO
import math
from PIL import Image
import numpy as np


def write_base64_to_image(base64_str: str, output_path: str) -> None:
    if base64_str.startswith("data:image"):
        base64_str = base64_str.split(",")[1]
    with open(output_path, "wb") as image_file:
        image_file.write(base64.b64decode(base64_str))


def get_used_pixels(image: Image.Image):
    used_pixels = 0
    for wp in range(image.width):
        for hp in range(image.height):
            used_pixels = used_pixels + (image.getpixel((wp, hp)) != (0, 0, 0))
    return used_pixels


def encode_image(image_path: str):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def encode_image(image: Image.Image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def se(image1: Image.Image, image2: Image.Image):
    """Compute Mean Squared Error between two PIL images."""
    arr1 = np.array(image1, dtype=np.float32)
    arr2 = np.array(image2, dtype=np.float32)

    if arr1.shape != arr2.shape:
        raise ValueError("Images must have the same dimensions")

    max_val = max(arr1.max(), arr2.max())
    if max_val == 0:
        return 0.0  # Avoid divide-by-zero

    # Normalize to [0, 1]
    arr1 /= max_val
    arr2 /= max_val

    return np.sum((arr1 - arr2) ** 2)


def mse(image1: Image.Image, image2: Image.Image):
    return se(image1, image2) / (math.prod(image1.size))


def adjust_bbox(box, image: Image.Image):
    adjust = lambda box_k, cursize: (box_k / 1000) * cursize
    box_2d = box["box_2d"]
    new_box = (
        int(adjust(box_2d[1], image.width)),
        int(adjust(box_2d[0], image.height)),
        int(adjust(box_2d[3], image.width)),
        int(adjust(box_2d[2], image.height)),
    )
    box["box_2d"] = new_box
    return box


import cv2 as cv
import numpy.typing as npt


def get_template_matches(
    template: Image.Image, image: npt.NDArray
) -> list[tuple[int, int, int, int]]:
    """Identifies templates in an image

    Args:
        template (Image.Image): the template to identify in the image.
        image (Image.Image): the image to identify the template in.
    """
    w = template.width
    h = template.height
    method = cv.TM_SQDIFF_NORMED  # best for exact pattern match
    template = cv.cvtColor(np.array(template), cv.COLOR_RGB2BGR)
    # Apply template Matching
    res = cv.matchTemplate(image, template, method)
    threshold = 0.1  # lower is better
    loc = np.where(res <= threshold)

    detected_boxes: list[tuple[int, int, int, int]] = []
    for pt in zip(*loc[::-1]):
        detected_boxes.append((pt[0], pt[1], pt[0] + w, pt[1] + h))

    return box_similarity_removal(detected_boxes)


def parallel_template_match(
    templates: dict[str, Image.Image], image: Image.Image
) -> dict[str, tuple[int, int, int, int]]:
    """Does parallel computation of template match on an image, given a dictionnary of "feature" to image

    Returns:
        dict[str, tuple[int, int, int, int]]: Mapping from the feature to a list of template matches
    """
    image = cv.cvtColor(np.array(image), cv.COLOR_RGB2BGR)
    with ThreadPoolExecutor() as executor:
        tasks: list[Future] = []
        for template in templates.values():
            tasks.append(executor.submit(get_template_matches, template, image))
        results = []
        for task in tasks:
            results.append(task.result())
    return {feat_name: res for feat_name, res in zip(templates, results)}


def box_similarity_removal(boxes: list[tuple[int, int, int, int]], iou_threshold=0.5):
    """Removes boxes that have too much overlap using IoU

    Args:
        boxes (list[tuple[int, int, int, int]]): The list of boxes to filter

    """
    i = len(boxes) - 1
    while i > 0:
        boxA = boxes[i]
        for j in range(i - 1, -1, -1):
            boxB = boxes[j]
            if IoU(boxA, boxB) > iou_threshold:
                boxes.pop(i)
                i -= 1
                break
        else:
            i -= 1
    return boxes


def IoU(boxA: tuple[int, int, int, int], boxB: tuple[int, int, int, int]):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


# TODO
