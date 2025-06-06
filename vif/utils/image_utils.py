import base64
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


import uuid
