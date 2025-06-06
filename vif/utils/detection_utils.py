import json
import math
import os
import re
import shutil

from loguru import logger
from build.lib.vif_agent.prompt import DETECTION_PROMPT
from vif.utils.debug_utils import adjust_bbox, encode_image, mse
from PIL import Image
from openai import OpenAI


def get_boxes(image: Image.Image, client: OpenAI, features, model, temperature):
    encoded_image = encode_image(image=image)

    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": DETECTION_PROMPT.format(
                            labels=", ".join(features)
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"},
                    },
                ],
            }
        ],
    )
    pattern = r"```(?:\w+)?\n([\s\S]+?)```"
    id_match = re.search(pattern, response.choices[0].message.content)

    if not id_match:

        return None

    json_boxes = id_match.group(1)
    detected_boxes = json.loads(json_boxes)
    detected_boxes = [adjust_bbox(box, image) for box in detected_boxes]
    return detected_boxes


def dsim_box(
    detected_boxes: list, base_image: Image.Image, compared_images: list[Image.Image]
):
    """Given a list of detected boxes, an original image and a list of images to compare, computes for each box a list of
    pair (disimilarity,image_index) with descending sort by the mse, and the image index of the corresponding image in the comprared_image list
    the dismilarity is computed via mse(cropped(box,base),cropped(box,other))/prod(box.size)

    Args:
        detected_boxes (list): list of json objects containing boxes, with attribue "box_2d"
        base_image (Image.Image): original image to compare the other images with
        compared_images (list[Image.Image]): list of images to compare the original image with

    Returns:
        list[list[tuple[float, int]]]: a list (in order of detected_boxes) of lists of pair (disimilarity,image_index)
    """

    box_image_map: list[list[tuple[float, int]]] = []

    for box in detected_boxes:
        base_image_mask = base_image.crop(box["box_2d"])

        cur_mse_map: list[tuple[float, int]] = []
        for i, image in enumerate(compared_images):
            mutant_image_mask = image.crop(box["box_2d"])
            cur_mse_map.append(
                (
                    mse(base_image_mask, mutant_image_mask)
                    / math.prod(base_image_mask.size),
                    i,
                )  # normalized MSE divided by the size of the image, to favoritize small specific features
            )

        sorted_mse_map: list[tuple[float, int]] = sorted(
            filter(lambda m: m[0] != 0, cur_mse_map),
            key=lambda m: m[0],
            reverse=True,
        )
        box_image_map.append(sorted_mse_map)

    return box_image_map
