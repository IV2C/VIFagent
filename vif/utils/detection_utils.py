import base64
import dataclasses
import io
import json
import math
import re
from typing import Any

from PIL import Image
import numpy as np
from openai import OpenAI

from vif.prompts.identification_prompts import DETECTION_PROMPT, SEGMENTATION_PROMPT
from vif.utils.image_utils import adjust_bbox, encode_image, mse


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
                            labels=", ".join(
                                ['"' + feature + '"' for feature in features]
                            )
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


def get_segmentation_masks(
    image: Image.Image, client: OpenAI, features, model, temperature
):
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
                        "text": SEGMENTATION_PROMPT.format(
                            labels=", ".join(
                                ['"' + feature + '"' for feature in features]
                            )
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

    json_res = id_match.group(1)
    detected = json.loads(json_res)
    seg_masks = parse_segmentation_masks(detected, image.height, image.width)
    return seg_masks


@dataclasses.dataclass(frozen=True)
class SegmentationMask:
    # bounding box pixel coordinates (not normalized)
    y0: int  # in [0..height - 1]
    x0: int  # in [0..width - 1]
    y1: int  # in [0..height - 1]
    x1: int  # in [0..width - 1]
    mask: np.array  # [img_height, img_width] with values 0..255
    label: str


def parse_segmentation_masks(
    items: Any, img_height, img_width
) -> list[SegmentationMask]:

    masks = []
    for item in items:
        raw_box = item["box_2d"]
        abs_y0 = int(item["box_2d"][0] / 1000 * img_height)
        abs_x0 = int(item["box_2d"][1] / 1000 * img_width)
        abs_y1 = int(item["box_2d"][2] / 1000 * img_height)
        abs_x1 = int(item["box_2d"][3] / 1000 * img_width)
        if abs_y0 >= abs_y1 or abs_x0 >= abs_x1:
            print("Invalid bounding box", item["box_2d"])
            continue
        label = item["label"]
        png_str = item["mask"]
        if not png_str.startswith("data:image/png;base64,"):
            print("Invalid mask")
            continue
        png_str = png_str.removeprefix("data:image/png;base64,")
        png_str = base64.b64decode(png_str)
        mask = Image.open(io.BytesIO(png_str))
        bbox_height = abs_y1 - abs_y0
        bbox_width = abs_x1 - abs_x0
        if bbox_height < 1 or bbox_width < 1:
            print("Invalid bounding box")
            continue
        mask = mask.resize(
            (bbox_width, bbox_height), resample=Image.Resampling.BILINEAR
        )
        np_mask = np.zeros((img_height, img_width), dtype=np.uint8)
        np_mask[abs_y0:abs_y1, abs_x0:abs_x1] = mask
        masks.append(SegmentationMask(abs_y0, abs_x0, abs_y1, abs_x1, np_mask, label))
    return masks


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
