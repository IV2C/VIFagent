import base64
import dataclasses
import hashlib
import io
import json
import math
import re
from typing import Any

from PIL import Image, ImageFont, ImageDraw, ImageColor
from loguru import logger
import numpy as np
from openai import OpenAI

from vif.env import SEGMENTATION_ATTEMPTS
from vif.models.detection import BoundingBox, SegmentationMask
from vif.models.exceptions import InvalidMasksError, JsonFormatError, ParsingError
from vif.prompts.identification_prompts import DETECTION_PROMPT, SEGMENTATION_PROMPT
from vif.utils.caching import CachedRequest, instantiate_cache
from vif.utils.image_utils import  encode_image, nmse

from google import genai
from google.genai import types as genTypes


def get_bounding_boxes(
    image: Image.Image,
    client: genai.Client,
    features,
    model,
) -> list[BoundingBox]:
    encoded_image = encode_image(image=image)

    logger.info(f"Getting boxes for features : {','.join(features)}")

    contents = [
        genTypes.Content(
            role="user",
            parts=[
                genTypes.Part.from_bytes(
                    mime_type="image/png",
                    data=encoded_image,
                ),
                genTypes.Part.from_text(
                    text=DETECTION_PROMPT.format(
                        labels=", ".join(['"' + feature + '"' for feature in features])
                    )
                ),
            ],
        ),
    ]

    token_data = []
    for attempt_nb in range(SEGMENTATION_ATTEMPTS):

        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=genTypes.GenerateContentConfig(
                temperature=0.5,
                thinking_config=genTypes.ThinkingConfig(thinking_budget=0),
                response_logprobs=True,
                logprobs=1,
            ),
        )

        pattern = r"```(?:\w+)?\n([\s\S]+?)```"
        logger.info("LLM segmentation response: " + str(response.text))
        res_meta = response.usage_metadata

        ## handling none text error ##
        if response.text is None:
            error_msg = (
                f"Error while parsing response, response is None: {str(response)}"
            )
            log_and_append_token_data(token_data, res_meta, error_msg)
            if attempt_nb == SEGMENTATION_ATTEMPTS - 1:
                raise ParsingError(error_msg)
            continue

        id_match = re.search(pattern, response.text)

        ## handling parsing error ##
        if not id_match:
            error_msg = f"Error while parsing :{response.text}"

            log_and_append_token_data(token_data, res_meta, error_msg)
            if attempt_nb == SEGMENTATION_ATTEMPTS - 1:
                raise ParsingError(error_msg)
            continue

        ## handling json decoding error ##
        json_res = id_match.group(1)
        try:
            detected = json.loads(json_res)
        except json.JSONDecodeError as jde:
            error_msg = f"Error while decoding the json {json_res} : {jde}"
            log_and_append_token_data(token_data, res_meta, error_msg)
            if attempt_nb == SEGMENTATION_ATTEMPTS - 1:
                raise JsonFormatError(
                    f"Error while decoding the json {json_res} : {jde}"
                )
            continue

        ## handling segmentation mask parsing error ##
        try:
            bounding_boxes = parse_bounding_boxes(detected, image.height, image.width)
        except InvalidMasksError as ime:
            log_and_append_token_data(token_data, res_meta, str(ime))
            if attempt_nb == SEGMENTATION_ATTEMPTS - 1:
                raise ime
            attempt_nb += 1
            continue
        log_and_append_token_data(token_data, res_meta, "Segmentation worked.")
        break

    return (bounding_boxes,token_data)


full_seg_cache = instantiate_cache(True, ".tmp/cache", "seg_cache")


def key_function(func, *args, **kwargs):
    image = args[0]
    features = args[2]
    model = args[3]
    func_name = func.__name__

    input_hash = hashlib.sha1(
        str(
            (hashlib.sha1(image.tobytes()).hexdigest(), features, model, func_name)
        ).encode("utf8")
    ).hexdigest()
    return input_hash


def get_mask_seg_logbprob(response):
    """
    Print log probabilities for each token in the response
    """

    probs = []
    if response.candidates and response.candidates[0].logprobs_result:
        logprobs_result = response.candidates[0].logprobs_result
        i = 0
        while i < len(logprobs_result.chosen_candidates):
            chosen_candidate = logprobs_result.chosen_candidates[i]
            if "{" in chosen_candidate.token:
                current_detection = {}
                i += 1
                current_string = ""
                while "}" not in chosen_candidate.token:
                    chosen_candidate = logprobs_result.chosen_candidates[i]
                    current_string += chosen_candidate.token
                    if 'box_2d": [' in current_string:
                        i += 1
                        chosen_candidate = logprobs_result.chosen_candidates[i]
                        box_prob_l = []
                        while "]" not in chosen_candidate.token:
                            if chosen_candidate.token.isdigit():
                                box_prob_l.append(chosen_candidate.log_probability)
                            i += 1
                            chosen_candidate = logprobs_result.chosen_candidates[i]
                        current_detection["box_prob"] = math.exp(
                            sum(box_prob_l) / len(box_prob_l)
                        )
                        current_string = ""
                    elif 'label": "' in current_string:
                        i += 1
                        chosen_candidate = logprobs_result.chosen_candidates[i]
                        cur_label = ""
                        while '"' not in chosen_candidate.token:
                            cur_label += chosen_candidate.token
                            i += 1
                            chosen_candidate = logprobs_result.chosen_candidates[i]
                        current_detection["label"] = cur_label
                        current_string = ""
                    elif 'mask": "' in current_string:
                        i += 1
                        chosen_candidate = logprobs_result.chosen_candidates[i]
                        seg_prob_l = []
                        while '"' not in chosen_candidate.token:
                            if (
                                "<" in chosen_candidate.token
                                and not "<start_of_mask>" in chosen_candidate.token
                            ):
                                seg_prob_l.append(chosen_candidate.log_probability)
                            i += 1
                            chosen_candidate = logprobs_result.chosen_candidates[i]
                        current_detection["seg_prob"] = math.exp(
                            sum(seg_prob_l) / len(seg_prob_l)
                        )
                        current_string = ""
                    else:
                        i += 1
                probs.append(current_detection)
            i += 1
    return probs


@CachedRequest(full_seg_cache, key_function, True)
def get_segmentation_masks(
    image: Image.Image,
    client: genai.Client,
    features,
    model,
) -> tuple[list[SegmentationMask], list]:
    encoded_image = encode_image(image=image)

    logger.info(f"Getting masks for features : {','.join(features)}")

    contents = [
        genTypes.Content(
            role="user",
            parts=[
                genTypes.Part.from_bytes(
                    mime_type="image/png",
                    data=encoded_image,
                ),
                genTypes.Part.from_text(
                    text=SEGMENTATION_PROMPT.format(
                        labels=", ".join(['"' + feature + '"' for feature in features])
                    )
                ),
            ],
        ),
    ]

    token_data = []
    for attempt_nb in range(SEGMENTATION_ATTEMPTS):

        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=genTypes.GenerateContentConfig(
                temperature=0.5,
                thinking_config=genTypes.ThinkingConfig(thinking_budget=0),
                response_logprobs=True,
                logprobs=1,
            ),
        )

        pattern = r"```(?:\w+)?\n([\s\S]+?)```"
        logger.info("LLM segmentation response: " + str(response.text))
        res_meta = response.usage_metadata

        ## handling none text error ##
        if response.text is None:
            error_msg = (
                f"Error while parsing response, response is None: {str(response)}"
            )
            log_and_append_token_data(token_data, res_meta, error_msg)
            if attempt_nb == SEGMENTATION_ATTEMPTS - 1:
                raise ParsingError(error_msg)
            continue

        id_match = re.search(pattern, response.text)

        ## handling parsing error ##
        if not id_match:
            error_msg = f"Error while parsing :{response.text}"

            log_and_append_token_data(token_data, res_meta, error_msg)
            if attempt_nb == SEGMENTATION_ATTEMPTS - 1:
                raise ParsingError(error_msg)
            continue

        ## handling json decoding error ##
        json_res = id_match.group(1)
        try:
            detected = json.loads(json_res)
        except json.JSONDecodeError as jde:
            error_msg = f"Error while decoding the json {json_res} : {jde}"
            log_and_append_token_data(token_data, res_meta, error_msg)
            if attempt_nb == SEGMENTATION_ATTEMPTS - 1:
                raise JsonFormatError(
                    f"Error while decoding the json {json_res} : {jde}"
                )
            continue

        ## handling segmentation mask parsing error ##
        try:
            seg_masks = parse_segmentation_masks(detected, image.height, image.width)
        except InvalidMasksError as ime:
            log_and_append_token_data(token_data, res_meta, str(ime))
            if attempt_nb == SEGMENTATION_ATTEMPTS - 1:
                raise ime
            attempt_nb += 1
            continue
        ## handling feature number detection error ##
        if len(seg_masks) < len(features):
            error_msg = f"The features {','.join(features)} were not detected."
            log_and_append_token_data(token_data, res_meta, error_msg)

            if attempt_nb == SEGMENTATION_ATTEMPTS - 1:
                raise InvalidMasksError(error_msg)
            continue
        log_and_append_token_data(token_data, res_meta, "Segmentation worked.")
        break

    log_probs = get_mask_seg_logbprob(response)

    for seg_mask in seg_masks:
        seg_mask.box_prob = next(
            log_prob["box_prob"]
            for log_prob in log_probs
            if log_prob["label"] == seg_mask.label
        )
        seg_mask.seg_prob = next(
            log_prob["seg_prob"]
            for log_prob in log_probs
            if log_prob["label"] == seg_mask.label
        )

    return (seg_masks, token_data)


def log_and_append_token_data(token_data: list, res_meta, error_info):
    logger.warning(error_info)
    token_data.append(
        {
            "completion_token": res_meta.candidates_token_count,
            "prompt_token": res_meta.prompt_token_count,
            "total_tokens": res_meta.total_token_count,
            "error_info": error_info,
        }
    )


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
            raise InvalidMasksError(items)
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
        np_mask = np_mask.astype(bool)
        masks.append(SegmentationMask(abs_y0, abs_x0, abs_y1, abs_x1, np_mask, label))
    return masks


def parse_bounding_boxes(items: Any, img_height, img_width) -> list[BoundingBox]:

    boxes = []
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
        bbox_height = abs_y1 - abs_y0
        bbox_width = abs_x1 - abs_x0
        if bbox_height < 1 or bbox_width < 1:
            print("Invalid bounding box")
            continue

        boxes.append(BoundingBox(abs_y0, abs_x0, abs_y1, abs_x1, label))
    return boxes


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
                    nmse(base_image_mask, mutant_image_mask)
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


def plot_segmentation_masks(
    img: Image.Image, segmentation_masks: list[SegmentationMask]
) -> Image.Image:
    """
    Plots bounding boxes on an image with markers for each a name, using PIL, normalized coordinates, and different colors.

    Args:
        img: The PIL.Image.
        segmentation_masks: A string encoding as JSON a list of segmentation masks containing the name of the object,
         their positions in normalized [y1 x1 y2 x2] format, and the png encoded segmentation mask.
    """
    # Define a list of colors
    colors = [
        "red",
        "green",
        "blue",
        "yellow",
        "orange",
        "pink",
        "purple",
        "brown",
        "gray",
        "beige",
        "turquoise",
        "cyan",
        "magenta",
        "lime",
        "navy",
        "maroon",
        "teal",
        "olive",
        "coral",
        "lavender",
        "violet",
        "gold",
        "silver",
    ]
    font = ImageFont.truetype("NotoSansCJK-Regular.ttc", size=14)

    # Do this in 3 passes to make sure the boxes and text are always visible.

    # Overlay the mask
    for i, mask in enumerate(segmentation_masks):
        color = colors[i % len(colors)]
        img = overlay_mask_on_img(img, mask.mask, color)

    # Create a drawing object
    draw = ImageDraw.Draw(img)

    # Draw the bounding boxes
    for i, mask in enumerate(segmentation_masks):
        color = colors[i % len(colors)]
        draw.rectangle(((mask.x0, mask.y0), (mask.x1, mask.y1)), outline=color, width=4)

    # Draw the text labels
    for i, mask in enumerate(segmentation_masks):
        color = colors[i % len(colors)]
        if mask.label != "":
            draw.text((mask.x0 + 8, mask.y0 - 20), mask.label, fill=color, font=font)
    return img


def overlay_mask_on_img(
    img: Image, mask: np.ndarray, color: str, alpha: float = 0.7
) -> Image.Image:
    """
    Overlays a single mask onto a PIL Image using a named color.

    The mask image defines the area to be colored. Non-zero pixels in the
    mask image are considered part of the area to overlay.

    Args:
        img: The base PIL Image object.
        mask: A PIL Image object representing the mask.
              Should have the same height and width as the img.
              Modes '1' (binary) or 'L' (grayscale) are typical, where
              non-zero pixels indicate the masked area.
        color: A standard color name string (e.g., 'red', 'blue', 'yellow').
        alpha: The alpha transparency level for the overlay (0.0 fully
               transparent, 1.0 fully opaque). Default is 0.7 (70%).

    Returns:
        A new PIL Image object (in RGBA mode) with the mask overlaid.

    Raises:
        ValueError: If color name is invalid, mask dimensions mismatch img
                    dimensions, or alpha is outside the 0.0-1.0 range.
    """
    if not (0.0 <= alpha <= 1.0):
        raise ValueError("Alpha must be between 0.0 and 1.0")

    # Convert the color name string to an RGB tuple
    try:
        color_rgb: tuple[int, int, int] = ImageColor.getrgb(color)
    except ValueError as e:
        # Re-raise with a more informative message if color name is invalid
        raise ValueError(
            f"Invalid color name '{color}'. Supported names are typically HTML/CSS color names. Error: {e}"
        )

    # Prepare the base image for alpha compositing
    img_rgba = img.convert("RGBA")
    width, height = img_rgba.size

    # Create the colored overlay layer
    # Calculate the RGBA tuple for the overlay color
    alpha_int = int(alpha * 255)
    overlay_color_rgba = color_rgb + (alpha_int,)

    # Create an RGBA layer (all zeros = transparent black)
    colored_mask_layer_np = np.zeros((height, width, 4), dtype=np.uint8)

    # Mask has values between 0 and 255, threshold at 127 to get binary mask.
    mask_np_logical = mask > 127

    # Apply the overlay color RGBA tuple where the mask is True
    colored_mask_layer_np[mask_np_logical] = overlay_color_rgba

    # Convert the NumPy layer back to a PIL Image
    colored_mask_layer_pil = Image.fromarray(colored_mask_layer_np, "RGBA")

    # Composite the colored mask layer onto the base image
    result_img = Image.alpha_composite(img_rgba, colored_mask_layer_pil)

    return result_img
