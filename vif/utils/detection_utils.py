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
from vif.utils.image_utils import encode_image, nmse

from google import genai
from google.genai import types as genTypes


full_seg_cache = instantiate_cache(True, ".tmp/cache", "seg_cache")


def key_function(func, *args, **kwargs):
    image = args[0]
    features = args[2]
    model = args[3]
    func_name = func.__name__

    input_hash = hashlib.sha1(
        str(
            (
                hashlib.sha1(image.tobytes()).hexdigest(),
                features,
                model,
                func_name,
            )
        ).encode("utf8")
    ).hexdigest()
    return input_hash


def get_res_detection_logbprob(response):
    """
    get log probabilities for each values in the segm mask response
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


def get_bounding_boxes(
    image: Image.Image,
    client: genai.Client,
    feature,
    model="gemini-2.0-flash",
    enable_logprob: bool = True,
) -> list[BoundingBox]:
    encoded_image = encode_image(image=image)

    logger.info(f"Getting boxe for feature : {feature}")

    contents = [
        genTypes.Content(
            role="user",
            parts=[
                genTypes.Part.from_bytes(
                    mime_type="image/png",
                    data=encoded_image,
                ),
                genTypes.Part.from_text(text=DETECTION_PROMPT.format(label=feature)),
            ],
        ),
    ]
    logger.info(genTypes.Part.from_text(text=DETECTION_PROMPT.format(label=feature)))

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
                raise ParsingError(error_msg, token_data=token_data)
            continue

        id_match = re.search(pattern, response.text)

        ## handling parsing error ##
        if not id_match:
            error_msg = f"Error while parsing :{response.text}"

            log_and_append_token_data(token_data, res_meta, error_msg)

            if attempt_nb == SEGMENTATION_ATTEMPTS - 1:
                raise ParsingError(error_msg, token_data=token_data)
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
                    f"Error while decoding the json {json_res} : {jde}",
                    token_data=token_data,
                )
            continue

        ## handling segmentation mask parsing error ##
        try:
            bounding_boxes = parse_bounding_boxes(detected, image.height, image.width)
        except InvalidMasksError as ime:
            log_and_append_token_data(token_data, res_meta, str(ime))
            if attempt_nb == SEGMENTATION_ATTEMPTS - 1:
                ime.token_data = token_data
                raise ime
            attempt_nb += 1
            continue
        log_and_append_token_data(token_data, res_meta, "box detection worked.")
        break

    if enable_logprob:

        log_probs = get_res_detection_logbprob(response)

        for bound_box in bounding_boxes:
            bound_box.box_prob = next(
                log_prob["box_prob"]
                for log_prob in log_probs
                if log_prob["label"] == bound_box.label
            )

    return (bounding_boxes, token_data)


# @CachedRequest(full_seg_cache, key_function, True)#TODO fix
def get_segmentation_masks(
    image: Image.Image,
    client: genai.Client,
    feature: str,
    model="gemini-2.5-flash",
    enable_logprob: bool = False,
) -> tuple[list[SegmentationMask], list]:
    encoded_image = encode_image(image=image)

    logger.info(f"Getting masks for features : {feature}")

    contents = [
        genTypes.Content(
            role="user",
            parts=[
                genTypes.Part.from_bytes(
                    mime_type="image/png",
                    data=encoded_image,
                ),
                genTypes.Part.from_text(text=SEGMENTATION_PROMPT.format(label=feature)),
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
                **(
                    {
                        "response_logprobs": True,
                        "logprobs": 1,
                    }
                    if enable_logprob
                    else {}
                ),
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
                raise ParsingError(error_msg, token_data=token_data)
            continue

        id_match = re.search(pattern, response.text)

        ## handling parsing error ##
        if not id_match:
            error_msg = f"Error while parsing :{response.text}"

            log_and_append_token_data(token_data, res_meta, error_msg)
            if attempt_nb == SEGMENTATION_ATTEMPTS - 1:
                raise ParsingError(error_msg, token_data=token_data)
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
                    f"Error while decoding the json {json_res} : {jde}", token_data=token_data
                )
            continue

        ## handling segmentation mask parsing error ##
        try:
            seg_masks = parse_segmentation_masks(detected, image.height, image.width)
        except InvalidMasksError as ime:
            log_and_append_token_data(token_data, res_meta, str(ime))
            if attempt_nb == SEGMENTATION_ATTEMPTS - 1:
                ime.token_data=token_data
                raise ime
            attempt_nb += 1
            continue
        log_and_append_token_data(token_data, res_meta, "Segmentation worked.")
        break

    if enable_logprob:

        log_probs = get_res_detection_logbprob(response)

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
