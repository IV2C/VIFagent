from __future__ import annotations

import numpy as np
import re
import torch
from dateutil import parser as dateparser
from PIL import Image
from torchvision import transforms
from torchvision.ops import box_iou
from typing import Union, List
from word2number import w2n

from vif.baselines.verifiers_baseline.ViperGPT_adapt import (
    ViperGPT_clip_utils,
)
from vif.baselines.verifiers_baseline.ViperGPT_adapt.ViperGPT_config import (
    ViperGPTConfig,
)
from vif.baselines.verifiers_baseline.ViperGPT_adapt.utils import (
    load_json,
    show_single_image,
)
from vif.models.detection import BoundingBox
from vif.utils.detection_utils import get_bounding_boxes
from vif.utils.image_utils import encode_image


class ImagePatch:
    """A Python class containing a crop of an image centered around a particular object, as well as relevant
    information.
    Attributes
    ----------
    cropped_image : array_like
        An array-like of the cropped image taken from the original image.
    left : int
        An int describing the position of the left border of the crop's bounding box in the original image.
    lower : int
        An int describing the position of the bottom border of the crop's bounding box in the original image.
    right : int
        An int describing the position of the right border of the crop's bounding box in the original image.
    upper : int
        An int describing the position of the top border of the crop's bounding box in the original image.

    Methods
    -------
    find(object_name: str)->List[ImagePatch]
        Returns a list of new ImagePatch objects containing crops of the image centered around any objects found in the
        image matching the object_name.
    exists(object_name: str)->bool
        Returns True if the object specified by object_name is found in the image, and False otherwise.
    verify_property(property: str)->bool
        Returns True if the property is met, and False otherwise.
    best_text_match(option_list: List[str], prefix: str)->str
        Returns the string that best matches the image.
    simple_query(question: str=None)->str
        Returns the answer to a basic question asked about the image. If no question is provided, returns the answer
        to "What is this?".
    compute_depth()->float
        Returns the median depth of the image crop.
    crop(left: int, lower: int, right: int, upper: int)->ImagePatch
        Returns a new ImagePatch object containing a crop of the image at the given coordinates.
    """

    def __init__(
        self,
        image: Union[Image.Image, torch.Tensor, np.ndarray],
        left: int = None,
        lower: int = None,
        right: int = None,
        upper: int = None,
        parent_left=0,
        parent_lower=0,
        queues=None,
        parent_img_patch=None,
    ):
        """Initializes an ImagePatch object by cropping the image at the given coordinates and stores the coordinates as
        attributes. If no coordinates are provided, the image is left unmodified, and the coordinates are set to the
        dimensions of the image.

        Parameters
        -------
        image : array_like
            An array-like of the original image.
        left : int
            An int describing the position of the left border of the crop's bounding box in the original image.
        lower : int
            An int describing the position of the bottom border of the crop's bounding box in the original image.
        right : int
            An int describing the position of the right border of the crop's bounding box in the original image.
        upper : int
            An int describing the position of the top border of the crop's bounding box in the original image.

        """

        if isinstance(image, Image.Image):
            image = transforms.ToTensor()(image)
        elif isinstance(image, np.ndarray):
            image = torch.tensor(image).permute(1, 2, 0)
        elif isinstance(image, torch.Tensor) and image.dtype == torch.uint8:
            image = image / 255

        if left is None and right is None and upper is None and lower is None:
            self.cropped_image = image
            self.left = 0
            self.lower = 0
            self.right = image.shape[2]  # width
            self.upper = image.shape[1]  # height
        else:
            self.cropped_image = image[
                :, image.shape[1] - upper : image.shape[1] - lower, left:right
            ]
            self.left = left + parent_left
            self.upper = upper + parent_lower
            self.right = right + parent_left
            self.lower = lower + parent_lower

        self.height = self.cropped_image.shape[1]
        self.width = self.cropped_image.shape[2]

        self.cache = {}
        self.queues = (None, None) if queues is None else queues

        self.parent_img_patch = parent_img_patch

        self.horizontal_center = (self.left + self.right) / 2
        self.vertical_center = (self.lower + self.upper) / 2

        if self.cropped_image.shape[1] == 0 or self.cropped_image.shape[2] == 0:
            raise Exception("ImagePatch has no area")

        self.possible_options = load_json(
            "vif/baselines/verifiers_baseline/ViperGPT_adapt/useful_lists/possible_options.json"
        )

    @property
    def original_image(self):
        if self.parent_img_patch is None:
            return self.cropped_image
        else:
            return self.parent_img_patch.original_image

    def find(self, object_name: str) -> list[ImagePatch]:
        """Returns a list of ImagePatch objects matching object_name contained in the crop if any are found.
        Otherwise, returns an empty list.
        Parameters
        ----------
        object_name : str
            the name of the object to be found

        Returns
        -------
        List[ImagePatch]
            a list of ImagePatch objects matching object_name contained in the crop
        """
        boxes: list[BoundingBox] = get_bounding_boxes(
            self.cropped_image, ViperGPTConfig.visual_client, object_name
        )
        all_coordinates = [(box.x0, box.y1, box.x1, box.y0) for box in boxes]

        if len(all_coordinates) == 0:
            return []

        return [self.crop(*coordinates) for coordinates in all_coordinates]

    def exists(self, object_name) -> bool:
        """Returns True if the object specified by object_name is found in the image, and False otherwise.
        Parameters
        -------
        object_name : str
            A string describing the name of the object to be found in the image.
        """
        if object_name.isdigit() or object_name.lower().startswith("number"):
            object_name = object_name.lower().replace("number", "").strip()

            object_name = w2n.word_to_num(object_name)
            answer = self.simple_query(
                "What number is written in the image (in digits)?"
            )
            return w2n.word_to_num(answer) == object_name

        patches = self.find(object_name)

        filtered_patches = []
        for patch in patches:
            if "yes" in patch.simple_query(f"Is this a {object_name}?"):
                filtered_patches.append(patch)
        return len(filtered_patches) > 0

    def _score(self, category: str, negative_categories=None, model="clip") -> float:
        """
        Returns a binary score for the similarity between the image and the category.
        The negative categories are used to compare to (score is relative to the scores of the negative categories).
        """
        res = ViperGPT_clip_utils.score(
            image=self.cropped_image,
            prompt=category,
            negative_categories=negative_categories,
        )

        return res

    def _detect(
        self, category: str, thresh, negative_categories=None, model="clip"
    ) -> bool:
        return self._score(category, negative_categories, model) > thresh

    def verify_property(self, object_name: str, attribute: str) -> bool:
        """Returns True if the object possesses the property, and False otherwise.
        Differs from 'exists' in that it presupposes the existence of the object specified by object_name, instead
        checking whether the object possesses the property.
        Parameters
        -------
        object_name : str
            A string describing the name of the object to be found in the image.
        attribute : str
            A string describing the property to be checked.
        """
        name = f"{attribute} {object_name}"
        negative_categories = [
            f"{att} {object_name}" for att in self.possible_options["attributes"]
        ]

        return self._detect(
            name,
            negative_categories=negative_categories,
            thresh=ViperGPTConfig.thresh_clip,
        )

    def best_text_match(self, option_list: list[str] = None, prefix: str = None) -> str:
        """Returns the string that best matches the image.
        Parameters
        -------
        option_list : str
            A list with the names of the different options
        prefix : str
            A string with the prefixes to append to the options
        """
        option_list_to_use = option_list
        if prefix is not None:
            option_list_to_use = [prefix + " " + option for option in option_list]

        image = self.cropped_image
        text = option_list_to_use

        selected = ViperGPT_clip_utils.classify(image, text)

        return option_list[selected]

    def simple_query(self, question: str = None):
        """Returns the answer to a basic question asked about the image. If no question is provided, returns the answer
        to "What is this?". The questions are about basic perception, and are not meant to be used for complex reasoning
        or external knowledge.
        Parameters
        -------
        question : str
            A string describing the question to be asked.
        """
        if question == None:
            question = "What is this?"
        encoded_image = encode_image(self.cropped_image)

        messages = (
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": question,
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{encoded_image}"
                            },
                        },
                    ],
                },
            ],
        )

        response = ViperGPTConfig.qa_client.chat.completions.create(
            messages=messages,
            model=ViperGPTConfig.qa_model,
            temperature=ViperGPTConfig.qa_temperature,
        )
        return self.forward("blip", self.cropped_image, question, task="qa")

    def crop(self, left: int, lower: int, right: int, upper: int) -> ImagePatch:
        """Returns a new ImagePatch containing a crop of the original image at the given coordinates.
        Parameters
        ----------
        left : int
            the position of the left border of the crop's bounding box in the original image
        lower : int
            the position of the bottom border of the crop's bounding box in the original image
        right : int
            the position of the right border of the crop's bounding box in the original image
        upper : int
            the position of the top border of the crop's bounding box in the original image

        Returns
        -------
        ImagePatch
            a new ImagePatch containing a crop of the original image at the given coordinates
        """
        # make all inputs ints
        left = int(left)
        lower = int(lower)
        right = int(right)
        upper = int(upper)

        return ImagePatch(
            self.cropped_image,
            left,
            lower,
            right,
            upper,
            self.left,
            self.lower,
            queues=self.queues,
            parent_img_patch=self,
        )

    def overlaps_with(self, left, lower, right, upper):
        """Returns True if a crop with the given coordinates overlaps with this one,
        else False.
        Parameters
        ----------
        left : int
            the left border of the crop to be checked
        lower : int
            the lower border of the crop to be checked
        right : int
            the right border of the crop to be checked
        upper : int
            the upper border of the crop to be checked

        Returns
        -------
        bool
            True if a crop with the given coordinates overlaps with this one, else False
        """
        return (
            self.left <= right
            and self.right >= left
            and self.lower <= upper
            and self.upper >= lower
        )

    def llm_query(self, question: str, long_answer: bool = True) -> str:
        return llm_query(question, None, long_answer)

    def print_image(self, size: tuple[int, int] = None):
        show_single_image(self.cropped_image, size)

    def __repr__(self):
        return "ImagePatch({}, {}, {}, {})".format(
            self.left, self.lower, self.right, self.upper
        )


def best_image_match(
    list_patches: list[ImagePatch], content: List[str], return_index: bool = False
) -> Union[ImagePatch, None]:
    """Returns the patch most likely to contain the content.
    Parameters
    ----------
    list_patches : List[ImagePatch]
    content : List[str]
        the object of interest
    return_index : bool
        if True, returns the index of the patch most likely to contain the object

    Returns
    -------
    int
        Patch most likely to contain the object
    """
    if len(list_patches) == 0:
        return None

    scores = []
    for cont in content:
        res = ViperGPT_clip_utils.compare(
            [p.cropped_image for p in list_patches], cont, True
        )
        scores.append(res)
    scores = torch.stack(scores).mean(dim=0)
    scores = scores.argmax().item()  # Argmax over all image patches

    if return_index:
        return scores
    return list_patches[scores]


def distance(
    patch_a: Union[ImagePatch, float], patch_b: Union[ImagePatch, float]
) -> float:
    """
    Returns the distance between the edges of two ImagePatches, or between two floats.
    If the patches overlap, it returns a negative distance corresponding to the negative intersection over union.
    """

    if isinstance(patch_a, ImagePatch) and isinstance(patch_b, ImagePatch):
        a_min = np.array([patch_a.left, patch_a.lower])
        a_max = np.array([patch_a.right, patch_a.upper])
        b_min = np.array([patch_b.left, patch_b.lower])
        b_max = np.array([patch_b.right, patch_b.upper])

        u = np.maximum(0, a_min - b_max)
        v = np.maximum(0, b_min - a_max)

        dist = np.sqrt((u**2).sum() + (v**2).sum())

        if dist == 0:
            box_a = torch.tensor(
                [patch_a.left, patch_a.lower, patch_a.right, patch_a.upper]
            )[None]
            box_b = torch.tensor(
                [patch_b.left, patch_b.lower, patch_b.right, patch_b.upper]
            )[None]
            dist = -box_iou(box_a, box_b).item()

    else:
        dist = abs(patch_a - patch_b)

    return dist


def llm_query(query):
    """Answers a text question using an LLM. The input question is always a formatted string with a variable in it.

    Parameters
    ----------
    query: str
        the text question to ask. Must not contain any reference to 'the image' or 'the photo', etc.
    """

    messages = (
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": query,
                    }
                ],
            },
        ],
    )

    response = ViperGPTConfig.query_client.chat.completions.create(
        messages=messages,
        model=ViperGPTConfig.query_model,
        temperature=ViperGPTConfig.query_temperature,
    )
    return response.choices[0].message.content
