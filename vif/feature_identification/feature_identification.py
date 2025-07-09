import json
import os
import re
from PIL import Image
from loguru import logger
from openai import OpenAI


from vif.feature_identification.utils import plot_segmentation_masks
from vif.models.detection import SegmentationMask
from vif.models.module import LLMmodule
from vif.utils.debug_utils import save_conversation
from vif.utils.detection_utils import get_segmentation_masks
from vif.utils.image_utils import encode_image
from vif.prompts.search_prompt import FEATURE_SEARCH_PROMPT


class IdentificationModule:
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def debug_instance_creation(self, debug: bool, debug_folder: str):
        self.debug = debug
        self.debug_folder = os.path.join(debug_folder, "identification")
        self.segmentation_call_nb = 0
        if debug:
            os.mkdir(self.debug_folder)
            os.mkdir(os.path.join(self.debug_folder, "search"))

    def segments_from_image(self, image: Image.Image) -> list[SegmentationMask]:
        pass

    def segments_from_features(
        self, features: list[str], base_image: Image.Image
    ) -> list[SegmentationMask]:
        pass


class SimpleIdentificationModule(IdentificationModule, LLMmodule):
    """ "Simple" search module that makes two requests to a gemini-like model to get a full segmentation of the image"""

    def __init__(
        self,
        *,
        client: OpenAI,
        model: str,
        temperature: float = 0.3,
    ):
        super().__init__(
            client=client,
            model=model,
            temperature=temperature,
        )

    def segments_from_image(self, image) -> list[SegmentationMask]:
        logger.info("Getting the features from the image")
        features = self.get_features(image)
        logger.info("getting the segments from the image")
        return self.segments_from_features(features, image)

    def get_features(self, image):
        encoded_image = encode_image(image=image)

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": FEATURE_SEARCH_PROMPT,
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"},
                    },
                ],
            }
        ]
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=messages,
        )
        messages.append(response.choices[0].message)

        if self.debug:
            save_conversation(messages, os.path.join(self.debug_folder, "search"))

        pattern = r"```(?:\w+)?\n([\s\S]+?)```"
        search_match = re.search(pattern, response.choices[0].message.content)

        if not search_match:
            logger.warning(
                f"Feature search failed, using un-identified code, unparseable response {response.choices[0].message.content}"
            )
            return None

        features_match = search_match.group(1)
        return list(json.loads(features_match)["features"])

    def segments_from_features(
        self, features: list[str], base_image: Image.Image
    ) -> list[SegmentationMask]:
        segments = get_segmentation_masks(
            base_image,
            self.client,
            features,
            self.model,
            self.temperature,
        )
        if self.debug:
            with open(
                os.path.join(
                    self.debug_folder, f"segments_{self.segmentation_call_nb}.txt"
                ),"w"
            ) as seg_file:
                seg_file.write(segments)
            debug_seg_image = plot_segmentation_masks(base_image, segments)
            debug_seg_image.save(
                os.path.join(
                    self.debug_folder,
                    f"segmented_image_{self.segmentation_call_nb}.png",
                )
            )
            self.segmentation_call_nb += 1
        return segments
