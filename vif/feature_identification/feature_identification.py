import json
import os
import re
from PIL import Image
from loguru import logger


from vif.feature_identification.utils import plot_segmentation_masks
from vif.models.detection import SegmentationMask, dataclassJSONEncoder
from vif.models.exceptions import InvalidMasksError, JsonFormatError, ParsingError
from vif.models.module import LLMmodule
from vif.utils.debug_utils import save_conversation, save_conversation_google
from vif.utils.detection_utils import get_segmentation_masks
from vif.utils.image_utils import encode_image
from vif.prompts.search_prompt import FEATURE_SEARCH_PROMPT

from google import genai
from google.genai import types as genTypes


class IdentificationModule:
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def debug_instance_creation(self, debug: bool, debug_folder: str):
        self.debug = debug
        self.debug_folder = os.path.join(debug_folder, "identification")
        self.segmentation_call_nb = 0
        if debug:
            if not os.path.exists(self.debug_folder):
                os.mkdir(self.debug_folder)
                os.mkdir(os.path.join(self.debug_folder, "search"))

    def segments_from_image(self, image: Image.Image) -> list[SegmentationMask]:
        pass

    def segments_from_features(
        self, features: list[str], base_image: Image.Image
    ) -> list[SegmentationMask]:
        pass


class SimpleGeminiIdentificationModule(IdentificationModule):
    """ "Simple" search module that makes two requests to a gemini-like model to get a full segmentation of the image"""

    def __init__(
        self,
        *,
        client: genai.Client,
        model: str,
        temperature: float = 0.3,
    ):
        self.client = client
        self.model = model
        self.temperature = temperature
        self.generate_content_config = genTypes.GenerateContentConfig(
            thinking_config=genTypes.ThinkingConfig(
                thinking_budget=-1,
            ),
            response_mime_type="text/plain",
            temperature=self.temperature
        )
        super().__init__()

    def segments_from_image(self, image) -> list[SegmentationMask]:
        logger.info("Getting the features from the image")
        features = self.get_features(image)
        logger.debug("Found features " + ",".join(features))
        logger.info("getting the segments from the image")

        try:
            segments = self.segments_from_features(features, image)
        except ParsingError as pe:
            logger.error(pe)
        except JsonFormatError as jfe:
            logger.error(jfe)

        logger.debug("Found segments " + ",".join(segments))

    def get_features(self, image):
        encoded_image = encode_image(image=image)
        contents = [
            genTypes.Content(
                role="user",
                parts=[
                    genTypes.Part.from_bytes(
                        mime_type="image/png",
                        data=encoded_image,
                    ),
                    genTypes.Part.from_text(text=FEATURE_SEARCH_PROMPT),
                ],
            ),
        ]
        
        response = self.client.models.generate_content(
            model=self.model,
            contents=contents,
            config=self.generate_content_config,
        )
        contents.append(response.candidates[0].content)
        if self.debug:
            save_conversation_google(contents, os.path.join(self.debug_folder, "search"))

        pattern = r"```(?:\w+)?\n([\s\S]+?)```"
        search_match = re.search(pattern, response.text)

        if not search_match:
            logger.warning(
                f"Feature search failed, using un-identified code, unparseable response {response.text}"
            )
            raise ParsingError(
                f"unparseable response {response.text}"
            )

        features_match = search_match.group(1)
        return list(json.loads(features_match)["features"])

    def segments_from_features(
        self, features: list[str], base_image: Image.Image
    ) -> list[SegmentationMask]:
        try:
            segments = get_segmentation_masks(
                base_image,
                self.client,
                features,
                self.model,
                self.generate_content_config,
            )
        except InvalidMasksError as ime:
            if self.debug:
                with open(
                    os.path.join(
                        self.debug_folder,
                        f"segments_{self.segmentation_call_nb}_res.json",
                    ),
                    "w",
                ) as seg_file:
                    json.dump(ime, seg_file)
        if self.debug:
            with open(
                os.path.join(
                    self.debug_folder, f"segments_{self.segmentation_call_nb}.json"
                ),
                "w",
            ) as seg_file:
                json.dump(segments, seg_file,cls=dataclassJSONEncoder)
            debug_seg_image = plot_segmentation_masks(base_image, segments)
            debug_seg_image.save(
                os.path.join(
                    self.debug_folder,
                    f"segmented_image_{self.segmentation_call_nb}.png",
                )
            )
            self.segmentation_call_nb += 1
        return segments
