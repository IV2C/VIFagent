from abc import abstractmethod
import json
import re
from PIL import Image
from loguru import logger
from openai import OpenAI

from vif.models.module import LLMmodule
from vif.prompts.search_prompt import FEATURE_SEARCH_PROMPT
from vif.utils.image_utils import encode_image


class SearchModule:
    def __init__(self,*, debug: bool = False, debug_folder: str = ".tmp/debug", **kwargs):
        super().__init__(**kwargs) 
        self.debug = debug
        self.debug_folder = debug_folder

    @abstractmethod
    def get_features(self, image: Image.Image) -> list[str]:
        pass


class VLMSearchModule(SearchModule,LLMmodule):
    def __init__(
        self,
        *,
        client: OpenAI,
        model: str,
        temperature: float = 0.3,
        debug=False,
        debug_folder=".tmp/debug",
    ):
        super().__init__(
            debug=debug,
            debug_folder=debug_folder,
            client=client,
            model=model,
            temperature=temperature,
        )

    def get_features(self, image):
        encoded_image = encode_image(image=image)
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": FEATURE_SEARCH_PROMPT,
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{encoded_image}"
                            },
                        },
                    ],
                }
            ],
        )
        pattern = r"```(?:\w+)?\n([\s\S]+?)```"
        search_match = re.search(pattern, response.choices[0].message.content)
        if not search_match:
            logger.warning(
                f"Feature search failed, using un-identified code, unparseable response {response.choices[0].message.content}"
            )
            return None

        features_match = search_match.group(1)
        return list(json.loads(features_match)["features"])
