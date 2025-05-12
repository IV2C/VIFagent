from vif_agent.modules.identification.identification import IdentificationModule
from PIL import Image
from collections.abc import Callable
from openai import Client
from vif_agent.modules.identification.prompt import PINPOINT_PROMPT
from vif_agent.modules.identification.utils import get_boxes


class IdentificationOracleBoxModule(IdentificationModule):
    def __init__(
        self,
        *,
        model,
        client: Client,
        temperature=0.3,
        debug: bool = False,
        debug_folder: str = ".tmp/debug",
        pinpoint_client: Client = None,
        pinpoint_model=None,
        pinpoint_model_temperature=None,
    ):
        self.pinpoint_client = pinpoint_client or client
        self.pinpoint_model = pinpoint_model or model
        self.pinpoint_model_temperature = pinpoint_model_temperature or temperature

        self.identification_client = client
        self.identification_model = model
        self.identification_model_temperature = temperature
        super().__init__(debug=debug, debug_folder=debug_folder)

    def get_oracle(
        self, features: list[str], instruction: str, base_image: Image.Image
    ) -> Callable[[Image.Image], bool]:
        feature_string = ",".join([f for f in features])
        pinpoint_instructions = PINPOINT_PROMPT.format(
            features=feature_string, instruction=instruction
        )
        response = self.pinpoint_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": pinpoint_instructions}],
        )

        detected_boxes = get_boxes(
            base_image,
            self.identification_client,
            self.identification_model,
            self.identification_model_temperature,
        )
        #create the function that takes an image and returns the most edited box 
        # => refactor BoxIdentificationModule(mapping.py at line ~144 `for box in detected_boxes:`) make a function that return the sorted mse map

        pass

    def __str__(self):
        return (
            f"{self.__class__.__name__}(\n"
            f"  identification_model={self.identification_model},\n"
            f"  identification_model_temperature={self.identification_model_temperature},\n"
            f"  pinpoint_model={self.pinpoint_model},\n"
            f"  pinpoint_model_temperature={self.pinpoint_model_temperature},\n"
            f"  debug={self.debug},\n"
            f"  debug_folder='{self.debug_folder}'\n"
            f")"
        )
