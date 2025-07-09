from abc import abstractmethod
import ast
from collections.abc import Callable
from dataclasses import dataclass
import os
from typing import Any
from openai import Client
from vif.models.module import LLMmodule
from PIL import Image


@dataclass
class OracleResponse:
    condition: bool
    feedbacks: list[str]
    score_object: Any = None


class OracleModule(LLMmodule):
    def __init__(self, *, model, client: Client, temperature=0.3):
        super().__init__(
            client=client,
            temperature=temperature,
            model=model,
        )

    def debug_instance_creation(self, debug: bool, debug_folder: str):
        self.debug = debug
        self.debug_folder = os.path.join(debug_folder, "oracle")

        if debug:
            os.mkdir(self.debug_folder)

    @abstractmethod
    def get_oracle(
        self, features: list[str], instruction: str, base_image: Image.Image
    ) -> Callable[[Image.Image], OracleResponse]: ...
