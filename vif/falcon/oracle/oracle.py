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
    evaluation_code:str = None


class OracleModule(LLMmodule):
    def __init__(self, *, model, client: Client, temperature=0.3):
        super().__init__(
            client=client,
            temperature=temperature,
            model=model,
        )


    @abstractmethod
    def get_oracle(
        self, instruction: str, base_image: Image.Image
    ) -> Callable[[Image.Image], OracleResponse]: ...
