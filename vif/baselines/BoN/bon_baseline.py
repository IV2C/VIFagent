from dataclasses import dataclass
from typing import Any
from openai import Client


#TODO
class TexBonBaseline:
    def __init__(
        self,
        *,
        edit_client: Client,
        edit_model: str,
        edit_temperature: float,
        feedback_client: Client,
        feedback_model: str,
        feedback_temperature: float,
        max_iterations:int,
        **kwargs,
    ):
        super().__init__()
        self.edit_client = edit_client
        self.edit_model = edit_model
        self.edit_temperature = edit_temperature
        self.feedback_client = feedback_client
        self.feedback_model = feedback_model
        self.feedback_temperature = feedback_temperature
        self.max_iteration = max_iterations

    def customize(code: str, instruction: str) -> Any:
        pass
