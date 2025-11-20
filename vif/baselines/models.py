from collections import defaultdict
import json
from typing import Any
from pydantic import BaseModel, ConfigDict
from PIL.Image import Image


class VerifierException(Exception):
    def json_dump(self):
        def default_serializer(obj):
            try:
                return str(obj)
            except Exception:
                return f"<unserializable {type(obj).__name__}>"

        data = dict(self.__dict__)
        data["name"] = type(self).__name__
        return json.dumps(data, ensure_ascii=False, default=default_serializer)


class RegexException(VerifierException):
    def __init__(self, pattern: str, content: str, *args):
        self.pattern = pattern
        self.content = content
        super().__init__(*args)


class RequestException(VerifierException):

    def __init__(self, messages: list, wrapped_exception: Any, *args):
        self.messages = messages
        self.wrapped_exception = wrapped_exception
        super().__init__(*args)


class CodeExecException(VerifierException):

    def __init__(self, code: str, wrapped_exception: Any, *args):
        self.code = code
        self.wrapped_exception = wrapped_exception
        super().__init__(*args)


class CompletionUsage(BaseModel):
    completion_tokens: int
    prompt_tokens: int
    total_tokens: int


class ChatMessage(BaseModel):
    # list of responses, can be from len 1 to n, with n being the max tries we give until considered failure
    # the last of this list is the final answer we consider
    content: list[str]
    role: str
    usage: list[CompletionUsage]


class VerEvaluation(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    #####Set before the call#####
    id: str
    # config
    approach_name: str
    config_metadata: dict
    # taken from the bench dataset
    initial_code: str
    initial_image: Image
    initial_instruction: str
    initial_solution: str
    initial_solution_image: Image
    # expected and actual output
    expected: bool
    #####Set by the call#####
    classified_score: float = None
    # if failing
    failed: bool = False
    retries: int = 0
    errors: dict[str, list[str]] = {}
    # Contains data specific to the approach(number of tool calls, code generation errors, etc)
    additional_metadata: dict = {}
    usage_metadata: dict[str, list[CompletionUsage]] = (
        dict()
    )  # mapping between model config/usage and token usages
    theoretical_perfect_image: Image = None


### corresponding hg features

import datasets

VerEval_Features = datasets.Features(
    {
        "id": datasets.Value("string"),
        "approach_name": datasets.Value("string"),
        "config_metadata": datasets.Value("string"),
        "initial_code": datasets.Value("string"),
        "initial_image": datasets.Image(),
        "initial_instruction": datasets.Value("string"),
        "initial_solution": datasets.Value("string"),
        "initial_solution_image": datasets.Image(),
        "expected": datasets.Value("bool"),
        "classified_score": datasets.Value("float64"),
        "failed": datasets.Value("bool"),
        "retries": datasets.Value("int64"),
        "try": datasets.Value("int64"),
        "index": datasets.Value("int64"),
        "errors": datasets.Value("string"),
        "additional_metadata": datasets.Value("string"),
        "usage_metadata": datasets.Value("string"),
        "theoretical_perfect_image": datasets.Image(),
    }
)
