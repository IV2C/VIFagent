from collections import defaultdict
from typing import Any
from pydantic import BaseModel
from PIL.Image import Image


class VerifierException(Exception): ...


class RegexException(VerifierException):
    def __init__(self, pattern: str, content: str, *args):
        self.pattern = pattern
        self.content = content
        super().__init__(*args)


class RequestException(VerifierException):
    """See here for more details https://platform.openai.com/docs/guides/error-codes/api-errors"""

    def __init__(self, messages: list, wrapped_exception: Any, *args):
        self.messages = messages
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
    classified: bool = None
    # if failing
    failed: bool = False
    retries: int = 0
    errors: list[VerifierException]
    # Contains data specific to the approach(number of tool calls, code generation errors, etc)
    additional_metadata: dict
    usage_metadata: dict[str, list[CompletionUsage]] = (
        dict()
    )  # mapping between model config/usage and token usages
