from pydantic import BaseModel


class CompletionUsage(BaseModel):
    completion_tokens: int
    prompt_tokens: int
    total_tokens: int

class ChatMessage(BaseModel):
    #list of responses, can be from len 1 to n, with n being the max tries we give until considered failure
    # the last of this list is the final answer we consider
    content: list[str]
    role: str
    usage: list[CompletionUsage]



class IteCustomizationOutput(BaseModel):
    #taken from the bench dataset
    initial_code: str
    initial_instruction: str
    #exchanges with the llm
    messages: list[ChatMessage]
    # Contains data specific to the approach(number of tool calls, code generation errors, etc)
    additional_metadata: dict
    # config
    approach_name: str
    edit_model_temperature: float
    edit_model_name: str
    feedback_model_temperature: float
    feedback_model_name: str
    
    
class BonCustomizationOutput(BaseModel):
    #Maybe todo?
    pass