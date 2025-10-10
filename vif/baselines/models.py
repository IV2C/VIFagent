from pydantic import BaseModel
from PIL.Image import Image

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



class VerEvaluation(BaseModel):
    
    #####Set before the call#####
    id:str
    # config
    approach_name: str
    config_metadata:dict
    #taken from the bench dataset
    initial_code: str
    initial_image: Image
    initial_instruction: str
    initial_solution:str
    initial_solution_image: Image
    #expected and actual output
    expected:bool
    #####Set by the call#####
    classified:bool
    # Contains data specific to the approach(number of tool calls, code generation errors, etc)
    additional_metadata: dict

    
    