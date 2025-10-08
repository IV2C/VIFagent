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



class VerEvaluationOutput(BaseModel):
    id:str
    #taken from the bench dataset
    initial_code: str
    initial_instruction: str
    initial_solution:str
    #expected and actual output
    expected:bool
    classified:bool
    # Contains data specific to the approach(model config,number of tool calls, code generation errors, etc)
    additional_metadata: dict
    # config
    approach_name: str
    
    