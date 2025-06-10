from openai import OpenAI


class LLMmodule:
    def __init__(self, *, client: OpenAI, model: str, temperature: float = 0.3, **kwargs):
        super().__init__()
        self.client = client
        self.model = model
        self.temperature = temperature
