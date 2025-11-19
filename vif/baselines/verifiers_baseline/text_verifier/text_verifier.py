import re
from openai import Client
from vif.baselines.models import RegexException,RequestException
from vif.baselines.verifiers_baseline.ver_baseline import TexVerBaseline

TEXT_VERIFY_SYSTEM_PROMPT: str = """
You are a verification agent, your task is to assess whether a code applied a given customization instruction or not.
You will be given the initial code, the customized code, and the instruction.

Your response must always contain the final answer in the format:
\\boxed{score}

With score being a float between 0 and 1.
0.0 => not applied at all.
1.0 => Perfectly applied.
"""

TEXT_VERIFY_PROMPT: str = """
INITIAL CODE:
```
{initial_code}
```
CUSTOMIZED CODE:
```
{customized_code}
```

INSTRUCTION:
{instruction}
"""


class TextVerifier(TexVerBaseline):
    def __init__(self, *args, model, client: Client, temperature, **kwargs):

        self.model = model
        self.client = client
        self.temperature = temperature

        super().__init__(*args, **kwargs)

    def get_config_metadata(self):
        return {
            "name": "TextVerifier",
            "model": self.model,
            "temperature": self.temperature,
        }

    def assess_customization(self, ver_eval_input):
        ver_eval_input.errors["base"] =[]
        messages = [
            {
                "role": "system",
                "content": TEXT_VERIFY_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": TEXT_VERIFY_PROMPT.format(
                            instruction=ver_eval_input.initial_instruction,
                            initial_code=ver_eval_input.initial_code,
                            customized_code=ver_eval_input.initial_solution,
                        ),
                    }
                ],
            },
        ]
        try:
            response = self.client.chat.completions.create(
                messages=messages, model=self.model, temperature=self.temperature
            )
        except Exception as e:
            ver_eval_input.errors["base"].append(
                RequestException(messages=messages, wrapped_exception=e).json_dump()
            )
            return ver_eval_input

        cnt = response.choices[0].message.content
        ver_eval_input.additional_metadata["response_content"] = cnt
        pattern = r"\\boxed{([0-1]\.?[0-9]?)}"
        id_match = re.search(pattern, cnt)

        if not id_match:
            ver_eval_input.errors["base"].append(RegexException(pattern=pattern, content=cnt).json_dump())
            return ver_eval_input

        ver_eval_input.classified_score = float(id_match.group(1))

        # token usage:
        ver_eval_input.usage_metadata = {"Base": [response.usage]}

        return ver_eval_input
