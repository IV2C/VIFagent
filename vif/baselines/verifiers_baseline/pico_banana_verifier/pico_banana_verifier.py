import re
from openai import Client
from vif.baselines.models import RegexException,RequestException
from vif.baselines.verifiers_baseline.ver_baseline import TexVerBaseline
from vif.utils.image_utils import concat_images_horizontally, encode_image

PICO_VERIFY_SYSTEM_PROMPT: str = """
System Prompt:
You are a professional image quality evaluator specializing in image editing assessment.

Your task is to evaluate edited images by analyzing the following items in sequence:

1. Edited Image: The final edited result (primary evaluation target)
2. Input Image: A reference images used for the edit operation (1–N images)
3. Editing Instruction: The specific editing prompt or instruction used

Multi-Image Evaluation Context:
You will receive the two concatenated image (input and edited images), and the editing instruction. Use all of these to make your assessment.

Evaluation Criteria (Weighted Scoring for Image Editing):
• Edit Instruction Compliance (40% weight): Does the edited image fulfill the specific instruction? Are the requested changes clearly visible and properly implemented? Does the result match the intended edit?
• Editing Quality & Seamlessness (25% weight): Are the edits natural and realistic? Are there visible artifacts, inconsistencies, or blending issues? Is lighting and perspective preserved?
• Preservation vs. Change Balance (20% weight): Are appropriate elements from the original preserved? Are unrelated regions unaffected? Is the editing focused and not overly destructive?
• Technical Quality (15% weight): Overall sharpness, color consistency, exposure, and absence of artifacts or distortions.

Comparative Analysis:
Compare the edited result against the original image to assess:
• What changes were successfully made
• What elements were properly preserved
• Whether the instruction was accurately interpreted

Scoring:
Provide a final weighted score from 0.0 to 1.0 based on the evaluation criteria above.

Final Answer:
Return the final result as \\boxed{score}, where score is the computed score.
"""


PICO_VERIFY_PROMPT: str = """
EDIT INSTRUCTION:
{instruction}
"""


class PicoBananaVerifier(TexVerBaseline):
    def __init__(
        self,
        *args,
        model,
        client: Client,
        temperature,
        **kwargs,
    ):

        self.model = model
        self.client = client
        self.temperature = temperature

        super().__init__(*args, **kwargs)

    def get_config_metadata(self):
        return {
            "name": "PicoBananaVerifier",
            "model": self.model,
            "temperature": self.temperature,
        }

    def assess_customization(self, ver_eval_input):
        ver_eval_input.errors["base"] =[]
        concat_image = concat_images_horizontally(
            [ver_eval_input.initial_image, ver_eval_input.initial_solution_image]
        )
        encoded_image = encode_image(concat_image)

        messages = [
            {
                "role": "system",
                "content": PICO_VERIFY_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": PICO_VERIFY_PROMPT.format(
                            instruction=ver_eval_input.initial_instruction
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{encoded_image}"},
                    },
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
        pattern = r"\\boxed{([0-1]\.?[0-9]?)}"  # r"\\boxed{(True|False)}"
        id_match = re.search(pattern, cnt)

        if not id_match:
            ver_eval_input.errors["base"].append(
                RegexException(pattern=pattern, content=cnt).json_dump()
            )
            return ver_eval_input

        ver_eval_input.classified_score = float(id_match.group(1))
        ver_eval_input.usage_metadata = {"Base": [response.usage]}

        return ver_eval_input
