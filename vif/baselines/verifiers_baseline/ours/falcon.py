from dataclasses import asdict, dataclass
import re
from vif.baselines.models import RegexException
from vif.baselines.verifiers_baseline.ver_baseline import TexVerBaseline
from openai import Client
import os
from google import genai
from google.genai import types as genTypes
from vif.falcon.oracle.guided_oracle.guided_code_oracle import OracleGuidedCodeModule
from vif.utils.renderer.tex_renderer import TexRenderer


@dataclass
class FalconVerifierMetadata:
    generated_code: str


class FalconVerifier(TexVerBaseline):
    def __init__(self, *args, **kwargs):

        self.oracle_gen_model = "meta-llama/llama-4-maverick:free"
        self.oracle_gen_model_temperature = 0.3
        self.vision_model = "gemini-2.5-flash"
        self.property_model = "mistralai/mistral-small-3.2-24b-instruct:free"
        self.property_model_temperature = 0.3

        self.gclient = genai.Client(
            api_key=os.environ.get("GOOGLE_API_KEY"),
            http_options=genTypes.HttpOptions(api_version="v1alpha"),
        )

        self.oclient = Client(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ.get("OPENROUTER_API_KEY"),
        )

        self.oracle_module = OracleGuidedCodeModule(
            model=self.oracle_gen_model,
            temperature=self.oracle_gen_model_temperature,
            client=self.oclient,
            visual_client=self.gclient,
            visual_model=self.vision_model,
            property_client=self.oclient,
            property_model=self.property_model,
            property_model_temperature=self.property_model_temperature,
        )

        super().__init__(*args, **kwargs)

    def get_config_metadata(self):
        return {
            "name": "Falcon",
            "oracle_gen_model": self.oracle_gen_model,
            "oracle_gen_model_temperature": self.oracle_gen_model_temperature,
            "vision_model": self.vision_model,
            "property_model": self.property_model,
            "property_model_temperature": self.property_model_temperature,
        }

    def assess_customization(self, ver_eval_input):

        oracle, metrics = self.oracle_module.get_oracle(
            ver_eval_input.initial_instruction, ver_eval_input.initial_image
        )
        ver_eval_input.usage_metadata[
            f"oracle_generation-{self.oracle_gen_model}_{self.oracle_gen_model_temperature}"
        ] = [metrics]

        or_response = oracle(ver_eval_input.initial_solution_image)

        ver_eval_input.classified = or_response.condition

        fal_metadata = FalconVerifierMetadata(
            generated_code=or_response.evaluation_code,
            feedback=or_response.feedbacks,
        )
        ver_eval_input.additional_metadata = asdict(fal_metadata)

        ver_eval_input.usage_metadata[f"segmentation-{self.vision_model}"] = (
            or_response.seg_token_usage
        )

        # TODO add token usage for property

        return ver_eval_input
