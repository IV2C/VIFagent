from dataclasses import asdict, dataclass
from vif.baselines.verifiers_baseline.ver_baseline import TexVerBaseline


@dataclass
class FalconVerifierMetadata:
    generated_code: str


class FalconVerifier(TexVerBaseline):
    def __init__(
        self,
        *args,
        oracle_gen_model,
        oracle_gen_model_temperature,
        vision_model,
        property_model,
        property_model_temperature,
        gclient,
        oclient,
        **kwargs,
    ):
        self.oracle_gen_model = oracle_gen_model
        self.oracle_gen_model_temperature = oracle_gen_model_temperature
        self.vision_model = vision_model
        self.property_model = property_model
        self.property_model_temperature = property_model_temperature

        self.gclient = gclient

        self.oclient = oclient

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
            f"oracle_generation"
        ] = [metrics]

        or_response = oracle(ver_eval_input.initial_solution_image)

        ver_eval_input.classified = or_response.condition

        fal_metadata = FalconVerifierMetadata(
            generated_code=or_response.evaluation_code,
            feedback=or_response.feedbacks,
        )
        ver_eval_input.additional_metadata = asdict(fal_metadata)

        ver_eval_input.usage_metadata[f"segmentation"] = (
            or_response.seg_token_usage
        )
        ver_eval_input.usage_metadata[f"property"] = (
            or_response.prop_token_usage
        )
        # TODO add token usage for property

        return ver_eval_input
