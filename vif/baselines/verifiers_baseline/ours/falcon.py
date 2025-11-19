from dataclasses import asdict, dataclass
from vif.baselines.verifiers_baseline.ver_baseline import TexVerBaseline
from vif.falcon.oracle.guided_oracle.guided_code_oracle import OracleGuidedCodeModule


@dataclass
class FalconVerifierMetadata:
    generated_code: str


class FalconVerifier(TexVerBaseline):
    def __init__(
        self,
        *args,
        oracle_gen_model,
        oracle_gen_model_temperature,
        box_model,
        segmentation_model,
        property_model,
        property_model_temperature,
        gclient,
        oclient,
        **kwargs,
    ):
        self.oracle_gen_model = oracle_gen_model
        self.oracle_gen_model_temperature = oracle_gen_model_temperature
        self.box_model = box_model
        self.segmentation_model = segmentation_model
        self.property_model = property_model
        self.property_model_temperature = property_model_temperature

        self.gclient = gclient

        self.oclient = oclient

        self.oracle_module: OracleGuidedCodeModule = OracleGuidedCodeModule(
            client=oclient,
            model=oracle_gen_model,
            temperature=oracle_gen_model_temperature,
            visual_client=oclient,
            box_model=box_model,
            segmentation_model=segmentation_model,
            property_client=oclient,
            property_model=property_model,
            property_model_temperature=property_model_temperature,
        )

        super().__init__(*args, **kwargs)

    def get_config_metadata(self):
        return {
            "name": "Falcon",
            "oracle_gen_model": self.oracle_gen_model,
            "oracle_gen_model_temperature": self.oracle_gen_model_temperature,
            "box_model": self.box_model,
            "segmentation_model": self.segmentation_model,
            "property_model": self.property_model,
            "property_model_temperature": self.property_model_temperature,
        }

    def assess_customization(self, ver_eval_input):
        try:
            oracle, metrics = self.oracle_module.get_oracle(
                ver_eval_input.initial_instruction, ver_eval_input.initial_image
            )
            ver_eval_input.usage_metadata[f"oracle_generation"] = [metrics]
        except Exception as e:
            ver_eval_input.errors["oracle_gen"] = [str(e)]
            return ver_eval_input
        
        try:
            or_response = oracle(ver_eval_input.initial_solution_image)
        except Exception as e:
            ver_eval_input.errors["oracle_exec"] = [str(e)]

        ver_eval_input.classified_score = 1.0 if or_response.condition else 0.0

        fal_metadata = FalconVerifierMetadata(
            generated_code=or_response.evaluation_code,
            feedback=or_response.feedbacks,
        )
        ver_eval_input.additional_metadata = asdict(fal_metadata)

        ver_eval_input.usage_metadata[f"segmentation"] = or_response.seg_token_usage
        ver_eval_input.usage_metadata[f"box"] = or_response.box_token_usage
        ver_eval_input.usage_metadata[f"property"] = or_response.prop_token_usage

        ver_eval_input.errors

        return ver_eval_input
