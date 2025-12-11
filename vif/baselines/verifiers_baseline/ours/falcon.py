from dataclasses import asdict, dataclass
import json
from typing import Any
from vif.baselines.verifiers_baseline.ver_baseline import TexVerBaseline
from vif.falcon.oracle.guided_oracle.guided_code_oracle import OracleGuidedCodeModule
from vif.models.detection import BoundingBox,SegmentationMask
import inspect
import traceback

@dataclass
class FalconVerifierMetadata:
    generated_code: str = None
    feedback:str = None
    boxes:list[BoundingBox] = None
    segments:list[dict] = None#will be a SegmentationMask without the mask 


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
            visual_client=gclient,
            box_model=box_model,
            segmentation_model=segmentation_model,
            property_client=oclient,
            property_model=property_model,
            property_model_temperature=property_model_temperature,
        )

        super().__init__(*args, **kwargs)

    def get_config_metadata(self):
        return {
            "name": "ours",
            "oracle_gen_model": self.oracle_gen_model,
            "oracle_gen_model_temperature": self.oracle_gen_model_temperature,
            "box_model": self.box_model,
            "segmentation_model": self.segmentation_model,
            "property_model": self.property_model,
            "property_model_temperature": self.property_model_temperature,
        }

    def assess_customization(self, ver_eval_input):
        
        fal_metadata = FalconVerifierMetadata()
        try:
            oracle, metrics,oracle_code = self.oracle_module.get_oracle(
                ver_eval_input.initial_instruction, ver_eval_input.initial_image
            )
            ver_eval_input.usage_metadata[f"oracle_generation"] = [metrics]
            ver_eval_input.additional_metadata = {"generated_code":oracle_code}
        except Exception as e:
            ver_eval_input.errors["oracle_gen"] = [traceback.format_exc()]
            return ver_eval_input
        
        try:
            or_response = oracle(ver_eval_input.initial_solution_image)
        except Exception as e:
            ver_eval_input.errors["oracle_exec"] = [traceback.format_exc()]
            return ver_eval_input


        ver_eval_input.classified_score = or_response.feedbacks.score
        
        new_segmasks=[]
        for mask in or_response.segments:
            d_mask = asdict(mask)
            d_mask.pop("mask")
            new_segmasks.append(d_mask)
        
        fal_metadata = FalconVerifierMetadata(
            generated_code=or_response.evaluation_code,
            feedback=json.dumps(or_response.feedbacks.tojson(1.0)),
            boxes=or_response.boxes,
            segments=new_segmasks
        )
        ver_eval_input.additional_metadata = asdict(fal_metadata)

        ver_eval_input.usage_metadata[f"segmentation"] = or_response.seg_token_usage
        ver_eval_input.usage_metadata[f"box"] = or_response.box_token_usage
        ver_eval_input.usage_metadata[f"property"] = or_response.prop_token_usage

        return ver_eval_input
